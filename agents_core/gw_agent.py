"""GravityWell review-agent harness — a read-only tool-loop backed by GW.

call_gw_agent() runs a multi-step agent on GravityWell with access to read-only tools
(read_file, grep, git, mem). The harness manages the doorman lease, tool execution,
loop control, and provenance tracking.

Unlike call_claude_cli() or call_operator() (stateless single-shot), this agent
conducts adaptive archaeology by requesting tools, executing them locally, and
feeding results back in multi-turn messages until reaching a verdict.

GW_URL is read from the environment (default http://203.0.113.11:8081);
doorman is imported from agents_core.doorman_client.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

from agents_core.doorman_client import DoormanClient, DoormanUnreachable

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")
GW_AGENT_TOOL_OUTPUT_CAP = 8192
GW_AGENT_TOOL_INPUT_CAP = 65536
GW_AGENT_CTX_CAP = 120000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Registry & Executors
# ---------------------------------------------------------------------------

class ToolExecutor:
    """Base class for tool executors."""

    def execute(self, arguments: dict) -> str | dict:
        """Execute the tool with the given arguments.

        Returns: str (tool output) or dict with 'error' key on failure.
        """
        raise NotImplementedError


class ReadFileExecutor(ToolExecutor):
    """Execute read_file(path, start_line?, end_line?)."""

    def __init__(self, cwd: str | None = None):
        self.cwd = Path(cwd or "/srv/agents").resolve()

    def execute(self, arguments: dict) -> str | dict:
        try:
            path_arg = arguments["path"]
            # Resolve relative to cwd, then verify it's still under cwd
            path = (self.cwd / path_arg).resolve()
            if not str(path).startswith(str(self.cwd)):
                return {"error": f"path outside cwd: {path}"}

            try:
                content = path.read_text()
            except FileNotFoundError:
                return {"error": f"file not found: {path}"}

            start_line = arguments.get("start_line", 1)
            end_line = arguments.get("end_line")

            lines = content.splitlines()
            if start_line < 1:
                start_line = 1
            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if end_line is None else min(end_line, len(lines))

            output_lines = lines[start_idx:end_idx]
            result = "\n".join(output_lines)

            if len(result) > GW_AGENT_TOOL_OUTPUT_CAP:
                result = result[:GW_AGENT_TOOL_OUTPUT_CAP] + "\n…[truncated]"

            return result
        except Exception as e:
            return {"error": f"read_file failed: {e}"}


class GrepExecutor(ToolExecutor):
    """Execute grep(pattern, path_glob?)."""

    def __init__(self, cwd: str | None = None):
        self.cwd = Path(cwd or "/srv/agents").resolve()

    def execute(self, arguments: dict) -> str | dict:
        try:
            pattern = arguments["pattern"]
            path_glob = arguments.get("path_glob", "**/*")

            # Build ripgrep command - search under cwd for the glob pattern
            # Use -l (files only), -m 100 (max 100 matches)
            cmd = ["rg", pattern, "-l", "-m", "100", str(self.cwd)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                output = result.stdout
            else:
                output = ""

            if len(output) > GW_AGENT_TOOL_OUTPUT_CAP:
                output = output[:GW_AGENT_TOOL_OUTPUT_CAP] + "\n…[truncated]"

            return output or "(no matches)"
        except subprocess.TimeoutExpired:
            return {"error": "grep timeout"}
        except Exception as e:
            return {"error": f"grep failed: {e}"}


class GitExecutor(ToolExecutor):
    """Execute git(args) with read-only allowlist."""

    ALLOWLIST = {"log", "show", "diff", "status", "blame", "ls-files", "rev-list", "cat-file", "describe", "shortlog", "fetch"}

    def __init__(self, cwd: str | None = None):
        self.cwd = Path(cwd or "/srv/agents").resolve()

    def execute(self, arguments: dict) -> str | dict:
        try:
            args = arguments.get("args", "")
            if isinstance(args, list):
                args = " ".join(args)

            # Check for shell metacharacters (no pipes, semicolons, etc.; spaces are OK)
            if any(c in args for c in ";|&$()`\n\r"):
                return {"error": "shell metacharacters not allowed in git args"}

            # Parse subcommand
            tokens = args.split()
            if not tokens:
                return {"error": "no git subcommand provided"}

            if tokens[0] not in self.ALLOWLIST:
                return {"error": f"git subcommand {tokens[0]!r} not allowed (read-only allowlist)"}

            cmd = ["git", "-C", str(self.cwd)] + tokens
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = result.stdout
            if result.returncode != 0:
                output = result.stderr or f"(git {tokens[0]} exited {result.returncode})"

            if len(output) > GW_AGENT_TOOL_OUTPUT_CAP:
                output = output[:GW_AGENT_TOOL_OUTPUT_CAP] + "\n…[truncated]"

            return output
        except subprocess.TimeoutExpired:
            return {"error": "git timeout"}
        except Exception as e:
            return {"error": f"git failed: {e}"}


class MemExecutor(ToolExecutor):
    """Execute mem(action, query) with read-only allowlist."""

    ALLOWLIST = {"search", "get"}

    def __init__(self, cwd: str | None = None):
        # mem is global and doesn't need cwd, but accept it for API consistency
        pass

    def execute(self, arguments: dict) -> str | dict:
        try:
            action = arguments.get("action", "")
            query = arguments.get("query", "")

            if action not in self.ALLOWLIST:
                return {"error": f"mem action {action!r} not allowed (read-only allowlist)"}

            cmd = ["mem", action, query]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = result.stdout
            if result.returncode != 0:
                output = result.stderr or f"(mem {action} exited {result.returncode})"

            if len(output) > GW_AGENT_TOOL_OUTPUT_CAP:
                output = output[:GW_AGENT_TOOL_OUTPUT_CAP] + "\n…[truncated]"

            return output
        except subprocess.TimeoutExpired:
            return {"error": "mem timeout"}
        except Exception as e:
            return {"error": f"mem failed: {e}"}


class OpenPrsExecutor(ToolExecutor):
    """Execute list_open_prs(repo, with_files?) to enumerate open PRs with optional file lists."""

    def __init__(self, cwd: str | None = None):
        # PRs are remote resources, no cwd confinement; accept for API consistency
        pass

    def execute(self, arguments: dict) -> str | dict:
        try:
            repo = arguments.get("repo", "")
            with_files = arguments.get("with_files", False)

            if not repo:
                return {"error": "repo parameter is required"}

            # Import here to avoid circular dependency
            from agents_core import forgejo

            # Fetch open PRs
            try:
                prs = forgejo.get_open_prs(repo, owner=None)
            except Exception as e:
                return {"error": f"failed to fetch open PRs: {e}"}

            # Format result: keep all PRs, cap only body snippet
            result = []
            for pr in prs:
                pr_record = {
                    "number": pr.get("number"),
                    "title": pr.get("title", ""),
                    "head": pr.get("head", {}).get("ref", ""),
                    "base": pr.get("base", {}).get("ref", ""),
                    "updated_at": pr.get("updated_at", ""),
                }

                # Snip body to ~200 chars, mark if truncated
                body = pr.get("body", "")
                if body and len(body) > 200:
                    pr_record["body"] = body[:200] + "…"
                else:
                    pr_record["body"] = body

                # Optionally fetch changed files
                if with_files:
                    changed_files = self._get_changed_files(repo, pr.get("number"))
                    pr_record["changed_files"] = changed_files

                result.append(pr_record)

            # Return as JSON string (body and changed_files are already capped per-PR)
            return json.dumps(result)
        except Exception as e:
            return {"error": f"list_open_prs failed: {e}"}

    def _get_changed_files(self, repo: str, pr_number: int) -> list[str]:
        """Fetch changed files for a PR, with fallback to diff parsing.

        Prefers the PR files API endpoint if available, falls back to diff parsing.
        Caps list to ~50 files per PR, marking truncation if needed.
        """
        from agents_core import forgejo

        changed_files = []

        # Try the PR files endpoint first
        try:
            files_data = forgejo.get_pr_files(repo, pr_number)
            if isinstance(files_data, list):
                for f in files_data:
                    if f.get("filename"):
                        changed_files.append(f["filename"])
                    if len(changed_files) >= 50:
                        remaining = len(files_data) - 50
                        if remaining > 0:
                            changed_files.append(f"…(+{remaining} more)")
                        break
                return changed_files
        except Exception:
            # Fall through to diff parsing if files endpoint fails
            pass

        # Fall back to diff parsing
        try:
            diff = forgejo.get_pr_diff(repo, pr_number)
            changed_files = self._parse_diff_for_paths(diff)
            return changed_files[:50] if len(changed_files) <= 50 else changed_files[:50] + [f"…(+{len(changed_files) - 50} more)"]
        except Exception:
            # If diff parsing also fails, return empty list
            return []

    def _parse_diff_for_paths(self, diff: str) -> list[str]:
        """Extract file paths from a diff robustly.

        Looks for `+++ b/<path>` headers and extracts the full path
        without whitespace-tokenization (to preserve paths with spaces).
        """
        paths = []
        for line in diff.split("\n"):
            if line.startswith("+++ b/"):
                # Extract everything after "+++ b/" to end of line
                path = line[6:]  # len("+++ b/") == 6
                if path:
                    paths.append(path)
        return list(dict.fromkeys(paths))  # Remove duplicates while preserving order


class WriteFileExecutor(ToolExecutor):
    """Execute write_file(path, content): create/overwrite a file under cwd."""

    def __init__(self, cwd: str | None = None):
        self.cwd = Path(cwd or "/srv/agents").resolve()

    def execute(self, arguments: dict) -> str | dict:
        try:
            path_arg = arguments["path"]
            content = arguments["content"]

            path = (self.cwd / path_arg).resolve()
            if not str(path).startswith(str(self.cwd)):
                return {"error": f"path outside cwd: {path}"}

            if len(content) > GW_AGENT_TOOL_INPUT_CAP:
                return {"error": f"content too large: {len(content)} bytes > {GW_AGENT_TOOL_INPUT_CAP}"}

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"wrote {len(content)} bytes to {path_arg}"
        except Exception as e:
            return {"error": f"write_file failed: {e}"}


class ApplyEditExecutor(ToolExecutor):
    """Execute apply_edit(path, old_string, new_string): exact-string unique replace."""

    def __init__(self, cwd: str | None = None):
        self.cwd = Path(cwd or "/srv/agents").resolve()

    def execute(self, arguments: dict) -> str | dict:
        try:
            path_arg = arguments["path"]
            old_string = arguments["old_string"]
            new_string = arguments["new_string"]

            path = (self.cwd / path_arg).resolve()
            if not str(path).startswith(str(self.cwd)):
                return {"error": f"path outside cwd: {path}"}

            try:
                content = path.read_text()
            except FileNotFoundError:
                return {"error": f"file not found: {path_arg}"}

            count = content.count(old_string)
            if count == 0:
                return {"error": f"old_string not found in {path_arg}"}
            if count > 1:
                return {"error": f"old_string not unique in {path_arg}: found {count} occurrences"}

            new_content = content.replace(old_string, new_string, 1)
            path.write_text(new_content)
            return f"applied edit to {path_arg}"
        except Exception as e:
            return {"error": f"apply_edit failed: {e}"}


class RunTestsExecutor(ToolExecutor):
    """Execute run_tests(target?, k_expr?): run pytest in cwd."""

    _SHELL_METACHARS = set(";|&$()`\n\r")

    def __init__(self, cwd: str | None = None, run_timeout: int = 180):
        self.cwd = Path(cwd or "/srv/agents").resolve()
        self.run_timeout = run_timeout

    def execute(self, arguments: dict) -> str | dict:
        try:
            target = arguments.get("target")
            k_expr = arguments.get("k_expr")

            for val in [target, k_expr]:
                if val and any(c in val for c in self._SHELL_METACHARS):
                    return {"error": "shell metacharacters not allowed in test args"}

            cmd = [sys.executable, "-m", "pytest"]
            if target:
                cmd.append(target)
            if k_expr:
                cmd.extend(["-k", k_expr])
            cmd.append("-q")

            timed_out = False
            returncode = -1
            output = ""
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.run_timeout,
                    cwd=str(self.cwd),
                    shell=False,
                )
                output = result.stdout + result.stderr
                returncode = result.returncode
            except subprocess.TimeoutExpired as e:
                output = (
                    (e.stdout or b"").decode(errors="replace")
                    + (e.stderr or b"").decode(errors="replace")
                    + f"\n[TIMEOUT after {self.run_timeout}s]"
                )
                timed_out = True

            if len(output) > GW_AGENT_TOOL_OUTPUT_CAP:
                output = output[:GW_AGENT_TOOL_OUTPUT_CAP] + "\n…[truncated]"

            return _parse_pytest_outcome(output, returncode, timed_out)
        except Exception as e:
            return {"error": f"run_tests failed: {e}"}


def _parse_pytest_outcome(output: str, returncode: int, timed_out: bool) -> dict:
    """Parse pytest -q output into a structured outcome dict."""
    lines = output.strip().splitlines()
    summary = lines[-1] if lines else ""

    passed = 0
    failed = 0
    errors = 0

    passed_m = re.search(r"(\d+) passed", summary)
    failed_m = re.search(r"(\d+) failed", summary)
    error_m = re.search(r"(\d+) error", summary)
    if passed_m:
        passed = int(passed_m.group(1))
    if failed_m:
        failed = int(failed_m.group(1))
    if error_m:
        errors = int(error_m.group(1))

    tail_lines = lines[-20:] if len(lines) > 20 else lines

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "timed_out": timed_out,
        "returncode": returncode,
        "summary": summary,
        "output_tail": "\n".join(tail_lines),
    }


# FixerResult is the structured return value of a writeable call_gw_agent run.
# final_diff: git diff output (empty string if no changes).
# last_test_outcome: last run_tests structured dict, or None if never called.
# concluded: True iff the run ended on finish_reason=="stop".
# steps: per-step transcript list (same entries as return_transcript mode).
FixerResult = dict  # alias for documentation; shape enforced by _build_fixer_result


def _build_fixer_result(
    cwd: str,
    transcript: list[dict],
    concluded: bool,
    max_steps_reached: bool = False,
    no_progress: bool = False,
    budget_forced: bool = False,
    interrupted: bool = False,
    interrupt_reason: str = "",
) -> dict:
    """Build a FixerResult dict from the completed writeable run."""
    diff_result = subprocess.run(
        ["git", "-C", cwd, "diff"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    final_diff = diff_result.stdout if diff_result.returncode == 0 else ""

    last_test_outcome = None
    for entry in reversed(transcript):
        if entry.get("tool_name") == "run_tests" and entry.get("error") is None:
            try:
                last_test_outcome = json.loads(entry["result"])
            except (json.JSONDecodeError, KeyError):
                pass
            break

    return {
        "final_diff": final_diff,
        "last_test_outcome": last_test_outcome,
        "concluded": concluded,
        "max_steps_reached": max_steps_reached,
        "no_progress": no_progress,
        "budget_forced": budget_forced,
        "interrupted": interrupted,
        "interrupt_reason": interrupt_reason,
        "steps": transcript,
    }


DEFAULT_READONLY_TOOLS: dict[str, dict[str, Any]] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository, optionally within a line range. Path is resolved and confined to cwd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to cwd).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-indexed, default 1).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (1-indexed, default EOF).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    "grep": {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files using ripgrep. Returns matching file paths (up to 100 matches).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path_glob": {
                        "type": "string",
                        "description": "Optional glob pattern for files (default '**/*').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    "git": {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Execute a read-only git command (log, show, diff, status, blame, ls-files, rev-list, cat-file, describe, shortlog). Output is capped at 8KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Git subcommand and arguments (e.g., 'log --oneline -10', 'show HEAD:file.txt').",
                    },
                },
                "required": ["args"],
            },
        },
    },
    "mem": {
        "type": "function",
        "function": {
            "name": "mem",
            "description": "Query the memory store (read-only: search or get entries). Use 'search' to find keys by topic, 'get' to retrieve a full entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "get"],
                        "description": "Action: 'search' finds keys by substring/tags, 'get' retrieves a full entry.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for 'search') or key name (for 'get').",
                    },
                },
                "required": ["action", "query"],
            },
        },
    },
    "list_open_prs": {
        "type": "function",
        "function": {
            "name": "list_open_prs",
            "description": "List open pull requests in a repository, optionally including the file paths each PR touches for overlap detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository name (e.g., 'agents-core').",
                    },
                    "with_files": {
                        "type": "boolean",
                        "description": "Optional: if true, include changed_files list per PR for overlap detection (default false).",
                    },
                },
                "required": ["repo"],
            },
        },
    },
}

DEFAULT_FIXER_TOOLS: dict[str, dict[str, Any]] = {
    **DEFAULT_READONLY_TOOLS,
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file under cwd. Creates parent dirs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "apply_edit": {
        "type": "function",
        "function": {
            "name": "apply_edit",
            "description": "Replace an exact, unique old_string with new_string in a file under cwd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "run_tests": {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest in cwd. Optional target (path/node-id) and k_expr (-k filter).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "k_expr": {"type": "string"},
                },
                "required": [],
            },
        },
    },
}


def _get_tool_executors(cwd: str | None = None, writeable: bool = False) -> dict[str, ToolExecutor]:
    """Instantiate tool executors with a given cwd. When writeable=True adds write executors."""
    result: dict[str, ToolExecutor] = {
        "read_file": ReadFileExecutor(cwd),
        "grep": GrepExecutor(cwd),
        "git": GitExecutor(cwd),
        "mem": MemExecutor(cwd),
        "list_open_prs": OpenPrsExecutor(cwd),
    }
    if writeable:
        result["write_file"] = WriteFileExecutor(cwd)
        result["apply_edit"] = ApplyEditExecutor(cwd)
        result["run_tests"] = RunTestsExecutor(cwd)
    return result


# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------


def call_gw_agent(
    prompt: str,
    system: str = "",
    cwd: str | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
    max_steps: int = 24,
    timeout: int = 300,
    json_mode: bool = False,
    think: bool = False,
    on_wake_fail: str = "skip",
    work_id: str | None = None,
    return_transcript: bool = False,
    log: Callable[[str], None] | None = None,
    backend_url: str | None = None,
    acquire_lease: bool = True,
    writeable: bool = False,
    no_progress_steps: int = 8,
    principal: str | None = None,
    verdict_schema: dict | None = None,
    tool_executors: dict[str, ToolExecutor] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    before_tool: Callable[[str, dict], dict] | None = None,
    reason_out: list[str] | None = None,
    served_model_out: list | None = None,
    model: str | None = None,
) -> str | None | tuple[str | None, list[dict]] | tuple[dict, list[dict]]:
    """Run a multi-step read-only tool-loop on GravityWell.

    The agent conducts adaptive archaeology via tools (read_file, grep, git, mem),
    executing them locally and feeding results back until reaching a verdict.

    Args:
        prompt: User prompt / task description.
        system: Optional system prompt (instruction set). If not provided,
                a default reviewer system prompt with attribution grammar is used.
        cwd: Working directory for tool execution and git context.
             Defaults to "/srv/agents".
        tools: Optional dict of tool definitions (OpenAI format). If None,
               uses DEFAULT_READONLY_TOOLS.
        max_steps: Max number of tool-call iterations (default 24). When exhausted,
                   a forced-conclusion turn attempts to emit a parseable verdict.
        timeout: Wall-clock timeout for the entire run (default 300s).
        json_mode: If True, appends "respond with JSON only" to the system prompt.
        think: If True, enables GW's thinking mode (default False).
        on_wake_fail: Policy when the doorman cannot wake GW:
                      - "skip" → return None (default)
                      - "error" → raise an exception
                      - "claude" → fall back to call_claude_cli with Sonnet (paid)
        work_id: Trace ID for the doorman lease. If None, generates internally.
        return_transcript: If True, return (text, transcript) tuple instead of just text.
        log: Optional logging function for progress/debug output.
        backend_url: Optional backend URL override (default None → GW_URL). Used by
                     swarm consumers to post to a different endpoint.
        acquire_lease: If False, skip doorman lease acquisition entirely (default True).
                       With defaults (True), behavior is byte-identical: acquire/release
                       are called, POST is to GW_URL. Only set both backend_url and
                       acquire_lease=False when running on swarm.
        writeable: If True, add write tools (write_file, apply_edit, run_tests) and return
                   (FixerResult, transcript). Default False keeps behavior byte-identical to
                   read-only callers. The return_transcript argument is ignored for writeable
                   runs — the tuple form is always used.
        tool_executors: Optional executor map {tool_name: ToolExecutor}. When provided,
                        used instead of the default registry built by _get_tool_executors.
                        When None (default), behavior is unchanged. Supply together with
                        a matching `tools` param (OpenAI tool defs).
        cancel_check: Optional callable () -> bool. When provided, called at the top of each
                      step and immediately before each tool execution. Truthy return halts the
                      loop with an interrupted result (reason="user_cancel"). Raising halts
                      with reason="cancel_check_failed" (fail-safe: a broken STOP must never
                      silently continue). When None (default), never called.
        before_tool: Optional callable (tool_name, tool_args) -> dict. When provided, called
                     before each tool execution. Return value is a gate dict with key
                     "decision": "proceed" (execute normally), "reject" (skip execution, feed
                     {"error": "rejected: <reason>"} back to the model), or "stop" (halt
                     loop, interrupted result). Raising is fail-closed: the tool is skipped
                     with {"error": "gate_failure: <detail>"} fed back, loop continues. When
                     None (default), never called.
        reason_out: Optional list. When provided, on a `writeable=False` (readonly/json_mode)
                    call that collapses to an empty result, one of the following category
                    strings is appended: "gw_unreachable", "gw_not_serving", "request_failed",
                    "no_choices", "grounding_failed", "budget_exhausted", "max_steps_exhausted",
                    "interrupted". Left untouched on a genuinely successful (non-empty) result.
                    Stays empty/unpopulated for `writeable=True` calls regardless of cause. Pure
                    side channel - does not change the return type. When None (default), never
                    touched.
        served_model_out: Optional list. When provided, the top-level "model" field echoed by
                           each completion response (main tool-loop steps and forced-conclusion
                           turns) is appended to it as observed - never overwritten/reset mid-run,
                           so a run with N model-echoing steps produces N entries. A caller
                           wanting the run's final/deciding served model reads
                           `served_model_out[-1]` after this function returns (a forced-conclusion
                           turn appends last, so it naturally wins). Silent (no append) when a
                           response never echoes a "model" field. Pure side channel - does not
                           change the return type. When None (default), never touched.
        model: Optional model name to request from the backend. Included as the "model"
               field in both POST payloads (main loop + forced-conclusion) when provided.
               When None (default), the field is omitted entirely — backward compatible
               with single-model vLLM endpoints that serve whatever is loaded.

    Returns:
        - str or None (or (str|None, list) when return_transcript=True).
        - None means "did not run" (only on on_wake_fail="skip" + doorman failure).
        - Non-None with "[gw_agent: max_steps reached ...]" suffix means loop exhausted.
        - Transcript (if return_transcript) is a list of dicts with tool execution details.
        - When writeable=True: always (FixerResult, transcript) regardless of return_transcript.

    The doorman lease is acquired once and held for the entire run; released in finally
    (unless acquire_lease=False). Tool errors are recovered gracefully: a malformed call
    returns a tool-error message so GW can adapt (the loop never crashes on tool execution).
    """
    if system == "":
        system = _default_reviewer_system_prompt()

    if json_mode:
        system = system + "\n\nYour FINAL answer must be valid JSON, no markdown fences."

    if cwd is None:
        cwd = "/srv/agents"

    if work_id is None:
        work_id = uuid.uuid4().hex[:8]

    # Capture swarm flag BEFORE backend_url is reassigned to GW_URL.
    # After the reassignment backend_url is never None, so testing it downstream is useless.
    _is_swarm = (backend_url is not None) and (not acquire_lease)

    if tools is None:
        tools = DEFAULT_FIXER_TOOLS if writeable else DEFAULT_READONLY_TOOLS

    if backend_url is None:
        backend_url = GW_URL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    transcript: list[dict] = []
    if tool_executors is None:
        tool_executors = _get_tool_executors(cwd, writeable=writeable)
    repeated_calls: dict[str, int] = {}
    ctx_tokens = 0
    # No-progress guard state (writeable mode): track consecutive steps with no semantic progress.
    consecutive_no_progress = 0
    last_test_counts: tuple | None = None
    # Grounding guard state (json_mode review runs): track verified (error-free) tool calls.
    grounding_count = 0  # tool calls with error is None
    grounding_nudged = False  # True after the first 0-tool-call stop nudge
    # Interrupt state: set by cancel_check or before_tool stop.
    _interrupted = False
    _interrupt_reason = ""
    _interrupted_step = 0

    # Acquire doorman lease for the whole run (unless acquire_lease=False for swarm).
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable, _gw_acquire_timeout

    client = DoormanClient()
    try:
        if acquire_lease:
            try:
                res = client.acquire(
                    "gravitywell",
                    work_id,
                    ttl_sec=timeout + 60,
                    reason="gw_agent",
                    timeout=_gw_acquire_timeout(),
                    principal=principal,
                )
            except DoormanUnreachable as e:
                if log:
                    log(f"[gw_agent] doorman unreachable: {e}")
                if on_wake_fail == "skip":
                    if writeable:
                        return (_build_fixer_result(cwd, transcript, concluded=False), transcript)
                    if reason_out is not None:
                        reason_out.append("gw_unreachable")
                    return (None, transcript) if return_transcript else None
                elif on_wake_fail == "error":
                    raise
                elif on_wake_fail == "claude":
                    return _fallback_claude_cli(
                        prompt, system, cwd, json_mode, log, return_transcript, transcript
                    )
                else:
                    raise ValueError(f"unknown on_wake_fail: {on_wake_fail}")

            if res.get("status") != "serving":
                if log:
                    log(f"[gw_agent] GW not serving: {res.get('status')}")
                if on_wake_fail == "skip":
                    if writeable:
                        return (_build_fixer_result(cwd, transcript, concluded=False), transcript)
                    if reason_out is not None:
                        reason_out.append("gw_not_serving")
                    return (None, transcript) if return_transcript else None
                elif on_wake_fail == "error":
                    raise Exception(f"GW not serving: {res.get('status')}")
                elif on_wake_fail == "claude":
                    return _fallback_claude_cli(
                        prompt, system, cwd, json_mode, log, return_transcript, transcript
                    )
                else:
                    raise ValueError(f"unknown on_wake_fail: {on_wake_fail}")

        # Loop: request → tool execution → result → request → ...
        # Wall-clock deadline tracking for budget-forced conclusion.
        _loop_start = time.monotonic()
        _deadline = _loop_start + timeout
        _avg_step_s = 18.0  # seed before any step completes (typical 122B latency)
        _step_times: list[float] = []
        _step_start: float | None = None

        for step_num in range(max_steps):
            # Update rolling avg using the wall-clock of the just-completed step (if any).
            _now = time.monotonic()
            if _step_start is not None:
                _step_times.append(_now - _step_start)
                _avg_step_s = sum(_step_times) / len(_step_times)
            _step_start = _now

            # Pre-step budget check: stop exploring if too close to the deadline to
            # fit another step AND still have time for a forced-conclusion call.
            _conclusion_reserve_s = max(2.0 * _avg_step_s, 0.20 * timeout)
            if _deadline - _now <= _conclusion_reserve_s:
                if log:
                    log(
                        f"[gw_agent] budget deadline approaching at step {step_num + 1}: "
                        f"{_deadline - _now:.1f}s remaining, reserve={_conclusion_reserve_s:.1f}s — "
                        "forcing conclusion"
                    )
                _elapsed = _now - _loop_start
                _budget_suffix = (
                    f"[gw_agent: budget-forced conclusion at step {step_num + 1}/"
                    f"elapsed {_elapsed:.0f}s]"
                )
                _fc_timeout = max(20.0, _deadline - _now)
                _forced_content = _force_conclusion(
                    messages, backend_url, timeout, json_mode, log, _is_swarm,
                    call_timeout=_fc_timeout, partial=True,
                    served_model_out=served_model_out,
                    model=model,
                )
                if _forced_content:
                    return _finalize_writeable_or_readonly(
                        messages, _forced_content, return_transcript, transcript,
                        writeable, cwd, concluded=False,
                        budget_forced=True,
                        budget_forced_suffix=_budget_suffix,
                        reason_out=reason_out,
                    )
                return _finalize_writeable_or_readonly(
                    messages, "", return_transcript, transcript,
                    writeable, cwd, concluded=False,
                    budget_forced=True,
                    budget_forced_suffix=_budget_suffix,
                    reason_out=reason_out,
                )

            if log:
                log(f"[gw_agent] step {step_num + 1}/{max_steps}")
            step_made_progress = False

            # Step-top cancel check (fail-safe: raising halts the loop)
            if cancel_check is not None:
                try:
                    if cancel_check():
                        _interrupted = True
                        _interrupt_reason = "user_cancel"
                        _interrupted_step = step_num + 1
                except Exception as _cc_exc:
                    logger.error(f"[gw_agent] cancel_check raised at step top: {_cc_exc}")
                    _interrupted = True
                    _interrupt_reason = "cancel_check_failed"
                    _interrupted_step = step_num + 1
                if _interrupted:
                    break

            # Per-step timeout: leave headroom for the forced-conclusion model call.
            # Never cap below 20s (a legitimate slow step on a loaded 122B can take minutes).
            _per_step_timeout = max(20.0, _deadline - _now - _conclusion_reserve_s)

            # POST to the backend (GW or swarm) with current message state.
            try:
                resp = requests.post(
                    f"{backend_url}/v1/chat/completions",
                    json={
                        **({} if model is None else {"model": model}),
                        "messages": messages,
                        "tools": list(tools.values()),
                        "tool_choice": "auto",
                        "temperature": 0.7,
                        **({} if _is_swarm else {"chat_template_kwargs": {"enable_thinking": think}}),
                    },
                    timeout=_per_step_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if served_model_out is not None and "model" in data and data["model"] is not None:
                    served_model_out.append(data["model"])
            except Exception as e:
                if log:
                    log(f"[gw_agent] GW request failed: {e}")
                # Return best-effort content accumulated so far
                return _finalize_writeable_or_readonly(
                    messages, "", return_transcript, transcript, writeable, cwd, concluded=False,
                    reason_out=reason_out, reason="request_failed",
                )

            # Extract response.
            if "choices" not in data or not data["choices"]:
                if log:
                    log(f"[gw_agent] GW returned no choices")
                return _finalize_writeable_or_readonly(
                    messages, "", return_transcript, transcript, writeable, cwd, concluded=False,
                    reason_out=reason_out, reason="no_choices",
                )

            choice = data["choices"][0]
            assistant_message = choice.get("message", {})
            content = assistant_message.get("content") or ""
            tool_calls_list = assistant_message.get("tool_calls") or []
            finish_reason = choice.get("finish_reason", "")

            # Update context token count.
            if "usage" in data:
                ctx_tokens = data["usage"].get("total_tokens", ctx_tokens)
            else:
                ctx_tokens = len(json.dumps(messages)) // 4

            # Append assistant message (with content + tool_calls reference).
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_list})

            # Precedence: tool_calls > content.
            if tool_calls_list:
                for tool_call in tool_calls_list:
                    tool_call_id = tool_call.get("id", f"call_{step_num}_{len(transcript)}")
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

                    # Parse tool arguments.
                    try:
                        if isinstance(tool_args_str, str):
                            tool_args = json.loads(tool_args_str)
                        else:
                            tool_args = tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Check for repeated calls (no-progress detection).
                    call_sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    repeated_calls[call_sig] = repeated_calls.get(call_sig, 0) + 1

                    if repeated_calls[call_sig] == 3:
                        # Nudge once.
                        if log:
                            log(f"[gw_agent] repeated call detected (3x): {tool_name}")
                        nudge_msg = f"You already ran '{tool_name}' with those arguments. Conclude or try something else."
                        messages.append({"role": "user", "content": nudge_msg})
                    elif repeated_calls[call_sig] >= 4:
                        # Break after 4th repeat (after nudge); try forced conclusion.
                        if log:
                            log(f"[gw_agent] breaking due to repeated call (4x): {tool_name}")
                        forced_content = _force_conclusion(
                            messages, backend_url, timeout, json_mode, log, _is_swarm,
                            served_model_out=served_model_out,
                            model=model,
                        )
                        if forced_content:
                            return _finalize_writeable_or_readonly(
                                messages, forced_content, return_transcript, transcript,
                                writeable, cwd, concluded=True,
                                reason_out=reason_out,
                            )
                        # Forced conclusion failed; fall back to exhaustion marker.
                        return _finalize_writeable_or_readonly(
                            messages, content, return_transcript, transcript,
                            writeable, cwd, concluded=False, max_steps_reached=True,
                            reason_out=reason_out,
                        )

                    # Pre-tool cancel check (fail-safe: raising halts the loop)
                    if cancel_check is not None:
                        try:
                            if cancel_check():
                                _interrupted = True
                                _interrupt_reason = "user_cancel"
                                _interrupted_step = step_num + 1
                        except Exception as _cc_exc:
                            logger.error(f"[gw_agent] cancel_check raised pre-tool: {_cc_exc}")
                            _interrupted = True
                            _interrupt_reason = "cancel_check_failed"
                            _interrupted_step = step_num + 1
                    if _interrupted:
                        break

                    # Before-tool gate (fail-closed: raising rejects this tool, loop continues)
                    _gate_override: dict | None = None
                    if before_tool is not None:
                        try:
                            _gate = before_tool(tool_name, tool_args)
                            _decision = _gate.get("decision", "proceed") if isinstance(_gate, dict) else "proceed"
                            if _decision == "stop":
                                _interrupted = True
                                _interrupt_reason = "user_cancel"
                                _interrupted_step = step_num + 1
                            elif _decision == "reject":
                                _reason_text = _gate.get("reason", "gate rejected") if isinstance(_gate, dict) else "gate rejected"
                                _gate_override = {"error": f"rejected: {_reason_text}"}
                        except Exception as _bt_exc:
                            logger.warning(f"[gw_agent] before_tool raised: {_bt_exc}")
                            _gate_override = {"error": f"gate_failure: {_bt_exc}"}
                    if _interrupted:
                        break

                    # Execute tool (or use gate override for reject/gate_failure)
                    if _gate_override is not None:
                        tool_result = _gate_override
                    elif tool_name in tool_executors:
                        try:
                            tool_result = tool_executors[tool_name].execute(tool_args)
                        except Exception as e:
                            tool_result = {"error": f"tool execution exception: {e}"}
                    else:
                        tool_result = {"error": f"unknown tool: {tool_name}"}

                    # Convert result to string.
                    if isinstance(tool_result, dict):
                        result_str = json.dumps(tool_result)
                    else:
                        result_str = str(tool_result)

                    # Record transcript.
                    # error field: None if tool succeeded, error message string if it failed
                    error_value = None
                    if isinstance(tool_result, dict) and "error" in tool_result:
                        error_value = tool_result["error"]

                    transcript.append(
                        {
                            "step": step_num + 1,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "arguments": tool_args,
                            "result": result_str,
                            "error": error_value,
                        }
                    )

                    # Track grounding: count error-free tool calls (verified, not merely attempted).
                    if error_value is None:
                        grounding_count += 1

                    # Append tool result message.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result_str,
                        }
                    )

                    # Semantic-progress tracking for no-progress guard (writeable only).
                    if writeable and no_progress_steps > 0:
                        if tool_name in ("apply_edit", "write_file"):
                            if not (isinstance(tool_result, dict) and "error" in tool_result):
                                step_made_progress = True
                        elif tool_name == "run_tests":
                            if isinstance(tool_result, dict) and "error" not in tool_result:
                                tc = (
                                    int(tool_result.get("passed") or 0),
                                    int(tool_result.get("failed") or 0),
                                    int(tool_result.get("errors") or 0),
                                )
                                if tc != last_test_counts:
                                    step_made_progress = True
                                    last_test_counts = tc

                if _interrupted:
                    break

                # No-progress guard: abort if K consecutive steps made no semantic progress.
                if writeable and no_progress_steps > 0:
                    if step_made_progress:
                        consecutive_no_progress = 0
                    else:
                        consecutive_no_progress += 1
                        if consecutive_no_progress >= no_progress_steps:
                            if log:
                                log(
                                    f"[gw_agent] no-progress guard: {consecutive_no_progress} "
                                    f"consecutive steps with no semantic progress - aborting"
                                )
                            return _finalize_writeable_or_readonly(
                                messages, "", return_transcript, transcript, writeable, cwd,
                                concluded=False, no_progress=True,
                            )

                # Context-growth guard: truncate oldest tool-result messages if needed.
                if ctx_tokens > GW_AGENT_CTX_CAP:
                    if log:
                        log(f"[gw_agent] context cap exceeded ({ctx_tokens} > {GW_AGENT_CTX_CAP}); truncating")
                    messages = _truncate_messages(messages)

            elif finish_reason == "stop" or finish_reason not in ("tool_calls", "stop"):
                # Agent concluded voluntarily (finish_reason == "stop", or unknown treated as stop).
                if finish_reason != "stop" and log:
                    log(
                        f"[gw_agent] agent ended with finish_reason={finish_reason} "
                        f"(expected stop or tool_calls)"
                    )
                if log and finish_reason == "stop":
                    log(f"[gw_agent] agent concluded at step {step_num + 1}")

                # §1c: Grounding guard — json_mode review runs only, not writeable fixer runs.
                if json_mode and not writeable and grounding_count == 0:
                    if not grounding_nudged:
                        # First ungrounded stop: nudge and continue the loop.
                        grounding_nudged = True
                        if log:
                            log("[gw_agent] grounding guard: 0 verified tool calls — nudging")
                        messages.append({
                            "role": "user",
                            "content": (
                                "You concluded without investigating. A verdict with no successful "
                                "tool call is not acceptable — use the available tools to read the "
                                "spec target and the relevant code, THEN produce your verdict."
                            ),
                        })
                        continue
                    else:
                        # Second ungrounded stop: UNFOUNDED — do not accept as a verdict.
                        if log:
                            log("[gw_agent] grounding guard: second ungrounded stop — UNFOUNDED")
                        return _finalize_writeable_or_readonly(
                            messages, "", return_transcript, transcript, writeable, cwd,
                            concluded=False,
                            reason_out=reason_out, reason="grounding_failed",
                        )

                # §1b: Validate JSON on voluntary stop for json_mode runs.
                if json_mode and not writeable:
                    _stripped = re.sub(
                        r"^```(?:json)?\s*\n?(.+?)\n?```$", r"\1", content.strip(), flags=re.DOTALL
                    )
                    try:
                        json.loads(_stripped)
                        # Already valid JSON — finalize directly, no extra turn.
                        return _finalize_writeable_or_readonly(
                            messages, _stripped, return_transcript, transcript, writeable, cwd,
                            concluded=True,
                            reason_out=reason_out,
                        )
                    except (json.JSONDecodeError, ValueError):
                        # Not valid JSON — re-emit under grammar constraint.
                        if log:
                            log("[gw_agent] voluntary stop: content not parseable JSON — re-emitting")
                        _re_emitted = _force_conclusion(
                            messages, backend_url, timeout, json_mode, log, _is_swarm,
                            verdict_schema=verdict_schema,
                            served_model_out=served_model_out,
                            model=model,
                            reason=(
                                "You stopped without emitting a valid JSON verdict. "
                                "Based only on what you have already gathered, produce "
                                "your final answer now as valid JSON only."
                            ),
                        )
                        return _finalize_writeable_or_readonly(
                            messages, _re_emitted if _re_emitted else content,
                            return_transcript, transcript, writeable, cwd, concluded=True,
                            reason_out=reason_out, reason="no_choices",
                        )

                # Non-json_mode or writeable: byte-identical to previous behavior.
                # reason="no_choices" reuses the closest existing category for a voluntary
                # stop whose content came back empty (mirrors the json_mode re-emit fallback
                # above, which reuses the same category for its analogous empty-content case).
                return _finalize_writeable_or_readonly(
                    messages, content, return_transcript, transcript, writeable, cwd, concluded=True,
                    reason_out=reason_out, reason="no_choices",
                )

        # Interrupted: cancel_check or before_tool stop halted the loop.
        if _interrupted:
            if log:
                log(f"[gw_agent] interrupted at step {_interrupted_step} reason={_interrupt_reason}")
            return _finalize_writeable_or_readonly(
                messages, "", return_transcript, transcript, writeable, cwd, concluded=False,
                interrupted=True, interrupt_reason=_interrupt_reason,
                interrupted_step=_interrupted_step,
                reason_out=reason_out,
            )

        # Exhausted max_steps without conclusion; try forced conclusion.
        if log:
            log(f"[gw_agent] max_steps ({max_steps}) reached without conclusion")
        # Get the last actual content before calling _force_conclusion (which mutates messages)
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_content = msg.get("content", "")
                break
        forced_content = _force_conclusion(
            messages, backend_url, timeout, json_mode, log, _is_swarm,
            served_model_out=served_model_out,
            model=model,
        )
        if forced_content:
            return _finalize_writeable_or_readonly(
                messages, forced_content, return_transcript, transcript, writeable, cwd, concluded=False,
                reason_out=reason_out,
            )
        # Forced conclusion failed; fall back to exhaustion marker.
        return _finalize_writeable_or_readonly(
            messages, last_content, return_transcript, transcript,
            writeable, cwd, concluded=False, max_steps_reached=True,
            reason_out=reason_out,
        )

    finally:
        if acquire_lease:
            try:
                client.release("gravitywell", work_id)
            except Exception as e:
                if log:
                    log(f"[gw_agent] failed to release lease: {e}")
        client.close()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _default_reviewer_system_prompt() -> str:
    """Default system prompt for a review agent.

    Instructs the agent to credit human authorship when the evidence carries it,
    never frames itself as author of the code or verdict, and does not invent
    attribution when metadata is absent.
    """
    return """You are a code review agent. Your task is to analyze code, commits, and
related artifacts to provide thoughtful feedback.

When you discover authorship evidence in the gathered metadata (git blame, commit logs,
file headers), credit the human authors explicitly. Frame your findings as observations
of human work, not as your own creation.

Never frame yourself as the author of the code under review or the sole author of the
verdict. You are an instrument reporting on human work and human authorship decisions.

When authorship metadata is absent, state your findings plainly without inventing or
hallucinating attribution. Prefer neutral or attributed phrasing over possessive language
that erases authorship.

You have access to read-only tools (read_file, grep, git, mem) to gather evidence."""


def _truncate_messages(messages: list[dict]) -> list[dict]:
    """Truncate oldest tool-result messages to free context space.

    Keeps: system (if present), user, and the most recent 3-4 turns of assistant+tool pairs.
    """
    result = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            result.append(msg)
        elif msg.get("role") == "user":
            result.append(msg)

    assistant_blocks = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant":
            block = [msg]
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                block.append(messages[i])
                i += 1
            assistant_blocks.append(block)
        else:
            i += 1

    if len(assistant_blocks) > 4:
        assistant_blocks = assistant_blocks[-4:]

    for block in assistant_blocks:
        result.extend(block)

    return result


def _force_conclusion(
    messages: list[dict],
    backend_url: str,
    timeout: int,
    json_mode: bool,
    log: Callable[[str], None] | None,
    is_swarm: bool = False,
    call_timeout: int | float | None = None,
    partial: bool = False,
    verdict_schema: dict | None = None,
    reason: str | None = None,
    served_model_out: list | None = None,
    model: str | None = None,
) -> str:
    """Emit a forced conclusion when the agent exhausts its tool budget.

    Makes one final inference call with tools disabled, forcing the model to conclude
    based on accumulated evidence. Returns the model's content or empty string on failure.

    Args:
        call_timeout: Actual seconds to allow for this one model call. When budget-forced,
                      pass the remaining wall-clock budget here so the conclusion call gets
                      real time to complete. Defaults to `timeout` (full budget) for the
                      existing max_steps and repeated-call paths.
        partial: When True, instructs the model to acknowledge its incomplete investigation
                 in the verdict text — required for the budget-forced path so a partial
                 review is not presented as complete.
        verdict_schema: OpenAI json_schema object for grammar-constrained JSON emission.
                        When provided (GW path only), sets response_format to json_schema.
        reason: Truthful framing for the re-emission prompt. When provided, replaces the
                default "reached your investigation budget" opening so a voluntary-stop
                re-emission does not lie about why the model is being asked to conclude.
        served_model_out: Optional list, passed through from the caller's own
                           `served_model_out` (same append-only contract). When provided, the
                           top-level "model" field echoed by this forced-conclusion response is
                           appended to it if present.
        model: Optional model name, passed through from the caller's own `model` param.
               Included as the "model" field in the POST payload when provided; omitted
               when None (default), matching the main loop's behavior.

    Validates that the response is not a leaked tool-call (content-integrity check).
    Does NOT raise exceptions or add to transcript.
    """
    post_timeout = call_timeout if call_timeout is not None else timeout

    # Clean the conversation tail: clear unmatched tool_calls from the trailing
    # assistant message to ensure the conversation ends on a clean boundary
    # (required for OpenAI-compatible backends to accept the following user turn).
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            last_msg["tool_calls"] = []

    # Append the conclusion instruction with explicit negative constraints
    # forbidding tool use.
    if reason is not None:
        # Truthful framing for voluntary-stop and grounding-guard re-emissions.
        conclusion_instruction = (
            f"{reason} You may NOT call any tools, and you MUST NOT emit a tool call."
        )
    else:
        conclusion_instruction = (
            "You have reached your investigation budget. You may NOT call any tools, "
            "and you MUST NOT emit a tool call. Based only on what you have already gathered, "
            "produce your final answer now as plain content."
        )
    if partial:
        if json_mode:
            # json_mode requires no leading prose; embed the caveat as a JSON field instead
            # so the Truth-Integrity requirement is met without conflicting with the
            # "JSON only" instruction that follows.
            conclusion_instruction += (
                " IMPORTANT: This is a PARTIAL review — you ran out of time before completing"
                " your investigation. Add a \"partial_review_note\" field to your JSON verdict"
                " that briefly states this is a partial review, approximately how many steps"
                " you completed, and what areas you could not examine. Do not omit this field"
                " and do not present an incomplete review as if it were complete."
            )
        else:
            conclusion_instruction += (
                " IMPORTANT: This is a PARTIAL review — you ran out of time before completing"
                " your investigation. You MUST begin your verdict with a brief caveat stating"
                " that this is a partial review, approximately how many steps you completed,"
                " and what areas you could not examine. Do not present an incomplete review"
                " as if it were complete."
            )
    if json_mode:
        conclusion_instruction += " Respond with the required JSON verdict only — no prose, no tool calls."

    messages.append({"role": "user", "content": conclusion_instruction})

    # §1a: Build response_format for grammar-constrained emission (GW path only; swarm excluded).
    _response_format: dict | None = None
    if not is_swarm:
        if verdict_schema is not None:
            _response_format = {"type": "json_schema", "json_schema": verdict_schema}
        elif json_mode:
            _response_format = {"type": "json_object"}

    # Make the final POST with tools strictly omitted (not tool_choice: "none").
    try:
        resp = requests.post(
            f"{backend_url}/v1/chat/completions",
            json={
                **({} if model is None else {"model": model}),
                "messages": messages,
                "temperature": 0.3,
                **({} if is_swarm else {"chat_template_kwargs": {"enable_thinking": False}}),
                **({} if _response_format is None else {"response_format": _response_format}),
            },
            timeout=post_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if served_model_out is not None and "model" in data and data["model"] is not None:
            served_model_out.append(data["model"])
    except Exception as e:
        if log:
            log(f"[gw_agent] forced conclusion POST failed: {e}")
        return ""

    # Extract and validate the response.
    if "choices" not in data or not data["choices"]:
        if log:
            log(f"[gw_agent] forced conclusion returned no choices")
        return ""

    choice = data["choices"][0]
    assistant_message = choice.get("message", {})
    content = assistant_message.get("content") or ""
    tool_calls_leaked = assistant_message.get("tool_calls") or []

    # Reject if the response contains leaked tool calls (re-entered tool loop).
    if tool_calls_leaked:
        if log:
            log(f"[gw_agent] forced conclusion response leaked tool_calls; rejecting")
        return ""

    return content


def _finalize_writeable_or_readonly(
    messages: list[dict],
    content: str,
    return_transcript: bool,
    transcript: list[dict],
    writeable: bool,
    cwd: str,
    concluded: bool,
    max_steps_reached: bool = False,
    no_progress: bool = False,
    budget_forced: bool = False,
    budget_forced_suffix: str = "",
    interrupted: bool = False,
    interrupt_reason: str = "",
    interrupted_step: int = 0,
    reason_out: list[str] | None = None,
    reason: str | None = None,
) -> str | None | tuple:
    """Route to FixerResult or plain result based on writeable flag.

    reason_out/reason are only consulted on the readonly (writeable=False) leg - a
    writeable=True call always returns a FixerResult here and never touches reason_out.
    """
    if writeable:
        fixer = _build_fixer_result(
            cwd, transcript,
            concluded=concluded and not max_steps_reached and not no_progress and not budget_forced and not interrupted,
            max_steps_reached=max_steps_reached,
            no_progress=no_progress,
            budget_forced=budget_forced,
            interrupted=interrupted,
            interrupt_reason=interrupt_reason,
        )
        return (fixer, transcript)
    return _finalize_result(
        messages, content, return_transcript, transcript, max_steps_reached, budget_forced_suffix,
        interrupted=interrupted, interrupt_reason=interrupt_reason, interrupted_step=interrupted_step,
        reason_out=reason_out, reason=reason,
    )


def _finalize_result(
    messages: list[dict],
    content: str,
    return_transcript: bool,
    transcript: list[dict],
    max_steps_reached: bool = False,
    budget_forced_suffix: str = "",
    interrupted: bool = False,
    interrupt_reason: str = "",
    interrupted_step: int = 0,
    reason_out: list[str] | None = None,
    reason: str | None = None,
) -> str | None | tuple[str | None, list[dict]]:
    """Finalize the return value with optional max_steps or budget-forced marker.

    reason_out (when not None) receives one category string if the underlying `content`
    passed in was empty - gated on that pre-marker content, not on the marker-synthesized
    `text` computed below, since interrupted/max_steps_reached/budget_forced_suffix all
    synthesize non-empty marker text even when the underlying result was empty.
    """
    content_was_empty = not content
    text = content or ""
    if interrupted:
        marker = f"[gw_agent: interrupted at step {interrupted_step} - reason: {interrupt_reason}]"
        text = (text + f"\n\n{marker}") if text else marker
    if max_steps_reached and text:
        text = text + "\n\n[gw_agent: max_steps reached — verdict may be incomplete]"
    elif max_steps_reached:
        text = "[gw_agent: max_steps reached — no verdict reached]"
    if budget_forced_suffix and text:
        # If content is valid JSON (json_mode=True case), inject as a field so
        # json.loads() by callers (e.g. spec_review.py:1671) still succeeds.
        # Appending a text suffix to JSON causes JSONDecodeError → false error verdict.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed["_budget_forced"] = budget_forced_suffix
                text = json.dumps(parsed)
            else:
                text = text + f"\n\n{budget_forced_suffix}"
        except (json.JSONDecodeError, ValueError):
            text = text + f"\n\n{budget_forced_suffix}"
    elif budget_forced_suffix:
        text = budget_forced_suffix

    if reason_out is not None and content_was_empty:
        if interrupted:
            reason_out.append("interrupted")
        elif max_steps_reached:
            reason_out.append("max_steps_exhausted")
        elif budget_forced_suffix:
            reason_out.append("budget_exhausted")
        elif reason:
            reason_out.append(reason)

    if return_transcript:
        return (text if text else None, transcript)
    else:
        return text if text else None


def _fallback_claude_cli(
    prompt: str,
    system: str,
    cwd: str,
    json_mode: bool,
    log: Callable[[str], None] | None,
    return_transcript: bool,
    transcript: list[dict],
) -> str | None | tuple[str | None, list[dict]]:
    """Fallback to call_claude_cli when doorman cannot wake GW.

    Logged as paid spend per decision/claude-p-api-pricing-june11.
    """
    if log:
        log(
            "[gw_agent] falling back to call_claude_cli(sonnet) "
            "(paid spend per claude-p-api-pricing-june11)"
        )

    from agents_core.llm import call_claude_cli

    text = call_claude_cli(
        prompt,
        system=system,
        model="sonnet",
        cwd=cwd,
        json_mode=json_mode,
        log=log,
    )

    if return_transcript:
        return (text, transcript)
    else:
        return text
