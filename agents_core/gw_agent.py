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

    ALLOWLIST = {"log", "show", "diff", "status", "blame", "ls-files", "rev-list", "cat-file", "describe", "shortlog"}

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
    tool_executors = _get_tool_executors(cwd, writeable=writeable)
    repeated_calls: dict[str, int] = {}
    ctx_tokens = 0
    # No-progress guard state (writeable mode): track consecutive steps with no semantic progress.
    consecutive_no_progress = 0
    last_test_counts: tuple | None = None

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
                )
            except DoormanUnreachable as e:
                if log:
                    log(f"[gw_agent] doorman unreachable: {e}")
                if on_wake_fail == "skip":
                    if writeable:
                        return (_build_fixer_result(cwd, transcript, concluded=False), transcript)
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
        for step_num in range(max_steps):
            if log:
                log(f"[gw_agent] step {step_num + 1}/{max_steps}")
            step_made_progress = False

            # POST to the backend (GW or swarm) with current message state.
            try:
                resp = requests.post(
                    f"{backend_url}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "tools": list(tools.values()),
                        "tool_choice": "auto",
                        "temperature": 0.7,
                        **({} if _is_swarm else {"chat_template_kwargs": {"enable_thinking": think}}),
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                if log:
                    log(f"[gw_agent] GW request failed: {e}")
                # Return best-effort content accumulated so far
                return _finalize_writeable_or_readonly(
                    messages, "", return_transcript, transcript, writeable, cwd, concluded=False
                )

            # Extract response.
            if "choices" not in data or not data["choices"]:
                if log:
                    log(f"[gw_agent] GW returned no choices")
                return _finalize_writeable_or_readonly(
                    messages, "", return_transcript, transcript, writeable, cwd, concluded=False
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
                            messages, backend_url, timeout, json_mode, log, _is_swarm
                        )
                        if forced_content:
                            return _finalize_writeable_or_readonly(
                                messages, forced_content, return_transcript, transcript,
                                writeable, cwd, concluded=True,
                            )
                        # Forced conclusion failed; fall back to exhaustion marker.
                        return _finalize_writeable_or_readonly(
                            messages, content, return_transcript, transcript,
                            writeable, cwd, concluded=False, max_steps_reached=True,
                        )

                    # Execute tool.
                    if tool_name in tool_executors:
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
                                    f"consecutive steps with no semantic progress — aborting"
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

            elif finish_reason == "stop":
                # Agent concluded (no tool_calls, just content).
                if log:
                    log(f"[gw_agent] agent concluded at step {step_num + 1}")
                return _finalize_writeable_or_readonly(
                    messages, content, return_transcript, transcript, writeable, cwd, concluded=True
                )
            else:
                # finish_reason is neither tool_calls nor stop; treat as stop.
                if log:
                    log(
                        f"[gw_agent] agent ended with finish_reason={finish_reason} "
                        f"(expected stop or tool_calls)"
                    )
                return _finalize_writeable_or_readonly(
                    messages, content, return_transcript, transcript, writeable, cwd, concluded=True
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
        forced_content = _force_conclusion(messages, backend_url, timeout, json_mode, log, _is_swarm)
        if forced_content:
            return _finalize_writeable_or_readonly(
                messages, forced_content, return_transcript, transcript, writeable, cwd, concluded=False
            )
        # Forced conclusion failed; fall back to exhaustion marker.
        return _finalize_writeable_or_readonly(
            messages, last_content, return_transcript, transcript,
            writeable, cwd, concluded=False, max_steps_reached=True,
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
) -> str:
    """Emit a forced conclusion when the agent exhausts its tool budget.

    Makes one final inference call with tools disabled, forcing the model to conclude
    based on accumulated evidence. Returns the model's content or empty string on failure.

    Validates that the response is not a leaked tool-call (content-integrity check).
    Does NOT raise exceptions or add to transcript.
    """
    # Clean the conversation tail: clear unmatched tool_calls from the trailing
    # assistant message to ensure the conversation ends on a clean boundary
    # (required for OpenAI-compatible backends to accept the following user turn).
    if messages:
        last_msg = messages[-1]
        if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            last_msg["tool_calls"] = []

    # Append the conclusion instruction with explicit negative constraints
    # forbidding tool use.
    conclusion_instruction = (
        "You have reached your investigation budget. You may NOT call any tools, "
        "and you MUST NOT emit a tool call. Based only on what you have already gathered, "
        "produce your final answer now as plain content."
    )
    if json_mode:
        conclusion_instruction += " Respond with the required JSON verdict only — no prose, no tool calls."

    messages.append({"role": "user", "content": conclusion_instruction})

    # Make the final POST with tools strictly omitted (not tool_choice: "none").
    try:
        resp = requests.post(
            f"{backend_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0.3,
                **({} if is_swarm else {"chat_template_kwargs": {"enable_thinking": False}}),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
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
) -> str | None | tuple:
    """Route to FixerResult or plain result based on writeable flag."""
    if writeable:
        fixer = _build_fixer_result(
            cwd, transcript,
            concluded=concluded and not max_steps_reached and not no_progress,
            max_steps_reached=max_steps_reached,
            no_progress=no_progress,
        )
        return (fixer, transcript)
    return _finalize_result(messages, content, return_transcript, transcript, max_steps_reached)


def _finalize_result(
    messages: list[dict],
    content: str,
    return_transcript: bool,
    transcript: list[dict],
    max_steps_reached: bool = False,
) -> str | None | tuple[str | None, list[dict]]:
    """Finalize the return value with optional max_steps marker."""
    text = content or ""
    if max_steps_reached and text:
        text = text + "\n\n[gw_agent: max_steps reached — verdict may be incomplete]"
    elif max_steps_reached:
        text = "[gw_agent: max_steps reached — no verdict reached]"

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
