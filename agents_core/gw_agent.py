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
import uuid
from pathlib import Path
from typing import Any, Callable

import requests

from agents_core.doorman_client import DoormanClient, DoormanUnreachable

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")
GW_AGENT_TOOL_OUTPUT_CAP = 8192
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
}


def _get_tool_executors(cwd: str | None = None) -> dict[str, ToolExecutor]:
    """Instantiate all default tool executors with a given cwd."""
    return {
        "read_file": ReadFileExecutor(cwd),
        "grep": GrepExecutor(cwd),
        "git": GitExecutor(cwd),
        "mem": MemExecutor(cwd),
    }


# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------


def call_gw_agent(
    prompt: str,
    system: str = "",
    cwd: str | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
    max_steps: int = 12,
    timeout: int = 300,
    json_mode: bool = False,
    think: bool = False,
    on_wake_fail: str = "skip",
    work_id: str | None = None,
    return_transcript: bool = False,
    log: Callable[[str], None] | None = None,
) -> str | None | tuple[str | None, list[dict]]:
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
        max_steps: Max number of tool-call iterations (default 12).
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

    Returns:
        - str or None (or (str|None, list) when return_transcript=True).
        - None means "did not run" (only on on_wake_fail="skip" + doorman failure).
        - Non-None with "[gw_agent: max_steps reached ...]" suffix means loop exhausted.
        - Transcript (if return_transcript) is a list of dicts with tool execution details.

    The doorman lease is acquired once and held for the entire run; released in finally.
    Tool errors are recovered gracefully: a malformed call returns a tool-error message
    so GW can adapt (the loop never crashes on tool execution).
    """
    if system == "":
        system = _default_reviewer_system_prompt()

    if json_mode:
        system = system + "\n\nYour FINAL answer must be valid JSON, no markdown fences."

    if cwd is None:
        cwd = "/srv/agents"

    if work_id is None:
        work_id = uuid.uuid4().hex[:8]

    if tools is None:
        tools = DEFAULT_READONLY_TOOLS

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    transcript: list[dict] = []
    tool_executors = _get_tool_executors(cwd)
    repeated_calls: dict[str, int] = {}
    ctx_tokens = 0

    # Acquire doorman lease for the whole run.
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    client = DoormanClient()
    try:
        try:
            res = client.acquire(
                "gravitywell",
                work_id,
                ttl_sec=timeout + 60,
                reason="gw_agent",
            )
        except DoormanUnreachable as e:
            if log:
                log(f"[gw_agent] doorman unreachable: {e}")
            if on_wake_fail == "skip":
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

            # POST to GW with current message state.
            try:
                resp = requests.post(
                    f"{GW_URL}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "tools": list(tools.values()),
                        "tool_choice": "auto",
                        "temperature": 0.7,
                        "chat_template_kwargs": {"enable_thinking": think},
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                if log:
                    log(f"[gw_agent] GW request failed: {e}")
                # Return best-effort content accumulated so far
                return _finalize_result(messages, "", return_transcript, transcript)

            # Extract response.
            if "choices" not in data or not data["choices"]:
                if log:
                    log(f"[gw_agent] GW returned no choices")
                return _finalize_result(messages, "", return_transcript, transcript)

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
                        # Break after 4th repeat (after nudge).
                        if log:
                            log(f"[gw_agent] breaking due to repeated call (4x): {tool_name}")
                        return _finalize_result(
                            messages,
                            content,
                            return_transcript,
                            transcript,
                            max_steps_reached=True,
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

                # Context-growth guard: truncate oldest tool-result messages if needed.
                if ctx_tokens > GW_AGENT_CTX_CAP:
                    if log:
                        log(f"[gw_agent] context cap exceeded ({ctx_tokens} > {GW_AGENT_CTX_CAP}); truncating")
                    messages = _truncate_messages(messages)

            elif finish_reason == "stop":
                # Agent concluded (no tool_calls, just content).
                if log:
                    log(f"[gw_agent] agent concluded at step {step_num + 1}")
                return _finalize_result(messages, content, return_transcript, transcript)
            else:
                # finish_reason is neither tool_calls nor stop; treat as stop.
                if log:
                    log(
                        f"[gw_agent] agent ended with finish_reason={finish_reason} "
                        f"(expected stop or tool_calls)"
                    )
                return _finalize_result(messages, content, return_transcript, transcript)

        # Exhausted max_steps without conclusion.
        if log:
            log(f"[gw_agent] max_steps ({max_steps}) reached without conclusion")
        last_content = messages[-1].get("content", "") if messages else ""
        return _finalize_result(
            messages,
            last_content,
            return_transcript,
            transcript,
            max_steps_reached=True,
        )

    finally:
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
