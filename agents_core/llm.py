#!/usr/bin/env python3
"""Shared LLM client — talks to llama-server and Claude CLI (Max subscription).

Backends:
  - call_llm()         → local llama-server (qwen3.6-35b-a3b, GPU, free)
  - call_operator()    → multi-operator routing (qwen / sonnet / opus / haiku)
  - call_claude_cli()  → claude -p subprocess (Haiku/Sonnet, Max subscription)

All conductor/agent scripts should import from here.
"""

import json
import os
import re
import subprocess
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

TAILSCALE_IP = "203.0.113.12"
LLAMACPP_URL = f"http://{TAILSCALE_IP}:8081"
PACIFIC = ZoneInfo("America/Los_Angeles")

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")


class OperatorUnreachableError(Exception):
    """Raised when an operator backend cannot be reached after exhausting retries.

    Distinguishes "server down / network failure" from "model returned empty content."
    Carries the backend URL and the last underlying error for diagnostic context.
    Empty operator responses (content stripped to "") still return None from call_llm;
    this exception fires only when the HTTP layer itself fails repeatedly.
    """

    def __init__(self, url: str, last_error: Exception):
        self.url = url
        self.last_error = last_error
        super().__init__(f"Operator unreachable at {url} after retries: {last_error}")


# ---------------------------------------------------------------------------
# Multi-operator routing
# ---------------------------------------------------------------------------

OPERATOR_DEFAULTS: dict[str, str] = {
    "qwen":        "qwen3.6-35b-a3b",
    "sonnet":      "claude-sonnet-4-6",
    "opus":        "claude-opus-4-7",
    "haiku":       "claude-haiku-4-5-20251001",
    "gravitywell": "gravitywell-122b",
}


def _call_qwen_backend(prompt: str, system: str = None, timeout: int = 600,
                       json_mode: bool = False, temperature: float = 0.7,
                       log=None, bundle_ids: list[str] = None) -> str | None:
    """Send a completion request to the local llama-server (Qwen endpoint).

    Context selection priority:
    1. Explicit system= override (task-specific prompts)
    2. Explicit bundle_ids= (chub bundles by ID)
    3. Default: inertia-ecosystem bundle

    Returns the response text, or None on failure.
    """
    messages = []

    # System prompt: explicit > bundles > default
    if system is not None:
        sys_prompt = system
    elif bundle_ids is not None:
        from chub_broker import select_bundles_by_ids
        sel = select_bundles_by_ids(bundle_ids)
        sys_prompt = sel.composed
    else:
        from chub_broker import select_bundles_by_ids
        sel = select_bundles_by_ids(["conductor/inertia-ecosystem"])
        sys_prompt = sel.composed

    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "cache_prompt": True,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{LLAMACPP_URL}/v1/chat/completions",
                json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text if text.strip() else None
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                backoff = 10 * (2 ** attempt)
                if log:
                    log(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(backoff)
            else:
                if log:
                    log(f"LLM call failed after {max_retries} attempts: {e}")
                raise OperatorUnreachableError(LLAMACPP_URL, e)
        except Exception as e:
            if log:
                log(f"LLM call error: {e}")
            return None


def _call_gravitywell_backend(
    prompt: str,
    system: str = None,
    timeout: int = 600,
    json_mode: bool = False,
    temperature: float = 0.7,
    log=None,
    think: bool = False,
) -> str | None:
    """Send a completion request to the GravityWell llama.cpp endpoint.

    GW is a Qwen3.5-122B reasoning model. By default think=False injects
    chat_template_kwargs={"enable_thinking": false} to suppress the think-trace
    and keep responses clean (~2-4s). Callers may pass think=True for quality-mode
    reasoning with a large max_tokens.

    GW_URL coupling: reads the same GW_URL env var as doorman_server. Both must
    be kept in sync (see doorman-server.env and operator environment docs).

    Does NOT accept bundle_ids — GW gets system verbatim, no chub-bundle injection.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{GW_URL}/v1/chat/completions",
                json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text if text.strip() else None
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                backoff = 10 * (2 ** attempt)
                if log:
                    log(f"GW call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(backoff)
            else:
                if log:
                    log(f"GW call failed after {max_retries} attempts: {e}")
                raise OperatorUnreachableError(GW_URL, e)
        except Exception as e:
            if log:
                log(f"GW call error: {e}")
            return None


def _apply_wake_fail(
    on_wake_fail: str | None,
    operator_class: str,
    prompt: str,
    **kwargs,
) -> str | None:
    """Apply the declared on_wake_fail policy when GW cannot be woken.

    Policies:
      "haiku" / "sonnet" / "opus" — re-dispatch to that operator class (paid fallback;
          logs loudly before spending tokens per claude-p-api-pricing-june11).
      "skip" / None — return None (batch/optional surfaces; no paid spend).
      "error" — raise OperatorUnreachableError so the caller decides.

    If the fallback call_operator() itself raises, the exception propagates unchanged.
    """
    policy = on_wake_fail or "skip"

    if policy in ("haiku", "sonnet", "opus"):
        warnings.warn(
            f"[gravitywell] wake_failed — falling back to {policy} "
            f"(paid Claude spend per claude-p-api-pricing-june11). "
            f"operator_class={operator_class!r}",
            RuntimeWarning,
            stacklevel=3,
        )
        # Remove gravitywell-specific kwargs that the fallback operator doesn't accept
        fallback_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("think", "on_wake_fail")
        }
        return call_operator(policy, prompt, **fallback_kwargs)

    if policy == "error":
        raise OperatorUnreachableError(
            GW_URL,
            Exception(f"GW wake_failed and on_wake_fail='error' for {operator_class!r}"),
        )

    # "skip" or None
    return None


def call_operator(operator_class: str, prompt: str, model: str = None,
                  **kwargs) -> str | None:
    """Route a completion request to the appropriate backend operator.

    operator_class ∈ {"qwen", "sonnet", "opus", "haiku"}.
    Raises ValueError for unknown classes.

    Default models:
        qwen   → "qwen3.6-35b-a3b"
        sonnet → "claude-sonnet-4-6"
        opus   → "claude-opus-4-7"
        haiku  → "claude-haiku-4-5-20251001"

    qwen routes via the local llama-server (same path as call_llm()).

    sonnet / opus / haiku route via ClaudeQueue → `claude -p` (Max subscription
    path). The task is submitted to ClaudeQueue with task_type="llm_call";
    `submit_and_wait` blocks the caller's thread until the daemon's
    `_run_llm_call_task` handler completes the task and writes the output file.
    No direct Anthropic-API calls — kill-switched per
    decision/no-anthropic-api-direct.

    Return contract:
      - str on success (the model's response text, identical to call_llm()'s).
      - Raises RuntimeError if the queued task fails (subprocess error, missing
        output_path, queue.fail()'d).
      - Raises TimeoutError if the wall-clock budget elapses (default
        timeout + 30s queue overhead).
    """
    if operator_class not in OPERATOR_DEFAULTS:
        raise ValueError(
            f"Unknown operator_class {operator_class!r}. "
            f"Must be one of: {sorted(OPERATOR_DEFAULTS)}"
        )

    if operator_class == "qwen":
        if model is not None and model != OPERATOR_DEFAULTS["qwen"]:
            raise ValueError(
                f"call_operator(operator_class='qwen', model={model!r}): "
                "the local llama.cpp backend serves a single fixed model "
                f"({OPERATOR_DEFAULTS['qwen']!r}); model swaps are an "
                "infrastructure operation (stop / swap weights / restart), "
                "not a per-call parameter. Either pass model=None to use the "
                "default, or do the model swap out-of-band first."
            )
        return _call_qwen_backend(prompt=prompt, **kwargs)

    if operator_class == "gravitywell":
        if model is not None and model != OPERATOR_DEFAULTS["gravitywell"]:
            raise ValueError(
                f"call_operator(operator_class='gravitywell', model={model!r}): "
                "the GravityWell endpoint serves a single fixed model "
                f"({OPERATOR_DEFAULTS['gravitywell']!r}); model swaps are an "
                "infrastructure operation (gw-serve), "
                "not a per-call parameter. Either pass model=None to use the "
                "default, or do the model swap out-of-band first."
            )
        # Extract GW-specific kwargs; discard bundle_ids and other qwen-specific keys
        gw_kwargs = {
            k: kwargs[k] for k in (
                "system", "timeout", "json_mode", "temperature", "log"
            ) if k in kwargs
        }
        think = kwargs.get("think", False)
        on_wake_fail = kwargs.get("on_wake_fail", "skip")
        timeout = int(kwargs.get("timeout", 300))
        work_id = f"op-gravitywell-{uuid.uuid4().hex}"

        # kwargs forwarded to _apply_wake_fail must not include on_wake_fail
        # (it's a positional arg there) or gravitywell-internal keys.
        wake_fail_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("on_wake_fail", "think", "bundle_ids")
        }

        from agents_core.doorman_client import DoormanClient, DoormanUnreachable
        client = DoormanClient()
        try:
            res = client.acquire(
                "gravitywell", work_id, ttl_sec=timeout + 60, reason="call_operator"
            )
            if res.get("status") != "serving":
                return _apply_wake_fail(on_wake_fail, operator_class, prompt, **wake_fail_kwargs)
            try:
                return _call_gravitywell_backend(prompt=prompt, think=think, **gw_kwargs)
            finally:
                client.release("gravitywell", work_id)
        except (DoormanUnreachable, OperatorUnreachableError):
            return _apply_wake_fail(on_wake_fail, operator_class, prompt, **wake_fail_kwargs)

    # Anthropic-family: route via ClaudeQueue → call_claude_cli.
    # No direct Anthropic-API code path (decision/no-anthropic-api-direct).
    resolved_model = model or OPERATOR_DEFAULTS[operator_class]
    from agents_core.claude_queue_sync import submit_and_wait
    timeout = int(kwargs.get("timeout", 300))
    return submit_and_wait(
        {
            "task_type": "llm_call",
            "model": resolved_model,
            "submitted_by": "call_operator",
            "description": f"call_operator/{operator_class}",
            "timeout_seconds": timeout,
            "payload": {
                "operator_class": operator_class,
                "prompt": prompt,
                "system": kwargs.get("system", ""),
                "json_mode": kwargs.get("json_mode", False),
                "_ignore_intention_registry": True,
            },
        },
        timeout_s=timeout + 30,  # wrapper budget = task timeout + queue overhead
    )


def call_llm(prompt: str, system: str = None, timeout: int = 600,
             json_mode: bool = False, temperature: float = 0.7,
             log=None, bundle_ids: list[str] = None) -> str | None:
    """Send a completion request to llama-server via /v1/chat/completions.

    Drop-in replacement for ollama_client.call_ollama() and
    ollama_utils.call_ollama().

    Thin wrapper around call_operator(operator_class="qwen", ...).

    Context selection priority:
    1. Explicit system= override (task-specific prompts)
    2. Explicit bundle_ids= (chub bundles by ID)
    3. Default: inertia-ecosystem bundle

    Returns the response text, or None if the operator returned empty content.
    May raise OperatorUnreachableError if the operator backend is unreachable
    after retries.
    """
    return call_operator(
        operator_class="qwen",
        prompt=prompt,
        system=system,
        timeout=timeout,
        json_mode=json_mode,
        temperature=temperature,
        log=log,
        bundle_ids=bundle_ids,
    )


def call_llm_streaming(prompt: str, system: str = None, timeout: int = 600,
                       temperature: float = 0.7):
    """Streaming LLM call — yields content chunks as they arrive.

    Used by awake_core heartbeat mode for GPU preemption support.
    Caller is responsible for preemption logic (closing the response).

    Yields (chunk_text, done) tuples. done=True on the final chunk.
    Returns a context manager wrapping the streaming response.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "cache_prompt": True,
        "stream": True,
    }

    resp = requests.post(
        f"{LLAMACPP_URL}/v1/chat/completions",
        json=payload, timeout=timeout, stream=True)
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8") if isinstance(line, bytes) else line
        if not line_str.startswith("data: "):
            continue
        data_str = line_str[6:]
        if data_str.strip() == "[DONE]":
            yield "", True
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content") or ""
            finish = chunk["choices"][0].get("finish_reason") is not None
            yield content, finish
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


# ---------------------------------------------------------------------------
# Claude CLI client (Max subscription via `claude -p`)
# ---------------------------------------------------------------------------

def call_claude_cli(
    prompt: str,
    system: str = "",
    model: str = "haiku",
    timeout: int = 300,
    json_mode: bool = False,
    log=None,
    return_envelope: bool = False,
    cwd: str | None = None,
    permission_mode: str | None = None,
):
    """Call Claude via `claude -p` CLI (Max subscription).

    Uses the same subprocess pattern proven in claude_heartbeat.py.

    Args:
        prompt: User prompt text
        system: Optional system prompt (passed via --append-system-prompt)
        model: "haiku" or "sonnet" (resolved by CLI to current model versions)
        timeout: Subprocess timeout in seconds
        json_mode: If True, instruct Claude to respond with JSON only
        log: Optional logging function
        return_envelope: If True, return (text, envelope) tuple instead of just
            text. Envelope is the parsed --output-format json response (or None
            on failure). Used by lapis-pm shaped-agent runner for tool-use
            detection / confabulation heuristics.
        cwd: Working directory for the `claude` subprocess. Determines which
            CLAUDE.md (and thus SessionStart hooks like chub-inject.py) the
            subprocess picks up. Defaults to "/srv/agents" to preserve the
            behavior this call had before the kwarg existed; shaped-agent
            dispatch passes the repo working clone so the agent inherits
            repo CLAUDE.md + chub injection + per-project auto-memory.
        permission_mode: Optional `claude -p` permission mode — one of
            "acceptEdits" | "auto" | "bypassPermissions" | "default" |
            "dontAsk" | "plan". When `-p` is used, the workspace-trust
            dialog is skipped, so Write/Edit to an un-trusted cwd returns
            a "please allow writes" message rather than succeeding. Shaped
            agents running headless should pass "bypassPermissions" so
            their Write/Edit work without a human to approve. None means
            don't pass the flag (current behavior).

    Returns:
        str | None on default (text or None on failure), or
        (str | None, dict | None) when return_envelope=True.
    """
    cmd = [
        "claude", "-p",
        "--model", model,
        "--no-session-persistence",
        "--output-format", "json",
    ]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if system:
        cmd += ["--append-system-prompt", system]

    user_input = prompt
    if json_mode:
        user_input = prompt + "\n\nRespond ONLY with valid JSON. No markdown fences."

    def _ret(text, envelope):
        return (text, envelope) if return_envelope else text

    try:
        result = subprocess.run(
            cmd,
            input=user_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or "/srv/agents",
        )
    except subprocess.TimeoutExpired:
        if log:
            log(f"claude -p timed out after {timeout}s (model={model})")
        return _ret(None, None)
    except Exception as e:
        if log:
            log(f"claude -p failed to launch: {e}")
        return _ret(None, None)

    if result.returncode != 0:
        stderr = result.stderr[:500] if result.stderr else "(no stderr)"
        if log:
            log(f"claude -p exited {result.returncode}: {stderr}")
        return _ret(None, None)

    # Parse the JSON envelope from --output-format json
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError:
        if log:
            log(f"claude -p returned invalid JSON envelope: {result.stdout[:300]}")
        return _ret(None, None)

    # Envelope should be a dict with a "result" key, but guard against
    # unexpected shapes (e.g., list) that would crash callers with
    # "'list' object has no attribute 'get'"
    if not isinstance(envelope, dict):
        if log:
            log(f"claude -p returned non-dict envelope: {type(envelope).__name__}")
        return _ret(str(envelope) if envelope else None, None)

    text = envelope.get("result") or ""
    if not isinstance(text, str):
        if log:
            log(f"claude -p 'result' field is {type(text).__name__}, not str: {str(text)[:200]}")
        text = str(text) if text else ""
    if not text.strip():
        # Log the full envelope so we can see error/type fields if present
        if log:
            log(f"claude -p returned empty result. envelope keys: {list(envelope.keys())}, "
                f"is_error={envelope.get('is_error')}, type={envelope.get('type')}")
        return _ret(None, envelope)
    return _ret(text, envelope)


# --- JSON parsing utilities (from ollama_utils.py) ---

def parse_json_object(text: str) -> dict | None:
    """Extract a single JSON object from LLM response text.

    Strips markdown fences, then tries direct parse. Falls back to
    json.JSONDecoder.raw_decode to find the first valid object.
    """
    text = _strip_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    return _raw_decode_first(text, "{")


def parse_json_array(text: str) -> list[dict] | None:
    """Extract a JSON array from LLM response text.

    Strips markdown fences, then tries direct parse. Falls back to
    json.JSONDecoder.raw_decode to find the first valid array.
    """
    text = _strip_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return _raw_decode_first(text, "[")


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _raw_decode_first(text: str, opener: str):
    """Use json.JSONDecoder.raw_decode starting from the first opener char."""
    decoder = json.JSONDecoder()
    idx = text.find(opener)
    while idx != -1:
        try:
            result, _ = decoder.raw_decode(text, idx)
            return result
        except json.JSONDecodeError:
            idx = text.find(opener, idx + 1)
    return None


# --- Logging utilities (from ollama_client.py / ollama_utils.py) ---

def make_logger(log_file: str):
    """Create a log function that writes to both stdout and the given file."""
    def _log(msg):
        ts = datetime.now(PACIFIC).strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
    return _log


def log(msg, log_file=None):
    """Timestamped log to stdout + optional file. Compat with ollama_client.log."""
    ts = datetime.now(PACIFIC).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
