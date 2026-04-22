#!/usr/bin/env python3
"""Shared LLM client — talks to llama-server and Claude CLI (Max subscription).

Two backends:
  - call_llm()         → local llama-server (qwen3.5-35b-a3b, GPU, free)
  - call_claude_cli()  → claude -p subprocess (Haiku/Sonnet, Max subscription)

All conductor/agent scripts should import from here.
"""

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

TAILSCALE_IP = "203.0.113.12"
LLAMACPP_URL = f"http://{TAILSCALE_IP}:8081"
PACIFIC = ZoneInfo("America/Los_Angeles")


def call_llm(prompt: str, system: str = None, timeout: int = 600,
             json_mode: bool = False, temperature: float = 0.7,
             log=None, bundle_ids: list[str] = None) -> str | None:
    """Send a completion request to llama-server via /v1/chat/completions.

    Drop-in replacement for ollama_client.call_ollama() and
    ollama_utils.call_ollama().

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
                return None
        except Exception as e:
            if log:
                log(f"LLM call error: {e}")
            return None


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
            cwd="/srv/agents",
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
