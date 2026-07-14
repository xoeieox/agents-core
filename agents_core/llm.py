#!/usr/bin/env python3
"""Shared LLM client — talks to llama-server and Claude CLI (Max subscription).

Backends:
  - call_llm()         → local llama-server (qwen3.6-35b-a3b, GPU, free)
  - call_operator()    → multi-operator routing (qwen / sonnet / opus / haiku)
  - call_claude_cli()  → claude -p subprocess (Haiku/Sonnet, Max subscription)

All conductor/agent scripts should import from here.
"""

import inspect
import json
import logging
import os
import re
import subprocess
import threading
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from requests.exceptions import Timeout, ConnectionError, HTTPError, ChunkedEncodingError

TAILSCALE_IP = "203.0.113.12"
LLAMACPP_URL = f"http://{TAILSCALE_IP}:8081"
PACIFIC = ZoneInfo("America/Los_Angeles")

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")
GW_CREATIVE_URL = os.environ.get("GW_CREATIVE_URL", "http://203.0.113.11:8093")
QUEST_URL = os.environ.get("QUEST_URL", "http://203.0.113.11:8080")
SWARM_URL = os.environ.get("SWARM_URL", GW_URL)
SWARM_MAX_CONCURRENT = int(os.environ.get("SWARM_MAX_CONCURRENT", "4"))

# GW admission provenance vocabulary — all known tuples appended to _provenance_out.
#
#   admission_off_master_passthrough — enforce mode on a non-master node; request passed through.
#   admission_shadow:<decision>      — shadow mode dry-run result ("would-admit" or "would-wait").
#   admission_shadow:principal_group_collision_risk — shadow mode: unique work_id principal used.
#   drain_count_unavailable          — doorman honored acquire but drain_cleared absent (pre-atomic doorman); proceeding on elevator gate alone.
#   doorman_unreachable              — doorman acquire failed; routed to wake_fail.
#   gw_deferred_swarm                — doorman deferred to swarm; requeueing (precedence ladder).
#   gw_member_deadline               — per-member watchdog fired (AC2); ticket failed, lease released.
#   gw_member_error                  — unexpected exception from backend dispatch (AC1); ticket failed.
#   gw_not_serving                   — doorman responded not-serving; bounded backoff requeue (precedence ladder).
#   serving_http_error               — OperatorUnreachableError from backend HTTP layer.
#   slot_pool_down                   — GW slot pool unavailable (precedence ladder).
#   slot_queued_timeout              — wait deadline expired before admission (precedence ladder).
#   success                          — backend returned successfully; ticket ack'd.
#
# GW_PROVENANCE_PRECEDENCE orders the gating reasons for gw_highest_precedence_reason (AC11).
GW_PROVENANCE_PRECEDENCE = (
    "gw_deferred_swarm",
    "slot_queued_timeout",
    "slot_pool_down",
    "gw_not_serving",
)

_log = logging.getLogger(__name__)


def gw_highest_precedence_reason(provenance: list) -> str | None:
    """Return the highest-precedence GW reason from a provenance list (AC11)."""
    reasons = {p[0] if isinstance(p, (tuple, list)) else p for p in provenance}
    for reason in GW_PROVENANCE_PRECEDENCE:
        if reason in reasons:
            return reason
    return None


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


class CreativeOperatorUnavailable(Exception):
    """Raised when the gravitywell-creative (Llama-70B) endpoint cannot be reached.

    Distinct from OperatorUnreachableError so callers can tell "creative serve not up"
    from generic GW failures. Never silently falls back to a paid or other model.
    """

    def __init__(self, url: str, last_error: Exception):
        self.url = url
        self.last_error = last_error
        super().__init__(f"Creative operator (Llama-70B) unreachable at {url}: {last_error}")


class GWServingModeMismatchError(Exception):
    """Raised when the GravityWell endpoint serves a different model than GW_BACKEND/GW_MODEL
    claim (config/serving-mode drift) — e.g. GW_BACKEND=vllm resolves to gravitywell-27b but
    the port is still serving the llama.cpp 122B. This is a *misconfiguration*, distinct from
    genuine GW unavailability (box asleep, doorman down, HTTP exhausted), which keeps its
    existing on_wake_fail behavior unchanged.

    Per Erah's ruling (2026-07-07, decision/gw-voicing-drift-hard-fail-not-fallback-2026-07-07):
    "the issuing session restores the right model and re-runs; fallback should be a deliberate
    choice, not an automatic affordance." Callers must NOT catch this alongside
    OperatorUnreachableError / route it through on_wake_fail.
    """

    def __init__(self, url: str, expected_model: str, actual_model: str):
        self.url = url
        self.expected_model = expected_model
        self.actual_model = actual_model
        super().__init__(
            f"GravityWell serving-mode mismatch at {url}: expected {expected_model!r}, "
            f"port serves {actual_model!r} — restore the serving mode and re-run."
        )


# ---------------------------------------------------------------------------
# Multi-operator routing
# ---------------------------------------------------------------------------

OPERATOR_DEFAULTS: dict[str, str] = {
    "qwen":                 "qwen3.6-35b-a3b",
    "quest":                "quest-35b-rl",
    "sonnet":               "claude-sonnet-4-6",
    "opus":                 "claude-opus-4-7",
    "haiku":                "claude-haiku-4-5-20251001",
    "gravitywell":          "gravitywell-122b",
    "gravitywell-creative": "gravitywell-llama-70b",
}


def _gw_explicit_model() -> str | None:
    """Return the explicit GW_MODEL value, or None if unset."""
    return os.environ.get("GW_MODEL") or None


def _gw_explicit_backend() -> str | None:
    """Return the validated explicit GW_BACKEND value, or None if unset.

    Raises ValueError on a set-but-unrecognized value (unchanged from before this seam
    existed - a misconfiguration is never silently normalized).
    """
    val = os.environ.get("GW_BACKEND")
    if val is None:
        return None
    if val not in ("llamacpp", "vllm"):
        raise ValueError(
            f"Unknown GW_BACKEND={val!r}. Must be 'llamacpp' or 'vllm' (or unset, which "
            "auto-detects the currently-served model)."
        )
    return val


def _gw_backend(discovered_model: str | None = None) -> str:
    """Resolve the llama.cpp-vs-vLLM payload dialect.

    Explicit GW_BACKEND always wins (set-but-unrecognized raises ValueError - a
    misconfigured switch must force a deliberate fix, never silently normalize).
    Otherwise, when auto-detecting, `discovered_model` (the model _gw_discover_serving()
    found being served) decides: "llamacpp" iff it equals OPERATOR_DEFAULTS["gravitywell"]
    ("gravitywell-122b", the only llama.cpp-served model name in this ecosystem today),
    else "vllm". With no explicit backend and no discovered_model (e.g. called standalone
    with no call context), falls back to "llamacpp".
    """
    explicit = _gw_explicit_backend()
    if explicit is not None:
        return explicit
    if discovered_model is not None:
        return "llamacpp" if discovered_model == OPERATOR_DEFAULTS["gravitywell"] else "vllm"
    return "llamacpp"


# Pre-flight serving-mode handshake cache (1d.1): (url, resolved-model) -> verified.
# Holds only a positive "verified" marker; elides the redundant startup /v1/models probe.
# It never suppresses the per-call response-echo check (_call_gravitywell_backend re-verifies
# model identity from the actual response on every call, unconditionally — see 1d.2).
_gw_handshake_cache: dict[tuple[str, str], bool] = {}
_gw_handshake_lock = threading.Lock()


def _gw_probe_served_model(url: str, timeout: int = 10, log=None) -> str | None:
    """Pure transport: GET {url}/v1/models and return the served model id, or None on
    any failure (connect error, timeout, malformed response, empty data). No caching, no
    assertion - callers own both. Shared by _gw_verify_serving_mode (explicit-mode,
    process-lifetime cache, hard-fail-on-drift) and _gw_discover_serving (auto-detect,
    TTL-bounded cache, no assertion) so the two only differ in caching/assertion
    semantics, not in how they talk to GW.
    """
    try:
        resp = requests.get(f"{url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0].get("id") if data.get("data") else None
    except Exception as e:
        if log:
            log(f"[gravitywell] /v1/models probe unreachable (treated as availability, "
                f"not drift): {e}")
        return None


def _gw_verify_serving_mode(url: str, model: str, log=None) -> None:
    """Pre-flight handshake (once per process, cached): assert {url}/v1/models serves `model`.

    Reachable-but-wrong-name -> raises GWServingModeMismatchError (drift; Erah ruling
    2026-07-07: hard-fail, not fallback). A *connect failure* on the probe is unavailability,
    not drift — silently returns so the caller's normal call/retry path runs and raises
    OperatorUnreachableError (routed to on_wake_fail) exactly as it does today.

    Facets fans 4 personas out via ThreadPoolExecutor, so the first call in a fresh process
    can race across threads; the lock guards cache reads/writes only (redundant concurrent
    probes are idempotent and harmless — the /v1/models GET is read-only).
    """
    cache_key = (url, model)
    with _gw_handshake_lock:
        if _gw_handshake_cache.get(cache_key):
            return

    served = _gw_probe_served_model(url, log=log)
    if served is None:
        return

    if served != model:
        raise GWServingModeMismatchError(url, model, served)

    with _gw_handshake_lock:
        _gw_handshake_cache[cache_key] = True


# TTL-bounded discovery cache (auto-detect case): url -> (served_model, discovered_at).
# Deliberately NOT a process-lifetime cache like _gw_handshake_cache above - a long-lived
# process (claude-queue-runner above all) must still notice a genuine mode flip within a
# bounded window. See _gw_discover_serving() for the full rationale.
_gw_discovery_cache: dict[str, tuple[str, float]] = {}
GW_DISCOVERY_TTL_S_DEFAULT = 30.0


def _gw_discovery_ttl_s() -> float:
    """Read at call time (not module load) - matches every other tunable in this file
    (GW_IDLE_GAP_SECS, GW_FIRST_TOKEN_GAP_SECS, etc.), and lets tests override via
    monkeypatch.setenv."""
    return float(os.environ.get("GW_DISCOVERY_TTL_S", str(GW_DISCOVERY_TTL_S_DEFAULT)))


def _gw_discover_serving(url: str, log=None) -> str | None:
    """Auto-detect (unset GW_MODEL and GW_BACKEND): return what `url` is currently
    serving, probing at most once per _gw_discovery_ttl_s() seconds per url.

    Deliberately NOT cached for a process's full lifetime the way the explicit-mode
    handshake is cached (_gw_verify_serving_mode) - a long-lived process must still notice
    a genuine mode flip within a bounded window. The TTL bounds re-probe frequency so a
    burst of concurrent calls doesn't hammer /v1/models once per call.

    A probe failure (GW unreachable) returns None and is never cached - the caller falls
    back to the legacy 122b name, and the real completion call then also fails to
    connect, raising OperatorUnreachableError via the existing connect-retry path exactly
    as it does today. This fallback model name is never actually sent to a live server in
    that case.

    Lock scope matches _gw_verify_serving_mode's existing pattern: held only around the
    cache dict read and the cache dict write, never across the network call itself -
    redundant concurrent probes are idempotent and harmless (the /v1/models GET is
    read-only), and holding the lock across the network call would serialize every
    concurrent GW probe process-wide (both functions share _gw_handshake_lock).
    """
    now = time.monotonic()
    with _gw_handshake_lock:
        cached = _gw_discovery_cache.get(url)
        if cached is not None and (now - cached[1]) < _gw_discovery_ttl_s():
            return cached[0]
    served = _gw_probe_served_model(url, log=log)
    if served is not None:
        with _gw_handshake_lock:
            _gw_discovery_cache[url] = (served, now)
    return served


def _gw_default_model(url: str = None, log=None) -> str:
    """Resolve the default gravitywell model name, read at call time (not module-load).

    Explicit GW_MODEL always wins. Otherwise, explicit GW_BACKEND=vllm -> "gravitywell-27b",
    GW_BACKEND=llamacpp -> OPERATOR_DEFAULTS["gravitywell"] ("gravitywell-122b"). With
    neither set, auto-detects by asking `url` (or GW_URL) what it is currently serving via
    _gw_discover_serving() - correct whether GW is resting in big (122B) or dual (27B), and
    immune to the boot-default posture changing again without a code edit. Falls back to
    OPERATOR_DEFAULTS["gravitywell"] if the discovery probe fails (GW unreachable); the
    subsequent real call then fails to connect too, surfacing as OperatorUnreachableError.
    """
    explicit_model = _gw_explicit_model()
    if explicit_model:
        return explicit_model
    explicit_backend = _gw_explicit_backend()
    if explicit_backend is not None:
        return "gravitywell-27b" if explicit_backend == "vllm" else OPERATOR_DEFAULTS["gravitywell"]
    discovered = _gw_discover_serving(url or GW_URL, log=log)
    return discovered if discovered is not None else OPERATOR_DEFAULTS["gravitywell"]


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


def _post_chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int = 600,
    json_mode: bool = False,
    temperature: float = 0.7,
    think: bool = False,
    max_retries: int = 3,
    log=None,
    cache_prompt: bool | None = None,
    chat_template_kwargs: dict | None = None,
    _no_thinking: bool = False,
) -> str | None:
    """Shared POST core for OpenAI-compatible chat/completions endpoints.

    Posts to {base_url}/v1/chat/completions with messages and optional chat_template_kwargs.
    Retries up to max_retries on transient errors (timeout, connection, chunked encoding).
    Returns response text on success, None on parse errors, raises OperatorUnreachableError
    on persistent HTTP/network failures.

    think=False (default) sends chat_template_kwargs={"enable_thinking": false} explicitly,
    matching _call_gravitywell_backend's pattern. _no_thinking=True structurally omits
    chat_template_kwargs entirely, for models that do not support the thinking knob.

    Used by _call_gravitywell_backend, call_swarm, and other chat-completion callers.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    elif not _no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            return text if text.strip() else None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ChunkedEncodingError) as e:
            if attempt < max_retries - 1:
                if attempt == 0:
                    backoff = 2
                elif attempt == 1:
                    backoff = 4
                else:
                    backoff = 0
                if log:
                    log(f"Chat completion call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(backoff)
            else:
                if log:
                    log(f"Chat completion call failed after {max_retries} attempts: {e}")
                raise OperatorUnreachableError(base_url, e)
        except Exception as e:
            if log:
                log(f"Chat completion call error: {e}")
            return None


def _gw_stream_attempt(base_url, model, payload, idle_gap, first_token_gap, hard_ceiling, call_start, log):
    """One streaming attempt. Returns (text_or_None, cull_tuple_or_None, served_model_or_None).

    Raises requests exceptions on connect failure (for caller to retry).
    cull_tuple = (reason_str, elapsed_secs, idle_secs) or None.
    served_model_or_None is the "model" field echoed by the response's SSE chunks (1c/1d.2
    true-mirror provenance) — None if the response never echoed one (e.g. a test double).
    """
    # Socket must outlast all Python timers so the watchdog fires first.
    # idle_gap * 2 would cause ReadTimeout before phase-1 first_token_gap (600s default).
    sock_timeout = (30, hard_ceiling + 60)

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=sock_timeout,
        stream=True,
    )
    resp.raise_for_status()

    _state = {
        "cull": None,          # (reason, elapsed, idle) when fired
        "first_token_at": None,
        "last_chunk_at": time.monotonic(),
        "served_model": None,  # "model" field echoed by the response (1c/1d.2)
    }
    _done_event = threading.Event()

    # Watchdog poll interval: responsive but not CPU-burning
    poll_interval = max(0.2, min(0.5, idle_gap / 10))

    def _watchdog():
        while not _done_event.wait(poll_interval):
            now = time.monotonic()
            elapsed = now - call_start
            if elapsed >= hard_ceiling:
                _state["cull"] = ("hard_ceiling_exceeded", elapsed, now - _state["last_chunk_at"])
                resp.close()
                return
            if _state["first_token_at"] is None:
                # Phase 1: check first-token grace
                if elapsed >= first_token_gap:
                    idle = elapsed
                    _state["cull"] = ("first_token_grace_exceeded", elapsed, idle)
                    resp.close()
                    return
            else:
                # Phase 2: check idle gap since last content chunk
                idle = now - _state["last_chunk_at"]
                if idle >= idle_gap:
                    _state["cull"] = ("idle_gap_exceeded", elapsed, idle)
                    resp.close()
                    return

    wt = threading.Thread(target=_watchdog, daemon=True)
    wt.start()

    content_parts = []
    reasoning_parts = []
    clean_end = False

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                clean_end = True
                break
            try:
                chunk = json.loads(data_str)
                if _state["served_model"] is None:
                    chunk_model = chunk.get("model")
                    if chunk_model:
                        _state["served_model"] = chunk_model
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if content or reasoning:
                    # First token received - transition to phase 2
                    if _state["first_token_at"] is None:
                        _state["first_token_at"] = time.monotonic()
                    _state["last_chunk_at"] = time.monotonic()
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    except Exception:
        # Connection closed by watchdog or network error - check cull state below
        pass
    finally:
        _done_event.set()
        wt.join(timeout=2.0)

    # If stream ended cleanly, ignore any watchdog cull (race condition safe-fallback)
    if clean_end:
        text = "".join(content_parts) or "".join(reasoning_parts)
        return (text if text.strip() else None, None, _state["served_model"])

    if _state["cull"]:
        return (None, _state["cull"], _state["served_model"])

    # Stream ended without [DONE] and no cull - return what we have
    text = "".join(content_parts) or "".join(reasoning_parts)
    return (text if text.strip() else None, None, _state["served_model"])


def _call_gravitywell_backend(
    prompt: str,
    system: str = None,
    timeout: int = 600,
    json_mode: bool = False,
    temperature: float = 0.7,
    log=None,
    think: bool = False,
    _url: str = None,
    _model: str = None,
    _no_thinking: bool = False,
    _served_model_out: list | None = None,
) -> str | None:
    """Send a completion request to a GravityWell endpoint via streaming SSE.

    By default targets GW_URL (:8081) with the model resolved by _gw_default_model()
    (GW_MODEL / GW_BACKEND env-driven when set; when both are unset, auto-detects the
    currently-served model via _gw_discover_serving() instead of assuming a fixed
    default). Internal _url/_model params route to alternate endpoints (e.g. the creative
    Llama-70B at :8093) without exposing that routing on the public 122B operator path.

    The default GW path (_url is None and _model is None) only:
    - Payload dialect gates on GW_BACKEND when explicit ("llamacpp" byte-identical to
      today; "vllm" omits the llama.cpp-only cache_prompt field), or on the auto-detected
      model when GW_BACKEND/GW_MODEL are both unset - see _gw_backend().
    - Runs the pre-flight serving-mode handshake (cached) when GW_BACKEND/GW_MODEL are
      explicit - skipped when auto-detecting, since the discovery probe already
      established what's served - and always runs the per-call response-echo assertion,
      raising GWServingModeMismatchError on reachable-but-wrong-model drift (explicit
      case) or a mid-flight flip between discovery and this call's response (auto-detect
      case).
    The gravitywell-creative path (_url/_model explicit) is untouched by either: always
    llama.cpp cache_prompt dialect, no handshake, no GW_BACKEND coupling.

    think=False (default) injects chat_template_kwargs={"enable_thinking": false} to
    suppress the think-trace. Callers may pass think=True for quality-mode reasoning.
    _no_thinking=True structurally omits chat_template_kwargs entirely (for models that
    do not support the thinking knob, e.g. Llama-3.3-70B-Instruct).
    _served_model_out: optional list to append the response-echoed "model" field to (1c
    true-mirror provenance) — the model the server actually reported serving, not merely
    the request's model field.

    Dual-timer liveness model:
    - Phase 1 (pre-first-token): cull after GW_FIRST_TOKEN_GAP_SECS (default 600).
    - Phase 2 (post-first-token): cull after GW_IDLE_GAP_SECS (default 45) of chunk silence.
    - Hard ceiling: GW_LIVENESS_HARD_CEILING_SECS (default 1800) total.
    - Stall retry: if idle_gap_exceeded, retry once within the same ceiling budget.

    Connection-level retry: 3 attempts, 2s/4s backoff on network errors.
    Persistent errors raise OperatorUnreachableError; parse errors return None.
    Serving-mode drift raises GWServingModeMismatchError (never caught by on_wake_fail).
    """
    url = _url if _url is not None else GW_URL
    is_default_gw_path = _url is None and _model is None
    is_auto_detecting = (
        is_default_gw_path and _gw_explicit_model() is None and _gw_explicit_backend() is None
    )
    model = _model if _model is not None else _gw_default_model(url=url, log=log)

    if is_default_gw_path:
        if not is_auto_detecting:
            _gw_verify_serving_mode(url, model, log=log)
        backend = _gw_backend(discovered_model=model if is_auto_detecting else None)
    else:
        backend = "llamacpp"

    idle_gap = float(os.environ.get("GW_IDLE_GAP_SECS", "45"))
    first_token_gap = float(os.environ.get("GW_FIRST_TOKEN_GAP_SECS", "600"))
    hard_ceiling = float(os.environ.get("GW_LIVENESS_HARD_CEILING_SECS", "1800"))
    # Caller-supplied timeout caps hard_ceiling in non-enforce mode; enforce mode
    # applies _member_deadline externally via Future.result(timeout=...) and doesn't
    # rely on this, but honor a tighter caller deadline here too.
    hard_ceiling = min(hard_ceiling, float(timeout))

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if backend == "llamacpp":
        payload["cache_prompt"] = True
    payload["stream"] = True
    if not _no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    call_start = time.monotonic()

    def _attempt_with_connect_retry():
        """Run _gw_stream_attempt with up to 3 connect-level retries."""
        last_exc = None
        for attempt in range(3):
            try:
                return _gw_stream_attempt(
                    url, model,
                    payload, idle_gap, first_token_gap, hard_ceiling,
                    call_start, log,
                )
            except (Timeout, ConnectionError, HTTPError, ChunkedEncodingError) as e:
                last_exc = e
                if attempt < 2:
                    backoff = 2 if attempt == 0 else 4
                    if log:
                        log(f"[gw-stream] connect error attempt {attempt+1}/3: {e}")
                    time.sleep(backoff)
                else:
                    raise OperatorUnreachableError(url, e)

    # First attempt
    text, cull, served_model = _attempt_with_connect_retry()

    # Stall retry: if idle_gap_exceeded, retry once within the same ceiling budget
    if cull is not None and cull[0] == "idle_gap_exceeded":
        cull_reason, cull_elapsed, cull_idle = cull
        _log.error(
            "[gw-liveness] cull on first attempt reason=%s elapsed=%.1fs idle=%.1fs — retrying",
            cull_reason, cull_elapsed, cull_idle,
        )
        text, cull, served_model = _attempt_with_connect_retry()

    if cull is not None:
        cull_reason, cull_elapsed, cull_idle = cull
        _log.error(
            "[gw-liveness] stream culled reason=%s elapsed=%.1fs idle=%.1fs",
            cull_reason, cull_elapsed, cull_idle,
        )
        return None

    # Per-call response-echo assertion (1d.2) — runs on every call, never cached, so a
    # mid-flight serving-mode flip is caught on the very next call even after a cached
    # handshake. Silent (no signal) when the response never echoed a model field at all.
    if is_default_gw_path and served_model is not None and served_model != model:
        raise GWServingModeMismatchError(url, model, served_model)

    if served_model is not None:
        _log.info("[gravitywell] voicing ok model=%s", served_model)
        if _served_model_out is not None:
            _served_model_out.append(served_model)

    return text


def _apply_wake_fail(
    on_wake_fail: str | None,
    operator_class: str,
    prompt: str,
    _provenance_out: list | None = None,
    **kwargs,
) -> str | None:
    """Apply the declared on_wake_fail policy when GW cannot be woken.

    Policies:
      "haiku" / "sonnet" / "opus" — re-dispatch to that operator class (paid fallback;
          logs loudly before spending tokens per claude-p-api-pricing-june11).
      "skip" / None — return None (batch/optional surfaces; no paid spend).
      "error" — raise OperatorUnreachableError so the caller decides.

    If the fallback call_operator() itself raises, the exception propagates unchanged.

    _provenance_out: optional list to append (operator_class, reason) tuples for tracking
                     which operator actually answered (used by adapters for observability).
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
            if k not in ("think", "on_wake_fail", "_provenance_out")
        }
        result = call_operator(policy, prompt, _provenance_out=_provenance_out, **fallback_kwargs)
        if _provenance_out is not None:
            _provenance_out.append(("fallback", policy))
        return result

    if policy == "error":
        raise OperatorUnreachableError(
            GW_URL,
            Exception(f"GW wake_failed and on_wake_fail='error' for {operator_class!r}"),
        )

    # "skip" or None
    return None


def _forward_supported_kwargs(func, kwargs: dict) -> dict:
    """Signature-normalize kwargs at the dispatcher boundary before forwarding to `func`.

    call_operator() accepts a superset of kwargs across operator classes (on_wake_fail,
    think, _provenance_out, ...); not every backend function accepts all of them. Filtering
    here, once, at the dispatcher means a backend simply not accepting some kwarg can never
    surface as a TypeError again — the backend's own signature stays the source of truth,
    with no per-backend kwarg-swallowing shim to keep in sync.

    A func whose signature includes **kwargs (e.g. a test spy/mock wrapping the real
    backend) declares it accepts anything, so nothing is filtered in that case.
    """
    params = inspect.signature(func).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def call_operator(operator_class: str, prompt: str, model: str = None,
                  _provenance_out: list | None = None,
                  principal: str | None = None,
                  _admission_bypass: bool = False,
                  **kwargs) -> str | None:
    """Route a completion request to the appropriate backend operator.

    operator_class ∈ {"qwen", "quest", "sonnet", "opus", "haiku", "gravitywell", "gravitywell-creative"}.
    Raises ValueError for unknown classes.

    Default models:
        qwen                 → "qwen3.6-35b-a3b"
        quest                → "quest-35b-rl"       (vLLM swarm, QUEST_URL :8080, OpenAI-compat)
        sonnet               → "claude-sonnet-4-6"
        opus                 → "claude-opus-4-7"
        haiku                → "claude-haiku-4-5-20251001"
        gravitywell          → "gravitywell-122b"   (122B reasoning, :8081, DoormanClient)
        gravitywell-creative → "gravitywell-llama-70b" (Llama-70B instruct, :8093, direct)

    qwen routes via the local llama-server (same path as call_llm()).

    sonnet / opus / haiku route via ClaudeQueue → `claude -p` (Max subscription
    path). The task is submitted to ClaudeQueue with task_type="llm_call";
    `submit_and_wait` blocks the caller's thread until the daemon's
    `_run_llm_call_task` handler completes the task and writes the output file.
    No direct Anthropic-API calls — kill-switched per
    decision/no-anthropic-api-direct.

    gravitywell routes via the doorman to the GravityWell llama.cpp endpoint (:8081).
    On unreachable, falls back per on_wake_fail policy.

    gravitywell-creative routes directly to the Llama-70B endpoint (:8093, GW_CREATIVE_URL).
    No DoormanClient, no admission control, no wake_fail fallback. Raises
    CreativeOperatorUnavailable on network failure — never silently falls back.

    _provenance_out: optional list to append (reason, effective_operator) tuples
                     for tracking which operator actually answered. Used by adapters
                     for observability (e.g., recording effective voicing in council runs).

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
        return _call_qwen_backend(prompt=prompt, **_forward_supported_kwargs(_call_qwen_backend, kwargs))

    if operator_class == "quest":
        if model is not None and model != OPERATOR_DEFAULTS["quest"]:
            raise ValueError(
                f"call_operator(operator_class='quest', model={model!r}): "
                "the QUEST vLLM swarm endpoint serves a single fixed model "
                f"({OPERATOR_DEFAULTS['quest']!r}); model swaps are an "
                "infrastructure operation, not a per-call parameter. "
                "Either pass model=None to use the default, or do the swap out-of-band."
            )
        on_wake_fail = kwargs.get("on_wake_fail", "skip")
        quest_kwargs = {
            k: kwargs[k] for k in ("system", "timeout", "json_mode", "temperature", "log")
            if k in kwargs
        }
        try:
            return _post_chat_completion(
                base_url=QUEST_URL,
                model=OPERATOR_DEFAULTS["quest"],
                messages=(
                    ([{"role": "system", "content": quest_kwargs["system"]}]
                     if quest_kwargs.get("system") else [])
                    + [{"role": "user", "content": prompt}]
                ),
                timeout=int(quest_kwargs.get("timeout", 300)),
                json_mode=bool(quest_kwargs.get("json_mode", False)),
                temperature=float(quest_kwargs.get("temperature", 0.7)),
                log=quest_kwargs.get("log"),
                _no_thinking=True,
            )
        except OperatorUnreachableError:
            if on_wake_fail == "skip" or on_wake_fail is None:
                return None
            if on_wake_fail == "error":
                raise
            # paid fallback operators
            return _apply_wake_fail(
                on_wake_fail, operator_class, prompt,
                _provenance_out=_provenance_out, **{
                    k: v for k, v in kwargs.items() if k not in ("on_wake_fail",)
                },
            )

    if operator_class == "gravitywell":
        _gw_resolved_default = _gw_default_model()
        if model is not None and model != _gw_resolved_default:
            raise ValueError(
                f"call_operator(operator_class='gravitywell', model={model!r}): "
                "the GravityWell endpoint serves a single fixed model "
                f"({_gw_resolved_default!r}); model swaps are an "
                "infrastructure operation (gw-serve), "
                "not a per-call parameter. Either pass model=None to use the "
                "default, or do the model swap out-of-band first."
            )
        gw_kwargs = {
            k: kwargs[k] for k in (
                "system", "timeout", "json_mode", "temperature", "log"
            ) if k in kwargs
        }
        think = kwargs.get("think", False)
        on_wake_fail = kwargs.get("on_wake_fail", "skip")
        timeout = int(kwargs.get("timeout", 300))
        work_id = f"op-gravitywell-{uuid.uuid4().hex}"
        wake_fail_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("on_wake_fail", "think", "bundle_ids", "_provenance_out")
        }

        from agents_core.doorman_client import DoormanClient, DoormanUnreachable, _gw_acquire_timeout

        admission_mode = os.environ.get("GW_ADMISSION_MODE", "off")
        if principal is not None and not principal:
            raise ValueError(
                "[gw-admission] empty-string principal is rejected; pass None to auto-group "
                "by work_id, or pass a concrete non-empty principal"
            )
        effective_principal = principal or work_id
        is_unique_work_id_principal = (principal is None)

        # --- SHADOW MODE (AC10): dry-run admission decision, dispatch directly ---
        if admission_mode == "shadow" and not _admission_bypass:
            _shadow_claimed: set = set()
            try:
                from agents_core.elevator import ElevatorStore as _ES, DB_DIR as _ELEV_DB_DIR_S
                from pathlib import Path as _Path_S
                _elev_db_s = _Path_S(os.environ.get(
                    "ELEVATOR_DB_PATH", str(_ELEV_DB_DIR_S / "queue.db")
                ))
                _ss = _ES(_elev_db_s)
                _shadow_claimed = _ss._claimed_principals_on_lane("deliberation")
                _ss.close()
            except Exception:
                pass
            other_claimed = _shadow_claimed - {effective_principal}
            decision = "would-wait" if other_claimed else "would-admit"
            _log.info(
                "[gw-admission] shadow: %s principal=%r%s",
                decision, effective_principal,
                " [collision_risk]" if is_unique_work_id_principal else "",
            )
            if _provenance_out is not None:
                _provenance_out.append((f"admission_shadow:{decision}", "gravitywell"))
                if is_unique_work_id_principal:
                    _provenance_out.append(
                        ("admission_shadow:principal_group_collision_risk", "gravitywell")
                    )
            # Fall through to direct dispatch below.

        # --- ENFORCE MODE (AC2-AC9): self-serve admission loop ---
        if admission_mode == "enforce" and not _admission_bypass:
            from agents_core.elevator import IS_MASTER as _ELEV_IS_MASTER
            if not _ELEV_IS_MASTER:
                # AC9: off-master passthrough - no enqueue, no deadlock
                _log.warning(
                    "[gw-admission] enforce mode but not IS_MASTER - passthrough work_id=%s", work_id
                )
                if _provenance_out is not None:
                    _provenance_out.append(("admission_off_master_passthrough", "gravitywell"))
                # Fall through to direct dispatch below.
            else:
                import concurrent.futures as _cf
                from agents_core.elevator import ElevatorStore, DB_DIR as _ELEV_DB_DIR
                _max_wait = int(os.environ.get("GW_ADMISSION_MAX_WAIT_SEC", "900"))
                # AC4: claim TTL aligned to the member's backend timeout, not _max_wait.
                _claim_ttl = timeout + 90
                _poll = float(os.environ.get("GW_ADMISSION_POLL_INTERVAL_SEC", "1.5"))
                _max_wf = int(os.environ.get("MAX_WAKE_FAIL_RETRIES", "5"))

                from pathlib import Path as _Path
                _elev_db = _Path(os.environ.get("ELEVATOR_DB_PATH",
                                                 str(_ELEV_DB_DIR / "queue.db")))
                elevator = ElevatorStore(db_path=_elev_db)
                ticket = elevator.enqueue(
                    lane="deliberation",
                    kind="gw-admission",
                    payload={"work_id": work_id},
                    principal=effective_principal,
                    latency_class="batch",
                )
                admitted = False
                is_ride_along = False
                wf_retries = 0
                _loop_ticket_settled = False
                deadline = time.monotonic() + _max_wait
                client = DoormanClient()
                try:
                    while True:
                        if time.monotonic() >= deadline:
                            _log.warning(
                                "[gw-admission] slot_queued_timeout principal=%r max_wait=%ss",
                                effective_principal, _max_wait,
                            )
                            if _provenance_out is not None:
                                _provenance_out.append(("slot_queued_timeout", "gravitywell"))
                            elevator.fail(ticket)
                            _loop_ticket_settled = True
                            return _apply_wake_fail(
                                on_wake_fail, operator_class, prompt,
                                _provenance_out=_provenance_out, **wake_fail_kwargs,
                            )

                        if not admitted:
                            elevator.reclaim_stale("deliberation")  # AC3a: opportunistic reclaim
                            ok, is_ride_along = elevator.try_admit(
                                ticket, "deliberation", effective_principal, claim_ttl_sec=_claim_ttl
                            )
                            if ok:
                                admitted = True
                            else:
                                time.sleep(_poll)
                                continue

                        # Acquire doorman lease with atomic drain-gate check for fresh groups.
                        # require_drain_clear=True collapses the old two-step drain_count() +
                        # acquire() into one server-side critical section, closing the TOCTOU
                        # where concurrent distinct-principal callers all observed drain=0 before
                        # any registered. Ride-alongs share the admitted group's slot — no drain
                        # wait, unconditional acquire (require_drain_clear=False).
                        try:
                            res = client.acquire(
                                "gravitywell", work_id, ttl_sec=timeout + 60,
                                reason="call_operator", timeout=_gw_acquire_timeout(),
                                principal=effective_principal,
                                require_drain_clear=(not is_ride_along),
                            )
                        except DoormanUnreachable:
                            # AC5b: transport failure — fail-ticket, not loud-proceed (would reopen race).
                            if _provenance_out is not None:
                                _provenance_out.append(("doorman_unreachable", "gravitywell"))
                            elevator.fail(ticket)
                            _loop_ticket_settled = True
                            return _apply_wake_fail(
                                on_wake_fail, operator_class, prompt,
                                _provenance_out=_provenance_out, **wake_fail_kwargs,
                            )
                        except Exception as _acquire_err:
                            # AC5b: soft-error (e.g. HTTP 5xx) — treat as not-drain-clear, retry.
                            _log.warning(
                                "[gw-admission] acquire_soft_error work_id=%s: %s",
                                work_id, _acquire_err,
                            )
                            time.sleep(_poll)
                            continue

                        if DoormanClient.is_contended(res):
                            # Drain-gate: another group holds a worker lease. Retry within deadline.
                            time.sleep(_poll)
                            continue

                        if DoormanClient.is_deferred(res):
                            # AC6: requeue and continue waiting
                            elevator.requeue(ticket)
                            admitted = False
                            is_ride_along = False
                            if _provenance_out is not None:
                                _provenance_out.append(("gw_deferred_swarm", "gravitywell"))
                            time.sleep(_poll)
                            continue

                        elif res.get("status") != "serving":
                            # AC8: wake_failed - bounded backoff requeue
                            if _provenance_out is not None:
                                _provenance_out.append(("gw_not_serving", "gravitywell"))
                            if wf_retries >= _max_wf:
                                elevator.fail(ticket)
                                _loop_ticket_settled = True
                                return _apply_wake_fail(
                                    on_wake_fail, operator_class, prompt,
                                    _provenance_out=_provenance_out, **wake_fail_kwargs,
                                )
                            backoff = min(2 ** wf_retries, 16)
                            wf_retries += 1
                            elevator.requeue(ticket)
                            admitted = False
                            is_ride_along = False
                            time.sleep(backoff)
                            continue

                        else:
                            # AC5a: older doorman ignores require_drain_clear and serves
                            # without drain_cleared (transient deploy window only; same PR lands both).
                            if not is_ride_along and not res.get("drain_cleared"):
                                _log.warning(
                                    "[gw-admission] drain_count_unavailable - doorman did not "
                                    "confirm drain check (pre-atomic-acquire doorman?); proceeding "
                                    "on elevator gate alone work_id=%s", work_id,
                                )
                                if _provenance_out is not None:
                                    _provenance_out.append(
                                        ("drain_count_unavailable", "gravitywell")
                                    )

                            # serving: run backend via thread watchdog (AC2)
                            # _call_gravitywell_backend is a GIL-releasing HTTP socket read.
                            # ASSUMPTION: if this ever does CPU-bound/GIL-holding work, escalate to multiprocessing.
                            ticket_settled = False
                            lease_released = False
                            _member_deadline = timeout
                            _executor = _cf.ThreadPoolExecutor(max_workers=1)
                            try:
                                _future = _executor.submit(
                                    _call_gravitywell_backend,
                                    prompt=prompt, think=think, **gw_kwargs,
                                )
                                try:
                                    result = _future.result(timeout=_member_deadline)
                                except _cf.TimeoutError:
                                    # AC2/AC2a: self-triggered cleanup on member deadline.
                                    # The abandoned thread is parked in a GIL-free socket wait;
                                    # the underlying request timeout will reap it eventually.
                                    if _provenance_out is not None:
                                        _provenance_out.append(("gw_member_deadline", "gravitywell"))
                                    elevator.fail(ticket)
                                    ticket_settled = True
                                    _loop_ticket_settled = True
                                    client.release("gravitywell", work_id)
                                    lease_released = True
                                    return _apply_wake_fail(
                                        on_wake_fail, operator_class, prompt,
                                        _provenance_out=_provenance_out, **wake_fail_kwargs,
                                    )
                                except OperatorUnreachableError:
                                    if _provenance_out is not None:
                                        _provenance_out.append(("serving_http_error", "gravitywell"))
                                    elevator.fail(ticket)
                                    ticket_settled = True
                                    _loop_ticket_settled = True
                                    return _apply_wake_fail(
                                        on_wake_fail, operator_class, prompt,
                                        _provenance_out=_provenance_out, **wake_fail_kwargs,
                                    )
                                except Exception:
                                    # AC1: universal ticket release on unexpected backend error.
                                    if _provenance_out is not None:
                                        _provenance_out.append(("gw_member_error", "gravitywell"))
                                    if not ticket_settled:
                                        elevator.fail(ticket)
                                        ticket_settled = True
                                        _loop_ticket_settled = True
                                    raise
                                if _provenance_out is not None:
                                    _provenance_out.append(("success", "gravitywell"))
                                elevator.ack(
                                    ticket,
                                    provenance={
                                        "gw_provenance": [
                                            {"reason": p[0], "operator": p[1]}
                                            for p in (_provenance_out or [])
                                        ]
                                    },
                                )
                                ticket_settled = True
                                _loop_ticket_settled = True
                                return result
                            finally:
                                _executor.shutdown(wait=False)
                                if not lease_released:
                                    client.release("gravitywell", work_id)
                finally:
                    # AC3: guarantee no pending-wait ticket leaks on any catchable exit.
                    if not _loop_ticket_settled:
                        try:
                            if _provenance_out is not None:
                                _provenance_out.append(
                                    ("gw_admission_loop_aborted", "gravitywell")
                                )
                            elevator.fail(ticket)
                        except Exception:
                            pass
                    client.close()
                    elevator.close()

        # --- DIRECT DISPATCH (off, shadow, bypass, off-master) ---
        client = DoormanClient()
        try:
            res = client.acquire(
                "gravitywell", work_id, ttl_sec=timeout + 60, reason="call_operator",
                timeout=_gw_acquire_timeout(), principal=effective_principal
            )
            if DoormanClient.is_deferred(res):
                if _provenance_out is not None:
                    _provenance_out.append(("gw_deferred_swarm", "gravitywell"))
                return _apply_wake_fail(on_wake_fail, operator_class, prompt,
                                       _provenance_out=_provenance_out, **wake_fail_kwargs)
            elif res.get("status") != "serving":
                if _provenance_out is not None:
                    _provenance_out.append(("gw_not_serving", "gravitywell"))
                return _apply_wake_fail(on_wake_fail, operator_class, prompt,
                                       _provenance_out=_provenance_out, **wake_fail_kwargs)
            try:
                result = _call_gravitywell_backend(prompt=prompt, think=think, **gw_kwargs)
                if _provenance_out is not None:
                    _provenance_out.append(("success", "gravitywell"))
                return result
            finally:
                client.release("gravitywell", work_id)
        except DoormanUnreachable:
            if _provenance_out is not None:
                _provenance_out.append(("doorman_unreachable", "gravitywell"))
            return _apply_wake_fail(on_wake_fail, operator_class, prompt,
                                   _provenance_out=_provenance_out, **wake_fail_kwargs)
        except OperatorUnreachableError:
            if _provenance_out is not None:
                _provenance_out.append(("serving_http_error", "gravitywell"))
            return _apply_wake_fail(on_wake_fail, operator_class, prompt,
                                   _provenance_out=_provenance_out, **wake_fail_kwargs)

    if operator_class == "gravitywell-creative":
        if model is not None and model != OPERATOR_DEFAULTS["gravitywell-creative"]:
            raise ValueError(
                f"call_operator(operator_class='gravitywell-creative', model={model!r}): "
                "the creative endpoint serves a single fixed model "
                f"({OPERATOR_DEFAULTS['gravitywell-creative']!r}); model swaps are an "
                "infrastructure operation (gw-collider-up/down), "
                "not a per-call parameter. Either pass model=None or do the swap out-of-band."
            )
        if kwargs.get("think"):
            raise ValueError(
                "call_operator(operator_class='gravitywell-creative', think=True): "
                "Llama-3.3-70B-Instruct is not a reasoning model and cannot honor think=True. "
                "Pass think=False or omit it."
            )
        gw_kwargs = {
            k: kwargs[k] for k in (
                "system", "timeout", "json_mode", "temperature", "log"
            ) if k in kwargs
        }
        try:
            result = _call_gravitywell_backend(
                prompt=prompt,
                _url=GW_CREATIVE_URL,
                _model=OPERATOR_DEFAULTS["gravitywell-creative"],
                _no_thinking=True,
                **gw_kwargs,
            )
        except OperatorUnreachableError as exc:
            raise CreativeOperatorUnavailable(GW_CREATIVE_URL, exc) from exc
        if _provenance_out is not None:
            _provenance_out.append(("success", "gravitywell-creative"))
        return result

    # Anthropic-family: route via ClaudeQueue → call_claude_cli.
    # No direct Anthropic-API code path (decision/no-anthropic-api-direct).
    resolved_model = model or OPERATOR_DEFAULTS[operator_class]
    from agents_core.claude_queue_sync import submit_and_wait
    timeout = int(kwargs.get("timeout", 300))
    if _provenance_out is not None:
        _provenance_out.append(("success", operator_class))
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


# ---------------------------------------------------------------------------
# Lease-free swarm client
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed

def swarm_serving(swarm_url: str = SWARM_URL, timeout: int = 4) -> bool:
    """Check if the swarm endpoint is ready to serve.

    Probes {swarm_url}/v1/models (must be 200 + non-empty) AND
    {swarm_url}/health (must be 200). Both must succeed for serving=True.

    Does NOT use the doorman (which returns False for a healthy vLLM).
    Does NOT check systemctl (a unit can be active while the model is still loading).
    Does NOT assume phase (liveness ≠ phase): both big-llama.cpp and swarm-vLLM bind
    the same :8081 and speak OpenAI, so this probe proves only that *an* OpenAI
    endpoint is live — phase (swarm resident vs big) is the caller's responsibility.

    Returns False on any error (timeout, connection error, HTTP error, empty models).
    """
    try:
        models_resp = requests.get(f"{swarm_url}/v1/models", timeout=timeout)
        models_resp.raise_for_status()
        models_data = models_resp.json()
        if not models_data.get("data") or len(models_data["data"]) == 0:
            return False

        health_resp = requests.get(f"{swarm_url}/health", timeout=timeout)
        health_resp.raise_for_status()
        return True
    except Exception:
        return False


def swarm_model(swarm_url: str = SWARM_URL, timeout: int = 4) -> str | None:
    """Get the served model ID from the swarm endpoint.

    Returns the model id (e.g., 'Qwen2.5-3B') from {swarm_url}/v1/models data[0].id.
    Returns None if the endpoint is not serving or the response is malformed.

    For observability and phase discrimination: a caller comparing swarm_model()
    against its expected swarm model asserts the model phase (vs big).
    """
    try:
        resp = requests.get(f"{swarm_url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data") and len(data["data"]) > 0:
            return data["data"][0].get("id")
        return None
    except Exception:
        return None


def call_swarm(
    prompts: list[str],
    system: str | None = None,
    max_concurrent: int | None = None,
    timeout: int = 300,
    temperature: float = 0.7,
    model: str | None = None,
    swarm_url: str = SWARM_URL,
    think: bool = False,
    log=None,
) -> list[str | None]:
    """Lease-free N-wide completion client for the swarm endpoint.

    Posts one /v1/chat/completions request per prompt to {swarm_url}, fans out
    bounded by max_concurrent (default SWARM_MAX_CONCURRENT=4, env overridable).
    Preserves input order, isolates per-prompt failures to None (one bad prompt
    never raises the batch).

    Args:
        prompts: List of user prompts to complete.
        system: Optional system prompt (same for all).
        max_concurrent: Max concurrent requests (default 4, env SWARM_MAX_CONCURRENT).
        timeout: Per-prompt timeout in seconds (default 300).
        temperature: Sampling temperature (default 0.7).
        model: Model name (default None → resolve from /v1/models, model-agnostic).
        swarm_url: Swarm endpoint URL (default SWARM_URL env).
        think: Enable thinking/CoT mode (default False — structured/dispatch work
            wants clean output, not reasoning traces). Pass True to opt into
            quality-mode reasoning.
        log: Optional logging function.

    Returns:
        list[str | None]: One entry per prompt, same order. None if that prompt
        failed (timeout, HTTP error, parse error, endpoint not serving, etc.).

    Safety guarantee (load-bearing):
        - This module does NOT import DoormanClient (static guarantee).
        - No doorman acquire, no mode flip, no lease. Posts directly to the swarm.
        - A behavioral test asserts acquire is never called even in misconfigured env.
    """
    if max_concurrent is None:
        max_concurrent = SWARM_MAX_CONCURRENT

    if not prompts:
        return []

    if model is None:
        model = swarm_model(swarm_url, timeout=4)
        if model is None:
            if log:
                log(f"[call_swarm] swarm_not_serving — /v1/models failed or empty")
            return [None] * len(prompts)

    messages_template = []
    if system:
        messages_template.append({"role": "system", "content": system})

    results = [None] * len(prompts)
    results_lock = __import__("threading").Lock()

    def _post_prompt(index: int, prompt: str) -> tuple[int, str | None, str]:
        """Post a single prompt and return (index, result, error_label)."""
        messages = messages_template.copy()
        messages.append({"role": "user", "content": prompt})

        try:
            result = _post_chat_completion(
                base_url=swarm_url,
                model=model,
                messages=messages,
                timeout=timeout,
                json_mode=False,
                temperature=temperature,
                think=think,
                max_retries=3,
                log=log,
            )
            if result is not None:
                return (index, result, "success")
            else:
                return (index, None, "prompt_parse_error")
        except OperatorUnreachableError as e:
            # Check for timeout first (takes precedence)
            if isinstance(e.last_error, requests.exceptions.Timeout):
                return (index, None, "prompt_timeout")
            # Distinguish endpoint unreachable vs per-prompt HTTP error
            if "Connection" in str(e.last_error) or "resolve" in str(e.last_error).lower():
                return (index, None, "endpoint_unreachable")
            else:
                return (index, None, "prompt_http_error")
        except Exception as e:
            if isinstance(e, requests.exceptions.Timeout):
                return (index, None, "prompt_timeout")
            return (index, None, "prompt_error")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(_post_prompt, i, prompt): i
            for i, prompt in enumerate(prompts)
        }

        for future in as_completed(futures):
            index, result, error_label = future.result()
            with results_lock:
                results[index] = result
                if error_label != "success":
                    if log:
                        log(f"[call_swarm] prompt[{index}] {error_label}")

    return results


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
