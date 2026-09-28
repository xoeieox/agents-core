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
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import yaml
from requests.exceptions import Timeout, ConnectionError, HTTPError, ChunkedEncodingError

PACIFIC = ZoneInfo("America/Los_Angeles")

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")
GW_CREATIVE_URL = os.environ.get("GW_CREATIVE_URL", "http://203.0.113.11:8093")
QUEST_URL = os.environ.get("QUEST_URL", "http://203.0.113.11:8080")
SWARM_URL = os.environ.get("SWARM_URL", GW_URL)
SWARM_MAX_CONCURRENT = int(os.environ.get("SWARM_MAX_CONCURRENT", "4"))
# Local phala-test-key.service (--user unit, 127.0.0.1:8413) — an
# OpenAI-compatible loopback endpoint over agents_core.phala_tee.PhalaTeeClient.
# Sealed, non-Anthropic, zero-local-watts inference seat (agents-core-phala-gate-voicing-v0).
PHALA_URL = os.environ.get("PHALA_URL", "http://127.0.0.1:8413")


def _llamacpp_url() -> str:
    """Resolve the qwen-operator local-LLM endpoint.

    Read at call time (not module load) so tests can monkeypatch.setenv - same
    call-time-vs-import-time discipline as _gw_max_tokens_default(). Defaults to
    GW_URL (GravityWell), which is itself env-overridable; LOCAL_LLM_URL wins when
    set. StarHouse (the old bare-literal TAILSCALE_IP default) kernel-panicked
    2026-08-03 and is being held off deliberately - see
    agents-core-local-llm-gw-repoint-v0.
    """
    return os.environ.get("LOCAL_LLM_URL", GW_URL)


LLAMACPP_URL = _llamacpp_url()


def _local_llm_think_enabled() -> bool:
    """Whether the qwen-operator path should preserve GW's reasoning trace.

    Default off. Measured (finding 3/4, agents-core-local-llm-gw-repoint-v0): a
    brief-shaped prompt with the trace on ran 28.0s wall-clock and blew
    state_brief.py's 30s guard; suppressed via chat_template_kwargs it ran 5.4s -
    a 5.2x reduction. Every current qwen-operator consumer is a summarizer under a
    wall-clock guard, not a reasoner. Opt back in with LOCAL_LLM_THINK=1.
    """
    return os.environ.get("LOCAL_LLM_THINK", "0") == "1"

# flip-controller — sole mode/units/in-flight-flip oracle (gw-serving-state-resolver-v0).
FLIP_CONTROLLER_URL = os.environ.get("FLIP_CONTROLLER_URL", "http://203.0.113.10:8408")
GW_MODEL_REGISTRY_PATH = Path(__file__).parent / "data" / "gw_models.yaml"

# Generation guards (spec-review-gw-generation-guards-v0): an unbounded GW call
# can degenerate and run to the 300s gw-liveness hard ceiling before being culled.
# max_tokens bounds a single turn's length; repeat_penalty (llama.cpp only) breaks
# repetition loops before they start.
GW_MAX_TOKENS_DEFAULT = 4096
GW_REPEAT_PENALTY_DEFAULT = 1.1

# Marker appended to a salvaged culled-partial so callers/tests can detect
# degradation without changing the str|None return contract.
GW_DEGRADED_MARKER = "[gw-degraded:"


def _gw_max_tokens_default() -> int:
    """Read at call time (not module load) so tests can monkeypatch.setenv."""
    return int(os.environ.get("GW_MAX_TOKENS", str(GW_MAX_TOKENS_DEFAULT)))


def _gw_repeat_penalty_default() -> float:
    """Read at call time (not module load) so tests can monkeypatch.setenv."""
    return float(os.environ.get("GW_REPEAT_PENALTY", str(GW_REPEAT_PENALTY_DEFAULT)))


def _gw_degraded_marker(reason: str, elapsed: float, idle: float) -> str:
    return f"\n\n{GW_DEGRADED_MARKER} reason={reason} elapsed={elapsed:.1f}s idle={idle:.1f}s]"


def _is_gw_result_degraded(text) -> bool:
    """True if `text` is a GW result carrying the culled-partial marker (AC3/AC6)."""
    return isinstance(text, str) and GW_DEGRADED_MARKER in text


def _persist_gw_cull_partial(
    text: str, model: str, url: str, cull_reason: str, cull_elapsed: float, cull_idle: float,
) -> None:
    """Best-effort persistence of a culled GW stream's partial text (AC4).

    Today the evidence of a runaway generation is destroyed on cull. Writes the
    partial to a run artifact path so future runaways are inspectable. Never
    raises — an artifact-write failure must not affect the caller's result.
    """
    try:
        from agents_core.room_paths import room_path
        out_dir = room_path("council.gw_cull", write=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(PACIFIC).strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"gw-cull-{ts}-{uuid.uuid4().hex[:8]}.txt"
        out_path.write_text(
            f"model={model} url={url} reason={cull_reason} "
            f"elapsed={cull_elapsed:.1f}s idle={cull_idle:.1f}s\n\n{text}"
        )
    except Exception as e:
        _log.warning("[gw-liveness] failed to persist culled partial: %s", e)


# GW admission provenance vocabulary — all known tuples appended to _provenance_out.
#
#   admission_off_master_passthrough — enforce mode on a non-master node; request passed through.
#   admission_shadow:<decision>      — shadow mode dry-run result ("would-admit" or "would-wait").
#   admission_shadow:principal_group_collision_risk — shadow mode: unique work_id principal used.
#   drain_count_unavailable          — doorman honored acquire but drain_cleared absent (pre-atomic doorman); proceeding on elevator gate alone.
#   doorman_unreachable              — doorman acquire failed; routed to wake_fail.
#   fallback                         — _apply_wake_fail re-dispatched to a paid operator (haiku/sonnet/opus) after wake_fail; the effective_operator is the fallback policy, not "gravitywell".
#   gw_deferred_swarm                — doorman deferred to swarm; requeueing (precedence ladder).
#   gw_member_deadline               — per-member watchdog fired (AC2); ticket failed, lease released.
#   gw_member_error                  — unexpected exception from backend dispatch (AC1); ticket failed.
#   gw_not_serving                   — doorman responded not-serving; bounded backoff requeue (precedence ladder).
#   serving_http_error               — OperatorUnreachableError from backend HTTP layer.
#   slot_pool_down                   — GW slot pool unavailable (precedence ladder).
#   slot_queued_timeout              — wait deadline expired before admission (precedence ladder).
#   stream_culled                    — backend result carries the GW_DEGRADED_MARKER (a culled/salvaged partial stream), not a clean completion.
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

# doorman-lease-class-consumers-v0: sentinel distinguishing "caller omitted
# lease_class" from "caller explicitly passed 'deferrable'" - both resolve to
# the same effective class, but only the former should trip the once-per-
# process silence warning below. A plain str default can't make this
# distinction (Python binds the default before the function body runs).
_LEASE_CLASS_UNSET = object()
_LEASE_CLASS_DEFAULT = "deferrable"
_lease_class_default_warned: set[str] = set()


def _warn_lease_class_defaulted() -> None:
    """WARN once per calling module when call_operator's gravitywell path
    takes the default lease_class because the caller passed none.

    Once-per-process (not per-call): a busy night DAG calling an unclassed
    path repeatedly should not drown the log in an identical warning.
    """
    try:
        caller_frame = sys._getframe(2)
        calling_module = caller_frame.f_globals.get("__name__", "<unknown>")
    except ValueError:
        calling_module = "<unknown>"
    if calling_module in _lease_class_default_warned:
        return
    _lease_class_default_warned.add(calling_module)
    _log.warning(
        "[lease-class] call_operator invoked without lease_class from module=%s "
        "- defaulting to %r. Pass lease_class explicitly (protected for measured "
        "gates/interactive sessions, deferrable for background work).",
        calling_module, _LEASE_CLASS_DEFAULT,
    )


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


class PhalaOperatorUnavailable(OperatorUnreachableError):
    """Raised when the local phala-test-key (127.0.0.1:8413) cannot be reached.

    There is nothing to wake — unlike gravitywell, phala is a single sealed
    HTTP seat with no doorman, no lease, no cold-wake path — so unreachable
    means fail-closed, structurally, not policy-configurable. Never falls
    back to a paid Anthropic operator (agents-core-phala-gate-voicing-v0).
    """

    def __init__(self, url: str, last_error: Exception):
        self.url = url
        self.last_error = last_error
        Exception.__init__(
            self,
            f"[phala] unavailable at {url} after retries — no paid fallback was "
            f"attempted (phala has no wake path; fail-closed by design). "
            f"last_error={last_error}",
        )


class FlashnextLaneUnavailable(OperatorUnreachableError):
    """Raised when the registry-resolved flashnext gate lane cannot be voiced on.

    gate-lanes-registry-driven-flashnext-v0-agents-core (S2): the flashnext
    operator is a REGISTRY lane, so "unavailable" has two shapes and they are
    NOT interchangeable — ``reason`` carries which one this is:

      * ``"registry_blind"`` — the gw-seats registry is unreachable/malformed
        (no information). This is the ONLY shape whose caller may fall back to
        the legacy gravitywell path (GW_URL/SWARM_URL) byte-identically,
        because that is what a blind gate leg ran on before this target.
      * anything else (``"flashnext_not_serving"``, ``"flashnext_unreachable"``)
        — the registry is readable and the explicitly-requested lane is
        inactive. The caller records an honest ``leg_down`` and NEVER
        re-routes the leg to the gravitywell lane; a silent legacy fallback
        here is the "lying leg" the parent's re-gate fold kills.

    Never falls back to a paid operator: the lane is a local seat, so
    unreachable is fail-closed (phala precedent).
    """

    def __init__(self, url: str, reason: str, last_error: Exception | None = None):
        self.url = url
        self.reason = reason
        self.last_error = last_error
        # Exception.__init__ (not super()): OperatorUnreachableError's own
        # __init__ demands a last_error it would then describe as an
        # "unreachable after retries" — this error carries a lane STATE, not a
        # retry exhaustion (phala precedent).
        Exception.__init__(
            self,
            f"[flashnext] gate lane unavailable at {url!r} (reason={reason!r}) — "
            "no legacy or paid fallback was attempted (fail-closed by design; only "
            "reason='registry_blind' may fall back to the gravitywell path). "
            f"last_error={last_error}",
        )


class GWParkedError(OperatorUnreachableError):
    """Raised when GW cannot be woken and on_wake_fail='park' (fail-closed default).

    Per decision/independence-blueprint-ratified-2026-07-28 and Machine Rhythm ruling 6:
    "before any unattended loop runs, GW-unavailable must PARK, not degrade... a silent
    degrade is a 76x cost event." No paid fallback is attempted and no result is returned
    silently — the caller must retry once GravityWell is serving again.
    """

    def __init__(self, operator_class: str, url: str, last_error: Exception):
        self.operator_class = operator_class
        Exception.__init__(
            self,
            f"[gravitywell] parked: GW unavailable for operator_class={operator_class!r} "
            f"at {url} — no paid fallback was attempted (on_wake_fail='park'). "
            f"Retry once GravityWell is serving again. last_error={last_error}",
        )
        self.url = url
        self.last_error = last_error


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

OPERATOR_DEFAULTS: dict[str, str | None] = {
    "qwen":                 "qwen3.6-35b-a3b",
    "quest":                "quest-35b-rl",
    "sonnet":               "claude-sonnet-4-6",
    "opus":                 "claude-opus-4-7",
    "haiku":                "claude-haiku-4-5-20251001",
    "gravitywell":          "gravitywell-122b",
    "gravitywell-creative": "gravitywell-llama-70b",
    "phala":                "deepseek/deepseek-v4-flash-0731",
    # flashnext (gate-lanes-registry-driven-flashnext-v0-agents-core, S2): the
    # served id is REGISTRY-resolved at call time, never a literal — the value
    # stored here is a None marker and _OperatorDefaults resolves it through
    # agents_core.lane_registry on every read (the f0fb039 fixer_flash
    # precedent: served-id + backend resolved, never hardcoded). None-valued
    # also means "registry blind => operator unavailable": a read raises
    # KeyError rather than inventing a model id.
    "flashnext":            None,
}

# Operators whose OPERATOR_DEFAULTS entry resolves through the gw-seats
# registry at CALL time (mirrors the _gw_default_model() / corroboration
# _llm_url call-time-read precedent; a module-load read would freeze a seat
# that comes and goes with the GPU handover).
_CALL_TIME_RESOLVED_OPERATORS = ("flashnext",)


class _OperatorDefaults(dict):
    """OPERATOR_DEFAULTS with call-time resolution for the registry lanes.

    Keys/iteration/``in`` behave like the plain dict literal above (so
    ``"flashnext" in OPERATOR_DEFAULTS`` is True and the unknown-operator
    ValueError still lists it), but reading a call-time-resolved key goes
    through the gw-seats registry instead of a stored literal:

      * resolved lane -> the registry's served model id (the
        served-model-name pin).
      * registry blind / lane not serving -> KeyError (and ``.get`` returns
        its default). That is the honest "operator unavailable" shape: there
        is no model id to claim, so none is invented, and the caller decides
        what an unavailable lane means (blind -> legacy fallback; dead lane ->
        honest leg_down).
    """

    def __getitem__(self, key):
        if key in _CALL_TIME_RESOLVED_OPERATORS:
            model = _registry_resolved_model(key)
            if not model:
                raise KeyError(key)
            return model
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _registry_resolved_model(operator_class: str) -> str | None:
    """The registry-resolved served id for a call-time-resolved operator.

    Returns None when the registry is blind or the lane is not serving —
    never a guessed/hardcoded id. ``fetcher`` (lane_registry.lane_state /
    resolve_gate_lane) is the hermetic test seam.
    """
    from agents_core import lane_registry

    lane_obj, _reason = lane_registry.lane_state(lane=operator_class)
    return lane_obj.served_model if lane_obj is not None else None


OPERATOR_DEFAULTS = _OperatorDefaults(OPERATOR_DEFAULTS)


def _flashnext_lane() -> tuple[object, str]:
    """Resolve the flashnext gate lane at CALL time (S2).

    Returns (GateLane | None, reason) straight from
    ``agents_core.lane_registry.lane_state`` — see that function for the
    reason taxonomy ("" / "registry_blind" / "flashnext_not_serving").
    """
    from agents_core import lane_registry

    return lane_registry.lane_state(lane=lane_registry.FLASHNEXT_LANE_NAME)



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


def _gw_backend(discovered_model: str | None = None, owned_by: str | None = None) -> str:
    """Resolve the llama.cpp-vs-vLLM payload dialect.

    Precedence:
    1. Explicit GW_BACKEND always wins (set-but-unrecognized raises ValueError - a
       misconfigured switch must force a deliberate fix, never silently normalize).
    2. `owned_by` - the `owned_by` field the discovery probe's /v1/models response
       carries alongside the served model id - decides directly when recognized:
       "llamacpp" -> "llamacpp", "vllm" -> "vllm". The server names its own engine, so
       this is correct for every current and future llama.cpp/vLLM seat (today
       gravitywell-122b and gravitywell-v4flash both serve llama.cpp under different
       names) without a model-name allowlist to keep in sync.
    3. An unrecognized `owned_by` (a future engine string, or a garbled value) falls
       through to the literal comparison below and logs one WARNING naming the value -
       resolution must never raise here, this runs inside every GW call. Absent
       `owned_by` (older servers, or no discovery context) is not an anomaly and logs
       nothing; it falls to the same literal comparison: `discovered_model` equals
       OPERATOR_DEFAULTS["gravitywell"] ("gravitywell-122b") -> "llamacpp", else "vllm".
       This literal fallback is known-stale for any second/future llama.cpp seat that
       omits or garbles `owned_by` - it exists only as a safety net for those servers.
    4. With no explicit backend, no owned_by, and no discovered_model (e.g. called
       standalone with no call context), falls back to "llamacpp".
    """
    explicit = _gw_explicit_backend()
    if explicit is not None:
        return explicit
    if owned_by is not None:
        if owned_by == "llamacpp":
            return "llamacpp"
        if owned_by == "vllm":
            return "vllm"
        _log.warning(
            "[gravitywell] unrecognized owned_by=%r reported for model=%r; falling back "
            "to literal model-name comparison for backend dialect resolution",
            owned_by, discovered_model,
        )
    if discovered_model is not None:
        return "llamacpp" if discovered_model == OPERATOR_DEFAULTS["gravitywell"] else "vllm"
    return "llamacpp"


# Pre-flight serving-mode handshake cache (1d.1): (url, resolved-model) -> verified.
# Holds only a positive "verified" marker; elides the redundant startup /v1/models probe.
# It never suppresses the per-call response-echo check (_call_gravitywell_backend re-verifies
# model identity from the actual response on every call, unconditionally — see 1d.2).
_gw_handshake_cache: dict[tuple[str, str], bool] = {}
_gw_handshake_lock = threading.Lock()


def _gw_probe_served_model(
    url: str, timeout: int = 10, log=None, _owned_by_out: list | None = None
) -> str | None:
    """Pure transport: GET {url}/v1/models and return the served model id, or None on
    any failure (connect error, timeout, malformed response, empty data). No caching, no
    assertion - callers own both. Shared by _gw_verify_serving_mode (explicit-mode,
    process-lifetime cache, hard-fail-on-drift) and _gw_discover_serving (auto-detect,
    TTL-bounded cache, no assertion) so the two only differ in caching/assertion
    semantics, not in how they talk to GW.

    _owned_by_out: optional list to append the response entry's `owned_by` field to (the
    same response this already fetches, no second request) - agents-core-gw-backend-
    owned-by-resolver-v0 D1. None is appended when the field is absent, the entry is
    missing, or the probe fails, so a caller can distinguish "no owned_by" from "didn't
    ask". Omitted by default so existing callers are unaffected.
    """
    try:
        resp = requests.get(f"{url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        entry = data["data"][0] if data.get("data") else None
        if _owned_by_out is not None:
            _owned_by_out.append(entry.get("owned_by") if entry else None)
        return entry.get("id") if entry else None
    except Exception as e:
        if _owned_by_out is not None:
            _owned_by_out.append(None)
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


# TTL-bounded discovery cache (auto-detect case): url -> (served_model, owned_by, discovered_at).
# Deliberately NOT a process-lifetime cache like _gw_handshake_cache above - a long-lived
# process (claude-queue-runner above all) must still notice a genuine mode flip within a
# bounded window. See _gw_discover_serving() for the full rationale.
_gw_discovery_cache: dict[str, tuple[str, str | None, float]] = {}
GW_DISCOVERY_TTL_S_DEFAULT = 30.0


def _gw_discovery_ttl_s() -> float:
    """Read at call time (not module load) - matches every other tunable in this file
    (GW_IDLE_GAP_SECS, GW_FIRST_TOKEN_GAP_SECS, etc.), and lets tests override via
    monkeypatch.setenv."""
    return float(os.environ.get("GW_DISCOVERY_TTL_S", str(GW_DISCOVERY_TTL_S_DEFAULT)))


def _gw_discover_serving(url: str, log=None, _owned_by_out: list | None = None) -> str | None:
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

    _owned_by_out: optional list to append the discovered `owned_by` value to (cached
    alongside the model id, so a TTL-hit reuses it too - no extra HTTP request either way;
    agents-core-gw-backend-owned-by-resolver-v0 D1).
    """
    now = time.monotonic()
    with _gw_handshake_lock:
        cached = _gw_discovery_cache.get(url)
        if cached is not None and (now - cached[2]) < _gw_discovery_ttl_s():
            if _owned_by_out is not None:
                _owned_by_out.append(cached[1])
            return cached[0]
    probe_owned_by: list = []
    served = _gw_probe_served_model(url, log=log, _owned_by_out=probe_owned_by)
    owned_by = probe_owned_by[0] if probe_owned_by else None
    if served is not None:
        with _gw_handshake_lock:
            _gw_discovery_cache[url] = (served, owned_by, now)
    if _owned_by_out is not None:
        _owned_by_out.append(owned_by if served is not None else None)
    return served


def _gw_default_model(url: str = None, log=None, _owned_by_out: list | None = None) -> str:
    """Resolve the default gravitywell model name, read at call time (not module-load).

    Explicit GW_MODEL always wins. Otherwise, explicit GW_BACKEND=vllm -> "gravitywell-27b",
    GW_BACKEND=llamacpp -> OPERATOR_DEFAULTS["gravitywell"] ("gravitywell-122b"). With
    neither set, auto-detects by asking `url` (or GW_URL) what it is currently serving via
    _gw_discover_serving() - correct whether GW is resting in big (122B) or dual (27B), and
    immune to the boot-default posture changing again without a code edit. Falls back to
    OPERATOR_DEFAULTS["gravitywell"] if the discovery probe fails (GW unreachable); the
    subsequent real call then fails to connect too, surfacing as OperatorUnreachableError.

    _owned_by_out: optional list to append the discovered `owned_by` value to (only
    populated on the auto-detect path; empty when GW_MODEL/GW_BACKEND are explicit, since
    no discovery ran) - _call_gravitywell_backend threads this into _gw_backend() so the
    backend dialect resolves from the server's own self-report, not a model-name literal
    (agents-core-gw-backend-owned-by-resolver-v0 D1).
    """
    explicit_model = _gw_explicit_model()
    if explicit_model:
        return explicit_model
    explicit_backend = _gw_explicit_backend()
    if explicit_backend is not None:
        return "gravitywell-27b" if explicit_backend == "vllm" else OPERATOR_DEFAULTS["gravitywell"]
    discovered = _gw_discover_serving(url or GW_URL, log=log, _owned_by_out=_owned_by_out)
    return discovered if discovered is not None else OPERATOR_DEFAULTS["gravitywell"]


def _call_qwen_backend(prompt: str, system: str = None, timeout: int = 600,
                       json_mode: bool = False, temperature: float = 0.7,
                       log=None, bundle_ids: list[str] = None) -> str | None:
    """Send a completion request to the local llama-server (Qwen endpoint).

    Context selection priority:
    1. Explicit system= override (task-specific prompts)
    2. Explicit bundle_ids= (chub bundles by ID)
    3. Neither given: no system context (empty messages list stays user-only)

    chub_broker is imported only when bundle_ids= is explicitly passed — this
    is the sole non-lazy runtime dep agents_core/CLAUDE.md permits ("No hard
    deps on /srv/agents/scripts/ ... imported only when bundle_ids= is used.
    Don't add more."). There is no implicit default-bundle fallback: a caller
    that wants no context passes nothing (system stays None); a caller that
    wants a specific bundle passes bundle_ids= explicitly
    (cr-bundle-item-agents-core-1a6e095197).

    Returns the response text, or None on failure.
    """
    messages = []

    # System prompt: explicit system= > explicit bundle_ids= > none.
    if system is not None:
        sys_prompt = system
    elif bundle_ids is not None:
        from chub_broker import select_bundles_by_ids
        sel = select_bundles_by_ids(bundle_ids)
        sys_prompt = sel.composed
    else:
        sys_prompt = None

    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": temperature,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": _local_llm_think_enabled()},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    max_retries = 3
    call_start = time.monotonic()
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{_llamacpp_url()}/v1/chat/completions",
                json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
            return text if text.strip() else None
        except (requests.exceptions.HTTPError,
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                # Budget-aware backoff: never sleep past the caller's remaining
                # `timeout` wall-clock budget (state_brief.py's 30s guard abandons
                # the executor mid-first-backoff otherwise - finding 6,
                # agents-core-local-llm-gw-repoint-v0). Control flow (which
                # exceptions retry vs raise vs swallow) is unchanged.
                remaining = timeout - (time.monotonic() - call_start)
                backoff = min(10 * (2 ** attempt), max(0.0, remaining))
                if backoff <= 0:
                    if log:
                        log(f"LLM call failed (attempt {attempt + 1}/{max_retries}), "
                            f"no budget left for retry: {e}")
                    raise OperatorUnreachableError(_llamacpp_url(), e)
                if log:
                    log(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(backoff)
            else:
                if log:
                    log(f"LLM call failed after {max_retries} attempts: {e}")
                raise OperatorUnreachableError(_llamacpp_url(), e)
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
    max_tokens: int | None = None,
) -> str | None:
    """Shared POST core for OpenAI-compatible chat/completions endpoints.

    Posts to {base_url}/v1/chat/completions with messages and optional chat_template_kwargs.
    Retries up to max_retries on transient errors (timeout, connection, chunked encoding).
    Returns response text on success, None on parse errors, raises OperatorUnreachableError
    on persistent HTTP/network failures.

    think=False (default) sends chat_template_kwargs={"enable_thinking": false} explicitly,
    matching _call_gravitywell_backend's pattern. _no_thinking=True structurally omits
    chat_template_kwargs entirely, for models that do not support the thinking knob.

    max_tokens bounds generation length (env-overridable default via GW_MAX_TOKENS,
    per spec-review-gw-generation-guards-v0) — pass an explicit value to override.

    json_mode=True degrade (agents-core-post-chat-json-object-degrade-v0): some
    OpenAI-compat engines reject response_format:{"type":"json_object"} with HTTP 400.
    On the first such 400, response_format is stripped and the request is retried once
    with a plain-text payload — this never alters the first (json_object) request, so
    the happy path stays byte-identical. The loop reserves one extra "breath" slot
    beyond max_retries so the degrade is guaranteed to fire even if the 400 lands on
    the final transient attempt; the degraded payload can still use its own max_retries
    budget for further transient errors. Non-400 errors and json_mode=False are
    unaffected. The degraded response is returned as-is — this core does not validate
    its JSON-ness; that is the downstream caller's job.

    Used by _call_gravitywell_backend, call_swarm, and other chat-completion callers.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens if max_tokens is not None else _gw_max_tokens_default(),
    }
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = chat_template_kwargs
    elif not _no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": think}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    degrade_used = False
    for attempt in range(max_retries + 1):
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
            if (not degrade_used and json_mode
                    and isinstance(e, requests.exceptions.HTTPError)
                    and e.response is not None and e.response.status_code == 400
                    and "response_format" in payload):
                degrade_used = True
                payload = {k: v for k, v in payload.items() if k != "response_format"}
                _log.warning(
                    "response_format:json_object rejected (HTTP 400) by %s; "
                    "retrying WITHOUT response_format - downstream may receive "
                    "unstructured text", base_url,
                )
                continue
            # last_attempt is the boundary before the terminal raise. It is
            # max_retries - 1 normally, but shifts out by one once the degrade
            # has consumed its reserved "extra breath" slot, so the degraded
            # payload still gets its own full max_retries budget.
            last_attempt = max_retries if degrade_used else max_retries - 1
            if attempt < last_attempt:
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
        if content_parts:
            text = "".join(content_parts)
        elif reasoning_parts:
            text = "".join(reasoning_parts) + _gw_degraded_marker(
                "reasoning_only_no_content", time.monotonic() - call_start, 0.0
            )
        else:
            text = ""
        return (text if text.strip() else None, None, _state["served_model"])

    if _state["cull"]:
        # Only hard_ceiling_exceeded is salvageable (AC3/B) - idle_gap_exceeded gets one
        # stall retry at the _call_gravitywell_backend layer and first_token_grace_exceeded
        # means no token ever arrived, so both keep their existing text=None contract.
        if _state["cull"][0] == "hard_ceiling_exceeded":
            text = "".join(content_parts) or "".join(reasoning_parts)
            return (text if text.strip() else None, _state["cull"], _state["served_model"])
        return (None, _state["cull"], _state["served_model"])

    # Stream ended without [DONE] and no cull - return what we have
    if content_parts:
        text = "".join(content_parts)
    elif reasoning_parts:
        text = "".join(reasoning_parts) + _gw_degraded_marker(
            "reasoning_only_no_content", time.monotonic() - call_start, 0.0
        )
    else:
        text = ""
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
    max_tokens: int | None = None,
) -> str | None:
    """Send a completion request to a GravityWell endpoint via streaming SSE.

    By default targets GW_URL (:8081) with the model resolved by _gw_default_model()
    (GW_MODEL / GW_BACKEND env-driven when set; when both are unset, auto-detects the
    currently-served model via _gw_discover_serving() instead of assuming a fixed
    default). Internal _url/_model params route to alternate endpoints (e.g. the creative
    Llama-70B at :8093) without exposing that routing on the public 122B operator path.

    The default GW path (_url is None and _model is None) only:
    - Payload dialect gates on GW_BACKEND when explicit ("llamacpp" byte-identical to
      today; "vllm" omits the llama.cpp-only cache_prompt field), or - when GW_BACKEND/
      GW_MODEL are both unset - on the `owned_by` field the discovery probe's response
      already carried alongside the auto-detected model (falling back to a literal
      model-name comparison only if `owned_by` is absent or unrecognized) - see
      _gw_backend().
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
    max_tokens: bounds generation length (env-overridable default via GW_MAX_TOKENS,
    default 4096 — spec-review-gw-generation-guards-v0). Always present in the payload;
    pass an explicit value to override the default for this call.

    Dual-timer liveness model:
    - Phase 1 (pre-first-token): cull after GW_FIRST_TOKEN_GAP_SECS (default 600).
    - Phase 2 (post-first-token): cull after GW_IDLE_GAP_SECS (default 45) of chunk silence.
    - Hard ceiling: GW_LIVENESS_HARD_CEILING_SECS (default 1800) total.
    - Stall retry: if idle_gap_exceeded, retry once within the same ceiling budget.
    - On hard_ceiling_exceeded, the accumulated partial is salvaged and returned with a
      GW_DEGRADED_MARKER suffix (and persisted to a run artifact) instead of dropping the
      voice entirely — a degenerate turn completes-with-partial rather than empty
      (spec-review-gw-generation-guards-v0, AC3/AC4). Other cull reasons (no token ever
      arrived) still return None.

    Connection-level retry: 3 attempts, 2s/4s backoff on network errors.
    Persistent errors raise OperatorUnreachableError; parse errors return None.
    Serving-mode drift raises GWServingModeMismatchError (never caught by on_wake_fail).
    """
    url = _url if _url is not None else GW_URL
    is_default_gw_path = _url is None and _model is None
    is_auto_detecting = (
        is_default_gw_path and _gw_explicit_model() is None and _gw_explicit_backend() is None
    )
    if _model is not None:
        model = _model
        discovered_owned_by = None
    else:
        _owned_by_out: list = []
        model = _gw_default_model(url=url, log=log, _owned_by_out=_owned_by_out)
        discovered_owned_by = _owned_by_out[0] if _owned_by_out else None

    if is_default_gw_path:
        if not is_auto_detecting:
            _gw_verify_serving_mode(url, model, log=log)
        backend = _gw_backend(
            discovered_model=model if is_auto_detecting else None,
            owned_by=discovered_owned_by if is_auto_detecting else None,
        )
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
        "max_tokens": max_tokens if max_tokens is not None else _gw_max_tokens_default(),
    }
    if backend == "llamacpp":
        payload["cache_prompt"] = True
        payload["repeat_penalty"] = _gw_repeat_penalty_default()
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
        if cull_reason == "hard_ceiling_exceeded" and text:
            # Salvage: a runaway turn completes-with-partial instead of empty (AC3).
            _persist_gw_cull_partial(text, model, url, cull_reason, cull_elapsed, cull_idle)
            return text + _gw_degraded_marker(cull_reason, cull_elapsed, cull_idle)
        # No tokens ever arrived (or a non-hard-ceiling cull) - nothing to salvage.
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
      "park" — raise GWParkedError; no paid fallback is attempted and nothing is
          returned silently (fail-closed default per
          decision/independence-blueprint-ratified-2026-07-28, Machine Rhythm ruling 6:
          "GW-unavailable must PARK, not degrade").

    If the fallback call_operator() itself raises, the exception propagates unchanged.

    _provenance_out: optional list to append (operator_class, reason) tuples for tracking
                     which operator actually answered (used by adapters for observability).
    """
    policy = on_wake_fail or "skip"

    if policy == "park":
        raise GWParkedError(
            operator_class,
            GW_URL,
            Exception(f"GW wake_failed and on_wake_fail='park' for {operator_class!r}"),
        )

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


def _call_operator_impl(operator_class: str, prompt: str, model: str = None,
                       _provenance_out: list | None = None,
                       principal: str | None = None,
                       lease_class: str = _LEASE_CLASS_UNSET,
                       _admission_bypass: bool = False,
                       **kwargs) -> str | None:
    """Route a completion request to the appropriate backend operator.

    operator_class ∈ {"qwen", "quest", "sonnet", "opus", "haiku", "gravitywell",
    "gravitywell-creative", "phala", "flashnext"}. Raises ValueError for unknown classes.

    Default models:
        qwen                 → "qwen3.6-35b-a3b"
        quest                → "quest-35b-rl"       (vLLM swarm, QUEST_URL :8080, OpenAI-compat)
        sonnet               → "claude-sonnet-4-6"
        opus                 → "claude-opus-4-7"
        haiku                → "claude-haiku-4-5-20251001"
        gravitywell          → "gravitywell-122b"   (122B reasoning, :8081, DoormanClient)
        gravitywell-creative → "gravitywell-llama-70b" (Llama-70B instruct, :8093, direct)
        phala                → "deepseek/deepseek-v4-flash-0731" (sealed TEE seat,
                                PHALA_URL :8413, OpenAI-compat, direct)
        flashnext            → registry-reserved (None in the table): the served id
                                AND the base_url resolve through the gw-seats registry
                                at call time (agents_core.lane_registry), never a
                                hardcoded string.

    qwen routes via the local llama-server (same path as call_llm()).

    sonnet / opus / haiku route via ClaudeQueue → `claude -p` (Max subscription
    path). The task is submitted to ClaudeQueue with task_type="llm_call";
    `submit_and_wait` blocks the caller's thread until the daemon's
    `_run_llm_call_task` handler completes the task and writes the output file.
    No direct Anthropic-API calls — kill-switched per
    decision/no-anthropic-api-direct.

    gravitywell routes via the doorman to the GravityWell llama.cpp endpoint (:8081).
    On unreachable, falls back per on_wake_fail policy.

    lease_class: foreground-priority gate class for the doorman lease this call
    acquires (gw-router-phase1-foreground-gate / doorman-lease-class-consumers-v0).
    "protected" for measured gates and interactive sessions (never deferred);
    "deferrable" (the default) for background/worker callers. Only meaningful
    for operator_class="gravitywell" - ignored by every other operator, since
    only the gravitywell path takes a doorman lease. Do NOT infer this from
    principal/operator_class/reason; pass it explicitly. Omitting it emits a
    WARN once per calling module (not per call) naming the module, so an
    unclassed caller is discoverable without trawling the doorman log - the
    default is a safety net, not a declaration.

    gravitywell also accepts acquire_lease: bool = True (via kwargs). If False,
    skip doorman lease acquisition entirely - no acquire, no enforce/shadow
    admission, no release - and dispatch straight to the backend. For callers
    that already hold a GravityWell mode-controller lease (e.g. a gate.flip
    node), so a plain call_operator() call doesn't self-deadlock against their
    own lease. With defaults (True), behavior is byte-identical. Opt-in trust
    contract identical to call_gw_agent(acquire_lease=False) - the caller's word
    is not verified against actual lease ownership.

    gravitywell-creative routes directly to the Llama-70B endpoint (:8093, GW_CREATIVE_URL).
    No DoormanClient, no admission control, no wake_fail fallback. Raises
    CreativeOperatorUnavailable on network failure — never silently falls back.

    phala routes to the local phala-test-key.service (:8413, PHALA_URL) — a sealed,
    non-Anthropic TEE inference seat over PhalaTeeClient. No DoormanClient, no lease,
    no admission control, and (unlike gravitywell/quest) no wake-fail fallback of any
    kind: there is nothing to wake, so unreachable is always fail-closed
    (PhalaOperatorUnavailable, never a paid Anthropic escalation). Unlike quest/gravitywell,
    `model=` overrides are accepted and passed through — Phala fronts a live swappable
    model catalog, not a single pinned weight set, so refusing a swap here would fight
    the seat's actual design (agents-core-phala-gate-voicing-v0).

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
        # No single-fixed-model guard here (agents-core-local-llm-gw-repoint-v0):
        # the backend is GravityWell/vLLM now, not the old llama.cpp box, and
        # OPERATOR_DEFAULTS["qwen"] ("qwen3.6-35b-a3b") is a name GW does not
        # serve (confirmed live: posting it 404s). _call_qwen_backend never sends
        # a `model` field at all (finding 1: the unmodified llama.cpp-shaped
        # payload works against vLLM unchanged, which omits `model`), so `model`
        # is accepted for API compatibility and otherwise ignored rather than
        # validated against a dead constant.
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
                "system", "timeout", "json_mode", "temperature", "log", "max_tokens",
                "_served_model_out",
            ) if k in kwargs
        }
        think = kwargs.get("think", False)
        on_wake_fail = kwargs.get("on_wake_fail", "skip")
        timeout = int(kwargs.get("timeout", 300))
        acquire_lease = kwargs.get("acquire_lease", True)
        if acquire_lease:
            if lease_class is _LEASE_CLASS_UNSET:
                _warn_lease_class_defaulted()
                effective_lease_class = _LEASE_CLASS_DEFAULT
            else:
                effective_lease_class = lease_class
        else:
            # acquire_lease=False takes no lease at all - nothing to classify (C2b).
            effective_lease_class = None
        work_id = f"op-gravitywell-{uuid.uuid4().hex}"
        wake_fail_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("on_wake_fail", "think", "bundle_ids", "_provenance_out", "acquire_lease")
        }

        # --- LEASE-HOLDER BYPASS: caller already owns a GW mode-controller lease,
        # so doorman admission would self-deadlock. Skip acquire/enforce/shadow
        # entirely (regardless of GW_ADMISSION_MODE) and dispatch straight to the
        # backend. Opt-in trust contract, mirrors call_gw_agent(acquire_lease=False)
        # - no lease-ownership check is performed here; see gw_agent.py:953.
        if not acquire_lease:
            try:
                result = _call_gravitywell_backend(prompt=prompt, think=think, **gw_kwargs)
                if _provenance_out is not None:
                    _provenance_out.append(
                        ("stream_culled", "gravitywell")
                        if _is_gw_result_degraded(result)
                        else ("success", "gravitywell")
                    )
                return result
            except OperatorUnreachableError:
                if _provenance_out is not None:
                    _provenance_out.append(("serving_http_error", "gravitywell"))
                return _apply_wake_fail(on_wake_fail, operator_class, prompt,
                                       _provenance_out=_provenance_out, **wake_fail_kwargs)

        from agents_core.doorman_client import (
            DoormanClient,
            DoormanUnreachable,
            _gw_acquire_timeout,
            is_creative_occupied,
            is_flashnext_occupied,
        )

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
                # Leg 1: per-branch ceilings for the two branches that used to hold the
                # claim across an unbounded sleep, modelled on AC8's wf_retries/_max_wf.
                _max_ct = int(os.environ.get("GW_ADMISSION_MAX_CONTENDED_RETRIES", "5"))
                _max_se = int(os.environ.get("GW_ADMISSION_MAX_SOFT_ERROR_RETRIES", "5"))

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
                ct_retries = 0
                se_retries = 0
                _loop_ticket_settled = False
                deadline = time.monotonic() + _max_wait
                client = DoormanClient()

                def _release_lease_swallow_unreachable(_work_id):
                    # Leg 2: a failed release must never convert a retry into a crash —
                    # the lease TTL + doorman _gc_stale remain the backstop. Catch only
                    # the declared transport failure so a real logic fault still surfaces.
                    try:
                        client.release("gravitywell", _work_id)
                    except DoormanUnreachable as _release_err:
                        _log.warning(
                            "[gw-admission] lease_release_failed work_id=%s: %s",
                            _work_id, _release_err,
                        )

                if _provenance_out is not None and is_unique_work_id_principal:
                    # Leg 4 / AC10: emit under enforce too, not just shadow.
                    _provenance_out.append(
                        ("admission_enforce:principal_group_collision_risk", "gravitywell")
                    )

                try:
                    while True:
                        if time.monotonic() >= deadline:
                            _log.warning(
                                "[gw-admission] slot_queued_timeout principal=%r max_wait=%ss",
                                effective_principal, _max_wait,
                            )
                            if _provenance_out is not None:
                                _provenance_out.append(("slot_queued_timeout", "gravitywell"))
                            elevator.fail(ticket, reason="slot_queued_timeout")
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
                                lease_class=effective_lease_class,
                            )
                        except DoormanUnreachable:
                            # AC5b: transport failure — fail-ticket, not loud-proceed (would reopen race).
                            if _provenance_out is not None:
                                _provenance_out.append(("doorman_unreachable", "gravitywell"))
                            elevator.fail(ticket, reason="doorman_unreachable")
                            _loop_ticket_settled = True
                            return _apply_wake_fail(
                                on_wake_fail, operator_class, prompt,
                                _provenance_out=_provenance_out, **wake_fail_kwargs,
                            )
                        except Exception as _acquire_err:
                            # Leg 1: acquire_soft_error (e.g. HTTP 5xx) was previously a bare
                            # sleep-and-continue that held the claim indefinitely. Now mirrors
                            # AC8's wake_failed pattern — release the claim before sleeping,
                            # with its own retry counter and exponential backoff ceiling.
                            _log.warning(
                                "[gw-admission] acquire_soft_error work_id=%s: %s",
                                work_id, _acquire_err,
                            )
                            if _provenance_out is not None:
                                _provenance_out.append(("acquire_soft_error", "gravitywell"))
                            if se_retries >= _max_se:
                                elevator.fail(ticket, reason="acquire_soft_error_ceiling")
                                _loop_ticket_settled = True
                                return _apply_wake_fail(
                                    on_wake_fail, operator_class, prompt,
                                    _provenance_out=_provenance_out, **wake_fail_kwargs,
                                )
                            backoff = min(2 ** se_retries, 16)
                            se_retries += 1
                            elevator.requeue(ticket, reason="acquire_soft_error_retry")
                            admitted = False
                            is_ride_along = False
                            time.sleep(backoff)
                            continue

                        if is_flashnext_occupied(res) or is_creative_occupied(res):
                            # S3 dict-side named outcome
                            # (doorman-flashnext-serving-admission-v0). Pre-S1 this
                            # refusal raised httpx.HTTPStatusError out of acquire() and
                            # landed in the acquire_soft_error branch above, which re-paid
                            # the whole admission loop up to _max_se times (the measured
                            # 6-probe soft-retry re-pay loop this unit kills: a named
                            # outcome is not a soft error). It is also NOT retryable
                            # in-run: the flash-next handover window does not self-clear
                            # while the seat is resident, and the creative collider is a
                            # different seat — so settle the ticket once and NAME the
                            # state instead of sleeping on it.
                            _seat_state = (
                                "gw_flashnext_window" if is_flashnext_occupied(res)
                                else "gw_seat_occupied"
                            )
                            _log.warning(
                                "[gw-admission] acquire refused work_id=%s state=%s "
                                "(no in-run retry: the seat refusal does not self-clear)",
                                work_id, _seat_state,
                            )
                            if _provenance_out is not None:
                                _provenance_out.append((_seat_state, "gravitywell"))
                            elevator.fail(ticket, reason=_seat_state)
                            _loop_ticket_settled = True
                            return _apply_wake_fail(
                                on_wake_fail, operator_class, prompt,
                                _provenance_out=_provenance_out, **wake_fail_kwargs,
                            )

                        if DoormanClient.is_contended(res):
                            # Leg 1: drain-gate contended was previously a bare sleep-and-continue
                            # that held the claim indefinitely (root cause of the two-principal
                            # deadlock — see the spec). Now releases the claim before sleeping,
                            # with its own retry counter and exponential backoff ceiling.
                            if _provenance_out is not None:
                                _provenance_out.append(("gw_contended", "gravitywell"))
                            if ct_retries >= _max_ct:
                                elevator.fail(ticket, reason="gw_contended_ceiling")
                                _loop_ticket_settled = True
                                return _apply_wake_fail(
                                    on_wake_fail, operator_class, prompt,
                                    _provenance_out=_provenance_out, **wake_fail_kwargs,
                                )
                            backoff = min(2 ** ct_retries, 16)
                            ct_retries += 1
                            elevator.requeue(ticket, reason="gw_contended_retry")
                            admitted = False
                            is_ride_along = False
                            time.sleep(backoff)
                            continue

                        if DoormanClient.is_deferred(res):
                            # AC6: requeue and continue waiting (unchanged except Leg 2's release).
                            _release_lease_swallow_unreachable(work_id)
                            elevator.requeue(ticket, reason="gw_deferred_swarm")
                            admitted = False
                            is_ride_along = False
                            if _provenance_out is not None:
                                _provenance_out.append(("gw_deferred_swarm", "gravitywell"))
                            time.sleep(_poll)
                            continue

                        elif res.get("status") != "serving":
                            # AC8: wake_failed - bounded backoff requeue (unchanged except Leg 2's release).
                            _release_lease_swallow_unreachable(work_id)
                            if _provenance_out is not None:
                                _provenance_out.append(("gw_not_serving", "gravitywell"))
                            if wf_retries >= _max_wf:
                                elevator.fail(ticket, reason="gw_not_serving_ceiling")
                                _loop_ticket_settled = True
                                return _apply_wake_fail(
                                    on_wake_fail, operator_class, prompt,
                                    _provenance_out=_provenance_out, **wake_fail_kwargs,
                                )
                            backoff = min(2 ** wf_retries, 16)
                            wf_retries += 1
                            elevator.requeue(ticket, reason="gw_not_serving_retry")
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

                            # Leg 3: re-validate the ticket before dispatch. A reaper may have
                            # requeued it behind our back (elevator.py's stale-claim reclaim) between
                            # try_admit and here; dispatching while holding no claim would silently
                            # invert the admission invariant. Lost the race -> don't dispatch,
                            # release the just-acquired doorman lease, and re-admit from the top.
                            _current_ticket = elevator.get(ticket)
                            if (
                                _current_ticket is None
                                or _current_ticket.get("status") != "claimed"
                                or _current_ticket.get("claim_owner") != effective_principal
                            ):
                                if _provenance_out is not None:
                                    _provenance_out.append(
                                        ("gw_claim_lost_before_dispatch", "gravitywell")
                                    )
                                _release_lease_swallow_unreachable(work_id)
                                admitted = False
                                is_ride_along = False
                                continue

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
                                    elevator.fail(ticket, reason="gw_member_deadline")
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
                                    elevator.fail(ticket, reason="serving_http_error")
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
                                        elevator.fail(ticket, reason="gw_member_error")
                                        ticket_settled = True
                                        _loop_ticket_settled = True
                                    raise
                                if _provenance_out is not None:
                                    _provenance_out.append(
                                        ("stream_culled", "gravitywell")
                                        if _is_gw_result_degraded(result)
                                        else ("success", "gravitywell")
                                    )
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
                            elevator.fail(ticket, reason="gw_admission_loop_aborted")
                        except Exception:
                            pass
                    client.close()
                    elevator.close()

        # --- DIRECT DISPATCH (off, shadow, bypass, off-master) ---
        client = DoormanClient()
        try:
            res = client.acquire(
                "gravitywell", work_id, ttl_sec=timeout + 60, reason="call_operator",
                timeout=_gw_acquire_timeout(), principal=effective_principal,
                lease_class=effective_lease_class,
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
                    _provenance_out.append(
                        ("stream_culled", "gravitywell")
                        if _is_gw_result_degraded(result)
                        else ("success", "gravitywell")
                    )
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

    if operator_class == "phala":
        # No fixed-model guard (deliberate divergence from quest/gravitywell — see
        # docstring above): Phala fronts a swappable catalog, so a model= override
        # is accepted and falls back to the default when unset.
        resolved_model = model or OPERATOR_DEFAULTS["phala"]
        phala_kwargs = {
            k: kwargs[k] for k in ("system", "timeout", "json_mode", "temperature", "log")
            if k in kwargs
        }
        try:
            result = _post_chat_completion(
                base_url=PHALA_URL,
                model=resolved_model,
                messages=(
                    ([{"role": "system", "content": phala_kwargs["system"]}]
                     if phala_kwargs.get("system") else [])
                    + [{"role": "user", "content": prompt}]
                ),
                timeout=int(phala_kwargs.get("timeout", 300)),
                json_mode=bool(phala_kwargs.get("json_mode", False)),
                temperature=float(phala_kwargs.get("temperature", 0.7)),
                log=phala_kwargs.get("log"),
                _no_thinking=True,
            )
        except OperatorUnreachableError as exc:
            if _provenance_out is not None:
                _provenance_out.append(("serving_http_error", "phala"))
            raise PhalaOperatorUnavailable(PHALA_URL, exc) from exc
        if _provenance_out is not None:
            _provenance_out.append(("success", "phala"))
        return result

    if operator_class == "flashnext":
        # S2 (gate-lanes-registry-driven-flashnext-v0-agents-core): the
        # flashnext lane resolves through the gw-seats registry AT CALL TIME
        # — base_url and served id both come from the registry row, never a
        # hardcoded string (the f0fb039 fixer_flash precedent). Registry-blind
        # is reported as reason="registry_blind" and is the ONLY shape whose
        # caller may fall back to the legacy gravitywell path; a readable
        # registry with the lane down raises a non-blind reason, which the
        # caller must surface as an honest leg_down, never a silent legacy
        # fallback (the "lying leg"). No doorman lease: the seat is the
        # flash-next sglang box, leased by the doorman's flashnext window, not
        # by this call site. No paid fallback of any kind (phala precedent).
        #
        # ``_lane`` (optional): either a caller-resolved lane (a GateLane-shaped
        # object with base_url/served_model — the council adapter passes the
        # lane it was constructed with, S4, so the endpoint it was built for is
        # exactly the endpoint it dials) or the (lane_obj, reason) pair the
        # call_operator wrapper already resolved for the locality record. Both
        # shapes skip the registry re-read; every other caller leaves it unset
        # and gets the call-time registry read.
        lane_arg = kwargs.get("_lane")
        if isinstance(lane_arg, tuple) and len(lane_arg) == 2:
            resolved, reason = lane_arg
        elif lane_arg is not None and isinstance(getattr(lane_arg, "base_url", None), str):
            resolved, reason = lane_arg, ""
        else:
            resolved, reason = _flashnext_lane()
        if resolved is None:
            # PM-review fold (fix 4): the default reason for a None lane is
            # "flashnext_unavailable", NEVER "registry_blind". registry_blind
            # is the caller-licensed key for the legacy gravitywell fallback,
            # so inventing it when the resolver gave no reason would hand a
            # caller permission to silently re-route a lane that may simply be
            # down. An absent reason is fail-closed (provenance already
            # defaults the same way one line above — the two must not disagree).
            if _provenance_out is not None:
                _provenance_out.append((reason or "flashnext_unavailable", "flashnext"))
            raise FlashnextLaneUnavailable("", reason or "flashnext_unavailable")
        resolved_model = resolved.served_model or swarm_model(resolved.base_url)
        if not resolved_model:
            if _provenance_out is not None:
                _provenance_out.append(("flashnext_not_serving", "flashnext"))
            raise FlashnextLaneUnavailable(
                resolved.base_url, f"{resolved.name}_not_serving"
            )
        if model is not None and model != resolved_model:
            raise ValueError(
                f"call_operator(operator_class='flashnext', model={model!r}): the "
                f"registry-resolved lane serves a single fixed model "
                f"({resolved_model!r}); model swaps are an infrastructure operation "
                "(the seat's serve script), not a per-call parameter. Either pass "
                "model=None to use the registry pin, or do the swap out-of-band."
            )
        # max_tokens is on the allowlist (PM-review fold, fix 2): wave-mode
        # seats pass WAVE_SEAT_MAX_TOKENS (500) through the adapter, and the
        # D6 compounding-prefill cap only exists if the lane path forwards it.
        # Dropping it here silently re-inflated every flashnext-voiced wave
        # seat to the GW_MAX_TOKENS default (4096) — the exact cost shape D6
        # exists to prevent. Unset stays unset (None -> _post_chat_completion's
        # own env-overridable default, byte-identical to the pre-fold shape).
        fx_kwargs = {
            k: kwargs[k] for k in (
                "system", "timeout", "json_mode", "temperature", "log", "max_tokens"
            ) if k in kwargs
        }
        try:
            result = _post_chat_completion(
                base_url=resolved.base_url,
                model=resolved_model,
                messages=(
                    ([{"role": "system", "content": fx_kwargs["system"]}]
                     if fx_kwargs.get("system") else [])
                    + [{"role": "user", "content": prompt}]
                ),
                timeout=int(fx_kwargs.get("timeout", 300)),
                json_mode=bool(fx_kwargs.get("json_mode", False)),
                temperature=float(fx_kwargs.get("temperature", 0.7)),
                log=fx_kwargs.get("log"),
                max_tokens=(int(fx_kwargs["max_tokens"])
                            if fx_kwargs.get("max_tokens") is not None else None),
                _no_thinking=True,
            )
        except OperatorUnreachableError as exc:
            if _provenance_out is not None:
                _provenance_out.append(("serving_http_error", "flashnext"))
            raise FlashnextLaneUnavailable(
                resolved.base_url, f"{resolved.name}_unreachable", exc
            ) from exc
        if _provenance_out is not None:
            _provenance_out.append(("success", "flashnext"))
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


# Locality ledger chokepoint B (cost_class derivation table, agents-core-locality-ledger-v0).
_LOCALITY_COST_CLASS_BY_OPERATOR = {
    "qwen": "local-sh",
    "quest": "local-gw",
    "gravitywell": "local-gw",
    "gravitywell-creative": "local-gw",
    # flashnext: the flash-next sglang seat on the GravityWell box — a local
    # seat, zero paid watts (gate-lanes-registry-driven-flashnext-v0-agents-core,
    # S2). Its host is registry-resolved at call time (see
    # _locality_record_call_operator), so it is deliberately absent from
    # _LOCALITY_HOST_BY_OPERATOR below.
    "flashnext": "local-gw",
    "sonnet": "paid-anthropic",
    "opus": "paid-anthropic",
    "haiku": "paid-anthropic",
    "phala": "paid-phala-tee",
}

_LOCALITY_HOST_BY_OPERATOR = {
    "qwen": _llamacpp_url(),
    "quest": QUEST_URL,
    "gravitywell": GW_URL,
    "gravitywell-creative": GW_CREATIVE_URL,
    "sonnet": "claude-cli",
    "opus": "claude-cli",
    "haiku": "claude-cli",
    "phala": PHALA_URL,
    # "flashnext" is intentionally absent: its base_url is a registry row that
    # comes and goes with the GPU handover, so a module-load literal would
    # record a host the call never dialed.
}


def _locality_record_call_operator(*, operator_class, model, prov, served, start, ok,
                                   lane_obj=None):
    """Derive and write one ledger record for a call_operator() invocation.

    prov is the (reason, effective_operator) list _call_operator_impl populated
    (whether or not the caller supplied their own — see call_operator() below).
    fallback_fired/fallback_reason are read from it: _apply_wake_fail appends
    ("fallback", policy) after its own recursive call_operator() returns, so the
    reason immediately preceding that entry (skipping "success"/"fallback"
    entries, which can belong to the nested fallback call sharing this same
    list) is why the original operator failed.

    `ok` reflects both the exception path (set False by the caller when the
    implementation raised) and a clean-return refusal (implementation
    returned None, e.g. on_wake_fail="skip") — see call_operator() below. On
    a failed call nothing answered, so served_model is recorded as None
    rather than guessed; on success with no observed served model the
    OPERATOR_DEFAULTS guess is kept (needed by by_requested_operator) but
    tagged served_model_observed=False in extra so the guess is legible
    rather than indistinguishable from a real observation.
    """
    try:
        from agents_core.locality import record as _locality_record

        fallback_fired = False
        fallback_reason = None
        for i, (reason, _effective_operator) in enumerate(prov):
            if reason == "fallback":
                fallback_fired = True
                for prior_reason, _ in reversed(prov[:i]):
                    if prior_reason not in ("success", "fallback"):
                        fallback_reason = prior_reason
                        break
                break

        if ok:
            served_model = served[-1] if served else (model or OPERATOR_DEFAULTS.get(operator_class))
            extra = {"served_model_observed": bool(served)}
        else:
            served_model = None
            extra = None
        cost_class = _LOCALITY_COST_CLASS_BY_OPERATOR.get(operator_class, "unknown")
        host = _LOCALITY_HOST_BY_OPERATOR.get(operator_class)
        if operator_class == "flashnext":
            # Registry-resolved host: the lane the call actually resolved
            # (passed in by the wrapper — S2). None (blind) records no host
            # rather than a stale one; the ledger never blocks a call.
            host = lane_obj.base_url if lane_obj is not None else None
        duration_ms = (time.monotonic() - start) * 1000

        _locality_record(
            requested_operator=operator_class,
            served_model=served_model,
            host=host,
            cost_class=cost_class,
            fallback_fired=fallback_fired,
            fallback_reason=fallback_reason,
            seam="call_operator",
            duration_ms=duration_ms,
            ok=ok,
            extra=extra,
        )
    except Exception as e:
        _log.warning("[locality] ledger write failed in call_operator: %s", e)


def call_operator(operator_class: str, prompt: str, model: str = None,
                  _provenance_out: list | None = None,
                  principal: str | None = None,
                  lease_class: str = _LEASE_CLASS_UNSET,
                  _admission_bypass: bool = False,
                  **kwargs) -> str | None:
    """Locality-ledger side-write wrapper around _call_operator_impl().

    Pure side-write: same public signature, same return value, same raised
    exceptions as the implementation below — the only addition is one
    locality.record() call per invocation (chokepoint B), which never raises
    and never changes what's returned. See _call_operator_impl for the full
    docstring (operator classes, defaults, kwargs, return contract).

    A GW→paid fallback recurses through this same wrapper (via
    _apply_wake_fail's own call_operator() call), so a fallback produces two
    ledger records — the failed local attempt and the paid fallback — both
    tagged seam="call_operator" and distinguishable via fallback_fired/
    fallback_reason. This is the double-counting the seam field exists to
    make attributable, not a bug.
    """
    _locality_start = time.monotonic()
    _locality_prov = _provenance_out if _provenance_out is not None else []
    _locality_served: list = []
    _impl_kwargs = dict(kwargs)
    _locality_lane_obj = _impl_kwargs.get("_lane")
    if operator_class == "gravitywell":
        # Only the gravitywell branch's gw_kwargs allowlist forwards this key
        # (llm.py ~1078-1082); injecting it for other operator classes would
        # leak an unused kwarg into backends that don't expect it (e.g. a
        # non-autospec test mock of _call_qwen_backend accepts and records
        # any kwarg, breaking assert_called_once_with(prompt=...) assertions).
        _impl_kwargs.setdefault("_served_model_out", _locality_served)
    elif operator_class == "flashnext":
        # S2: resolve the registry lane ONCE per call and hand the same
        # resolved lane to the implementation, so the ledger records the host
        # the call actually dialed instead of reading the registry twice (a
        # second read could disagree with the first across a seat handover,
        # and a blind second read would record a stale host). Accept either
        # shape a caller may have passed: a GateLane object (adapter-built,
        # S4) or an already-resolved (lane_obj, reason) pair.
        if isinstance(_locality_lane_obj, tuple) and len(_locality_lane_obj) == 2:
            _locality_lane_obj = _locality_lane_obj[0]
        elif not isinstance(getattr(_locality_lane_obj, "base_url", None), str):
            _locality_lane_obj, _locality_reason = _flashnext_lane()
            _impl_kwargs["_lane"] = (_locality_lane_obj, _locality_reason)

    ok = True
    _locality_result = None
    try:
        _locality_result = _call_operator_impl(
            operator_class, prompt, model=model,
            _provenance_out=_locality_prov,
            principal=principal, lease_class=lease_class,
            _admission_bypass=_admission_bypass,
            **_impl_kwargs,
        )
        return _locality_result
    except Exception:
        ok = False
        raise
    finally:
        if ok:
            ok = _locality_result is not None
        _locality_record_call_operator(
            operator_class=operator_class,
            # S2: for the registry lane, reuse the already-resolved served id
            # rather than letting the record helper re-read the registry for
            # its OPERATOR_DEFAULTS fallback guess (a second read could
            # disagree with the call across a seat handover). served_model_observed
            # stays False: a registry pin is a contract, not a wire observation.
            model=(model or (getattr(_locality_lane_obj, "served_model", None)
                             if operator_class == "flashnext" else None)),
            prov=_locality_prov, served=_locality_served,
            start=_locality_start, ok=ok,
            lane_obj=_locality_lane_obj if operator_class == "flashnext" else None,
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
    3. Neither given: no system context

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
        f"{_llamacpp_url()}/v1/chat/completions",
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

def _write_stream_line(path, line: str) -> None:
    """Append one complete JSONL line to the stream log, corruption-safe.

    open, write, flush, close per line — no fsync. A killed *process* cannot
    touch data flush() has already handed to the OS's file buffer; fsync()
    would additionally force it to physical disk, which only matters against
    a full host crash (not this unit's threat model, a queue-runner
    SIGKILL/timeout) and risks blocking if the disk subsystem stalls. Any
    failure here is swallowed so a bad line can never kill the read loop.
    """
    try:
        f = open(path, "a", encoding="utf-8", errors="replace")
        try:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
            f.flush()
        finally:
            f.close()
    except OSError:
        pass


def _call_claude_cli_streaming(cmd, user_input, cwd, timeout, stream_log_path, log, _ret):
    """Popen + line-by-line stdout read for the opt-in stream-json mode.

    Reconstructs the same (text, envelope) shape the --output-format json
    path returns, from the stream's terminal type:"result" event. Never
    accumulates the full stream in memory — the file on disk is the
    authoritative record; this only tracks the terminal event needed to
    build the return value.
    """
    log_path = Path(stream_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)

    terminal_envelope = None
    proc = None
    watchdog = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd or "/srv/agents",
        )
        try:
            proc.stdin.write(user_input)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        # A blocking readline() below can stall past `timeout` waiting on the
        # child; a watchdog kills the process from outside the read loop
        # rather than checking a deadline between reads.
        watchdog = threading.Timer(timeout, proc.kill)
        watchdog.daemon = True
        watchdog.start()

        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            _write_stream_line(log_path, line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = None
            if isinstance(event, dict) and event.get("type") == "result":
                terminal_envelope = event

        proc.wait()
    except Exception as e:
        if log:
            log(f"claude -p (stream) failed to launch: {e}")
        if proc is not None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
        return _ret(None, None)
    finally:
        if watchdog is not None:
            watchdog.cancel()

    if terminal_envelope is None:
        if log:
            log(f"claude -p (stream) ended without a terminal result event "
                f"(returncode={proc.returncode})")
        return _ret(None, None)

    if not isinstance(terminal_envelope, dict):
        if log:
            log(f"claude -p (stream) returned non-dict envelope: "
                f"{type(terminal_envelope).__name__}")
        return _ret(str(terminal_envelope) if terminal_envelope else None, None)

    text = terminal_envelope.get("result") or ""
    if not isinstance(text, str):
        text = str(text) if text else ""
    if not text.strip():
        if log:
            log(f"claude -p (stream) returned empty result. envelope keys: "
                f"{list(terminal_envelope.keys())}, "
                f"is_error={terminal_envelope.get('is_error')}, "
                f"type={terminal_envelope.get('type')}")
        return _ret(None, terminal_envelope)
    return _ret(text, terminal_envelope)


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
    stream_log_path: str | None = None,
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
        stream_log_path: Opt-in streaming mode. When set, invokes with
            `--output-format stream-json` and writes each JSONL event line
            to this path as it arrives (one open/write/flush/close per
            line), so a killed subprocess leaves a partial, uncorrupted
            record on disk. The file, once it exists, is the authoritative
            record of what happened during the run; the in-memory return
            value here is always a bounded summary, never a substitute for
            it. Default None reproduces today's `--output-format json`
            behavior byte-for-byte.

    Returns:
        str | None on default (text or None on failure), or
        (str | None, dict | None) when return_envelope=True.
    """
    cmd = [
        "claude", "-p",
        "--model", model,
        "--no-session-persistence",
        "--output-format", "stream-json" if stream_log_path else "json",
    ]
    if permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if system:
        cmd += ["--append-system-prompt", system]

    user_input = prompt
    if json_mode:
        user_input = prompt + "\n\nRespond ONLY with valid JSON. No markdown fences."

    _locality_call_start = time.monotonic()

    def _ret(text, envelope):
        # Locality ledger chokepoint A — covers all paid Anthropic spend via
        # `claude -p`, after the envelope is parsed so cost_usd is available
        # when the CLI provided one. Never raises (locality.record() is
        # itself exception-safe); this call must never affect the return
        # value below.
        try:
            from agents_core.locality import record as _locality_record

            cost_usd = None
            duration_ms = None
            if isinstance(envelope, dict):
                raw_cost = envelope.get("total_cost_usd")
                if isinstance(raw_cost, (int, float)):
                    cost_usd = raw_cost
                raw_duration = envelope.get("duration_ms")
                if isinstance(raw_duration, (int, float)):
                    duration_ms = raw_duration
            if duration_ms is None:
                duration_ms = (time.monotonic() - _locality_call_start) * 1000

            _locality_record(
                requested_operator=model,
                served_model=model,
                host="claude-cli",
                cost_class="paid-anthropic",
                seam="call_claude_cli",
                cost_usd=cost_usd,
                duration_ms=duration_ms,
                ok=text is not None,
            )
        except Exception as e:
            _log.warning("[locality] ledger write failed in call_claude_cli: %s", e)
        return (text, envelope) if return_envelope else text

    if stream_log_path:
        return _call_claude_cli_streaming(
            cmd, user_input, cwd, timeout, stream_log_path, log, _ret,
        )

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
# Canonical GW serving-state + model-name resolver (gw-serving-state-resolver-v0)
# ---------------------------------------------------------------------------
#
# Contract: mode/units/in-flight-flip are owned solely by the flip-controller
# (:8408/v0/status); served model id(s) solely by the endpoint's /v1/models;
# name reconciliation solely by the static alias registry below (gw_models.yaml).
# No consumer re-derives these — everyone calls gw_serving_state().
#
# Silent-oracle epistemology ("both, never conflated", enforced architecturally):
# when the flip-controller is unreachable, `mode` is None and `authority_gap` is
# True — the resolver never writes an inferred value into `mode`. `mode_inferred`
# is a distinct, read-only field a caller must opt into by name;
# require_authoritative_mode() raises rather than let a caller silently act on it.


class GwRegistryError(Exception):
    """Raised when agents_core/data/gw_models.yaml is malformed or absent.

    A loud, load-time failure — never a blank-map silent start (C2)."""


class AuthorityGapError(Exception):
    """Raised by require_authoritative_mode() when the flip-controller mode
    oracle is unreachable (state.mode is None) — the caller may not proceed
    on state.mode_inferred as a substitute."""


@dataclass(frozen=True)
class ModelEntry:
    canonical_id: str
    mode_alias: str
    operator_alias: str
    display_label: str
    weights_hint: str


_GW_MODEL_ENTRY_FIELDS = ("canonical_id", "mode_alias", "operator_alias", "display_label", "weights_hint")


def _load_gw_model_registry(path: Path = GW_MODEL_REGISTRY_PATH) -> list[ModelEntry]:
    """Parse gw_models.yaml into ModelEntry rows. Raises GwRegistryError loudly
    on any malformed or absent file — never a blank-map silent start."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise GwRegistryError(f"gw_models.yaml unreadable at {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise GwRegistryError(f"gw_models.yaml at {path} missing top-level 'models' list")

    entries = []
    for row in raw["models"]:
        if not isinstance(row, dict) or not all(k in row for k in _GW_MODEL_ENTRY_FIELDS):
            raise GwRegistryError(f"gw_models.yaml at {path} has a malformed model entry: {row!r}")
        entries.append(ModelEntry(**{k: row[k] for k in _GW_MODEL_ENTRY_FIELDS}))
    return entries


_GW_MODEL_REGISTRY = _load_gw_model_registry()


def _gw_registry_lookup(alias: str | None) -> ModelEntry | None:
    """Lookup a ModelEntry by ANY of its aliases (canonical_id, display_label,
    operator_alias, mode_alias). Returns None if unrecognised (resolve-time
    unknown_model, never a load-time error — the file is valid, the world moved)."""
    if alias is None:
        return None
    for entry in _GW_MODEL_REGISTRY:
        if alias in (entry.canonical_id, entry.display_label, entry.operator_alias, entry.mode_alias):
            return entry
    return None


def gw_slot2_url(primary_url: str | None = None) -> str:
    """Resolve the slot-2 GW URL (unified helper, Sonnet #3).

    GW_SLOT2_URL env override wins; else primary_url (or GW_URL) host with
    GW_SLOT2_PORT (default 8082)."""
    override = os.environ.get("GW_SLOT2_URL")
    if override:
        return override
    base = primary_url or GW_URL
    parsed = urlparse(base)
    port = os.environ.get("GW_SLOT2_PORT", "8082")
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def _gw_freshness(status: str, checked_at: int | None = None) -> dict:
    return {"status": status, "checked_at": checked_at}


@dataclass(frozen=True)
class GwServingState:
    endpoint: str
    reachable: bool
    serving: bool
    served_id: str | None
    served_ids: list
    canonical: ModelEntry | None
    unknown_model: bool
    mode: str | None
    mode_inferred: str | None
    authority_gap: bool
    units: dict
    in_flight_flip: bool
    distinct_second_model: bool
    source_freshness: dict


def require_authoritative_mode(state: GwServingState) -> str:
    """Return state.mode, or raise AuthorityGapError when the flip-controller
    oracle was silent (state.mode is None). State-transition / lease-acceptance
    code calls this — it structurally cannot proceed on mode_inferred."""
    if state.mode is None:
        raise AuthorityGapError(
            f"flip-controller mode oracle unreachable for {state.endpoint} "
            f"(authority_gap=True) — refusing to substitute mode_inferred={state.mode_inferred!r}"
        )
    return state.mode


def gw_serving_state(endpoint: str | None = None, timeout: float = 4.0, log=None) -> GwServingState:
    """The single resolver for GW serving-state + model-name (C3).

    Composes three independently-owned sources — the endpoint's /health and
    /v1/models, and the flip-controller's :8408/v0/status — reconciled against
    the static alias registry (gw_models.yaml). Degrades soft on any single-
    source outage (never raises); source_freshness reports which source
    answered. Never makes a live network call from a test — all HTTP here is
    the live-verification path callers opt into by invoking this function."""
    resolved_endpoint = endpoint or GW_URL

    source_freshness = {
        "flip_controller": _gw_freshness("unreachable"),
        "models_endpoint": _gw_freshness("unreachable"),
        "health": _gw_freshness("unreachable"),
        "slot2": _gw_freshness("unreachable"),
    }

    reachable = False
    try:
        resp = requests.get(f"{resolved_endpoint}/health", timeout=timeout)
        if resp.status_code == 200:
            reachable = True
            source_freshness["health"] = _gw_freshness("answered", int(time.time()))
    except Exception:
        pass

    serving = False
    served_ids: list = []
    try:
        resp = requests.get(f"{resolved_endpoint}/v1/models", timeout=timeout)
        if resp.status_code == 200:
            data = (resp.json() or {}).get("data") or []
            served_ids = [d.get("id") for d in data if d.get("id")]
            if served_ids:
                serving = True
                source_freshness["models_endpoint"] = _gw_freshness("answered", int(time.time()))
    except Exception:
        pass

    served_id = served_ids[0] if served_ids else None
    canonical = _gw_registry_lookup(served_id)
    unknown_model = served_id is not None and canonical is None
    if unknown_model and log:
        log(f"[gw_serving_state] unknown_model: served_id={served_id!r} not in registry")

    mode = None
    units: dict = {}
    in_flight_flip = False
    authority_gap = True
    try:
        resp = requests.get(f"{FLIP_CONTROLLER_URL}/v0/status", timeout=timeout)
        if resp.status_code == 200:
            fc = resp.json() or {}
            mode = fc.get("mode")
            units = fc.get("units") or {}
            in_flight_flip = bool(fc.get("in_flight_flip"))
            authority_gap = False
            source_freshness["flip_controller"] = _gw_freshness("answered", int(time.time()))
    except Exception:
        pass

    mode_inferred = canonical.mode_alias if canonical else None

    primary_key = canonical.canonical_id if canonical else served_id
    distinct_second_model = False
    if served_id is not None:
        for other_id in served_ids[1:]:
            other_canonical = _gw_registry_lookup(other_id)
            other_key = other_canonical.canonical_id if other_canonical else other_id
            if other_key != primary_key:
                distinct_second_model = True
                break

        if not distinct_second_model:
            slot2_url = gw_slot2_url(resolved_endpoint)
            try:
                resp = requests.get(f"{slot2_url}/v1/models", timeout=timeout)
                if resp.status_code == 200:
                    slot2_data = (resp.json() or {}).get("data") or []
                    if slot2_data:
                        source_freshness["slot2"] = _gw_freshness("answered", int(time.time()))
                        slot2_id = slot2_data[0].get("id")
                        slot2_canonical = _gw_registry_lookup(slot2_id)
                        slot2_key = slot2_canonical.canonical_id if slot2_canonical else slot2_id
                        if slot2_key != primary_key:
                            distinct_second_model = True
            except Exception:
                pass

    return GwServingState(
        endpoint=resolved_endpoint,
        reachable=reachable,
        serving=serving,
        served_id=served_id,
        served_ids=served_ids,
        canonical=canonical,
        unknown_model=unknown_model,
        mode=mode,
        mode_inferred=mode_inferred,
        authority_gap=authority_gap,
        units=units,
        in_flight_flip=in_flight_flip,
        distinct_second_model=distinct_second_model,
        source_freshness=source_freshness,
    )


# ---------------------------------------------------------------------------
# Lease-free swarm client
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed

def swarm_serving(swarm_url: str = SWARM_URL, timeout: int = 4) -> bool:
    """Check if the swarm endpoint is ready to serve.

    Delegates to gw_serving_state() (gw-serving-state-resolver-v0) — the one
    definition of "serving" lives there. True iff {swarm_url}/v1/models is
    200 + non-empty AND {swarm_url}/health is 200.

    Does NOT use the doorman (which returns False for a healthy vLLM).
    Does NOT check systemctl (a unit can be active while the model is still loading).
    Does NOT assume phase (liveness ≠ phase): both big-llama.cpp and swarm-vLLM bind
    the same :8081 and speak OpenAI, so this probe proves only that *an* OpenAI
    endpoint is live — phase (swarm resident vs big) is the caller's responsibility.

    Returns False on any error (timeout, connection error, HTTP error, empty models).
    """
    state = gw_serving_state(endpoint=swarm_url, timeout=timeout)
    return state.serving and state.reachable


def swarm_model(swarm_url: str = SWARM_URL, timeout: int = 4) -> str | None:
    """Get the served model ID from the swarm endpoint.

    Delegates to gw_serving_state() (gw-serving-state-resolver-v0). Returns
    the model id (e.g., 'Qwen2.5-3B') from {swarm_url}/v1/models data[0].id.
    Returns None if the endpoint is not serving or the response is malformed.

    For observability and phase discrimination: a caller comparing swarm_model()
    against its expected swarm model asserts the model phase (vs big).
    """
    state = gw_serving_state(endpoint=swarm_url, timeout=timeout)
    return state.served_id


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
