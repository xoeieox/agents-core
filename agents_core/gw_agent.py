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

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Final

import requests

from agents_core.doorman_client import DoormanClient, DoormanUnreachable

GW_URL = os.environ.get("GW_URL", "http://203.0.113.11:8081")
GW_AGENT_TOOL_OUTPUT_CAP = 8192
GW_AGENT_TOOL_INPUT_CAP = 65536
GW_AGENT_CTX_CAP = 120000

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-step POST failure vocabulary (D1: agents-core-gw-agent-failure-reasons-
# bounded-retry-v0). Additive extension of the reason_out vocabulary documented
# on call_gw_agent (see its reason_out docstring) - "gw_unreachable",
# "gw_not_serving", "no_choices", etc. stay as they are; these are the
# machine-consumable classes for the per-step POST exception/response site,
# aligned with lapis_pm/contractor_seat.py's classification so the two surfaces
# speak one language. Defined once as typed module-level constants so no
# inline string literal ever drifts from this vocabulary at a return site.
# GW_REASON_REQUEST_FAILED remains the terminal fallback for anything
# unclassified - existing consumers matching that plain string keep working.
# ---------------------------------------------------------------------------
GW_REASON_RATE_LIMITED: Final[str] = "rate_limited"
GW_REASON_BACKEND_UNREACHABLE: Final[str] = "backend_unreachable"
GW_REASON_REQUEST_TIMEOUT: Final[str] = "request_timeout"
GW_REASON_SERVER_ERROR: Final[str] = "server_error"
GW_REASON_REQUEST_FAILED: Final[str] = "request_failed"
GW_REASON_NO_CHOICES: Final[str] = "no_choices"

# D2: only these classes are transient enough to warrant a bounded in-step
# retry. A 4xx other than 429, a malformed-JSON success body, or anything
# unclassified (GW_REASON_REQUEST_FAILED) returns on the first attempt.
GW_TRANSIENT_REASONS: Final[frozenset] = frozenset({
    GW_REASON_RATE_LIMITED,
    GW_REASON_BACKEND_UNREACHABLE,
    GW_REASON_REQUEST_TIMEOUT,
    GW_REASON_SERVER_ERROR,
})

GW_STEP_MAX_RETRIES: Final[int] = 2
GW_RETRY_AFTER_CAP_S: Final[float] = 30.0
GW_RETRY_BACKOFF_BASE_S: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Deferrable-acquire retry (agents-core-gw-agent-deferrable-acquire-retry-v0).
#
# The doorman's foreground-priority gate (gw-router-phase1-foreground-gate)
# returns "pending_defer" immediately (unregistered, no lease) when a
# `deferrable`-class acquire collides with an active `protected` lease. The
# contract (doorman_client.py's acquire()/is_pending_defer() docstrings) is
# that the caller retries the SAME work_id until it either clears or a
# client-owned budget is exhausted.
#
# GW_DEFER_RETRY_BUDGET_SEC is deliberately NOT sized off
# _defer_wait_timeout()/DOORMAN_MAX_HOLD_TIMEOUT_SEC (900s default): that
# value bounds the *server's* wait-list hold, not how long a caller's worker
# thread should block synchronously on one acquire. A short, separate,
# client-owned ceiling is used instead so a caller that can't afford to sit
# for 15 minutes still fails within a bounded, predictable window.
# ---------------------------------------------------------------------------
GW_DEFER_RETRY_BUDGET_SEC: Final[float] = float(
    os.environ.get("GW_DEFER_RETRY_BUDGET_SEC", "45")
)
GW_DEFER_RETRY_INITIAL_SLEEP_S: Final[float] = 1.5
GW_DEFER_RETRY_BACKOFF_MULT: Final[float] = 1.8
GW_DEFER_RETRY_MAX_SLEEP_S: Final[float] = 9.0
GW_DEFER_RETRY_JITTER_FRAC: Final[float] = 0.25

# Additive to the reason_out vocabulary (see GW_REASON_* above and
# call_gw_agent's reason_out docstring): distinguishable from
# "gw_not_serving" so a defer-retry budget exhaustion is never confused with
# a genuine same-call not-serving response.
GW_REASON_DEFER_TIMEOUT: Final[str] = "gw_defer_timeout"


def _compute_defer_retry_sleep_s(attempt: int, rand_fn: Callable[[], float] = random.random) -> float:
    """Compute the jittered, capped-exponential sleep for defer-retry attempt N (1-indexed).

    Shape (Council/Facets requirement, gate run 2026-08-04): bounded exponential
    backoff with jitter, not a flat interval — a fixed interval risks every
    deferred caller in a batch re-polling the doorman on the same tick.
    """
    base_sleep = min(
        GW_DEFER_RETRY_INITIAL_SLEEP_S * (GW_DEFER_RETRY_BACKOFF_MULT ** max(0, attempt - 1)),
        GW_DEFER_RETRY_MAX_SLEEP_S,
    )
    jitter = base_sleep * GW_DEFER_RETRY_JITTER_FRAC * (2.0 * rand_fn() - 1.0)
    return max(0.0, base_sleep + jitter)


def _acquire_with_defer_retry(
    client: DoormanClient,
    work_id: str,
    *,
    ttl_sec: float,
    reason: str,
    timeout: float,
    principal: str | None,
    lease_class: str | None,
    budget_sec: float | None = None,
    log: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    rand_fn: Callable[[], float] = random.random,
) -> tuple[dict, bool]:
    """Retry a `deferrable`-class acquire on "pending_defer" up to a client-owned budget.

    Isolated from the surrounding lease-acquisition flow (Council/Facets note,
    gate run 2026-08-04): all retry/backoff state lives in this helper's own
    locals, and a failure inside the sleep/retry cycle itself (e.g. sleep_fn
    raising) is caught here and treated as a bounded timeout rather than
    propagating in a way that could corrupt the caller's acquire state.

    Only "pending_defer" responses are retried, per DoormanClient.is_pending_defer().
    Any other status (including exceptions raised by client.acquire() itself,
    e.g. DoormanUnreachable) is NOT retried here — it propagates/returns
    immediately for the caller's existing on_wake_fail handling.

    Returns (res, timed_out):
      - timed_out is False and res["status"] == "serving" -> lease acquired.
      - timed_out is False and res["status"] != "serving" -> genuine non-serving
        status on the first check (or after a status other than pending_defer
        was returned); caller falls into today's "gw_not_serving" handling.
      - timed_out is True -> still "pending_defer" when the budget was
        exhausted; res is the last pending_defer response seen. Caller should
        use a reason distinguishable from "gw_not_serving" (GW_REASON_DEFER_TIMEOUT).
    """
    if budget_sec is None:
        budget_sec = GW_DEFER_RETRY_BUDGET_SEC
    if sleep_fn is None:
        sleep_fn = time.sleep

    start = time.monotonic()
    attempt = 0
    res = client.acquire(
        "gravitywell", work_id, ttl_sec=ttl_sec, reason=reason, timeout=timeout,
        principal=principal, lease_class=lease_class,
    )
    while DoormanClient.is_pending_defer(res):
        elapsed = time.monotonic() - start
        remaining = budget_sec - elapsed
        if remaining <= 0:
            return res, True
        attempt += 1
        sleep_s = min(_compute_defer_retry_sleep_s(attempt, rand_fn=rand_fn), remaining)
        if log:
            log(
                f"[gw_agent] acquire pending_defer for work_id={work_id!r} "
                f"(attempt {attempt}, {remaining:.1f}s of budget left) — "
                f"retrying in {sleep_s:.2f}s"
            )
        try:
            sleep_fn(sleep_s)
        except Exception as e:  # defensive: a sleep_fn failure must not corrupt caller state
            if log:
                log(f"[gw_agent] defer-retry sleep failed: {e}")
            return res, True
        res = client.acquire(
            "gravitywell", work_id, ttl_sec=ttl_sec, reason=reason, timeout=timeout,
            principal=principal, lease_class=lease_class,
        )
    return res, False


def _classify_response(resp: "requests.Response") -> tuple[str, float | None]:
    """Classify a non-2xx HTTP response into (reason, retry_after_s).

    retry_after_s is only ever populated for a 429 with a parseable
    Retry-After header; every other case returns None for it.
    """
    status = resp.status_code
    if status == 429:
        retry_after = None
        header = resp.headers.get("Retry-After")
        if header is not None:
            try:
                retry_after = float(header)
            except (TypeError, ValueError):
                retry_after = None
        return GW_REASON_RATE_LIMITED, retry_after
    if 500 <= status < 600:
        return GW_REASON_SERVER_ERROR, None
    # Any other 4xx (or a status raise_for_status flagged for another reason)
    # is not classified as transient - falls to the terminal fallback.
    return GW_REASON_REQUEST_FAILED, None


def _classify_exception(exc: Exception) -> tuple[str, float | None]:
    """Classify a requests exception raised by the per-step POST into (reason, retry_after_s).

    Timeout is checked before ConnectionError because requests.exceptions.
    ConnectTimeout subclasses BOTH - a connect that times out should classify
    as request_timeout, matching "the per-step timeout fired" (D1), not
    backend_unreachable.
    """
    if isinstance(exc, requests.exceptions.Timeout):
        return GW_REASON_REQUEST_TIMEOUT, None
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return _classify_response(exc.response)
    if isinstance(exc, requests.exceptions.ConnectionError):
        return GW_REASON_BACKEND_UNREACHABLE, None
    return GW_REASON_REQUEST_FAILED, None


def build_step_payload(
    *,
    model: str | None,
    messages: list[dict],
    tools: dict[str, dict[str, Any]] | list[dict],
    is_swarm: bool = False,
    think: bool = False,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> dict:
    """Single construction path for a chat-completions step payload.

    Used by both the real per-step POST inside `_call_gw_agent_impl`'s loop and
    the reviewer-seat probe (agents-core-reviewer-seat-tool-call-probe-v0, D6/
    5b) — the probe must send a structurally identical payload (same `model`,
    `tools` shape, `chat_template_kwargs`/`response_format` handling) to the
    real reviewer call, or it tests a different request shape and can report a
    healthy seat while the real call fails. `tools` may be the OpenAI-format
    dict keyed by name (as stored in DEFAULT_READONLY_TOOLS/DEFAULT_FIXER_TOOLS)
    or an already-flattened list — dict is flattened via `.values()`.
    """
    _tools = list(tools.values()) if isinstance(tools, dict) else tools
    return {
        **({} if model is None else {"model": model}),
        "messages": messages,
        "tools": _tools,
        "tool_choice": "auto",
        "temperature": temperature,
        **({} if is_swarm else {"chat_template_kwargs": {"enable_thinking": think}}),
        **({} if max_tokens is None else {"max_tokens": max_tokens}),
    }


# ---------------------------------------------------------------------------
# Reviewer-seat tool-call probe (agents-core-reviewer-seat-tool-call-probe-v0)
#
# The local reviewer seat can be alive, serving, and correctly configured, and
# still emit zero tool_calls for every request — a fault that is invisible
# above this layer (a dead-seat reviewer exits in 2-4s with reason=
# grounding_failed, indistinguishable from a real "model looked and declined
# to investigate" outcome without this probe) and NOT stable across GW
# restarts (established by live probe 2026-08-06, see
# correction/reviewer-tool-set-rule-refuted-seat-is-epoch-fragile-2026-08-06).
#
# D6 (amended): this is a DIRECT HTTP POST, not a call_gw_agent run. Routing
# through call_gw_agent would take a second doorman lease (acquire_lease
# defaults True) or, if acquire_lease=False were passed to avoid that, would
# silently flip _is_swarm (gw_agent.py _is_swarm = backend_url is not None and
# not acquire_lease) and change the payload shape (drops chat_template_kwargs,
# alters response_format handling) — testing a payload the real call never
# sends. The probe therefore posts once, directly, with the exact payload
# shape (build_step_payload above) the real call uses, takes no lease, and
# cannot trip the swarm flag.
#
# D5: fail OPEN on probe error (transport/timeout/non-200 — seat health
# unknown, never block a real review on a probe outage), fail CLOSED on probe
# refusal (a clean response with zero tool_calls — a positive determination
# the seat is dead).
# ---------------------------------------------------------------------------

PROBE_PROMPT: Final[str] = (
    "Call the read_file tool on the path \"/tmp\" to confirm your tool "
    "surface is working. This is a liveness probe — do not explain, just "
    "call the tool."
)

# Default number of perturbation attempts probe_seat_tool_call makes before
# concluding the seat is genuinely dead (agents-core-reviewer-seat-prefix-
# perturbation-retry-v0). See perturb_tool_order below for why this is a
# deterministic rotation, not a synthetic-tool injection or random shuffle.
PROBE_DEFAULT_ATTEMPTS: Final[int] = 3


def perturb_tool_order(
    tools: dict[str, dict[str, Any]], attempt: int
) -> dict[str, dict[str, Any]]:
    """Return a semantically-equivalent tool mapping, differently serialized.

    Live-measured 2026-08-07 (see the reviewer-seat-prefix-perturbation-retry-v0
    spec): the native serialization of DEFAULT_READONLY_TOOLS/DEFAULT_FIXER_TOOLS
    emits zero tool_calls deterministically for a given vLLM-server-lifetime,
    while any structurally-different serialization of the SAME tool set
    recovers it. The fault is localized to the cached KV blocks for that exact
    prefix, not to anything semantic about the tools.

    The perturbation is a rotation of tool order, not injection of a synthetic
    tool — a synthetic tool would enlarge the model's action space and could
    itself be called. This never adds, removes, or renames a tool, and never
    touches a tool's parameter schema; only the dict's key order (and hence
    the serialized byte sequence downstream) changes.

    Deterministic and stateless: derived from `attempt` alone via modulo
    arithmetic over the tool count, so it is well-defined for any attempt
    index, including 0 (returns `tools` unchanged — the native order),
    an index equal to the tool count (shift wraps to 0, same as attempt 0),
    and an index exceeding the tool count (`attempt % len(tools)` folds it
    back into range). Randomized/salted perturbation is deliberately rejected
    — a non-reproducible variant would make it impossible for the real review
    call (Part 3) to send the exact variant the probe validated, and would
    make post-hoc log analysis of which prefixes get poisoned impossible.
    """
    if not tools:
        return tools
    names = list(tools.keys())
    shift = attempt % len(names)
    if shift == 0:
        return tools
    rotated_names = names[shift:] + names[:shift]
    return {name: tools[name] for name in rotated_names}


def _tool_block_hash(tools: dict[str, dict[str, Any]] | list[dict]) -> str:
    """Short, stable hash of a serialized tool block for dispatch-log correlation.

    Used by probe_seat_tool_call's log line (Part 4) so a seat that needs
    perturbation is traceable back to the exact refused prefix — makes
    post-hoc analysis of which prefixes get poisoned, and how often,
    possible. Deterministic: same tool mapping/order -> same hash, always.
    """
    _tools = list(tools.values()) if isinstance(tools, dict) else tools
    blob = json.dumps(_tools, sort_keys=False, default=str).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:12]


def probe_seat_tool_call(
    *,
    backend_url: str | None,
    model: str | None,
    tools: dict[str, dict[str, Any]],
    timeout: float = 15.0,
    attempts: int = PROBE_DEFAULT_ATTEMPTS,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Direct-POST probe: does this seat, with this tool surface, emit a
    tool call at all? Bounded attempt loop (default 3) over perturbed
    tool-order variants (see perturb_tool_order) — a single dead attempt at
    the native prefix no longer condemns the seat, since the fault is a
    per-server-lifetime cache-poisoning of one exact serialized prefix, not
    a property of the tool set itself.

    Returns a dict:
      {"outcome": "tool_call" | "no_tool_call" | "error",
       "served_model": str | None,
       "detail": str | None,
       "variant": dict — the tool mapping that produced this outcome (the
         successful variant on "tool_call"; the caller should reuse this
         EXACT variant for the real review — see Part 3 of the spec),
       "attempt": int | None — 0-indexed attempt that succeeded, or None if
         no attempt succeeded (outcome != "tool_call"),
       "attempts_made": int — total attempts actually made,
       "refused_prefix_hash": str — short stable hash of the native
         (attempt-0) serialized tool block, for dispatch-log correlation}

    "tool_call": some attempt (any of the `attempts`) carried >=1
      tool_calls — seat is alive. Outcome semantics are NOT widened relative
      to the pre-retry probe: this is still "the seat can tool-call", now
      established across up to `attempts` differently-serialized variants
      of the identical tool set instead of just the native one.
    "no_tool_call": every attempt returned a clean zero-tool-call response —
      seat is dead (D5 fail CLOSED; caller should short-circuit with reason
      "seat_no_tool_calls").
    "error": transport failure, timeout, non-200, or a malformed body on ANY
      attempt — seat health is UNKNOWN (D5 fail OPEN; caller should proceed
      to the real review, never treat this as a dead seat). A transport
      error terminates the attempt loop immediately rather than being
      retried as if it were a clean refusal — it must never be converted
      into a dead-seat ("no_tool_call") verdict.
    """
    _url = backend_url if backend_url is not None else GW_URL
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    native_hash = _tool_block_hash(tools)
    _attempts = max(1, attempts)

    served_model = None
    for attempt in range(_attempts):
        attempts_made = attempt + 1
        variant = perturb_tool_order(tools, attempt)
        payload = build_step_payload(
            model=model,
            messages=messages,
            tools=variant,
            is_swarm=False,
            think=False,
            temperature=0.0,
            max_tokens=64,
        )
        try:
            resp = requests.post(f"{_url}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            if log:
                log(f"[gw_agent] reviewer-seat-probe: request failed on attempt {attempt}: {exc}")
            return {
                "outcome": "error",
                "served_model": None,
                "detail": str(exc),
                "variant": variant,
                "attempt": None,
                "attempts_made": attempts_made,
                "refused_prefix_hash": native_hash,
            }

        served_model = data.get("model")
        choices = data.get("choices") or []
        if not choices:
            if log:
                log(f"[gw_agent] reviewer-seat-probe: response had no choices on attempt {attempt}")
            return {
                "outcome": "error",
                "served_model": served_model,
                "detail": "no_choices",
                "variant": variant,
                "attempt": None,
                "attempts_made": attempts_made,
                "refused_prefix_hash": native_hash,
            }

        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if log:
                log(
                    f"[gw_agent] reviewer-seat-probe: outcome=tool_call "
                    f"succeeded_attempt={attempt} attempts_made={attempts_made} "
                    f"served_model={served_model}"
                )
            return {
                "outcome": "tool_call",
                "served_model": served_model,
                "detail": None,
                "variant": variant,
                "attempt": attempt,
                "attempts_made": attempts_made,
                "refused_prefix_hash": native_hash,
            }
        # Clean zero-tool-call response on this attempt — try the next
        # perturbation rather than concluding the seat is dead on one attempt.

    if log:
        log(
            f"[gw_agent] reviewer-seat-probe: outcome=no_tool_call "
            f"all {_attempts} perturbations refused served_model={served_model}"
        )
    return {
        "outcome": "no_tool_call",
        "served_model": served_model,
        "detail": None,
        "variant": tools,
        "attempt": None,
        "attempts_made": _attempts,
        "refused_prefix_hash": native_hash,
    }


def _post_step_with_bounded_retry(
    backend_url: str,
    payload: dict,
    now: float,
    deadline: float,
    conclusion_reserve_s: float,
    log: Callable[[str], None] | None,
    step_num: int,
) -> tuple[dict | None, str | None]:
    """POST one agent step with bounded retry for transient failure classes (D2).

    `now` is the caller's already-computed time.monotonic() reading for this loop
    iteration (the same one used to derive `_per_step_timeout`) - reused for the
    first attempt so a successful first attempt costs zero extra monotonic() calls
    over the pre-retry implementation. time.monotonic() is called again only when
    actually retrying, i.e. only on the failure path.

    Returns (data, None) on success (data is the parsed JSON response body), or
    (None, reason) on terminal failure - reason is always one of the typed
    GW_REASON_* constants (D1), never a raw exception/string built ad hoc.

    Retry is double-bounded: GW_STEP_MAX_RETRIES caps the attempt count, AND the
    `_per_step_timeout` envelope (deadline - conclusion_reserve_s) is the
    absolute governor - remaining budget is recomputed before every attempt
    (shrinking that attempt's request timeout) and again before every retry
    sleep, aborting immediately with the already-classified reason whenever the
    required wait would not fit. A fixed retry count alone would not honor
    that bound. Only classes in GW_TRANSIENT_REASONS retry; everything else
    (a 4xx other than 429, an unclassified exception) returns on attempt one.
    """
    attempt = 0
    reason = GW_REASON_REQUEST_FAILED
    while True:
        if attempt > 0:
            now = time.monotonic()
        remaining = deadline - now - conclusion_reserve_s
        if attempt > 0 and remaining <= 0:
            if log:
                log(
                    f"[gw_agent] step {step_num + 1}: no budget remaining for retry "
                    f"attempt {attempt + 1}, returning '{reason}'"
                )
            return None, reason
        per_step_timeout = max(20.0, remaining)

        retry_after = None
        try:
            resp = requests.post(
                f"{backend_url}/v1/chat/completions",
                json=payload,
                timeout=per_step_timeout,
            )
            resp.raise_for_status()
            return resp.json(), None
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                reason, retry_after = _classify_response(e.response)
            else:
                reason = GW_REASON_REQUEST_FAILED
        except Exception as e:
            reason, retry_after = _classify_exception(e)
            if reason == GW_REASON_REQUEST_FAILED and log:
                log(f"[gw_agent] step {step_num + 1}: unclassified request exception: {e}")

        if log:
            log(
                f"[gw_agent] step {step_num + 1} attempt {attempt + 1} failed: {reason}"
            )

        if reason not in GW_TRANSIENT_REASONS or attempt >= GW_STEP_MAX_RETRIES:
            return None, reason

        if reason == GW_REASON_RATE_LIMITED and retry_after is not None:
            wait_s = min(max(retry_after, 0.0), GW_RETRY_AFTER_CAP_S)
        else:
            wait_s = GW_RETRY_BACKOFF_BASE_S * (2 ** attempt)

        now = time.monotonic()
        remaining = deadline - now - conclusion_reserve_s
        if wait_s > remaining:
            if log:
                log(
                    f"[gw_agent] step {step_num + 1}: retry wait {wait_s:.1f}s exceeds "
                    f"remaining budget {remaining:.1f}s, aborting with '{reason}'"
                )
            return None, reason

        if log:
            log(
                f"[gw_agent] step {step_num + 1}: retrying in {wait_s:.1f}s "
                f"(attempt {attempt + 2}/{GW_STEP_MAX_RETRIES + 1})"
            )
        time.sleep(wait_s)
        attempt += 1


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
            if not path.is_relative_to(self.cwd):
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
            cmd = ["rg", pattern, "-l", "-m", "100"]
            if path_glob != "**/*":
                cmd += ["--glob", path_glob]
            cmd.append(str(self.cwd))
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
                return {
                    "error": (
                        f"git {tokens[0]!r} not allowed (read-only); origin/main is already "
                        "your base — use read_file/grep to inspect, apply_edit/write_file to change"
                    )
                }

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


def _resolve_owner_repo_from_cwd(cwd: str | None) -> tuple[str | None, str | None]:
    """Resolve (owner, repo) from a cwd's git `origin` remote.

    Pure local git inspection - no network round-trip. A linked worktree shares
    its parent's .git config (including origin), so this is authoritative ground
    truth for the repo a given dispatch is actually running against, unlike a
    bare model-supplied repo name which carries no owner/org information.

    Returns (None, None) on any failure and prints a single WARN line to stderr
    so a resolution failure is distinguishable from a resolution that succeeded
    and happened to land on the Erah default.
    """
    if not cwd or not os.path.isdir(cwd):
        print(
            f"WARN: local-fixer: openprs owner-resolution failed for cwd={cwd} "
            "(no such directory); falling back to model-supplied repo argument",
            file=sys.stderr,
        )
        return None, None

    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(
            f"WARN: local-fixer: openprs owner-resolution failed for cwd={cwd} "
            f"(git remote get-url failed: {exc}); falling back to model-supplied repo argument",
            file=sys.stderr,
        )
        return None, None

    if result.returncode != 0:
        stderr_lower = (result.stderr or "").lower()
        reason = "not a git repo" if "not a git repository" in stderr_lower else "git remote get-url failed"
        print(
            f"WARN: local-fixer: openprs owner-resolution failed for cwd={cwd} "
            f"({reason}); falling back to model-supplied repo argument",
            file=sys.stderr,
        )
        return None, None

    url = result.stdout.strip()
    if url.endswith(".git"):
        url = url[: -len(".git")]

    segments = [s for s in re.split(r"[/:]", url) if s]
    if len(segments) < 2:
        print(
            f"WARN: local-fixer: openprs owner-resolution failed for cwd={cwd} "
            f"(unparseable origin URL: {url}); falling back to model-supplied repo argument",
            file=sys.stderr,
        )
        return None, None

    owner, repo = segments[-2], segments[-1]
    return owner, repo


class OpenPrsExecutor(ToolExecutor):
    """Execute list_open_prs(repo, with_files?) to enumerate open PRs with optional file lists."""

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def execute(self, arguments: dict) -> str | dict:
        try:
            repo = arguments.get("repo", "")
            with_files = arguments.get("with_files", False)

            if not repo:
                return {"error": "repo parameter is required"}

            # Import here to avoid circular dependency
            from agents_core import forgejo

            # cwd-derived owner/repo (from the dispatch's own worktree origin remote) is
            # authoritative when available; it replaces the model-supplied repo argument
            # entirely rather than just filling in a missing owner.
            cwd_owner, cwd_repo = _resolve_owner_repo_from_cwd(self.cwd)
            if cwd_repo is not None:
                owner, repo = cwd_owner, cwd_repo
            else:
                owner = None

            # Fetch open PRs
            try:
                prs = forgejo.get_open_prs(repo, owner=owner)
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
                    changed_files = self._get_changed_files(repo, pr.get("number"), owner)
                    pr_record["changed_files"] = changed_files

                result.append(pr_record)

            # Return as JSON string (body and changed_files are already capped per-PR)
            return json.dumps(result)
        except Exception as e:
            return {"error": f"list_open_prs failed: {e}"}

    def _get_changed_files(self, repo: str, pr_number: int, owner: str | None = None) -> list[str]:
        """Fetch changed files for a PR, with fallback to diff parsing.

        Prefers the PR files API endpoint if available, falls back to diff parsing.
        Caps list to ~50 files per PR, marking truncation if needed.
        """
        from agents_core import forgejo

        changed_files = []

        # Try the PR files endpoint first
        try:
            files_data = forgejo.get_pr_files(repo, pr_number, owner=owner)
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
            diff = forgejo.get_pr_diff(repo, pr_number, owner=owner)
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
            if not path.is_relative_to(self.cwd):
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
            if not path.is_relative_to(self.cwd):
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
    _SHELL_TOKENS = {"pwd", "ls", "cd", "echo", "git", "cat", "grep", "find"}

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

            if target:
                _first_token = target.strip().split(" ", 1)[0] if target.strip() else ""
                if any(ch.isspace() for ch in target) or _first_token in self._SHELL_TOKENS:
                    return {
                        "error": (
                            "run_tests(target=...) takes a pytest path or node id, e.g. "
                            "'tests/test_foo.py' or ''; there is no shell — use read_file/grep "
                            "to inspect"
                        )
                    }

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
                        "description": (
                            "Repository name (e.g., 'agents-core'). A bare name is fine - "
                            "the correct org/owner is resolved automatically from the "
                            "dispatch's working directory when possible."
                        ),
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
# Tool-surface truth block (writeable runs) + novelty-aware progress tracking
# ---------------------------------------------------------------------------


def _render_tool_line(name: str, tool_spec: dict) -> str:
    """One-line usage description for a single tool, driven by its live spec.

    apply_edit/write_file/run_tests get exact hand-authored signatures (required so a
    small model can't miscall them); every other tool gets a generic signature derived
    from its own parameter schema, so an unfamiliar/renamed tool still renders correctly.
    """
    if name == "apply_edit":
        return (
            "- apply_edit(path, old_string, new_string) — replace an exact, unique "
            "old_string with new_string in a file under cwd. This is how you edit code."
        )
    if name == "write_file":
        return (
            "- write_file(path, content) — create or overwrite a file under cwd with "
            "content. This is how you create a new file."
        )
    if name == "run_tests":
        return (
            "- run_tests(target=\"\", k_expr=\"\") — run pytest. target is a pytest path "
            "or node-id (e.g. 'tests/test_foo.py' or 'tests/test_foo.py::test_bar'), or "
            "\"\" for the whole suite. There is no shell — target is NOT a shell command."
        )
    func = (tool_spec or {}).get("function", {}) or {}
    description = func.get("description", "")
    props = ((func.get("parameters") or {}).get("properties") or {})
    sig = f"{name}({', '.join(props.keys())})" if props else f"{name}()"
    return f"- {sig} — {description}" if description else f"- {sig}"


def _build_tool_block(tools: dict[str, dict], writeable: bool = True) -> str:
    """Auto-generate the '## Your actual tools' block from the live tools dict.

    Never hardcodes a tool-name list — iterates `tools` so this can't drift from what's
    actually wired up (the harness generates its own truth instead of trusting prose
    that describes a different agent's toolset).

    writeable=True (fixer path, default — byte-identical to pre-existing behavior)
    renders the edit-capable framing. writeable=False (read-only reviewer path) renders
    the read-only contract instead: this run investigates and returns a verdict, a
    verdict with no tool call is refused, and any preceding fixer-shaped instruction
    (edits/diffs/commits/branches/pushes/PRs) does not apply to this run.
    """
    lines = [
        "## Your actual tools",
        "",
        "These are the ONLY tools available to you in this run. Call them exactly as named below:",
        "",
    ]
    for name, spec in tools.items():
        lines.append(_render_tool_line(name, spec))
    lines.append("")
    if writeable:
        lines.append(
            "Any earlier statement that you have Bash, Read, Write, Edit, Grep, or mem-CLI shell "
            "access is FALSE CONTEXT inherited from a different agent — ignore it entirely. Your "
            "ONLY tools are the ones listed above. To change code you MUST call `apply_edit` or "
            "`write_file`; describing a change in your response does nothing."
        )
    else:
        lines.append(
            "Any earlier statement that you have Bash, Read, Write, Edit, Grep, or mem-CLI shell "
            "access is FALSE CONTEXT inherited from a different agent — ignore it entirely. Your "
            "ONLY tools are the ones listed above."
        )
        lines.append("")
        lines.append(
            "This is a READ-ONLY investigation run. You do not have apply_edit, write_file, or "
            "any tool that changes code, and no shell. Any instruction earlier in this system "
            "prompt describing edits, diffs, commits, branches, pushes, or opening a PR does NOT "
            "apply to this run — ignore it. Your job is to investigate using the tools above and "
            "return a verdict. A verdict produced with zero successful tool calls is not "
            "acceptable and will be refused — call at least one of the tools above before you "
            "conclude."
        )
    return "\n".join(lines)


def _normalize_for_novelty(result_str: str) -> str:
    """Normalize a tool result string before hashing it for novelty tracking.

    Strips whitespace and, for JSON payloads, re-serializes with sorted keys so
    trivially-reordered or re-whitespaced output isn't counted as fresh progress.
    """
    stripped = result_str.strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped
    return json.dumps(parsed, sort_keys=True)


def _novelty_hash(tool_name: str, tool_args: dict, result_str: str) -> str:
    """Hash a (call-args -> normalized-result) pair for grep/mem novelty tracking."""
    payload = json.dumps(
        {"tool": tool_name, "args": tool_args, "result": _normalize_for_novelty(result_str)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _build_handler_context(
    transcript: list[dict],
    messages: list[dict],
    handler_objective: str,
    consecutive_no_progress: int,
    step_num: int,
    explore_steps: int,
) -> dict:
    """Build the bounded context dict passed to handler_hook at the no-progress threshold.

    transcript_slice is the last 6 transcript entries, each stripped down to
    {step, tool, args, error} - tool RESULT bodies are omitted entirely, and each
    string arg value is truncated to 200 chars. Plus the most recent assistant
    message content (truncated to ~1000 chars), which is what reveals drift
    ("I'll wait for the background suite"). No full message history, no tool results.
    """
    _slice = []
    for entry in transcript[-6:]:
        _args = entry.get("arguments")
        _trunc_args: dict = {}
        if isinstance(_args, dict):
            for k, v in _args.items():
                _trunc_args[k] = v[:200] if isinstance(v, str) and len(v) > 200 else v
        _slice.append({
            "step": entry.get("step"),
            "tool": entry.get("tool_name"),
            "args": _trunc_args,
            "error": entry.get("error"),
        })

    _last_assistant = None
    for _m in reversed(messages):
        if _m.get("role") == "assistant":
            _last_assistant = _m.get("content") or ""
            break
    if _last_assistant and len(_last_assistant) > 1000:
        _last_assistant = _last_assistant[:1000]

    return {
        "objective": handler_objective,
        "transcript_slice": _slice,
        "consecutive_no_progress": consecutive_no_progress,
        "step_num": step_num,
        "explore_steps": explore_steps,
        "last_assistant_message": _last_assistant,
    }


def _resolve_int_env(env_name: str, default: int, log: Callable[[str], None] | None) -> int:
    """Read an int override from the environment; fall back (and log once) on bad input."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _msg = f"[gw_agent] invalid {env_name}={raw!r}; falling back to default {default}"
        if log:
            log(_msg)
        else:
            logger.warning(_msg)
        return default


# ---------------------------------------------------------------------------
# Main Agent Loop
# ---------------------------------------------------------------------------


def _call_gw_agent_impl(
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
    no_progress_steps: int | None = None,
    principal: str | None = None,
    lease_class: str = "deferrable",
    verdict_schema: dict | None = None,
    tool_executors: dict[str, ToolExecutor] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    before_tool: Callable[[str, dict], dict] | None = None,
    reason_out: list[str] | None = None,
    served_model_out: list | None = None,
    model: str | None = None,
    handler_hook: Callable[[dict], dict] | None = None,
    handler_objective: str = "",
    handler_max_interventions: int = 2,
    skip_probe: bool = False,
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
        lease_class: foreground-priority gate class for the doorman lease this run
                     acquires (doorman-lease-class-consumers-v0). "protected" for a
                     measured gate or live interactive session (never deferred);
                     "deferrable" (the default) for background/worker runs. call_gw_agent
                     serves both, so the caller must pass this explicitly rather than
                     it being inferred here. Ignored when acquire_lease=False (no lease
                     is taken).
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
                    strings is appended: "gw_unreachable", "gw_not_serving", "gw_defer_timeout"
                    (a `deferrable`-class acquire stayed "pending_defer" past
                    GW_DEFER_RETRY_BUDGET_SEC - see _acquire_with_defer_retry - distinguishable
                    from "gw_not_serving" which is a genuine same-call non-serving status),
                    "request_failed", "rate_limited", "backend_unreachable", "request_timeout",
                    "server_error" (the last four are per-step POST failure classes - see
                    GW_REASON_* / GW_TRANSIENT_REASONS - that retry up to GW_STEP_MAX_RETRIES times within
                    the per-step timeout envelope before landing here), "no_choices",
                    "grounding_failed", "budget_exhausted", "max_steps_exhausted",
                    "interrupted". Left untouched on a genuinely successful (non-empty) result,
                    with one deliberate exception: "grounded_after_perturbation" is appended
                    on a successful (non-empty) result that only grounded after the grounding
                    guard's tool-order perturbation retry engaged (agents-core-gw-agent-
                    grounding-retry-parity-v0, skip_probe=False) - a caller counting on
                    "untouched reason_out == no retry happened" needs this one visible, since
                    a silent recovery here is exactly the invisibility this remedy exists to
                    fix. Stays empty/unpopulated for `writeable=True` calls regardless of cause. Pure
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
        handler_hook: Optional callable (dict) -> dict. When provided (writeable mode
                      only), called at the no-progress nudge threshold - same trigger
                      point that fires the static nudge - in place of the static string,
                      as long as the intervention budget (handler_max_interventions) is
                      not exhausted. Called with a bounded context dict (objective, a
                      6-entry transcript slice, consecutive_no_progress, step_num,
                      explore_steps, last_assistant_message); expected to return a
                      verdict dict {"decision": "continue"|"redirect", "redirect": str
                      or None, "note": str, "anomaly": str or None}. "redirect" (non-empty
                      after strip) appends the redirect as a user message and resets the
                      no-progress counter; "continue" resets the counter with no message
                      appended (strategic pause); any other decision (including a stray
                      "stop" - v0 is redirect-only, run-halting is not supported), a
                      non-dict, or None falls through to the static nudge. The MODEL CALL
                      lives in the caller, not here - gw_agent stays model-agnostic and
                      never imports llm/call_claude_cli, exactly like cancel_check/
                      before_tool. Fail-safe: wrapped in try/except - any exception
                      (including a self-raised TimeoutError) logs a WARN and falls
                      through to the static nudge; the hook is REQUIRED to be
                      self-time-bounding (pass an explicit short timeout to its own model
                      call) since gw_agent applies no timeout of its own around the call.
                      When None (default), never called - behavior is byte-identical to
                      before this param existed.
        handler_objective: Objective string threaded into the handler_hook context dict's
                           "objective" field. Ignored when handler_hook is None.
        handler_max_interventions: Caps the total number of Handler-driven no-progress-
                                   counter resets (redirect + continue-extend combined) in
                                   one run (default 2). Once exhausted, every subsequent
                                   threshold hit takes the fall-through (static nudge)
                                   path and the existing hard-abort stands - the Handler
                                   can never create an infinite supervision loop. Ignored
                                   when handler_hook is None.
        skip_probe: If True, the grounding guard's second-ungrounded-stop retry (below)
                    stands down and returns "grounding_failed" immediately, exactly like
                    before this parameter existed (agents-core-gw-agent-grounding-retry-
                    parity-v0, DoD-5). Set this when the caller already probed the seat's
                    tool-calling ability itself (shaped_runner.py's _run_local_reviewer
                    calls probe_seat_tool_call before this function) — engaging both would
                    double the model calls on the hottest dispatch route in the system.
                    When False (default), a run that reaches the second ungrounded stop
                    (json_mode, not writeable, zero verified tool calls) retries with a
                    perturbed tool order via perturb_tool_order() instead of giving up,
                    capped at PROBE_DEFAULT_ATTEMPTS total tool-order variants — the same
                    remedy shaped_runner already applies externally via probe_seat_tool_
                    call, now reachable by every other call_gw_agent consumer. A run whose
                    first step makes a tool call never engages this path (lazy, not eager).

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

    # Effective local-fixer no-progress default raised 8 -> 12 (see novelty-aware guard
    # below); GW_AGENT_NO_PROGRESS_STEPS overrides, GW_AGENT_MAX_EXPLORE_STEPS bounds
    # total exploration regardless of novelty grace.
    if no_progress_steps is None:
        no_progress_steps = _resolve_int_env("GW_AGENT_NO_PROGRESS_STEPS", 12, log)
    _max_explore_steps = _resolve_int_env("GW_AGENT_MAX_EXPLORE_STEPS", 20, log)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Authoritative tool-surface truth, generated from the live `tools` dict and
    # appended AFTER any composed preamble so it wins on ordering (final/highest
    # priority system content). Appended for both writeable (fixer) and read-only
    # (reviewer) runs — the read-only reviewer needs to be told it has tools too,
    # and must additionally be told the fixer-shaped preamble above it doesn't apply.
    _tool_block = _build_tool_block(tools, writeable=writeable)
    _sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
    if _sys_idx is not None:
        messages[_sys_idx]["content"] = messages[_sys_idx]["content"] + "\n\n" + _tool_block
    else:
        messages.insert(0, {"role": "system", "content": _tool_block})

    transcript: list[dict] = []
    if tool_executors is None:
        tool_executors = _get_tool_executors(cwd, writeable=writeable)
    repeated_calls: dict[str, int] = {}
    ctx_tokens = 0
    # No-progress guard state (writeable mode): track consecutive steps with no semantic progress.
    consecutive_no_progress = 0
    last_test_counts: tuple | None = None
    # Novelty-aware progress state (writeable mode): seen read_file paths and normalized
    # grep/mem result hashes, plus a hard ceiling on total non-edit steps and a
    # fired-once nudge flag.
    _seen_read_paths: set[str] = set()
    _seen_result_hashes: set[str] = set()
    _explore_steps = 0
    _nudge_fired = False
    # Handler supervision state (writeable mode, handler_hook only): bounded intervention
    # budget for Handler-driven counter resets (redirect + continue-extend combined).
    _handler_interventions_used = 0
    # Grounding guard state (json_mode review runs): track verified (error-free) tool calls.
    grounding_count = 0  # tool calls with error is None
    grounding_nudged = False  # True after the first 0-tool-call stop nudge
    # Grounding-retry-parity state (agents-core-gw-agent-grounding-retry-parity-v0):
    # 0 = still on the native tool order; > 0 = this many perturb_tool_order() variants
    # tried so far. _restart_for_perturbation signals the step loop below to re-enter
    # with a fresh conversation and the next variant instead of returning grounding_failed.
    _grounding_perturb_attempt = 0
    _restart_for_perturbation = False
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
                res, _defer_timed_out = _acquire_with_defer_retry(
                    client,
                    work_id,
                    ttl_sec=timeout + 60,
                    reason="gw_agent",
                    timeout=_gw_acquire_timeout(),
                    principal=principal,
                    lease_class=lease_class,
                    log=log,
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

            if _defer_timed_out:
                if log:
                    log(
                        f"[gw_agent] GW acquire still pending_defer after "
                        f"{GW_DEFER_RETRY_BUDGET_SEC}s retry budget"
                    )
                if on_wake_fail == "skip":
                    if writeable:
                        return (_build_fixer_result(cwd, transcript, concluded=False), transcript)
                    if reason_out is not None:
                        reason_out.append(GW_REASON_DEFER_TIMEOUT)
                    return (None, transcript) if return_transcript else None
                elif on_wake_fail == "error":
                    raise Exception(
                        f"GW acquire still pending_defer after {GW_DEFER_RETRY_BUDGET_SEC}s "
                        "retry budget"
                    )
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
        # Base tool set for grounding-retry perturbation (agents-core-gw-agent-grounding-
        # retry-parity-v0) — `tools` itself is reassigned to a perturbed variant on retry,
        # so the native mapping must be captured once, up front, to perturb from.
        _original_tools = tools

        while True:
            for step_num in range(max_steps):
                # Update rolling avg using the wall-clock of the just-completed step (if any).
                _now = time.monotonic()
                if _step_start is not None:
                    _step_times.append(_now - _step_start)
                    _avg_step_s = sum(_step_times) / len(_step_times)
                _step_start = _now

                # Pre-step budget check: stop exploring if too close to the deadline to
                # fit another step AND still have time for a forced-conclusion call.
                #
                # First-step guarantee: at step_num == 0 no real step has run yet, so
                # _avg_step_s is only a seed (18s). The seeded 2*_avg_step_s reserve term
                # can spuriously exceed a modest timeout (e.g. 36s > a 10s timeout),
                # force-concluding at step 0 with ZERO real work — a conclusion from the
                # prompt alone. So at step 0 we use only the real proportional reserve
                # (0.20*timeout) and ignore the unvalidated seed term; steps >= 1 use the
                # full reserve once _avg_step_s reflects measured latency. A genuine
                # deadline breach at step 0 (little/no wall-clock left) still force-
                # concludes below, salvaging a partial via the 20s-floored conclusion call.
                _conclusion_reserve_s = max(2.0 * _avg_step_s, 0.20 * timeout)
                _effective_reserve_s = (
                    0.20 * timeout if step_num == 0 else _conclusion_reserve_s
                )
                if _deadline - _now <= _effective_reserve_s:
                    if log:
                        log(
                            f"[gw_agent] budget deadline approaching at step {step_num + 1}: "
                            f"{_deadline - _now:.1f}s remaining, reserve={_effective_reserve_s:.1f}s — "
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

                # Per-step timeout envelope: leave headroom for the forced-conclusion model
                # call. Never cap below 20s (a legitimate slow step on a loaded 122B can
                # take minutes). _post_step_with_bounded_retry recomputes this same
                # (deadline - conclusion_reserve) bound before every attempt/sleep below.
                #
                # POST to the backend (GW or swarm) with current message state, with
                # bounded retry for transient failure classes (D1/D2) - never exceeds the
                # _per_step_timeout envelope above (recomputed per attempt/sleep inside).
                _payload = build_step_payload(
                    model=model,
                    messages=messages,
                    tools=tools,
                    is_swarm=_is_swarm,
                    think=think,
                )
                data, _post_fail_reason = _post_step_with_bounded_retry(
                    backend_url, _payload, _now, _deadline, _conclusion_reserve_s, log, step_num,
                )
                if data is None:
                    if log:
                        log(f"[gw_agent] GW request failed: {_post_fail_reason}")
                    # Return best-effort content accumulated so far
                    return _finalize_writeable_or_readonly(
                        messages, "", return_transcript, transcript, writeable, cwd, concluded=False,
                        reason_out=reason_out, reason=_post_fail_reason,
                    )
                if served_model_out is not None and "model" in data and data["model"] is not None:
                    served_model_out.append(data["model"])

                # Extract response.
                if "choices" not in data or not data["choices"]:
                    if log:
                        log(f"[gw_agent] GW returned no choices")
                    return _finalize_writeable_or_readonly(
                        messages, "", return_transcript, transcript, writeable, cwd, concluded=False,
                        reason_out=reason_out, reason=GW_REASON_NO_CHOICES,
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
                            # First verified tool call of a perturbation-retry variant: the
                            # seat that refused the native order just grounded on a
                            # differently-serialized one. Distinguishable from a plain,
                            # never-retried success (reason_out untouched there) and from
                            # a final grounding_failed (DoD-4).
                            if grounding_count == 1 and _grounding_perturb_attempt > 0:
                                if reason_out is not None:
                                    reason_out.append("grounded_after_perturbation")
                                if log:
                                    log(
                                        "[gw_agent] grounding guard: recovered via perturbed "
                                        f"tool order on attempt {_grounding_perturb_attempt}"
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
                        # Novelty accounting: a repeated non-novel action (re-reading an
                        # already-read path, an unchanged grep/mem/test) does NOT reset the
                        # guard — it counts toward exhaustion via _explore_steps instead.
                        if writeable and no_progress_steps > 0:
                            if tool_name in ("apply_edit", "write_file"):
                                if not (isinstance(tool_result, dict) and "error" in tool_result):
                                    step_made_progress = True
                            else:
                                _explore_steps += 1
                                if tool_name == "run_tests":
                                    if isinstance(tool_result, dict) and "error" not in tool_result:
                                        tc = (
                                            int(tool_result.get("passed") or 0),
                                            int(tool_result.get("failed") or 0),
                                            int(tool_result.get("errors") or 0),
                                        )
                                        if tc != last_test_counts:
                                            step_made_progress = True
                                            last_test_counts = tc
                                elif tool_name == "read_file":
                                    # Novelty keyed on path only — a path already read this run
                                    # is never novel again, even if its bytes changed (defeats
                                    # a live-timestamp-in-file state-flip loop).
                                    _path_key = tool_args.get("path")
                                    if _path_key is not None:
                                        if _path_key not in _seen_read_paths:
                                            step_made_progress = True
                                        _seen_read_paths.add(_path_key)
                                elif tool_name in ("grep", "mem"):
                                    _rhash = _novelty_hash(tool_name, tool_args, result_str)
                                    if _rhash not in _seen_result_hashes:
                                        step_made_progress = True
                                    _seen_result_hashes.add(_rhash)

                    if _interrupted:
                        break

                    # No-progress guard: abort if K consecutive steps made no semantic progress,
                    # OR if total exploration steps exceed the hard ceiling (grace can never
                    # mask an infinite loop of "novel" reads).
                    if writeable and no_progress_steps > 0:
                        if step_made_progress:
                            consecutive_no_progress = 0
                        else:
                            consecutive_no_progress += 1

                        _nudge_threshold = max(no_progress_steps - 2, 1)
                        _explore_nudge_threshold = max(_max_explore_steps - 2, 1)
                        if (
                            consecutive_no_progress >= _nudge_threshold
                            or _explore_steps >= _explore_nudge_threshold
                        ):
                            _handler_acted = False
                            if handler_hook is not None and _handler_interventions_used < handler_max_interventions:
                                try:
                                    _handler_ctx = _build_handler_context(
                                        transcript, messages, handler_objective,
                                        consecutive_no_progress, step_num + 1, _explore_steps,
                                    )
                                    _verdict = handler_hook(_handler_ctx)
                                except Exception as _hh_exc:
                                    logger.warning(f"[gw_agent] handler_hook raised: {_hh_exc}")
                                    _verdict = None

                                _decision = _verdict.get("decision") if isinstance(_verdict, dict) else None
                                if _decision == "redirect":
                                    _redirect_raw = _verdict.get("redirect")
                                    _redirect_text = _redirect_raw.strip() if isinstance(_redirect_raw, str) else ""
                                    if _redirect_text:
                                        messages.append({"role": "user", "content": _redirect_text})
                                        consecutive_no_progress = 0
                                        _handler_interventions_used += 1
                                        _handler_acted = True
                                elif _decision == "continue":
                                    # Strategic pause: Handler vouches the Operative is legitimately
                                    # still gathering context. Extend the budget, append nothing -
                                    # the Operative is judged on-track, don't pressure it.
                                    consecutive_no_progress = 0
                                    _handler_interventions_used += 1
                                    _handler_acted = True
                                # else: malformed/absent/unrecognized decision (incl. a stray "stop" -
                                # v0 is redirect-only, run-halting is deferred) or empty/null redirect
                                # falls through to the static nudge below, uncounted against budget.

                            if not _handler_acted and not _nudge_fired:
                                _nudge_fired = True
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        "You now have enough context to act. Make your first "
                                        "`apply_edit`/`write_file` now - continued reading without "
                                        "an edit will end this run without a result."
                                    ),
                                })

                        if consecutive_no_progress >= no_progress_steps or _explore_steps >= _max_explore_steps:
                            if log:
                                log(
                                    f"[gw_agent] no-progress guard: {consecutive_no_progress} "
                                    f"consecutive steps with no semantic progress "
                                    f"({_explore_steps} total exploration steps) - aborting"
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
                            _tool_names = ", ".join(tools.keys())
                            messages.append({
                                "role": "user",
                                "content": (
                                    "You concluded without investigating. A verdict with no successful "
                                    f"tool call is not acceptable — use one of your available tools "
                                    f"({_tool_names}) to read the spec target and the relevant code, "
                                    "THEN produce your verdict."
                                ),
                            })
                            continue
                        else:
                            # Second ungrounded stop at this tool-order variant. Before
                            # ruling UNFOUNDED, retry with a perturbed tool order
                            # (agents-core-gw-agent-grounding-retry-parity-v0) — the
                            # reviewer-seat-prefix-perturbation-retry-v0 arc established
                            # that native tool-list order alone can suppress tool_calls
                            # deterministically, and a differently-serialized SAME tool set
                            # recovers it. skip_probe stands this down for a caller
                            # (shaped_runner) that already probed the seat externally, so
                            # the two remedies never stack (DoD-5).
                            if (
                                not skip_probe
                                and _grounding_perturb_attempt + 1 < PROBE_DEFAULT_ATTEMPTS
                            ):
                                _grounding_perturb_attempt += 1
                                if log:
                                    log(
                                        "[gw_agent] grounding guard: second ungrounded stop — "
                                        f"perturbing tool order (attempt "
                                        f"{_grounding_perturb_attempt}/{PROBE_DEFAULT_ATTEMPTS - 1}) "
                                        "and retrying"
                                    )
                                tools = perturb_tool_order(_original_tools, _grounding_perturb_attempt)
                                _tool_block = _build_tool_block(tools, writeable=writeable)
                                messages = [
                                    {"role": "system", "content": system + "\n\n" + _tool_block},
                                    {"role": "user", "content": prompt},
                                ]
                                grounding_nudged = False
                                grounding_count = 0
                                _restart_for_perturbation = True
                                break

                            # Every perturbation variant (or skip_probe) refused — UNFOUNDED.
                            if log:
                                if _grounding_perturb_attempt:
                                    log(
                                        "[gw_agent] grounding guard: grounding_failed after "
                                        f"{_grounding_perturb_attempt} perturbation attempt(s) "
                                        "— all refused"
                                    )
                                else:
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

            # Grounding-retry-parity: the guard below armed a perturbed-tool-order retry and
            # broke out of the step loop above to get here — re-enter with the fresh
            # conversation/tool order it already installed, consuming another full max_steps
            # budget rather than falling through to the interrupted/max_steps handling below,
            # which would misreport this as either.
            if _restart_for_perturbation:
                _restart_for_perturbation = False
                continue

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


def _locality_record_call_gw_agent(*, model, on_wake_fail, served, start, ok):
    """Derive and write one locality-ledger record for a call_gw_agent() run
    (chokepoint C, agents-core-locality-ledger-v0).

    call_gw_agent makes N POSTs per run, so served[-1] (the last echoed
    "model" field) is the deciding served model. A run that fell back to
    call_claude_cli (on_wake_fail="claude", doorman unreachable/not-serving)
    never reaches a POST at all, so `served` stays empty — that emptiness,
    combined with on_wake_fail=="claude", is the only signal available here
    to flag fallback_fired without threading a new side-channel through the
    wake-fail branches (which don't populate reason_out today). The actual
    paid spend is recorded separately and precisely by call_claude_cli's own
    chokepoint (_fallback_claude_cli calls it directly).

    `ok` is False both when an exception escaped and when the run returned
    cleanly with no result payload (e.g. on_wake_fail="skip" refusing rather
    than raising — see call_gw_agent() below). On a failed run nothing
    answered, so served_model is recorded as None rather than guessed; on a
    successful run with no observed served model the `model` guess is kept
    (needed by by_requested_operator) but tagged
    served_model_observed=False in extra.
    """
    try:
        from agents_core.locality import record as _locality_record

        fallback_fired = (on_wake_fail == "claude") and not served
        fallback_reason = "gw_agent_wake_fail" if fallback_fired else None
        if ok:
            served_model = served[-1] if served else model
            extra = {"served_model_observed": bool(served)}
        else:
            served_model = None
            extra = None
        duration_ms = (time.time() - start) * 1000

        _locality_record(
            requested_operator="gravitywell",
            served_model=served_model,
            host=GW_URL,
            cost_class="local-gw",
            fallback_fired=fallback_fired,
            fallback_reason=fallback_reason,
            seam="call_gw_agent",
            duration_ms=duration_ms,
            ok=ok,
            extra=extra,
        )
    except Exception as e:
        logger.warning("[locality] ledger write failed in call_gw_agent: %s", e)


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
    no_progress_steps: int | None = None,
    principal: str | None = None,
    lease_class: str = "deferrable",
    verdict_schema: dict | None = None,
    tool_executors: dict[str, ToolExecutor] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    before_tool: Callable[[str, dict], dict] | None = None,
    reason_out: list[str] | None = None,
    served_model_out: list | None = None,
    model: str | None = None,
    handler_hook: Callable[[dict], dict] | None = None,
    handler_objective: str = "",
    handler_max_interventions: int = 2,
    skip_probe: bool = False,
) -> str | None | tuple[str | None, list[dict]] | tuple[dict, list[dict]]:
    """Locality-ledger side-write wrapper around _call_gw_agent_impl().

    Pure side-write: same public signature, same return value, same raised
    exceptions as the implementation below — the only addition is one
    locality.record() call per run (chokepoint C, at run exit), which never
    raises and never changes what's returned. See _call_gw_agent_impl for the
    full docstring.

    Uses time.time() (wall clock), not time.monotonic(), purely so this
    side-write never consumes values from the tightly-calibrated
    time.monotonic() side_effect sequences several existing tests patch onto
    this module for the impl's own deadline arithmetic (test_gw_agent_budget_
    conclusion.py) — an extra call here would exhaust their iterators.
    """
    _locality_start = time.time()
    _locality_served = served_model_out if served_model_out is not None else []

    ok = True
    _locality_result = None
    try:
        _locality_result = _call_gw_agent_impl(
            prompt, system=system, cwd=cwd, tools=tools, max_steps=max_steps,
            timeout=timeout, json_mode=json_mode, think=think, on_wake_fail=on_wake_fail,
            work_id=work_id, return_transcript=return_transcript, log=log,
            backend_url=backend_url, acquire_lease=acquire_lease, writeable=writeable,
            no_progress_steps=no_progress_steps, principal=principal, lease_class=lease_class,
            verdict_schema=verdict_schema, tool_executors=tool_executors,
            cancel_check=cancel_check, before_tool=before_tool,
            reason_out=reason_out, served_model_out=_locality_served,
            model=model, handler_hook=handler_hook, handler_objective=handler_objective,
            handler_max_interventions=handler_max_interventions, skip_probe=skip_probe,
        )
        return _locality_result
    except Exception:
        ok = False
        raise
    finally:
        if ok:
            # return_transcript=True yields a (payload, transcript) tuple —
            # the None-check belongs on the payload, not the tuple wrapper,
            # so a (None, transcript) refusal is still recorded as failed.
            _locality_payload = (
                _locality_result[0]
                if isinstance(_locality_result, tuple)
                else _locality_result
            )
            ok = _locality_payload is not None
        _locality_record_call_gw_agent(
            model=model, on_wake_fail=on_wake_fail,
            served=_locality_served, start=_locality_start, ok=ok,
        )


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
