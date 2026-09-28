"""Lane-reality preflight for plain-tier local dispatches.

agents-core-lane-reality-preflight-v0. The daemon's dispatch path
(``Shaper.dispatch`` -> ClaudeQueue -> shaped_runner) fires plain-tier
local-fixer / local-reviewer work at a seat WITHOUT asking the registry
whether that seat is alive. The 2026-09-27 deaths this closes:

  * reviewer dispatches for PR #369 died "local reviewer produced no
    verdict" - the reviewer row pins :8081 and :8081 was down;
  * a plain fixer for flashnext-trigger-ssh-default-fix-v0 died "local seat
    returned no text" -> lost-dispatch, no branch, no durable error;
  * reviewer-attempt ceiling keys accumulated ``gw_seat_occupied`` rows
    (pr=363/364/365) - cycles burned by a down lane.

Deliverable 1: before the dispatch fires, resolve the dispatching registry
row's model -> seat through the gw-seats REGISTRY ROOT endpoint (the same
root path the canonical client polls; NEVER a ``/v0/status`` guess - that
endpoint does not exist, see
finding/night2-flip-postmortem-registry-404-endpoint-2026-09-27). If that
port's ``seat.state != "serving"``: DO NOT FIRE. The dispatch is parked with
the tagged result ``noop:lane_down:model=<model>`` and NOTHING is submitted
to the queue, so no fixer/reviewer cycle is burned and the attempt counter
cannot increment (an attempt is only ever counted from a queue record; no
record, no attempt).

Deliverable 2: the preflight must never become the new single point of
failure. A registry that is unreachable / non-200 / malformed / seat-less is
UNKNOWN, not down: the first dispatch for a target inside a 10-minute tick
fires as a PROBE with the WARNING tag ``lane_unknown`` (fail-open), and only
that one probe fires per target per tick. Later dispatches for the same
target inside the window park with ``noop:lane_unknown:probe_in_flight`` so a
blind registry cannot stampede the seat with doomed dispatches, and the
window rolls over so a permanently-blind registry still lets one dispatch
through every tick.

Deliverable 3: reviewer-family rows follow the ACTIVE registry lane instead
of the :8081 pin (the reviewer half of the seat-gate paradox). When the
registry is readable and the row's pinned seat is not serving, the row is
re-routed to the registry's active gate lane (base_url + served id from the
registry row - never a hardcoded port). Registry-blind leaves the row
byte-identical to today.

Deliverable 4: every decision is appended to the lane-status ledger
(``<claude-queue>/lane-status.jsonl``) keyed by target, and
``ClaudeQueue.status()`` / ``ClaudeQueue.lane_status()`` surface the latest
tag per target, so the PM/Erah see WHY a target sits instead of having to
tail JSONL.

Invariants (spec, verbatim):
  * SENSE ONLY - this module reads the registry, it never wakes GW and never
    flips a seat (the standing fence).
  * No new env knobs - the registry URL reuses the seat-check convention
    (root path) and the ``GW_SEATS_URL`` env already established by
    ``agents_core.lane_registry``; the ledger rides the existing
    ``claude_queue`` room-path key.
  * The fixer_flash defer machinery in ``claude_queue`` is the PATTERN
    reference (claim-time deferral, never a failure record), not a
    dependency - the plain tiers get their own gate here.
  * Backward compatible: when the lane is up, behavior is byte-identical to
    today (the preflight returns ``fire`` and the caller mutates nothing).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents_core.room_paths import room_path

# ---------------------------------------------------------------------------
# Tags (the operator-facing decision vocabulary, spec Deliverables 1/2/4)
# ---------------------------------------------------------------------------

#: The seat is registered and declared not-serving: park, do not fire.
TAG_LANE_DOWN = "noop:lane_down"
#: The registry could not be read: fail-open probe (WARNING), once per tick.
TAG_LANE_UNKNOWN = "lane_unknown"
#: A probe already went out for this target inside the current tick.
TAG_LANE_UNKNOWN_PROBE_IN_FLIGHT = "noop:lane_unknown:probe_in_flight"

#: Registry root-path cache TTL (spec: 15 s cache on the registry read).
REGISTRY_CACHE_TTL_S = 15.0
#: The "one probe dispatch per target per 10-min tick" window (Deliverable 2).
PROBE_TICK_S = 600.0

#: Lane-status ledger rotation bounds (the surface is read tail-first, so
#: rotation loses nothing the PM reads).
_LEDGER_ROTATE_BYTES = 256 * 1024
_LEDGER_KEEP_LINES = 2000

#: Engines whose seat is a gw-seats registered lane (the plain tiers). The
#: claude/API tiers have no local seat to preflight; local-opencode names its
#: seat through opencode_model rather than a dialable port and is deferred.
GATED_ENGINES = ("local-fixer", "local-fixer-staged", "local-reviewer", "local-auditor")

#: Reviewer-family engines that FOLLOW the active lane (Deliverable 3) rather
#: than parking on a dead pin. Plain fixers park (auto-routing plain fixers to
#: the active lane is spec-Known-Deferred: it needs the active-lane policy from
#: the sole-model proofs).
REVIEWER_FAMILY_ENGINES = ("local-reviewer", "local-auditor")

# ---------------------------------------------------------------------------
# Registry read (sense-only, root path, 15 s cache)
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {"key": None, "payload": None, "expires": 0.0}

# Test seam: an explicit fetcher installed process-wide (the repo conftest
# installs an all-serving stub so no existing test depends on which seat
# happens to hold the GPU while it runs). An installed override bypasses the
# 15 s cache - deterministic per call, which is what a test needs.
_FETCHER_OVERRIDE: Optional[Callable[[], dict]] = None


def set_registry_fetcher(
    fetcher: Optional[Callable[[], dict]],
) -> Optional[Callable[[], dict]]:
    """Install/clear the process-wide registry fetcher (test seam).

    Returns the previous value so a caller can restore it. Production leaves
    this unset and gets the cached live root-path read.
    """
    global _FETCHER_OVERRIDE
    with _CACHE_LOCK:
        prev = _FETCHER_OVERRIDE
        _FETCHER_OVERRIDE = fetcher
        return prev


# Probe throttle: (target_id, model) -> monotonic ts of the last probe fired.
_PROBE_LOCK = threading.Lock()
_PROBE_SEEN: dict[tuple[str, Optional[str]], float] = {}


def reset_state_for_tests() -> None:
    """Drop the registry cache and the probe throttle (test seam)."""
    with _CACHE_LOCK:
        _CACHE.update({"key": None, "fetcher": None, "payload": None, "expires": 0.0})
    with _PROBE_LOCK:
        _PROBE_SEEN.clear()


def _registry_url() -> str:
    """The registry base URL - the lane_registry convention (GW_SEATS_URL
    env, default the GW box). Reused, never re-derived, and the ROOT path is
    the only endpoint that exists (gw_seats.py serves / and /health)."""
    from agents_core import lane_registry

    return lane_registry._gw_seats_url()


def read_registry(
    fetcher: Optional[Callable[[], dict]] = None,
    now: Optional[float] = None,
    ttl_s: float = REGISTRY_CACHE_TTL_S,
) -> dict:
    """The cached registry-root read (15 s TTL, spec Deliverable 1).

    ``fetcher`` is the hermetic test seam (canned payloads); None = the live
    sense read through ``lane_registry._fetch_payload``. Any transport /
    parse failure collapses to ``{}`` = registry blind. The cache is keyed on
    the registry URL so an env flip (tests, multi-GW) cannot serve a stale
    payload from a different origin.
    """
    url = _registry_url()
    now_s = time.monotonic() if now is None else now
    with _CACHE_LOCK:
        override = _FETCHER_OVERRIDE
        if (
            override is None
            and _CACHE["key"] == url
            # Cache is keyed on the fetcher IDENTITY too: a payload fetched
            # through one stub must never be served to a different stub (or
            # to the live path) inside the TTL window.
            and _CACHE.get("fetcher") is fetcher
            and _CACHE["expires"] > now_s
            and _CACHE["payload"] is not None
        ):
            return _CACHE["payload"]
    if fetcher is not None:
        try:
            payload = fetcher()
        except Exception:
            payload = {}
    elif override is not None:
        try:
            payload = override()
        except Exception:
            payload = {}
    else:
        from agents_core import lane_registry

        try:
            payload = lane_registry._fetch_payload()
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    with _CACHE_LOCK:
        _CACHE["key"] = url
        _CACHE["fetcher"] = fetcher
        _CACHE["payload"] = payload
        _CACHE["expires"] = now_s + ttl_s
    return payload


def registry_blind(payload: dict) -> bool:
    """True when the payload carries no usable seat rows (UNKNOWN, not DOWN).

    Registry unreachable / non-200 / malformed / seat-less all land here; the
    caller must NOT read this as "the lane is down" (Deliverable 2 - the
    preflight is never a new single point of failure).
    """
    from agents_core import lane_registry

    return not (isinstance(payload, dict) and lane_registry._seat_rows(payload))


def seat_state(payload: dict, port: Optional[int]) -> str:
    """The registry's own state for ``port``: "serving" | "down" | "absent" |
    "unknown" (blind) | "not_a_gate_lane" (a seat this preflight does not own).

    Only the registered gate-lane ports are gated; an unknown endpoint (a
    swarm URL, a third-party seat) is out of this spec's scope and fires.
    """
    from agents_core import lane_registry

    if registry_blind(payload):
        return "unknown"
    if port is None or port not in lane_registry.GATE_LANE_PORTS:
        return "not_a_gate_lane"
    for seat in lane_registry._seat_rows(payload):
        if seat.get("port") == port:
            return "serving" if seat.get("state") == "serving" else "down"
    return "absent"


def model_port(payload: dict, model) -> Optional[int]:
    """Resolve a registry row's MODEL to its seat port (spec Deliverable 1:
    "resolve the dispatching registry row's model -> seat").

    Used when the row's ``backend_url`` names no port: match the requested
    model against the registry's own seat rows (exact served id / model_root,
    then the lane markers - "flash" for the :30000 sglang seat, "27b" for the
    :8081 slot1 seat). None = the model does not name a registered gate-lane
    seat, which the caller treats as not port-addressable (fire).
    """
    from agents_core import lane_registry

    if registry_blind(payload) or not isinstance(model, str) or not model.strip():
        return None
    want = model.strip().lower()
    for seat in lane_registry._seat_rows(payload):
        for field_name in ("model", "model_root"):
            val = seat.get(field_name)
            if isinstance(val, str) and val.strip().lower() == want:
                port = seat.get("port")
                if isinstance(port, int):
                    return port
    if lane_registry.FLASHNEXT_MODEL_ROOT_MARKER in want:
        return lane_registry.FLASHNEXT_LANE_PORT
    if lane_registry.SLOT1_MODEL_ROOT_MARKER in want:
        return lane_registry.SLOT1_LANE_PORT
    return None


def active_lane(payload: dict):
    """The registry's ACTIVE gate lane (GateLane | None) - the row the
    reviewer follows (Deliverable 3). None when blind or nothing serves."""
    from agents_core import lane_registry

    try:
        return lane_registry._resolve_from_payload(payload, None)
    except Exception:
        return None


def row_port(spec: dict) -> Optional[int]:
    """The seat port a dispatching row targets: its spec ``backend_url`` port,
    falling back to the GW_URL default port (the :8081 pin the reviewer rows
    carry today). None when neither names a port (not port-addressable here).
    """
    from urllib.parse import urlparse

    url = spec.get("backend_url")
    if not isinstance(url, str) or not url.strip():
        import os

        from agents_core import gw_agent

        url = os.environ.get("GW_URL", gw_agent.GW_URL)
    try:
        return urlparse(url.strip()).port
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@dataclass
class LaneDecision:
    """What the preflight says to do with one dispatching row.

    ``action``: "fire" (dispatch normally) | "park" (DO NOT FIRE).
    ``tag``: the operator-facing tag ("" on a clean fire); ``noop:lane_down:...``
        parks, ``lane_unknown`` is the fail-open WARNING.
    ``reroute_*``: Deliverable 3 - the reviewer-family row's registry-active
        lane (base_url + served id) when the pinned seat is dead.
    """

    action: str = "fire"
    tag: str = ""
    reason: str = ""
    model: Optional[str] = None
    port: Optional[int] = None
    seat: str = "unknown"
    probe: bool = False
    registry_blind: bool = False
    reroute_base_url: Optional[str] = None
    reroute_model: Optional[str] = None
    extras: dict = field(default_factory=dict)

    @property
    def fire(self) -> bool:
        return self.action == "fire"


def _probe_slot_claimed(key: tuple, now_s: float) -> bool:
    """Claim the one-probe-per-target-per-tick slot. True when THIS caller
    owns the probe for the current window (and records it)."""
    with _PROBE_LOCK:
        last = _PROBE_SEEN.get(key)
        if last is not None and (now_s - last) < PROBE_TICK_S:
            return False
        _PROBE_SEEN[key] = now_s
        return True


def preflight(
    spec: dict,
    *,
    fetcher: Optional[Callable[[], dict]] = None,
    now: Optional[float] = None,
    allow_reviewer_reroute: bool = True,
    probe_throttle: bool = True,
    probe_key: Optional[tuple] = None,
) -> LaneDecision:
    """Decide whether a plain-tier local dispatch may fire.

    Reads the registry ONCE (through the 15 s cache) and classifies the row's
    seat:

      * serving / not-a-gate-lane / claude-tier -> ``fire`` (byte-identical
        to today).
      * registered but not serving -> ``park`` with
        ``noop:lane_down:model=<model>``; reviewer-family rows additionally
        carry the active-lane re-route when the registry names one.
      * blind -> first dispatch per (target, model) in a 10-min tick ``fire``
        as a probe with the WARNING tag ``lane_unknown``; the rest of the
        window parks with ``noop:lane_unknown:probe_in_flight``.

    Never raises: an internal fault degrades to ``fire`` (fail-open) - a
    broken preflight must not become the reason nothing dispatches.
    """
    try:
        engine = str(spec.get("engine") or "")
        if engine not in GATED_ENGINES:
            return LaneDecision(action="fire", reason="engine_ungated")

        model = spec.get("model")
        target_id = str(spec.get("target_id") or spec.get("task_id") or "<unknown>")
        now_s = time.monotonic() if now is None else now
        payload = read_registry(fetcher=fetcher, now=now)
        blind = registry_blind(payload)
        port = row_port(spec)
        if port is None:
            # No port on the row: resolve the registry row's model -> seat
            # (Deliverable 1) before deciding there is nothing to gate.
            port = model_port(payload, model)
        state = seat_state(payload, port)

        if state in ("serving", "not_a_gate_lane"):
            return LaneDecision(
                action="fire", model=model, port=port, seat=state,
                registry_blind=blind,
            )

        if blind:
            key = probe_key if probe_key is not None else (
                target_id, model if isinstance(model, str) else None,
            )
            if not probe_throttle or _probe_slot_claimed(key, now_s):
                return LaneDecision(
                    action="fire", tag=TAG_LANE_UNKNOWN, reason="registry_blind",
                    model=model, port=port, seat="unknown", probe=True,
                    registry_blind=True,
                )
            return LaneDecision(
                action="park", tag=TAG_LANE_UNKNOWN_PROBE_IN_FLIGHT,
                reason="registry_blind_probe_in_flight",
                model=model, port=port, seat="unknown", registry_blind=True,
            )

        # Readable registry, seat declared down/absent. Reviewer-family rows
        # FOLLOW the active registry lane instead of dying on the :8081 pin
        # (Deliverable 3 - the reviewer half of the seat-gate paradox); plain
        # fixers PARK (auto-routing them is spec-Known-Deferred).
        lane = active_lane(payload) if allow_reviewer_reroute else None
        if engine in REVIEWER_FAMILY_ENGINES and lane is not None:
            return LaneDecision(
                action="fire", tag=f"INFO:{TAG_LANE_DOWN}:model={model}->lane={lane.name}",
                reason=f"seat_{state}_routed_active_lane",
                model=model, port=port, seat=state, registry_blind=False,
                reroute_base_url=lane.base_url, reroute_model=lane.served_model,
                extras={"active_lane": lane.name},
            )

        tag = f"{TAG_LANE_DOWN}:model={model}"
        decision = LaneDecision(
            action="park", tag=tag, reason=f"seat_{state}",
            model=model, port=port, seat=state, registry_blind=False,
        )
        if lane is not None:
            decision.extras["active_lane"] = lane.name
        return decision
    except Exception as exc:  # fail-open: the preflight is never a SPOF
        return LaneDecision(
            action="fire", tag=TAG_LANE_UNKNOWN, reason=f"preflight_fault:{type(exc).__name__}",
            model=spec.get("model") if isinstance(spec, dict) else None,
            extras={"fault": str(exc)[:200]},
        )


# ---------------------------------------------------------------------------
# The ledger (Deliverable 4: visible per target, not only in JSONL)
# ---------------------------------------------------------------------------


def ledger_path():
    """The lane-status ledger (one JSONL line per preflight decision).

    Rides the existing claude_queue room-path key - no new env knob, and it
    sits beside history.jsonl so the PM's queue-status read picks it up.
    """
    return room_path("claude_queue") / "lane-status.jsonl"


_LEDGER_LOCK = threading.Lock()


def record_lane_tag(
    decision: LaneDecision,
    *,
    target_id: str,
    task_id: str = "",
    agent_type: str = "",
    engine: str = "",
) -> None:
    """Append one decision line to the ledger. Best-effort: never raises.

    Recorded for EVERY decision - including a clean fire - because
    ``read_lane_status`` reports the LAST line per target: if only parks were
    written, a target whose lane came back up would keep showing
    ``noop:lane_down`` forever, and a stale WHY is as misleading as a missing
    one.
    """
    try:
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target_id": target_id,
            "task_id": task_id,
            "agent_type": agent_type,
            "engine": engine,
            "model": decision.model,
            "port": decision.port,
            "seat": decision.seat,
            "action": decision.action,
            "tag": decision.tag,
            "reason": decision.reason,
            "probe": decision.probe,
            "registry": _registry_url(),
        }
        if decision.reroute_base_url:
            line["reroute_base_url"] = decision.reroute_base_url
            line["reroute_model"] = decision.reroute_model
        if decision.extras:
            line["extras"] = decision.extras
        with _LEDGER_LOCK:
            with path.open("a") as fh:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            # Bound the surface: a parked row is deduped per (task, tag) by the
            # queue, but a long-lived daemon still accumulates one line per new
            # dispatch forever. Rotate past the cap (the tail is what
            # read_lane_status reads, so rotation is lossless for the surface).
            try:
                if path.stat().st_size > _LEDGER_ROTATE_BYTES:
                    keep = path.read_text().splitlines()[-_LEDGER_KEEP_LINES:]
                    tmp = path.with_suffix(".jsonl.tmp")
                    tmp.write_text("\n".join(keep) + "\n")
                    tmp.replace(path)
            except OSError:
                pass
    except Exception:
        pass  # the ledger is a sense surface; a write failure never blocks


def read_lane_status(
    target_id: Optional[str] = None,
    limit: int = 500,
) -> dict:
    """Latest lane tag per target (Deliverable 4).

    Returns ``{target_id: {tag, action, reason, model, port, seat, ts,
    agent_type}}`` - the shape lapis-pm's per-target status surfaces so the
    PM/Erah can see WHY a target sits without tailing JSONL.
    """
    out: dict[str, dict] = {}
    try:
        path = ledger_path()
        if not path.exists():
            return {}
        lines = path.read_text().splitlines()[-max(1, limit):]
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            tid = str(rec.get("target_id") or "")
            if not tid or (target_id is not None and tid != target_id):
                continue
            out[tid] = {
                "tag": rec.get("tag", ""),
                "action": rec.get("action", ""),
                "reason": rec.get("reason", ""),
                "model": rec.get("model"),
                "port": rec.get("port"),
                "seat": rec.get("seat"),
                "ts": rec.get("ts"),
                "agent_type": rec.get("agent_type", ""),
            }
    except Exception:
        return out
    return out
