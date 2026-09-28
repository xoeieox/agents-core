"""Registry-resolved gate-lane resolver (canonical home, S1-PROMOTE).

gate-lanes-registry-driven-flashnext-v0-agents-core, S1-PROMOTE: this module
is the promoted canonical home of the resolver that the lapis-pm side vendored
as ``lapis_pm/gate_lane.py`` while the parent PR was in flight. The vendored
shim delegates here when this module is importable (its protocol boundary), so
the contract below is kept identical to the shim's local implementation for
identical payloads: same constants, same seat-row acceptance, same
base_url/served_model derivation, same None cases.

Gate legs (spec_review facets/council, corroboration node2, many-eyes lens
pins, council/triage voicing) resolve their model lane through the gw-seats
registry (the :8408 reality_view, served-model-name pins), never through
hardcoded ports. :8081 stays the default only as the registry-absent fallback
— the same pattern the fixer lane established (fixer_flash registry row
f0fb039, PR #385: served-id + backend :30000, resolved, never hardcoded).

Caller contract (inherited from the parent's stand-aside folds, binding here):

* ``None`` (registry blind — the registry is unreachable, the payload is
  malformed, or no gate lane is registered) is the ONLY case in which callers
  fall back to the existing GW_URL behavior byte-identically. A blind registry
  is "no information", and the legacy env-var-driven path (GW_URL / SWARM_URL)
  is what a blind gate leg ran on before this target — so the blind path must
  be indistinguishable from today.
* When the registry is readable and an explicitly-requested lane (e.g.
  ``--facets-operator flashnext`` / ``--voicing flashnext``) is inactive or not
  serving, the caller records an honest ``leg_down`` and NEVER falls back to
  the legacy lane — a requested-but-dead lane is reported as absent, not
  masked. A silent legacy fallback here is a "lying leg": it degrades the truth
  of the seat's absence (honesty invariant).

Registry URL discovery is explicit in code: ``GW_SEATS_URL`` env override,
default ``http://203.0.113.11:8408`` (mirroring the GW_URL/SWARM_URL
env-default pattern; the registry URL is marked in code rather than assumed).

Registry shape (verified live 2026-09-25 against the :8408 status): the payload
carries ``reality_view`` (``reality`` in {"slot1-solo", "flashnext-solo", ...},
``primary`` with the 27B seat's port/model_root) and ``seats`` — one row per
seat port (8081, 8082, 8500, 30000) with ``state`` ("serving" | "down"),
``model`` (served id), ``model_root``, ``bind``. A gate lane is a seat row
whose port is a known gate-lane port and whose state is "serving". The
flashnext row follows the f0fb039 row shape: served id
``Qwen3.8-Flash-Next-NVFP4-SSD-Stream`` on backend :30000.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

# Registry base URL — explicit env override, documented default (the
# Erah-canonical stand-aside: discovery mechanism marked in code).
DEFAULT_GW_SEATS_URL = "http://203.0.113.11:8408"
GW_SEATS_TIMEOUT_S = 4.0

# Gate-lane ports. :8081 is the 27B slot1 seat (the legacy gravitywell
# lane); :30000 is the flash-next sglang seat (the flashnext lane,
# f0fb039 row shape). Only these two ports are gate lanes — :8082 (slot2
# reference leg) and :8500 are not.
GATE_LANE_PORTS = (8081, 30000)

# The flashnext lane identity (f0fb039 row shape). The served id is the
# EXACT id :30000/v1/models data[0].id advertises (the sglang seat serves
# the concrete id — the slot-alias pitfall is a vLLM-seat property).
FLASHNEXT_LANE_NAME = "flashnext"
FLASHNEXT_LANE_PORT = 30000
FLASHNEXT_SERVED_ID = "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
FLASHNEXT_MODEL_ROOT_MARKER = "flash"  # model_root substring (case-insensitive)

# The 27B lane identity (slot1, :8081).
SLOT1_LANE_NAME = "slot1"
SLOT1_LANE_PORT = 8081
SLOT1_MODEL_ROOT_MARKER = "27b"


@dataclass
class GateLane:
    """One registry-resolved gate lane.

    ``name``: the lane name ("flashnext" | "slot1").
    ``base_url``: the lane's base URL (no trailing path) — the endpoint
        callers build their /v1/* probes and completions POSTs against.
    ``served_model``: the served model id the lane advertises (the
        registry's served-model-name pin; ``None`` when the registry row
        did not carry one and the caller must probe /v1/models live).
    """
    name: str
    base_url: str
    served_model: Optional[str] = None


def _gw_seats_url() -> str:
    """The gw-seats registry base URL, read at call time (env-overridable
    so tests never hit the live registry)."""
    return os.environ.get("GW_SEATS_URL", DEFAULT_GW_SEATS_URL)


def _registry_host() -> str:
    """The CLIENT-REACHABLE host for lane dialing: the gw-seats registry's
    own origin host (GW_SEATS_URL), default the GW tailscale IP.

    Why not the seat row's ``bind``? ``bind`` is the SERVING bind (the
    sglang/vllm bind-address on the GW box) — the live :8408 row for the
    serving :30000 seat advertises ``bind: "0.0.0.0"`` (gw-seats v0,
    verified live 2026-09-25), which is a wildcard listen address, not a
    dialable client host. The f0fb039 fixer_flash precedent pins the full
    URL (http://203.0.113.11:30000) for exactly this reason."""
    try:
        from urllib.parse import urlparse

        host = urlparse(_gw_seats_url()).hostname
        if isinstance(host, str) and host:
            return host
    except Exception:
        pass
    return "203.0.113.11"


def _lane_name_for_port(port: int) -> str:
    if port == FLASHNEXT_LANE_PORT:
        return FLASHNEXT_LANE_NAME
    if port == SLOT1_LANE_PORT:
        return SLOT1_LANE_NAME
    return f"port-{port}"


def _seat_rows(payload: dict) -> list[dict]:
    """The registry payload's seat rows (defensive: malformed -> [])."""
    seats = payload.get("seats")
    if not isinstance(seats, list):
        return []
    return [s for s in seats if isinstance(s, dict)]


def _lane_from_seats(seats: list[dict], lane_name: str) -> Optional[GateLane]:
    """Build a GateLane for ``lane_name`` from the seat rows, or None when
    the lane is absent or not serving.

    "Serving" is the registry's own state (state == "serving") — the
    registry is the seat-state contract's source of truth. A lane the
    registry declares down is NOT a lane; the caller's honest leg_down
    fires, never a masked legacy fallback.
    """
    for seat in seats:
        port = seat.get("port")
        if port != _lane_port(lane_name):
            continue
        if seat.get("state") != "serving":
            return None
        # ``bind`` is the serving-bind, NOT a client host (review HIGH-1,
        # 2026-09-25): the live serving row advertises "0.0.0.0", which
        # would build a phantom http://0.0.0.0:<port> lane. Wildcards (and
        # absent bind) dial the registry's own origin host instead; a
        # concrete bind is still honored (registry-forwarded rows).
        host = seat.get("bind")
        if not isinstance(host, str) or host.strip() in ("", "0.0.0.0", "::", "*", "[::]"):
            host = _registry_host()
        base_url = f"http://{host}:{port}"
        served_model = seat.get("model")
        if not isinstance(served_model, str) or not served_model:
            served_model = None
        return GateLane(name=lane_name, base_url=base_url, served_model=served_model)
    return None


def _lane_port(lane_name: str) -> int:
    if lane_name == FLASHNEXT_LANE_NAME:
        return FLASHNEXT_LANE_PORT
    if lane_name == SLOT1_LANE_NAME:
        return SLOT1_LANE_PORT
    raise ValueError(f"unknown gate lane {lane_name!r}")


def _reality_is_flashnext_solo(view: dict) -> bool:
    """True when the registry's reality_view says flashnext holds the seat
    (flashnext-solo: the 27B is down, :30000 is the live lane)."""
    reality = str(view.get("reality", "")).lower()
    if "flashnext" in reality:
        return True
    # Belt-and-braces: a reality string that names the flash model root as
    # the anchor is flashnext reality even if the label drifts.
    anchor = str(view.get("anchor", "")).lower()
    return FLASHNEXT_MODEL_ROOT_MARKER in anchor


def _reality_is_slot1(view: dict) -> bool:
    """True when the registry's reality_view says the 27B slot1 seat holds
    the seat (slot1-solo / 27B up)."""
    reality = str(view.get("reality", "")).lower()
    if "slot1" in reality or "27b" in reality:
        return True
    anchor = str(view.get("anchor", "")).lower()
    return SLOT1_MODEL_ROOT_MARKER in anchor


def _resolve_from_payload(payload: dict, lane: Optional[str] = None) -> Optional[GateLane]:
    """Resolve a gate lane from a parsed registry payload (the pure
    function the fetcher seam feeds).

    ``lane``: "flashnext" | "slot1" | None. None = "the live lane": the
    flashnext lane when reality is flashnext-solo, else the slot1 lane
    when the 27B is up, else None (registry readable but no gate lane is
    serving — callers treat this as the requested-lane-absent case, NOT
    as blind).

    Returns None ONLY for the blind/malformed cases (payload not a dict,
    no seats) — a readable registry with no serving gate lane returns
    None through the same shape, and callers distinguish the two by
    probing the registry separately when the distinction matters
    (``lane_availability`` below).
    """
    if not isinstance(payload, dict):
        return None
    seats = _seat_rows(payload)
    if not seats:
        return None

    if lane is None:
        view = payload.get("reality_view")
        view = view if isinstance(view, dict) else {}
        if _reality_is_flashnext_solo(view):
            return _lane_from_seats(seats, FLASHNEXT_LANE_NAME)
        if _reality_is_slot1(view):
            return _lane_from_seats(seats, SLOT1_LANE_NAME)
        # Reality label unrecognized: fall back to the seat-state truth —
        # whichever known gate lane is actually serving. (A readable
        # registry with an unlabeled reality is NOT blind.)
        for name in (FLASHNEXT_LANE_NAME, SLOT1_LANE_NAME):
            lane_obj = _lane_from_seats(seats, name)
            if lane_obj is not None:
                return lane_obj
        return None

    if lane not in (FLASHNEXT_LANE_NAME, SLOT1_LANE_NAME):
        return None
    return _lane_from_seats(seats, lane)


def _fetch_payload(timeout: float = GW_SEATS_TIMEOUT_S) -> dict:
    """Live registry read (GET {GW_SEATS_URL}/). Any error -> {} (blind).

    The registry is a SENSE read (GET), never a seat flip. Fail-soft:
    unreachable / non-200 / malformed JSON all collapse to the empty
    payload, which ``_resolve_from_payload`` maps to None (blind).
    """
    import httpx

    resp = httpx.get(f"{_gw_seats_url()}/", timeout=timeout)
    if resp.status_code != 200:
        return {}
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def resolve_gate_lane(
    lane: Optional[str] = None,
    fetcher: Optional[Callable[[], dict]] = None,
) -> Optional[GateLane]:
    """Resolve a gate lane through the gw-seats registry.

    Args:
        lane: "flashnext" | "slot1" | None. None = the live lane (the
            flashnext row when reality is flashnext-solo, the slot1 row
            when the 27B is up).
        fetcher: injectable callable returning the raw registry payload
            (the test seam — never a live call from a test). None = the
            live httpx read of ``GW_SEATS_URL``.

    Returns:
        GateLane (name, base_url, served_model) when the registry is
        readable and the lane is registered + serving.

        None in the ONLY blind case: the registry is unreachable, the
        payload is malformed, no seats are registered, or (for an
        explicit ``lane``) the lane is not registered / not serving.
        Callers fall back to the existing GW_URL behavior byte-identically
        on None — and an explicitly-requested inactive lane is reported as
        an honest leg_down by the caller, NEVER a silent legacy fallback
        (the "lying leg" the re-gate fold kills). Use ``lane_availability``
        to tell blind apart from lane-absent.
    """
    if fetcher is not None:
        payload = fetcher()
    else:
        try:
            payload = _fetch_payload()
        except Exception:
            return None  # blind — transport error
    return _resolve_from_payload(payload, lane)


def lane_availability(
    lane: Optional[str] = None,
    fetcher: Optional[Callable[[], dict]] = None,
) -> tuple[bool, str]:
    """Blind-vs-lane-absent sense for callers that must report an honest
    leg_down (mirrors the reason strings the vendored shim's
    ``gate_lane_serving`` reports, minus the local-voicing lease guard,
    which is the parent's S8 scope on the lapis-pm side).

    Returns (available, reason):
      * (True, "") — the lane resolved (registry readable + serving row).
      * (False, "registry_blind") — the ONLY case a caller may fall back to
        legacy GW_URL behavior.
      * (False, "<lane>_not_serving") — registry readable, lane absent or
        declared down. An honest leg_down; NEVER a masked legacy fallback.
    """
    lane_obj = resolve_gate_lane(lane=lane, fetcher=fetcher)
    if lane_obj is not None:
        return (True, "")
    if fetcher is not None:
        payload = fetcher()
    else:
        try:
            payload = _fetch_payload()
        except Exception:
            payload = None
    if isinstance(payload, dict) and _seat_rows(payload):
        return (False, f"{lane or 'gate'}_not_serving")
    return (False, "registry_blind")
