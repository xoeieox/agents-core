"""Hermetic tests for the promoted gate-lane registry (S1/S2/S4).

gate-lanes-registry-driven-flashnext-v0-agents-core. Every test is hermetic:
the gw-seats registry is stubbed through the ``fetcher`` seam (or a stubbed
httpx read), so ZERO live net. Coverage follows the spec's Tests section:

  (1) helper importable + flashnext row for a flashnext-solo stub payload
  (2) shim-parity: promoted helper output == the vendored shim's own local
      implementation for identical payloads (acceptance vectors pinned here,
      plus a live cross-check against the shim's pure resolver when the
      companion lapis_pm package is importable)
  (3) blind registry -> None, and operator resolution is unavailable
      (no invented model id, no legacy URL)
  (4) requested-inactive lane -> no legacy URL returned (honest leg_down)
  (5) adapter construction asserts base_url + served-id from the resolved lane
  (6) 27B-up stub -> slot1 row, gravitywell path unchanged
"""

import json

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    """Module-level hermeticity guard (PM-review fold, fix 1).

    Every test in this module must reach the gw-seats registry through the
    ``fetcher`` seam or an explicitly patched read. The guard makes any real
    socket connect from here fail loudly rather than silently passing on
    whatever the live registry happens to say — without it, an unstubbed
    ``lane_state`` call reads the real :8408 and the test's verdict depends on
    which seat holds the GPU at run time (that is exactly how the
    OPERATOR_DEFAULTS assertion passed while proving nothing).
    """
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "hermetic test attempted a live network call "
            "(stub agents_core.lane_registry.lane_state / resolve_gate_lane, "
            "or pass fetcher=...)"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    # Companion pin: the resolved base_url's HOST follows GW_SEATS_URL at call
    # time, so an ambient GW_SEATS_URL in the runner's environment would flip
    # every expected base_url (the payloads here are stubs, but their host is
    # not). Pin it to the contract default; tests that deliberately exercise a
    # different registry origin set the var themselves inside the test body,
    # which overrides this.
    from agents_core import lane_registry as _lr

    monkeypatch.setenv("GW_SEATS_URL", _lr.DEFAULT_GW_SEATS_URL)

# ---------------------------------------------------------------------------
# Stub registry payloads (shapes verified live against :8408 2026-09-25;
# see agents_core/lane_registry module docstring)
# ---------------------------------------------------------------------------

FLASHNEXT_SOLO_PAYLOAD = {
    "reality_view": {
        "reality": "flashnext-solo",
        "anchor": "/data/models/flash-next",
        "primary": {"port": 30000, "model_root": "/data/models/flash-next"},
    },
    "seats": [
        # The live serving row advertises bind "0.0.0.0" (a wildcard serving
        # bind, NOT a client host — review HIGH-1).
        {"port": 30000, "state": "serving", "bind": "0.0.0.0",
         "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
         "model_root": "/data/models/flash-next"},
        {"port": 8081, "state": "down", "bind": "0.0.0.0", "model": None},
        {"port": 8082, "state": "down", "bind": "0.0.0.0", "model": None},
    ],
}

SLOT1_UP_PAYLOAD = {
    "reality_view": {
        "reality": "slot1-solo",
        "anchor": "/data/models/qwen3-27b",
        "primary": {"port": 8081, "model_root": "/data/models/qwen3-27b"},
    },
    "seats": [
        {"port": 8081, "state": "serving", "bind": "0.0.0.0",
         "model": "gravitywell-27b", "model_root": "/data/models/qwen3-27b"},
        {"port": 30000, "state": "down", "bind": "0.0.0.0", "model": None},
    ],
}

FLASHNEXT_DOWN_PAYLOAD = {
    "reality_view": {"reality": "flashnext-solo", "anchor": "/data/models/flash-next"},
    "seats": [
        {"port": 30000, "state": "down", "bind": "0.0.0.0",
         "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"},
        {"port": 8081, "state": "serving", "bind": "0.0.0.0", "model": "gravitywell-122b"},
    ],
}

BLIND_PAYLOADS = [
    {},                                                     # empty (transport collapse)
    {"seats": "not-a-list"},                               # malformed seats
    {"seats": [1, "x", None]},                             # no dict rows
    {"reality_view": {"reality": "flashnext-solo"}},        # no seats at all
    "not-a-dict",                                           # malformed payload
]


def _fetcher(payload):
    """The injected-fetcher seam — a stubbed registry read, never a live call."""
    return lambda: payload


def _tup(lane_obj):
    """Normalize a GateLane for cross-module comparison (dataclass __eq__ is
    class-strict, so compare the contract fields, not the objects)."""
    if lane_obj is None:
        return None
    return (lane_obj.name, lane_obj.base_url, lane_obj.served_model)


# ---------------------------------------------------------------------------
# (1) helper importable + flashnext row for a flashnext-solo stub payload
# ---------------------------------------------------------------------------

def test_helper_importable_and_exports_contract():
    from agents_core import lane_registry

    assert callable(lane_registry.resolve_gate_lane)
    assert lane_registry.DEFAULT_GW_SEATS_URL == "http://203.0.113.11:8408"
    assert lane_registry.GATE_LANE_PORTS == (8081, 30000)
    assert lane_registry.FLASHNEXT_LANE_PORT == 30000
    assert lane_registry.FLASHNEXT_SERVED_ID == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"


def test_flashnext_solo_stub_returns_flashnext_row():
    from agents_core import lane_registry

    lane_obj = lane_registry.resolve_gate_lane(
        lane="flashnext", fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD)
    )
    assert _tup(lane_obj) == (
        "flashnext",
        "http://203.0.113.11:30000",
        "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
    )


def test_live_lane_follows_reality_view():
    """lane=None resolves the live lane from reality_view, not from a port."""
    from agents_core import lane_registry

    lane_obj = lane_registry.resolve_gate_lane(fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD))
    assert _tup(lane_obj) == (
        "flashnext",
        "http://203.0.113.11:30000",
        "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
    )
    lane_obj = lane_registry.resolve_gate_lane(fetcher=_fetcher(SLOT1_UP_PAYLOAD))
    assert _tup(lane_obj) == (
        "slot1",
        "http://203.0.113.11:8081",
        "gravitywell-27b",
    )


def test_concrete_bind_is_honored():
    """A concrete (registry-forwarded) bind is dialed as-is; only wildcards
    fall back to the registry origin host."""
    from agents_core import lane_registry

    payload = {
        "reality_view": {"reality": "flashnext-solo"},
        "seats": [{"port": 30000, "state": "serving", "bind": "10.0.0.7",
                   "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"}],
    }
    lane_obj = lane_registry.resolve_gate_lane(fetcher=_fetcher(payload))
    assert lane_obj.base_url == "http://10.0.0.7:30000"


def test_wildcard_bind_never_builds_phantom_lane():
    """HIGH-1 fold: bind 0.0.0.0 must not produce http://0.0.0.0:<port>."""
    from agents_core import lane_registry

    for bind in ("0.0.0.0", "::", "*", "", "  ", None):
        payload = {
            "reality_view": {"reality": "flashnext-solo"},
            "seats": [{"port": 30000, "state": "serving", "bind": bind,
                       "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"}],
        }
        lane_obj = lane_registry.resolve_gate_lane(fetcher=_fetcher(payload))
        assert lane_obj.base_url == "http://203.0.113.11:30000", bind


def test_registry_host_follows_gw_seats_url(monkeypatch):
    """base_url host = the registry's own origin host, read at CALL time."""
    from agents_core import lane_registry

    monkeypatch.setenv("GW_SEATS_URL", "http://10.9.8.7:8408")
    lane_obj = lane_registry.resolve_gate_lane(fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD))
    assert lane_obj.base_url == "http://10.9.8.7:30000"


def test_live_read_path_uses_root_path_and_env(monkeypatch):
    """The unstubbed path GETs {GW_SEATS_URL}/ (root path) via httpx."""
    from agents_core import lane_registry

    monkeypatch.setenv("GW_SEATS_URL", "http://registry.test:8408")
    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return FLASHNEXT_SOLO_PAYLOAD

    def _fake_get(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("httpx.get", _fake_get)
    lane_obj = lane_registry.resolve_gate_lane(lane="flashnext")
    assert captured["url"] == "http://registry.test:8408/"
    assert captured["timeout"] == lane_registry.GW_SEATS_TIMEOUT_S
    assert lane_obj.name == "flashnext"


@pytest.mark.parametrize("payload", BLIND_PAYLOADS)
def test_blind_payloads_resolve_none(payload):
    from agents_core import lane_registry

    assert lane_registry.resolve_gate_lane(fetcher=_fetcher(payload)) is None
    assert lane_registry.resolve_gate_lane(lane="flashnext", fetcher=_fetcher(payload)) is None
    lane_obj, reason = lane_registry.lane_state(fetcher=_fetcher(payload))
    assert lane_obj is None and reason == "registry_blind"


def test_transport_error_is_blind():
    """The unstubbed live-read path collapses a transport error to blind.

    (The shim's contract, kept byte-identically: an INJECTED fetcher's own
    errors propagate — only the live ``_fetch_payload`` read is swallowed.)
    """
    from agents_core import lane_registry

    def _boom(*args, **kwargs):
        raise RuntimeError("registry unreachable")

    with patch.object(lane_registry, "_fetch_payload", _boom):
        assert lane_registry.resolve_gate_lane(lane="flashnext") is None
        lane_obj, reason = lane_registry.lane_state(lane="flashnext")
    assert lane_obj is None and reason == "registry_blind"


def test_injected_fetcher_errors_propagate_shim_contract():
    """The shim lets an injected fetcher's exception escape (only the live
    read is fail-soft); the promoted helper must do the same."""
    from agents_core import lane_registry

    def _boom():
        raise RuntimeError("caller-side fetcher bug")

    with pytest.raises(RuntimeError, match="caller-side fetcher bug"):
        lane_registry.resolve_gate_lane(lane="flashnext", fetcher=_boom)


# ---------------------------------------------------------------------------
# (2) shim-parity: promoted helper == the vendored shim's local implementation
# ---------------------------------------------------------------------------

# Acceptance vectors transcribed from the vendored shim's own local
# implementation (lapis_pm/gate_lane.py, the protocol boundary): identical
# payload -> identical (name, base_url, served_model) / identical None case.
SHIM_ACCEPTANCE_VECTORS = [
    ("flashnext", FLASHNEXT_SOLO_PAYLOAD,
     ("flashnext", "http://203.0.113.11:30000", "Qwen3.8-Flash-Next-NVFP4-SSD-Stream")),
    (None, FLASHNEXT_SOLO_PAYLOAD,
     ("flashnext", "http://203.0.113.11:30000", "Qwen3.8-Flash-Next-NVFP4-SSD-Stream")),
    ("slot1", SLOT1_UP_PAYLOAD,
     ("slot1", "http://203.0.113.11:8081", "gravitywell-27b")),
    (None, SLOT1_UP_PAYLOAD,
     ("slot1", "http://203.0.113.11:8081", "gravitywell-27b")),
    ("flashnext", FLASHNEXT_DOWN_PAYLOAD, None),
    # reality_view is the authority for lane=None: reality says flashnext-solo,
    # so the live lane IS flashnext — and it is down, so None (the shim's
    # reality-first contract; the slot1 row serving underneath does NOT
    # silently become the live lane).
    (None, FLASHNEXT_DOWN_PAYLOAD, None),
    ("flashnext", {}, None),
    ("bogus", FLASHNEXT_SOLO_PAYLOAD, None),
]


@pytest.mark.parametrize("lane,payload,expected", SHIM_ACCEPTANCE_VECTORS)
def test_shim_acceptance_vectors(lane, payload, expected):
    from agents_core import lane_registry

    assert _tup(lane_registry.resolve_gate_lane(lane=lane, fetcher=_fetcher(payload))) == expected
    # The pure payload function (what the shim's local path feeds) agrees.
    assert _tup(lane_registry._resolve_from_payload(payload, lane)) == expected


def test_shim_parity_against_vendored_module():
    """Cross-check against the actual vendored shim's local implementation.

    The shim's ``resolve_gate_lane`` DELEGATES to this module when importable,
    so comparing it directly would be vacuous — the parity that matters is
    with the shim's own fallback implementation, which lives in its pure
    ``_resolve_from_payload`` / ``_seat_rows`` / ``_reality_is_*`` helpers.
    """
    pytest.importorskip("lapis_pm.gate_lane", reason="companion lapis_pm not installed")
    from lapis_pm import gate_lane as shim

    from agents_core import lane_registry

    for lane, payload, _expected in SHIM_ACCEPTANCE_VECTORS:
        mine = _tup(lane_registry._resolve_from_payload(payload, lane))
        theirs = _tup(shim._resolve_from_payload(payload, lane))
        assert mine == theirs, (lane, payload, mine, theirs)
        assert bool(lane_registry._seat_rows(payload)) == bool(shim._seat_rows(payload))

    # Constants that form the contract (the shim's own literals).
    assert lane_registry.FLASHNEXT_LANE_PORT == shim.FLASHNEXT_LANE_PORT
    assert lane_registry.SLOT1_LANE_PORT == shim.SLOT1_LANE_PORT
    assert lane_registry.FLASHNEXT_SERVED_ID == shim.FLASHNEXT_SERVED_ID
    assert lane_registry.DEFAULT_GW_SEATS_URL == shim.DEFAULT_GW_SEATS_URL
    assert lane_registry.GATE_LANE_PORTS == shim.GATE_LANE_PORTS

    # The shim's delegation boundary: with this module importable, the shim
    # routes through the promoted helper (protocol boundary made primary).
    seen = {}

    def _spy(lane=None, fetcher=None):
        seen["lane"] = lane
        return lane_registry.GateLane(name="flashnext", base_url="sentinel",
                                      served_model="sentinel-id")

    with patch.object(lane_registry, "resolve_gate_lane", _spy):
        out = shim.resolve_gate_lane(lane="flashnext", fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD))
    assert seen.get("lane") == "flashnext"
    assert out.base_url == "sentinel"


def test_shim_delegation_fail_closed():
    """A companion-side runtime error degrades the shim, never the gate: the
    shim falls through to its own local implementation (LOW-7)."""
    shim = pytest.importorskip("lapis_pm.gate_lane", reason="companion lapis_pm not installed")
    from agents_core import lane_registry

    def _raise(*args, **kwargs):
        raise RuntimeError("companion bug")

    with patch.object(lane_registry, "resolve_gate_lane", _raise):
        out = shim.resolve_gate_lane(lane="flashnext", fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD))
    assert _tup(out) == (
        "flashnext", "http://203.0.113.11:30000", "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    )


# ---------------------------------------------------------------------------
# (3) blind registry -> operator unavailable, no invented model id
# ---------------------------------------------------------------------------

def test_flashnext_in_operator_defaults_but_none_valued():
    """The operator is registered (so call_operator accepts it and the
    unknown-operator error lists it) but its stored default is a None marker —
    never a hardcoded served id."""
    from agents_core.llm import OPERATOR_DEFAULTS

    assert "flashnext" in OPERATOR_DEFAULTS
    assert dict.__getitem__(OPERATOR_DEFAULTS, "flashnext") is None
    # Every other operator keeps its literal default (byte-identical).
    assert OPERATOR_DEFAULTS["gravitywell"] == "gravitywell-122b"
    assert OPERATOR_DEFAULTS["phala"] == "deepseek/deepseek-v4-flash-0731"


def test_operator_default_resolves_served_id_from_registry(monkeypatch):
    """A readable registry resolves the model to the registry's served id —
    the pin, not a literal.

    PM-review fold (fix 1): the serving case is STUBBED at the top of the
    test. Previously the first assert ran unstubbed and live-read the gw-seats
    registry at :8408, so it passed only because the flash-next seat happened
    to hold the GPU — a blind registry (27B up, or registry down) would have
    flipped it to KeyError, and the module's socket guard now makes that live
    read fail loudly instead of passing on luck. The post-stub blind asserts
    are kept: they pin the "blind => no model id to claim" shape.
    """
    from agents_core import lane_registry
    from agents_core.llm import OPERATOR_DEFAULTS

    monkeypatch.setattr(
        lane_registry, "lane_state",
        lambda lane=None, fetcher=None: (
            lane_registry.GateLane(
                name="flashnext",
                base_url="http://203.0.113.11:30000",
                served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
            "")),

    assert OPERATOR_DEFAULTS["flashnext"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert OPERATOR_DEFAULTS.get("flashnext") == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"

    monkeypatch.setattr(
        lane_registry, "lane_state",
        lambda lane=None, fetcher=None: (None, "registry_blind"),
    )
    with pytest.raises(KeyError):
        OPERATOR_DEFAULTS["flashnext"]
    assert OPERATOR_DEFAULTS.get("flashnext") is None
    assert OPERATOR_DEFAULTS.get("flashnext", "fallback") == "fallback"


def test_wave_max_tokens_reaches_the_lane_post(monkeypatch):
    """PM-review fold (fix 2): max_tokens must survive the flashnext allowlist.

    Wave mode passes WAVE_SEAT_MAX_TOKENS (500) through GravityWellAdapter
    (D6: cap the compounding re-prefill a long seat statement causes across
    every later round). If the lane path dropped the kwarg, every
    flashnext-voiced wave seat would silently fall back to the GW_MAX_TOKENS
    default (4096) — the exact cost shape D6 exists to prevent.
    """
    from agents_core import lane_registry
    from agents_core.council.cli import WAVE_SEAT_MAX_TOKENS
    import agents_core.llm as llm

    captured = {}

    def _fake_post(**kwargs):
        captured.update(kwargs)
        return "voiced"

    monkeypatch.setattr(lane_registry, "lane_state",
                        lambda lane=None, fetcher=None: (
                            lane_registry.GateLane(
                                name="flashnext",
                                base_url="http://203.0.113.11:30000",
                                served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
                            ""))
    monkeypatch.setattr(llm, "_post_chat_completion", _fake_post)

    prov = []
    out = llm.call_operator("flashnext", prompt="seat statement",
                            max_tokens=WAVE_SEAT_MAX_TOKENS, _provenance_out=prov)

    assert out == "voiced"
    assert captured["max_tokens"] == 500 == WAVE_SEAT_MAX_TOKENS
    assert prov == [("success", "flashnext")]


def test_max_tokens_unset_stays_unset(monkeypatch):
    """The fold must not invent a cap: unset -> None, letting
    _post_chat_completion apply its own env-overridable default
    (byte-identical to the pre-fold shape for non-wave callers)."""
    from agents_core import lane_registry
    import agents_core.llm as llm

    captured = {}

    def _fake_post(**kwargs):
        captured.update(kwargs)
        return "voiced"

    monkeypatch.setattr(lane_registry, "lane_state",
                        lambda lane=None, fetcher=None: (
                            lane_registry.GateLane(
                                name="flashnext",
                                base_url="http://203.0.113.11:30000",
                                served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
                            ""))
    monkeypatch.setattr(llm, "_post_chat_completion", _fake_post)

    llm.call_operator("flashnext", prompt="hi")

    assert captured["max_tokens"] is None


def test_adapter_max_tokens_threads_through_the_lane(monkeypatch):
    """The wave seat's adapter-level cap reaches call_operator on the lane
    path (the D6 plumbing end-to-end, adapter -> call_operator)."""
    from agents_core import lane_registry
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter(
        temperature=0.8, max_tokens=500,
        lane=lane_registry.GateLane(
            name="flashnext", base_url="http://203.0.113.11:30000",
            served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"))

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="voiced") as mock_op:
        adapter.chat("CARD", [MagicMock(role="user", content="hi")])

    _args, kwargs = mock_op.call_args
    assert kwargs["max_tokens"] == 500


def test_no_reason_defaults_to_unavailable_not_blind(monkeypatch):
    """PM-review fold (fix 4): a None lane with NO reason must not be
    reported as registry_blind. registry_blind is the key that licenses the
    caller's legacy gravitywell fallback, so inventing it for an
    unknown-state lane would hand a silent re-route to the 122B — fail-closed
    means the default is the non-blind flashnext_unavailable.
    """
    from agents_core import lane_registry
    from agents_core.llm import FlashnextLaneUnavailable, call_operator

    monkeypatch.setattr(lane_registry, "lane_state",
                        lambda lane=None, fetcher=None: (None, ""))

    prov = []
    with patch("agents_core.llm._post_chat_completion") as mock_post:
        with pytest.raises(FlashnextLaneUnavailable) as exc:
            call_operator("flashnext", prompt="p", _provenance_out=prov)

    mock_post.assert_not_called()
    assert exc.value.reason == "flashnext_unavailable"
    assert exc.value.reason != "registry_blind"
    # Provenance and the raised reason must agree — a caller reading either
    # must reach the same conclusion about whether legacy fallback is licensed.
    assert prov == [("flashnext_unavailable", "flashnext")]


def test_blind_registry_call_operator_is_unavailable_blind():
    """Registry blind -> operator unavailable with reason=registry_blind, the
    ONLY reason a caller may fall back to the gravitywell path. No HTTP call
    is made and no legacy URL is returned."""
    from agents_core import lane_registry
    from agents_core.llm import FlashnextLaneUnavailable, call_operator

    prov = []
    with patch.object(lane_registry, "lane_state",
                      return_value=(None, "registry_blind")), \
         patch("agents_core.llm._post_chat_completion") as mock_post:
        with pytest.raises(FlashnextLaneUnavailable) as exc:
            call_operator("flashnext", prompt="p", _provenance_out=prov)
    mock_post.assert_not_called()
    assert exc.value.reason == "registry_blind"
    assert prov == [("registry_blind", "flashnext")]


# ---------------------------------------------------------------------------
# (4) requested-inactive lane -> no legacy URL, honest leg_down
# ---------------------------------------------------------------------------

def test_requested_inactive_lane_returns_no_legacy_url():
    """Readable registry, lane down: reason is NOT registry_blind (so no
    caller may fall back) and no legacy GW_URL is dialed."""
    from agents_core import lane_registry
    from agents_core.llm import FlashnextLaneUnavailable, call_operator

    lane_obj, reason = lane_registry.lane_state(
        lane="flashnext", fetcher=_fetcher(FLASHNEXT_DOWN_PAYLOAD)
    )
    assert lane_obj is None
    assert reason == "flashnext_not_serving"
    assert reason != "registry_blind"

    prov = []
    with patch.object(lane_registry, "lane_state", return_value=(None, reason)), \
         patch("agents_core.llm._post_chat_completion") as mock_post:
        with pytest.raises(FlashnextLaneUnavailable) as exc:
            call_operator("flashnext", prompt="p", _provenance_out=prov)
    mock_post.assert_not_called()
    assert exc.value.reason == "flashnext_not_serving"
    assert prov == [("flashnext_not_serving", "flashnext")]


def test_flashnext_call_dials_registry_lane_never_gw_url(monkeypatch):
    """A serving flashnext row dials the registry base_url with the registry
    served id — and NOT GW_URL / the gravitywell backend."""
    from agents_core import lane_registry
    from agents_core.llm import GW_URL, call_operator

    captured = {}

    def _fake_post(**kwargs):
        captured.update(kwargs)
        return "voiced"

    prov = []
    with patch.object(lane_registry, "lane_state",
                      lambda lane=None, fetcher=None: (
                          lane_registry.GateLane(
                              name="flashnext",
                              base_url="http://203.0.113.11:30000",
                              served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
                          "")), \
         patch("agents_core.llm._post_chat_completion", _fake_post), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=AssertionError("gravitywell path must not be used")):
        out = call_operator("flashnext", prompt="hi", system="sys", _provenance_out=prov)

    assert out == "voiced"
    assert captured["base_url"] == "http://203.0.113.11:30000"
    assert captured["base_url"] != GW_URL
    assert captured["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert captured["_no_thinking"] is True
    assert prov == [("success", "flashnext")]


def test_flashnext_http_failure_is_lane_down_not_legacy():
    """An unreachable lane raises the lane's own error (fail-closed), never a
    silent gravitywell/paid re-route."""
    from agents_core import lane_registry
    from agents_core.llm import (
        FlashnextLaneUnavailable, OperatorUnreachableError, call_operator,
    )

    def _boom(**kwargs):
        raise OperatorUnreachableError("http://203.0.113.11:30000", RuntimeError("down"))

    prov = []
    with patch.object(lane_registry, "lane_state",
                      return_value=(lane_registry.GateLane(
                          name="flashnext",
                          base_url="http://203.0.113.11:30000",
                          served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"), "")), \
         patch("agents_core.llm._post_chat_completion", _boom):
        with pytest.raises(FlashnextLaneUnavailable) as exc:
            call_operator("flashnext", prompt="p", _provenance_out=prov)

    assert exc.value.reason == "flashnext_unreachable"
    assert prov == [("serving_http_error", "flashnext")]


def test_flashnext_model_swap_rejects_non_registry_id():
    from agents_core import lane_registry
    from agents_core.llm import call_operator

    with patch.object(lane_registry, "lane_state",
                      return_value=(lane_registry.GateLane(
                          name="flashnext",
                          base_url="http://203.0.113.11:30000",
                          served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"), "")), \
         patch("agents_core.llm._post_chat_completion") as mock_post:
        with pytest.raises(ValueError, match="single fixed model"):
            call_operator("flashnext", prompt="p", model="gravitywell-122b")
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# (5) adapter construction asserts base_url + served-id from the resolved lane
# ---------------------------------------------------------------------------

def test_build_adapter_flashnext_carries_registry_lane():
    from agents_core import lane_registry
    from agents_core.council.cli import _build_adapter
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    with patch.object(lane_registry, "lane_state",
                      lambda lane=None, fetcher=None: (
                          lane_registry.GateLane(
                              name="flashnext",
                              base_url="http://203.0.113.11:30000",
                              served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
                          "")):
        adapter = _build_adapter("flashnext", ClaudeAdapter=None, LlamaAdapter=None,
                                 run_id="r1")

    assert isinstance(adapter, GravityWellAdapter)
    assert adapter.lane.base_url == "http://203.0.113.11:30000"
    assert adapter.lane.served_model == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"


def test_build_adapter_flashnext_refuses_inactive_lane():
    """No lying leg: an explicitly-requested inactive lane raises here rather
    than constructing a gravitywell adapter.

    PM-review fold (fix 3): the raise is the typed FlashnextLaneUnavailable
    carrying .reason, so a caller can tell the one legacy-licensed shape
    (registry_blind) from an honest leg_down (<lane>_not_serving) without
    string-matching the message.
    """
    from agents_core import lane_registry
    from agents_core.council.cli import _build_adapter
    from agents_core.llm import FlashnextLaneUnavailable

    with patch.object(lane_registry, "lane_state",
                      return_value=(None, "flashnext_not_serving")):
        with pytest.raises(FlashnextLaneUnavailable,
                           match="flashnext voicing unavailable") as exc:
            _build_adapter("flashnext", ClaudeAdapter=None, LlamaAdapter=None)
    assert exc.value.reason == "flashnext_not_serving"
    assert exc.value.reason != "registry_blind"

    with patch.object(lane_registry, "lane_state",
                      return_value=(None, "registry_blind")):
        with pytest.raises(FlashnextLaneUnavailable, match="registry_blind") as exc:
            _build_adapter("flashnext", ClaudeAdapter=None, LlamaAdapter=None)
    assert exc.value.reason == "registry_blind"

    # The reason must be carried on the exception object, not only in the
    # message — that is the whole point of the typed raise.
    assert "reason=" in str(exc.value)


def test_flashnext_adapter_voices_lane_without_gw_lease():
    """The lane-voiced adapter routes through the flashnext operator with the
    lane it was built with, and requests no gravitywell lease/principal."""
    from agents_core import lane_registry
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter(temperature=0.8, lane=lane_registry.GateLane(
        name="flashnext",
        base_url="http://203.0.113.11:30000",
        served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"))

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="voiced") as mock_op:
        out = adapter.chat("CARD", [MagicMock(role="user", content="hi")])

    assert out == "voiced"
    args, kwargs = mock_op.call_args
    assert args[0] == "flashnext"
    assert kwargs["_lane"].base_url == "http://203.0.113.11:30000"
    assert "principal" not in kwargs and "lease_class" not in kwargs
    assert adapter.voicing_events == []  # provenance comes from call_operator


def test_gravitywell_adapter_unchanged_without_lane():
    """Byte-identical invariant: lane unset -> the gravitywell call shape is
    exactly what it was before this target."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter(temperature=0.8, principal="council-delib-x")
    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="voiced") as mock_op:
        adapter.chat("CARD", [MagicMock(role="user", content="hi")])

    args, kwargs = mock_op.call_args
    assert args[0] == "gravitywell"
    assert kwargs == {
        "system": "CARD", "temperature": 0.8, "timeout": 300,
        "on_wake_fail": "park", "principal": "council-delib-x",
        "_provenance_out": [], "lease_class": "protected",
    }
    assert adapter.lane is None


def test_voicing_provenance_labels_lane_not_gravitywell():
    """A flashnext-voiced run must never read as a gravitywell one (the phala
    explicit-banner rule), and a clean lane run is not degraded."""
    from agents_core import lane_registry
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter(
        lane=lane_registry.GateLane(
            name="flashnext",
            base_url="http://203.0.113.11:30000",
            served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
        voicing_events=[{"effective_operator": "flashnext", "reason": "success"}],
    )
    run = {"voicing": "flashnext", "turns": [{"step": 1}]}
    _apply_voicing_provenance(run, adapter)
    assert run["effective_voicing"] == "flashnext:Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert run["voicing_degraded"] is False
    assert run["turns"][0]["effective_voicing"] == "flashnext"


def test_voicing_provenance_records_lane_failure_honestly():
    from agents_core import lane_registry
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter(
        lane=lane_registry.GateLane(
            name="flashnext", base_url="http://203.0.113.11:30000",
            served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"),
        voicing_events=[{"effective_operator": "flashnext",
                         "reason": "serving_http_error"}],
    )
    run = {"voicing": "flashnext", "turns": []}
    _apply_voicing_provenance(run, adapter)
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "serving_http_error"
    assert "gravitywell" not in run["effective_voicing"]


def test_wave_seats_share_registry_lane():
    """Wave mode resolves the lane once for the whole wave and refuses a dead
    lane rather than waving on the gravitywell seat.

    PM-review fold (fix 3): the refusal is the typed FlashnextLaneUnavailable
    carrying .reason (callers branch on it), not a bare ValueError.
    """
    from agents_core import lane_registry
    from agents_core.llm import FlashnextLaneUnavailable
    import agents_core.council.cli as cli

    with patch.object(lane_registry, "lane_state",
                      return_value=(None, "flashnext_not_serving")):
        with pytest.raises(FlashnextLaneUnavailable, match="wave mode") as exc:
            cli._run_wave_deliberation(
                run={"voicing": "flashnext", "selected_entities": [], "decision": "d",
                     "turns_cap": 2, "turns": [], "gw_principal": None},
                run_id="r", Engine=MagicMock(), hold_active=False, doorman=None,
                hold_work_id="w", hold_principal="p", refresh_threads=[],
                CharacterEntity=MagicMock(), NarratorEntity=MagicMock(),
            )
    assert exc.value.reason == "flashnext_not_serving"


# ---------------------------------------------------------------------------
# (6) 27B-up stub -> slot1 row, gravitywell path unchanged
# ---------------------------------------------------------------------------

def test_slot1_up_stub_returns_slot1_row():
    from agents_core import lane_registry

    lane_obj = lane_registry.resolve_gate_lane(lane="slot1", fetcher=_fetcher(SLOT1_UP_PAYLOAD))
    assert _tup(lane_obj) == ("slot1", "http://203.0.113.11:8081", "gravitywell-27b")


def test_27b_up_gravitywell_path_unchanged():
    """With the 27B up, nothing on the gravitywell path changes: the flashnext
    lane is simply not the live lane, and the gravitywell operator still
    resolves its own default from GW discovery."""
    from agents_core import lane_registry
    from agents_core.llm import GW_URL, OPERATOR_DEFAULTS

    assert _tup(lane_registry.resolve_gate_lane(
        lane="flashnext", fetcher=_fetcher(SLOT1_UP_PAYLOAD))) is None
    assert OPERATOR_DEFAULTS["gravitywell"] == "gravitywell-122b"

    prov = []
    from agents_core.doorman_client import DoormanClient
    with patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer") as gw, \
         patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"):
        from agents_core.llm import call_operator
        out = call_operator("gravitywell", prompt="hi", _provenance_out=prov)

    assert out == "gw answer"
    assert gw.call_args.kwargs.get("_url", GW_URL) == GW_URL
    assert prov[-1] == ("success", "gravitywell")


def test_locality_record_uses_registry_host():
    """The ledger records the registry-resolved host for a flashnext call
    (never a stale literal), and never raises into the call."""
    from agents_core import lane_registry
    import agents_core.llm as llm

    records = []

    def _rec(**kwargs):
        records.append(kwargs)

    with patch.object(lane_registry, "lane_state",
                      return_value=(lane_registry.GateLane(
                          name="flashnext",
                          base_url="http://203.0.113.11:30000",
                          served_model="Qwen3.8-Flash-Next-NVFP4-SSD-Stream"), "")), \
         patch("agents_core.llm._post_chat_completion", return_value="voiced"), \
         patch("agents_core.locality.record", _rec):
        llm.call_operator("flashnext", prompt="hi")

    assert len(records) == 1
    assert records[0]["host"] == "http://203.0.113.11:30000"
    assert records[0]["requested_operator"] == "flashnext"
    assert records[0]["cost_class"] == "local-gw"
    assert records[0]["served_model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert records[0]["ok"] is True


def test_no_live_net_in_this_module(monkeypatch):
    """Hermeticity guard: any real socket use from these tests fails loudly."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("hermetic test attempted a live network call")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    from agents_core import lane_registry

    assert lane_registry.resolve_gate_lane(
        lane="flashnext", fetcher=_fetcher(FLASHNEXT_SOLO_PAYLOAD)
    ).name == "flashnext"
