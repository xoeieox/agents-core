"""D1/D2/D3 stop-path partition tests (gw-doorman-flashnext-idle-awareness-v0).

The 2026-09-22 GPU1-wedge incident: the doorman idle-ejected a SERVING box
whose vLLM gauges were idle while the live brain was the flash-next seat on
:30000 — the D3 window guard withholds only the handover case (seat up +
day seat DOWN); the overlap case (seat up + day seat up-idle) left the
window "none", the guard inert, and the eject enabled a guard-correct
suspend that killed the unsupervised seat.

These tests drive the REAL _decide_idle_stop() (hermetic mocks; the verb
asserts on the subprocess.run call, never on a re-implemented branch) over
the actual state space:

  flashnext_state in {down, blind, up_registered, up_unverified,
                      up_foreign, None (cold)}
  x day-seat-serving (the D3 window's other half)
  x vLLM probe tri-state (serving / idle / blind)
  x lease-held

plus the D1 bookkeeping in the tick's probe pass (_refresh_serving_cache
with the probes mocked), the D2 legibility probe, and the /status
rendering. Non-regression: the D3 card_held_flashnext path and the D9
close re-anchor one-shot semantics are asserted unchanged.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from agents_core import doorman_server as ds
from agents_core.doorman_server import (
    _NodeState,
    _SGLANG_ACTIVITY_METRICS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def node(tmp_path, monkeypatch):
    """In-process _NodeState; the idle log points at a tmp file so the
    tests can assert the D3 idle-log rows (the durable machine-parseable
    audit substrate), not just the journal.

    HERMETIC GUARD: status_snapshot() reads the gw_topology importability
    check (the "actuator_available" field) — on this host the live deploy
    tree makes that import take a slow path that can exceed the test
    budget. The field is out of scope for this spec, so it is pinned to a
    fast constant here (the value is never asserted)."""
    idle_log = tmp_path / "idle.jsonl"
    monkeypatch.setattr(ds, "DOORMAN_IDLE_LOG", str(idle_log))
    monkeypatch.setattr(ds, "_gw_topology_importable", lambda: False)
    state = _NodeState(gw_url="http://mock.internal/", node_name="gravitywell")
    yield state


def idle_rows(node) -> list[dict]:
    """Parse the idle-log file written during the test."""
    import pathlib
    p = pathlib.Path(ds.DOORMAN_IDLE_LOG)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def set_seat(node, state, served_id=None, registered=None, error_class=None):
    """Populate the seat-state fields the way the tick's probe pass does."""
    with node.lock:
        node._flashnext_state = state
        node._flashnext_served_id = served_id
        node._flashnext_registered = registered
        node._flashnext_error_class = error_class


def run_stop(node, **state_kwargs) -> tuple[bool, list]:
    """Run the real _decide_idle_stop() with the stop verb and the
    serving re-check mocked; return (return_value, subprocess.run calls)."""
    calls = []

    def _fake_run(*a, **kw):
        calls.append((a, kw))
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run), \
         patch.object(node, "_is_serving", return_value=False):
        result = node._decide_idle_stop()
    return result, calls


def stopped_reason(node) -> str | None:
    """The reason field of the last 'stopped' idle-log row (None if the
    stop verb never fired)."""
    rows = idle_rows(node)
    stopped = [r for r in rows if r["event"] == "stopped"]
    return stopped[-1]["reason"] if stopped else None


# ---------------------------------------------------------------------------
# D1: the seat-state partition inside _decide_idle_stop()
# ---------------------------------------------------------------------------

class TestStopPathSeatPartition:
    """The verb fires only in the idle-ok case: seat definitively down
    (or foreign) AND the vLLM axis idle per its existing rules."""

    def _idle_node(self, node, idle_secs=632.0):
        with node.lock:
            node.idle_since = time.time() - idle_secs
            node._cached_serving = True
            node.service_stopped = False
            node._stop_in_flight = False
            node.leases = {}
            node._probe_indeterminate = False
            node._serving_is_big = False
            node._cached_topology_state = None
        return node

    # -- the incident's overlap case -------------------------------------

    @pytest.mark.parametrize("seat_state", ["up_registered", "up_unverified"])
    def test_overlap_withhold_day_seat_up_idle_seat_up(self, node, seat_state):
        """The incident shape: day seat UP-and-idle (vLLM gauges idle past
        grace), the :30000 seat up and serving. NO verb; the idle log
        carries a flashnext_withheld_* row (D3)."""
        self._idle_node(node)
        set_seat(node, seat_state, served_id="gravitywell-flashnext-27b",
                 registered=(seat_state == "up_registered"))
        with node.lock:
            node._flashnext_last_activity_ts = time.time() - 300.0
        result, calls = run_stop(node)
        assert result is True  # the loop continues; the stop is withheld
        assert calls == []  # the gw-serve stop verb NEVER fires
        rows = idle_rows(node)
        events = [r["event"] for r in rows]
        assert "flashnext_withheld_active" in events
        assert "stopped" not in events
        row = next(r for r in rows if r["event"] == "flashnext_withheld_active")
        assert row["seat_state"] == seat_state
        assert row["idle_secs"] is not None and row["idle_secs"] > 600
        # a withhold is a normal safety state, not a failure (D3)
        assert node.last_error is None

    @pytest.mark.parametrize("seat_state", ["up_registered", "up_unverified"])
    def test_seat_up_withholds_unconditionally_regardless_of_activity(
            self, node, seat_state):
        """D1: the seat-up withhold is UNCONDITIONAL — a live run with a
        >600s request gap (the 292k-run shape) must not become stop-
        eligible; a stale/absent activity stamp renders the substate
        withheld-up-idle, not idle-ok."""
        self._idle_node(node)
        set_seat(node, seat_state)
        with node.lock:
            node._flashnext_last_activity_ts = None  # never observed
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        events = [r["event"] for r in rows]
        assert "flashnext_withheld_up_idle" in events
        assert "flashnext_withheld_active" not in events
        assert "stopped" not in events

    # -- the blind bound ---------------------------------------------------

    def test_blind_within_bound_withholds(self, node):
        """Seat blind, idle past grace, grace clock within the bound
        (grace + DOORMAN_PROBE_BLINDNESS_SEC): no verb; a
        flashnext_blind_hold row. The bound is measured on idle_elapsed
        (the grace clock — the spec's grace-pause semantics and the
        vLLM-axis precedent), so the blind-since fixture is armed at
        idle_start here (both clocks coincide in this fixture)."""
        self._idle_node(node)
        set_seat(node, "blind", error_class="Timeout")
        with node.lock:
            node._flashnext_blind_since = time.time() - 400.0
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        row = next(r for r in rows if r["event"] == "flashnext_blind_hold")
        assert row["seat_state"] == "blind"
        assert row["error_class"] == "Timeout"
        assert "stopped" not in [r["event"] for r in rows]

    def test_blind_after_idle_accumulated_bound_is_idle_elapsed(self, node):
        """The two clocks coincide ONLY when the seat went blind at
        idle_start. Here the seat went blind 400s AFTER idle already
        had 632s accumulated: the continuous-blindness duration (400s)
        is within the bound, but the spec's grace-pause semantics bind
        on idle_elapsed (632s + 900s bound -> still withheld; this
        fixture pins the divergence case the blind-since fixtures
        cannot). No verb."""
        self._idle_node(node, idle_secs=632.0)
        set_seat(node, "blind", error_class="Timeout")
        bound = ds.GW_STOP_GRACE_SEC + ds.DOORMAN_PROBE_BLINDNESS_SEC
        assert 400.0 < bound  # blind duration alone is within the bound
        with node.lock:
            node._flashnext_blind_since = time.time() - 400.0
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        assert any(r["event"] == "flashnext_blind_hold" for r in rows)
        assert "stopped" not in [r["event"] for r in rows]

    def test_blind_after_idle_accumulated_past_bound_proceeds(self, node):
        """The divergence case in the other direction: idle 1000s
        accumulated, the seat went blind only 400s ago. The continuous-
        blindness duration (400s) is within the bound, but idle_elapsed
        (1000s) is past it — the grace-pause semantics release the
        withhold on idle_elapsed (the vLLM-axis precedent), so the stop
        proceeds with the distinct stop_reason. Under the old
        blind-since clock this fixture would have withheld."""
        self._idle_node(node, idle_secs=1000.0)
        set_seat(node, "blind", error_class="Timeout")
        bound = ds.GW_STOP_GRACE_SEC + ds.DOORMAN_PROBE_BLINDNESS_SEC
        assert 400.0 < bound < 1000.0  # blind duration within, idle past
        with node.lock:
            node._flashnext_blind_since = time.time() - 400.0
        with patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="", stderr="")) as run_mock, \
             patch.object(node, "_is_serving", return_value=False), \
             patch.object(ds.log, "critical") as crit_mock:
            result = node._decide_idle_stop()
        assert result is True
        assert run_mock.called
        assert stopped_reason(node) == "flashnext_probe_blind_bound_exceeded"
        assert crit_mock.called

    def test_blind_bound_exceeded_stop_proceeds_distinct_reason(self, node):
        """Continuous blind past GW_STOP_GRACE_SEC +
        DOORMAN_PROBE_BLINDNESS_SEC (armed at idle_start, so the blind
        clock and idle_elapsed coincide): this axis stops withholding;
        the vLLM axis is confirmed idle, so the stop proceeds with
        stop_reason EXACTLY flashnext_probe_blind_bound_exceeded (D1)
        and the event logs at CRITICAL."""
        bound = ds.GW_STOP_GRACE_SEC + ds.DOORMAN_PROBE_BLINDNESS_SEC
        self._idle_node(node, idle_secs=bound + 60.0)
        set_seat(node, "blind", error_class="Timeout")
        with node.lock:
            node._flashnext_blind_since = time.time() - (bound + 60.0)
        with patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="", stderr="")) as run_mock, \
             patch.object(node, "_is_serving", return_value=False), \
             patch.object(ds.log, "critical") as crit_mock:
            result = node._decide_idle_stop()
        assert result is True
        assert run_mock.called  # the gw-serve stop verb fired
        assert stopped_reason(node) == "flashnext_probe_blind_bound_exceeded"
        assert crit_mock.called  # the bound-exceeded event is CRITICAL
        crit_text = " ".join(str(c.args[0]) for c in crit_mock.call_args_list
                             if c.args)
        assert "flashnext_probe_blind_bound_exceeded" in crit_text

    def test_double_blind_past_both_bounds_stop_proceeds(self, node):
        """Both axes blind past both bounds (idle past the vLLM-axis bound
        too): the stop proceeds with the vLLM-axis's own distinct reason
        (probe_blind_bound_exceeded — the vLLM axis is the deciding
        factor); the seat axis has already fallen through its bound.
        The seat's blind clock is armed at idle_start here, so it is
        past the bound too (both clocks coincide in this fixture)."""
        bound = ds.GW_STOP_GRACE_SEC + ds.DOORMAN_PROBE_BLINDNESS_SEC
        self._idle_node(node, idle_secs=bound + 60.0)
        set_seat(node, "blind")
        with node.lock:
            node._flashnext_blind_since = time.time() - (bound + 60.0)
            node._probe_indeterminate = True
        with patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="", stderr="")) as run_mock, \
             patch.object(node, "_is_serving", return_value=False):
            result = node._decide_idle_stop()
        assert result is True
        assert run_mock.called
        assert stopped_reason(node) == "probe_blind_bound_exceeded"

    # -- definitive states: down / up_foreign -----------------------------

    @pytest.mark.parametrize("seat_state", ["down", "up_foreign"])
    def test_definitive_states_do_not_withhold(self, node, seat_state):
        """Seat definitively down (or a foreign occupant — not our seat):
        this axis does not withhold; the vLLM axis decides, as today.
        vLLM confirmed idle past grace -> the stop fires confirmed_idle."""
        self._idle_node(node)
        set_seat(node, seat_state,
                 served_id="other-model" if seat_state == "up_foreign" else None,
                 registered=False if seat_state == "up_foreign" else None)
        with patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="", stderr="")) as run_mock, \
             patch.object(node, "_is_serving", return_value=False):
            result = node._decide_idle_stop()
        assert result is True
        assert run_mock.called
        assert stopped_reason(node) == "confirmed_idle"

    # -- cold start --------------------------------------------------------

    def test_cold_start_never_idle_ok(self, node):
        """COLD START (D1): before the first definitive probe read the
        state is None — treated as BLIND (bounded withhold), never as
        down/idle-ok. Fresh state (never-observed clock + seat state
        None) x grace elapsed -> no verb."""
        self._idle_node(node)
        # _flashnext_state is None by construction (cold).
        assert node._flashnext_state is None
        with node.lock:
            node._flashnext_blind_since = time.time() - 100.0
            node._flashnext_last_activity_ts = None
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        assert any(r["event"] == "flashnext_blind_hold"
                   and r["seat_state"] is None for r in rows)
        assert "stopped" not in [r["event"] for r in rows]

    # -- lease-held / vLLM-serving preconditions ---------------------------

    def test_lease_held_no_stop(self, node):
        """A held lease means idle_since is None — the whole stop block is
        skipped; the seat partition never runs."""
        with node.lock:
            node.idle_since = None
            node.leases = {"wid": {"acquired_at": time.time(), "ttl_sec": 60,
                                   "reason": "t", "role": "worker"}}
        result, calls = run_stop(node)
        assert result is False
        assert calls == []
        assert idle_rows(node) == []

    def test_vllm_serving_no_stop(self, node):
        """A serving vLLM axis blocks the stop: while the day seat is
        serving the refresh loop re-anchors idle_since every tick
        (source 'probe', _refresh_serving_cache), so the stop block
        never sees idle_since at all — and _decide_idle_stop() with
        idle_since=None issues no verb (the documented precondition
        this method relies on)."""
        self._idle_node(node)
        set_seat(node, "down")
        with node.lock:
            node.idle_since = None  # the serving read re-anchored the clock
            node._idle_since_source = "probe"
        result, calls = run_stop(node)
        assert result is False  # the stop block is skipped entirely
        assert calls == []  # the gw-serve stop verb NEVER fires
        assert idle_rows(node) == []

    def test_seat_down_idle_past_grace_stops_confirmed_idle(self, node):
        """The vLLM axis's own idle reading is what produced idle_since:
        seat definitively down + idle past grace -> the stop fires with
        confirmed_idle (the matrix cell the misnamed predecessor test
        actually covered)."""
        self._idle_node(node)
        set_seat(node, "down")
        with patch("subprocess.run", return_value=MagicMock(
                returncode=0, stdout="", stderr="")) as run_mock, \
             patch.object(node, "_is_serving", return_value=False):
            result = node._decide_idle_stop()
        assert result is True
        assert run_mock.called
        assert stopped_reason(node) == "confirmed_idle"

    # -- sentinel 0: unbounded blind-withhold ------------------------------

    def test_blind_sentinel_zero_unbounded(self, node, monkeypatch):
        """DOORMAN_PROBE_BLINDNESS_SEC == 0: withhold on blind forever —
        the operator chooses burning fuel over risking the seat."""
        self._idle_node(node)
        set_seat(node, "blind")
        with node.lock:
            node._flashnext_blind_since = time.time() - 100000.0
        monkeypatch.setattr(ds, "DOORMAN_PROBE_BLINDNESS_SEC", 0)
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        assert any(r["event"] == "flashnext_blind_hold" for r in rows)
        assert "stopped" not in [r["event"] for r in rows]


# ---------------------------------------------------------------------------
# Non-regression: the D3 window guard and the D9 close re-anchor
# ---------------------------------------------------------------------------

class TestStopPathNonRegression:
    """The D3 card_held_flashnext path and the D9 one-shot semantics are
    unchanged by the new partition."""

    def _idle_node(self, node, idle_secs=632.0):
        with node.lock:
            node.idle_since = time.time() - idle_secs
            node._cached_serving = True
            node.service_stopped = False
            node._stop_in_flight = False
            node.leases = {}
            node._probe_indeterminate = False
            node._serving_is_big = False
        return node

    def test_d3_window_active_card_held(self, node):
        """D3 (handover case: seat up + day seat DOWN): the window guard
        withholds BEFORE the new partition runs; the idle log carries the
        unchanged card_held_flashnext row."""
        self._idle_node(node)
        with node.lock:
            node._flashnext_window = "active"
            node._flashnext_window_since = time.time() - 300.0
        set_seat(node, "up_registered", served_id="gravitywell-flashnext-27b",
                 registered=True)
        result, calls = run_stop(node)
        assert result is True
        assert calls == []
        rows = idle_rows(node)
        events = [r["event"] for r in rows]
        assert "card_held_flashnext" in events
        assert "stopped" not in events
        # the new partition's rows must NOT also be present — D3 returned
        # first (precedence preserved)
        assert "flashnext_withheld_active" not in events
        assert "flashnext_withheld_up_idle" not in events

    def test_d9_close_reanchor_one_shot(self, node):
        """D9: the close re-anchor fires exactly once on the close
        transition (window_closed_at consumed), re-anchoring idle_since
        so the grace clock starts fresh."""
        self._idle_node(node, idle_secs=5000.0)
        with node.lock:
            node._flashnext_window_closed_at = time.time() - 45.0
            node._flashnext_window = "none"
        set_seat(node, "down")
        before = node.idle_since
        result, calls = run_stop(node)
        # the re-anchor reset idle_since to now, so idle_elapsed < grace:
        # no stop this tick, and the one-shot flag is consumed.
        assert result is True
        assert calls == []
        assert node._flashnext_window_closed_at is None
        assert node.idle_since > before
        rows = idle_rows(node)
        assert "flashnext_window_closed" in [r["event"] for r in rows]
        # second call: the flag is consumed — no re-anchor, no row.
        result2, calls2 = run_stop(node)
        assert calls2 == []
        # one-shot: the second tick wrote NO further flashnext_window_closed
        # row (the flag was consumed on the first) and no stop fired.
        rows2 = idle_rows(node)
        assert sum(1 for r in rows2 if r["event"] == "flashnext_window_closed") == 1
        assert "stopped" not in [r["event"] for r in rows2]


# ---------------------------------------------------------------------------
# D1 bookkeeping: the tick's probe pass (blind clock + D2 stamp rule)
# ---------------------------------------------------------------------------

class TestProbePassBookkeeping:
    """_refresh_serving_cache with the probes mocked: the blind clock is
    armed on the first blind (or cold) read, cleared on a definitive
    read; the D2 stamp rule holds; a definitive down clears the stamp."""

    def _run_tick(self, node, seat_result, activity_result,
                  serving=True):
        with patch.object(node, "_probe_flashnext_seat",
                          return_value=seat_result) as seat_mock, \
             patch.object(node, "_probe_flashnext_activity",
                          return_value=activity_result), \
             patch.object(node, "_is_serving", return_value=serving), \
             patch.object(node, "_is_creative_serving", return_value=False):
            node._refresh_serving_cache()
        assert seat_mock.called
        return seat_mock

    def test_blind_arms_clock_then_definitive_clears(self, node):
        r = ("blind", None, None, "Timeout")
        self._run_tick(node, r, None)
        assert node._flashnext_blind_since is not None
        assert node._flashnext_state == "blind"
        assert node._flashnext_error_class == "Timeout"
        armed = node._flashnext_blind_since
        # a second blind tick does NOT re-arm the clock
        self._run_tick(node, r, None)
        assert node._flashnext_blind_since == armed
        # a definitive read clears the clock
        self._run_tick(node, ("down", None, None, "ConnectionRefusedError"), None)
        assert node._flashnext_blind_since is None

    def test_down_clears_activity_stamp(self, node):
        with node.lock:
            node._flashnext_last_activity_ts = time.time() - 100.0
        self._run_tick(node, ("down", None, None, "ConnectionRefusedError"), None)
        assert node._flashnext_last_activity_ts is None

    def test_stamp_rule_only_on_confirmed_activity(self, node):
        with node.lock:
            node._flashnext_last_activity_ts = None
        # successful zero-activity read: the stamp is NOT set
        self._run_tick(node, ("up_registered", "m", True, None), False)
        assert node._flashnext_last_activity_ts is None
        # failed read: unchanged
        self._run_tick(node, ("up_registered", "m", True, None), None)
        assert node._flashnext_last_activity_ts is None
        # confirmed activity: stamped
        self._run_tick(node, ("up_registered", "m", True, None), True)
        assert node._flashnext_last_activity_ts is not None
        stamped = node._flashnext_last_activity_ts
        # blind reads leave it unchanged
        self._run_tick(node, ("blind", None, None, "Timeout"), None)
        assert node._flashnext_last_activity_ts == stamped

    def test_cold_start_arms_blind_clock(self, node):
        """COLD START: the first read (state None before any probe) — a
        blind first read arms the clock on the very first tick."""
        assert node._flashnext_state is None
        self._run_tick(node, ("blind", None, None, "Timeout"), None)
        assert node._flashnext_blind_since is not None

    def test_activity_probe_runs_unconditionally(self, node):
        """The D2 probe is unconditional in _refresh_serving_cache (it
        does not join the flag-gated _probe_slot_activity pool)."""
        with patch.object(node, "_probe_flashnext_seat",
                          return_value=("down", None, None, "ConnectionRefusedError")), \
             patch.object(node, "_probe_flashnext_activity",
                          return_value=None) as act_mock, \
             patch.object(node, "_is_serving", return_value=True), \
             patch.object(node, "_is_creative_serving", return_value=False):
            node._refresh_serving_cache()
        assert act_mock.called


# ---------------------------------------------------------------------------
# D2: the legibility probe itself
# ---------------------------------------------------------------------------

class TestFlashnextActivityProbe:
    """_probe_flashnext_activity: tri-state parsing of the SGLang
    /metrics gauges. A 404 / non-200 / unparseable gauge is "unknown"
    (None) — never an idle reading, never a stop authorization."""

    def _resp(self, status=200, text=""):
        m = MagicMock(spec=requests.Response)
        m.status_code = status
        m.text = text
        return m

    def test_confirmed_activity(self):
        node = _NodeState(gw_url="http://mock.internal/")
        body = (
            "# HELP sglang:num_running_requests ...\n"
            "sglang:num_running_requests{engine=\"0\"} 3.0\n"
            "sglang:num_queue_requests{engine=\"0\"} 0.0\n"
        )
        with patch("requests.get", return_value=self._resp(200, body)):
            assert node._probe_flashnext_activity() is True

    def test_confirmed_idle(self):
        node = _NodeState(gw_url="http://mock.internal/")
        body = (
            "sglang:num_running_requests{engine=\"0\"} 0.0\n"
            "sglang:num_queue_requests{engine=\"0\"} 0.0\n"
        )
        with patch("requests.get", return_value=self._resp(200, body)):
            assert node._probe_flashnext_activity() is False

    @pytest.mark.parametrize("status", [404, 500])
    def test_non_200_is_unknown(self, status):
        node = _NodeState(gw_url="http://mock.internal/")
        with patch("requests.get", return_value=self._resp(status, "x")):
            assert node._probe_flashnext_activity() is None

    def test_missing_gauges_is_unknown(self):
        node = _NodeState(gw_url="http://mock.internal/")
        body = "some_other_metric 1.0\n"
        with patch("requests.get", return_value=self._resp(200, body)):
            assert node._probe_flashnext_activity() is None

    def test_unparsable_value_is_unknown(self):
        node = _NodeState(gw_url="http://mock.internal/")
        body = "sglang:num_running_requests{engine=\"0\"} notanumber\n"
        with patch("requests.get", return_value=self._resp(200, body)):
            assert node._probe_flashnext_activity() is None

    def test_connection_error_is_unknown(self):
        node = _NodeState(gw_url="http://mock.internal/")
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            assert node._probe_flashnext_activity() is None

    def test_gauge_names_are_the_sglang_pair(self):
        assert set(_SGLANG_ACTIVITY_METRICS) == {
            "sglang:num_running_requests", "sglang:num_queue_requests",
        }


# ---------------------------------------------------------------------------
# D3: /status rendering
# ---------------------------------------------------------------------------

class TestStatusFlashnextBlock:
    """last_activity_ts + eject_state nest INSIDE the existing
    "flashnext" object; a withhold never sets last_error."""

    def _status(self, node):
        with node.lock:
            return node.status_snapshot()

    def test_withheld_active(self, node):
        with node.lock:
            node._flashnext_state = "up_registered"
            node._flashnext_last_activity_ts = time.time() - 100.0
        snap = self._status(node)
        assert snap["flashnext"]["eject_state"] == "withheld-active"
        assert snap["flashnext"]["last_activity_ts"] is not None
        assert snap["last_error"] is None

    def test_withheld_up_idle(self, node):
        with node.lock:
            node._flashnext_state = "up_unverified"
            node._flashnext_last_activity_ts = None
        snap = self._status(node)
        assert snap["flashnext"]["eject_state"] == "withheld-up-idle"
        assert snap["flashnext"]["last_activity_ts"] is None

    def test_withheld_blind_cold(self, node):
        with node.lock:
            node._flashnext_state = "blind"
        snap = self._status(node)
        assert snap["flashnext"]["eject_state"] == "withheld-blind"
        # cold (None) is the same substate
        with node.lock:
            node._flashnext_state = None
        assert self._status(node)["flashnext"]["eject_state"] == "withheld-blind"

    def test_idle_ok(self, node):
        with node.lock:
            node._flashnext_state = "down"
        snap = self._status(node)
        assert snap["flashnext"]["eject_state"] == "idle-ok"
        with node.lock:
            node._flashnext_state = "up_foreign"
        assert self._status(node)["flashnext"]["eject_state"] == "idle-ok"

    def test_stale_stamp_is_up_idle_not_active(self, node):
        """A stamp older than GW_STOP_GRACE_SEC renders
        withheld-up-idle (the substate is legibility-only; the stop
        decision withholds either way)."""
        with node.lock:
            node._flashnext_state = "up_registered"
            node._flashnext_last_activity_ts = time.time() - (ds.GW_STOP_GRACE_SEC + 60)
        snap = self._status(node)
        assert snap["flashnext"]["eject_state"] == "withheld-up-idle"
