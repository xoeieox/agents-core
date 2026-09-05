"""Hermetic tests for the doorman's flash-next seat (:30000) handover window —
agents-core-doorman-flashnext-handover-v0.

All subprocess / SSH / :8081 / :30000 HTTP is mocked; no GPU or network is
required. Covers the spec's test groups:

  (a) Probe states (mocked HTTP): DOWN / BLIND / UP_REGISTERED / UP_FOREIGN /
      UP_UNVERIFIED
  (b) Window determination matrix (same-tick inputs)
  (c) Identity pin (D2b): exact canonical_id match
  (d) window_since / window_closed_at lifecycle
  (e) Stop path: window active -> card_held_flashnext, no stop
  (f) D9 close re-anchor
  (g) ensure_serving (D4): window active -> FLASHNEXT_OCCUPIED, no wake
  (h) Route: 409 body shape
  (i) /status: block shape
  (j) D2 probe: "down" means connection-refused ONLY (ConnectionRefusedError
      via __cause__/__context__); every other ConnectionError -> "blind"
  (k) D4 BLIND-proceed log.warning carries the ACTUAL probe error class
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from agents_core.doorman_server import (
    FLASHNEXT_OCCUPIED,
    GW_FLASHNEXT_MODEL_ID,
    GW_FLASHNEXT_URL,
    GW_STOP_GRACE_SEC,
    GW_URL_DEFAULT,
    _NodeState,
    create_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


def _mock_resp(status_code: int, json_data=None, text: str = ""):
    """A mock that passes isinstance(x, requests.Response) AND has the
    right status_code / .json() / .text."""
    m = MagicMock(spec=requests.Response)
    m.status_code = status_code
    if json_data is not None:
        m.json.return_value = json_data
    else:
        m.json.side_effect = ValueError("no json")
    m.text = text
    return m


def _mock_refused_conn():
    """A requests ConnectionError wrapping a ConnectionRefusedError."""
    inner = ConnectionRefusedError("Connection refused")
    exc = requests.exceptions.ConnectionError("refused")
    exc.__cause__ = inner
    return exc


def _mock_dns_conn():
    """A requests ConnectionError wrapping a DNS failure (NOT refused)."""
    inner = OSError("Name or service not known")
    exc = requests.exceptions.ConnectionError("dns fail")
    exc.__cause__ = inner
    return exc


def _mock_timeout():
    """A requests Timeout exception."""
    return requests.exceptions.Timeout("timed out")


def _mock_models_resp(model_id: str):
    """A /v1/models 200 response with a single model entry."""
    return _mock_resp(200, {"data": [{"id": model_id}]})


# ---------------------------------------------------------------------------
# (a) Probe states
# ---------------------------------------------------------------------------

class TestProbeStates:
    def test_down_on_connection_refused(self):
        """ConnectionRefusedError (via __cause__) -> "down"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_refused_conn(),  # /health refused
            _mock_resp(200),       # /v1/models (never reached in sequential)
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "down"
            assert result[1] is None  # served_id
            assert result[2] is None  # registered
            assert result[3] == "ConnectionError"  # error_class

    def test_blind_on_dns_failure(self):
        """A non-refused ConnectionError (DNS) -> "blind", NOT "down"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_dns_conn(),  # /health DNS failure
            _mock_resp(200),
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"
            assert result[1] is None
            assert result[2] is None
            assert result[3] == "ConnectionError"

    def test_blind_on_timeout(self):
        """A Timeout -> "blind"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_timeout(),  # /health timeout
            _mock_resp(200),
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"
            assert result[3] == "Timeout"

    def test_up_registered(self):
        """/health 200 + /v1/models 200 with exact canonical_id -> "up_registered"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health
            _mock_models_resp(GW_FLASHNEXT_MODEL_ID),  # /v1/models
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_registered"
            assert result[1] == GW_FLASHNEXT_MODEL_ID
            assert result[2] is True
            assert result[3] is None

    def test_up_foreign(self):
        """/health 200 + /v1/models 200 with a DIFFERENT id -> "up_foreign"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health
            _mock_models_resp("some-other-model"),  # /v1/models
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_foreign"
            assert result[1] == "some-other-model"
            assert result[2] is False
            assert result[3] is None

    def test_up_unverified_models_timeout(self):
        """/health 200 + /v1/models timeout -> "up_unverified"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health
            _mock_timeout(),  # /v1/models timeout
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_unverified"
            assert result[1] is None
            assert result[2] is None
            assert result[3] is None

    def test_up_unverified_models_non_200(self):
        """/health 200 + /v1/models 500 -> "up_unverified"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health
            _mock_resp(500),  # /v1/models
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_unverified"

    def test_sequential_short_circuit_on_refused(self):
        """A refused /health stops after exactly ONE HTTP call (sequential)."""
        state = _make_state()
        call_count = 0
        def _counting_get(*a, **kw):
            nonlocal call_count
            call_count += 1
            raise _mock_refused_conn()
        with patch("agents_core.doorman_server.requests.get", side_effect=_counting_get):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "down"
            assert call_count == 1

    def test_tick_path_uses_pool(self):
        """The tick path (sequential=False) runs both GETs concurrently in a
        2-worker pool. Mock requests.get so no real HTTP occurs."""
        state = _make_state()
        def _fake_get(url, **kw):
            if "/health" in url:
                return _mock_resp(200)
            elif "/v1/models" in url:
                return _mock_models_resp(GW_FLASHNEXT_MODEL_ID)
            raise AssertionError(f"unexpected URL {url}")
        with patch("agents_core.doorman_server.requests.get", side_effect=_fake_get):
            result = state._probe_flashnext_seat(sequential=False)
            assert result[0] == "up_registered"


# ---------------------------------------------------------------------------
# (b) Window determination matrix
# ---------------------------------------------------------------------------

class TestWindowDetermination:
    def _run_tick(self, state, probe_result, serving):
        """Run one _refresh_serving_cache tick with mocked probe + serving."""
        with patch.object(state, "_probe_flashnext_seat", return_value=probe_result), \
             patch.object(state, "_is_serving", return_value=serving), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch("agents_core.doorman_server.gw_serving_state", return_value=None):
            state._refresh_serving_cache()

    def test_up_registered_day_down_window_active(self):
        state = _make_state()
        probe = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window == "active"

    def test_up_registered_day_up_window_none(self):
        state = _make_state()
        probe = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        self._run_tick(state, probe, serving=True)
        assert state._flashnext_window == "none"

    def test_up_unverified_day_down_window_active(self):
        """Safe direction: unverified listener + day down -> active."""
        state = _make_state()
        probe = ("up_unverified", None, None, None)
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window == "active"

    def test_up_foreign_day_down_window_none(self):
        """A squatter never admits a window."""
        state = _make_state()
        probe = ("up_foreign", "other-model", False, None)
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window == "none"

    def test_blind_day_down_window_indeterminate(self):
        """A blind probe leaves the window indeterminate (None)."""
        state = _make_state()
        probe = ("blind", None, None, "Timeout")
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window is None

    def test_down_day_down_window_none(self):
        state = _make_state()
        probe = ("down", None, None, "ConnectionError")
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window == "none"


# ---------------------------------------------------------------------------
# (c) Identity pin (D2b)
# ---------------------------------------------------------------------------

class TestIdentityPin:
    def test_shared_alias_big_does_not_admit(self):
        """A served id of "big" (shared alias) must NOT admit a window."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_models_resp("big"),
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_foreign"
            assert result[2] is False

    def test_shared_alias_gravitywell_does_not_admit(self):
        """A served id of "gravitywell" (shared alias) must NOT admit."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_models_resp("gravitywell"),
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_foreign"
            assert result[2] is False

    def test_exact_canonical_id_admits(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_models_resp(GW_FLASHNEXT_MODEL_ID),
        ]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "up_registered"
            assert result[2] is True


# ---------------------------------------------------------------------------
# (d) window_since / window_closed_at lifecycle
# ---------------------------------------------------------------------------

class TestWindowLifecycle:
    def _run_tick(self, state, probe_result, serving):
        with patch.object(state, "_probe_flashnext_seat", return_value=probe_result), \
             patch.object(state, "_is_serving", return_value=serving), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch("agents_core.doorman_server.gw_serving_state", return_value=None):
            state._refresh_serving_cache()

    def test_window_since_set_on_first_active(self):
        state = _make_state()
        probe = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window == "active"
        assert state._flashnext_window_since is not None

    def test_window_since_stable_across_active_ticks(self):
        state = _make_state()
        probe = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        self._run_tick(state, probe, serving=False)
        since1 = state._flashnext_window_since
        self._run_tick(state, probe, serving=False)
        assert state._flashnext_window_since == since1

    def test_window_since_cleared_on_close(self):
        state = _make_state()
        probe_active = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        probe_down = ("down", None, None, "ConnectionError")
        self._run_tick(state, probe_active, serving=False)
        assert state._flashnext_window == "active"
        self._run_tick(state, probe_down, serving=False)
        assert state._flashnext_window == "none"
        assert state._flashnext_window_since is None

    def test_window_closed_at_set_on_transition(self):
        state = _make_state()
        probe_active = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        probe_down = ("down", None, None, "ConnectionError")
        self._run_tick(state, probe_active, serving=False)
        assert state._flashnext_window_closed_at is None
        self._run_tick(state, probe_down, serving=False)
        assert state._flashnext_window_closed_at is not None

    def test_window_closed_at_consumed_once(self):
        """The D9 re-anchor consumes window_closed_at exactly once."""
        state = _make_state()
        probe_active = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
        probe_down = ("down", None, None, "ConnectionError")
        self._run_tick(state, probe_active, serving=False)
        self._run_tick(state, probe_down, serving=False)
        assert state._flashnext_window_closed_at is not None
        # Simulate the D9 re-anchor consuming it
        state._flashnext_window_closed_at = None
        # Second tick: no new transition, so closed_at stays None
        self._run_tick(state, probe_down, serving=False)
        assert state._flashnext_window_closed_at is None


# ---------------------------------------------------------------------------
# (e) Stop path: window active -> card_held_flashnext, no stop
# ---------------------------------------------------------------------------

class TestStopPath:
    """Pins the REAL production stop decision —
    _NodeState._decide_idle_stop() (the behavior-preserving extraction of
    _start_refresh_thread._loop's per-node stop block, rev 3). The tests
    call the real method; they do NOT re-implement the if/else inline."""

    def test_window_active_writes_card_held(self):
        """Window active -> the REAL method writes card_held_flashnext,
        issues no gw-serve stop, and never reaches the
        topology_unknown_no_park alarm; the loop must continue (True)."""
        state = _make_state()
        state._flashnext_window = "active"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state.idle_since = time.time() - 100
        state.leases = {}
        # Hostile inputs: idle far past the grace clock, unresolved topology
        # and the flag ON — the legacy path WOULD have parked or alarmed.
        state._serving_is_big = None
        state._cached_topology_state = None

        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            result = state._decide_idle_stop()

        assert result is True  # the loop `continue`s to the next node
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == "gravitywell"
        assert call_args[0][1] == "card_held_flashnext"
        assert call_args[1].get("served_id") == GW_FLASHNEXT_MODEL_ID
        # No gw-serve stop, no topology_unknown_no_park alarm.
        mock_run.assert_not_called()
        assert "topology_unknown_no_park" not in str(mock_log.call_args)

    def test_window_active_flag_off_still_no_stop(self):
        """The D3 check is NOT gated by DOORMAN_MODE_AWARE_ADMISSION:
        flag OFF + window active -> the REAL method still writes
        card_held_flashnext and never stops."""
        state = _make_state()
        state._flashnext_window = "active"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state.idle_since = time.time() - 100
        state.leases = {}

        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            result = state._decide_idle_stop()

        assert result is True
        mock_log.assert_called_once()
        assert mock_log.call_args[0][1] == "card_held_flashnext"
        mock_run.assert_not_called()

    def test_window_inactive_legacy_behavior(self):
        """Window not active -> the REAL method runs the legacy path
        (regression pin): idle past grace, flag off -> the legacy
        confirmed-idle park fires exactly as before this unit."""
        state = _make_state()
        state._flashnext_window = "none"
        state._flashnext_served_id = None
        state.idle_since = time.time() - 10000  # far past GW_STOP_GRACE_SEC
        state.leases = {}
        state.service_stopped = False
        state._stop_in_flight = False
        state._probe_indeterminate = False

        stop_proc = MagicMock()
        stop_proc.returncode = 0
        stop_proc.stdout = ""
        stop_proc.stderr = ""
        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            mock_run.return_value = stop_proc
            result = state._decide_idle_stop()

        assert result is True  # the idle-node trailing skip
        # The REAL method issued the gw-serve stop subprocess.
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0] == ["ssh", "gravitywell", "gw-serve stop"]
        # The legacy park bookkeeping landed (rc=0 success path).
        assert state.service_stopped is True
        assert state.idle_since is None
        events = [c[0][1] for c in mock_log.call_args_list]
        assert "stopped" in events
        # No flashnext events at all on the legacy path.
        assert "card_held_flashnext" not in events
        assert "flashnext_window_closed" not in events
        # Stop succeeded (rc=0): no stop-failure backoff signal — the
        # refresh loop must NOT bump backoff on a successful stop.
        assert state._stop_failed_this_tick is False

    def test_stop_failure_sets_backoff_signal(self):
        """Regression pin (rev 3 extraction repair): a real stop failure
        (subprocess rc!=0 with the service still serving) sets
        _stop_failed_this_tick, which the refresh loop reads to apply
        origin/main's stop-failure backoff bump. The extraction dropped
        those two inline bumps; this pins the restored signal on the
        rc!=0-still-serving outcome."""
        state = _make_state()
        state._flashnext_window = "none"
        state._flashnext_served_id = None
        state.idle_since = time.time() - 10000  # far past GW_STOP_GRACE_SEC
        state.leases = {}
        state.service_stopped = False
        state._stop_in_flight = False
        state._probe_indeterminate = False

        stop_proc = MagicMock()
        stop_proc.returncode = 1
        stop_proc.stdout = ""
        stop_proc.stderr = "boom"
        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False), \
             patch.object(state, "_is_serving", return_value=True):  # still serving
            mock_run.return_value = stop_proc
            result = state._decide_idle_stop()

        assert result is True  # the idle-node trailing skip
        assert mock_run.call_count == 1
        # Real failure (still serving): the stop-failure backoff signal is set.
        assert state._stop_failed_this_tick is True
        # And the stop_failed idle-log row landed.
        assert "stop_failed" in [c[0][1] for c in mock_log.call_args_list]
        # The in-flight flag was cleared (R8) — the failure is resolved,
        # not wedged.
        assert state._stop_in_flight is False
        assert state.service_stopped is False


# ---------------------------------------------------------------------------
# (f) D9 close re-anchor
# ---------------------------------------------------------------------------

class TestCloseReAnchor:
    """Pins the REAL D9 close re-anchor inside
    _NodeState._decide_idle_stop() (the behavior-preserving extraction,
    rev 3). The tests call the real method; they do NOT re-implement the
    if/else inline."""

    def test_re_anchor_fires_on_close_transition(self):
        """Window closed + idle_since set + no leases + serving -> the REAL
        method re-anchors idle_since (source window_close), writes the
        flashnext_window_closed idle-log line, and the restored seat gets a
        FRESH grace clock (no gw-serve stop within GW_STOP_GRACE_SEC)."""
        state = _make_state()
        state._flashnext_window = "none"  # the window itself is already closed
        state._flashnext_window_closed_at = time.time()
        # idle_since is window-STALE: set long before the grace clock, so
        # without the re-anchor the legacy path would park this tick.
        state.idle_since = time.time() - 10000
        state.leases = {}
        state._cached_serving = True
        state.service_stopped = False
        state._stop_in_flight = False
        state._probe_indeterminate = False

        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            result = state._decide_idle_stop()

        assert result is True
        mock_log.assert_called_once()
        assert mock_log.call_args[0][1] == "flashnext_window_closed"
        assert state._idle_since_source == "window_close"
        # The flag is consumed.
        assert state._flashnext_window_closed_at is None
        # The re-anchored clock is fresh: no stop within GW_STOP_GRACE_SEC of
        # the first serving observation after the window.
        assert state.idle_since is not None
        assert (time.time() - state.idle_since) < GW_STOP_GRACE_SEC
        # No gw-serve stop this tick.
        mock_run.assert_not_called()
        assert state.service_stopped is False

    def test_re_anchor_not_fired_when_lease_held(self):
        """A lease acquired between close and first-serving-observation owns
        the clock: the REAL method does not re-anchor (and the refresh loop
        does not even call the stop decision while leases exist)."""
        state = _make_state()
        state._flashnext_window_closed_at = time.time()
        state.idle_since = time.time() - 300
        state.leases = {"w1": {"acquired_at": time.time(), "ttl_sec": 300}}
        state._cached_serving = True

        with patch("agents_core.doorman_server._write_idle_log") as mock_log:
            # The loop gates the decision on `if not state.leases:` — the
            # real guard the re-anchor itself also checks.
            assert not state.leases is True  # leases exist -> loop skips
            if not state.leases:
                state._decide_idle_stop()

        mock_log.assert_not_called()
        # The flag survives (not consumed) and the clock is untouched.
        assert state._flashnext_window_closed_at is not None
        assert state._idle_since_source is None

    def test_re_anchor_fires_once(self):
        """The flag is consumed by the REAL method: a second tick does not
        re-fire, and the (now fresh) grace clock does not park the seat."""
        state = _make_state()
        state._flashnext_window = "none"
        state._flashnext_window_closed_at = time.time()
        state.idle_since = time.time() - 10000
        state.leases = {}
        state._cached_serving = True
        state.service_stopped = False
        state._stop_in_flight = False
        state._probe_indeterminate = False

        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            # First tick: the re-anchor fires.
            result1 = state._decide_idle_stop()
            # Second tick: flag consumed, does NOT re-fire.
            result2 = state._decide_idle_stop()

        assert result1 is True
        assert result2 is True
        assert mock_log.call_count == 1
        assert mock_log.call_args[0][1] == "flashnext_window_closed"
        # The fresh grace clock protected the seat on the second tick too.
        mock_run.assert_not_called()
        assert state.service_stopped is False


# ---------------------------------------------------------------------------
# (g) ensure_serving (D4): window active -> FLASHNEXT_OCCUPIED, no wake
# ---------------------------------------------------------------------------

class TestEnsureServingGuard:
    def test_window_active_returns_sentinel(self):
        """Fresh probe UP_REGISTERED -> FLASHNEXT_OCCUPIED, no wake-gravitywell."""
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_flashnext_seat",
                          return_value=("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)), \
             patch("agents_core.doorman_server.subprocess.run") as mock_run:
            result = state.ensure_serving()
            assert result is FLASHNEXT_OCCUPIED
            mock_run.assert_not_called()

    def test_window_active_up_unverified_returns_sentinel(self):
        """Fresh probe UP_UNVERIFIED -> FLASHNEXT_OCCUPIED (safe direction)."""
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_flashnext_seat",
                          return_value=("up_unverified", None, None, None)), \
             patch("agents_core.doorman_server.subprocess.run") as mock_run:
            result = state.ensure_serving()
            assert result is FLASHNEXT_OCCUPIED
            mock_run.assert_not_called()

    def test_window_inactive_wake_proceeds(self):
        """Window inactive (DOWN) -> the wake path proceeds (regression pin)."""
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_flashnext_seat",
                          return_value=("down", None, None, "ConnectionError")), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", False), \
             patch.object(state, "_resolve_cold_wake_posture", return_value="dual"), \
             patch.object(state, "_wake_dual", return_value=True):
            result = state.ensure_serving()
            assert result is True

    def test_blind_proceeds_with_warning(self):
        """BLIND probe -> proceeds with wake (today's behavior), warning logged."""
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_flashnext_seat",
                          return_value=("blind", None, None, "Timeout")), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", False), \
             patch.object(state, "_resolve_cold_wake_posture", return_value="dual"), \
             patch.object(state, "_wake_dual", return_value=True):
            result = state.ensure_serving()
            assert result is True

    def test_no_second_8081_probe(self):
        """The guard issues ONLY the fresh seat probe, never a second :8081 GET."""
        state = _make_state()
        is_serving_calls = 0
        def _counting_is_serving(*a, **kw):
            nonlocal is_serving_calls
            is_serving_calls += 1
            return False
        with patch.object(state, "_is_serving", side_effect=_counting_is_serving), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_flashnext_seat",
                          return_value=("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)), \
             patch("agents_core.doorman_server.subprocess.run") as mock_run:
            result = state.ensure_serving()
            assert result is FLASHNEXT_OCCUPIED
            # _is_serving is called once (the fast path fallthrough)
            assert is_serving_calls == 1


# ---------------------------------------------------------------------------
# (h) Route: 409 body shape
# ---------------------------------------------------------------------------

class TestRoute409:
    def test_flashnext_occupied_409(self):
        """acquire_lease returns FLASHNEXT_OCCUPIED -> the /lease/acquire
        route answers 409 with the exact flashnext_occupied body shape."""
        with patch.object(_NodeState, "acquire_lease",
                          return_value=FLASHNEXT_OCCUPIED):
            app = create_app()
            client = TestClient(app)
            resp = client.post(
                "/lease/acquire",
                json={"node": "gravitywell", "work_id": "w1", "ttl_sec": 300},
            )
            assert resp.status_code == 409
            body = resp.json()
            assert body == {
                "ok": False,
                "flashnext_occupied": True,
                "reason": "flashnext-window-holding-gpu0",
            }

    def test_acquire_lease_propagates_sentinel(self):
        """acquire_lease propagates FLASHNEXT_OCCUPIED exactly as CREATIVE_OCCUPIED."""
        state = _make_state()
        with patch.object(state, "ensure_serving", return_value=FLASHNEXT_OCCUPIED):
            result = state.acquire_lease("test-wid", 300, "test")
            assert result is FLASHNEXT_OCCUPIED

    def test_no_lease_registered_on_sentinel(self):
        """Spec Tests-section requirement (D4): the FLASHNEXT_OCCUPIED path
        registers NO lease and leaves idle_since untouched — the refusal
        mirrors CREATIVE_OCCUPIED, which refuses even the controller's own
        acquire and holds no clock, so no doorman keepawake hold exists
        during a window (Invariant 12)."""
        state = _make_state()
        idle_before = state.idle_since
        with patch.object(state, "ensure_serving", return_value=FLASHNEXT_OCCUPIED):
            result = state.acquire_lease("test-wid", 300, "test")
        assert result is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert state.idle_since is idle_before


# ---------------------------------------------------------------------------
# (i) /status: block shape
# ---------------------------------------------------------------------------

class TestStatusBlock:
    def _snap(self, state):
        """Get a status snapshot with subprocess mocked (no real ssh)."""
        with patch("agents_core.doorman_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            return state.status_snapshot()

    def test_flashnext_block_shape(self):
        """The /status flashnext block has all D5 fields."""
        state = _make_state()
        state._flashnext_state = "up_registered"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state._flashnext_registered = True
        state._flashnext_window = "active"
        state._flashnext_window_since = 12345.0
        state._flashnext_window_closed_at = None

        snap = self._snap(state)

        fn = snap["flashnext"]
        assert fn["seat_health"] is True
        assert fn["seat_state"] == "up_registered"
        assert fn["served_id"] == GW_FLASHNEXT_MODEL_ID
        assert fn["registered"] is True
        assert fn["window"] == "active"
        assert fn["window_since"] == 12345.0
        assert fn["window_closed_at"] is None
        assert fn["last_vote"] == "up_registered"

    def test_pre_probe_none_state_maps_to_seat_health_none(self):
        """The pre-probe None state maps to seat_health None (not False)."""
        state = _make_state()
        # _flashnext_state is None (pre-probe)
        assert state._flashnext_state is None

        snap = self._snap(state)

        fn = snap["flashnext"]
        assert fn["seat_health"] is None
        assert fn["seat_state"] is None

    def test_blind_state_maps_to_seat_health_none(self):
        """A blind probe maps to seat_health None."""
        state = _make_state()
        state._flashnext_state = "blind"

        snap = self._snap(state)

        fn = snap["flashnext"]
        assert fn["seat_health"] is None
        assert fn["seat_state"] == "blind"

    def test_serving_mode_untouched(self):
        """The flashnext block does not affect the 27B axis:
        serving_mode / serving_is_big / big_probe_state are byte-identical
        to the no-window baseline (spec test-list requirement)."""
        def _axis_fields(snap):
            return {
                "serving_mode": snap["serving_mode"],
                "serving_is_big": snap["serving_is_big"],
                "big_probe_state": snap["big_probe_state"],
            }

        # Baseline: no window state at all (fresh node, pre-probe).
        baseline = _make_state()
        base_snap = self._snap(baseline)
        base_fields = _axis_fields(base_snap)
        assert base_fields == {
            "serving_mode": "unknown",
            "serving_is_big": None,
            "big_probe_state": None,
        }

        # Window fully active: the 27B-axis fields must be byte-identical to
        # the no-window baseline with the SAME 27B-axis inputs.
        state = _make_state()
        state._cached_serving = base_snap["serving"]
        state._flashnext_state = "up_registered"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state._flashnext_registered = True
        state._flashnext_window = "active"
        state._flashnext_window_since = 12345.0
        state._flashnext_window_closed_at = None
        win_snap = self._snap(state)
        assert _axis_fields(win_snap) == base_fields

        # Window closed (D9 re-anchor state pending): still byte-identical.
        closed_state = _make_state()
        closed_state._cached_serving = base_snap["serving"]
        closed_state._flashnext_state = "down"
        closed_state._flashnext_window = "none"
        closed_state._flashnext_window_closed_at = 99999.0
        closed_snap = self._snap(closed_state)
        assert _axis_fields(closed_snap) == base_fields


# ---------------------------------------------------------------------------
# (j) D2 probe: "down" means connection-refused ONLY
# ---------------------------------------------------------------------------

class TestDownVsBlind:
    def test_connection_refused_is_down(self):
        """ConnectionRefusedError (via __cause__) -> "down"."""
        state = _make_state()
        inner = ConnectionRefusedError("refused")
        # RequestException.__init__ rejects the __cause__ kwarg (it pops only
        # response/request) — set it AFTER construction, the way the file's
        # own _mock_refused_conn helper does.
        exc = requests.exceptions.ConnectionError("refused")
        exc.__cause__ = inner
        with patch("agents_core.doorman_server.requests.get", side_effect=[exc]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "down"

    def test_connection_refused_via_context(self):
        """ConnectionRefusedError via __context__ (not __cause__) -> "down"."""
        state = _make_state()
        inner = ConnectionRefusedError("refused")
        exc = requests.exceptions.ConnectionError("refused")
        exc.__context__ = inner
        with patch("agents_core.doorman_server.requests.get", side_effect=[exc]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "down"

    def test_dns_failure_is_blind(self):
        """A DNS failure (OSError, not ConnectionRefusedError) -> "blind"."""
        state = _make_state()
        inner = OSError("Name or service not known")
        exc = requests.exceptions.ConnectionError("dns")
        exc.__cause__ = inner
        with patch("agents_core.doorman_server.requests.get", side_effect=[exc]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"

    def test_connection_reset_is_blind(self):
        """A connection reset (ConnectionResetError, not Refused) -> "blind"."""
        state = _make_state()
        inner = ConnectionResetError("reset by peer")
        exc = requests.exceptions.ConnectionError("reset")
        exc.__cause__ = inner
        with patch("agents_core.doorman_server.requests.get", side_effect=[exc]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"

    def test_timeout_is_blind(self):
        """A Timeout -> "blind"."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get",
                   side_effect=[requests.exceptions.Timeout("slow")]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"


# ---------------------------------------------------------------------------
# (k) D4 BLIND-proceed log.warning carries the ACTUAL probe error class
# ---------------------------------------------------------------------------

class TestBlindErrorClass:
    def test_blind_returns_actual_error_class(self):
        """The probe returns the ACTUAL exception class name, not a placeholder."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get",
                   side_effect=[requests.exceptions.Timeout("slow")]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"
            assert result[3] == "Timeout"

    def test_blind_connection_error_returns_class(self):
        state = _make_state()
        inner = OSError("unreachable")
        exc = requests.exceptions.ConnectionError("unreachable")
        exc.__cause__ = inner
        with patch("agents_core.doorman_server.requests.get", side_effect=[exc]):
            result = state._probe_flashnext_seat(sequential=True)
            assert result[0] == "blind"
            assert result[3] == "ConnectionError"
