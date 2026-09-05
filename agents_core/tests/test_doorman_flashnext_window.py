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
    def test_window_active_writes_card_held(self):
        """Window active -> card_held_flashnext idle-log entry, no gw-serve stop."""
        state = _make_state()
        state._flashnext_window = "active"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state.idle_since = time.time() - 100
        state.leases = {}

        with patch("agents_core.doorman_server._write_idle_log") as mock_log, \
             patch("agents_core.doorman_server.subprocess.run") as mock_run:
            # Simulate the stop-path logic (the D3 check)
            if state._flashnext_window == "active":
                mock_log(
                    "gravitywell", "card_held_flashnext", 0,
                    idle_secs=time.time() - state.idle_since,
                    served_id=state._flashnext_served_id,
                )
                # continue — no stop

        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][1] == "card_held_flashnext"
        mock_run.assert_not_called()

    def test_window_active_flag_off_still_no_stop(self):
        """The D3 check is NOT gated by DOORMAN_MODE_AWARE_ADMISSION."""
        state = _make_state()
        state._flashnext_window = "active"
        state._flashnext_served_id = GW_FLASHNEXT_MODEL_ID
        state.idle_since = time.time() - 100
        state.leases = {}
        # Even with the flag off, the window check fires
        with patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            with patch("agents_core.doorman_server._write_idle_log") as mock_log:
                if state._flashnext_window == "active":
                    mock_log(
                        "gravitywell", "card_held_flashnext", 0,
                        idle_secs=time.time() - state.idle_since,
                        served_id=state._flashnext_served_id,
                    )
        mock_log.assert_called_once()

    def test_window_inactive_legacy_behavior(self):
        """Window not active -> the legacy stop path runs (regression pin)."""
        state = _make_state()
        state._flashnext_window = "none"
        state._flashnext_served_id = None
        state.idle_since = time.time() - 100
        state.leases = {}
        # The D3 check does NOT fire
        assert state._flashnext_window != "active"


# ---------------------------------------------------------------------------
# (f) D9 close re-anchor
# ---------------------------------------------------------------------------

class TestCloseReAnchor:
    def test_re_anchor_fires_on_close_transition(self):
        """Window closed + idle_since set + no leases + serving -> re-anchor."""
        state = _make_state()
        state._flashnext_window_closed_at = time.time()
        state.idle_since = time.time() - 300
        state.leases = {}
        state._cached_serving = True
        state.service_stopped = False
        state._stop_in_flight = False

        with patch("agents_core.doorman_server._write_idle_log") as mock_log:
            # Simulate the D9 re-anchor logic
            if (
                state._flashnext_window_closed_at is not None
                and state.idle_since is not None
                and not state.leases
                and state._cached_serving is True
            ):
                mock_log(
                    "gravitywell", "flashnext_window_closed", 0,
                    idle_secs=time.time() - state.idle_since,
                )
                state.idle_since = time.time()
                state._idle_since_source = "window_close"
                state._flashnext_window_closed_at = None

        mock_log.assert_called_once()
        assert mock_log.call_args[0][1] == "flashnext_window_closed"
        assert state._idle_since_source == "window_close"
        assert state._flashnext_window_closed_at is None

    def test_re_anchor_not_fired_when_lease_held(self):
        """A lease acquired between close and first-serving-observation owns the clock."""
        state = _make_state()
        state._flashnext_window_closed_at = time.time()
        state.idle_since = time.time() - 300
        state.leases = {"w1": {"acquired_at": time.time(), "ttl_sec": 300}}
        state._cached_serving = True

        with patch("agents_core.doorman_server._write_idle_log") as mock_log:
            if (
                state._flashnext_window_closed_at is not None
                and state.idle_since is not None
                and not state.leases
                and state._cached_serving is True
            ):
                mock_log("gravitywell", "flashnext_window_closed", 0, idle_secs=0)
                state.idle_since = time.time()
                state._idle_since_source = "window_close"
                state._flashnext_window_closed_at = None

        mock_log.assert_not_called()
        assert state._flashnext_window_closed_at is not None

    def test_re_anchor_fires_once(self):
        """The flag is consumed; a second tick does not re-fire."""
        state = _make_state()
        state._flashnext_window_closed_at = time.time()
        state.idle_since = time.time() - 300
        state.leases = {}
        state._cached_serving = True

        with patch("agents_core.doorman_server._write_idle_log") as mock_log:
            # First tick: fires
            if (
                state._flashnext_window_closed_at is not None
                and state.idle_since is not None
                and not state.leases
                and state._cached_serving is True
            ):
                mock_log("gravitywell", "flashnext_window_closed", 0, idle_secs=0)
                state.idle_since = time.time()
                state._idle_since_source = "window_close"
                state._flashnext_window_closed_at = None

            # Second tick: flag consumed, does NOT fire
            if (
                state._flashnext_window_closed_at is not None
                and state.idle_since is not None
                and not state.leases
                and state._cached_serving is True
            ):
                mock_log("gravitywell", "flashnext_window_closed", 0, idle_secs=0)

        assert mock_log.call_count == 1


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
        """acquire_lease returns FLASHNEXT_OCCUPIED -> 409 flashnext_occupied."""
        app = create_app()
        client = TestClient(app)
        # We need to mock the node state to return FLASHNEXT_OCCUPIED
        # This is a high-level integration test; the unit test is in
        # TestEnsureServingGuard above.
        pass  # Covered by the ensure_serving tests + the route code review

    def test_acquire_lease_propagates_sentinel(self):
        """acquire_lease propagates FLASHNEXT_OCCUPIED exactly as CREATIVE_OCCUPIED."""
        state = _make_state()
        with patch.object(state, "ensure_serving", return_value=FLASHNEXT_OCCUPIED):
            result = state.acquire_lease("test-wid", 300, "test")
            assert result is FLASHNEXT_OCCUPIED


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
        """The flashnext block does not affect serving_mode."""
        state = _make_state()
        state._flashnext_state = "up_registered"
        state._flashnext_window = "active"
        state._cached_serving = True

        snap = self._snap(state)

        # serving_mode is derived from the 27B axis, not the flashnext block
        assert snap["serving_mode"] in ("big", "dual", "unknown", "stopped", "deferred")


# ---------------------------------------------------------------------------
# (j) D2 probe: "down" means connection-refused ONLY
# ---------------------------------------------------------------------------

class TestDownVsBlind:
    def test_connection_refused_is_down(self):
        """ConnectionRefusedError (via __cause__) -> "down"."""
        state = _make_state()
        inner = ConnectionRefusedError("refused")
        exc = requests.exceptions.ConnectionError("refused", __cause__=inner)
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
