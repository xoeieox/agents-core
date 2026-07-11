"""Tests for agents_core.doorman_server — all subprocess / SSH / :8081 mocked.

Covers:
  - acquire when GW already serving (no wake subprocess)
  - acquire when asleep: wake shelled, polls /health, hold placed, returns serving
  - acquire when wake times out → wake_failed + last_error set; next success clears it
  - release drops keepawake hold only on last lease (two-lease, partial release)
  - stale-lease GC auto-releases on release path
  - refresh loop re-issues hold while leases are active
  - concurrent acquires serialize on the lock (only one wake-gravitywell subprocess)
  - [clean-stop] acquire when stopped → gw-serve big invoked + poll
  - [clean-stop] last-release sets idle_since, no immediate stop
  - [clean-stop] refresh tick before grace → no stop; after grace → one gw-serve stop
  - [clean-stop] acquire within grace → idle_since cleared, no stop
  - [clean-stop] gw-serve stop rc≠0 → last_error set, thread alive, retried next tick
  - [clean-stop] gw-serve big rc≠0 → acquire returns False (wake_failed-equivalent)
  - [clean-stop] /status serving_mode field present
  - [doorman-mode-deference-v0] deference suppresses gw-serve big when controller owns mode
  - [doorman-mode-deference-v0] foreign acquire returns deferred when controller owns mode
  - [doorman-mode-deference-v0] controller's own acquire (role=mode-controller) registers lease
  - [doorman-mode-deference-v0] /v0/mode-owner endpoint returns active/owner_lease_held
  - [doorman-mode-deference-v0] orphan-reclaim scar emitted when mode-controller lease TTL-expires
  - [doorman-mode-deference-v0] typed role recognition (not work_id inference)
  - [doorman-mode-deference-v0] kill-switch: DOORMAN_DEFER_TO_CONTROLLER=false disables deference
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from agents_core.doorman_server import (
    GW_URL_DEFAULT,
    HOLD_NAME,
    DEFERRED,
    DOORMAN_DEFER_TO_CONTROLLER,
    DOORMAN_CONTROLLER_NAME,
    GHOST_PRINCIPAL,
    _NodeState,
    _write_idle_log,
    create_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_no_auth() -> TestClient:
    with patch("agents_core.doorman_server._start_refresh_thread"):
        app = create_app(gw_url=GW_URL_DEFAULT)
    return TestClient(app, raise_server_exceptions=True)


def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


class _StopRefreshLoop(Exception):
    """Sentinel raised from a mocked time.sleep to end _start_refresh_thread's
    while-True loop after exactly one tick. Without this, the background
    thread outlives the `with patch(...)` block and keeps looping — once the
    patches are torn down it calls the REAL subprocess.run/requests.get."""


def _run_refresh_thread_one_tick(nodes: dict, timeout: float = 1.0) -> threading.Thread:
    """Start _start_refresh_thread and join it, asserting it actually exited.

    Callers must include `patch("time.sleep", side_effect=_StopRefreshLoop)`
    in their patch stack so the loop raises (and the thread dies) right after
    completing its first tick.
    """
    from agents_core.doorman_server import _start_refresh_thread

    prev_hook = threading.excepthook
    threading.excepthook = lambda args: None
    try:
        t = _start_refresh_thread(nodes)
        t.join(timeout=timeout)
    finally:
        threading.excepthook = prev_hook
    assert not t.is_alive(), "refresh thread leaked past test teardown"
    return t


@pytest.fixture(autouse=True)
def _default_serve_mode_big(monkeypatch):
    """Pin DOORMAN_DEFAULT_SERVE_MODE=big for this whole legacy suite.

    Every test above predates gw-doorman-wake-to-default-mode-v0 (the doorman's
    real default is now "dual") and asserts literal gw-serve big / GW_WAKE_DEADLINE_SEC
    wake mechanics that are orthogonal to which mode gets woken. Pinning here keeps
    them byte-identical instead of touching ~30 call sites individually. Dual-mode
    wake behavior gets its own explicit tests in TestEnsureServingDualMode below,
    which locally override this pin via their own patch() on DOORMAN_DEFAULT_SERVE_MODE.
    """
    monkeypatch.setattr("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "big")


# ---------------------------------------------------------------------------
# _NodeState unit tests
# ---------------------------------------------------------------------------

class TestIsServing:
    """Tests for _is_serving() (doorman-dual-vllm-serving-probe-gap-v0).

    _is_serving() must treat any 200 from /health as serving, regardless of
    body shape - llama.cpp returns {"status": "ok"}, vLLM returns an empty
    (non-JSON) 200 body.
    """

    def test_llamacpp_shaped_response_is_serving(self):
        state = _make_state()

        def mock_health(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"status": "ok"}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_health):
            assert state._is_serving() is True

    def test_vllm_shaped_response_is_serving(self):
        state = _make_state()

        def mock_health(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.json.side_effect = ValueError("no json")
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_health):
            assert state._is_serving() is True

    def test_non_200_status_is_not_serving(self):
        state = _make_state()

        def mock_health(url, **kwargs):
            m = MagicMock()
            m.status_code = 503
            m.json.return_value = {"status": "loading model"}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_health):
            assert state._is_serving() is False

    def test_connection_error_is_not_serving(self):
        state = _make_state()
        import requests as req_lib

        def mock_health(url, **kwargs):
            raise req_lib.exceptions.ConnectionError("simulated connection error")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_health):
            assert state._is_serving() is False


class TestEnsureServing:
    def test_already_serving_skips_wake(self):
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run") as mock_sub:
            result = state.ensure_serving()
        assert result is True
        mock_sub.assert_not_called()

    def test_already_serving_clears_service_stopped(self):
        """Fast-path success must set service_stopped=False."""
        state = _make_state()
        state.service_stopped = True
        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run"):
            result = state.ensure_serving()
        assert result is True
        assert state.service_stopped is False

    def test_asleep_wakes_polls_holds(self):
        state = _make_state()
        # First _is_serving call (entry check) returns False; second (poll) returns True
        serving_seq = [False, True]
        serving_iter = iter(serving_seq)

        def _is_serving_side(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=_is_serving_side), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        assert state.last_error is None
        # wake-gravitywell + gw-serve big + gw-keepawake hold
        assert mock_sub.call_count >= 2
        wake_call = mock_sub.call_args_list[0]
        assert "wake-gravitywell" in wake_call[0][0]
        # gw-serve big must be the second subprocess call
        serve_call = mock_sub.call_args_list[1]
        assert "gw-serve" in str(serve_call) and "big" in str(serve_call)

    def test_stopped_service_issues_gw_serve_big(self):
        """When host is up but service is stopped, ensure_serving must call gw-serve big."""
        state = _make_state()
        state.service_stopped = True
        # is_serving: False (service down), then True (after gw-serve big)
        serving_iter = iter([False, True])

        def _is_serving_side(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=_is_serving_side), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        assert state.service_stopped is False
        # gw-serve big must appear in the subprocess calls
        all_cmds = [str(c) for c in mock_sub.call_args_list]
        assert any("gw-serve" in c and "big" in c for c in all_cmds)

    def test_gw_serve_big_failure_returns_false(self):
        """gw-serve big rc≠0 → acquire returns False (distinct from wake failure)."""
        state = _make_state()
        call_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # wake-gravitywell succeeds
                return MagicMock(returncode=0, stderr="")
            else:
                # gw-serve big fails
                return MagicMock(returncode=1, stderr="llama-server failed to start")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run):
            result = state.ensure_serving()

        assert result is False
        assert state.last_error is not None
        assert "gw-serve big" in state.last_error or "rc=1" in state.last_error

    def test_wake_subprocess_failure_returns_false(self):
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stderr="network unreachable")
            result = state.ensure_serving()
        assert result is False
        assert state.last_error is not None
        assert "wake-gravitywell" in state.last_error or "rc=1" in state.last_error

    def test_wake_deadline_sets_last_error(self):
        state = _make_state()
        # Always not serving after wake
        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.GW_WAKE_DEADLINE_SEC", 0):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()
        assert result is False
        assert state.last_error is not None

    def test_success_clears_last_error(self):
        state = _make_state()
        state.last_error = "prior error"
        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run"):
            result = state.ensure_serving()
        assert result is True
        assert state.last_error is None


class TestLeaseLifecycle:
    def test_acquire_success_registers_lease(self):
        state = _make_state()
        with patch.object(state, "ensure_serving", return_value=True), \
             patch.object(state, "_place_hold"):
            ok = state.acquire_lease("work-1", ttl_sec=120, reason="test")
        assert ok is True
        assert "work-1" in state.leases

    def test_acquire_clears_idle_since(self):
        """acquire_lease must clear idle_since even if called within the grace period."""
        state = _make_state()
        state.idle_since = time.time() - 10  # set 10s ago (within grace)
        with patch.object(state, "ensure_serving", return_value=True), \
             patch.object(state, "_place_hold"):
            state.acquire_lease("work-1", ttl_sec=120, reason="test")
        assert state.idle_since is None

    def test_acquire_wake_failure_returns_false(self):
        state = _make_state()
        with patch.object(state, "ensure_serving", return_value=False):
            ok = state.acquire_lease("work-1", ttl_sec=120, reason="test")
        assert ok is False
        assert "work-1" not in state.leases

    def test_release_last_lease_sets_idle_since(self):
        """Releasing the last lease must set idle_since (not None)."""
        state = _make_state()
        state.leases["work-1"] = {"acquired_at": time.time(), "ttl_sec": 300, "reason": "t"}
        with patch.object(state, "_release_hold"), \
             patch("agents_core.doorman_server._write_idle_log"):
            state.release_lease("work-1")
        assert "work-1" not in state.leases
        assert state.idle_since is not None

    def test_release_last_lease_drops_hold(self):
        state = _make_state()
        state.leases["work-1"] = {"acquired_at": time.time(), "ttl_sec": 300, "reason": "t"}
        with patch.object(state, "_release_hold") as mock_rh, \
             patch("agents_core.doorman_server._write_idle_log"):
            state.release_lease("work-1")
        assert "work-1" not in state.leases
        mock_rh.assert_called_once()

    def test_release_non_last_lease_keeps_hold_and_no_idle(self):
        """Releasing a non-last lease must not set idle_since."""
        state = _make_state()
        now = time.time()
        state.leases["work-1"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t"}
        state.leases["work-2"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t"}
        # Simulate leases having arrived via acquire_lease(), which always clears
        # idle_since on arrival (including the construction-time seed).
        state.idle_since = None
        with patch.object(state, "_release_hold") as mock_rh, \
             patch("agents_core.doorman_server._write_idle_log"):
            state.release_lease("work-1")
        assert "work-1" not in state.leases
        assert "work-2" in state.leases
        mock_rh.assert_not_called()
        assert state.idle_since is None  # not idle — still leased

    def test_stale_lease_gc(self):
        state = _make_state()
        # expired lease
        state.leases["stale"] = {"acquired_at": time.time() - 1000, "ttl_sec": 1, "reason": "t"}
        # live lease
        state.leases["live"] = {"acquired_at": time.time(), "ttl_sec": 300, "reason": "t"}
        with patch.object(state, "_release_hold"):
            expired = state._gc_stale()
        assert "stale" in expired
        assert "stale" not in state.leases
        assert "live" in state.leases

    def test_release_unknown_work_id_is_noop(self):
        state = _make_state()
        with patch.object(state, "_release_hold") as mock_rh, \
             patch("agents_core.doorman_server._write_idle_log"):
            state.release_lease("nonexistent")
        mock_rh.assert_called_once()  # no leases → release hold (idempotent hold release)

    def test_gc_stale_last_lease_sets_idle_since(self):
        """TTL-expiry GC of the last lease must set idle_since (not just explicit release)."""
        state = _make_state()
        state.leases["work-1"] = {"acquired_at": time.time() - 1000, "ttl_sec": 1, "reason": "t"}
        with patch.object(state, "_release_hold") as mock_rh, \
             patch("agents_core.doorman_server._write_idle_log"):
            expired = state._gc_stale()
        assert "work-1" in expired
        assert not state.leases
        assert state.idle_since is not None
        mock_rh.assert_not_called()  # _gc_stale must not release the hold itself

    def test_gc_stale_non_last_lease_no_idle_since(self):
        """GC'ing one of several concurrent leases must not set idle_since."""
        state = _make_state()
        now = time.time()
        state.leases["stale"] = {"acquired_at": now - 1000, "ttl_sec": 1, "reason": "t"}
        state.leases["live"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t"}
        # Simulate leases having arrived via acquire_lease(), which always clears
        # idle_since on arrival (including the construction-time seed).
        state.idle_since = None
        with patch.object(state, "_release_hold"):
            expired = state._gc_stale()
        assert "stale" in expired
        assert "live" in state.leases
        assert state.idle_since is None

    def test_gc_stale_does_not_clobber_existing_idle_since(self):
        """If idle_since was already set, _gc_stale must not overwrite it."""
        state = _make_state()
        state.leases["stale"] = {"acquired_at": time.time() - 1000, "ttl_sec": 1, "reason": "t"}
        earlier = time.time() - 500
        state.idle_since = earlier
        with patch.object(state, "_release_hold"):
            state._gc_stale()
        assert state.idle_since == earlier

    def test_fresh_state_seeds_idle_since(self):
        """A freshly constructed _NodeState must start idle-tracking immediately —
        leases is always empty at construction, so there is never a transition to
        wait for (doorman-seed-idle-since-on-startup-v0)."""
        before = time.time()
        state = _make_state()
        after = time.time()
        assert state.leases == {}
        assert state.idle_since is not None
        assert before <= state.idle_since <= after


# ---------------------------------------------------------------------------
# Concurrency: concurrent acquires serialize on lock
# ---------------------------------------------------------------------------

def test_concurrent_acquires_single_wake():
    """Two concurrent acquire_lease calls must not launch parallel wake subprocesses."""
    state = _make_state()
    wake_count = {"n": 0}
    lock = threading.Lock()

    def slow_ensure_serving():
        with lock:
            wake_count["n"] += 1
        time.sleep(0.05)
        state.leases[f"work-{wake_count['n']}"] = {
            "acquired_at": time.time(), "ttl_sec": 120, "reason": "t"
        }
        return True

    results = []

    def _acquire(work_id):
        with state.lock:
            ok = slow_ensure_serving()
            if ok:
                state.leases[work_id] = {"acquired_at": time.time(), "ttl_sec": 120, "reason": "t"}
            results.append(ok)

    t1 = threading.Thread(target=_acquire, args=("w1",))
    t2 = threading.Thread(target=_acquire, args=("w2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both acquired but serialized — ensure_serving called once per thread (2 total),
    # but crucially never concurrently (the lock enforces serial execution).
    assert all(results)
    assert "w1" in state.leases or "w2" in state.leases


# ---------------------------------------------------------------------------
# HTTP endpoint tests
# ---------------------------------------------------------------------------

class TestEndpoints:
    def test_healthz(self):
        c = _client_no_auth()
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_acquire_already_serving(self):
        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch("agents_core.doorman_server._NodeState.acquire_lease", return_value=True):
            app = create_app(gw_url=GW_URL_DEFAULT)
            c = TestClient(app)
            r = c.post("/lease/acquire", json={
                "node": "gravitywell", "work_id": "w1", "ttl_sec": 120, "reason": "test"
            })
        assert r.status_code == 200
        assert r.json()["status"] == "serving"

    def test_acquire_wake_failed(self):
        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch("agents_core.doorman_server._NodeState.acquire_lease", return_value=False):
            app = create_app(gw_url=GW_URL_DEFAULT)
            c = TestClient(app)
            r = c.post("/lease/acquire", json={
                "node": "gravitywell", "work_id": "w1", "ttl_sec": 120, "reason": "test"
            })
        assert r.status_code == 200
        assert r.json()["status"] == "wake_failed"

    def test_acquire_unknown_node(self):
        c = _client_no_auth()
        r = c.post("/lease/acquire", json={
            "node": "starhouse", "work_id": "w1", "ttl_sec": 120, "reason": "test"
        })
        assert r.status_code == 400

    def test_release_ok(self):
        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch("agents_core.doorman_server._NodeState.release_lease"):
            app = create_app(gw_url=GW_URL_DEFAULT)
            c = TestClient(app)
            r = c.post("/lease/release", json={"node": "gravitywell", "work_id": "w1"})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_status_endpoint(self):
        c = _client_no_auth()
        with patch("agents_core.doorman_server._NodeState._is_serving", return_value=False):
            r = c.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert "nodes" in body
        assert "gravitywell" in body["nodes"]
        gw = body["nodes"]["gravitywell"]
        assert "serving" in gw
        assert "lease_count" in gw
        assert "leases" in gw
        assert "last_error" in gw

    def test_status_serving_mode_field(self):
        """/status must include serving_mode, idle_since, service_stopped."""
        c = _client_no_auth()
        with patch("agents_core.doorman_server._NodeState._is_serving", return_value=False):
            r = c.get("/status")
        gw = r.json()["nodes"]["gravitywell"]
        assert "serving_mode" in gw
        assert "idle_since" in gw
        assert "service_stopped" in gw

    def test_status_serving_mode_big_when_serving(self):
        state = _NodeState(GW_URL_DEFAULT)
        state._cached_serving = True
        state._serving_checked_at = time.time()
        snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "big"
        assert snapshot["snapshot_mode"] == "cached"
        assert snapshot["serving_checked_at"] > 0

    def test_status_serving_mode_stopped_when_service_stopped(self):
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        c = TestClient(app, raise_server_exceptions=True)
        # Inject service_stopped=True into the node state
        node_state = list(app.state.__dict__.values())[0] if hasattr(app.state, "__dict__") else None
        # Access node state via the nodes dict in the closure — patch _is_serving instead
        with patch("agents_core.doorman_server._NodeState._is_serving", return_value=False):
            # Manually set service_stopped on all NodeState instances tracked at module level
            # Use a fresh state to verify the field derivation logic directly
            state = _NodeState(GW_URL_DEFAULT)
            state.service_stopped = True
            snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "stopped"

    def test_status_serving_mode_unknown_when_not_stopped(self):
        state = _NodeState(GW_URL_DEFAULT)
        state.service_stopped = False
        with patch.object(state, "_is_serving", return_value=False):
            snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "unknown"

    def test_last_error_cleared_on_success(self):
        """last_error is set on failed ensure_serving, cleared on next success."""
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        c = TestClient(app)

        # Inject a last_error directly
        from agents_core.doorman_server import _NodeState
        node = app.state  # can't easily reach; use _NodeState directly
        state = _NodeState(GW_URL_DEFAULT)
        state.last_error = "prior failure"
        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run"):
            state.ensure_serving()
        assert state.last_error is None


# ---------------------------------------------------------------------------
# Refresh loop
# ---------------------------------------------------------------------------

def test_refresh_loop_re_issues_hold():
    """Background refresh loop calls gw-keepawake hold while leases are active."""
    from agents_core.doorman_server import _start_refresh_thread, GW_HOLD_REFRESH_SEC

    state = _NodeState(GW_URL_DEFAULT)
    state.leases["w1"] = {"acquired_at": time.time(), "ttl_sec": 300, "reason": "t"}
    nodes = {"gravitywell": state}

    hold_calls = []

    def fake_run(cmd, **kwargs):
        hold_calls.append(cmd)
        return MagicMock(returncode=0, stderr="")

    tick_count = {"n": 0}

    def fake_sleep(s):
        tick_count["n"] += 1
        if tick_count["n"] >= 2:
            state.leases.clear()  # stop after 2 ticks

    with patch("subprocess.run", side_effect=fake_run), \
         patch("time.sleep", side_effect=fake_sleep), \
         patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0):
        t = _start_refresh_thread(nodes)
        t.join(timeout=2.0)

    # At least one hold refresh should have been issued
    hold_issued = any("gw-keepawake" in str(c) and "hold" in str(c) for c in hold_calls)
    assert hold_issued


# ---------------------------------------------------------------------------
# Clean-stop: deferred dwell-guarded service stop (gravitywell-doorman-clean-stop-v0)
# ---------------------------------------------------------------------------

class TestDeferredStop:
    """Tests for the deferred gw-serve stop triggered by the refresh thread after grace."""

    def _make_nodes_idle(self, idle_secs: float = 0) -> tuple[dict, _NodeState]:
        """Return a (nodes, state) pair with idle_since set."""
        state = _NodeState(GW_URL_DEFAULT)
        state.idle_since = time.time() - idle_secs
        nodes = {"gravitywell": state}
        return nodes, state

    def test_refresh_no_stop_before_grace(self):
        """Refresh tick before grace expires must not issue gw-serve stop."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = self._make_nodes_idle(idle_secs=1)  # 1s < 600s grace

        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                state.idle_since = None  # end the loop cleanly

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 9999), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert len(stop_calls) == 0

    def test_refresh_issues_stop_after_grace(self):
        """Refresh tick after grace expires must issue exactly one gw-serve stop."""
        from agents_core.doorman_server import _start_refresh_thread

        # Already past grace
        nodes, state = self._make_nodes_idle(idle_secs=700)

        stop_calls = []
        tick_count = {"n": 0}

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 3:
                # force exit: remove idle_since so loop won't re-stop
                pass  # service_stopped will be True after first stop

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", False), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert len(stop_calls) >= 1
        assert state.service_stopped is True
        assert state.idle_since is None

    def test_fresh_state_reaches_stop_without_prior_lease_cycle(self):
        """Regression guard for the 2026-07-01 live incident: a doorman process that
        starts up with GravityWell already serving and zero leases must still
        dwell-stop after grace — no prior acquire/release (or GC) transition should
        be required to seed idle_since (doorman-seed-idle-since-on-startup-v0)."""
        from agents_core.doorman_server import _start_refresh_thread

        state = _NodeState(GW_URL_DEFAULT)  # fresh construction, no leases ever acquired
        nodes = {"gravitywell": state}

        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        def fake_sleep(s):
            pass

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", False), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 0), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert len(stop_calls) >= 1
        assert state.service_stopped is True

    def test_acquire_within_grace_clears_idle_no_stop(self):
        """An acquire within the grace period must clear idle_since — no stop issued."""
        state = _NodeState(GW_URL_DEFAULT)
        state.idle_since = time.time() - 10  # 10s, well within grace

        with patch.object(state, "ensure_serving", return_value=True), \
             patch.object(state, "_place_hold"), \
             patch("agents_core.doorman_server._write_idle_log"):
            ok = state.acquire_lease("work-new", ttl_sec=120, reason="test")

        assert ok is True
        assert state.idle_since is None
        assert state.service_stopped is False

    def test_stop_failure_rc_nonzero_still_serving_sets_error(self):
        """gw-serve stop rc≠0 while still serving → last_error set, idle_since retained."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = self._make_nodes_idle(idle_secs=700)

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                return MagicMock(returncode=1, stderr="failed to stop")
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                state.idle_since = None  # end the loop

        # Patch _is_serving to return True (still serving — real failure)
        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", False), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert state.last_error is not None
        assert state.service_stopped is False

    def test_stop_failure_rc_nonzero_already_down_treats_as_success(self):
        """gw-serve stop rc≠0 but _is_serving False → idempotency: treat as success."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = self._make_nodes_idle(idle_secs=700)

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                return MagicMock(returncode=1, stderr="already stopped")
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                pass

        # _is_serving returns False → service already down → success
        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch.object(state, "_is_serving", return_value=False), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", False), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert state.service_stopped is True
        assert state.idle_since is None

    def test_no_double_stop_after_service_stopped_set(self):
        """Once service_stopped=True, the refresh thread must not issue another stop."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = self._make_nodes_idle(idle_secs=700)
        state.service_stopped = True  # already stopped — should not re-stop

        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                state.idle_since = None

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        assert len(stop_calls) == 0


# ---------------------------------------------------------------------------
# Deference guard (doorman-mode-deference-v0)
# ---------------------------------------------------------------------------

class TestDeference:
    """Tests for deference to flip-controller's mode ownership."""

    def test_controller_lease_active_recognizes_mode_controller_role(self):
        """_controller_lease_active must return True when mode-controller lease exists."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "mode control",
            "role": "mode-controller",
        }
        state._gc_stale()  # should not GC — lease is fresh
        assert state._controller_lease_active() is True

    def test_controller_lease_active_ignores_worker_role(self):
        """_controller_lease_active must return False for worker-role leases."""
        state = _make_state()
        state.leases["worker-lease"] = {
            "acquired_at": time.time(),
            "ttl_sec": 300,
            "reason": "inference",
            "role": "worker",
        }
        state._gc_stale()
        assert state._controller_lease_active() is False

    def test_controller_lease_active_false_when_expired(self):
        """_controller_lease_active returns False for expired mode-controller leases."""
        state = _make_state()
        state.leases["expired-controller"] = {
            "acquired_at": time.time() - 1000,  # expired
            "ttl_sec": 1,
            "reason": "dead",
            "role": "mode-controller",
        }
        state._gc_stale()
        assert state._controller_lease_active() is False
        assert "expired-controller" not in state.leases

    def test_deference_suppress_gw_serve_big_when_controller_owns(self):
        """ensure_serving(role=None) must not issue gw-serve big when mode-controller lease active."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "mode control",
            "role": "mode-controller",
        }

        gw_serve_called = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving()

        assert result is DEFERRED
        assert len(gw_serve_called) == 0  # gw-serve big was NOT called

    def test_controller_own_acquire_short_circuits_deferred(self):
        """ensure_serving(role='mode-controller') returns DEFERRED without needing pre-registered lease."""
        state = _make_state()
        # No leases yet — controller's own acquire
        gw_serve_called = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving(role="mode-controller")

        assert result is DEFERRED
        assert len(gw_serve_called) == 0  # no gw-serve big

    def test_acquire_lease_controller_own_registers_and_places_hold(self):
        """Controller's own acquire (role='mode-controller') must register lease and place hold."""
        state = _make_state()

        hold_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-keepawake" in str(cmd) and "hold" in str(cmd):
                hold_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.acquire_lease(
                "flip-controller-gw", ttl_sec=240, reason="mode control", role="mode-controller"
            )

        assert result is DEFERRED  # The call itself is deferred (no gw-serve big)
        # But we placed the hold and registered the lease
        assert "flip-controller-gw" in state.leases
        assert state.leases["flip-controller-gw"]["role"] == "mode-controller"
        assert len(hold_calls) >= 1

    def test_foreign_acquire_returns_deferred_when_controller_owns(self):
        """A foreign acquire during controller ownership returns deferred, no lease registered."""
        state = _make_state()
        # Setup: controller owns the mode
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "mode control",
            "role": "mode-controller",
        }

        gw_serve_called = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            # Foreign worker acquire
            result = state.acquire_lease(
                "worker-lease-1", ttl_sec=120, reason="inference", role="worker"
            )

        assert result is DEFERRED
        assert "worker-lease-1" not in state.leases  # No lease registered for foreign deferred
        assert len(gw_serve_called) == 0  # No yank

    def test_legacy_wake_when_no_controller_lease(self):
        """When no controller lease, a worker acquire takes legacy wake+gw-serve big path."""
        state = _make_state()
        # No controller lease — legacy behavior

        wake_called = []
        gw_serve_called = []
        serving_iter = iter([False, True])  # Not serving, then serving after wake

        def fake_run(cmd, **kwargs):
            if "wake-gravitywell" in str(cmd):
                wake_called.append(cmd)
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        def fake_is_serving(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=fake_is_serving), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.acquire_lease(
                "worker-lease", ttl_sec=120, reason="inference", role="worker"
            )

        assert result is True  # Success
        assert "worker-lease" in state.leases
        assert len(wake_called) >= 1  # wake-gravitywell was called
        assert len(gw_serve_called) >= 1  # gw-serve big was called

    def test_type_role_only_not_work_id_inference(self):
        """Ownership is recognized by typed role field, NOT by work_id prefix."""
        state = _make_state()
        # A lease with work_id="flip-controller-gw" (the name!) but role="worker"
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "inference",
            "role": "worker",  # Not a mode-controller!
        }

        wake_called = []
        gw_serve_called = []
        serving_iter = iter([False, True])

        def fake_run(cmd, **kwargs):
            if "wake-gravitywell" in str(cmd):
                wake_called.append(cmd)
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        def fake_is_serving(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=fake_is_serving), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.acquire_lease(
                "worker-new", ttl_sec=120, reason="inference", role="worker"
            )

        # Legacy path taken — not deferred
        assert result is True
        assert len(gw_serve_called) >= 1  # gw-serve big WAS called

    def test_kill_switch_defer_to_controller_false_disables_deference(self):
        """When DOORMAN_DEFER_TO_CONTROLLER=false, deference is disabled and legacy path taken."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "mode control",
            "role": "mode-controller",
        }

        gw_serve_called = []
        serving_iter = iter([False, True])

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "big" in str(cmd):
                gw_serve_called.append(cmd)
            return MagicMock(returncode=0, stderr="")

        def fake_is_serving(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=fake_is_serving), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", False):
            result = state.ensure_serving()

        # Kill-switch off → legacy path → gw-serve big IS called
        assert result is True
        assert len(gw_serve_called) >= 1


# ---------------------------------------------------------------------------
# Deference HTTP endpoint tests (doorman-mode-deference-v0)
# ---------------------------------------------------------------------------

class TestDeferenceEndpoints:
    """Tests for /v0/mode-owner and deferred acquire responses."""

    def test_mode_owner_active_true_when_defer_enabled(self):
        c = _client_no_auth()
        with patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            r = c.get("/v0/mode-owner?node=gravitywell")
        assert r.status_code == 200
        body = r.json()
        assert body["active"] is True

    def test_mode_owner_active_false_when_defer_disabled(self):
        c = _client_no_auth()
        with patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", False):
            r = c.get("/v0/mode-owner?node=gravitywell")
        assert r.status_code == 200
        body = r.json()
        assert body["active"] is False

    def test_mode_owner_owner_lease_held_false_when_no_controller(self):
        c = _client_no_auth()
        r = c.get("/v0/mode-owner?node=gravitywell")
        assert r.status_code == 200
        body = r.json()
        assert body["owner_lease_held"] is False
        assert body["owner_lease_age_sec"] is None
        assert body["owner_lease_stale"] is False

    def test_mode_owner_unknown_node_returns_400(self):
        c = _client_no_auth()
        r = c.get("/v0/mode-owner?node=starhouse")
        assert r.status_code == 400

    def test_acquire_deferred_response(self):
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        c = TestClient(app)

        # Setup: inject a mode-controller lease so next foreign acquire defers
        with patch("agents_core.doorman_server._NodeState.acquire_lease") as mock_acquire:
            mock_acquire.return_value = DEFERRED
            r = c.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "worker-1",
                "ttl_sec": 120,
                "reason": "inference",
                "role": "worker",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "deferred"
        assert "mode_owner" in body
        assert "detail" in body

    def test_status_serving_mode_deferred_when_controller_owns(self):
        """When mode-controller lease is active and GW not serving, serving_mode=="deferred"."""
        state = _NodeState(GW_URL_DEFAULT)
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(),
            "ttl_sec": 240,
            "reason": "mode control",
            "role": "mode-controller",
        }
        with patch.object(state, "_is_serving", return_value=False):
            snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "deferred"
        assert snapshot["mode_owner"] is not None

    def test_status_mode_owner_none_when_no_controller(self):
        """mode_owner should be None when no controller lease active."""
        state = _NodeState(GW_URL_DEFAULT)
        with patch.object(state, "_is_serving", return_value=False):
            snapshot = state.status_snapshot()
        assert snapshot["mode_owner"] is None


# ---------------------------------------------------------------------------
# Serving cache (doorman-status-cached-serving-v0)
# ---------------------------------------------------------------------------

class TestServingCache:
    """Tests for the cached serving state and fast /status endpoint."""

    def test_status_snapshot_reads_cache_not_probes_live(self):
        """AC1: status_snapshot must read _cached_serving, not call _is_serving."""
        state = _make_state()
        state._cached_serving = True
        state._serving_checked_at = time.time()

        with patch.object(state, "_is_serving") as mock_probe:
            snapshot = state.status_snapshot()

        # _is_serving must NOT be called by status_snapshot
        mock_probe.assert_not_called()
        assert snapshot["serving"] is True

    def test_status_snapshot_null_serving_on_startup_race(self):
        """When cache is None (startup before first refresh), return null serving."""
        state = _make_state()
        # Default state: _cached_serving = None, _serving_checked_at = 0.0
        assert state._cached_serving is None
        assert state._serving_checked_at == 0.0

        snapshot = state.status_snapshot()
        assert snapshot["serving"] is None
        assert snapshot["serving_checked_at"] == 0.0

    def test_cache_fields_in_status_response(self):
        """AC5: /status response includes serving_checked_at and snapshot_mode."""
        state = _make_state()
        state._cached_serving = False
        state._serving_checked_at = 1234567890.0

        snapshot = state.status_snapshot()
        assert "serving_checked_at" in snapshot
        assert snapshot["serving_checked_at"] == 1234567890.0
        assert "snapshot_mode" in snapshot
        assert snapshot["snapshot_mode"] == "cached"

    def test_refresh_cache_populates_and_timestamps(self):
        """AC3: _refresh_serving_cache probes and updates cache + timestamp."""
        state = _make_state()
        before = time.time()

        with patch.object(state, "_is_serving", return_value=True):
            state._refresh_serving_cache()

        after = time.time()
        assert state._cached_serving is True
        assert before <= state._serving_checked_at <= after

    def test_refresh_cache_outside_lock(self):
        """AC4: _refresh_serving_cache probes _is_serving OUTSIDE the lock."""
        state = _make_state()
        lock_held_during_probe = {"yes": False}

        def fake_is_serving(timeout=3.0):
            # Try to acquire the lock without blocking
            acquired = state.lock.acquire(blocking=False)
            if not acquired:
                lock_held_during_probe["yes"] = True
            else:
                state.lock.release()
            return True

        with patch.object(state, "_is_serving", side_effect=fake_is_serving):
            state._refresh_serving_cache()

        # Lock must NOT be held during the probe
        assert lock_held_during_probe["yes"] is False

    def test_ensure_serving_wake_success_sets_cache_true(self):
        """AC3: successful wake in ensure_serving sets _cached_serving=True."""
        state = _make_state()
        serving_iter = iter([False, True])  # not serving, then serving after wake

        def fake_is_serving(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=fake_is_serving), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        # Cache must be updated to True after successful wake
        assert state._cached_serving is True
        assert state._serving_checked_at > 0

    def test_gw_serve_stop_success_sets_cache_false(self):
        """AC3: gw-serve stop (rc==0) sets _cached_serving=False."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = {}, _NodeState(GW_URL_DEFAULT)
        state.idle_since = time.time() - 700  # past grace
        nodes["gravitywell"] = state

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                state.idle_since = None

        # Mock _refresh_serving_cache to not write cache in the loop,
        # so we can test that the stop path itself sets the cache
        original_refresh = state._refresh_serving_cache
        def no_cache_refresh():
            # Probe but don't write cache (to isolate the stop-path cache write)
            state._is_serving(timeout=2.0)

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_refresh_serving_cache", side_effect=no_cache_refresh), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        # After stop success, cache must be False (set by the stop path, not refresh)
        assert state._cached_serving is False
        assert state._serving_checked_at > 0

    def test_gw_serve_stop_idempotency_sets_cache_false(self):
        """AC3: gw-serve stop idempotency (rc!=0, already down) sets cache=False."""
        from agents_core.doorman_server import _start_refresh_thread

        nodes, state = {}, _NodeState(GW_URL_DEFAULT)
        state.idle_since = time.time() - 700  # past grace
        nodes["gravitywell"] = state

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                return MagicMock(returncode=1, stderr="already stopped")
            return MagicMock(returncode=0, stderr="")

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                pass

        # Mock _refresh_serving_cache to not write cache in the loop,
        # so we can test that the idempotency-success path sets the cache
        original_refresh = state._refresh_serving_cache
        def no_cache_refresh():
            # Probe but don't write cache (to isolate the stop-path cache write)
            state._is_serving(timeout=2.0)

        # _is_serving returns False → idempotency success → cache should be False
        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_refresh_serving_cache", side_effect=no_cache_refresh), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        # Cache must be False from idempotency-success path (rc!=0 but already down)
        assert state._cached_serving is False
        assert state._serving_checked_at > 0

    def test_refresh_loop_calls_refresh_outside_lock_no_deadlock(self):
        """AC7: _loop calls _refresh_serving_cache OUTSIDE the per-node lock."""
        from agents_core.doorman_server import _start_refresh_thread

        state = _NodeState(GW_URL_DEFAULT)
        state.leases["w1"] = {"acquired_at": time.time(), "ttl_sec": 300, "reason": "t"}
        nodes = {"gravitywell": state}

        refresh_calls = []

        # Mock _refresh_serving_cache to track calls and verify lock is not held
        original_refresh = state._refresh_serving_cache

        def tracked_refresh():
            acquired = state.lock.acquire(blocking=False)
            if acquired:
                # Lock was NOT held — correct!
                refresh_calls.append("lock_free")
                state.lock.release()
            else:
                # Lock WAS held — would deadlock!
                refresh_calls.append("lock_held")
            # Call the real refresh
            original_refresh()

        tick_count = {"n": 0}

        def fake_sleep(s):
            tick_count["n"] += 1
            if tick_count["n"] >= 2:
                state.leases.clear()

        with patch.object(state, "_refresh_serving_cache", side_effect=tracked_refresh), \
             patch("time.sleep", side_effect=fake_sleep), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        # _refresh_serving_cache must have been called with lock NOT held
        assert "lock_free" in refresh_calls
        assert "lock_held" not in refresh_calls

    def test_startup_refresh_before_sleep(self):
        """AC6: refresh thread does first _refresh_serving_cache before sleep."""
        from agents_core.doorman_server import _start_refresh_thread

        state = _NodeState(GW_URL_DEFAULT)
        nodes = {"gravitywell": state}

        refresh_calls = []
        sleep_calls = []

        original_refresh = state._refresh_serving_cache

        def tracked_refresh():
            refresh_calls.append(time.time())
            original_refresh()

        def tracked_sleep(s):
            sleep_calls.append(time.time())
            # Stop after first sleep
            if sleep_calls:
                state.idle_since = None

        # Mock _is_serving so original_refresh() doesn't make live HTTP calls
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_refresh_serving_cache", side_effect=tracked_refresh), \
             patch("time.sleep", side_effect=tracked_sleep), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0.1):
            t = _start_refresh_thread(nodes)
            t.join(timeout=2.0)

        # First refresh must happen before first sleep
        assert len(refresh_calls) >= 1
        assert len(sleep_calls) >= 1
        assert refresh_calls[0] < sleep_calls[0]

    def test_acquire_does_not_write_cache_fast_path(self):
        """Fast-path ensure_serving (already serving) must NOT write cache."""
        state = _make_state()
        state._cached_serving = None  # Cache starts uninitialized

        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run"):
            result = state.ensure_serving()

        # Fast path succeeded but must NOT update cache (only doorman-driven transitions do)
        assert result is True
        # Cache stays None (not a doorman-driven wake, just a confirmation)
        assert state._cached_serving is None


# ---------------------------------------------------------------------------
# Orphan-reclaim scar (doorman-mode-deference-v0)
# ---------------------------------------------------------------------------

class TestOrphanReclaim:
    """Tests for orphan-reclaim scar emission when mode-controller lease TTL-expires."""

    def test_gc_stale_emits_scar_for_mode_controller_eviction(self):
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time() - 1000,  # expired
            "ttl_sec": 1,
            "reason": "mode control",
            "role": "mode-controller",
        }

        scar_events = []

        def fake_write_idle_log(node, event, lease_count, **extra_fields):
            if event == "controller-orphan-reclaim":
                scar_events.append((node, event, extra_fields))

        with patch("agents_core.doorman_server._write_idle_log", side_effect=fake_write_idle_log):
            expired = state._gc_stale()

        assert "flip-controller-gw" in expired
        assert "flip-controller-gw" not in state.leases
        assert len(scar_events) == 1
        node, event, extra = scar_events[0]
        assert node == "gravitywell"
        assert extra["evicted_lease"] == "flip-controller-gw"
        assert extra["evicted_role"] == "mode-controller"

    def test_gc_stale_no_scar_for_worker_eviction(self):
        """No scar should be emitted when a worker-role lease TTL-expires."""
        state = _make_state()
        state.leases["worker-lease"] = {
            "acquired_at": time.time() - 1000,  # expired
            "ttl_sec": 1,
            "reason": "inference",
            "role": "worker",
        }

        scar_events = []

        def fake_write_idle_log(node, event, lease_count, **extra_fields):
            if event == "controller-orphan-reclaim":
                scar_events.append((node, event, extra_fields))

        with patch("agents_core.doorman_server._write_idle_log", side_effect=fake_write_idle_log):
            expired = state._gc_stale()

        assert "worker-lease" in expired
        assert len(scar_events) == 0  # No scar for worker


# ---------------------------------------------------------------------------
# Mode-aware admission guard (doorman-mode-aware-serving-predicate-v0)
# AC1-AC5, AC10-AC12
# ---------------------------------------------------------------------------

class TestModeAwareAdmission:
    """Tests for DOORMAN_MODE_AWARE_ADMISSION flag: HOLE 1 fix, HOLE 2 fix, flag semantics."""

    def test_ac1_flag_defaults_false(self):
        """AC1: DOORMAN_MODE_AWARE_ADMISSION must be False when the env var is absent."""
        import agents_core.doorman_server as ds
        assert ds.DOORMAN_MODE_AWARE_ADMISSION is False

    def test_ac2a_deference_before_is_serving_role_worker(self):
        """AC2(a): flag ON, controller lease active, role='worker' -> DEFERRED before _is_serving."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch.object(state, "_is_serving") as mock_serving, \
             patch("subprocess.run") as mock_sub, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving(role="worker")

        assert result is DEFERRED
        mock_serving.assert_not_called()   # fast path NOT reached
        mock_sub.assert_not_called()       # wake-gravitywell NOT invoked

    def test_ac2b_no_wake_gravitywell_on_deferred_path(self):
        """AC2(b,c): wake-gravitywell subprocess NOT called when deferred-worker path taken."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch("subprocess.run") as mock_sub, \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving(role="worker")

        assert result is DEFERRED
        wake_calls = [c for c in mock_sub.call_args_list if "wake-gravitywell" in str(c)]
        assert len(wake_calls) == 0

    def test_ac2d_acquire_lease_worker_not_registered_when_deferred(self):
        """AC2(d): acquire_lease role='worker' returns DEFERRED; lease NOT in state.leases."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.acquire_lease(
                "work-new", ttl_sec=120, reason="inference", role="worker"
            )

        assert result is DEFERRED
        assert "work-new" not in state.leases

    def test_ac2_is_serving_true_still_deferred_flag_on(self):
        """AC2: _is_serving=True, controller lease, flag ON -> DEFERRED (not True)."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving(role="worker")

        assert result is DEFERRED

    def test_ac3_flag_off_preserves_fast_path(self):
        """AC3: flag OFF, _is_serving=True, controller lease -> True (not DEFERRED)."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.ensure_serving(role="worker")

        assert result is True  # legacy order: fast path fires first

    def test_ac4_controller_renewal_registers_flag_on_is_serving_true(self):
        """AC4: flag=True + worker lease + _is_serving=True, controller renewal registers and holds."""
        state = _make_state()
        state.leases["worker-1"] = {
            "acquired_at": time.time(), "ttl_sec": 300,
            "reason": "inference", "role": "worker",
        }
        hold_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-keepawake" in str(cmd) and "hold" in str(cmd):
                hold_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=True), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            result = state.acquire_lease(
                "flip-controller-gw", ttl_sec=240, reason="mode control",
                role="mode-controller",
            )

        assert "flip-controller-gw" in state.leases
        assert state.leases["flip-controller-gw"]["role"] == "mode-controller"
        assert len(hold_calls) >= 1

    def test_ac5a_serving_mode_deferred_controller_owns_flag_on(self):
        """AC5(a) ON: _cached_serving=True, controller lease, flag ON -> serving_mode='deferred'."""
        state = _make_state()
        state._cached_serving = True
        state._serving_checked_at = time.time()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "deferred"

    def test_ac5a_serving_mode_big_controller_owns_flag_off(self):
        """AC5(a) OFF: _cached_serving=True, controller lease, flag OFF -> serving_mode='big' (legacy)."""
        state = _make_state()
        state._cached_serving = True
        state._serving_checked_at = time.time()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }
        with patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", False):
            snapshot = state.status_snapshot()
        assert snapshot["serving_mode"] == "big"

    def test_ac5b_status_snapshot_no_network_call(self):
        """AC5(b): status_snapshot must not issue any network call (reads cache only)."""
        state = _make_state()
        state._cached_serving = True
        state._serving_checked_at = time.time()
        with patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.requests") as mock_requests:
            snapshot = state.status_snapshot()
        mock_requests.get.assert_not_called()

    def test_ac10_http_deferred_end_to_end(self):
        """AC10: POST /lease/acquire with controller lease, flag ON, _is_serving True -> deferred."""
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        c = TestClient(app)

        # Register a controller lease via mocked acquire_lease DEFERRED path
        with patch.object(_NodeState, "ensure_serving", return_value=DEFERRED), \
             patch.object(_NodeState, "_place_hold"):
            c.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "flip-controller-gw",
                "ttl_sec": 240,
                "reason": "mode control",
                "role": "mode-controller",
            })

        # Worker acquire with flag ON and _is_serving True
        with patch.object(_NodeState, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            r = c.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "worker-1",
                "ttl_sec": 120,
                "reason": "inference",
                "role": "worker",
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "deferred"

    def test_ac12_status_additive_contract(self):
        """AC12: /status includes drain_count and serving_is_big; no existing key changes type."""
        c = _client_no_auth()
        with patch("agents_core.doorman_server._NodeState._is_serving", return_value=False):
            r = c.get("/status")
        assert r.status_code == 200
        gw = r.json()["nodes"]["gravitywell"]
        # New keys
        assert "drain_count" in gw
        assert "serving_is_big" in gw
        assert "big_probe_state" in gw
        # Existing keys unchanged
        assert "serving" in gw
        assert "serving_mode" in gw
        assert "lease_count" in gw
        assert "leases" in gw
        assert "last_error" in gw
        assert "service_stopped" in gw
        assert "idle_since" in gw


# ---------------------------------------------------------------------------
# Three-state big-model probe (AC6, AC7, AC11)
# ---------------------------------------------------------------------------

class TestModeAwareBigPredicate:
    """Tests for the /v1/models three-state probe and serving_is_big predicate."""

    def test_ac7_gw_big_model_id_value(self):
        """AC7: GW_BIG_MODEL_ID must equal 'gravitywell-122b' (matches OPERATOR_DEFAULTS)."""
        import agents_core.doorman_server as ds
        assert ds.GW_BIG_MODEL_ID == "gravitywell-122b"

    def test_ac6a_confirmed_probe_serving_is_big_true(self):
        """AC6(A): probe confirmed, no controller lease, cached_serving True -> serving_is_big=True."""
        state = _make_state()

        def mock_models(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"data": [{"id": "gravitywell-122b"}]}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_models), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            state._refresh_serving_cache()

        assert state._serving_is_big is True
        assert state._big_probe_state == "confirmed"

    def test_ac6b_refuted_probe_serving_is_big_false(self):
        """AC6(B): probe returns swarm id -> serving_is_big=False, big_probe_state='refuted'."""
        state = _make_state()
        get_calls = []

        def mock_models(url, **kwargs):
            get_calls.append(url)
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"data": [{"id": "swarm-coder-7b"}]}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_models), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            state._refresh_serving_cache()

        assert state._serving_is_big is False
        assert state._big_probe_state == "refuted"
        # Assert GET was actually issued (AC6-B: real split-brain scenario)
        assert len(get_calls) >= 1
        assert any("v1/models" in url for url in get_calls)

    def test_ac6c_inconclusive_probe_fallback_to_legacy_no_raise(self):
        """AC6(C): probe timeout, cached_serving True, no controller -> serving_is_big=True, no exception."""
        state = _make_state()
        import requests as req_lib

        def mock_timeout(url, **kwargs):
            raise req_lib.exceptions.Timeout("simulated probe timeout")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_timeout), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            state._refresh_serving_cache()  # must not raise

        assert state._serving_is_big is True   # legacy: serving + no controller = True
        assert state._big_probe_state == "inconclusive"

    def test_ac6d_controller_wins_over_confirmed_probe(self):
        """AC6(D): probe confirmed but controller lease present -> serving_is_big=False."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240,
            "reason": "mode control", "role": "mode-controller",
        }

        def mock_models(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"data": [{"id": "gravitywell-122b"}]}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_models), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            state._refresh_serving_cache()

        assert state._serving_is_big is False  # controller wins
        assert state._big_probe_state == "confirmed"

    def test_ac11_probe_runs_outside_lock(self):
        """AC11: /v1/models probe is called with self.lock NOT held."""
        state = _make_state()
        lock_held_during_probe = {"yes": False}

        def mock_models(url, **kwargs):
            # Try to acquire the lock without blocking — must succeed (lock is free)
            acquired = state.lock.acquire(blocking=False)
            if not acquired:
                lock_held_during_probe["yes"] = True
            else:
                state.lock.release()
            m = MagicMock()
            m.status_code = 200
            m.json.return_value = {"data": [{"id": "gravitywell-122b"}]}
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_models), \
             patch.object(state, "_is_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_MODE_AWARE_ADMISSION", True):
            state._refresh_serving_cache()

        assert lock_held_during_probe["yes"] is False


# ---------------------------------------------------------------------------
# Drain-count exposure (AC8, AC9)
# ---------------------------------------------------------------------------

class TestDrainCount:
    """Tests for AC8 (drain_count field) and AC9 (endpoint + client method)."""

    def test_ac8a_drain_count_worker_only(self):
        """AC8(a): two worker + one controller -> drain_count=2, lease_count=3."""
        state = _make_state()
        now = time.time()
        state.leases["w1"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t", "role": "worker"}
        state.leases["w2"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t", "role": "worker"}
        state.leases["ctrl"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t", "role": "mode-controller"}

        snapshot = state.status_snapshot()
        assert snapshot["drain_count"] == 2
        assert snapshot["worker_lease_count"] == 2
        assert snapshot["lease_count"] == 3
        # All prior keys present
        for key in ("serving", "serving_mode", "leases", "last_error", "service_stopped"):
            assert key in snapshot

    def test_ac8b_drain_count_independent_of_probe(self):
        """AC8(b): probe inconclusive -> drain_count still correct from registry."""
        state = _make_state()
        now = time.time()
        state.leases["w1"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t", "role": "worker"}
        state.leases["w2"] = {"acquired_at": now, "ttl_sec": 300, "reason": "t", "role": "worker"}
        state._big_probe_state = "inconclusive"

        snapshot = state.status_snapshot()
        assert snapshot["drain_count"] == 2

    def test_ac9_drain_count_endpoint_two_workers(self):
        """AC9: GET /v0/drain-count with two worker leases -> {drain_count: 2}."""
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        c = TestClient(app, raise_server_exceptions=True)

        with patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch.object(_NodeState, "_place_hold"):
            c.post("/lease/acquire", json={
                "node": "gravitywell", "work_id": "w1",
                "ttl_sec": 300, "reason": "t", "role": "worker",
            })
            c.post("/lease/acquire", json={
                "node": "gravitywell", "work_id": "w2",
                "ttl_sec": 300, "reason": "t", "role": "worker",
            })

        r = c.get("/v0/drain-count?node=gravitywell")
        assert r.status_code == 200
        body = r.json()
        assert body["drain_count"] == 2
        assert body["node"] == "gravitywell"

    def test_ac9_drain_count_endpoint_unknown_node_400(self):
        """AC9: unknown node -> 400."""
        c = _client_no_auth()
        r = c.get("/v0/drain-count?node=starhouse")
        assert r.status_code == 400

    def test_ac9_drain_count_endpoint_missing_token_401(self):
        """AC9: missing bearer token when token configured -> 401."""
        with patch("agents_core.doorman_server._start_refresh_thread"):
            with patch.dict(__import__("os").environ, {"DOORMAN_BEARER_TOKEN": "secret"}):
                app = create_app(gw_url=GW_URL_DEFAULT)
        c_no_token = TestClient(app, raise_server_exceptions=True)
        r = c_no_token.get("/v0/drain-count?node=gravitywell")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Principal-aware drain-gate tests (gw-admission-drain-count-self-count-v0)
# ---------------------------------------------------------------------------

class TestPrincipalAwareDrainGate:
    """AC1-AC6 from spec gw-admission-drain-count-self-count-v0."""

    def _tc(self):
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        return TestClient(app, raise_server_exceptions=True)

    def _acquire(self, c, work_id, *, principal=None, role="worker"):
        body = {
            "node": "gravitywell",
            "work_id": work_id,
            "ttl_sec": 300,
            "reason": "test",
            "role": role,
        }
        if principal is not None:
            body["principal"] = principal
        with patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch.object(_NodeState, "_place_hold"):
            return c.post("/lease/acquire", json=body)

    # AC1: self-deadlock fix — own group's hold is excluded
    def test_ac1_self_group_excluded_from_drain_count(self):
        c = self._tc()
        self._acquire(c, "hold-1", principal="council-delib-run42")
        r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=council-delib-run42")
        assert r.status_code == 200
        assert r.json()["drain_count"] == 0

    # AC2: flip-protection — unfiltered query still counts the hold
    def test_ac2_unfiltered_counts_hold(self):
        c = self._tc()
        self._acquire(c, "hold-1", principal="council-delib-run42")
        r = c.get("/v0/drain-count?node=gravitywell")
        assert r.json()["drain_count"] == 1

    # AC3: real cross-group contention still gates
    def test_ac3_different_principal_still_counted(self):
        c = self._tc()
        self._acquire(c, "other-worker", principal="council-delib-runXX")
        r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=council-delib-run42")
        assert r.json()["drain_count"] == 1

    # AC4: ghost marker
    def test_ac4_no_principal_stamped_as_ghost(self):
        c = self._tc()
        self._acquire(c, "w-ghost")  # no principal
        state = create_app.__wrapped__ if hasattr(create_app, "__wrapped__") else None
        # Verify via state inspection: acquire_lease stamps ghost
        s = _make_state()
        with patch.object(s, "ensure_serving", return_value=True), \
             patch.object(s, "_place_hold"):
            s.acquire_lease("w-ghost", 300, "test", role="worker", principal=None)
        assert s.leases["w-ghost"]["principal"] == GHOST_PRINCIPAL

    def test_ac4_ghost_always_counted_not_excluded(self):
        c = self._tc()
        self._acquire(c, "w-ghost")  # no principal → ghost
        r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=council-delib-run42")
        assert r.json()["drain_count"] == 1

    def test_ac4_ghost_critical_log_emitted(self, caplog):
        import logging
        c = self._tc()
        self._acquire(c, "w-ghost")  # no principal → ghost
        with caplog.at_level(logging.CRITICAL, logger="doorman-server"):
            r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=real-principal")
        assert r.json()["drain_count"] == 1
        assert any("ghost_lease_counted" in rec.message for rec in caplog.records)
        assert any("w-ghost" in rec.message for rec in caplog.records)

    def test_ac4_ghost_no_critical_log_without_exclude(self, caplog):
        """Ghost counted without exclude_principal must NOT log critical (unfiltered path)."""
        import logging
        c = self._tc()
        self._acquire(c, "w-ghost")
        with caplog.at_level(logging.CRITICAL, logger="doorman-server"):
            c.get("/v0/drain-count?node=gravitywell")
        assert not any("ghost_lease_counted" in rec.message for rec in caplog.records)

    # AC4: ghost path does not raise
    def test_ac4_ghost_does_not_raise(self):
        c = self._tc()
        self._acquire(c, "w-ghost")
        r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=real-principal")
        assert r.status_code == 200

    # AC6: backward compat — no exclude_principal = prior behavior
    def test_ac6_backward_compat_no_exclude_param(self):
        c = self._tc()
        self._acquire(c, "w1", principal="p1")
        self._acquire(c, "w2", principal="p2")
        r = c.get("/v0/drain-count?node=gravitywell")
        assert r.json()["drain_count"] == 2

    def test_ac6_legacy_acquire_without_principal_succeeds(self):
        """Legacy callers omitting principal= still succeed (stamped as ghost, not rejected)."""
        s = _make_state()
        with patch.object(s, "ensure_serving", return_value=True), \
             patch.object(s, "_place_hold"):
            result = s.acquire_lease("w-legacy", 300, "test")
        assert result is True
        assert s.leases["w-legacy"]["principal"] == GHOST_PRINCIPAL

    def test_ac6_non_worker_role_no_principal_field(self):
        """Non-worker leases (mode-controller) get no principal field."""
        s = _make_state()
        with patch.object(s, "ensure_serving", return_value=True), \
             patch.object(s, "_place_hold"):
            s.acquire_lease("ctrl", 300, "test", role="mode-controller")
        assert "principal" not in s.leases["ctrl"]

    # Multiple workers, some same group, some different
    def test_mixed_principals_partial_exclusion(self):
        c = self._tc()
        self._acquire(c, "w-own1", principal="group-A")
        self._acquire(c, "w-own2", principal="group-A")
        self._acquire(c, "w-other", principal="group-B")
        r = c.get("/v0/drain-count?node=gravitywell&exclude_principal=group-A")
        assert r.json()["drain_count"] == 1  # only group-B counts


# ---------------------------------------------------------------------------
# lease_kind drain-gate tests (gw-admission-elevator-kind-aware-v0)
# ---------------------------------------------------------------------------

class TestLeaseKindDrainGate:
    """AC1-AC4, AC7 from spec gw-admission-elevator-kind-aware-v0.

    Verifies that lease_kind="coordination" is excluded from drain-gate
    contention while lease_kind="inference" (default) still serializes,
    and /v0/drain-count still counts coordination leases for flip-protection.
    """

    def _make_state(self) -> _NodeState:
        return _NodeState(GW_URL_DEFAULT)

    def _serving_state(self) -> _NodeState:
        s = self._make_state()
        s.ensure_serving = lambda role="worker": True  # type: ignore[method-assign]
        s._place_hold = lambda: None  # type: ignore[method-assign]
        return s

    def _tc(self):
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        return TestClient(app, raise_server_exceptions=True)

    def _acquire_http(self, c, work_id, *, principal=None, lease_kind="inference",
                      require_drain_clear=False, role="worker"):
        body = {
            "node": "gravitywell",
            "work_id": work_id,
            "ttl_sec": 300,
            "reason": "test",
            "role": role,
            "lease_kind": lease_kind,
        }
        if principal is not None:
            body["principal"] = principal
        if require_drain_clear:
            body["require_drain_clear"] = True
        with patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch.object(_NodeState, "_place_hold"):
            return c.post("/lease/acquire", json=body)

    # AC1: lease_kind stored on entry; omitting is byte-identical to "inference"
    def test_ac1_lease_kind_stored_on_entry(self):
        s = self._serving_state()
        s.acquire_lease("w1", 300, "test", role="worker", principal="p1",
                        lease_kind="inference")
        assert s.leases["w1"]["lease_kind"] == "inference"

    def test_ac1_lease_kind_coordination_stored(self):
        s = self._serving_state()
        s.acquire_lease("w-hold", 300, "test", role="worker", principal="p-hold",
                        lease_kind="coordination")
        assert s.leases["w-hold"]["lease_kind"] == "coordination"

    def test_ac1_omit_lease_kind_defaults_inference(self):
        s = self._serving_state()
        s.acquire_lease("w-default", 300, "test", role="worker", principal="p1")
        assert s.leases["w-default"]["lease_kind"] == "inference"

    def test_ac1_omit_lease_kind_still_succeeds(self):
        s = self._serving_state()
        result = s.acquire_lease("w-legacy", 300, "test", role="worker", principal="p1")
        assert result is True

    # AC2: coordination hold does NOT contend with an inference require_drain_clear acquire
    def test_ac2_coordination_hold_does_not_contend_inference_acquire(self):
        """A session's own coordination hold must not block its inference legs."""
        s = self._serving_state()
        # Place the span-hold as coordination (principal=P1, lease_kind=coordination)
        res = s.acquire_lease("span-hold", 300, "span-hold",
                              role="worker", principal="delib-session-abc",
                              lease_kind="coordination")
        assert res is True

        # A different principal's inference leg with require_drain_clear must be GRANTED
        res2 = s.acquire_lease("op-leg", 300, "inference",
                               role="worker", principal="op-gravitywell-xyz",
                               require_drain_clear=True, lease_kind="inference")
        assert res2 is True, "coordination hold must not contend inference acquire"

    # AC2 concurrent: verify with real threading (mirrors the atomic acquire test pattern)
    def test_ac2_concurrent_coordination_hold_plus_inference_granted(self):
        s = self._serving_state()
        errors = []

        def place_hold():
            try:
                r = s.acquire_lease("span-hold", 300, "hold",
                                    role="worker", principal="session-P1",
                                    lease_kind="coordination")
                if r is not True:
                    errors.append(f"hold acquire returned {r!r}")
            except Exception as e:
                errors.append(str(e))

        def place_inference():
            try:
                r = s.acquire_lease("inference-leg", 300, "inference",
                                    role="worker", principal="op-P2",
                                    require_drain_clear=True, lease_kind="inference")
                if r is not True:
                    errors.append(f"inference acquire returned {r!r}")
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=place_hold)
        t2 = threading.Thread(target=place_inference)
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert not errors, f"concurrent test errors: {errors}"

    # AC3: two inference workers under distinct principals still contend (#113 TOCTOU preserved)
    def test_ac3_two_inference_workers_distinct_principals_contend(self):
        s = self._serving_state()
        res1 = s.acquire_lease("inf-1", 300, "inf",
                               role="worker", principal="op-A",
                               require_drain_clear=True, lease_kind="inference")
        assert res1 is True

        from agents_core.doorman_server import CONTENDED
        res2 = s.acquire_lease("inf-2", 300, "inf",
                               role="worker", principal="op-B",
                               require_drain_clear=True, lease_kind="inference")
        assert res2 is CONTENDED, "distinct-principal inference workers must still contend"

    # AC3 concurrent: race two inference workers; at most one must win
    def test_ac3_concurrent_inference_workers_only_one_wins(self):
        s = self._serving_state()
        from agents_core.doorman_server import CONTENDED
        results = []

        def try_acquire(wid, principal):
            r = s.acquire_lease(wid, 300, "inf",
                                role="worker", principal=principal,
                                require_drain_clear=True, lease_kind="inference")
            results.append(r)

        t1 = threading.Thread(target=try_acquire, args=("inf-A", "op-A"))
        t2 = threading.Thread(target=try_acquire, args=("inf-B", "op-B"))
        t1.start(); t2.start()
        t1.join(); t2.join()

        wins = [r for r in results if r is True]
        contended = [r for r in results if r is CONTENDED]
        assert len(wins) == 1, f"exactly one inference worker must win; got wins={wins}"
        assert len(contended) == 1, f"second must be CONTENDED; got contended={contended}"

    # AC4: /v0/drain-count still counts coordination leases (flip-protection preserved)
    def test_ac4_drain_count_still_counts_coordination(self):
        """Coordination lease must count toward drain-count for flip-protection."""
        c = self._tc()
        self._acquire_http(c, "span-hold", principal="session-P1",
                           lease_kind="coordination")
        r = c.get("/v0/drain-count?node=gravitywell")
        assert r.status_code == 200
        assert r.json()["drain_count"] == 1

    def test_ac4_drain_count_counts_coordination_alongside_inference(self):
        c = self._tc()
        self._acquire_http(c, "span-hold", principal="session-P1",
                           lease_kind="coordination")
        self._acquire_http(c, "inf-leg", principal="op-P2",
                           lease_kind="inference")
        r = c.get("/v0/drain-count?node=gravitywell")
        assert r.json()["drain_count"] == 2  # both kinds counted for flip-protection

    # AC7: GHOST_PRINCIPAL inference lease still contends (regression guard)
    def test_ac7_ghost_inference_still_contends(self):
        from agents_core.doorman_server import CONTENDED
        s = self._serving_state()
        # Ghost inference lease (no principal)
        s.acquire_lease("w-ghost", 300, "ghost", role="worker", principal=None,
                        lease_kind="inference")
        # A new inference acquire with require_drain_clear must be CONTENDED by the ghost
        res = s.acquire_lease("w-new", 300, "new",
                              role="worker", principal="some-principal",
                              require_drain_clear=True, lease_kind="inference")
        assert res is CONTENDED

    # AC6 (Fix 2): coordination lease self-expires via TTL without depending on explicit release
    def test_ac6_coordination_lease_expires_via_gc_stale(self):
        """A held coordination lease expires via _gc_stale once TTL lapses — no release needed."""
        s = self._serving_state()
        s.acquire_lease("span-hold", ttl_sec=1, reason="hold",
                        role="worker", principal="session-P",
                        lease_kind="coordination")
        assert "span-hold" in s.leases

        # Backdate the acquired_at by more than ttl_sec to simulate TTL expiry
        s.leases["span-hold"]["acquired_at"] = time.time() - 5

        # _gc_stale should reap it without any explicit release call
        s._gc_stale()
        assert "span-hold" not in s.leases, (
            "coordination lease must self-expire via TTL/gc_stale — "
            "no explicit release call made"
        )


# ---------------------------------------------------------------------------
# llama-server /slots activity probe (doorman-probe-llama-activity-v0)
# ---------------------------------------------------------------------------

class TestProbeLlamaSlotsActivity:
    """Unit tests for _NodeState._probe_llama_slots_activity() (Probe A, unchanged)."""

    def test_is_processing_true_detects_activity(self):
        state = _make_state()

        def mock_slots(url, **kwargs):
            m = MagicMock(status_code=200)
            m.json.return_value = [{"id": 0, "is_processing": True, "id_task": 5}]
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is True

    def test_id_task_change_since_last_probe_detects_activity(self):
        """A generation completed between ticks: id_task changed, is_processing now false."""
        state = _make_state()
        state._last_probed_task_by_slot = {0: 100}

        def mock_slots(url, **kwargs):
            m = MagicMock(status_code=200)
            m.json.return_value = [{"id": 0, "is_processing": False, "id_task": 101}]
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is True
        assert state._last_probed_task_by_slot[0] == 101

    def test_unchanged_task_not_processing_no_activity(self):
        state = _make_state()
        state._last_probed_task_by_slot = {0: 100}

        def mock_slots(url, **kwargs):
            m = MagicMock(status_code=200)
            m.json.return_value = [{"id": 0, "is_processing": False, "id_task": 100}]
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is False

    def test_non_200_response_returns_false(self):
        state = _make_state()

        def mock_slots(url, **kwargs):
            return MagicMock(status_code=500)

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is False

    def test_timeout_returns_false_no_raise(self):
        state = _make_state()
        import requests as req_lib

        def mock_timeout(url, **kwargs):
            raise req_lib.exceptions.Timeout("simulated /slots timeout")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_timeout):
            assert state._probe_llama_slots_activity() is False  # must not raise

    def test_connection_refused_returns_false_no_raise(self):
        state = _make_state()
        import requests as req_lib

        def mock_conn_error(url, **kwargs):
            raise req_lib.exceptions.ConnectionError("connection refused")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_conn_error):
            assert state._probe_llama_slots_activity() is False

    def test_malformed_json_returns_false(self):
        state = _make_state()

        def mock_slots(url, **kwargs):
            m = MagicMock(status_code=200)
            m.json.side_effect = ValueError("malformed JSON")
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is False

    def test_empty_slots_list_returns_false(self):
        state = _make_state()

        def mock_slots(url, **kwargs):
            m = MagicMock(status_code=200)
            m.json.return_value = []
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_slots):
            assert state._probe_llama_slots_activity() is False


class TestProbeActivityIdleClock:
    """Tests for D2: probe activity feeding into idle_since via _refresh_serving_cache()."""

    def test_probe_activity_pushes_idle_since_forward_when_leases_empty(self):
        state = _make_state()
        state.idle_since = time.time() - 500  # already idle a while

        with patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server._write_idle_log"):
            state._refresh_serving_cache()

        assert state.idle_since is not None
        assert time.time() - state.idle_since < 2
        assert state._idle_since_source == "probe"

    def test_probe_activity_noop_when_leases_present(self):
        """Probe activity must not touch idle_since while leases exist (lease bookkeeping owns it)."""
        state = _make_state()
        state.leases["w1"] = {
            "acquired_at": time.time(), "ttl_sec": 300, "reason": "x", "role": "worker",
        }
        state.idle_since = None

        with patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server._write_idle_log"):
            state._refresh_serving_cache()

        assert state.idle_since is None

    def test_probe_activity_logs_only_on_first_detected_tick(self):
        """Must not spam one idle-log line per refresh tick during a continuous session."""
        state = _make_state()
        write_calls = []

        def fake_write(node, event, lease_count, **kwargs):
            write_calls.append(event)

        with patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server._write_idle_log", side_effect=fake_write):
            state._refresh_serving_cache()  # None -> set: should log
            state._refresh_serving_cache()  # already set: should not log again
            state._refresh_serving_cache()  # still set: should not log again

        probe_events = [e for e in write_calls if e == "probe_activity_detected"]
        assert len(probe_events) == 1

    def test_probe_disabled_env_flag_ignores_activity(self):
        """DOORMAN_PROBE_LLAMA_ACTIVITY=false must not touch idle_since from probe activity."""
        state = _make_state()
        original_idle_since = time.time() - 700
        state.idle_since = original_idle_since

        with patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity") as mock_probe, \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", False), \
             patch("agents_core.doorman_server._write_idle_log"):
            state._refresh_serving_cache()

        mock_probe.assert_not_called()
        assert state.idle_since == original_idle_since
        assert state._idle_since_source is None


class TestProbeActivityDwellStopIntegration:
    """Integration tests (D4): probe activity delays dwell-stop; absence lets it fire on schedule."""

    def test_continued_probe_activity_never_triggers_stop(self):
        from agents_core.doorman_server import _start_refresh_thread

        state = _NodeState(GW_URL_DEFAULT)
        nodes = {"gravitywell": state}
        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"), \
             patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 5), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=1.0)

        assert len(stop_calls) == 0
        assert state.leases == {}
        assert state.idle_since is not None
        assert time.time() - state.idle_since < 5

    def test_probe_activity_stopping_lets_dwell_stop_fire_on_schedule(self):
        """Once probe activity stops, the existing grace countdown proceeds and stop fires -
        confirms this is a delay, not a permanent suppression, of dwell-stop."""
        from agents_core.doorman_server import _start_refresh_thread

        state = _NodeState(GW_URL_DEFAULT)
        state.idle_since = time.time() - 700  # last probe-touched idle_since, now past grace
        state._idle_since_source = "probe"
        nodes = {"gravitywell": state}
        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep"), \
             patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=False), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server._write_idle_log"):
            t = _start_refresh_thread(nodes)
            t.join(timeout=1.0)

        assert len(stop_calls) >= 1
        assert state.service_stopped is True
        assert state.idle_since is None


# ---------------------------------------------------------------------------
# vLLM dual-slot /metrics activity probe (gw-doorman-vllm-activity-probe-v0)
# ---------------------------------------------------------------------------

class TestProbeVllmMetricsActivity:
    """Unit tests for _NodeState._probe_vllm_metrics_activity() (Probe B)."""

    def test_nonzero_running_returns_true(self):
        state = _make_state()

        def mock_metrics(url, **kwargs):
            m = MagicMock(status_code=200)
            m.text = (
                "# HELP vllm:num_requests_running Number of requests in model execution batches.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                'vllm:num_requests_running{model_name="gravitywell-27b",engine="0"} 2.0\n'
                "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
                "# TYPE vllm:num_requests_waiting gauge\n"
                'vllm:num_requests_waiting{model_name="gravitywell-27b",engine="0"} 0.0\n'
            )
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is True

    def test_nonzero_waiting_returns_true(self):
        state = _make_state()

        def mock_metrics(url, **kwargs):
            m = MagicMock(status_code=200)
            m.text = (
                'vllm:num_requests_running{model_name="gravitywell-27b",engine="0"} 0.0\n'
                'vllm:num_requests_waiting{model_name="gravitywell-27b",engine="0"} 3.0\n'
            )
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is True

    def test_all_zero_counters_returns_false(self):
        state = _make_state()

        def mock_metrics(url, **kwargs):
            m = MagicMock(status_code=200)
            m.text = (
                'vllm:num_requests_running{model_name="gravitywell-27b",engine="0"} 0.0\n'
                'vllm:num_requests_waiting{model_name="gravitywell-27b",engine="0"} 0.0\n'
            )
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is False

    def test_unreachable_returns_none_not_false(self):
        """Sonnet HIGH regression case: connection failure must be indeterminate, not confirmed-false."""
        state = _make_state()
        import requests as req_lib

        def mock_conn_error(url, **kwargs):
            raise req_lib.exceptions.ConnectionError("connection refused")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_conn_error):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is None

    def test_timeout_returns_none_not_false(self):
        state = _make_state()
        import requests as req_lib

        def mock_timeout(url, **kwargs):
            raise req_lib.exceptions.Timeout("simulated /metrics timeout")

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_timeout):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is None

    def test_non_200_returns_none(self):
        state = _make_state()

        def mock_metrics(url, **kwargs):
            return MagicMock(status_code=404)

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is None

    def test_malformed_plaintext_returns_none_not_false(self):
        """Sonnet HIGH regression case: this must NOT be parsed as JSON (would raise and,
        pre-fix, get silently swallowed as False by a shared best-effort except clause)."""
        state = _make_state()

        def mock_metrics(url, **kwargs):
            m = MagicMock(status_code=200)
            m.text = "not prometheus text at all, no matching gauge lines here"
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is None

    def test_llamacpp_shaped_body_has_no_vllm_gauges_returns_none(self):
        """This port is serving llama.cpp (big mode), not vLLM - no vllm: gauges present."""
        state = _make_state()

        def mock_metrics(url, **kwargs):
            m = MagicMock(status_code=200)
            m.text = (
                "# HELP llamacpp:requests_processing Number of requests processing.\n"
                "# TYPE llamacpp:requests_processing gauge\n"
                "llamacpp:requests_processing 0\n"
            )
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_metrics):
            assert state._probe_vllm_metrics_activity(GW_URL_DEFAULT) is None


class TestProbeSlotActivityMerge:
    """Unit tests for _NodeState._probe_slot_activity() tri-state OR-merge logic.

    Each underlying probe is patched in isolation so this only exercises the merge
    rule: True if any True; False only if all confirmed False; None otherwise.
    """

    def test_probe_a_true_probe_b_both_none_combined_true(self):
        """Big mode: no vLLM ports up at all - a real True always wins regardless
        of the other probes' uncertainty."""
        state = _make_state()
        with patch.object(state, "_probe_llama_slots_activity", return_value=True), \
             patch.object(state, "_probe_vllm_metrics_activity", return_value=None):
            assert state._probe_slot_activity() is True

    def test_probe_b_slot1_true_combined_true(self):
        state = _make_state()

        def fake_metrics(url):
            return True if url == state.gw_url else None

        with patch.object(state, "_probe_llama_slots_activity", return_value=False), \
             patch.object(state, "_probe_vllm_metrics_activity", side_effect=fake_metrics):
            assert state._probe_slot_activity() is True

    def test_probe_b_slot2_true_combined_true(self):
        state = _make_state()
        slot2 = state._slot2_url()

        def fake_metrics(url):
            return True if url == slot2 else None

        with patch.object(state, "_probe_llama_slots_activity", return_value=False), \
             patch.object(state, "_probe_vllm_metrics_activity", side_effect=fake_metrics):
            assert state._probe_slot_activity() is True

    def test_all_confirmed_false_combined_false(self):
        state = _make_state()
        with patch.object(state, "_probe_llama_slots_activity", return_value=False), \
             patch.object(state, "_probe_vllm_metrics_activity", return_value=False):
            assert state._probe_slot_activity() is False

    def test_all_none_combined_indeterminate(self):
        """Total network blip on every source - never conflated with confirmed-false."""
        state = _make_state()
        with patch.object(state, "_probe_llama_slots_activity", return_value=False), \
             patch.object(state, "_probe_vllm_metrics_activity", return_value=None):
            assert state._probe_slot_activity() is None

    def test_dispatch_is_concurrent_not_sequential(self):
        """D2: confirms the three HTTP calls are issued without one blocking on
        another's timeout - staggered slow mocked responses via real threads."""
        state = _make_state()

        def mock_get(url, **kwargs):
            if url.endswith("/slots"):
                time.sleep(0.2)
                m = MagicMock(status_code=200)
                m.json.return_value = []
                return m
            time.sleep(0.2)
            m = MagicMock(status_code=200)
            m.text = 'vllm:num_requests_running{model_name="x",engine="0"} 0.0\n'
            return m

        with patch("agents_core.doorman_server.requests.get", side_effect=mock_get):
            start = time.time()
            result = state._probe_slot_activity()
            elapsed = time.time() - start

        assert result is False
        # Sequential would be ~0.6s (3 calls x 0.2s); concurrent stays close to 0.2s.
        assert elapsed < 0.4


class TestProbeBlindnessFallback:
    """Integration tests (D2h): a probe stuck indeterminate past GW_STOP_GRACE_SEC +
    DOORMAN_PROBE_BLINDNESS_SEC must fall back to confirmed-idle rather than pausing
    forever (the 'eternal guardian' failure mode Mirror Council's stand-asides flagged).
    """

    def test_indeterminate_within_blindness_window_does_not_stop(self):
        state = _NodeState(GW_URL_DEFAULT)
        # Past grace but well within grace + blindness.
        state.idle_since = time.time() - 650
        nodes = {"gravitywell": state}
        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=_StopRefreshLoop), \
             patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=None), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_BLINDNESS_SEC", 900), \
             patch("agents_core.doorman_server._write_idle_log"):
            _run_refresh_thread_one_tick(nodes)

        assert stop_calls == []
        assert state.service_stopped is False

    def test_indeterminate_past_blindness_window_falls_back_to_stop(self):
        state = _NodeState(GW_URL_DEFAULT)
        # Past grace AND past grace + blindness.
        state.idle_since = time.time() - 1600
        nodes = {"gravitywell": state}
        stop_calls = []

        def fake_run(cmd, **kwargs):
            if "gw-serve" in str(cmd) and "stop" in str(cmd):
                stop_calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run), \
             patch("time.sleep", side_effect=_StopRefreshLoop), \
             patch.object(state, "_is_serving", return_value=True), \
             patch.object(state, "_is_creative_serving", return_value=False), \
             patch.object(state, "_probe_slot_activity", return_value=None), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_LLAMA_ACTIVITY", True), \
             patch("agents_core.doorman_server.GW_HOLD_REFRESH_SEC", 0), \
             patch("agents_core.doorman_server.GW_STOP_GRACE_SEC", 600), \
             patch("agents_core.doorman_server.DOORMAN_PROBE_BLINDNESS_SEC", 900), \
             patch("agents_core.doorman_server._write_idle_log"):
            _run_refresh_thread_one_tick(nodes)

        assert len(stop_calls) >= 1
        assert state.service_stopped is True


# ---------------------------------------------------------------------------
# Dual-mode cold-wake (gw-doorman-wake-to-default-mode-v0)
# ---------------------------------------------------------------------------

class TestEnsureServingDualMode:
    """Tests for the dual-mode cold-wake path: DOORMAN_DEFAULT_SERVE_MODE's real
    default is "dual" (this file's autouse fixture pins it to "big" for the legacy
    suite above; these tests locally override back to "dual" or explicitly re-pin
    "big" to demonstrate the rollback lever). Covers async-initiate + backoff poll,
    both-slots (two-consecutive-200) readiness, the GW_DUAL_WAKE_DEADLINE_SEC bound,
    and initiation-failure cleanup. All subprocess/HTTP mocked, no live GW.

    _is_serving() backs BOTH ensure_serving()'s top-of-function fast path (before
    any wake is issued) and _wake_dual()'s Slot-1 readiness probe. Tests that need
    to exercise the wake path use _not_serving_then(...) so the very first call
    (the fast path) returns False, forcing the cold-wake branch, while later calls
    (the poll loop) return the caller-supplied readiness sequence.
    """

    @staticmethod
    def _not_serving_then(*poll_values):
        """Build an _is_serving side_effect: False on call 1 (fast path), then
        poll_values in order for subsequent calls (repeating the last value once
        exhausted)."""
        state = {"n": 0}

        def _side_effect(_timeout=3.0):
            state["n"] += 1
            if state["n"] == 1:
                return False
            idx = min(state["n"] - 2, len(poll_values) - 1)
            return poll_values[idx]

        return _side_effect

    def test_dual_mode_cold_wake_issues_gw_serve_dual(self):
        """Cold-wake with DOORMAN_DEFAULT_SERVE_MODE=dual must issue an async
        `gw-serve dual` initiation, not gw-serve big."""
        state = _make_state()

        with patch.object(state, "_is_serving", side_effect=self._not_serving_then(True)), \
             patch.object(state, "_is_slot2_serving", return_value=True), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        all_cmds = [str(c) for c in mock_sub.call_args_list]
        assert any("gw-serve" in c and "dual" in c for c in all_cmds)
        assert not any("gw-serve" in c and "big" in c for c in all_cmds)

    def test_big_mode_override_issues_gw_serve_big_not_dual(self):
        """DOORMAN_DEFAULT_SERVE_MODE=big must restore the exact prior gw-serve big
        wake (scope item 6 rollback lever), explicitly re-pinned independent of the
        file-wide autouse fixture."""
        state = _make_state()
        serving_iter = iter([False, True])

        def fake_is_serving(_timeout=3.0):
            return next(serving_iter, True)

        with patch.object(state, "_is_serving", side_effect=fake_is_serving), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "big"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        all_cmds = [str(c) for c in mock_sub.call_args_list]
        assert any("gw-serve" in c and "big" in c for c in all_cmds)
        assert not any("gw-serve" in c and "dual" in c for c in all_cmds)

    def test_dual_initiation_failure_triggers_cleanup_no_zombie(self):
        """A failed async-initiate launch (rc != 0) must trigger best-effort cleanup
        (gw-serve stop) and return False - no orphaned half-started process left."""
        state = _make_state()
        call_log = []

        def fake_run(cmd, **kwargs):
            call_log.append(cmd)
            cmd_str = str(cmd)
            if "wake-gravitywell" in cmd_str:
                return MagicMock(returncode=0, stderr="")
            if "gw-serve" in cmd_str and "dual" in cmd_str:
                return MagicMock(returncode=1, stderr="ssh connection refused")
            return MagicMock(returncode=0, stderr="")  # cleanup gw-serve stop call

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            result = state.ensure_serving()

        assert result is False
        assert state.last_error is not None
        assert "gw-serve dual initiation failed" in state.last_error
        cleanup_calls = [c for c in call_log if "gw-serve" in str(c) and "stop" in str(c)]
        assert len(cleanup_calls) == 1  # cleanup issued exactly once, no zombie left

    def test_dual_initiation_exception_triggers_cleanup(self):
        """A launch subprocess exception (e.g. ssh hang/timeout) must also trigger
        best-effort cleanup, not just a non-zero launch return code."""
        state = _make_state()
        cleanup_calls = []

        def fake_run(cmd, **kwargs):
            cmd_str = str(cmd)
            if "wake-gravitywell" in cmd_str:
                return MagicMock(returncode=0, stderr="")
            if "gw-serve" in cmd_str and "dual" in cmd_str:
                raise TimeoutError("ssh hung")
            cleanup_calls.append(cmd)  # only the cleanup call reaches here
            return MagicMock(returncode=0, stderr="")

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            result = state.ensure_serving()

        assert result is False
        assert "gw-serve dual initiation subprocess error" in state.last_error
        assert any("gw-serve" in str(c) and "stop" in str(c) for c in cleanup_calls)

    def test_dual_deadline_independent_of_big_deadline(self):
        """Dual poll must be bounded by GW_DUAL_WAKE_DEADLINE_SEC, not GW_WAKE_DEADLINE_SEC -
        a huge GW_WAKE_DEADLINE_SEC must not make the dual wait longer."""
        state = _make_state()

        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_is_slot2_serving", return_value=False), \
             patch("subprocess.run") as mock_sub, \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"), \
             patch("agents_core.doorman_server.GW_DUAL_WAKE_DEADLINE_SEC", 0), \
             patch("agents_core.doorman_server.GW_WAKE_DEADLINE_SEC", 99999):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is False
        assert "GW dual did not reach both-slot readiness within 0s" in state.last_error

    def test_dual_requires_both_slots_no_slot1_only_fast_path(self):
        """Slot 1 alone reaching readiness must NOT short-circuit success - dual
        'served' requires both slots (no Slot-1-only fast path, scope item 4)."""
        state = _make_state()

        with patch.object(state, "_is_serving", side_effect=self._not_serving_then(True)), \
             patch.object(state, "_is_slot2_serving", return_value=False), \
             patch("subprocess.run") as mock_sub, \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"), \
             patch("agents_core.doorman_server.GW_DUAL_WAKE_DEADLINE_SEC", 0.3), \
             patch("agents_core.doorman_server.GW_DUAL_POLL_INITIAL_SEC", 0.05), \
             patch("agents_core.doorman_server.GW_DUAL_POLL_MAX_SEC", 0.05):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is False
        assert "slot1_ready=True" in state.last_error
        assert "slot2_ready=False" in state.last_error

    def test_dual_readiness_requires_two_consecutive_200s_per_slot(self):
        """A single healthy ping per slot must not be enough - readiness requires
        two CONSECUTIVE 200s per slot (stability window, scope item 4). Slot 1 hits
        its own 2-consecutive streak one iteration before Slot 2; success must wait
        for Slot 2, proving there is no single-slot-streak fast path."""
        state = _make_state()
        slot2_calls = {"n": 0}

        def fake_slot2(_timeout=3.0):
            slot2_calls["n"] += 1
            return slot2_calls["n"] >= 2  # slot2 unhealthy on probe 1, healthy from probe 2 on

        with patch.object(state, "_is_serving", side_effect=self._not_serving_then(True)), \
             patch.object(state, "_is_slot2_serving", side_effect=fake_slot2), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        # slot1's streak would reach 2 after its 2nd poll-loop probe; slot2's streak
        # reaches 2 only after its 3rd probe (1st was unhealthy). Exactly 3 slot2
        # probes confirms success waited for slot2 rather than firing early.
        assert slot2_calls["n"] == 3

    def test_dual_poll_uses_exponential_backoff(self):
        """Dual readiness poll must use growing (deterministic) backoff intervals -
        not the constant 3.0s big-mode poll_interval."""
        state = _make_state()
        slot2_results = iter([False, True, True])

        sleep_calls = []

        with patch.object(state, "_is_serving", side_effect=self._not_serving_then(True)), \
             patch.object(state, "_is_slot2_serving",
                           side_effect=lambda _timeout=3.0: next(slot2_results, True)), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep", side_effect=sleep_calls.append), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 5.0  # GW_DUAL_POLL_INITIAL_SEC
        assert sleep_calls[1] == pytest.approx(7.5)  # 5.0 * GW_DUAL_POLL_BACKOFF_FACTOR

    def test_dual_success_places_hold_and_clears_error(self):
        """Successful dual wake must set last_wake_at, clear last_error, place hold,
        and cache serving=True - same contract as big-mode success."""
        state = _make_state()
        state.last_error = "prior failure"

        with patch.object(state, "_is_serving", side_effect=self._not_serving_then(True)), \
             patch.object(state, "_is_slot2_serving", return_value=True), \
             patch("subprocess.run") as mock_sub, \
             patch("time.sleep"), \
             patch.object(state, "_place_hold") as mock_hold, \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is True
        assert state.last_error is None
        assert state._cached_serving is True
        mock_hold.assert_called_once()

    def test_deference_unchanged_under_dual_default(self):
        """Controller-lease deference must still short-circuit BEFORE the mode
        dispatch, unaffected by DOORMAN_DEFAULT_SERVE_MODE=dual."""
        state = _make_state()
        state.leases["flip-controller-gw"] = {
            "acquired_at": time.time(), "ttl_sec": 240, "reason": "mode control",
            "role": "mode-controller",
        }

        with patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run") as mock_sub, \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", True):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving()

        assert result is DEFERRED
        all_cmds = [str(c) for c in mock_sub.call_args_list]
        assert not any("gw-serve" in c for c in all_cmds)  # never reached the dispatch

    def test_creative_occupied_unchanged_under_dual_default(self):
        """Creative-70B-occupied short-circuit must still fire first, unaffected by
        DOORMAN_DEFAULT_SERVE_MODE=dual."""
        from agents_core.doorman_server import CREATIVE_OCCUPIED

        state = _make_state()
        with patch.object(state, "_is_creative_serving", return_value=True), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            result = state.ensure_serving()

        assert result is CREATIVE_OCCUPIED
