"""Regression tests for the doorman's post-stop keepawake-hold leak —
agents-core-doorman-poststop-keepawake-v0.

The rev-3 extraction (PR #295) moved the inline `if not state.leases:`
idle-stop block into _NodeState._decide_idle_stop(). That method returns
False for a node that is already stopped (service_stopped=True) or has a
stop in flight (_stop_in_flight=True), and the refresh loop's `continue`
only fires on a True return — so a stopped idle node fell through to the
keepawake-hold refresh and re-issued `gw-keepawake hold` every
GW_HOLD_REFRESH_SEC tick, holding the box awake forever after a clean
stop (the GW idle-suspend guard HOLDs on the hold file).

The fix: the loop now `continue`s for every no-lease node after the
_decide_idle_stop() check, so the hold refresh is reachable only when
state.leases is non-empty (the stated intent of the "Leases are active"
comment).

These tests drive the ACTUAL _start_refresh_thread._loop closure — one
tick, hermetically (all network probes and subprocesses mocked) — and
assert on the argv of any subprocess.run invocations:

  (A) no leases + service_stopped=True  -> NO gw-keepawake hold call
  (B) one active lease                  -> the doorman-refresh hold call
  (C) stop in flight, no leases         -> NO gw-keepawake hold call

The thread is stopped after exactly one tick using the EXISTING
test_doorman_server.py pattern: `patch("time.sleep",
side_effect=_StopRefreshLoop)` plus `_run_refresh_thread_one_tick(nodes)`,
which joins the thread and asserts it actually exited (no Thread.cancel()
— threading.Thread has no such method).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agents_core.doorman_server import (
    GW_URL_DEFAULT,
    _NodeState,
)
from agents_core.tests.test_doorman_server import (
    _StopRefreshLoop,
    _run_refresh_thread_one_tick,
)


# ---------------------------------------------------------------------------
# Hermetic pins (mirror the autouse fixtures in test_doorman_server.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _default_serve_mode_big(monkeypatch):
    """Pin DOORMAN_DEFAULT_SERVE_MODE=big so the suite's wake/stop
    dispatch is byte-identical regardless of the host's live default."""
    monkeypatch.setattr("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "big")


@pytest.fixture(autouse=True)
def _no_declared_posture_by_default(monkeypatch):
    """Pin the declared-posture read to "unreadable" (None) so tests never
    pick up a REAL /srv/agents/config/conductor.env off the host."""
    monkeypatch.setattr(
        "agents_core.doorman_server._NodeState._read_declared_home_posture",
        lambda self: None,
    )


def _mock_requests_get():
    """A requests.get that fails fast with a ConnectionError — every probe
    in the tick path (health, /slots, /metrics, flash-next seat, glances)
    is hermetically unreachable, so _refresh_serving_cache degrades to
    "not serving / indeterminate" without a single real network call.
    The probes never raise out of the loop (each catches its own errors),
    and _decide_idle_stop()'s stop path is unreachable in all three cases
    below (stopped / in-flight / leased), so no stop subprocess fires."""
    import requests as req_lib

    def _get(url, **kwargs):
        raise req_lib.exceptions.ConnectionError("hermetic test: no network")

    return _get


def _run_one_tick(nodes: dict[str, _NodeState]) -> MagicMock:
    """Start the REAL refresh thread with the given nodes, let it complete
    exactly one tick (the loop's first iteration skips the sleep), stop the
    thread via the _StopRefreshLoop sentinel, and return the mocked
    subprocess.run.

    All network probes are mocked (requests.get -> ConnectionError) and
    time.sleep is the loop-terminator, so the tick under test does no real
    I/O and the thread is guaranteed dead before the patch stack tears
    down (no leaked thread hitting the real subprocess/requests).
    """
    with patch("agents_core.doorman_server.subprocess.run") as mock_run, \
         patch("agents_core.doorman_server.requests.get", side_effect=_mock_requests_get()), \
         patch("time.sleep", side_effect=_StopRefreshLoop):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _run_refresh_thread_one_tick(nodes)
        return mock_run


def _keepawake_hold_calls(mock_run: MagicMock) -> list[list[str]]:
    """argv of every mocked subprocess.run call that issues a keepawake hold."""
    calls = []
    for call in mock_run.call_args_list:
        args, kwargs = call
        argv = args[0] if args else kwargs.get("args")
        if isinstance(argv, (list, tuple)) and argv:
            joined = " ".join(str(a) for a in argv)
            if "gw-keepawake" in joined and "hold" in joined:
                calls.append(list(argv))
    return calls


def _stopped_node() -> _NodeState:
    """A node in the post-clean-stop posture: no leases, service stopped."""
    state = _NodeState(GW_URL_DEFAULT)
    state.leases = {}
    state.service_stopped = True
    state.idle_since = None
    state._stop_in_flight = False
    state._stop_in_flight_since = None
    state._flashnext_window = "none"
    state._flashnext_window_closed_at = None
    state._probe_indeterminate = False
    return state


def _lease_node() -> _NodeState:
    """The same node but with one active lease."""
    state = _stopped_node()
    state.leases["work-1"] = {
        "acquired_at": time.time(),
        "ttl_sec": 300,
        "reason": "test",
        "role": "worker",
    }
    return state


def _stop_in_flight_node() -> _NodeState:
    """No leases, service NOT yet stopped, but a stop is in flight."""
    state = _stopped_node()
    state.service_stopped = False
    state.idle_since = time.time()  # grace clock running, stop pending
    state._stop_in_flight = True
    state._stop_in_flight_since = time.time()
    return state


class TestPostStopKeepawakeGating:
    """The loop's hold-refresh gating: the keepawake hold is refreshed only
    when state.leases is non-empty."""

    def test_a_stopped_idle_node_issues_no_hold(self):
        """Case A (the bug): no leases + service_stopped=True -> one refresh
        tick issues NO gw-keepawake hold subprocess call."""
        state = _stopped_node()
        mock_run = _run_one_tick({"gravitywell": state})
        hold_calls = _keepawake_hold_calls(mock_run)
        assert hold_calls == [], (
            f"stopped idle node must not refresh the keepawake hold; "
            f"got {hold_calls!r}"
        )

    def test_b_active_lease_issues_hold(self):
        """Case B (regression guard): one active lease -> the tick issues the
        doorman-refresh hold call."""
        state = _lease_node()
        mock_run = _run_one_tick({"gravitywell": state})
        hold_calls = _keepawake_hold_calls(mock_run)
        assert len(hold_calls) == 1, (
            f"lease-active node must refresh the keepawake hold exactly once "
            f"per tick; got {hold_calls!r}"
        )
        argv = hold_calls[0]
        assert argv[0] == "ssh"
        assert argv[1] == "gravitywell"
        assert "gw-keepawake" in argv[2]
        assert "hold" in argv[2]
        assert "doorman-refresh" in argv[2]

    def test_c_stop_in_flight_issues_no_hold(self):
        """Case C: stop in flight (_stop_in_flight=True), no leases -> no
        hold call."""
        state = _stop_in_flight_node()
        mock_run = _run_one_tick({"gravitywell": state})
        hold_calls = _keepawake_hold_calls(mock_run)
        assert hold_calls == [], (
            f"stop-in-flight node must not refresh the keepawake hold; "
            f"got {hold_calls!r}"
        )
