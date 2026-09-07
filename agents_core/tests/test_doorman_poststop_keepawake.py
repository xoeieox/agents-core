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

These tests drive the ACTUAL _start_refresh_thread._loop closure (started
via _start_refresh_thread, with the loop thread cancelled after its first
iteration) and assert on the argv of any subprocess.run invocations:

  (A) no leases + service_stopped=True  -> NO gw-keepawake hold call
  (B) one active lease                  -> the doorman-refresh hold call
  (C) stop in flight, no leases         -> NO gw-keepawake hold call
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from agents_core.doorman_server import (
    GW_URL_DEFAULT,
    _NodeState,
    _start_refresh_thread,
)

DOORMAN_THREAD_NAME = "doorman-refresh"


def _run_one_tick(nodes: dict[str, _NodeState]) -> MagicMock:
    """Start the REAL refresh thread with the given nodes, let it complete
    exactly one tick, cancel it, and return the mocked subprocess.run.

    The loop's first iteration runs immediately (first_iteration skips the
    sleep), so cancelling the thread right after start guarantees exactly
    one tick of the per-node body — the code under test.
    """
    with patch("agents_core.doorman_server.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        t = _start_refresh_thread(nodes)
        try:
            # The first tick runs synchronously inside the thread start path
            # only after the thread body begins; join briefly so the tick
            # completes, then cancel before the next sleep.
            t.join(timeout=10.0)
        finally:
            t.cancel()
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
