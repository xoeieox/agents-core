"""Tests for gw-admission-pending-orphan-reclaim-v0.

AC1  — dead-enqueuer pending orphan is reaped within one reap() call.
AC1b — PID recycling: start_time mismatch → dead; match → alive.
AC1c — presumed-dead backstop: old ticket reaped even when liveness unknown.
AC2  — wedge regression: dead-orphan at FIFO head cleared; live waiter then admits.
AC3  — in-process leak: enforce loop abort leaves no pending ticket.
AC4  — parent-kill cleanup: fail_pending_by_pid fails matching tickets; does not raise.
AC5  — no regressions: GW_ADMISSION_MODE=off takes direct path (zero elevator interaction).
AC6  — reclaim_stale semantics unchanged: only touches claimed tickets.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.elevator import (
    ElevatorStore,
    GW_ADMISSION_MAX_WAIT_SEC,
    GW_ADMISSION_ORPHAN_GRACE_SEC,
    HOSTNAME,
    _get_process_start_time,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    s = ElevatorStore(db_path=tmp_path / "q.db")
    yield s
    s.close()


def _enqueue_gw_admission(store, principal="p-test", pid=None, host=None, start_time=None):
    """Helper: enqueue a gw-admission ticket with controlled enqueuer stamp."""
    # We patch os.getpid and _get_process_start_time to control the stamp.
    fake_pid = pid if pid is not None else os.getpid()
    fake_host = host if host is not None else HOSTNAME
    fake_start = start_time  # None → _get_process_start_time not mocked

    with patch("agents_core.elevator.HOSTNAME", fake_host), \
         patch("agents_core.elevator.os.getpid", return_value=fake_pid), \
         patch("agents_core.elevator._get_process_start_time",
               return_value=fake_start) as _mst:
        ticket = store.enqueue(
            lane="deliberation",
            kind="gw-admission",
            payload={"work_id": f"op-gw-{principal}"},
            principal=principal,
            latency_class="batch",
        )
    return ticket


def _age_ticket(store, ticket_id, seconds):
    """Back-date a ticket's created_at by `seconds` so the reaper sees it as aged."""
    new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with store._lock:
        store._conn.execute(
            "UPDATE queue_items SET created_at=? WHERE item_id=?",
            (new_ts, ticket_id),
        )
        store._conn.commit()


# ---------------------------------------------------------------------------
# AC1 — dead-enqueuer pending orphan is reaped
# ---------------------------------------------------------------------------

def test_dead_pid_reaped_within_one_reap(store):
    """A pending ticket whose stamped pid is dead is failed after one reap()."""
    dead_pid = 99999999  # almost certainly not a live pid

    ticket = _enqueue_gw_admission(store, pid=dead_pid, start_time=12345.0)
    _age_ticket(store, ticket, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    with patch("agents_core.elevator.os.kill", side_effect=OSError("no such process")):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "failed", f"expected failed, got {item['status']}"
    assert result["orphan_reaped"] == 1


def test_live_pid_matching_start_time_spared(store):
    """A pending ticket whose pid+start_time both match a live process is NOT reaped."""
    live_pid = os.getpid()
    live_start = _get_process_start_time(live_pid) or 12345.0

    ticket = _enqueue_gw_admission(store, pid=live_pid, start_time=live_start)
    _age_ticket(store, ticket, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    # pid is alive (os.kill succeeds), start_time matches
    with patch("agents_core.elevator.os.kill", return_value=None), \
         patch("agents_core.elevator._get_process_start_time", return_value=live_start):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "pending", f"live ticket should be pending, got {item['status']}"
    assert result["orphan_reaped"] == 0


def test_grace_period_protects_fresh_ticket(store):
    """A ticket younger than GW_ADMISSION_ORPHAN_GRACE_SEC is never reaped."""
    dead_pid = 99999999

    ticket = _enqueue_gw_admission(store, pid=dead_pid, start_time=12345.0)
    # Do NOT age it — it's fresh.

    with patch("agents_core.elevator.os.kill", side_effect=OSError("no such process")):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "pending"
    assert result["orphan_reaped"] == 0


# ---------------------------------------------------------------------------
# AC1b — PID recycling does not cause a false verdict
# ---------------------------------------------------------------------------

def test_pid_recycled_start_time_mismatch_reaped(store):
    """A ticket whose pid is alive but start_time mismatches is treated as dead (PID recycled)."""
    recycled_pid = os.getpid()  # some live pid
    original_start = 99.0       # not the actual start time of this process

    ticket = _enqueue_gw_admission(store, pid=recycled_pid, start_time=original_start)
    _age_ticket(store, ticket, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    current_start = original_start + 1000.0  # clearly different

    with patch("agents_core.elevator.os.kill", return_value=None), \
         patch("agents_core.elevator._get_process_start_time", return_value=current_start):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "failed", "recycled PID should be reaped"
    prov = item["provenance"] or {}
    assert prov.get("orphan_reap") == "pid_recycled"


def test_pid_alive_start_time_matches_spared(store):
    """Exact start_time match: pid alive, same start_time → ticket spared."""
    live_pid = os.getpid()
    live_start = 500.0

    ticket = _enqueue_gw_admission(store, pid=live_pid, start_time=live_start)
    _age_ticket(store, ticket, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    with patch("agents_core.elevator.os.kill", return_value=None), \
         patch("agents_core.elevator._get_process_start_time", return_value=live_start):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "pending"
    assert result["orphan_reaped"] == 0


# ---------------------------------------------------------------------------
# AC1c — presumed-dead backstop
# ---------------------------------------------------------------------------

def test_presumed_dead_backstop_no_stamp(store):
    """A pending gw-admission ticket older than max_wait is reaped even without a stamp."""
    # Enqueue without the _enqueuer_id stamp by inserting directly.
    with store._lock:
        store._conn.execute(
            "INSERT INTO queue_items "
            "(item_id, lane, kind, principal, payload, latency_class, status, "
            " attempts, created_at) "
            "VALUES ('nostamp', 'deliberation', 'gw-admission', 'p', '{}', "
            "'batch', 'pending', 0, ?)",
            ((datetime.now(timezone.utc) - timedelta(
                seconds=GW_ADMISSION_MAX_WAIT_SEC + 10
            )).isoformat(),),
        )
        store._conn.commit()

    result = store.reap()

    item = store.get("nostamp")
    assert item["status"] == "failed"
    prov = item["provenance"] or {}
    assert prov.get("orphan_reap") == "presumed_dead_backstop"


def test_presumed_dead_backstop_live_pid(store):
    """Ticket older than max_wait is reaped even if the enqueuer pid appears alive."""
    live_pid = os.getpid()
    live_start = _get_process_start_time(live_pid) or 100.0

    ticket = _enqueue_gw_admission(store, pid=live_pid, start_time=live_start)
    _age_ticket(store, ticket, GW_ADMISSION_MAX_WAIT_SEC + 10)

    # Even though pid appears alive, the backstop fires unconditionally.
    with patch("agents_core.elevator.os.kill", return_value=None), \
         patch("agents_core.elevator._get_process_start_time", return_value=live_start):
        result = store.reap()

    item = store.get(ticket)
    assert item["status"] == "failed"
    prov = item["provenance"] or {}
    assert prov.get("orphan_reap") == "presumed_dead_backstop"


# ---------------------------------------------------------------------------
# AC2 — wedge regression test
# ---------------------------------------------------------------------------

def test_dead_orphan_wedge_cleared_live_waiter_admits(store):
    """AC2 (D2): dead orphan at FIFO head is squeezed past inline; live waiter admits in same call.

    With liveness-aware admission (gw-admission-liveness-aware-admission-v0), try_admit
    itself fails the dead pending head and admits the live waiter — no reaper cycle needed.
    """
    dead_pid = 99999999

    # Dead-enqueuer orphan is head-of-line (enqueued first).
    orphan = _enqueue_gw_admission(store, principal="dead-voice", pid=dead_pid, start_time=1.0)
    time.sleep(0.01)
    # Live waiter behind it.
    waiter = _enqueue_gw_admission(store, principal="live-council", pid=os.getpid(),
                                   start_time=_get_process_start_time(os.getpid()) or 100.0)

    _age_ticket(store, orphan, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    # D2: single try_admit on the live waiter succeeds — dead head is failed inline,
    # live waiter is admitted in the same call (AC2 regression gate).
    admitted, _ = store.try_admit(waiter, "deliberation", "live-council")
    assert admitted is True, "live waiter should admit via squeeze-past (dead head failed inline)"
    assert store.get(orphan)["status"] == "failed", "dead head should be failed inline"
    store.ack(waiter)


# ---------------------------------------------------------------------------
# AC3 — in-process leak: enforce loop abort leaves no pending ticket
# ---------------------------------------------------------------------------

def test_enforce_loop_abort_fails_ticket(tmp_path, monkeypatch):
    """An exception escaping the enforce while-loop triggers the finally cleanup."""
    from agents_core.llm import call_operator

    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")
    monkeypatch.setenv("GW_ADMISSION_MAX_WAIT_SEC", "60")

    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    # _call_gravitywell_backend raises after lease acquisition to trigger the finally abort path.
    # (drain_count is no longer called; the atomic acquire path replaced the two-step check.)
    fake_client = MagicMock()
    fake_client.acquire.return_value = {"status": "serving", "drain_cleared": True}
    fake_client.release = MagicMock()
    fake_client.close = MagicMock()
    fake_dc = MagicMock(return_value=fake_client)
    fake_dc.is_deferred = lambda r: r.get("status") == "deferred"
    fake_dc.is_contended = lambda r: bool(r.get("contended"))

    # IS_MASTER is lazy-imported from agents_core.elevator inside call_operator.
    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", fake_dc), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=RuntimeError("probe exploded")):
        with pytest.raises(RuntimeError, match="probe exploded"):
            call_operator("gravitywell", "test prompt",
                          principal="test-principal",
                          on_wake_fail="error")

    # Open a fresh store on the same DB to check the ticket status.
    check_store = ElevatorStore(db_path=db_path)
    try:
        with check_store._lock:
            rows = check_store._conn.execute(
                "SELECT item_id, status FROM queue_items WHERE kind='gw-admission'"
            ).fetchall()
    finally:
        check_store.close()

    assert rows, "ticket was never enqueued"
    for row in rows:
        assert row["status"] in ("failed", "expired"), \
            f"expected failed/expired after loop abort, got {row['status']} for {row['item_id']}"


# ---------------------------------------------------------------------------
# AC4 — parent-kill cleanup via fail_pending_by_pid
# ---------------------------------------------------------------------------

def test_fail_pending_by_pid_matches_and_fails(store):
    """fail_pending_by_pid() fails tickets stamped with the given host+pid."""
    target_pid = 77777
    other_pid = 88888

    t_target = _enqueue_gw_admission(store, principal="voice-a", pid=target_pid, start_time=1.0)
    t_other = _enqueue_gw_admission(store, principal="voice-b", pid=other_pid, start_time=1.0)
    _age_ticket(store, t_target, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)
    _age_ticket(store, t_other, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)

    count = store.fail_pending_by_pid(HOSTNAME, target_pid)

    assert count == 1
    assert store.get(t_target)["status"] == "failed"
    assert store.get(t_other)["status"] == "pending"  # untouched


def test_fail_pending_by_pid_never_raises(store):
    """fail_pending_by_pid() is best-effort and never raises even on internal error."""
    # Pass a nonsensical host — should silently return 0, not raise.
    result = store.fail_pending_by_pid("no-such-host-ever", 0)
    assert result == 0


def test_fail_pending_by_pid_cross_host_ignored(store):
    """Tickets stamped with a different host are not failed (cross-host)."""
    t = _enqueue_gw_admission(store, principal="voice-c", host="other-box",
                               pid=1234, start_time=1.0)
    _age_ticket(store, t, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)

    count = store.fail_pending_by_pid(HOSTNAME, 1234)
    assert count == 0
    assert store.get(t)["status"] == "pending"


# ---------------------------------------------------------------------------
# AC5 — GW_ADMISSION_MODE=off: zero elevator interaction
# ---------------------------------------------------------------------------

def test_off_mode_no_elevator(monkeypatch):
    """GW_ADMISSION_MODE=off takes the direct path; ElevatorStore is never instantiated."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "off")

    fake_client = MagicMock()
    fake_client.acquire.return_value = {"status": "serving"}
    fake_client.release = MagicMock()
    fake_client.close = MagicMock()
    fake_dc = MagicMock(return_value=fake_client)
    fake_dc.is_deferred = lambda r: r.get("status") == "deferred"
    fake_dc.is_contended = lambda r: bool(r.get("contended"))

    elevator_instantiated = []

    real_es = ElevatorStore

    def spy_es(*a, **kw):
        elevator_instantiated.append(True)
        return real_es(*a, **kw)

    # DoormanClient and ElevatorStore are lazy-imported from their source modules
    # inside call_operator; patch at the source, not agents_core.llm.
    with patch("agents_core.doorman_client.DoormanClient", fake_dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="pong"), \
         patch("agents_core.elevator.ElevatorStore", spy_es):
        from agents_core.llm import call_operator
        result = call_operator("gravitywell", "ping", on_wake_fail="skip")

    assert result == "pong"
    assert not elevator_instantiated, "off mode must not touch ElevatorStore"


# ---------------------------------------------------------------------------
# AC6 — reclaim_stale semantics unchanged
# ---------------------------------------------------------------------------

def test_reclaim_stale_only_claimed(store):
    """reclaim_stale() acts only on claimed tickets; pending orphans are untouched by it."""
    dead_pid = 99999999
    orphan = _enqueue_gw_admission(store, pid=dead_pid, start_time=1.0)
    _age_ticket(store, orphan, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

    # reclaim_stale should return 0 (no claimed items expired).
    reclaimed = store.reclaim_stale("deliberation")
    assert reclaimed == 0

    # Orphan must still be pending (reclaim_stale did not touch it).
    assert store.get(orphan)["status"] == "pending"


def test_enqueuer_id_stamped_in_payload(store):
    """enqueue() injects _enqueuer_id into the stored payload."""
    fake_start = 42.5
    with patch("agents_core.elevator._get_process_start_time", return_value=fake_start):
        tid = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": "w1"},
            principal="p", latency_class="batch",
        )
    item = store.get(tid)
    eid = item["payload"]["_enqueuer_id"]
    assert eid["host"] == HOSTNAME
    assert eid["pid"] == os.getpid()
    assert eid["start_time"] == fake_start
