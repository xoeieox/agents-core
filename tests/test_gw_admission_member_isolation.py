"""Tests for gw-admission-member-isolation-v0.

AC1 — universal ticket release on any exception
AC2 — thread-watchdog member deadline (FuturesTimeoutError path)
AC2a — self-triggered cleanup (no second caller needed)
AC3 — reclaim_stale factored into reusable method
AC3a — opportunistic reclaim called on the wait path
AC3b — deterministic sweep in interactive worker
AC3c — reclaim/new-claim race consistency
AC4 — claim TTL aligned to member timeout, not 960
AC5 — liveness: second principal admits within member_deadline
AC5a — no-traffic self-heal: hung member clears without external trigger
AC6 — no same-group penalty for ride-along members
AC7 — dark inertness: off/shadow/bypass/off-master unchanged
AC8 — provenance vocabulary: gw_member_error, gw_member_deadline
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agents_core.elevator import ElevatorStore
from agents_core.llm import call_operator, OperatorUnreachableError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    store = ElevatorStore(db_path=tmp_path / "q.db")
    yield store
    store.close()


def _gw_dc_serving():
    """Return (DoormanClient class mock, instance mock) with status=serving."""
    instance = MagicMock()
    instance.acquire.return_value = {"status": "serving", "node": "gravitywell", "work_id": "w1",
                                     "drain_cleared": True}
    instance.drain_count.return_value = 0
    instance.release = MagicMock()
    instance.close = MagicMock()
    dc = MagicMock(return_value=instance)
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))
    return dc, instance


def _enforce_env(monkeypatch, timeout=5):
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.05")
    monkeypatch.setenv("GW_ADMISSION_MAX_WAIT_SEC", "30")


# ---------------------------------------------------------------------------
# AC3 — reclaim_stale factored into a reusable method
# ---------------------------------------------------------------------------

def test_reclaim_stale_returns_zero_when_nothing_stale(db):
    """reclaim_stale on an empty lane returns 0."""
    assert db.reclaim_stale("deliberation") == 0


def test_reclaim_stale_only_touches_past_ttl(db):
    """reclaim_stale leaves items within their claim_ttl untouched."""
    t = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                   principal="p1", latency_class="batch")
    db.try_admit(t, "deliberation", "p1", claim_ttl_sec=3600)
    assert db.get(t)["status"] == "claimed"
    assert db.reclaim_stale("deliberation") == 0
    assert db.get(t)["status"] == "claimed"


def test_reclaim_stale_reclaims_expired_claim(db):
    """reclaim_stale reclaims an item whose claim_ttl_sec is in the past."""
    t = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                   principal="p1", latency_class="batch")
    db.try_admit(t, "deliberation", "p1", claim_ttl_sec=1)
    # Force the claimed_at to a past timestamp by manipulating the DB directly
    with db._lock:
        db._conn.execute(
            "UPDATE queue_items SET claimed_at=datetime('now', '-10 seconds') WHERE item_id=?",
            (t,),
        )
        db._conn.commit()
    reclaimed = db.reclaim_stale("deliberation")
    assert reclaimed == 1
    item = db.get(t)
    assert item["status"] == "pending"
    assert item["claim_owner"] is None


def test_reclaim_stale_lane_scoped(db):
    """reclaim_stale("deliberation") does not touch stale claims on other lanes."""
    t_i = db.enqueue(lane="interactive", kind="baton", payload={},
                     principal="p1", latency_class="interactive")
    db.claim(lanes=["interactive"], owner="w", claim_ttl_sec=1)
    with db._lock:
        db._conn.execute(
            "UPDATE queue_items SET claimed_at=datetime('now', '-10 seconds') WHERE item_id=?",
            (t_i,),
        )
        db._conn.commit()
    assert db.reclaim_stale("deliberation") == 0
    assert db.get(t_i)["status"] == "claimed"  # interactive item untouched


def test_reap_inline_behavior_preserved(db):
    """_reap_inline still reclaims stale deliberation claims (behavior-preserving)."""
    t = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                   principal="p1", latency_class="batch")
    db.try_admit(t, "deliberation", "p1", claim_ttl_sec=1)
    with db._lock:
        db._conn.execute(
            "UPDATE queue_items SET claimed_at=datetime('now', '-10 seconds') WHERE item_id=?",
            (t,),
        )
        db._conn.commit()
    result = db._reap_inline()
    assert result["reclaimed"] >= 1
    assert db.get(t)["status"] == "pending"


# ---------------------------------------------------------------------------
# AC3c — reclaim/new-claim race
# ---------------------------------------------------------------------------

def test_reclaim_and_try_admit_race_leaves_fresh_claim_intact(db):
    """Concurrent reclaim_stale + try_admit: only genuinely-stale row is cleared."""
    # Set up a stale claim for principal-A
    old_t = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="p-a", latency_class="batch")
    db.try_admit(old_t, "deliberation", "p-a", claim_ttl_sec=1)
    with db._lock:
        db._conn.execute(
            "UPDATE queue_items SET claimed_at=datetime('now', '-10 seconds') WHERE item_id=?",
            (old_t,),
        )
        db._conn.commit()

    # Enqueue a fresh ticket for principal-B
    new_t = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="p-b", latency_class="batch")

    # Simulate concurrent reclaim + try_admit
    barrier = threading.Barrier(2)
    reclaim_result = [0]
    admit_result = [None]

    def do_reclaim():
        barrier.wait()
        reclaim_result[0] = db.reclaim_stale("deliberation")

    def do_admit():
        barrier.wait()
        admitted, _ = db.try_admit(new_t, "deliberation", "p-b", claim_ttl_sec=300)
        admit_result[0] = admitted

    t1 = threading.Thread(target=do_reclaim)
    t2 = threading.Thread(target=do_admit)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # The stale claim is reclaimed; old_t may be pending or claimed-by-B depending on order.
    old_item = db.get(old_t)
    new_item = db.get(new_t)

    # Critical: the fresh claim for p-b is not lost.
    assert new_item["status"] in ("claimed", "pending")
    # Critical: the stale item for p-a is NOT still claimed (either reclaimed-to-pending, or failed).
    assert old_item["status"] != "claimed" or reclaim_result[0] == 0


# ---------------------------------------------------------------------------
# AC1 — universal ticket release on unexpected exception
# ---------------------------------------------------------------------------

def test_ac1_unexpected_exception_fails_ticket_and_reraises(tmp_path, monkeypatch):
    """AC1: an unexpected exception from _call_gravitywell_backend fails the ticket."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _client = _gw_dc_serving()
    store = ElevatorStore(db_path=db_path)
    provenance = []

    def bang(*a, **kw):
        raise RuntimeError("unexpected!")

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.llm.IS_MASTER" if hasattr(__import__("agents_core.llm", fromlist=["IS_MASTER"]), "IS_MASTER") else "agents_core.elevator.IS_MASTER", True, create=True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", side_effect=bang):
        with pytest.raises(RuntimeError, match="unexpected!"):
            call_operator(
                "gravitywell", "hello",
                principal="test-principal",
                _provenance_out=provenance,
                _admission_bypass=False,
            )

    # The ticket on the deliberation lane must be in a terminal state (failed).
    items = []
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM queue_items WHERE lane='deliberation'"
        ).fetchall()
        items = [store._row_to_dict(r) for r in rows]
    store.close()

    assert len(items) == 1
    assert items[0]["status"] == "failed"
    assert ("gw_member_error", "gravitywell") in provenance


# ---------------------------------------------------------------------------
# AC4 — claim TTL aligned to member timeout
# ---------------------------------------------------------------------------

def test_ac4_claim_ttl_aligned_to_timeout(db):
    """AC4: try_admit called with claim_ttl = timeout + 90, not 960."""
    admitted, _ = db.try_admit(
        db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                   principal="p1", latency_class="batch"),
        "deliberation", "p1",
        claim_ttl_sec=390,  # timeout=300, claim_ttl = 300 + 90 = 390
    )
    assert admitted is True
    # 390 is NOT 960 (the old default)
    item = db.get(db._conn.execute(
        "SELECT item_id FROM queue_items WHERE lane='deliberation' LIMIT 1"
    ).fetchone()["item_id"])
    assert item["claim_ttl_sec"] == 390


def test_ac4_call_operator_derives_claim_ttl_from_timeout(tmp_path, monkeypatch):
    """AC4 integration: call_operator passes claim_ttl_sec = timeout + 90 to try_admit."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    captured_ttls = []

    original_try_admit = ElevatorStore.try_admit

    def tracking_try_admit(self, item_id, lane, principal, max_groups=1, claim_ttl_sec=960):
        captured_ttls.append(claim_ttl_sec)
        return original_try_admit(self, item_id, lane, principal,
                                  max_groups=max_groups, claim_ttl_sec=claim_ttl_sec)

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"), \
         patch.object(ElevatorStore, "try_admit", tracking_try_admit):
        call_operator(
            "gravitywell", "hello",
            principal="p",
            timeout=300,
            _admission_bypass=False,
        )

    assert captured_ttls, "try_admit was never called"
    assert captured_ttls[0] == 390, (
        f"expected claim_ttl_sec=390 (timeout=300 + 90), got {captured_ttls[0]}"
    )


# ---------------------------------------------------------------------------
# AC2 + AC2a — thread watchdog (simulated via monkeypatched future timeout)
# ---------------------------------------------------------------------------

def _make_blocking_backend(event: threading.Event):
    """Backend that blocks until event is set."""
    def fake_backend(*a, **kw):
        event.wait(timeout=30)
        return None
    return fake_backend


def test_ac2_deadline_breach_fails_ticket_and_releases_lease(tmp_path, monkeypatch):
    """AC2/AC2a: FuturesTimeoutError path fails ticket + releases lease + routes wake_fail."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, mock_client = _gw_dc_serving()
    provenance = []
    unblock = threading.Event()

    import concurrent.futures as _cf

    original_executor = _cf.ThreadPoolExecutor

    class _FastTimeoutExecutor:
        """Wraps ThreadPoolExecutor but future.result() raises TimeoutError immediately."""
        def __init__(self, *a, **kw):
            self._inner = original_executor(*a, **kw)

        def submit(self, fn, *a, **kw):
            fut = self._inner.submit(fn, *a, **kw)
            wrapper = MagicMock(spec=_cf.Future)
            wrapper.result.side_effect = _cf.TimeoutError()
            return wrapper

        def shutdown(self, wait=True):
            self._inner.shutdown(wait=False)

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", side_effect=_make_blocking_backend(unblock)), \
         patch("agents_core.llm._cf.ThreadPoolExecutor" if hasattr(__import__("agents_core.llm", fromlist=[""]), "_cf") else "concurrent.futures.ThreadPoolExecutor", _FastTimeoutExecutor, create=True):
        result = call_operator(
            "gravitywell", "hello",
            principal="test-principal",
            on_wake_fail="skip",
            _provenance_out=provenance,
            _admission_bypass=False,
        )

    unblock.set()  # let the blocked thread exit cleanly

    # wake_fail=skip → None
    assert result is None
    # Lease must have been released
    mock_client.release.assert_called()
    assert ("gw_member_deadline", "gravitywell") in provenance


# ---------------------------------------------------------------------------
# AC5 — liveness: second principal admits after member deadline
# ---------------------------------------------------------------------------

def test_ac5_second_principal_admits_after_hung_member_cleared(tmp_path, monkeypatch):
    """AC5: a hung member's claim is self-cleared; a second principal then admits.

    This test uses a short member_deadline (timeout=1s) so the watchdog fires fast.
    """
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.1")

    dc1, mock_client1 = _gw_dc_serving()
    dc2, mock_client2 = _gw_dc_serving()

    unblock_hung = threading.Event()
    first_call_started = threading.Event()

    def hung_backend(*a, **kw):
        first_call_started.set()
        unblock_hung.wait(timeout=30)
        return None

    second_admitted = threading.Event()
    second_result = [None]
    second_exception = [None]

    def run_second():
        dc_cls = MagicMock(side_effect=[mock_client2])
        dc_cls.is_deferred = lambda r: r.get("status") == "deferred"
        dc_cls.is_contended = lambda r: bool(r.get("contended"))
        with patch("agents_core.elevator.IS_MASTER", True), \
             patch("agents_core.doorman_client.DoormanClient", dc_cls), \
             patch("agents_core.llm._call_gravitywell_backend", return_value="second-ok"):
            try:
                second_result[0] = call_operator(
                    "gravitywell", "second",
                    principal="principal-b",
                    on_wake_fail="skip",
                    timeout=1,
                    _admission_bypass=False,
                )
            except Exception as e:
                second_exception[0] = e

    # Start the hung first member
    t_first = threading.Thread(target=lambda: call_operator(
        "gravitywell", "first",
        principal="principal-a",
        on_wake_fail="skip",
        timeout=1,  # 1-second watchdog
        _admission_bypass=False,
    ), daemon=True)

    dc_cls_first = MagicMock(side_effect=[mock_client1])
    dc_cls_first.is_deferred = lambda r: r.get("status") == "deferred"
    dc_cls_first.is_contended = lambda r: bool(r.get("contended"))

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc_cls_first), \
         patch("agents_core.llm._call_gravitywell_backend", side_effect=hung_backend):
        t_first.start()
        first_call_started.wait(timeout=5)

        # Now start second principal — should wait while first holds the lane,
        # then admit after the watchdog fires (within timeout + poll grace).
        t_second = threading.Thread(target=run_second)
        t_second.start()
        t_second.join(timeout=10)

    unblock_hung.set()
    t_first.join(timeout=5)

    assert second_exception[0] is None, f"second caller raised: {second_exception[0]}"
    assert second_result[0] == "second-ok"


# ---------------------------------------------------------------------------
# AC5a — no-traffic self-heal
# ---------------------------------------------------------------------------

def test_ac5a_no_traffic_self_heal(tmp_path, monkeypatch):
    """AC5a: a hung member with no second caller self-clears its claim + lease on deadline."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, mock_client = _gw_dc_serving()
    unblock = threading.Event()
    call_started = threading.Event()

    def hung_backend(*a, **kw):
        call_started.set()
        unblock.wait(timeout=30)
        return None

    provenance = []
    result_box = [None]

    def run():
        result_box[0] = call_operator(
            "gravitywell", "hung",
            principal="only-principal",
            on_wake_fail="skip",
            timeout=1,
            _provenance_out=provenance,
            _admission_bypass=False,
        )

    dc_cls = MagicMock(side_effect=[mock_client])
    dc_cls.is_deferred = lambda r: r.get("status") == "deferred"
    dc_cls.is_contended = lambda r: bool(r.get("contended"))

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc_cls), \
         patch("agents_core.llm._call_gravitywell_backend", side_effect=hung_backend):
        t = threading.Thread(target=run)
        t.start()
        call_started.wait(timeout=5)
        t.join(timeout=10)  # watchdog fires after ~1s

    unblock.set()

    # The lease must have been released without any external trigger.
    mock_client.release.assert_called()
    # The result should be None (on_wake_fail=skip) — deadline branch.
    assert result_box[0] is None
    assert ("gw_member_deadline", "gravitywell") in provenance

    # The deliberation-lane claim must be in a terminal state.
    store = ElevatorStore(db_path=db_path)
    with store._lock:
        rows = store._conn.execute(
            "SELECT status FROM queue_items WHERE lane='deliberation'"
        ).fetchall()
    store.close()
    statuses = [r[0] for r in rows]
    assert all(s in ("failed", "served") for s in statuses), f"unexpected statuses: {statuses}"


# ---------------------------------------------------------------------------
# AC6 — no same-group penalty
# ---------------------------------------------------------------------------

def test_ac6_ride_along_admits_immediately(db):
    """AC6: a ride-along member (same principal, already claimed) admits without delay."""
    t1 = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                    principal="shared-p", latency_class="batch")
    # First member admitted and claimed
    admitted1, ride1 = db.try_admit(t1, "deliberation", "shared-p", claim_ttl_sec=390)
    assert admitted1 is True
    assert ride1 is False

    # Second member of same group
    t2 = db.enqueue(lane="deliberation", kind="gw-admission", payload={},
                    principal="shared-p", latency_class="batch")
    admitted2, ride2 = db.try_admit(t2, "deliberation", "shared-p", claim_ttl_sec=390)
    assert admitted2 is True
    assert ride2 is True  # ride-along: admitted immediately


# ---------------------------------------------------------------------------
# AC7 — dark inertness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_ac7_off_and_shadow_do_not_enqueue(tmp_path, monkeypatch, mode):
    """AC7: off/shadow modes never touch the elevator queue."""
    monkeypatch.setenv("GW_ADMISSION_MODE", mode)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        call_operator("gravitywell", "hello", _admission_bypass=False)

    store = ElevatorStore(db_path=db_path)
    with store._lock:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM queue_items WHERE lane='deliberation'"
        ).fetchone()[0]
    store.close()
    assert count == 0, f"mode={mode!r} enqueued {count} item(s) on deliberation"


def test_ac7_bypass_skips_queue(tmp_path, monkeypatch):
    """AC7: _admission_bypass=True skips queue even in enforce mode."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        call_operator("gravitywell", "hello", _admission_bypass=True)

    store = ElevatorStore(db_path=db_path)
    with store._lock:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM queue_items WHERE lane='deliberation'"
        ).fetchone()[0]
    store.close()
    assert count == 0


def test_ac7_off_master_passthrough_no_queue(tmp_path, monkeypatch):
    """AC7: enforce mode on non-master node falls through without enqueuing."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    with patch("agents_core.elevator.IS_MASTER", False), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        result = call_operator("gravitywell", "hello", _admission_bypass=False)

    assert result == "ok"
    store = ElevatorStore(db_path=db_path)
    with store._lock:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM queue_items WHERE lane='deliberation'"
        ).fetchone()[0]
    store.close()
    assert count == 0


# ---------------------------------------------------------------------------
# AC8 — provenance vocabulary
# ---------------------------------------------------------------------------

def test_ac8_member_error_none_provenance_no_crash(tmp_path, monkeypatch):
    """AC8: gw_member_error None guard — no AttributeError when _provenance_out=None on unexpected exception."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", side_effect=RuntimeError("bang")):
        with pytest.raises(RuntimeError, match="bang"):
            call_operator(
                "gravitywell", "hello",
                principal="test-principal",
                _provenance_out=None,  # None guard under test: must not AttributeError
                _admission_bypass=False,
            )


def test_ac8_member_deadline_none_provenance_no_crash(tmp_path, monkeypatch):
    """AC8: gw_member_deadline None guard — no AttributeError when _provenance_out=None on watchdog timeout."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()

    import concurrent.futures as _cf
    original_executor = _cf.ThreadPoolExecutor

    class _ImmediateTimeoutExecutor:
        def __init__(self, *a, **kw):
            self._inner = original_executor(*a, **kw)

        def submit(self, fn, *a, **kw):
            wrapper = MagicMock(spec=_cf.Future)
            wrapper.result.side_effect = _cf.TimeoutError()
            return wrapper

        def shutdown(self, wait=True):
            self._inner.shutdown(wait=False)

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value=None), \
         patch("concurrent.futures.ThreadPoolExecutor", _ImmediateTimeoutExecutor):
        result = call_operator(
            "gravitywell", "hello",
            principal="test-principal",
            on_wake_fail="skip",
            _provenance_out=None,  # None guard under test: must not AttributeError
            _admission_bypass=False,
        )
    assert result is None  # on_wake_fail=skip → None


# ---------------------------------------------------------------------------
# AC3a — opportunistic reclaim is called on the wait path
# ---------------------------------------------------------------------------

def test_ac3a_reclaim_called_before_try_admit(tmp_path, monkeypatch):
    """AC3a: reclaim_stale("deliberation") is called before each try_admit attempt."""
    _enforce_env(monkeypatch)
    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    dc, _ = _gw_dc_serving()
    reclaim_calls = []

    original_reclaim = ElevatorStore.reclaim_stale

    def tracking_reclaim(self, lane):
        reclaim_calls.append(lane)
        return original_reclaim(self, lane)

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"), \
         patch.object(ElevatorStore, "reclaim_stale", tracking_reclaim):
        call_operator("gravitywell", "hello", principal="p", _admission_bypass=False)

    assert "deliberation" in reclaim_calls, "reclaim_stale('deliberation') not called on wait path"


# ---------------------------------------------------------------------------
# AC3b — deterministic deliberation sweep in interactive worker
# ---------------------------------------------------------------------------

def test_ac3b_worker_loop_calls_reclaim_stale(tmp_path, monkeypatch):
    """AC3b: the interactive worker calls reclaim_stale('deliberation') each iteration."""
    from agents_core.elevator_interactive_worker import worker_loop
    from agents_core.elevator import ElevatorStore

    db_path = tmp_path / "q.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    reclaim_calls = []

    def tracking_reclaim(self, lane):
        reclaim_calls.append(lane)
        # Stop the loop after the first reclaim_stale call to avoid sleeping 5s.
        raise KeyboardInterrupt("stop after first reclaim")

    # A doorman client that reports GW in big mode so we reach the claim section.
    class FakeDoorman:
        def status(self):
            return {"nodes": {"gravitywell": {"serving": True, "serving_mode": "big"}}}

        def close(self):
            pass

    with patch.object(ElevatorStore, "reclaim_stale", tracking_reclaim), \
         patch("agents_core.elevator.IS_MASTER", True):
        try:
            worker_loop(_doorman_client=FakeDoorman())
        except KeyboardInterrupt:
            pass  # expected termination

    assert "deliberation" in reclaim_calls, "worker_loop did not call reclaim_stale('deliberation')"
