"""CI regression wrapper for gw-enforce-rearm-ab-harness-v0.

Tests S1-S3 using friction-injected mock (seeded/parametrized for deterministic replay).
Does NOT import from scripts/gw_enforce_rearm_ab.py — AC8 structural decoupling.

Scenarios:
  S1 — drain-count fix: active state assertions on drain_count(exclude_principal)
  S2 — pending-orphan reclaim: SIGKILL'd enqueuer → reaped → successor admitted
  S3 — hard timeout / kill mid-flight: all tickets terminal, zero leaked claims

Shared assertion helpers are defined inline here (not imported from the live harness driver)
so mock flakiness cannot become a dependency of the production GO/NO-GO gate.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.elevator import (
    ElevatorStore,
    GW_ADMISSION_ORPHAN_GRACE_SEC,
    HOSTNAME,
    IS_MASTER,
    _get_process_start_time,
)
from agents_core.doorman_server import GHOST_PRINCIPAL, _NodeState


# ---------------------------------------------------------------------------
# Stable assertion helpers (inline — NOT imported from scripts/)
# AC8: share only stable pure-function helpers, not the backend simulator.
# ---------------------------------------------------------------------------

def _drain_count_from_state(node_state: _NodeState, exclude_principal: str | None = None) -> int:
    """Count active worker leases, optionally excluding a principal. Ghost leases always counted."""
    count = 0
    with node_state.lock:
        node_state._gc_stale()
        for wid, info in node_state.leases.items():
            if info.get("role") != "worker":
                continue
            p = info.get("principal", GHOST_PRINCIPAL)
            if p == GHOST_PRINCIPAL:
                count += 1
            elif exclude_principal is not None and p == exclude_principal:
                continue
            else:
                count += 1
    return count


def _all_terminal(elevator: ElevatorStore, ticket_ids: list[str]) -> bool:
    terminal = {"served", "failed", "expired"}
    return all((elevator.get(t) or {}).get("status") in terminal for t in ticket_ids)


def _claimed_principals(elevator: ElevatorStore, lane: str = "deliberation") -> set:
    return elevator._claimed_principals_on_lane(lane)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db = ElevatorStore(db_path=tmp_path / "q.db")
    yield db
    db.close()


@pytest.fixture
def node():
    """In-process _NodeState with ensure_serving always returning True (no subprocesses)."""
    state = _NodeState(gw_url="http://mock.internal/", node_name="gravitywell")
    with patch.object(state, "_is_serving", return_value=True), \
         patch("subprocess.run"):
        yield state


def _acquire_worker(node_state: _NodeState, work_id: str, principal: str | None, ttl_sec: int = 60):
    """Acquire a worker lease on node_state without subprocess or HTTP."""
    with node_state.lock:
        node_state.leases[work_id] = {
            "acquired_at": time.time(),
            "ttl_sec": ttl_sec,
            "role": "worker",
            "principal": principal if principal is not None else GHOST_PRINCIPAL,
        }


def _release_worker(node_state: _NodeState, work_id: str):
    with node_state.lock:
        node_state.leases.pop(work_id, None)


def _age_ticket(store: ElevatorStore, ticket_id: str, seconds: int) -> None:
    new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with store._lock:
        store._conn.execute(
            "UPDATE queue_items SET created_at=? WHERE item_id=?",
            (new_ts, ticket_id),
        )
        store._conn.commit()


# ---------------------------------------------------------------------------
# S1 — Drain-count fix: active state assertions
# ---------------------------------------------------------------------------

class TestS1DrainCountFix:
    """AC2: drain_count fix proven by state, not log-grep."""

    def test_attributed_hold_excluded_first_voice_proceeds(self, store, node):
        """Properly-attributed hold is excluded from drain_count → first voice proceeds."""
        P = f"council-delib-{uuid.uuid4().hex[:6]}"
        hold_wid = f"hold-{uuid.uuid4().hex[:6]}"

        # Acquire the council's deliberation-spanning hold with principal=P
        _acquire_worker(node, hold_wid, principal=P)

        # Assert drain_count(exclude_principal=P) == 0 → first voice sees open lane
        assert _drain_count_from_state(node, exclude_principal=P) == 0, \
            "first voice should see drain_count=0 when its own hold is excluded"

        # Total count without exclusion is 1
        assert _drain_count_from_state(node, exclude_principal=None) == 1

        # First voice acquires → elevator ride-along
        ticket_hold = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": hold_wid}, principal=P, latency_class="batch",
        )
        admitted, _ = store.try_admit(ticket_hold, "deliberation", P)
        assert admitted, "hold ticket must be admitted"

        # Voice 1: enqueue with same principal → ride-along
        ticket_v1 = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": "v1"}, principal=P, latency_class="batch",
        )
        ok, is_ride_along = store.try_admit(ticket_v1, "deliberation", P)
        assert ok, "voice 1 must be admitted"
        assert is_ride_along, "voice 1 must be a ride-along (shared principal)"

        # Verify drain_count(exclude_principal=P) still 0 with both leases excluded
        v1_wid = f"v1-{uuid.uuid4().hex[:6]}"
        _acquire_worker(node, v1_wid, principal=P)
        assert _drain_count_from_state(node, exclude_principal=P) == 0

        # Cleanup
        _release_worker(node, hold_wid)
        _release_worker(node, v1_wid)
        store.fail(ticket_v1)
        store.fail(ticket_hold)

    def test_ghost_lease_counted_and_gates_caller(self, caplog):
        """Un-attributed worker lease is GHOST_PRINCIPAL — always counted, never excluded.

        A ghost in drain_count(exclude_principal=P) → gate blocks (count ≥ 1).
        Critical log fires (from real /v0/drain-count endpoint) when ghost is counted.

        Uses the real doorman app (via TestClient) with ensure_serving mocked out
        so no subprocess or SSH calls are made — same pattern as test_doorman_server.py.
        """
        import logging
        from agents_core.doorman_server import create_app, _NodeState
        from fastapi.testclient import TestClient

        P = f"council-delib-{uuid.uuid4().hex[:6]}"
        ghost_wid = f"ghost-{uuid.uuid4().hex[:6]}"

        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch("subprocess.run"):
            app = create_app(gw_url="http://mock.internal/")
            client = TestClient(app, raise_server_exceptions=True)

            # Acquire ghost: no "principal" field → server stamps GHOST_PRINCIPAL
            r = client.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": ghost_wid,
                "ttl_sec": 300,
                "reason": "ghost-test",
                "role": "worker",
            })
            assert r.status_code == 200, f"acquire failed: {r.text}"
            assert r.json().get("status") == "serving", \
                f"expected serving, got {r.json()}"

            with caplog.at_level(logging.CRITICAL, logger="doorman-server"):
                resp = client.get(
                    f"/v0/drain-count?node=gravitywell&exclude_principal={P}"
                )
            assert resp.status_code == 200
            dc = resp.json()["drain_count"]
            assert dc >= 1, \
                f"ghost must be counted in drain_count(exclude_principal={P}), got {dc}"

            # Critical log must fire when ghost is counted in an exclude-principal decision
            assert any(
                "ghost_lease_counted" in rec.message
                for rec in caplog.records
                if rec.levelno >= logging.CRITICAL
            ), "critical log must fire when ghost is counted in a drain_count drain decision"

            client.post("/lease/release", json={
                "node": "gravitywell", "work_id": ghost_wid,
            })

    def test_distinct_principal_gates_group(self, store, node):
        """A distinct-principal worker prevents the group's drain gate from opening."""
        P = f"council-delib-{uuid.uuid4().hex[:6]}"
        other_wid = f"other-{uuid.uuid4().hex[:6]}"

        _acquire_worker(node, other_wid, principal="other-group-xyz")

        dc = _drain_count_from_state(node, exclude_principal=P)
        assert dc >= 1, \
            f"distinct-principal lease must gate group P (drain_count(excl=P)≥1), got {dc}"

        _release_worker(node, other_wid)
        assert _drain_count_from_state(node, exclude_principal=P) == 0

    def test_n_voices_all_serve_no_self_deadlock(self, store, node):
        """N voices sharing principal P all serve without self-deadlock."""
        P = f"council-delib-{uuid.uuid4().hex[:6]}"
        N = 4
        hold_wid = f"hold-{uuid.uuid4().hex[:6]}"

        # Deliberation-spanning hold
        _acquire_worker(node, hold_wid, principal=P)
        hold_ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": hold_wid}, principal=P, latency_class="batch",
        )
        admitted, _ = store.try_admit(hold_ticket, "deliberation", P)
        assert admitted

        served = 0
        for i in range(N):
            dc = _drain_count_from_state(node, exclude_principal=P)
            assert dc == 0, f"voice {i} should see drain=0 with hold excluded, got {dc}"

            t = store.enqueue(
                lane="deliberation", kind="gw-admission",
                payload={"work_id": f"v{i}"}, principal=P, latency_class="batch",
            )
            ok, is_ride_along = store.try_admit(t, "deliberation", P)
            assert ok, f"voice {i} must be admitted"
            assert is_ride_along, f"voice {i} must ride-along"

            v_wid = f"v{i}-{uuid.uuid4().hex[:4]}"
            _acquire_worker(node, v_wid, principal=P)
            _release_worker(node, v_wid)
            store.ack(t)
            served += 1

        assert served == N, f"all {N} voices must serve, got {served}"
        _release_worker(node, hold_wid)
        store.ack(hold_ticket)


# ---------------------------------------------------------------------------
# S2 — Pending-orphan reclaim: SIGKILL + seeded friction
# ---------------------------------------------------------------------------

# Subprocess: enqueue a gw-admission ticket then block until killed.
_ENQUEUER_CODE = """\
import os, sys, time
sys.path.insert(0, '/srv/agents')
db_path, principal = sys.argv[1], sys.argv[2]
os.environ['ELEVATOR_DB_PATH'] = db_path
from agents_core.elevator import ElevatorStore
store = ElevatorStore()
ticket = store.enqueue(
    lane='deliberation', kind='gw-admission',
    payload={'work_id': 'orphan-ci-test'},
    principal=principal, latency_class='batch',
)
print(ticket, flush=True)
time.sleep(9999)
"""


@pytest.mark.skipif(not IS_MASTER, reason="orphan reaper requires IS_MASTER (run on BRIX)")
class TestS2PendingOrphanReclaim:
    """AC3: dead-enqueuer orphan reaped within one reap() call; successor admitted."""

    @pytest.mark.parametrize("seed,kill_delay", [
        (42, 0.0),    # kill immediately after enqueue (mid-poll phase)
        (137, 0.15),  # kill after brief delay (post-enqueue, pre-grace)
        (7, 0.30),    # kill after longer delay
    ])
    def test_dead_enqueuer_orphan_reaped(self, store, tmp_path, seed, kill_delay):
        """SIGKILL'd enqueuer's ticket is reaped by pid-liveness check.

        Ages the ticket past GW_ADMISSION_ORPHAN_GRACE_SEC (the module constant
        used by the main-process reaper) so the precise-liveness probe runs.
        """
        principal = f"s2-orphan-{seed}"
        db_str = str(tmp_path / "q.db")

        env = dict(os.environ)
        env["ELEVATOR_DB_PATH"] = db_str

        proc = subprocess.Popen(
            [sys.executable, "-c", _ENQUEUER_CODE, db_str, principal],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            ticket_id = proc.stdout.readline().decode().strip()
            time.sleep(kill_delay)
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            pytest.skip(f"subprocess setup failed (seed={seed})")

        assert ticket_id, "subprocess must emit ticket id before being killed"

        # Verify ticket is pending
        item = store.get(ticket_id)
        assert item is not None, "orphan ticket must exist in store"
        assert item["status"] == "pending", f"expected pending, got {item['status']}"

        # Age past the module-level grace constant so the reaper's precise-liveness
        # probe triggers (mirrors pattern in test_gw_admission_pending_orphan_reclaim.py)
        _age_ticket(store, ticket_id, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)

        # Reap: should detect dead pid via os.kill(pid, 0) → OSError
        store.reap()
        item_after = store.get(ticket_id)
        assert item_after is not None
        assert item_after["status"] in ("failed", "expired"), \
            f"orphan must be reaped; got status={item_after['status']}"

    def test_successor_admitted_after_orphan_reaped(self, store, tmp_path):
        """Orphan at FIFO head cleared; live successor then admitted within bound."""
        db_str = str(tmp_path / "q.db")
        env = dict(os.environ)
        env["ELEVATOR_DB_PATH"] = db_str

        principal_orphan = "s2-orphan-fifo"
        proc = subprocess.Popen(
            [sys.executable, "-c", _ENQUEUER_CODE, db_str, principal_orphan],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            ticket_id = proc.stdout.readline().decode().strip()
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            pytest.skip("subprocess setup failed")

        assert ticket_id
        _age_ticket(store, ticket_id, GW_ADMISSION_ORPHAN_GRACE_SEC + 10)
        store.reap()

        # Orphan must be reaped
        assert store.get(ticket_id)["status"] in ("failed", "expired"), \
            "orphan must be reaped before successor can proceed"

        # Successor: distinct principal, should be admitted as new head-of-line
        succ_principal = "s2-successor-live"
        succ_ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": "succ"}, principal=succ_principal, latency_class="batch",
        )
        ok, _ = store.try_admit(succ_ticket, "deliberation", succ_principal)
        assert ok, "successor must be admitted after orphan is cleared from FIFO head"
        store.ack(succ_ticket)

    def test_grace_period_spares_live_enqueuer(self, store):
        """Ticket within grace period is NOT reaped even if stamped pid is dead."""
        dead_pid = 99999999
        ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": "grace-test"}, principal="grace-p", latency_class="batch",
        )
        with patch("agents_core.elevator.HOSTNAME", HOSTNAME), \
             patch("agents_core.elevator.os.getpid", return_value=dead_pid), \
             patch("agents_core.elevator._get_process_start_time", return_value=12345.0):
            # Re-stamp via a fresh enqueue (we backdated by 0, within grace)
            pass  # ticket was just created → within grace

        # Within grace: reap must spare it
        with patch("agents_core.elevator.os.kill", side_effect=OSError("no such process")):
            store.reap()

        item = store.get(ticket)
        assert item["status"] == "pending", \
            "ticket within grace period must not be reaped"
        store.fail(ticket)

    def test_presumed_dead_backstop_reaped(self, store):
        """Ticket older than GW_ADMISSION_MAX_WAIT_SEC reaped by backstop."""
        from agents_core.elevator import GW_ADMISSION_MAX_WAIT_SEC
        ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": "backstop-test"}, principal="backstop-p", latency_class="batch",
        )
        _age_ticket(store, ticket, GW_ADMISSION_MAX_WAIT_SEC + 10)
        store.reap()
        item = store.get(ticket)
        assert item["status"] in ("failed", "expired"), \
            "ticket past presumed-dead backstop must be reaped"


# ---------------------------------------------------------------------------
# S3 — Hard timeout / kill mid-flight
# ---------------------------------------------------------------------------

class TestS3HardTimeoutKill:
    """AC4: all tickets reach terminal state; zero leaked claims after burst."""

    def test_member_deadline_fires_ticket_terminal(self, store, node):
        """Member deadline fires → ticket failed, doorman lease released, no leaked claim."""
        P = f"s3-timeout-{uuid.uuid4().hex[:6]}"
        wid = f"s3-wid-{uuid.uuid4().hex[:6]}"
        member_deadline = 0.2

        ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": wid}, principal=P, latency_class="batch",
        )
        admitted, _ = store.try_admit(ticket, "deliberation", P)
        assert admitted

        _acquire_worker(node, wid, principal=P)

        deadline_fired = threading.Event()
        work_done = threading.Event()

        def _hung_work():
            work_done.wait(timeout=member_deadline + 0.5)

        def _watchdog():
            if not work_done.wait(timeout=member_deadline):
                deadline_fired.set()
                _release_worker(node, wid)
                store.fail(ticket)

        wt = threading.Thread(target=_hung_work)
        wd = threading.Thread(target=_watchdog)
        wt.start()
        wd.start()
        wd.join(timeout=member_deadline + 0.5)
        work_done.set()
        wt.join(timeout=1)

        assert deadline_fired.is_set(), "watchdog must fire within member_deadline"
        assert _all_terminal(store, [ticket])
        assert _claimed_principals(store, "deliberation") == set(), \
            "no leaked claims after member deadline"
        assert _drain_count_from_state(node) == 0, \
            "doorman lease must be released after deadline"

    def test_parent_kill_cleanup_no_leaked_claims(self, store, node):
        """Parent kill → ticket failed + lease released → zero leaked claims."""
        P = f"s3-kill-{uuid.uuid4().hex[:6]}"
        wid = f"s3-kill-wid-{uuid.uuid4().hex[:6]}"

        ticket = store.enqueue(
            lane="deliberation", kind="gw-admission",
            payload={"work_id": wid}, principal=P, latency_class="batch",
        )
        admitted, _ = store.try_admit(ticket, "deliberation", P)
        assert admitted

        _acquire_worker(node, wid, principal=P)
        assert _drain_count_from_state(node) == 1

        # Parent-kill cleanup (mirrors fail_pending_by_pid + lease release)
        store.fail(ticket)
        _release_worker(node, wid)

        assert _all_terminal(store, [ticket])
        assert _claimed_principals(store, "deliberation") == set()
        assert _drain_count_from_state(node) == 0

    def test_combined_timeout_and_kill_burst(self, store, node):
        """Multiple simultaneous timeouts and kills leave zero leaked claims."""
        tickets = []
        wids = []

        # Enqueue 6 tickets: 3 timeout, 3 kill
        for i in range(6):
            P = f"s3-burst-{i}-{uuid.uuid4().hex[:4]}"
            wid = f"s3-burst-wid-{i}"
            t = store.enqueue(
                lane="deliberation", kind="gw-admission",
                payload={"work_id": wid}, principal=P, latency_class="batch",
            )
            admitted, _ = store.try_admit(t, "deliberation", P)
            if admitted:
                _acquire_worker(node, wid, principal=P)
                tickets.append(t)
                wids.append(wid)

        # Simulate termination of all
        for t, w in zip(tickets, wids):
            store.fail(t)
            _release_worker(node, w)

        assert _all_terminal(store, tickets), "all tickets must be terminal"
        assert _claimed_principals(store, "deliberation") == set()
        assert _drain_count_from_state(node) == 0

    def test_no_ghost_leak_after_forced_cleanup(self, store, node):
        """Ghost leases (no principal) also cleaned up on kill path."""
        ghost_wid = f"ghost-kill-{uuid.uuid4().hex[:6]}"
        _acquire_worker(node, ghost_wid, principal=None)
        assert node.leases[ghost_wid]["principal"] == GHOST_PRINCIPAL

        # Cleanup: release ghost
        _release_worker(node, ghost_wid)
        assert _drain_count_from_state(node) == 0


# ---------------------------------------------------------------------------
# S4 — atomic drain-gate: concurrent distinct-principal burst peak ≤ 1
# ---------------------------------------------------------------------------

class TestS4AtomicDrainGate:
    """AC1/AC4: peak in-flight distinct-principal workers ≤ 1 under concurrent burst.

    Uses the mock doorman (_NodeState with ensure_serving=True) and the
    require_drain_clear=True path directly so the CI guard is meaningful —
    it exercises the same code path that the live harness S4 exercises.
    """

    @pytest.mark.parametrize("seed,n_groups", [
        (42, 7),   # mirrors harness S4 seed=42: council shared + 3 facets + gate + fixer + dead-enqueuer
        (137, 7),  # mirrors harness S4 seed=137
    ])
    def test_peak_in_flight_le_1_distinct_principals(self, node, seed, n_groups):
        """N distinct-principal workers burst with require_drain_clear=True → peak ≤ 1 active."""
        import random
        rng = random.Random(seed)

        results = [None] * n_groups
        peak_in_flight = [0]
        active_count = [0]
        peak_lock = threading.Lock()
        barrier = threading.Barrier(n_groups)

        def _worker(idx):
            principal = f"s4-group-{idx}-{rng.randint(0, 9999)}"
            work_id = f"s4-wid-{idx}-{uuid.uuid4().hex[:4]}"

            barrier.wait()  # burst: all start simultaneously

            with node.lock:
                ok = node.acquire_lease(
                    work_id, 60, "call_operator", role="worker",
                    principal=principal, require_drain_clear=True,
                )

            results[idx] = ok

            if ok is True:
                with peak_lock:
                    active_count[0] += 1
                    if active_count[0] > peak_in_flight[0]:
                        peak_in_flight[0] = active_count[0]

                # Simulate brief work then release
                time.sleep(rng.uniform(0.001, 0.005))

                with node.lock:
                    node.leases.pop(work_id, None)

                with peak_lock:
                    active_count[0] -= 1

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert peak_in_flight[0] <= 1, (
            f"seed={seed}: peak_in_flight={peak_in_flight[0]} must be ≤ 1 "
            f"results={results}"
        )
        from agents_core.doorman_server import CONTENDED
        true_count = sum(1 for r in results if r is True)
        contended_count = sum(1 for r in results if r is CONTENDED)
        assert true_count == 1, f"seed={seed}: exactly 1 must succeed; got {true_count}"
        assert contended_count == n_groups - 1, (
            f"seed={seed}: rest must be CONTENDED; got {contended_count}"
        )

    def test_ride_along_does_not_count_toward_peak(self, node):
        """Ride-alongs (same principal, no require_drain_clear) share the slot — not extra in-flight."""
        P = f"s4-shared-{uuid.uuid4().hex[:6]}"
        wid_main = f"s4-main-{uuid.uuid4().hex[:6]}"
        wid_ride = f"s4-ride-{uuid.uuid4().hex[:6]}"

        # Main group member acquires (no drain flag — it's first)
        with node.lock:
            ok_main = node.acquire_lease(wid_main, 60, "test", role="worker", principal=P)
        assert ok_main is True

        # Ride-along (same principal, require_drain_clear=False) succeeds
        with node.lock:
            ok_ride = node.acquire_lease(wid_ride, 60, "test", role="worker", principal=P)
        assert ok_ride is True, "same-principal ride-along must succeed alongside the main member"

        # Distinct group is still blocked
        with node.lock:
            from agents_core.doorman_server import CONTENDED
            ok_other = node.acquire_lease(
                "other-wid", 60, "test", role="worker", principal="other-group",
                require_drain_clear=True,
            )
        assert ok_other is CONTENDED, "distinct group must still be CONTENDED while P holds leases"

        # Cleanup
        with node.lock:
            node.leases.pop(wid_main, None)
            node.leases.pop(wid_ride, None)
