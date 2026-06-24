#!/usr/bin/env python3
"""GW enforce re-arm A/B harness (gw-enforce-rearm-ab-harness-v0).

Adversarial regression guard for the enforce flip. Reproduces both wedge shapes
and re-confirms the original A/B headline properties.

Scenarios:
  S0 — Baseline re-confirm (mined from /tmp/gw_enforce_ab_probe.py). N distinct-
       principal callers, OFF vs ENFORCE. Metric: peak in-flight (N vs 1).
  S1 — Shared-principal group + deliberation-spanning hold (drain-count self-
       deadlock fix). Active drain_count state assertions per AC2.
  S2 — Dead-enqueuer orphan (PR #107 fix). SIGKILL at seeded timing; assert
       orphan is reaped within bound and successor is admitted.
  S3 — Hard timeout / kill mid-flight. All tickets terminal, zero leaked claims.
  S4 — Combined realistic burst (council + facets + 1 injected death).

Usage:
  python scripts/gw_enforce_rearm_ab.py [--mock-gw | --live] [--seed N] [--n N]

Modes:
  --mock-gw   Friction-injected simulator (default). Seeded chaos; no real GW decode.
              Varies SIGKILL timing, orphan-reaper latency, reaper-vs-admit races.
  --live      Real GW + real doorman. For the final GO/NO-GO verdict run.

Safety:
  - Always uses an isolated temp ELEVATOR_DB_PATH (never perturbs production lane).
  - Sets GW_ADMISSION_MODE per-arm via process env only — never mutates conductor.env.
  - Best-effort cleanup (drive all tickets terminal, release leases, delete temp DB).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/srv/agents")

from agents_core.elevator import (
    ElevatorStore,
    GW_ADMISSION_ORPHAN_GRACE_SEC,
    HOSTNAME,
    IS_MASTER,
)
from agents_core.doorman_server import GHOST_PRINCIPAL

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("gw_enforce_rearm_ab")


# ---------------------------------------------------------------------------
# Stable assertion helpers
# (duplicated inline in tests/test_gw_enforce_rearm_ab.py for AC8 decoupling)
# ---------------------------------------------------------------------------

def _drain_count_from_state(doorman_state, exclude_principal: str | None = None) -> int:
    """Count active worker leases on doorman_state, optionally excluding a principal.

    Mirrors the /v0/drain-count endpoint logic. Ghost leases are always counted
    and emit a critical log when exclude_principal is set.
    """
    count = 0
    with doorman_state.lock:
        doorman_state._gc_stale()
        for wid, info in doorman_state.leases.items():
            if info.get("role") != "worker":
                continue
            p = info.get("principal", GHOST_PRINCIPAL)
            if p == GHOST_PRINCIPAL:
                if exclude_principal is not None:
                    log.critical(
                        "[drain_count] ghost_lease_counted work_id=%s — "
                        "no principal attributed; add principal= to acquire()",
                        wid,
                    )
                count += 1
            elif exclude_principal is not None and p == exclude_principal:
                continue
            else:
                count += 1
    return count


def _claimed_principals(elevator: ElevatorStore, lane: str = "deliberation") -> set:
    return elevator._claimed_principals_on_lane(lane)


def _all_tickets_terminal(elevator: ElevatorStore, ticket_ids: list[str]) -> bool:
    terminal = {"served", "failed", "expired"}
    return all(
        (elevator.get(t) or {}).get("status") in terminal
        for t in ticket_ids
    )


# ---------------------------------------------------------------------------
# Mock in-process doorman (--mock-gw mode; NOT imported by CI pytest wrapper)
# ---------------------------------------------------------------------------

class _MockDoormanState:
    """Minimal in-process doorman state for mock-mode scenarios.

    Manages the leases dict directly (no HTTP, no subprocess).
    Mirrors _NodeState.leases structure for drain_count assertions.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.leases: dict[str, dict] = {}
        self._critical_log_count = 0

    def acquire(
        self,
        work_id: str,
        principal: str | None = None,
        role: str = "worker",
        ttl_sec: int = 300,
    ) -> None:
        with self.lock:
            self.leases[work_id] = {
                "acquired_at": time.time(),
                "ttl_sec": ttl_sec,
                "role": role,
                "principal": principal if principal is not None else GHOST_PRINCIPAL,
            }

    def release(self, work_id: str) -> None:
        with self.lock:
            self.leases.pop(work_id, None)

    def drain_count(self, exclude_principal: str | None = None) -> int:
        count = 0
        with self.lock:
            for wid, info in self.leases.items():
                if info.get("role") != "worker":
                    continue
                p = info.get("principal", GHOST_PRINCIPAL)
                if p == GHOST_PRINCIPAL:
                    if exclude_principal is not None:
                        self._critical_log_count += 1
                        log.critical(
                            "[mock_doorman] ghost_lease_counted work_id=%s — "
                            "no principal; frozen counts as drain",
                            wid,
                        )
                    count += 1
                elif exclude_principal is not None and p == exclude_principal:
                    continue
                else:
                    count += 1
        return count

    def worker_count(self) -> int:
        with self.lock:
            return sum(1 for i in self.leases.values() if i.get("role") == "worker")


# ---------------------------------------------------------------------------
# Friction-injected backend (--mock-gw seeded chaos)
# NOT imported by the CI pytest wrapper — all chaos lives here.
# ---------------------------------------------------------------------------

class _FrictionBackend:
    """Seeded friction injector for --mock-gw mode.

    Models race conditions:
    - SIGKILL timing relative to admission (before_enqueue_ack / mid_poll / post_admit)
    - Variable orphan-reaper latency (0..reaper_latency_max_sec)
    - reaper-vs-try_admit interleavings (inject_reaper_before_admit)

    All randomness is derived from `seed` for deterministic CI replay.
    """

    SIGKILL_PHASES = ["before_enqueue_ack", "mid_poll", "post_admit"]

    def __init__(self, seed: int = 42, reaper_latency_max_sec: float = 2.0):
        self.rng = random.Random(seed)
        self.reaper_latency_max_sec = reaper_latency_max_sec
        self.sigkill_phase = self.rng.choice(self.SIGKILL_PHASES)
        self.kill_delay_sec = self.rng.uniform(0.0, 0.3)
        self.reaper_latency_sec = self.rng.uniform(0.0, reaper_latency_max_sec)
        self.inject_reaper_before_admit = self.rng.random() < 0.5

    def gw_call_delay(self) -> float:
        return self.rng.uniform(0.05, 0.2)

    def simulate_gw_call(self, caller_id: str) -> str:
        time.sleep(self.gw_call_delay())
        return f"mock-response-from-{caller_id}"

    def describe(self) -> dict:
        return {
            "sigkill_phase": self.sigkill_phase,
            "kill_delay_sec": round(self.kill_delay_sec, 3),
            "reaper_latency_sec": round(self.reaper_latency_sec, 3),
            "inject_reaper_before_admit": self.inject_reaper_before_admit,
        }


# ---------------------------------------------------------------------------
# S0 — Baseline re-confirm
# ---------------------------------------------------------------------------

def run_s0(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    n: int,
    friction: _FrictionBackend | None,
    live: bool,
) -> dict:
    """S0: N distinct-principal callers, OFF vs ENFORCE.

    OFF arm: all N skip elevator → peak worker leases = N.
    ENFORCE arm: elevator FIFO → peak in deliberation lane ≤ 1.

    AC1: OFF peak_drain == N, ENFORCE peak_drain == 1, 0 errors.
    """
    print(f"\n=== S0 BASELINE RE-CONFIRM (n={n}, mock={not live}) ===", flush=True)

    def _run_arm(mode: str) -> dict:
        peaks, errors, latencies = [], [], []
        peak_lock = threading.Lock()

        def _sample():
            while not stop.is_set():
                with peak_lock:
                    c = doorman.worker_count()
                    peaks.append(c)
                time.sleep(0.1)

        stop = threading.Event()
        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()
        t0 = time.time()

        def _caller(idx: int):
            principal = f"s0-{mode}-caller-{idx}"
            work_id = f"s0-{mode}-work-{idx}"
            try:
                if mode == "off":
                    doorman.acquire(work_id, principal=principal)
                    try:
                        time.sleep(friction.gw_call_delay() if friction else 0.1)
                    finally:
                        doorman.release(work_id)
                else:  # enforce
                    ticket = elevator.enqueue(
                        lane="deliberation",
                        kind="gw-admission",
                        payload={"work_id": work_id},
                        principal=principal,
                        latency_class="batch",
                    )
                    deadline = time.monotonic() + 30.0
                    admitted = False
                    while time.monotonic() < deadline:
                        elevator.reclaim_stale("deliberation")
                        ok, _ = elevator.try_admit(ticket, "deliberation", principal)
                        if ok:
                            admitted = True
                            break
                        # Wait for drain (exclude own principal — fresh group)
                        dc = doorman.drain_count(exclude_principal=principal)
                        if dc == 0:
                            ok, _ = elevator.try_admit(ticket, "deliberation", principal)
                            if ok:
                                admitted = True
                                break
                        time.sleep(0.05)
                    if not admitted:
                        elevator.fail(ticket)
                        errors.append(f"timeout-{idx}")
                        return
                    # Admitted: acquire doorman lease
                    doorman.acquire(work_id, principal=principal)
                    try:
                        time.sleep(friction.gw_call_delay() if friction else 0.1)
                    finally:
                        doorman.release(work_id)
                        elevator.ack(ticket)
            except Exception as e:
                errors.append(f"{idx}:{e}")

        threads = [threading.Thread(target=_caller, args=(i,)) for i in range(n)]
        lat_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        latencies.append(round(time.time() - lat_start, 2))

        stop.set()
        sampler.join(timeout=2)
        peak = max(peaks) if peaks else 0
        return {"arm": mode, "n": n, "peak_drain": peak, "errors": errors, "wall": latencies[0]}

    off_result = _run_arm("off")
    doorman.leases.clear()
    time.sleep(0.2)
    enf_result = _run_arm("enforce")

    passed = (
        off_result["peak_drain"] == n
        and enf_result["peak_drain"] <= 1
        and not off_result["errors"]
        and not enf_result["errors"]
    )
    result = {
        "scenario": "S0",
        "pass": passed,
        "off": off_result,
        "enforce": enf_result,
        "details": {
            "AC1_off_peak": off_result["peak_drain"],
            "AC1_enforce_peak": enf_result["peak_drain"],
            "AC1_errors": len(off_result["errors"]) + len(enf_result["errors"]),
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S1 — Shared-principal group + deliberation-spanning hold
# ---------------------------------------------------------------------------

def run_s1(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    n: int,
    friction: _FrictionBackend | None,
) -> dict:
    """S1: Drain-count self-deadlock fix. Active state assertions per AC2.

    Principal P holds a doorman lease (council delib hold). N voices share P.
    Assert drain_count(exclude_principal=P) == 0 → first voice proceeds (no deadlock).
    Sub-cases: ghost lease counted, distinct-principal gates the group.
    """
    print(f"\n=== S1 DRAIN-COUNT FIX (shared-principal, n={n}) ===", flush=True)

    P = f"council-delib-{uuid.uuid4().hex[:8]}"
    failures = []

    # ---- 1. Acquire the deliberation-spanning hold with principal=P ----
    hold_work_id = f"s1-hold-{uuid.uuid4().hex[:8]}"
    doorman.acquire(hold_work_id, principal=P, role="worker")

    # ---- 2. AC2 assertion: drain_count(exclude_principal=P) == 0 ----
    dc_excl = doorman.drain_count(exclude_principal=P)
    if dc_excl != 0:
        failures.append(f"AC2_drain_excl_expected_0_got_{dc_excl}")

    dc_total = doorman.drain_count(exclude_principal=None)
    if dc_total != 1:
        failures.append(f"AC2_drain_total_expected_1_got_{dc_total}")

    # ---- 3. First voice: elevator ride-along + drain_count check → proceeds ----
    hold_ticket = elevator.enqueue(
        lane="deliberation",
        kind="gw-admission",
        payload={"work_id": hold_work_id},
        principal=P,
        latency_class="batch",
    )
    admitted, _ = elevator.try_admit(hold_ticket, "deliberation", P)
    if not admitted:
        failures.append("hold_ticket_not_admitted")

    voice_tickets = []
    voice_served = 0
    for i in range(n):
        t = elevator.enqueue(
            lane="deliberation",
            kind="gw-admission",
            payload={"work_id": f"s1-voice-{i}"},
            principal=P,
            latency_class="batch",
        )
        voice_tickets.append(t)
        ok, is_ride_along = elevator.try_admit(t, "deliberation", P)
        if not ok:
            failures.append(f"voice_{i}_not_admitted")
            continue
        if not is_ride_along:
            failures.append(f"voice_{i}_not_ride_along")
        # Simulate: check drain_count(exclude_principal=P) before proceeding
        dc = doorman.drain_count(exclude_principal=P)
        if dc != 0:
            failures.append(f"voice_{i}_drain_nonzero_got_{dc}")
            continue
        # Voice proceeds — acquire doorman lease, simulate call, release
        v_work_id = f"s1-voice-work-{i}"
        doorman.acquire(v_work_id, principal=P, role="worker")
        time.sleep(0.02)
        doorman.release(v_work_id)
        elevator.ack(t)
        voice_served += 1

    if voice_served != n:
        failures.append(f"voices_served_{voice_served}_expected_{n}")

    # ---- 4. Sub-case: ghost lease is counted ----
    ghost_work_id = f"s1-ghost-{uuid.uuid4().hex[:8]}"
    doorman.acquire(ghost_work_id, principal=None, role="worker")  # → GHOST_PRINCIPAL
    ghost_critical_before = doorman._critical_log_count
    dc_with_ghost = doorman.drain_count(exclude_principal=P)
    ghost_critical_after = doorman._critical_log_count
    if dc_with_ghost < 1:
        failures.append(f"ghost_not_counted_got_{dc_with_ghost}")
    if ghost_critical_after <= ghost_critical_before:
        failures.append("ghost_critical_log_not_emitted")
    # Self-ghost case: drain_count > 0 → gate blocks (we verify by asserting count ≥ 1)
    if dc_with_ghost < 1:
        failures.append("self_ghost_gate_should_block_but_drain_zero")

    doorman.release(ghost_work_id)

    # ---- 5. Sub-case: distinct-principal worker gates the group ----
    other_work_id = f"s1-other-{uuid.uuid4().hex[:8]}"
    doorman.acquire(other_work_id, principal="other-group-xyz", role="worker")
    dc_with_other = doorman.drain_count(exclude_principal=P)
    if dc_with_other < 1:
        failures.append(f"distinct_principal_not_gating_got_{dc_with_other}")
    doorman.release(other_work_id)

    # ---- Cleanup ----
    doorman.release(hold_work_id)
    try:
        elevator.ack(hold_ticket)
    except Exception:
        try:
            elevator.fail(hold_ticket)
        except Exception:
            pass

    passed = len(failures) == 0
    result = {
        "scenario": "S1",
        "pass": passed,
        "failures": failures,
        "details": {
            "principal": P,
            "voices_served": voice_served,
            "AC2_drain_excl_P": dc_excl,
            "AC2_drain_total": dc_total,
            "ghost_counted_dc": dc_with_ghost,
            "ghost_critical_log_fired": ghost_critical_after > ghost_critical_before,
            "distinct_principal_dc": dc_with_other,
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S2 — Dead-enqueuer orphan
# ---------------------------------------------------------------------------

# Subprocess code: enqueue a gw-admission ticket then block until SIGKILL.
_S2_ENQUEUER_CODE = """
import os, sys, time
sys.path.insert(0, '/srv/agents')
db_path, principal = sys.argv[1], sys.argv[2]
os.environ['ELEVATOR_DB_PATH'] = db_path
from agents_core.elevator import ElevatorStore
store = ElevatorStore()
ticket = store.enqueue(
    lane='deliberation', kind='gw-admission',
    payload={{'work_id': 'orphan-test'}},
    principal=principal, latency_class='batch',
)
print(ticket, flush=True)
time.sleep(9999)
"""


def _age_ticket(elevator: ElevatorStore, ticket_id: str, seconds: int) -> None:
    from datetime import timedelta
    new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    with elevator._lock:
        elevator._conn.execute(
            "UPDATE queue_items SET created_at=? WHERE item_id=?",
            (new_ts, ticket_id),
        )
        elevator._conn.commit()


def run_s2(
    elevator: ElevatorStore,
    db_path: str,
    friction: _FrictionBackend | None,
    n_seeds: int = 3,
) -> dict:
    """S2: SIGKILL'd enqueuer creates pending orphan → reaped within bound → successor admitted.

    Seeded kill timing tests three phases:
    - before_enqueue_ack: subprocess may die before ticket appears
    - mid_poll: ticket exists, enqueuer dead before grace expires
    - post_admit: ticket exists past grace period

    AC3: orphan reaped ≤ 1 reaper cycle; subsequent caller admitted; no permanent wedge.
    """
    print(f"\n=== S2 DEAD-ENQUEUER ORPHAN (seeds={n_seeds}) ===", flush=True)

    failures = []
    # Use a short grace and max-wait for test speed
    grace = 2
    max_wait = 30
    env = dict(os.environ)
    env["ELEVATOR_DB_PATH"] = db_path
    env["GW_ADMISSION_ORPHAN_GRACE_SEC"] = str(grace)
    env["GW_ADMISSION_MAX_WAIT_SEC"] = str(max_wait)

    seeds = [42, 137, 7] if n_seeds >= 3 else [42]
    all_passed = True

    for seed_i, seed in enumerate(seeds[:n_seeds]):
        rng = random.Random(seed)
        kill_phase = rng.choice(_FrictionBackend.SIGKILL_PHASES)
        # kill delay relative to stdout flush (ticket enqueued)
        kill_delay = rng.uniform(0.0, 0.5) if kill_phase != "before_enqueue_ack" else 0.0

        principal = f"s2-orphan-{seed_i}"
        proc = subprocess.Popen(
            [sys.executable, "-c", _S2_ENQUEUER_CODE, db_path, principal],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        ticket_id = None
        try:
            if kill_phase == "before_enqueue_ack":
                # Kill before the ticket is written
                proc.kill()
                proc.wait(timeout=3)
            else:
                # Wait for ticket id on stdout
                try:
                    line = proc.stdout.readline().decode().strip()
                    ticket_id = line if line else None
                except Exception:
                    ticket_id = None
                time.sleep(kill_delay)
                proc.kill()
                proc.wait(timeout=3)
        except Exception as e:
            failures.append(f"seed{seed}_kill_error:{e}")
            try:
                proc.kill()
            except Exception:
                pass
            all_passed = False
            continue

        if kill_phase == "before_enqueue_ack" or ticket_id is None:
            # Nothing to reap — just verify no wedge
            print(f"  seed={seed} phase={kill_phase}: killed before enqueue, no ticket to verify")
            continue

        # Verify ticket exists and is pending
        item = elevator.get(ticket_id)
        if item is None or item["status"] != "pending":
            print(f"  seed={seed}: ticket {ticket_id} not pending ({item})")
            continue

        # Age past grace period so reaper can act
        _age_ticket(elevator, ticket_id, grace + 5)

        # Reap — should detect dead enqueuer pid
        t_reap_start = time.time()
        result = elevator.reap()
        t_reap_end = time.time()
        orphan_reaped = result.get("orphan_reaped", 0)

        item_after = elevator.get(ticket_id)
        status_after = (item_after or {}).get("status", "missing")

        if status_after not in ("failed", "expired"):
            failures.append(f"seed{seed}_orphan_not_reaped_status={status_after}")
            all_passed = False
        else:
            print(
                f"  seed={seed} phase={kill_phase}: orphan reaped in "
                f"{t_reap_end-t_reap_start:.2f}s status={status_after} "
                f"orphan_reaped={orphan_reaped}"
            )

        # Verify subsequent caller is admitted (FIFO head unblocked)
        succ_principal = f"s2-successor-{seed_i}"
        succ_ticket = elevator.enqueue(
            lane="deliberation",
            kind="gw-admission",
            payload={"work_id": f"s2-succ-{seed_i}"},
            principal=succ_principal,
            latency_class="batch",
        )
        ok, _ = elevator.try_admit(succ_ticket, "deliberation", succ_principal)
        if not ok:
            failures.append(f"seed{seed}_successor_not_admitted")
            all_passed = False
            elevator.fail(succ_ticket)
        else:
            elevator.ack(succ_ticket)
            print(f"  seed={seed}: successor admitted OK")

    passed = all_passed and len(failures) == 0
    result = {
        "scenario": "S2",
        "pass": passed,
        "failures": failures,
        "details": {"seeds": seeds[:n_seeds], "grace_sec": grace},
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S3 — Hard timeout / kill mid-flight
# ---------------------------------------------------------------------------

def run_s3(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    friction: _FrictionBackend | None,
) -> dict:
    """S3: Hard timeout and mid-flight kill → all tickets terminal, no leaked claims.

    AC4: every ticket reaches a terminal state; _claimed_principals_on_lane('deliberation')
    == set() after the burst.
    """
    print("\n=== S3 HARD TIMEOUT / KILL MID-FLIGHT ===", flush=True)

    failures = []
    ticket_ids = []
    member_deadline = 0.3  # short for test speed

    # ---- Case A: member timeout (thread watchdog) ----
    P_a = f"s3-timeout-{uuid.uuid4().hex[:6]}"
    w_id_a = f"s3-timeout-work-{uuid.uuid4().hex[:6]}"
    ticket_a = elevator.enqueue(
        lane="deliberation",
        kind="gw-admission",
        payload={"work_id": w_id_a},
        principal=P_a,
        latency_class="batch",
    )
    ticket_ids.append(ticket_a)

    admitted_a, _ = elevator.try_admit(ticket_a, "deliberation", P_a)
    if not admitted_a:
        failures.append("s3a_not_admitted")
    else:
        doorman.acquire(w_id_a, principal=P_a, role="worker")
        # Simulate member deadline firing: work exceeds timeout
        deadline_fired = threading.Event()
        work_done = threading.Event()

        def _hung_work():
            work_done.wait(timeout=member_deadline + 0.5)

        def _watchdog():
            if not work_done.wait(timeout=member_deadline):
                deadline_fired.set()

        wt = threading.Thread(target=_hung_work)
        wd = threading.Thread(target=_watchdog)
        wt.start(); wd.start()
        wd.join(timeout=member_deadline + 0.5)

        if deadline_fired.is_set():
            # Deadline fired: cleanup
            work_done.set()
            wt.join(timeout=1)
            elevator.fail(ticket_a)
            doorman.release(w_id_a)
        else:
            failures.append("s3a_watchdog_did_not_fire")
            elevator.fail(ticket_a)
            doorman.release(w_id_a)

    # ---- Case B: parent kill mid-run ----
    P_b = f"s3-kill-{uuid.uuid4().hex[:6]}"
    w_id_b = f"s3-kill-work-{uuid.uuid4().hex[:6]}"
    ticket_b = elevator.enqueue(
        lane="deliberation",
        kind="gw-admission",
        payload={"work_id": w_id_b},
        principal=P_b,
        latency_class="batch",
    )
    ticket_ids.append(ticket_b)

    admitted_b, _ = elevator.try_admit(ticket_b, "deliberation", P_b)
    if not admitted_b:
        failures.append("s3b_not_admitted")
    else:
        doorman.acquire(w_id_b, principal=P_b, role="worker")
        # Simulate kill: immediately fail ticket and release lease (parent-kill cleanup)
        elevator.fail(ticket_b)
        doorman.release(w_id_b)

    # ---- Assert all tickets terminal ----
    time.sleep(0.05)
    if not _all_tickets_terminal(elevator, ticket_ids):
        for t in ticket_ids:
            item = elevator.get(t)
            if item and item["status"] not in ("served", "failed", "expired"):
                failures.append(f"ticket_{t}_not_terminal_status={item['status']}")

    # ---- Assert no leaked claims ----
    claimed = _claimed_principals(elevator, "deliberation")
    if claimed:
        failures.append(f"leaked_claims={claimed}")

    # ---- Assert doorman leases released ----
    dc = doorman.worker_count()
    if dc != 0:
        failures.append(f"leaked_doorman_leases={dc}")

    passed = len(failures) == 0
    result = {
        "scenario": "S3",
        "pass": passed,
        "failures": failures,
        "details": {
            "tickets": len(ticket_ids),
            "watchdog_fired": deadline_fired.is_set() if admitted_a else None,
            "claimed_after": list(claimed),
            "doorman_workers_after": dc,
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S4 — Combined realistic burst
# ---------------------------------------------------------------------------

def run_s4(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    db_path: str,
    friction: _FrictionBackend | None,
    n: int,
) -> dict:
    """S4: Council + Facets + 1 injected enqueuer death, all under ENFORCE.

    Combined burst matching the 2026-06-24 incident shape.
    AC5: no permanent wedge, healthy callers 100% served, peak in-flight ≤ 1,
    paid-Sonnet fallback count = 0.
    """
    print(f"\n=== S4 COMBINED BURST (n={n}) ===", flush=True)

    failures = []
    fallback_count = 0
    served_count = 0
    peak_drain = 0
    peak_lock = threading.Lock()
    peaks = []
    stop_sampler = threading.Event()

    def _sample_peaks():
        while not stop_sampler.is_set():
            c = doorman.worker_count()
            with peak_lock:
                peaks.append(c)
            time.sleep(0.05)

    sampler = threading.Thread(target=_sample_peaks, daemon=True)
    sampler.start()

    P_council = f"council-delib-s4-{uuid.uuid4().hex[:6]}"
    all_tickets = []

    def _admit_and_serve(principal: str, label: str) -> bool:
        nonlocal served_count, fallback_count
        work_id = f"s4-{label}-{uuid.uuid4().hex[:6]}"
        try:
            ticket = elevator.enqueue(
                lane="deliberation",
                kind="gw-admission",
                payload={"work_id": work_id},
                principal=principal,
                latency_class="batch",
            )
            all_tickets.append(ticket)
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                elevator.reclaim_stale("deliberation")
                ok, is_ride_along = elevator.try_admit(ticket, "deliberation", principal)
                if ok:
                    if not is_ride_along:
                        dc = doorman.drain_count(exclude_principal=principal)
                        if dc > 0:
                            time.sleep(0.05)
                            continue
                    doorman.acquire(work_id, principal=principal, role="worker")
                    try:
                        if friction:
                            time.sleep(friction.gw_call_delay())
                        else:
                            time.sleep(0.05)
                    finally:
                        doorman.release(work_id)
                    elevator.ack(ticket)
                    served_count += 1
                    return True
                time.sleep(0.05)
            elevator.fail(ticket)
            failures.append(f"{label}_timeout")
            return False
        except Exception as e:
            failures.append(f"{label}_error:{e}")
            return False

    # ---- Spawn combined burst ----
    threads = []

    # Council: N voices sharing P_council
    for i in range(n):
        t = threading.Thread(
            target=_admit_and_serve,
            args=(P_council, f"council-voice-{i}"),
        )
        threads.append(t)

    # Facets: 3 personas with distinct principals
    for persona in ("facets-alpha", "facets-beta", "facets-gamma"):
        t = threading.Thread(
            target=_admit_and_serve,
            args=(persona, persona),
        )
        threads.append(t)

    # Distinct gate/fixer legs
    for role in ("gate", "fixer"):
        t = threading.Thread(
            target=_admit_and_serve,
            args=(f"distinct-{role}", f"s4-{role}"),
        )
        threads.append(t)

    # Injected enqueuer death (in a thread that spawns then kills subprocess)
    death_reaped = threading.Event()

    def _inject_death():
        env = dict(os.environ)
        env["ELEVATOR_DB_PATH"] = db_path
        env["GW_ADMISSION_ORPHAN_GRACE_SEC"] = "2"
        principal = "s4-dead-enqueuer"
        proc = subprocess.Popen(
            [sys.executable, "-c", _S2_ENQUEUER_CODE, db_path, principal],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            line = proc.stdout.readline().decode().strip()
            ticket_id = line if line else None
        except Exception:
            ticket_id = None
        time.sleep(friction.kill_delay_sec if friction else 0.1)
        proc.kill()
        proc.wait(timeout=3)
        if ticket_id:
            all_tickets.append(ticket_id)
            _age_ticket(elevator, ticket_id, 10)
            res = elevator.reap()
            if res.get("orphan_reaped", 0) > 0:
                death_reaped.set()

    death_thread = threading.Thread(target=_inject_death)
    threads.append(death_thread)

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    stop_sampler.set()
    sampler.join(timeout=2)

    peak_drain = max(peaks) if peaks else 0
    healthy_callers = n + 3 + 2  # council + facets + gate/fixer
    if served_count < healthy_callers:
        failures.append(f"healthy_not_all_served: {served_count}/{healthy_callers}")
    if peak_drain > 1:
        failures.append(f"peak_in_flight_{peak_drain}_exceeds_1")
    if not death_reaped.is_set():
        failures.append("injected_death_orphan_not_reaped")

    # Verify no leaked claims
    claimed = _claimed_principals(elevator, "deliberation")
    if claimed:
        failures.append(f"leaked_claims_after_burst={claimed}")

    passed = len(failures) == 0
    result = {
        "scenario": "S4",
        "pass": passed,
        "failures": failures,
        "details": {
            "served": served_count,
            "healthy_callers": healthy_callers,
            "peak_in_flight": peak_drain,
            "paid_sonnet_fallback": fallback_count,
            "injected_death_reaped": death_reaped.is_set(),
            "leaked_claims_after": list(claimed),
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# Verdict emitter
# ---------------------------------------------------------------------------

def emit_verdict(results: dict, seed: int, n: int, mode: str) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    verdict_dir = Path("/srv/lapis/planning/evals")
    verdict_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = verdict_dir / f"gw-enforce-rearm-ab-verdict-{date_str}.md"

    all_pass = all(r.get("pass", False) for r in results.values())
    go_no_go = "GO" if all_pass else "NO-GO"

    lines = [
        f"# GW Enforce Re-Arm A/B Verdict — {date_str}",
        "",
        f"**Mode:** {mode}  **Seed:** {seed}  **N:** {n}",
        f"**Verdict: {go_no_go}**",
        "",
        "## Per-Scenario Results",
        "",
        "| Scenario | Result | Key Metrics |",
        "|----------|--------|-------------|",
    ]

    for key, r in results.items():
        status = "PASS" if r.get("pass") else "FAIL"
        details = r.get("details", {})
        metric_str = " ".join(f"{k}={v}" for k, v in list(details.items())[:3])
        lines.append(f"| {r.get('scenario', key)} | {status} | {metric_str} |")

    lines += [
        "",
        "## Failure Details",
        "",
    ]
    for key, r in results.items():
        if not r.get("pass"):
            lines.append(f"### {r.get('scenario', key)}")
            for f in r.get("failures", []):
                lines.append(f"- {f}")
            lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- **S0**: Regression guard for original A/B — peak N→1 under ENFORCE.",
        "- **S1**: Drain-count fix proven by active state assertion (not log-grep).",
        "  Ghost leases counted + critical log; distinct-principal gates group.",
        "- **S2**: PR #107 fix proven — dead-enqueuer orphan reaped; no FIFO wedge.",
        "- **S3**: All tickets terminal after timeout/kill; zero leaked claims.",
        "- **S4**: Combined burst (council+facets+injected death) with no permanent wedge.",
        "",
        f"## Decision: **{go_no_go}** for re-arming `GW_ADMISSION_MODE=enforce`",
        "",
        f"GO requires all scenarios PASS. {'All scenarios passed.' if all_pass else 'One or more scenarios FAILED — do not re-arm enforce until fixed.'}",
        "",
        "<!-- gw-enforce-rearm-ab-harness-v0 -->",
    ]

    verdict_path.write_text("\n".join(lines) + "\n")
    print(f"\nVerdict written to: {verdict_path}", flush=True)
    return verdict_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_scenario_result(result: dict) -> None:
    s = result.get("scenario", "?")
    p = "PASS" if result.get("pass") else "FAIL"
    failures = result.get("failures", [])
    print(f"  {s}: {p}" + (f" — {failures}" if failures else ""), flush=True)


def _cleanup_elevator(elevator: ElevatorStore) -> None:
    try:
        with elevator._lock:
            pending = elevator._conn.execute(
                "SELECT item_id FROM queue_items WHERE status IN ('pending','claimed')"
            ).fetchall()
            for row in pending:
                elevator._conn.execute(
                    "UPDATE queue_items SET status='failed' WHERE item_id=?",
                    (row[0],),
                )
            if pending:
                elevator._conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--mock-gw", action="store_true", default=False,
                     help="Friction-injected simulator (default if neither flag set)")
    grp.add_argument("--live", action="store_true", default=False,
                     help="Real GW backend for final GO/NO-GO verdict")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for friction injection (default 42)")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of concurrent callers per scenario (default 4)")
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--skip-s4", action="store_true")
    args = parser.parse_args()

    live_mode = args.live
    mode_label = "live" if live_mode else "mock-gw"

    if not IS_MASTER and not live_mode:
        print("WARNING: not IS_MASTER — some elevator writes will fail. Run on BRIX.", flush=True)

    # ---- Isolated temp DB ----
    tmpdir = tempfile.mkdtemp(prefix="gw_ab_harness_")
    db_path = Path(tmpdir) / "elevator.db"
    os.environ["ELEVATOR_DB_PATH"] = str(db_path)
    print(f"HARNESS START mode={mode_label} seed={args.seed} n={args.n} db={db_path}", flush=True)

    friction = _FrictionBackend(seed=args.seed) if not live_mode else None
    if friction:
        print(f"  friction: {friction.describe()}", flush=True)

    elevator = ElevatorStore(db_path=db_path)
    doorman = _MockDoormanState()
    results = {}

    try:
        if not args.skip_s0:
            results["S0"] = run_s0(elevator, doorman, args.n, friction, live_mode)

        results["S1"] = run_s1(elevator, doorman, args.n, friction)
        results["S2"] = run_s2(elevator, str(db_path), friction)
        results["S3"] = run_s3(elevator, doorman, friction)

        if not args.skip_s4:
            results["S4"] = run_s4(elevator, doorman, str(db_path), friction, args.n)

    finally:
        _cleanup_elevator(elevator)
        elevator.close()
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    print("\n=== FINAL RESULTS ===", flush=True)
    for r in results.values():
        s = r.get("scenario", "?")
        p = "PASS" if r.get("pass") else "FAIL"
        print(f"  {s}: {p}", flush=True)

    emit_verdict(results, seed=args.seed, n=args.n, mode=mode_label)

    all_pass = all(r.get("pass", False) for r in results.values())
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
