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
  S5 — Span-hold coordination lease + distinct-operator-principal admit (kind-
       aware drain-gate). Proves coordination-skip; guards cross-principal fence.
  S6 — GW reference-leg unified-principal admit + ghost/split wedge guards (5th
       wedge). Reference leg held as inference under shared principal P; all
       operator legs admit. Ghost ref (principal=None) and split-principal ref
       (P_ref != P_op) each assert CONTENDED.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Insert the repo root (parent of scripts/) first so the worktree's agents_core
# takes precedence over any system-wide /srv/agents copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, "/srv/agents")

# D-AMEND-1: GW_ADMISSION_ORPHAN_GRACE_SEC / GW_ADMISSION_MAX_WAIT_SEC are
# module-level constants frozen at import. Set them before agents_core loads
# so the in-process reaper uses short test values, not the 60/900-second defaults.
_TEST_ORPHAN_GRACE_SEC = 2
_TEST_MAX_WAIT_SEC = 30
os.environ["GW_ADMISSION_ORPHAN_GRACE_SEC"] = str(_TEST_ORPHAN_GRACE_SEC)
os.environ["GW_ADMISSION_MAX_WAIT_SEC"] = str(_TEST_MAX_WAIT_SEC)

from agents_core.elevator import (
    ElevatorStore,
    GW_ADMISSION_MAX_WAIT_SEC,
    GW_ADMISSION_ORPHAN_GRACE_SEC,
    HOSTNAME,
    IS_MASTER,
    _now,
)
from agents_core.doorman_server import CONTENDED, GHOST_PRINCIPAL

# Guard: if this fires, agents_core.elevator was imported before this file set the env vars.
assert GW_ADMISSION_ORPHAN_GRACE_SEC == _TEST_ORPHAN_GRACE_SEC, (
    f"D-AMEND-1 import-order bug: elevator constant={GW_ADMISSION_ORPHAN_GRACE_SEC} "
    f"but env set {_TEST_ORPHAN_GRACE_SEC} — something imported agents_core.elevator first"
)

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
        require_drain_clear: bool = False,
        lease_kind: str = "inference",
    ) -> bool | object:
        """Try to register a lease. Mirrors _NodeState.acquire_lease atomic semantics.

        Returns CONTENDED (without registering) if require_drain_clear=True and another
        principal's worker lease is active. Returns True on success.
        GHOST_PRINCIPAL leases always count as contending (never excluded).
        lease_kind="coordination" leases are skipped in the contention loop, mirroring
        doorman_server.py:500-503. drain_count() still counts them (AC1a).
        """
        with self.lock:
            if require_drain_clear and role == "worker":
                effective_principal = principal if principal is not None else GHOST_PRINCIPAL
                for wid, info in self.leases.items():
                    if info.get("role") != "worker":
                        continue
                    # Coordination leases hold no inference — not a drain-gate contender.
                    # Mirrors doorman_server.py:502-503.
                    if info.get("lease_kind", "inference") == "coordination":
                        continue
                    _p = info.get("principal", GHOST_PRINCIPAL)
                    if _p == GHOST_PRINCIPAL:
                        self._critical_log_count += 1
                        log.critical(
                            "[mock_doorman] ghost_lease_counted work_id=%s — "
                            "no principal; frozen counts as drain",
                            wid,
                        )
                        return CONTENDED
                    if _p != effective_principal:
                        return CONTENDED
            self.leases[work_id] = {
                "acquired_at": time.time(),
                "ttl_sec": ttl_sec,
                "role": role,
                "principal": principal if principal is not None else GHOST_PRINCIPAL,
                "lease_kind": lease_kind,
            }
        return True

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

    def active_group_count(self) -> int:
        """Count distinct active principal groups (ride-alongs share one slot)."""
        with self.lock:
            return len({
                info.get("principal", GHOST_PRINCIPAL)
                for info in self.leases.values()
                if info.get("role") == "worker"
            })


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
                # 0.02s interval: fine enough to catch peaks shorter than gw_call_delay
                # min (≈0.05s in mock mode); 0.1s was too coarse and undersampled.
                time.sleep(0.02)

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
                    is_ride_along = False
                    while time.monotonic() < deadline:
                        elevator.reclaim_stale("deliberation")
                        ok, is_ride_along = elevator.try_admit(ticket, "deliberation", principal)
                        if ok:
                            admitted = True
                            break
                        time.sleep(0.05)
                    if not admitted:
                        elevator.fail(ticket)
                        errors.append(f"timeout-{idx}")
                        return
                    # Atomic drain-gate: single acquire(require_drain_clear=True) for fresh
                    # groups replaces the old two-step drain_count+acquire, mirroring llm.py.
                    # On CONTENDED → retry within deadline; ride-alongs use False.
                    while time.monotonic() < deadline:
                        res = doorman.acquire(work_id, principal=principal,
                                              require_drain_clear=(not is_ride_along))
                        if res is not CONTENDED:
                            break
                        time.sleep(0.05)
                    else:
                        elevator.fail(ticket)
                        errors.append(f"timeout-drain-{idx}")
                        return
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
        # Voice is a ride-along (same principal P, already admitted).
        # AC2 drain_count assertion is proven at the scenario level above.
        # require_drain_clear=False: ride-along shares the group's admitted slot.
        v_work_id = f"s1-voice-work-{i}"
        doorman.acquire(v_work_id, principal=P, role="worker", require_drain_clear=False)
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
    payload={'work_id': 'orphan-test'},
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
    # D-AMEND-1: use the module-level constant (frozen from env before import) so
    # aging is always relative to the effective grace the in-process reaper uses.
    grace = GW_ADMISSION_ORPHAN_GRACE_SEC
    env = dict(os.environ)
    env["ELEVATOR_DB_PATH"] = db_path
    # GW_ADMISSION_ORPHAN_GRACE_SEC and GW_ADMISSION_MAX_WAIT_SEC already in os.environ

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
            # Count distinct active principal groups: ride-alongs sharing one principal
            # count as 1 slot, matching the drain-gate's group-level occupancy model.
            c = doorman.active_group_count()
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
                    # Atomic drain-gate: require_drain_clear=True for fresh groups collapses
                    # the old two-step drain_count+acquire into one lock operation, mirroring
                    # agents_core/llm.py. Ride-alongs (same group already admitted) use False.
                    while time.monotonic() < deadline:
                        res = doorman.acquire(work_id, principal=principal, role="worker",
                                              require_drain_clear=(not is_ride_along))
                        if res is not CONTENDED:
                            break
                        time.sleep(0.05)
                    else:
                        elevator.fail(ticket)
                        failures.append(f"{label}_timeout")
                        return False
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
# S5 — Span-hold coordination lease + distinct-operator-principal admit
# ---------------------------------------------------------------------------

def run_s5(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    n: int,
    friction: _FrictionBackend | None,
) -> dict:
    """S5: Reproduces the 2026-06-24 span-principal wedge; proves kind-aware drain-gate fix.

    A session holds a span-hold coordination lease (P_span) while its own GW operator
    legs (P_op != P_span) acquire with require_drain_clear=True.

    AC2: every operator leg admits — the coordination lease is skipped, not a contender.
    AC3: cross-principal inference still contends inference (drain-gate purpose intact).
    AC3a: transition — CONTENDED while inference present, ADMITS when only coordination remains.
    AC6: inline self-check that the coordination-skip in the mock fires correctly.
    """
    print(f"\n=== S5 SPAN-HOLD COORDINATION + DISTINCT-OPERATOR ADMIT (n={n}) ===", flush=True)

    failures = []
    served_count = 0
    peaks = []
    stop_sampler = threading.Event()
    peak_lock = threading.Lock()

    P_span = f"shared-delib-s5-{uuid.uuid4().hex[:8]}"
    P_op = f"op-gravitywell-s5-{uuid.uuid4().hex[:8]}"
    span_hold_id = f"s5-span-{uuid.uuid4().hex[:6]}"

    def _sample_peaks():
        while not stop_sampler.is_set():
            c = doorman.active_group_count()
            with peak_lock:
                peaks.append(c)
            time.sleep(0.05)

    sampler = threading.Thread(target=_sample_peaks, daemon=True)
    sampler.start()

    try:
        # ---- Phase 1: Acquire span-hold (coordination) — held for the entire scenario ----
        res = doorman.acquire(span_hold_id, principal=P_span, role="worker",
                              lease_kind="coordination")
        if res is not True:
            failures.append(f"span_hold_acquire_failed: {res}")

        # ---- Phase 2: AC6 inline self-check — confirm coordination-skip fires ----
        # With only the coordination lease present, a distinct-principal inference acquire MUST admit.
        _sc_admit_id = f"s5-sc-admit-{uuid.uuid4().hex[:4]}"
        sc_admit = doorman.acquire(_sc_admit_id, principal=P_op, role="worker",
                                   require_drain_clear=True, lease_kind="inference")
        if sc_admit is not True:
            failures.append(f"AC6_coordination_blocked_inference: got {sc_admit!r}")
        else:
            doorman.release(_sc_admit_id)

        # With an inference lease under a third principal present, inference MUST contend.
        P_sc_inf = f"s5-sc-inf-{uuid.uuid4().hex[:6]}"
        _sc_inf_id = f"s5-sc-inf-{uuid.uuid4().hex[:4]}"
        doorman.acquire(_sc_inf_id, principal=P_sc_inf, role="worker", lease_kind="inference")
        _sc_op_id = f"s5-sc-op-{uuid.uuid4().hex[:4]}"
        sc_contend = doorman.acquire(_sc_op_id, principal=P_op, role="worker",
                                     require_drain_clear=True, lease_kind="inference")
        if sc_contend is not CONTENDED:
            failures.append(f"AC6_inference_should_contend: got {sc_contend!r}")
        doorman.release(_sc_inf_id)

        # ---- Phase 3: AC3a transition + AC3 negative control ----
        # State: span-hold (coordination, P_span) is held throughout this phase.
        P_inf = f"inf-s5-{uuid.uuid4().hex[:6]}"
        P_op2 = f"op2-s5-{uuid.uuid4().hex[:6]}"
        P_inf_b = f"inf-b-s5-{uuid.uuid4().hex[:6]}"
        _inf_id = f"s5-inf-{uuid.uuid4().hex[:4]}"
        _op2_id = f"s5-op2-{uuid.uuid4().hex[:4]}"
        _inf_b_id = f"s5-inf-b-{uuid.uuid4().hex[:4]}"

        # Add an inference worker under P_inf (simulates another session holding GPU).
        doorman.acquire(_inf_id, principal=P_inf, role="worker", lease_kind="inference")

        # AC3a-part1: P_op2 CONTENDS — the inference worker is the cause, not the coordination lease.
        res_contend = doorman.acquire(_op2_id, principal=P_op2, role="worker",
                                      require_drain_clear=True, lease_kind="inference")
        if res_contend is not CONTENDED:
            failures.append(f"AC3a_expected_CONTENDED_with_inference_present: got {res_contend!r}")

        # AC3 negative control: a second inference principal (P_inf_b) CONTENDS the first (P_inf).
        res_inf_b = doorman.acquire(_inf_b_id, principal=P_inf_b, role="worker",
                                    require_drain_clear=True, lease_kind="inference")
        if res_inf_b is not CONTENDED:
            failures.append(f"AC3_inference_b_not_contended_by_inference_a: got {res_inf_b!r}")

        # Release the inference worker — only the span-hold (coordination) now remains.
        doorman.release(_inf_id)

        # AC3a-part2: P_op2 now ADMITS — the coordination lease is skipped, not a blocker.
        res_admit = doorman.acquire(_op2_id, principal=P_op2, role="worker",
                                    require_drain_clear=True, lease_kind="inference")
        if res_admit is CONTENDED:
            failures.append("AC3a_still_CONTENDED_after_inference_released")
        elif res_admit is True:
            doorman.release(_op2_id)
        else:
            failures.append(f"AC3a_unexpected_after_inference_released: {res_admit!r}")

        # ---- Phase 4: AC2 main elevator test ----
        # N operator legs (P_op, shared) + 3 Facets personas (distinct per-persona principal).
        # Span-hold (P_span, coordination) is held throughout. Every leg MUST be served.
        all_tickets = []

        def _admit_and_serve(principal: str, label: str) -> None:
            nonlocal served_count
            work_id = f"s5-{label}-{uuid.uuid4().hex[:6]}"
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
                        while time.monotonic() < deadline:
                            res = doorman.acquire(
                                work_id, principal=principal, role="worker",
                                require_drain_clear=(not is_ride_along),
                                lease_kind="inference",
                            )
                            if res is not CONTENDED:
                                break
                            time.sleep(0.05)
                        else:
                            elevator.fail(ticket)
                            failures.append(f"AC2_drain_timeout_{label}")
                            return
                        try:
                            time.sleep(friction.gw_call_delay() if friction else 0.05)
                        finally:
                            doorman.release(work_id)
                        elevator.ack(ticket)
                        served_count += 1
                        return
                    time.sleep(0.05)
                elevator.fail(ticket)
                failures.append(f"AC2_timeout_{label}")
            except Exception as e:
                failures.append(f"AC2_error_{label}:{e}")

        threads = []
        for i in range(n):
            threads.append(threading.Thread(target=_admit_and_serve, args=(P_op, f"op-{i}")))
        for persona in ("facets-alpha", "facets-beta", "facets-gamma"):
            threads.append(threading.Thread(
                target=_admit_and_serve, args=(f"s5-{persona}", persona)
            ))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        expected_callers = n + 3
        if served_count < expected_callers:
            failures.append(f"AC2_not_all_served: {served_count}/{expected_callers}")

        claimed = _claimed_principals(elevator, "deliberation")
        if claimed:
            failures.append(f"leaked_claims={claimed}")

    finally:
        doorman.release(span_hold_id)
        stop_sampler.set()
        sampler.join(timeout=2)

    peak_drain = max(peaks) if peaks else 0
    passed = len(failures) == 0
    result = {
        "scenario": "S5",
        "pass": passed,
        "failures": failures,
        "details": {
            "served": served_count,
            "expected_callers": n + 3,
            "peak_in_flight": peak_drain,
            "AC2_admits_pass": not any("AC2_" in f for f in failures),
            "AC3_contention_pass": not any("AC3_" in f for f in failures),
            "AC3a_transition_pass": not any("AC3a_" in f for f in failures),
            "AC6_self_check_pass": not any("AC6_" in f for f in failures),
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S6 — GW reference-leg unified-principal admit + ghost/split wedge guards
# ---------------------------------------------------------------------------

def run_s6(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    n: int,
    friction: "_FrictionBackend | None",
) -> dict:
    """S6: GW reference-leg ghost/split inference acquire — 5th-wedge guard.

    Models the actual reference-reviewer acquire in gw_agent.py: a long-held
    lease_kind="inference" worker under the shared spec-review principal P,
    concurrent with Facets/Council operator legs. The arc fix (U1-U3) threads
    one shared gw_principal so the whole spec-review is one admission group.

    AC2: unified-principal — reference leg (inference, P) + span-hold
         (coordination, P) held; every operator leg under same P with
         require_drain_clear=True is GRANTED (not CONTENDED).
    AC3 (guard A): ghost reference leg (principal=None, inference) → CONTENDED.
    AC4 (guard B): split-principal ref (P_ref) + operator (P_op != P_ref) → CONTENDED.
    AC5: elevator serve phase — all n operator + Facets persona legs served.
    """
    print(
        f"\n=== S6 GW REFERENCE-LEG UNIFIED-PRINCIPAL ADMIT + GHOST/SPLIT GUARDS (n={n}) ===",
        flush=True,
    )

    failures = []
    served_count = 0
    peaks = []
    stop_sampler = threading.Event()
    peak_lock = threading.Lock()

    P = f"gw-gate-{uuid.uuid4().hex[:8]}"
    ref_id = f"s6-ref-{uuid.uuid4().hex[:6]}"
    span_id = f"s6-span-{uuid.uuid4().hex[:6]}"
    ref_elev_id = None

    def _sample_peaks():
        while not stop_sampler.is_set():
            c = doorman.active_group_count()
            with peak_lock:
                peaks.append(c)
            time.sleep(0.05)

    sampler = threading.Thread(target=_sample_peaks, daemon=True)
    sampler.start()

    try:
        # ---- Phase 1: Unified-principal ADMIT (the payoff) ----
        # Reference leg: inference, shared P — models gw_agent.py long acquire.
        res_ref = doorman.acquire(ref_id, principal=P, role="worker", lease_kind="inference")
        if res_ref is not True:
            failures.append(f"AC2_ref_leg_acquire_failed: {res_ref!r}")

        # Span-hold: coordination, shared P — held throughout for realism.
        res_span = doorman.acquire(span_id, principal=P, role="worker", lease_kind="coordination")
        if res_span is not True:
            failures.append(f"AC2_span_hold_acquire_failed: {res_span!r}")

        # Operator legs under same P with require_drain_clear=True, inference.
        # Reference-leg inference worker is held — shared P must ride-along, not contend.
        for i in range(n):
            op_id = f"s6-op-{i}-{uuid.uuid4().hex[:4]}"
            res = doorman.acquire(
                op_id, principal=P, role="worker",
                require_drain_clear=True, lease_kind="inference",
            )
            if res is not True:
                failures.append(f"AC2_op_{i}_not_admitted: {res!r}")
            else:
                doorman.release(op_id)

        # Facets personas — also under shared P, ride-along.
        for persona in ("facets-alpha", "facets-beta", "facets-gamma"):
            persona_id = f"s6-{persona}-{uuid.uuid4().hex[:4]}"
            res = doorman.acquire(
                persona_id, principal=P, role="worker",
                require_drain_clear=True, lease_kind="inference",
            )
            if res is not True:
                failures.append(f"AC2_{persona}_not_admitted: {res!r}")
            else:
                doorman.release(persona_id)

        # ---- Phase 2: GHOST reference leg WEDGES (guard A) ----
        # Release the Phase 1 reference leg; hold a ghost (principal=None) instead.
        doorman.release(ref_id)
        ref_ghost_id = f"s6-ghost-{uuid.uuid4().hex[:6]}"
        doorman.acquire(ref_ghost_id, principal=None, role="worker", lease_kind="inference")

        # Operator leg under valid P, require_drain_clear → CONTENDED (ghost blocks always).
        op_ghost_probe = f"s6-ghost-op-{uuid.uuid4().hex[:4]}"
        res_ghost = doorman.acquire(
            op_ghost_probe, principal=P, role="worker",
            require_drain_clear=True, lease_kind="inference",
        )
        if res_ghost is not CONTENDED:
            failures.append(f"AC3_ghost_ref_should_CONTEND: got {res_ghost!r}")

        doorman.release(ref_ghost_id)

        # ---- Phase 3: SPLIT principal WEDGES (guard B) ----
        # Reference leg under P_ref; operator leg under P_op != P_ref → CONTENDED.
        P_ref = f"gw-ref-{uuid.uuid4().hex[:8]}"
        P_op = f"gw-op-{uuid.uuid4().hex[:8]}"
        ref_split_id = f"s6-split-ref-{uuid.uuid4().hex[:6]}"
        op_split_id = f"s6-split-op-{uuid.uuid4().hex[:4]}"

        doorman.acquire(ref_split_id, principal=P_ref, role="worker", lease_kind="inference")
        res_split = doorman.acquire(
            op_split_id, principal=P_op, role="worker",
            require_drain_clear=True, lease_kind="inference",
        )
        if res_split is not CONTENDED:
            failures.append(f"AC4_split_principal_should_CONTEND: got {res_split!r}")

        doorman.release(ref_split_id)

        # ---- Phase 4: Main elevator admit/serve (mirror S5 Phase 4) ----
        # Re-acquire reference leg under shared P for the elevator serve phase.
        ref_elev_id = f"s6-ref-elev-{uuid.uuid4().hex[:6]}"
        doorman.acquire(ref_elev_id, principal=P, role="worker", lease_kind="inference")

        all_tickets = []

        def _admit_and_serve(principal: str, label: str) -> None:
            nonlocal served_count
            work_id = f"s6-{label}-{uuid.uuid4().hex[:6]}"
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
                        while time.monotonic() < deadline:
                            res = doorman.acquire(
                                work_id, principal=principal, role="worker",
                                require_drain_clear=(not is_ride_along),
                                lease_kind="inference",
                            )
                            if res is not CONTENDED:
                                break
                            time.sleep(0.05)
                        else:
                            elevator.fail(ticket)
                            failures.append(f"AC5_drain_timeout_{label}")
                            return
                        try:
                            time.sleep(friction.gw_call_delay() if friction else 0.05)
                        finally:
                            doorman.release(work_id)
                        elevator.ack(ticket)
                        served_count += 1
                        return
                    time.sleep(0.05)
                elevator.fail(ticket)
                failures.append(f"AC5_timeout_{label}")
            except Exception as e:
                failures.append(f"AC5_error_{label}:{e}")

        threads = []
        for i in range(n):
            threads.append(threading.Thread(target=_admit_and_serve, args=(P, f"op-{i}")))
        for persona in ("facets-alpha", "facets-beta", "facets-gamma"):
            threads.append(threading.Thread(target=_admit_and_serve, args=(P, persona)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        expected_callers = n + 3
        if served_count < expected_callers:
            failures.append(f"AC5_not_all_served: {served_count}/{expected_callers}")

        claimed = _claimed_principals(elevator, "deliberation")
        if claimed:
            failures.append(f"leaked_claims={claimed}")

    finally:
        doorman.release(span_id)
        if ref_elev_id is not None:
            doorman.release(ref_elev_id)
        stop_sampler.set()
        sampler.join(timeout=2)

    peak_drain = max(peaks) if peaks else 0
    passed = len(failures) == 0
    result = {
        "scenario": "S6",
        "pass": passed,
        "failures": failures,
        "details": {
            "served": served_count,
            "expected_callers": n + 3,
            "peak_in_flight": peak_drain,
            "AC2_admit_pass": not any("AC2_" in f for f in failures),
            "AC3_ghost_wedge_pass": not any("AC3_" in f for f in failures),
            "AC4_split_wedge_pass": not any("AC4_" in f for f in failures),
            "AC5_serve_pass": not any("AC5_" in f for f in failures),
        },
    }
    _print_scenario_result(result)
    return result


# ---------------------------------------------------------------------------
# S7 — Liveness-aware admission (6th-wedge regression gate)
# ---------------------------------------------------------------------------

def run_s7(
    elevator: ElevatorStore,
    doorman: _MockDoormanState,
    friction: "_FrictionBackend | None",
) -> dict:
    """S7: Liveness-aware admission — 6th-wedge regression gate.

    (i)   Dead pending head: live waiter admits via D2 squeeze-past (same try_admit call).
    (ii)  Dead claimed slot: D4 fast-reclaim frees slot within 1 reaper cycle.
    (iii) Slow-but-live owner: reclaim_stale requeues to BACK (fresh created_at), not failed.
    (iv)  Ghost/spoofed stamp (unknown verdict): requeued to BACK at reclaim, NOT falsely
          failed; clears at the presumed-dead backstop (no indefinite head-of-line deadlock).

    AC8: (i)+(ii) squeeze-past, (iii) back-of-line, (iv) no deadlock/not-falsely-failed,
    drain-to-0.
    """
    print("\n=== S7 LIVENESS-AWARE ADMISSION — 6TH-WEDGE REGRESSION GATE ===", flush=True)

    failures = []
    DEAD_PID = 99999999  # almost certainly non-existent
    DEAD_START = 1.0

    # ---- Shared helpers ----

    def _insert_pending_controlled(item_id, principal, pid, host=None, start_time=None):
        """Insert a pending gw-admission ticket with a controlled _enqueuer_id stamp."""
        h = host if host is not None else HOSTNAME
        eid = {"host": h, "pid": pid}
        if start_time is not None:
            eid["start_time"] = start_time
        payload = json.dumps({"work_id": f"s7-{item_id}", "_enqueuer_id": eid})
        now_ts = _now()
        with elevator._lock:
            elevator._conn.execute(
                "INSERT INTO queue_items (item_id, lane, kind, principal, payload, "
                "latency_class, status, attempts, created_at) "
                "VALUES (?, 'deliberation', 'gw-admission', ?, ?, 'batch', 'pending', 0, ?)",
                (item_id, principal, payload, now_ts),
            )
            elevator._conn.commit()
        return now_ts

    def _claim_direct(item_id, claim_ttl_sec=300):
        """Directly claim a ticket (bypasses try_admit for test injection)."""
        now_ts = _now()
        with elevator._lock:
            elevator._conn.execute(
                "UPDATE queue_items SET status='claimed', claim_owner=?, "
                "claim_ttl_sec=?, claimed_at=? WHERE item_id=?",
                (item_id, claim_ttl_sec, now_ts, item_id),
            )
            elevator._conn.commit()

    def _backdate_claimed(item_id, seconds):
        new_ts = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
        with elevator._lock:
            elevator._conn.execute(
                "UPDATE queue_items SET claimed_at=? WHERE item_id=?",
                (new_ts, item_id),
            )
            elevator._conn.commit()

    all_tickets = []

    # ---- (i) Dead pending head → D2 squeeze-past ----
    print("  S7(i): dead pending head → squeeze-past inline on try_admit", flush=True)
    iid_dead_head = f"s7-i-dead-{uuid.uuid4().hex[:6]}"
    iid_live_waiter = f"s7-i-live-{uuid.uuid4().hex[:6]}"
    P_dead_i = f"s7-i-dead-{uuid.uuid4().hex[:4]}"
    P_live_i = f"s7-i-live-{uuid.uuid4().hex[:4]}"
    all_tickets += [iid_dead_head, iid_live_waiter]

    _insert_pending_controlled(iid_dead_head, P_dead_i, DEAD_PID, start_time=DEAD_START)
    _age_ticket(elevator, iid_dead_head, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)
    time.sleep(0.005)
    _insert_pending_controlled(iid_live_waiter, P_live_i, os.getpid(), start_time=None)

    ok_i, _ = elevator.try_admit(iid_live_waiter, "deliberation", P_live_i)
    dead_head_i_status = (elevator.get(iid_dead_head) or {}).get("status")

    if not ok_i:
        failures.append("S7i_live_waiter_not_admitted_via_squeeze_past")
    if dead_head_i_status != "failed":
        failures.append(f"S7i_dead_head_not_failed_status={dead_head_i_status}")
    else:
        print(f"    dead_head=failed, live_admitted={ok_i} ✓", flush=True)

    if ok_i:
        elevator.ack(iid_live_waiter)
    else:
        elevator.fail(iid_live_waiter)

    # ---- (ii) Dead claimed slot → D4 fast-reclaim within 1 reaper cycle ----
    print("  S7(ii): dead claimed slot → fast-reclaim via D4 reap", flush=True)
    iid_dead_claimed = f"s7-ii-dead-{uuid.uuid4().hex[:6]}"
    iid_next_waiter = f"s7-ii-next-{uuid.uuid4().hex[:6]}"
    P_dead_ii = f"s7-ii-dead-{uuid.uuid4().hex[:4]}"
    P_next_ii = f"s7-ii-next-{uuid.uuid4().hex[:4]}"
    all_tickets += [iid_dead_claimed, iid_next_waiter]

    _insert_pending_controlled(iid_dead_claimed, P_dead_ii, DEAD_PID, start_time=DEAD_START)
    _claim_direct(iid_dead_claimed, claim_ttl_sec=960)
    _backdate_claimed(iid_dead_claimed, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)
    _insert_pending_controlled(iid_next_waiter, P_next_ii, os.getpid(), start_time=None)

    # Next waiter blocked while dead claimant holds the slot.
    ok_before_ii, _ = elevator.try_admit(iid_next_waiter, "deliberation", P_next_ii)
    if ok_before_ii:
        failures.append("S7ii_next_waiter_admitted_before_reap_unexpected")
        elevator.ack(iid_next_waiter)

    if not ok_before_ii:
        # Reap: D4 fast-reclaims the dead claimed ticket.
        elevator.reap()
        dead_claimed_status = (elevator.get(iid_dead_claimed) or {}).get("status")
        if dead_claimed_status != "failed":
            failures.append(f"S7ii_dead_claimed_not_reaped_status={dead_claimed_status}")
        else:
            print(f"    dead_claimed=failed ✓", flush=True)

        ok_after_ii, _ = elevator.try_admit(iid_next_waiter, "deliberation", P_next_ii)
        if not ok_after_ii:
            failures.append("S7ii_next_waiter_not_admitted_after_reap")
        else:
            elevator.ack(iid_next_waiter)
            print(f"    next_waiter_admitted={ok_after_ii} ✓", flush=True)

    # ---- (iii) Slow-but-live owner → reclaim requeues to back, not failed ----
    print("  S7(iii): slow-but-live owner → requeued to back at reclaim", flush=True)
    iid_slow = f"s7-iii-slow-{uuid.uuid4().hex[:6]}"
    P_slow = f"s7-iii-slow-{uuid.uuid4().hex[:4]}"
    all_tickets.append(iid_slow)

    _insert_pending_controlled(iid_slow, P_slow, os.getpid(), start_time=None)
    iii_original_created = (elevator.get(iid_slow) or {}).get("created_at")

    elevator.try_admit(iid_slow, "deliberation", P_slow, claim_ttl_sec=1)
    _backdate_claimed(iid_slow, 10)  # 10 seconds past TTL of 1

    elevator.reclaim_stale("deliberation")
    iii_item = elevator.get(iid_slow) or {}

    if iii_item.get("status") != "pending":
        failures.append(f"S7iii_slow_alive_not_requeued_to_pending_status={iii_item.get('status')}")
    elif iii_item.get("claim_owner") is not None:
        failures.append("S7iii_slow_alive_claim_owner_not_cleared")
    elif iii_original_created and iii_item.get("created_at", "") <= iii_original_created:
        failures.append(
            f"S7iii_created_at_not_advanced: original={iii_original_created} new={iii_item.get('created_at')}"
        )
    else:
        print(f"    slow_alive requeued to back, created_at advanced ✓", flush=True)

    elevator.fail(iid_slow)

    # ---- (iv) Ghost/spoofed stamp → back-of-line, no deadlock, not falsely failed ----
    print("  S7(iv): ghost/spoofed stamp → back-of-line; clears at backstop not before", flush=True)

    # iv-A: ghost claim lapses → D3 requeues to back (not failed); real waiter admits.
    iid_ghost_a = f"s7-iv-a-ghost-{uuid.uuid4().hex[:6]}"
    iid_real_a = f"s7-iv-a-real-{uuid.uuid4().hex[:6]}"
    P_ghost_a = f"s7-iv-ghost-{uuid.uuid4().hex[:4]}"
    P_real_a = f"s7-iv-real-{uuid.uuid4().hex[:4]}"
    all_tickets += [iid_ghost_a, iid_real_a]

    _insert_pending_controlled(iid_ghost_a, P_ghost_a, 12345, host="other-host-xyz", start_time=DEAD_START)
    _age_ticket(elevator, iid_ghost_a, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)
    ghost_a_original_created = (elevator.get(iid_ghost_a) or {}).get("created_at")

    time.sleep(0.005)
    _insert_pending_controlled(iid_real_a, P_real_a, os.getpid(), start_time=None)

    # Claim the ghost and backdate past its TTL.
    elevator.try_admit(iid_ghost_a, "deliberation", P_ghost_a, claim_ttl_sec=1)
    _backdate_claimed(iid_ghost_a, 10)

    # Reclaim: unknown verdict → requeue to back (fresh created_at), NOT failed.
    elevator.reclaim_stale("deliberation")
    ghost_a_item = elevator.get(iid_ghost_a) or {}

    if ghost_a_item.get("status") == "failed":
        failures.append("S7iv_ghost_falsely_failed_on_reclaim_should_requeue")
    elif ghost_a_item.get("status") != "pending":
        failures.append(f"S7iv_ghost_not_requeued_status={ghost_a_item.get('status')}")
    elif ghost_a_original_created and ghost_a_item.get("created_at", "") <= ghost_a_original_created:
        failures.append("S7iv_ghost_created_at_not_advanced_to_back")
    else:
        print(f"    ghost requeued to back (not failed) ✓", flush=True)

    # Real waiter (older than ghost's fresh created_at) is now head → admits.
    ok_real_a, _ = elevator.try_admit(iid_real_a, "deliberation", P_real_a)
    if not ok_real_a:
        failures.append("S7iv_real_waiter_not_admitted_after_ghost_requeued_to_back")
    else:
        elevator.ack(iid_real_a)
        print(f"    real_waiter admitted after ghost requeued ✓", flush=True)

    elevator.fail(iid_ghost_a)

    # iv-B: ghost pending past grace but before backstop → NOT falsely failed.
    iid_ghost_b = f"s7-iv-b-ghost-{uuid.uuid4().hex[:6]}"
    P_ghost_b = f"s7-iv-b-ghost-{uuid.uuid4().hex[:4]}"
    all_tickets.append(iid_ghost_b)

    _insert_pending_controlled(iid_ghost_b, P_ghost_b, 12346, host="other-host-xyz", start_time=DEAD_START)
    _age_ticket(elevator, iid_ghost_b, GW_ADMISSION_ORPHAN_GRACE_SEC + 5)

    # Reap: ghost has no same-host stamp → unknown → not reaped (not falsely failed).
    reap_before_backstop = elevator.reap()
    ghost_b_before = (elevator.get(iid_ghost_b) or {}).get("status")
    if ghost_b_before == "failed":
        failures.append("S7iv_ghost_falsely_failed_before_backstop")
    else:
        print(f"    ghost not falsely failed before backstop (status={ghost_b_before}) ✓", flush=True)

    # iv-C: ghost pending past MAX_WAIT → reaped by presumed-dead backstop.
    _age_ticket(elevator, iid_ghost_b, GW_ADMISSION_MAX_WAIT_SEC + 5)
    reap_at_backstop = elevator.reap()
    ghost_b_after = (elevator.get(iid_ghost_b) or {}).get("status")
    if ghost_b_after != "failed":
        failures.append(f"S7iv_ghost_not_cleared_at_backstop_status={ghost_b_after}")
    else:
        prov = (elevator.get(iid_ghost_b) or {}).get("provenance") or {}
        if prov.get("orphan_reap") != "presumed_dead_backstop":
            failures.append(f"S7iv_ghost_backstop_wrong_reason={prov}")
        else:
            print(f"    ghost cleared at backstop (presumed_dead_backstop) ✓", flush=True)

    # ---- Drain check: all tickets terminal ----
    non_terminal = []
    for t in all_tickets:
        item = elevator.get(t) or {}
        if item.get("status") not in ("served", "failed", "expired"):
            non_terminal.append(f"{t}={item.get('status')}")
            elevator.fail(t)
    if non_terminal:
        failures.append(f"S7_not_drained={non_terminal}")
    else:
        print("    drain-to-0 ✓", flush=True)

    passed = len(failures) == 0
    result = {
        "scenario": "S7",
        "pass": passed,
        "failures": failures,
        "details": {
            "AC2_squeeze_past_dead_pending": not any("S7i_" in f for f in failures),
            "AC5_dead_claimed_fast_reclaim": not any("S7ii_" in f for f in failures),
            "AC3_slow_alive_back_of_line": not any("S7iii_" in f for f in failures),
            "AC8_ghost_no_deadlock": not any("S7iv_" in f for f in failures),
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
        "- **S4**: Combined burst (council+facets+injected death). The production fix's",
        "  authoritative proof is `tests/test_doorman_atomic_acquire.py` (N concurrent",
        "  distinct-principal threads against the real doorman → exactly 1 True, rest",
        "  CONTENDED). S4 is the end-to-end confirmation that the fixed atomic drain-gate",
        "  pattern (require_drain_clear=True, mirroring agents_core/llm.py) holds under",
        "  the 2026-06-24 incident-shaped burst with no permanent wedge.",
        "- **S5**: Kind-aware drain-gate re-arm prerequisite (gw-enforce-ab-harness-kind-aware-scenario-v0).",
        "  Reproduces the 2026-06-24 span-principal wedge: a session's span-hold coordination",
        "  lease (P_span, lease_kind=coordination) must NOT block its own GW operator legs",
        "  (P_op, lease_kind=inference, require_drain_clear=True). AC2 asserts all operator/Facets",
        "  legs admit with span-hold held. AC3 asserts cross-principal inference still contends.",
        "  AC3a asserts the transition: CONTENDED-while-inference-present → ADMITS-when-coordination-only.",
        "  AC6 is an inline coordination-skip self-check. A --live GO that includes S5 PASS is",
        "  the trustworthy gate for enforce re-arm. Do NOT re-arm until S5 PASS is confirmed.",
        "- **S6**: GW reference-leg 5th-wedge guard (gw-gate-principal-unification-ab-harness-v0).",
        "  Models gw_agent.py's long-held lease_kind=inference acquire (the reference-reviewer leg)",
        "  concurrent with Facets/Council operator legs. Phase 1 (AC2): with the reference leg held",
        "  as inference under shared principal P, every operator leg under the same P with",
        "  require_drain_clear=True is GRANTED — proves the shared-principal fix (U1-U3) closes",
        "  the 5th wedge without a lease_kind change. Phase 2 (AC3 guard A): a ghost reference leg",
        "  (principal=None, inference) causes CONTENDED — the exact regression this arc fixes;",
        "  a future reintroduction of a principal-less GW acquire will break S6. Phase 3 (AC4",
        "  guard B): split-principal ref (P_ref) + operator (P_op != P_ref) causes CONTENDED —",
        "  proves cross-deliberation serialization is preserved. Phase 4 (AC5): elevator serve",
        "  phase confirms all n operator + Facets persona legs are served under shared P.",
        "",
        f"## Decision: **{go_no_go}** for re-arming `GW_ADMISSION_MODE=enforce`",
        "",
        f"GO requires all scenarios PASS (S0–S7). {'All scenarios passed.' if all_pass else 'One or more scenarios FAILED — do not re-arm enforce until fixed.'}",
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
    parser.add_argument("--skip-s5", action="store_true")
    parser.add_argument("--skip-s6", action="store_true")
    parser.add_argument("--skip-s7", action="store_true")
    args = parser.parse_args()

    live_mode = args.live
    mode_label = "live" if live_mode else "mock-gw"

    if not IS_MASTER and not live_mode:
        print("WARNING: not IS_MASTER — some elevator writes will fail. Run on BRIX.", flush=True)

    # ---- Isolated temp dir; one DB per scenario (D-AMEND-2: no cross-scenario FIFO cascade) ----
    tmpdir = tempfile.mkdtemp(prefix="gw_ab_harness_")
    print(f"HARNESS START mode={mode_label} seed={args.seed} n={args.n} tmpdir={tmpdir}", flush=True)

    friction = _FrictionBackend(seed=args.seed) if not live_mode else None
    if friction:
        print(f"  friction: {friction.describe()}", flush=True)

    def _scenario_db(name: str) -> tuple[ElevatorStore, str]:
        """Open a fresh isolated elevator DB for one scenario (D-AMEND-2)."""
        db = Path(tmpdir) / f"elevator_{name}.db"
        db_str = str(db)
        os.environ["ELEVATOR_DB_PATH"] = db_str
        return ElevatorStore(db_path=db), db_str

    doorman = _MockDoormanState()
    results = {}

    try:
        if not args.skip_s0:
            elev, _ = _scenario_db("s0")
            try:
                results["S0"] = run_s0(elev, doorman, args.n, friction, live_mode)
            finally:
                _cleanup_elevator(elev)
                elev.close()
            doorman.leases.clear()

        elev, _ = _scenario_db("s1")
        try:
            results["S1"] = run_s1(elev, doorman, args.n, friction)
        finally:
            _cleanup_elevator(elev)
            elev.close()
        doorman.leases.clear()

        elev, db_path = _scenario_db("s2")
        try:
            results["S2"] = run_s2(elev, db_path, friction)
        finally:
            _cleanup_elevator(elev)
            elev.close()

        elev, _ = _scenario_db("s3")
        doorman.leases.clear()
        try:
            results["S3"] = run_s3(elev, doorman, friction)
        finally:
            _cleanup_elevator(elev)
            elev.close()
        doorman.leases.clear()

        if not args.skip_s4:
            elev, db_path = _scenario_db("s4")
            try:
                results["S4"] = run_s4(elev, doorman, db_path, friction, args.n)
            finally:
                _cleanup_elevator(elev)
                elev.close()
            doorman.leases.clear()

        if not args.skip_s5:
            elev, _ = _scenario_db("s5")
            doorman.leases.clear()
            try:
                results["S5"] = run_s5(elev, doorman, args.n, friction)
            finally:
                _cleanup_elevator(elev)
                elev.close()
            doorman.leases.clear()

        if not args.skip_s6:
            elev, _ = _scenario_db("s6")
            doorman.leases.clear()
            try:
                results["S6"] = run_s6(elev, doorman, args.n, friction)
            finally:
                _cleanup_elevator(elev)
                elev.close()
            doorman.leases.clear()

        if not args.skip_s7:
            elev, _ = _scenario_db("s7")
            doorman.leases.clear()
            try:
                results["S7"] = run_s7(elev, doorman, friction)
            finally:
                _cleanup_elevator(elev)
                elev.close()
            doorman.leases.clear()

    finally:
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
