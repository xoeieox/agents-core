"""Tests for the FIXER_MAX_CONCURRENT sub-cap in claude_queue_runner.

Mirrors the council sub-cap precedent (tests/test_council_queue_runner.py,
tests/test_claude_queue_runner.py council-cap cases): module-monkeypatch
fixtures, async tests, explicit coverage of
- default wiring (cap + _FIXER_SEM._value pair)
- env wiring via subprocess-isolated import (override + invalid fallback)
- Semaphore ordering (_FIXER_SEM acquired before self.sem)
- Concurrency cap (subset cap: restricts, never expands past the global)
- Cross-family non-interference (reviewer / generic subprocess / council)
- _is_fixer_task predicate (incl. the council.run description-collision case)
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

import agents_core.claude_queue_runner as runner_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_type="subprocess", task_id="test-task-001",
               description=None, **kwargs):
    task = {
        "id": task_id,
        "task_type": task_type,
        "timeout_seconds": 300,
        "notify": False,
        "payload": {"spec_path": "/fake/spec.yaml"},
    }
    if description is not None:
        task["description"] = description
    task.update(kwargs)
    return task


def _import_time_fixer_cap(env_value: str | None) -> tuple[int, int]:
    """Import claude_queue_runner fresh in a subprocess under a controlled
    FIXER_MAX_CONCURRENT env, and report (FIXER_MAX_CONCURRENT,
    _FIXER_SEM._value). Subprocess isolation (rather than importlib.reload
    in-process) avoids leaving a second Daemon/Semaphore class generation
    behind for every other test module that imported this one at collection
    time. Cloned from _import_time_council_cap (test_claude_queue_runner.py)."""
    env = dict(os.environ)
    if env_value is None:
        env.pop("FIXER_MAX_CONCURRENT", None)
    else:
        env["FIXER_MAX_CONCURRENT"] = env_value
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, agents_core.claude_queue_runner as m; "
            "print(json.dumps([m.FIXER_MAX_CONCURRENT, m._FIXER_SEM._value]))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return tuple(json.loads(proc.stdout.strip().splitlines()[-1]))


async def _poll_until(pred, timeout: float = 2.0, step: float = 0.01) -> bool:
    """Poll pred() every step seconds until True or timeout. Returns whether
    pred() went True in time. (sem._value / stub-entry observables only —
    see the sem.locked() caveat in test_fixer_task_acquires_fixer_sem_before_global.)"""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(step)
    return False


# ---------------------------------------------------------------------------
# Cases 1-2: default wiring + env wiring (mirror the council precedent)
# ---------------------------------------------------------------------------

def test_fixer_default_is_5():
    """FIXER_MAX_CONCURRENT and _FIXER_SEM are module-level, parsed once at
    import. In this process (no FIXER_MAX_CONCURRENT set at import time) the
    default is 5 — proving the literal-default semaphore has been wired to
    the configurable cap, not hardcoded. Delenving after import cannot change
    an already-computed module constant; env behavior is pinned by the
    subprocess-isolated test below (mirror: test_council_max_concurrent_default_wiring)."""
    assert runner_mod.FIXER_MAX_CONCURRENT == 5
    assert runner_mod._FIXER_SEM._value == 5


def test_fixer_env_wiring_via_subprocess():
    """FIXER_MAX_CONCURRENT is honoured at import time in a fresh process,
    and invalid/<1 values fall back to 1 (helper semantics, same as council).
    Mirror: test_council_max_concurrent_honours_env_override /
    ..._invalid_env_falls_back_to_1."""
    cap, sem_value = _import_time_fixer_cap("5")
    assert (cap, sem_value) == (5, 5)
    cap, sem_value = _import_time_fixer_cap("0")
    assert (cap, sem_value) == (1, 1)
    cap, sem_value = _import_time_fixer_cap("abc")
    assert (cap, sem_value) == (1, 1)


# ---------------------------------------------------------------------------
# Case 3: _FIXER_SEM acquired BEFORE self.sem (exact 9-step protocol)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fixer_task_acquires_fixer_sem_before_global(monkeypatch):
    """A claimed fixer task waiting on _FIXER_SEM holds NO global slot.

    EXACT protocol (spec Part D case 3):
    (i)   Daemon(workers=2) — real ClaudeQueue() (harmless mkdir on the live
          /srv/lapis/claude-queue dirs; the _make_queue mock is for _run_task-path
          tests only);
    (ii)  monkeypatch runner_mod._run_task with an async stub gated on an
          asyncio.Event (enter the stub, record entry, then await the event);
    (iii) monkeypatch runner_mod._FIXER_SEM = asyncio.Semaphore(1);
    (iv)  create task 1 (description="fixer:t1") and start d._worker(task1);
    (v)   await until _FIXER_SEM.locked() is True AND the stub has been
          entered (task 1 holds both sems);
    (vi)  baseline = d.sem._value (== 1 for workers=2);
    (vii) create task 2 (description="fixer:t2") and start d._worker(task2);
    (viii) yield until task 2 is blocked on _FIXER_SEM (short sleep poll,
          council-harness style);
    (ix)  assert d.sem._value == baseline (task 2 holds NO global slot).

    OBSERVABLE: sem._value (already established in-suite,
    test_claude_queue_runner.py:735/:760). Do NOT use sem.locked() as the
    assertion — on the host's Python 3.12, locked() returns True whenever
    ANY waiter exists even with _value > 0
    (/usr/lib/python3.12/asyncio/locks.py:358-361).
    """
    # (i)
    d = runner_mod.Daemon(workers=2)

    # (ii)
    entered = asyncio.Event()
    release = asyncio.Event()
    entries = []

    async def _stub(queue, task):
        entries.append(task["id"])
        entered.set()
        await release.wait()

    monkeypatch.setattr(runner_mod, "_run_task", _stub)

    # (iii)
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(1))

    # (iv)
    task1 = _make_task(description="fixer:t1", task_id="fixer-t1")
    w1 = asyncio.create_task(d._worker(task1))

    # (v)
    assert await _poll_until(lambda: runner_mod._FIXER_SEM.locked() and entered.is_set()), \
        "task 1 must hold both _FIXER_SEM and the global slot (stub entered)"

    # (vi)
    baseline = d.sem._value
    assert baseline == 1, f"workers=2 with task 1 holding a slot -> _value 1, got {baseline}"

    # (vii)
    task2 = _make_task(description="fixer:t2", task_id="fixer-t2")
    w2 = asyncio.create_task(d._worker(task2))

    # (viii) yield until task 2 is blocked on _FIXER_SEM (short sleep poll)
    await asyncio.sleep(0.05)

    # (ix)
    assert d.sem._value == baseline, (
        f"task 2 must hold NO global slot while waiting on _FIXER_SEM "
        f"(baseline={baseline}, got {d.sem._value})"
    )
    assert entries == ["fixer-t1"], "only task 1 may have entered the stub"

    release.set()
    await asyncio.gather(w1, w2)
    assert entries == ["fixer-t1", "fixer-t2"], "task 2 runs after task 1 releases"


# ---------------------------------------------------------------------------
# Case 4: fixer_retry is subject to the same sub-cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fixer_retry_task_capped(monkeypatch):
    d = runner_mod.Daemon(workers=2)

    entered = asyncio.Event()
    release = asyncio.Event()
    entries = []

    async def _stub(queue, task):
        entries.append(task["id"])
        entered.set()
        await release.wait()

    monkeypatch.setattr(runner_mod, "_run_task", _stub)
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(1))

    task1 = _make_task(description="fixer_retry:t1", task_id="frt-t1")
    task2 = _make_task(description="fixer_retry:t2", task_id="frt-t2")
    w1 = asyncio.create_task(d._worker(task1))
    assert await _poll_until(lambda: runner_mod._FIXER_SEM.locked() and entered.is_set())
    baseline = d.sem._value

    w2 = asyncio.create_task(d._worker(task2))
    await asyncio.sleep(0.05)

    # Second fixer_retry waits on _FIXER_SEM: holds no global slot, did not run.
    assert d.sem._value == baseline
    assert entries == ["frt-t1"]

    release.set()
    await asyncio.gather(w1, w2)
    assert entries == ["frt-t1", "frt-t2"]


# ---------------------------------------------------------------------------
# Case 5: reviewer bypasses _FIXER_SEM entirely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reviewer_task_not_capped(monkeypatch):
    d = runner_mod.Daemon(workers=2)

    release = asyncio.Event()
    entries = []

    async def _stub(queue, task):
        entries.append(task["id"])
        await release.wait()

    monkeypatch.setattr(runner_mod, "_run_task", _stub)
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(1))

    # Saturate _FIXER_SEM with a fixer task...
    fixer = _make_task(description="fixer:t1", task_id="rv-t1")
    w_fixer = asyncio.create_task(d._worker(fixer))
    assert await _poll_until(lambda: runner_mod._FIXER_SEM.locked() and "rv-t1" in entries), \
        "_FIXER_SEM must be saturated by the fixer task"

    # ...a reviewer task must STILL acquire self.sem and run.
    reviewer = _make_task(description="reviewer:t1", task_id="rv-t2")
    w_reviewer = asyncio.create_task(d._worker(reviewer))
    assert await _poll_until(lambda: "rv-t2" in entries), \
        "reviewer task must not be gated by _FIXER_SEM"

    release.set()
    await asyncio.gather(w_fixer, w_reviewer)
    assert set(entries) == {"rv-t1", "rv-t2"}


# ---------------------------------------------------------------------------
# Case 6: council path unchanged; cross-interference is one-directional
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_council_path_unchanged(monkeypatch):
    """council still acquires _COUNCIL_SEM before self.sem; a saturated
    _COUNCIL_SEM blocks council but NOT fixer or reviewer tasks.

    workers=4 so the one-directional-interference half has room: the two
    saturated sub-cap holders (council + fixer) occupy 2 of 4 global slots,
    leaving 2 for the fixer-family and reviewer tasks to prove they are NOT
    gated by the saturated _COUNCIL_SEM.
    """
    d = runner_mod.Daemon(workers=4)

    release = asyncio.Event()
    entries = []

    async def _stub(queue, task):
        entries.append(task["id"])
        await release.wait()

    monkeypatch.setattr(runner_mod, "_run_task", _stub)
    monkeypatch.setattr(runner_mod, "_COUNCIL_SEM", asyncio.Semaphore(1))
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(1))

    # Saturated _COUNCIL_SEM (council task 1) + saturated _FIXER_SEM (fixer task).
    c1 = _make_task(task_type="council.run", task_id="cp-c1")
    f1 = _make_task(description="fixer:t1", task_id="cp-f1")
    w_c1 = asyncio.create_task(d._worker(c1))
    w_f1 = asyncio.create_task(d._worker(f1))
    assert await _poll_until(
        lambda: "cp-c1" in entries and "cp-f1" in entries,
        timeout=3.0,
    ), "both sub-cap tasks must be running (council on _COUNCIL_SEM, fixer on _FIXER_SEM)"
    assert runner_mod._COUNCIL_SEM.locked() and runner_mod._FIXER_SEM.locked()

    # Second council task: blocked on _COUNCIL_SEM, holds no global slot.
    c2 = _make_task(task_type="council.run", task_id="cp-c2")
    w_c2 = asyncio.create_task(d._worker(c2))
    await asyncio.sleep(0.05)
    assert "cp-c2" not in entries, "council task 2 must be waiting on _COUNCIL_SEM"
    assert d.sem._value == 3, (
        f"blocked council task must hold no global slot (4 - 2 held = 3, got {d.sem._value})"
    )

    # ...but a fixer-family and a reviewer task still run (one-directional
    # interference: each sub-cap only gates its own family). The fixer task is
    # additionally held by the saturated _FIXER_SEM — its global-slot claim is
    # observed via sem._value, not via the stub entry (which requires the
    # sub-cap). The reviewer task has no sub-cap: it must enter the stub.
    f2 = _make_task(description="fixer:t2", task_id="cp-f2")
    r1 = _make_task(description="reviewer:t1", task_id="cp-r1")
    w_f2 = asyncio.create_task(d._worker(f2))
    w_r1 = asyncio.create_task(d._worker(r1))
    assert await _poll_until(lambda: "cp-r1" in entries and d.sem._value == 1), \
        "reviewer must run and the fixer must hold a global slot despite the saturated _COUNCIL_SEM"

    release.set()
    await asyncio.gather(w_c1, w_f1, w_c2, w_f2, w_r1)
    assert set(entries) == {"cp-c1", "cp-c2", "cp-f1", "cp-f2", "cp-r1"}


# ---------------------------------------------------------------------------
# Case 7: subset cap can never expand beyond the global
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fixer_cap_cannot_expand_beyond_global(monkeypatch):
    """FIXER_MAX_CONCURRENT effectively 4 with Daemon(workers=2): at most 2
    fixer tasks run concurrently — the global self.sem stays the ceiling."""
    d = runner_mod.Daemon(workers=2)
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(4))

    peak = [0]
    current = [0]
    release = asyncio.Event()

    async def _stub(queue, task):
        current[0] += 1
        peak[0] = max(peak[0], current[0])
        await release.wait()
        current[0] -= 1

    monkeypatch.setattr(runner_mod, "_run_task", _stub)

    workers = [
        asyncio.create_task(d._worker(_make_task(description=f"fixer:t{i}", task_id=f"cap-t{i}")))
        for i in range(4)
    ]
    assert await _poll_until(lambda: current[0] == 2, timeout=3.0), \
        "two fixer tasks must be running (global workers=2)"
    await asyncio.sleep(0.05)
    assert peak[0] == 2, f"at most 2 fixer tasks may run concurrently, got peak={peak[0]}"

    release.set()
    await asyncio.gather(*workers)


# ---------------------------------------------------------------------------
# Case 8: generic (non-fixer) subprocess bypasses _FIXER_SEM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_fixer_subprocess_bypass(monkeypatch):
    d = runner_mod.Daemon(workers=2)

    release = asyncio.Event()
    entries = []

    async def _stub(queue, task):
        entries.append(task["id"])
        await release.wait()

    monkeypatch.setattr(runner_mod, "_run_task", _stub)
    monkeypatch.setattr(runner_mod, "_FIXER_SEM", asyncio.Semaphore(1))

    fixer = _make_task(description="fixer:t1", task_id="np-f1")
    w_fixer = asyncio.create_task(d._worker(fixer))
    assert await _poll_until(lambda: runner_mod._FIXER_SEM.locked() and "np-f1" in entries)

    # Generic subprocess with no fixer prefix: runs with the sub-cap saturated.
    generic = _make_task(description="smoke-test fake job", task_id="np-g1")
    w_generic = asyncio.create_task(d._worker(generic))
    assert await _poll_until(lambda: "np-g1" in entries), \
        "non-fixer subprocess must not be gated by _FIXER_SEM"

    release.set()
    await asyncio.gather(w_fixer, w_generic)
    assert set(entries) == {"np-f1", "np-g1"}


# ---------------------------------------------------------------------------
# Case 9: _is_fixer_task predicate unit checks
# ---------------------------------------------------------------------------

def test_fixer_predicate_helper():
    assert runner_mod._is_fixer_task(_make_task(description="fixer:x")) is True
    assert runner_mod._is_fixer_task(_make_task(description="fixer_retry:x")) is True
    assert runner_mod._is_fixer_task(_make_task(description="reviewer:x")) is False
    # Predicate-level council exclusion: council descriptions are free-form
    # decision text (council/cli.py:1986-1993) and can legally start with
    # "fixer:" — this colliding case is constructible in production.
    assert runner_mod._is_fixer_task(
        _make_task(task_type="council.run", description="fixer:probe")
    ) is False
    # Missing description.
    assert runner_mod._is_fixer_task(_make_task()) is False
    # Prefix must include the colon.
    assert runner_mod._is_fixer_task(_make_task(description="fixerx:x")) is False
