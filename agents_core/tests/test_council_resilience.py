"""Tests for spec-review-council-keepawake-resilience-v0, extended by

agents-core-council-liveness-queue-aware-v0.

Covers:
  AC1  — Deliberation-spanning GW keepawake hold: placed at start, heartbeat-coupled
         TTL so zombie hold auto-expires (no unconditional pinning).
  AC2  — _poll_council fast-fails within COUNCIL_STALL_S on stale heartbeat (not 1800s).
  AC2b — Queue-aware liveness ladder: queued != dead. A run still waiting in
         ClaudeQueue's serial queue (single-model GW, --parallel 1) must not
         read as a dead worker. started_at distinguishes running-pre-heartbeat
         from still-queued; COUNCIL_STARTUP_GRACE_S is the startup-grace clock,
         separate from COUNCIL_STALL_S.
  AC3  — Bounded retry (once) on a liveness fast-fail, not on terminal status or
         a genuine timeout_s backstop; structured failure signal is well-formed
         after retries are exhausted. Revises AC3's prior "never retry" rule.
  AC4  — Happy path: healthy deliberation still reaches terminal status (stub regression).
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, call, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# AC1 — Deliberation-spanning hold: TTL and heartbeat-coupling
# ---------------------------------------------------------------------------

def test_deliberation_hold_uses_council_stall_s_as_ttl(monkeypatch, tmp_path):
    """Initial hold acquire uses COUNCIL_STALL_S as lease TTL (zombie-hold guard).

    A dead worker's hold auto-expires after COUNCIL_STALL_S, not infinity.
    """
    from agents_core.council import cli

    monkeypatch.setenv("COUNCIL_STALL_S", "42")
    # Re-read constant (it's read at import time, so patch the module attr)
    monkeypatch.setattr(cli, "COUNCIL_STALL_S", 42)

    run_id = "2026-01-01-000000-hold01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test",
        "voicing": "gravitywell",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file.write_text(yaml.safe_dump(run_data))

    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving"}

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    # DoormanClient is a local import inside run_deliberation — patch at source module
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=210.0):
        cli.run_deliberation(run_id)

    # First acquire call should use COUNCIL_STALL_S (42) as ttl_sec
    first_call = mock_client.acquire.call_args_list[0]
    ttl_used = first_call.kwargs.get("ttl_sec") or (first_call.args[2] if len(first_call.args) > 2 else None)
    assert ttl_used == 42, f"Expected ttl_sec=42 on initial hold acquire, got: {first_call}"
    # Hold must be released on exit
    mock_client.release.assert_called_once()

    run_file.unlink(missing_ok=True)


def test_deliberation_hold_released_on_stub_exit(monkeypatch):
    """Deliberation hold is released via finally block even on stub-mode return."""
    from agents_core.council import cli

    run_id = "2026-01-01-000000-hold02"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test",
        "voicing": "gravitywell",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving"}

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=210.0):
        cli.run_deliberation(run_id)

    mock_client.release.assert_called_once_with("gravitywell", f"council-delib-{run_id}")
    mock_client.close.assert_called_once()

    run_file.unlink(missing_ok=True)


def test_deliberation_hold_not_acquired_for_non_gw_voicing(monkeypatch):
    """No deliberation hold acquired when voicing != gravitywell."""
    from agents_core.council import cli

    run_id = "2026-01-01-000000-hold03"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test",
        "voicing": "local",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    with patch("agents_core.doorman_client.DoormanClient") as mock_dc_cls:
        cli.run_deliberation(run_id)

    # DoormanClient should not be instantiated when voicing != gravitywell
    mock_dc_cls.assert_not_called()

    run_file.unlink(missing_ok=True)


def test_no_per_voice_gw_rewake(monkeypatch):
    """Single deliberation-spanning hold prevents per-voice GW re-wake.

    Simulates NUM_STEPS engine turns via a fake Engine. DoormanClient.acquire
    must be called exactly once (initial hold) + NUM_STEPS (heartbeat refreshes).
    No additional per-voice re-wake calls — the deliberation-spanning hold keeps
    GW warm across all voices without re-acquiring per turn.

    AC1: 'a test asserts GW is not re-woken per-voice (single ensure_serving + held)'
    """
    import sys
    from agents_core.council import cli

    NUM_STEPS = 3
    run_id = "2026-01-01-000000-norestart01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test no per-voice re-wake",
        "voicing": "gravitywell",
        "turns_cap": NUM_STEPS,
        "turns": [],
        "selected_entities": [{"id": "char1"}, {"id": "char2"}],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    class _FakeEvent:
        type = "deliberation_turn"

    class _FakeStepData:
        def __init__(self, n):
            self.step = n
            self.acting_entity_id = "char1"
            self.response = f"turn {n} content"
            self.events = [_FakeEvent()]

    class _FakeEngine:
        def run(self, director, entities, on_step):
            for n in range(1, NUM_STEPS + 1):
                on_step(_FakeStepData(n))

    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving"}

    fake_lapis_mod = MagicMock()
    fake_lapis_mod.Engine.return_value = _FakeEngine()

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=210.0), \
         patch.dict(sys.modules, {
             "lapis_engine": fake_lapis_mod,
             "archetypes": MagicMock(),
             "archetypes.engine": MagicMock(),
             "archetypes.engine.character_entity": MagicMock(),
             "agents_core.council.narrator_entity": MagicMock(),
         }), \
         patch.object(cli, "_build_adapter", return_value=MagicMock()), \
         patch.object(cli, "_build_entity", return_value=MagicMock()), \
         patch.object(cli, "_build_director", return_value=MagicMock()), \
         patch.object(cli, "_apply_voicing_provenance"), \
         patch.object(cli, "_emit_voicing_signal"), \
         patch.object(cli, "_apply_position_cast_tail"), \
         patch.object(cli, "_parse_synthesis", return_value={}), \
         patch.object(cli, "_calculate_paid_spend", return_value=False):
        cli.run_deliberation(run_id)

    acquire_calls = mock_client.acquire.call_args_list
    # 1 initial hold + NUM_STEPS heartbeat refreshes = NUM_STEPS + 1 total.
    # Any higher count would indicate per-voice re-wake calls.
    assert len(acquire_calls) == 1 + NUM_STEPS, (
        f"Expected 1 initial hold + {NUM_STEPS} heartbeat-refresh acquires "
        f"(={1 + NUM_STEPS} total), got {len(acquire_calls)}: {acquire_calls}"
    )

    # First call is the deliberation-level hold, not a per-voice re-wake
    initial_reason = acquire_calls[0].kwargs.get("reason")
    assert initial_reason == "council-deliberation-hold", (
        f"Initial acquire should be deliberation hold, got reason={initial_reason!r}"
    )

    # All subsequent calls are heartbeat refreshes (per step, not per voice start)
    for i, c in enumerate(acquire_calls[1:], 1):
        reason = c.kwargs.get("reason")
        assert reason == "council-deliberation-heartbeat", (
            f"Acquire call {i + 1} should be heartbeat refresh (not per-voice re-wake), "
            f"got reason={reason!r}"
        )

    mock_client.release.assert_called_once()
    run_file.unlink(missing_ok=True)


def test_deliberation_hold_failure_does_not_abort(monkeypatch):
    """A failed hold acquire logs a warning but does not abort the deliberation."""
    from agents_core.council import cli

    run_id = "2026-01-01-000000-hold04"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test",
        "voicing": "gravitywell",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    mock_client = MagicMock()
    mock_client.acquire.side_effect = Exception("doorman down")

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=210.0):
        # Must not raise even though hold failed
        cli.run_deliberation(run_id)

    run = yaml.safe_load(run_file.read_text())
    assert run["status"] in ("resolved", "open", "laid-down", "closed")

    run_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC2 — _poll_council fast-fail on stale heartbeat / no heartbeat
#
# NOTE on isolation: room_path("council") is resolved fresh on every
# _poll_council call from ROOM_ROOT (env, default /room) + "council", so
# pointing ROOM_ROOT at tmp_path is what actually isolates these tests from
# real /srv/lapis/council data (a prior version of these tests patched
# orchestrator.Path, which room_path() never consults — dead mock, silently
# reading real production /srv/lapis/council and hanging to the outer timeout).
# ---------------------------------------------------------------------------

def _install_council_run(tmp_path, monkeypatch, run_id, run_data):
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    council_dir = tmp_path / "council"
    council_dir.mkdir(parents=True, exist_ok=True)
    (council_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))


def _mock_queue(get_pending=None, get_active=None, side_effect=None):
    """Build a patch context for agents_core.claude_queue.ClaudeQueue.

    _queue_task_alive does `from agents_core.claude_queue import ClaudeQueue`
    lazily inside the function, so patching the class at its source module is
    picked up on every call regardless of import order.
    """
    mock_cls = MagicMock()
    if side_effect is not None:
        mock_cls.side_effect = side_effect
    else:
        mock_instance = MagicMock()
        mock_instance.get_pending.return_value = get_pending or []
        mock_instance.get_active.return_value = get_active or []
        mock_cls.return_value = mock_instance
    return patch("agents_core.claude_queue.ClaudeQueue", mock_cls), mock_cls


def test_poll_council_fast_fails_on_stale_heartbeat(tmp_path, monkeypatch):
    """_poll_council fast-fails when heartbeat_at is stale beyond COUNCIL_STALL_S.

    Should return within COUNCIL_STALL_S, not wait for timeout_s (1800s).
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-stale01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        "heartbeat_at": "2020-01-01T00:00:01",  # very stale
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STALL_S", "5")

    t0 = time.time()
    data, error = orchestrator._poll_council(run_id, timeout_s=3600)
    elapsed = time.time() - t0

    assert data is None
    assert error is not None
    assert "died/stalled" in error
    assert run_id in error
    assert "heartbeat_stale" in error
    # Should fast-fail well within timeout_s
    assert elapsed < 30, f"Expected fast-fail, took {elapsed:.1f}s"


def test_poll_council_fast_fails_on_no_heartbeat_after_startup(tmp_path, monkeypatch):
    """_poll_council fast-fails when no heartbeat_at/started_at and the queue

    has no record of the task (never enqueued, or already reaped) — the
    dead-worker fallback path of the queue-aware ladder.
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-stale02"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        # No heartbeat_at, no started_at — worker died before first turn/dequeue
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STARTUP_GRACE_S", "5")

    queue_patch, _ = _mock_queue(get_pending=[], get_active=[])
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=3600)

    assert data is None
    assert error is not None
    assert "died/stalled" in error
    assert run_id in error
    assert "no_heartbeat_after_startup" in error


def test_poll_council_does_not_fast_fail_with_fresh_heartbeat(tmp_path, monkeypatch):
    """_poll_council does not fast-fail when heartbeat_at is recent."""
    from agents_core.shared_deliberation import orchestrator
    from datetime import datetime

    run_id = "2026-01-01-000000-fresh01"
    # heartbeat_at set to 'now' — should not trigger stall detection
    fresh_ts = datetime.now().isoformat(timespec="seconds")
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": fresh_ts,
        "heartbeat_at": fresh_ts,
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    # Tiny timeout so the test completes quickly
    monkeypatch.setenv("COUNCIL_STALL_S", "300")

    # timeout_s=1 so we exit via wall-clock timeout, NOT stall detection
    data, error = orchestrator._poll_council(run_id, timeout_s=1)

    # Must time-out normally, not produce a stale-heartbeat error
    assert data is None
    assert error is not None
    assert "timeout" in error
    assert "died/stalled" not in error


def test_poll_council_returns_terminal_status(tmp_path, monkeypatch):
    """_poll_council returns council data when YAML reaches a terminal status."""
    from agents_core.shared_deliberation import orchestrator
    from datetime import datetime

    run_id = "2026-01-01-000000-term01"
    run_data = {
        "run_id": run_id,
        "status": "resolved",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "voicing": "gravitywell",
        "selection_degraded": False,
        "synthesis": {
            "landing": "all agree",
            "confidence": "converged",
            "open_questions": [],
            "positions": [],
        },
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)

    data, error = orchestrator._poll_council(run_id, timeout_s=10)

    assert error is None
    assert data is not None
    assert data["status"] == "resolved"
    assert data["landing"] == "all agree"


# ---------------------------------------------------------------------------
# AC2b — Queue-aware liveness ladder (agents-core-council-liveness-queue-aware-v0)
#
# Root cause: the poller was measuring worker-startup from enqueue-time
# created_at, so a run still waiting in ClaudeQueue's serial queue (GW
# --parallel 1) looked indistinguishable from a dead worker. Fix: queued !=
# dead. See orchestrator._poll_council docstring for the full ladder.
# ---------------------------------------------------------------------------

def test_poll_council_queued_and_pending_in_queue_does_not_fast_fail(tmp_path, monkeypatch):
    """Core regression: a queued (not yet started) run whose task is still

    pending in ClaudeQueue must NOT fast-fail, even with created_at aged well
    past COUNCIL_STALL_S. It keeps polling, bounded only by the timeout_s
    backstop — asserted here by driving to that backstop and checking the
    error is a plain timeout, not died/stalled.
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-queued01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",  # ancient — would trip old logic
        # No started_at, no heartbeat_at — still sitting in the serial queue
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STALL_S", "5")
    monkeypatch.setenv("COUNCIL_STARTUP_GRACE_S", "5")

    queue_patch, mock_cls = _mock_queue(get_pending=[{"id": run_id}], get_active=[])
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=1)

    assert data is None
    assert error is not None
    assert "timeout" in error
    assert "died/stalled" not in error
    mock_cls.assert_called()


def test_poll_council_started_at_within_grace_does_not_fast_fail(tmp_path, monkeypatch):
    """Running-pre-heartbeat, started_at recent → grace clock, no fast-fail.

    created_at is ancient (would trip the old created_at-based logic); only
    started_at should matter once the worker has actually dequeued.
    """
    from agents_core.shared_deliberation import orchestrator
    from datetime import datetime

    run_id = "2026-01-01-000000-grace01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        # No heartbeat_at yet
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)

    queue_patch, mock_cls = _mock_queue(side_effect=AssertionError("queue must not be consulted"))
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=1)

    assert data is None
    assert error is not None
    assert "timeout" in error
    assert "died/stalled" not in error
    # started_at present resolves rung 2 directly — the queue is never touched.
    mock_cls.assert_not_called()


def test_poll_council_started_at_past_grace_fast_fails(tmp_path, monkeypatch):
    """Running-pre-heartbeat, started_at aged past COUNCIL_STARTUP_GRACE_S → fast-fail."""
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-grace02"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        "started_at": "2020-01-01T00:00:01",
        # No heartbeat_at
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STARTUP_GRACE_S", "5")

    t0 = time.time()
    data, error = orchestrator._poll_council(run_id, timeout_s=3600)
    elapsed = time.time() - t0

    assert data is None
    assert error is not None
    assert "died/stalled" in error
    assert "no_heartbeat_after_startup" in error
    assert elapsed < 30, f"Expected fast-fail, took {elapsed:.1f}s"


def test_poll_council_queue_absent_task_uses_grace_fallback_from_created_at(tmp_path, monkeypatch):
    """Queued rung, queue reachable but task absent (never enqueued / reaped) →

    dead-worker fallback: COUNCIL_STARTUP_GRACE_S clock from created_at.
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-absent01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        # No started_at, no heartbeat_at
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STARTUP_GRACE_S", "5")

    queue_patch, mock_cls = _mock_queue(get_pending=[], get_active=[])
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=3600)

    assert data is None
    assert error is not None
    assert "died/stalled" in error
    assert "no_heartbeat_after_startup" in error
    mock_cls.assert_called()


def test_poll_council_queue_lookup_raises_degrades_gracefully(tmp_path, monkeypatch):
    """Hung/lying-queue guard (Facets Trickster + Council OQ#3): if the queue

    lookup itself raises on the queued rung, the poller must not propagate
    that as a poll error, and must not false-positive as died/stalled. It
    degrades to the tolerant grace fallback and stays bounded by the
    timeout_s backstop — a queue that never stops reporting "pending" (or
    can't be read at all) can at worst delay the verdict, never hang the gate.
    """
    from agents_core.shared_deliberation import orchestrator
    from datetime import datetime

    run_id = "2026-01-01-000000-hungqueue01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": datetime.now().isoformat(timespec="seconds"),  # recent
        # No started_at, no heartbeat_at — queued rung
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)

    queue_patch, mock_cls = _mock_queue(side_effect=RuntimeError("queue backend unreachable"))
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=1)

    assert data is None
    assert error is not None
    assert "timeout" in error
    assert "died/stalled" not in error
    mock_cls.assert_called()


def test_poll_council_queue_unavailable_started_at_present_is_tolerant(tmp_path, monkeypatch):
    """Graceful degrade: with started_at present (no heartbeat_at yet), the

    queue is irrelevant to the ladder decision — even a broken/unavailable
    ClaudeQueue must not cause a false-positive fast-fail against the tight
    COUNCIL_STALL_S clock. Same scenario as
    test_poll_council_started_at_within_grace_does_not_fast_fail, asserted
    from the "queue unavailable" angle named explicitly in the DoD.
    """
    from agents_core.shared_deliberation import orchestrator
    from datetime import datetime

    run_id = "2026-01-01-000000-degrade01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)
    monkeypatch.setenv("COUNCIL_STALL_S", "5")

    queue_patch, mock_cls = _mock_queue(side_effect=RuntimeError("queue import failure"))
    with queue_patch:
        data, error = orchestrator._poll_council(run_id, timeout_s=1)

    assert data is None
    assert error is not None
    assert "timeout" in error
    assert "died/stalled" not in error


def test_poll_council_absolute_backstop_bounds_queued_rung(tmp_path, monkeypatch):
    """Absolute timeout_s backstop is never exceeded, even when the queue

    keeps reporting the task as pending forever.
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-backstop01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
    }
    _install_council_run(tmp_path, monkeypatch, run_id, run_data)

    queue_patch, mock_cls = _mock_queue(get_pending=[{"id": run_id}], get_active=[])
    with queue_patch:
        t0 = time.time()
        data, error = orchestrator._poll_council(run_id, timeout_s=2)
        elapsed = time.time() - t0

    assert data is None
    assert "timeout" in error
    assert "died/stalled" not in error
    # Loop must exit at/near the timeout_s backstop, not hang indefinitely.
    assert elapsed < 30, f"Expected bounded exit near timeout_s=2, took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# AC3 — Bounded retry on liveness fast-fail; structured failure signal is
# well-formed after retries are exhausted (Part B,
# agents-core-council-liveness-queue-aware-v0).
#
# Deliberate invariant revision: the prior "submit exactly once, never retry"
# rule (spec-review-council-keepawake-resilience-v0 AC3) is intentionally
# replaced by "retry once, then fast-fail with the contract preserved" — a
# single liveness fast-fail can be a false positive from serial-queue
# contention on the single-model GW endpoint, so one bounded re-submit with a
# fresh run_id is defense-in-depth atop the queued != dead fix in AC2b.
# ---------------------------------------------------------------------------

def test_retry_once_then_fast_fail():
    """_council_subprocess retries once on a liveness fast-fail, then returns

    the failure. _submit_council is called twice (fresh run_id on retry); the
    final error still carries run_id/last_heartbeat/reason (Unit 2 contract).
    """
    from agents_core.shared_deliberation import orchestrator
    import asyncio

    run_id_1 = "2026-01-01-000000-retry01a"
    run_id_2 = "2026-01-01-000000-retry01b"

    def fake_error(rid):
        return (
            f"council worker died/stalled "
            f"(run_id={rid}, last_heartbeat='2020-01-01T00:00:01', "
            f"reason=heartbeat_stale)"
        )

    with patch.object(
        orchestrator, "_submit_council", side_effect=[run_id_1, run_id_2]
    ) as mock_submit, patch.object(
        orchestrator,
        "_poll_council",
        side_effect=[(None, fake_error(run_id_1)), (None, fake_error(run_id_2))],
    ) as mock_poll:
        ok, run_id, data, error = asyncio.run(
            orchestrator._council_subprocess("some decision", "gravitywell")
        )

    assert ok is False
    assert data is None
    # The returned failure is from the retry (fresh run_id), not the first attempt.
    assert run_id == run_id_2
    assert error == fake_error(run_id_2)
    assert mock_submit.call_count == 2
    assert mock_poll.call_count == 2
    # Unit 2 trigger contract preserved after retries are exhausted.
    assert run_id_2 in error
    assert "last_heartbeat=" in error
    assert "reason=" in error


def test_no_retry_on_terminal_council_status():
    """Retry does NOT fire on a terminal Council status — only on the

    liveness died/stalled fast-fail.
    """
    from agents_core.shared_deliberation import orchestrator
    import asyncio

    run_id = "2026-01-01-000000-terminal01"
    council_data = {"status": "resolved", "positions": []}

    with patch.object(
        orchestrator, "_submit_council", return_value=run_id
    ) as mock_submit, patch.object(
        orchestrator, "_poll_council", return_value=(council_data, None)
    ) as mock_poll:
        ok, out_run_id, data, error = asyncio.run(
            orchestrator._council_subprocess("some decision", "gravitywell")
        )

    assert ok is True
    assert out_run_id == run_id
    assert data == council_data
    assert error is None
    mock_submit.assert_called_once()
    mock_poll.assert_called_once()


def test_no_retry_on_timeout_backstop():
    """Retry does NOT fire on a genuine timeout_s backstop — doubling GW load

    on a real 1800s hang would be worse, not better.
    """
    from agents_core.shared_deliberation import orchestrator
    import asyncio

    run_id = "2026-01-01-000000-timeout01"
    fake_error = "Council poll timeout after 1800s"

    with patch.object(
        orchestrator, "_submit_council", return_value=run_id
    ) as mock_submit, patch.object(
        orchestrator, "_poll_council", return_value=(None, fake_error)
    ) as mock_poll:
        ok, out_run_id, data, error = asyncio.run(
            orchestrator._council_subprocess("some decision", "gravitywell")
        )

    assert ok is False
    assert out_run_id == run_id
    assert data is None
    assert error == fake_error
    assert "died/stalled" not in error
    mock_submit.assert_called_once()
    mock_poll.assert_called_once()


def test_fast_fail_signal_is_structured():
    """Fast-fail error message carries run_id, last_heartbeat, and reason fields.

    These three fields form the Unit 2 trigger contract.
    """
    from agents_core.shared_deliberation import orchestrator
    import asyncio

    fake_run_id = "2026-06-20-153344-40f04a"
    fake_error = (
        f"council worker died/stalled "
        f"(run_id={fake_run_id}, last_heartbeat='2026-06-20T15:34:55', "
        f"reason=heartbeat_stale)"
    )

    with patch.object(orchestrator, "_submit_council", return_value=fake_run_id), \
         patch.object(orchestrator, "_poll_council", return_value=(None, fake_error)):
        ok, run_id, data, error = asyncio.run(
            orchestrator._council_subprocess("some decision", "gravitywell")
        )

    assert ok is False
    assert not data

    # Signal payload must carry run_id
    assert fake_run_id in error
    # Signal payload must carry last_heartbeat
    assert "last_heartbeat=" in error
    # Signal payload must carry reason
    assert "reason=" in error
    # Must identify as died/stalled, not generic timeout
    assert "died/stalled" in error
    assert "timeout" not in error


# ---------------------------------------------------------------------------
# AC4 — Happy-path regression: healthy deliberation reaches terminal status
# ---------------------------------------------------------------------------

def test_stub_deliberation_reaches_terminal_status(monkeypatch):
    """Stub-mode deliberation reaches a terminal status (resolved/open/laid-down).

    Regression test: the hold machinery must not break the happy path.
    """
    from agents_core.council import cli

    run_id = "2026-01-01-000000-happy01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "Should we proceed?",
        "voicing": "gravitywell",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    # Allow hold to fail gracefully (no real doorman in tests)
    mock_client = MagicMock()
    mock_client.acquire.side_effect = Exception("no doorman in test")

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=210.0):
        cli.run_deliberation(run_id)

    run = yaml.safe_load(run_file.read_text())
    assert run["status"] in ("resolved", "open", "laid-down"), (
        f"Expected terminal status, got: {run['status']}"
    )
    # heartbeat_at must be stamped even in stub mode
    assert "heartbeat_at" in run, "heartbeat_at not found in stub run YAML"
    # Zero paid spend
    assert run.get("paid_spend") is False

    run_file.unlink(missing_ok=True)


def test_stub_deliberation_stamps_heartbeat_at(monkeypatch):
    """run_deliberation stamps heartbeat_at in the YAML (poller can detect liveness)."""
    from agents_core.council import cli

    run_id = "2026-01-01-000000-hb01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test heartbeat",
        "voicing": "gravitywell",
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    with patch("agents_core.doorman_client.DoormanClient", side_effect=Exception("no doorman")):
        cli.run_deliberation(run_id)

    run = yaml.safe_load(run_file.read_text())
    assert "heartbeat_at" in run
    # heartbeat_at must be an ISO timestamp
    from datetime import datetime
    ts = datetime.fromisoformat(run["heartbeat_at"])
    assert ts is not None

    run_file.unlink(missing_ok=True)


def test_run_deliberation_stamps_started_at(monkeypatch):
    """run_deliberation stamps started_at, distinct from the enqueue-time

    created_at (agents-core-council-liveness-queue-aware-v0, Part A.1). This
    lets the poller's liveness ladder tell a running-but-pre-heartbeat worker
    apart from one still sitting in the serial GW queue.
    """
    from agents_core.council import cli
    from datetime import datetime, timedelta

    run_id = "2026-01-01-000000-started01"
    created_at = (datetime.now() - timedelta(seconds=30)).isoformat(timespec="seconds")
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "test started_at stamp",
        "voicing": "gravitywell",
        "created_at": created_at,
        "turns_cap": 2,
        "turns": [],
        "selected_entities": [],
    }
    cli.COUNCIL_DIR.mkdir(parents=True, exist_ok=True)
    run_file = cli.COUNCIL_DIR / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    with patch("agents_core.doorman_client.DoormanClient", side_effect=Exception("no doorman")):
        cli.run_deliberation(run_id)

    run = yaml.safe_load(run_file.read_text())
    assert "started_at" in run
    assert datetime.fromisoformat(run["started_at"]) >= datetime.fromisoformat(created_at)

    run_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC1 — Atomic save_run and typed load_run parse error
# ---------------------------------------------------------------------------

def test_save_run_is_atomic(tmp_path):
    """Concurrent reads during a large save_run never observe a partial/empty YAML.

    Simulates the write side of the save_run → load_run fork race: writes a
    large YAML repeatedly while a reader thread reads concurrently; asserts
    that every successful read is a complete, parseable dict.
    """
    import threading
    import yaml as _yaml
    from agents_core.council import cli

    run_id = "2026-01-01-000000-atomic01"
    run_file = tmp_path / f"{run_id}.yaml"
    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path

    run = {
        "run_id": run_id,
        "status": "deliberating",
        "decision": "x" * 60_000,
        "turns": [],
    }
    cli.save_run(run)

    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                data = _yaml.safe_load(run_file.read_text())
                if not isinstance(data, dict):
                    errors.append(f"got non-dict: {type(data).__name__}")
            except Exception as e:
                errors.append(f"parse error: {e}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    for _ in range(20):
        cli.save_run(run)

    stop.set()
    t.join(timeout=2)
    cli.COUNCIL_DIR = orig_dir
    run_file.unlink(missing_ok=True)

    assert not errors, f"Observed partial/corrupt reads during atomic writes: {errors[:3]}"


def test_load_run_raises_on_corrupt_yaml(tmp_path):
    """load_run raises CouncilRunParseError (not None) when YAML is empty or non-dict."""
    from agents_core.council import cli

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path
    run_id = "2026-01-01-000000-parse01"
    run_file = tmp_path / f"{run_id}.yaml"

    run_file.write_text("")
    with pytest.raises(cli.CouncilRunParseError):
        cli.load_run(run_id)

    run_file.write_text("just a string\n")
    with pytest.raises(cli.CouncilRunParseError):
        cli.load_run(run_id)

    cli.COUNCIL_DIR = orig_dir
    run_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# AC2 — Worker self-capture of startup exceptions
# ---------------------------------------------------------------------------

def test_run_deliberation_captures_load_run_failure(tmp_path):
    """A CouncilRunParseError during startup is written to the run YAML as status:failed."""
    from agents_core.council import cli

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path

    run_id = "2026-01-01-000000-capture01"
    (tmp_path / f"{run_id}.yaml").write_text("")

    with pytest.raises(cli.CouncilRunParseError):
        cli.run_deliberation(run_id)

    run = yaml.safe_load((tmp_path / f"{run_id}.yaml").read_text())
    assert isinstance(run, dict), "run YAML should be a dict after watchdog write"
    assert run.get("status") == "failed", f"Expected status:failed, got {run.get('status')!r}"
    assert "worker_error" in run, "worker_error field missing"

    cli.COUNCIL_DIR = orig_dir


def test_run_deliberation_self_capture_does_not_swallow_exception(tmp_path):
    """After self-capture, run_deliberation still re-raises so callers see the error."""
    from agents_core.council import cli

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path
    run_id = "2026-01-01-000000-reraise01"
    (tmp_path / f"{run_id}.yaml").write_text("")

    with pytest.raises(Exception):
        cli.run_deliberation(run_id)

    cli.COUNCIL_DIR = orig_dir


# ---------------------------------------------------------------------------
# AC3 — Watchdog forces status:failed on pre-heartbeat worker death
# ---------------------------------------------------------------------------

def test_watch_startup_forces_failed_on_early_death(tmp_path):
    """_watch_startup writes status:failed when the child exits without a heartbeat."""
    import subprocess
    from unittest.mock import MagicMock
    from agents_core.council import cli

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path

    run_id = "2026-01-01-000000-watchdog01"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "decision": "test watchdog",
    }
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))
    log_file = tmp_path / f"{run_id}.log"
    log_file.write_text("crash output here\n")

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 1

    cli._watch_startup(mock_proc, run_id, log_file, timeout=5.0)

    run = yaml.safe_load(run_file.read_text())
    assert run.get("status") == "failed", f"Expected status:failed, got {run.get('status')!r}"
    assert "worker_error" in run
    assert "code=1" in run["worker_error"]

    cli.COUNCIL_DIR = orig_dir
    run_file.unlink(missing_ok=True)


def test_watch_startup_does_not_override_if_heartbeat_present(tmp_path):
    """_watch_startup does NOT write status:failed if the child wrote a heartbeat."""
    import subprocess
    from unittest.mock import MagicMock
    from agents_core.council import cli
    from datetime import datetime as _dt

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path

    run_id = "2026-01-01-000000-watchdog02"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "decision": "test",
        "heartbeat_at": _dt.now().isoformat(timespec="seconds"),
    }
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))
    log_file = tmp_path / f"{run_id}.log"

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 0

    cli._watch_startup(mock_proc, run_id, log_file, timeout=5.0)

    run = yaml.safe_load(run_file.read_text())
    assert run.get("status") == "deliberating", (
        f"Expected deliberating (heartbeat present), got {run.get('status')!r}"
    )

    cli.COUNCIL_DIR = orig_dir
    run_file.unlink(missing_ok=True)


def test_queue_runner_watchdog_forces_failed_on_nonzero_exit(tmp_path):
    """_force_council_run_failed writes status:failed + worker_error atomically."""
    from agents_core.claude_queue_runner import _force_council_run_failed

    run_id = "2026-01-01-000000-qwatchdog01"
    run_file = tmp_path / f"{run_id}.yaml"
    run_data = {"run_id": run_id, "status": "deliberating", "decision": "test"}
    run_file.write_text(yaml.safe_dump(run_data))

    _force_council_run_failed(run_file, run_id, "council worker exited code=1")

    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed"
    assert "worker_error" in result
    assert "code=1" in result["worker_error"]


def test_queue_runner_watchdog_does_not_override_terminal_status(tmp_path):
    """_force_council_run_failed skips if status is already terminal."""
    from agents_core.claude_queue_runner import _force_council_run_failed

    run_id = "2026-01-01-000000-qwatchdog02"
    run_file = tmp_path / f"{run_id}.yaml"
    run_data = {"run_id": run_id, "status": "resolved", "decision": "test"}
    run_file.write_text(yaml.safe_dump(run_data))

    _force_council_run_failed(run_file, run_id, "should be ignored")

    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "resolved", "Must not overwrite a terminal status"
    assert "worker_error" not in result


def test_watch_startup_preserves_self_captured_traceback(tmp_path):
    """_watch_startup does NOT overwrite a self-captured traceback (status:failed + worker_error).

    AC3: if the worker ran its own error handler and wrote worker_error before dying,
    the parent watchdog's generic 'exited code=N' string must not clobber it.
    """
    import subprocess
    from unittest.mock import MagicMock
    from agents_core.council import cli

    orig_dir = cli.COUNCIL_DIR
    cli.COUNCIL_DIR = tmp_path

    run_id = "2026-01-01-000000-preserve01"
    captured_tb = "Traceback (most recent call last):\n  File cli.py line 3\nRuntimeError: engine exploded"
    run_data = {
        "run_id": run_id,
        "status": "failed",
        "worker_error": captured_tb,
    }
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump(run_data))
    log_file = tmp_path / f"{run_id}.log"

    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 1

    cli._watch_startup(mock_proc, run_id, log_file, timeout=5.0)

    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed"
    assert result.get("worker_error") == captured_tb, (
        f"Self-captured traceback was overwritten by parent watchdog. "
        f"Got: {result.get('worker_error')!r}"
    )

    cli.COUNCIL_DIR = orig_dir
    run_file.unlink(missing_ok=True)


def test_queue_runner_watchdog_preserves_self_captured_traceback(tmp_path):
    """_force_council_run_failed does NOT overwrite a self-captured traceback.

    AC3: when the child already wrote status:failed + worker_error (self-capture),
    the parent's generic fallback message must not clobber the diagnostic traceback.
    """
    from agents_core.claude_queue_runner import _force_council_run_failed

    run_id = "2026-01-01-000000-preserve02"
    run_file = tmp_path / f"{run_id}.yaml"
    captured_tb = "Traceback (most recent call last):\n  File bar.py line 5\nValueError: bad yaml"
    run_data = {
        "run_id": run_id,
        "status": "failed",
        "worker_error": captured_tb,
    }
    run_file.write_text(yaml.safe_dump(run_data))

    _force_council_run_failed(run_file, run_id, "council worker exited code=1")

    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed"
    assert result.get("worker_error") == captured_tb, (
        f"Self-captured traceback was overwritten. Got: {result.get('worker_error')!r}"
    )


# ---------------------------------------------------------------------------
# AC5 — Orchestrator span-hold acquires with lease_kind="coordination"
# (gw-admission-elevator-kind-aware-v0 / Fix 1 end-to-end integration)
# ---------------------------------------------------------------------------

def test_orchestrator_span_hold_uses_coordination_kind(monkeypatch):
    """AC5: orchestrator span-hold must acquire with lease_kind='coordination'.

    A coordination hold does not count as a drain-gate contender, so the
    session's own GW operator legs can admit while the span-hold is held.
    """
    import asyncio
    from agents_core.shared_deliberation import orchestrator
    from agents_core.shared_deliberation.envelope import DeliberationRequest

    acquire_calls = []

    class FakeDoorman:
        def acquire(self, *args, **kwargs):
            acquire_calls.append(kwargs)
            return {"status": "serving"}

        def release(self, *args, **kwargs):
            pass

        def close(self):
            pass

    # (ok, result_dict, facets_id, error) — 4-tuple
    fake_facets = (True, {"answer": "stub"}, "facets-id-stub", None)
    fake_council = (True, "run-stub", {"status": "resolved", "landing": "stub",
                                        "confidence": "high", "open_questions": [],
                                        "positions": [], "voicing_effective": "gravitywell",
                                        "voicing_degraded": False,
                                        "voicing_degraded_reason": None}, None)

    with patch("agents_core.doorman_client.DoormanClient",
               return_value=FakeDoorman()), \
         patch("agents_core.doorman_client._gw_acquire_timeout",
               return_value=5.0), \
         patch.object(orchestrator, "_facets_subprocess",
                      return_value=fake_facets), \
         patch.object(orchestrator, "_council_subprocess",
                      return_value=fake_council):
        asyncio.run(orchestrator.run_deliberation(
            DeliberationRequest(
                text="test",
                context={},
                council_voicing="gravitywell",
                facets_operator="gravitywell",
            )
        ))

    # The initial span-hold acquire must use lease_kind="coordination"
    hold_calls = [c for c in acquire_calls if c.get("lease_kind") == "coordination"]
    assert hold_calls, (
        f"Expected at least one acquire call with lease_kind='coordination'; "
        f"all calls: {acquire_calls}"
    )


# ---------------------------------------------------------------------------
# AC6 / Fix 2 — Refresh loop stops at deliberation deadline
# ---------------------------------------------------------------------------

def test_orchestrator_span_hold_uses_short_refresh_ttl(monkeypatch):
    """AC6 / Fix 2: initial span-hold uses refresh_ttl (2x interval), not the long span_ttl.

    If the refresh thread dies, the hold expires within 2x SHARED_DELIB_SPAN_REFRESH_S
    without depending on the finally-release firing.
    """
    import asyncio
    from agents_core.shared_deliberation import orchestrator
    from agents_core.shared_deliberation.envelope import DeliberationRequest

    monkeypatch.setenv("SHARED_DELIB_SPAN_REFRESH_S", "300")
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_TIMEOUT_S", "1800")

    ttl_values = []

    class FakeDoorman:
        def acquire(self, node, work_id, ttl_sec, reason, **kwargs):
            ttl_values.append(ttl_sec)
            return {"status": "serving"}

        def release(self, *args, **kwargs):
            pass

        def close(self):
            pass

    fake_facets = (True, {"answer": "stub"}, "facets-id-stub", None)
    fake_council = (True, "run-stub", {"status": "resolved", "landing": "stub",
                                        "confidence": "high", "open_questions": [],
                                        "positions": [], "voicing_effective": "gravitywell",
                                        "voicing_degraded": False,
                                        "voicing_degraded_reason": None}, None)

    with patch("agents_core.doorman_client.DoormanClient",
               return_value=FakeDoorman()), \
         patch("agents_core.doorman_client._gw_acquire_timeout",
               return_value=5.0), \
         patch.object(orchestrator, "_facets_subprocess",
                      return_value=fake_facets), \
         patch.object(orchestrator, "_council_subprocess",
                      return_value=fake_council):
        asyncio.run(orchestrator.run_deliberation(
            DeliberationRequest(
                text="test",
                context={},
                council_voicing="gravitywell",
                facets_operator="gravitywell",
            )
        ))

    assert ttl_values, "span-hold acquire must have been called"
    # Initial TTL must be _refresh_interval * 2 (600s), not the long span_ttl (2400s)
    initial_ttl = ttl_values[0]
    expected_refresh_ttl = 300 * 2  # SHARED_DELIB_SPAN_REFRESH_S * 2
    assert initial_ttl == expected_refresh_ttl, (
        f"Initial span-hold TTL must be refresh_ttl={expected_refresh_ttl}; "
        f"got {initial_ttl}. A long TTL means a leaked hold pins flip-protection indefinitely."
    )
