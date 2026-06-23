"""Tests for spec-review-council-keepawake-resilience-v0.

Covers:
  AC1 — Deliberation-spanning GW keepawake hold: placed at start, heartbeat-coupled
         TTL so zombie hold auto-expires (no unconditional pinning).
  AC2 — _poll_council fast-fails within COUNCIL_STALL_S on stale heartbeat (not 1800s).
  AC3 — No automatic retry on fast-fail; structured failure signal is well-formed.
  AC4 — Happy path: healthy deliberation still reaches terminal status (stub regression).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
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
# ---------------------------------------------------------------------------

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
    (tmp_path / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_STALL_S", "5")

    real_path = Path

    def path_redirect(*args, **kwargs):
        p = real_path(*args, **kwargs)
        if str(p) == "/srv/lapis/council":
            return tmp_path
        return p

    with patch("agents_core.shared_deliberation.orchestrator.Path", side_effect=path_redirect):
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
    """_poll_council fast-fails when no heartbeat_at and created_at is stale.

    Covers the case where the worker died before emitting any heartbeat.
    """
    from agents_core.shared_deliberation import orchestrator

    run_id = "2026-01-01-000000-stale02"
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": "2020-01-01T00:00:00",
        # No heartbeat_at — worker died before first turn
    }
    (tmp_path / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    monkeypatch.setenv("COUNCIL_STALL_S", "5")

    real_path = Path

    def path_redirect(*args, **kwargs):
        p = real_path(*args, **kwargs)
        if str(p) == "/srv/lapis/council":
            return tmp_path
        return p

    with patch("agents_core.shared_deliberation.orchestrator.Path", side_effect=path_redirect):
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
    (tmp_path / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    # Tiny timeout so the test completes quickly
    monkeypatch.setenv("COUNCIL_STALL_S", "300")

    real_path = Path

    def path_redirect(*args, **kwargs):
        p = real_path(*args, **kwargs)
        if str(p) == "/srv/lapis/council":
            return tmp_path
        return p

    with patch("agents_core.shared_deliberation.orchestrator.Path", side_effect=path_redirect):
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
    (tmp_path / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    real_path = Path

    def path_redirect(*args, **kwargs):
        p = real_path(*args, **kwargs)
        if str(p) == "/srv/lapis/council":
            return tmp_path
        return p

    with patch("agents_core.shared_deliberation.orchestrator.Path", side_effect=path_redirect):
        data, error = orchestrator._poll_council(run_id, timeout_s=10)

    assert error is None
    assert data is not None
    assert data["status"] == "resolved"
    assert data["landing"] == "all agree"


# ---------------------------------------------------------------------------
# AC3 — No automatic retry; structured failure signal is well-formed
# ---------------------------------------------------------------------------

def test_no_retry_on_council_fast_fail():
    """_council_subprocess does not re-submit when _poll_council fast-fails.

    Verify _submit_council called exactly once and the failure signal carries
    run_id + reason (Unit 2 trigger contract).
    """
    from agents_core.shared_deliberation import orchestrator
    import asyncio

    fake_run_id = "2026-01-01-000000-noretry"
    fake_error = (
        f"council worker died/stalled "
        f"(run_id={fake_run_id}, last_heartbeat='2020-01-01T00:00:01', "
        f"reason=heartbeat_stale)"
    )

    with patch.object(orchestrator, "_submit_council", return_value=fake_run_id) as mock_submit, \
         patch.object(orchestrator, "_poll_council", return_value=(None, fake_error)):
        ok, run_id, data, error = asyncio.run(
            orchestrator._council_subprocess("some decision", "gravitywell")
        )

    assert ok is False
    assert run_id == fake_run_id
    assert data is None
    assert error == fake_error
    # _submit_council must be called exactly once — no retry
    mock_submit.assert_called_once()


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
