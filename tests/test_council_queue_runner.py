"""Tests for council task dispatch in claude_queue_runner.

Covers:
- Dispatch routing: council.run → _run_council_task, others → _run_shaped_task
- Terminal status handling for deliberation and scene modes
- Failure paths: invalid mode, no terminal status, negative/nonzero returncode
- Notification gating (notify flag)
- ops-primitives skip for council tasks
- Semaphore ordering (_COUNCIL_SEM acquired before self.sem)
- Concurrency cap (at most one council subprocess)
- startup_sweep orphan recovery for council runs
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_task(task_type="subprocess", task_id="test-task-001", mode="deliberation",
               notify=False, timeout=300, **kwargs):
    task = {
        "id": task_id,
        "task_type": task_type,
        "timeout_seconds": timeout,
        "notify": notify,
        "payload": {"mode": mode, "spec_path": "/fake/spec.yaml"},
    }
    task.update(kwargs)
    return task


def _make_queue(tmp_path):
    """Return a minimal fake ClaudeQueue with queue_dir set to tmp_path."""
    q = MagicMock()
    q.queue_dir = tmp_path
    q.active_dir = tmp_path / "active"
    q.active_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("pending", "completed", "failed"):
        (tmp_path / subdir).mkdir(exist_ok=True)
    return q


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------

def test_run_task_routes_council_to_council_handler():
    """council.run task_type must reach _run_council_task, not _run_shaped_task."""
    from agents_core.claude_queue_runner import _run_task

    shaped_calls = []
    council_calls = []

    async def _fake_shaped(queue, task): shaped_calls.append(task)
    async def _fake_council(queue, task): council_calls.append(task)

    import agents_core.claude_queue_runner as runner_mod
    orig_shaped = runner_mod._run_shaped_task
    orig_council = runner_mod._run_council_task
    runner_mod._run_shaped_task = _fake_shaped
    runner_mod._run_council_task = _fake_council
    try:
        asyncio.run(_run_task(MagicMock(), _make_task(task_type="council.run")))
    finally:
        runner_mod._run_shaped_task = orig_shaped
        runner_mod._run_council_task = orig_council

    assert len(council_calls) == 1
    assert len(shaped_calls) == 0


def test_run_task_routes_subprocess_to_shaped_handler():
    from agents_core.claude_queue_runner import _run_task

    shaped_calls = []
    council_calls = []

    async def _fake_shaped(queue, task): shaped_calls.append(task)
    async def _fake_council(queue, task): council_calls.append(task)

    import agents_core.claude_queue_runner as runner_mod
    orig_shaped = runner_mod._run_shaped_task
    orig_council = runner_mod._run_council_task
    runner_mod._run_shaped_task = _fake_shaped
    runner_mod._run_council_task = _fake_council
    try:
        asyncio.run(_run_task(MagicMock(), _make_task(task_type="subprocess")))
    finally:
        runner_mod._run_shaped_task = orig_shaped
        runner_mod._run_council_task = orig_council

    assert len(shaped_calls) == 1
    assert len(council_calls) == 0


# ---------------------------------------------------------------------------
# _run_council_task: terminal status handling
# ---------------------------------------------------------------------------

def _make_council_run_yaml(tmp_path, run_id, status, mode="deliberation", error=None):
    data = {
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "decision": "test",
        "turns": [],
    }
    if error:
        data["error"] = error
    run_yaml = tmp_path / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(data))
    return run_yaml


@pytest.mark.asyncio
async def test_council_task_deliberation_resolved(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000000-aabbcc"
    _make_council_run_yaml(tmp_path, run_id, "resolved", mode="deliberation")

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.complete.assert_called_once()
    call_kwargs = queue.complete.call_args
    assert "council deliberation resolved" in call_kwargs[1].get("result_summary", "")
    queue.fail.assert_not_called()


@pytest.mark.asyncio
async def test_council_task_scene_closed(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000001-bbccdd"
    _make_council_run_yaml(tmp_path, run_id, "closed", mode="scene")

    task = _make_task(task_type="council.run", task_id=run_id, mode="scene", timeout=2400)
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.complete.assert_called_once()
    assert "council scene closed" in queue.complete.call_args[1].get("result_summary", "")


# ---------------------------------------------------------------------------
# _run_council_task: failure paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_council_task_invalid_mode(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    task = _make_task(task_type="council.run", task_id="bad-mode-001", mode="interview")
    queue = MagicMock()

    await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    error_msg = queue.fail.call_args[1].get("error", "")
    assert "invalid payload.mode" in error_msg


@pytest.mark.asyncio
async def test_council_task_no_terminal_status(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000002-ccddee"
    _make_council_run_yaml(tmp_path, run_id, "deliberating", mode="deliberation")  # still deliberating

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    assert "runtime_did_not_set_terminal_status" in queue.fail.call_args[1].get("error", "")


@pytest.mark.asyncio
async def test_council_task_run_failed_status(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000003-ddeeff"
    _make_council_run_yaml(tmp_path, run_id, "failed", mode="deliberation", error="RuntimeError: engine exploded")

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    assert "RuntimeError: engine exploded" in queue.fail.call_args[1].get("error", "")


@pytest.mark.asyncio
async def test_council_task_negative_returncode(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    task = _make_task(task_type="council.run", task_id="sig-kill-001", mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=-9)
        mock_proc.returncode = -9
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    assert "interrupted signal 9" in queue.fail.call_args[1].get("error", "")


@pytest.mark.asyncio
async def test_council_task_nonzero_returncode(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    task = _make_task(task_type="council.run", task_id="exit1-001", mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    error = queue.fail.call_args[1].get("error", "")
    assert "EXIT 1" in error


# ---------------------------------------------------------------------------
# Notification gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_council_task_no_notify_by_default(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000004-eeffgg"
    _make_council_run_yaml(tmp_path, run_id, "resolved")

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation", notify=False)
    queue = MagicMock()
    notifications = []

    import agents_core.claude_queue_runner as r
    orig_notify_completion = r.notify_completion
    orig_notify_failure = r.notify_failure
    r.notify_completion = lambda t, p: notifications.append(("complete", t.get("notify")))
    r.notify_failure = lambda t, p: notifications.append(("fail", t.get("notify")))

    try:
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            await runner_mod._run_council_task(queue, task)
    finally:
        r.notify_completion = orig_notify_completion
        r.notify_failure = orig_notify_failure

    # notify_completion called but notify=False means send_notification won't fire
    # Just verify the call path happened (gating tested in notify_completion itself)
    assert len(notifications) == 1
    assert notifications[0] == ("complete", False)


@pytest.mark.asyncio
async def test_council_task_notify_true_calls_notify_completion(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000005-ffgghh"
    _make_council_run_yaml(tmp_path, run_id, "resolved")

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation", notify=True)
    queue = MagicMock()
    notifications = []

    import agents_core.claude_queue_runner as r
    orig_nc = r.notify_completion
    r.notify_completion = lambda t, p: notifications.append(("complete", t.get("notify")))

    try:
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            await runner_mod._run_council_task(queue, task)
    finally:
        r.notify_completion = orig_nc

    assert any(n[1] is True for n in notifications)


# ---------------------------------------------------------------------------
# ops-primitives skip for council tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_council_task_does_not_call_ops_primitives(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-05-06-000006-gghhi"
    _make_council_run_yaml(tmp_path, run_id, "resolved")

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()
    ops_calls = []

    orig_ops = runner_mod._extract_ops_primitives
    runner_mod._extract_ops_primitives = lambda *a, **kw: ops_calls.append(a)

    try:
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock(return_value=0)
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            await runner_mod._run_council_task(queue, task)
    finally:
        runner_mod._extract_ops_primitives = orig_ops

    assert ops_calls == [], "_extract_ops_primitives must NOT be called for council tasks"


# ---------------------------------------------------------------------------
# startup_sweep: council orphan recovery
# ---------------------------------------------------------------------------

def test_startup_sweep_marks_old_orphan_deliberating_run_failed(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "_COUNCIL_ORPHAN_AGE_SECS", 3600)
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    council_dir = tmp_path / "council"
    council_dir.mkdir()

    run_id = "2026-05-05-120000-orphan1"
    # Created 2 hours ago (older than 1h threshold)
    created_at = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": created_at,
        "mode": "deliberation",
        "decision": "orphan test",
        "turns": [],
    }
    (council_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    # Fake queue with empty queue dirs
    fake_queue = MagicMock()
    fake_queue.active_dir = tmp_path / "active"
    fake_queue.active_dir.mkdir()
    fake_queue.queue_dir = tmp_path / "queue"
    for sub in ("pending", "active", "completed", "failed"):
        (fake_queue.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    # Patch git/worktree operations
    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    result = yaml.safe_load((council_dir / f"{run_id}.yaml").read_text())
    assert result["status"] == "failed"
    assert result["error"] == "runner_crash_recovery"
    assert "completed_at" in result


def test_startup_sweep_skips_young_deliberating_run(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "_COUNCIL_ORPHAN_AGE_SECS", 3600)
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    council_dir = tmp_path / "council"
    council_dir.mkdir()

    run_id = "2026-05-06-110000-young1"
    # Created 30 minutes ago (younger than 1h threshold)
    created_at = (datetime.now() - timedelta(minutes=30)).isoformat(timespec="seconds")
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": created_at,
        "mode": "deliberation",
        "decision": "young run test",
        "turns": [],
    }
    (council_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    fake_queue = MagicMock()
    fake_queue.active_dir = tmp_path / "active"
    fake_queue.active_dir.mkdir()
    fake_queue.queue_dir = tmp_path / "queue"
    for sub in ("pending", "active", "completed", "failed"):
        (fake_queue.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    result = yaml.safe_load((council_dir / f"{run_id}.yaml").read_text())
    assert result["status"] == "deliberating", "Young run must NOT be marked failed"


def test_startup_sweep_skips_run_with_active_queue_task(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "_COUNCIL_ORPHAN_AGE_SECS", 3600)
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    council_dir = tmp_path / "council"
    council_dir.mkdir()

    run_id = "2026-05-04-100000-active1"
    # Old run, but has a queue task
    created_at = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
    run_data = {
        "run_id": run_id,
        "status": "deliberating",
        "created_at": created_at,
        "mode": "deliberation",
        "decision": "active queue run test",
        "turns": [],
    }
    (council_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    fake_queue = MagicMock()
    fake_queue.active_dir = tmp_path / "active"
    fake_queue.active_dir.mkdir()
    fake_queue.queue_dir = tmp_path / "queue"
    for sub in ("pending", "active", "completed", "failed"):
        (fake_queue.queue_dir / sub).mkdir(parents=True, exist_ok=True)
    # Put the run_id in the active queue subdir
    (fake_queue.queue_dir / "active" / f"{run_id}.yaml").write_text(
        yaml.safe_dump({"id": run_id, "task_type": "council.run"})
    )

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    result = yaml.safe_load((council_dir / f"{run_id}.yaml").read_text())
    assert result["status"] == "deliberating", "Run with active queue task must NOT be marked failed"


def test_startup_sweep_skips_already_terminal_run(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "_COUNCIL_ORPHAN_AGE_SECS", 3600)
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    council_dir = tmp_path / "council"
    council_dir.mkdir()

    for final_status in ("resolved", "closed", "failed", "diverged"):
        run_id = f"2026-05-05-120000-{final_status}"
        created_at = (datetime.now() - timedelta(hours=5)).isoformat(timespec="seconds")
        run_data = {
            "run_id": run_id,
            "status": final_status,
            "created_at": created_at,
            "mode": "deliberation",
            "decision": "terminal run",
            "turns": [],
        }
        (council_dir / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))

    fake_queue = MagicMock()
    fake_queue.active_dir = tmp_path / "active"
    fake_queue.active_dir.mkdir()
    fake_queue.queue_dir = tmp_path / "queue"
    for sub in ("pending", "active", "completed", "failed"):
        (fake_queue.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    for final_status in ("resolved", "closed", "failed", "diverged"):
        run_id = f"2026-05-05-120000-{final_status}"
        result = yaml.safe_load((council_dir / f"{run_id}.yaml").read_text())
        assert result["status"] == final_status, f"Terminal run {final_status} must not be overwritten"


# ---------------------------------------------------------------------------
# Semaphore ordering and concurrency cap
# ---------------------------------------------------------------------------

def test_council_sem_is_module_level_semaphore():
    """_COUNCIL_SEM must be a module-level asyncio.Semaphore with value 1."""
    import agents_core.claude_queue_runner as runner_mod
    sem = runner_mod._COUNCIL_SEM
    assert isinstance(sem, asyncio.Semaphore)
    # Semaphore with value 1 should not be locked initially
    assert not sem.locked()


def test_daemon_worker_acquires_council_sem_before_self_sem():
    """Verify the code acquires _COUNCIL_SEM before self.sem for council tasks.
    This is a structural test — reads the source to verify ordering."""
    import inspect
    import agents_core.claude_queue_runner as runner_mod
    source = inspect.getsource(runner_mod.Daemon._worker)
    # Both semaphores must appear; _COUNCIL_SEM must appear before self.sem
    council_idx = source.index("_COUNCIL_SEM")
    self_sem_idx = source.index("self.sem")
    assert council_idx < self_sem_idx, (
        "_COUNCIL_SEM must be acquired before self.sem in _worker"
    )


@pytest.mark.asyncio
async def test_council_concurrency_cap_one_at_a_time():
    """At most one council task runs at a time under _COUNCIL_SEM."""
    import agents_core.claude_queue_runner as runner_mod

    concurrent_peak = [0]
    current_running = [0]

    async def _slow_council(queue, task):
        current_running[0] += 1
        concurrent_peak[0] = max(concurrent_peak[0], current_running[0])
        await asyncio.sleep(0.05)
        current_running[0] -= 1

    orig = runner_mod._run_council_task
    runner_mod._run_council_task = _slow_council
    try:
        sem = asyncio.Semaphore(4)  # plenty of general capacity

        async def _worker_sim(task):
            async with runner_mod._COUNCIL_SEM:
                async with sem:
                    await runner_mod._run_task(MagicMock(), task)

        tasks = [
            asyncio.create_task(_worker_sim(_make_task(task_type="council.run", task_id=f"c{i}")))
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
    finally:
        runner_mod._run_council_task = orig

    assert concurrent_peak[0] == 1, (
        f"Expected at most 1 concurrent council task, got peak={concurrent_peak[0]}"
    )


# ---------------------------------------------------------------------------
# AC5 — Watchdog end-to-end: _run_council_task exit branches write run YAML
#
# These tests drive _run_council_task's actual exit-branch wiring (non-zero,
# signal, timeout) and assert the run YAML lands status:failed — not just that
# queue.fail() was called.  AC3 guarantee: no run left deliberating after the
# worker process exits, verified via the run file itself.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_council_task_nonzero_exit_writes_run_yaml_failed(tmp_path, monkeypatch):
    """Non-zero exit: _run_council_task writes status:failed + worker_error to run YAML."""
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-06-23-000000-e2e-nz01"
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump({"run_id": run_id, "status": "deliberating"}))

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=2)
        mock_proc.returncode = 2
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed", f"Expected status:failed, got {result.get('status')!r}"
    assert "worker_error" in result, "worker_error field must be written to run YAML"
    assert "code=2" in result["worker_error"]


@pytest.mark.asyncio
async def test_run_council_task_signal_exit_writes_run_yaml_failed(tmp_path, monkeypatch):
    """Signal exit (negative rc): _run_council_task writes status:failed with signal info."""
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-06-23-000000-e2e-sig01"
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump({"run_id": run_id, "status": "deliberating"}))

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=-9)
        mock_proc.returncode = -9
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed", f"Expected status:failed, got {result.get('status')!r}"
    assert "worker_error" in result, "worker_error field must be written to run YAML"
    assert "signal 9" in result["worker_error"]


@pytest.mark.asyncio
async def test_run_council_task_timeout_writes_run_yaml_failed(tmp_path, monkeypatch):
    """Timeout: _run_council_task drives watchdog and writes status:failed to run YAML."""
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-06-23-000000-e2e-to01"
    run_file = tmp_path / f"{run_id}.yaml"
    run_file.write_text(yaml.safe_dump({"run_id": run_id, "status": "deliberating"}))

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation", timeout=300)
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec, \
         patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    queue.fail.assert_called_once()
    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed", f"Expected status:failed, got {result.get('status')!r}"
    assert "worker_error" in result, "worker_error field must be written to run YAML"
    assert "timeout" in result["worker_error"]


@pytest.mark.asyncio
async def test_run_council_task_nonzero_preserves_self_captured_traceback(tmp_path, monkeypatch):
    """Watchdog does NOT overwrite self-captured traceback when child exits non-zero.

    AC3: if the child ran its own error handler and wrote worker_error before dying,
    _force_council_run_failed must preserve the diagnostic traceback, not clobber it
    with the generic 'council worker exited code=N' fallback.
    """
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_COUNCIL_LOG_DIR", tmp_path / "logs")

    run_id = "2026-06-23-000000-e2e-tb01"
    run_file = tmp_path / f"{run_id}.yaml"
    captured_tb = "Traceback (most recent call last):\n  File cli.py\nValueError: bad yaml"
    run_file.write_text(yaml.safe_dump({
        "run_id": run_id,
        "status": "failed",
        "worker_error": captured_tb,
    }))

    task = _make_task(task_type="council.run", task_id=run_id, mode="deliberation")
    queue = MagicMock()

    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock(return_value=1)
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc
        await runner_mod._run_council_task(queue, task)

    result = yaml.safe_load(run_file.read_text())
    assert result.get("status") == "failed"
    assert result.get("worker_error") == captured_tb, (
        f"Self-captured traceback was overwritten by watchdog. "
        f"Got: {result.get('worker_error')!r}"
    )
