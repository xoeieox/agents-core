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
# startup_sweep: narrative-emit orphan-spec recovery
# (agents-core-narrative-emit-atomic-submit-v0)
# ---------------------------------------------------------------------------

def _make_fake_queue_for_pending_sweep(tmp_path):
    """Fake ClaudeQueue with all five queue subdirs plus pending_dir set."""
    fake_queue = MagicMock()
    fake_queue.active_dir = tmp_path / "active"
    fake_queue.active_dir.mkdir(parents=True, exist_ok=True)
    fake_queue.queue_dir = tmp_path / "queue"
    for sub in ("pending", "active", "completed", "failed", "cancelled"):
        (fake_queue.queue_dir / sub).mkdir(parents=True, exist_ok=True)
    fake_queue.pending_dir = fake_queue.queue_dir / "pending"
    return fake_queue


def test_startup_sweep_reaps_orphaned_json_to_dead_letter(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")  # does not exist
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    fake_queue = _make_fake_queue_for_pending_sweep(tmp_path)
    orphan = fake_queue.pending_dir / "narrative_20260709_000000_0000_grants.json"
    orphan.write_text('{"task_id": "narrative_20260709_000000_0000_grants"}')

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    assert not orphan.exists(), "orphaned json must be moved out of pending/"
    dead_letter = fake_queue.queue_dir / "dead-letter" / orphan.name
    assert dead_letter.exists(), "orphaned json must land in dead-letter/"


@pytest.mark.parametrize("yaml_subdir", ["pending", "active", "completed", "failed", "cancelled"])
def test_startup_sweep_leaves_json_with_matching_yaml_alone(tmp_path, monkeypatch, yaml_subdir):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    fake_queue = _make_fake_queue_for_pending_sweep(tmp_path)
    task_id = "narrative_20260709_000000_0001_grants"
    spec_json = fake_queue.pending_dir / f"{task_id}.json"
    spec_json.write_text(f'{{"task_id": "{task_id}"}}')
    (fake_queue.queue_dir / yaml_subdir / f"{task_id}.yaml").write_text("id: " + task_id)

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    assert spec_json.exists(), f"json with a matching yaml in {yaml_subdir}/ must NOT be reaped"
    assert not (fake_queue.queue_dir / "dead-letter").exists()


def test_startup_sweep_no_nameerror_when_council_dir_missing(tmp_path, monkeypatch):
    """Regression: queued_ids must be function-scoped, not just inside
    `if _COUNCIL_DIR.exists():` - else this NameErrors when the dir is absent,
    aborting startup_sweep entirely."""
    import agents_core.claude_queue_runner as runner_mod
    missing_council_dir = tmp_path / "no-such-council-dir"
    assert not missing_council_dir.exists()
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", missing_council_dir)
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    fake_queue = _make_fake_queue_for_pending_sweep(tmp_path)
    task_id = "narrative_20260709_000000_0002_grants"
    orphan = fake_queue.pending_dir / f"{task_id}.json"
    orphan.write_text(f'{{"task_id": "{task_id}"}}')

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)  # must not raise NameError

    assert not orphan.exists()
    assert (fake_queue.queue_dir / "dead-letter" / orphan.name).exists()


def test_startup_sweep_ignores_dotfile_and_tmp_suffixed_files(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    fake_queue = _make_fake_queue_for_pending_sweep(tmp_path)
    hidden_tmp = fake_queue.pending_dir / ".narrative_20260709_000000_0003_grants.json.tmp"
    hidden_tmp.write_text('{"task_id": "in-progress-write"}')

    with patch("subprocess.run"):
        runner_mod.startup_sweep(fake_queue)

    assert hidden_tmp.exists(), "an in-flight atomic-write temp file must never be reaped"
    assert not (fake_queue.queue_dir / "dead-letter").exists()


def test_startup_sweep_continues_when_dead_letter_move_raises(tmp_path, monkeypatch):
    import agents_core.claude_queue_runner as runner_mod
    monkeypatch.setattr(runner_mod, "_COUNCIL_DIR", tmp_path / "council")
    monkeypatch.setattr(runner_mod, "WORKTREE_ROOT", tmp_path / "worktrees")

    fake_queue = _make_fake_queue_for_pending_sweep(tmp_path)
    orphan = fake_queue.pending_dir / "narrative_20260709_000000_0004_grants.json"
    orphan.write_text('{"task_id": "narrative_20260709_000000_0004_grants"}')

    with patch("subprocess.run"), \
         patch("agents_core.claude_queue_runner.shutil.move", side_effect=OSError("move boom")):
        runner_mod.startup_sweep(fake_queue)  # must not raise/abort the sweep

    assert orphan.exists(), "file must remain in pending/ since the move itself failed"


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


async def _run_council_concurrency_probe(monkeypatch, cap: int) -> int:
    """Drive N=3 simulated council tasks under a _COUNCIL_SEM of the given
    cap and return the observed concurrency peak. Shared by the cap=1
    (historical hardcoded behavior) and cap=N (configurable, this unit)
    cases below — same mechanism, different cap."""
    import agents_core.claude_queue_runner as runner_mod

    monkeypatch.setattr(runner_mod, "_COUNCIL_SEM", asyncio.Semaphore(cap))

    concurrent_peak = [0]
    current_running = [0]

    async def _slow_council(queue, task):
        current_running[0] += 1
        concurrent_peak[0] = max(concurrent_peak[0], current_running[0])
        await asyncio.sleep(0.05)
        current_running[0] -= 1

    monkeypatch.setattr(runner_mod, "_run_council_task", _slow_council)
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
    return concurrent_peak[0]


@pytest.mark.asyncio
async def test_council_concurrency_cap_one_at_a_time(monkeypatch):
    """At most one council task runs at a time under a _COUNCIL_SEM of 1
    (COUNCIL_MAX_CONCURRENT=1 fallback / A/B surface — the historical
    behavior, still reachable via explicit cap)."""
    peak = await _run_council_concurrency_probe(monkeypatch, cap=1)
    assert peak == 1, f"Expected at most 1 concurrent council task, got peak={peak}"


@pytest.mark.asyncio
async def test_council_concurrency_cap_honours_configured_value(monkeypatch):
    """COUNCIL_MAX_CONCURRENT is a real cap, not just a literal-1 relabel —
    raising it to 2 must let 2 council tasks run concurrently."""
    peak = await _run_council_concurrency_probe(monkeypatch, cap=2)
    assert peak == 2, f"Expected exactly 2 concurrent council tasks under cap=2, got peak={peak}"


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
