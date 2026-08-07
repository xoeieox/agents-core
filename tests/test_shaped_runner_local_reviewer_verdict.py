"""Regression tests for agents-core-local-reviewer-no-silent-empty-verdict-v0.

Covers the defect at shaped_runner.py's ``_run_local_reviewer``: a None
result from ``call_gw_agent`` (no verdict produced — budget/step exhaustion,
request failure, etc.) was being coerced to ``""`` via ``result or ""``,
which shaped_runner then printed and exited 0 for, and which
claude_queue_runner then recorded as ``status=completed / error=null``.

DoD covered:
  1. ``_run_local_reviewer`` returns None (never "") on a no-verdict result,
     and the reason from reason_out reaches stderr.
  2. An unpopulated reason_out yields "no_content_no_reason", not an empty
     or invented reason.
  3. A no-verdict result causes a non-zero process exit end-to-end, and
     therefore a queue record of status=failed with a non-null error —
     asserted against the queue record, not just the return value.
  4. The happy path (a real verdict) is unchanged: exits 0, queue record is
     completed.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1 & 2: _run_local_reviewer itself
# ---------------------------------------------------------------------------


_PROBE_OK = {"outcome": "tool_call", "served_model": "gravitywell-122b", "detail": None}


def test_run_local_reviewer_returns_none_and_logs_reason(capsys):
    import agents_core.shaped_runner as sr

    spec = {"prompt": "review this", "task_id": "t-1"}

    def fake_call_gw_agent(**kwargs):
        kwargs["reason_out"].append("max_steps_exhausted")
        return None

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=_PROBE_OK),
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result is None, "must return None, not '' — '' silently becomes a fake success"
    err = capsys.readouterr().err
    assert "reason=max_steps_exhausted" in err


def test_run_local_reviewer_empty_reason_out_falls_back(capsys):
    """reason_out populated with nothing -> 'no_content_no_reason', not a guess."""
    import agents_core.shaped_runner as sr

    spec = {"prompt": "review this", "task_id": "t-2"}

    def fake_call_gw_agent(**kwargs):
        return None  # reason_out left untouched

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=_PROBE_OK),
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result is None
    err = capsys.readouterr().err
    assert "reason=no_content_no_reason" in err


def test_run_local_reviewer_happy_path_returns_verdict_text():
    """Regression guard: a normal verdict is returned unchanged."""
    import agents_core.shaped_runner as sr

    spec = {"prompt": "review this", "task_id": "t-3"}

    def fake_call_gw_agent(**kwargs):
        assert "reason_out" in kwargs and kwargs["reason_out"] == []
        return '{"verdict": "clean"}'

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=_PROBE_OK),
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result == '{"verdict": "clean"}'


# ---------------------------------------------------------------------------
# 3: main() exits non-zero when _run_local_reviewer yields no verdict
# ---------------------------------------------------------------------------


def _make_reviewer_spec(tmp_path: Path) -> Path:
    shaped = tmp_path / "shaped"
    shaped.mkdir(exist_ok=True)
    spec = {
        "model": "gravitywell-122b",
        "engine": "local-reviewer",
        "system": "you are a reviewer",
        "prompt": "review the diff",
        "timeout_s": 900,
        "capture_meta": False,
        "target_id": "t-rev",
        "repo": "agents-core",
        "task_id": "abc123",
        "base_branch": "main",
        "worktree_required": False,
        "cwd": str(tmp_path / "shared-clone"),
    }
    p = shaped / f"{spec['target_id']}-reviewer-abc123.json"
    p.write_text(json.dumps(spec))
    return p


def test_main_exits_nonzero_when_local_reviewer_returns_none(tmp_path, capsys):
    import agents_core.shaped_runner as sr

    spec_path = _make_reviewer_spec(tmp_path)

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner._run_local_reviewer", return_value=None),
        pytest.raises(SystemExit) as exc_info,
    ):
        sr.main()

    assert exc_info.value.code == 1


def test_main_exits_zero_when_local_reviewer_returns_verdict(tmp_path, capsys):
    import agents_core.shaped_runner as sr

    spec_path = _make_reviewer_spec(tmp_path)

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner._run_local_reviewer", return_value='{"verdict": "clean"}'),
    ):
        sr.main()  # must not raise

    out = capsys.readouterr().out
    assert '{"verdict": "clean"}' in out


# ---------------------------------------------------------------------------
# 4: end-to-end — queue record, not just return value
# ---------------------------------------------------------------------------


class _FakeQueue:
    def __init__(self):
        self.failed = []
        self.completed = []

    def fail(self, task_id, error=""):
        self.failed.append((task_id, error))

    def complete(self, task_id, **kwargs):
        self.completed.append((task_id, kwargs))


class _FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_run_shaped_task_no_verdict_records_failed_queue_record(monkeypatch, tmp_path):
    """The end-to-end assertion that matters: a no-verdict local-reviewer run
    must produce a queue record of status=failed with the reason visible in
    the recorded error, not status=completed/error=null."""
    from agents_core import claude_queue_runner as runner_mod

    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_cage_buildable", lambda: (True, ""))
    monkeypatch.setattr(runner_mod, "_check_slice_has_cpu_quota", lambda: True)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod, "notify_failure", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod, "notify_completion", lambda *a, **kw: None)

    stderr = (
        b"ERROR: local reviewer produced no verdict (reason=max_steps_exhausted)\n"
    )

    async def fake_spawn(*args, **kwargs):
        return _FakeProcess(stdout=b"", stderr=stderr, returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    queue = _FakeQueue()
    task = {
        "id": "local-rev-1",
        "payload": {"spec_path": str(tmp_path / "spec.json")},
        "timeout_seconds": 30,
        "notify": False,
    }

    await runner_mod._run_shaped_task(queue, task)

    assert not queue.completed, "must not be recorded as completed"
    assert len(queue.failed) == 1
    failed_task_id, error = queue.failed[0]
    assert failed_task_id == "local-rev-1"
    assert error, "error must be non-null — this is the whole point of the fix"
    assert "local reviewer produced no verdict" in error
    assert "max_steps_exhausted" in error


@pytest.mark.asyncio
async def test_run_shaped_task_happy_path_still_completes(monkeypatch, tmp_path):
    """Regression guard: a normal verdict still records status=completed."""
    from agents_core import claude_queue_runner as runner_mod

    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_cage_buildable", lambda: (True, ""))
    monkeypatch.setattr(runner_mod, "_check_slice_has_cpu_quota", lambda: True)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod, "notify_failure", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod, "notify_completion", lambda *a, **kw: None)

    async def fake_spawn(*args, **kwargs):
        return _FakeProcess(stdout=b'{"verdict": "clean"}', stderr=b"", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    queue = _FakeQueue()
    task = {
        "id": "local-rev-2",
        "payload": {"spec_path": str(tmp_path / "spec.json")},
        "timeout_seconds": 30,
        "notify": False,
    }

    await runner_mod._run_shaped_task(queue, task)

    assert not queue.failed, "must not be recorded as failed"
    assert len(queue.completed) == 1
    completed_task_id, kwargs = queue.completed[0]
    assert completed_task_id == "local-rev-2"
    output_path = Path(kwargs["output_path"])
    assert '{"verdict": "clean"}' in output_path.read_text()
