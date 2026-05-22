"""Unit tests for agents_core.claude_queue_sync.submit_and_wait.

Covers:
  - Completed task returns file contents
  - Failed task raises RuntimeError with error string
  - Timeout raises TimeoutError
  - Missing output_path field in completed YAML raises RuntimeError
  - Missing output_path file on disk raises RuntimeError
  - ClaudeQueue import failure raises RuntimeError
  - Callback emits "pending" immediately
  - Callback emits "active" on active_dir transition
  - Callback emits "completed" exactly once before return
  - Callback emits "failed" exactly once before raise
  - Callback exceptions don't break the wrapper
  - Default on_state_change=None preserves byte-for-byte parity
  - No duplicate state emissions
"""
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(directory: Path, task_id: str, data: dict) -> None:
    path = directory / f"{task_id}.yaml"
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# Fixture: minimal ClaudeQueue with tmp dirs
# ---------------------------------------------------------------------------

class _FakeQueue:
    """Minimal ClaudeQueue stand-in for submit_and_wait tests."""

    def __init__(self, tmp_path: Path):
        self.pending_dir = tmp_path / "pending"
        self.active_dir = tmp_path / "active"
        self.completed_dir = tmp_path / "completed"
        self.failed_dir = tmp_path / "failed"
        for d in (self.pending_dir, self.active_dir,
                  self.completed_dir, self.failed_dir):
            d.mkdir(parents=True)
        self._counter = 0

    def submit(self, task_dict: dict) -> str:
        self._counter += 1
        return f"task_{self._counter:04d}"


# ---------------------------------------------------------------------------
# 1. Completed task returns file contents
# ---------------------------------------------------------------------------

def test_submit_and_wait_completed(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "output.md"
    output_file.write_text("hello world")

    def _complete_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return f"task_{fq._counter + 1:04d}"

    fq.submit = _complete_immediately

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        result = submit_and_wait({"task_type": "llm_call"}, poll_interval_s=0.01)

    assert result == "hello world"


# ---------------------------------------------------------------------------
# 2. Failed task raises RuntimeError with error string
# ---------------------------------------------------------------------------

def test_submit_and_wait_failed(tmp_path):
    fq = _FakeQueue(tmp_path)

    def _fail_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.failed_dir, task_id,
                    {"error": "subprocess died", "status": "failed"})
        return task_id

    fq.submit = _fail_immediately

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(RuntimeError, match="subprocess died"):
            submit_and_wait({"task_type": "llm_call"}, poll_interval_s=0.01)


# ---------------------------------------------------------------------------
# 3. Timeout raises TimeoutError
# ---------------------------------------------------------------------------

def test_submit_and_wait_timeout(tmp_path):
    fq = _FakeQueue(tmp_path)

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(TimeoutError):
            submit_and_wait({"task_type": "llm_call"},
                            timeout_s=0.05, poll_interval_s=0.01)


# ---------------------------------------------------------------------------
# 4. Missing output_path field in completed YAML raises RuntimeError
# ---------------------------------------------------------------------------

def test_submit_and_wait_no_output_path_field(tmp_path):
    fq = _FakeQueue(tmp_path)

    def _complete_no_output(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id, {"status": "completed"})
        return task_id

    fq.submit = _complete_no_output

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(RuntimeError, match="output_path field is missing"):
            submit_and_wait({"task_type": "llm_call"}, poll_interval_s=0.01)


# ---------------------------------------------------------------------------
# 5. Missing output_path file on disk raises RuntimeError
# ---------------------------------------------------------------------------

def test_submit_and_wait_output_file_missing(tmp_path):
    fq = _FakeQueue(tmp_path)

    def _complete_missing_file(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": "/nonexistent/path/output.md", "status": "completed"})
        return task_id

    fq.submit = _complete_missing_file

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(RuntimeError, match="output file is unreadable"):
            submit_and_wait({"task_type": "llm_call"}, poll_interval_s=0.01)


# ---------------------------------------------------------------------------
# 6. ClaudeQueue import failure raises RuntimeError
# ---------------------------------------------------------------------------

def test_submit_and_wait_no_claude_queue():
    with patch("agents_core.claude_queue_sync.ClaudeQueue", None):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(RuntimeError, match="agents_core.claude_queue not available"):
            submit_and_wait({"task_type": "llm_call"})


# ---------------------------------------------------------------------------
# 7. Callback emits "pending" immediately
# ---------------------------------------------------------------------------

def test_callback_pending_emitted(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("done")

    def _complete_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return task_id

    fq.submit = _complete_immediately

    spy = MagicMock()

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        submit_and_wait({"task_type": "llm_call"},
                        poll_interval_s=0.01, on_state_change=spy)

    first_call = spy.call_args_list[0][0][0]
    assert first_call.state == "pending"
    assert first_call.elapsed_s >= 0.0


# ---------------------------------------------------------------------------
# 8. Callback emits "active" on active_dir transition
# ---------------------------------------------------------------------------

def test_callback_active_emitted(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("done")

    call_count = [0]

    def _submit_then_progress(task_dict):
        call_count[0] += 1
        task_id = f"task_{call_count[0]:04d}"
        # Write active YAML immediately so first poll sees it
        _write_yaml(fq.active_dir, task_id, {"status": "active"})
        return task_id

    fq.submit = _submit_then_progress

    spy = MagicMock()
    states = []

    def _on_change(update):
        states.append(update.state)
        # After "active" is emitted, write completed YAML
        if update.state == "active":
            task_id = f"task_{call_count[0]:04d}"
            _write_yaml(fq.completed_dir, task_id,
                        {"output_path": str(output_file), "status": "completed"})

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        submit_and_wait({"task_type": "llm_call"},
                        poll_interval_s=0.01, on_state_change=_on_change)

    assert "active" in states


# ---------------------------------------------------------------------------
# 9. Callback emits "completed" exactly once before return
# ---------------------------------------------------------------------------

def test_callback_completed_once(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("result text")

    def _complete_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return task_id

    fq.submit = _complete_immediately

    spy = MagicMock()

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        result = submit_and_wait({"task_type": "llm_call"},
                                 poll_interval_s=0.01, on_state_change=spy)

    assert result == "result text"
    completed_calls = [c for c in spy.call_args_list
                       if c[0][0].state == "completed"]
    assert len(completed_calls) == 1


# ---------------------------------------------------------------------------
# 10. Callback emits "failed" exactly once before raise
# ---------------------------------------------------------------------------

def test_callback_failed_once(tmp_path):
    fq = _FakeQueue(tmp_path)

    def _fail_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.failed_dir, task_id,
                    {"error": "oops", "status": "failed"})
        return task_id

    fq.submit = _fail_immediately

    spy = MagicMock()

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        with pytest.raises(RuntimeError):
            submit_and_wait({"task_type": "llm_call"},
                            poll_interval_s=0.01, on_state_change=spy)

    failed_calls = [c for c in spy.call_args_list
                    if c[0][0].state == "failed"]
    assert len(failed_calls) == 1


# ---------------------------------------------------------------------------
# 11. Callback exceptions don't break the wrapper
# ---------------------------------------------------------------------------

def test_callback_exception_ignored(tmp_path, caplog):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("ok")

    def _complete_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return task_id

    fq.submit = _complete_immediately

    def _bad_cb(update):
        raise ValueError("callback explosion")

    with caplog.at_level(logging.WARNING, logger="agents_core.claude_queue_sync"):
        with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
            from agents_core.claude_queue_sync import submit_and_wait
            result = submit_and_wait({"task_type": "llm_call"},
                                     poll_interval_s=0.01, on_state_change=_bad_cb)

    assert result == "ok"
    assert any("callback" in r.message.lower() or "callback explosion" in r.message
               for r in caplog.records)


# ---------------------------------------------------------------------------
# 12. Default on_state_change=None preserves byte-for-byte parity
# ---------------------------------------------------------------------------

def test_no_callback_parity(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("parity result")

    def _complete_immediately(task_dict):
        task_id = f"task_{fq._counter + 1:04d}"
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return task_id

    fq.submit = _complete_immediately

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        result = submit_and_wait({"task_type": "llm_call"}, poll_interval_s=0.01)

    assert result == "parity result"


# ---------------------------------------------------------------------------
# 13. No duplicate state emissions (rapid pending→active→completed)
# ---------------------------------------------------------------------------

def test_no_duplicate_state_emissions(tmp_path):
    fq = _FakeQueue(tmp_path)
    output_file = tmp_path / "out.md"
    output_file.write_text("fast result")

    call_count = [0]

    def _rapid_submit(task_dict):
        call_count[0] += 1
        task_id = f"task_{call_count[0]:04d}"
        # Pre-populate active before poll loop starts
        _write_yaml(fq.active_dir, task_id, {"status": "active"})
        # Also pre-populate completed so first poll sees terminal state
        _write_yaml(fq.completed_dir, task_id,
                    {"output_path": str(output_file), "status": "completed"})
        return task_id

    fq.submit = _rapid_submit

    states = []

    with patch("agents_core.claude_queue_sync.ClaudeQueue", return_value=fq):
        from agents_core.claude_queue_sync import submit_and_wait
        submit_and_wait({"task_type": "llm_call"},
                        poll_interval_s=0.01,
                        on_state_change=lambda u: states.append(u.state))

    # Each state emitted at most once
    from collections import Counter
    counts = Counter(states)
    for state, n in counts.items():
        assert n == 1, f"state {state!r} emitted {n} times (expected 1)"
    # pending always emitted; completed always emitted; active optional
    assert "pending" in states
    assert "completed" in states
