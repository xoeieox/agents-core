"""Integration tests for the llm_call runner handler (_run_llm_call_task).

Exercises _run_llm_call_task via _run_task with a mocked call_claude_cli.
Uses a real ClaudeQueue wired to tmp directories so file/directory interactions
are exercised without touching the live /srv/lapis/claude-queue directory.
"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path):
    """Return a ClaudeQueue wired to tmp_path."""
    from agents_core.claude_queue import ClaudeQueue
    q = ClaudeQueue(queue_dir=tmp_path / "queue")
    return q


def _make_task(q, **payload_overrides):
    """Submit a minimal llm_call task and claim it (returns the task dict)."""
    payload = {
        "operator_class": "sonnet",
        "prompt": "say hi",
        "_ignore_intention_registry": True,
    }
    payload.update(payload_overrides)
    task_id = q.submit({
        "task_type": "llm_call",
        "model": "claude-sonnet-4-6",
        "submitted_by": "test",
        "payload": payload,
    })
    task = q.claim()
    assert task is not None
    return task


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# 1. Successful llm_call — output file written, task completed
# ---------------------------------------------------------------------------

def test_llm_call_success(tmp_path):
    from agents_core.claude_queue_runner import _run_task, OUTPUT_DIR

    q = _make_queue(tmp_path)
    task = _make_task(q)
    task_id = task["id"]

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", return_value="hello"):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    completed_yaml = q.completed_dir / f"{task_id}.yaml"
    assert completed_yaml.exists(), "task should be in completed dir"
    data = yaml.safe_load(completed_yaml.read_text())
    assert data["status"] == "completed"
    output_path = Path(data["output_path"])
    assert output_path.exists()
    assert output_path.read_text() == "hello"


# ---------------------------------------------------------------------------
# 2. call_claude_cli returns None → task fails
# ---------------------------------------------------------------------------

def test_llm_call_cli_returns_none(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q)
    task_id = task["id"]

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", return_value=None):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    assert failed_yaml.exists()
    data = yaml.safe_load(failed_yaml.read_text())
    assert "call_claude_cli returned None" in data["error"]


# ---------------------------------------------------------------------------
# 3. Unknown operator_class fails fast without calling call_claude_cli
# ---------------------------------------------------------------------------

def test_llm_call_unknown_operator_class(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, operator_class="gpt-4")
    task_id = task["id"]

    cli_mock = MagicMock()

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    assert failed_yaml.exists()
    data = yaml.safe_load(failed_yaml.read_text())
    assert "gpt-4" in data["error"] or "operator_class" in data["error"]
    cli_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 4. operator_class="qwen" fails fast with a clear error
# ---------------------------------------------------------------------------

def test_llm_call_qwen_rejected(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, operator_class="qwen")
    task_id = task["id"]

    cli_mock = MagicMock()

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    assert failed_yaml.exists()
    data = yaml.safe_load(failed_yaml.read_text())
    assert "qwen" in data["error"].lower()
    cli_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Missing payload.prompt fails fast
# ---------------------------------------------------------------------------

def test_llm_call_missing_prompt(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, prompt="")  # empty prompt = same as missing
    task_id = task["id"]

    cli_mock = MagicMock()

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    assert failed_yaml.exists()
    data = yaml.safe_load(failed_yaml.read_text())
    assert "prompt" in data["error"].lower()
    cli_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 6. payload.system and payload.json_mode propagate to call_claude_cli
# ---------------------------------------------------------------------------

def test_llm_call_system_json_mode_forwarded(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, system="be terse", json_mode=True)

    cli_mock = MagicMock(return_value='{"ok": true}')

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    cli_mock.assert_called_once()
    kwargs = cli_mock.call_args.kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["json_mode"] is True


# ---------------------------------------------------------------------------
# 7. operator_class="opus" maps to CLI model "opus"
# ---------------------------------------------------------------------------

def test_llm_call_opus_maps_to_cli_model(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, operator_class="opus", prompt="test")
    # Patch the model field in submitted task to opus default
    task["model"] = "claude-opus-4-7"

    cli_mock = MagicMock(return_value="opus reply")

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    cli_mock.assert_called_once()
    kwargs = cli_mock.call_args.kwargs
    assert kwargs["model"] == "opus"


# ---------------------------------------------------------------------------
# 8. operator_class="gravitywell" is rejected with a clear sync-path error
# ---------------------------------------------------------------------------

def test_llm_call_gravitywell_rejected(tmp_path):
    from agents_core.claude_queue_runner import _run_task

    q = _make_queue(tmp_path)
    task = _make_task(q, operator_class="gravitywell")
    task_id = task["id"]

    cli_mock = MagicMock()

    with patch("agents_core.claude_queue_runner.OUTPUT_DIR", tmp_path / "out"), \
         patch("agents_core.llm.call_claude_cli", cli_mock):
        (tmp_path / "out").mkdir(parents=True, exist_ok=True)
        _run(_run_task(q, task))

    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    assert failed_yaml.exists()
    data = yaml.safe_load(failed_yaml.read_text())
    assert "gravitywell" in data["error"].lower()
    assert "sync" in data["error"].lower() or "path" in data["error"].lower()
    cli_mock.assert_not_called()
