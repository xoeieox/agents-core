"""Regression tests for agents-core-shaped-runner-stream-log-v0, DoD 5.

shaped_runner.py must only pass stream_log_path to call_claude_cli for
agent_type="spec_reviewer" tasks landing in the generic (non local-fixer,
non local-reviewer, non capture_meta) else-branch. Every other agent_type
must keep calling call_claude_cli with stream_log_path=None.
"""
import json
import sys
from unittest.mock import patch

import pytest


def _write_spec(tmp_path, **overrides):
    shaped = tmp_path / "shaped"
    shaped.mkdir(exist_ok=True)
    spec = {
        "model": "haiku",
        "prompt": "do the review",
        "timeout_s": 30,
        "capture_meta": False,
        "task_id": "task-99",
        "cwd": str(tmp_path / "clone"),
    }
    spec.update(overrides)
    spec_path = shaped / "task-99.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path


def test_spec_reviewer_agent_type_passes_stream_log_path(tmp_path, monkeypatch):
    import agents_core.shaped_runner as sr

    monkeypatch.setattr(sr, "STREAM_LOG_DIR", tmp_path / "stream-logs")
    spec_path = _write_spec(tmp_path, agent_type="spec_reviewer")

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="ok") as mock_cli,
    ):
        sr.main()

    mock_cli.assert_called_once()
    assert mock_cli.call_args.kwargs["stream_log_path"] == str(tmp_path / "stream-logs" / "task-99.jsonl")
    assert (tmp_path / "stream-logs").is_dir()


@pytest.mark.parametrize("agent_type", [None, "fixer", "handler", "council"])
def test_other_agent_types_pass_stream_log_path_none(tmp_path, monkeypatch, agent_type):
    import agents_core.shaped_runner as sr

    monkeypatch.setattr(sr, "STREAM_LOG_DIR", tmp_path / "stream-logs")
    overrides = {} if agent_type is None else {"agent_type": agent_type}
    spec_path = _write_spec(tmp_path, **overrides)

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="ok") as mock_cli,
    ):
        sr.main()

    mock_cli.assert_called_once()
    assert mock_cli.call_args.kwargs["stream_log_path"] is None
    assert not (tmp_path / "stream-logs").exists()
