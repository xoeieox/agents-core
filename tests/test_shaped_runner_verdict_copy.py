"""Tests for verdict sidecar copy in shaped_runner.main() finally block."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VERDICT_PAYLOAD = json.dumps({"verdict": "already_satisfied", "reason": "tests passed"})


def _make_spec_file(shaped_dir: Path, spec_id: str = "abc123") -> Path:
    """Write a minimal dispatch spec and return its path."""
    spec_path = shaped_dir / f"target1-fixer-{spec_id}.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": "haiku",
                "system": "",
                "prompt": "do something",
                "timeout_s": 30,
                "capture_meta": False,
                "worktree_required": True,
                "task_id": "task-1",
                "cwd": str(shaped_dir),
                "base_branch": "main",
            }
        )
    )
    return spec_path


def _run_main(spec_path: Path, worktree_path: Path) -> None:
    """Drive shaped_runner.main() with mocked worktree and LLM call."""
    import agents_core.shaped_runner as sr

    fake_handle = MagicMock()
    fake_handle.path = worktree_path
    fake_handle.env = {}

    with (
        patch.object(sys, "argv", ["shaped_runner", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="done"),
        patch("agents_core.worktree.setup_worktree", return_value=fake_handle),
        patch("agents_core.worktree.teardown_worktree"),
    ):
        sr.main()


# ---------------------------------------------------------------------------
# Test: verdict file present → copied to shaped/ dir
# ---------------------------------------------------------------------------


def test_verdict_copied_when_present(tmp_path):
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    spec_id = "abc123"
    spec_path = _make_spec_file(shaped_dir, spec_id)

    verdict_src = worktree / ".lapis-pm-verdict.json"
    verdict_src.write_text(VERDICT_PAYLOAD)

    _run_main(spec_path, worktree)

    verdict_dest = shaped_dir / f"{spec_id}-verdict.json"
    assert verdict_dest.exists(), "verdict sidecar should be copied to shaped/"
    assert verdict_dest.read_text() == VERDICT_PAYLOAD


# ---------------------------------------------------------------------------
# Test: verdict file absent → no sidecar created, no error
# ---------------------------------------------------------------------------


def test_no_verdict_when_absent(tmp_path, capsys):
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    spec_id = "def456"
    spec_path = _make_spec_file(shaped_dir, spec_id)

    # Do NOT write .lapis-pm-verdict.json in the worktree
    _run_main(spec_path, worktree)

    verdict_dest = shaped_dir / f"{spec_id}-verdict.json"
    assert not verdict_dest.exists(), "no sidecar should appear if verdict absent"
    captured = capsys.readouterr()
    assert "verdict" not in captured.err.lower()


# ---------------------------------------------------------------------------
# Test: verdict file unreadable → WARN logged, teardown still runs
# ---------------------------------------------------------------------------


def test_verdict_unreadable_logs_warn_and_continues(tmp_path, capsys):
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    spec_id = "ghi789"
    spec_path = _make_spec_file(shaped_dir, spec_id)

    # Place a directory where the verdict file is expected — read_text() raises IsADirectoryError
    verdict_as_dir = worktree / ".lapis-pm-verdict.json"
    verdict_as_dir.mkdir()

    import agents_core.shaped_runner as sr

    fake_handle = MagicMock()
    fake_handle.path = worktree
    fake_handle.env = {}

    teardown_called = []

    def fake_teardown(*args, **kwargs):
        teardown_called.append(True)

    with (
        patch.object(sys, "argv", ["shaped_runner", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="done"),
        patch("agents_core.worktree.setup_worktree", return_value=fake_handle),
        patch("agents_core.worktree.teardown_worktree", side_effect=fake_teardown),
    ):
        sr.main()

    captured = capsys.readouterr()
    assert "WARN: verdict sidecar copy failed" in captured.err
    assert teardown_called, "teardown_worktree must still run after copy failure"
