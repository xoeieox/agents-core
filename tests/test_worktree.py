"""Tests for agents_core.worktree.setup_worktree soft-warning behaviour.

These tests validate that missing CLAUDE.md and/or .claude/settings.json
in the parent clone produce stderr warnings rather than AssertionError —
the fix landed in the `lapis-pm-worktree-assertion-softening` spec.

Moved from lapis-pm tests/ on 2026-04-28 as part of the shaper promotion
to agents-core (agents-core-promote-shaper).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents_core.worktree import WORKTREE_ROOT, WorktreeHandle, setup_worktree, teardown_worktree


def _init_minimal_clone(tmp_path: Path) -> Path:
    """Create a bare repo + an initial main branch + a clone with no `.claude/`
    and no `CLAUDE.md`. Return the clone path.
    """
    bare = tmp_path / "bare.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "clone"

    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)

    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True, capture_output=True)
    (seed / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(seed), "config", "user.email", "test@test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "add", "README.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "remote", "add", "origin", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"], check=True, capture_output=True)

    subprocess.run(["git", "clone", str(bare), str(clone)], check=True, capture_output=True)
    return clone


def test_setup_worktree_warns_on_missing_claude_md_and_settings(tmp_path, capfd):
    """No `.claude/`, no `CLAUDE.md` in the parent clone → warnings on stderr,
    worktree path returned successfully, no AssertionError.
    """
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-missing-claude-md-and-settings"

    try:
        handle = setup_worktree(task_id, str(clone))
        assert isinstance(handle, WorktreeHandle)
        assert handle.path == WORKTREE_ROOT / task_id
        assert handle.path.exists()

        captured = capfd.readouterr()
        stderr = captured.err
        assert "CLAUDE.md missing" in stderr, f"expected CLAUDE.md warning, got: {stderr!r}"
        assert "settings.json missing" in stderr, f"expected settings.json warning, got: {stderr!r}"
        assert "WARN:" in stderr
    finally:
        teardown_worktree(task_id, str(clone))


def test_setup_worktree_silent_when_both_present(tmp_path, capfd):
    """If both CLAUDE.md and .claude/settings.json exist, no WARN lines are emitted."""
    clone = _init_minimal_clone(tmp_path)
    (clone / "CLAUDE.md").write_text("# test\n")
    (clone / ".claude").mkdir(exist_ok=True)
    (clone / ".claude" / "settings.json").write_text('{"hooks": {}}\n')
    # Commit CLAUDE.md so worktree add sees it (`.claude/` is per-repo state,
    # symlinked into the worktree by setup_worktree — not committed).
    subprocess.run(["git", "-C", str(clone), "config", "user.email", "test@test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "add", "CLAUDE.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "claude.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "push", "origin", "main"], check=True, capture_output=True)

    task_id = "test-both-present"
    try:
        handle = setup_worktree(task_id, str(clone))
        assert handle.path.exists()

        captured = capfd.readouterr()
        assert "WARN:" not in captured.err, f"expected silent setup, got warnings: {captured.err!r}"
    finally:
        teardown_worktree(task_id, str(clone))
