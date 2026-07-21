"""Tests for local-reviewer engine worktree/branch-checkout routing
(agents-core-reviewer-worktree-branch-checkout-v0).

Covers DoD 3-6 of the spec:
  - engine="local-reviewer" with a verified existing_branch routes through
    setup_worktree(ref=existing_branch), _run_local_reviewer(cwd=worktree),
    teardown_worktree.
  - existing_branch not found on origin (nonzero git ls-remote exit code)
    aborts with sys.exit(2) and an ERROR line naming the branch — no silent
    fallback to base_branch.
  - a git ls-remote timeout is treated the same as a verification failure —
    same abort path, not a silent fallback.
  - no existing_branch in the spec (e.g. a spec_reviewer dispatch) resolves
    the worktree ref to base_branch, unchanged from today's behavior.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_spec(tmp_path: Path, **overrides) -> Path:
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
        "worktree_required": True,
        "cwd": str(tmp_path / "shared-clone"),
    }
    spec.update(overrides)
    p = shaped / f"{spec['target_id']}-reviewer-abc123.json"
    p.write_text(json.dumps(spec))
    return p


def _fake_handle(worktree: Path) -> MagicMock:
    h = MagicMock()
    h.path = worktree
    h.env = {}
    return h


# ---------------------------------------------------------------------------
# DoD 3: valid existing_branch -> worktree checked out to that ref
# ---------------------------------------------------------------------------


def test_local_reviewer_uses_existing_branch_when_verified(tmp_path):
    spec_path = _make_spec(tmp_path, existing_branch="lapis/t-rev/forced")
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)) as mock_setup,
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.shaped_runner._run_local_reviewer", return_value="verdict text") as mock_lr,
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
    ):
        sr.main()

    mock_setup.assert_called_once()
    assert mock_setup.call_args.args[2] == "lapis/t-rev/forced"
    mock_lr.assert_called_once()
    called_spec, called_cwd = mock_lr.call_args.args
    assert called_cwd == str(worktree)
    mock_teardown.assert_called_once()


# ---------------------------------------------------------------------------
# DoD 4: existing_branch not found on origin -> abort, no fallback
# ---------------------------------------------------------------------------


def test_local_reviewer_aborts_when_branch_not_on_origin(tmp_path, capsys):
    spec_path = _make_spec(tmp_path, existing_branch="lapis/t-rev/forced")

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.worktree.setup_worktree") as mock_setup,
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.shaped_runner._run_local_reviewer") as mock_lr,
        patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="branch not found")),
        pytest.raises(SystemExit) as exc_info,
    ):
        sr.main()

    assert exc_info.value.code == 2
    mock_setup.assert_not_called()
    mock_lr.assert_not_called()
    mock_teardown.assert_not_called()
    err = capsys.readouterr().err
    assert "ERROR: worktree_setup: existing_branch lapis/t-rev/forced not found on origin" in err


# ---------------------------------------------------------------------------
# DoD 5: git ls-remote timeout -> treated as verification failure, same abort
# ---------------------------------------------------------------------------


def test_local_reviewer_aborts_on_ls_remote_timeout(tmp_path, capsys):
    spec_path = _make_spec(tmp_path, existing_branch="lapis/t-rev/forced")

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.worktree.setup_worktree") as mock_setup,
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.shaped_runner._run_local_reviewer") as mock_lr,
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=30)),
        pytest.raises(SystemExit) as exc_info,
    ):
        sr.main()

    assert exc_info.value.code == 2
    mock_setup.assert_not_called()
    mock_lr.assert_not_called()
    mock_teardown.assert_not_called()
    err = capsys.readouterr().err
    assert "ERROR: worktree_setup: existing_branch lapis/t-rev/forced not found on origin" in err


# ---------------------------------------------------------------------------
# DoD 6: no existing_branch -> resolves to base_branch, unchanged from today
# ---------------------------------------------------------------------------


def test_local_reviewer_no_existing_branch_uses_base_branch(tmp_path):
    """e.g. a spec_reviewer dispatch, which never has an existing_branch."""
    spec_path = _make_spec(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)) as mock_setup,
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.shaped_runner._run_local_reviewer", return_value="verdict text") as mock_lr,
        patch("subprocess.run") as mock_run,
    ):
        sr.main()

    mock_setup.assert_called_once()
    assert mock_setup.call_args.args[2] == "main"
    mock_run.assert_not_called()
    mock_lr.assert_called_once()
    mock_teardown.assert_called_once()
