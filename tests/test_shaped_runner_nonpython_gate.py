"""Tests for the repo-aware D1 test gate + tail log capture.

Spec: agents-core-local-fixer-gate-nonpython-v0 (local-fixer test gate:
repo-aware bypass + tail log capture).

AC1: _has_python_test_infra detection (cases a-i, including the Facets
     mandate: a root pyproject.toml carrying only [tool.pytest.ini_options]
     is still infra-present - no bypass).
AC2: bypass path - non-Python worktree -> PR IS created, body carries the
     '## Test gate' section + lapis-test-gate comment, tail log has BYPASSED.
AC3: fail-closed preserved - Python worktree, no tests touched, last outcome
     passed=0 -> NO PR (today's behavior, pinned).
AC4: positive-only untouched - Python worktree, touched test green -> PR
     created, body byte-identical to the non-bypassed shape (no marker).
AC5: _tail_log - appends timestamped lines; a raising filesystem is
     swallowed without propagating; no token-looking substrings.
AC6: module green - this file passes under the harness's own in-dispatch
     run_tests (touched-tests path).

Mocking conventions mirrored from tests/test_local_fixer_dispatch.py:
call_gw_agent -> (fixer_result, transcript); setup/teardown_worktree;
forgejo.create_pr; subprocess.run for git.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agents_core.shaped_runner as sr
from agents_core.shaped_runner import _has_python_test_infra, _tail_log


# ---------------------------------------------------------------------------
# Helpers (conventions mirrored from tests/test_local_fixer_dispatch.py)
# ---------------------------------------------------------------------------

DIFF = (
    "diff --git a/src/lib.rs b/src/lib.rs\n"
    "--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n"
)


def _make_spec(tmp_path: Path, **overrides) -> Path:
    spec = {
        "model": "gravitywell-122b",
        "engine": "local-fixer",
        "system": "you are a fixer",
        "prompt": "fix the bug",
        "timeout_s": 60,
        "capture_meta": False,
        "target_id": "my-target-v0",
        "repo": "agents-core",
        "task_id": "abc123",
        "base_branch": "main",
        "slot_id": "abc123",
    }
    spec.update(overrides)
    shaped = tmp_path / "shaped"
    shaped.mkdir(exist_ok=True)
    p = shaped / f"{spec['target_id']}-fixer_local-abc123.json"
    p.write_text(json.dumps(spec))
    return p


def _fixer_result(diff: str = DIFF,
                  last_test_outcome=None,
                  transcript=None,
                  concluded: bool = True) -> dict:
    return {
        "final_diff": diff,
        "concluded": concluded,
        "last_test_outcome": last_test_outcome,
        "steps": [{"tool": "write_file"}],
        "transcript": transcript if transcript is not None else [],
    }


def _run(tmp_path: Path, worktree: Path, result: dict, mock_pr):
    """Drive sr._run_local_fixer with the standard mock stack.

    Returns (url, pr_body, shaped_dir). The shaped dir is pointed at
    tmp_path via a patched room_path so _tail_log / transcript artifacts
    land in the test sandbox instead of the live /srv/lapis/gpu-queue/shaped.
    """
    spec_path = _make_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    pr_response = {"html_url": "http://x/pulls/1"}
    mock_pr.return_value = pr_response
    shaped_dir = tmp_path / "shaped-rt"
    mock_friction = MagicMock()
    with (
        patch("agents_core.gw_agent.call_gw_agent",
              return_value=(result, result.get("transcript") or [])),
        patch("agents_core.worktree.setup_worktree",
              return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", mock_pr),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(sr, "_write_friction_entry", mock_friction),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))
    if mock_pr.call_args:
        body = mock_pr.call_args.kwargs.get("body", "")
    else:
        body = ""
    return url, body, shaped_dir, mock_friction


def _fake_handle(worktree: Path) -> MagicMock:
    h = MagicMock()
    h.path = worktree
    h.env = {}
    return h


# ---------------------------------------------------------------------------
# AC1: _has_python_test_infra (cases a-i)
# ---------------------------------------------------------------------------

def test_ac1a_empty_tree_is_not_infra(tmp_path):
    assert _has_python_test_infra(tmp_path) is False


def test_ac1b_root_pyproject_toml_is_infra(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert _has_python_test_infra(tmp_path) is True


def test_ac1c_root_conftest_is_infra(tmp_path):
    (tmp_path / "conftest.py").write_text("# conftest\n")
    assert _has_python_test_infra(tmp_path) is True


def test_ac1d_nested_tests_dir_test_file_at_depth_2(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    assert _has_python_test_infra(tmp_path) is True


def test_ac1e_tests_dir_with_only_non_py_files_is_not_infra(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "readme.md").write_text("hi\n")
    (tmp_path / "tests" / "helper.ts").write_text("export {};\n")
    assert _has_python_test_infra(tmp_path) is False


def test_ac1f_oserror_in_scan_fails_safe_to_infra(tmp_path, monkeypatch):
    real_iterdir = Path.iterdir

    def boom(self):
        raise OSError("eperm")

    monkeypatch.setattr(Path, "iterdir", boom, raising=True)
    assert _has_python_test_infra(tmp_path) is True


def test_ac1g_ts_only_tree_is_not_infra(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "x"}\n')
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.test.ts").write_text("test('x', () => {});\n")
    (tmp_path / "src" / "lib.ts").write_text("export const x = 1;\n")
    assert _has_python_test_infra(tmp_path) is False


def test_ac1h_setup_cfg_without_tool_pytest_is_not_infra(tmp_path):
    (tmp_path / "setup.cfg").write_text("[metadata]\nname = x\n")
    assert _has_python_test_infra(tmp_path) is False


def test_ac1h2_setup_cfg_with_tool_pytest_is_infra(tmp_path):
    (tmp_path / "setup.cfg").write_text("[metadata]\nname = x\n[tool:pytest]\naddopts = -q\n")
    assert _has_python_test_infra(tmp_path) is True


def test_ac1i_pyproject_with_only_pytest_ini_options_is_infra(tmp_path):
    """Facets mandate: configured pytest is infrastructure - the bypass
    must not fire when pytest is configured even if no test files exist."""
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-q'\n")
    assert _has_python_test_infra(tmp_path) is True


# ---------------------------------------------------------------------------
# AC2: bypass path (non-Python worktree)
# ---------------------------------------------------------------------------

def test_ac2_non_python_worktree_bypass_creates_pr_with_marker(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "package.json").write_text('{"name": "claude-view"}\n')
    (worktree / "Cargo.toml").write_text("[package]\nname = 'x'\n")
    # Concluded, non-empty diff, NO run_tests outcomes at all.
    result = _fixer_result(last_test_outcome=None)
    pr = MagicMock()
    url, body, shaped_dir, mock_friction = _run(tmp_path, worktree, result, pr)

    assert url == "http://x/pulls/1"
    pr.assert_called_once()
    # PR body carries the Test-gate section AND the machine-readable marker.
    assert "## Test gate" in body
    assert "lapis-test-gate: bypassed-no-python-test-infra" in body
    assert "## Test outcome" in body
    # Friction mem entry written via the existing D6 helper (witnessed).
    mock_friction.assert_called_once()
    kw = mock_friction.call_args.kwargs
    assert kw["node_id"] == "test-gate-bypassed-no-python-test-infra"
    assert kw["task_id"] == "abc123"
    # Tail log exists with a BYPASSED line.
    tail = shaped_dir / "abc123-tail.log"
    assert tail.exists()
    assert any("BYPASSED" in line for line in tail.read_text().splitlines())


# ---------------------------------------------------------------------------
# AC3: fail-closed preserved (Python worktree, gate math unchanged)
# ---------------------------------------------------------------------------

def test_ac3_python_worktree_no_tests_touched_last_outcome_zero_no_pr(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    # Model shipped a production-only diff; last outcome: no tests ran.
    result = _fixer_result(last_test_outcome={"passed": 0, "failed": 0})
    pr = MagicMock()
    url, body, shaped_dir, mock_friction = _run(tmp_path, worktree, result, pr)

    assert url == ""
    pr.assert_not_called()
    mock_friction.assert_not_called()
    # Tail log carries the gate-failed reason.
    tail = shaped_dir / "abc123-tail.log"
    assert tail.exists()
    assert any("gate FAILED" in line for line in tail.read_text().splitlines())
    assert body == ""


# ---------------------------------------------------------------------------
# AC4: positive-only path untouched (Python worktree, touched test green)
# ---------------------------------------------------------------------------

def test_ac4_python_worktree_touched_test_green_pr_unmarked(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    transcript = [{
        "tool_name": "write_file",
        "arguments": {"path": "tests/test_foo.py", "content": "def test_x():\n    pass\n"},
        "result": "wrote",
    }]
    result = _fixer_result(
        last_test_outcome={"passed": 2, "failed": 0, "errors": 0},
        transcript=transcript,
    )
    pr = MagicMock()
    url, body, shaped_dir, mock_friction = _run(tmp_path, worktree, result, pr)

    assert url == "http://x/pulls/1"
    pr.assert_called_once()
    mock_friction.assert_not_called()
    # Non-bypassed body: NO Test-gate section, NO marker comment.
    assert "## Test gate" not in body
    assert "lapis-test-gate" not in body
    assert "## Test outcome" in body
    assert "## Steps" in body
    # Tail log shows gate PASSED (not BYPASSED).
    tail = (shaped_dir / "abc123-tail.log").read_text()
    assert "gate PASSED" in tail
    assert "BYPASSED" not in tail


# AC5: _tail_log helper
# ---------------------------------------------------------------------------

def test_ac5_tail_log_appends_timestamped_lines(tmp_path):
    shaped = tmp_path / "shaped"
    with patch.object(sr, "room_path", lambda name: shaped):
        _tail_log("t-1", "gate PASSED (model_touched_tests=[])")
        _tail_log("t-1", "create_pr OK url=http://x/pulls/9")
    lines = (shaped / "t-1-tail.log").read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        ts = line[:20]
        assert ts[10] == "T" and ts.endswith("Z"), f"bad timestamp prefix: {ts}"
    assert lines[0][21:] == "gate PASSED (model_touched_tests=[])"
    assert lines[1][21:] == "create_pr OK url=http://x/pulls/9"


def test_ac5_tail_log_swallows_oserror(tmp_path):
    # room_path resolves to an EXISTING DIRECTORY: open(dir, "a") raises
    # IsADirectoryError (an OSError) - must be swallowed, not propagated.
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    with patch.object(sr, "room_path", lambda name: blocker):
        _tail_log("t-2", "this must not raise")  # no exception == pass


def test_ac5_tail_log_writes_only_the_given_reason(tmp_path):
    """No token-looking substrings: the written line is exactly the
    timestamp + the (already-redacted, caller-truncated) reason string."""
    shaped = tmp_path / "shaped"
    reason = (
        "git push failed rc=128: remote: HTTP Basic: Access denied "
        "(token=***); <500-char stderr tail truncated by caller>"
    )
    with patch.object(sr, "room_path", lambda name: shaped):
        _tail_log("t-3", reason)
    line = (shaped / "t-3-tail.log").read_text().splitlines()[0]
    assert line[21:] == reason


# ---------------------------------------------------------------------------
# AC6: module green (meta)
# ---------------------------------------------------------------------------

def test_ac6_module_green_meta():
    """AC6 is proven by the harness's own in-dispatch run_tests (the
    touched-tests gate path running THIS file). In the local test env we
    pin the structural contract instead: the module imports clean and the
    deliverables' entry points exist; the marker string is present in the
    module source (bypassed runs emit it; non-bypassed runs do not)."""
    import agents_core.shaped_runner as mod
    assert callable(mod._has_python_test_infra)
    assert callable(mod._tail_log)
    src = Path(mod.__file__).read_text()
    assert "lapis-test-gate: bypassed-no-python-test-infra" in src