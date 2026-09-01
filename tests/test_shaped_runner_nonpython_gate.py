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


# ---------------------------------------------------------------------------
# agents-core-local-fixer-gate-perception-v0 (D1-D5 + the 8-test skeleton)
#
# Full-drive pattern: integration tests 3-8 drive sr._run_local_fixer /
# sr._run_local_opencode end-to-end via the _run helper above (the
# _gate_decision mirror in tests/test_shaped_runner_test_gate.py is the
# WRONG home - it cannot observe the tail-log rerun=true line).
#
# Canned outputs are the real D3 strings (mem
# finding/local-fixer-gate-0-0-last-call-wins-poison-file-2026-08-31):
# the D3 dispatch's own tests were green ("54 passed in 0.09s") while its
# last repo-wide run_tests hit a pre-existing collection error in an
# unrelated file (conductor scripts/forgejo_webhook.py:30 opens
# /var/log/rag-index.log at import time -> PermissionError for the
# non-root fixer -> collection aborts -> passed=0 -> bare 0/0 gate
# failure).
# ---------------------------------------------------------------------------

import subprocess as _subprocess  # noqa: E402

D3_GREEN_SUMMARY = "54 passed in 0.09s\n"
D3_RED_SUMMARY = "2 failed, 1 passed in 0.1s\n"
RC4_LAST_LINE = "no tests ran in 0.00s\n"  # rc=4 AND rc=5 ground truth


def _tail_lines(shaped_dir) -> list[str]:
    tail = shaped_dir / "abc123-tail.log"
    assert tail.exists(), "tail log missing"
    return tail.read_text().splitlines()


def _touched_transcript(path: str = "tests/test_foo.py") -> list[dict]:
    return [{
        "tool_name": "write_file",
        "arguments": {"path": path, "content": "def test_x():\n    pass\n"},
        "result": "wrote",
    }]


def _py_outcome(stdout: str, rc: int) -> dict:
    from agents_core.gw_agent import _parse_pytest_outcome
    return _parse_pytest_outcome(stdout, rc, False)


def _run_with_rerun(tmp_path, worktree, result, mock_pr, rerun_side_effect,
                    mock_rerun=None):
    """_run + a mock for sr._gate_targeted_rerun. The global subprocess.run
    patch (git args) is left alone: the re-run is mocked at the helper
    boundary so the two subprocess.run consumers never collide."""
    mock_friction = MagicMock()
    if mock_rerun is None:
        mock_rerun = MagicMock(side_effect=rerun_side_effect)
    spec_path = _make_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    mock_pr.return_value = {"html_url": "http://x/pulls/1"}
    shaped_dir = tmp_path / "shaped-rt"
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
        patch.object(sr, "_gate_targeted_rerun", mock_rerun),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))
    if mock_pr.call_args:
        body = mock_pr.call_args.kwargs.get("body", "")
    else:
        body = ""
    return url, body, shaped_dir, mock_friction, mock_rerun


# 1. helper-level: the re-run parses a green targeted run (the real D3
#    string: the D3 tests themselves were green).
def test_rerun_on_collection_error_last_outcome(tmp_path, monkeypatch):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    fake = MagicMock(returncode=0, stdout=D3_GREEN_SUMMARY, stderr="")
    monkeypatch.setattr(_subprocess, "run", MagicMock(return_value=fake))
    out = sr._gate_targeted_rerun(str(worktree), {"tests/test_foo.py"})
    assert out is not None
    assert out["passed"] == 54
    assert out["failed"] == 0
    assert out["returncode"] == 0
    # The re-run mirrors the fixer's own run_tests invocation.
    argv = _subprocess.run.call_args.args[0]
    assert argv[1:3] == ["-m", "pytest"]
    assert argv[-1] == "-q"
    assert "tests/test_foo.py" in argv


# 2a. the re-run fires when the model's last outcome is None (aborted run,
#     the D3-retry shape).
def test_rerun_fires_when_last_outcome_none(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    result = _fixer_result(
        last_test_outcome=None,
        transcript=_touched_transcript(),
    )
    mock_rerun = MagicMock(return_value=_py_outcome(D3_GREEN_SUMMARY, 0))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    mock_rerun.assert_called_once()
    assert mock_rerun.call_args.args[0] == str(worktree)
    assert mock_rerun.call_args.args[1] == {"tests/test_foo.py"}
    # rc==0 decision rule: the gate flips fail->pass.
    assert url == "http://x/pulls/1"
    assert any("gate PASSED" in line and "rerun=true" in line
               for line in _tail_lines(shaped_dir))


# 2b. the re-run fires on an error-dict last outcome (D3: no-valid-last-
#     result is a re-run trigger, not a spurious 0/0 count).
def test_rerun_fires_on_error_dict_last_outcome(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    result = _fixer_result(
        last_test_outcome={"error": "shell metacharacters not allowed in test args"},
        transcript=_touched_transcript(),
    )
    mock_rerun = MagicMock(return_value=_py_outcome(D3_GREEN_SUMMARY, 0))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    mock_rerun.assert_called_once()
    assert mock_rerun.call_args.args[1] == {"tests/test_foo.py"}
    assert url == "http://x/pulls/1"


# 3. integration: red last outcome (D3 collection-error shape), green
#    re-run -> gate passes; tail log + PR body carry the rerun provenance.
def test_gate_passes_when_rerun_green_despite_red_last_outcome(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    red = _py_outcome(
        "ERRORS!\n"
        "ERROR tests/test_foo.py\n"
        "!!!! Interrupted: 1 error during collection !!!\n"
        "1 error in 0.05s\n",
        2,
    )
    result = _fixer_result(last_test_outcome=red,
                           transcript=_touched_transcript())
    mock_rerun = MagicMock(return_value=_py_outcome(D3_GREEN_SUMMARY, 0))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    assert url == "http://x/pulls/1"
    lines = _tail_lines(shaped_dir)
    gate_line = next(l for l in lines if "gate PASSED" in l)
    assert "rerun=true" in gate_line
    assert "errors=" in gate_line
    assert "rc=0" in gate_line
    # D5: the PR body reads the DECIDING (re-run) outcome + the annotation.
    assert "54 passed, 0 failed" in body
    assert "rerun=true" in body
    # M-8: the re-run firing writes a distinct friction entry.
    fired = [c for c in mock_friction.call_args_list
             if c.kwargs.get("error_signature") == "gate-rerun:fired"]
    assert len(fired) == 1
    assert fired[0].kwargs["task_id"] == "abc123"


# 4. integration: red re-run -> gate still fails; the journal WARN + tail
#    log carry rc= and errors= (the D3 disaster line now speaks).
def test_gate_still_fails_when_rerun_red(tmp_path, capsys):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")
    red_last = _py_outcome(
        "ERRORS!\n"
        "ERROR tests/test_foo.py\n"
        "!!!! Interrupted: 1 error during collection !!!\n"
        "1 error in 0.05s\n",
        2,
    )
    result = _fixer_result(last_test_outcome=red_last,
                           transcript=_touched_transcript())
    mock_rerun = MagicMock(
        return_value=_py_outcome(D3_RED_SUMMARY, 1))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    assert url == ""
    assert body == ""
    # Daemon-journal WARN (the "no PR" line) carries rc=1 + errors=.
    err = capsys.readouterr().err
    warn_line = next(l for l in err.splitlines()
                     if "WARN: local-fixer: test gate failed" in l)
    assert "rc=1" in warn_line
    assert "errors=" in warn_line
    # Tail log carries the same disambiguation.
    failed_line = next(l for l in _tail_lines(shaped_dir) if "gate FAILED" in l)
    assert "rc=1" in failed_line
    assert "errors=" in failed_line


# 5. the helper returns None on timeout AND on spawn error; the gate stays
#    fail-closed in both cases.
def test_rerun_returns_none_on_timeout_and_on_spawn_error(tmp_path, monkeypatch,
                                                          capsys):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (worktree / "tests").mkdir()
    (worktree / "tests" / "test_foo.py").write_text("def test_x():\n    pass\n")

    def _timeout(*a, **kw):
        raise _subprocess.TimeoutExpired(
            cmd="pytest", timeout=180, output=b"", stderr=b"")

    monkeypatch.setattr(_subprocess, "run", MagicMock(side_effect=_timeout))
    assert sr._gate_targeted_rerun(str(worktree), {"tests/test_foo.py"}) is None
    err = capsys.readouterr().err
    assert "timed out after 180s" in err

    monkeypatch.setattr(_subprocess, "run",
                        MagicMock(side_effect=OSError("spawn failed")))
    assert sr._gate_targeted_rerun(str(worktree), {"tests/test_foo.py"}) is None
    err = capsys.readouterr().err
    assert "spawn error" in err

    # Gate integration: an unusable re-run keeps the fail-closed verdict.
    # (A 0/0 last outcome - NOT None: a None last outcome with a touched
    # file is exactly the D3-retry shape the re-run exists to rescue, and
    # the harness's own transcript write would have crashed the run before
    # the gate. The 0/0 shape is what the gate historically mislabeled.)
    result = _fixer_result(
        last_test_outcome={"passed": 0, "failed": 0, "errors": 0,
                          "timed_out": False, "returncode": 2,
                          "summary": "1 error in 0.05s",
                          "output_tail": "1 error in 0.05s"},
        transcript=_touched_transcript(),
    )
    # A re-run that returns a usable but RED outcome (rc=2 collection
    # error in the re-run itself): the gate stays fail-closed and the
    # deciding (re-run) outcome is named.
    mock_rerun = MagicMock(
        return_value=_py_outcome(
            "ERRORS!\n"
            "ERROR tests/test_foo.py\n"
            "!!!! Interrupted: 1 error during collection !!!\n"
            "1 error in 0.05s\n",
            2,
        ))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    assert url == ""
    lines = _tail_lines(shaped_dir)
    assert any("gate FAILED" in l for l in lines)
    # The fail-closed verdict is now disambiguated (D2): the 0/0 shape is
    # named with errors/rc/summary instead of a bare passed=0 failed=0.
    failed_line = next(l for l in lines if "gate FAILED" in l)
    assert "errors=1" in failed_line
    assert "rc=2" in failed_line
    assert "rerun=true" in failed_line


# 6. argv option-injection guard: a leading-dash touched path is ./-prefixed
#    so pytest cannot consume it as an option.
def test_rerun_prefixed_dash_path(tmp_path, monkeypatch):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    fake = MagicMock(returncode=0, stdout=D3_GREEN_SUMMARY, stderr="")
    monkeypatch.setattr(_subprocess, "run", MagicMock(return_value=fake))
    out = sr._gate_targeted_rerun(str(worktree), {"-x/tests/test_foo.py"})
    assert out is not None
    argv = _subprocess.run.call_args.args[0]
    assert "./-x/tests/test_foo.py" in argv
    assert "-x/tests/test_foo.py" not in argv


# 7. the legacy no-touched-tests branch is UNCHANGED: no touched tests ->
#    fail-closed, and the re-run never fires.
def test_legacy_branch_unchanged_no_touched_tests(tmp_path):
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    # Production-only diff, no test writes, last outcome None.
    result = _fixer_result(last_test_outcome=None)
    mock_rerun = MagicMock(return_value=_py_outcome(D3_GREEN_SUMMARY, 0))
    url, body, shaped_dir, mock_friction, _ = _run_with_rerun(
        tmp_path, worktree, result, MagicMock(), None, mock_rerun=mock_rerun)
    assert url == ""
    mock_rerun.assert_not_called()
    assert any("gate FAILED" in l for l in _tail_lines(shaped_dir))


# 8. opencode path (D4): the F4 re-run's rc=4 is named explicitly in the
#    gate-failed WARN instead of folding into bare 0/0.
#
# Rev-4 mechanics (LOAD-BEARING - two prior attempts died here):
#   (a) spec timeout_s=600 (> TAIL_BUDGET 300): the S4d budget check fails
#       closed BEFORE the gate when timeout_s <= 300 (loop_budget =
#       timeout_s - TAIL_BUDGET - setup_lag <= 0 -> "no model-loop budget"
#       ERROR + return "" before the gate-failed WARN is ever reached);
#   (b) subprocess.Popen is mocked (the opencode-binary argv) and completes
#       immediately with rc=0;
#   (c) a test file is seeded into the worktree AND staged (git add) - F4's
#       touched source is the STAGED DIFF (_collect_diff_touched_tests),
#       not the transcript;
#   (d) the F4 re-run's single subprocess.run is argv-dispatched: git args
#       -> rc=0 empty output; the [sys.executable, "-m", "pytest", ...]
#       invocation -> the canned rc=4 result (last line "no tests ran in
#       0.00s").
def test_f4_rc4_rc5_logged(tmp_path, monkeypatch, capsys):
    import subprocess
    from agents_core import shaped_runner, worktree

    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(["git", "init", "--bare", str(origin)],
                   check=True, capture_output=True)
    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "-C", str(seed), "init", "-b", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    (seed / "f.py").write_text("VALUE = 0\n")
    (seed / "tests").mkdir()
    (seed / "tests" / "test_fast.py").write_text(
        "def test_ok():\n    assert True\n")
    subprocess.run(["git", "-C", str(seed), "add", "-A"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "remote", "add", "origin",
                    str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                   check=True, capture_output=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)

    spec = {
        "task_id": "loco-t8",
        "target_id": "t8-target",
        "repo": "agents-core",
        "base_branch": "main",
        "slug": "loco",
        "prompt": "fix the thing",
        # (a) MUST be > TAIL_BUDGET (300) - see the rev-4 note.
        "timeout_s": 600,
        "opencode_model": "gravitywell/gravitywell-slot1",
    }

    # (b) the opencode model loop: a REAL fake opencode binary (no Popen
    # patch - patching subprocess.Popen would also break the real
    # subprocess.run calls subprocess.run makes internally, since it
    # instantiates through the module-global Popen). The engine resolves
    # the binary from OPENCODE_BIN; the fake plays the model: it leaves a
    # staged test file in the worktree (its cwd) and exits 0. The engine's
    # setup_worktree force-removes any pre-existing worktree, so the seed
    # MUST come from the model loop, not from a pre-created worktree.
    opencode_fake = tmp_path / "fake-opencode"
    opencode_fake.write_text(
        "#!/bin/sh\n"
        "# fake opencode: seed + stage the touched test file, exit 0\n"
        "mkdir -p tests\n"
        "printf 'def test_t():\\n    assert True\\n' > tests/test_touched.py\n"
        "git add tests/test_touched.py\n"
        "exit 0\n"
    )
    opencode_fake.chmod(0o755)
    monkeypatch.setenv("OPENCODE_BIN", str(opencode_fake))

    # Capture the real subprocess.run BEFORE the patch context: the engine's
    # deterministic tail needs TRUE answers from the real worktree (F1's
    # `diff --cached`/`status --porcelain`/`add -A`, F2's rev-parse pair),
    # and a mock that re-invokes the patched subprocess.run would recurse.
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        argv0 = cmd[0] if isinstance(cmd, (list, tuple)) else str(cmd)
        if argv0 == "git" or argv0.endswith("/git"):
            # git args -> run for real against the real worktree (the fake
            # opencode staged tests/test_touched.py, so the true staged diff
            # and status are non-empty; the F2 rev-parse pair is equal on
            # both sides).
            return real_run(list(cmd), *a, **kw)
        # the [sys.executable, "-m", "pytest", ...] F4 re-run -> canned rc=4
        # (ground truth: last line "no tests ran in 0.00s").
        return MagicMock(returncode=4,
                         stdout="no tests ran in 0.00s\n",
                         stderr="")

    monkeypatch.setattr(worktree, "WORKTREE_ROOT", tmp_path / "wtroot")
    monkeypatch.setattr(shaped_runner, "room_path",
                        lambda key, *parts, **kw: tmp_path / "artifacts")
    with patch("agents_core.doorman_client.DoormanClient") as mock_dm, \
         patch("agents_core.forgejo.create_pr") as mock_pr, \
         patch.object(subprocess, "run", side_effect=fake_run):
        mock_dm.return_value.acquire.return_value = {
            "status": "serving", "work_id": "loco-t8-berth-sup",
        }
        url = shaped_runner._run_local_opencode(spec, str(clone))

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert any("test gate failed" in l for l in err.splitlines()), (
        "gate-failed WARN missing; stderr was:\n" + err)
    warn_line = next(l for l in err.splitlines()
                     if "WARN: local-opencode: test gate failed" in l)
    # (d) the D4 change names f4-rc=4 explicitly instead of bare 0/0.
    assert "f4-rc=4 touched-path-missing" in warn_line
    assert "rc=4" in warn_line