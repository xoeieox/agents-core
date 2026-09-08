"""fixer-reception-v0 (leg 1, D1): the slug re-home regression tests.

The legacy local path must push to the VERIFIED existing branch when the
worktree setup resolved existing_branch (fixer_retry with a spec-carried
branch), NOT the `lapis/<tid>/local` slug default - and the post-push
open-PR scan must then find the SAME PR advanced. A push failure logs at
ERROR + records a pm:push-failed observation in the run log.

Pattern: tests/test_shaper_backend_url_passthrough.py (patches
subprocess.run + call_gw_agent + worktree + forgejo, calls
sr._run_local_fixer directly); a subprocess.run side_effect captures the
push ref.

PATCHING NOTE (reviewer cycle-1, HIGH): `_run_local_fixer` resolves
`call_gw_agent` / `setup_worktree` / `teardown_worktree` via function-local
imports (`from agents_core.gw_agent import call_gw_agent` etc. inside the
try block), so a module-level patch of the SOURCE attribute
(`agents_core.gw_agent.call_gw_agent`) is NOT sufficient on its own: the
local import re-binds the name from the source module at call time - which
works only while the patch is active, and the shared `subprocess.run`
attribute is only intercepted by patching the real subprocess module
(`patch("subprocess.run")` patches the module attribute that every
`import subprocess` / `import subprocess as _x` resolves to). These tests
patch the source-module attributes (the names the function-local imports
re-resolve at call time) AND the real subprocess module's run attribute,
so each test genuinely exercises the `_run_local_fixer` push path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agents_core.shaped_runner as sr


def _make_retry_spec(tmp_path: Path, existing_branch: str) -> dict:
    return {
        "model": "gravitywell-122b",
        "engine": "local-fixer",
        "system": "you are a fixer",
        "prompt": "fix the bug",
        "timeout_s": 60,
        "capture_meta": False,
        "target_id": "my-target-v0",
        "repo": "agents-core",
        "task_id": "retry-abc123",
        "slot_id": "retry-abc123",
        "base_branch": "main",
        "agent_type": "fixer_retry",
        "existing_branch": existing_branch,
    }


def _good_result() -> dict:
    return {
        "final_diff": "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n",
        "concluded": True,
        "last_test_outcome": {"passed": 3, "failed": 0},
        "steps": [{"tool": "read_file"}],
    }


def _fake_handle(worktree: Path) -> MagicMock:
    h = MagicMock()
    h.path = worktree
    h.env = {}
    return h


def _make_git_repo(worktree: Path) -> None:
    """Make the worktree a real git repo with one committed file.

    The tail's `git commit` fails on an empty index (nothing to commit) -
    a failed commit returns "" before the push, so the push never happens
    and the push-ref assertions cannot fire. A real repo with a tracked
    file lets the tail's checkout -B / add -A / commit / push run against
    the fake subprocess.run (the push ref is captured from the argv).
    """
    subprocess.run(["git", "init", "-q", str(worktree)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t"], check=True, capture_output=True, cwd=str(worktree))
    subprocess.run(["git", "config", "user.name", "t"], check=True, capture_output=True, cwd=str(worktree))
    (worktree / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], check=True, capture_output=True, cwd=str(worktree))
    subprocess.run(["git", "commit", "-q", "-m", "seed"], check=True, capture_output=True, cwd=str(worktree))


@pytest.fixture
def fake_py_infra(monkeypatch):
    """Force _has_python_test_infra True so the tail reaches the push
    (the gate does NOT bypass on an empty tmp worktree). The test
    outcome (3 passed, 0 failed) carries the gate through."""
    monkeypatch.setattr(sr, "_has_python_test_infra", lambda cwd: True)


def test_retry_run_pushes_to_verified_existing_branch(tmp_path, fake_py_infra, monkeypatch):
    """AC1: a retry-shaped run with existing_branch set pushes HEAD to that
    branch (not lapis/<tid>/local) and the post-push open-PR scan reports
    the EXISTING PR advanced (the returned URL is the existing PR's, not a
    create_pr'd second one)."""
    existing = "lapis/my-target-v0/forced"
    spec = _make_retry_spec(tmp_path, existing)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _make_git_repo(worktree)
    monkeypatch.setenv("GPU_QUEUE_DIR", str(tmp_path / "gpu-queue"))
    push_refs: list[str] = []

    def fake_subprocess_run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 3 and cmd[1] == "-C" and cmd[3] == "push":
            # cmd = ["git", "-C", cwd, "push", "origin", "HEAD:<branch>"]
            # (the subcommand is at index 3)
            push_refs.append(cmd[-1])
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", side_effect=AssertionError(
            "create_pr must NOT be called - the open-PR scan finds the existing PR")),
        patch("agents_core.forgejo.get_open_prs",
              return_value=[{"head": {"ref": existing},
                             "html_url": "http://x/pulls/945"}]),
        patch("subprocess.run", side_effect=fake_subprocess_run),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == "http://x/pulls/945"
    # The push ref is the verified existing branch, not the slug default.
    assert push_refs, "no push command captured"
    assert push_refs[0] == f"HEAD:{existing}"
    assert "lapis/my-target-v0/local" not in push_refs[0]


def test_retry_run_worktree_set_up_at_existing_branch(tmp_path, fake_py_infra, monkeypatch):
    """The worktree is provisioned at the verified existing branch (the
    setup path is unchanged; the fix threads the SAME ref through to the
    push)."""
    existing = "lapis/my-target-v0/forced"
    spec = _make_retry_spec(tmp_path, existing)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _make_git_repo(worktree)
    monkeypatch.setenv("GPU_QUEUE_DIR", str(tmp_path / "gpu-queue"))

    setup_refs: list[str] = []

    def fake_setup(task_id, effective_cwd, ref):
        setup_refs.append(ref)
        return _fake_handle(worktree)

    def fake_subprocess_run(cmd, **kwargs):
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_result(), [])),
        patch("agents_core.worktree.setup_worktree", side_effect=fake_setup),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.get_open_prs",
              return_value=[{"head": {"ref": existing},
                             "html_url": "http://x/pulls/945"}]),
        patch("subprocess.run", side_effect=fake_subprocess_run),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert setup_refs == [existing]


def test_initial_fixer_run_still_pushes_slug_default(tmp_path, fake_py_infra, monkeypatch):
    """Behavior unchanged for the initial-dispatch case (no
    existing_branch): the tail still creates and pushes
    lapis/<tid>/local and opens a fresh PR via create_pr."""
    spec = _make_retry_spec(tmp_path, "")
    spec.pop("existing_branch")
    spec.pop("agent_type")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _make_git_repo(worktree)
    monkeypatch.setenv("GPU_QUEUE_DIR", str(tmp_path / "gpu-queue"))

    push_refs: list[str] = []

    def fake_subprocess_run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 3 and cmd[1] == "-C" and cmd[3] == "push":
            # cmd = ["git", "-C", cwd, "push", "origin", "HEAD:<branch>"]
            # (the subcommand is at index 3)
            push_refs.append(cmd[-1])
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr",
              return_value={"html_url": "http://x/pulls/100"}),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch("subprocess.run", side_effect=fake_subprocess_run),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == "http://x/pulls/100"
    assert push_refs == ["HEAD:lapis/my-target-v0/local"]


def test_push_failure_records_pm_push_failed_observation(tmp_path, fake_py_infra, monkeypatch):
    """D1 push-failure partition: a failed push logs at ERROR (the
    pm:push-failed observation lands in the run log) and returns "" with
    no PR; the work survives in the transcript + local refs/wip (no
    salvage-PR fallback on this path)."""
    existing = "lapis/my-target-v0/forced"
    spec = _make_retry_spec(tmp_path, existing)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    _make_git_repo(worktree)
    # _tail_log resolves <shaped-dir> via room_path("gpu_queue.shaped"),
    # whose classmap entry has NO env_var override - the GPU_QUEUE_DIR env
    # only redirects the gpu_queue key. Point ROOM_ROOT at tmp_path so the
    # tail log lands under tmp_path/srv/lapis/gpu-queue/shaped.
    monkeypatch.setenv("ROOM_ROOT", str(tmp_path / "room"))

    def fake_subprocess_run(cmd, **kwargs):
        if isinstance(cmd, list) and len(cmd) > 3 and cmd[1] == "-C" and cmd[3] == "push":
            # cmd = ["git", "-C", cwd, "push", "origin", "HEAD:<branch>"]
            # (the subcommand is at index 3)
            return MagicMock(returncode=1, stdout="",
                             stderr="! [rejected] branch (non-fast-forward)")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch("agents_core.forgejo.create_pr", side_effect=AssertionError(
            "create_pr must NOT be called after a push failure")),
        patch("subprocess.run", side_effect=fake_subprocess_run),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    # The pm:push-failed observation landed in the run log
    # (<shaped-dir>/<task_id>-tail.log).
    tail_log = tmp_path / "room" / "gpu-queue" / "shaped" / "retry-abc123-tail.log"
    assert tail_log.exists(), f"run log missing at {tail_log}"
    content = tail_log.read_text()
    assert "pm:push-failed" in content
    assert "rc=1" in content
    assert existing in content
