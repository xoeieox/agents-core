"""Integration test for agents_core.shaped_runner subprocess invocation.

Verifies that `python3 -m agents_core.shaped_runner <spec>` is importable
and executes correctly from inside a synthesized worktree environment
(PYTHONUSERBASE + PIP_USER set, as setup_worktree would produce).

This is the load-bearing test for the path → module resolution change:
the runner is now invoked as `python3 -m agents_core.shaped_runner <spec>`
instead of `python3 /abs/path/to/_runner.py <spec>`. Without this test,
the `-m` invocation could be silently broken until production.

agents_core.llm.call_claude_cli is stubbed via a sitecustomize.py injected
into PYTHONPATH so that no real `claude -p` subprocess is spawned.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stub_site(tmp_path):
    """Create a temp site dir with sitecustomize.py that stubs call_claude_cli.

    Python's site module runs `import sitecustomize` early during interpreter
    startup. Because PYTHONPATH entries are prepended to sys.path before
    site-packages, our stub_site/sitecustomize.py is found first and patches
    agents_core.llm.call_claude_cli before any user code runs.
    """
    stub_dir = tmp_path / "stub_site"
    stub_dir.mkdir()
    (stub_dir / "sitecustomize.py").write_text(textwrap.dedent("""\
        # Injected by test_shaped_runner_integration.py via PYTHONPATH.
        # Stubs agents_core.llm.call_claude_cli to return "ok"
        # without spawning a real `claude -p` subprocess.
        try:
            import agents_core.llm as _llm
            def _stub_call_claude_cli(*args, **kwargs):
                if kwargs.get("return_envelope"):
                    return "ok", {"num_turns": 2, "is_error": False, "usage": {}}
                return "ok"
            _llm.call_claude_cli = _stub_call_claude_cli
        except Exception:
            pass
    """))
    return stub_dir


def test_shaped_runner_module_invocation(tmp_path, stub_site):
    """python3 -m agents_core.shaped_runner <spec> returns rc=0 from a worktree env.

    The subprocess runs with:
      - PYTHONPATH prepended with stub_site so sitecustomize.py stubs call_claude_cli
      - PYTHONUSERBASE + PIP_USER set (as setup_worktree produces)
      - cwd set to a synthesized "worktree" directory
      - worktree_required=False (no git ops needed; isolation env is what matters)

    Asserts:
      - rc == 0
      - "ok" in stdout (from stubbed call_claude_cli)
      - spec file is unlinked by shaped_runner after execution
      - no Traceback in combined output
    """
    # Synthesize a worktree-like directory with pip isolation env vars.
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()
    pyuserbase = worktree_dir / ".pyuserbase"
    (pyuserbase / "lib" / "python3.12" / "site-packages").mkdir(parents=True)

    # Spec file in the shaped/ dir.
    spec_dir = tmp_path / "shaped"
    spec_dir.mkdir()
    spec_id = "testintegration0"
    spec_path = spec_dir / f"test-t-1-stub-{spec_id}.json"
    spec = {
        "agent_type": "stub",
        "target_id": "test-t-1",
        "model": "haiku",
        "timeout_s": 30,
        "system": "You are a test agent.",
        "prompt": "say ok",
        "cwd": str(worktree_dir),
        "permission_mode": "bypassPermissions",
        "capture_meta": False,
        "worktree_required": False,
    }
    spec_path.write_text(json.dumps(spec))

    # Build PYTHONPATH: prepend stub_site so sitecustomize.py is found first.
    # Also include the agents-core source root so `agents_core` is importable
    # from the worktree cwd (which is not the repo root — simulating production
    # where agents-core is editable-installed but subprocess cwd is arbitrary).
    import agents_core as _ac
    agents_core_root = str(Path(_ac.__file__).parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(stub_site), agents_core_root]
    if existing:
        parts.append(existing)
    pythonpath = ":".join(parts)

    env = {**os.environ}
    env["PYTHONPATH"] = pythonpath
    env["PYTHONUSERBASE"] = str(pyuserbase)
    env["PIP_USER"] = "yes"

    result = subprocess.run(
        [sys.executable, "-m", "agents_core.shaped_runner", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(worktree_dir),
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"shaped_runner exited {result.returncode}\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "ok" in result.stdout, (
        f"expected 'ok' in stdout; got: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert not spec_path.exists(), (
        f"spec file was not unlinked by shaped_runner: {spec_path}"
    )
    assert "Traceback" not in combined, (
        f"Traceback found in output:\n{combined}"
    )


# ---------------------------------------------------------------------------
# Regression: worktree_required=True, non-reviewer, no existing_branch
# (agents-core-reviewer-worktree-branch-checkout-v0, DoD 8)
#
# The reviewer worktree-branch-checkout fix restructures main()'s
# worktree-setup block to resolve existing_branch for ANY worktree_required
# dispatch, not just local-reviewer. A plain "claude"-engine agent type
# (e.g. "fixer") that never populates spec["existing_branch"] must still
# resolve its worktree ref to base_branch and call call_claude_cli exactly
# as before this change.
# ---------------------------------------------------------------------------


def test_claude_engine_worktree_required_without_existing_branch_unchanged(tmp_path):
    shaped = tmp_path / "shaped"
    shaped.mkdir()
    spec = {
        "model": "sonnet",
        "engine": "claude",
        "system": "",
        "prompt": "do the fix",
        "timeout_s": 30,
        "capture_meta": False,
        "task_id": "task-42",
        "base_branch": "main",
        "worktree_required": True,
        "cwd": str(tmp_path / "shared-clone"),
        "target_id": "t-1",
        "repo": "agents-core",
    }
    spec_path = shaped / "t-1-fixer-abc.json"
    spec_path.write_text(json.dumps(spec))

    worktree = tmp_path / "wt"
    worktree.mkdir()
    fake_handle = MagicMock()
    fake_handle.path = worktree
    fake_handle.env = {}

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="ok") as mock_cli,
        patch("agents_core.shaped_runner._run_local_reviewer") as mock_lr,
        patch("agents_core.worktree.setup_worktree", return_value=fake_handle) as mock_setup,
        patch("agents_core.worktree.teardown_worktree"),
        patch("subprocess.run") as mock_run,
    ):
        sr.main()

    mock_setup.assert_called_once()
    assert mock_setup.call_args.args[2] == "main", "ref must resolve to base_branch, unchanged"
    # no existing_branch -> git ls-remote verification must be skipped
    mock_run.assert_not_called()
    mock_cli.assert_called_once()
    mock_lr.assert_not_called()
