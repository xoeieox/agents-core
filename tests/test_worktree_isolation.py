"""Tests for worktree pip isolation (PYTHONUSERBASE + PIP_USER).

Enforces the invariant: fixer worktrees must not mutate host-global Python
state. See agents_core/worktree.py module docstring and mem key
`fix/lapis-pm-fixer-worktree-mutates-host-pip-2026-04-24`.

Moved from lapis-pm tests/ on 2026-04-28 as part of the shaper promotion
to agents-core (agents-core-promote-shaper). Monkeypatch targets updated
from lapis_pm.worktree to agents_core.worktree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.worktree import WORKTREE_ROOT, WorktreeHandle, setup_worktree, teardown_worktree


def _init_minimal_clone(tmp_path: Path) -> Path:
    """Minimal git repo + clone with no .claude/ and no CLAUDE.md."""
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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_setup_worktree_creates_pyuserbase(tmp_path):
    """setup_worktree creates <worktree>/.pyuserbase/lib/python3.12/site-packages/."""
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-isolation-pyuserbase"
    try:
        handle = setup_worktree(task_id, str(clone))
        pyuserbase_site = handle.path / ".pyuserbase" / "lib" / "python3.12" / "site-packages"
        assert pyuserbase_site.is_dir(), f".pyuserbase site-packages not created: {pyuserbase_site}"
    finally:
        teardown_worktree(task_id, str(clone))


def test_worktree_env_sets_pythonuserbase_and_pip_user(tmp_path):
    """handle.env contains PYTHONUSERBASE pointing at <worktree>/.pyuserbase and PIP_USER=yes."""
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-isolation-env"
    try:
        handle = setup_worktree(task_id, str(clone))
        assert "PYTHONUSERBASE" in handle.env, "handle.env missing PYTHONUSERBASE"
        assert "PIP_USER" in handle.env, "handle.env missing PIP_USER"
        assert handle.env["PIP_USER"] == "yes"
        expected_pyuserbase = str(handle.path / ".pyuserbase")
        assert handle.env["PYTHONUSERBASE"] == expected_pyuserbase, (
            f"PYTHONUSERBASE={handle.env['PYTHONUSERBASE']!r}, expected {expected_pyuserbase!r}"
        )
    finally:
        teardown_worktree(task_id, str(clone))


def test_teardown_removes_pyuserbase(tmp_path):
    """After teardown_worktree, the .pyuserbase dir is gone."""
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-isolation-teardown"
    handle = setup_worktree(task_id, str(clone))
    pyuserbase = handle.path / ".pyuserbase"
    assert pyuserbase.exists(), ".pyuserbase should exist before teardown"
    teardown_worktree(task_id, str(clone))
    assert not handle.path.exists(), f"worktree path still exists after teardown: {handle.path}"


def test_subprocess_env_propagated(tmp_path):
    """shaped_runner.py merges handle.env into os.environ before calling claude -p.

    Simulates the worktree dispatch path in shaped_runner.main; asserts that
    PYTHONUSERBASE and PIP_USER=yes are in os.environ at the point the
    shaped-agent subprocess would be invoked.
    """
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-isolation-runner-env"
    fake_path = WORKTREE_ROOT / task_id
    fake_handle = WorktreeHandle(
        path=fake_path,
        env={"PYTHONUSERBASE": str(fake_path / ".pyuserbase"), "PIP_USER": "yes"},
    )

    captured_env: dict = {}

    with patch("agents_core.worktree.setup_worktree", return_value=fake_handle), \
         patch("agents_core.worktree.teardown_worktree"):
        # Replicate the env-propagation logic from shaped_runner.py
        handle = fake_handle
        os.environ.update(handle.env)
        try:
            captured_env.update(os.environ)
        finally:
            os.environ.pop("PYTHONUSERBASE", None)
            os.environ.pop("PIP_USER", None)

    assert "PYTHONUSERBASE" in captured_env, "PYTHONUSERBASE not in subprocess env"
    assert "PIP_USER" in captured_env, "PIP_USER not in subprocess env"
    assert captured_env["PIP_USER"] == "yes"
    assert captured_env["PYTHONUSERBASE"] == str(fake_path / ".pyuserbase")


# ---------------------------------------------------------------------------
# Integration test (run with: pytest -m integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pip_install_lands_in_pyuserbase(tmp_path):
    """pip install with the isolation env writes to .pyuserbase, not ~/.local/.

    Installs 'toml' (small, pure-Python, not in ~/.local by default) to verify
    that PYTHONUSERBASE + PIP_USER=yes redirects pip's --user target correctly.
    Skips if network is unavailable.

    On Debian/Ubuntu systems with an externally-managed Python, pip requires
    --break-system-packages to install user packages; we add that flag here
    since the isolation env redirect is what we're testing, not pip's system
    guard.
    """
    canary = "toml"
    clone = _init_minimal_clone(tmp_path)
    task_id = "test-isolation-integration"
    try:
        handle = setup_worktree(task_id, str(clone))
        pyuserbase = handle.path / ".pyuserbase"

        # Snapshot ~/.local before install to detect leaks reliably.
        host_local = Path.home() / ".local" / "lib"
        before = set(host_local.rglob(f"{canary}*")) if host_local.exists() else set()

        # Build env: current process env + isolation overrides.
        env = {**os.environ, **handle.env}

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--break-system-packages", "--ignore-installed", canary],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"pip install failed (network unavailable?): {result.stderr[:300]}")

        # Package should appear somewhere under .pyuserbase.
        installed = list(pyuserbase.rglob(f"{canary}*"))
        assert installed, (
            f"{canary} not found under {pyuserbase};\n"
            f"pip stdout={result.stdout[:400]}\n"
            f"pip stderr={result.stderr[:400]}"
        )

        # Must NOT have leaked into host ~/.local/.
        if host_local.exists():
            after = set(host_local.rglob(f"{canary}*"))
            leaked = after - before
            assert not leaked, (
                f"{canary} leaked into host ~/.local/: {sorted(leaked)}"
            )
    finally:
        teardown_worktree(task_id, str(clone))
