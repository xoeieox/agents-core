"""Per-task git worktree lifecycle for shaped-agent dispatch.

Why this module exists: `shaped_runner.py` operated directly on
`/srv/git/<repo>-working/` until 2026-04-23, which meant concurrent
shaped-agent runs interleaved their `git checkout/commit/push` sequences
— producing PR #15 (empty) and PR #16 (two components' work mashed
together) on 2026-04-22. See mem `fix/lapis-pm-runner-shared-working-tree-race`.

WORKTREE_ROOT is the single source of truth for the worktree path,
imported by `agents_core.claude_queue_runner` for its startup sweep.

## Python pip isolation invariant

Fixer worktrees must not mutate host-global Python state. On 2026-04-24 a
fixer ran `pip install -e .` from inside its worktree, writing an editable-
install pointer into `~/.local/lib/python3.12/site-packages/`. When
`teardown_worktree` later removed the worktree, `pip show agents-core` still
claimed installed but `from agents_core import ...` raised `ModuleNotFoundError`
host-wide, breaking `mem`, `lapis-pm`, and all transitive consumers.
See mem key `fix/lapis-pm-fixer-worktree-mutates-host-pip-2026-04-24`.

The fix: `setup_worktree` sets `PYTHONUSERBASE=<worktree>/.pyuserbase` and
`PIP_USER=yes` in the shaped agent's subprocess environment. Any `pip install`
the agent runs defaults to `--user` and lands under the worktree. When
`teardown_worktree` removes the worktree tree, `.pyuserbase` is cleaned up
automatically. The host `~/.local/` is never touched.

This isolation covers *Python pip installs only*. systemd unit modifications,
mem.db writes outside the fixer's target scope, and other side effects are
separate concerns (file them as separate targets if they bite).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WORKTREE_ROOT = Path("/tmp/lapis-pm-worktrees")

_PYUSERBASE_SITE = "lib/python3.12/site-packages"


@dataclass
class WorktreeHandle:
    """Return value of setup_worktree — bundles the worktree path with the
    isolation env vars that must be forwarded to the shaped agent subprocess.
    """
    path: Path
    env: dict[str, str]


def setup_worktree(task_id: str, repo_cwd: str, base_branch: str = "main") -> WorktreeHandle:
    """Create a detached worktree at WORKTREE_ROOT/<task_id>.

    Symlinks the parent clone's `.claude/` into the worktree root because
    `git worktree add` only materializes the tracked working tree, and
    `.claude/` (containing settings.json, hooks, and per-project
    auto-memory) is gitignored. The load-bearing `chub-inject.py`
    SessionStart hook is registered at user level in
    `~/.claude/settings.json` and fires for every `claude -p` invocation
    regardless of this symlink — so the living-context invariant
    (agents-core #7 / lapis-pm #2) is enforced there. The symlink
    carries per-project auto-memory + any local per-repo permissions
    into the worktree cwd.

    A missing `CLAUDE.md` or `.claude/settings.json` in the parent clone
    is a soft warning (emitted to stderr, captured by the claude-queue
    runner into the task output markdown), not a fatal. Dispatch
    proceeds; the shaped agent simply runs without repo-specific
    `@chub:` expansion and/or without per-repo hooks/permissions.

    Known constraint: the symlink means concurrent workers share one
    `.claude/settings.json` + `.claude/projects/*/memory/`. Safe at
    CLAUDE_QUEUE_WORKERS=2; revisit before raising beyond 4.

    Returns a WorktreeHandle whose .env dict must be merged into the
    shaped-agent subprocess environment (see shaped_runner.py). This prevents
    `pip install` from writing to the host's ~/.local/ — see module
    docstring and mem `fix/lapis-pm-fixer-worktree-mutates-host-pip-2026-04-24`.
    """
    WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORKTREE_ROOT / task_id

    if path.exists():
        _force_remove(path, repo_cwd)

    subprocess.run(
        ["git", "-C", repo_cwd, "fetch", "origin", base_branch],
        check=True, capture_output=True, timeout=60,
    )
    subprocess.run(
        ["git", "-C", repo_cwd, "worktree", "add", "--detach",
         str(path), f"origin/{base_branch}"],
        check=True, capture_output=True, timeout=60,
    )

    src_claude = Path(repo_cwd) / ".claude"
    dst_claude = path / ".claude"
    if src_claude.exists() and not dst_claude.exists():
        dst_claude.symlink_to(src_claude, target_is_directory=True)

    if not (path / "CLAUDE.md").exists():
        print(
            f"WARN: worktree_setup: CLAUDE.md missing at {path} — "
            f"shaped agent will run without repo-specific @chub: expansion "
            f"(user-level chub-inject hook still fires but finds no directives)",
            file=sys.stderr,
        )
    settings = path / ".claude" / "settings.json"
    if not settings.exists():
        print(
            f"WARN: worktree_setup: .claude/settings.json missing at {settings} — "
            f"per-repo hooks/permissions absent; user-level settings still apply",
            file=sys.stderr,
        )

    # Create the per-worktree pip user-base so pip --user installs land here
    # instead of ~/.local/. The directory is cleaned up automatically by
    # teardown_worktree (which rmtrees the entire worktree path).
    pyuserbase = path / ".pyuserbase"
    (pyuserbase / _PYUSERBASE_SITE).mkdir(parents=True, exist_ok=True)

    env = {
        "PYTHONUSERBASE": str(pyuserbase),
        "PIP_USER": "yes",
    }
    return WorktreeHandle(path=path, env=env)


def teardown_worktree(task_id: str, repo_cwd: str) -> None:
    """Best-effort cleanup. Always called from the `finally` in shaped_runner.py.

    `git worktree remove --force` + `shutil.rmtree` + `git worktree prune`
    — each step is independent, failures are swallowed (the startup sweep
    in claude_queue_runner handles anything this misses).
    """
    path = WORKTREE_ROOT / task_id
    _force_remove(path, repo_cwd)


def _force_remove(path: Path, repo_cwd: str) -> None:
    subprocess.run(
        ["git", "-C", repo_cwd, "worktree", "remove", "--force", str(path)],
        check=False, capture_output=True, timeout=60,
    )
    # Unlink the .claude symlink BEFORE rmtree. shutil.rmtree follows
    # symlinks only for the top-level path; a symlink inside the tree is
    # left alone (unlinked, not recursed). Belt-and-braces: unlink
    # explicitly so a rare rmtree implementation quirk can't destroy the
    # real .claude dir.
    claude_link = path / ".claude"
    if claude_link.is_symlink():
        try:
            claude_link.unlink()
        except OSError:
            pass
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    subprocess.run(
        ["git", "-C", repo_cwd, "worktree", "prune"],
        check=False, capture_output=True, timeout=60,
    )
