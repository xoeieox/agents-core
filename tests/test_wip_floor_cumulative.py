# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D1 (attestation-contract-v0, leg 1): the cumulative WIP floor.

The 2026-09-08 finding (finding/wip-floor-non-cumulative-salvage-prs-
2026-09-08): the WIP floor's two salvage PRs carried ONLY the last step's
test file - each WIP commit's tree was base-HEAD + this step's files (the
implicit `git reset` + `add -- <this step's paths>`), so a later step
clobbered the floor's record of earlier steps' files.

The contract: the WIP tip's tree after N steps contains every step's files
at their final content; a later step never clobbers an earlier step's file
it did not touch.

The hook is a closure inside `_run_local_fixer` (it captures `wip_ref`,
`task_id`, `wip_steps`, `wip_commit_count`, `wip_head_sha`, `_wip_git`),
so these tests drive the EXACT hook body verbatim (extracted from the
source - a drift in the hook's cumulative seam breaks the extraction and
the test fails loudly).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents_core import shaped_runner


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"git {args} failed: {r.stderr}"
    return r.stdout.strip()


def _init_repo(cwd: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=cwd, check=True)
    (cwd / "base.py").write_text("base\n")
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=cwd, check=True)


def _extract_hook_source() -> str:
    """Extract the `_wip_commit_hook` closure source verbatim (the hook is
    a closure - the test drives the exact production body, not a copy)."""
    import inspect

    src = inspect.getsource(shaped_runner._run_local_fixer)
    start = src.index("    def _wip_commit_hook(ctx: dict) -> None:")
    # the hook ends at the next line at the same indent (4 spaces) that is
    # a `def ` or a comment-block start after the hook's body.
    lines = src[start:].splitlines()
    body_lines = [lines[0]]
    for line in lines[1:]:
        if line.startswith("    ") and not line.startswith("        "):
            # a new 4-space-indented statement after the hook's 8-space body
            break
        body_lines.append(line)
    body = "\n".join(body_lines)
    assert "read-tree" in body, "the D1 cumulative seam is missing from the source"
    assert "commit-tree" in body
    return body


def _make_hook(cwd: Path, wip_ref: str, task_id: str):
    """Build the hook from the verbatim source body (the closure's
    nonlocal captures are provided by the module-level exec scope - the
    hook's `nonlocal` names resolve to the enclosing module scope of the
    exec'd wrapper)."""
    import sys as _sys

    body = _extract_hook_source()
    # de-indent the hook by 4 spaces (it's nested in _run_local_fixer)
    dedented = "\n".join(
        line[4:] if line.startswith("    ") else line for line in body.splitlines()
    )
    # the hook's `nonlocal wip_commit_count, wip_head_sha` needs those in
    # an enclosing function scope - wrap the body in a factory function
    # that provides the nonlocal bindings as its locals (the exec'd
    # module scope cannot serve as a nonlocal target).
    wrapper = (
        "def _make_hook_factory():\n"
        "    wip_commit_count = 0\n"
        "    wip_head_sha = \"\"\n"
        + "\n".join(
            "    " + l if l else l for l in dedented.splitlines()
        )
        + "\n    return _wip_commit_hook\n"
    )
    module_ns = {
        "py_compile": __import__("py_compile"),
        "subprocess": subprocess,
        "Path": Path,
        "sys": _sys,
        "wip_ref": wip_ref,
        "task_id": task_id,
        "wip_steps": [],
        "_wip_git": lambda *a: subprocess.run(
            ["git", "-C", str(cwd), *a],
            capture_output=True, text=True, timeout=30,
        ),
    }
    exec(compile(wrapper, "<wip-hook-test>", "exec"), module_ns)
    module_ns["_wip_commit_hook"] = module_ns["_make_hook_factory"]()

    def _hook(paths: list[str]) -> None:
        ctx = {
            "writeable": True,
            "cwd": str(cwd),
            "step_num": len(paths) + 1,
            "transcript": [
                {"tool_name": "write_file", "arguments": {"path": p}}
                for p in paths
            ],
        }
        # write the files into the worktree (the model's writes)
        for p in paths:
            (cwd / p).write_text(f"content-{p}\n")
        module_ns["_wip_commit_hook"](ctx)

    return _hook


class TestCumulativeFloor:
    def test_three_steps_tip_tree_contains_all_files(self, tmp_path: Path):
        """3-step synthetic run: the WIP tip's tree contains all three
        files at final content (the contract)."""
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")

        hook(["a.py"])
        hook(["b.py"])
        # step 3: b.py again (final content) + c.py
        hook(["b.py", "c.py"])

        tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")
        tree_files = _git(tmp_path, "ls-tree", "-r", "--name-only", tip)
        assert set(tree_files.splitlines()) == {"base.py", "a.py", "b.py", "c.py"}

        # final content: b.py is step 3's content
        assert _git(tmp_path, "show", f"{tip}:b.py") == "content-b.py\n"
        assert _git(tmp_path, "show", f"{tip}:a.py") == "content-a.py\n"

    def test_later_step_does_not_clobber_earlier_file(self, tmp_path: Path):
        """A step touching file B does not alter file A's content at the
        tip (the non-cumulative bug: A would be ABSENT from the tip)."""
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")

        hook(["a.py"])
        hook(["b.py"])

        tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")
        # the non-cumulative bug: the tip's tree = base + {b.py} only.
        # The cumulative floor keeps a.py.
        assert _git(tmp_path, "show", f"{tip}:a.py") == "content-a.py\n"
        assert _git(tmp_path, "show", f"{tip}:b.py") == "content-b.py\n"

    def test_parent_is_prior_wip_tip(self, tmp_path: Path):
        """commit-tree -p is the prior WIP tip (the chain is cumulative),
        not the base HEAD."""
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")

        hook(["a.py"])
        first_tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")
        hook(["b.py"])

        second_tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")
        parent = _git(tmp_path, "rev-parse", f"{second_tip}^")
        assert parent == first_tip

    def test_read_tree_ref_and_commit_parent_are_same_sha(self, tmp_path: Path):
        """REV 2 pin: the ref given to read-tree and the ref consumed by
        commit-tree -p are the same resolved value captured once. The
        observable: the second commit's parent IS the first tip (a ref
        that moved between the two calls would desync the tree from its
        declared parent - the parent would not contain a.py's step-1
        content)."""
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")

        hook(["a.py"])
        hook(["b.py"])

        tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")
        parent = _git(tmp_path, "rev-parse", f"{tip}^")
        # the parent's tree carries step 1's file (read-tree loaded it)
        assert _git(tmp_path, "show", f"{parent}:a.py") == "content-a.py\n"

    def test_py_compile_floor_skips_commit_leaves_prior_tip(self, tmp_path: Path):
        """The py_compile floor is unchanged: a step whose touched .py
        fails to compile skips its commit, leaving the prior cumulative
        tip as the floor."""
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")

        hook(["a.py"])
        prior_tip = _git(tmp_path, "rev-parse", f"refs/heads/{ref}")

        # step 2: b.py fails py_compile -> no commit (the hook's floor)
        (tmp_path / "b.py").write_text("def broken(:\n")
        hook(["b.py"])
        assert _git(tmp_path, "rev-parse", f"refs/heads/{ref}") == prior_tip
        # and the prior tip still carries a.py (cumulative)
        assert _git(tmp_path, "show", f"{prior_tip}:a.py") == "content-a.py\n"

    def test_no_wip_commit_when_no_paths(self, tmp_path: Path):
        _init_repo(tmp_path)
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")
        # no paths -> the hook returns early (no ref created)
        hook([])
        r = subprocess.run(
            ["git", "rev-parse", f"refs/heads/{ref}"],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode != 0  # ref never created

    def test_worktree_head_unchanged(self, tmp_path: Path):
        """The hook never moves the worktree HEAD (read-tree/commit-tree
        are plumbing - HEAD stays at the base commit)."""
        _init_repo(tmp_path)
        base_head = _git(tmp_path, "rev-parse", "HEAD")
        ref = "wip-floor-test"
        hook = _make_hook(tmp_path, ref, "task-d1")
        hook(["a.py"])
        hook(["b.py"])
        assert _git(tmp_path, "rev-parse", "HEAD") == base_head
        # the worktree files are the model's (the hook staged them)
        assert (tmp_path / "a.py").read_text() == "content-a.py\n"
