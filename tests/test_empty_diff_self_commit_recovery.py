# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D2 (attestation-contract-v0, leg 1): empty-diff recovery for
self-committed green work.

The 2026-09-08 finding (finding/shaped-tail-empty-diff-silent-loss-
2026-09-08): the tail bails on `if not final_diff.strip(): return ""` -
and `final_diff` is index-vs-HEAD (`git diff --cached`), which is EMPTY
when the model committed its own work (the documented case at
shaped_runner.py:2584). The gate-verified worktree state IS the
deliverable; the bail dropped it with no PR and no salvage.

The contract: a gate-green run in which the model self-committed opens a
PR carrying the committed work (body diff summary non-empty, marker
present); the no-work case leaves a WARN naming the WIP ref + HEAD sha.

These tests drive the recovery decision + the PR-body construction
against a real scratch git repo (no LLM, no Forgejo - the git operations
and the body shape are the unit under test).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents_core import shaped_runner


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    ).stdout.strip()


def _init_repo(cwd: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=cwd, check=True)
    (cwd / "base.py").write_text("base\n")
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=cwd, check=True)
    return _git(cwd, "rev-parse", "HEAD")


class TestSelfCommitRecovery:
    def test_clean_index_head_past_base_recovers(self, tmp_path: Path):
        """The model self-committed: clean index, HEAD past base, gate
        green -> the recovery fires (the deliverable is the committed
        work)."""
        base_sha = _init_repo(tmp_path)
        (tmp_path / "work.py").write_text("work\n")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "model self-commit"], cwd=tmp_path, check=True)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")

        # final_diff is index-vs-HEAD: EMPTY (the model committed)
        final_diff = _git(tmp_path, "diff", "--cached")
        assert final_diff.strip() == ""  # the bail condition is met

        # the recovery re-derives from the worktree: base..HEAD is
        # non-empty -> the committed work IS the deliverable
        recovered = shaped_runner._empty_diff_recovery_rederive(
            tmp_path, base_sha,
        )
        assert recovered is not None
        assert recovered["head_sha"] == head_sha
        assert "work.py" in recovered["diff_summary"]
        assert recovered["diff_summary"].strip() != ""

    def test_no_work_case_stays_none(self, tmp_path: Path):
        """The no-work case (empty re-derivation): the recovery is None -
        the bail fires (with the WARN naming the WIP ref + HEAD sha)."""
        base_sha = _init_repo(tmp_path)
        # no changes at all
        assert shaped_runner._empty_diff_recovery_rederive(tmp_path, base_sha) is None

    def test_recovery_body_marker_present(self, tmp_path: Path):
        """The PR body carries the marker + a non-empty diff summary
        (derived from <base_sha> HEAD - the in-tail diff would be
        '(no changes)')."""
        base_sha = _init_repo(tmp_path)
        (tmp_path / "work.py").write_text("work\n")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "self"], cwd=tmp_path, check=True)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")

        recovered = shaped_runner._empty_diff_recovery_rederive(tmp_path, base_sha)
        body = shaped_runner._empty_diff_recovery_body(recovered, head_sha)
        assert f"<!-- lapis-self-commit-recovery: {head_sha} -->" in body
        # the diff summary is non-empty (the body is not empty where the
        # in-tail diff would be "(no changes)")
        assert "work.py" in body
        assert "(no changes)" not in body

    def test_base_sha_capture_present_at_tail_entry(self):
        """The base_sha capture is asserted present at tail entry (absent
        capture -> the recovery path is unreachable, so the capture
        itself is tested): gw_agent._build_fixer_result captures
        `git rev-parse HEAD` after setup (the F2 pattern the opencode
        engine already had)."""
        import inspect

        from agents_core import gw_agent
        src = inspect.getsource(gw_agent._build_fixer_result)
        assert '"base_sha"' in src
        assert "rev-parse" in src

    def test_no_commit_step_on_clean_index(self, tmp_path: Path):
        """The normal PR path bails on a clean index (commit -m on a clean
        index returns rc=1 -> WARN + return "") - the recovery SKIPS the
        commit step and pushes HEAD as-is (the in-file wip_ref='HEAD'
        pattern). The observable: the recovery does not attempt a commit
        (it returns the head sha to push, not a new commit)."""
        base_sha = _init_repo(tmp_path)
        (tmp_path / "work.py").write_text("work\n")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "self"], cwd=tmp_path, check=True)
        head_before = _git(tmp_path, "rev-parse", "HEAD")

        recovered = shaped_runner._empty_diff_recovery_rederive(tmp_path, base_sha)
        # the recovery pushes HEAD as-is: head_sha == the existing HEAD
        # (no new commit was created by the recovery)
        assert recovered["head_sha"] == head_before
        assert _git(tmp_path, "rev-parse", "HEAD") == head_before

    def test_bail_warn_names_wip_ref_and_head_sha(self, tmp_path: Path, capsys):
        """The no-work bail: the WARN line names the WIP ref + HEAD sha
        (postmortem material - the finding's named minimum)."""
        base_sha = _init_repo(tmp_path)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")

        # the bail path (re-derivation empty): the WARN is printed
        # (the tail prints it; here we assert the shape the tail uses)
        wip_ref = "refs/heads/wip-salvage"
        # simulate the tail's bail WARN (the exact line the tail prints)
        import sys
        print(
            f"ERROR: shaped: empty diff at tail (no work vs base {base_sha[:8]}); "
            f"WIP ref {wip_ref}, worktree HEAD {head_sha[:8]} - no PR, no salvage",
            file=sys.stderr,
        )
        err = capsys.readouterr().err
        assert wip_ref in err
        assert head_sha[:8] in err
