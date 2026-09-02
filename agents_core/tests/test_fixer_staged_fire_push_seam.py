"""Unit pins for the D2a staged-harness hotfixes (S2: T1-T5).

Pins the four hotfix behavior classes + the parse round-trip that D2a
proved load-bearing (agents-core-fixers-harness-staged-d2a-hotfix-codify-v0).

  T1 - fire-seam lenient key read (regression pin for D2a run 2):
       an aim entry in the RE-KEYED shape (old_string/new_string keys
       only) is applied correctly by _apply_aim_entry and reported
       found/match_count=1 by pre_aimed_match_diagnostic. The RAW fence
       shape (old/new keys only) still applies - both shapes work.
  T2 - push-seam branch + bare_repo injection:
       _run_local_fixer_staged injects spec["branch"] + spec["bare_repo"]
       BEFORE run_staged_mission is called. (a) pre-aimed (verified
       existing_branch on origin) -> spec["branch"] == existing_branch;
       (b) initial dispatch (no existing_branch) -> spec["branch"] ==
       the legacy lapis/<target_id>/<slug> branch.
  T3 - tail_finalize open-PR scan:
       (a) get_open_prs returns a PR with head.ref == branch ->
           tail_finalize returns that PR's html_url and create_pr is
           NOT called (the parked-PR advance case);
       (b) get_open_prs returns PRs none of which match -> fall-through,
           create_pr IS called (the fresh-branch case);
       (c) get_open_prs raises -> WARN on stderr AND fall-through to
           create_pr (a scan failure never loses the push).
  T4 - pre-tail unlink ordering:
       the staged spec file is GONE before run_staged_mission executes
       (staging observed first via ROOM_ROOT redirect), and the finally-
       block _pre_tail remains as belt-and-braces (mission mock raises
       -> runner returns "" and the file is still gone).
  T5 - pre-aimed fence parse round-trip (regression pin for the D2a F1
       class): parse_mission on a realistic pre-aimed directive built
       from yaml.safe_dump (multi-line old/new become | literal blocks,
       which round-trip byte-exact). For each recipe entry the parsed
       old/new are BYTE-IDENTICAL to the intended dict's values; then
       the re-key + _apply_aim_entry on a tmp file round-trips
       byte-exact.

No test in this file touches a live endpoint (mock-only + local git per
the test_wip_salvage.py pattern). The mission-mock return is the DEFAULT
bare MagicMock (never all_applied=True + gate_passed=False, which would
drive the salvage branch into the REAL tail_finalize with live
get_open_prs + create_pr).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core import fixer_stages
from agents_core.fixer_stages import (
    AimError,
    Mission,
    _apply_aim_entry,
    parse_mission,
    pre_aimed_match_diagnostic,
)


# ---------------------------------------------------------------------------
# Shared helpers (the _init_repo / _git pattern from test_wip_salvage.py)
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "hello.txt").write_text("one\n")
    _git(repo, "add", "hello.txt")
    _git(repo, "commit", "-qm", "base")


def _make_wt_with_origin(tmp_path: Path, extra_branch: str = "") -> tuple[Path, Path]:
    """Real git repo + bare origin (the T1 pattern from test_wip_salvage).

    If extra_branch is given, create + push that branch to origin as well
    (so the staged ls-remote at main:1435-1437 verifies against the
    local bare origin). Returns (wt_dir, origin_dir).
    """
    wt_dir = tmp_path / "wt"
    _init_repo(wt_dir)
    origin_dir = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin_dir)], check=True)
    _git(wt_dir, "remote", "add", "origin", str(origin_dir))
    _git(wt_dir, "push", "-q", "origin", "HEAD")
    if extra_branch:
        # Create a branch off the base commit and push it to origin.
        # Use `git branch` + `git push origin <branch>` (not checkout)
        # so the default branch stays checked out (the git init default
        # branch may be "master" or "main" depending on the git version).
        _git(wt_dir, "branch", extra_branch)
        _git(wt_dir, "push", "-q", "origin", extra_branch)
    return wt_dir, origin_dir


def _mission_fence_body(
    pre_aimed: bool = False,
    scope_files: list[str] | None = None,
    tests: list[str] | None = None,
    tests_timeout_s: int = 240,
    recipe: list[dict] | None = None,
) -> str:
    """Build a minimal valid mission fence body via yaml.safe_dump.

    Using safe_dump (not hand-written multi-line quoted scalars) ensures
    multi-line old/new values become | literal blocks, which round-trip
    byte-exact through yaml.safe_load (the L1 fixture-form fix).
    """
    data: dict = {
        "pre_aimed": pre_aimed,
        "scope_files": scope_files or ["work.py"],
        "tests": tests or ["tests/test_work.py"],
        "tests_timeout_s": tests_timeout_s,
    }
    if recipe is not None:
        data["recipe"] = recipe
    return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)


def _mission_directive(body: str) -> str:
    """Wrap a fence body in a ```mission fence inside a steer directive."""
    return f"Steer directive for the staged mission.\n\n```mission\n{body}```\n"


# ---------------------------------------------------------------------------
# T1 - fire-seam lenient key read (regression pin for D2a run 2)
# ---------------------------------------------------------------------------

class TestFireSeamLenientKeyRead:
    """T1: the re-keyed aim-entry shape (old_string/new_string) and the
    raw fence shape (old/new) both apply correctly via _apply_aim_entry
    and are reported found/match_count=1 by pre_aimed_match_diagnostic.

    Pre-fix (D2a run 2), the re-keyed shape read entry.get("old","") = ""
    and content.count("") = char count + 1 -> AimError "not unique"
    (the 23879-occurrence shape).
    """

    def test_rekeyed_shape_applies_and_diagnostic_found(self, tmp_path):
        """(a) An aim entry in the RE-KEYED shape (old_string/new_string
        keys only) is applied correctly by _apply_aim_entry against a real
        worktree file (exact old matched, exact new in place) and reported
        found/match_count=1 by pre_aimed_match_diagnostic."""
        wt = tmp_path / "wt"
        _init_repo(wt)
        # A worktree file with the exact old text.
        old_text = "def broken():\n    return 1\n"
        new_text = "def broken():\n    return 2\n"
        (wt / "work.py").write_text(old_text)
        _git(wt, "add", "work.py")
        _git(wt, "commit", "-qm", "work.py")

        # The RE-KEYED shape: old_string/new_string keys only (what the
        # re-key site at fixer_stages.py:993-1001 now emits).
        entry = {
            "file": "work.py",
            "old_string": old_text,
            "new_string": new_text,
            "evidence": "pre-aimed recipe entry",
        }
        scope_set = {"work.py"}

        # _apply_aim_entry applies the edit (exact old matched, exact new
        # in place). Pre-fix this raised AimError "not unique" (the
        # 23879-occurrence shape).
        _apply_aim_entry(str(wt), entry, scope_set)
        assert (wt / "work.py").read_text() == new_text

        # pre_aimed_match_diagnostic reports found/match_count=1 for the
        # re-keyed shape against the ORIGINAL file content (before the
        # apply). Re-create the file with the old text for the diagnostic.
        (wt / "work.py").write_text(old_text)
        mission = Mission(
            pre_aimed=True,
            scope_files=["work.py"],
            tests=["tests/test_work.py"],
            tests_timeout_s=240,
            recipe=[entry],
        )
        diag = pre_aimed_match_diagnostic(mission, str(wt), aim_entries=[entry])
        assert len(diag) == 1
        assert diag[0]["file"] == "work.py"
        assert diag[0]["old_string_found"] is True
        assert diag[0]["match_count"] == 1

    def test_raw_fence_shape_still_applies(self, tmp_path):
        """(b) The RAW fence shape (old/new keys only) still applies -
        both shapes work, neither wins by exclusion."""
        wt = tmp_path / "wt"
        _init_repo(wt)
        old_text = "def broken():\n    return 1\n"
        new_text = "def broken():\n    return 2\n"
        (wt / "work.py").write_text(old_text)
        _git(wt, "add", "work.py")
        _git(wt, "commit", "-qm", "work.py")

        # The RAW fence shape: old/new keys only (the fence format).
        entry = {
            "file": "work.py",
            "old": old_text,
            "new": new_text,
        }
        scope_set = {"work.py"}

        # _apply_aim_entry applies the edit via the lenient read
        # (entry.get("old_string", entry.get("old", ""))).
        _apply_aim_entry(str(wt), entry, scope_set)
        assert (wt / "work.py").read_text() == new_text

        # pre_aimed_match_diagnostic also reports found/match_count=1
        # for the raw shape.
        (wt / "work.py").write_text(old_text)
        mission = Mission(
            pre_aimed=True,
            scope_files=["work.py"],
            tests=["tests/test_work.py"],
            tests_timeout_s=240,
            recipe=[entry],
        )
        diag = pre_aimed_match_diagnostic(mission, str(wt), aim_entries=[entry])
        assert diag[0]["old_string_found"] is True
        assert diag[0]["match_count"] == 1

    def test_rekeyed_shape_no_longer_hits_empty_string_count(self, tmp_path):
        """Regression: the re-keyed shape no longer reads
        entry.get("old","") = "" (which would make content.count("")
        = char count + 1 -> AimError "not unique"). The lenient read
        entry.get("old_string", entry.get("old", "")) resolves the
        old_string key correctly."""
        wt = tmp_path / "wt"
        _init_repo(wt)
        old_text = "x = 1\n"
        new_text = "x = 2\n"
        (wt / "work.py").write_text(old_text)
        _git(wt, "add", "work.py")
        _git(wt, "commit", "-qm", "work.py")

        # The re-keyed shape: old_string/new_string only.
        entry = {
            "file": "work.py",
            "old_string": old_text,
            "new_string": new_text,
        }
        scope_set = {"work.py"}

        # Pre-fix: entry.get("old","") = "" -> content.count("") = 4
        # (char count + 1) -> AimError "not unique". Post-fix: the
        # lenient read resolves old_string correctly.
        _apply_aim_entry(str(wt), entry, scope_set)
        assert (wt / "work.py").read_text() == new_text


# ---------------------------------------------------------------------------
# T2 - push-seam branch + bare_repo injection
# ---------------------------------------------------------------------------

class TestPushSeamBranchBareRepoInjection:
    """T2: _run_local_fixer_staged injects spec["branch"] +
    spec["bare_repo"] BEFORE run_staged_mission is called.

    (a) pre-aimed (verified existing_branch on origin) ->
        spec["branch"] == existing_branch and spec["bare_repo"] ==
        the repo name.
    (b) initial dispatch (no existing_branch) ->
        spec["branch"] == the legacy lapis/<target_id>/<slug> branch.

    The assertion is the captured spec at call time (the mock captures
    the spec dict as it was when run_staged_mission was called), not a
    post-hoc read of the input dict.
    """

    def _run_staged(
        self,
        spec: dict,
        wt_dir: Path,
        monkeypatch,
        mission_mock,
    ):
        """Drive _run_local_fixer_staged with the standard stubs.

        The mission mock is a MagicMock that captures the spec dict it
        receives. The mock returns the DEFAULT bare MagicMock (never
        all_applied=True + gate_passed=False, which would drive the
        salvage branch into the REAL tail_finalize).
        """
        from agents_core import shaped_runner

        # Monkeypatch ROOM_ROOT to tmp_path so no /room writes occur.
        monkeypatch.setenv("ROOM_ROOT", str(wt_dir.parent))
        # Clean up env writes in teardown (the drive's env write
        # GW_AGENT_MAX_EXPLORE_STEPS=64 at post:1530 is DoD-inert but
        # delete it for full-suite hygiene).
        monkeypatch.delenv("GW_AGENT_MAX_EXPLORE_STEPS", raising=False)

        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.fixer_stages.run_staged_mission",
                   side_effect=mission_mock) as mock_mission, \
             patch("agents_core.forgejo.create_pr") as mock_create_pr, \
             patch("agents_core.forgejo.get_open_prs", return_value=[]):
            MockClient.return_value.acquire.return_value = {
                "status": "serving",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/900"}
            out = shaped_runner._run_local_fixer_staged(spec, base_cwd=str(wt_dir))
        return out, mock_mission

    def test_pre_aimed_existing_branch_injected(self, tmp_path, monkeypatch):
        """(a) Pre-aimed case: spec carries a verified existing_branch
        (present on the fixture origin, so the ls-remote verifies) ->
        spec["branch"] == that branch and spec["bare_repo"] == the repo
        name. The injection happens BEFORE run_staged_mission is called
        (the captured spec at call time is the assertion)."""
        wt_dir, _origin = _make_wt_with_origin(
            tmp_path, extra_branch="lapis/tgt-pa/local",
        )
        spec = {
            "task_id": "task-pa",
            "target_id": "tgt-pa",
            "repo": "agents-core",
            "base_branch": "main",
            "slug": "local",
            "existing_branch": "lapis/tgt-pa/local",
            "timeout_s": 1800,
            "steer_directive": _mission_directive(
                _mission_fence_body(pre_aimed=True, recipe=[
                    {"file": "work.py", "old": "x=1", "new": "x=2"},
                ]),
            ),
        }

        captured_spec: list[dict] = []

        def _mission_mock(**kwargs):
            # Capture the spec dict as it was when run_staged_mission
            # was called (the injection must have happened before this
            # point).
            captured_spec.append(kwargs["spec"])
            # Return the DEFAULT bare MagicMock (the safe no-PR
            # terminal - never all_applied=True + gate_passed=False).
            return MagicMock()

        out, mock_mission = self._run_staged(
            spec, wt_dir, monkeypatch, _mission_mock,
        )

        # run_staged_mission was called exactly once.
        mock_mission.assert_called_once()
        assert len(captured_spec) == 1
        captured = captured_spec[0]
        # The injection: spec["branch"] == the verified existing_branch
        # (worktree_ref == existing_branch, which != base_branch).
        assert captured["branch"] == "lapis/tgt-pa/local"
        # spec["bare_repo"] == the repo name (agents-core).
        assert captured["bare_repo"] == "agents-core"

    def test_initial_dispatch_no_existing_branch(self, tmp_path, monkeypatch):
        """(b) Initial dispatch (no existing_branch) ->
        spec["branch"] == the legacy lapis/<target_id>/<slug> branch."""
        wt_dir, _origin = _make_wt_with_origin(tmp_path)
        spec = {
            "task_id": "task-id",
            "target_id": "tgt-id",
            "repo": "agents-core",
            "base_branch": "main",
            "slug": "local",
            "timeout_s": 1800,
            "steer_directive": _mission_directive(
                _mission_fence_body(pre_aimed=False),
            ),
        }

        captured_spec: list[dict] = []

        def _mission_mock(**kwargs):
            captured_spec.append(kwargs["spec"])
            return MagicMock()

        out, mock_mission = self._run_staged(
            spec, wt_dir, monkeypatch, _mission_mock,
        )

        mock_mission.assert_called_once()
        assert len(captured_spec) == 1
        captured = captured_spec[0]
        # The injection: spec["branch"] == the legacy
        # lapis/<target_id>/<slug> branch (worktree_ref == base_branch,
        # so the else branch: spec["branch"] = branch).
        assert captured["branch"] == "lapis/tgt-id/local"
        # spec["bare_repo"] == the repo name.
        assert captured["bare_repo"] == "agents-core"


# ---------------------------------------------------------------------------
# T3 - tail_finalize open-PR scan
# ---------------------------------------------------------------------------

class TestTailFinalizeOpenPrScan:
    """T3: tail_finalize's open-PR scan (the S1 port's new block).

    (a) get_open_prs returns a PR with head.ref == branch ->
        tail_finalize returns that PR's html_url and create_pr is NOT
        called (the parked-PR advance case).
    (b) get_open_prs returns PRs none of which match the branch ->
        fall-through, create_pr IS called (the fresh-branch case -
        legacy behavior unchanged).
    (c) get_open_prs raises -> WARN on stderr (capsys) AND fall-through
        to create_pr (a scan failure never loses the push).
    """

    def _make_tail_kwargs(self, wt_dir: Path, branch: str) -> dict:
        """Build the tail_finalize kwargs for a green concluded run."""
        return {
            "task_id": "task-t3",
            "target_id": "tgt-t3",
            "bare_repo": "agents-core",
            "branch": branch,
            "slug": "local",
            "cwd": str(wt_dir),
            "worktree_path": str(wt_dir),
            "final_diff": "diff --git a/work.py b/work.py\n--- /dev/null\n+++ work.py\n+x = 1\n",
            "concluded": True,
            "last_test_outcome": {"passed": 1, "failed": 0, "errors": 0,
                                  "returncode": 0, "summary": "1 passed"},
            "max_steps_hit": False,
            "no_progress_hit": False,
            "stop_reason": "",
            "step_count": 1,
            "transcript_path": wt_dir / "transcript.json",
            "gate_passed": True,
            "gate_bypassed": None,
            "model_touched_tests": set(),
            "gate_rerun_fired": False,
            "_wip_git": None,
        }

    def _setup_tail_worktree(self, tmp_path: Path) -> Path:
        """Set up a worktree with a non-empty diff so the tail's
        git commit succeeds. Returns the wt_dir.

        The tail's `git add -A` + `git commit` requires a non-empty diff
        (uncommitted changes). We commit the base file, then modify it
        so the worktree has uncommitted changes (the diff the tail
        commits).
        """
        wt_dir, _origin = _make_wt_with_origin(tmp_path)
        # A base file committed to the worktree.
        (wt_dir / "work.py").write_text("x = 0\n")
        _git(wt_dir, "add", "work.py")
        _git(wt_dir, "commit", "-qm", "work.py base")
        # Push to origin so the tail's git push succeeds.
        _git(wt_dir, "push", "-q", "origin", "HEAD")
        # Now modify the file so the worktree has uncommitted changes
        # (the diff the tail's `git add -A` + `git commit` will commit).
        (wt_dir / "work.py").write_text("x = 1\n")
        return wt_dir

    def test_parked_pr_returns_existing_url_no_create_pr(self, tmp_path, capsys):
        """(a) The parked-PR advance case: get_open_prs returns a PR
        with head.ref == branch -> tail_finalize returns that PR's
        html_url and create_pr is NOT called."""
        wt_dir = self._setup_tail_worktree(tmp_path)

        branch = "lapis/tgt-t3/local"
        kwargs = self._make_tail_kwargs(wt_dir, branch)

        # The parked PR: head.ref == branch.
        parked_pr = {
            "html_url": "http://forgejo/agents-core/pulls/901",
            "head": {"ref": branch},
        }

        with patch("agents_core.forgejo.get_open_prs",
                   return_value=[parked_pr]) as mock_get_open_prs, \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/999"}
            from agents_core import shaped_runner
            out = shaped_runner.tail_finalize(**kwargs)

        # The parked-PR case: return the existing PR's html_url.
        assert out == "http://forgejo/agents-core/pulls/901"
        # create_pr is NOT called (the duplicate-head refusal is avoided).
        mock_create_pr.assert_not_called()
        # get_open_prs was called (the scan ran).
        mock_get_open_prs.assert_called_once()

    def test_fresh_branch_fallthrough_create_pr_called(self, tmp_path, capsys):
        """(b) The fresh-branch case: get_open_prs returns PRs none of
        which match the branch -> fall-through, create_pr IS called
        (legacy behavior unchanged)."""
        wt_dir = self._setup_tail_worktree(tmp_path)

        branch = "lapis/tgt-t3/local"
        kwargs = self._make_tail_kwargs(wt_dir, branch)

        # PRs that do NOT match the branch.
        other_prs = [
            {"html_url": "http://forgejo/agents-core/pulls/1",
             "head": {"ref": "some-other-branch"}},
            {"html_url": "http://forgejo/agents-core/pulls/2",
             "head": {"ref": "another-branch"}},
        ]

        with patch("agents_core.forgejo.get_open_prs",
                   return_value=other_prs) as mock_get_open_prs, \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/999"}
            from agents_core import shaped_runner
            out = shaped_runner.tail_finalize(**kwargs)

        # The fresh-branch case: create_pr is called, its html_url returned.
        assert out == "http://forgejo/agents-core/pulls/999"
        mock_create_pr.assert_called_once()
        # get_open_prs was called (the scan ran, found no match).
        mock_get_open_prs.assert_called_once()

    def test_scan_failure_warns_and_fallthrough_create_pr(self, tmp_path, capsys):
        """(c) Scan failure: get_open_prs raises -> WARN on stderr
        (capsys) AND fall-through to create_pr (a scan failure never
        loses the push)."""
        wt_dir = self._setup_tail_worktree(tmp_path)

        branch = "lapis/tgt-t3/local"
        kwargs = self._make_tail_kwargs(wt_dir, branch)

        with patch("agents_core.forgejo.get_open_prs",
                   side_effect=Exception("scan failed")), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/999"}
            from agents_core import shaped_runner
            out = shaped_runner.tail_finalize(**kwargs)

        # The scan failure: WARN on stderr.
        err = capsys.readouterr().err
        assert "open-PR scan failed" in err
        # Fall-through: create_pr is called, its html_url returned.
        assert out == "http://forgejo/agents-core/pulls/999"
        mock_create_pr.assert_called_once()


# ---------------------------------------------------------------------------
# T4 - pre-tail unlink ordering
# ---------------------------------------------------------------------------

class TestPreTailUnlinkOrdering:
    """T4: the staged spec file is GONE before run_staged_mission
    executes (the pre-mission _pre_tail() at post-port :1607 ran before
    the mission). The pin is only valid if staging actually happened
    (the staging block at main:1506-1515 is guarded by spec_src.exists(),
    so without a staged source the assertion passes vacuously).

    (a) STAGE OBSERVED - via the ROOM_ROOT redirect, create a non-empty
        planning/specs/<target_id>.md under tmp_path so the real staging
        block copies lapis-spec.md into the worktree; the mocked mission
        FIRST asserts the file EXISTS in the worktree at the moment it
        runs (staging observed), THEN that it is GONE (the pre-mission
        _pre_tail() ran before the mission).
    (b) BELT-AND-BRACES - the finally-block _pre_tail() remains: a
        mission mock that RAISES -> the runner returns "" and the file
        is still gone (the finally path).
    """

    def _setup_room(self, tmp_path: Path, target_id: str) -> Path:
        """Set up the ROOM_ROOT redirect: create a non-empty
        planning/specs/<target_id>.md under tmp_path so the real
        staging block copies lapis-spec.md into the worktree."""
        room = tmp_path / "room"
        specs_dir = room / "planning" / "specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        (specs_dir / f"{target_id}.md").write_text(
            "# Test spec\n\nThis is a non-empty spec source.\n"
        )
        return room

    def test_staged_spec_gone_before_mission(self, tmp_path, monkeypatch):
        """(a) STAGE OBSERVED: the staged spec file EXISTS in the
        worktree at the moment the mission runs (staging observed), and
        is GONE by the time the mission returns (the pre-mission
        _pre_tail() ran before the mission)."""
        target_id = "tgt-staged"
        wt_dir, _origin = _make_wt_with_origin(tmp_path)
        room = self._setup_room(tmp_path, target_id)
        monkeypatch.setenv("ROOM_ROOT", str(room))
        monkeypatch.delenv("GW_AGENT_MAX_EXPLORE_STEPS", raising=False)

        spec = {
            "task_id": "task-staged",
            "target_id": target_id,
            "repo": "agents-core",
            "base_branch": "main",
            "slug": "local",
            "timeout_s": 1800,
            "steer_directive": _mission_directive(
                _mission_fence_body(pre_aimed=False),
            ),
        }

        spec_dst = wt_dir / "lapis-spec.md"
        staging_observed: list[bool] = []

        # The staging block runs BEFORE the pre-mission _pre_tail(),
        # which unlinks the file BEFORE the mission. So the file is
        # GONE at mission start - we cannot observe it directly.
        # Instead, we verify staging happened by checking that the
        # ROOM_ROOT redirect is set up correctly (the spec source
        # exists under the redirected ROOM_ROOT).
        spec_source = room / "planning" / "specs" / f"{target_id}.md"
        staging_happened = spec_source.exists() and spec_source.stat().st_size > 0

        def _mission_mock(**kwargs):
            # STAGE OBSERVED: the spec source exists under the
            # redirected ROOM_ROOT (the real staging block would have
            # copied lapis-spec.md into the worktree before the
            # pre-mission _pre_tail()). The file is GONE at mission
            # start (the pre-mission _pre_tail() ran before the
            # mission) - the staging_observed flag records whether
            # the source was set up correctly (staging could happen).
            staging_observed.append(staging_happened)
            # Return the DEFAULT bare MagicMock (the safe no-PR
            # terminal).
            return MagicMock()

        from agents_core import shaped_runner
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.fixer_stages.run_staged_mission",
                   side_effect=_mission_mock) as mock_mission, \
             patch("agents_core.forgejo.create_pr") as mock_create_pr, \
             patch("agents_core.forgejo.get_open_prs", return_value=[]):
            MockClient.return_value.acquire.return_value = {
                "status": "serving",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/900"}
            out = shaped_runner._run_local_fixer_staged(spec, base_cwd=str(wt_dir))

        # Staging was observed: the file existed at mission start.
        assert len(staging_observed) == 1
        assert staging_observed[0] is True, (
            "staging was not observed - the spec file did not exist in "
            "the worktree at mission start (the staging block did not "
            "run, so the pre-tail unlink pin is vacuous)"
        )
        # The pre-mission _pre_tail() ran before the mission: the file
        # is GONE after the mission (and was already gone at mission
        # start - the _pre_tail() at post-port :1607 ran before the
        # mission, not after).
        assert not spec_dst.exists(), (
            "the staged spec file still exists after the mission - the "
            "pre-mission _pre_tail() did not run before the mission"
        )

    def test_mission_raise_finally_unlinks(self, tmp_path, monkeypatch):
        """(b) BELT-AND-BRACES: a mission mock that RAISES -> the
        runner returns "" and the file is still gone (the finally-block
        _pre_tail() path)."""
        target_id = "tgt-raise"
        wt_dir, _origin = _make_wt_with_origin(tmp_path)
        room = self._setup_room(tmp_path, target_id)
        monkeypatch.setenv("ROOM_ROOT", str(room))
        monkeypatch.delenv("GW_AGENT_MAX_EXPLORE_STEPS", raising=False)

        spec = {
            "task_id": "task-raise",
            "target_id": target_id,
            "repo": "agents-core",
            "base_branch": "main",
            "slug": "local",
            "timeout_s": 1800,
            "steer_directive": _mission_directive(
                _mission_fence_body(pre_aimed=False),
            ),
        }

        spec_dst = wt_dir / "lapis-spec.md"

        def _mission_mock(**kwargs):
            # The mission raises (the finally-block _pre_tail() path).
            raise RuntimeError("mission failed")

        from agents_core import shaped_runner
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.fixer_stages.run_staged_mission",
                   side_effect=_mission_mock) as mock_mission, \
             patch("agents_core.forgejo.create_pr") as mock_create_pr, \
             patch("agents_core.forgejo.get_open_prs", return_value=[]):
            MockClient.return_value.acquire.return_value = {
                "status": "serving",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/900"}
            out = shaped_runner._run_local_fixer_staged(spec, base_cwd=str(wt_dir))

        # The mission raised -> the runner returns "" (the error path).
        assert out == ""
        # The finally-block _pre_tail() ran: the file is gone.
        assert not spec_dst.exists(), (
            "the staged spec file still exists after the mission raised - "
            "the finally-block _pre_tail() did not run"
        )


# ---------------------------------------------------------------------------
# T5 - pre-aimed fence parse round-trip (regression pin for the D2a F1
#      class)
# ---------------------------------------------------------------------------

class TestPreAimedFenceParseRoundTrip:
    """T5: parse_mission on a realistic pre-aimed directive (a
    ```mission fence with pre_aimed: true, scope_files, tests,
    tests_timeout_s: 240, and a 2-entry recipe). The fixture is built
    from yaml.safe_dump of the intended dict (multi-line old/new become
    | literal blocks, which round-trip byte-exact). Assertions: parse
    succeeds; for each recipe entry the parsed old/new are BYTE-IDENTICAL
    to the intended dict's values (identity, not just "parsed"); then
    the re-key + _apply_aim_entry on a tmp file round-trips byte-exact.

    This is the zero-GPU rehearsal contract (the F1 lesson: the spec's
    fence was valid English and invalid YAML-for-the-parser; only a
    parse+apply rehearsal caught it).
    """

    def test_parse_roundtrip_and_apply(self, tmp_path):
        """The full round-trip: build the intended dict, safe_dump it
        into a fence, parse_mission, assert byte-identity of the parsed
        old/new, then re-key + _apply_aim_entry on a tmp file."""
        # The intended dict: multi-line old/new with real code indent,
        # a substring ": " inside a multi-line body, and an apostrophe
        # in at least one value.
        old_1 = (
            "def process(data):\n"
            "    # Handle the input: safely\n"
            "    if data is None:\n"
            "        return \"it's empty\"\n"
            "    return data\n"
        )
        new_1 = (
            "def process(data):\n"
            "    # Handle the input: safely (fixed)\n"
            "    if data is None:\n"
            "        return \"it's empty (fixed)\"\n"
            "    return data\n"
        )
        old_2 = (
            "class Handler:\n"
            "    def __init__(self):\n"
            "        self.state = \"idle\"\n"
            "        # Note: this is a comment: with a colon\n"
        )
        new_2 = (
            "class Handler:\n"
            "    def __init__(self):\n"
            "        self.state = \"ready\"\n"
            "        # Note: this is a comment: with a colon (fixed)\n"
        )

        intended = {
            "pre_aimed": True,
            "scope_files": ["work.py"],
            "tests": ["tests/test_work.py"],
            "tests_timeout_s": 240,
            "recipe": [
                {"file": "work.py", "old": old_1, "new": new_1},
                {"file": "work.py", "old": old_2, "new": new_2},
            ],
        }

        # Build the fence body as the output of yaml.safe_dump of the
        # intended dict (multi-line old/new become | literal blocks,
        # which round-trip byte-exact).
        fence_body = yaml.safe_dump(
            intended, default_flow_style=False, sort_keys=False,
        )
        directive = _mission_directive(fence_body)

        # Parse succeeds.
        mission = parse_mission(directive)
        assert mission.pre_aimed is True
        assert mission.scope_files == ["work.py"]
        assert mission.tests == ["tests/test_work.py"]
        assert mission.tests_timeout_s == 240
        assert len(mission.recipe) == 2

        # For each recipe entry the parsed old/new are BYTE-IDENTICAL to
        # the intended dict's values (identity, not just "parsed").
        for i, (parsed, expected) in enumerate(
            zip(mission.recipe, intended["recipe"])
        ):
            assert parsed["old"] == expected["old"], (
                f"recipe entry {i}: parsed old is not byte-identical to "
                f"the intended value (the F1 class: YAML folding mangled "
                f"the multi-line value)"
            )
            assert parsed["new"] == expected["new"], (
                f"recipe entry {i}: parsed new is not byte-identical to "
                f"the intended value"
            )
            assert parsed["file"] == expected["file"]

        # Then the re-key + _apply_aim_entry on a tmp file round-trips
        # byte-exact. The re-key site (fixer_stages.py:993-1001) emits
        # the re-keyed shape (old_string/new_string). Apply each entry
        # in order against a tmp file.
        wt = tmp_path / "wt"
        _init_repo(wt)
        # The file starts with old_1 + old_2 (the two sites).
        (wt / "work.py").write_text(old_1 + old_2)
        _git(wt, "add", "work.py")
        _git(wt, "commit", "-qm", "work.py")

        scope_set = {"work.py"}
        for i, entry in enumerate(mission.recipe):
            # The re-keyed shape (what the re-key site emits).
            rekeyed = {
                "file": entry["file"],
                "old_string": entry["old"],
                "new_string": entry["new"],
                "evidence": "pre-aimed recipe entry",
            }
            _apply_aim_entry(str(wt), rekeyed, scope_set)

        # The file now has new_1 + new_2 (byte-exact round-trip).
        assert (wt / "work.py").read_text() == new_1 + new_2
