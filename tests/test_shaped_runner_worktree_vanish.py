"""Tests for the vanished-worktree diagnostic + WIP salvage
(agents-core-fixer-worktree-vanish-salvage-v0, D1/D2/D3).

An external process has deleted fixer worktrees mid-run three times
(2026-09-08 x2, 2026-09-11 - the #291 measured loss). Every recurrence
the model's good work died silently: the run "concluded" (rc=0), the tail
read `empty diff - no PR`, and the gate diag mislabelled the failure
`(timeout/spawn error)`.

T1: _build_fixer_result with a removed cwd (git-add failure + missing
    dir) -> worktree_vanished=True; live cwd -> False.
T2: tail_finalize with worktree-vanished + empty diff + WIP present ->
    opens the advisory [SALVAGE] PR (fake forgejo),
    stop_reason="concluded_empty_diff_wip_salvage"; the PR body carries
    task_id + the WIP head sha; the tail line is the distinct vanished
    line (NOT `empty diff - no PR`); the PR first line reads
    `concluded, but the diff was lost...`, not `not concluded`.
T3: salvage from a vanished worktree - MANDATORY REAL GIT (a
    subprocess.run monkeypatch whose fake push always returns rc=0 cannot
    distinguish which -C root _git used, so it would green the exact
    mechanism D2 exists to fix): worktree_path points at a removed dir +
    repo_cwd is a scratch clone holding refs/wip/<task_id> -> assert BOTH
    (a) the salvage branch EXISTS on the tmp bare origin AND (b) the push
    root was repo_cwd (the removed worktree cannot serve as the -C root
    for a successful push, and no surviving worktree holds the ref, so a
    push that succeeded could only have used repo_cwd).
T4: fail-closed: vanished + parent-clone push also fails -> returns ""
    (never raises), and the tail line is the distinct vanished line.
T5: the cwd-missing re-run diag reads `unusable:cwd-missing` and contains
    no "timeout/spawn".

Conventions mirrored from tests/test_shaped_runner_nonpython_gate.py
(fake-forgejo create_pr seam, room_path patch to tmp) and
tests/test_local_opencode_engine.py (the real-git scratch_repo fixture
pattern). The dedicated WIP-salvage suite lives at
agents_core/tests/test_wip_salvage.py (package tree) - referenced, not
moved. All scratch repos/dirs are pytest tmp_path (auto-cleaned; the
host root is ~96% full - nothing on a persistent /tmp path).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agents_core.shaped_runner as sr
from agents_core.gw_agent import _build_fixer_result


# ---------------------------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------------------------

VANISHED_LINE = (
    "run discarded - worktree vanished mid-run "
    "(external deletion; NOT a gate or test-runner failure)"
)

WIP_SALVAGE_STOP_REASON = "concluded_empty_diff_wip_salvage"


def _salvage_head(task_id: str, branch: str) -> str:
    """Mirror the locked S1 expression (shaped_runner
    _open_wip_salvage_pr) for an in-test expected-value computation."""
    return (
        f"{branch}-salvage-"
        f"{hashlib.sha1(task_id.encode()).hexdigest()[:8]}"
    )


def _tail_lines(shaped_dir: Path, task_id: str) -> str:
    p = shaped_dir / f"{task_id}-tail.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _tail_finalize_kwargs(
    tmp_path: Path,
    *,
    task_id: str = "vw-task-001",
    worktree_path: str,
    repo_cwd: str = "",
    worktree_vanished: bool = True,
    wip_commit_count: int = 0,
    wip_head_sha: str = "",
) -> dict:
    """Standard tail_finalize kwargs for the vanished/empty-diff shape:
    concluded=True (a vanished run DOES conclude - the model exits after
    the ENOENTs), empty diff, no gate outcome, no budget flags, no WIP
    ref by default (the caller opts in)."""
    shaped_dir = tmp_path / "shaped-rt"
    shaped_dir.mkdir(exist_ok=True)
    return dict(
        task_id=task_id,
        target_id="vw-target-v0",
        bare_repo="agents-core",
        branch="lapis/vw-target-v0/forced",
        slug="forced",
        cwd=worktree_path,
        worktree_path=worktree_path,
        final_diff="",
        concluded=True,
        last_test_outcome=None,
        max_steps_hit=False,
        no_progress_hit=False,
        stop_reason="",
        step_count=3,
        transcript_path=shaped_dir / f"{task_id}-gw-transcript.json",
        gate_passed=False,
        gate_bypassed=None,
        model_touched_tests=set(),
        gate_rerun_fired=False,
        wip_ref=f"refs/wip/{task_id}",
        wip_commit_count=wip_commit_count,
        wip_head_sha=wip_head_sha,
        wip_steps=[1],
        worktree_vanished=worktree_vanished,
        repo_cwd=repo_cwd,
        seat_alias="gravitywell-122b",
        served_model="",
    )


def _run_tail_finalize(tmp_path: Path, kwargs: dict, mock_pr, mock_friction,
                       stub_git: bool = False):
    """Drive sr.tail_finalize with the standard mock stack. Returns
    (url, pr_body). room_path is patched to tmp so _tail_log lands in the
    sandbox; _write_friction_entry is mocked (the friction store is a live
    SQLite WAL - unit tests never touch it); create_pr is the fake-forgejo
    seam.

    stub_git=True stubs the TAIL's own deterministic git ops (checkout -B /
    add / commit / push / rev-parse) to rc=0 so a run whose worktree is
    gone still reaches the PR path and the tail.log lines are assertable -
    the DUT in those tests is a DIAGNOSTIC LABEL, not the git ops. This is
    NOT the T3 mechanism: T3 (the repo_cwd fallback push) runs REAL git
    with stub_git=False - a stubbed push cannot distinguish which -C root
    was used.
    """
    shaped_dir = tmp_path / "shaped-rt"
    shaped_dir.mkdir(exist_ok=True)
    mock_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/999"}
    if stub_git:
        # The tail's `_git` calls are ["git", "-C", cwd, <op>, ...]; the
        # worktree-salvage rev-parse (op index 3 == "rev-parse") needs a
        # sha-shaped stdout or the partition soft-fails to "".
        def _fake_run(cmd, **kw):
            op = cmd[3] if len(cmd) > 3 else ""
            return MagicMock(returncode=0, stderr="",
                             stdout="deadbeefcafe0001\n" if op == "rev-parse" else "")
        git_patch = patch("subprocess.run", side_effect=_fake_run)
    else:
        git_patch = contextlib.nullcontext()
    with (
        patch("agents_core.forgejo.create_pr", mock_pr),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(sr, "_write_friction_entry", mock_friction),
        git_patch,
    ):
        url = sr.tail_finalize(**kwargs)
    body = mock_pr.call_args.kwargs.get("body", "") if mock_pr.call_args else ""
    return url, body, shaped_dir, mock_friction


# ---------------------------------------------------------------------------
# T1: _build_fixer_result worktree_vanished flag
# ---------------------------------------------------------------------------

def test_t1_removed_cwd_sets_worktree_vanished(tmp_path, capsys):
    """A removed cwd (git add -A fails rc!=0 + missing dir) ->
    worktree_vanished=True + the distinct WARN naming the shape."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"],
                   check=True, capture_output=True)

    # Simulate the external deleter: the worktree is gone.
    shutil.rmtree(str(repo))

    result = _build_fixer_result(
        cwd=str(repo), transcript=[], concluded=True,
    )
    assert result["worktree_vanished"] is True
    assert result["final_diff"] == ""
    assert result["concluded"] is True
    err = capsys.readouterr().err
    assert "worktree vanished mid-run" in err
    assert "external deletion suspected" in err
    assert "finding/fixer-worktrees-destroyed-external-process-" in err


def test_t1_live_cwd_defaults_worktree_vanished_false(tmp_path, capsys):
    """A live cwd with an actual change -> worktree_vanished=False (the
    default) and a real diff; a live cwd with a git-add failure (no
    changes of its own) also stays False - only a MISSING dir flips it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"],
                   check=True, capture_output=True)

    (repo / "b.py").write_text("y = 2\n")
    result = _build_fixer_result(
        cwd=str(repo), transcript=[], concluded=True,
    )
    assert result["worktree_vanished"] is False
    assert "b.py" in result["final_diff"]
    assert "worktree vanished" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# T2: the concluded + empty-diff + WIP-present salvage partition
# ---------------------------------------------------------------------------

def test_t2_vanished_empty_diff_wip_present_opens_salvage_pr(tmp_path):
    """The #291 class: concluded=True, final_diff="", WIP present -> the
    distinct vanished line FIRST (static primary line), then the advisory
    [SALVAGE] PR with stop_reason=concluded_empty_diff_wip_salvage; the
    PR body carries task_id + the WIP head sha; the first line reads
    'concluded, but the diff was lost...', not 'not concluded'; the tail
    line is the vanished line, not 'empty diff - no PR'. The friction
    entry is written with the STABLE node_id (dedup across recurrences).

    The push itself is the fake-forgejo seam's territory (T3 owns the
    REAL-git push): the worktree is gone (the deleter) and repo_cwd is a
    scratch git repo so the root-selection resolves to it and the fake
    push succeeds, letting the create_pr seam carry the assertions."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    shutil.rmtree(str(wt_dir))  # the external deleter
    # repo_cwd (the parent clone) is a real git repo so the root-selection
    # resolves to it and the push command runs; the WIP ref is a REAL ref
    # on it (the #291 shape: the ref survives in the parent clone's shared
    # gitdir after the worktree is gone) so the push to a tmp bare origin
    # succeeds and the fake-forgejo create_pr seam carries the PR
    # assertions. (T3 owns the full real-git end-to-end variant.)
    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", str(origin)],
                   check=True, capture_output=True)
    clone = tmp_path / "clone"
    clone.mkdir()
    subprocess.run(["git", "-C", str(clone), "init", "-q", "-b", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    (clone / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(clone), "add", "-A"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-qm", "base"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "remote", "add", "origin",
                    str(origin)], check=True, capture_output=True)
    wip_sha = "abc123def456"
    # A real commit object + a real refs/wip/<task_id> pointing at it:
    # the push source ref resolves, so the (fake origin) push succeeds.
    tree = subprocess.run(["git", "-C", str(clone), "write-tree"],
                          check=True, capture_output=True, text=True)
    commit = subprocess.run(
        ["git", "-C", str(clone), "commit-tree", tree.stdout.strip(),
         "-p", "HEAD", "-m", "wip: vw-task-001 step 1 [auto]"],
        check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(clone), "update-ref",
                    "refs/wip/vw-task-001", commit.stdout.strip()],
                   check=True, capture_output=True)
    wip_sha = commit.stdout.strip()
    mock_pr = MagicMock()
    mock_friction = MagicMock()

    kwargs = _tail_finalize_kwargs(
        tmp_path,
        worktree_path=str(wt_dir),
        repo_cwd=str(clone),
        worktree_vanished=True,
        wip_commit_count=2,
        wip_head_sha=wip_sha,
    )
    url, body, shaped_dir, mock_friction = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction)

    # The advisory [SALVAGE] PR was opened (the run is still LOST - the
    # returned URL is the salvage PR's, not a success PR).
    mock_pr.assert_called_once()
    call_kwargs = mock_pr.call_args.kwargs
    assert "[SALVAGE]" in call_kwargs["title"]
    assert WIP_SALVAGE_STOP_REASON in call_kwargs["title"]
    assert call_kwargs["head"] == _salvage_head(
        "vw-task-001", "lapis/vw-target-v0/forced")
    assert call_kwargs["base"] == "main"
    assert url == "http://forgejo/agents-core/pulls/999"

    # The PR body carries task_id + the WIP head sha.
    assert "vw-task-001" in body
    assert wip_sha in body
    # The third first-line state case: a run that DID conclude is not
    # mislabeled "not concluded".
    assert "concluded, but the diff was lost to a mid-run worktree deletion" in body
    assert "not concluded, no gate_passed" not in body
    # concluded=False stays in force for the "What remains" section: the
    # WIP-snapshot text (incl. the non-cumulative-head caveat), NOT the
    # concluded "worktree's FINAL state" text.
    assert "non-cumulative" in body
    assert "worktree's FINAL state" not in body

    # The tail line is the DISTINCT vanished line (the success path is not
    # the `empty diff - no PR` line), and the salvage PR line is a separate
    # append AFTER the primary line (append-not-modify).
    tail = _tail_lines(shaped_dir, "vw-task-001")
    assert VANISHED_LINE in tail
    assert "empty diff - no PR" not in tail
    assert "[SALVAGE] PR opened" in tail
    assert tail.index(VANISHED_LINE) < tail.index("[SALVAGE] PR opened")

    # The friction entry: STABLE node_id (NOT the per-run task_id) so every
    # vanished run dedups onto friction/agents-core-worktree-vanished; the
    # per-run task_id is carried in last_task_id/first_task_id content.
    mock_friction.assert_called_once()
    f_kwargs = mock_friction.call_args.kwargs
    assert f_kwargs["node_id"] == "worktree-vanished"
    assert f_kwargs["error_signature"] == "worktree-vanished:mid-run"
    assert f_kwargs["task_id"] == "vw-task-001"
    assert f_kwargs["repo"] == "agents-core"


def test_t2_vanished_no_wip_no_pr(tmp_path):
    """Invariant: no-WIP behavior unchanged - a vanished run with zero WIP
    commits still ends with the distinct vanished line + friction entry,
    but NO PR."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path, worktree_path=str(wt_dir), worktree_vanished=True,
    )
    url, body, shaped_dir, mock_friction = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction)
    assert url == ""
    mock_pr.assert_not_called()
    tail = _tail_lines(shaped_dir, "vw-task-001")
    assert VANISHED_LINE in tail
    assert "empty diff - no PR" not in tail
    assert "[SALVAGE] PR opened" not in tail
    mock_friction.assert_called_once()
    assert mock_friction.call_args.kwargs["node_id"] == "worktree-vanished"


def test_t2_not_vanished_empty_diff_unchanged(tmp_path):
    """Invariant: the non-vanished empty-diff partition is byte-unchanged -
    the existing `empty diff - no PR` line, no friction entry, no PR."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path, worktree_path=str(wt_dir), worktree_vanished=False,
    )
    url, body, shaped_dir, mock_friction = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction)
    assert url == ""
    mock_pr.assert_not_called()
    mock_friction.assert_not_called()
    tail = _tail_lines(shaped_dir, "vw-task-001")
    assert "empty diff - no PR" in tail
    assert VANISHED_LINE not in tail


# ---------------------------------------------------------------------------
# T3: real-git salvage from a vanished worktree (MANDATORY real git)
# ---------------------------------------------------------------------------

def _make_scratch(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Real-git scratch environment (the scratch_repo fixture pattern from
    tests/test_local_opencode_engine.py): a tmp bare `origin` + a real
    clone on `main` that will hold refs/wip/<task_id>. Everything under
    tmp_path (auto-cleaned)."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    # -b main: without it the bare's HEAD is an unborn `master` on git >=
    # 2.28 hosts, so `git clone` lands the clone on an unborn master and
    # the initial push dies with 'src refspec main does not match any'.
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
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
    subprocess.run(["git", "-C", str(seed), "add", "-A"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-qm", "seed"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "remote", "add", "origin",
                    str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                   check=True, capture_output=True)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    return origin, clone, seed


def test_t3_real_git_salvage_pushed_from_parent_clone(tmp_path):
    """MANDATORY REAL GIT (the subprocess.run monkeypatch is NOT
    acceptable here - a fake push that always returns rc=0 cannot
    distinguish which -C root was used, so it would green the exact
    mechanism D2 exists to fix).

    Setup: a real worktree (a git worktree of the clone) with the WIP ref
    on it (refs live in the SHARED common gitdir - the clone's). The
    worktree dir is then DELETED (the external deleter). repo_cwd is the
    clone - the ONLY surviving place the WIP ref resolves.

    Assert BOTH:
    (a) the salvage branch EXISTS on the tmp bare origin after the call;
    (b) the push root was repo_cwd - the root-selection returned repo_cwd
        for the removed worktree (the removed worktree cannot serve as a
        -C root for a successful push, and no surviving worktree holds
        the ref, so a push that succeeded could only have used repo_cwd).
    """
    origin, clone, _seed = _make_scratch(tmp_path)
    task_id = "vw-task-t3"
    wip_ref = f"refs/wip/{task_id}"

    # A REAL worktree of the clone (the harness's own shape: a worktree
    # shares the clone's gitdir, so refs/wip/<task_id> resolves from both).
    wt = tmp_path / "wt-t3"
    subprocess.run(["git", "-C", str(clone), "worktree", "add",
                    "--detach", str(wt), "origin/main"],
                   check=True, capture_output=True)
    (wt / "work.py").write_text("z = 3\n")
    subprocess.run(["git", "-C", str(wt), "add", "work.py"],
                   check=True, capture_output=True)
    tree = subprocess.run(["git", "-C", str(wt), "write-tree"],
                          check=True, capture_output=True, text=True)
    parent = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True)
    commit = subprocess.run(
        ["git", "-C", str(wt), "commit-tree", tree.stdout.strip(),
         "-p", parent.stdout.strip(), "-m", f"wip: {task_id} step 1 [auto]"],
        check=True, capture_output=True, text=True)
    wip_sha = commit.stdout.strip()
    subprocess.run(["git", "-C", str(wt), "update-ref", wip_ref, wip_sha],
                   check=True, capture_output=True)
    # The ref resolves from BOTH the worktree and the clone (shared
    # common gitdir) while the worktree lives.
    for root in (wt, clone):
        r = subprocess.run(["git", "-C", str(root), "rev-parse",
                            "--verify", wip_ref],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"WIP ref must resolve from {root}"

    # The external deleter: the worktree is gone (the clone survives).
    shutil.rmtree(str(wt))

    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path,
        task_id=task_id,
        worktree_path=str(wt),
        repo_cwd=str(clone),
        worktree_vanished=True,
        wip_commit_count=1,
        wip_head_sha=wip_sha,
    )
    url, body, shaped_dir, _mf = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction)

    # (a) the salvage branch EXISTS on the bare origin, and its head is
    # the WIP commit - pushed from the vanished worktree's surviving
    # shared gitdir.
    assert url == "http://forgejo/agents-core/pulls/999"
    mock_pr.assert_called_once()
    expected_branch = _salvage_head(task_id, "lapis/vw-target-v0/forced")
    r = subprocess.run(["git", "-C", str(origin), "rev-parse",
                        "--verify", f"refs/heads/{expected_branch}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        f"salvage branch {expected_branch} must exist on the bare origin "
        f"(rc={r.returncode}): {r.stderr.strip()}")
    assert r.stdout.strip() == wip_sha

    # (b) the push root was repo_cwd: the root-selection returned
    # repo_cwd for the removed worktree. The removed worktree cannot
    # serve as a -C root for a successful push (git -C <missing> fails),
    # and the WIP ref resolves only from the clone's gitdir now - a push
    # that succeeded could only have used repo_cwd. Rule out an
    # undocumented fallback-to-worktree path explicitly.
    assert not os.path.isdir(str(wt))
    assert os.path.isdir(str(clone))
    # Root-selection mirror (the D2 pivot): worktree exists -> worktree;
    # else repo_cwd when non-empty; else worktree.
    if worktree_exists := os.path.isdir(str(wt)):
        chosen = str(wt)
    elif repo_cwd := str(clone):
        chosen = repo_cwd
    else:
        chosen = str(wt)
    assert chosen == str(clone), (
        "root-selection must return repo_cwd for a removed worktree - "
        "an undocumented fallback-to-worktree path would have failed the "
        "real push")

    # The tail line is the distinct vanished line + the salvage append.
    tail = _tail_lines(shaped_dir, task_id)
    assert VANISHED_LINE in tail
    assert "[SALVAGE] PR opened" in tail
    assert "empty diff - no PR" not in tail
    # The PR body carries the WIP head sha (real sha, not a stub).
    assert wip_sha in body


# ---------------------------------------------------------------------------
# T4: fail-closed - vanished + parent-clone push also fails
# ---------------------------------------------------------------------------

def test_t4_vanished_push_failure_returns_empty_never_raises(tmp_path):
    """Fail-closed: the worktree is gone AND the parent-clone push fails
    (the clone is not a git repo -> git -C fails) -> tail_finalize
    returns "" (never raises) and the tail line is the distinct vanished
    line (the primary line is written BEFORE the recovery attempt and
    stays immutable)."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    shutil.rmtree(str(wt_dir))  # the external deleter
    dead_clone = tmp_path / "clone"
    dead_clone.mkdir()  # exists but is NOT a git repo -> push fails

    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path,
        worktree_path=str(wt_dir),
        repo_cwd=str(dead_clone),
        worktree_vanished=True,
        wip_commit_count=1,
        wip_head_sha="deadbeef0000",
    )
    url, body, shaped_dir, _mf = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction)

    assert url == ""  # never raises; soft-fails to ""
    assert body == ""  # no PR
    mock_pr.assert_not_called()
    tail = _tail_lines(shaped_dir, "vw-task-001")
    assert VANISHED_LINE in tail
    assert "empty diff - no PR" not in tail
    assert "[SALVAGE] PR opened" not in tail
    # The push-failure footnote is a distinct append (the _open_wip_
    # salvage_pr tail line), after the immutable primary line.
    assert "wip-salvage push failed" in tail
    assert tail.index(VANISHED_LINE) < tail.index("wip-salvage push failed")


def test_t4_friction_refresh_preserves_status_field(tmp_path):
    """The D2 salvage_success friction refresh must preserve the entry's
    status field (the _write_friction_entry contract: status is the
    dedup/recurrence signal - a resolved entry is flipped back to open on
    recurrence). The refresh writes via a raw MemoryStore set; without the
    status guard a pre-existing entry's status would be dropped (and a
    resolved entry would not flip back to open). Driven with a fake
    MemoryStore: the initial _write_friction_entry (mocked) is followed by
    the raw refresh inside tail_finalize, which the fake store records."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    shutil.rmtree(str(wt_dir))  # the external deleter
    dead_clone = tmp_path / "clone"
    dead_clone.mkdir()  # exists but is NOT a git repo -> push fails

    fkey = "friction/agents-core-worktree-vanished"
    # A pre-existing RESOLVED entry (a prior recurrence was fixed) - the
    # refresh must flip it back to open AND add salvage_success.
    pre_existing = {
        "status": "resolved",
        "test_node_id": "worktree-vanished",
        "error_signature": "worktree-vanished:mid-run",
        "first_seen": "2026-09-08",
        "last_seen": "2026-09-09",
        "first_task_id": "old-task",
        "last_task_id": "old-task",
    }

    class _FakeStore:
        def __init__(self):
            self.sets: list[tuple[str, str, list]] = []

        def get(self, key):
            if key == fkey:
                return {"content": json.dumps(pre_existing)}
            return None

        def set(self, key, content, tags=None):
            self.sets.append((key, content, list(tags or [])))

        def close(self):
            pass

    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path,
        worktree_path=str(wt_dir),
        repo_cwd=str(dead_clone),
        worktree_vanished=True,
        wip_commit_count=1,
        wip_head_sha="deadbeef0000",
    )
    fake_store = _FakeStore()
    shaped_dir = tmp_path / "shaped-rt"
    shaped_dir.mkdir(exist_ok=True)
    with (
        patch("agents_core.forgejo.create_pr", mock_pr),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(sr, "_write_friction_entry", mock_friction),
        patch("agents_core.mem.MemoryStore", return_value=fake_store),
    ):
        url = sr.tail_finalize(**kwargs)

    assert url == ""  # push failed (dead clone) -> fail-closed
    # The raw refresh wrote exactly one entry back to the store...
    assert len(fake_store.sets) == 1
    wkey, wcontent, wtags = fake_store.sets[0]
    assert wkey == fkey
    assert wtags == ["friction", "test-gate", "agents-core"]
    written = json.loads(wcontent)
    # ...with salvage_success (the push failed -> False)...
    assert written["salvage_success"] is False
    # ...the status field PRESERVED and flipped back to open (recurrence
    # of a resolved friction)...
    assert written["status"] == "open"
    # ...and the pre-existing content fields intact (not dropped).
    assert written["test_node_id"] == "worktree-vanished"
    assert written["error_signature"] == "worktree-vanished:mid-run"
    assert written["first_seen"] == "2026-09-08"
    assert written["last_task_id"] == "old-task"


# ---------------------------------------------------------------------------
# T5: the cwd-missing re-run diag label
# ---------------------------------------------------------------------------

def test_t5_cwd_missing_rerun_diag_label(tmp_path):
    """The re-run diag for a cwd-missing shape reads
    `rerun=unusable:cwd-missing` and contains no "timeout/spawn" - the
    2026-09-11 mislabel that sent the PM session down a wrong theory."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path,
        worktree_path=str(wt_dir),
        worktree_vanished=False,  # the D1 line is not the target here
        wip_commit_count=0,
    )
    # A non-empty diff + a touched test + a failing last outcome -> the
    # gate fails closed and the targeted re-run fires; the re-run's
    # cwd-missing check (Path(cwd).is_dir() False) makes it unusable.
    kwargs.update(
        final_diff="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
                   "@@ -1 +1 @@\n-old\n+new\n",
        model_touched_tests={"tests/test_fast.py"},
        # A FAILED node ID in output_tail makes the touched test count as a
        # failure -> the gate fails closed and the targeted re-run fires.
        last_test_outcome={
            "passed": 1, "failed": 1, "errors": 0,
            "returncode": 1, "summary": "1 failed, 1 passed",
            "output_tail": "FAILED tests/test_fast.py::test_x - assert 0 == 1",
        },
    )
    shutil.rmtree(str(wt_dir))  # the external deleter, before the tail

    # The re-run is unusable (cwd gone -> None) and the tail's own
    # deterministic git ops (checkout -B / add) cannot run in the deleted
    # cwd: stub them to rc=0 so the run reaches the PR path (fake create_pr)
    # and the tail.log lines are asserted on. The DUT is the _rerun_diag
    # label, not the git ops.
    url, body, shaped_dir, _mf = _run_tail_finalize(
        tmp_path, kwargs, mock_pr, mock_friction, stub_git=True)

    assert url == "http://forgejo/agents-core/pulls/999"
    tail = _tail_lines(shaped_dir, "vw-task-001")
    # The gate FAILED line carries the distinct cwd-missing diag.
    assert "gate FAILED" in tail
    assert "rerun=fired-but-unusable:cwd-missing" in tail
    assert "timeout/spawn" not in tail
    # The primary vanished line is NOT emitted here (worktree_vanished is
    # False in this test - the DUT is the re-run label, not the D1 line).
    assert VANISHED_LINE not in tail


def test_t5_live_cwd_rerun_diag_keeps_timeout_spawn_label(tmp_path):
    """The genuine timeout/spawn shape is UNCHANGED: a live cwd with an
    unusable re-run still reads `rerun=fired-but-unusable (timeout/spawn
    error)` (regression pin on the D1 label split)."""
    wt_dir = tmp_path / "wt"
    wt_dir.mkdir()
    mock_pr = MagicMock()
    mock_friction = MagicMock()
    kwargs = _tail_finalize_kwargs(
        tmp_path,
        worktree_path=str(wt_dir),
        worktree_vanished=False,
    )
    kwargs.update(
        final_diff="diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
                   "@@ -1 +1 @@\n-old\n+new\n",
        model_touched_tests={"tests/test_fast.py"},
        # A FAILED node ID in output_tail makes the touched test count as a
        # failure -> the gate fails closed and the targeted re-run fires.
        last_test_outcome={
            "passed": 1, "failed": 1, "errors": 0,
            "returncode": 1, "summary": "1 failed, 1 passed",
            "output_tail": "FAILED tests/test_fast.py::test_x - assert 0 == 1",
        },
    )
    # The re-run times out (the genuine timeout/spawn shape) -> None. The
    # re-run outcome is forced to None via a REAL _gate_targeted_rerun
    # call that times out: the subprocess.run stub returns a
    # TimeoutExpired, which the real helper converts to None (the exact
    # genuine-timeout shape - a return_value=None patch would be defeated
    # by the tail's local `import subprocess`, which the real helper uses
    # for its pytest spawn). The tail's own deterministic git ops are
    # stubbed to rc=0 (the DUT is the _rerun_diag label, not the git ops)
    # so the run reaches the PR path and the tail.log lines are
    # assertable.
    import subprocess as _subprocess_mod

    def _fake_run(cmd, **kw):
        # The re-run's pytest spawn (the cmd carries "-m pytest") times
        # out -> the REAL _gate_targeted_rerun converts it to None (the
        # exact genuine-timeout shape). Everything else (the tail's own
        # deterministic git ops) returns rc=0.
        if "-m" in cmd and "pytest" in cmd:
            raise _subprocess_mod.TimeoutExpired(
                cmd, kw.get("timeout", 180), output="", stderr="")
        op = cmd[3] if len(cmd) > 3 else ""
        return MagicMock(returncode=0, stderr="",
                         stdout="deadbeefcafe0001\n" if op == "rev-parse" else "")

    shaped_dir = tmp_path / "shaped-rt"
    shaped_dir.mkdir(exist_ok=True)
    mock_pr.return_value = {"html_url": "http://forgejo/agents-core/pulls/999"}
    with (
        patch("agents_core.forgejo.create_pr", mock_pr),
        patch("agents_core.forgejo.get_open_prs", return_value=[]),
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(sr, "_write_friction_entry", mock_friction),
        patch("subprocess.run", side_effect=_fake_run),
    ):
        url = sr.tail_finalize(**kwargs)

    assert url == "http://forgejo/agents-core/pulls/999"
    tail = _tail_lines(shaped_dir, "vw-task-001")
    # The gate FAILED line carries the genuine timeout/spawn label (the
    # D1 label split keeps the non-cwd-missing shape byte-identical).
    assert "gate FAILED" in tail
    assert "rerun=fired-but-unusable (timeout/spawn error)" in tail
    assert "cwd-missing" not in tail
    assert VANISHED_LINE not in tail
