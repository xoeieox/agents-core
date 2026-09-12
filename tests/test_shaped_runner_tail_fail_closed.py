"""Fail-closed tail for unclassified terminal deaths (agents-core-
shaperunner-fail-closed-v0).

The tail_finalize catch-all partition (the else branch of the
`salvaged = False / if not concluded:` partition) used to label EVERY
remaining unclassified terminal death "DoormanUnreachable or wake
timeout" and return "" - which main() printed as an empty stdout line
and exited 0 with. A ghost death (model-down / context death / seat
flip) then showed as a SUCCESSFUL dispatch in the claude-queue-runner
log (rc is the only signal the runner logs), the death was mislabeled
as a doorman problem, and the daemon's lost-dispatch detection never
fired from the record.

This file pins the fix:
  (a) unclassified death with a named stop_reason -> the tail returns
      the TAIL_UNCLASSIFIED_DEATH sentinel, the label carries the
      ACTUAL stop_reason, and the false "DoormanUnreachable or wake
      timeout" string is absent from the tail source.
  (b) unclassified death with no stop_reason in the record -> the
      label says exactly "no stop_reason recorded" (no invented cause).
  (c) regressions: the no_progress_hit and max_steps_hit named paths
      are UNCHANGED (same WARN string, same return ""), and the
      WIP-salvage partition is UNCHANGED (a non-budget death with >=1
      WIP commit still opens the advisory [SALVAGE] PR).
  (d) the rc=3 mapping: main() has a testable seam - the local-fixer
      engine block calls _engine_dispatch_exit(pr_url, engine) - so the
      mapping is tested by driving the REAL main() with a mocked
      _run_local_fixer that returns the sentinel (no subprocess
      needed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core import shaped_runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )


def _make_wt_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    """A worktree whose origin is a bare repo (mirrors the fixture
    pattern in agents_core/tests/test_fixer_staged_fire_push_seam.py)."""
    origin = tmp_path / "origin.git"
    _git(Path("/"), "init", "-q", "--bare", str(origin))
    wt = tmp_path / "wt"
    wt.mkdir()
    _git(wt, "init", "-q", "-b", "main")
    _git(wt, "config", "user.email", "test@example.com")
    _git(wt, "config", "user.name", "test")
    _git(wt, "commit", "-q", "--allow-empty", "-m", "base")
    _git(wt, "remote", "add", "origin", str(origin))
    _git(wt, "push", "-q", "origin", "main")
    return wt, origin


def _make_tail_kwargs(wt_dir: Path, **overrides) -> dict:
    """Build the tail_finalize kwargs for an UNCLASSIFIED terminal
    death: not concluded, no WIP commits, no guard flags, empty diff.
    That is exactly the shape that reaches the else catch-all."""
    kwargs = {
        "task_id": "task-fc",
        "target_id": "tgt-fc",
        "bare_repo": "agents-core",
        "branch": "lapis/tgt-fc/local",
        "slug": "local",
        "cwd": str(wt_dir),
        "worktree_path": str(wt_dir),
        "final_diff": "",
        "concluded": False,
        "last_test_outcome": None,
        "max_steps_hit": False,
        "no_progress_hit": False,
        "stop_reason": "",
        "step_count": 12,
        "transcript_path": wt_dir / "transcript.json",
        "gate_passed": False,
        "gate_bypassed": None,
        "model_touched_tests": set(),
        "gate_rerun_fired": False,
        "wip_ref": "",
        "wip_commit_count": 0,
        "wip_head_sha": "",
        "wip_steps": [],
        "_wip_git": None,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# (a) unclassified death with a named stop_reason
# ---------------------------------------------------------------------------


def test_unclassified_death_with_stop_reason_returns_sentinel(tmp_path, capsys):
    """(a) A model-down death (synthetic record: not concluded, no WIP,
    no guard flags, stop_reason named) -> the tail returns the
    TAIL_UNCLASSIFIED_DEATH sentinel and the label carries the ACTUAL
    stop_reason."""
    wt_dir, _ = _make_wt_with_origin(tmp_path)
    kwargs = _make_tail_kwargs(wt_dir, stop_reason="model_down")

    out = shaped_runner.tail_finalize(**kwargs)

    assert out == shaped_runner.TAIL_UNCLASSIFIED_DEATH
    assert out != ""  # the sentinel must NOT be the empty-string success shape
    err = capsys.readouterr().err
    assert "unclassified terminal death" in err
    assert "stop_reason=model_down" in err
    # The false doorman label must not leak out of the tail.
    assert "DoormanUnreachable" not in err
    assert "wake timeout" not in err


# ---------------------------------------------------------------------------
# (b) unclassified death with no stop_reason in the record
# ---------------------------------------------------------------------------


def test_unclassified_death_without_stop_reason_says_none_recorded(tmp_path, capsys):
    """(b) No stop_reason in the record -> the label says exactly
    "no stop_reason recorded" (no invented cause)."""
    wt_dir, _ = _make_wt_with_origin(tmp_path)
    kwargs = _make_tail_kwargs(wt_dir, stop_reason="")

    out = shaped_runner.tail_finalize(**kwargs)

    assert out == shaped_runner.TAIL_UNCLASSIFIED_DEATH
    err = capsys.readouterr().err
    assert "unclassified terminal death" in err
    assert "no stop_reason recorded" in err
    assert "DoormanUnreachable" not in err
    assert "wake timeout" not in err


# ---------------------------------------------------------------------------
# (a, source-level) the false label is gone from the tail
# ---------------------------------------------------------------------------


def test_doorman_label_absent_from_tail_source():
    """The string "DoormanUnreachable or wake timeout" must not appear
    anywhere in the shaped_runner source after this change (a real
    DoormanUnreachable is caught and soft-failed at the lease-acquire
    sites, never at the tail)."""
    src = shaped_runner.__file__
    text = Path(src).read_text()
    assert "DoormanUnreachable or wake timeout" not in text


# ---------------------------------------------------------------------------
# (c) regressions: the named paths and the WIP-salvage partition are
#     UNCHANGED
# ---------------------------------------------------------------------------


def test_no_progress_path_unchanged(tmp_path, capsys):
    """(c) no_progress_hit (no salvageable green diff) -> the SAME WARN
    string and the SAME return "" (rc=0) as before this change."""
    wt_dir, _ = _make_wt_with_origin(tmp_path)
    kwargs = _make_tail_kwargs(wt_dir, no_progress_hit=True)

    out = shaped_runner.tail_finalize(**kwargs)

    assert out == ""  # unchanged: the named path still exits 0
    err = capsys.readouterr().err
    assert (
        "WARN: local-fixer: run aborted - no semantic progress after "
        "consecutive idle steps (spinning wheels)" in err
    )


def test_max_steps_path_unchanged(tmp_path, capsys):
    """(c) max_steps_hit (no salvageable green diff) -> the SAME WARN
    string and the SAME return "" (rc=0) as before this change."""
    wt_dir, _ = _make_wt_with_origin(tmp_path)
    kwargs = _make_tail_kwargs(wt_dir, max_steps_hit=True)

    out = shaped_runner.tail_finalize(**kwargs)

    assert out == ""  # unchanged: the named path still exits 0
    err = capsys.readouterr().err
    assert (
        "WARN: local-fixer: run not concluded - max_steps ceiling reached "
        "(no passing tests or empty diff)" in err
    )


def test_wip_salvage_partition_unchanged(tmp_path, capsys):
    """(c) The WIP-salvage partition is UNCHANGED: a non-budget death
    (not concluded, no guard flags) with >=1 WIP commit still opens the
    advisory [SALVAGE] PR - it never reaches the catch-all."""
    wt_dir, _ = _make_wt_with_origin(tmp_path)
    (wt_dir / "work.py").write_text("x = 1\n")
    _git(wt_dir, "add", "work.py")
    _git(wt_dir, "commit", "-qm", "wip: task-fc step 1 [auto]")
    _git(wt_dir, "update-ref", "refs/wip/task-fc", "HEAD")

    kwargs = _make_tail_kwargs(
        wt_dir,
        stop_reason="seat_loss",
        wip_ref="refs/wip/task-fc",
        wip_commit_count=1,
        wip_head_sha=_git(wt_dir, "rev-parse", "HEAD").stdout.strip(),
    )

    with patch("agents_core.forgejo.create_pr") as mock_create_pr:
        mock_create_pr.return_value = {
            "html_url": "http://forgejo/agents-core/pulls/777",
        }
        out = shaped_runner.tail_finalize(**kwargs)

    assert out == "http://forgejo/agents-core/pulls/777"
    mock_create_pr.assert_called_once()
    # The salvage PR title carries the [SALVAGE] marker.
    title = mock_create_pr.call_args.kwargs.get("title") or ""
    assert "[SALVAGE]" in title
    # The catch-all sentinel must NOT be in play here.
    err = capsys.readouterr().err
    assert "unclassified terminal death" not in err


# ---------------------------------------------------------------------------
# (d) the rc=3 mapping via the main() seam
# ---------------------------------------------------------------------------


def _make_spec_file(shaped_dir: Path, engine: str) -> Path:
    """A minimal dispatch spec for the given engine."""
    spec_path = shaped_dir / f"tgt-fc-{engine}.json"
    spec_path.write_text(
        json.dumps(
            {
                "engine": engine,
                "model": "haiku",
                "system": "",
                "prompt": "do something",
                "timeout_s": 30,
                "task_id": "task-fc",
                "cwd": str(shaped_dir),
            }
        )
    )
    return spec_path


def test_main_maps_sentinel_to_exit_3_local_fixer(tmp_path, capsys):
    """(d) The main() engine-dispatch seam: _run_local_fixer returns the
    sentinel -> main() exits 3 with a loud ERROR line on stderr (and
    does NOT print the sentinel as a stdout PR URL)."""
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    spec_path = _make_spec_file(shaped_dir, "local-fixer")

    with (
        patch.object(sys, "argv", ["shaped_runner", str(spec_path)]),
        patch.object(
            shaped_runner, "_run_local_fixer",
            return_value=(
                shaped_runner.TAIL_UNCLASSIFIED_DEATH,
                {"seat": "haiku", "served": ""},
            ),
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            shaped_runner.main()

    assert exc_info.value.code == 3
    captured = capsys.readouterr()
    assert "ERROR" in captured.err
    assert "unclassified terminal death" in captured.err
    # The sentinel must never be printed as a stdout PR URL.
    assert shaped_runner.TAIL_UNCLASSIFIED_DEATH not in captured.out


def test_main_maps_sentinel_to_exit_3_local_fixer_staged(tmp_path, capsys):
    """(d) Same mapping on the local-fixer-staged engine block."""
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    spec_path = _make_spec_file(shaped_dir, "local-fixer-staged")

    with (
        patch.object(sys, "argv", ["shaped_runner", str(spec_path)]),
        patch.object(
            shaped_runner, "_run_local_fixer_staged",
            return_value=shaped_runner.TAIL_UNCLASSIFIED_DEATH,
        ),
    ):
        with pytest.raises(SystemExit) as exc_info:
            shaped_runner.main()

    assert exc_info.value.code == 3
    assert "unclassified terminal death" in capsys.readouterr().err


def test_main_normal_url_still_exits_0(tmp_path, capsys):
    """(d) Regression: a normal PR URL (and the named paths' "") still
    print and exit 0 - the mapping fires ONLY on the sentinel. The
    local-fixer block returns after _engine_dispatch_exit, so a clean
    return here IS the rc=0 path (the claude engine's sys.exit(1) is
    unreachable on this block)."""
    shaped_dir = tmp_path / "shaped"
    shaped_dir.mkdir()
    spec_path = _make_spec_file(shaped_dir, "local-fixer")

    with (
        patch.object(sys, "argv", ["shaped_runner", str(spec_path)]),
        patch.object(
            shaped_runner, "_run_local_fixer",
            # main()'s local-fixer block unpacks the (pr_url, provenance)
            # tuple (L1.D3 contract) - the mock must return the tuple shape.
            return_value=(
                "http://forgejo/agents-core/pulls/123",
                {"seat": "haiku", "served": ""},
            ),
        ),
    ):
        # main() must NOT sys.exit(3) here - the local-fixer block
        # returns after _engine_dispatch_exit (rc=0). The claude
        # engine's sys.exit(1) tail is unreachable on this block, so a
        # clean return is the honest rc=0 assertion.
        shaped_runner.main()

    captured = capsys.readouterr()
    assert "http://forgejo/agents-core/pulls/123" in captured.out


def test_engine_dispatch_exit_named_empty_string_is_rc0(capsys):
    """(d) The named failure paths (no_progress / max_steps / empty
    diff) return "" - the mapping must NOT fire on "" (those stay
    rc=0, loud in the log; the guard-scaling item's territory)."""
    shaped_runner._engine_dispatch_exit("", "local-fixer")  # no SystemExit
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # the empty line, exactly as before
    assert "ERROR" not in captured.err
