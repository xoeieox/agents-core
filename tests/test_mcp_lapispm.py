"""mcp_lapispm tests: each MCP tool must construct the correct `lapis-pm`
argv, and the parse-contract must hold:
  (1) non-zero exit -> parse_status == "cli_error", no parsed fields
  (2) clean status/list stdout -> parse_status == "ok" with expected fields
  (3) an unexpected stdout shape -> parse_status == "partial", ambiguous
      fields omitted (never filled with a guess)
`raw_output` must be present in all three cases.
"""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents_core import mcp_lapispm


def _proc(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["lapis-pm"], returncode=returncode, stdout=stdout, stderr=stderr)


STATUS_OK_STDOUT = """=== review-gate ===
  counter:       0 / 10
  paused:        False

=== my-target ===
  title:         Some Title
  pm_bound:      True
  pm_repo:       agents-core
  pm_authority:  advisory
  pm_verification: machine
  paused:        False
  cursor:        3
  dispatched:    2 total, 0 pending
  outstanding:   (none)
"""


def test_lapispm_status_argv_no_target():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(STATUS_OK_STDOUT)) as run:
        mcp_lapispm.lapispm_status()
    run.assert_called_once_with(["status"])


def test_lapispm_status_argv_with_target_and_explain():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(STATUS_OK_STDOUT)) as run:
        mcp_lapispm.lapispm_status(target_id="my-target", explain=True)
    run.assert_called_once_with(["status", "my-target", "--explain"])


def test_lapispm_status_cli_error_on_nonzero_exit():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout="", stderr="boom", returncode=2)):
        result = mcp_lapispm.lapispm_status(target_id="nope")
    assert result["parse_status"] == "cli_error"
    assert result["exit_code"] == 2
    assert result["raw_output"] == ""
    assert result["stderr"] == "boom"
    assert "title" not in result
    assert "pm_bound" not in result


def test_lapispm_status_ok_parses_expected_fields():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(STATUS_OK_STDOUT)):
        result = mcp_lapispm.lapispm_status(target_id="my-target")
    assert result["parse_status"] == "ok"
    assert result["raw_output"] == STATUS_OK_STDOUT
    assert result["exit_code"] == 0
    assert result["title"] == "Some Title"
    assert result["pm_bound"] is True
    assert result["pm_repo"] == "agents-core"
    assert result["pm_authority"] == "advisory"
    assert result["cursor"] == "3"
    assert result["outstanding"] == "(none)"


def test_lapispm_status_partial_on_unrecognized_shape():
    weird_stdout = "=== review-gate ===\n  counter: 0/10\n\nSomething unexpected happened.\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(weird_stdout)):
        result = mcp_lapispm.lapispm_status(target_id="my-target")
    assert result["parse_status"] == "partial"
    assert result["raw_output"] == weird_stdout
    assert "title" not in result
    assert "pm_bound" not in result


def test_lapispm_list_argv():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("[]")) as run:
        mcp_lapispm.lapispm_list()
    run.assert_called_once_with(["list", "--json"])


def test_lapispm_list_ok_parses_json():
    payload = [{"target_id": "foo", "pm_repo": "agents-core"}]
    with patch.object(mcp_lapispm, "_run", return_value=_proc(json.dumps(payload))):
        result = mcp_lapispm.lapispm_list()
    assert result["parse_status"] == "ok"
    assert result["targets"] == payload
    assert result["raw_output"] == json.dumps(payload)


def test_lapispm_list_cli_error():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout="", stderr="fail", returncode=1)):
        result = mcp_lapispm.lapispm_list()
    assert result["parse_status"] == "cli_error"
    assert "targets" not in result
    assert result["raw_output"] == ""


def test_lapispm_list_partial_on_bad_json():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("(no pm-bound targets)")):
        result = mcp_lapispm.lapispm_list()
    assert result["parse_status"] == "partial"
    assert "targets" not in result
    assert result["raw_output"] == "(no pm-bound targets)"


def test_lapispm_brief_argv():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")) as run:
        mcp_lapispm.lapispm_brief()
    run.assert_called_once_with(["brief", "--period", "live"])


def test_lapispm_brief_ok():
    stdout = "Brief written: /srv/lapis/briefs/2026-07-01-live.md\nSymlink updated: /srv/lapis/briefs/latest-live.md\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_brief()
    assert result["parse_status"] == "ok"
    assert result["brief_path"] == "/srv/lapis/briefs/2026-07-01-live.md"
    assert result["symlink"] == "/srv/lapis/briefs/latest-live.md"


def test_lapispm_brief_partial_on_empty_output():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")):
        result = mcp_lapispm.lapispm_brief()
    assert result["parse_status"] == "partial"
    assert "brief_path" not in result


def test_lapispm_bind_argv_minimal():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")) as run:
        mcp_lapispm.lapispm_bind("my-target", "/path/to/spec.md", "agents-core")
    run.assert_called_once_with([
        "bind", "my-target",
        "--spec-from", "/path/to/spec.md",
        "--repo", "agents-core",
        "--authority", "advisory",
    ])


def test_lapispm_bind_argv_with_optional_flags():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")) as run:
        mcp_lapispm.lapispm_bind(
            "my-target", "/path/to/spec.md", "agents-core",
            authority="hold", force=True, create=True, title="My Title",
        )
    run.assert_called_once_with([
        "bind", "my-target",
        "--spec-from", "/path/to/spec.md",
        "--repo", "agents-core",
        "--authority", "hold",
        "--force",
        "--create",
        "--title", "My Title",
    ])


def test_lapispm_bind_ok_parses_fields():
    stdout = "Bound my-target → repo=agents-core, authority=advisory, verification=machine\nSpec: 1234 chars\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_bind("my-target", "/spec.md", "agents-core")
    assert result["parse_status"] == "ok"
    assert result["bound_target_id"] == "my-target"
    assert result["bound_repo"] == "agents-core"
    assert result["bound_authority"] == "advisory"
    assert result["bound_verification"] == "machine"
    assert result["spec_chars"] == 1234


def test_lapispm_bind_cli_error():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout="", stderr="ERROR: repo not found", returncode=2)):
        result = mcp_lapispm.lapispm_bind("my-target", "/spec.md", "agents-core")
    assert result["parse_status"] == "cli_error"
    assert "bound_target_id" not in result


def test_lapispm_tick_argv_no_args():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")) as run:
        mcp_lapispm.lapispm_tick()
    run.assert_called_once_with(["tick"])


def test_lapispm_tick_argv_with_target_and_force_dispatch():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")) as run:
        mcp_lapispm.lapispm_tick(target_id="my-target", force_dispatch="fixer:smoke")
    run.assert_called_once_with(["tick", "--target", "my-target", "--force-dispatch", "fixer:smoke"])


def test_lapispm_tick_force_dispatch_ok():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("Dispatched: fixer task_id=abc123\n")):
        result = mcp_lapispm.lapispm_tick(target_id="my-target", force_dispatch="fixer:smoke")
    assert result["parse_status"] == "ok"
    assert result["agent_type"] == "fixer"
    assert result["task_id"] == "abc123"


def test_lapispm_tick_normal_ok():
    stdout = "[my-target] skipped=False reason=- encoded=True decision=dispatch:fixer\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_tick(target_id="my-target")
    assert result["parse_status"] == "ok"
    assert result["ticks"] == [{
        "target_id": "my-target",
        "skipped": "False",
        "reason": "-",
        "encoded": "True",
        "decision": "dispatch:fixer",
    }]


def test_lapispm_tick_reason_with_spaces_parses_ok():
    stdout = (
        "[my-target] skipped=True reason=target not pm_bound encoded=False decision=skip\n"
        "[other-target] skipped=True reason=exception: connection refused encoded=False decision=skip\n"
    )
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_tick()
    assert result["parse_status"] == "ok"
    assert result["ticks"] == [
        {
            "target_id": "my-target",
            "skipped": "True",
            "reason": "target not pm_bound",
            "encoded": "False",
            "decision": "skip",
        },
        {
            "target_id": "other-target",
            "skipped": "True",
            "reason": "exception: connection refused",
            "encoded": "False",
            "decision": "skip",
        },
    ]


def test_lapispm_tick_reason_with_spaces_and_reconciled_parses_ok():
    stdout = "[my-target] skipped=True reason=target not pm_bound reconciled=True encoded=False decision=skip\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_tick()
    assert result["parse_status"] == "ok"
    assert result["ticks"] == [{
        "target_id": "my-target",
        "skipped": "True",
        "reason": "target not pm_bound",
        "reconciled": "True",
        "encoded": "False",
        "decision": "skip",
    }]


def test_lapispm_status_timeout_is_cli_error():
    with patch.object(
        mcp_lapispm.subprocess, "run",
        side_effect=subprocess.TimeoutExpired(cmd=["lapis-pm", "status"], timeout=30.0),
    ):
        result = mcp_lapispm.lapispm_status(target_id="my-target")
    assert result["parse_status"] == "cli_error"
    assert result["exit_code"] == 124
    assert "timed out" in result["stderr"]


def test_lapispm_tick_no_bound_targets_is_ok_empty():
    with patch.object(mcp_lapispm, "_run", return_value=_proc("")):
        result = mcp_lapispm.lapispm_tick()
    assert result["parse_status"] == "ok"
    assert result["ticks"] == []


def test_lapispm_tick_partial_on_unrecognized_line():
    stdout = "something the parser has never seen before\n"
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout)):
        result = mcp_lapispm.lapispm_tick()
    assert result["parse_status"] == "partial"
    assert result["ticks"] == []


def test_lapispm_tick_cli_error():
    with patch.object(mcp_lapispm, "_run", return_value=_proc(stdout="", stderr="ERROR", returncode=2)):
        result = mcp_lapispm.lapispm_tick(target_id="my-target")
    assert result["parse_status"] == "cli_error"
    assert "ticks" not in result


# ---------------------------------------------------------------------------
# spec-review: background-launch + poll pair
# ---------------------------------------------------------------------------

SPEC_REVIEW_DONE_LOG = """# Spec Review: my-target

**Spec:** /srv/lapis/planning/specs/my-target.md
**Repo:** agents-core
**Elapsed:** 123.4s
**Recommendation:** proceed-to-bind

## Mirror Council deliberation
- **Status:** resolved, confidence converged
- **Run ID:** council-abc123
- **Landing:** ship it
- **Positions:**
    - (none)
- **Open questions:**
    - (none)

## Suggested next step
Both passes returned clean signals. Proceed to `lapis-pm bind` after Erah confirms.
"""


def _patch_spec_review_dirs(tmp_path):
    runs_dir = tmp_path / "runs"
    lock_path = tmp_path / "lock"
    return (
        patch.object(mcp_lapispm, "_spec_review_runs_dir", return_value=runs_dir),
        patch.object(mcp_lapispm, "_spec_review_lock_path", return_value=lock_path),
        runs_dir,
        lock_path,
    )


def _write_registry_entry(runs_dir, run_id, **overrides):
    runs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run_id,
        "pid": 424242,
        "spec_path": "/srv/lapis/planning/specs/my-target.md",
        "log_path": str(runs_dir / f"{run_id}.log"),
        "started_at": "2026-07-03T10:00:00Z",
        "cli_argv": ["lapis-pm", "spec-review", "/srv/lapis/planning/specs/my-target.md"],
    }
    entry.update(overrides)
    (runs_dir / f"{run_id}.json").write_text(json.dumps(entry), encoding="utf-8")
    return entry


def test_lapispm_spec_review_start_argv_and_returns_immediately(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    fake_proc = MagicMock()
    fake_proc.pid = 5555
    with p_runs, p_lock, \
         patch.object(mcp_lapispm.subprocess, "Popen", return_value=fake_proc) as popen:
        result = mcp_lapispm.lapispm_spec_review_start(
            spec_path="/srv/lapis/planning/specs/my-target.md",
            council_voicing="gravitywell",
            facets_operator="gravitywell",
            no_sonnet_reviewer=True,
            no_facets=False,
            authority="advisory",
            timeout_s=900,
        )

    assert popen.call_count == 1
    call_args, call_kwargs = popen.call_args
    argv = call_args[0]
    assert argv == [
        "lapis-pm", "spec-review", "/srv/lapis/planning/specs/my-target.md",
        "--council-voicing", "gravitywell",
        "--facets-operator", "gravitywell",
        "--timeout", "900",
        "--no-sonnet-reviewer",
        "--authority", "advisory",
    ]
    assert call_kwargs["start_new_session"] is True
    assert call_kwargs["env"]["PYTHONUNBUFFERED"] == "1"

    fake_proc.wait.assert_not_called()
    fake_proc.communicate.assert_not_called()

    assert result["status"] == "started"
    assert result["pid"] == 5555
    assert "run_id" in result
    assert "log_path" in result
    assert (runs_dir / f"{result['run_id']}.json").exists()


def test_lapispm_spec_review_start_durability_env_unbuffered(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    fake_proc = MagicMock()
    fake_proc.pid = 5556
    with p_runs, p_lock, \
         patch.object(mcp_lapispm.subprocess, "Popen", return_value=fake_proc) as popen:
        mcp_lapispm.lapispm_spec_review_start(spec_path="/srv/lapis/planning/specs/x.md")
    _, call_kwargs = popen.call_args
    assert call_kwargs["env"]["PYTHONUNBUFFERED"] == "1"


def test_lapispm_spec_review_start_collision_own_registry(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    entry = _write_registry_entry(runs_dir, "run-1", pid=os.getpid())
    with p_runs, p_lock, \
         patch.object(mcp_lapispm.subprocess, "Popen") as popen:
        result = mcp_lapispm.lapispm_spec_review_start(spec_path=entry["spec_path"])
    popen.assert_not_called()
    assert result == {
        "status": "already_running",
        "held_by_pid": os.getpid(),
        "spec_path": entry["spec_path"],
        "started_at": entry["started_at"],
        "run_id": "run-1",
    }


def test_lapispm_spec_review_start_collision_cli_lock_file(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    lock_path.write_text(json.dumps({
        "pid": os.getpid(),
        "spec_path": "/srv/lapis/planning/specs/other-target.md",
        "started_at": "2026-07-03T09:00:00Z",
        "host": "brix",
    }), encoding="utf-8")
    with p_runs, p_lock, \
         patch.object(mcp_lapispm.subprocess, "Popen") as popen:
        result = mcp_lapispm.lapispm_spec_review_start(spec_path="/srv/lapis/planning/specs/my-target.md")
    popen.assert_not_called()
    assert result == {
        "status": "already_running",
        "held_by_pid": os.getpid(),
        "spec_path": "/srv/lapis/planning/specs/other-target.md",
        "started_at": "2026-07-03T09:00:00Z",
        "run_id": None,
    }


def test_lapispm_spec_review_start_stale_lock_never_deletes_lock(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    original_lock_bytes = json.dumps({
        "pid": 999999,
        "spec_path": "/srv/lapis/planning/specs/other-target.md",
        "started_at": "2026-07-03T09:00:00Z",
        "host": "brix",
    }).encode()
    lock_path.write_bytes(original_lock_bytes)
    with p_runs, p_lock, \
         patch.object(mcp_lapispm, "_pid_alive", return_value=False), \
         patch.object(
             mcp_lapispm, "_try_nonblocking_flock",
             return_value=(False, "[Errno 11] Resource temporarily unavailable"),
         ), \
         patch.object(mcp_lapispm.subprocess, "Popen") as popen:
        result = mcp_lapispm.lapispm_spec_review_start(spec_path="/srv/lapis/planning/specs/my-target.md")
    popen.assert_not_called()
    assert result == {
        "status": "stale_lock",
        "held_by_pid": 999999,
        "spec_path": "/srv/lapis/planning/specs/other-target.md",
        "flock_error": "[Errno 11] Resource temporarily unavailable",
    }
    # never deleted or rewritten — kernel owns lock state
    assert lock_path.read_bytes() == original_lock_bytes


def test_lapispm_spec_review_poll_running(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    entry = _write_registry_entry(runs_dir, "run-2", pid=os.getpid())
    Path(entry["log_path"]).write_text("partial output so far...\n", encoding="utf-8")
    with p_runs, p_lock:
        result = mcp_lapispm.lapispm_spec_review_poll(run_id="run-2")
    assert result["status"] == "running"
    assert "elapsed_s" in result
    assert result["tail"] == "partial output so far...\n"


def test_lapispm_spec_review_poll_done_parses_brief(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    entry = _write_registry_entry(runs_dir, "run-3", pid=999999)
    Path(entry["log_path"]).write_text(SPEC_REVIEW_DONE_LOG, encoding="utf-8")
    with p_runs, p_lock, patch.object(mcp_lapispm, "_pid_alive", return_value=False):
        result = mcp_lapispm.lapispm_spec_review_poll(run_id="run-3")
    assert result["status"] == "done"
    assert result["exit_code"] == 0
    assert result["parse_status"] == "ok"
    assert result["raw_output"] == SPEC_REVIEW_DONE_LOG
    assert result["recommendation"] == "proceed-to-bind"
    assert result["target_id"] == "my-target"
    assert result["council_status"] == "resolved"
    assert result["council_run_id"] == "council-abc123"
    # strictly exit_code + raw_output + the existing ok/partial parsed-brief contract —
    # never an invented soft/hard-failure classification (spec-review round 2)
    assert "assessment" not in result
    assert not any("assessment" in str(k).lower() for k in result)


def test_lapispm_spec_review_poll_not_found(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    with p_runs, p_lock:
        result = mcp_lapispm.lapispm_spec_review_poll(run_id="does-not-exist")
    assert result == {"status": "not_found"}


def test_lapispm_spec_review_poll_stale_lock_by_spec_path(tmp_path):
    p_runs, p_lock, runs_dir, lock_path = _patch_spec_review_dirs(tmp_path)
    lock_path.write_text(json.dumps({
        "pid": 999999,
        "spec_path": "/srv/lapis/planning/specs/my-target.md",
        "started_at": "2026-07-03T09:00:00Z",
        "host": "brix",
    }), encoding="utf-8")
    with p_runs, p_lock, \
         patch.object(mcp_lapispm, "_pid_alive", return_value=False), \
         patch.object(
             mcp_lapispm, "_try_nonblocking_flock",
             return_value=(False, "[Errno 11] Resource temporarily unavailable"),
         ):
        result = mcp_lapispm.lapispm_spec_review_poll(spec_path="/srv/lapis/planning/specs/my-target.md")
    assert result == {
        "status": "stale_lock",
        "held_by_pid": 999999,
        "spec_path": "/srv/lapis/planning/specs/my-target.md",
        "flock_error": "[Errno 11] Resource temporarily unavailable",
    }
