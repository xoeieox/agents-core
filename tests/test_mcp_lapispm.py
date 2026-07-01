"""mcp_lapispm tests: each MCP tool must construct the correct `lapis-pm`
argv, and the parse-contract must hold:
  (1) non-zero exit -> parse_status == "cli_error", no parsed fields
  (2) clean status/list stdout -> parse_status == "ok" with expected fields
  (3) an unexpected stdout shape -> parse_status == "partial", ambiguous
      fields omitted (never filled with a guess)
`raw_output` must be present in all three cases.
"""

import json
import subprocess
from unittest.mock import patch

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
