"""Tests for the local-auditor engine (auditor-diagnosability-v0, leg 1).

The engine's first test surface (R6: ``rg auditor`` over the agents-core
test trees at the parent head returned zero hits - the R1 wiring gap was
invisible to the suite).

AC1 (D-A1, the grant):
  1. ``_get_tool_executors(cwd, auditor=True)`` yields a map with
     ``run_tests`` -> RunTestsExecutor and a git executor restricted to
     AUDITOR_GIT_ALLOWLIST; the default map (no args) has no run_tests;
     the writeable map is unchanged (regression).
  2. ``run_tests`` under the auditor grant actually works: a
     RunTestsExecutor pointed at a tmp cwd containing a trivial passing
     pytest file returns a structured outcome dict (passed >= 1, no
     "unknown tool" error).
  3. ``_run_local_auditor`` passes a ``tool_executors`` containing a
     working run_tests to call_gw_agent (writeable=False, json_mode=True,
     tools=AUDITOR_TOOLS, think=False, timeout/max_steps from the spec).

AC2 (D-A2, the transcript):
  4. A synthetic success run writes {task_id}-gw-transcript.json to the
     shaped dir - valid JSON list, per-step schema.
  5. The no-verdict path (call_gw_agent -> (None, [...])) STILL writes
     the transcript and returns None (stderr contract preserved).
  6. The bare-None shape does not crash the defensive unpack (shape-
     drift guard).
  7. call_gw_agent raising -> the transcript file exists ([]), the WARN
     line is on stderr, and the engine returns None.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agents_core.shaped_runner as sr
from agents_core import gw_agent as gw
from agents_core.gw_agent import (
    AUDITOR_GIT_ALLOWLIST,
    AUDITOR_TOOLS,
    GitExecutor,
    RunTestsExecutor,
    _get_tool_executors,
)

_STEP_KEYS = {"step", "tool_name", "tool_call_id", "arguments", "result", "error"}


def _step(step: int, tool_name: str = "read_file") -> dict:
    return {
        "step": step,
        "tool_name": tool_name,
        "tool_call_id": f"call_{step}",
        "arguments": {"path": "x"},
        "result": "ok",
        "error": None,
    }


# ---------------------------------------------------------------------------
# AC1 / D-A1: the executor grant
# ---------------------------------------------------------------------------


def test_auditor_executors_carry_run_tests_and_allowlisted_git():
    m = _get_tool_executors("/some/cwd", writeable=False, auditor=True)
    assert isinstance(m["run_tests"], RunTestsExecutor)
    git = m["git"]
    assert isinstance(git, GitExecutor)
    assert git._allowlist == AUDITOR_GIT_ALLOWLIST
    # No write tools: the auditor grant is read-only + run_tests + git.
    assert "write_file" not in m
    assert "apply_edit" not in m
    assert "run_command" not in m
    assert "web_fetch" not in m


def test_default_executors_have_no_run_tests():
    m = _get_tool_executors("/some/cwd")
    assert "run_tests" not in m
    # Default git executor keeps the class-default allowlist (None ->
    # GitExecutor.ALLOWLIST), not the auditor allowlist.
    assert m["git"]._allowlist == GitExecutor.ALLOWLIST


def test_writeable_executors_unchanged():
    m = _get_tool_executors("/some/cwd", writeable=True)
    assert isinstance(m["run_tests"], RunTestsExecutor)
    assert "write_file" in m
    assert "apply_edit" in m
    assert "run_command" in m
    assert "web_fetch" in m
    assert m["git"]._allowlist == GitExecutor.ALLOWLIST


def test_run_tests_under_auditor_grant_returns_real_counts(tmp_path):
    # A trivial passing pytest file in a tmp cwd: the executor returns a
    # structured outcome dict, not the unknown-tool error.
    (tmp_path / "test_trivial.py").write_text(
        "def test_ok():\n    assert 1 == 1\n"
    )
    m = _get_tool_executors(str(tmp_path), writeable=False, auditor=True)
    out = m["run_tests"].execute({"target": "test_trivial.py"})
    assert isinstance(out, dict)
    assert "unknown tool" not in json.dumps(out)
    assert out["passed"] >= 1
    assert out["failed"] == 0
    assert out["returncode"] == 0


def _spec() -> dict:
    return {
        "task_id": "claude_20260908_test_audit",
        "prompt": "audit this",
        "system": "you are an auditor",
        "timeout_s": 1234,
        "max_steps": 77,
        "model": "test-model",
        "backend_url": None,
        "acquire_lease": True,
    }


def test_run_local_auditor_passes_working_tool_executors(tmp_path):
    shaped_dir = tmp_path / "shaped"
    captured: dict = {}

    def fake_call_gw_agent(**kwargs):
        captured.update(kwargs)
        return ("{\"ok\": true}", [_step(1)])

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(gw, "call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result == "{\"ok\": true}"
    kwargs = captured
    assert kwargs["writeable"] is False
    assert kwargs["json_mode"] is True
    assert kwargs["tools"] is AUDITOR_TOOLS
    assert kwargs["think"] is False
    assert kwargs["timeout"] == 1234
    assert kwargs["max_steps"] == 77
    assert kwargs["return_transcript"] is True
    ex = kwargs["tool_executors"]
    assert isinstance(ex, dict)
    assert isinstance(ex["run_tests"], RunTestsExecutor)
    # Same cwd the engine resolves.
    assert ex["run_tests"].cwd == Path("/some/cwd").resolve()
    assert ex["git"]._allowlist == AUDITOR_GIT_ALLOWLIST


# ---------------------------------------------------------------------------
# AC2 / D-A2: the transcript artifact
# ---------------------------------------------------------------------------


def test_success_run_writes_transcript(tmp_path):
    shaped_dir = tmp_path / "shaped"
    steps = [_step(1), _step(2, "run_tests")]

    def fake_call_gw_agent(**kwargs):
        return ("{\"ok\": true}", steps)

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(gw, "call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result == "{\"ok\": true}"
    path = shaped_dir / f"{_spec()['task_id']}-gw-transcript.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    for entry in data:
        assert _STEP_KEYS.issubset(entry.keys())


def test_no_verdict_still_writes_transcript_and_returns_none(tmp_path, capsys):
    shaped_dir = tmp_path / "shaped"

    def fake_call_gw_agent(**kwargs):
        kwargs["reason_out"].append("max_steps_exhausted")
        return (None, [_step(1)])

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(gw, "call_gw_agent", side_effect=fake_call_gw_agent),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result is None
    err = capsys.readouterr().err
    assert "reason=max_steps_exhausted" in err
    path = shaped_dir / f"{_spec()['task_id']}-gw-transcript.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1


def test_bare_none_shape_does_not_crash_unpack(tmp_path):
    """Shape-drift guard: bare None (only reachable with
    return_transcript=False) must not crash the defensive unpack; an
    empty-list transcript is written."""
    shaped_dir = tmp_path / "shaped"

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(gw, "call_gw_agent", return_value=None),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result is None
    path = shaped_dir / f"{_spec()['task_id']}-gw-transcript.json"
    assert path.exists()
    assert json.loads(path.read_text()) == []


def test_raise_writes_empty_transcript_warns_and_returns_none(tmp_path, capsys):
    shaped_dir = tmp_path / "shaped"

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(
            gw, "call_gw_agent", side_effect=Exception("seat exploded")
        ),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result is None
    err = capsys.readouterr().err
    assert "WARN: local-auditor: call_gw_agent raised: seat exploded" in err
    path = shaped_dir / f"{_spec()['task_id']}-gw-transcript.json"
    assert path.exists()
    assert json.loads(path.read_text()) == []


def test_transcript_write_failure_warns_not_raises(tmp_path, capsys):
    """OSError on the artifact write -> WARN on stderr, no crash, the
    verdict still returns (mirrors the local-fixer write shape)."""
    shaped_dir = tmp_path / "shaped"
    steps = [_step(1)]

    def fake_call_gw_agent(**kwargs):
        return ("{\"ok\": true}", steps)

    with (
        patch.object(sr, "room_path", lambda name: shaped_dir),
        patch.object(gw, "call_gw_agent", side_effect=fake_call_gw_agent),
        patch.object(Path, "write_text", side_effect=OSError("disk full")),
    ):
        result = sr._run_local_auditor(_spec(), "/some/cwd")

    assert result == "{\"ok\": true}"
    err = capsys.readouterr().err
    assert "WARN: local-auditor: transcript write failed: disk full" in err
