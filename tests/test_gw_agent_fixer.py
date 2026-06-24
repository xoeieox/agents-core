"""Unit tests for the write-capable local fixer harness (local-fixer-write-harness-v0).

All tests are fully offline (no live GW, no live swarm, no network).
AC1-AC7 per spec.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from agents_core.gw_agent import (
    GW_AGENT_TOOL_INPUT_CAP,
    GW_AGENT_TOOL_OUTPUT_CAP,
    ApplyEditExecutor,
    DEFAULT_FIXER_TOOLS,
    DEFAULT_READONLY_TOOLS,
    RunTestsExecutor,
    WriteFileExecutor,
    _build_fixer_result,
    _get_tool_executors,
    _parse_pytest_outcome,
    call_gw_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo in tmp_path."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    # Initial commit so git diff works
    (tmp_path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "."], check=True, capture_output=True, cwd=str(tmp_path))
    subprocess.run(
        ["git", "commit", "-m", "init"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    return tmp_path


# ---------------------------------------------------------------------------
# AC1 / AC1b: WriteFileExecutor
# ---------------------------------------------------------------------------

class TestWriteFileExecutor:
    def test_write_creates_file(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        result = ex.execute({"path": "hello.txt", "content": "world"})
        assert result == "wrote 5 bytes to hello.txt"
        assert (tmp_path / "hello.txt").read_text() == "world"

    def test_write_creates_parent_dir(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        result = ex.execute({"path": "new_subdir/file.py", "content": "x = 1\n"})
        assert isinstance(result, str) and result.startswith("wrote")
        assert (tmp_path / "new_subdir" / "file.py").read_text() == "x = 1\n"

    def test_write_overwrites_existing(self, tmp_path):
        (tmp_path / "f.txt").write_text("old")
        ex = WriteFileExecutor(str(tmp_path))
        ex.execute({"path": "f.txt", "content": "new"})
        assert (tmp_path / "f.txt").read_text() == "new"

    def test_path_escape_rejected(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        result = ex.execute({"path": "../outside.txt", "content": "evil"})
        assert isinstance(result, dict) and "error" in result
        assert not (tmp_path.parent / "outside.txt").exists()

    def test_content_over_cap_returns_error_no_write(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        big = "x" * (GW_AGENT_TOOL_INPUT_CAP + 1)
        result = ex.execute({"path": "big.txt", "content": big})
        assert isinstance(result, dict) and "error" in result
        assert "too large" in result["error"]
        assert not (tmp_path / "big.txt").exists()

    def test_content_at_cap_succeeds(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        at_cap = "a" * GW_AGENT_TOOL_INPUT_CAP
        result = ex.execute({"path": "cap.txt", "content": at_cap})
        assert isinstance(result, str) and "wrote" in result
        assert len((tmp_path / "cap.txt").read_text()) == GW_AGENT_TOOL_INPUT_CAP

    def test_error_returned_not_raised(self, tmp_path):
        ex = WriteFileExecutor(str(tmp_path))
        # Force an exception by passing a non-string content
        result = ex.execute({"path": "../escape", "content": "x"})
        assert isinstance(result, dict)
        assert "error" in result


# ---------------------------------------------------------------------------
# AC2: ApplyEditExecutor
# ---------------------------------------------------------------------------

class TestApplyEditExecutor:
    def test_unique_match_succeeds(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({
            "path": "code.py",
            "old_string": "return 1",
            "new_string": "return 42",
        })
        assert isinstance(result, str) and "applied" in result
        assert "return 42" in f.read_text()

    def test_missing_old_string_errors(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({
            "path": "code.py",
            "old_string": "return 999",
            "new_string": "return 0",
        })
        assert isinstance(result, dict) and "error" in result
        assert "not found" in result["error"]

    def test_non_unique_old_string_errors(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("x = 1\nx = 1\n")
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({
            "path": "code.py",
            "old_string": "x = 1",
            "new_string": "x = 2",
        })
        assert isinstance(result, dict) and "error" in result
        assert "not unique" in result["error"]

    def test_path_escape_rejected(self, tmp_path):
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({
            "path": "../outside.py",
            "old_string": "a",
            "new_string": "b",
        })
        assert isinstance(result, dict) and "error" in result

    def test_file_not_found_errors(self, tmp_path):
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({
            "path": "nonexistent.py",
            "old_string": "x",
            "new_string": "y",
        })
        assert isinstance(result, dict) and "error" in result
        assert "not found" in result["error"]

    def test_error_returned_not_raised(self, tmp_path):
        ex = ApplyEditExecutor(str(tmp_path))
        result = ex.execute({"path": "../escape.py", "old_string": "x", "new_string": "y"})
        assert isinstance(result, dict) and "error" in result


# ---------------------------------------------------------------------------
# AC3: RunTestsExecutor
# ---------------------------------------------------------------------------

class TestRunTestsExecutor:
    def test_passing_tests(self, tmp_path):
        (tmp_path / "test_pass.py").write_text(
            "def test_ok():\n    assert True\n"
        )
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({})
        assert isinstance(result, dict)
        assert result["passed"] >= 1
        assert result["failed"] == 0
        assert result["timed_out"] is False

    def test_failing_tests(self, tmp_path):
        (tmp_path / "test_fail.py").write_text(
            "def test_bad():\n    assert False\n"
        )
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({})
        assert isinstance(result, dict)
        assert result["failed"] >= 1

    def test_collection_error_tolerant(self, tmp_path):
        (tmp_path / "test_broken.py").write_text(
            "import nonexistent_module_xyz\ndef test_x(): pass\n"
        )
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({})
        # Should not raise; should return structured dict with error info
        assert isinstance(result, dict)
        assert "output_tail" in result

    def test_timeout_tolerant(self, tmp_path):
        (tmp_path / "test_slow.py").write_text(
            "import time\ndef test_slow():\n    time.sleep(999)\n"
        )
        ex = RunTestsExecutor(str(tmp_path), run_timeout=1)
        result = ex.execute({})
        assert isinstance(result, dict)
        assert result["timed_out"] is True

    def test_output_capped(self, tmp_path):
        # Write a test that produces lots of output
        tests = "\n".join(
            f"def test_{i}(): print('x' * 500)" for i in range(100)
        )
        (tmp_path / "test_verbose.py").write_text(tests)
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({})
        assert isinstance(result, dict)
        assert len(result["output_tail"]) <= GW_AGENT_TOOL_OUTPUT_CAP + 50  # small buffer for truncation marker

    def test_shell_metachar_rejected(self, tmp_path):
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({"target": "tests; rm -rf /"})
        assert isinstance(result, dict) and "error" in result

    def test_k_expr_with_spaces_accepted(self, tmp_path):
        # k_expr with spaces is valid pytest syntax
        (tmp_path / "test_k.py").write_text(
            "def test_alpha(): pass\ndef test_beta(): pass\n"
        )
        ex = RunTestsExecutor(str(tmp_path))
        result = ex.execute({"k_expr": "alpha or beta"})
        assert isinstance(result, dict)
        assert "timed_out" in result

    def test_shell_false_enforced(self, tmp_path):
        # Verify RunTestsExecutor passes shell=False by patching subprocess.run
        with patch("agents_core.gw_agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="1 passed in 0.01s\n", stderr=""
            )
            ex = RunTestsExecutor(str(tmp_path))
            ex.execute({})
            args, kwargs = mock_run.call_args
            assert kwargs.get("shell") is False or kwargs.get("shell") is None
            # The cmd must be a list, not a string (enforces shell=False at argv level)
            assert isinstance(args[0], list)

    def test_error_returned_not_raised(self, tmp_path):
        ex = RunTestsExecutor(str(tmp_path))
        with patch("agents_core.gw_agent.subprocess.run", side_effect=OSError("boom")):
            result = ex.execute({})
        assert isinstance(result, dict) and "error" in result


# ---------------------------------------------------------------------------
# AC4: call_gw_agent(writeable=True) end-to-end with mocked backend
# ---------------------------------------------------------------------------

def _make_tool_call_response(tool_name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"total_tokens": 100},
    }


def _make_stop_response(content: str = "Done.") -> dict:
    return {
        "choices": [{
            "message": {"content": content, "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"total_tokens": 200},
    }


class TestCallGwAgentWriteable:
    def test_writeable_run_mutates_repo_and_returns_fixer_result(self, tmp_path):
        repo = _tmp_git_repo(tmp_path)
        # Write a tracked file so git diff captures the edit
        (repo / "src.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "."], check=True, capture_output=True, cwd=str(repo))
        subprocess.run(["git", "commit", "-m", "add src"], check=True, capture_output=True, cwd=str(repo))

        # Step 1: apply_edit on a tracked file so git diff shows a change
        step1 = _make_tool_call_response(
            "apply_edit", {"path": "src.py", "old_string": "x = 1", "new_string": "x = 42"}, "c1"
        )
        step2 = _make_tool_call_response(
            "run_tests", {}, "c2"
        )
        step3 = _make_stop_response("All done.")

        responses = [
            MagicMock(status_code=200, json=MagicMock(return_value=step1)),
            MagicMock(status_code=200, json=MagicMock(return_value=step2)),
            MagicMock(status_code=200, json=MagicMock(return_value=step3)),
        ]
        for r in responses:
            r.raise_for_status = MagicMock()

        with patch("agents_core.gw_agent.requests.post", side_effect=responses), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            result = call_gw_agent(
                prompt="Fix the code.",
                cwd=str(repo),
                writeable=True,
                acquire_lease=True,
                backend_url=None,
            )

        assert isinstance(result, tuple)
        fixer_result, transcript = result

        # src.py must have been modified
        assert (repo / "src.py").read_text() == "x = 42\n"

        # final_diff must be non-empty (tracked file was edited)
        assert isinstance(fixer_result["final_diff"], str)
        assert len(fixer_result["final_diff"]) > 0

        # last_test_outcome must be populated
        assert fixer_result["last_test_outcome"] is not None
        assert "passed" in fixer_result["last_test_outcome"]

        # concluded=True because step3 ended on stop
        assert fixer_result["concluded"] is True

        # steps == transcript
        assert fixer_result["steps"] is transcript
        assert len(transcript) >= 2

    def test_writeable_ignores_return_transcript_arg(self, tmp_path):
        repo = _tmp_git_repo(tmp_path)
        step1 = _make_stop_response("Done.")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        responses[0].raise_for_status = MagicMock()

        with patch("agents_core.gw_agent.requests.post", side_effect=responses), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            # return_transcript=False but writeable=True -> should still return tuple
            result = call_gw_agent(
                prompt="Fix.",
                cwd=str(repo),
                writeable=True,
                return_transcript=False,
                acquire_lease=True,
                backend_url=None,
            )

        assert isinstance(result, tuple)
        fixer, transcript = result
        assert isinstance(fixer, dict)
        assert "final_diff" in fixer

    def test_no_run_tests_last_test_outcome_is_none(self, tmp_path):
        repo = _tmp_git_repo(tmp_path)
        step1 = _make_stop_response("Done without tests.")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        responses[0].raise_for_status = MagicMock()

        with patch("agents_core.gw_agent.requests.post", side_effect=responses), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            result = call_gw_agent(
                prompt="Fix.",
                cwd=str(repo),
                writeable=True,
                acquire_lease=True,
                backend_url=None,
            )

        fixer, _ = result
        assert fixer["last_test_outcome"] is None


# ---------------------------------------------------------------------------
# AC5: writeable=False is read-only (regression guard)
# ---------------------------------------------------------------------------

class TestReadOnlyRegression:
    def test_writeable_false_tools_are_readonly(self):
        executors = _get_tool_executors("/tmp", writeable=False)
        assert "write_file" not in executors
        assert "apply_edit" not in executors
        assert "run_tests" not in executors

    def test_writeable_false_default_tools_unchanged(self):
        step1 = _make_stop_response("OK")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        responses[0].raise_for_status = MagicMock()

        captured_bodies = []

        def fake_post(url, json=None, timeout=None):
            captured_bodies.append(json)
            return responses.pop(0)

        with patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            result = call_gw_agent(
                prompt="Review this.",
                cwd="/tmp",
                writeable=False,
                acquire_lease=True,
                backend_url=None,
            )

        # tools sent to backend must match DEFAULT_READONLY_TOOLS exactly
        body = captured_bodies[0]
        sent_tool_names = {t["function"]["name"] for t in body["tools"]}
        expected_names = set(DEFAULT_READONLY_TOOLS.keys())
        assert sent_tool_names == expected_names

    def test_writeable_true_default_tools_are_fixer(self):
        step1 = _make_stop_response("OK")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        responses[0].raise_for_status = MagicMock()

        captured_bodies = []

        def fake_post(url, json=None, timeout=None):
            captured_bodies.append(json)
            return responses.pop(0)

        with patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            with tempfile.TemporaryDirectory() as td:
                _tmp_git_repo(Path(td))
                call_gw_agent(
                    prompt="Fix.",
                    cwd=td,
                    writeable=True,
                    acquire_lease=True,
                    backend_url=None,
                )

        body = captured_bodies[0]
        sent_tool_names = {t["function"]["name"] for t in body["tools"]}
        assert "write_file" in sent_tool_names
        assert "apply_edit" in sent_tool_names
        assert "run_tests" in sent_tool_names


# ---------------------------------------------------------------------------
# AC6: swarm-payload safety (no chat_template_kwargs on swarm path)
# ---------------------------------------------------------------------------

class TestSwarmPayloadSafety:
    """Assert that chat_template_kwargs is omitted on the swarm path and present on GW path,
    for both the main tool-loop POST and the _force_conclusion POST."""

    def _run_with_captured_posts(self, responses, is_swarm: bool, backend_url: str = "http://swarm:8080"):
        """Helper: run call_gw_agent and return all captured POST JSON bodies."""
        captured = []

        def fake_post(url, json=None, timeout=None):
            captured.append(json)
            r = responses.pop(0)
            r.raise_for_status = MagicMock()
            return r

        cwd = tempfile.mkdtemp()
        _tmp_git_repo(Path(cwd))

        with patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            call_gw_agent(
                prompt="Fix.",
                cwd=cwd,
                writeable=False,
                acquire_lease=(not is_swarm),
                backend_url=backend_url if is_swarm else None,
                max_steps=2,
            )

        return captured

    def test_swarm_main_loop_no_chat_template_kwargs(self):
        step1 = _make_stop_response("OK")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        bodies = self._run_with_captured_posts(responses, is_swarm=True)
        assert len(bodies) >= 1
        assert "chat_template_kwargs" not in bodies[0], (
            "chat_template_kwargs must be absent on swarm path (main loop POST)"
        )

    def test_gw_main_loop_has_chat_template_kwargs(self):
        step1 = _make_stop_response("OK")
        responses = [MagicMock(status_code=200, json=MagicMock(return_value=step1))]
        bodies = self._run_with_captured_posts(responses, is_swarm=False)
        assert len(bodies) >= 1
        assert "chat_template_kwargs" in bodies[0], (
            "chat_template_kwargs must be present on GW path"
        )

    def test_swarm_force_conclusion_no_chat_template_kwargs(self):
        """Exhaust max_steps so _force_conclusion is exercised; verify its POST is also clean."""
        # Two tool-call responses to exhaust max_steps=2, then _force_conclusion fires
        tool_resp = _make_tool_call_response("read_file", {"path": "README.md"})
        tool_resp2 = _make_tool_call_response("read_file", {"path": "README.md"})
        stop_resp = _make_stop_response("conclusion")

        responses_list = [
            MagicMock(status_code=200, json=MagicMock(return_value=tool_resp)),
            MagicMock(status_code=200, json=MagicMock(return_value=tool_resp2)),
            # _force_conclusion fires after max_steps
            MagicMock(status_code=200, json=MagicMock(return_value=stop_resp)),
        ]
        for r in responses_list:
            r.raise_for_status = MagicMock()

        captured = []
        cwd = tempfile.mkdtemp()
        _tmp_git_repo(Path(cwd))

        def fake_post(url, json=None, timeout=None):
            captured.append(json)
            r = responses_list.pop(0)
            return r

        with patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            call_gw_agent(
                prompt="Fix.",
                cwd=cwd,
                acquire_lease=False,
                backend_url="http://swarm:8080",
                max_steps=2,
            )

        # All POSTs on swarm path must omit chat_template_kwargs
        assert len(captured) >= 2, "Expected at least main-loop POST + force-conclusion POST"
        for i, body in enumerate(captured):
            assert "chat_template_kwargs" not in body, (
                f"POST #{i} on swarm path must not contain chat_template_kwargs"
            )

    def test_gw_force_conclusion_has_chat_template_kwargs(self):
        """Exhaust max_steps on GW path; verify _force_conclusion POST includes chat_template_kwargs."""
        tool_resp = _make_tool_call_response("read_file", {"path": "README.md"})
        tool_resp2 = _make_tool_call_response("read_file", {"path": "README.md"})
        stop_resp = _make_stop_response("conclusion")

        responses_list = [
            MagicMock(status_code=200, json=MagicMock(return_value=tool_resp)),
            MagicMock(status_code=200, json=MagicMock(return_value=tool_resp2)),
            MagicMock(status_code=200, json=MagicMock(return_value=stop_resp)),
        ]
        for r in responses_list:
            r.raise_for_status = MagicMock()

        captured = []
        cwd = tempfile.mkdtemp()

        def fake_post(url, json=None, timeout=None):
            captured.append(json)
            r = responses_list.pop(0)
            return r

        with patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            mock_client = MagicMock()
            mock_client.acquire.return_value = {"status": "serving"}
            MockDoorman.return_value = mock_client

            call_gw_agent(
                prompt="Review.",
                cwd=cwd,
                acquire_lease=True,
                backend_url=None,
                max_steps=2,
            )

        # All POSTs on GW path must include chat_template_kwargs
        assert len(captured) >= 2
        for i, body in enumerate(captured):
            assert "chat_template_kwargs" in body, (
                f"POST #{i} on GW path must contain chat_template_kwargs"
            )


# ---------------------------------------------------------------------------
# AC7: Parse pytest output helper unit tests (no network)
# ---------------------------------------------------------------------------

class TestParsePytestOutcome:
    def test_passed(self):
        output = ".\n1 passed in 0.05s\n"
        result = _parse_pytest_outcome(output, 0, False)
        assert result["passed"] == 1
        assert result["failed"] == 0
        assert result["errors"] == 0
        assert result["timed_out"] is False

    def test_failed(self):
        output = "F\n1 failed in 0.05s\n"
        result = _parse_pytest_outcome(output, 1, False)
        assert result["failed"] == 1
        assert result["passed"] == 0

    def test_error(self):
        output = "E\n1 error in 0.05s\n"
        result = _parse_pytest_outcome(output, 2, False)
        assert result["errors"] == 1

    def test_timeout(self):
        output = "[TIMEOUT after 1s]"
        result = _parse_pytest_outcome(output, -1, True)
        assert result["timed_out"] is True

    def test_output_tail_last_20_lines(self):
        lines = [f"line{i}" for i in range(30)]
        output = "\n".join(lines)
        result = _parse_pytest_outcome(output, 0, False)
        tail_lines = result["output_tail"].splitlines()
        assert len(tail_lines) == 20
        assert tail_lines[0] == "line10"
