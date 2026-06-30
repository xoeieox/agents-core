"""Tests for gw-agent-actuation-hooks-v0: injectable executors + cancel + step-gate hooks.

All tests are fully offline (no live GW, no doorman, no network).
Covers: custom_executor, cancel_check, before_tool, fail-safe, fail-closed, regression.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import (
    ToolExecutor,
    _get_tool_executors,
    call_gw_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    (tmp_path / "README.md").write_text("init\n")
    subprocess.run(["git", "add", "."], check=True, capture_output=True, cwd=str(tmp_path))
    subprocess.run(
        ["git", "commit", "-m", "init"],
        check=True, capture_output=True, cwd=str(tmp_path),
    )
    return tmp_path


def _mock_gw_stop(content: str) -> dict:
    """Build a GW response that returns content and stops."""
    return {
        "choices": [
            {
                "message": {"content": content, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 50},
    }


def _mock_gw_tool_call(tool_name: str, args: dict, call_id: str = "call_1") -> dict:
    """Build a GW response requesting one tool call."""
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"total_tokens": 50},
    }


def _setup_doorman(mock_doorman_class):
    mock_doorman = MagicMock()
    mock_doorman_class.return_value = mock_doorman
    mock_doorman.acquire.return_value = {"status": "serving"}
    return mock_doorman


# ---------------------------------------------------------------------------
# Custom tool_executors
# ---------------------------------------------------------------------------


class FakeEchoExecutor(ToolExecutor):
    """Returns a fixed string for any call."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    def execute(self, arguments: dict) -> str:
        self.calls.append(arguments)
        return self.response


class TestCustomExecutors:
    def test_custom_executor_is_invoked_and_result_fed_back(self):
        """Caller-supplied executor map is used instead of the default registry."""
        echo_ex = FakeEchoExecutor("custom_tool_result")
        custom_executors = {"my_tool": echo_ex}
        custom_tools = {
            "my_tool": {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "Custom tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        # Step 1: GW calls my_tool. Step 2: GW stops with final verdict.
        responses = [
            _mock_gw_tool_call("my_tool", {"key": "val"}),
            _mock_gw_stop("done"),
        ]

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            _setup_doorman(mock_dc)
            mock_post.return_value.json.side_effect = responses

            result = call_gw_agent(
                prompt="run my_tool",
                tools=custom_tools,
                tool_executors=custom_executors,
                acquire_lease=True,
            )

        assert result == "done"
        assert len(echo_ex.calls) == 1
        assert echo_ex.calls[0] == {"key": "val"}

    def test_custom_executor_result_in_tool_message(self):
        """Custom executor output appears as a tool message fed back to the model."""
        echo_ex = FakeEchoExecutor("my_output")
        custom_executors = {"my_tool": echo_ex}
        custom_tools = {
            "my_tool": {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "Custom tool.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        posted_payloads: list[dict] = []

        def capture_post(url, json=None, timeout=None):
            posted_payloads.append(json)
            if len(posted_payloads) == 1:
                resp = MagicMock()
                resp.json.return_value = _mock_gw_tool_call("my_tool", {})
                return resp
            resp = MagicMock()
            resp.json.return_value = _mock_gw_stop("ok")
            return resp

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post", side_effect=capture_post):
            _setup_doorman(mock_dc)

            call_gw_agent(
                prompt="test",
                tools=custom_tools,
                tool_executors=custom_executors,
            )

        # Second POST should include the tool result message with custom output
        msgs2 = posted_payloads[1]["messages"]
        tool_msgs = [m for m in msgs2 if m.get("role") == "tool"]
        assert any("my_output" in m.get("content", "") for m in tool_msgs)

    def test_default_registry_used_when_tool_executors_none(self):
        """When tool_executors=None, the default registry is built (byte-identical regression)."""
        with patch("agents_core.gw_agent._get_tool_executors") as mock_get_ex, \
             patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            _setup_doorman(mock_dc)
            mock_get_ex.return_value = {}
            mock_post.return_value.json.return_value = _mock_gw_stop("done")

            call_gw_agent(prompt="hello", tool_executors=None)

            mock_get_ex.assert_called_once()


# ---------------------------------------------------------------------------
# cancel_check
# ---------------------------------------------------------------------------


class TestCancelCheck:
    def test_cancel_after_step1_stops_before_next_tool(self, tmp_path):
        """cancel_check flipping true after step 1 stops the loop before the next tool."""
        _tmp_git_repo(tmp_path)
        step_counter = [0]
        tool_calls_executed = [0]

        def cancel_check() -> bool:
            # True on second check (at step-top of step 2)
            step_counter[0] += 1
            return step_counter[0] >= 3  # first call at step1 top, second at pre-tool step1, third at step2 top

        class CountingExecutor(ToolExecutor):
            def execute(self, arguments: dict) -> str:
                tool_calls_executed[0] += 1
                return "ok"

        executors = {"fake_tool": CountingExecutor()}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        # Step 1: tool call. Step 2: would be another tool call (but interrupted at step top)
        responses = [
            _mock_gw_tool_call("fake_tool", {}),
            _mock_gw_tool_call("fake_tool", {}),
            _mock_gw_stop("should not reach"),
        ]

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.side_effect = responses

            fixer, transcript = call_gw_agent(
                prompt="loop",
                cwd=str(tmp_path),
                tools=tools,
                tool_executors=executors,
                cancel_check=cancel_check,
                writeable=True,
            )

        assert fixer["interrupted"] is True
        assert fixer["interrupt_reason"] == "user_cancel"
        assert fixer["concluded"] is False
        mock_doorman.release.assert_called_once()

    def test_cancel_check_at_step_top_returns_interrupted(self, tmp_path):
        """cancel_check returning True at step-top immediately halts and returns interrupted."""
        _tmp_git_repo(tmp_path)

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_stop("never")

            fixer, _ = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                cancel_check=lambda: True,
                writeable=True,
            )

        # GW should never have been called (interrupted before first POST)
        mock_post.assert_not_called()
        assert fixer["interrupted"] is True
        assert fixer["interrupt_reason"] == "user_cancel"
        mock_doorman.release.assert_called_once()

    def test_cancel_check_readonly_returns_marker(self, tmp_path):
        """Read-only interrupted result carries the [gw_agent: interrupted ...] marker."""
        _tmp_git_repo(tmp_path)

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_stop("never")

            result = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                cancel_check=lambda: True,
            )

        assert result is not None
        assert "interrupted" in result
        assert "user_cancel" in result

    def test_cancel_check_lease_released_on_interrupt(self, tmp_path):
        """Doorman lease is always released when cancel_check stops the loop."""
        _tmp_git_repo(tmp_path)

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_stop("never")

            call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                cancel_check=lambda: True,
                writeable=True,
            )

        mock_doorman.release.assert_called_once()


# ---------------------------------------------------------------------------
# Fail-safe: cancel_check raising -> interrupted, cancel_check_failed
# ---------------------------------------------------------------------------


class TestCancelCheckFailSafe:
    def test_cancel_check_raising_halts_not_continues(self, tmp_path):
        """cancel_check raising must halt the loop (fail-safe), never silently continue."""
        _tmp_git_repo(tmp_path)

        def bad_cancel_check() -> bool:
            raise RuntimeError("cancel check broken")

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_stop("should not reach")

            fixer, _ = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                cancel_check=bad_cancel_check,
                writeable=True,
            )

        # Must be interrupted with fail-safe reason
        assert fixer["interrupted"] is True
        assert fixer["interrupt_reason"] == "cancel_check_failed"
        assert fixer["concluded"] is False
        # GW should not have been called (interrupted at step top)
        mock_post.assert_not_called()
        # Lease must be released
        mock_doorman.release.assert_called_once()

    def test_cancel_check_failing_pre_tool_halts(self, tmp_path):
        """cancel_check raising in pre-tool position also halts (fail-safe)."""
        _tmp_git_repo(tmp_path)
        call_count = [0]

        def fragile_cancel() -> bool:
            call_count[0] += 1
            if call_count[0] >= 2:  # pass step-top, raise on pre-tool
                raise RuntimeError("exploded")
            return False

        executors = {"fake_tool": FakeEchoExecutor("ok")}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_tool_call("fake_tool", {})

            fixer, _ = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                cancel_check=fragile_cancel,
                tool_executors=executors,
                tools=tools,
                writeable=True,
            )

        assert fixer["interrupted"] is True
        assert fixer["interrupt_reason"] == "cancel_check_failed"
        mock_doorman.release.assert_called_once()


# ---------------------------------------------------------------------------
# before_tool step-gate
# ---------------------------------------------------------------------------


class TestBeforeTool:
    def test_proceed_executes_tool_normally(self, tmp_path):
        """Gate returning proceed -> tool executes normally."""
        _tmp_git_repo(tmp_path)
        echo_ex = FakeEchoExecutor("tool_ran")
        executors = {"fake_tool": echo_ex}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        responses = [_mock_gw_tool_call("fake_tool", {}), _mock_gw_stop("finished")]

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            _setup_doorman(mock_dc)
            mock_post.return_value.json.side_effect = responses

            result = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                tools=tools,
                tool_executors=executors,
                before_tool=lambda name, args: {"decision": "proceed"},
            )

        assert result == "finished"
        assert len(echo_ex.calls) == 1

    def test_reject_skips_tool_feeds_error_back(self, tmp_path):
        """Gate returning reject skips the tool and feeds the error back to the model."""
        _tmp_git_repo(tmp_path)
        echo_ex = FakeEchoExecutor("should_not_run")
        executors = {"fake_tool": echo_ex}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        posted_payloads: list[dict] = []

        def capture_post(url, json=None, timeout=None):
            posted_payloads.append(json)
            resp = MagicMock()
            if len(posted_payloads) == 1:
                resp.json.return_value = _mock_gw_tool_call("fake_tool", {})
            else:
                resp.json.return_value = _mock_gw_stop("loop continued after reject")
            return resp

        def gate(name: str, args: dict) -> dict:
            return {"decision": "reject", "reason": "not allowed right now"}

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post", side_effect=capture_post):
            _setup_doorman(mock_dc)

            result = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                tools=tools,
                tool_executors=executors,
                before_tool=gate,
            )

        # Tool must NOT have executed
        assert len(echo_ex.calls) == 0
        # Loop continued: GW was called twice (once to get tool call, once after reject)
        assert len(posted_payloads) == 2
        # Second POST must include a tool result with the rejection error
        msgs2 = posted_payloads[1]["messages"]
        tool_msgs = [m for m in msgs2 if m.get("role") == "tool"]
        assert any("rejected" in m.get("content", "") and "not allowed right now" in m.get("content", "")
                   for m in tool_msgs)
        assert result == "loop continued after reject"

    def test_stop_halts_loop_interrupted(self, tmp_path):
        """Gate returning stop halts the loop with interrupted result."""
        _tmp_git_repo(tmp_path)
        echo_ex = FakeEchoExecutor("should_not_run")
        executors = {"fake_tool": echo_ex}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_tool_call("fake_tool", {})

            fixer, _ = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                tools=tools,
                tool_executors=executors,
                before_tool=lambda name, args: {"decision": "stop"},
                writeable=True,
            )

        assert fixer["interrupted"] is True
        assert fixer["interrupt_reason"] == "user_cancel"
        assert len(echo_ex.calls) == 0
        mock_doorman.release.assert_called_once()

    def test_before_tool_raising_fail_closed_loop_continues(self, tmp_path):
        """before_tool raising is fail-closed: tool skipped with gate_failure error, loop continues."""
        _tmp_git_repo(tmp_path)
        echo_ex = FakeEchoExecutor("should_not_run")
        executors = {"fake_tool": echo_ex}
        tools = {
            "fake_tool": {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "Fake.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        }

        posted_payloads: list[dict] = []

        def capture_post(url, json=None, timeout=None):
            posted_payloads.append(json)
            resp = MagicMock()
            if len(posted_payloads) == 1:
                resp.json.return_value = _mock_gw_tool_call("fake_tool", {})
            else:
                resp.json.return_value = _mock_gw_stop("continued after gate_failure")
            return resp

        def exploding_gate(name: str, args: dict) -> dict:
            raise RuntimeError("gate logic crashed")

        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post", side_effect=capture_post):
            _setup_doorman(mock_dc)

            result = call_gw_agent(
                prompt="test",
                cwd=str(tmp_path),
                tools=tools,
                tool_executors=executors,
                before_tool=exploding_gate,
            )

        # Tool must NOT have executed
        assert len(echo_ex.calls) == 0
        # Loop must have continued (two GW calls)
        assert len(posted_payloads) == 2
        # gate_failure error must appear as a tool message
        msgs2 = posted_payloads[1]["messages"]
        tool_msgs = [m for m in msgs2 if m.get("role") == "tool"]
        assert any("gate_failure" in m.get("content", "") for m in tool_msgs)
        assert result == "continued after gate_failure"


# ---------------------------------------------------------------------------
# Regression: default params -> unchanged behavior
# ---------------------------------------------------------------------------


class TestRegressionDefaultParams:
    def test_default_registry_built_when_all_hooks_none(self):
        """With all three new params None, _get_tool_executors is called (byte-identical path)."""
        with patch("agents_core.gw_agent._get_tool_executors") as mock_get_ex, \
             patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            _setup_doorman(mock_dc)
            mock_get_ex.return_value = {}
            mock_post.return_value.json.return_value = _mock_gw_stop("result")

            result = call_gw_agent(
                prompt="test",
                tool_executors=None,
                cancel_check=None,
                before_tool=None,
            )

        mock_get_ex.assert_called_once()
        assert result == "result"

    def test_no_hooks_normal_loop_concludes(self, tmp_path):
        """With all hooks at defaults, agent concludes normally (byte-identical behavior)."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_dc, \
             patch("requests.post") as mock_post:
            mock_doorman = _setup_doorman(mock_dc)
            mock_post.return_value.json.return_value = _mock_gw_stop("verdict text")

            result = call_gw_agent(prompt="analyze")

        assert result == "verdict text"
        mock_doorman.acquire.assert_called_once()
        mock_doorman.release.assert_called_once()
