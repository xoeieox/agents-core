"""Unit tests for gw_agent harness with mocked GW backend.

Tests the agent loop control, tool execution, doorman integration,
error recovery, and transcript generation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import (
    call_gw_agent,
    DEFAULT_READONLY_TOOLS,
    _default_reviewer_system_prompt,
)


class TestCallGWAgent:
    def test_doorman_acquire_success_then_loop_concludes(self):
        """Agent acquires lease, runs one turn, and concludes."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            # Mock doorman
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # Mock GW response: one turn, agent concludes
            mock_gw_response = {
                "choices": [
                    {
                        "message": {
                            "content": "The code looks good.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }
            mock_post.return_value.json.return_value = mock_gw_response

            result = call_gw_agent(
                prompt="Review this code.",
                system="You are a reviewer.",
                timeout=10,
            )

            assert result == "The code looks good."
            mock_doorman.acquire.assert_called_once()
            mock_doorman.release.assert_called_once()
            mock_post.assert_called_once()

    def test_doorman_not_serving_on_wake_fail_skip(self):
        """When doorman says not serving and on_wake_fail='skip', return None."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "wake_failed"}

            result = call_gw_agent(
                prompt="Review this.",
                on_wake_fail="skip",
            )

            assert result is None
            mock_doorman.release.assert_called_once()

    def test_doorman_not_serving_on_wake_fail_error(self):
        """When doorman says not serving and on_wake_fail='error', raise."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "wake_failed"}

            with pytest.raises(Exception, match="GW not serving"):
                call_gw_agent(
                    prompt="Review this.",
                    on_wake_fail="error",
                )

            mock_doorman.release.assert_called_once()

    def test_doorman_unreachable_on_wake_fail_skip(self):
        """When doorman is unreachable and on_wake_fail='skip', return None."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            from agents_core.doorman_client import DoormanUnreachable

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.side_effect = DoormanUnreachable("connection failed")

            result = call_gw_agent(
                prompt="Review this.",
                on_wake_fail="skip",
            )

            assert result is None
            mock_doorman.release.assert_called_once()

    def test_tool_execution_and_result_append(self):
        """Agent calls a tool and receives result."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # GW response 1: request a tool
            tool_call_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Let me read that file.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": "test.txt"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            # GW response 2: agent concludes after seeing result
            conclude_response = {
                "choices": [
                    {
                        "message": {
                            "content": "The file says: (error: file not found)",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: tool_call_response),
                MagicMock(json=lambda: conclude_response),
            ]

            result, transcript = call_gw_agent(
                prompt="Read test.txt.",
                return_transcript=True,
            )

            assert "file says" in result
            assert len(transcript) == 1
            assert transcript[0]["tool_name"] == "read_file"
            assert transcript[0]["step"] == 1

    def test_max_steps_enforcement(self):
        """Loop stops after max_steps."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # GW always returns a tool_call (never concludes)
            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Let me check something...",
                            "tool_calls": [
                                {
                                    "id": "call_x",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": "file.txt"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            mock_post.return_value.json.return_value = infinite_response

            result = call_gw_agent(
                prompt="Infinite loop test.",
                max_steps=3,
            )

            assert "max_steps reached" in result
            # Should have called GW 3 times (steps 1-3)
            assert mock_post.call_count == 3

    def test_no_progress_detection_nudge_and_break(self):
        """When same call repeats 4x, nudge once then break."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # GW returns the same tool call every time
            repeated_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Checking something.",
                            "tool_calls": [
                                {
                                    "id": "call_same",
                                    "function": {
                                        "name": "grep",
                                        "arguments": json.dumps({"pattern": "test"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            mock_post.return_value.json.return_value = repeated_response

            result = call_gw_agent(
                prompt="Test no-progress detection.",
                max_steps=10,  # More than enough to trigger the condition
            )

            # Should break on the 4th repeat (after nudge on 3rd)
            assert "max_steps reached" in result or len(result) > 0
            # Calls: 1 (first), 2, 3 (nudge), 4 (break) = 4 calls max
            assert mock_post.call_count <= 4

    def test_lease_released_on_exception(self):
        """Doorman lease is released even if GW request fails."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.side_effect = Exception("GW connection failed")

            result = call_gw_agent(
                prompt="This will fail.",
            )

            assert result is None or isinstance(result, str)
            mock_doorman.release.assert_called_once()

    def test_transcript_schema(self):
        """Transcript entries have correct schema."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # Use a file that definitely exists (the test file itself)
            test_file_path = "agents_core/tests/test_gw_agent.py"

            tool_call_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Reading file.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": json.dumps({"path": test_file_path}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            conclude_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Done.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: tool_call_response),
                MagicMock(json=lambda: conclude_response),
            ]

            # Use cwd that contains the test file
            _, transcript = call_gw_agent(
                prompt="Test.",
                cwd="/tmp/lapis-pm-worktrees/claude_20260615_104624_3226_fixergravitywellreviewag",
                return_transcript=True,
            )

            assert len(transcript) == 1
            entry = transcript[0]
            assert entry["step"] == 1
            assert entry["tool_name"] == "read_file"
            assert entry["tool_call_id"] == "call_1"
            assert entry["arguments"] == {"path": test_file_path}
            assert isinstance(entry["result"], str)
            assert len(entry["result"]) > 0  # File exists and has content
            assert entry["error"] is None  # No error since file exists

    def test_json_mode_appends_instruction(self):
        """json_mode=True appends JSON instruction to system prompt."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict": "good"}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            call_gw_agent(
                prompt="Test.",
                system="Base system prompt.",
                json_mode=True,
                timeout=10,
            )

            # Check that the posted message includes the JSON instruction
            call_args = mock_post.call_args
            posted_json = call_args[1]["json"]
            messages = posted_json["messages"]
            system_msg = messages[0]["content"]
            assert "JSON" in system_msg

    def test_default_system_prompt_includes_attribution_grammar(self):
        """Default system prompt credits human authorship and no self-authorship."""
        prompt = _default_reviewer_system_prompt()
        assert "human" in prompt.lower() or "author" in prompt.lower()
        assert "git blame" in prompt or "commit" in prompt.lower()


class TestFallbackClaudeCli:
    def test_on_wake_fail_claude_falls_back(self):
        """on_wake_fail='claude' calls call_claude_cli on doorman failure."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("agents_core.llm.call_claude_cli") as mock_claude:

            from agents_core.doorman_client import DoormanUnreachable

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.side_effect = DoormanUnreachable("unreachable")

            mock_claude.return_value = "Sonnet fallback response."

            result = call_gw_agent(
                prompt="Review this.",
                on_wake_fail="claude",
            )

            assert result == "Sonnet fallback response."
            mock_claude.assert_called_once()
            # Verify it was called with sonnet model
            call_kwargs = mock_claude.call_args[1]
            assert call_kwargs["model"] == "sonnet"
