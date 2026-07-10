"""Unit tests for gw_agent harness with mocked GW backend.

Tests the agent loop control, tool execution, doorman integration,
error recovery, and transcript generation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import (
    call_gw_agent,
    DEFAULT_READONLY_TOOLS,
    _default_reviewer_system_prompt,
    _force_conclusion,
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
        """Loop stops after max_steps, then attempts forced conclusion."""
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

            # Forced conclusion fails (empty response) so falls back to marker
            failed_conclusion = {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: failed_conclusion),
            ]

            result = call_gw_agent(
                prompt="Infinite loop test.",
                max_steps=3,
            )

            assert "max_steps reached" in result
            # Should have called GW 3 times (steps 1-3) + 1 (forced conclusion)
            assert mock_post.call_count == 4

    def test_no_progress_detection_nudge_and_break(self):
        """When same call repeats 4x, nudge once then break with forced conclusion."""
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

            # Forced conclusion succeeds
            verdict_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Analysis done.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            # Loop: 1, 2, 3 (nudge), 4 (break), then forced conclusion
            mock_post.side_effect = [
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: verdict_response),
            ]

            result = call_gw_agent(
                prompt="Test no-progress detection.",
                max_steps=10,  # More than enough to trigger the condition
            )

            # Should break on the 4th repeat (after nudge on 3rd) and use forced conclusion
            assert result == "Analysis done."
            # Calls: 1, 2, 3 (nudge), 4 (break + forced conclusion) = 5 calls total
            assert mock_post.call_count == 5

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
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # Create a test file in the temporary directory
            test_file_path = "test_content.txt"
            test_file_full_path = Path(tmpdir) / test_file_path
            test_file_full_path.write_text("This is test content.\n")

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

            # Use the temporary directory as cwd
            _, transcript = call_gw_agent(
                prompt="Test.",
                cwd=tmpdir,
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


class TestBackendURLAndAcquireLease:
    def test_default_behavior_unchanged_with_acquire_lease_true(self):
        """With defaults (acquire_lease=True, backend_url=None), behavior is byte-identical."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

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
            # Verify acquire/release were called
            mock_doorman.acquire.assert_called_once()
            mock_doorman.release.assert_called_once()
            # Verify POST was to GW_URL (the default backend)
            from agents_core.gw_agent import GW_URL
            called_url = mock_post.call_args[0][0]
            assert called_url == f"{GW_URL}/v1/chat/completions"

    def test_backend_url_changes_post_endpoint(self):
        """With backend_url set, POST goes to the custom URL instead of GW_URL."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            mock_gw_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Swarm answer.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }
            mock_post.return_value.json.return_value = mock_gw_response

            custom_url = "http://swarm:8081"
            result = call_gw_agent(
                prompt="Review on swarm.",
                backend_url=custom_url,
                timeout=10,
            )

            assert result == "Swarm answer."
            # Verify POST was to the custom URL
            called_url = mock_post.call_args[0][0]
            assert called_url == f"{custom_url}/v1/chat/completions"

    def test_acquire_lease_false_skips_doorman(self):
        """With acquire_lease=False, doorman.acquire is never called."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman

            mock_gw_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Swarm answer.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }
            mock_post.return_value.json.return_value = mock_gw_response

            result = call_gw_agent(
                prompt="Review on swarm.",
                acquire_lease=False,
                timeout=10,
            )

            assert result == "Swarm answer."
            # Verify acquire was NOT called
            mock_doorman.acquire.assert_not_called()
            # Verify release was NOT called
            mock_doorman.release.assert_not_called()

    def test_acquire_lease_false_and_backend_url_together(self):
        """With both acquire_lease=False and backend_url set, skip doorman and use custom URL."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman

            mock_gw_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Swarm grounding answer.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }
            mock_post.return_value.json.return_value = mock_gw_response

            custom_url = "http://swarm:8081"
            result = call_gw_agent(
                prompt="Grounding on swarm.",
                backend_url=custom_url,
                acquire_lease=False,
                timeout=10,
            )

            assert result == "Swarm grounding answer."
            # Verify doorman was not touched
            mock_doorman.acquire.assert_not_called()
            mock_doorman.release.assert_not_called()
            # Verify POST was to custom URL
            called_url = mock_post.call_args[0][0]
            assert called_url == f"{custom_url}/v1/chat/completions"


class TestForcedConclusion:
    def test_max_steps_exhaustion_concludes_with_forced_turn(self):
        """When max_steps reached, forced conclusion emits a parseable verdict."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # GW always returns tool_calls (never concludes)
            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            # Forced conclusion returns a verdict
            verdict_response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict": "good"}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            # First max_steps calls return infinite_response, then forced conclusion returns verdict
            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            result = call_gw_agent(
                prompt="Infinite loop test.",
                max_steps=3,
            )

            # Should return the forced verdict, not the exhaustion marker
            assert '{"verdict": "good"}' in result
            assert "max_steps reached — no verdict" not in result
            # Should have called GW 3 times (loop) + 1 (forced conclusion)
            assert mock_post.call_count == 4

    def test_forced_conclusion_omits_tools_field(self):
        """Forced conclusion POST must omit tools field entirely."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            verdict_response = {
                "choices": [
                    {
                        "message": {
                            "content": "The analysis is complete.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            call_gw_agent(
                prompt="Test.",
                max_steps=2,
            )

            # The last POST (forced conclusion) must not have a tools field
            last_call = mock_post.call_args_list[-1]
            posted_json = last_call[1]["json"]
            assert "tools" not in posted_json

    def test_forced_conclusion_negative_constraints_forbid_tool_calls(self):
        """Forced conclusion user message explicitly forbids tool calls."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            verdict_response = {
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
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            call_gw_agent(
                prompt="Test.",
                max_steps=1,
            )

            # Check that the last POST's user message contains explicit no-tool constraints
            last_call = mock_post.call_args_list[-1]
            posted_json = last_call[1]["json"]
            messages = posted_json["messages"]
            user_msg = messages[-1]  # Should be the conclusion user message
            assert user_msg["role"] == "user"
            content = user_msg["content"]
            assert "may NOT call any tools" in content
            assert "MUST NOT emit a tool call" in content

    def test_forced_conclusion_with_json_mode(self):
        """Forced conclusion with json_mode=True includes JSON instruction."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            verdict_response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict": "analyzed"}',
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            call_gw_agent(
                prompt="Test.",
                json_mode=True,
                max_steps=1,
            )

            # Check that the forced conclusion user message mentions JSON
            last_call = mock_post.call_args_list[-1]
            posted_json = last_call[1]["json"]
            messages = posted_json["messages"]
            user_msg = messages[-1]
            content = user_msg["content"]
            assert "JSON" in content

    def test_forced_conclusion_rejects_leaked_tool_calls(self):
        """If forced conclusion response has tool_calls, it's rejected (leaked tool loop)."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            # Forced conclusion leaks a tool_call
            leaked_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating more...",
                            "tool_calls": [
                                {
                                    "id": "call_y",
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
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: leaked_response),
            ]

            result = call_gw_agent(
                prompt="Test.",
                max_steps=1,
            )

            # Should fall back to exhaustion marker since forced conclusion was rejected
            assert "max_steps reached" in result

    def test_forced_conclusion_fallback_on_empty_response(self):
        """If forced conclusion returns empty or fails, fall back to exhaustion marker."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            # Forced conclusion fails (empty response)
            failed_response = {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: failed_response),
            ]

            result = call_gw_agent(
                prompt="Test.",
                max_steps=1,
            )

            # Should fall back to exhaustion marker
            assert "max_steps reached" in result

    def test_repeated_call_break_with_forced_conclusion(self):
        """Repeated call break path uses forced conclusion and cleans trailing tool_calls."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # Same tool call every time (to trigger repeated-call detection)
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

            # Forced conclusion returns a verdict
            verdict_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Analysis complete.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            # 1st call (grep), 2nd call (grep), 3rd call (grep + nudge), 4th call (grep + break), forced conclusion
            mock_post.side_effect = [
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: repeated_response),
                MagicMock(json=lambda: verdict_response),
            ]

            result = call_gw_agent(
                prompt="Test no-progress detection.",
                max_steps=10,
            )

            # Should have forced conclusion verdict, not exhaustion marker
            assert result == "Analysis complete."
            # Verify that forced conclusion was called (4 repeated + 1 forced conclusion = 5 POST calls)
            assert mock_post.call_count == 5

    def test_forced_conclusion_not_recorded_in_transcript(self):
        """Forced conclusion turn is not recorded in the transcript."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            verdict_response = {
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
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            _, transcript = call_gw_agent(
                prompt="Test.",
                max_steps=1,
                return_transcript=True,
            )

            # Transcript should only have the tool execution (read_file), not the forced conclusion
            assert len(transcript) == 1
            assert transcript[0]["tool_name"] == "read_file"

    def test_forced_conclusion_disables_think(self):
        """Forced conclusion turn disables thinking (enable_thinking: False)."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "Investigating...",
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

            verdict_response = {
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
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: verdict_response),
            ]

            call_gw_agent(
                prompt="Test.",
                think=True,  # Request thinking in the main loop
                max_steps=1,
            )

            # The forced conclusion POST must have enable_thinking: False
            last_call = mock_post.call_args_list[-1]
            posted_json = last_call[1]["json"]
            chat_template_kwargs = posted_json.get("chat_template_kwargs", {})
            assert chat_template_kwargs.get("enable_thinking") is False

    def test_default_max_steps_is_24(self):
        """Default max_steps parameter is 24."""
        from inspect import signature
        sig = signature(call_gw_agent)
        assert sig.parameters["max_steps"].default == 24

    def test_clean_path_untouched(self):
        """Clean path (agent concludes mid-budget) still works without forced turn."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # Agent concludes immediately
            clean_response = {
                "choices": [
                    {
                        "message": {
                            "content": "The code is good.",
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            mock_post.return_value.json.return_value = clean_response

            result = call_gw_agent(
                prompt="Review this.",
                max_steps=10,
            )

            # Should return the conclusion without any exhaustion marker
            assert result == "The code is good."
            # Should only have called GW once (no forced conclusion turn needed)
            assert mock_post.call_count == 1


class TestReasonOut:
    """Tests for the reason_out side-channel (agents-core-gw-agent-empty-result-reason-v0).

    reason_out is only populated on the writeable=False (readonly/json_mode) leg. A caller
    that never passes it sees zero behavior change (side-channel convention, see llm.py's
    _provenance_out for prior art).
    """

    def test_reason_out_gw_unreachable(self):
        """Doorman unreachable populates reason_out=["gw_unreachable"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            from agents_core.doorman_client import DoormanUnreachable

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.side_effect = DoormanUnreachable("connection failed")

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                on_wake_fail="skip",
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["gw_unreachable"]

    def test_reason_out_gw_not_serving(self):
        """Doorman reachable but not serving populates reason_out=["gw_not_serving"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "wake_failed"}

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                on_wake_fail="skip",
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["gw_not_serving"]

    def test_reason_out_request_failed(self):
        """A request exception talking to GW populates reason_out=["request_failed"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.side_effect = Exception("GW connection failed")

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["request_failed"]

    def test_reason_out_no_choices(self):
        """GW responding with no choices populates reason_out=["no_choices"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = {
                "choices": [],
                "usage": {"total_tokens": 10},
            }

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["no_choices"]

    def test_reason_out_voluntary_stop_empty_content(self):
        """Plain voluntary stop (non-json_mode, non-writeable) with empty content
        populates reason_out=["no_choices"] - the closest existing category, reused
        the same way the json_mode re-emit fallback reuses it for its analogous case.
        """
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
            }

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["no_choices"]

    def test_reason_out_stays_empty_on_success(self):
        """A genuinely successful call leaves reason_out == [] unchanged."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "All good.", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                reason_out=reason_out,
            )

            assert result == "All good."
            assert reason_out == []

    def test_reason_out_omitted_backward_compatible(self):
        """Calling call_gw_agent without reason_out at all behaves identically to today."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "All good.", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 100},
            }

            # No TypeError, no return-shape change, when reason_out is not passed at all.
            result = call_gw_agent(prompt="Review this.")

            assert result == "All good."

    def test_reason_out_interrupted(self):
        """Interrupted run populates reason_out=["interrupted"] despite the non-empty marker text."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                cancel_check=lambda: True,
                reason_out=reason_out,
            )

            # Underlying content was empty, but the final text carries a synthesized marker.
            assert result is not None
            assert "interrupted" in result
            assert reason_out == ["interrupted"]
            mock_post.assert_not_called()

    def test_reason_out_max_steps_exhausted(self):
        """Max-steps exhaustion with an empty forced conclusion populates reason_out=["max_steps_exhausted"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            # GW always returns a tool_call with empty content (never concludes).
            infinite_response = {
                "choices": [
                    {
                        "message": {
                            "content": "",
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
            # Forced conclusion also comes back empty.
            failed_conclusion = {
                "choices": [
                    {
                        "message": {"content": "", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 150},
            }

            mock_post.side_effect = [
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: infinite_response),
                MagicMock(json=lambda: failed_conclusion),
            ]

            reason_out = []
            result = call_gw_agent(
                prompt="Infinite loop test.",
                max_steps=3,
                reason_out=reason_out,
            )

            # Final text carries the bare "no verdict reached" marker (underlying content was empty).
            assert "max_steps reached" in result
            assert "no verdict reached" in result
            assert reason_out == ["max_steps_exhausted"]

    def test_reason_out_budget_exhausted(self, tmp_path):
        """Budget-forced conclusion whose forced content itself is empty populates reason_out=["budget_exhausted"]."""
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 250.0]), \
             patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.DoormanClient") as mock_doorman_class:
            mock_doorman_class.return_value = MagicMock()

            empty_stop = MagicMock()
            empty_stop.json.return_value = {
                "choices": [
                    {
                        "message": {"content": "", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 10},
            }
            mock_post.return_value = empty_stop

            reason_out = []
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(tmp_path),
                writeable=False,
                acquire_lease=False,
                backend_url="http://gw-test:8081",
                max_steps=5,
                timeout=300,
                reason_out=reason_out,
            )

            # Final text carries the bare budget-forced suffix (underlying content was empty).
            assert result is not None
            assert "budget-forced conclusion" in result
            assert reason_out == ["budget_exhausted"]

    def test_reason_out_grounding_failed(self):
        """Grounding guard's second ungrounded stop populates reason_out=["grounding_failed"]."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}

            ungrounded_stop = {
                "choices": [
                    {
                        "message": {"content": "no tools used", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 50},
            }
            mock_post.side_effect = [
                MagicMock(json=lambda: ungrounded_stop),
                MagicMock(json=lambda: ungrounded_stop),
            ]

            reason_out = []
            result = call_gw_agent(
                prompt="Review this.",
                json_mode=True,
                reason_out=reason_out,
            )

            assert result is None
            assert reason_out == ["grounding_failed"]

    def test_reason_out_writeable_true_unaffected(self, tmp_path):
        """writeable=True calls leave reason_out unpopulated and still return a FixerResult,
        regardless of which underlying empty-collapse cause fires (scope boundary)."""
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class:
            from agents_core.doorman_client import DoormanUnreachable

            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.side_effect = DoormanUnreachable("connection failed")

            reason_out = []
            result = call_gw_agent(
                prompt="Fix this.",
                cwd=str(tmp_path),
                writeable=True,
                on_wake_fail="skip",
                reason_out=reason_out,
            )

            fixer, _transcript = result
            assert isinstance(fixer, dict)
            assert fixer.get("concluded") is False
            assert reason_out == []
