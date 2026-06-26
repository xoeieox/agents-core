"""Tests for grammar-constrained verdict emission + grounding guard (gw-agent-grammar-constrained-verdict-emission-v0).

All tests are fully offline (no live GW, no network). Two helpers:
  _run_swarm()  — acquire_lease=False + backend_url set (_is_swarm=True); no response_format added.
  _run_gw()     — acquire_lease=True + backend_url=None (_is_swarm=False);  response_format IS added.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import call_gw_agent

_FAKE_BACKEND = "http://gw-test:8081"
_GW_URL_PATCH = "http://gw-gw:8081"


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


def _make_tool_call_response(tool_name: str, args: dict, call_id: str = "c1") -> dict:
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
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


def _mock_resp(data: dict) -> MagicMock:
    r = MagicMock(status_code=200)
    r.json = MagicMock(return_value=data)
    r.raise_for_status = MagicMock()
    return r


def _run_swarm(tmp_path, post_responses, json_mode=False, writeable=False,
               max_steps=10, timeout=300, verdict_schema=None):
    """Swarm path: _is_swarm=True (acquire_lease=False + backend_url set).
    No response_format added in this mode."""
    repo = _tmp_git_repo(tmp_path)
    posts = []
    monotonic_values = [float(i) for i in range(max_steps + 20)]

    def fake_post(url, json=None, timeout=None):
        posts.append({"url": url, "body": json, "timeout": timeout})
        return post_responses.pop(0)

    with patch("agents_core.gw_agent.time.monotonic", side_effect=monotonic_values), \
         patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
         patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
        MockDoorman.return_value = MagicMock()
        result = call_gw_agent(
            prompt="Review the spec.",
            cwd=str(repo),
            writeable=writeable,
            acquire_lease=False,
            backend_url=_FAKE_BACKEND,
            max_steps=max_steps,
            timeout=timeout,
            json_mode=json_mode,
            verdict_schema=verdict_schema,
        )
    return result, posts


def _run_gw(tmp_path, post_responses, json_mode=False, writeable=False,
            max_steps=10, timeout=300, verdict_schema=None):
    """GW path: _is_swarm=False (acquire_lease=True + backend_url=None).
    response_format IS added on json_mode/verdict_schema."""
    repo = _tmp_git_repo(tmp_path)
    posts = []
    monotonic_values = [float(i) for i in range(max_steps + 20)]

    def fake_post(url, json=None, timeout=None):
        posts.append({"url": url, "body": json, "timeout": timeout})
        return post_responses.pop(0)

    with patch("agents_core.gw_agent.time.monotonic", side_effect=monotonic_values), \
         patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
         patch("agents_core.gw_agent.GW_URL", _GW_URL_PATCH), \
         patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
        mock_client = MagicMock()
        mock_client.acquire.return_value = {"status": "serving"}
        MockDoorman.return_value = mock_client
        result = call_gw_agent(
            prompt="Review the spec.",
            cwd=str(repo),
            writeable=writeable,
            acquire_lease=True,
            backend_url=None,
            max_steps=max_steps,
            timeout=timeout,
            json_mode=json_mode,
            verdict_schema=verdict_schema,
        )
    return result, posts


# ---------------------------------------------------------------------------
# §1b: Voluntary-stop, unparseable content, json_mode → re-emit
# Note: grounding guard (§1c) takes priority; all §1b tests must establish
# at least one successful tool call before the voluntary stop.
# ---------------------------------------------------------------------------

class TestVoluntaryStopReemit:
    def test_prose_stop_reemits_and_returns_valid_json(self, tmp_path):
        """Grounded voluntary stop with prose content triggers one _force_conclusion re-emission."""
        # Step 0: successful tool call (grounding established).
        # Step 1: prose stop → §1b: not JSON → re-emit.
        # Step 2: fc_json from _force_conclusion → return.
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("here is my review: looks fine"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "clean", "issues": []}'))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, prose_stop, fc_json],
            json_mode=True,
        )

        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "clean"
        # 3 POSTs: tool step + prose stop + _force_conclusion re-emit.
        assert len(posts) == 3

    def test_already_parseable_stop_no_extra_turn(self, tmp_path):
        """Grounded voluntary stop with valid JSON is returned directly — no _force_conclusion POST."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        valid_json_stop = _mock_resp(_make_stop_response('{"verdict": "clean", "issues": []}'))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, valid_json_stop],
            json_mode=True,
        )

        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "clean"
        # 2 POSTs: tool step + stop (no re-emission).
        assert len(posts) == 2

    def test_fenced_json_no_extra_turn(self, tmp_path):
        """Fenced JSON (```json\\n{...}\\n```) is stripped and returned without re-emission."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        fenced = "```json\n{\"verdict\": \"clean\", \"issues\": []}\n```"
        fenced_stop = _mock_resp(_make_stop_response(fenced))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, fenced_stop],
            json_mode=True,
        )

        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "clean"
        # 2 POSTs: tool step + fenced stop (fence-strip + parse → no re-emission).
        assert len(posts) == 2

    def test_reemit_falls_back_to_original_when_force_conclusion_empty(self, tmp_path):
        """If _force_conclusion returns empty, original prose content is returned (never worse)."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("here is my review: looks fine"))
        post_responses = [tool_step, prose_stop]

        def spy_fc(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                   call_timeout=None, partial=False, verdict_schema=None, reason=None):
            return ""

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=list(range(30))), \
             patch("agents_core.gw_agent.requests.post",
                   side_effect=lambda url, **kw: post_responses.pop(0)), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_fc), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=10,
                timeout=300,
                json_mode=True,
            )

        # Falls back to original content rather than None.
        assert result == "here is my review: looks fine"


# ---------------------------------------------------------------------------
# §1b: Truthful framing on voluntary-stop re-emission
# ---------------------------------------------------------------------------

class TestTruthfulReemitFraming:
    def test_voluntary_stop_reemit_uses_reason_not_budget_preamble(self, tmp_path):
        """Voluntary-stop re-emission passes reason= and does NOT say 'investigation budget'."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("looks good to me"))
        post_responses = [tool_step, prose_stop]
        captured_args = []

        def spy_fc(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                   call_timeout=None, partial=False, verdict_schema=None, reason=None):
            captured_args.append({"reason": reason})
            return '{"verdict": "clean"}'

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=list(range(30))), \
             patch("agents_core.gw_agent.requests.post",
                   side_effect=lambda url, **kw: post_responses.pop(0)), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_fc), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=10,
                timeout=300,
                json_mode=True,
            )

        assert len(captured_args) == 1
        reason = captured_args[0]["reason"]
        assert reason is not None
        assert "investigation budget" not in reason
        assert "valid JSON verdict" in reason

    def test_exhaustion_path_still_uses_budget_preamble(self, tmp_path):
        """Exhaustion path calls _force_conclusion with reason=None (budget preamble unchanged)."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        captured_args = []

        def spy_fc(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                   call_timeout=None, partial=False, verdict_schema=None, reason=None):
            captured_args.append({"reason": reason})
            return '{"verdict": "clean"}'

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=list(range(30))), \
             patch("agents_core.gw_agent.requests.post", return_value=tool_step), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_fc), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=1,  # exhausts after 1 tool step
                timeout=300,
            )

        # Exhaustion _force_conclusion call has reason=None.
        assert len(captured_args) == 1
        assert captured_args[0]["reason"] is None


# ---------------------------------------------------------------------------
# §1a: response_format shape in the conclusion POST body (GW path, non-swarm)
# ---------------------------------------------------------------------------

class TestResponseFormatShape:
    def test_json_mode_no_schema_sends_json_object(self, tmp_path):
        """GW path, json_mode=True, no verdict_schema → response_format={"type":"json_object"}."""
        # Tool step (grounding), then prose stop → re-emit via _force_conclusion.
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("not json"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_gw(
            tmp_path,
            [tool_step, prose_stop, fc_json],
            json_mode=True,
        )

        # 3 POSTs: tool step + prose stop + _force_conclusion.
        assert len(posts) == 3
        fc_body = posts[2]["body"]
        assert "response_format" in fc_body
        assert fc_body["response_format"] == {"type": "json_object"}

    def test_verdict_schema_sends_json_schema(self, tmp_path):
        """GW path, verdict_schema provided → response_format={"type":"json_schema",...}."""
        schema = {"name": "verdict", "strict": True, "schema": {"type": "object"}}
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("not json"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_gw(
            tmp_path,
            [tool_step, prose_stop, fc_json],
            json_mode=True,
            verdict_schema=schema,
        )

        assert len(posts) == 3
        fc_body = posts[2]["body"]
        assert fc_body["response_format"] == {"type": "json_schema", "json_schema": schema}

    def test_no_json_mode_no_schema_no_response_format(self, tmp_path):
        """GW path, json_mode=False, no schema → no response_format key in any POST."""
        stop = _mock_resp(_make_stop_response("plain text verdict"))

        result, posts = _run_gw(
            tmp_path,
            [stop],
            json_mode=False,
        )

        for post in posts:
            assert "response_format" not in (post["body"] or {})

    def test_swarm_path_no_response_format(self, tmp_path):
        """Swarm path: no response_format in the re-emit POST regardless of json_mode."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        prose_stop = _mock_resp(_make_stop_response("not json"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, prose_stop, fc_json],
            json_mode=True,
        )

        # On swarm, no response_format in any POST.
        for post in posts:
            assert "response_format" not in (post["body"] or {})


# ---------------------------------------------------------------------------
# §1c: Grounding guard — nudge then UNFOUNDED
# ---------------------------------------------------------------------------

class TestGroundingGuard:
    def test_first_ungrounded_stop_nudges_and_loop_continues(self, tmp_path):
        """0 grounding + first stop → nudge appended, loop continues; tool call → stop accepted."""
        # Step 0 (step_num=0): stop with no prior tool calls → nudge fires, continue.
        # Step 1 (step_num=1): tool call → grounding_count=1.
        # Step 2 (step_num=2): valid JSON stop → accepted.
        ungrounded_stop = _mock_resp(_make_stop_response("no tools used"))
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c1"))
        grounded_json_stop = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_swarm(
            tmp_path,
            [ungrounded_stop, tool_step, grounded_json_stop],
            json_mode=True,
            max_steps=10,
        )

        # 3 POSTs: ungrounded stop (nudge) + tool step + final json stop.
        assert len(posts) == 3
        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "clean"

    def test_second_ungrounded_stop_returns_unfounded_error(self, tmp_path):
        """0 grounding + two consecutive stops → UNFOUNDED (None / not-concluded)."""
        ungrounded_stop1 = _mock_resp(_make_stop_response("no tools used first"))
        ungrounded_stop2 = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_swarm(
            tmp_path,
            [ungrounded_stop1, ungrounded_stop2],
            json_mode=True,
            max_steps=10,
        )

        # 2 POSTs: first stop (nudge) + second stop (UNFOUNDED).
        assert len(posts) == 2
        # Result is None (UNFOUNDED / not-concluded).
        assert result is None

    def test_errored_tool_call_counts_as_zero_grounding(self, tmp_path):
        """Tool call that returns an error counts as 0 grounding; grounding guard fires."""
        # Step 0: tool call to nonexistent path → error → grounding_count stays 0.
        # Step 1: stop → grounding guard: nudge fires, continue.
        # Step 2: successful tool call → grounding_count=1.
        # Step 3: valid JSON stop → accepted.
        error_tool = _mock_resp(_make_tool_call_response("read_file", {"path": "/nonexistent/x"}, "c1"))
        stop_after_error = _mock_resp(_make_stop_response("checked but errored"))
        good_tool = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c2"))
        grounded_json = _mock_resp(_make_stop_response('{"verdict": "fixable", "findings": []}'))

        result, posts = _run_swarm(
            tmp_path,
            [error_tool, stop_after_error, good_tool, grounded_json],
            json_mode=True,
            max_steps=10,
        )

        assert len(posts) == 4
        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "fixable"

    def test_grounding_guard_does_not_fire_after_successful_tool_call(self, tmp_path):
        """≥1 error-free tool call → no nudge; stop finalizes immediately per §1b."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        json_stop = _mock_resp(_make_stop_response('{"verdict": "clean", "issues": []}'))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, json_stop],
            json_mode=True,
        )

        # 2 POSTs: tool step + stop (no nudge).
        assert len(posts) == 2
        assert result is not None
        parsed = json.loads(result)
        assert parsed.get("verdict") == "clean"

    def test_grounding_guard_does_not_fire_for_non_json_mode(self, tmp_path):
        """json_mode=False: 0-tool-call stop is accepted with no nudge."""
        ungrounded_stop = _mock_resp(_make_stop_response("plain text verdict no tools"))

        result, posts = _run_swarm(
            tmp_path,
            [ungrounded_stop],
            json_mode=False,
            max_steps=5,
        )

        # Exactly 1 POST — no nudge.
        assert len(posts) == 1
        assert result == "plain text verdict no tools"

    def test_grounding_guard_does_not_fire_for_writeable(self, tmp_path):
        """writeable=True: 0-tool-call stop is accepted with no nudge (fixer runs unaffected)."""
        ungrounded_stop = _mock_resp(_make_stop_response("fixer output no tools"))

        result, posts = _run_swarm(
            tmp_path,
            [ungrounded_stop],
            json_mode=False,
            writeable=True,
            max_steps=5,
        )

        # 1 POST — no nudge.
        assert len(posts) == 1
        fixer, transcript = result
        assert fixer.get("concluded") is True


# ---------------------------------------------------------------------------
# §1d + regression: Non-json_mode + writeable unchanged
# ---------------------------------------------------------------------------

class TestNonJsonModeBytIdentical:
    def test_writeable_run_returns_fixer_result_no_json_reemit(self, tmp_path):
        """writeable=True run returns (FixerResult, transcript) with no JSON re-emission."""
        stop = _mock_resp(_make_stop_response("fixer result prose"))

        result, posts = _run_swarm(
            tmp_path,
            [stop],
            writeable=True,
            json_mode=False,
            max_steps=5,
        )

        fixer, transcript = result
        assert "final_diff" in fixer
        assert "steps" in fixer
        # Only 1 POST — no re-emission.
        assert len(posts) == 1

    def test_readonly_non_json_mode_returns_prose(self, tmp_path):
        """json_mode=False readonly: prose content returned as-is."""
        stop = _mock_resp(_make_stop_response("plain verdict text"))

        result, posts = _run_swarm(
            tmp_path,
            [stop],
            json_mode=False,
            max_steps=5,
        )

        assert result == "plain verdict text"
        assert len(posts) == 1


# ---------------------------------------------------------------------------
# §4: #87 regression guard — exhaustion and repeated-call paths unchanged
# ---------------------------------------------------------------------------

class TestPrior87Regression:
    def test_exhaustion_path_still_routes_through_force_conclusion(self, tmp_path):
        """max_steps exhaustion still calls _force_conclusion (the #87 path)."""
        tool_step = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "clean"}'))

        result, posts = _run_swarm(
            tmp_path,
            [tool_step, fc_json],
            json_mode=True,
            max_steps=1,
        )

        assert result is not None
        # 2 POSTs: tool step + _force_conclusion.
        assert len(posts) == 2

    def test_max_steps_default_is_24(self):
        """Default max_steps is 24."""
        import inspect
        sig = inspect.signature(call_gw_agent)
        assert sig.parameters["max_steps"].default == 24

    def test_leaked_tool_call_in_force_conclusion_rejected(self, tmp_path):
        """_force_conclusion still rejects a leaked tool-call response (the #87 guard)."""
        from agents_core.gw_agent import _force_conclusion

        messages = [{"role": "user", "content": "test"}]
        leaked = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }]
        }
        leaked_resp = MagicMock(status_code=200)
        leaked_resp.json = MagicMock(return_value=leaked)
        leaked_resp.raise_for_status = MagicMock()

        with patch("agents_core.gw_agent.requests.post", return_value=leaked_resp):
            content = _force_conclusion(
                messages, _FAKE_BACKEND, 30, True, None, is_swarm=False
            )

        assert content == ""
