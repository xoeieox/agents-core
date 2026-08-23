"""Tests for budget-aware forced conclusion in call_gw_agent (gw-agent-budget-aware-conclusion-v0).

All tests are fully offline (no live GW, no network). Clock is stubbed via
patch("agents_core.gw_agent.time.monotonic").

We use acquire_lease=False throughout (swarm-path mock pattern) to get a
predictable, minimal number of time.monotonic calls — exactly 1 (_loop_start)
+ 1 per loop iteration (_now). This mirrors test_gw_agent_fixer.py's swarm
tests and avoids mock-internal monotonic calls from DoormanClient.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import (
    _build_fixer_result,
    call_gw_agent,
)

_FAKE_BACKEND = "http://gw-test:8081"


# ---------------------------------------------------------------------------
# Helpers (mirrors test_gw_agent_fixer.py patterns)
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


def _run_readonly(tmp_path, monotonic_values, post_responses, max_steps=5, timeout=300):
    """Run call_gw_agent(writeable=False, acquire_lease=False) with stubbed clock."""
    repo = _tmp_git_repo(tmp_path)
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append({"timeout": timeout, "json": json})
        return post_responses.pop(0)

    with patch("agents_core.gw_agent.time.monotonic", side_effect=monotonic_values), \
         patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
         patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
        MockDoorman.return_value = MagicMock()
        result = call_gw_agent(
            prompt="Review.",
            cwd=str(repo),
            writeable=False,
            acquire_lease=False,
            backend_url=_FAKE_BACKEND,
            max_steps=max_steps,
            timeout=timeout,
        )
    return result, captured


# ---------------------------------------------------------------------------
# AC1: budget fires before max_steps and lands a verdict
# ---------------------------------------------------------------------------

class TestBudgetForcesConclusion:
    """Core regression: slow model hits wall-clock before max_steps; must get a verdict."""

    def test_budget_forces_conclusion_before_timeout(self, tmp_path):
        """Stubbed slow clock fires budget at step 3 (< max_steps=5); verdict returned, not error.

        monotonic call sequence (acquire_lease=False):
          call 1: _loop_start = 0.0            deadline = 300.0
          call 2: step0 _now = 0.0             avg=18  reserve=max(36,60)=60  300-0=300>60 → OK
          call 3: step1 _now = 50.0            avg=50  reserve=max(100,60)=100 300-50=250>100 → OK
          call 4: step2 _now = 265.0           avg=(50+215)/2=132.5 reserve=max(265,60)=265 300-265=35<265 → FIRE
        """
        monotonic_values = [0.0, 0.0, 50.0, 265.0]

        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        step1 = _mock_resp(_make_tool_call_response("grep", {"pattern": "init"}, "c1"))
        fc_stop = _mock_resp(_make_stop_response('{"verdict": "ok", "findings": []}'))

        result, captured = _run_readonly(
            tmp_path,
            list(monotonic_values),
            [step0, step1, fc_stop],
            max_steps=5, timeout=300,
        )

        # Non-None string result (not an error).
        assert result is not None
        assert isinstance(result, str)
        # Must carry the budget-forced suffix.
        assert "budget-forced conclusion" in result
        # Verdict content reached.
        assert "verdict" in result or "ok" in result
        # Exactly 3 POSTs: step0 tool, step1 tool, _force_conclusion (not 5 steps + fc).
        assert len(captured) == 3

    def test_budget_fires_immediately_when_already_past_deadline(self, tmp_path):
        """If clock starts near the deadline, budget fires on the very first step."""
        # timeout=300; loop_start=0; step0_now=250; reserve=max(36,60)=60; 300-250=50<60 → fire
        monotonic_values = [0.0, 250.0]

        fc_stop = _mock_resp(_make_stop_response('{"verdict": "early", "findings": []}'))

        def spy_force(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                      call_timeout=None, partial=False, served_model_out=None, model=None):
            return '{"verdict": "early", "findings": []}'

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=monotonic_values), \
             patch("agents_core.gw_agent.requests.post", return_value=fc_stop), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_force), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert result is not None
        assert "budget-forced conclusion" in result
        # Fire at step 1 (step_num=0, step_num+1=1).
        assert "step 1" in result

    def test_budget_forced_conclusion_instruction_is_partial(self, tmp_path):
        """_force_conclusion is called with partial=True so the model is told to narrate incompleteness."""
        repo = _tmp_git_repo(tmp_path)
        captured_partial = []

        def spy_force(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                      call_timeout=None, partial=False, served_model_out=None, model=None):
            captured_partial.append(partial)
            return '{"verdict": "partial ok", "findings": []}'

        # Fire at step1 (loop_start=0, step0_now=0 ok, step1_now=265 fires).
        # Needs one tool call response for step0, then budget fires.
        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))

        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 265.0]), \
             patch("agents_core.gw_agent.requests.post", return_value=step0), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_force), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert len(captured_partial) == 1
        assert captured_partial[0] is True


# ---------------------------------------------------------------------------
# AC2: _force_conclusion receives the remaining budget as call_timeout
# ---------------------------------------------------------------------------

class TestConclusionGetsRemainingBudget:
    def test_conclusion_gets_remaining_budget(self, tmp_path):
        """_force_conclusion call_timeout equals the remaining wall-clock at budget-fire time.

        At step1 _now=265: remaining=300-265=35; _fc_timeout=max(20,35)=35.
        """
        repo = _tmp_git_repo(tmp_path)
        captured = []

        def spy_force(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                      call_timeout=None, partial=False, served_model_out=None, model=None):
            captured.append(call_timeout)
            return '{"verdict": "ok", "findings": []}'

        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))

        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 265.0]), \
             patch("agents_core.gw_agent.requests.post", return_value=step0), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_force), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert len(captured) == 1
        # remaining = 300-265=35; _fc_timeout = max(20, 35) = 35.
        assert captured[0] == pytest.approx(35.0, abs=1.0)

    def test_conclusion_timeout_floored_at_20s_when_past_deadline(self, tmp_path):
        """When the deadline has already passed, _fc_timeout is still at least 20s."""
        repo = _tmp_git_repo(tmp_path)
        captured = []

        def spy_force(messages, backend_url, timeout, json_mode, log, is_swarm=False,
                      call_timeout=None, partial=False, served_model_out=None, model=None):
            captured.append(call_timeout)
            return "partial verdict"

        # Budget fires at step0 _now=305 (past the deadline: 300-305=-5 < 60=reserve).
        # _fc_timeout = max(20, 300-305) = max(20, -5) = 20.
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 305.0]), \
             patch("agents_core.gw_agent.requests.post", return_value=MagicMock()), \
             patch("agents_core.gw_agent._force_conclusion", side_effect=spy_force), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert len(captured) == 1
        assert captured[0] == pytest.approx(20.0, abs=0.1)


# ---------------------------------------------------------------------------
# AC3: per-step POST timeout is bounded by deadline - reserve, not full timeout
# ---------------------------------------------------------------------------

class TestPerStepCapBoundsSingleStep:
    def test_per_step_timeout_capped_by_reserve(self, tmp_path):
        """Step POST timeout = max(20, deadline - now - reserve), not the full `timeout`.

        timeout=300, t=0, avg_step_s=18 (seed):
          reserve = max(2*18, 0.20*300) = max(36, 60) = 60
          per_step_timeout = max(20, 300 - 0 - 60) = max(20, 240) = 240
        """
        stop_resp = _mock_resp(_make_stop_response("done"))
        captured = []

        def fake_post(url, json=None, timeout=None):
            captured.append(timeout)
            return stop_resp

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0]), \
             patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert len(captured) == 1
        # reserve=60, per_step=max(20, 300-0-60)=240
        assert captured[0] == pytest.approx(240.0, abs=1.0)

    def test_per_step_timeout_floored_at_20s(self, tmp_path):
        """When remaining budget barely exceeds reserve, step timeout floors at 20s.

        timeout=300; at step0 _now=0: remaining=300; reserve=60; per_step=240.
        But with a shorter timeout (timeout=50): reserve=max(36,10)=36;
        remaining=50; per_step=max(20, 50-0-36)=max(20,14)=20.
        """
        stop_resp = _mock_resp(_make_stop_response("done"))
        captured = []

        def fake_post(url, json=None, timeout=None):
            captured.append(timeout)
            return stop_resp

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0]), \
             patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5,
                timeout=50,  # short timeout: reserve=max(36, 10)=36; per_step=max(20,14)=20
            )

        assert len(captured) == 1
        assert captured[0] == pytest.approx(20.0, abs=1.0)


# ---------------------------------------------------------------------------
# AC4: existing max_steps path unchanged when deadline never binds
# ---------------------------------------------------------------------------

class TestMaxStepsPathStillWorks:
    def test_max_steps_path_fires_when_clock_is_fast(self, tmp_path):
        """Fast model (1s/step): deadline never binds; max_steps path runs as before.

        max_steps=2, timeout=300:
          call1: _loop_start=0
          call2: step0 _now=0  reserve=60; 300>60 → OK
          call3: step1 _now=1  avg=1; reserve=max(2,60)=60; 299>60 → OK
          (max_steps exhausted → _force_conclusion fires)
        """
        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        step1 = _mock_resp(_make_tool_call_response("grep", {"pattern": "init"}, "c1"))
        fc_stop = _mock_resp(_make_stop_response('{"verdict": "max_steps done", "findings": []}'))
        post_responses = [step0, step1, fc_stop]

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 1.0]), \
             patch("agents_core.gw_agent.requests.post", side_effect=lambda url, **kw: post_responses.pop(0)), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=2, timeout=300,
            )

        # Must return a non-None verdict.
        assert result is not None
        assert isinstance(result, str)
        # Must NOT contain the budget-forced suffix (this is max_steps path).
        assert "budget-forced conclusion" not in result
        # All 3 POSTs consumed.
        assert len(post_responses) == 0

    def test_max_steps_result_has_budget_forced_false(self):
        """max_steps forced-conclusion produces budget_forced=False in FixerResult."""
        with tempfile.TemporaryDirectory() as td:
            repo = _tmp_git_repo(Path(td))

            step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
            fc_stop = _mock_resp(_make_stop_response('{"verdict": "done"}'))
            post_responses = [step0, fc_stop]

            with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0]), \
                 patch("agents_core.gw_agent.requests.post", side_effect=lambda url, **kw: post_responses.pop(0)), \
                 patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
                MockDoorman.return_value = MagicMock()
                fixer, _ = call_gw_agent(
                    prompt="Fix.",
                    cwd=td,
                    writeable=True,
                    acquire_lease=False,
                    backend_url=_FAKE_BACKEND,
                    max_steps=1, timeout=300,
                )

        assert fixer.get("budget_forced") is False
        assert fixer.get("max_steps_reached") is False  # _force_conclusion succeeded


# ---------------------------------------------------------------------------
# AC5: budget_forced metadata is distinct from error; parseable verdict returned
# ---------------------------------------------------------------------------

class TestBudgetForcedMetadataNotError:
    def test_budget_forced_metadata_writeable(self):
        """Writeable path: budget_forced=True in FixerResult, concluded=False, not error-shaped."""
        with tempfile.TemporaryDirectory() as td:
            repo = _tmp_git_repo(Path(td))

            step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
            fc = _mock_resp(_make_stop_response('{"verdict": "partial", "findings": []}'))
            post_responses = [step0, fc]

            with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 265.0]), \
                 patch("agents_core.gw_agent.requests.post", side_effect=lambda url, **kw: post_responses.pop(0)), \
                 patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
                MockDoorman.return_value = MagicMock()
                fixer, transcript = call_gw_agent(
                    prompt="Fix.",
                    cwd=td,
                    writeable=True,
                    acquire_lease=False,
                    backend_url=_FAKE_BACKEND,
                    max_steps=5, timeout=300,
                )

        assert fixer.get("budget_forced") is True
        assert fixer.get("concluded") is False
        # Intact FixerResult shape (not an error blob).
        assert "final_diff" in fixer
        assert "steps" in fixer
        assert "last_test_outcome" in fixer

    def test_budget_forced_readonly_suffix_present(self, tmp_path):
        """Readonly path: result string contains the budget-forced suffix marker."""
        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        fc = _mock_resp(_make_stop_response('{"verdict": "partial", "findings": []}'))
        post_responses = [step0, fc]

        repo = _tmp_git_repo(tmp_path)
        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 265.0]), \
             patch("agents_core.gw_agent.requests.post", side_effect=lambda url, **kw: post_responses.pop(0)), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5, timeout=300,
            )

        assert result is not None
        assert "budget-forced conclusion" in result
        # Suffix includes step number and elapsed time.
        assert "elapsed" in result

    def test_budget_forced_result_has_budget_forced_field_false(self):
        """_build_fixer_result includes budget_forced=False by default."""
        with tempfile.TemporaryDirectory() as td:
            result = _build_fixer_result(td, [], concluded=True, budget_forced=False)
        assert "budget_forced" in result
        assert result["budget_forced"] is False

    def test_budget_forced_result_has_budget_forced_field_true(self):
        """_build_fixer_result propagates budget_forced=True and sets concluded=False."""
        with tempfile.TemporaryDirectory() as td:
            result = _build_fixer_result(td, [], concluded=False, budget_forced=True)
        assert result["budget_forced"] is True
        assert result["concluded"] is False

    def test_budget_forced_json_mode_result_is_valid_json(self, tmp_path):
        """json_mode=True + budget-forced: returned string must still be valid JSON.

        Regression for: budget_forced_suffix was appended as text to JSON content,
        causing json.loads() in spec_review.py:1671 to raise JSONDecodeError.

        Clock sequence (acquire_lease=False, timeout=300):
          call 1: _loop_start=0    deadline=300
          call 2: step0 _now=0     avg=18  reserve=max(36,60)=60  300>60 → OK
          call 3: step1 _now=265   avg=(0+265)/2  reserve≥60  300-265=35<reserve → FIRE
        """
        repo = _tmp_git_repo(tmp_path)

        step0 = _mock_resp(_make_tool_call_response("read_file", {"path": "README.md"}, "c0"))
        fc_json = _mock_resp(_make_stop_response('{"verdict": "partial", "findings": [], "partial_review_note": "ran 1 step"}'))
        post_responses = [step0, fc_json]

        def fake_post(url, json=None, timeout=None):
            return post_responses.pop(0)

        with patch("agents_core.gw_agent.time.monotonic", side_effect=[0.0, 0.0, 265.0]), \
             patch("agents_core.gw_agent.requests.post", side_effect=fake_post), \
             patch("agents_core.gw_agent.DoormanClient") as MockDoorman:
            MockDoorman.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.",
                cwd=str(repo),
                writeable=False,
                acquire_lease=False,
                backend_url=_FAKE_BACKEND,
                max_steps=5,
                timeout=300,
                json_mode=True,
            )

        assert result is not None
        # Must be parseable JSON — appending the text suffix would break this.
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        # Budget-forced info injected as JSON field, not trailing text.
        assert "_budget_forced" in parsed
        assert "budget-forced conclusion" in parsed["_budget_forced"]
        # Original verdict fields intact.
        assert parsed.get("verdict") == "partial"
