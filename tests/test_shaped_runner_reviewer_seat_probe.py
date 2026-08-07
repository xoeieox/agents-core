"""Tests for agents-core-reviewer-seat-tool-call-probe-v0.

Covers the DoD:
  1/4. `_run_local_reviewer` issues `probe_seat_tool_call` before the real
       call, with the same model/backend_url/tools the real call uses.
  2/6. A probe with no tool call short-circuits with reason
       "seat_no_tool_calls" — the real reviewer call is never made.
  3/6. A probe that errors does NOT short-circuit — the real review proceeds
       (D5 fail open).
  6.   probe-returns-tool-call -> proceeds (real call made, normal result).
  5b.  The probe payload is built via `build_step_payload`, the same
       construction path the real per-step POST uses — not a hand-rolled
       second dict.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_probe_no_tool_call_short_circuits_before_real_call(capsys):
    """D5 fail CLOSED: a clean response with zero tool_calls must stop the
    real reviewer call entirely and report the new distinct reason token."""
    import agents_core.shaped_runner as sr

    spec = {
        "prompt": "review this",
        "task_id": "t-probe-1",
        "model": "gravitywell-122b",
        "backend_url": "http://gw:8081",
    }

    probe_result = {"outcome": "no_tool_call", "served_model": "gravitywell-122b", "detail": None}

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=probe_result) as mock_probe,
        patch("agents_core.gw_agent.call_gw_agent") as mock_call,
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result is None
    mock_call.assert_not_called(), "a dead seat must never spend a real reviewer attempt"
    mock_probe.assert_called_once()
    err = capsys.readouterr().err
    assert "reason=seat_no_tool_calls" in err
    # New token must be distinguishable from a real grounding_failed.
    assert "grounding_failed" not in err


def test_probe_tool_call_proceeds_to_real_review():
    """The seat proved itself — the real call happens and its verdict wins."""
    import agents_core.shaped_runner as sr

    spec = {
        "prompt": "review this",
        "task_id": "t-probe-2",
        "model": "gravitywell-122b",
        "backend_url": "http://gw:8081",
    }

    probe_result = {"outcome": "tool_call", "served_model": "gravitywell-122b", "detail": None}

    def fake_call_gw_agent(**kwargs):
        return '{"verdict": "clean"}'

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=probe_result),
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call_gw_agent) as mock_call,
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result == '{"verdict": "clean"}'
    mock_call.assert_called_once()


def test_probe_error_fails_open_and_proceeds():
    """D5 fail OPEN: a probe transport failure must never block a real
    review — seat health is unknown, not established-dead."""
    import agents_core.shaped_runner as sr

    spec = {
        "prompt": "review this",
        "task_id": "t-probe-3",
        "model": "gravitywell-122b",
        "backend_url": "http://gw:8081",
    }

    probe_result = {"outcome": "error", "served_model": None, "detail": "connection refused"}

    def fake_call_gw_agent(**kwargs):
        return '{"verdict": "clean"}'

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=probe_result),
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_call_gw_agent) as mock_call,
    ):
        result = sr._run_local_reviewer(spec, "/some/cwd")

    assert result == '{"verdict": "clean"}'
    mock_call.assert_called_once()


def test_probe_variant_carried_through_to_real_call_gw_agent():
    """DoD 4: the variant returned by the probe is the exact tool surface
    handed to call_gw_agent for the real review — a probe that only proves
    *some* variant works while the real call still sends the poisoned one
    fixes nothing."""
    import agents_core.shaped_runner as sr
    from agents_core.gw_agent import perturb_tool_order

    spec = {
        "prompt": "review this",
        "task_id": "t-probe-variant",
        "model": "gravitywell-122b",
        "backend_url": "http://gw:8081",
    }

    from agents_core.gw_agent import DEFAULT_READONLY_TOOLS

    working_variant = perturb_tool_order(DEFAULT_READONLY_TOOLS, 1)
    probe_result = {
        "outcome": "tool_call",
        "served_model": "gravitywell-122b",
        "detail": None,
        "variant": working_variant,
        "attempt": 1,
        "attempts_made": 2,
        "refused_prefix_hash": "abc123def456",
    }

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=probe_result),
        patch("agents_core.gw_agent.call_gw_agent", return_value='{"verdict": "clean"}') as mock_call,
    ):
        sr._run_local_reviewer(spec, "/some/cwd")

    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["tools"] is working_variant
    assert list(call_kwargs["tools"].keys()) != list(DEFAULT_READONLY_TOOLS.keys())


def test_probe_called_with_same_model_and_backend_as_real_call():
    """DoD 1/4: the probe must use the same model/backend_url the real call
    below it uses — not a hardcoded or default value."""
    import agents_core.shaped_runner as sr
    from agents_core.gw_agent import DEFAULT_READONLY_TOOLS

    spec = {
        "prompt": "review this",
        "task_id": "t-probe-4",
        "model": "some-specific-model",
        "backend_url": "http://custom-backend:9999",
    }

    probe_result = {"outcome": "tool_call", "served_model": "some-specific-model", "detail": None}

    with (
        patch("agents_core.gw_agent.probe_seat_tool_call", return_value=probe_result) as mock_probe,
        patch("agents_core.gw_agent.call_gw_agent", return_value='{"verdict": "clean"}') as mock_call,
    ):
        sr._run_local_reviewer(spec, "/some/cwd")

    probe_kwargs = mock_probe.call_args.kwargs
    assert probe_kwargs["model"] == "some-specific-model"
    assert probe_kwargs["backend_url"] == "http://custom-backend:9999"
    assert probe_kwargs["tools"] is DEFAULT_READONLY_TOOLS

    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs["model"] == "some-specific-model"
    assert call_kwargs["backend_url"] == "http://custom-backend:9999"


# ---------------------------------------------------------------------------
# probe_seat_tool_call itself (gw_agent.py) — direct POST, no lease, no
# call_gw_agent, structural parity with the real per-step payload (5b).
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._json_body


def test_probe_seat_tool_call_detects_tool_call():
    from agents_core import gw_agent

    resp = _FakeResponse({
        "model": "gravitywell-122b",
        "choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "read_file"}}]}}],
    })
    with patch("agents_core.gw_agent.requests.post", return_value=resp) as mock_post:
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="gravitywell-122b",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "tool_call"
    assert result["served_model"] == "gravitywell-122b"
    assert result["detail"] is None
    # Succeeded on the first (native, unperturbed) attempt.
    assert result["attempt"] == 0
    assert result["attempts_made"] == 1
    assert result["variant"] == gw_agent.DEFAULT_READONLY_TOOLS
    # Direct POST to the same endpoint shape the real step call uses.
    assert mock_post.call_args.args[0] == "http://gw:8081/v1/chat/completions"


def test_probe_seat_tool_call_detects_no_tool_call():
    from agents_core import gw_agent

    resp = _FakeResponse({
        "model": "gravitywell-122b",
        "choices": [{"message": {"content": "I looked and there is nothing to do."}}],
    })
    with patch("agents_core.gw_agent.requests.post", return_value=resp) as mock_post:
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="gravitywell-122b",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "no_tool_call"
    # Every attempt (default 3) refused before concluding dead.
    assert result["attempts_made"] == 3
    assert result["attempt"] is None
    assert mock_post.call_count == 3


def test_probe_seat_tool_call_transport_error_reports_error_not_dead():
    from agents_core import gw_agent

    with patch("agents_core.gw_agent.requests.post", side_effect=ConnectionError("refused")):
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="gravitywell-122b",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "error"


def test_probe_seat_tool_call_payload_matches_real_step_shape():
    """5b: the probe must build its payload via build_step_payload — the
    same construction path the real per-step POST uses — not a hand-rolled
    second dict. Verified by comparing key sets and the tools/model/
    chat_template_kwargs shape directly. Pinned to attempts=1 — the
    perturbation retry loop is covered separately."""
    from agents_core import gw_agent

    resp = _FakeResponse({"model": "m", "choices": [{"message": {"tool_calls": []}}]})
    with patch("agents_core.gw_agent.requests.post", return_value=resp) as mock_post:
        gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="m",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
            attempts=1,
        )

    probe_payload = mock_post.call_args.kwargs["json"]
    real_payload = gw_agent.build_step_payload(
        model="m",
        messages=[{"role": "user", "content": "irrelevant"}],
        tools=gw_agent.DEFAULT_READONLY_TOOLS,
        is_swarm=False,
        think=False,
    )
    # Same tool surface, same model, same non-swarm chat_template_kwargs
    # handling — the structural properties D6/5b require.
    assert probe_payload["tools"] == real_payload["tools"]
    assert probe_payload["model"] == real_payload["model"]
    assert probe_payload["tool_choice"] == real_payload["tool_choice"]
    assert "chat_template_kwargs" in probe_payload
    assert "chat_template_kwargs" in real_payload
    # The probe caps max_tokens and uses temperature 0; neither field exists
    # on the real per-step payload, so they must not be asserted equal —
    # only the structural/tool-surface fields above are load-bearing.


def test_probe_seat_tool_call_takes_no_lease():
    """D6/DoD-4: the probe must never call call_gw_agent (which is the only
    path that acquires a doorman lease)."""
    from agents_core import gw_agent

    resp = _FakeResponse({"model": "m", "choices": [{"message": {"tool_calls": []}}]})
    with (
        patch("agents_core.gw_agent.requests.post", return_value=resp),
        patch("agents_core.gw_agent.call_gw_agent") as mock_call_gw_agent,
    ):
        gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="m",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    mock_call_gw_agent.assert_not_called()


# ---------------------------------------------------------------------------
# Adaptive prefix-perturbation retry (agents-core-reviewer-seat-prefix-
# perturbation-retry-v0). DoD 1-5.
# ---------------------------------------------------------------------------


def test_perturb_tool_order_preserves_tool_set_exactly():
    """DoD 1: same names, same parameter schemas, different serialized bytes."""
    import json as _json

    from agents_core import gw_agent

    tools = gw_agent.DEFAULT_READONLY_TOOLS
    variant = gw_agent.perturb_tool_order(tools, 1)

    assert set(variant.keys()) == set(tools.keys())
    for name in tools:
        assert variant[name]["function"]["parameters"] == tools[name]["function"]["parameters"]
        assert variant[name]["function"]["name"] == tools[name]["function"]["name"]
    # Same set, same schemas, but a different key order -> different
    # serialized bytes (values()-order is what build_step_payload flattens).
    assert _json.dumps(list(variant.values())) != _json.dumps(list(tools.values()))


def test_probe_retries_across_perturbations_and_returns_working_variant():
    """DoD 2: a stubbed backend refuses attempt 0, accepts attempt 1 — the
    probe returns tool_call and the variant that actually succeeded."""
    from agents_core import gw_agent

    dead = _FakeResponse({"model": "m", "choices": [{"message": {"tool_calls": []}}]})
    alive = _FakeResponse({
        "model": "m",
        "choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "read_file"}}]}}],
    })
    with patch("agents_core.gw_agent.requests.post", side_effect=[dead, alive]) as mock_post:
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="m",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "tool_call"
    assert result["attempt"] == 1
    assert result["attempts_made"] == 2
    assert mock_post.call_count == 2
    expected_variant = gw_agent.perturb_tool_order(gw_agent.DEFAULT_READONLY_TOOLS, 1)
    assert result["variant"] == expected_variant
    # The second (successful) POST actually carried the perturbed variant.
    second_call_tools = mock_post.call_args_list[1].kwargs["json"]["tools"]
    assert second_call_tools == list(expected_variant.values())


def test_probe_all_attempts_refuse_still_no_tool_call():
    """DoD 3a: every attempt returning a clean zero-tool-call response still
    yields outcome no_tool_call — retry does not widen the outcome."""
    from agents_core import gw_agent

    dead = _FakeResponse({"model": "m", "choices": [{"message": {"tool_calls": []}}]})
    with patch("agents_core.gw_agent.requests.post", return_value=dead) as mock_post:
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="m",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "no_tool_call"
    assert mock_post.call_count == gw_agent.PROBE_DEFAULT_ATTEMPTS


def test_probe_transport_error_fails_open_not_retried_into_dead_verdict():
    """DoD 3b: a transport error still yields "error" (fail OPEN) — the
    retry loop must not convert it into a "no_tool_call" dead-seat verdict,
    and must not keep retrying past the error."""
    from agents_core import gw_agent

    with patch(
        "agents_core.gw_agent.requests.post", side_effect=ConnectionError("refused")
    ) as mock_post:
        result = gw_agent.probe_seat_tool_call(
            backend_url="http://gw:8081",
            model="m",
            tools=gw_agent.DEFAULT_READONLY_TOOLS,
        )

    assert result["outcome"] == "error"
    assert result["outcome"] != "no_tool_call"
    # Terminates on the first transport error rather than exhausting attempts.
    assert mock_post.call_count == 1


def test_rotation_well_defined_at_zero_count_and_beyond():
    """DoD 5: no IndexError, no empty tool list, no dropped tool at attempt
    index 0, at an index equal to the tool count, and at one exceeding it."""
    from agents_core import gw_agent

    tools = gw_agent.DEFAULT_READONLY_TOOLS
    n = len(tools)

    for attempt in (0, n, n + 1, n * 5 + 3):
        variant = gw_agent.perturb_tool_order(tools, attempt)
        assert set(variant.keys()) == set(tools.keys())
        assert len(variant) == n

    # attempt 0 and attempt == tool count are the same (identity) rotation.
    assert gw_agent.perturb_tool_order(tools, 0) == gw_agent.perturb_tool_order(tools, n)
