"""Tests for inferred invariants (Qwen-mocked)."""
from unittest.mock import patch, MagicMock

import pytest

from agents_core.friction_test.critique import (
    InvariantResult,
    critique,
    infer_invariants,
)
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


def _make_obs(scenario_id: str) -> Observation:
    return Observation(
        scenario_id=scenario_id,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        http_calls=[{"method": "GET", "url": "/api/status", "status": 200, "latency_ms": 50.0}],
    )


def _mock_qwen_response(invs_json: str):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": invs_json}}]
    }
    return mock_resp


def test_infer_invariants_with_mocked_qwen():
    obs = [_make_obs("abc123456789")]
    qwen_json = '''[
        {"id": "inferred_01", "description": "latency stays under 7s",
         "dimension": "latency_ms", "expected": 7000, "tolerance": 500,
         "qwen_check": "Is the latency under 7000ms?"}
    ]'''

    with patch("httpx.post", return_value=_mock_qwen_response(qwen_json)):
        result = infer_invariants(obs, qwen_endpoint="http://fake-qwen/v1/chat/completions")

    assert len(result) >= 1
    inv = result[0]
    assert inv.invariant_id == "inferred_01"
    assert inv.proposed_by == "qwen"
    assert inv.provisional is True
    assert inv.check is None
    assert inv.qwen_check is not None


def test_infer_invariants_unreachable_returns_empty():
    """If Qwen is unreachable, infer_invariants returns [] without crashing."""
    obs = [_make_obs("abc123456789")]
    import httpx

    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        result = infer_invariants(obs, qwen_endpoint="http://127.0.0.1:19996/v1/chat/completions")

    assert result == []


def test_infer_invariants_empty_obs_returns_empty():
    result = infer_invariants([])
    assert result == []


def test_inferred_invariants_reported_separately():
    """Inferred invariants should carry provisional=True and proposed_by=qwen."""
    obs = [_make_obs("abc123456789")]
    qwen_json = '''[
        {"id": "inferred_02", "description": "error rate under 5%",
         "dimension": "error_rate", "expected": 0.05, "tolerance": 0.01,
         "qwen_check": "What is the error rate?"}
    ]'''

    with patch("httpx.post", return_value=_mock_qwen_response(qwen_json)):
        result = infer_invariants(obs, qwen_endpoint="http://fake-qwen/v1/chat/completions")

    assert all(inv.provisional is True for inv in result)
    assert all(inv.proposed_by == "qwen" for inv in result)


def test_inferred_dissonance_classifies_as_model_likely():
    """Dissonance against a provisional inferred invariant => dissonant_model_likely."""
    from agents_core.friction_test.critique import Invariant

    # Make a simple inferred invariant that will always dissonant
    def always_dissonant(s, o):
        return 9999.0, 9999.0  # big distance

    inv = Invariant(
        invariant_id="inferred_test",
        description="will always dissonant",
        dimension="latency_ms",
        expected=100,
        tolerance=10,
        proposed_by="qwen",
        provisional=True,
        check=always_dissonant,
    )

    sid = _make_scenario_id("radio-op", "test", {})
    scenario = Scenario(
        scenario_id=sid,
        target="radio-op",
        family_id="test",
        kind="happy",
        inputs={},
        expected_class="test",
    )
    obs = _make_obs(sid)

    results = critique([scenario], [obs], [inv], qwen_endpoint="http://127.0.0.1:19996")
    assert len(results) == 1
    r = results[0]
    assert r.status == "dissonant"
    assert r.classification == "dissonant_model_likely"
