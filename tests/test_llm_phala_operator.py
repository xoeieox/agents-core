"""Tests for the "phala" call_operator() class (agents-core-phala-gate-voicing-v0).

Covers DoD 1/2/6/7 from the spec:
  1. call_operator("phala", ...) returns a completion from the default model
     with no DoormanClient interaction.
  2. A locality row with cost_class="paid-phala-tee" is written per call.
  6. content:null + populated reasoning_content yields the reasoning text.
  7. An unreachable Phala endpoint fails closed — no paid Anthropic fallback.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

from agents_core.llm import (
    call_operator,
    OPERATOR_DEFAULTS,
    OperatorUnreachableError,
    PhalaOperatorUnavailable,
    PHALA_URL,
)


def _fake_resp(content=None, reasoning_content=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": content, "reasoning_content": reasoning_content}}]
    }
    return resp


def test_phala_in_operator_defaults():
    assert "phala" in OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["phala"] == "deepseek/deepseek-v4-flash-0731"


def test_phala_returns_completion_default_model():
    """DoD 1: call_operator("phala", ...) returns a completion from the default
    model, and posts to PHALA_URL with that model."""
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        captured["url"] = url
        captured["payload"] = json
        return _fake_resp(content="sealed reply")

    with patch("agents_core.llm.requests.post", side_effect=fake_post):
        result = call_operator("phala", prompt="hi")

    assert result == "sealed reply"
    assert captured["url"].startswith(PHALA_URL)
    assert captured["payload"]["model"] == "deepseek/deepseek-v4-flash-0731"


def test_phala_no_doorman_client_interaction():
    """DoD 1: no DoormanClient is constructed or touched for a phala call."""
    with patch("agents_core.llm.requests.post",
               return_value=_fake_resp(content="ok")), \
         patch("agents_core.doorman_client.DoormanClient") as mock_dc:
        call_operator("phala", prompt="hi")

    mock_dc.assert_not_called()


def test_phala_model_override_passes_through():
    """Deliberate divergence from quest/gravitywell: model= is accepted and forwarded,
    since Phala fronts a swappable catalog rather than a single pinned weight set."""
    captured = {}

    def fake_post(url, json=None, timeout=None, **kw):
        captured["payload"] = json
        return _fake_resp(content="ok")

    with patch("agents_core.llm.requests.post", side_effect=fake_post):
        result = call_operator("phala", prompt="hi", model="qwen/qwen3.5-122b-a10b")

    assert result == "ok"
    assert captured["payload"]["model"] == "qwen/qwen3.5-122b-a10b"


def test_phala_locality_row_cost_class(tmp_path, monkeypatch):
    """DoD 2: a locality row with cost_class="paid-phala-tee" is written for
    every phala call."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))

    with patch("agents_core.llm.requests.post",
               return_value=_fake_resp(content="ok")):
        call_operator("phala", prompt="hi")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) == 1
    import json as _json
    entry = _json.loads(lines[0])
    assert entry["cost_class"] == "paid-phala-tee"
    assert entry["seam"] == "call_operator"
    assert entry["requested_operator"] == "phala"


def test_phala_reasoning_content_fallback_when_content_null():
    """DoD 6: content:null + populated reasoning_content yields the reasoning
    text, not an empty string (openai/gpt-oss-120b, qwen/qwen3.5-122b-a10b shape,
    verified live 2026-08-04)."""
    with patch("agents_core.llm.requests.post",
               return_value=_fake_resp(content=None, reasoning_content="the reasoning text")):
        result = call_operator("phala", prompt="hi")

    assert result == "the reasoning text"


def test_phala_unreachable_fails_closed_no_paid_fallback():
    """DoD 7: an unreachable Phala endpoint raises PhalaOperatorUnavailable and
    never falls back to a paid Anthropic operator, even implicitly."""
    from requests.exceptions import ConnectionError as ReqConnError

    with patch("agents_core.llm.requests.post", side_effect=ReqConnError("refused")), \
         patch("time.sleep"), \
         patch("agents_core.claude_queue_sync.submit_and_wait") as mock_saw:
        with pytest.raises(PhalaOperatorUnavailable) as exc_info:
            call_operator("phala", prompt="hi")

    assert isinstance(exc_info.value, OperatorUnreachableError)
    mock_saw.assert_not_called()


def test_phala_unreachable_locality_row_marks_not_ok(tmp_path, monkeypatch):
    """A failed phala call still writes a locality row (served_model=None, ok=False),
    not silence."""
    from requests.exceptions import ConnectionError as ReqConnError

    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))

    with patch("agents_core.llm.requests.post", side_effect=ReqConnError("refused")), \
         patch("time.sleep"):
        with pytest.raises(PhalaOperatorUnavailable):
            call_operator("phala", prompt="hi")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    import json as _json
    entry = _json.loads(files[0].read_text().strip().splitlines()[0])
    assert entry["ok"] is False
    assert entry["served_model"] is None
    assert entry["cost_class"] == "paid-phala-tee"
