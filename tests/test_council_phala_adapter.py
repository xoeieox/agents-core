"""Tests for PhalaAdapter and phala voicing dispatch (agents-core-phala-gate-voicing-v0).

Covers: adapter chat() routing, message flattening, voicing_events provenance
(the explicit-banner mechanism), _build_adapter dispatch, and
_apply_voicing_provenance surfacing "phala:<model-id>" by name.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_phala_adapter_chat_routes_to_operator():
    """PhalaAdapter.chat() calls call_operator with correct args."""
    from agents_core.council.phala_adapter import PhalaAdapter

    with patch("agents_core.council.phala_adapter.call_operator",
               return_value="sealed response") as mock_op:
        adapter = PhalaAdapter(temperature=0.8)
        result = adapter.chat("ENTITY_CARD", [MagicMock(role="user", content="hi")])

    assert result == "sealed response"
    mock_op.assert_called_once()
    call_args, call_kwargs = mock_op.call_args
    assert call_args[0] == "phala"
    assert call_args[1] == "hi"
    assert call_kwargs["system"] == "ENTITY_CARD"
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["timeout"] == 300
    assert call_kwargs["model"] == "deepseek/deepseek-v4-flash-0731"


def test_phala_adapter_model_override():
    from agents_core.council.phala_adapter import PhalaAdapter

    with patch("agents_core.council.phala_adapter.call_operator",
               return_value="response") as mock_op:
        adapter = PhalaAdapter(model="qwen/qwen3.5-122b-a10b")
        adapter.chat("system", [MagicMock(role="user", content="test")])

    _, call_kwargs = mock_op.call_args
    assert call_kwargs["model"] == "qwen/qwen3.5-122b-a10b"


def test_phala_adapter_voicing_events_names_model():
    """Explicit banner (Erah 2026-08-04): voicing_events must name Phala AND
    the model actually requested, not fold into an anonymous success entry."""
    from agents_core.council.phala_adapter import PhalaAdapter

    with patch("agents_core.council.phala_adapter.call_operator",
               return_value="response"):
        adapter = PhalaAdapter()
        adapter.chat("system", [MagicMock(role="user", content="test")])

    assert len(adapter.voicing_events) == 1
    assert adapter.voicing_events[0]["effective_operator"] == "phala:deepseek/deepseek-v4-flash-0731"
    assert adapter.voicing_events[0]["reason"] == "success"


def test_phala_adapter_unreachable_propagates_never_swallowed():
    """Unreachable Phala raises straight through the adapter — no fallback,
    no swallow, exactly like GWParkedError propagates through GravityWellAdapter."""
    from agents_core.council.phala_adapter import PhalaAdapter
    from agents_core.llm import PhalaOperatorUnavailable

    with patch("agents_core.council.phala_adapter.call_operator",
               side_effect=PhalaOperatorUnavailable("http://127.0.0.1:8413", Exception("refused"))):
        adapter = PhalaAdapter()
        with pytest.raises(PhalaOperatorUnavailable):
            adapter.chat("system", [MagicMock(role="user", content="test")])


def test_flatten_messages_single_and_multi():
    from agents_core.council.phala_adapter import _flatten_messages

    single = [MagicMock(role="user", content="only one")]
    assert _flatten_messages(single) == "only one"

    multi = [
        MagicMock(role="user", content="first"),
        MagicMock(role="assistant", content="second"),
    ]
    flattened = _flatten_messages(multi)
    assert "user: first" in flattened
    assert "assistant: second" in flattened


# ---------------------------------------------------------------------------
# _build_adapter dispatch and --voicing choices (council/cli.py)
# ---------------------------------------------------------------------------

def test_build_adapter_phala_returns_phala_adapter():
    from agents_core.council.cli import _build_adapter
    from agents_core.council.phala_adapter import PhalaAdapter

    adapter = _build_adapter("phala", ClaudeAdapter=None, LlamaAdapter=None)
    assert isinstance(adapter, PhalaAdapter)


def test_voicing_choices_include_phala():
    from agents_core.council.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["submit", "test decision", "--voicing", "phala"])
    assert args.voicing == "phala"


def test_apply_voicing_provenance_names_phala_and_model():
    """DoD 5: run["effective_voicing"] names Phala and the served model id —
    reusing the requested-vs-effective machinery, never folded into 'gravitywell'
    or a bare 'phala' with no model."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.phala_adapter import PhalaAdapter

    adapter = PhalaAdapter()
    adapter.voicing_events.append({
        "effective_operator": "phala:deepseek/deepseek-v4-flash-0731",
        "reason": "success",
    })
    run = {"voicing": "phala", "turns": [{}]}

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "phala:deepseek/deepseek-v4-flash-0731"
    assert run["voicing_degraded"] is False
    assert run["turns"][0]["effective_voicing"] == "phala:deepseek/deepseek-v4-flash-0731"


def test_refuse_wave_against_122b_does_not_refuse_phala():
    """Explicit decision (spec DoD 8): phala is a remote HTTP endpoint with no
    --parallel 1 pin, so wave mode is not refused for it."""
    from agents_core.council.cli import _refuse_wave_against_122b

    # Must not raise.
    _refuse_wave_against_122b("phala")
