"""Tests for GravityWellAdapter and gravitywell voicing dispatch.

Covers: adapter chat() routing, message flattening, _build_adapter dispatch,
serving-then-HTTP-fail fallback, and zero-engine-import invariant.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Test 1: chat() routes to call_operator with correct shape
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_chat_routes_to_operator():
    """GravityWellAdapter.chat() calls call_operator with correct args."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="voiced response") as mock_op:
        adapter = GravityWellAdapter(temperature=0.8)
        result = adapter.chat("ENTITY_CARD", [MagicMock(role="user", content="hi")])

    assert result == "voiced response"
    mock_op.assert_called_once()
    call_args, call_kwargs = mock_op.call_args
    assert call_args[0] == "gravitywell"  # operator_class positional
    assert call_args[1] == "hi"  # prompt positional
    assert call_kwargs["system"] == "ENTITY_CARD"
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["timeout"] == 300
    assert call_kwargs["on_wake_fail"] == "sonnet"
    assert "model" not in call_kwargs  # MUST NOT pass model=


def test_gravitywell_adapter_no_model_kwarg():
    """Verify model= is never passed to call_operator (contract violation guard)."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="response") as mock_op:
        adapter = GravityWellAdapter()
        adapter.chat("system", [MagicMock(role="user", content="test")])

    # Verify model= was not passed
    _, call_kwargs = mock_op.call_args
    assert "model" not in call_kwargs


# ---------------------------------------------------------------------------
# Test 2: System is passed through unchanged
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_system_passthrough():
    """System card / entity persona is passed to call_operator unchanged."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    entity_card = "You are Ada Lovelace, inventor of the algorithm."
    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="response") as mock_op:
        adapter = GravityWellAdapter()
        adapter.chat(entity_card, [MagicMock(role="user", content="prompt")])

    _, call_kwargs = mock_op.call_args
    assert call_kwargs["system"] == entity_card


def test_gravitywell_adapter_empty_system_passthrough():
    """Empty string system (not None) is passed unchanged (backend handles it)."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value="response") as mock_op:
        adapter = GravityWellAdapter()
        adapter.chat("", [MagicMock(role="user", content="test")])

    _, call_kwargs = mock_op.call_args
    assert call_kwargs["system"] == ""


# ---------------------------------------------------------------------------
# Test 3: Message flattening — single message fast path + multi-message join
# ---------------------------------------------------------------------------

def test_flatten_messages_single_user():
    """Single user message returns bare content."""
    from agents_core.council.gravitywell_adapter import _flatten_messages

    msg = MagicMock(role="user", content="hello")
    result = _flatten_messages([msg])
    assert result == "hello"


def test_flatten_messages_multi():
    """Multi-message list returns role-tagged join."""
    from agents_core.council.gravitywell_adapter import _flatten_messages

    messages = [
        MagicMock(role="user", content="prompt"),
        MagicMock(role="assistant", content="response"),
    ]
    result = _flatten_messages(messages)
    assert "user: prompt" in result
    assert "assistant: response" in result


def test_flatten_messages_missing_attrs_defaults():
    """Messages missing role/content attrs get defaults."""
    from agents_core.council.gravitywell_adapter import _flatten_messages

    msg = MagicMock(spec=[])  # No role/content attrs
    msg.role = "user"  # Only role, no content
    msg.content = "test"
    result = _flatten_messages([msg])
    assert result == "test"


# ---------------------------------------------------------------------------
# Test 4: Empty-GW guard (result=None → "")
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_empty_gw_returns_empty_string():
    """call_operator returns None (GW empty) → chat() returns '' (never None)."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    with patch("agents_core.council.gravitywell_adapter.call_operator",
               return_value=None):
        adapter = GravityWellAdapter()
        result = adapter.chat("system", [MagicMock(role="user", content="test")])

    assert result == ""
    assert result is not None


# ---------------------------------------------------------------------------
# Test 5: _build_adapter dispatch
# ---------------------------------------------------------------------------

def test_build_adapter_gravitywell():
    """_build_adapter('gravitywell', ...) returns a GravityWellAdapter."""
    from agents_core.council.cli import _build_adapter
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    claude_stub = MagicMock()
    llama_stub = MagicMock()
    result = _build_adapter("gravitywell", claude_stub, llama_stub)

    assert isinstance(result, GravityWellAdapter)
    assert result.temperature == 0.8


def test_build_adapter_local():
    """_build_adapter('local', ...) still returns LlamaAdapter (unchanged)."""
    from agents_core.council.cli import _build_adapter

    claude_stub = MagicMock()
    llama_stub = MagicMock()
    _build_adapter("local", claude_stub, llama_stub)
    llama_stub.assert_called_once_with(temperature=0.8, max_tokens=900)


def test_build_adapter_sonnet_raises():
    """_build_adapter('sonnet', ...) raises ValueError (paid-model voicing removed)."""
    from agents_core.council.cli import _build_adapter

    claude_stub = MagicMock()
    llama_stub = MagicMock()
    with pytest.raises(ValueError, match="Paid-model voicing"):
        _build_adapter("sonnet", claude_stub, llama_stub)


def test_build_adapter_unknown_raises():
    """_build_adapter with unknown voicing raises ValueError."""
    from agents_core.council.cli import _build_adapter

    with pytest.raises(ValueError, match="Unknown voicing"):
        _build_adapter("unknown", MagicMock(), MagicMock())


# ---------------------------------------------------------------------------
# Test 6: serving-then-HTTP-fail falls back (Edit 4 fix)
# ---------------------------------------------------------------------------

def test_gravitywell_serving_then_http_fail_falls_back_to_sonnet():
    """When GW acquires (status=serving) then HTTP fails with OperatorUnreachableError,
    on_wake_fail='sonnet' routes to Sonnet fallback (never raises)."""
    from agents_core.llm import call_operator, OperatorUnreachableError

    # Create a mock doorman client (imported locally in the call_operator path)
    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving"}

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw", Exception("timeout"))), \
         patch("agents_core.claude_queue_sync.submit_and_wait",
               return_value="sonnet-fallback"), \
         warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = call_operator("gravitywell", prompt="test", on_wake_fail="sonnet")

    # Should have fallen back to Sonnet (no crash)
    assert result == "sonnet-fallback"
    # Verify release was called
    assert mock_client.release.call_count == 1
    # Verify loud warning was emitted
    assert any("sonnet" in str(warning.message).lower() or "paid" in str(warning.message).lower()
               for warning in w)


def test_gravitywell_serving_then_http_fail_skip_returns_none():
    """When GW fails with OperatorUnreachableError and on_wake_fail='skip',
    returns None (no paid spend, no crash)."""
    from agents_core.llm import call_operator, OperatorUnreachableError

    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving"}

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw", Exception("timeout"))):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")

    assert result is None
    # Verify release was called (the finally block ran)
    assert mock_client.release.call_count == 1


# ---------------------------------------------------------------------------
# Test 7: Zero lapis_engine import (mechanical gate)
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_no_lapis_engine_import():
    """GravityWellAdapter source contains no actual lapis_engine import statements."""
    import re
    from pathlib import Path

    adapter_file = Path(__file__).parent.parent / "agents_core" / "council" / "gravitywell_adapter.py"
    source = adapter_file.read_text()

    # Check for actual import statements (not in docstrings)
    # Strip docstrings and comments to avoid false positives
    lines = source.split("\n")
    in_docstring = False
    code_only = []
    for line in lines:
        if '"""' in line or "'''" in line:
            in_docstring = not in_docstring
            continue
        if not in_docstring and not line.strip().startswith("#"):
            code_only.append(line)

    code = "\n".join(code_only)
    # Check for actual imports
    assert not re.search(r"^\s*(from|import)\s+lapis_engine", code, re.MULTILINE), \
        "GravityWellAdapter must not import from lapis_engine"
