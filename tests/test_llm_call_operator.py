"""Tests for call_operator() — qwen model guard and operator routing."""
import pytest
from unittest.mock import patch, MagicMock

from agents_core.llm import call_operator, OPERATOR_DEFAULTS


# ---------------------------------------------------------------------------
# qwen: default model (None or explicit default) must reach the backend
# ---------------------------------------------------------------------------

def test_call_operator_qwen_default_model_works():
    """model=None and model=<default> both reach _call_qwen_backend without error."""
    with patch("agents_core.llm._call_qwen_backend", return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", model=None)
        assert result == "ok"
        mock_qwen.assert_called_once_with(prompt="hi")

    with patch("agents_core.llm._call_qwen_backend", return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", model=OPERATOR_DEFAULTS["qwen"])
        assert result == "ok"
        mock_qwen.assert_called_once_with(prompt="hi")


# ---------------------------------------------------------------------------
# qwen: non-default model must raise ValueError with the directive message
# ---------------------------------------------------------------------------

def test_call_operator_qwen_non_default_model_raises():
    """Passing a non-default model to the qwen operator raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        call_operator("qwen", prompt="hi", model="qwen-other-7b")
    assert "infrastructure operation" in str(exc_info.value)
    assert "qwen-other-7b" in str(exc_info.value)
    assert OPERATOR_DEFAULTS["qwen"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# sonnet: non-default model must NOT be rejected — passes through to ClaudeQueue
# ---------------------------------------------------------------------------

def test_call_operator_sonnet_model_override_passes_through():
    """A non-default model on sonnet is not rejected; it propagates into the task dict."""
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="ok") as mock_saw:
        result = call_operator("sonnet", prompt="hi", model="claude-sonnet-4-5")

    assert result == "ok"
    mock_saw.assert_called_once()
    task_dict = mock_saw.call_args[0][0]
    assert task_dict["model"] == "claude-sonnet-4-5"
