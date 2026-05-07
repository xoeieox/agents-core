"""Tests for call_operator() — qwen model guard and operator routing."""
import pytest
from unittest.mock import patch, MagicMock

from agents_core.llm import call_operator, _OPERATOR_DEFAULTS


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
        result = call_operator("qwen", prompt="hi", model=_OPERATOR_DEFAULTS["qwen"])
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
    assert _OPERATOR_DEFAULTS["qwen"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# sonnet: non-default model must NOT be rejected — passes through to ClaudeQueue
# ---------------------------------------------------------------------------

def test_call_operator_sonnet_model_override_passes_through():
    """A non-default model on sonnet is not rejected; it routes to ClaudeQueue."""
    mock_queue = MagicMock()
    mock_queue.submit.return_value = "task-123"

    with patch("agents_core.claude_queue.ClaudeQueue", return_value=mock_queue):
        with pytest.raises(NotImplementedError) as exc_info:
            call_operator("sonnet", prompt="hi", model="claude-sonnet-4-5")

    # Confirm it reached ClaudeQueue (not rejected before)
    mock_queue.submit.assert_called_once()
    submitted = mock_queue.submit.call_args[0][0]
    assert submitted["model"] == "claude-sonnet-4-5"
    # Confirm the NotImplementedError is the v0 gap, not a model-guard error
    assert "task_id" in str(exc_info.value) or "submitted to ClaudeQueue" in str(exc_info.value)
