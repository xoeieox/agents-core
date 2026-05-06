"""Unit tests for agents_core.llm multi-operator routing (v0).

Covers:
  - call_operator() dispatches to the right backend per operator class
  - Unknown operator class raises ValueError
  - call_llm() legacy behavior is byte-identical to pre-bind

No real network calls — all backends are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llama_response(text: str):
    """Minimal requests.Response mock for a successful llama-server reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text, "reasoning_content": None}}]
    }
    return resp


# ---------------------------------------------------------------------------
# test_call_operator_qwen_dispatches_to_existing_endpoint
# ---------------------------------------------------------------------------

def test_call_operator_qwen_dispatches_to_existing_endpoint():
    """qwen operator_class posts to the local llama-server URL."""
    from agents_core import llm as llm_mod

    fake_resp = _make_llama_response("qwen answer")

    with patch("agents_core.llm.requests.post", return_value=fake_resp) as mock_post, \
         patch("agents_core.llm._call_qwen_backend",
               wraps=llm_mod._call_qwen_backend) as spy:

        result = llm_mod.call_operator(
            "qwen",
            prompt="hello",
            system="sys",
            bundle_ids=None,
        )

    assert result == "qwen answer"
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert "8081" in called_url
    assert "/v1/chat/completions" in called_url


def test_call_operator_qwen_uses_default_model_constant():
    """The qwen default model string is the expected production value."""
    from agents_core.llm import _OPERATOR_DEFAULTS
    assert _OPERATOR_DEFAULTS["qwen"] == "qwen3.6-35b-a3b"


# ---------------------------------------------------------------------------
# test_call_operator_sonnet_dispatches_to_claude_queue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator_class,expected_model", [
    ("sonnet", "claude-sonnet-4-6"),
    ("opus",   "claude-opus-4-7"),
    ("haiku",  "claude-haiku-4-5-20251001"),
])
def test_call_operator_anthropic_dispatches_to_claude_queue(
        operator_class, expected_model):
    """Anthropic-family classes submit to ClaudeQueue with the right model.

    v0 gap: the function raises NotImplementedError after submit (no sync surface).
    The test verifies both the dispatch and the gap signal.
    """
    mock_queue = MagicMock()
    mock_queue.submit.return_value = f"claude_fake_{operator_class}_id"

    with patch("agents_core.claude_queue.ClaudeQueue", return_value=mock_queue) as mock_cls:
        with pytest.raises(NotImplementedError) as exc_info:
            from agents_core import llm as llm_mod
            llm_mod.call_operator(operator_class, prompt="test prompt")

    mock_queue.submit.assert_called_once()
    submitted = mock_queue.submit.call_args[0][0]
    assert submitted["model"] == expected_model
    assert submitted["task_type"] == "llm_call"
    assert submitted["payload"]["operator_class"] == operator_class
    assert submitted["payload"]["prompt"] == "test prompt"

    # Gap signal: exception message names the task_id
    err_msg = str(exc_info.value)
    assert operator_class in err_msg
    assert f"claude_fake_{operator_class}_id" in err_msg


def test_call_operator_sonnet_dispatches_to_claude_queue():
    """Alias for the parametrized test — spec-named test function."""
    test_call_operator_anthropic_dispatches_to_claude_queue(
        "sonnet", "claude-sonnet-4-6"
    )


# ---------------------------------------------------------------------------
# test_call_operator_unknown_class_raises
# ---------------------------------------------------------------------------

def test_call_operator_unknown_class_raises():
    """Unknown operator_class raises ValueError with the class name."""
    from agents_core.llm import call_operator
    with pytest.raises(ValueError, match="gpt4"):
        call_operator("gpt4", prompt="irrelevant")


def test_call_operator_unknown_class_names_valid_options():
    """ValueError message lists the valid operator classes."""
    from agents_core.llm import call_operator, _OPERATOR_DEFAULTS
    with pytest.raises(ValueError) as exc_info:
        call_operator("unknown-op", prompt="x")
    err = str(exc_info.value)
    for valid in _OPERATOR_DEFAULTS:
        assert valid in err


# ---------------------------------------------------------------------------
# test_call_llm_legacy_behavior_byte_identical
# ---------------------------------------------------------------------------

def test_call_llm_legacy_behavior_byte_identical():
    """call_llm() returns the same result as the pre-bind qwen path.

    Regression guard: same inputs → same output shape and same HTTP endpoint.
    """
    fake_resp = _make_llama_response("legacy response text")

    with patch("agents_core.llm.requests.post", return_value=fake_resp) as mock_post:
        from agents_core.llm import call_llm
        result = call_llm(prompt="legacy prompt", system="sys override")

    assert result == "legacy response text"
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert "/v1/chat/completions" in called_url
    payload = mock_post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "sys override"}
    assert payload["messages"][1] == {"role": "user", "content": "legacy prompt"}


def test_call_llm_returns_none_on_connection_error():
    """call_llm() returns None after exhausting retries (legacy behavior)."""
    import requests as req

    with patch("agents_core.llm.requests.post",
               side_effect=req.exceptions.ConnectionError("down")), \
         patch("agents_core.llm.time.sleep"):
        from agents_core.llm import call_llm
        result = call_llm(prompt="will fail", system="s")

    assert result is None


def test_call_llm_delegates_to_call_operator():
    """call_llm() is a thin wrapper — it calls call_operator('qwen', ...)."""
    from agents_core import llm as llm_mod

    with patch.object(llm_mod, "call_operator", return_value="delegated") as mock_op:
        result = llm_mod.call_llm(
            prompt="p", system="s", timeout=99, json_mode=True,
            temperature=0.1, log=None, bundle_ids=["x"]
        )

    assert result == "delegated"
    mock_op.assert_called_once_with(
        operator_class="qwen",
        prompt="p",
        system="s",
        timeout=99,
        json_mode=True,
        temperature=0.1,
        log=None,
        bundle_ids=["x"],
    )
