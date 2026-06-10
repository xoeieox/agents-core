"""Tests for call_operator() — qwen/gravitywell model guards and operator routing."""
import warnings
import pytest
from unittest.mock import patch, MagicMock, call

from agents_core.llm import call_operator, OPERATOR_DEFAULTS, OperatorUnreachableError


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


# ---------------------------------------------------------------------------
# gravitywell: in OPERATOR_DEFAULTS
# ---------------------------------------------------------------------------

def test_gravitywell_in_operator_defaults():
    assert "gravitywell" in OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["gravitywell"] == "gravitywell-122b"


# ---------------------------------------------------------------------------
# gravitywell: non-default model must raise ValueError
# ---------------------------------------------------------------------------

def test_call_operator_gravitywell_non_default_model_raises():
    """Passing a non-default model to gravitywell raises ValueError."""
    mock_client = MagicMock()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client):
        with pytest.raises(ValueError) as exc_info:
            call_operator("gravitywell", prompt="hi", model="gw-other")
    assert "infrastructure operation" in str(exc_info.value)
    assert "gw-other" in str(exc_info.value)


# ---------------------------------------------------------------------------
# gravitywell: default payload has enable_thinking=False
# ---------------------------------------------------------------------------

def _gw_mock_client(status="serving"):
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.acquire.return_value = {"status": status, "node": "gravitywell", "work_id": "w1"}
    return mock


def test_gravitywell_default_think_off():
    """Default call injects enable_thinking=False into the payload."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}', "reasoning_content": None}}]
        }
        return resp

    mock_client = _gw_mock_client()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell", prompt="test", json_mode=True)

    assert result is not None
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False


def test_gravitywell_think_true():
    """think=True injects enable_thinking=True."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "answer", "reasoning_content": "trace"}}]
        }
        return resp

    mock_client = _gw_mock_client()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell", prompt="test", think=True)

    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is True


# ---------------------------------------------------------------------------
# gravitywell: bundle_ids passed through call_operator does NOT raise TypeError
# ---------------------------------------------------------------------------

def test_gravitywell_bundle_ids_discarded_no_typeerror():
    """bundle_ids kwarg is discarded before reaching _call_gravitywell_backend."""
    mock_client = _gw_mock_client()

    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok", "reasoning_content": None}}]
        }
        return resp

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("requests.post", side_effect=fake_post):
        # Must not raise TypeError
        result = call_operator("gravitywell", prompt="test", bundle_ids=["some/bundle"])
    assert result == "ok"


# ---------------------------------------------------------------------------
# gravitywell: on_wake_fail policies
# ---------------------------------------------------------------------------

def test_gravitywell_wake_failed_skip_returns_none():
    mock_client = _gw_mock_client(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None


def test_gravitywell_wake_failed_error_raises():
    mock_client = _gw_mock_client(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client):
        with pytest.raises(OperatorUnreachableError):
            call_operator("gravitywell", prompt="test", on_wake_fail="error")


def test_gravitywell_wake_failed_haiku_logs_and_falls_back():
    """on_wake_fail='haiku' emits a loud warning then falls back to haiku operator."""
    mock_client = _gw_mock_client(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="haiku-reply") as mock_saw, \
         warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = call_operator("gravitywell", prompt="test", on_wake_fail="haiku")

    assert result == "haiku-reply"
    assert len(w) >= 1
    assert any("haiku" in str(warning.message).lower() or "paid" in str(warning.message).lower()
               for warning in w)


def test_gravitywell_doorman_unreachable_applies_wake_fail():
    """DoormanUnreachable is treated like wake_failed."""
    from agents_core.doorman_client import DoormanUnreachable
    mock_client = MagicMock()
    mock_client.acquire.side_effect = DoormanUnreachable("can't reach doorman")
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None


def test_gravitywell_haiku_fallback_raises_propagates():
    """If the fallback call_operator raises, it propagates (not suppressed to None)."""
    mock_client = _gw_mock_client(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("agents_core.claude_queue_sync.submit_and_wait",
               side_effect=RuntimeError("ClaudeQueue down")):
        with pytest.raises(RuntimeError, match="ClaudeQueue down"):
            call_operator("gravitywell", prompt="test", on_wake_fail="haiku")


# ---------------------------------------------------------------------------
# gravitywell: release is called even when backend raises
# ---------------------------------------------------------------------------

def test_gravitywell_release_called_on_backend_error():
    """client.release is called in the finally block even when the backend raises."""
    mock_client = _gw_mock_client(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client), \
         patch("requests.post", side_effect=Exception("backend exploded")):
        result = call_operator("gravitywell", prompt="test")
    assert mock_client.release.call_count == 1
    release_args = mock_client.release.call_args[0]
    assert release_args[0] == "gravitywell"
