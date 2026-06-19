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
    from agents_core.llm import OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["qwen"] == "qwen3.6-35b-a3b"


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
    """Anthropic-family classes route via submit_and_wait and return the response string."""
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="y") as mock_saw:
        from agents_core import llm as llm_mod
        result = llm_mod.call_operator(operator_class, prompt="x")

    assert result == "y"
    mock_saw.assert_called_once()
    task_dict = mock_saw.call_args[0][0]
    assert task_dict["task_type"] == "llm_call"
    assert task_dict["model"] == expected_model
    assert task_dict["payload"]["operator_class"] == operator_class
    assert task_dict["payload"]["prompt"] == "x"
    assert task_dict["payload"]["system"] == ""
    assert task_dict["payload"]["json_mode"] is False
    assert task_dict["payload"]["_ignore_intention_registry"] is True


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
    from agents_core.llm import call_operator, OPERATOR_DEFAULTS
    with pytest.raises(ValueError) as exc_info:
        call_operator("unknown-op", prompt="x")
    err = str(exc_info.value)
    for valid in OPERATOR_DEFAULTS:
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


def test_call_llm_raises_on_unreachable_after_retries():
    """call_llm() raises OperatorUnreachableError after exhausting retries on ConnectionError."""
    import requests as req
    from agents_core.llm import call_llm, OperatorUnreachableError, LLAMACPP_URL

    conn_err = req.exceptions.ConnectionError("down")

    with patch("agents_core.llm.requests.post", side_effect=conn_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            call_llm(prompt="will fail", system="s")

    exc = exc_info.value
    assert exc.url == LLAMACPP_URL
    assert exc.last_error is conn_err


def test_call_llm_raises_on_http_error_after_retries():
    """call_llm() raises OperatorUnreachableError after exhausting retries on HTTPError."""
    import requests as req
    from agents_core.llm import call_llm, OperatorUnreachableError, LLAMACPP_URL

    http_err = req.exceptions.HTTPError("503")

    with patch("agents_core.llm.requests.post", side_effect=http_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            call_llm(prompt="will fail", system="s")

    exc = exc_info.value
    assert exc.url == LLAMACPP_URL
    assert exc.last_error is http_err


def test_call_llm_returns_none_on_empty_content():
    """call_llm() returns None (not exception) when operator returns empty content."""
    from agents_core.llm import call_llm

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": None}}]
    }

    with patch("agents_core.llm.requests.post", return_value=resp):
        result = call_llm(prompt="empty reply", system="s")

    assert result is None


def test_call_llm_retries_then_succeeds():
    """call_llm() retries on ConnectionError and returns content on eventual success."""
    import requests as req
    from agents_core.llm import call_llm

    success_resp = _make_llama_response("recovered")
    conn_err = req.exceptions.ConnectionError("transient")

    with patch("agents_core.llm.requests.post",
               side_effect=[conn_err, conn_err, success_resp]), \
         patch("agents_core.llm.time.sleep"):
        result = call_llm(prompt="retry me", system="s")

    assert result == "recovered"


def test_call_llm_returns_none_on_generic_exception():
    """call_llm() returns None (not exception) for non-Connection/HTTP exceptions."""
    from agents_core.llm import call_llm

    with patch("agents_core.llm.requests.post", side_effect=ValueError("unexpected")):
        result = call_llm(prompt="generic error", system="s")

    assert result is None


def test_operator_unreachable_error_carries_diagnostic():
    """OperatorUnreachableError exposes .url and .last_error; str() includes both."""
    from agents_core.llm import OperatorUnreachableError

    cause = ValueError("boom")
    exc = OperatorUnreachableError("http://x:1234", cause)

    assert exc.url == "http://x:1234"
    assert exc.last_error is cause
    msg = str(exc)
    assert "http://x:1234" in msg
    assert "boom" in msg


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


# ---------------------------------------------------------------------------
# Voicing Provenance Tests (GravityWell effective-operator tracking)
# ---------------------------------------------------------------------------

def test_gravitywell_successful_call_records_provenance():
    """call_operator('gravitywell') on success appends ('success', 'gravitywell') to provenance."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "gw answer"
    assert ("success", "gravitywell") in provenance


def test_gravitywell_fallback_on_doorman_unreachable():
    """call_operator('gravitywell') appends doorman_unreachable reason when doorman is down."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    provenance = []
    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("doorman down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("doorman_unreachable", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_gravitywell_fallback_on_gw_not_serving():
    """call_operator('gravitywell') appends gw_not_serving reason when doorman says not serving."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "offline"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("gw_not_serving", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_gravitywell_fallback_on_http_error():
    """call_operator('gravitywell') appends serving_http_error when GW returns HTTP error."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient
    from agents_core.llm import OperatorUnreachableError

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw:8081", Exception("500"))), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("serving_http_error", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_non_gravitywell_operator_records_success_provenance():
    """call_operator('sonnet') appends ('success', 'sonnet') to provenance."""
    from agents_core import llm as llm_mod

    provenance = []
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="sonnet answer"):
        result = llm_mod.call_operator(
            "sonnet", prompt="test",
            _provenance_out=provenance,
        )

    assert result == "sonnet answer"
    assert ("success", "sonnet") in provenance


# ---------------------------------------------------------------------------
# GravityWell Retry Hardening Tests (gw-backend-retry-hardening-v0)
# ---------------------------------------------------------------------------

def _make_gw_response(text: str):
    """Minimal requests.Response mock for a successful GravityWell reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text, "reasoning_content": None}}]
    }
    return resp


def test_gravitywell_retries_on_timeout():
    """_call_gravitywell_backend retries on ReadTimeout and returns completion on success."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    timeout_err = req.exceptions.Timeout("read timeout")
    success_resp = _make_gw_response("recovered after timeout")

    with patch("agents_core.llm.requests.post",
               side_effect=[timeout_err, timeout_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry me")

    assert result == "recovered after timeout"
    # Verify backoff was called: attempt 0 (2s) and attempt 1 (4s)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(2)
    mock_sleep.assert_any_call(4)


def test_gravitywell_raises_on_persistent_timeout():
    """_call_gravitywell_backend raises OperatorUnreachableError after timeout exhaustion."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend, OperatorUnreachableError, GW_URL

    timeout_err = req.exceptions.Timeout("persistent timeout")

    with patch("agents_core.llm.requests.post", side_effect=timeout_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            _call_gravitywell_backend(prompt="will fail")

    exc = exc_info.value
    assert exc.url == GW_URL
    assert isinstance(exc.last_error, req.exceptions.Timeout)


def test_gravitywell_retries_on_chunked_encoding_error():
    """_call_gravitywell_backend retries on ChunkedEncodingError (stateless calls)."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    chunked_err = req.exceptions.ChunkedEncodingError("connection broken")
    success_resp = _make_gw_response("recovered from chunked error")

    with patch("agents_core.llm.requests.post",
               side_effect=[chunked_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry chunked")

    assert result == "recovered from chunked error"
    # Only one backoff (after attempt 0)
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(2)


def test_gravitywell_empty_content_returns_none():
    """_call_gravitywell_backend returns None for successful 200 with empty content (unchanged behavior)."""
    from agents_core.llm import _call_gravitywell_backend

    empty_resp = MagicMock()
    empty_resp.raise_for_status = MagicMock()
    empty_resp.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": None}}]
    }

    with patch("agents_core.llm.requests.post", return_value=empty_resp):
        result = _call_gravitywell_backend(prompt="empty reply")

    assert result is None


def test_gravitywell_malformed_body_returns_none_via_generic_handler():
    """_call_gravitywell_backend returns None for malformed response (KeyError/IndexError)."""
    from agents_core.llm import _call_gravitywell_backend

    # Missing "choices" key
    malformed_resp = MagicMock()
    malformed_resp.raise_for_status = MagicMock()
    malformed_resp.json.return_value = {"error": "unexpected structure"}

    with patch("agents_core.llm.requests.post", return_value=malformed_resp):
        result = _call_gravitywell_backend(prompt="malformed response")

    assert result is None


def test_gravitywell_empty_choices_list_returns_none():
    """_call_gravitywell_backend returns None for empty choices list (IndexError)."""
    from agents_core.llm import _call_gravitywell_backend

    empty_choices_resp = MagicMock()
    empty_choices_resp.raise_for_status = MagicMock()
    empty_choices_resp.json.return_value = {"choices": []}

    with patch("agents_core.llm.requests.post", return_value=empty_choices_resp):
        result = _call_gravitywell_backend(prompt="empty choices")

    assert result is None


def test_gravitywell_backoff_timing_cumulative():
    """_call_gravitywell_backend total backoff (6s) does not exceed timeout_per_persona budget."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend, OperatorUnreachableError

    timeout_err = req.exceptions.Timeout("timeout")

    with patch("agents_core.llm.requests.post", side_effect=timeout_err), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        with pytest.raises(OperatorUnreachableError):
            _call_gravitywell_backend(prompt="backoff test")

    # Two backoffs: 2s (after attempt 0) and 4s (after attempt 1)
    assert mock_sleep.call_count == 2
    total_sleep = sum(call[0][0] for call in mock_sleep.call_args_list)
    assert total_sleep == 6
    assert total_sleep <= 6  # Should not exceed the 6s cap


def test_gravitywell_http_error_is_retryable():
    """_call_gravitywell_backend retries on HTTPError (e.g., 503)."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    http_err = req.exceptions.HTTPError("503 Service Unavailable")
    success_resp = _make_gw_response("recovered from 503")

    with patch("agents_core.llm.requests.post",
               side_effect=[http_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry http")

    assert result == "recovered from 503"
    mock_sleep.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# Swarm Tests: call_swarm, swarm_serving, swarm_model
# ---------------------------------------------------------------------------

def _make_swarm_response(text: str):
    """Minimal requests.Response mock for a successful swarm reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return resp


def _make_models_response(models: list[str]):
    """Mock /v1/models response with a list of model IDs."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": [{"id": model} for model in models]
    }
    return resp


def test_swarm_serving_returns_true_when_both_healthy():
    """swarm_serving() returns True when /v1/models (200+non-empty) and /health (200)."""
    from agents_core.llm import swarm_serving

    models_resp = _make_models_response(["Qwen2.5-3B"])
    health_resp = MagicMock()
    health_resp.raise_for_status = MagicMock()

    with patch("agents_core.llm.requests.get") as mock_get:
        mock_get.side_effect = [models_resp, health_resp]
        result = swarm_serving("http://localhost:8081")

    assert result is True
    assert mock_get.call_count == 2


def test_swarm_serving_false_when_models_empty():
    """swarm_serving() returns False when /v1/models is empty."""
    from agents_core.llm import swarm_serving

    empty_models = MagicMock()
    empty_models.raise_for_status = MagicMock()
    empty_models.json.return_value = {"data": []}

    with patch("agents_core.llm.requests.get", return_value=empty_models):
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_serving_false_when_health_down():
    """swarm_serving() returns False when /health fails."""
    from agents_core.llm import swarm_serving

    models_resp = _make_models_response(["Qwen2.5-3B"])
    health_err = MagicMock()
    health_err.raise_for_status = MagicMock(side_effect=Exception("health check failed"))

    with patch("agents_core.llm.requests.get") as mock_get:
        mock_get.side_effect = [models_resp, health_err]
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_serving_false_when_unreachable():
    """swarm_serving() returns False when endpoint is unreachable."""
    from agents_core.llm import swarm_serving
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_model_returns_model_id():
    """swarm_model() returns the first model id from /v1/models."""
    from agents_core.llm import swarm_model

    models_resp = _make_models_response(["Qwen2.5-3B", "other-model"])

    with patch("agents_core.llm.requests.get", return_value=models_resp):
        result = swarm_model("http://localhost:8081")

    assert result == "Qwen2.5-3B"


def test_swarm_model_returns_none_when_unreachable():
    """swarm_model() returns None when endpoint is unreachable."""
    from agents_core.llm import swarm_model
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = swarm_model("http://localhost:8081")

    assert result is None


def test_call_swarm_single_prompt_success():
    """call_swarm() with one prompt returns [result]."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("swarm answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp):
        result = call_swarm(["hello"])

    assert result == ["swarm answer"]


def test_call_swarm_multiple_prompts_preserves_order():
    """call_swarm() with multiple prompts returns results in input order."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    resp1 = _make_swarm_response("answer1")
    resp2 = _make_swarm_response("answer2")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post") as mock_post:
        # Return different responses in order
        mock_post.side_effect = [resp1, resp2]
        result = call_swarm(["prompt1", "prompt2"])

    assert len(result) == 2
    assert result[0] == "answer1"
    assert result[1] == "answer2"


def test_call_swarm_returns_all_none_when_not_serving():
    """call_swarm() returns [None, None, ...] when swarm not serving."""
    from agents_core.llm import call_swarm
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = call_swarm(["p1", "p2", "p3"])

    assert result == [None, None, None]


def test_call_swarm_isolates_per_prompt_errors():
    """call_swarm() maps per-prompt errors to None, preserves successes."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    success_resp = _make_swarm_response("success")
    error_resp = MagicMock()
    error_resp.raise_for_status = MagicMock(side_effect=Exception("error"))

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post") as mock_post:
        mock_post.side_effect = [success_resp, error_resp, success_resp]
        result = call_swarm(["p1", "p2", "p3"])

    assert len(result) == 3
    assert result[0] == "success"
    assert result[1] is None
    assert result[2] == "success"


def test_call_swarm_uses_provided_model():
    """call_swarm() uses the model param when provided (no /v1/models call)."""
    from agents_core.llm import call_swarm

    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get") as mock_get, \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        result = call_swarm(["hello"], model="custom-model")

    # Should NOT call /v1/models when model is provided
    mock_get.assert_not_called()
    assert result == ["answer"]

    # POST payload should include the custom model
    posted_data = mock_post.call_args[1]["json"]
    assert posted_data["model"] == "custom-model"


def test_call_swarm_does_not_import_doorman_client():
    """call_swarm module scope does NOT import DoormanClient (static guarantee)."""
    import agents_core.llm as llm_module

    # Verify that DoormanClient is not in the module's top-level namespace
    # (it may be imported inside call_operator, but not at module scope)
    assert "DoormanClient" not in dir(llm_module)
    assert not hasattr(llm_module, "DoormanClient")


def test_call_swarm_acquire_never_called():
    """Behavioral test: call_swarm never calls DoormanClient.acquire (even if imported)."""
    from agents_core.llm import call_swarm
    from agents_core.doorman_client import DoormanClient

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch.object(DoormanClient, "acquire") as mock_acquire, \
         patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp):
        result = call_swarm(["hello"])

    mock_acquire.assert_not_called()
    assert result == ["answer"]


def test_call_swarm_empty_prompts_list_returns_empty():
    """call_swarm([]) returns []."""
    from agents_core.llm import call_swarm

    result = call_swarm([])

    assert result == []


def test_call_swarm_respects_max_concurrent():
    """call_swarm() fans out bounded by max_concurrent (default SWARM_MAX_CONCURRENT)."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    # Patch ThreadPoolExecutor at the agents_core.llm module level where it's used
    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp), \
         patch("agents_core.llm.ThreadPoolExecutor") as mock_executor_class:

        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=None)
        mock_executor.submit = MagicMock()
        mock_executor.submit.return_value.result = MagicMock(return_value=(0, "answer", "success"))

        # Patch as_completed to return futures in order
        with patch("agents_core.llm.as_completed") as mock_as_completed:
            mock_as_completed.return_value = []
            call_swarm(["p1", "p2"], max_concurrent=2)

        mock_executor_class.assert_called_once_with(max_workers=2)


def test_call_swarm_post_includes_system_prompt():
    """call_swarm() includes system prompt in message payload."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        call_swarm(["hello"], system="You are helpful.")

    posted_data = mock_post.call_args[1]["json"]
    messages = posted_data["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "hello"}
