"""Tests for agents_core.phala_tee_proxy (agents-core-phala-test-key-v0).

Covers: request shaping (reasoning_effort stripping, max_tokens clamping),
non-streaming pass-through, the fake-SSE re-emit (the regression guard for
the MacBook bug that dropped tool_calls from the streamed delta), 502
framing on a PhalaTeeClient failure, and the /v1/models catalog. All unit
tests here mock PhalaTeeClient.chat_completion — no network. The one live
integration test is credential-gated and skips without PHALA_API_KEY, same
convention as tests/test_phala_tee.py.
"""
import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agents_core import phala_tee_proxy
from agents_core.phala_tee import PhalaTeeClient, ReasoningContentDecryptionError


def _make_client(mock_response=None, side_effect=None):
    client = MagicMock(spec=PhalaTeeClient)
    if side_effect is not None:
        client.chat_completion.side_effect = side_effect
    else:
        client.chat_completion.return_value = mock_response
    return client


def _app_with(mock_client):
    return TestClient(phala_tee_proxy.create_app(client=mock_client))


# ---------------------------------------------------------------------------
# Request shaping — reasoning_effort stripping, max_tokens clamping.
# ---------------------------------------------------------------------------


def test_reasoning_effort_none_is_stripped_before_call():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert "reasoning_effort" not in kwargs["extra_body"]


def test_reasoning_effort_other_values_pass_through():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["reasoning_effort"] == "high"


def test_max_tokens_defaults_to_ceiling_for_listed_model_when_absent():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["max_tokens"] == 8192


def test_max_tokens_under_ceiling_passes_through_unchanged():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 500,
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["max_tokens"] == 500


def test_max_tokens_over_ceiling_is_clamped_down():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64000,
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["max_tokens"] == 8192


def test_unlisted_model_max_tokens_passes_through_if_present():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999,
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["max_tokens"] == 999


def test_unlisted_model_max_tokens_defaults_to_4096_if_absent():
    mock_client = _make_client(mock_response={
        "id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    })
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    _, kwargs = mock_client.chat_completion.call_args
    assert kwargs["extra_body"]["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Non-streaming pass-through.
# ---------------------------------------------------------------------------


def test_non_streaming_returns_response_directly():
    mock_response = {
        "id": "r1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello there"}, "finish_reason": "stop"}],
    }
    mock_client = _make_client(mock_response=mock_response)
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 200
    assert resp.json() == mock_response


# ---------------------------------------------------------------------------
# Streaming re-emit — regression guard for the dropped-tool_calls bug.
# ---------------------------------------------------------------------------


def _parse_sse_data_lines(text: str) -> list[str]:
    return [line[len("data: "):] for line in text.splitlines() if line.startswith("data: ")]


def test_streaming_forwards_content_tool_calls_and_reasoning_content():
    mock_response = {
        "id": "r1",
        "model": "deepseek/deepseek-v4-flash",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "let me check",
                "reasoning_content": "the user wants weather data",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}},
                ],
            },
        }],
    }
    mock_client = _make_client(mock_response=mock_response)
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "weather?"}],
        "stream": True,
    })
    assert resp.status_code == 200
    data_lines = _parse_sse_data_lines(resp.text)
    assert data_lines[-1] == "[DONE]"

    content_chunk = json.loads(data_lines[0])
    delta = content_chunk["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta["content"] == "let me check"
    assert delta["reasoning_content"] == "the user wants weather data"
    assert delta["tool_calls"][0]["id"] == "call_1"
    assert delta["tool_calls"][0]["index"] == 0

    closing_chunk = json.loads(data_lines[1])
    assert closing_chunk["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------------------
# Error framing — a client failure surfaces as a plain 502, never a broken
# or partially-emitted stream.
# ---------------------------------------------------------------------------


def test_reasoning_content_decryption_error_surfaces_as_502():
    mock_client = _make_client(side_effect=ReasoningContentDecryptionError(0, "bad tag"))
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 502


def test_client_error_surfaces_as_502_for_non_streaming_too():
    mock_client = _make_client(side_effect=RuntimeError("upstream exploded"))
    app = _app_with(mock_client)
    resp = app.post("/v1/chat/completions", json={
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# /v1/models catalog.
# ---------------------------------------------------------------------------


def test_models_catalog_excludes_bad_includes_good():
    mock_client = _make_client()
    app = _app_with(mock_client)
    resp = app.get("/v1/models")
    assert resp.status_code == 200
    ids = {m["id"] for m in resp.json()["data"]}
    assert {"deepseek/deepseek-v3.2", "deepseek/deepseek-v4-flash", "openai/gpt-oss-120b"} <= ids
    assert "google/gemma-4-31b-it" not in ids
    assert "z-ai/glm-5.2" not in ids
    assert "moonshotai/kimi-k2.6" not in ids


# ---------------------------------------------------------------------------
# Live integration test — real network call through the proxy's own
# TestClient; skipped (not failed) when PHALA_API_KEY is unset, same
# convention as tests/test_phala_tee.py's live test.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("PHALA_API_KEY"),
    reason="PHALA_API_KEY not set — live Phala ACI integration test skipped",
)
def test_integration_proxy_live_round_trip():
    """Run locally:  PHALA_API_KEY=... pytest -m integration tests/test_phala_tee_proxy.py"""
    model = os.environ.get("PHALA_MODEL", "deepseek/deepseek-v3.2")
    app = TestClient(phala_tee_proxy.create_app())
    resp = app.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly the word: acknowledged"}],
    })
    assert resp.status_code == 200
    content = resp.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()
