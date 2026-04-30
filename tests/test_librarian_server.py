"""Tests for agents_core.librarian.server — FastAPI sidecar wire format."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agents_core.librarian import LibrarianUnavailable, SynthesisArtifact
from agents_core.librarian.server import _artifact_wire, _shift_for, app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_artifact() -> SynthesisArtifact:
    return SynthesisArtifact(
        schema_version="1",
        artifact_id="sha256:abc",
        claim="hello",
        scope_identity={"corpus_canonical": ["a"], "policy": "auto-update", "format_schema_hash": "sha256:00"},
        request_metadata={"format": None},
        answer={"text": "world"},
        citations=[
            {"path": "/vault/a.md", "content_hash": "sha256:11", "quoted_snippet": "hi"}
        ],
        librarian_id="starhouse-v0",
        model_id="qwen3.6-35b-a3b",
        corpus_snapshot="sha256:99",
        synthesized_at="2026-04-30T12:00:00+00:00",
        verification="full",
        policy="auto-update",
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---------------------------------------------------------------------------
# /v1/corroborate — happy path returns "artifact"
# ---------------------------------------------------------------------------

def test_corroborate_returns_artifact_envelope():
    art = _fake_artifact()
    with patch("agents_core.librarian.server.corroborate", return_value=art), \
         patch("agents_core.librarian.server._shift_for", return_value="no-shift"):
        resp = client.post(
            "/v1/corroborate",
            json={
                "claim": "hello",
                "scope": {"corpus": ["a"]},
                "freshness": 60,
                "policy": "auto-update",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "artifact"
    assert body["artifact_id"] == "sha256:abc"
    assert body["claim"] == "hello"
    assert body["answer"] == {"text": "world"}
    assert body["citations"][0]["path"] == "/vault/a.md"
    assert body["degree_of_shift"] == "no-shift"
    assert body["verification"] == "full"
    assert body["policy"] == "auto-update"


def test_corroborate_passes_through_format_param():
    """`format` in request body must reach corroborate(...)."""
    art = _fake_artifact()
    captured: dict[str, Any] = {}

    def fake_corroborate(claim, scope, *, freshness, policy, format):
        captured["format"] = format
        return art

    with patch("agents_core.librarian.server.corroborate", side_effect=fake_corroborate), \
         patch("agents_core.librarian.server._shift_for", return_value=None):
        resp = client.post(
            "/v1/corroborate",
            json={
                "claim": "q",
                "scope": {"corpus": []},
                "format": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
        )
    assert resp.status_code == 200
    assert captured["format"] == {"type": "object", "properties": {"x": {"type": "string"}}}


# ---------------------------------------------------------------------------
# /v1/corroborate — degraded path returns "unavailable"
# ---------------------------------------------------------------------------

def test_corroborate_unavailable_with_no_cache():
    unavail = LibrarianUnavailable(most_recent_cached=None, reason="LLM down")
    with patch("agents_core.librarian.server.corroborate", return_value=unavail):
        resp = client.post(
            "/v1/corroborate",
            json={"claim": "x", "scope": {"corpus": []}},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "unavailable"
    assert body["most_recent_cached"] is None
    assert body["degraded"] is True
    assert body["reason"] == "LLM down"


def test_corroborate_unavailable_with_cached_artifact():
    cached = _fake_artifact()
    unavail = LibrarianUnavailable(most_recent_cached=cached, reason="LLM timeout")
    with patch("agents_core.librarian.server.corroborate", return_value=unavail), \
         patch("agents_core.librarian.server._shift_for", return_value="paragraph"):
        resp = client.post(
            "/v1/corroborate",
            json={"claim": "x", "scope": {"corpus": ["a"]}},
        )

    body = resp.json()
    assert body["type"] == "unavailable"
    assert body["degraded"] is True
    assert body["reason"] == "LLM timeout"
    cached_wire = body["most_recent_cached"]
    assert cached_wire is not None
    assert cached_wire["artifact_id"] == "sha256:abc"
    assert cached_wire["degree_of_shift"] == "paragraph"
    # Nested cached artifact must NOT carry the discriminator field
    assert "type" not in cached_wire


# ---------------------------------------------------------------------------
# /v1/corroborate — unexpected exception is mapped to unavailable, never 5xx
# ---------------------------------------------------------------------------

def test_corroborate_internal_exception_returns_unavailable():
    with patch(
        "agents_core.librarian.server.corroborate",
        side_effect=RuntimeError("boom"),
    ):
        resp = client.post(
            "/v1/corroborate",
            json={"claim": "x", "scope": {"corpus": []}},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "unavailable"
    assert body["most_recent_cached"] is None
    assert body["degraded"] is True
    assert "boom" in body["reason"]


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

def test_corroborate_rejects_missing_claim():
    resp = client.post(
        "/v1/corroborate",
        json={"scope": {"corpus": []}},
    )
    assert resp.status_code == 422


def test_corroborate_defaults_freshness_and_policy():
    art = _fake_artifact()
    captured: dict[str, Any] = {}

    def fake_corroborate(claim, scope, *, freshness, policy, format):
        captured["freshness"] = freshness
        captured["policy"] = policy
        return art

    with patch("agents_core.librarian.server.corroborate", side_effect=fake_corroborate), \
         patch("agents_core.librarian.server._shift_for", return_value=None):
        resp = client.post(
            "/v1/corroborate",
            json={"claim": "x", "scope": {"corpus": []}},
        )
    assert resp.status_code == 200
    assert captured["freshness"] == 60
    assert captured["policy"] == "auto-update"


# ---------------------------------------------------------------------------
# _artifact_wire helper
# ---------------------------------------------------------------------------

def test_artifact_wire_shape():
    """Wire dict must include all fields claude-view's Rust deserializer expects."""
    art = _fake_artifact()
    with patch("agents_core.librarian.server._shift_for", return_value="word-line"):
        wire = _artifact_wire(art)

    assert set(wire.keys()) == {
        "artifact_id",
        "claim",
        "scope_identity",
        "answer",
        "citations",
        "synthesized_at",
        "degree_of_shift",
        "verification",
        "policy",
    }
    assert wire["degree_of_shift"] == "word-line"


# ---------------------------------------------------------------------------
# _shift_for helper — fault tolerance
# ---------------------------------------------------------------------------

def test_shift_for_returns_none_when_cache_lookup_fails():
    with patch(
        "agents_core.librarian.server.synthesis_cache.get_source_rows",
        side_effect=RuntimeError("cache error"),
    ):
        assert _shift_for("sha256:nope") is None


def test_shift_for_returns_none_for_no_rows():
    with patch(
        "agents_core.librarian.server.synthesis_cache.get_source_rows",
        return_value=[],
    ):
        assert _shift_for("sha256:nope") is None


def test_shift_for_returns_kebab_case_string():
    """ShiftLevel.__str__ converts ``no_shift`` to ``no-shift`` for wire compat."""
    rows = [{"source_path": "/x", "source_content_hash": "h1", "latest_content_hash": "h1"}]
    with patch(
        "agents_core.librarian.server.synthesis_cache.get_source_rows",
        return_value=rows,
    ):
        result = _shift_for("sha256:any")
    assert result == "no-shift"
