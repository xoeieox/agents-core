"""Tests for projector routing inside corroborate().

Covers the spec requirements:
- Registered claim short-circuits LLM (no httpx.post call)
- Unregistered claim still hits LLM
- Projector artifact cached and reused on second call
- Stale projector artifact re-runs the projector
- Projector failure propagates (no silent LLM substitution)
- request_metadata.projector == True in cached entry
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures: redirect all persistent paths to tmp dirs
# (mirrors smoke_librarian.py autouse fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    cache_root = tmp_path / "synthesis-cache"
    key_path = tmp_path / "librarian.key"
    audit_db = tmp_path / "vault-audit.db"
    events_jsonl = tmp_path / "vault-events.jsonl"

    monkeypatch.setenv("SYNTHESIS_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("LIBRARIAN_KEY_PATH", str(key_path))
    monkeypatch.setenv("VAULT_AUDIT_DB", str(audit_db))
    monkeypatch.setenv("VAULT_EVENTS_JSONL", str(events_jsonl))

    import agents_core.synthesis_cache as sc
    import agents_core.vault_audit as va
    sc.reset_connections()
    va.reset_connection()
    from agents_core import librarian
    librarian._reset_signing_key()

    yield

    sc.reset_connections()
    va.reset_connection()
    librarian._reset_signing_key()


# ---------------------------------------------------------------------------
# Helper: stub LLM response
# ---------------------------------------------------------------------------

def _llm_ok(answer: dict | None = None) -> MagicMock:
    payload = answer or {"text": "llm answer"}
    body = {
        "model": "mock",
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Test: registered claim short-circuits LLM
# ---------------------------------------------------------------------------

def test_registered_claim_short_circuits_llm():
    """Patching _call_llm to raise; registered claim must still return authoritative artifact."""
    from agents_core import librarian
    from agents_core.librarian.projectors import REGISTRY

    stub_answer = {"targets": [], "generated_at": "2026-01-01T00:00:00+00:00"}
    stub_projector = MagicMock(return_value=stub_answer)

    with patch.dict(REGISTRY, {"current-targets-state": stub_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("LLM must not be called")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):
        result = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=0,
        )

    assert isinstance(result, librarian.SynthesisArtifact)
    assert result.verification == "authoritative"
    assert result.model_id == "projector:current-targets-state"
    stub_projector.assert_called_once()


# ---------------------------------------------------------------------------
# Test: unregistered claim still hits LLM
# ---------------------------------------------------------------------------

def test_unregistered_claim_hits_llm():
    """Unregistered claim falls through to _synthesize → LLM called."""
    from agents_core import librarian

    with patch("agents_core.retrieval.retrieve", return_value=[]), \
         patch("httpx.post", return_value=_llm_ok()) as mock_post:
        result = librarian.corroborate(
            "some unregistered claim that is definitely not in the registry",
            librarian.Scope(corpus=[]),
            freshness=0,
        )

    assert mock_post.called, "LLM should have been called for an unregistered claim"
    assert isinstance(result, librarian.SynthesisArtifact)


# ---------------------------------------------------------------------------
# Test: projector artifact cached and reused on second call
# ---------------------------------------------------------------------------

def test_projector_artifact_cached_and_reused():
    """Back-to-back calls within freshness window: projector invoked exactly once."""
    from agents_core import librarian
    from agents_core.librarian.projectors import REGISTRY

    call_count = [0]

    def counting_projector(scope, freshness, policy):
        call_count[0] += 1
        return {"targets": [], "generated_at": "2026-01-01T00:00:00+00:00"}

    with patch.dict(REGISTRY, {"current-targets-state": counting_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("no LLM")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):

        result1 = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=3600,
        )
        result2 = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=3600,
        )

    assert isinstance(result1, librarian.SynthesisArtifact)
    assert isinstance(result2, librarian.SynthesisArtifact)
    assert result1.artifact_id == result2.artifact_id
    assert call_count[0] == 1  # projector called only once


# ---------------------------------------------------------------------------
# Test: stale projector artifact re-runs the projector
# ---------------------------------------------------------------------------

def test_stale_projector_artifact_reruns_projector():
    """Entry with synthesized_at far in the past → projector re-invoked on second call."""
    from agents_core import librarian, synthesis_cache
    from agents_core.librarian.projectors import REGISTRY

    call_count = [0]

    def counting_projector(scope, freshness, policy):
        call_count[0] += 1
        return {"targets": [], "generated_at": "2026-01-01T00:00:00+00:00"}

    with patch.dict(REGISTRY, {"current-targets-state": counting_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("no LLM")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):
        result1 = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=60,
        )

    assert call_count[0] == 1

    # Overwrite the cached entry with a stale synthesized_at
    artifact_id = result1.artifact_id
    entry = synthesis_cache.get(artifact_id)
    assert entry is not None
    entry["synthesized_at"] = "2020-01-01T00:00:00+00:00"  # 6 years ago
    synthesis_cache.put(entry, "")

    with patch.dict(REGISTRY, {"current-targets-state": counting_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("no LLM")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):
        result2 = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=60,  # 60s budget, entry is 6y old → stale
        )

    assert isinstance(result2, librarian.SynthesisArtifact)
    assert call_count[0] == 2  # projector re-ran


# ---------------------------------------------------------------------------
# Test: projector failure propagates — no silent LLM fallback
# ---------------------------------------------------------------------------

def test_projector_failure_propagates():
    """A broken projector propagates its exception — no silent LLM substitution."""
    from agents_core import librarian
    from agents_core.librarian.projectors import REGISTRY

    def broken_projector(scope, freshness, policy):
        raise RuntimeError("projector is broken")

    with patch.dict(REGISTRY, {"current-targets-state": broken_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("no LLM")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):
        with pytest.raises(RuntimeError, match="projector is broken"):
            librarian.corroborate(
                "current-targets-state",
                librarian.Scope(corpus=[]),
                freshness=0,
            )


# ---------------------------------------------------------------------------
# Test: request_metadata.projector == True
# ---------------------------------------------------------------------------

def test_projector_request_metadata_flag():
    """Cached projector entry has request_metadata['projector'] == True."""
    from agents_core import librarian, synthesis_cache
    from agents_core.librarian.projectors import REGISTRY

    stub_answer = {"targets": [], "generated_at": "2026-01-01T00:00:00+00:00"}
    stub_projector = MagicMock(return_value=stub_answer)

    with patch.dict(REGISTRY, {"current-targets-state": stub_projector}), \
         patch("agents_core.librarian._call_llm", side_effect=RuntimeError("no LLM")), \
         patch("agents_core.retrieval.retrieve", return_value=[]):
        result = librarian.corroborate(
            "current-targets-state",
            librarian.Scope(corpus=[]),
            freshness=0,
        )

    entry = synthesis_cache.get(result.artifact_id)
    assert entry is not None
    assert entry["request_metadata"]["projector"] is True
    assert entry["corpus_snapshot"] == "projector"
    assert entry["citations"] == []
