"""Smoke tests for agents_core.keeper_query — keeper-query-cli-v0 gate.

Covers (all LLM-free unless noted):

1.  Low-stakes retrieval-only: returns hits, no LLM call.
2.  High-stakes synthesis: calls corroborate(), returns artifact.
3.  Auto routing — lookup → low: short keyword query routes to low.
4.  Auto routing — synthesis → high: "why/compare" query routes to high.
5.  Empty-result fallback: scoped query returns nothing, retries with full corpus.
6.  Degraded path: LLM endpoint down → exit code 2, [DEGRADED] banner, no traceback.
7.  --dry-run: prints scope/stakes, no LLM call.

Run::

    python3 -m agents_core.tests.smoke_keeper_query
or::
    pytest agents_core/tests/smoke_keeper_query.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures: redirect persistent paths to tmp dirs (mirrors smoke_librarian)
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
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _make_hit(path: str, content: str, score: float = 0.9):
    from agents_core.retrieval import Hit
    return Hit(
        id=f"vault-rag:{path}",
        score=score,
        source="vault-rag",
        content=content,
        metadata={"file_path": path},
    )


def _llm_response(answer: dict, citations: list[dict] | None = None) -> MagicMock:
    payload = dict(answer)
    if citations:
        payload["citations"] = citations
    body = {
        "model": "mock-model",
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _llm_503() -> MagicMock:
    import httpx
    resp = MagicMock()
    resp.status_code = 503
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error 503", request=MagicMock(), response=MagicMock()
    )
    return resp


# ---------------------------------------------------------------------------
# Test 1: Low-stakes retrieval-only — no LLM call
# ---------------------------------------------------------------------------


def test_low_stakes_no_llm_call(capsys):
    """--stakes low returns ranked hits without calling the LLM endpoint."""
    from agents_core import keeper_query

    hit = _make_hit("/vault/compost.md", "The compost invariant says nothing is wasted.")

    with patch("agents_core.retrieval.retrieve", return_value=[hit]) as mock_retrieve, \
         patch("httpx.post") as mock_post:

        code = keeper_query.run(
            "compost invariant",
            stakes="low",
            top_k=8,
        )

    assert code == 0
    assert mock_post.call_count == 0, "LLM must not be called for low stakes"
    out = capsys.readouterr().out
    assert "compost.md" in out or "compost invariant" in out.lower()


# ---------------------------------------------------------------------------
# Test 2: High-stakes synthesis — calls corroborate()
# ---------------------------------------------------------------------------


def test_high_stakes_synthesis(capsys):
    """--stakes high returns synthesized answer with citations."""
    from agents_core import keeper_query

    path = "/vault/compost.md"
    content = "The compost invariant means all failed work feeds the next cycle."
    hit = _make_hit(path, content)
    chunk_hash = f"sha256:{_sha256_hex(content)}"
    mock_resp = _llm_response(
        {"text": "Compost invariant: failed work is not wasted, it feeds future cycles."},
        [{"path": path, "content_hash": chunk_hash, "quoted_snippet": "compost invariant means"}],
    )

    with patch("agents_core.retrieval.retrieve", return_value=[hit]), \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        code = keeper_query.run(
            "what is the compost invariant",
            stakes="high",
            freshness=0,
        )

    assert code == 0
    assert mock_post.call_count >= 1, "LLM must be called for high stakes"
    out = capsys.readouterr().out
    assert "Compost" in out or "compost" in out
    assert "Verification:" in out


# ---------------------------------------------------------------------------
# Test 3: Auto routing — lookup query → low
# ---------------------------------------------------------------------------


def test_auto_routes_lookup_to_low(capsys):
    """A short keyword lookup routes auto to low stakes."""
    from agents_core import keeper_query

    # Clear top hit with large score spread → should route low
    hits = [
        _make_hit("/vault/a.md", "content a", score=0.95),
        _make_hit("/vault/b.md", "content b", score=0.40),
    ]

    with patch("agents_core.retrieval.retrieve", return_value=hits), \
         patch("httpx.post") as mock_post:

        code = keeper_query.run("compost invariant", stakes="auto", top_k=8)

    assert code == 0
    # Low-stakes routing → no LLM call
    assert mock_post.call_count == 0
    out = capsys.readouterr().out
    assert "[auto → low:" in out


# ---------------------------------------------------------------------------
# Test 4: Auto routing — synthesis query → high
# ---------------------------------------------------------------------------


def test_auto_routes_synthesis_to_high(capsys):
    """A 'why/compare' shaped question routes auto to high stakes."""
    from agents_core import keeper_query

    path = "/vault/why.md"
    content = "Compost matters because it closes the resource loop."
    hit = _make_hit(path, content)
    chunk_hash = f"sha256:{_sha256_hex(content)}"
    mock_resp = _llm_response(
        {"text": "Compost closes resource loops and reduces waste."},
        [{"path": path, "content_hash": chunk_hash, "quoted_snippet": "closes the resource loop"}],
    )

    with patch("agents_core.retrieval.retrieve", return_value=[hit]), \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        code = keeper_query.run(
            "why does the compost invariant matter",
            stakes="auto",
            freshness=0,
        )

    assert code == 0
    assert mock_post.call_count >= 1, "auto → high should call LLM"
    out = capsys.readouterr().out
    assert "[auto → high:" in out


# ---------------------------------------------------------------------------
# Test 5: Empty-result fallback fires and is surfaced to user
# ---------------------------------------------------------------------------


def test_at_full_corpus_no_results_shows_not_found(capsys):
    """When already querying full corpus and retrieval returns nothing, exit 0 and report clean empty."""
    from agents_core import keeper_query

    with patch("agents_core.retrieval.retrieve", return_value=[]), \
         patch("httpx.post") as mock_post:

        code = keeper_query.run(
            "obscure topic",
            stakes="low",
            corpus=["vault-rag"],  # already the full corpus — fallback cannot fire
            top_k=8,
        )

    assert code == 0
    assert mock_post.call_count == 0
    out = capsys.readouterr().out
    assert "No results" in out
    # The fallback banner must NOT appear — already at full corpus, nothing to fall back to
    assert "fell back" not in out


def test_empty_result_fallback_message_shown(capsys):
    """When fallback triggers, the user-facing downgrade message is printed."""
    from agents_core import keeper_query

    path = "/vault/fallback.md"
    content = "Full corpus fallback content here."
    fallback_hit = _make_hit(path, content)

    call_count = {"n": 0}

    def _mock_retrieve(query, scopes, **kwargs):
        call_count["n"] += 1
        # Simulate narrowed scope returning nothing, full scope returning a hit
        if "room-rag" in scopes:
            return []
        return [fallback_hit]

    with patch("agents_core.retrieval.retrieve", side_effect=_mock_retrieve), \
         patch("httpx.post") as mock_post:

        # Use a "narrower" corpus token that maps to a different scope
        # We'll patch _corpus_to_retrieval_scopes to return room-rag for the custom token
        def _fake_scope_map(corpus):
            if corpus == ["room-rag"]:
                return ["room-rag"]
            return ["vault-rag"]

        with patch("agents_core.keeper_query._resolve_retrieval_scopes", side_effect=_fake_scope_map):
            code = keeper_query.run(
                "obscure topic",
                stakes="low",
                corpus=["room-rag"],
                top_k=8,
            )

    assert code == 0
    out = capsys.readouterr().out
    assert "fell back to full corpus" in out


# ---------------------------------------------------------------------------
# Test 6: Degraded path — LLM down → exit 2, [DEGRADED] banner
# ---------------------------------------------------------------------------


def test_degraded_llm_unavailable(capsys):
    """When LLM endpoint is down, exit code is 2 and [DEGRADED] banner is printed."""
    from agents_core import keeper_query

    with patch("agents_core.retrieval.retrieve", return_value=[]), \
         patch("httpx.post", return_value=_llm_503()):

        code = keeper_query.run(
            "what is the compost invariant",
            stakes="high",
            freshness=0,
        )

    assert code == 2
    out = capsys.readouterr().out
    assert "[DEGRADED" in out


def test_degraded_serves_cached_on_llm_down(capsys):
    """When LLM goes down after a warm cache, the cached answer is served with [DEGRADED]."""
    from agents_core import keeper_query

    path = "/vault/cached.md"
    content = "Cached synthesis content."
    hit = _make_hit(path, content)
    chunk_hash = f"sha256:{_sha256_hex(content)}"

    # First: populate cache via a successful synthesis
    good_resp = _llm_response(
        {"text": "Cached answer about compost."},
        [{"path": path, "content_hash": chunk_hash, "quoted_snippet": "Cached synthesis"}],
    )
    with patch("agents_core.retrieval.retrieve", return_value=[hit]), \
         patch("httpx.post", return_value=good_resp):
        code1 = keeper_query.run(
            "compost invariant cached",
            stakes="high",
            freshness=0,
        )
    assert code1 == 0

    # Second: LLM goes down
    with patch("agents_core.retrieval.retrieve", return_value=[hit]), \
         patch("httpx.post", return_value=_llm_503()):
        code2 = keeper_query.run(
            "compost invariant cached",
            stakes="high",
            freshness=0,
        )

    assert code2 == 2
    out = capsys.readouterr().out
    assert "[DEGRADED" in out
    assert "Cached answer" in out


# ---------------------------------------------------------------------------
# Test 7: --dry-run prints scope/stakes, no LLM call
# ---------------------------------------------------------------------------


def test_dry_run_no_llm_call(capsys):
    """--dry-run prints resolved scope + stakes without calling the LLM."""
    from agents_core import keeper_query

    with patch("agents_core.retrieval.retrieve", return_value=[]) as mock_retrieve, \
         patch("httpx.post") as mock_post:

        code = keeper_query.run(
            "compost invariant",
            stakes="auto",
            dry_run=True,
        )

    assert code == 0
    assert mock_post.call_count == 0
    out = capsys.readouterr().out
    assert "stakes" in out.lower() or "corpus" in out.lower()


def test_dry_run_json_output(capsys):
    """--dry-run --json emits valid JSON with required fields."""
    from agents_core import keeper_query

    with patch("agents_core.retrieval.retrieve", return_value=[]), \
         patch("httpx.post"):

        code = keeper_query.run(
            "compost invariant",
            stakes="high",
            dry_run=True,
            emit_json=True,
        )

    assert code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "corpus" in data
    assert "stakes_decision" in data
    assert "reason" in data
    assert data["stakes_decision"] == "high"


# ---------------------------------------------------------------------------
# Test 8: decide_stakes heuristic
# ---------------------------------------------------------------------------


def test_decide_stakes_low_clear_hit():
    """Short keyword query with clear top hit → low."""
    from agents_core.keeper_query import decide_stakes
    from agents_core.retrieval import Hit

    hits = [
        Hit(id="vault-rag:a", score=0.95, source="vault-rag", content="content a", metadata={}),
        Hit(id="vault-rag:b", score=0.40, source="vault-rag", content="content b", metadata={}),
    ]
    stakes, reason = decide_stakes("compost invariant", hits)
    assert stakes == "low"
    assert "low" in reason


def test_decide_stakes_high_why_marker():
    """Question with 'why' → high."""
    from agents_core.keeper_query import decide_stakes
    from agents_core.retrieval import Hit

    hits = [Hit(id="vault-rag:a", score=0.9, source="vault-rag", content="x", metadata={})]
    stakes, reason = decide_stakes("why does the compost invariant matter", hits)
    assert stakes == "high"
    assert "why" in reason


def test_decide_stakes_high_compare_marker():
    """Question with 'compare' → high."""
    from agents_core.keeper_query import decide_stakes
    from agents_core.retrieval import Hit

    hits = [Hit(id="vault-rag:a", score=0.9, source="vault-rag", content="x", metadata={})]
    stakes, reason = decide_stakes("compare lapis and inertia", hits)
    assert stakes == "high"


def test_decide_stakes_high_ambiguous_spread():
    """Close score spread → high regardless of question shape."""
    from agents_core.keeper_query import decide_stakes
    from agents_core.retrieval import Hit

    hits = [
        Hit(id="vault-rag:a", score=0.85, source="vault-rag", content="x", metadata={}),
        Hit(id="vault-rag:b", score=0.82, source="vault-rag", content="y", metadata={}),
    ]
    stakes, reason = decide_stakes("compost", hits)
    assert stakes == "high"
    assert "spread" in reason


def test_decide_stakes_high_no_hits():
    """Zero retrieval hits → high."""
    from agents_core.keeper_query import decide_stakes

    stakes, reason = decide_stakes("some query", [])
    assert stakes == "high"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent.parent,
    )
    sys.exit(result.returncode)
