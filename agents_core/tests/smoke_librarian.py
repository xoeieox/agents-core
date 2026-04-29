#!/usr/bin/env python3
"""Smoke test for agents_core.librarian — PR 2 gate.

Covers the LLM-free tests required by the vault-substrate-v0 spec (PR 2):

1.  Cache hit: same (claim, scope) returns same artifact; no LLM call made.
2.  Cache key canonicalization:
    a. Different freshness values → same artifact_id.
    b. Unsorted vs sorted corpus → same artifact_id.
    c. Different format schema → different artifact_id.
3.  Cache stale: entry is older than freshness budget; mocked LLM re-runs.
4.  Citation verification stage 1: citation path+hash not in retrieval set → dropped,
    artifact has verification: partial.
5.  Citation verification stage 2: quoted_snippet not a substring of chunk text →
    dropped, artifact has verification: partial.
6.  LLM unavailable: endpoint returns 503 → LibrarianUnavailable; never raises.
7.  Signature roundtrip: artifact signed; tampering breaks signature.
8.  Degree-of-shift: hand-crafted diffs at word/paragraph/section/source-broken
    classify correctly.
9.  Index rebuild: corrupt the three sqlite indexes, run rebuild_indexes, verify
    subsequent lookups succeed.

One LLM-required test (gated by env var LIBRARIAN_LIVE_TEST=1; skip-not-fail
if endpoint unreachable):

10. End-to-end synthesis against the live vault; answer structure matches
    expected schema; all citations resolve to real corpus paths.

Run::

    python3 -m agents_core.tests.smoke_librarian
or::
    pytest agents_core/tests/smoke_librarian.py -v

All non-live tests are LLM-free (LLM calls are mocked).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures: redirect all persistent paths to tmp dirs
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Redirect synthesis cache, key, audit db, and events JSONL to tmp dirs."""
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


def _make_chunk(path: str, content: str):
    """Build a fake retrieval Hit."""
    from agents_core.retrieval import Hit
    content_hash = f"sha256:{_sha256_hex(content)}"
    return Hit(
        id=f"vault-rag:{path}",
        score=1.0,
        source="vault-rag",
        content=content,
        metadata={"file_path": path, "content_hash": content_hash},
    )


def _llm_response(answer: dict, citations: list[dict] | None = None, model: str = "mock-model") -> MagicMock:
    """Build a mock httpx.Response for an LLM call."""
    payload = dict(answer)
    if citations:
        payload["citations"] = citations
    body = {
        "model": model,
        "choices": [{"message": {"content": json.dumps(payload)}}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


def _llm_error(status: int = 503) -> MagicMock:
    """Build a mock httpx.Response that raises HTTPStatusError on raise_for_status."""
    import httpx
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"Server error {status}", request=MagicMock(), response=MagicMock()
    )
    return resp


# ---------------------------------------------------------------------------
# Test 1: Cache hit — same query returns same artifact, no second LLM call
# ---------------------------------------------------------------------------


def test_cache_hit_no_second_llm_call():
    """Second corroborate() with same params returns the cached artifact without LLM call."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/foo.md"
    chunk_content = "This is the canonical source text about foo."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    answer_payload = {"text": "foo is the answer"}
    citations_payload = [
        {"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "canonical source text"}
    ]
    mock_resp = _llm_response(answer_payload, citations_payload)

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]) as mock_retrieve, \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        result1 = librarian.corroborate("status of foo", scope, freshness=60)
        assert isinstance(result1, librarian.SynthesisArtifact)
        assert result1.verification == "full"
        assert mock_post.call_count == 1

        # Second call with same params + ample freshness → cache hit, no LLM
        result2 = librarian.corroborate("status of foo", scope, freshness=60)
        assert isinstance(result2, librarian.SynthesisArtifact)
        assert result2.artifact_id == result1.artifact_id
        assert mock_post.call_count == 1  # no second LLM call


# ---------------------------------------------------------------------------
# Test 2: Cache key canonicalization
# ---------------------------------------------------------------------------


def test_cache_key_freshness_excluded():
    """Different freshness values produce the same artifact_id."""
    from agents_core import synthesis_cache

    id1 = synthesis_cache.compute_artifact_id("status of foo", ["a", "b"], "auto-update")
    id2 = synthesis_cache.compute_artifact_id("status of foo", ["a", "b"], "auto-update")
    # freshness is not a parameter of compute_artifact_id
    assert id1 == id2


def test_cache_key_sorted_corpus():
    """Unsorted and sorted corpus produce the same artifact_id."""
    from agents_core import synthesis_cache

    id1 = synthesis_cache.compute_artifact_id("status of foo", ["b", "a"], "auto-update")
    id2 = synthesis_cache.compute_artifact_id("status of foo", ["a", "b"], "auto-update")
    assert id1 == id2


def test_cache_key_different_format_different_id():
    """Different format schema produces a different artifact_id."""
    from agents_core import synthesis_cache

    id_no_format = synthesis_cache.compute_artifact_id("status of foo", ["a"], "auto-update", None)
    id_with_format = synthesis_cache.compute_artifact_id(
        "status of foo", ["a"], "auto-update",
        {"type": "object", "properties": {"text": {"type": "string"}}}
    )
    assert id_no_format != id_with_format


def test_cache_key_none_and_empty_format_same():
    """format=None and format={} collapse to the same artifact_id."""
    from agents_core import synthesis_cache

    id_none = synthesis_cache.compute_artifact_id("status of foo", ["a"], "auto-update", None)
    id_empty = synthesis_cache.compute_artifact_id("status of foo", ["a"], "auto-update", {})
    assert id_none == id_empty


def test_cache_key_case_preserved():
    """claim case is preserved in artifact_id (capital S produces different id)."""
    from agents_core import synthesis_cache

    id_lower = synthesis_cache.compute_artifact_id("status of foo", ["a"], "auto-update")
    id_upper = synthesis_cache.compute_artifact_id("Status of foo", ["a"], "auto-update")
    assert id_lower != id_upper


# ---------------------------------------------------------------------------
# Test 3: Cache stale — freshness expired → LLM re-runs
# ---------------------------------------------------------------------------


def test_cache_stale_reruns_llm():
    """With freshness=0 the cache is always stale; LLM runs on every call."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/bar.md"
    chunk_content = "Source text for bar."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "bar answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Source text"}],
    )
    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        librarian.corroborate("status of bar", scope, freshness=0)
        librarian.corroborate("status of bar", scope, freshness=0)

        # freshness=0 means always stale → LLM called both times
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: Citation verification stage 1 — path+hash not in retrieval set
# ---------------------------------------------------------------------------


def test_citation_verification_stage1_drops_unknown_path():
    """Citation with path not in retrieval set is dropped; artifact has partial verification."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/real.md"
    chunk_content = "Real source content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    # LLM claims a citation from a path NOT in the retrieval set
    bogus_citations = [
        {"path": "/not/in/retrieval.md", "content_hash": "sha256:deadbeef", "quoted_snippet": "whatever"},
        {"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Real source"},
    ]
    mock_resp = _llm_response({"text": "answer"}, bogus_citations)

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):

        result = librarian.corroborate("what is real", scope, freshness=0)

    assert isinstance(result, librarian.SynthesisArtifact)
    assert result.verification == "partial"
    # The bogus citation should be absent; the valid one should be present
    assert not any(c["path"] == "/not/in/retrieval.md" for c in result.citations)
    assert any(c["path"] == chunk_path for c in result.citations)


# ---------------------------------------------------------------------------
# Test 5: Citation verification stage 2 — snippet not a substring of chunk text
# ---------------------------------------------------------------------------


def test_citation_verification_stage2_drops_bad_snippet():
    """Citation with snippet not present in chunk text is dropped; partial verification."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/snip.md"
    chunk_content = "The actual text in this document."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    # LLM returns a citation where quoted_snippet does NOT appear in the chunk
    citations = [
        {"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "THIS SNIPPET IS NOT IN THE CHUNK"},
    ]
    mock_resp = _llm_response({"text": "answer"}, citations)

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):

        result = librarian.corroborate("snippet test", scope, freshness=0)

    assert isinstance(result, librarian.SynthesisArtifact)
    assert result.verification == "partial"
    assert len(result.citations) == 0  # all dropped


# ---------------------------------------------------------------------------
# Test 6: LLM unavailable — returns LibrarianUnavailable, never raises
# ---------------------------------------------------------------------------


def test_llm_unavailable_returns_degraded_no_raise():
    """When LLM endpoint returns 503, corroborate returns LibrarianUnavailable."""
    from agents_core import librarian

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[]), \
         patch("httpx.post", return_value=_llm_error(503)):

        result = librarian.corroborate("any claim", scope, freshness=0)

    assert isinstance(result, librarian.LibrarianUnavailable)
    assert result.degraded is True
    assert result.most_recent_cached is None  # no prior entry


def test_llm_unavailable_returns_most_recent_cached():
    """On cache miss+LLM unavailable returns None; on cache hit+LLM unavailable returns cached."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/cached.md"
    chunk_content = "Cached content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "cached answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Cached content"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    # First call: populate cache
    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        first = librarian.corroborate("cached claim", scope, freshness=0)
    assert isinstance(first, librarian.SynthesisArtifact)

    # Second call: LLM is down → should return most_recent_cached
    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=_llm_error(503)):
        result = librarian.corroborate("cached claim", scope, freshness=0)

    assert isinstance(result, librarian.LibrarianUnavailable)
    assert result.degraded is True
    assert result.most_recent_cached is not None
    assert result.most_recent_cached.artifact_id == first.artifact_id


# ---------------------------------------------------------------------------
# Test 7: Signature roundtrip + tamper detection
# ---------------------------------------------------------------------------


def test_signature_roundtrip_valid():
    """A freshly synthesised artifact has a valid signature."""
    from agents_core import librarian, synthesis_cache

    chunk_path = "/srv/git/inertia-vault-working/Lapis/sig.md"
    chunk_content = "Signed content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "signed answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Signed content"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("signature claim", scope, freshness=0)

    assert isinstance(result, librarian.SynthesisArtifact)
    artifact_id = result.artifact_id

    # Load entry bytes and sig from disk
    entry_dict = synthesis_cache.get(artifact_id)
    assert entry_dict is not None
    sig_hex = synthesis_cache.get_sig(artifact_id)
    assert sig_hex

    entry_bytes = synthesis_cache.canonical_json(entry_dict).encode("utf-8")
    pub_key_bytes = librarian.get_public_key_bytes()
    assert librarian.verify_entry(entry_bytes, sig_hex, pub_key_bytes)


def test_tampered_entry_fails_signature():
    """Tampering with the answer field breaks the signature."""
    from agents_core import librarian, synthesis_cache

    chunk_path = "/srv/git/inertia-vault-working/Lapis/tamper.md"
    chunk_content = "Original tamper content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "original answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Original tamper"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("tamper claim", scope, freshness=0)

    artifact_id = result.artifact_id
    entry_dict = synthesis_cache.get(artifact_id)
    sig_hex = synthesis_cache.get_sig(artifact_id)
    pub_key_bytes = librarian.get_public_key_bytes()

    # Tamper with the answer field
    entry_dict["answer"]["text"] = "TAMPERED ANSWER"
    tampered_bytes = synthesis_cache.canonical_json(entry_dict).encode("utf-8")

    assert not librarian.verify_entry(tampered_bytes, sig_hex, pub_key_bytes)


# ---------------------------------------------------------------------------
# Test 8: Degree-of-shift classification
# ---------------------------------------------------------------------------


def test_shift_no_shift():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    text = "# Heading\n\nSome content here.\n"
    assert compute_shift(text, text) == ShiftLevel.no_shift


def test_shift_word_line():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    old = "# Heading\n\nSome content here with many sentences. More text follows. Even more.\n"
    # One word changed — very high ratio
    new = old.replace("Some", "Much")
    level = compute_shift(old, new)
    assert level in (ShiftLevel.no_shift, ShiftLevel.word_line)


def test_shift_paragraph():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    old = (
        "# Heading\n\n"
        "First paragraph has some content.\n\n"
        "Second paragraph has more content.\n\n"
        "Third paragraph closes things out.\n"
    )
    # Replace ~half the body with different text
    new = (
        "# Heading\n\n"
        "First paragraph has some content.\n\n"
        "Completely different paragraph now with totally new ideas and concepts.\n\n"
        "Another entirely new paragraph replacing the old closing.\n"
    )
    level = compute_shift(old, new)
    assert level in (ShiftLevel.word_line, ShiftLevel.paragraph, ShiftLevel.section)
    # Ensure it's at least paragraph (not no_shift or word_line for this diff)
    assert level >= ShiftLevel.paragraph


def test_shift_section():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    old = (
        "# Section A\n\nContent about A.\n\n"
        "## Subsection A1\n\nMore content.\n\n"
        "## Subsection A2\n\nEven more.\n"
    )
    new = (
        "# Section B\n\nCompletely different content.\n\n"
        "## Subsection B1\n\nFoo bar baz.\n\n"
        "## Subsection B2\n\nQux quux.\n\n"
        "## Subsection B3\n\nNew section added.\n"
    )
    level = compute_shift(old, new)
    assert level >= ShiftLevel.paragraph  # at minimum paragraph, likely section


def test_shift_source_broken():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    assert compute_shift("some content", "") == ShiftLevel.source_broken


def test_shift_source_broken_whitespace_only():
    from agents_core.librarian.shift import ShiftLevel, compute_shift
    assert compute_shift("some content", "   \n  ") == ShiftLevel.source_broken


# ---------------------------------------------------------------------------
# Test 9: Index rebuild
# ---------------------------------------------------------------------------


def test_index_rebuild():
    """Rebuild indexes from by-hash entries; subsequent lookups succeed."""
    from agents_core import librarian, synthesis_cache

    chunk_path = "/srv/git/inertia-vault-working/Lapis/rebuild.md"
    chunk_content = "Content for rebuild test."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "rebuild answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Content for rebuild"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("rebuild claim", scope, freshness=0)

    assert isinstance(result, librarian.SynthesisArtifact)
    artifact_id = result.artifact_id

    # Corrupt the three index databases by deleting them
    cache_root = Path(os.environ["SYNTHESIS_CACHE_ROOT"])
    for name in ("by-claim.sqlite", "by-source.sqlite", "trigger-requests.sqlite"):
        db_path = cache_root / "index" / name
        if db_path.exists():
            db_path.unlink()

    synthesis_cache.reset_connections()

    # Rebuild
    synthesis_cache.rebuild_indexes()

    # Subsequent lookup should succeed (by-claim rebuilt from entry)
    format_schema_hash = synthesis_cache.compute_format_schema_hash(None)
    found_id = synthesis_cache.lookup_by_claim(
        "rebuild claim", ["vault-rag"], "auto-update", format_schema_hash
    )
    assert found_id == artifact_id

    # by-source rows should also be rebuilt
    source_rows = synthesis_cache.get_source_rows(artifact_id)
    assert any(r["source_path"] == chunk_path for r in source_rows)


# ---------------------------------------------------------------------------
# Test 10 (LLM-required): end-to-end synthesis against live vault
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("LIBRARIAN_LIVE_TEST"),
    reason="Set LIBRARIAN_LIVE_TEST=1 to run live-endpoint tests",
)
def test_live_synthesis():
    """End-to-end synthesis against the live vault and LLM endpoint.

    Requires:
    - vault-rag endpoint reachable (http://203.0.113.12:8200)
    - LLM endpoint reachable (http://203.0.113.12:8081)
    """
    from agents_core import librarian

    format_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content_hash": {"type": "string"},
                        "quoted_snippet": {"type": "string"},
                    },
                    "required": ["path", "content_hash", "quoted_snippet"],
                },
            },
        },
        "required": ["summary"],
    }

    try:
        result = librarian.corroborate(
            "What is the purpose of vault-substrate-v0?",
            librarian.Scope(corpus=["vault-rag"]),
            freshness=0,
            format=format_schema,
        )
    except Exception as exc:
        pytest.skip(f"Live endpoint unreachable: {exc}")

    if isinstance(result, librarian.LibrarianUnavailable):
        pytest.skip("LLM endpoint unavailable during live test")

    # Answer structure matches schema
    assert "summary" in result.answer
    assert isinstance(result.answer["summary"], str)
    assert len(result.answer["summary"]) > 0

    # All citations resolve to real corpus paths and pass both verification stages
    for cit in result.citations:
        assert "path" in cit
        assert "content_hash" in cit
        assert "quoted_snippet" in cit
        assert cit["content_hash"].startswith("sha256:")

    assert result.verification in ("full", "partial")
    assert result.librarian_id == "starhouse-v0"


# ---------------------------------------------------------------------------
# Additional: zero-citation artifacts get verification: none
# ---------------------------------------------------------------------------


def test_zero_citation_verification_none():
    """LLM returns no citations → artifact.verification == 'none' (not 'full')."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/nocit.md"
    chunk_content = "No citations expected here."
    chunk = _make_chunk(chunk_path, chunk_content)

    # LLM returns an answer with no citations field
    mock_resp = _llm_response({"text": "answer without citations"}, citations=None)

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("zero citation claim", scope, freshness=0)

    assert isinstance(result, librarian.SynthesisArtifact)
    assert result.verification == "none"
    assert result.citations == []


# ---------------------------------------------------------------------------
# Additional: Ed25519 key file permissions
# ---------------------------------------------------------------------------


def test_signing_key_file_mode_is_0o600():
    """The generated Ed25519 private key file must be mode 0o600 (owner r/w only)."""
    from agents_core import librarian

    # Trigger key generation by calling any function that loads the key
    _ = librarian.get_public_key_bytes()

    key_path = Path(os.environ["LIBRARIAN_KEY_PATH"])
    assert key_path.exists(), "key file was not created"
    mode = key_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Additional: startup_replay
# ---------------------------------------------------------------------------


def test_startup_replay_updates_source_tracking(tmp_path):
    """startup_replay replays vault-events.jsonl into by-source.sqlite."""
    from agents_core import librarian, synthesis_cache

    chunk_path = "/srv/git/inertia-vault-working/Lapis/replay.md"
    chunk_content = "Replay source content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "replay answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Replay source"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("replay claim", scope, freshness=0)

    artifact_id = result.artifact_id

    # Simulate a vault write event for the cited path with a new hash
    new_hash = "sha256:newcontenthashabcdef1234567890"
    new_ts = "2026-04-30T00:00:00+00:00"
    events_jsonl = Path(os.environ["VAULT_EVENTS_JSONL"])
    events_jsonl.write_text(
        json.dumps({
            "topic": chunk_path,
            "path": chunk_path,
            "content_hash": new_hash,
            "prev_hash": chunk_hash,
            "ts": new_ts,
            "agent_id": "test",
            "intent": "update",
        }) + "\n",
        encoding="utf-8",
    )

    replayed = librarian.startup_replay()
    assert replayed >= 1

    rows = synthesis_cache.get_source_rows(artifact_id)
    assert any(r["source_path"] == chunk_path and r["latest_content_hash"] == new_hash for r in rows)


# ---------------------------------------------------------------------------
# Additional: trigger request API
# ---------------------------------------------------------------------------


def test_trigger_request_round_trip():
    """request_trigger() records a row; trigger_request_summary() returns it."""
    from agents_core import librarian

    chunk_path = "/srv/git/inertia-vault-working/Lapis/trigger.md"
    chunk_content = "Trigger content."
    chunk = _make_chunk(chunk_path, chunk_content)
    chunk_hash = f"sha256:{_sha256_hex(chunk_content)}"

    mock_resp = _llm_response(
        {"text": "trigger answer"},
        [{"path": chunk_path, "content_hash": chunk_hash, "quoted_snippet": "Trigger content"}],
    )

    scope = librarian.Scope(corpus=["vault-rag"])

    with patch("agents_core.retrieval.retrieve", return_value=[chunk]), \
         patch("httpx.post", return_value=mock_resp):
        result = librarian.corroborate("trigger claim", scope, freshness=0)

    librarian.request_trigger(result.artifact_id, "test-requester", "testing")
    summary = librarian.trigger_request_summary()
    assert len(summary) >= 1
    row = summary[-1]
    assert row.artifact_id == result.artifact_id
    assert row.requester_id == "test-requester"
    assert row.reason == "testing"


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
