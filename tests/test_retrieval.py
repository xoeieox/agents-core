"""Tests for agents_core.retrieval — Synapse Retrieval Engine.

Unit tests use mocks; no real backend calls.
Integration smoke (``@pytest.mark.integration``) hits mem + chub + vault-rag for real.

Run integration tests locally:
    pytest -m integration tests/test_retrieval.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents_core.retrieval import Hit, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hit(source: str, key: str, score: float) -> Hit:
    return Hit(id=f"{source}:{key}", score=score, source=source, content=f"content-{key}",
               metadata={})


# ---------------------------------------------------------------------------
# Invariant: empty scope
# ---------------------------------------------------------------------------

def test_empty_scope_returns_empty_no_calls():
    with patch("agents_core.retrieval._search_mem") as m_mem, \
         patch("agents_core.retrieval._search_chub") as m_chub, \
         patch("agents_core.retrieval._search_rag") as m_rag:
        result = retrieve("anything", scope=[])
    assert result == []
    m_mem.assert_not_called()
    m_chub.assert_not_called()
    m_rag.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant: unknown scope token
# ---------------------------------------------------------------------------

def test_unknown_scope_raises_value_error():
    with pytest.raises(ValueError, match="Unknown scope token"):
        retrieve("query", scope=["mem", "no-such-backend"])


# ---------------------------------------------------------------------------
# Routing: single backend
# ---------------------------------------------------------------------------

def test_scope_mem_only_calls_mem():
    fake_hits = [_hit("mem", "k1", 0.9)]
    with patch("agents_core.retrieval._search_mem", return_value=fake_hits) as m_mem, \
         patch("agents_core.retrieval._search_chub") as m_chub, \
         patch("agents_core.retrieval._search_rag") as m_rag:
        result = retrieve("query", scope=["mem"])
    assert result == fake_hits
    m_mem.assert_called_once()
    m_chub.assert_not_called()
    m_rag.assert_not_called()


# ---------------------------------------------------------------------------
# Merge + re-sort across two backends
# ---------------------------------------------------------------------------

def test_scope_mem_chub_merges_and_sorts_by_score():
    mem_hits  = [_hit("mem",  "a", 0.4), _hit("mem",  "b", 0.9)]
    chub_hits = [_hit("chub", "x", 0.7), _hit("chub", "y", 0.2)]
    with patch("agents_core.retrieval._search_mem", return_value=mem_hits), \
         patch("agents_core.retrieval._search_chub", return_value=chub_hits):
        result = retrieve("query", scope=["mem", "chub"])

    assert [h.id for h in result] == ["mem:b", "chub:x", "mem:a", "chub:y"]


# ---------------------------------------------------------------------------
# top_k caps result count
# ---------------------------------------------------------------------------

def test_top_k_caps_results():
    hits = [_hit("mem", str(i), float(i) / 10) for i in range(8)]
    with patch("agents_core.retrieval._search_mem", return_value=hits):
        result = retrieve("q", scope=["mem"], top_k=3)
    assert len(result) == 3
    # highest scores come first
    assert result[0].score >= result[1].score >= result[2].score


# ---------------------------------------------------------------------------
# min_score filters below-threshold hits
# ---------------------------------------------------------------------------

def test_min_score_filters():
    hits = [_hit("mem", "hi", 0.8), _hit("mem", "lo", 0.1)]
    with patch("agents_core.retrieval._search_mem", return_value=hits):
        result = retrieve("q", scope=["mem"], min_score=0.5)
    assert len(result) == 1
    assert result[0].id == "mem:hi"


# ---------------------------------------------------------------------------
# exclude removes specified Hit.id values
# ---------------------------------------------------------------------------

def test_exclude_removes_hits():
    hits = [_hit("mem", "xyz", 0.9), _hit("mem", "abc", 0.8)]
    with patch("agents_core.retrieval._search_mem", return_value=hits):
        result = retrieve("q", scope=["mem"], exclude={"mem:xyz"})
    assert len(result) == 1
    assert result[0].id == "mem:abc"


# ---------------------------------------------------------------------------
# filters flow-through: tags → mem, tags key ignored by chub
# ---------------------------------------------------------------------------

def test_filters_tags_flow_to_mem_ignored_by_chub(caplog):
    """tags filter is consumed by mem; chub ignores it without error and logs DEBUG."""
    import json
    import logging
    from subprocess import CompletedProcess

    mem_hits = [_hit("mem", "m1", 0.9)]
    # chub subprocess returns one result so _search_chub runs real code paths
    chub_json = json.dumps({
        "results": [{"id": "docs/x", "description": "desc", "_score": 1.0,
                     "_type": "doc", "name": "x", "tags": [], "_source": "c"}]
    })

    captured_mem_filters: list[dict] = []

    def fake_mem(query, filters):
        captured_mem_filters.append(filters)
        return mem_hits

    with patch("agents_core.retrieval._search_mem", side_effect=fake_mem), \
         patch("subprocess.run",
               return_value=CompletedProcess(args=[], returncode=0, stdout=chub_json)):
        with caplog.at_level(logging.DEBUG, logger="agents_core.retrieval"):
            result = retrieve("q", scope=["mem", "chub"], filters={"tags": ["foo"]})

    assert captured_mem_filters[0]["tags"] == ["foo"]
    # chub logs at DEBUG about unrecognised key "tags"
    assert any("tags" in r.message for r in caplog.records if r.levelno == logging.DEBUG)
    # both backends contributed results
    sources = {h.source for h in result}
    assert "mem" in sources
    assert "chub" in sources


# ---------------------------------------------------------------------------
# *-rag payload adapter: real pagination field (n_results, not limit) + doc_type forwarding
# ---------------------------------------------------------------------------

def test_search_rag_sends_n_results_not_limit():
    """RoomRAG's SearchRequest schema field is n_results; the old 'limit' key was a no-op."""
    import json as _json
    from unittest.mock import MagicMock

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"results": []}
        return resp

    with patch("httpx.post", side_effect=fake_post):
        retrieve("q", scope=["room-rag"], top_k=25)

    assert "limit" not in captured["payload"]
    assert captured["payload"]["n_results"] == 25


def test_search_rag_forwards_doc_type_and_not_warned_as_unknown(caplog):
    import logging
    from unittest.mock import MagicMock

    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"results": [{"file_path": "a", "score": 1.0, "doc_type": "podcast-transcript"}]}
        return resp

    with patch("httpx.post", side_effect=fake_post):
        with caplog.at_level(logging.DEBUG, logger="agents_core.retrieval"):
            result = retrieve(
                "q", scope=["room-rag"], filters={"doc_type": "podcast-transcript"}, top_k=10
            )

    assert captured["payload"]["doc_type"] == "podcast-transcript"
    assert result and result[0].source == "room-rag"
    assert not any("doc_type" in r.message for r in caplog.records if "ignoring unrecognised" in r.message)


def test_search_rag_canary_warns_when_doc_type_appears_ignored(caplog):
    """If the response contains a doc_type other than the one requested, warn loudly."""
    import logging
    from unittest.mock import MagicMock

    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "results": [{"file_path": "a", "score": 1.0, "doc_type": "vault-note"}]
        }
        return resp

    with patch("httpx.post", side_effect=fake_post):
        with caplog.at_level(logging.WARNING, logger="agents_core.retrieval"):
            retrieve("q", scope=["room-rag"], filters={"doc_type": "podcast-transcript"})

    assert any("IGNORING doc_type" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# HTTP backend fail-soft: timeout → [] for that backend; others still contribute
# ---------------------------------------------------------------------------

def test_rag_timeout_is_failsoft():
    import httpx

    mem_hits = [_hit("mem", "k1", 0.8)]

    def raise_timeout(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out")

    with patch("agents_core.retrieval._search_mem", return_value=mem_hits), \
         patch("agents_core.retrieval._search_chub", return_value=[]), \
         patch("httpx.post", side_effect=raise_timeout):
        # vault-rag will timeout; mem hits must still come through
        result = retrieve("q", scope=["mem", "vault-rag"])

    assert any(h.source == "mem" for h in result)
    # vault-rag contributed nothing but did not crash the call
    assert all(h.source != "vault-rag" for h in result)


def test_rag_connection_error_is_failsoft():
    import httpx

    chub_hits = [_hit("chub", "c1", 0.7)]

    def raise_conn_error(*args, **kwargs):
        raise httpx.ConnectError("refused")

    with patch("agents_core.retrieval._search_mem", return_value=[]), \
         patch("agents_core.retrieval._search_chub", return_value=chub_hits), \
         patch("httpx.post", side_effect=raise_conn_error):
        result = retrieve("q", scope=["chub", "code-rag"])

    assert any(h.source == "chub" for h in result)
    assert all(h.source != "code-rag" for h in result)


# ---------------------------------------------------------------------------
# Score normalisation unit tests
# ---------------------------------------------------------------------------

def test_normalise_single_result_scores_1():
    """_normalise with a single value returns [1.0]."""
    from agents_core.retrieval import _normalise
    assert _normalise([0.42]) == pytest.approx([1.0])
    assert _normalise([-7.3], invert=True) == pytest.approx([1.0])


def test_normalise_range():
    """Verify min-max: best hit → 1.0, worst → 0.0."""
    from agents_core.retrieval import _normalise
    scores = _normalise([0.2, 0.5, 0.8])
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(1.0)


def test_normalise_invert():
    """FTS5 rank inversion: most-negative raw → highest normalised score."""
    from agents_core.retrieval import _normalise
    # FTS5 returns e.g. [-10, -5, -1]; -10 is best
    scores = _normalise([-10.0, -5.0, -1.0], invert=True)
    assert scores[0] == pytest.approx(1.0)   # -10 inverted → 10, normalised highest
    assert scores[2] == pytest.approx(0.0)   # -1 inverted → 1, normalised lowest


# ---------------------------------------------------------------------------
# Integration smoke — real backends; skip unless -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_mem_chub_vault_rag():
    """Real query against mem + chub + vault-rag.

    Run locally:  pytest -m integration tests/test_retrieval.py

    Asserts that the engine returns at least one Hit from at least one backend.
    Does not assert content — backend state varies.
    """
    results = retrieve(
        query="agents starhouse",
        scope=["mem", "chub", "vault-rag"],
        top_k=5,
    )
    assert isinstance(results, list)
    sources = {h.source for h in results}
    assert len(sources) >= 1, "expected hits from at least one backend"
    for h in results:
        assert h.source in {"mem", "chub", "vault-rag"}
        assert 0.0 <= h.score <= 1.0
        assert h.id.startswith(h.source + ":")
