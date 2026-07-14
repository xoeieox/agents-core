"""Tests for agents_core.corpus_reader — RoomRAG podcast read-funnel + fixed-answer scorer.

No live GW / RoomRAG / network. All operators, retrieve(), and the health probe are stubbed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from agents_core.retrieval import Hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query_gen_json(queries: list[str]) -> str:
    return json.dumps({"queries": queries})


def _make_triage_json(top_indices: list[int]) -> str:
    return json.dumps({"top_indices": top_indices})


def _make_findings_json(findings: str, citations: list[dict]) -> str:
    return json.dumps({"findings": findings, "citations": citations})


def _hit(idx: int, show: str = "dwarkesh", score: float = 0.8, content: str | None = None) -> Hit:
    content = content or f"[01:0{idx}:00] Some transcript content number {idx}. [01:0{idx}:30] more."
    return Hit(
        id=f"room-rag:library/podcasts/{show}/ep{idx}.md",
        score=score,
        source="room-rag",
        content=content,
        metadata={
            "file_path": f"library/podcasts/{show}/ep{idx}.md",
            "title": f"Episode {idx}",
            "folder": show,
            "date": "2026-05-0" + str(idx) if idx < 10 else "2026-05-10",
        },
    )


def _fake_read_op_factory(queries=None, triage_indices=None, findings="found it", citations=None):
    queries = queries or ["q1"]
    citations = citations if citations is not None else [{"chunk_index": 1, "excerpt": "direct quote"}]

    def fake(operator, prompt, json_mode=False):
        if "search strategist" in prompt:
            return _make_query_gen_json(queries)
        if "Rank these" in prompt:
            idx = triage_indices if triage_indices is not None else [0]
            return _make_triage_json(idx)
        if "research analyst" in prompt:
            return _make_findings_json(findings, citations)
        return "{}"

    return fake


# ---------------------------------------------------------------------------
# Draft shape parity with Dowser
# ---------------------------------------------------------------------------

class TestDraftShape:
    def test_matches_dowser_key_shape(self):
        hits = [_hit(1)]
        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=hits):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "what happened"}], read_operator="quest")

        draft = result["drafts"][0]
        assert set(draft.keys()) == {"intent", "findings", "citations", "outcome", "provenance"}

    def test_citations_carry_show_episode_timestamp_no_url(self):
        hits = [_hit(1, show="dwarkesh")]
        fake_read_op = _fake_read_op_factory(citations=[{"chunk_index": 1, "excerpt": "direct quote"}])

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=hits):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "what happened"}], read_operator="quest")

        draft = result["drafts"][0]
        assert draft["outcome"] == "sources-found"
        assert len(draft["citations"]) == 1
        c = draft["citations"][0]
        assert "url" not in c
        assert c["show"] == "dwarkesh"
        assert c["episode_title"] == "Episode 1"
        assert c["excerpt"] == "direct quote"
        assert c["timestamp_range"]


# ---------------------------------------------------------------------------
# No web-credibility prior
# ---------------------------------------------------------------------------

class TestNoWebCredibilityPrior:
    def test_no_credibility_symbols_present(self):
        import agents_core.corpus_reader as cr
        for sym in ("_CREDIBLE_DOMAINS", "_JUNK_PATTERNS", "_credibility_score"):
            assert not hasattr(cr, sym), f"web-credibility symbol {sym} leaked into corpus_reader"


# ---------------------------------------------------------------------------
# Stage 2 scoping: server-side doc_type + client-side folder filter
# ---------------------------------------------------------------------------

class TestShowFilterScoping:
    def test_doc_type_always_applied_server_side(self):
        captured = {}

        def fake_retrieve(query, scope, filters, top_k, min_score, timeout=None):
            captured["filters"] = filters
            captured["top_k"] = top_k
            captured["timeout"] = timeout
            return [_hit(1, show="creative-codex")]

        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", side_effect=fake_retrieve):
            from agents_core.corpus_reader import read_batch_corpus
            read_batch_corpus(
                [{"intent": "x"}], read_operator="quest", show_filter="creative-codex"
            )

        assert captured["filters"] == {"doc_type": "podcast-transcript"}

    def test_effective_fetch_floor_is_at_least_30(self):
        """A small caller-supplied retrieve_k must not shrink the candidate pool below 30."""
        captured = {}

        def fake_retrieve(query, scope, filters, top_k, min_score, timeout=None):
            captured["top_k"] = top_k
            return []

        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", side_effect=fake_retrieve), \
             patch("agents_core.corpus_reader._room_rag_alive", return_value=True):
            from agents_core.corpus_reader import read_batch_corpus
            read_batch_corpus(
                [{"intent": "x"}], read_operator="quest", budget={"retrieve_k": 8}
            )

        assert captured["top_k"] >= 30

    def test_generous_timeout_passed_to_retrieve(self):
        captured = {}

        def fake_retrieve(query, scope, filters, top_k, min_score, timeout=None):
            captured["timeout"] = timeout
            return []

        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", side_effect=fake_retrieve), \
             patch("agents_core.corpus_reader._room_rag_alive", return_value=True):
            from agents_core.corpus_reader import read_batch_corpus
            read_batch_corpus([{"intent": "x"}], read_operator="quest")

        assert captured["timeout"] >= 8

    def test_client_side_filter_drops_other_shows(self):
        """Even if retrieve() returns a hit outside the requested show, it must not surface."""
        mixed_hits = [_hit(1, show="creative-codex"), _hit(2, show="dwarkesh")]
        fake_read_op = _fake_read_op_factory(citations=[
            {"chunk_index": 1, "excerpt": "quote a"},
        ])

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=mixed_hits):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus(
                [{"intent": "x"}], read_operator="quest", show_filter="creative-codex"
            )

        draft = result["drafts"][0]
        assert draft["provenance"]["hits_count"] == 1
        assert all(c["show"] != "dwarkesh" for c in draft["citations"])

    def test_hit_missing_folder_key_is_dropped_not_passed_through(self):
        """A Hit with no folder metadata must never survive a show scope — defensive default."""
        no_folder_hit = Hit(
            id="room-rag:library/podcasts/mystery/ep9.md",
            score=0.9,
            source="room-rag",
            content="[01:00] some content",
            metadata={"title": "Mystery Episode"},  # no "folder" key
        )
        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=[no_folder_hit]), \
             patch("agents_core.corpus_reader._room_rag_alive", return_value=True):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus(
                [{"intent": "x"}], read_operator="quest", show_filter="dwarkesh"
            )

        assert result["drafts"][0]["provenance"]["hits_count"] == 0

    def test_per_request_show_overrides_batch_show_filter(self):
        captured_folders = []

        def fake_retrieve(query, scope, filters, top_k, min_score, timeout=None):
            return [_hit(1, show="no-priors"), _hit(2, show="dwarkesh")]

        fake_read_op = _fake_read_op_factory(citations=[
            {"chunk_index": 1, "excerpt": "quote a"},
        ])

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", side_effect=fake_retrieve):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus(
                [{"intent": "x", "show": "no-priors"}],
                read_operator="quest",
                show_filter="dwarkesh",
            )

        draft = result["drafts"][0]
        assert draft["provenance"]["hits_count"] == 1
        assert all(c["show"] == "no-priors" for c in draft["citations"])


# ---------------------------------------------------------------------------
# Outcome classification — all four branches
# ---------------------------------------------------------------------------

class TestOutcomeClassification:
    def test_sources_found(self):
        hits = [_hit(1)]
        fake_read_op = _fake_read_op_factory(findings="the answer is X", citations=[
            {"chunk_index": 1, "excerpt": "quote"},
        ])

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=hits):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "x"}], read_operator="quest")

        assert result["drafts"][0]["outcome"] == "sources-found"

    def test_no_answer_in_corpus_when_operator_disclaims(self):
        """Chunks retrieved, but operator reports the intent isn't answered — must NOT be sources-found."""
        hits = [_hit(1)]
        fake_read_op = _fake_read_op_factory(findings="", citations=[])

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=hits):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "x"}], read_operator="quest")

        draft = result["drafts"][0]
        assert draft["outcome"] == "no-answer-in-corpus"
        assert draft["outcome"] != "sources-found"
        assert draft["citations"] == []

    def test_no_relevant_chunks_when_empty_retrieval_and_probe_ok(self):
        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=[]), \
             patch("agents_core.corpus_reader._room_rag_alive", return_value=True):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "x"}], read_operator="quest")

        assert result["drafts"][0]["outcome"] == "no-relevant-chunks"

    def test_infra_unavailable_when_empty_retrieval_and_probe_fails(self):
        """Simulated RoomRAG outage must never masquerade as a corpus gap."""
        fake_read_op = _fake_read_op_factory()

        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=[]), \
             patch("agents_core.corpus_reader._room_rag_alive", return_value=False):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "x"}], read_operator="quest")

        draft = result["drafts"][0]
        assert draft["outcome"] == "infra-unavailable"
        assert draft["outcome"] != "no-relevant-chunks"


# ---------------------------------------------------------------------------
# Budget dict access — None budget must default, not AttributeError
# ---------------------------------------------------------------------------

class TestBudgetDefaulting:
    def test_none_budget_does_not_raise(self):
        fake_read_op = _fake_read_op_factory()
        with patch("agents_core.corpus_reader._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.corpus_reader.retrieve", return_value=[_hit(1)]):
            from agents_core.corpus_reader import read_batch_corpus
            result = read_batch_corpus([{"intent": "x"}], read_operator="quest", budget=None)
        assert result["drafts"][0]["intent"] == "x"


# ---------------------------------------------------------------------------
# No direct Anthropic API / new HTTP LLM transport
# ---------------------------------------------------------------------------

class TestNoDirectAnthropicAPI:
    def test_module_source_has_no_anthropic_api_reference(self):
        import inspect
        import agents_core.corpus_reader as cr
        src = inspect.getsource(cr)
        assert "anthropic" not in src.lower()
        assert "ANTHROPIC_API_KEY" not in src
        assert "call_operator" in src
        assert "call_claude_cli" in src


# ---------------------------------------------------------------------------
# score_fixed_answer
# ---------------------------------------------------------------------------

class TestScoreFixedAnswer:
    def _fixture(self):
        return {
            "questions": [
                {
                    "question": "answerable q",
                    "target_show": "dwarkesh",
                    "expected_answer_substrings": ["the target fact"],
                    "answer_present": True,
                    "category": "answerable",
                },
                {
                    "question": "absent trap q",
                    "target_show": "nonexistent-show",
                    "expected_answer_substrings": [],
                    "answer_present": False,
                    "category": "absent-trap",
                },
                {
                    "question": "difficult present q - suppressed",
                    "target_show": "mlst",
                    "expected_answer_substrings": ["uncomfortable fact"],
                    "answer_present": True,
                    "category": "difficult-present",
                },
            ]
        }

    def _drafts(self):
        return [
            {
                "intent": "answerable q",
                "findings": "the target fact is confirmed",
                "citations": [{"show": "dwarkesh", "episode_title": "e", "published": "d",
                               "timestamp_range": "01:00", "excerpt": "the target fact"}],
                "outcome": "sources-found",
                "provenance": {},
            },
            {
                "intent": "absent trap q",
                "findings": "",
                "citations": [],
                "outcome": "no-answer-in-corpus",
                "provenance": {},
            },
            {
                "intent": "difficult present q - suppressed",
                "findings": "",
                "citations": [{"show": "mlst", "episode_title": "e2", "published": "d",
                               "timestamp_range": "02:00", "excerpt": "unrelated"}],
                "outcome": "no-answer-in-corpus",
                "provenance": {},
            },
        ]

    def test_answerable_hit_and_match(self):
        from agents_core.corpus_reader import score_fixed_answer
        result = score_fixed_answer(self._drafts(), self._fixture())
        pq = {p["question"]: p for p in result["per_question"]}
        assert pq["answerable q"]["retrieval_hit"] is True
        assert pq["answerable q"]["answer_match"] is True

    def test_absent_trap_fabrication_avoidance(self):
        from agents_core.corpus_reader import score_fixed_answer
        result = score_fixed_answer(self._drafts(), self._fixture())
        pq = {p["question"]: p for p in result["per_question"]}
        assert pq["absent trap q"]["fabrication_avoidance"] is True

    def test_difficult_present_suppression_failure_detected(self):
        """Chunk WAS retrieved (citation carries mlst) but operator hedged — a suppression failure."""
        from agents_core.corpus_reader import score_fixed_answer
        result = score_fixed_answer(self._drafts(), self._fixture())
        pq = {p["question"]: p for p in result["per_question"]}
        item = pq["difficult present q - suppressed"]
        assert item["retrieval_hit"] is True
        assert item["suppression_avoidance"] is False

    def test_aggregate_reports_fabrication_vs_suppression_spread(self):
        from agents_core.corpus_reader import score_fixed_answer
        result = score_fixed_answer(self._drafts(), self._fixture())
        agg = result["aggregate"]
        assert agg["absent-trap"]["fabrication_avoidance_rate"] == 1.0
        assert agg["difficult-present"]["suppression_avoidance_rate"] == 0.0
        assert agg["fabrication_vs_suppression_spread"] == 1.0
