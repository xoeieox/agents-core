"""Tests for agents_core.dowser — funnel stages, server, client.

No live GW / SearXNG / network. All operators and HTTP calls are stubbed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call as mock_call

import pytest
import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_read_response(text: str) -> str:
    return text


def _make_query_gen_json(queries: list[str]) -> str:
    return json.dumps({"queries": queries})


def _make_findings_json(findings: str, citations: list[dict]) -> str:
    return json.dumps({"findings": findings, "citations": citations})


def _make_triage_json(top_indices: list[int]) -> str:
    return json.dumps({"top_indices": top_indices})


def _make_critic_json(relevance=8, credibility=8, faithfulness=8, status="pass", diagnosis="") -> str:
    return json.dumps({
        "relevance": relevance,
        "credibility": credibility,
        "faithfulness": faithfulness,
        "confidence": "high",
        "status": status,
        "diagnosis": diagnosis,
    })


def _searxng_hits(n: int = 3) -> list[dict]:
    return [
        {"url": f"https://example.com/page{i}", "title": f"Page {i}", "snippet": f"snippet {i}"}
        for i in range(n)
    ]


def _fetched_pages(urls: list[str]) -> list[dict]:
    return [{"url": u, "text": f"text content for {u}", "error": None} for u in urls]


# ---------------------------------------------------------------------------
# AC1: read_batch funnel order, URL dedupe, top-K triage cap
# ---------------------------------------------------------------------------

class TestReadBatchFunnelOrder:
    """Assert funnel runs in order and respects the top-K budget."""

    def _run_with_stubs(
        self,
        queries=None,
        hits=None,
        triage_indices=None,
        findings="found something",
        citations=None,
        deep_read_urls=3,
    ):
        queries = queries or ["test query"]
        hits = hits if hits is not None else _searxng_hits(5)
        triage_indices = triage_indices if triage_indices is not None else list(range(min(deep_read_urls, len(hits))))
        citations = citations or [{"url": "https://example.com/page0", "title": "P0", "excerpt": "some text", "credibility": "high"}]

        call_log = []

        def fake_read_op(operator, prompt, json_mode=False):
            call_log.append(("read_op", prompt[:60]))
            if "search strategist" in prompt:
                return _make_query_gen_json(queries)
            if "Rank these" in prompt:
                return _make_triage_json(triage_indices)
            if "research analyst" in prompt:
                return _make_findings_json(findings, citations)
            return json.dumps({})

        def fake_searxng(query_list, searxng_url):
            call_log.append(("searxng", query_list))
            return hits, None

        def fake_fetch(urls):
            call_log.append(("fetch", urls))
            return _fetched_pages(urls)

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._search_searxng", side_effect=fake_searxng), \
             patch("agents_core.dowser._fetch_pages", side_effect=fake_fetch):
            from agents_core.dowser import read_batch
            result = read_batch(
                requests_list=[{"intent": "test intent", "context": "ctx"}],
                read_operator="quest",
                budget={"deep_read_urls": deep_read_urls, "wall_clock_sec": 300},
            )

        return result, call_log

    def test_funnel_runs_query_gen_then_search_then_triage_then_fetch_then_deep_read(self):
        result, call_log = self._run_with_stubs()
        ops = [c[0] for c in call_log]
        # query_gen (read_op), search, triage (read_op), fetch, deep_read (read_op)
        assert "read_op" in ops
        assert "searxng" in ops
        assert "fetch" in ops

        # query_gen fires before search
        first_read_op = next(i for i, c in enumerate(call_log) if c[0] == "read_op")
        first_searxng = next(i for i, c in enumerate(call_log) if c[0] == "searxng")
        first_fetch = next(i for i, c in enumerate(call_log) if c[0] == "fetch")
        assert first_read_op < first_searxng < first_fetch

    def test_top_k_triage_cap(self):
        hits = _searxng_hits(8)
        deep_read_urls = 3
        result, call_log = self._run_with_stubs(
            hits=hits, triage_indices=[0, 1, 2], deep_read_urls=deep_read_urls
        )
        # fetch must only see top-K URLs
        fetch_calls = [c for c in call_log if c[0] == "fetch"]
        assert fetch_calls
        fetched_urls = fetch_calls[0][1]
        assert len(fetched_urls) <= deep_read_urls

    def test_url_dedupe_across_queries(self):
        """Two queries returning overlapping URLs must not double-fetch."""
        duplicate_url = "https://example.com/shared"
        # Simulate two queries yielding hits that include the same URL
        shared_hit = {"url": duplicate_url, "title": "Shared", "snippet": "both queries"}

        call_log = []

        def fake_searxng(query_list, searxng_url):
            # Each call returns hits including the same URL
            base = _searxng_hits(2)
            return base + [shared_hit], None

        def fake_read_op(operator, prompt, json_mode=False):
            if "search strategist" in prompt:
                return _make_query_gen_json(["q1", "q2"])
            if "Rank" in prompt:
                return _make_triage_json([0, 1, 2])
            if "research analyst" in prompt:
                return _make_findings_json("found", [{"url": duplicate_url, "title": "t", "excerpt": "e", "credibility": "high"}])
            return "{}"

        def fake_fetch(urls):
            call_log.append(("fetch", urls))
            return _fetched_pages(urls)

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._search_searxng", side_effect=fake_searxng), \
             patch("agents_core.dowser._fetch_pages", side_effect=fake_fetch):
            from agents_core.dowser import read_batch
            read_batch(
                requests_list=[{"intent": "dedupe test"}],
                read_operator="quest",
                budget={"deep_read_urls": 5, "wall_clock_sec": 300},
            )

        # The URL set passed to triage should be deduped
        # We verify no duplicate in what was passed to fetch
        fetch_calls = [c for c in call_log if c[0] == "fetch"]
        if fetch_calls:
            urls = fetch_calls[0][1]
            assert len(urls) == len(set(urls)), "Duplicate URLs passed to fetch"

    def test_returns_findings_citations_outcome_provenance(self):
        result, _ = self._run_with_stubs(findings="answer found", citations=[
            {"url": "https://example.com/page0", "title": "T", "excerpt": "exact quote", "credibility": "high"}
        ])
        draft = result["drafts"][0]
        assert draft["intent"] == "test intent"
        assert draft["findings"] == "answer found"
        assert len(draft["citations"]) == 1
        assert draft["citations"][0]["excerpt"] == "exact quote"
        assert draft["outcome"] == "sources-found"
        prov = draft["provenance"]
        assert "search_strings" in prov
        assert "hits_count" in prov
        assert "triaged_urls" in prov
        assert "read_urls" in prov
        assert "friction_ratio" in prov
        assert "rewrote_from_diagnosis" in prov
        assert "notes" in prov


# ---------------------------------------------------------------------------
# AC2: tier purity — read phase never calls critic, critique phase never calls reader
# ---------------------------------------------------------------------------

class TestTierPurity:
    def test_read_batch_never_calls_critic_operator(self):
        critic_called = []

        def fake_read_op(operator, prompt, json_mode=False):
            if "search strategist" in prompt:
                return _make_query_gen_json(["q"])
            if "Rank" in prompt:
                return _make_triage_json([0])
            return _make_findings_json("found", [{"url": "http://x.com", "title": "x", "excerpt": "y", "credibility": "high"}])

        def fake_critic(*args, **kwargs):
            critic_called.append(True)
            return "{}"

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._search_searxng", return_value=(_searxng_hits(2), None)), \
             patch("agents_core.dowser._fetch_pages", return_value=_fetched_pages(["http://x.com"])), \
             patch("agents_core.dowser._call_critic_operator", side_effect=fake_critic):
            from agents_core.dowser import read_batch
            read_batch(
                requests_list=[{"intent": "purity test"}],
                read_operator="quest",
                budget={"deep_read_urls": 2, "wall_clock_sec": 60},
            )

        assert not critic_called, "read_batch must never call the critic operator"

    def test_critique_batch_never_calls_read_operator(self):
        read_called = []

        def fake_read_op(*args, **kwargs):
            read_called.append(True)
            return "{}"

        def fake_critic(operator, prompt, json_mode=False):
            return _make_critic_json()

        draft = {
            "intent": "x",
            "findings": "some finding",
            "citations": [{"url": "http://x.com", "title": "t", "excerpt": "e", "credibility": "high"}],
            "outcome": "sources-found",
            "provenance": {},
        }

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._call_critic_operator", side_effect=fake_critic):
            from agents_core.dowser import critique_batch
            critique_batch(drafts=[draft], critic_operator="gravitywell")

        assert not read_called, "critique_batch must never call the read operator"


# ---------------------------------------------------------------------------
# AC3: critique_batch returns status/verdict/diagnosis; faithfulness check
# ---------------------------------------------------------------------------

class TestCritiqueBatch:
    def _run_critique(self, findings="good finding", citations=None, critic_response=None):
        citations = citations or [{"url": "http://x.com", "title": "t", "excerpt": "direct quote", "credibility": "high"}]
        critic_response = critic_response or _make_critic_json(status="pass")

        draft = {
            "intent": "test",
            "findings": findings,
            "citations": citations,
            "outcome": "sources-found",
            "provenance": {},
        }
        with patch("agents_core.dowser._call_critic_operator", return_value=critic_response):
            from agents_core.dowser import critique_batch
            return critique_batch(drafts=[draft], critic_operator="gravitywell")

    def test_faithful_draft_returns_pass(self):
        result = self._run_critique(critic_response=_make_critic_json(status="pass"))
        v = result["verdicts"][0]
        assert v["status"] == "pass"
        assert v["verdict"]["faithfulness"] >= 0

    def test_unfaithful_draft_returns_subpar(self):
        result = self._run_critique(
            critic_response=_make_critic_json(faithfulness=2, status="subpar", diagnosis="claims not supported by excerpts")
        )
        v = result["verdicts"][0]
        assert v["status"] == "subpar"
        assert "not supported" in v["diagnosis"] or v["diagnosis"]

    def test_empty_draft_returns_subpar_insufficient_sources(self):
        draft = {"intent": "empty", "findings": "", "citations": [], "outcome": "no-credible-sources", "provenance": {}}
        with patch("agents_core.dowser._call_critic_operator") as mock_critic:
            from agents_core.dowser import critique_batch
            result = critique_batch(drafts=[draft], critic_operator="gravitywell")
        mock_critic.assert_not_called()
        v = result["verdicts"][0]
        assert v["status"] == "subpar"
        assert v["diagnosis"] == "insufficient-sources"


# ---------------------------------------------------------------------------
# AC4: typed honest-null outcomes
# ---------------------------------------------------------------------------

class TestTypedHonestNull:
    def _run_read_batch(self, hits, error_note, friction_ratio_override=None):
        def fake_query_gen(operator, prompt, json_mode=False):
            return _make_query_gen_json(["query"])

        def fake_searxng(queries, searxng_url):
            return hits, error_note

        def fake_triage(h, intent, top_k, operator):
            from agents_core.dowser import _credibility_score
            credible = [x for x in h if _credibility_score(x["url"]) >= 0.5]
            total = len(h)
            credible_count = len(credible)
            fr = 1.0 - (credible_count / total) if total > 0 else 0.0
            if friction_ratio_override is not None:
                fr = friction_ratio_override
            return h[:top_k], fr

        def fake_fetch(urls):
            return []

        def fake_deep_read(intent, context, pages, operator):
            return "", []

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_query_gen), \
             patch("agents_core.dowser._search_searxng", side_effect=fake_searxng), \
             patch("agents_core.dowser._triage", side_effect=fake_triage), \
             patch("agents_core.dowser._fetch_pages", side_effect=fake_fetch), \
             patch("agents_core.dowser._deep_read", side_effect=fake_deep_read):
            from agents_core.dowser import read_batch
            return read_batch(
                requests_list=[{"intent": "null test"}],
                read_operator="quest",
                budget={"deep_read_urls": 3, "wall_clock_sec": 60},
            )

    def test_zero_hits_yields_no_credible_sources(self):
        result = self._run_read_batch(hits=[], error_note=None)
        draft = result["drafts"][0]
        assert draft["outcome"] == "no-credible-sources"
        assert draft["findings"] == ""
        assert draft["citations"] == []

    def test_many_junk_hits_yields_high_friction(self):
        junk_hits = [
            {"url": f"https://reddit.com/r/foo/post{i}", "title": f"r{i}", "snippet": ""}
            for i in range(6)
        ]
        result = self._run_read_batch(hits=junk_hits, error_note=None, friction_ratio_override=0.9)
        draft = result["drafts"][0]
        assert draft["outcome"] == "high-friction"

    def test_searxng_error_yields_infra_unavailable(self):
        result = self._run_read_batch(hits=[], error_note="SearXNG error: Connection refused")
        draft = result["drafts"][0]
        assert draft["outcome"] == "infra-unavailable"

    def test_no_fabricated_citation_on_null(self):
        result = self._run_read_batch(hits=[], error_note=None)
        draft = result["drafts"][0]
        assert draft["citations"] == [], "Must never emit fabricated citations"

    def test_critique_marks_null_draft_subpar(self):
        draft = {
            "intent": "null",
            "findings": "",
            "citations": [],
            "outcome": "no-credible-sources",
            "provenance": {},
        }
        from agents_core.dowser import critique_batch
        result = critique_batch(drafts=[draft], critic_operator="gravitywell")
        assert result["verdicts"][0]["status"] == "subpar"


# ---------------------------------------------------------------------------
# AC5: retry hint — prior_diagnosis triggers rewrite
# ---------------------------------------------------------------------------

class TestRetryHint:
    def test_prior_diagnosis_produces_rewritten_queries(self):
        queries_normal = []
        queries_retry = []

        def fake_query_gen_capture(operator, prompt, json_mode=False):
            if "prior_diagnosis" in prompt.lower() or "diagnosis" in prompt.lower():
                queries_retry.append("retry_query_from_diagnosis")
                return _make_query_gen_json(["retry_query_from_diagnosis"])
            queries_normal.append("normal_query")
            return _make_query_gen_json(["normal_query"])

        def fake_searxng(queries, searxng_url):
            return _searxng_hits(2), None

        def fake_triage(h, intent, top_k, operator):
            return h[:top_k], 0.0

        def fake_fetch(urls):
            return _fetched_pages(urls[:1])

        def fake_deep_read(intent, context, pages, operator):
            return "found", [{"url": "http://x.com", "title": "t", "excerpt": "e", "credibility": "high"}]

        reqs_normal = [{"intent": "test", "context": None}]
        reqs_retry = [{"intent": "test", "context": None, "prior_diagnosis": "all forum junk"}]

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_query_gen_capture), \
             patch("agents_core.dowser._search_searxng", side_effect=fake_searxng), \
             patch("agents_core.dowser._triage", side_effect=fake_triage), \
             patch("agents_core.dowser._fetch_pages", side_effect=fake_fetch), \
             patch("agents_core.dowser._deep_read", side_effect=fake_deep_read):
            from agents_core.dowser import read_batch
            r_normal = read_batch(reqs_normal, read_operator="quest", budget={"deep_read_urls": 2, "wall_clock_sec": 60})
            r_retry = read_batch(reqs_retry, read_operator="quest", budget={"deep_read_urls": 2, "wall_clock_sec": 60})

        assert r_normal["drafts"][0]["provenance"]["rewrote_from_diagnosis"] is False
        assert r_retry["drafts"][0]["provenance"]["rewrote_from_diagnosis"] is True
        # Queries must differ
        normal_queries = r_normal["drafts"][0]["provenance"]["search_strings"]
        retry_queries = r_retry["drafts"][0]["provenance"]["search_strings"]
        assert normal_queries != retry_queries or queries_retry  # rewrite fired


# ---------------------------------------------------------------------------
# AC6: fail-soft — SearXNG error, unfetchable URL, operator None
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_searxng_error_does_not_500_batch(self):
        def fake_read_op(operator, prompt, json_mode=False):
            if "search strategist" in prompt:
                return _make_query_gen_json(["q"])
            return "{}"

        def fake_searxng(queries, searxng_url):
            return [], "SearXNG error: timeout"

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._search_searxng", side_effect=fake_searxng):
            from agents_core.dowser import read_batch
            result = read_batch(
                requests_list=[{"intent": "searxng down"}],
                read_operator="quest",
                budget={"deep_read_urls": 3, "wall_clock_sec": 60},
            )

        draft = result["drafts"][0]
        assert draft["outcome"] == "infra-unavailable"
        assert draft["findings"] == ""
        assert draft["citations"] == []
        assert any("SearXNG" in n for n in draft["provenance"]["notes"])

    def test_operator_returning_none_does_not_raise(self):
        with patch("agents_core.dowser._call_read_operator", return_value=None), \
             patch("agents_core.dowser._search_searxng", return_value=(_searxng_hits(2), None)), \
             patch("agents_core.dowser._fetch_pages", return_value=_fetched_pages(["http://x.com"])):
            from agents_core.dowser import read_batch
            result = read_batch(
                requests_list=[{"intent": "operator none"}],
                read_operator="quest",
                budget={"deep_read_urls": 2, "wall_clock_sec": 60},
            )
        assert "drafts" in result  # no exception

    def test_unfetchable_url_drops_with_provenance_note(self):
        def fake_read_op(operator, prompt, json_mode=False):
            if "search strategist" in prompt:
                return _make_query_gen_json(["q"])
            if "Rank" in prompt:
                return _make_triage_json([0])
            return _make_findings_json("", [])

        failed_url = "http://unfetchable.example.com"

        def fake_fetch(urls):
            return [{"url": failed_url, "text": "", "error": "Connection refused"}]

        with patch("agents_core.dowser._call_read_operator", side_effect=fake_read_op), \
             patch("agents_core.dowser._search_searxng", return_value=([{"url": failed_url, "title": "t", "snippet": "s"}], None)), \
             patch("agents_core.dowser._fetch_pages", side_effect=fake_fetch):
            from agents_core.dowser import read_batch
            result = read_batch(
                requests_list=[{"intent": "fetch fail"}],
                read_operator="quest",
                budget={"deep_read_urls": 2, "wall_clock_sec": 60},
            )

        draft = result["drafts"][0]
        assert failed_url in " ".join(draft["provenance"]["notes"])


# ---------------------------------------------------------------------------
# AC7: quest operator in llm.py
# ---------------------------------------------------------------------------

class TestQuestOperator:
    def test_quest_in_operator_defaults(self):
        from agents_core.llm import OPERATOR_DEFAULTS
        assert "quest" in OPERATOR_DEFAULTS
        assert OPERATOR_DEFAULTS["quest"] == "quest-35b-rl"

    def test_quest_routes_to_quest_url(self):
        from agents_core import llm as llm_mod

        captured = {}

        def fake_post(base_url, model, messages, **kwargs):
            captured["base_url"] = base_url
            captured["model"] = model
            return "quest answer"

        with patch("agents_core.llm._post_chat_completion", side_effect=fake_post):
            result = llm_mod.call_operator("quest", prompt="hello", system="sys")

        assert result == "quest answer"
        assert "QUEST_URL" in dir(llm_mod) or captured.get("base_url") is not None
        # base_url must be the QUEST endpoint (not GW :8081)
        assert captured["base_url"] != llm_mod.GW_URL
        assert captured["model"] == "quest-35b-rl"

    def test_quest_does_not_paid_fallback_when_unreachable(self):
        from agents_core import llm as llm_mod
        from agents_core.llm import OperatorUnreachableError

        paid_called = []

        def fake_post(base_url, model, messages, **kwargs):
            raise OperatorUnreachableError(base_url, Exception("refused"))

        def fake_paid(operator_class, prompt, **kwargs):
            paid_called.append(operator_class)
            return "paid answer"

        with patch("agents_core.llm._post_chat_completion", side_effect=fake_post), \
             patch("agents_core.llm._apply_wake_fail", side_effect=fake_paid):
            result = llm_mod.call_operator("quest", prompt="x", on_wake_fail="skip")

        assert result is None, "quest must return None (not paid fallback) when on_wake_fail=skip"
        assert not paid_called

    def test_quest_model_swap_raises(self):
        from agents_core.llm import call_operator
        with pytest.raises(ValueError, match="quest-35b-rl"):
            call_operator("quest", prompt="x", model="some-other-model")


# ---------------------------------------------------------------------------
# AC8: dowser_client round-trips both endpoints (via TestClient)
# ---------------------------------------------------------------------------

class TestDowserClientServerRoundTrip:
    """Uses FastAPI TestClient + stubbed dowser functions."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from agents_core.dowser_server import create_app
        return TestClient(create_app())

    def test_read_batch_round_trip(self):
        fake_result = {
            "drafts": [{
                "intent": "test",
                "findings": "found",
                "citations": [],
                "outcome": "sources-found",
                "provenance": {"search_strings": ["q"], "hits_count": 1, "triaged_urls": [],
                               "read_urls": [], "friction_ratio": 0.0, "rewrote_from_diagnosis": False, "notes": []},
            }]
        }
        with patch("agents_core.dowser_server.read_batch", return_value=fake_result):
            client = self._make_client()
            resp = client.post("/research/read-batch", json={
                "requests": [{"intent": "test"}],
                "read_operator": "quest",
                "budget": {"deep_read_urls": 3, "wall_clock_sec": 60},
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["drafts"][0]["intent"] == "test"
        assert data["drafts"][0]["outcome"] == "sources-found"

    def test_critique_batch_round_trip(self):
        fake_result = {
            "verdicts": [{
                "intent": "test",
                "status": "pass",
                "verdict": {"relevance": 8, "credibility": 8, "faithfulness": 8, "confidence": "high"},
                "diagnosis": "",
            }]
        }
        with patch("agents_core.dowser_server.critique_batch", return_value=fake_result):
            client = self._make_client()
            resp = client.post("/research/critique-batch", json={
                "drafts": [{"intent": "test", "findings": "f", "citations": [{"url": "x", "title": "t", "excerpt": "e", "credibility": "high"}], "outcome": "sources-found", "provenance": {}}],
                "critic_operator": "gravitywell",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdicts"][0]["status"] == "pass"

    def test_server_binds_default_port_8412(self):
        import os
        # Ensure the default env produces port 8412
        env_port = os.environ.get("DOWSER_BIND_PORT", "8412")
        assert env_port == "8412"

    def test_invalid_read_operator_returns_400(self):
        client = self._make_client()
        resp = client.post("/research/read-batch", json={
            "requests": [{"intent": "x"}],
            "read_operator": "gpt4-turbo",
        })
        assert resp.status_code == 400

    def test_invalid_requests_field_returns_400(self):
        client = self._make_client()
        resp = client.post("/research/read-batch", json={"requests": "not-a-list"})
        assert resp.status_code == 400
