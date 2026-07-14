"""Corpus-reader — RoomRAG podcast-transcript read-funnel (query-gen -> retrieve -> triage -> deep-read).

Sibling to agents_core.dowser: same funnel shape, corpus body instead of the web. Tier-pure
(read operator only; never calls a critic). GW-topology-agnostic — no flip, no doorman, no
duty-cycle awareness; the caller owns serving-mode sequencing, same contract as Dowser.

Emits the same drafts record shape Dowser emits (intent, findings, citations, outcome,
provenance) so agents_core.dowser.critique_batch and downstream consumers work unchanged.
Citations reference show/episode + [MM:SS] excerpt instead of a URL. No SearXNG, no URL
fetch — agents_core.retrieval.retrieve() returns chunk text directly, so the whole
web-fetch stage Dowser has is absent here by design.
"""

from __future__ import annotations

import logging
import os
import re
import time

import httpx

from agents_core.llm import call_operator, call_claude_cli, parse_json_object
from agents_core.retrieval import retrieve, Hit

_log = logging.getLogger(__name__)

ROOM_RAG_URL = os.environ.get("ROOM_RAG_URL", "http://203.0.113.10:8201")
_DOC_TYPE_PODCAST = "podcast-transcript"
_MIN_CANDIDATE_POOL = 30
_DEFAULT_RAG_TIMEOUT_SEC = 10.0
HEALTH_TIMEOUT_SEC = 2.0
DEEP_READ_MAX_CONTEXT_CHARS = 4000


def _hit_label(h: Hit) -> str:
    md = h.metadata or {}
    return str(md.get("title") or md.get("file_path") or h.id)


_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def _timestamp_range(content: str) -> str:
    stamps = _TIMESTAMP_RE.findall(content)
    if not stamps:
        return ""
    if stamps[0] == stamps[-1]:
        return stamps[0]
    return f"{stamps[0]}-{stamps[-1]}"


# ---------------------------------------------------------------------------
# Stage 1: QUERY-GEN (read operator)
# ---------------------------------------------------------------------------

def _query_gen(
    intent: str,
    context: str | None,
    sub_intents: list[str] | None,
    prior_diagnosis: str | None,
    operator: str,
) -> tuple[list[str], bool]:
    """Returns (queries, rewrote_from_diagnosis)."""
    if prior_diagnosis:
        prompt = (
            f"You are a search strategist for a podcast-transcript corpus. A previous retrieval "
            f"pass for the following intent failed with this diagnosis:\n\n"
            f"DIAGNOSIS: {prior_diagnosis}\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + "Rewrite the retrieval queries to address the diagnosis. "
            "If 'off-topic', narrow the query. If 'thin', broaden or rephrase. "
            "Return JSON: {\"queries\": [\"...\", ...]}"
        )
    elif sub_intents:
        prompt = (
            f"You are a search strategist for a podcast-transcript corpus. Generate targeted "
            f"retrieval queries for this intent:\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + f"SUB-INTENTS (generate 1-2 queries per sub-intent): {sub_intents}\n"
            "Return JSON: {\"queries\": [\"...\", ...]}"
        )
    else:
        prompt = (
            f"You are a search strategist for a podcast-transcript corpus. Generate 2-4 targeted "
            f"retrieval queries for:\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + "Return JSON: {\"queries\": [\"...\", ...]}"
        )

    raw = _call_read_operator(operator, prompt, json_mode=True)
    if not raw:
        return [intent], False

    parsed = parse_json_object(raw)
    if not parsed or not isinstance(parsed.get("queries"), list):
        return [intent], False

    queries = [q for q in parsed["queries"] if isinstance(q, str) and q.strip()]
    rewrote = prior_diagnosis is not None
    return (queries or [intent]), rewrote


# ---------------------------------------------------------------------------
# Stage 2: RETRIEVE (no LLM)
# ---------------------------------------------------------------------------

def _retrieve_corpus(
    queries: list[str],
    show_filter: str | None,
    retrieve_k: int,
    min_score: float,
    rag_timeout_sec: float,
) -> list[Hit]:
    """Merge + dedupe Hits across queries, scoped server-side to the podcast corpus.

    Server-side scoping is filters={"doc_type": "podcast-transcript"} — the live RoomRAG
    :8201 /search endpoint honors doc_type (confirmed 2026-07-13); path_prefix has no
    field on that schema and is a silent no-op. A generous candidate pool (>= 30, the
    empirically-validated floor) is requested so per-show client-side filtering below has
    enough to work with; the floor overrides a smaller caller-supplied retrieve_k.

    Per-show scoping is client-side on the folder metadata field (folder == bare show
    slug) since RoomRAG's folder filter 500s live. A Hit missing the folder key is treated
    as a NON-match and dropped — never a silent pass-through.
    """
    effective_k = max(retrieve_k, _MIN_CANDIDATE_POOL)
    merged: dict[str, Hit] = {}
    for query in queries:
        hits = retrieve(
            query,
            scope=["room-rag"],
            filters={"doc_type": _DOC_TYPE_PODCAST},
            top_k=effective_k,
            min_score=min_score,
            timeout=rag_timeout_sec,
        )
        for h in hits:
            if show_filter and (h.metadata or {}).get("folder") != show_filter:
                continue
            if h.id not in merged:
                merged[h.id] = h
    return list(merged.values())


def _room_rag_alive() -> bool:
    try:
        resp = httpx.get(f"{ROOM_RAG_URL}/health", timeout=HEALTH_TIMEOUT_SEC)
        return resp.status_code == 200
    except Exception as e:
        _log.warning("[corpus_reader] RoomRAG health probe failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Stage 3: TRIAGE (read operator, no domain-credibility prior)
# ---------------------------------------------------------------------------

def _triage_corpus(
    hits: list[Hit],
    intent: str,
    top_k: int,
    operator: str,
) -> list[Hit]:
    if not hits:
        return []

    scored_sorted = sorted(hits, key=lambda h: h.score, reverse=True)
    candidates = scored_sorted[:20]
    hits_text = "\n".join(
        f"{i+1}. [{h.score:.2f}] {_hit_label(h)}\n   {h.content[:200]}"
        for i, h in enumerate(candidates)
    )
    prompt = (
        f"Rank these podcast-transcript excerpts by relevance to the research intent.\n\n"
        f"INTENT: {intent}\n\nEXCERPTS:\n{hits_text}\n\n"
        f"Return JSON: {{\"top_indices\": [0-based indices of the top {top_k} excerpts, best first]}}"
    )

    raw = _call_read_operator(operator, prompt, json_mode=True)
    top_indices = None
    if raw:
        parsed = parse_json_object(raw)
        if parsed and isinstance(parsed.get("top_indices"), list):
            top_indices = [
                i for i in parsed["top_indices"]
                if isinstance(i, int) and 0 <= i < len(candidates)
            ][:top_k]

    if not top_indices:
        top_indices = list(range(min(top_k, len(candidates))))

    return [candidates[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Stage 5: DEEP-READ (read operator)
# ---------------------------------------------------------------------------

def _deep_read_corpus(
    intent: str,
    context: str | None,
    chunks: list[Hit],
    operator: str,
) -> tuple[str, list[dict]]:
    """Returns (findings, citations)."""
    if not chunks:
        return "", []

    chunks_text = ""
    for i, h in enumerate(chunks):
        chunks_text += f"\n\n--- CHUNK {i+1}: {_hit_label(h)} ---\n{h.content[:DEEP_READ_MAX_CONTEXT_CHARS]}"

    prompt = (
        f"You are a research analyst. Synthesize findings from the podcast transcript excerpts "
        f"below for this intent.\n\n"
        f"INTENT: {intent}\n"
        + (f"CONTEXT: {context}\n" if context else "")
        + f"\nTRANSCRIPT EXCERPTS:{chunks_text}\n\n"
        "Rules:\n"
        "- Each claim MUST be supported by a direct excerpt from a transcript chunk.\n"
        "- Never fabricate citations or invent content not in the sources.\n"
        "- If sources don't answer the intent, say so explicitly.\n"
        "Return JSON:\n"
        '{"findings": "prose summary", "citations": [{"chunk_index": N (1-based, matching '
        'the CHUNK numbers above), "excerpt": "direct quote from the transcript chunk"}]}'
    )

    raw = _call_read_operator(operator, prompt, json_mode=True)
    if not raw:
        return "", []

    parsed = parse_json_object(raw)
    if not parsed:
        return "", []

    findings = parsed.get("findings", "") or ""
    citations_raw = parsed.get("citations") or []
    if not isinstance(citations_raw, list):
        citations_raw = []

    citations = []
    for c in citations_raw:
        if not isinstance(c, dict):
            continue
        excerpt = c.get("excerpt", "")
        chunk_index = c.get("chunk_index")
        if not excerpt or not isinstance(chunk_index, int):
            continue
        idx = chunk_index - 1
        if not (0 <= idx < len(chunks)):
            continue
        h = chunks[idx]
        md = h.metadata or {}
        citations.append({
            "show": md.get("folder", ""),
            "episode_title": md.get("title", ""),
            "published": md.get("date", ""),
            "timestamp_range": _timestamp_range(h.content),
            "excerpt": excerpt,
        })

    return findings, citations


# ---------------------------------------------------------------------------
# LLM dispatch helper (tier-pure)
# ---------------------------------------------------------------------------

def _call_read_operator(operator: str, prompt: str, json_mode: bool = False) -> str | None:
    """Call only the read operator. Never calls the critic."""
    try:
        if operator in ("sonnet", "haiku", "opus"):
            suffix = "\n\nRespond ONLY with valid JSON." if json_mode else ""
            return call_claude_cli(prompt + suffix, model=operator, timeout=120)
        else:
            return call_operator(operator, prompt, json_mode=json_mode, on_wake_fail="skip")
    except Exception as e:
        _log.warning("[corpus_reader] read operator %r failed: %s", operator, e)
        return None


# ---------------------------------------------------------------------------
# Typed outcome helpers
# ---------------------------------------------------------------------------

def _classify_outcome(hit_count: int, probe_alive: bool, findings: str) -> str:
    if hit_count == 0:
        return "no-relevant-chunks" if probe_alive else "infra-unavailable"
    if not findings:
        return "no-answer-in-corpus"
    return "sources-found"


def _infra_null_draft(intent: str, note: str) -> dict:
    return {
        "intent": intent,
        "findings": "",
        "citations": [],
        "outcome": "infra-unavailable",
        "provenance": {
            "search_strings": [],
            "hits_count": 0,
            "triaged_ids": [],
            "rewrote_from_diagnosis": False,
            "notes": [note],
        },
    }


# ---------------------------------------------------------------------------
# Public API: read_batch_corpus
# ---------------------------------------------------------------------------

def read_batch_corpus(
    requests_list: list[dict],
    read_operator: str = "quest",
    show_filter: str | None = None,
    budget: dict | None = None,
) -> dict:
    """Run the corpus read funnel for each request.

    Each request: {intent, show?, context?, sub_intents?, prior_diagnosis?}
    A per-request `show` scopes that request's retrieval to a single show (client-side,
    folder-metadata match); when a request omits `show`, the batch-level `show_filter`
    is used as the default.
    Returns: {drafts: [{intent, findings, citations, outcome, provenance}]}
    """
    retrieve_k = int((budget or {}).get("retrieve_k", 8))
    deep_read_k = int((budget or {}).get("deep_read_k", 5))
    min_score = float((budget or {}).get("min_score", 0.0))
    rag_timeout_sec = float((budget or {}).get("rag_timeout_sec", _DEFAULT_RAG_TIMEOUT_SEC))
    wall_clock_sec = int((budget or {}).get("wall_clock_sec", 300))
    deadline = time.monotonic() + wall_clock_sec

    drafts = []
    for req in requests_list:
        if time.monotonic() >= deadline:
            _log.warning("[corpus_reader] wall_clock_sec budget exhausted; skipping remaining requests")
            drafts.append(_infra_null_draft(req["intent"], "wall_clock_budget_exhausted"))
            continue

        intent = req["intent"]
        show = req.get("show", show_filter)
        context = req.get("context")
        sub_intents = req.get("sub_intents")
        prior_diagnosis = req.get("prior_diagnosis")

        # Stage 1: query-gen
        try:
            queries, rewrote = _query_gen(
                intent, context, sub_intents, prior_diagnosis, read_operator
            )
        except Exception as e:
            _log.warning("[corpus_reader] query_gen failed for %r: %s", intent, e)
            queries = [intent]
            rewrote = False

        # Stage 2: retrieve
        hits = _retrieve_corpus(queries, show, retrieve_k, min_score, rag_timeout_sec)

        notes: list[str] = []
        probe_alive = True
        if not hits:
            probe_alive = _room_rag_alive()
            notes.append(
                "RoomRAG liveness probe: OK" if probe_alive
                else f"RoomRAG liveness probe FAILED: {ROOM_RAG_URL}/health unreachable"
            )

        # Stage 3: triage
        try:
            triaged = _triage_corpus(hits, intent, deep_read_k, read_operator)
        except Exception as e:
            _log.warning("[corpus_reader] triage failed: %s", e)
            triaged = hits[:deep_read_k]

        # Stage 5: deep-read
        findings = ""
        citations: list[dict] = []
        if triaged:
            try:
                findings, citations = _deep_read_corpus(intent, context, triaged, read_operator)
            except Exception as e:
                _log.warning("[corpus_reader] deep_read failed: %s", e)

        outcome = _classify_outcome(len(hits), probe_alive, findings)

        drafts.append({
            "intent": intent,
            "findings": findings,
            "citations": citations,
            "outcome": outcome,
            "provenance": {
                "search_strings": queries,
                "hits_count": len(hits),
                "triaged_ids": [h.id for h in triaged],
                "rewrote_from_diagnosis": rewrote,
                "notes": notes,
            },
        })

    return {"drafts": drafts}


# ---------------------------------------------------------------------------
# Public API: score_fixed_answer
# ---------------------------------------------------------------------------

def _draft_for(question: str, drafts: list[dict]) -> dict | None:
    for d in drafts:
        if d.get("intent") == question:
            return d
    return None


def _retrieval_hit(draft: dict, target_show: str) -> bool:
    citations = draft.get("citations") or []
    return any(c.get("show") == target_show for c in citations)


def _answer_match(draft: dict, substrings: list[str]) -> bool:
    if not substrings:
        return False
    haystack = (draft.get("findings") or "").lower()
    haystack += " " + " ".join((c.get("excerpt") or "") for c in (draft.get("citations") or [])).lower()
    return any(s.lower() in haystack for s in substrings)


def score_fixed_answer(drafts: list[dict], fixture: dict) -> dict:
    """Score read_batch_corpus drafts against a geist_podcast_fixed_answer.json-shaped fixture.

    drafts are matched to fixture questions by intent == question text.
    Returns per-question detail plus an aggregate summary split by category, including the
    fabrication-avoidance-vs-suppression-avoidance spread (genuine-gatherer vs risk-averse
    discriminator the Mirror Council reservation asked for).
    """
    per_question = []
    for q in fixture.get("questions", []):
        question = q["question"]
        target_show = q.get("target_show", "")
        substrings = q.get("expected_answer_substrings") or []
        category = q["category"]
        answer_present = q["answer_present"]

        draft = _draft_for(question, drafts)
        if draft is None:
            per_question.append({
                "question": question,
                "category": category,
                "outcome": None,
                "retrieval_hit": None,
                "answer_match": None,
                "fabrication_avoidance": None,
                "suppression_avoidance": None,
                "note": "no matching draft (intent mismatch)",
            })
            continue

        outcome = draft.get("outcome")
        retrieval_hit = _retrieval_hit(draft, target_show) if answer_present else None
        answer_match = _answer_match(draft, substrings) if answer_present else None

        fabrication_avoidance = None
        suppression_avoidance = None

        if category == "absent-trap":
            fabrication_avoidance = outcome in (
                "no-answer-in-corpus", "no-relevant-chunks", "infra-unavailable",
            )
        elif category == "difficult-present":
            if retrieval_hit:
                suppression_avoidance = outcome == "sources-found" and bool(answer_match)

        per_question.append({
            "question": question,
            "category": category,
            "outcome": outcome,
            "retrieval_hit": retrieval_hit,
            "answer_match": answer_match,
            "fabrication_avoidance": fabrication_avoidance,
            "suppression_avoidance": suppression_avoidance,
        })

    def _rate(items: list[bool]) -> float | None:
        items = [i for i in items if i is not None]
        if not items:
            return None
        return sum(1 for i in items if i) / len(items)

    answerable = [p for p in per_question if p["category"] == "answerable"]
    absent_trap = [p for p in per_question if p["category"] == "absent-trap"]
    difficult_present = [p for p in per_question if p["category"] == "difficult-present"]

    fab_rate = _rate([p["fabrication_avoidance"] for p in absent_trap])
    suppression_rate = _rate([p["suppression_avoidance"] for p in difficult_present])
    spread = (
        fab_rate - suppression_rate
        if fab_rate is not None and suppression_rate is not None
        else None
    )

    aggregate = {
        "answerable": {
            "n": len(answerable),
            "retrieval_hit_rate": _rate([p["retrieval_hit"] for p in answerable]),
            "answer_match_rate": _rate([p["answer_match"] for p in answerable]),
        },
        "absent-trap": {
            "n": len(absent_trap),
            "fabrication_avoidance_rate": fab_rate,
        },
        "difficult-present": {
            "n": len(difficult_present),
            "retrieval_hit_rate": _rate([p["retrieval_hit"] for p in difficult_present]),
            "answer_match_rate": _rate([p["answer_match"] for p in difficult_present]),
            "suppression_avoidance_rate": suppression_rate,
        },
        "fabrication_vs_suppression_spread": spread,
    }

    return {"per_question": per_question, "aggregate": aggregate}
