"""Dowser — external-web research funnel (query-gen -> SearXNG -> triage -> fetch -> deep-read).

Two tier-pure batch operations:
  read_batch()     — read operator only (quest / sonnet)
  critique_batch() — critic operator only (gravitywell)

GW-topology-agnostic: no flip, no doorman, no duty-cycle awareness.
Consumer (backcaster-quest-leg-v0) owns flip sequencing between the two phases.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

from agents_core.llm import call_operator, call_claude_cli, parse_json_object

_log = logging.getLogger(__name__)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://203.0.113.10:8888")

# Domain-credibility prior: reward authoritative, penalize junk
_CREDIBLE_DOMAINS = {
    "arxiv.org", "github.com", "docs.python.org", "developer.mozilla.org",
    "w3.org", "ietf.org", "rfc-editor.org", "interledger.org",
    "openpayments.guide", "openpayments.dev",
    "docs.mojaloop.io", "mojaloop.io",
    "en.wikipedia.org",
    "pypi.org", "readthedocs.io", "readthedocs.org",
    "stackoverflow.com",  # moderate
}
_JUNK_PATTERNS = re.compile(
    r"(reddit\.com|quora\.com|pinterest\.com|medium\.com/tag|"
    r"fiverr\.com|upwork\.com|guru\.com|freelancer\.com|"
    r"seo|affiliate|blogspot|wordpress\.com/\d{4})",
    re.IGNORECASE,
)
_HIGH_CREDIBILITY_THRESHOLD = 0.3  # fraction of hits that are credible; below => high-friction

FETCH_TIMEOUT_SEC = 15
FETCH_MAX_REDIRECTS = 5
FETCH_MAX_BYTES = 512 * 1024  # 512 KB
FETCH_MAX_CONTEXT_CHARS = 8000  # chars fed to LLM per page


# ---------------------------------------------------------------------------
# Domain credibility helpers
# ---------------------------------------------------------------------------

def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _credibility_score(url: str) -> float:
    domain = _domain_of(url)
    if any(domain == d or domain.endswith("." + d) for d in _CREDIBLE_DOMAINS):
        return 1.0
    if _JUNK_PATTERNS.search(url):
        return 0.1
    return 0.5


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
    """Returns (search_strings, rewrote_from_diagnosis)."""
    if prior_diagnosis:
        prompt = (
            f"You are a search strategist. A previous research pass for the following intent "
            f"failed with this diagnosis:\n\nDIAGNOSIS: {prior_diagnosis}\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + "Rewrite the search queries to address the diagnosis. "
            "If the diagnosis says 'all forum/SEO junk', target authoritative domains (add 'site:' hints or domain qualifiers). "
            "If 'off-topic', narrow the query. If 'thin', broaden. "
            "Return JSON: {\"queries\": [\"...\", ...]}"
        )
    elif sub_intents:
        prompt = (
            f"You are a search strategist. Generate targeted web search queries for this intent:\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + f"SUB-INTENTS (generate 1-2 queries per sub-intent): {json.dumps(sub_intents)}\n"
            "Return JSON: {\"queries\": [\"...\", ...]}"
        )
    else:
        prompt = (
            f"You are a search strategist. Generate 2-4 targeted web search queries for:\n\n"
            f"INTENT: {intent}\n"
            + (f"CONTEXT: {context}\n" if context else "")
            + "Return JSON: {\"queries\": [\"...\", ...]}"
        )

    raw = _call_read_operator(operator, prompt, json_mode=True)
    if not raw:
        # fallback: use intent as-is; no rewrite occurred
        return [intent], False

    parsed = parse_json_object(raw)
    if not parsed or not isinstance(parsed.get("queries"), list):
        return [intent], False

    queries = [q for q in parsed["queries"] if isinstance(q, str) and q.strip()]
    # rewrote_from_diagnosis is True only when the LLM successfully returned rewritten queries
    rewrote = prior_diagnosis is not None
    return (queries or [intent]), rewrote


# ---------------------------------------------------------------------------
# Stage 2: SEARCH (no LLM)
# ---------------------------------------------------------------------------

def _search_searxng(queries: list[str], searxng_url: str) -> tuple[list[dict], str | None]:
    """Returns (hits, error_note). error_note is set if SearXNG was unreachable."""
    seen_urls: set[str] = set()
    hits: list[dict] = []
    error_note = None

    with httpx.Client(timeout=10, follow_redirects=True) as client:
        for query in queries:
            try:
                resp = client.get(
                    searxng_url + "/search",
                    params={"q": query, "format": "json"},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                error_note = f"SearXNG error: {e}"
                _log.warning("[dowser] SearXNG error for query %r: %s", query, e)
                continue

            for result in data.get("results", []):
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    hits.append({
                        "url": url,
                        "title": result.get("title", ""),
                        "snippet": result.get("content", ""),
                    })

    return hits, error_note


# ---------------------------------------------------------------------------
# Stage 3: TRIAGE (read operator + domain prior)
# ---------------------------------------------------------------------------

def _triage(
    hits: list[dict],
    intent: str,
    top_k: int,
    operator: str,
) -> tuple[list[dict], float]:
    """Returns (triaged_hits, friction_ratio).

    friction_ratio: fraction of hits with low credibility score.
    High friction = many hits but few credible ones.
    """
    if not hits:
        return [], 0.0

    # Pre-score with domain prior
    scored = []
    for h in hits:
        score = _credibility_score(h["url"])
        scored.append({**h, "_cred": score})

    total = len(scored)
    credible_count = sum(1 for h in scored if h["_cred"] >= 0.5)
    friction_ratio = 1.0 - (credible_count / total) if total > 0 else 0.0

    # Ask the read operator to rank the top candidates
    # Feed at most 20 hits to keep prompt size bounded
    candidates = scored[:20]
    hits_text = "\n".join(
        f"{i+1}. [{h['_cred']:.1f}] {h['title']} | {h['url']}\n   {h['snippet'][:200]}"
        for i, h in enumerate(candidates)
    )
    prompt = (
        f"Rank these search results by relevance and credibility for the research intent.\n\n"
        f"INTENT: {intent}\n\nRESULTS:\n{hits_text}\n\n"
        f"Return JSON: {{\"top_indices\": [0-based indices of the top {top_k} results, best first]}}"
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
        # fallback: sort by credibility score
        sorted_hits = sorted(scored, key=lambda h: h["_cred"], reverse=True)
        top_indices = list(range(min(top_k, len(sorted_hits))))
        result_hits = [sorted_hits[i] for i in top_indices]
    else:
        result_hits = [candidates[i] for i in top_indices]

    # Strip internal _cred field
    return [{k: v for k, v in h.items() if k != "_cred"} for h in result_hits], friction_ratio


# ---------------------------------------------------------------------------
# Stage 4: FETCH (no LLM)
# ---------------------------------------------------------------------------

def _strip_html(html: str) -> str:
    """Minimal HTML -> plaintext: remove tags, collapse whitespace."""
    # Remove scripts/styles
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    html = html.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _fetch_pages(urls: list[str]) -> list[dict]:
    """Fetch each URL; return list of {url, text, error}."""
    results = []
    for url in urls:
        try:
            with httpx.Client(
                timeout=FETCH_TIMEOUT_SEC,
                follow_redirects=True,
                max_redirects=FETCH_MAX_REDIRECTS,
            ) as client:
                resp = client.get(url, headers={"User-Agent": "Dowser/0.1 research-bot"})
                resp.raise_for_status()
                raw = resp.content[:FETCH_MAX_BYTES]
                content_type = resp.headers.get("content-type", "")
                if "html" in content_type:
                    text = _strip_html(raw.decode("utf-8", errors="replace"))
                else:
                    text = raw.decode("utf-8", errors="replace")
                text = text[:FETCH_MAX_CONTEXT_CHARS]
                results.append({"url": url, "text": text, "error": None})
        except Exception as e:
            results.append({"url": url, "text": "", "error": str(e)})
            _log.debug("[dowser] fetch failed %s: %s", url, e)
    return results


# ---------------------------------------------------------------------------
# Stage 5: DEEP-READ (read operator)
# ---------------------------------------------------------------------------

def _deep_read(
    intent: str,
    context: str | None,
    pages: list[dict],
    operator: str,
) -> tuple[str, list[dict]]:
    """Returns (findings, citations)."""
    readable = [p for p in pages if p["text"]]
    if not readable:
        return "", []

    pages_text = ""
    for i, p in enumerate(readable):
        pages_text += f"\n\n--- SOURCE {i+1}: {p['url']} ---\n{p['text'][:FETCH_MAX_CONTEXT_CHARS]}"

    prompt = (
        f"You are a research analyst. Synthesize findings from the sources below for this intent.\n\n"
        f"INTENT: {intent}\n"
        + (f"CONTEXT: {context}\n" if context else "")
        + f"\nSOURCES:{pages_text}\n\n"
        "Rules:\n"
        "- Each claim MUST be supported by a direct excerpt from a source.\n"
        "- Never fabricate citations or invent content not in the sources.\n"
        "- If sources don't answer the intent, say so explicitly.\n"
        "Return JSON:\n"
        '{"findings": "prose summary", "citations": [{"url": "...", "title": "...", '
        '"excerpt": "direct quote from source", "credibility": "high|medium|low"}]}'
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
        url = c.get("url", "")
        excerpt = c.get("excerpt", "")
        if not url or not excerpt:
            continue
        citations.append({
            "url": url,
            "title": c.get("title", ""),
            "excerpt": excerpt,
            "credibility": c.get("credibility", "medium"),
        })

    return findings, citations


# ---------------------------------------------------------------------------
# LLM dispatch helpers (tier-pure)
# ---------------------------------------------------------------------------

def _call_read_operator(operator: str, prompt: str, json_mode: bool = False) -> str | None:
    """Call only the read operator. Never calls the critic."""
    try:
        if operator in ("sonnet", "haiku", "opus"):
            # Route via call_claude_cli — no direct Anthropic API
            suffix = "\n\nRespond ONLY with valid JSON." if json_mode else ""
            return call_claude_cli(prompt + suffix, model=operator, timeout=120)
        else:
            # quest or other GW-side operators
            return call_operator(operator, prompt, json_mode=json_mode, on_wake_fail="skip",
                                 lease_class="deferrable")
    except Exception as e:
        _log.warning("[dowser] read operator %r failed: %s", operator, e)
        return None


def _call_critic_operator(operator: str, prompt: str, json_mode: bool = False) -> str | None:
    """Call only the critic operator. Never calls the reader."""
    try:
        return call_operator(operator, prompt, json_mode=json_mode, on_wake_fail="skip",
                             lease_class="deferrable")
    except Exception as e:
        _log.warning("[dowser] critic operator %r failed: %s", operator, e)
        return None


# ---------------------------------------------------------------------------
# Typed outcome helpers
# ---------------------------------------------------------------------------

def _classify_outcome(
    hits: list[dict],
    error_note: str | None,
    friction_ratio: float,
    findings: str,
) -> str:
    # Partial failure (error_note set but hits non-empty from other queries): the infra error
    # survives in provenance.notes but we let outcome reflect the actual content quality.
    # Full infra failure (no hits at all) is the only case we surface as infra-unavailable.
    if error_note and not hits:
        return "infra-unavailable"
    if not findings:
        if friction_ratio >= (1.0 - _HIGH_CREDIBILITY_THRESHOLD) and hits:
            return "high-friction"
        return "no-credible-sources"
    return "sources-found"


# ---------------------------------------------------------------------------
# Public API: read_batch
# ---------------------------------------------------------------------------

def read_batch(
    requests_list: list[dict],
    read_operator: str = "quest",
    budget: dict | None = None,
) -> dict:
    """Run the full read funnel for each request.

    Each request: {intent, context?, sub_intents?, prior_diagnosis?}
    Returns: {drafts: [{intent, findings, citations, outcome, provenance}]}
    """
    budget = budget or {}
    deep_read_urls = int(budget.get("deep_read_urls", 5))
    wall_clock_sec = int(budget.get("wall_clock_sec", 300))
    deadline = time.monotonic() + wall_clock_sec

    drafts = []
    for req in requests_list:
        if time.monotonic() >= deadline:
            _log.warning("[dowser] wall_clock_sec budget exhausted; skipping remaining requests")
            drafts.append(_infra_null_draft(req["intent"], "wall_clock_budget_exhausted"))
            continue

        intent = req["intent"]
        context = req.get("context")
        sub_intents = req.get("sub_intents")
        prior_diagnosis = req.get("prior_diagnosis")

        # Stage 1: query-gen
        try:
            queries, rewrote = _query_gen(
                intent, context, sub_intents, prior_diagnosis, read_operator
            )
        except Exception as e:
            _log.warning("[dowser] query_gen failed for %r: %s", intent, e)
            queries = [intent]
            rewrote = False

        # Stage 2: search
        hits, error_note = _search_searxng(queries, SEARXNG_URL)

        # Stage 3: triage
        if hits:
            try:
                triaged, friction_ratio = _triage(hits, intent, deep_read_urls, read_operator)
            except Exception as e:
                _log.warning("[dowser] triage failed: %s", e)
                triaged = hits[:deep_read_urls]
                friction_ratio = 0.0
        else:
            triaged = []
            friction_ratio = 0.0

        triaged_urls = [h["url"] for h in triaged]

        # Stage 4: fetch
        pages = _fetch_pages(triaged_urls)
        read_urls = [p["url"] for p in pages if not p["error"]]

        # Stage 5: deep-read
        findings = ""
        citations: list[dict] = []
        if read_urls:
            try:
                findings, citations = _deep_read(intent, context, pages, read_operator)
            except Exception as e:
                _log.warning("[dowser] deep_read failed: %s", e)

        outcome = _classify_outcome(hits, error_note, friction_ratio, findings)

        notes = []
        if error_note:
            notes.append(error_note)
        unfetchable = [p["url"] for p in pages if p["error"]]
        if unfetchable:
            notes.append(f"unfetchable: {unfetchable}")

        drafts.append({
            "intent": intent,
            "findings": findings,
            "citations": citations,
            "outcome": outcome,
            "provenance": {
                "search_strings": queries,
                "hits_count": len(hits),
                "triaged_urls": triaged_urls,
                "read_urls": read_urls,
                "friction_ratio": round(friction_ratio, 3),
                "rewrote_from_diagnosis": rewrote,
                "notes": notes,
            },
        })

    return {"drafts": drafts}


def _infra_null_draft(intent: str, note: str) -> dict:
    return {
        "intent": intent,
        "findings": "",
        "citations": [],
        "outcome": "infra-unavailable",
        "provenance": {
            "search_strings": [],
            "hits_count": 0,
            "triaged_urls": [],
            "read_urls": [],
            "friction_ratio": 0.0,
            "rewrote_from_diagnosis": False,
            "notes": [note],
        },
    }


# ---------------------------------------------------------------------------
# Public API: critique_batch
# ---------------------------------------------------------------------------

def critique_batch(
    drafts: list[dict],
    critic_operator: str = "gravitywell",
) -> dict:
    """Score each draft for relevance, credibility, and faithfulness.

    Faithfulness = real fetched excerpt supports each claim.
    Returns: {verdicts: [{intent, status, verdict, diagnosis}]}
    """
    verdicts = []
    for draft in drafts:
        intent = draft.get("intent", "")
        findings = draft.get("findings", "")
        citations = draft.get("citations") or []

        if not findings or not citations:
            verdicts.append({
                "intent": intent,
                "status": "subpar",
                "verdict": {
                    "relevance": 0,
                    "credibility": 0,
                    "faithfulness": 0,
                    "confidence": "high",
                },
                "diagnosis": "insufficient-sources",
            })
            continue

        citations_text = "\n".join(
            f"- [{c.get('credibility', 'medium')}] {c.get('url', '')}\n  EXCERPT: {c.get('excerpt', '')[:500]}"
            for c in citations[:10]
        )
        prompt = (
            f"You are a rigorous fact-checker. Evaluate this research finding.\n\n"
            f"INTENT: {intent}\n\n"
            f"FINDINGS:\n{findings[:2000]}\n\n"
            f"CITATIONS:\n{citations_text}\n\n"
            "Score the following (0-10):\n"
            "- relevance: does the finding actually address the intent?\n"
            "- credibility: are the sources authoritative and reliable?\n"
            "- faithfulness: is every claim in findings supported by a direct excerpt? "
            "(This is critical — if any claim lacks a real excerpt, score low)\n\n"
            "Then give a verdict: 'pass' if all scores >= 6, otherwise 'subpar'.\n"
            "Return JSON:\n"
            '{"relevance": N, "credibility": N, "faithfulness": N, '
            '"confidence": "high|medium|low", "status": "pass|subpar", '
            '"diagnosis": "one sentence explanation if subpar, else empty string"}'
        )

        raw = _call_critic_operator(critic_operator, prompt, json_mode=True)
        if not raw:
            verdicts.append({
                "intent": intent,
                "status": "subpar",
                "verdict": {"relevance": 0, "credibility": 0, "faithfulness": 0, "confidence": "low"},
                "diagnosis": "critic-operator-unavailable",
            })
            continue

        parsed = parse_json_object(raw)
        if not parsed:
            verdicts.append({
                "intent": intent,
                "status": "subpar",
                "verdict": {"relevance": 0, "credibility": 0, "faithfulness": 0, "confidence": "low"},
                "diagnosis": "critic-parse-error",
            })
            continue

        status = parsed.get("status", "subpar")
        if status not in ("pass", "subpar"):
            status = "subpar"

        verdicts.append({
            "intent": intent,
            "status": status,
            "verdict": {
                "relevance": int(parsed.get("relevance", 0)),
                "credibility": int(parsed.get("credibility", 0)),
                "faithfulness": int(parsed.get("faithfulness", 0)),
                "confidence": parsed.get("confidence", "medium"),
            },
            "diagnosis": parsed.get("diagnosis", ""),
        })

    return {"verdicts": verdicts}
