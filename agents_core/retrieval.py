"""Synapse Retrieval Engine — shared retrieval substrate for Synapse, Explorer, and PVC.

Execute queries against any combination of five backends and return merged, scored
results. No retrieval policy — policy lives at the caller layer.

Scope tokens and backend endpoints
===================================
  mem        — agents_core.mem.MemoryStore.search (SQLite FTS5, local DB)
  chub       — ``chub search <query> --json`` CLI (BM25-ish; honors filter key: ``type``)
  vault-rag  — POST http://<RAG_HOST>:8200/search (cosine sim; honors filter key: ``doc_type``)
  room-rag   — POST http://<RAG_HOST>:8201/search (cosine sim; honors filter key: ``doc_type``)
  code-rag   — POST http://<RAG_HOST>:8100/search (cosine sim; honors filter key: ``doc_type``)

Default RAG_HOST is 203.0.113.10. Override via env:
  RAG_HOST        — shared host for all three RAG backends
  VAULT_RAG_URL   — full base URL for vault-rag (overrides RAG_HOST for this backend)
  ROOM_RAG_URL    — full base URL for room-rag
  CODE_RAG_URL    — full base URL for code-rag
  RAG_HTTP_TIMEOUT — per-call wall-clock limit for RAG HTTP (default 1.5 s)
  CHUB_TIMEOUT    — subprocess wall-clock limit for chub (default 1.5 s)

Filter dict
===========
Pass a flat dict; each backend consumes the keys it understands and ignores the
rest. Recognised keys per backend:

  mem      → ``tags``        (list[str]) — AND-intersected against stored tag CSV
  chub     → ``type``        (str)       — filters results by ``_type`` (e.g. "doc", "skill")
  *-rag    → ``doc_type``    (str)       — passed via a per-backend payload adapter (see
                                            _RAG_PAYLOAD_ADAPTER); ``path_prefix`` is also
                                            still forwarded but is a confirmed no-op on the
                                            live room-rag server (kept only for backends that
                                            may honor it — do not rely on it for room-rag).

Unrecognised filter keys are silently dropped per backend but logged at DEBUG level
so callers can catch typos.

Pagination — ``top_k`` is forwarded to *-rag backends as their real pagination field via
the same per-backend adapter (room-rag confirmed live: ``n_results``, not ``limit``). If a
backend appears to have silently ignored ``doc_type`` or under-filled a request despite more
results being available, ``_search_rag`` logs a loud WARNING (fail-open canary) rather than
degrading invisibly — see ``_search_rag``.

Score normalisation
===================
Each backend's raw scores are min-max normalised to [0, 1] *per call*. FTS5 rank
(negative; lower = more relevant) is negated before normalising so that higher
values always mean more relevant across all backends. A single result from a backend
gets a normalised score of 1.0.

Fail-soft HTTP
==============
vault-rag, room-rag, and code-rag use a short connect + read timeout (RAG_HTTP_TIMEOUT,
default 1.5 s; connect timeout 0.5 s). Any network error or non-2xx response logs a
warning and contributes [] to the merged result — the engine never raises for HTTP
failures. A degraded engine is better than a failing engine.

Parallel fan-out
================
All backends are queried concurrently (ThreadPoolExecutor). retrieve() latency is
~max(backend), not the sum. A slow or dead backend degrades to [] without blocking
the others.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx

from agents_core.mem import MemoryStore

log = logging.getLogger(__name__)

KNOWN_SCOPE: frozenset[str] = frozenset({"mem", "chub", "vault-rag", "room-rag", "code-rag"})

_RAG_HOST: str = os.environ.get("RAG_HOST", "203.0.113.10")
_RAG_BASE_URLS: dict[str, str] = {
    "vault-rag": os.environ.get("VAULT_RAG_URL", f"http://{_RAG_HOST}:8200"),
    "room-rag":  os.environ.get("ROOM_RAG_URL",  f"http://{_RAG_HOST}:8201"),
    "code-rag":  os.environ.get("CODE_RAG_URL",  f"http://{_RAG_HOST}:8100"),
}

RAG_HTTP_TIMEOUT: float = float(os.environ.get("RAG_HTTP_TIMEOUT", "1.5"))
CHUB_TIMEOUT: float = float(os.environ.get("CHUB_TIMEOUT", "1.5"))


@dataclass(frozen=True)
class Hit:
    id: str        # stable across backends; format: "<source>:<backend_id>"
    score: float   # 0.0–1.0, normalised per backend per call
    source: str    # "mem" | "chub" | "vault-rag" | "room-rag" | "code-rag"
    content: str   # text content (may be truncated — see metadata["truncated"])
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


def retrieve(
    query: str,
    scope: list[str],
    filters: dict | None = None,
    top_k: int = 10,
    min_score: float = 0.0,
    exclude: set[str] | None = None,
    timeout: float | None = None,
) -> list[Hit]:
    """Return up to top_k hits across the requested scope, sorted by score desc.

    All backends are queried concurrently. A slow or dead backend degrades to []
    without blocking the others.

    Args:
        query:     Free-text search query.
        scope:     Subset of KNOWN_SCOPE tokens to query. Empty returns [].
        filters:   Flat dict of backend-specific filter keys. Each backend
                   consumes what it understands and ignores the rest.
        top_k:     Maximum number of hits returned after merging.
        min_score: Exclude hits with normalised score below this threshold
                   (applied after per-backend normalisation).
        exclude:   Set of Hit.id values to omit from results. Applied after
                   retrieval — backends don't need to support it natively.
        timeout:   Per-call wall-clock override for RAG HTTP backends (seconds).
                   Overrides RAG_HTTP_TIMEOUT for this call only. Default None
                   preserves the module-level RAG_HTTP_TIMEOUT for all callers.

    Returns:
        List of Hit objects, sorted by score descending, at most top_k items.

    Raises:
        ValueError: If scope contains unknown tokens.
    """
    if not scope:
        return []

    unknown = set(scope) - KNOWN_SCOPE
    if unknown:
        raise ValueError(
            f"Unknown scope token(s): {sorted(unknown)}. Valid: {sorted(KNOWN_SCOPE)}"
        )

    filters = filters or {}
    exclude = exclude or set()

    sources = list(dict.fromkeys(scope))  # deduplicated, preserve order

    def _query_source(src: str) -> list[Hit]:
        try:
            if src == "mem":
                return _search_mem(query, filters)
            elif src == "chub":
                return _search_chub(query, filters)
            else:
                return _search_rag(src, query, filters, top_k=top_k, timeout=timeout)
        except Exception as exc:
            log.warning("retrieval: %s raised unexpectedly: %s", src, exc)
            return []

    all_hits: list[Hit] = []
    with ThreadPoolExecutor(max_workers=max(len(sources), 1)) as ex:
        fut_to_src = {ex.submit(_query_source, src): src for src in sources}
        for fut in as_completed(fut_to_src):
            all_hits.extend(fut.result())

    # post-retrieval filtering
    if exclude:
        all_hits = [h for h in all_hits if h.id not in exclude]
    if min_score > 0.0:
        all_hits = [h for h in all_hits if h.score >= min_score]

    all_hits.sort(key=lambda h: h.score, reverse=True)
    return all_hits[:top_k]


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _normalise(raw_scores: list[float], *, invert: bool = False) -> list[float]:
    """Min-max normalise raw_scores to [0, 1]; invert negates first (for FTS5 rank)."""
    if not raw_scores:
        return []
    scores = [-s for s in raw_scores] if invert else list(raw_scores)
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


# ---------------------------------------------------------------------------
# Backend: mem
# ---------------------------------------------------------------------------

def _search_mem(query: str, filters: dict) -> list[Hit]:
    _warn_unknown_keys("mem", filters, {"tags"})

    tags: list[str] | None = filters.get("tags")
    store = MemoryStore()
    try:
        rows = store.search(query, limit=50)
    finally:
        store.close()

    if tags:
        tag_set = set(tags)
        rows = [
            r for r in rows
            if tag_set.issubset(
                {t.strip() for t in r.get("tags", "").split(",") if t.strip()}
            )
        ]

    if not rows:
        return []

    raw = [r["rank"] for r in rows]
    scores = _normalise(raw, invert=True)

    return [
        Hit(
            id=f"mem:{r['key']}",
            score=s,
            source="mem",
            content=r["content"],
            metadata={
                "key": r["key"],
                "tags": r.get("tags", ""),
                "source": r.get("source", ""),
                "updated_at": r.get("updated_at", ""),
            },
        )
        for r, s in zip(rows, scores)
    ]


# ---------------------------------------------------------------------------
# Backend: chub
# ---------------------------------------------------------------------------

def _search_chub(query: str, filters: dict) -> list[Hit]:
    _warn_unknown_keys("chub", filters, {"type"})

    try:
        result = subprocess.run(
            ["chub", "search", "--json", query],
            capture_output=True,
            text=True,
            timeout=CHUB_TIMEOUT,
        )
        if result.returncode != 0:
            log.warning("retrieval: chub exited %d: %s", result.returncode, result.stderr[:200])
            return []
        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        log.warning("retrieval: chub timed out after %.1fs (killed)", CHUB_TIMEOUT)
        return []
    except (json.JSONDecodeError, FileNotFoundError) as exc:
        log.warning("retrieval: chub search failed: %s", exc)
        return []

    results: list[dict] = data.get("results", [])

    # chub CLI has no --type flag; filter post-hoc
    type_filter: str | None = filters.get("type")
    if type_filter:
        results = [r for r in results if r.get("_type") == type_filter]

    if not results:
        return []

    raw = [r.get("_score", 0.0) for r in results]
    scores = _normalise(raw)

    return [
        Hit(
            id=f"chub:{r['id']}",
            score=s,
            source="chub",
            content=r.get("description", ""),
            metadata={
                "name": r.get("name", ""),
                "type": r.get("_type", ""),
                "tags": r.get("tags", []),
                "source_registry": r.get("_source", ""),
            },
        )
        for r, s in zip(results, scores)
    ]


# ---------------------------------------------------------------------------
# Backend: *-rag (vault-rag / room-rag / code-rag)
# ---------------------------------------------------------------------------

# Backend-agnostic payload adapter: maps retrieve()'s logical keys (top_k, doc_type) to
# each *-rag backend's actual SearchRequest field name. Isolates backend-specific field
# naming so a fix for one backend cannot silently regress another.
#
# room-rag confirmed live via its /openapi.json SearchRequest schema (2026-07-13): the
# pagination field is ``n_results`` (default 8), NOT ``limit`` — the old hardcoded
# ``{"limit": 50}`` payload was silently ignored, capping every query at 8 global
# candidates regardless of caller top_k. ``doc_type`` is honored as-is.
# vault-rag/code-rag are assumed to share the same SearchRequest schema (same FastAPI
# template) pending independent live confirmation — the fail-open canary below surfaces
# a loud warning if that assumption is ever wrong for either.
_RAG_PAYLOAD_ADAPTER: dict[str, dict[str, str]] = {
    "room-rag":  {"top_k": "n_results", "doc_type": "doc_type"},
    "vault-rag": {"top_k": "n_results", "doc_type": "doc_type"},
    "code-rag":  {"top_k": "n_results", "doc_type": "doc_type"},
}
_DEFAULT_RAG_ADAPTER = {"top_k": "n_results", "doc_type": "doc_type"}


def _search_rag(
    source: str, query: str, filters: dict, top_k: int = 10, timeout: float | None = None
) -> list[Hit]:
    _warn_unknown_keys(source, filters, {"path_prefix", "doc_type"})

    adapter = _RAG_PAYLOAD_ADAPTER.get(source, _DEFAULT_RAG_ADAPTER)
    base_url = _RAG_BASE_URLS[source]
    payload: dict = {"query": query, adapter["top_k"]: top_k}
    if "path_prefix" in filters:
        payload["path_prefix"] = filters["path_prefix"]
    requested_doc_type = filters.get("doc_type")
    if requested_doc_type:
        payload[adapter["doc_type"]] = requested_doc_type

    effective_timeout = timeout if timeout is not None else RAG_HTTP_TIMEOUT
    timeout = httpx.Timeout(effective_timeout, connect=min(0.5, effective_timeout))
    try:
        resp = httpx.post(f"{base_url}/search", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("retrieval: %s HTTP error: %s", source, exc)
        return []

    results: list[dict] = data.get("results", [])

    # Fail-open canary: never crash on this, but if a requested scoping/pagination field
    # was silently ignored by the backend, log loudly instead of degrading invisibly —
    # this is exactly the bug class (limit/path_prefix silently dropped) this fix addresses.
    if requested_doc_type:
        seen_doc_types = {r.get("doc_type") for r in results if r.get("doc_type")}
        if seen_doc_types - {requested_doc_type}:
            log.warning(
                "retrieval: %s may be IGNORING doc_type=%r — response contains other "
                "doc_type(s) %s; verify the backend's SearchRequest schema still honors "
                "this field",
                source, requested_doc_type, sorted(seen_doc_types),
            )
    total_results = data.get("total_results")
    if total_results is not None and len(results) < min(top_k, total_results):
        log.warning(
            "retrieval: %s returned %d result(s) but requested top_k=%d with %d available "
            "(total_results) — the pagination field %r may have been silently ignored",
            source, len(results), top_k, total_results, adapter["top_k"],
        )

    if not results:
        return []

    # RAG backends return 0-1 cosine similarity; still normalise for consistent
    # cross-backend ranking within a call.
    raw = [r.get("score", 0.0) for r in results]
    scores = _normalise(raw)

    return [
        Hit(
            id=f"{source}:{r.get('file_path', r.get('id', str(i)))}",
            score=s,
            source=source,
            content=r.get("content", ""),
            metadata={k: v for k, v in r.items() if k not in ("content", "score")},
        )
        for i, (r, s) in enumerate(zip(results, scores))
    ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _warn_unknown_keys(backend: str, filters: dict, known: set[str]) -> None:
    unexpected = set(filters) - known
    if unexpected:
        log.debug(
            "retrieval: backend %r ignoring unrecognised filter key(s): %s",
            backend,
            sorted(unexpected),
        )
