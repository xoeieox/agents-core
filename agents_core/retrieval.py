"""Synapse Retrieval Engine — shared retrieval substrate for Synapse, Explorer, and PVC.

Execute queries against any combination of five backends and return merged, scored
results. No retrieval policy — policy lives at the caller layer.

Scope tokens and backend endpoints
===================================
  mem        — agents_core.mem.MemoryStore.search (SQLite FTS5, local DB)
  chub       — ``chub search <query> --json`` CLI (BM25-ish; honors filter key: ``type``)
  vault-rag  — POST http://203.0.113.12:8200/search (cosine sim; honors filter key: ``path_prefix``)
  room-rag   — POST http://203.0.113.12:8201/search (cosine sim; honors filter key: ``path_prefix``)
  code-rag   — POST http://203.0.113.12:8100/search (cosine sim; honors filter key: ``path_prefix``)

Filter dict
===========
Pass a flat dict; each backend consumes the keys it understands and ignores the
rest. Recognised keys per backend:

  mem      → ``tags``        (list[str]) — AND-intersected against stored tag CSV
  chub     → ``type``        (str)       — filters results by ``_type`` (e.g. "doc", "skill")
  *-rag    → ``path_prefix`` (str)       — passed as ``path_prefix`` in the POST body

Unrecognised filter keys are silently dropped per backend but logged at DEBUG level
so callers can catch typos.

Score normalisation
===================
Each backend's raw scores are min-max normalised to [0, 1] *per call*. FTS5 rank
(negative; lower = more relevant) is negated before normalising so that higher
values always mean more relevant across all backends. A single result from a backend
gets a normalised score of 1.0.

Fail-soft HTTP
==============
vault-rag, room-rag, and code-rag use a 5 s connect + read timeout. Any network
error or non-2xx response logs a warning and contributes [] to the merged result —
the engine never raises for HTTP failures. A degraded engine is better than a
failing engine.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field

import httpx

from agents_core.mem import MemoryStore

log = logging.getLogger(__name__)

KNOWN_SCOPE: frozenset[str] = frozenset({"mem", "chub", "vault-rag", "room-rag", "code-rag"})

_RAG_BASE_URLS: dict[str, str] = {
    "vault-rag": "http://203.0.113.12:8200",
    "room-rag":  "http://203.0.113.12:8201",
    "code-rag":  "http://203.0.113.12:8100",
}

HTTP_TIMEOUT = 5.0  # seconds


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
) -> list[Hit]:
    """Return up to top_k hits across the requested scope, sorted by score desc.

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

    all_hits: list[Hit] = []
    for source in dict.fromkeys(scope):  # deduplicate, preserve order
        if source == "mem":
            all_hits.extend(_search_mem(query, filters))
        elif source == "chub":
            all_hits.extend(_search_chub(query, filters))
        else:
            all_hits.extend(_search_rag(source, query, filters))

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
            timeout=10,
        )
        if result.returncode != 0:
            log.warning("retrieval: chub exited %d: %s", result.returncode, result.stderr[:200])
            return []
        data = json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as exc:
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

def _search_rag(source: str, query: str, filters: dict) -> list[Hit]:
    _warn_unknown_keys(source, filters, {"path_prefix"})

    base_url = _RAG_BASE_URLS[source]
    payload: dict = {"query": query, "limit": 50}
    if "path_prefix" in filters:
        payload["path_prefix"] = filters["path_prefix"]

    try:
        resp = httpx.post(f"{base_url}/search", json=payload, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        log.warning("retrieval: %s HTTP error: %s", source, exc)
        return []

    results: list[dict] = data.get("results", [])
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
