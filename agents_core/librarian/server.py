"""FastAPI sidecar exposing ``agents_core.librarian.corroborate()`` over HTTP.

Consumed by claude-view's ``/api/librarian/panes/*`` routes. Default bind is
``127.0.0.1:9090``; the URL is overridable in claude-view via ``LIBRARIAN_URL``.

Wire format mirrors the Rust types in
``crates/server/src/routes/librarian.rs`` (claude-view repo): a discriminated
union tagged on ``type``, with values ``artifact`` or ``unavailable``.

Run::

    python3 -m uvicorn agents_core.librarian.server:app \\
        --host 127.0.0.1 --port 9090

Health probe::

    GET /health -> {"ok": true}
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents_core import synthesis_cache
from agents_core.librarian import (
    LibrarianUnavailable,
    SynthesisArtifact,
    _read_corpus_file,
    _read_old_content,
    corroborate,
)
from agents_core.librarian.shift import ShiftLevel, compute_shift_from_rows

log = logging.getLogger(__name__)

app = FastAPI(title="agents_core.librarian", version="0")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ScopeIn(BaseModel):
    corpus: list[str] = Field(default_factory=list)


class CorroborateIn(BaseModel):
    claim: str
    scope: ScopeIn
    freshness: int = 60
    policy: str = "auto-update"
    format: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


# ---------------------------------------------------------------------------
# Corroborate
# ---------------------------------------------------------------------------

@app.post("/v1/corroborate")
async def corroborate_endpoint(body: CorroborateIn) -> dict[str, Any]:
    """Translate HTTP request → ``corroborate()`` call → wire JSON.

    Never raises 5xx for downstream LLM/cache failures: ``corroborate()``
    returns ``LibrarianUnavailable`` on LLM unreachable, which is mapped to
    ``{"type": "unavailable", ...}``.  Unexpected exceptions become an
    unavailable response with the exception text in ``reason``.
    """
    try:
        result = await asyncio.to_thread(
            corroborate,
            body.claim,
            {"corpus": list(body.scope.corpus)},
            freshness=body.freshness,
            policy=body.policy,
            format=body.format,
        )
    except Exception as exc:
        log.exception("librarian sidecar: corroborate raised")
        return {
            "type": "unavailable",
            "most_recent_cached": None,
            "degraded": True,
            "reason": f"corroborate raised: {exc}",
        }

    if isinstance(result, LibrarianUnavailable):
        cached = (
            _artifact_wire(result.most_recent_cached)
            if result.most_recent_cached is not None
            else None
        )
        return {
            "type": "unavailable",
            "most_recent_cached": cached,
            "degraded": True,
            "reason": result.reason or "",
        }

    return {"type": "artifact", **_artifact_wire(result)}


# ---------------------------------------------------------------------------
# Artifact → wire dict
# ---------------------------------------------------------------------------

def _artifact_wire(art: SynthesisArtifact) -> dict[str, Any]:
    """Translate a SynthesisArtifact to the wire shape the Rust client expects.

    Adds a ``degree_of_shift`` field (computed from cached source rows) that
    the Python dataclass does not carry inline.
    """
    return {
        "artifact_id": art.artifact_id,
        "claim": art.claim,
        "scope_identity": art.scope_identity,
        "answer": art.answer,
        "citations": art.citations,
        "synthesized_at": art.synthesized_at,
        "degree_of_shift": _shift_for(art.artifact_id),
        "verification": art.verification,
        "policy": art.policy,
    }


def _shift_for(artifact_id: str) -> str | None:
    """Compute degree-of-shift for a cached artifact, or None if unknown."""
    try:
        rows = synthesis_cache.get_source_rows(artifact_id)
    except Exception:
        return None
    if not rows:
        return None
    try:
        level = compute_shift_from_rows(
            rows,
            read_old=_read_old_content,
            read_new=_read_corpus_file,
        )
    except Exception:
        return None
    return str(level)  # ShiftLevel.__str__ → "no-shift", "word-line", ...
