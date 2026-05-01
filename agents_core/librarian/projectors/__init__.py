"""agents_core.librarian.projectors — operational claim projectors.

A projector is a pure function over source-of-truth state (filesystem, mem.db)
that produces the same structured `answer` payload an LLM synthesis would have
produced — but without an LLM call, without citation verification, and with
``verification: "authoritative"`` in the resulting SynthesisArtifact.

Registered claims short-circuit ``corroborate()`` before its retrieve+LLM
path. Unregistered claims fall through to LLM synthesis as before.

Each projector is responsible for:
  - Reading source-of-truth state directly (no retrieval, no caching layer).
  - Returning a JSON-serialisable dict shaped to match the contract its
    consumer is already coded against (do NOT invent new shapes here —
    consumers exist).
  - Being side-effect-free: read-only, no mem writes, no file writes,
    no PR opens.

The registry is keyed by exact claim string (case-sensitive, whitespace-
significant). Lookup is O(1) and runs before any other corroborate work.
"""
from __future__ import annotations

from typing import Callable

from .targets import current_targets_state

# Projector signature: (scope, freshness, policy) -> dict (the `answer` payload).
# freshness and policy are passed for projectors that want to vary behaviour
# (e.g. include richer detail at low freshness), but most will ignore them.
ProjectorFn = Callable[[dict, int, str], dict]

REGISTRY: dict[str, ProjectorFn] = {
    "current-targets-state": current_targets_state,
}


def lookup(claim: str) -> ProjectorFn | None:
    """Return the projector for *claim*, or None if unregistered."""
    return REGISTRY.get(claim)
