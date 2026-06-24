"""ground() — unified ground-call substrate (ground-call-substrate-v0, H1.U1).

Assembles a token-bounded, cited context bundle from two halves:
  1. Decision substrate — retrieve() across mem / vault-rag / chub
  2. Live PM-state snapshot — MemoryStore + TargetStore (read-only, no lapis_pm import)

ground() NEVER raises. Degraded backends contribute provenance markers only.
ground() makes NO model call and references no ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

from agents_core.mem import MemoryStore
from agents_core.retrieval import retrieve
from agents_core.targets import TargetStore

log = logging.getLogger(__name__)

_DEFAULT_SCOPE = ["mem", "vault-rag", "chub"]
_PM_TIMEOUT_S = 0.2  # hard non-blocking guard for MemoryStore/TargetStore reads


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundBundle:
    """Assembled, token-bounded context bundle ready for injection into a downstream turn."""
    context_block: str
    provenance: list[dict]
    truncated: bool
    stale: bool
    token_estimate: int


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ground(
    query: str,
    *,
    pm_state: bool = True,
    token_budget: int = 3000,
    scope: list[str] | None = None,
    timeout_s: float = 4.0,
) -> GroundBundle:
    """Assemble a cited, token-bounded context bundle.

    Args:
        query:        Free-text question that guides retrieval.
        pm_state:     Include live PM-state snapshot when True.
        token_budget: Rough upper bound on output tokens (chars / 4 estimate).
        scope:        RAG/mem backends to query. Default: ["mem","vault-rag","chub"].
        timeout_s:    Per-backend HTTP timeout for RAG calls.

    Returns:
        GroundBundle (never raises).
    """
    effective_scope = scope if scope is not None else list(_DEFAULT_SCOPE)
    char_budget = token_budget * 4
    provenance: list[dict] = []
    stale = False

    # -------------------------------------------------------------------------
    # Half 2 first: reserve PM-state slice before RAG fill
    # -------------------------------------------------------------------------
    pm_block = ""
    pm_provenance: list[dict] = []
    if pm_state:
        pm_block, pm_provenance, pm_failed = _assemble_pm_state()
        stale = pm_failed
        provenance.extend(pm_provenance)

    pm_chars = len(pm_block)
    rag_budget = max(0, char_budget - pm_chars)

    # -------------------------------------------------------------------------
    # Half 1: retrieval substrate
    # -------------------------------------------------------------------------
    rag_block, rag_provenance, truncated = _assemble_rag(
        query, effective_scope, rag_budget, timeout_s
    )
    provenance.extend(rag_provenance)

    # -------------------------------------------------------------------------
    # Assemble context_block — only real retrieved substrate, no error prose
    # -------------------------------------------------------------------------
    parts = []
    if rag_block:
        parts.append(rag_block)
    if pm_block:
        parts.append(pm_block)
    context_block = "\n\n".join(parts)

    token_estimate = len(context_block) // 4

    return GroundBundle(
        context_block=context_block,
        provenance=provenance,
        truncated=truncated,
        stale=stale,
        token_estimate=token_estimate,
    )


# ---------------------------------------------------------------------------
# Half 1 — RAG / decision substrate
# ---------------------------------------------------------------------------

def _assemble_rag(
    query: str,
    scope: list[str],
    char_budget: int,
    timeout_s: float,
) -> tuple[str, list[dict], bool]:
    """Query retrieve() with per-backend fail-soft; return (block, provenance, truncated)."""
    provenance: list[dict] = []

    # Bias mem hits toward decision/project/architecture namespaces
    mem_filters: dict = {"tags": ["decision"]}
    # We ask retrieve() for all in scope; but mem uses the tag filter to prefer
    # decision/project/architecture content. retrieve() handles the bias internally
    # via the tags filter on the mem backend.
    try:
        hits = retrieve(
            query,
            scope=scope,
            filters=mem_filters,
            top_k=40,
            min_score=0.0,
        )
    except Exception as exc:
        log.warning("ground: retrieve() raised: %s", exc)
        # Mark each backend as ungrounded
        for src in scope:
            provenance.append({
                "tag": "ungrounded",
                "source": src,
                "why": f"retrieve-error: {type(exc).__name__}",
            })
        return "", provenance, False

    # If retrieve returned zero from mem (no decision-tagged hits), retry without
    # the tag filter so we don't starve the bundle of any mem substrate.
    if not any(h.source == "mem" for h in hits):
        try:
            mem_hits = retrieve(query, scope=["mem"], top_k=20, min_score=0.0)
            hits = list({h.id: h for h in (mem_hits + hits)}.values())
            hits.sort(key=lambda h: h.score, reverse=True)
        except Exception:
            pass

    # Check which backends came back empty and record ungrounded markers
    returned_sources = {h.source for h in hits}
    for src in scope:
        if src not in returned_sources:
            provenance.append({
                "tag": "ungrounded",
                "source": src,
                "why": "no-hits-or-unreachable",
            })

    # Greedy fill within char_budget
    lines: list[str] = []
    used = 0
    truncated = False

    for hit in hits:
        tag = _citation_tag(hit.source, hit.id)
        line = f"[{tag}] {hit.content.strip()}"
        cost = len(line) + 1  # +1 for newline
        if used + cost > char_budget and char_budget > 0:
            truncated = True
            break
        lines.append(line)
        used += cost
        provenance.append({
            "tag": tag,
            "source": hit.source,
            "score": round(hit.score, 4),
            "why": "retrieved",
        })

    block = "\n".join(lines)
    return block, provenance, truncated


def _citation_tag(source: str, hit_id: str) -> str:
    """Map a Hit source + id to a citation tag with a recognized prefix."""
    source_prefix_map = {
        "mem": "mem",
        "vault-rag": "vault",
        "room-rag": "vault",
        "chub": "chub",
        "code-rag": "code",
    }
    prefix = source_prefix_map.get(source, source)
    # Strip the "mem:" prefix already in the id (ids are "<source>:<backend_id>")
    bare = re.sub(r"^[^:]+:", "", hit_id)
    return f"{prefix}:{bare}"


# ---------------------------------------------------------------------------
# Half 2 — live PM-state snapshot
# ---------------------------------------------------------------------------

def _assemble_pm_state() -> tuple[str, list[dict], bool]:
    """Read active targets + router decisions + outstanding briefs.

    Returns (block, provenance, failed).
    Runs under a hard _PM_TIMEOUT_S wall-clock guard.
    Never raises.
    """
    result: list = [None]  # [0] = (block, provenance) or None on failure
    exc_holder: list = [None]

    def _read() -> None:
        try:
            result[0] = _do_pm_state_read()
        except Exception as exc:  # noqa: BLE001
            exc_holder[0] = exc

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout=_PM_TIMEOUT_S)

    if t.is_alive() or result[0] is None:
        why = "timeout" if t.is_alive() else (
            f"error:{type(exc_holder[0]).__name__}" if exc_holder[0] else "error:unknown"
        )
        log.warning("ground: PM-state read failed (%s); continuing substrate-only", why)
        return "", [{"tag": "ungrounded", "source": "pm_state", "why": why}], True

    block, prov = result[0]
    return block, prov, False


def _do_pm_state_read() -> tuple[str, list[dict]]:
    """Inner PM-state read — runs inside the timeout thread.

    Reads:
      - TargetStore.load_all() -> active + pm_bound targets (YAML, no mem)
      - router/lapis-pm/decisions/* (via mem key tag target:<id>)
      - pm/outstanding-brief/<id> (via direct mem get)
    """
    # TargetStore reads YAML files — open mem once and keep it for all lookups.
    mem = MemoryStore()
    try:
        ts = TargetStore()
        targets = ts.load_all()

        active_bound = [t for t in targets if t.status == "active" and t.pm_bound]
        if not active_bound:
            active_bound = [t for t in targets if t.status == "active"]

        lines: list[str] = []
        provenance: list[dict] = []

        for target in active_bound[:8]:  # cap at 8 to keep snapshot small
            tid = target.id
            tag = f"pm:{tid}"

            decision = _find_latest_decision_via_mem(mem, tid)
            brief_rec = mem.get(f"pm/outstanding-brief/{tid}")
            brief = brief_rec["content"] if brief_rec else None

            status_parts = [f"target:{tid} ({target.title}) [{target.urgency}]"]
            if decision:
                intent = decision.get("intent_summary", "")
                verdict = decision.get("verdict", "")
                expert = decision.get("expert_chosen", "")
                status_parts.append(
                    f"  latest-decision: {verdict} / expert={expert}"
                    + (f" — {intent[:120]}" if intent else "")
                )
            if brief:
                status_parts.append(f"  outstanding-brief: {brief[:80]}")

            entry = "\n".join(status_parts)
            lines.append(f"[{tag}]\n{entry}")
            provenance.append({
                "tag": tag,
                "source": "pm_state",
                "score": None,
                "why": "active-target",
            })
    finally:
        mem.close()

    block = "\n\n".join(lines) if lines else ""
    return block, provenance


def _find_latest_decision_via_mem(mem: MemoryStore, target_id: str) -> dict | None:
    """Replicate router_portfolio.find_latest_decision without importing lapis_pm."""
    rows = mem.list_all(tags=["lapis-pm", f"target:{target_id}"], limit=200)
    best: dict | None = None
    best_stamp = ""
    for row in rows:
        key = row.get("key", "")
        if not key.startswith("router/lapis-pm/decisions/"):
            continue
        try:
            data = json.loads(row["content"])
        except (json.JSONDecodeError, KeyError):
            continue
        stamp = data.get("freshness_stamp", "")
        if stamp > best_stamp:
            best_stamp = stamp
            best = dict(data)
            best["_mem_key"] = key
    return best
