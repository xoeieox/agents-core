"""ground() — unified ground-call substrate (ground-call-substrate-v0, H1.U1).

Assembles a token-bounded, cited context bundle from two halves:
  1. Decision substrate — retrieve() across mem / vault-rag / chub
  2. Live PM-state snapshot — MemoryStore + TargetStore (read-only, no lapis_pm import)

ground() NEVER raises. Degraded backends contribute provenance markers only.
ground() makes NO model call and references no ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

from agents_core.mem import MemoryStore
from agents_core.retrieval import retrieve
from agents_core.targets import TargetStore

log = logging.getLogger(__name__)

_DEFAULT_SCOPE = ["mem", "vault-rag", "chub"]
# Hard wall-clock guard for PM-state read. Widened from 200ms (which was below
# the measured ~412ms healthy read on BRIX) to 1.5s to catch only pathological
# lock/hang cases. Override without a code change via GROUND_PM_TIMEOUT_S env var.
_PM_TIMEOUT_S: float = float(os.environ.get("GROUND_PM_TIMEOUT_S", "1.5"))
_PM_WARN_FRACTION = 0.6  # emit warning when healthy read >= this fraction of the guard
_MEM_BIAS_NAMESPACES = {"decision", "project", "architecture"}


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
) -> GroundBundle:
    """Assemble a cited, token-bounded context bundle.

    Args:
        query:        Free-text question that guides retrieval.
        pm_state:     Include live PM-state snapshot when True.
        token_budget: Rough upper bound on output tokens (chars / 4 estimate).
        scope:        RAG/mem backends to query. Default: ["mem","vault-rag","chub"].

    Note:
        RAG backend timeouts are controlled by retrieve() and are env-overridable:
        RAG_HTTP_TIMEOUT (default 1.5 s) and CHUB_TIMEOUT (default 2.0 s).

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
        query, effective_scope, rag_budget
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

def _mem_biased_score(hit) -> float:
    """Return a biased sort key for mem hits in preferred namespaces.

    Adds 0.1 to the score of mem hits whose key starts with a preferred
    namespace (decision/, project/, architecture/) or whose tags include
    one of those terms. Capped at 1.0. All other hits are unchanged.
    """
    if hit.source != "mem":
        return hit.score
    key = hit.metadata.get("key", "")
    if not key and hit.id.startswith("mem:"):
        key = hit.id[4:]
    key_ns = key.split("/")[0] if "/" in key else key
    tags = {t.strip() for t in hit.metadata.get("tags", "").split(",") if t.strip()}
    if key_ns in _MEM_BIAS_NAMESPACES or bool(tags & _MEM_BIAS_NAMESPACES):
        return min(1.0, hit.score + 0.1)
    return hit.score


def _assemble_rag(
    query: str,
    scope: list[str],
    char_budget: int,
) -> tuple[str, list[dict], bool]:
    """Query retrieve() with per-backend fail-soft; return (block, provenance, truncated)."""
    provenance: list[dict] = []

    # No hard tag filter — retrieve mem unfiltered so project/architecture/decision
    # entries all reach the ranking step. Rank-bias applied post-retrieval via
    # _mem_biased_score().
    try:
        hits = retrieve(
            query,
            scope=scope,
            top_k=40,
            min_score=0.0,
        )
    except Exception as exc:
        log.warning("ground: retrieve() raised: %s", exc)
        for src in scope:
            provenance.append({
                "tag": "ungrounded",
                "source": src,
                "why": f"retrieve-error: {type(exc).__name__}",
            })
        return "", provenance, False

    # Rank-bias: boost mem hits in decision/project/architecture namespaces so they
    # float to the top without hard-excluding anything else.
    hits = sorted(hits, key=_mem_biased_score, reverse=True)

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
    t_start = time.monotonic()
    t.start()
    t.join(timeout=_PM_TIMEOUT_S)
    elapsed = time.monotonic() - t_start

    if t.is_alive() or result[0] is None:
        why = "timeout" if t.is_alive() else (
            f"error:{type(exc_holder[0]).__name__}" if exc_holder[0] else "error:unknown"
        )
        log.warning("ground: PM-state read failed (%s); continuing substrate-only", why)
        return "", [{"tag": "ungrounded", "source": "pm_state", "why": why}], True

    warn_threshold = _PM_TIMEOUT_S * _PM_WARN_FRACTION
    if elapsed >= warn_threshold:
        log.warning(
            "ground: PM-state read slow (%.0fms >= %.0fms soft threshold); "
            "approaching guard (%.0fms) - portfolio growth may cross the bound",
            elapsed * 1000,
            warn_threshold * 1000,
            _PM_TIMEOUT_S * 1000,
        )

    block, prov = result[0]
    return block, prov, False


def _do_pm_state_read() -> tuple[str, list[dict]]:
    """Inner PM-state read — runs inside the timeout thread.

    Reads:
      - TargetStore.active_targets() -> active + pm_bound targets (YAML, no mem)
      - router/lapis-pm/decisions/* (via mem key tag target:<id>)
      - pm/outstanding-brief/<id> (via direct mem get)
    """
    # TargetStore reads YAML files — open mem once and keep it for all lookups.
    mem = MemoryStore()
    try:
        ts = TargetStore()
        active = ts.active_targets()  # skip archived/dead targets; sorted by urgency

        active_bound = [t for t in active if t.pm_bound]
        if not active_bound:
            active_bound = active

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
    rows = mem.list_all(tags=["lapis-pm", f"target:{target_id}"], limit=20)
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
