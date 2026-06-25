"""roadmap — committed-plan snapshot + decision-lineage grounding for the roadmap-mirror.

roadmap-mirror-lineage-retriever-v0: Unit 1 of 3.
No LLM calls. No writes to mem.db. Read-only over MemoryStore.

Four public surfaces:
  materialize_committed_plan()   snapshot materializer (ledger → on-disk artifact)
  walk_lineage(seed_keys, ...)   typed key-mention edge walker
  ground_roadmap(query, ...)     GroundBundle for weaver injection
  ROADMAP_MIRROR_PREAMBLE        system preamble constant for GW-122B
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from agents_core.ground import GroundBundle
from agents_core.mem import MemoryStore
from agents_core.retrieval import retrieve

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROADMAP_NAMESPACES = frozenset({
    "decision", "project", "progress", "chain", "architecture", "strategy"
})

# Key-mention edge extraction (broader than filter — captures cross-namespace refs too)
_EDGE_PATTERN = re.compile(
    r"\b(decision|project|progress|chain|architecture|strategy"
    r"|infra|audit|incident|review|feedback)"
    r"/([a-z0-9][a-z0-9-]+)"
)

# Operators that make an edge "strong" when found near a key mention (±120 chars)
_STRONG_OP = re.compile(r"\b(supersed\w*|depends|sequences-before)\b", re.IGNORECASE)

_LANDED_RE = re.compile(r"\b(LANDED|MERGED|DEPLOYED|LIVE|CLOSED)\b", re.IGNORECASE)
_DEFERRED_RE = re.compile(r"\bdeferred?\b", re.IGNORECASE)

_MAX_WALK_NODES = 40
_SUMMARY_MAX_CHARS = 300

_LAPIS_STATE = Path(os.environ.get("LAPIS_STATE", "/data/lapis-state"))
_DEFAULT_SNAPSHOT_PATH = Path(
    os.environ.get(
        "ROADMAP_SNAPSHOT_PATH",
        str(_LAPIS_STATE / "roadmap" / "committed-plan.json"),
    )
)

EdgeType = Literal["strong", "mention"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RoadmapEdge:
    target_key: str
    edge_type: EdgeType
    operator: str = ""  # matched operator token for strong edges; "" for mention


@dataclass
class RoadmapItem:
    key: str
    namespace: str
    created_at: str
    updated_at: str
    tags: str
    summary: str
    outbound_edges: list[RoadmapEdge]
    status: str  # landed | in-flight | deferred | superseded | unknown


@dataclass
class LineageNode:
    key: str
    namespace: str
    created_at: str
    updated_at: str
    summary: str
    status: str
    edges_in: list[RoadmapEdge]  # edges that brought this node into the walk
    hop: int


# ---------------------------------------------------------------------------
# 4. Five-mode reply preamble
# ---------------------------------------------------------------------------

ROADMAP_MIRROR_PREAMBLE = """\
You are the ROADMAP-MIRROR for Erah's Lapis project — read-only and propose-only. \
You observe, reflect, and advise. You never auto-mutate the roadmap or assert \
ownership without evidence from the committed-plan snapshot.

## Reply modes (v0: ALREADY-HAVE and COMPOST enforced; SLOTS / LOSES / FORK deferred)

### ALREADY-HAVE — Echo-chamber guard (REQUIRED)
A proposal may only be confirmed as already owned when it maps to a CANONICAL KEY \
in the materialized committed-plan snapshot. Lexical keyword overlap is NOT confirmation.

- Confirmed canonical-key match: say "You already have this: <key> (status: <status>, \
date: <created_at>)." Cite the canonical ledger key.
- Lexical-only candidate with no canonical-key match: say "This looks related to <key> \
— is it the same thing?" A question, not an assertion. Never claim ownership on keywords alone.

### COMPOST — When a proposal does not slot into the current roadmap
Say the idea is composted into the idea corpus (simmering, retrievable), and name the \
real next step. Emit the structured marker on its own line so the system can act on it:
  <!-- LAPIS-COMPOST: <one-line summary of the composted idea> -->

### SLOTS — Deferred (v0)
Reordering existing roadmap items is not enabled in this version.

### LOSES — Deferred (v0)
Deprioritization logic is not wired in v0.

### FORK — Human-required direction decision
When a proposal requires a direction choice only Erah can make, surface it as a fork \
and ask for a yes/no or A/B choice. Do not decide it yourself.

## Posture
- Grounded: base replies on the ROADMAP-PORTRAIT and ROADMAP-LINEAGE context blocks injected above.
- When recall is flagged recall:fts-only or thin, treat lineage connections as uncertain \
and say so — do not fabricate a lineage.
- If context is absent or stale, say so before answering.
- Advisory authority only. All outputs are mirrors, not mandates.
"""


# ---------------------------------------------------------------------------
# 1. Committed-plan snapshot materializer
# ---------------------------------------------------------------------------

def materialize_committed_plan(
    *,
    snapshot_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict | None:
    """Snapshot roadmap-namespace entries from mem.db to a stable on-disk artifact.

    Returns the snapshot dict on success, None on hard failure. Never raises.
    Idempotent: same ledger state produces byte-identical file on disk (same content_hash;
    materialized_at is only updated when content changes).
    """
    dest = Path(snapshot_path) if snapshot_path else _DEFAULT_SNAPSHOT_PATH
    try:
        return _materialize(dest, db_path=Path(db_path) if db_path else None)
    except Exception as exc:
        log.error("roadmap: materialize_committed_plan failed: %s", exc)
        return None


def _materialize(dest: Path, *, db_path: Path | None) -> dict:
    store = MemoryStore(db_path) if db_path else MemoryStore()
    try:
        raw_rows = _load_roadmap_rows(store)
    finally:
        store.close()

    items_with_content: list[tuple[RoadmapItem, str]] = [
        (_build_item(row), row.get("content", ""))
        for row in raw_rows
    ]
    _mark_superseded(items_with_content)

    items = sorted(
        (item for item, _ in items_with_content),
        key=lambda i: i.key,
    )

    payload: dict = {"items": [_item_to_dict(i) for i in items]}
    content_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()

    # Idempotency: return existing artifact unchanged when content hasn't changed
    existing = _load_snapshot_file(dest)
    if existing and existing.get("content_hash") == content_hash:
        return existing

    artifact = {
        "content_hash": content_hash,
        "item_count": len(items),
        "materialized_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "roadmap-v0",
        **payload,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    log.info(
        "roadmap: snapshot written to %s (%d items, hash=%s)",
        dest, len(items), content_hash[:12],
    )
    return artifact


def _load_roadmap_rows(store: MemoryStore) -> list[dict]:
    rows: list[dict] = []
    for ns in sorted(_ROADMAP_NAMESPACES):
        rows.extend(store.list_by_prefix(f"{ns}/", limit=500))
    return rows


def _build_item(row: dict) -> RoadmapItem:
    key = row["key"]
    ns = key.split("/")[0] if "/" in key else ""
    content = row.get("content", "")
    return RoadmapItem(
        key=key,
        namespace=ns,
        created_at=row.get("created_at", ""),
        updated_at=row.get("updated_at", ""),
        tags=row.get("tags", ""),
        summary=_extract_summary(content),
        outbound_edges=_extract_edges(content),
        status=_infer_status(content),
    )


def _extract_summary(content: str) -> str:
    paras = re.split(r"\n\s*\n", content, maxsplit=1)
    return (paras[0].strip() if paras else "")[:_SUMMARY_MAX_CHARS]


def _extract_edges(content: str) -> list[RoadmapEdge]:
    """Extract outbound key-mention edges with typed classification.

    Uses best-match semantics: if a key is mentioned multiple times and at least
    one occurrence is near a strong operator, the edge is classified as strong.
    Insertion order (first mention) is preserved for determinism.
    """
    order: list[str] = []
    best: dict[str, RoadmapEdge] = {}

    for m in _EDGE_PATTERN.finditer(content):
        target = f"{m.group(1)}/{m.group(2)}"
        win_start = max(0, m.start() - 60)
        win_end = min(len(content), m.end() + 60)
        op_m = _STRONG_OP.search(content[win_start:win_end])
        edge_type: EdgeType = "strong" if op_m else "mention"
        operator = op_m.group(0).lower() if op_m else ""

        if target not in best:
            order.append(target)
            best[target] = RoadmapEdge(target_key=target, edge_type=edge_type, operator=operator)
        elif edge_type == "strong" and best[target].edge_type == "mention":
            best[target] = RoadmapEdge(target_key=target, edge_type=edge_type, operator=operator)

    return [best[k] for k in order]


def _infer_status(content: str) -> str:
    if _LANDED_RE.search(content):
        return "landed"
    if _DEFERRED_RE.search(content):
        return "deferred"
    return "in-flight"


def _mark_superseded(items_with_content: list[tuple[RoadmapItem, str]]) -> None:
    """Mark older items superseded when a newer entry has a supersed* strong edge to them."""
    key_to_item = {item.key: item for item, _ in items_with_content}
    for item, _ in items_with_content:
        for edge in item.outbound_edges:
            if edge.edge_type != "strong" or not edge.operator.startswith("supersed"):
                continue
            target = key_to_item.get(edge.target_key)
            if not target:
                continue
            if item.updated_at >= target.updated_at and target.status not in ("landed",):
                target.status = "superseded"


def _item_to_dict(item: RoadmapItem) -> dict:
    return {
        "key": item.key,
        "namespace": item.namespace,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "tags": item.tags,
        "summary": item.summary,
        "status": item.status,
        "outbound_edges": [
            {
                "edge_type": e.edge_type,
                "operator": e.operator,
                "target_key": e.target_key,
            }
            for e in item.outbound_edges
        ],
    }


def _dict_to_item(d: dict) -> RoadmapItem:
    return RoadmapItem(
        key=d["key"],
        namespace=d["namespace"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        tags=d.get("tags", ""),
        summary=d.get("summary", ""),
        outbound_edges=[
            RoadmapEdge(
                target_key=e["target_key"],
                edge_type=e["edge_type"],
                operator=e.get("operator", ""),
            )
            for e in d.get("outbound_edges", [])
        ],
        status=d.get("status", "unknown"),
    )


def _load_snapshot_file(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("roadmap: failed to load snapshot from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# 2. Decision-lineage walker
# ---------------------------------------------------------------------------

def walk_lineage(
    seed_keys: list[str] | str,
    *,
    max_hops: int = 2,
    snapshot: dict | None = None,
    db_path: Path | str | None = None,
) -> list[LineageNode]:
    """Expand seed keys along key-mention edges within roadmap namespaces.

    Returns nodes ordered oldest→newest (build-order). Bounded by _MAX_WALK_NODES.
    Never raises.
    """
    try:
        return _walk(seed_keys, max_hops=max_hops, snapshot=snapshot, db_path=db_path)
    except Exception as exc:
        log.error("roadmap: walk_lineage failed: %s", exc)
        return []


def _walk(
    seed_keys: list[str] | str,
    *,
    max_hops: int,
    snapshot: dict | None,
    db_path: Path | str | None,
) -> list[LineageNode]:
    if isinstance(seed_keys, str):
        seed_keys = [seed_keys]
    if not seed_keys:
        return []

    item_map = _build_item_map(snapshot=snapshot, db_path=db_path)

    visited: dict[str, LineageNode] = {}
    queue: list[tuple[str, int, list[RoadmapEdge]]] = [
        (k, 0, []) for k in seed_keys
    ]

    while queue and len(visited) < _MAX_WALK_NODES:
        key, hop, edges_in = queue.pop(0)
        if key in visited:
            continue
        item = item_map.get(key)
        if not item:
            continue

        visited[key] = LineageNode(
            key=key,
            namespace=item.namespace,
            created_at=item.created_at,
            updated_at=item.updated_at,
            summary=item.summary,
            status=item.status,
            edges_in=edges_in,
            hop=hop,
        )

        if hop < max_hops:
            for edge in item.outbound_edges:
                tgt = edge.target_key
                if tgt not in visited and tgt in item_map:
                    queue.append((tgt, hop + 1, [edge]))

    nodes = list(visited.values())
    nodes.sort(key=lambda n: (n.created_at or "", n.key))
    return nodes


def _build_item_map(
    snapshot: dict | None,
    db_path: Path | str | None,
) -> dict[str, RoadmapItem]:
    if snapshot and "items" in snapshot:
        return {d["key"]: _dict_to_item(d) for d in snapshot["items"]}

    dp = Path(db_path) if db_path else None
    store = MemoryStore(dp) if dp else MemoryStore()
    try:
        raw_rows = _load_roadmap_rows(store)
    finally:
        store.close()

    items_with_content = [(_build_item(r), r.get("content", "")) for r in raw_rows]
    _mark_superseded(items_with_content)
    return {item.key: item for item, _ in items_with_content}


# ---------------------------------------------------------------------------
# 3. Roadmap grounding entry point
# ---------------------------------------------------------------------------

def ground_roadmap(
    query: str,
    *,
    node_anchor: str | None = None,
    token_budget: int = 3000,
    snapshot_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> GroundBundle:
    """Assemble a GroundBundle for roadmap-mirror injection into a weaver turn.

    Returns GroundBundle with same shape as ground() so weaver's existing injection
    path consumes it unchanged. Never raises; degrades to a stale/thin bundle on error.
    """
    try:
        return _ground(
            query,
            node_anchor=node_anchor,
            token_budget=token_budget,
            snapshot_path=snapshot_path,
            db_path=db_path,
        )
    except Exception as exc:
        log.error("roadmap: ground_roadmap failed: %s", exc)
        return GroundBundle(
            context_block="",
            provenance=[{
                "tag": "ungrounded",
                "source": "roadmap",
                "why": f"error:{type(exc).__name__}",
            }],
            truncated=False,
            stale=True,
            token_estimate=0,
        )


def _ground(
    query: str,
    *,
    node_anchor: str | None,
    token_budget: int,
    snapshot_path: Path | str | None,
    db_path: Path | str | None,
) -> GroundBundle:
    char_budget = token_budget * 4
    provenance: list[dict] = []
    stale = False

    # Load materialized snapshot
    snap_path = Path(snapshot_path) if snapshot_path else _DEFAULT_SNAPSHOT_PATH
    snapshot = _load_snapshot_file(snap_path)
    if snapshot is None:
        stale = True
        provenance.append({
            "tag": "ungrounded",
            "source": "snapshot",
            "why": "missing-or-failed",
        })

    # (a) Portrait slice — reserve up to 40% of budget
    portrait_budget = char_budget * 2 // 5
    portrait_block, portrait_prov = _portrait_slice(snapshot, query, char_budget=portrait_budget)
    provenance.extend(portrait_prov)

    lineage_budget = max(0, char_budget - len(portrait_block))

    # (b) FTS over-recall for seeds
    seed_keys = _fts_seeds(query, node_anchor=node_anchor, db_path=db_path)

    # Echo-chamber guard: split confirmed_keys vs candidate_keys
    if snapshot:
        snapshot_key_set = {item["key"] for item in snapshot.get("items", [])}
        confirmed_keys = [k for k in seed_keys if k in snapshot_key_set]
        candidate_keys = [k for k in seed_keys if k not in snapshot_key_set]
        provenance.append({
            "tag": "already-have",
            "confirmed_keys": confirmed_keys,
            "candidate_keys": candidate_keys,
            "source": "snapshot",
        })

    # (c) Lineage walk from top seeds
    nodes = (
        walk_lineage(seed_keys[:5], max_hops=2, snapshot=snapshot, db_path=db_path)
        if seed_keys else []
    )

    # Thin-recall detection: flag when no strong edges were traversed
    has_strong = any(
        any(e.edge_type == "strong" for e in n.edges_in)
        for n in nodes
    )
    if not has_strong:
        provenance.append({
            "tag": "recall:fts-only",
            "source": "roadmap",
            "why": "no-strong-edges-traversed",
        })

    # (d) Compose context_block = portrait + focused lineage
    lineage_block, lineage_prov = _lineage_block(nodes, char_budget=lineage_budget)
    provenance.extend(lineage_prov)

    parts = [b for b in (portrait_block, lineage_block) if b]
    context_block = "\n\n".join(parts)

    return GroundBundle(
        context_block=context_block,
        provenance=provenance,
        truncated=len(context_block) >= char_budget,
        stale=stale,
        token_estimate=len(context_block) // 4,
    )


def _fts_seeds(
    query: str,
    *,
    node_anchor: str | None,
    db_path: Path | str | None,
) -> list[str]:
    """FTS over-recall within roadmap namespaces; node_anchor seeds retrieval."""
    try:
        q = f"{query} {node_anchor}" if node_anchor else query
        hits = retrieve(q, scope=["mem"], top_k=20, min_score=0.0)
        keys: list[str] = []
        for hit in hits:
            key = hit.metadata.get("key", "")
            if not key and hit.id.startswith("mem:"):
                key = hit.id[4:]
            ns = key.split("/")[0] if "/" in key else ""
            if ns in _ROADMAP_NAMESPACES:
                keys.append(key)
        # Prepend anchor if not already surfaced
        if node_anchor:
            anchor_ns = node_anchor.split("/")[0] if "/" in node_anchor else ""
            if anchor_ns in _ROADMAP_NAMESPACES and node_anchor not in keys:
                keys.insert(0, node_anchor)
        return keys
    except Exception as exc:
        log.warning("roadmap: FTS seed recall failed: %s", exc)
        return [node_anchor] if node_anchor else []


def _portrait_slice(
    snapshot: dict | None,
    query: str,
    *,
    char_budget: int,
) -> tuple[str, list[dict]]:
    if not snapshot:
        return "", []
    items = snapshot.get("items", [])
    if not items:
        return "", []

    query_terms = set(re.sub(r"[^\w]", " ", query.lower()).split())

    scored: list[tuple[int, dict]] = []
    for item in items:
        text = f"{item['key']} {item.get('summary', '')}".lower()
        score = sum(1 for t in query_terms if len(t) > 2 and t in text)
        scored.append((score, item))
    scored.sort(key=lambda x: (-x[0], x[1]["key"]))

    lines: list[str] = []
    used = 0
    prov: list[dict] = []

    for score, item in scored:
        tag = f"roadmap:{item['key']}"
        strong = [e for e in item.get("outbound_edges", []) if e.get("edge_type") == "strong"]
        dep = ""
        if strong:
            dep = " | depends: " + ", ".join(e["target_key"] for e in strong[:3])
        line = f"[{tag}] ({item.get('status', 'unknown')}) {item.get('summary', '')}{dep}"
        cost = len(line) + 1
        if used + cost > char_budget:
            break
        lines.append(line)
        used += cost
        prov.append({
            "tag": tag,
            "source": "roadmap-snapshot",
            "score": score,
            "why": "portrait",
        })

    if not lines:
        return "", prov

    return "[ROADMAP-PORTRAIT]\n" + "\n".join(lines), prov


def _lineage_block(
    nodes: list[LineageNode],
    *,
    char_budget: int,
) -> tuple[str, list[dict]]:
    if not nodes:
        return "", []

    lines: list[str] = []
    used = 0
    prov: list[dict] = []

    for node in nodes:
        tag = f"lineage:{node.key}"
        edge_label = ""
        if node.edges_in:
            e = node.edges_in[0]
            edge_label = f" [{e.edge_type}]"
        line = f"[{tag}] (hop={node.hop},{node.status}){edge_label} {node.summary}"
        cost = len(line) + 1
        if used + cost > char_budget:
            break
        lines.append(line)
        used += cost
        prov.append({
            "tag": tag,
            "source": "roadmap-lineage",
            "score": None,
            "why": "lineage-walk",
        })

    if not lines:
        return "", prov

    return "[ROADMAP-LINEAGE]\n" + "\n".join(lines), prov


# ---------------------------------------------------------------------------
# CLI entry (spec-authorized; snapshot refresh)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Refresh the committed-plan roadmap snapshot from mem.db"
    )
    parser.add_argument("--snapshot-path", help="Override snapshot output path")
    parser.add_argument("--db-path", help="Override mem.db path")
    args = parser.parse_args()

    result = materialize_committed_plan(
        snapshot_path=args.snapshot_path or None,
        db_path=args.db_path or None,
    )
    if result is None:
        print("ERROR: snapshot materialization failed", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK: {result['item_count']} items, hash={result['content_hash'][:12]}, "
        f"at={result['materialized_at']}"
    )


if __name__ == "__main__":
    main()
