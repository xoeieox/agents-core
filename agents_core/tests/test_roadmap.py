"""Tests for agents_core.roadmap — roadmap-mirror-lineage-retriever-v0 acceptance gate.

Acceptance criteria per spec:
  1. materialize_committed_plan(): namespace filter; items carry fields; supersed* marks older
     item superseded; artifact byte-identical on second run; hash changes on ledger change
  2. walk_lineage(): edge traversal to N hops; hop+node caps; timestamp-ordered;
     strong vs mention never conflated; seed with no edges returns itself
  3. Echo-chamber guard: keyword candidate ≠ confirmed; canonical-key match IS confirmable
  4. Thin-recall flag: fts-only result → recall:fts-only tag; strong-edge result → no tag
  5. ground_roadmap(): returns GroundBundle with portrait+lineage; node_anchor influences seeds;
     never raises on missing snapshot or mem error; respects token_budget
  6. ROADMAP_MIRROR_PREAMBLE: contains propose-only, ALREADY-HAVE, COMPOST, marker shape
  7. No paid-model import; no write path to mem.db (AST scans)
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.ground import GroundBundle
from agents_core.retrieval import Hit
from agents_core.roadmap import (
    ROADMAP_MIRROR_PREAMBLE,
    LineageNode,
    RoadmapEdge,
    _extract_edges,
    _infer_status,
    ground_roadmap,
    materialize_committed_plan,
    walk_lineage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem_db(tmp_path: Path, entries: list[dict]) -> Path:
    db = tmp_path / "mem.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            key TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            source TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            key, content, tags,
            content='memories', content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, key, content, tags)
            VALUES (new.rowid, new.key, new.content, new.tags);
        END;
    """)
    for e in entries:
        conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(key, content, tags, source, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (
                e["key"], e["content"], e.get("tags", ""), "test",
                e.get("created_at", "2026-01-01T00:00:00+00:00"),
                e.get("updated_at", "2026-06-01T00:00:00+00:00"),
            ),
        )
    conn.commit()
    conn.close()
    return db


def _make_hit(key: str, score: float = 0.8) -> Hit:
    return Hit(
        id=f"mem:{key}",
        score=score,
        source="mem",
        content=f"content for {key}",
        metadata={"key": key},
    )


# ---------------------------------------------------------------------------
# 1a. Namespace filter drops router/weather entries; included items carry fields
# ---------------------------------------------------------------------------

def test_materialize_namespace_filter(tmp_path):
    db = _make_mem_db(tmp_path, [
        {"key": "decision/foo-v0", "content": "We decided to build foo."},
        {"key": "project/bar-v0", "content": "Bar project planning."},
        {"key": "router/lapis-pm/decisions/evt-1", "content": "Router noise."},
        {"key": "weather/today", "content": "Sunny."},
    ])
    snap_path = tmp_path / "snap.json"
    result = materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    assert result is not None
    keys = [i["key"] for i in result["items"]]
    assert "decision/foo-v0" in keys
    assert "project/bar-v0" in keys
    assert "router/lapis-pm/decisions/evt-1" not in keys
    assert "weather/today" not in keys


def test_materialize_item_fields(tmp_path):
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/arch-v0",
            "content": "Architecture decision.\n\nMore details here.",
            "tags": "lapis,Erah",
            "created_at": "2026-01-10T00:00:00+00:00",
            "updated_at": "2026-06-10T00:00:00+00:00",
        },
    ])
    snap_path = tmp_path / "snap.json"
    result = materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    assert result is not None
    item = result["items"][0]
    assert item["key"] == "decision/arch-v0"
    assert item["namespace"] == "decision"
    assert item["created_at"] == "2026-01-10T00:00:00+00:00"
    assert item["updated_at"] == "2026-06-10T00:00:00+00:00"
    assert set(item["tags"].split(",")) == {"lapis", "Erah"}
    assert "Architecture decision" in item["summary"]
    assert isinstance(item["outbound_edges"], list)
    assert item["status"] in ("in-flight", "deferred", "landed", "superseded", "unknown")


# ---------------------------------------------------------------------------
# 1b. supersed* prose link marks older item superseded
# ---------------------------------------------------------------------------

def test_materialize_superseded_status(tmp_path):
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/old-v0",
            "content": "The old approach.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "key": "decision/new-v1",
            "content": "The new approach supersedes decision/old-v0 completely.",
            "created_at": "2026-06-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
        },
    ])
    snap_path = tmp_path / "snap.json"
    result = materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    assert result is not None
    by_key = {i["key"]: i for i in result["items"]}
    assert by_key["decision/old-v0"]["status"] == "superseded"
    assert by_key["decision/new-v1"]["status"] == "in-flight"


# ---------------------------------------------------------------------------
# 1c. Idempotency: same DB → same content_hash; changed entry → different hash
# ---------------------------------------------------------------------------

def test_materialize_idempotent(tmp_path):
    db = _make_mem_db(tmp_path, [
        {"key": "decision/foo-v0", "content": "Foo decision."},
    ])
    snap_path = tmp_path / "snap.json"

    r1 = materialize_committed_plan(snapshot_path=snap_path, db_path=db)
    assert r1 is not None
    snap_bytes_1 = snap_path.read_bytes()

    r2 = materialize_committed_plan(snapshot_path=snap_path, db_path=db)
    assert r2 is not None
    snap_bytes_2 = snap_path.read_bytes()

    assert r1["content_hash"] == r2["content_hash"]
    assert snap_bytes_1 == snap_bytes_2, "file must be byte-identical on second run"


def test_materialize_hash_changes_on_ledger_change(tmp_path):
    db = _make_mem_db(tmp_path, [
        {"key": "decision/foo-v0", "content": "Foo v1."},
    ])
    snap_path = tmp_path / "snap.json"

    r1 = materialize_committed_plan(snapshot_path=snap_path, db_path=db)
    assert r1 is not None
    hash1 = r1["content_hash"]

    # Modify the ledger entry
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET content=?, updated_at=? WHERE key=?",
        ("Foo v2 — updated content.", "2026-06-25T00:00:00+00:00", "decision/foo-v0"),
    )
    conn.commit()
    conn.close()

    r2 = materialize_committed_plan(snapshot_path=snap_path, db_path=db)
    assert r2 is not None
    assert r2["content_hash"] != hash1, "hash must change when ledger entry changes"


# ---------------------------------------------------------------------------
# 2a. walk_lineage: edge traversal, hop limit, timestamp order
# ---------------------------------------------------------------------------

def test_walk_lineage_follows_edges(tmp_path):
    """Walk from A reaches B (mentioned by A) and C (mentioned by B) at max_hops=2."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/a-v0",
            "content": "A decision. See decision/b-v0 for next step.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "key": "decision/b-v0",
            "content": "B decision. Depends on decision/c-v0.",
            "created_at": "2026-02-01T00:00:00+00:00",
            "updated_at": "2026-02-01T00:00:00+00:00",
        },
        {
            "key": "decision/c-v0",
            "content": "C decision, foundational.",
            "created_at": "2026-03-01T00:00:00+00:00",
            "updated_at": "2026-03-01T00:00:00+00:00",
        },
    ])
    nodes = walk_lineage("decision/a-v0", max_hops=2, db_path=db)
    keys = [n.key for n in nodes]
    assert "decision/a-v0" in keys
    assert "decision/b-v0" in keys
    assert "decision/c-v0" in keys


def test_walk_lineage_respects_hop_cap(tmp_path):
    """Nodes beyond max_hops are not reached."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/a-v0",
            "content": "See decision/b-v0.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "key": "decision/b-v0",
            "content": "See decision/c-v0.",
            "created_at": "2026-02-01T00:00:00+00:00",
            "updated_at": "2026-02-01T00:00:00+00:00",
        },
        {
            "key": "decision/c-v0",
            "content": "See decision/d-v0.",
            "created_at": "2026-03-01T00:00:00+00:00",
            "updated_at": "2026-03-01T00:00:00+00:00",
        },
        {
            "key": "decision/d-v0",
            "content": "Deep node.",
            "created_at": "2026-04-01T00:00:00+00:00",
            "updated_at": "2026-04-01T00:00:00+00:00",
        },
    ])
    nodes = walk_lineage("decision/a-v0", max_hops=2, db_path=db)
    keys = [n.key for n in nodes]
    assert "decision/a-v0" in keys
    assert "decision/b-v0" in keys
    assert "decision/c-v0" in keys
    assert "decision/d-v0" not in keys  # 3 hops away from A


def test_walk_lineage_respects_node_cap(tmp_path):
    """Node cap (_MAX_WALK_NODES=40) is enforced."""
    # Create 50 entries all chained from decision/root-v0 via mention
    entries = [{"key": "decision/root-v0", "content": " ".join(
        f"decision/node-{i:02d}" for i in range(50)
    ), "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"}]
    for i in range(50):
        entries.append({
            "key": f"decision/node-{i:02d}",
            "content": f"Node {i}.",
            "created_at": "2026-01-02T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
        })
    db = _make_mem_db(tmp_path, entries)
    nodes = walk_lineage("decision/root-v0", max_hops=2, db_path=db)
    assert len(nodes) <= 40


def test_walk_lineage_timestamp_order(tmp_path):
    """Nodes are returned oldest→newest (by created_at)."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/z-v0",
            "content": "Z decision. See decision/a-v0.",
            "created_at": "2026-06-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
        },
        {
            "key": "decision/a-v0",
            "content": "A decision, foundational.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ])
    nodes = walk_lineage("decision/z-v0", max_hops=2, db_path=db)
    keys = [n.key for n in nodes]
    assert keys.index("decision/a-v0") < keys.index("decision/z-v0")


def test_walk_lineage_seed_no_edges(tmp_path):
    """Seed with no roadmap-namespace outbound edges returns just itself."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/lonely-v0",
            "content": "Stand-alone decision with no key mentions.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ])
    nodes = walk_lineage("decision/lonely-v0", max_hops=2, db_path=db)
    assert len(nodes) == 1
    assert nodes[0].key == "decision/lonely-v0"


# ---------------------------------------------------------------------------
# 2b. Edge type classification: strong vs mention never conflated
# ---------------------------------------------------------------------------

def test_edge_type_strong_for_supersed():
    content = "This supersedes decision/old-v0 and replaces it."
    edges = _extract_edges(content)
    by_key = {e.target_key: e for e in edges}
    assert "decision/old-v0" in by_key
    assert by_key["decision/old-v0"].edge_type == "strong"
    assert by_key["decision/old-v0"].operator.startswith("supersed")


def test_edge_type_strong_for_depends():
    content = "This work depends on decision/infra-v0 to proceed."
    edges = _extract_edges(content)
    by_key = {e.target_key: e for e in edges}
    assert "decision/infra-v0" in by_key
    assert by_key["decision/infra-v0"].edge_type == "strong"
    assert "depends" in by_key["decision/infra-v0"].operator


def test_edge_type_mention_for_bare_reference():
    content = "See also decision/background-v0 for context."
    edges = _extract_edges(content)
    by_key = {e.target_key: e for e in edges}
    assert "decision/background-v0" in by_key
    assert by_key["decision/background-v0"].edge_type == "mention"
    assert by_key["decision/background-v0"].operator == ""


def test_edge_type_never_conflated():
    """Strong edge and mention edge in the same content are correctly distinguished.

    The operator must be near the key (within ~60 chars) to classify it as strong.
    A bare mention far from any operator is classified as mention.
    """
    # "supersedes" is within 60 chars of decision/old-v0 → strong
    # decision/context-v0 appears >120 chars away from any operator → mention
    strong_part = "This decision supersedes decision/old-v0 completely."
    gap = " " * 200  # ensures decision/context-v0 is far from any operator
    mention_part = "See also decision/context-v0 for additional background context."
    content = strong_part + gap + mention_part

    edges = _extract_edges(content)
    by_key = {e.target_key: e for e in edges}

    assert by_key["decision/old-v0"].edge_type == "strong"
    assert by_key["decision/context-v0"].edge_type == "mention"
    # The two must never be confused
    assert by_key["decision/old-v0"].edge_type != by_key["decision/context-v0"].edge_type


def test_walk_lineage_edge_types_in_nodes(tmp_path):
    """Nodes reached via strong edge have edge_type=strong in edges_in; mention has mention."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/root-v0",
            "content": (
                "Root decision. This supersedes decision/old-v0."
                + " " * 200
                + "See also decision/ref-v0 for context."
            ),
            "created_at": "2026-06-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
        },
        {
            "key": "decision/old-v0",
            "content": "The old approach.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "key": "decision/ref-v0",
            "content": "Background reference.",
            "created_at": "2026-02-01T00:00:00+00:00",
            "updated_at": "2026-02-01T00:00:00+00:00",
        },
    ])
    nodes = walk_lineage("decision/root-v0", max_hops=1, db_path=db)
    by_key = {n.key: n for n in nodes}

    # decision/old-v0 reached via strong (supersed*) edge
    assert "decision/old-v0" in by_key
    old_edges = by_key["decision/old-v0"].edges_in
    assert len(old_edges) == 1
    assert old_edges[0].edge_type == "strong"

    # decision/ref-v0 reached via mention edge
    assert "decision/ref-v0" in by_key
    ref_edges = by_key["decision/ref-v0"].edges_in
    assert len(ref_edges) == 1
    assert ref_edges[0].edge_type == "mention"

    # The two must not be equal
    assert old_edges[0].edge_type != ref_edges[0].edge_type


# ---------------------------------------------------------------------------
# 3. Echo-chamber guard: confirmed vs candidate keys
# ---------------------------------------------------------------------------

def test_echo_chamber_guard_confirmed_vs_candidate(tmp_path):
    """Keys present in snapshot → confirmed_keys; absent keys → candidate_keys."""
    db = _make_mem_db(tmp_path, [
        {"key": "decision/real-v0", "content": "Real decision in the snapshot."},
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    # FTS recall returns both a real key and a "false positive" key not in snapshot
    fake_hits = [
        _make_hit("decision/real-v0", 0.9),
        _make_hit("decision/phantom-v0", 0.7),
    ]

    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap(
            "some query",
            snapshot_path=snap_path,
            db_path=db,
        )

    already_have = next(
        (p for p in bundle.provenance if p.get("tag") == "already-have"),
        None,
    )
    assert already_have is not None, "already-have provenance must be present"
    assert "decision/real-v0" in already_have["confirmed_keys"]
    assert "decision/phantom-v0" in already_have["candidate_keys"]
    # Phantom must NOT be in confirmed
    assert "decision/phantom-v0" not in already_have["confirmed_keys"]


# ---------------------------------------------------------------------------
# 4. Thin-recall flag
# ---------------------------------------------------------------------------

def test_thin_recall_flag_when_fts_only(tmp_path):
    """When no strong edges are traversed, recall:fts-only provenance is present."""
    db = _make_mem_db(tmp_path, [
        {"key": "decision/a-v0", "content": "A decision with no strong key mentions."},
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    fake_hits = [_make_hit("decision/a-v0", 0.9)]

    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap("some query", snapshot_path=snap_path, db_path=db)

    thin_tags = [p for p in bundle.provenance if p.get("tag") == "recall:fts-only"]
    assert len(thin_tags) >= 1, "recall:fts-only must be in provenance when no strong edges"


def test_no_thin_recall_flag_when_strong_edges(tmp_path):
    """When strong edges are traversed, recall:fts-only is NOT in provenance."""
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/new-v1",
            "content": "New decision. This supersedes decision/old-v0.",
            "created_at": "2026-06-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
        },
        {
            "key": "decision/old-v0",
            "content": "Old decision.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    # Seed retrieval returns decision/new-v1 which has a strong edge to decision/old-v0
    fake_hits = [_make_hit("decision/new-v1", 0.9)]

    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap("some query", snapshot_path=snap_path, db_path=db)

    thin_tags = [p for p in bundle.provenance if p.get("tag") == "recall:fts-only"]
    assert len(thin_tags) == 0, "recall:fts-only must NOT be present when strong edges found"


# ---------------------------------------------------------------------------
# 5a. ground_roadmap returns GroundBundle with portrait + lineage
# ---------------------------------------------------------------------------

def test_ground_roadmap_returns_bundle(tmp_path):
    db = _make_mem_db(tmp_path, [
        {"key": "decision/foo-v0", "content": "Foo decision for the roadmap."},
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    fake_hits = [_make_hit("decision/foo-v0", 0.9)]
    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap("foo", snapshot_path=snap_path, db_path=db)

    assert isinstance(bundle, GroundBundle)
    assert isinstance(bundle.context_block, str)
    assert isinstance(bundle.provenance, list)
    assert isinstance(bundle.truncated, bool)
    assert isinstance(bundle.stale, bool)
    assert isinstance(bundle.token_estimate, int)
    assert "ROADMAP-PORTRAIT" in bundle.context_block


def test_ground_roadmap_portrait_and_lineage_present(tmp_path):
    db = _make_mem_db(tmp_path, [
        {
            "key": "decision/a-v0",
            "content": "A. See also decision/b-v0.",
            "created_at": "2026-06-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
        },
        {
            "key": "decision/b-v0",
            "content": "B foundational.",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    fake_hits = [_make_hit("decision/a-v0", 0.9)]
    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap("a b decision", snapshot_path=snap_path, db_path=db)

    assert "ROADMAP-PORTRAIT" in bundle.context_block
    assert "ROADMAP-LINEAGE" in bundle.context_block
    lineage_prov = [p for p in bundle.provenance if p.get("source") == "roadmap-lineage"]
    assert len(lineage_prov) >= 1


# ---------------------------------------------------------------------------
# 5b. node_anchor influences recalled seeds
# ---------------------------------------------------------------------------

def test_ground_roadmap_node_anchor_prepended(tmp_path):
    """node_anchor is prepended to seed_keys when not already surfaced by FTS."""
    db = _make_mem_db(tmp_path, [
        {"key": "decision/anchor-v0", "content": "The anchored decision."},
        {"key": "decision/other-v0", "content": "Another decision."},
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    # FTS returns other-v0 but NOT anchor-v0
    fake_hits = [_make_hit("decision/other-v0", 0.9)]
    captured_seeds: list[list[str]] = []

    original_walk = walk_lineage

    def capturing_walk(seeds, **kwargs):
        captured_seeds.append(list(seeds) if isinstance(seeds, list) else [seeds])
        return original_walk(seeds, **kwargs)

    with patch("agents_core.roadmap.retrieve", return_value=fake_hits), \
         patch("agents_core.roadmap.walk_lineage", side_effect=capturing_walk):
        ground_roadmap(
            "some query",
            node_anchor="decision/anchor-v0",
            snapshot_path=snap_path,
            db_path=db,
        )

    assert len(captured_seeds) >= 1
    first_call_seeds = captured_seeds[0]
    assert first_call_seeds[0] == "decision/anchor-v0", \
        "node_anchor must be first in seeds when not already surfaced"


# ---------------------------------------------------------------------------
# 5c. Never raises on missing snapshot or mem error
# ---------------------------------------------------------------------------

def test_ground_roadmap_missing_snapshot_does_not_raise(tmp_path):
    """Missing snapshot_path → stale=True bundle, no exception."""
    snap_path = tmp_path / "nonexistent.json"
    with patch("agents_core.roadmap.retrieve", return_value=[]):
        bundle = ground_roadmap("query", snapshot_path=snap_path)

    assert isinstance(bundle, GroundBundle)
    assert bundle.stale is True
    ungrounded = [p for p in bundle.provenance if p.get("tag") == "ungrounded"]
    assert len(ungrounded) >= 1


def test_ground_roadmap_retrieve_raises_does_not_raise(tmp_path):
    """retrieve() raising → no exception from ground_roadmap; bundle is returned."""
    snap_path = tmp_path / "nonexistent.json"
    with patch("agents_core.roadmap.retrieve", side_effect=ConnectionError("db down")):
        bundle = ground_roadmap("query", snapshot_path=snap_path)

    assert isinstance(bundle, GroundBundle)


def test_ground_roadmap_unexpected_error_does_not_raise(tmp_path):
    """Any unexpected error in the inner pipeline → graceful GroundBundle with stale=True."""
    with patch("agents_core.roadmap._ground", side_effect=RuntimeError("boom")):
        bundle = ground_roadmap("query")

    assert isinstance(bundle, GroundBundle)
    assert bundle.stale is True


# ---------------------------------------------------------------------------
# 5d. Respects token_budget
# ---------------------------------------------------------------------------

def test_ground_roadmap_respects_token_budget(tmp_path):
    db = _make_mem_db(tmp_path, [
        {"key": "decision/big-v0", "content": "x" * 2000},
        {"key": "decision/big-v1", "content": "y" * 2000},
    ])
    snap_path = tmp_path / "snap.json"
    materialize_committed_plan(snapshot_path=snap_path, db_path=db)

    fake_hits = [_make_hit("decision/big-v0", 0.9), _make_hit("decision/big-v1", 0.8)]
    with patch("agents_core.roadmap.retrieve", return_value=fake_hits):
        bundle = ground_roadmap("big", token_budget=200, snapshot_path=snap_path, db_path=db)

    assert bundle.token_estimate <= 200


# ---------------------------------------------------------------------------
# 6. ROADMAP_MIRROR_PREAMBLE content
# ---------------------------------------------------------------------------

def test_preamble_propose_only():
    assert "propose-only" in ROADMAP_MIRROR_PREAMBLE.lower()


def test_preamble_already_have():
    assert "ALREADY-HAVE" in ROADMAP_MIRROR_PREAMBLE


def test_preamble_compost():
    assert "COMPOST" in ROADMAP_MIRROR_PREAMBLE


def test_preamble_compost_marker_shape():
    assert "<!-- LAPIS-COMPOST:" in ROADMAP_MIRROR_PREAMBLE


def test_preamble_echo_chamber_guard():
    assert "canonical" in ROADMAP_MIRROR_PREAMBLE.lower()


# ---------------------------------------------------------------------------
# 7a. No paid-model import (AST scan)
# ---------------------------------------------------------------------------

def test_no_paid_model_import():
    """roadmap.py must not import any paid-model client at module level."""
    import agents_core.roadmap as rmod
    src = Path(rmod.__file__).read_text()
    tree = ast.parse(src)

    forbidden_prefixes = ("anthropic", "openai", "litellm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(alias.name.startswith(p) for p in forbidden_prefixes), \
                    f"roadmap.py imports forbidden module: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(mod.startswith(p) for p in forbidden_prefixes), \
                f"roadmap.py imports from forbidden module: {mod}"


# ---------------------------------------------------------------------------
# 7b. No write path to mem.db (AST scan)
# ---------------------------------------------------------------------------

def test_no_mem_write_calls():
    """roadmap.py must not call MemoryStore.set() or MemoryStore.delete()."""
    import agents_core.roadmap as rmod
    src = Path(rmod.__file__).read_text()
    tree = ast.parse(src)

    forbidden_methods = {"set", "delete"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_methods or \
                    not isinstance(node.func.value, ast.Name) or \
                    node.func.value.id not in ("store", "mem", "_store", "_mem"), \
                    f"roadmap.py calls write method: {node.func.attr}"


# ---------------------------------------------------------------------------
# Extra: status inference
# ---------------------------------------------------------------------------

def test_status_landed():
    assert _infer_status("This feature is LANDED as of 2026-06-01.") == "landed"


def test_status_deferred():
    assert _infer_status("This work is deferred to next quarter.") == "deferred"


def test_status_inflight():
    assert _infer_status("Work in progress, shipping soon.") == "in-flight"


# ---------------------------------------------------------------------------
# Extra: walk_lineage never raises
# ---------------------------------------------------------------------------

def test_walk_lineage_never_raises_on_bad_input():
    result = walk_lineage([], db_path="/nonexistent/path.db")
    assert result == []


def test_walk_lineage_string_seed(tmp_path):
    """walk_lineage accepts a bare string seed."""
    db = _make_mem_db(tmp_path, [
        {"key": "decision/str-seed-v0", "content": "A decision."},
    ])
    nodes = walk_lineage("decision/str-seed-v0", db_path=db)
    assert any(n.key == "decision/str-seed-v0" for n in nodes)


# ---------------------------------------------------------------------------
# DoD-6: roadmap._LAPIS_STATE default is /srv/lapis/lapis-state (anomaly fix)
# ---------------------------------------------------------------------------

def test_lapis_state_default_is_room():
    """roadmap._LAPIS_STATE is Path('/srv/lapis/lapis-state') when LAPIS_STATE is unset.

    Covers the anomaly fix: old default was Path(os.environ.get('LAPIS_STATE', '/data/lapis-state'))
    which silently pointed at the wrong /data location whenever the env var was absent.
    The fix canonicalises the default to /srv/lapis/lapis-state via room_path('lapis_state')
    while preserving LAPIS_STATE env-override precedence.

    Runs in a subprocess so that LAPIS_STATE is guaranteed absent at module import time
    (module-level constants bind once at first import).
    """
    code = (
        "import os; "
        "os.environ.pop('LAPIS_STATE', None); "
        "os.environ.pop('ROADMAP_SNAPSHOT_PATH', None); "
        "import agents_core.roadmap as rmod; "
        "from pathlib import Path; "
        "got = rmod._LAPIS_STATE; "
        "assert got == Path('/srv/lapis/lapis-state'), "
        "    f'_LAPIS_STATE={got!r}, expected Path(\"/srv/lapis/lapis-state\")'; "
        "print('OK')"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("LAPIS_STATE", "ROADMAP_SNAPSHOT_PATH")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, (
        f"roadmap._LAPIS_STATE check failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout
