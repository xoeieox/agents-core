"""Tests for agents_core.ground — ground-call-substrate-v0 acceptance gate.

Acceptance criteria per spec:
  1. ground() returns GroundBundle; context_block non-empty; token_estimate <= token_budget
  2. Oversize input sets truncated=True
  3. Every cited item in context_block has a matching provenance entry with a well-formed
     citation tag (mem:/vault:/chub:/pm: prefix)
  4. pm_state=True surfaces >=1 active target via MemoryStore/TargetStore; no lapis_pm import
  5. Degradation: retrieve() patched to raise -> non-empty bundle, no raise, ungrounded marker
  6. Budget reservation: flood of high-score RAG hits still leaves PM-state when pm_state=True
  7. ground.py does NOT import lapis_pm (AST scan)
"""

from __future__ import annotations

import ast
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.ground import GroundBundle, ground, _citation_tag
from agents_core.retrieval import Hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hit(source: str, key: str, content: str, score: float = 0.8) -> Hit:
    hit_id = f"{source}:{key}"
    return Hit(id=hit_id, score=score, source=source, content=content)


def _make_mem_db(tmp_path: Path, entries: list[dict]) -> Path:
    """Create a minimal mem.db with given key/content/tags entries."""
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
            "INSERT OR REPLACE INTO memories (key, content, tags, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (e["key"], e["content"], e.get("tags", ""), "test",
             "2026-01-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()
    return db


def _make_target_yaml(tmp_path: Path, target_id: str, title: str,
                      status: str = "active", pm_bound: bool = True,
                      urgency: str = "high") -> Path:
    d = tmp_path / "targets"
    d.mkdir(exist_ok=True)
    p = d / f"{target_id}.yaml"
    data = {
        "id": target_id,
        "title": title,
        "status": status,
        "urgency": urgency,
        "work_mode": "anywhere",
        "created": "2026-06-01",
        "touched": "2026-06-20",
        "updated": "2026-06-20",
        "decay_days": 0,
        "decay_threshold": 7,
        "pm_bound": pm_bound,
        "pm_repo": "agents-core",
        "pm_authority": "advisory",
    }
    p.write_text(yaml.dump(data))
    return d


# ---------------------------------------------------------------------------
# 1. ground() returns GroundBundle; token_estimate <= token_budget
# ---------------------------------------------------------------------------

def test_ground_returns_bundle_within_budget():
    """ground() returns a GroundBundle with token_estimate <= token_budget."""
    fake_hits = [_make_hit("mem", "decision/foo", "decision about foo", 0.9)]

    with patch("agents_core.ground.retrieve", return_value=fake_hits), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("what is the lapis PM loop?", pm_state=False, token_budget=500)

    assert isinstance(bundle, GroundBundle)
    assert bundle.token_estimate <= 500
    assert isinstance(bundle.context_block, str)
    assert isinstance(bundle.provenance, list)
    assert isinstance(bundle.truncated, bool)
    assert isinstance(bundle.stale, bool)


# ---------------------------------------------------------------------------
# 2. Oversize input sets truncated=True
# ---------------------------------------------------------------------------

def test_ground_truncated_when_hits_exceed_budget():
    """Many large hits with a small budget sets truncated=True."""
    big_hits = [
        _make_hit("mem", f"decision/item-{i}", "x" * 500, 0.9 - i * 0.01)
        for i in range(20)
    ]
    # budget of 100 tokens = 400 chars — far smaller than 20 * 500-char hits

    with patch("agents_core.ground.retrieve", return_value=big_hits), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("query", pm_state=False, token_budget=100)

    assert bundle.truncated is True


# ---------------------------------------------------------------------------
# 3. Citation tags well-formed and provenance matches context_block
# ---------------------------------------------------------------------------

def test_cited_items_have_matching_provenance():
    """Every [tag] in context_block has a corresponding provenance entry."""
    hits = [
        _make_hit("mem", "decision/arch-v0", "architecture decision", 0.95),
        _make_hit("vault-rag", "Lapis/Spec.md", "spec content", 0.8),
        _make_hit("chub", "conductor/lapis-ecosystem", "ecosystem doc", 0.7),
    ]

    with patch("agents_core.ground.retrieve", return_value=hits), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("lapis architecture", pm_state=False)

    import re
    cited_tags = re.findall(r"\[([^\]]+)\]", bundle.context_block)
    prov_tags = {p["tag"] for p in bundle.provenance}

    for tag in cited_tags:
        assert tag in prov_tags, f"Tag [{tag}] in context_block not found in provenance"
        assert any(tag.startswith(p) for p in ("mem:", "vault:", "chub:", "pm:", "code:")), \
            f"Tag [{tag}] has unrecognized prefix"


def test_citation_tag_prefixes():
    """_citation_tag maps source names to correct prefixes."""
    assert _citation_tag("mem", "mem:decision/foo").startswith("mem:")
    assert _citation_tag("vault-rag", "vault-rag:Lapis/Spec").startswith("vault:")
    assert _citation_tag("chub", "chub:conductor/lapis").startswith("chub:")
    assert _citation_tag("room-rag", "room-rag:some/doc").startswith("vault:")


# ---------------------------------------------------------------------------
# 4. pm_state=True surfaces >=1 active target; no lapis_pm import
# ---------------------------------------------------------------------------

def test_pm_state_surfaces_active_target(tmp_path):
    """pm_state=True returns >=1 active-target provenance; reads via MemoryStore/TargetStore."""
    targets_dir = _make_target_yaml(
        tmp_path, "my-target-v0", "My Target", pm_bound=True
    )
    db = _make_mem_db(tmp_path, [
        {
            "key": "router/lapis-pm/decisions/evt-001",
            "content": json.dumps({
                "freshness_stamp": "2026-06-20T10:00:00Z",
                "verdict": "proposed",
                "expert_chosen": "fixer",
                "intent_summary": "implement the ground() primitive",
                "target_id": "my-target-v0",
            }),
            "tags": "lapis-pm,router-portfolio,target:my-target-v0",
        },
    ])

    with patch("agents_core.ground.retrieve", return_value=[]), \
         patch("agents_core.mem.DB_PATH", db), \
         patch("agents_core.ground.MemoryStore") as mock_ms_cls, \
         patch("agents_core.ground.TargetStore") as mock_ts_cls:

        # Wire up fake MemoryStore
        fake_mem = MagicMock()
        mock_ms_cls.return_value = fake_mem
        fake_mem.list_all.return_value = [{
            "key": "router/lapis-pm/decisions/evt-001",
            "content": json.dumps({
                "freshness_stamp": "2026-06-20T10:00:00Z",
                "verdict": "proposed",
                "expert_chosen": "fixer",
                "intent_summary": "implement the ground() primitive",
                "target_id": "my-target-v0",
            }),
            "tags": "lapis-pm,router-portfolio,target:my-target-v0",
        }]
        fake_mem.get.return_value = None

        # Wire up fake TargetStore
        from agents_core.targets import Target
        fake_target = MagicMock(spec=Target)
        fake_target.id = "my-target-v0"
        fake_target.title = "My Target"
        fake_target.status = "active"
        fake_target.pm_bound = True
        fake_target.urgency = "high"
        mock_ts_cls.return_value.load_all.return_value = [fake_target]

        bundle = ground("PM loop query", pm_state=True, token_budget=3000)

    pm_prov = [p for p in bundle.provenance if p.get("tag", "").startswith("pm:")]
    assert len(pm_prov) >= 1, "pm_state=True should yield >=1 pm: provenance entry"
    assert pm_prov[0]["tag"] == "pm:my-target-v0"
    assert "pm:my-target-v0" in bundle.context_block


def test_no_lapis_pm_import():
    """ground.py does NOT import lapis_pm — verified by AST scan of source imports."""
    ground_src = Path(__file__).parent.parent / "ground.py"
    tree = ast.parse(ground_src.read_text())

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("lapis_pm"), \
                        f"ground.py imports lapis_pm: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not mod.startswith("lapis_pm"), \
                    f"ground.py imports from lapis_pm: {mod}"


# ---------------------------------------------------------------------------
# 5. Degradation: retrieve() raises -> no raise, ungrounded marker
# ---------------------------------------------------------------------------

def test_degradation_retrieve_raises():
    """retrieve() raising causes no raise from ground(); ungrounded provenance emitted."""
    with patch("agents_core.ground.retrieve", side_effect=ConnectionError("backend down")), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("query", pm_state=False)

    assert isinstance(bundle, GroundBundle)
    ungrounded = [p for p in bundle.provenance if p.get("tag") == "ungrounded"]
    assert len(ungrounded) >= 1


def test_degradation_pm_state_timeout():
    """PM-state timeout -> stale=True, ungrounded pm_state marker, no raise."""
    import time

    def _slow():
        time.sleep(10)  # far exceeds _PM_TIMEOUT_S

    with patch("agents_core.ground.retrieve", return_value=[]), \
         patch("agents_core.ground._do_pm_state_read", side_effect=_slow):
        # Temporarily lower the timeout guard to near-zero for test speed
        import agents_core.ground as gmod
        orig = gmod._PM_TIMEOUT_S
        gmod._PM_TIMEOUT_S = 0.05
        try:
            bundle = ground("query", pm_state=True)
        finally:
            gmod._PM_TIMEOUT_S = orig

    assert bundle.stale is True
    ungrounded = [p for p in bundle.provenance if p.get("tag") == "ungrounded"
                  and p.get("source") == "pm_state"]
    assert len(ungrounded) >= 1


def test_degradation_pm_state_raises():
    """PM-state raising -> stale=True, ungrounded marker, no raise from ground()."""
    with patch("agents_core.ground.retrieve", return_value=[]), \
         patch("agents_core.ground._do_pm_state_read", side_effect=RuntimeError("lock")):
        bundle = ground("query", pm_state=True)

    assert bundle.stale is True
    assert isinstance(bundle, GroundBundle)


# ---------------------------------------------------------------------------
# 6. Budget reservation: PM-state survives a flood of RAG hits
# ---------------------------------------------------------------------------

def test_budget_reservation_pm_state_not_starved():
    """A flood of high-score RAG hits cannot evict the PM-state block from the bundle."""
    flood = [
        _make_hit("mem", f"decision/item-{i}", "z" * 300, 1.0)
        for i in range(50)
    ]
    pm_text = "[pm:test-target]\ntarget:test-target (Test) [high]"

    with patch("agents_core.ground.retrieve", return_value=flood), \
         patch("agents_core.ground._assemble_pm_state",
               return_value=(pm_text, [{"tag": "pm:test-target", "source": "pm_state",
                                        "score": None, "why": "active-target"}], False)):
        bundle = ground("lapis", pm_state=True, token_budget=500)

    assert "pm:test-target" in bundle.context_block, \
        "PM-state block should be present even with flood of RAG hits"
    assert bundle.token_estimate <= 500
