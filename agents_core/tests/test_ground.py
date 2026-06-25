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
        mock_ts_cls.return_value.active_targets.return_value = [fake_target]

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


# ---------------------------------------------------------------------------
# 8. mem is retrieved UNFILTERED; decision AND project entries both appear
# ---------------------------------------------------------------------------

def test_mem_unfiltered_decision_and_project_both_appear():
    """retrieve() is called without a hard tag filter; decision AND project mem hits both appear."""
    decision_hit = _make_hit("mem", "decision/arch-v0", "architecture decision content", 0.85)
    project_hit = _make_hit("mem", "project/ground-call", "project planning content", 0.75)
    other_hit = _make_hit("mem", "notes/random", "random note", 0.60)
    all_hits = [decision_hit, project_hit, other_hit]

    captured_calls: list[dict] = []

    def fake_retrieve(query, scope, filters=None, top_k=10, min_score=0.0, **kwargs):
        captured_calls.append({"filters": filters})
        return all_hits

    with patch("agents_core.ground.retrieve", side_effect=fake_retrieve), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("architecture query", pm_state=False, token_budget=5000)

    # retrieve() must NOT have been called with a hard tag filter on mem
    for call in captured_calls:
        f = call.get("filters") or {}
        assert not f.get("tags"), \
            f"retrieve() called with hard tag filter tags={f.get('tags')} — must be unfiltered"

    # Both decision-namespace AND project-namespace entries must appear in the bundle
    assert "decision/arch-v0" in bundle.context_block, \
        "decision namespace entry must appear in context_block"
    assert "project/ground-call" in bundle.context_block, \
        "project namespace entry must appear in context_block"


# ---------------------------------------------------------------------------
# 9. PM-state read completes under guard with real stores (regression gate)
# ---------------------------------------------------------------------------

def test_pm_state_read_completes_under_guard(tmp_path):
    """ground(pm_state=True) returns stale=False with real MemoryStore/TargetStore.

    This is the regression test that would have caught the 200ms-guard defect:
    the old guard was below the healthy read time so stale was always True.
    """
    from agents_core.mem import MemoryStore as RealMemoryStore
    from agents_core.targets import TargetStore as RealTargetStore

    targets_dir = _make_target_yaml(tmp_path, "alpha-v0", "Alpha Target", pm_bound=True)
    _make_target_yaml(tmp_path, "beta-v0", "Beta Target", pm_bound=True)
    _make_target_yaml(tmp_path, "gamma-v0", "Gamma Target", pm_bound=False)

    db = _make_mem_db(tmp_path, [
        {
            "key": "router/lapis-pm/decisions/evt-alpha-001",
            "content": json.dumps({
                "freshness_stamp": "2026-06-24T10:00:00Z",
                "verdict": "proposed",
                "expert_chosen": "fixer",
                "intent_summary": "implement the PM-state guard fix",
                "target_id": "alpha-v0",
            }),
            "tags": "lapis-pm,router-portfolio,target:alpha-v0",
        },
        {
            "key": "router/lapis-pm/decisions/evt-beta-001",
            "content": json.dumps({
                "freshness_stamp": "2026-06-24T09:00:00Z",
                "verdict": "dispatched",
                "expert_chosen": "fixer",
                "intent_summary": "beta work in progress",
                "target_id": "beta-v0",
            }),
            "tags": "lapis-pm,router-portfolio,target:beta-v0",
        },
    ])

    with patch("agents_core.ground.MemoryStore", side_effect=lambda: RealMemoryStore(db)), \
         patch("agents_core.ground.TargetStore", side_effect=lambda: RealTargetStore(targets_dir)), \
         patch("agents_core.ground.retrieve", return_value=[]):
        bundle = ground("PM state query", pm_state=True, token_budget=3000)

    assert bundle.stale is False, "healthy PM-state read must not set stale=True"
    assert "pm:alpha-v0" in bundle.context_block, "active pm_bound target must appear"
    assert "pm:beta-v0" in bundle.context_block, "second pm_bound target must appear"
    ungrounded_pm = [
        p for p in bundle.provenance
        if p.get("tag") == "ungrounded" and p.get("source") == "pm_state"
    ]
    assert len(ungrounded_pm) == 0, "no ungrounded pm_state marker on healthy read"


# ---------------------------------------------------------------------------
# 10. GROUND_PM_TIMEOUT_S env override is honored
# ---------------------------------------------------------------------------

def test_pm_timeout_env_override(monkeypatch):
    """GROUND_PM_TIMEOUT_S env var is read at import time and sets _PM_TIMEOUT_S."""
    import importlib
    import agents_core.ground as gmod

    monkeypatch.setenv("GROUND_PM_TIMEOUT_S", "3.7")
    importlib.reload(gmod)
    try:
        assert gmod._PM_TIMEOUT_S == pytest.approx(3.7), \
            f"Expected _PM_TIMEOUT_S=3.7 after reload with env var, got {gmod._PM_TIMEOUT_S}"
    finally:
        monkeypatch.delenv("GROUND_PM_TIMEOUT_S", raising=False)
        importlib.reload(gmod)  # restore default for subsequent tests


# ---------------------------------------------------------------------------
# 11. Bounded read: semantic regression — active targets + decisions + briefs
# ---------------------------------------------------------------------------

def test_bounded_read_surfaces_active_targets_and_decisions(tmp_path):
    """Bounded _do_pm_state_read still surfaces active targets + latest decision + brief."""
    from agents_core.mem import MemoryStore as RealMemoryStore
    from agents_core.targets import TargetStore as RealTargetStore

    targets_dir = _make_target_yaml(tmp_path, "foo-v0", "Foo Target", pm_bound=True)

    db = _make_mem_db(tmp_path, [
        {
            "key": "router/lapis-pm/decisions/evt-foo-001",
            "content": json.dumps({
                "freshness_stamp": "2026-06-24T08:00:00Z",
                "verdict": "proposed",
                "expert_chosen": "fixer",
                "intent_summary": "do the foo thing",
                "target_id": "foo-v0",
            }),
            "tags": "lapis-pm,router-portfolio,target:foo-v0",
        },
        {
            "key": "pm/outstanding-brief/foo-v0",
            "content": "Awaiting Erah review of the fixer PR.",
            "tags": "lapis-pm,target:foo-v0",
        },
    ])

    with patch("agents_core.ground.MemoryStore", side_effect=lambda: RealMemoryStore(db)), \
         patch("agents_core.ground.TargetStore", side_effect=lambda: RealTargetStore(targets_dir)), \
         patch("agents_core.ground.retrieve", return_value=[]):
        bundle = ground("foo query", pm_state=True, token_budget=3000)

    assert bundle.stale is False
    assert "pm:foo-v0" in bundle.context_block
    assert "proposed" in bundle.context_block, "latest decision verdict must appear"
    assert "fixer" in bundle.context_block, "expert_chosen must appear"
    assert "do the foo thing" in bundle.context_block, "intent_summary must appear"
    assert "Awaiting Erah review" in bundle.context_block, "outstanding brief must appear"


# ---------------------------------------------------------------------------
# 12. Slow-leak warning fires for reads above soft threshold but under guard
# ---------------------------------------------------------------------------

def test_slow_leak_warning_fires(caplog):
    """A healthy read >= 60% of the guard emits log.warning; stale stays False."""
    import logging
    import time
    import agents_core.ground as gmod

    orig_timeout = gmod._PM_TIMEOUT_S
    gmod._PM_TIMEOUT_S = 0.5  # guard = 500ms; threshold = 300ms

    def _slow_but_ok():
        time.sleep(0.35)  # 70% of guard — above threshold, below guard
        return (
            "[pm:slow-target]\ntarget:slow-target (Slow) [low]",
            [{"tag": "pm:slow-target", "source": "pm_state", "score": None, "why": "active-target"}],
        )

    try:
        with patch("agents_core.ground._do_pm_state_read", side_effect=_slow_but_ok), \
             patch("agents_core.ground.retrieve", return_value=[]), \
             caplog.at_level(logging.WARNING, logger="agents_core.ground"):
            bundle = ground("q", pm_state=True)
    finally:
        gmod._PM_TIMEOUT_S = orig_timeout

    assert bundle.stale is False, "slow-but-healthy read must not set stale=True"
    slow_warnings = [r for r in caplog.records if "slow" in r.message.lower()]
    assert len(slow_warnings) >= 1, "warning must be emitted for read above soft threshold"
    assert "pm_state read failed" not in caplog.text, "must not emit failure message"


# ---------------------------------------------------------------------------
# 13. GROUND_TOTAL_BUDGET_S — RAG skipped when PM-state exhausts the budget
# ---------------------------------------------------------------------------

def test_rag_skipped_when_pm_exhausts_budget():
    """When PM-state consumes > GROUND_TOTAL_BUDGET_S, RAG is not queried and stale=True."""
    import time
    import agents_core.ground as gmod

    orig_budget = gmod._TOTAL_BUDGET_S
    gmod._TOTAL_BUDGET_S = 0.05  # 50ms budget

    retrieve_calls = []

    def slow_pm_state():
        time.sleep(0.15)  # 3x the budget
        return (
            "[pm:x]\ntarget:x (X) [high]",
            [{"tag": "pm:x", "source": "pm_state", "score": None, "why": "active-target"}],
            False,
        )

    try:
        with patch("agents_core.ground._assemble_pm_state", side_effect=slow_pm_state), \
             patch("agents_core.ground.retrieve",
                   side_effect=lambda *a, **kw: retrieve_calls.append(kw) or []):
            bundle = ground("q", pm_state=True)
    finally:
        gmod._TOTAL_BUDGET_S = orig_budget

    assert len(retrieve_calls) == 0, "retrieve must not be called when budget exhausted by PM-state"
    assert bundle.stale is True
    ungrounded = [p for p in bundle.provenance if p.get("why") == "budget-exhausted"]
    assert len(ungrounded) >= 1, "budget-exhausted ungrounded markers must appear in provenance"


# ---------------------------------------------------------------------------
# 14. Remaining budget is passed to retrieve() as effective timeout
# ---------------------------------------------------------------------------

def test_remaining_budget_passed_to_retrieve():
    """After PM-state, ground() passes min(RAG_HTTP_TIMEOUT, remaining) to retrieve."""
    import time
    import agents_core.ground as gmod
    import agents_core.retrieval as rmod

    orig_budget = gmod._TOTAL_BUDGET_S
    gmod._TOTAL_BUDGET_S = 0.4  # 400ms total budget

    captured_timeouts = []

    def pm_consumes_200ms():
        time.sleep(0.2)  # leaves ~200ms remaining, < 1.5s RAG_HTTP_TIMEOUT
        return ("", [], False)

    def capture_retrieve(query, scope, top_k=10, min_score=0.0, timeout=None, **kw):
        captured_timeouts.append(timeout)
        return []

    try:
        with patch("agents_core.ground._assemble_pm_state", side_effect=pm_consumes_200ms), \
             patch("agents_core.ground.retrieve", side_effect=capture_retrieve):
            bundle = ground("q", pm_state=True)
    finally:
        gmod._TOTAL_BUDGET_S = orig_budget

    assert len(captured_timeouts) == 1, "retrieve should be called exactly once"
    t = captured_timeouts[0]
    assert t is not None, "timeout kwarg must be passed to retrieve"
    assert t < rmod.RAG_HTTP_TIMEOUT, "effective timeout must be < RAG_HTTP_TIMEOUT when budget constrained"
    assert t <= 0.4, "effective timeout must not exceed the total budget"
    assert bundle.stale is True, "budget-constrained RAG timeout must set stale=True"


# ---------------------------------------------------------------------------
# 15. retrieve(timeout=...) override — unit test for the retrieval layer
# ---------------------------------------------------------------------------

def test_retrieve_timeout_override():
    """retrieve() passes an explicit timeout to _search_rag; None preserves module default."""
    from unittest.mock import patch as mpatch
    from agents_core import retrieval as rmod

    captured = []

    def fake_rag(source, query, filters, timeout=None):
        captured.append(timeout)
        return []

    with mpatch.object(rmod, "_search_rag", side_effect=fake_rag):
        rmod.retrieve("q", scope=["vault-rag"], timeout=0.3)
        rmod.retrieve("q", scope=["vault-rag"])

    assert len(captured) == 2
    assert captured[0] == pytest.approx(0.3), "explicit timeout must be threaded to _search_rag"
    assert captured[1] is None, "omitting timeout must pass None (module default applies in _search_rag)"


# ---------------------------------------------------------------------------
# 16. Healthy path with budget: stale=False when backends are fast
# ---------------------------------------------------------------------------

def test_healthy_path_budget_stale_false():
    """With fast backends and normal budget, the total-budget mechanism leaves stale=False."""
    import agents_core.ground as gmod

    # Verify the default budget is sane (> RAG_HTTP_TIMEOUT so normal calls are never stale)
    import agents_core.retrieval as rmod
    assert gmod._TOTAL_BUDGET_S > rmod.RAG_HTTP_TIMEOUT, \
        "default GROUND_TOTAL_BUDGET_S must exceed RAG_HTTP_TIMEOUT so healthy calls aren't stale"

    fake_hits = [_make_hit("mem", "decision/x", "content", 0.9)]

    with patch("agents_core.ground.retrieve", return_value=fake_hits), \
         patch("agents_core.ground._assemble_pm_state", return_value=("pm text", [], False)):
        bundle = ground("q")

    assert bundle.stale is False, "healthy fast call must not be marked stale"
    assert "decision/x" in bundle.context_block
    assert "pm text" in bundle.context_block


# ---------------------------------------------------------------------------
# 17. GROUND_TOTAL_BUDGET_S env override is honored (reload test)
# ---------------------------------------------------------------------------

def test_total_budget_env_override(monkeypatch):
    """GROUND_TOTAL_BUDGET_S env var sets _TOTAL_BUDGET_S at import time."""
    import importlib
    import agents_core.ground as gmod

    monkeypatch.setenv("GROUND_TOTAL_BUDGET_S", "4.2")
    importlib.reload(gmod)
    try:
        assert gmod._TOTAL_BUDGET_S == pytest.approx(4.2), \
            f"Expected _TOTAL_BUDGET_S=4.2, got {gmod._TOTAL_BUDGET_S}"
    finally:
        monkeypatch.delenv("GROUND_TOTAL_BUDGET_S", raising=False)
        importlib.reload(gmod)
