"""Tests for the dead-stream quarantine runner (mem-hygiene-automation-v0).

D-5: quarantine/restore round-trip on a fixture db; FTS integrity
post-run; live-producer guard (a router-style live prefix is never
quarantined); batch-cap and rollback-window math; the D4 write guard
rejects test sources without the env flag. All hermetic: tmp_path fixture
dbs, no live mem.db, no network.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents_core.mem import MemoryStore, TestWriteRejected
from agents_core.mem_hygiene import (
    ATOM_CLASS_PREFIXES,
    DEFAULT_BATCH_CAP,
    HygieneAborted,
    HygieneConfig,
    MemHygieneRunner,
    is_atom_class_key,
)

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, *, allowlist, dead_sources, **overrides) -> Path:
    registry = tmp_path / "dead_producers.json"
    registry.write_text(json.dumps({
        "version": 1,
        "dead_sources": {k: sorted(v) for k, v in dead_sources.items()},
    }))
    cfg = tmp_path / "mem_hygiene.json"
    payload = {
        "registry": "dead_producers.json",
        "allowlist": list(allowlist),
        "dead_stream_age_days": 30,
        "batch_cap": 5000,
        "rollback_window_days": 14,
    }
    payload.update(overrides)
    cfg.write_text(json.dumps(payload))
    return cfg


def _backdate(conn, prefix: str, age_days: float) -> None:
    """Back-date every row under `prefix` by `age_days` (fixture helper)."""
    ts = (NOW - timedelta(days=age_days)).isoformat()
    conn.execute(
        "UPDATE memories SET created_at=?, updated_at=? WHERE key LIKE ?",
        (ts, ts, prefix + "%"),
    )
    conn.commit()


def _seed_dead_stream(tmp_path: Path, prefix: str, source: str,
                      n: int = 5, age_days: float = 45) -> None:
    """Seed `prefix` rows.

    Non-exhaust prefixes go through store.set() (the normal write path);
    the tier-1 exhaust prefixes (elevator/, weather/) are seeded via
    direct SQL on the `memories` table — store.set() would route them to
    the sibling exhaust store, and the quarantine runner classifies the
    mem.db table (the spec's own counts, elevator/ 10,975 + weather/
    2,407, are mem.db rows).
    """
    from agents_core import mem_exhaust
    store = MemoryStore(db_path=tmp_path / "mem.db")
    ts = (NOW - timedelta(days=age_days)).isoformat()
    for i in range(n):
        key = f"{prefix}row-{i}"
        if mem_exhaust.route_to_exhaust(key):
            store._conn.execute(
                "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, f"machine state {i}", "machine", source, ts, ts),
            )
        else:
            store.set(key, f"machine state {i}", tags=["machine"], source=source)
            store._conn.execute(
                "UPDATE memories SET created_at=?, updated_at=? WHERE key=?",
                (ts, ts, key),
            )
    store._conn.commit()
    store.close()


def _make_runner(tmp_path: Path, cfg_path: Path, **kw) -> MemHygieneRunner:
    store = MemoryStore(db_path=tmp_path / "mem.db")
    config = HygieneConfig.load(cfg_path)
    return MemHygieneRunner(store, config, run_id="test-run",
                            artifact_dir=tmp_path, **kw)


# ---------------------------------------------------------------------------
# D2 — dead-stream classifier
# ---------------------------------------------------------------------------

def test_eligible_dead_stream_is_quarantine_eligible(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler")
    runner = _make_runner(tmp_path, cfg)
    prefixes = runner.classify_prefixes(now=NOW)
    assert len(prefixes) == 1
    p = prefixes[0]
    assert p.eligible is True
    assert p.row_count == 5
    assert p.observed_sources == {"elevator_scheduler"}


def test_live_stream_within_age_window_is_ineligible(tmp_path):
    """(b): sources registered-dead but last write inside the window."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", age_days=5)
    runner = _make_runner(tmp_path, cfg)
    p = runner.classify_prefixes(now=NOW)[0]
    assert p.eligible is False
    assert "window" in p.reason


def test_unregistered_sources_fail_closed(tmp_path):
    """(c): sources not in the registry (hostname-shaped) -> ineligible.
    This is exactly router/'s shape: it can never be quarantined without
    explicit manual registration."""
    cfg = _write_config(
        tmp_path,
        allowlist=["router/"],
        dead_sources={},
    )
    _seed_dead_stream(tmp_path, "router/", "brix", age_days=90)
    runner = _make_runner(tmp_path, cfg)
    p = runner.classify_prefixes(now=NOW)[0]
    assert p.eligible is False
    assert "fail-closed" in p.reason


def test_new_source_appeared_superset_fails_closed(tmp_path):
    """(c): a NEW source under a registered prefix (a resurrected or new
    producer) makes the observed set a strict superset -> ineligible."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=3, age_days=90)
    store = MemoryStore(db_path=tmp_path / "mem.db")
    ts = (NOW - timedelta(days=90)).isoformat()
    # Direct SQL: store.set() would route an elevator/ key to the exhaust
    # store; this row must land in mem.db's memories table.
    store._conn.execute(
        "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("elevator/new-producer-row", "resurrected", "", "elevator_v2", ts, ts),
    )
    store._conn.commit()
    store.close()
    runner = _make_runner(tmp_path, cfg)
    p = runner.classify_prefixes(now=NOW)[0]
    assert p.eligible is False
    assert "fail-closed" in p.reason


def test_dual_store_last_write_measured_across_exhaust(tmp_path):
    """(b) dual-store: the exhaust twin is newer than mem.db -> the prefix
    is NOT dead even though mem.db alone would look old."""
    cfg = _write_config(
        tmp_path,
        allowlist=["weather/"],
        dead_sources={"weather/": ["ops-primitives"]},
    )
    _seed_dead_stream(tmp_path, "weather/", "ops-primitives", age_days=45)
    # The exhaust twin (a sibling file next to the fixture db) is newer.
    exhaust = MemoryStore(db_path=tmp_path / "mem.db")._exhaust_store()
    exhaust.set("weather/2026-09-10/new-row", "still writing",
                source="ops-primitives")
    runner = _make_runner(tmp_path, cfg)
    p = runner.classify_prefixes(now=NOW)[0]
    assert p.eligible is False
    assert "window" in p.reason


def test_dual_store_exhaust_older_does_not_rescue(tmp_path):
    """(b) dual-store: mem.db newer than the exhaust twin -> the mem.db
    last write governs (here: eligible)."""
    cfg = _write_config(
        tmp_path,
        allowlist=["weather/"],
        dead_sources={"weather/": ["ops-primitives"]},
    )
    _seed_dead_stream(tmp_path, "weather/", "ops-primitives", age_days=45)
    exhaust = MemoryStore(db_path=tmp_path / "mem.db")._exhaust_store()
    old = (NOW - timedelta(days=60)).isoformat()
    exhaust._conn.execute(
        "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
        "VALUES (?, ?, '', ?, ?, ?)",
        ("weather/old-row", "old", "ops-primitives", old, old),
    )
    exhaust._conn.commit()
    runner = _make_runner(tmp_path, cfg)
    p = runner.classify_prefixes(now=NOW)[0]
    assert p.eligible is True


# ---------------------------------------------------------------------------
# D6 — atom-class backstop
# ---------------------------------------------------------------------------

def test_atom_class_prefix_never_eligible(tmp_path):
    for prefix in ATOM_CLASS_PREFIXES:
        assert is_atom_class_key(prefix + "some-key")
    cfg = _write_config(
        tmp_path,
        allowlist=["decision/", "elevator/"],
        dead_sources={"decision/": ["anything"], "elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "decision/", "anything", age_days=90)
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", age_days=90)
    runner = _make_runner(tmp_path, cfg)
    by_prefix = {p.prefix: p for p in runner.classify_prefixes(now=NOW)}
    assert by_prefix["decision/"].eligible is False
    assert "D6" in by_prefix["decision/"].reason
    assert by_prefix["elevator/"].eligible is True


# ---------------------------------------------------------------------------
# D1 — quarantine / restore round-trip
# ---------------------------------------------------------------------------

def test_quarantine_restore_round_trip(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=5, age_days=45)
    runner = _make_runner(tmp_path, cfg)

    verdict = runner.run_quarantine(dry_run=False)
    assert verdict.quarantined == 5
    assert verdict.already_quarantined == 0
    assert verdict.aborted is False
    assert verdict.fts_integrity_ok is True

    # The rows are gone from memories...
    store = MemoryStore(db_path=tmp_path / "mem.db")
    assert store.list_by_prefix("elevator/") == []

    # ...and searchable FTS agrees (trigger-covered DELETE).
    assert store.search("machine state") == []

    # Restore: the INSERT..SELECT + DELETE pair (memories_ai re-indexes).
    restored = runner.restore_prefix("elevator/")
    assert restored == 5
    rows = store.list_by_prefix("elevator/")
    assert len(rows) == 5
    assert {r["key"] for r in rows} == {f"elevator/row-{i}" for i in range(5)}
    # FTS re-indexed on the restore INSERT.
    hits = store.search("machine state")
    assert len(hits) == 5
    # Quarantine table is drained for the prefix.
    stats = runner.quarantine_stats()
    assert stats["total"] == 0
    store.close()


def test_quarantine_delete_fires_fts_trigger(tmp_path):
    """The quarantine DELETE must leave memories_fts in sync (healthz
    in_sync semantics)."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=3, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    runner.run_quarantine(dry_run=False)

    conn = runner._store._conn
    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    fts_count = conn.execute(
        "SELECT COUNT(*) FROM memories_fts_docsize"
    ).fetchone()[0]
    assert mem_count == fts_count  # in_sync


def test_quarantine_pk_key_run_id_allows_re_quarantine_after_restore(tmp_path):
    """PK (key, run_id): after restore + a second run with a new run_id,
    the same key quarantines again without a PK collision."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=2, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    runner.run_quarantine(dry_run=False)
    runner.restore_prefix("elevator/")

    runner2 = _make_runner(tmp_path, cfg)
    runner2.run_id = "test-run-2"
    v2 = runner2.run_quarantine(dry_run=False)
    assert v2.quarantined == 2
    # restore_prefix DRAINS the quarantine table for the restored prefix
    # (INSERT..SELECT + DELETE pair), so the first run's rows are gone.
    # Only the second run's (key, run_id) pairs remain: 2 keys x 1 run_id.
    rows = runner2._store._conn.execute(
        "SELECT key, run_id FROM memories_quarantine ORDER BY run_id"
    ).fetchall()
    assert len(rows) == 2  # 2 keys x 1 run_id (test-run-2)
    assert all(r["run_id"] == "test-run-2" for r in rows)


# ---------------------------------------------------------------------------
# D3 — dry-run, cap, abort
# ---------------------------------------------------------------------------

def test_dry_run_writes_artifact_and_mutates_nothing(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=4, age_days=45)
    runner = _make_runner(tmp_path, cfg)

    verdict = runner.run_quarantine(dry_run=True)
    assert verdict.mode == "dry-run"
    assert verdict.candidate_count == 4
    assert verdict.quarantined == 0
    assert verdict.candidate_artifact is not None

    artifact = json.loads(Path(verdict.candidate_artifact).read_text())
    assert artifact["candidate_count"] == 4
    assert len(artifact["candidates"]) == 4
    assert artifact["eligible_prefixes"] == ["elevator/"]
    # Default artifact mode is 'dry-run' (the snapshot is written
    # dry-run-style); a real run passes mode="run" explicitly (reviewer
    # medium, PR #331 cycle 1).
    assert artifact["mode"] == "dry-run"

    # No mutation: everything still in memories, quarantine table empty.
    store = MemoryStore(db_path=tmp_path / "mem.db")
    assert len(store.list_by_prefix("elevator/")) == 4
    assert runner.quarantine_stats()["total"] == 0
    store.close()


def test_batch_cap_aborts_scheduled_path(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
        batch_cap=3,
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=5, age_days=45)
    runner = _make_runner(tmp_path, cfg)

    with pytest.raises(HygieneAborted, match="exceeds batch cap"):
        runner.run_quarantine(dry_run=False, allow_over_cap=False)
    # Nothing mutated (the abort is pre-transaction).
    store = MemoryStore(db_path=tmp_path / "mem.db")
    assert len(store.list_by_prefix("elevator/")) == 5
    store.close()


def test_over_cap_allowed_with_flag(tmp_path):
    """The first pass: above-cap run with allow_over_cap=True (the
    mandatory db-file backup is the caller's job, not the runner's)."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
        batch_cap=3,
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=5, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    verdict = runner.run_quarantine(dry_run=False, allow_over_cap=True)
    assert verdict.quarantined == 5
    assert verdict.allow_over_cap is True


def test_no_candidates_is_a_clean_run(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    # Nothing seeded at all.
    runner = _make_runner(tmp_path, cfg)
    verdict = runner.run_quarantine(dry_run=False)
    assert verdict.candidate_count == 0
    assert verdict.quarantined == 0
    assert verdict.aborted is False


def test_quarantine_chunks_mutation_above_bound_parameter_cap(tmp_path):
    """The first-pass shape: a candidate set larger than sqlite's 999
    bound-parameter cap must not raise too-many-SQL-variables — the
    mutation path chunks the IN-lists the way list_candidates does."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
        batch_cap=5000,
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=1200, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    verdict = runner.run_quarantine(dry_run=False)
    assert verdict.quarantined == 1200
    assert verdict.already_quarantined == 0
    assert verdict.aborted is False
    assert verdict.fts_integrity_ok is True
    store = MemoryStore(db_path=tmp_path / "mem.db")
    assert store.list_by_prefix("elevator/") == []
    store.close()


def test_count_mismatch_aborts_and_rolls_back(tmp_path, monkeypatch):
    """The count-mismatch guard is on the DELETE rowcount (not a derived
    tautology): if the store drifts mid-run so fewer rows delete than
    were candidates, the run aborts and the whole batch rolls back."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=4, age_days=45)
    runner = _make_runner(tmp_path, cfg)

    real_conn = runner._store._conn

    class DriftConn:
        """Proxy over the real connection that simulates the store
        drifting mid-run: one candidate key vanishes from memories
        between enumeration and the DELETE."""

        def __init__(self, conn):
            self._conn = conn
            self._drifted = False

        def execute(self, sql, *args):
            if (not self._drifted
                    and isinstance(sql, str)
                    and sql.lstrip().startswith("DELETE FROM memories")):
                self._drifted = True
                key = args[0][0]
                self._conn.execute(
                    "DELETE FROM memories WHERE key = ?", (key,)
                )
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(runner._store, "_conn", DriftConn(real_conn))

    with pytest.raises(HygieneAborted, match="count mismatch"):
        runner.run_quarantine(dry_run=False)

    # Full rollback: the quarantined rows are gone, and the memories
    # rows are fully intact — the drift DELETE happened INSIDE the
    # transaction (it was issued on the same connection, before the
    # batch DELETE), so ROLLBACK undoes it too. The count is 4, not 3:
    # the drift is a mid-run artifact, not a committed external write.
    conn = runner._store._conn
    assert conn.execute(
        "SELECT COUNT(*) FROM memories_quarantine"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE key LIKE 'elevator/%'"
    ).fetchone()[0] == 4


def test_re_run_after_crash_is_idempotent(tmp_path):
    """A crashed run re-runs clean: the second run finds 0 candidates
    (the first committed) and quarantines nothing."""
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=3, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    v1 = runner.run_quarantine(dry_run=False)
    assert v1.quarantined == 3

    runner2 = _make_runner(tmp_path, cfg)
    v2 = runner2.run_quarantine(dry_run=False)
    assert v2.candidate_count == 0
    assert v2.quarantined == 0


# ---------------------------------------------------------------------------
# D1 — ageout (rollback window math)
# ---------------------------------------------------------------------------

def test_ageout_purges_only_past_window(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
        rollback_window_days=14,
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=3, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    runner.run_quarantine(dry_run=False)

    # Freshly quarantined rows are WITHIN the window: ageout purges 0.
    assert runner.ageout() == 0
    assert runner.quarantine_stats()["total"] == 3

    # Back-date the quarantine rows past the window.
    old = (NOW - timedelta(days=20)).isoformat()
    conn = runner._store._conn
    conn.execute("UPDATE memories_quarantine SET quarantined_at = ?", (old,))
    conn.commit()
    assert runner.ageout() == 3
    assert runner.quarantine_stats()["total"] == 0


def test_ageout_respects_explicit_window(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    _seed_dead_stream(tmp_path, "elevator/", "elevator_scheduler", n=2, age_days=45)
    runner = _make_runner(tmp_path, cfg)
    runner.run_quarantine(dry_run=False)
    # A 0-day window purges everything immediately.
    assert runner.ageout(window_days=0) == 2
    assert runner.quarantine_stats()["total"] == 0


def test_restore_refuses_atom_class_prefix(tmp_path):
    cfg = _write_config(
        tmp_path,
        allowlist=["elevator/"],
        dead_sources={"elevator/": ["elevator_scheduler"]},
    )
    runner = _make_runner(tmp_path, cfg)
    with pytest.raises(ValueError, match="atom-class"):
        runner.restore_prefix("decision/")


# ---------------------------------------------------------------------------
# D4 — write-path guard at MemoryStore.set()
# ---------------------------------------------------------------------------

def test_write_guard_rejects_test_source_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    store = MemoryStore(db_path=tmp_path / "guard.db")
    with pytest.raises(TestWriteRejected):
        store.set("pattern/whatever", "x", source="gw_topology_test")
    # Nothing landed.
    assert store.get("pattern/whatever") is None
    store.close()


@pytest.mark.parametrize("source", [
    "test",
    "tests",
    "unit_test_runner",
    "mock-source",
    "fake",
    "fixture",
    "test-run-123",
])
def test_write_guard_pattern_variants(tmp_path, monkeypatch, source):
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    store = MemoryStore(db_path=tmp_path / "guard.db")
    with pytest.raises(TestWriteRejected):
        store.set("pattern/x", "x", source=source)
    store.close()


@pytest.mark.parametrize("source", [
    "gw_topology",        # production source (the 08-11 finding's shape)
    "elevator_scheduler",
    "ops-primitives",
    "lapis-pm/opencode",
    "brix",
    "starhouse",
    "user/BRIX interactive PM 2026-08-11",
])
def test_write_guard_allows_production_sources(tmp_path, monkeypatch, source):
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    store = MemoryStore(db_path=tmp_path / "guard.db")
    created = store.set("pattern/x", "x", source=source)
    assert created is True
    store.close()


def test_write_guard_empty_source_uses_hostname_default(tmp_path, monkeypatch):
    """The guard matches the EXPLICIT source only — an empty source
    falls through to the hostname default and is never pattern-matched
    (a production writer that omits source on a test host is not
    locked out)."""
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    store = MemoryStore(db_path=tmp_path / "guard.db")
    created = store.set("pattern/x", "x")
    assert created is True
    row = store.get("pattern/x")
    assert row["source"]  # hostname default populated
    store.close()


def test_write_guard_env_flag_allows(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM_ALLOW_TEST_WRITE", "1")
    store = MemoryStore(db_path=tmp_path / "guard.db")
    created = store.set("pattern/x", "x", source="test-harness")
    assert created is True
    store.close()


def test_write_guard_env_flag_must_be_exactly_one(tmp_path, monkeypatch):
    monkeypatch.setenv("MEM_ALLOW_TEST_WRITE", "0")
    store = MemoryStore(db_path=tmp_path / "guard.db")
    with pytest.raises(TestWriteRejected):
        store.set("pattern/x", "x", source="test-harness")
    store.close()


def test_write_guard_applies_to_exhaust_routed_writes(tmp_path, monkeypatch):
    """The guard fires BEFORE the exhaust routing (the chokepoint is
    the chokepoint)."""
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    store = MemoryStore(db_path=tmp_path / "guard.db")
    with pytest.raises(TestWriteRejected):
        store.set("weather/2026-09-14/x", "x", source="test")
    store.close()
