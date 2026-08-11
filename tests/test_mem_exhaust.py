"""Tests for the exhaust sibling store and its wiring into MemoryStore
(agents-core-mem-exhaust-sibling-store-v0)."""
from __future__ import annotations

import logging
import sqlite3

import pytest

from agents_core import mem_exhaust
from agents_core.mem import MemoryStore
from agents_core.mem_exhaust import ExhaustStore, route_to_exhaust


# ---------------------------------------------------------------------------
# route_to_exhaust() — strict startswith matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [
    "elevator/proposals/2026-08-11T22-00-00Z",
    "weather/2026-08-11/task-42",
    "router/gw-review-divergence/f00fb861",
])
def test_route_to_exhaust_matches_tier1_prefixes(key):
    assert route_to_exhaust(key) is True


@pytest.mark.parametrize("key", [
    "router/lapis-pm/decisions/2026-08-11T22:38:45Z-77bcd29e",
    "pm/dispatched/some-target",
    "router/gw-review-divergen",  # not the full literal prefix
    "elevato/typo",
    "review/finding/1",
])
def test_route_to_exhaust_rejects_everything_else(key):
    assert route_to_exhaust(key) is False


def test_route_to_exhaust_no_substring_or_split_on_router(tmp_path):
    """A loose `router/` match would silently move router/lapis-pm/* (two
    live readers). Assert the specific case the spec calls out by name."""
    key = "router/lapis-pm/decisions/2026-08-11T22:38:45Z-77bcd29e"
    assert route_to_exhaust(key) is False

    store = MemoryStore(db_path=tmp_path / "mem.db")
    created = store.set(key, "some decision content", tags=["router-portfolio"])
    assert created is True

    # Landed in mem.db...
    row = store._conn.execute(
        "SELECT * FROM memories WHERE key = ?", (key,)
    ).fetchone()
    assert row is not None

    # ...and NOT in the sibling exhaust store — set() never routed there, so
    # exhaust.db was never even created.
    exhaust_path = tmp_path / "exhaust.db"
    assert not exhaust_path.exists()
    store.close()


# ---------------------------------------------------------------------------
# set() routing
# ---------------------------------------------------------------------------

def test_set_routes_weather_key_to_sibling_not_mem_db(tmp_path):
    mem_db = tmp_path / "mem.db"
    exhaust_db = tmp_path / "exhaust.db"
    store = MemoryStore(db_path=mem_db)

    key = "weather/2026-08-11/task-abc"
    created = store.set(key, "extracted primitives", tags=["operational"], source="ops-primitives")
    assert created is True

    # Not in mem.db.
    row = store._conn.execute("SELECT 1 FROM memories WHERE key = ?", (key,)).fetchone()
    assert row is None

    # In exhaust.db.
    assert exhaust_db.exists()
    exhaust_row = ExhaustStore(db_path=exhaust_db).get(key)
    assert exhaust_row is not None
    assert exhaust_row["content"] == "extracted primitives"
    store.close()


def test_set_exhaust_upsert_semantics_match_memorystore(tmp_path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    key = "elevator/proposals/p1"
    created = store.set(key, "v1")
    assert created is True
    updated = store.set(key, "v2")
    assert updated is False
    assert store.get(key)["content"] == "v2"
    store.close()


# ---------------------------------------------------------------------------
# get() fall-through + cold-path logging (DoD #4)
# ---------------------------------------------------------------------------

def test_get_fallthrough_returns_sibling_row_and_logs_once(tmp_path, caplog):
    mem_db = tmp_path / "mem.db"
    exhaust_db = tmp_path / "exhaust.db"

    # Seed the sibling directly, simulating a key already migrated/routed —
    # bypassing MemoryStore.set() so this test exercises get()'s fall-through
    # in isolation from set()'s routing.
    seed = ExhaustStore(db_path=exhaust_db)
    seed.set("elevator/proposals/already-migrated", "proposal payload", tags=["elevator"])
    seed.close()

    store = MemoryStore(db_path=mem_db)
    with caplog.at_level(logging.WARNING, logger="agents_core.mem.cold_path"):
        row = store.get("elevator/proposals/already-migrated")

    assert row is not None
    assert row["content"] == "proposal payload"

    cold_path_records = [r for r in caplog.records if r.name == "agents_core.mem.cold_path"]
    assert len(cold_path_records) == 1
    assert "elevator/proposals/already-migrated" in cold_path_records[0].message
    store.close()


def test_get_true_miss_does_not_log(tmp_path, caplog):
    """A key absent from BOTH stores is a real miss, not a cold-path event."""
    store = MemoryStore(db_path=tmp_path / "mem.db")
    with caplog.at_level(logging.WARNING, logger="agents_core.mem.cold_path"):
        row = store.get("nonexistent/key/entirely")
    assert row is None
    cold_path_records = [r for r in caplog.records if r.name == "agents_core.mem.cold_path"]
    assert len(cold_path_records) == 0
    store.close()


# ---------------------------------------------------------------------------
# router/lapis-pm/* must land in mem.db, never exhaust (DoD #3)
# ---------------------------------------------------------------------------

def test_router_lapis_pm_write_lands_in_mem_db(tmp_path):
    store = MemoryStore(db_path=tmp_path / "mem.db")
    key = "router/lapis-pm/decisions/2026-08-11T22:38:45Z-77bcd29e"
    store.set(key, json_content := '{"verdict": "proposed"}', tags=["router-portfolio"])

    got = store.get(key)
    assert got is not None
    assert got["content"] == json_content

    row = store._conn.execute("SELECT 1 FROM memories WHERE key = ?", (key,)).fetchone()
    assert row is not None, "router/lapis-pm/* must land in mem.db, not the exhaust sibling"
    store.close()


# ---------------------------------------------------------------------------
# list_by_prefix() merge behaviour
# ---------------------------------------------------------------------------

def test_list_by_prefix_merges_exhaust_rows_for_overlapping_prefix(tmp_path):
    mem_db = tmp_path / "mem.db"
    exhaust_db = tmp_path / "exhaust.db"

    store = MemoryStore(db_path=mem_db)
    # Legacy (pre-migration) row still sitting in mem.db under the prefix.
    store._conn.execute(
        "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
        "VALUES (?, ?, '', '', '2026-08-01T00:00:00Z', '2026-08-01T00:00:00Z')",
        ("elevator/proposals/legacy-1", "legacy content"),
    )
    store._conn.commit()

    # New write routes straight to exhaust.
    store.set("elevator/proposals/new-1", "new content")

    results = store.list_by_prefix("elevator/proposals/")
    keys = {r["key"] for r in results}
    assert keys == {"elevator/proposals/legacy-1", "elevator/proposals/new-1"}
    store.close()


def test_list_by_prefix_unrelated_prefix_never_opens_exhaust(tmp_path):
    mem_db = tmp_path / "mem.db"
    exhaust_db = tmp_path / "exhaust.db"
    store = MemoryStore(db_path=mem_db)
    store._conn.execute(
        "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
        "VALUES ('review/finding/1', 'x', '', '', '2026-08-01T00:00:00Z', '2026-08-01T00:00:00Z')"
    )
    store._conn.commit()

    results = store.list_by_prefix("review/finding/")
    assert [r["key"] for r in results] == ["review/finding/1"]
    assert not exhaust_db.exists()
    store.close()


# ---------------------------------------------------------------------------
# exhaust.db schema — no FTS5 (DoD #1)
# ---------------------------------------------------------------------------

def test_exhaust_db_has_no_fts5_table(tmp_path):
    exhaust_db = tmp_path / "exhaust.db"
    ExhaustStore(db_path=exhaust_db).close()

    conn = sqlite3.connect(str(exhaust_db))
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()}
    conn.close()

    assert "memories" in tables
    assert not any("fts" in t.lower() for t in tables)


# ---------------------------------------------------------------------------
# checkpoint_wal() covers both files in one pass (leg 4)
# ---------------------------------------------------------------------------

def test_checkpoint_wal_covers_sibling_once_opened(tmp_path):
    mem_db = tmp_path / "mem.db"
    exhaust_db = tmp_path / "exhaust.db"
    store = MemoryStore(db_path=mem_db)

    # Not opened yet -> checkpoint is a no-op for the sibling, and must not
    # conjure exhaust.db out of nothing.
    store.checkpoint_wal()
    assert not exhaust_db.exists()

    # A routed write opens the sibling in-process...
    store.set("weather/2026-08-11/task-x", "content")
    assert exhaust_db.exists()

    # ...so the next checkpoint call covers both files in one pass.
    store.checkpoint_wal()  # must not raise
    store.close()


def test_default_exhaust_path_is_sibling_of_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("MEM_EXHAUST_DB_PATH", raising=False)
    db_path = tmp_path / "somewhere" / "mem.db"
    assert mem_exhaust.default_exhaust_path(db_path) == db_path.parent / "exhaust.db"


def test_default_exhaust_path_env_override_wins(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere" / "exhaust.db"
    monkeypatch.setenv("MEM_EXHAUST_DB_PATH", str(override))
    assert mem_exhaust.default_exhaust_path(tmp_path / "mem.db") == override
