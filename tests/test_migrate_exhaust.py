"""Tests for the tier-1 exhaust migration tool (scripts/migrate_exhaust.py),
run against a scratch copy — never production (agents-core-mem-exhaust-
sibling-store-v0, leg 3)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from migrate_exhaust import migrate  # noqa: E402

from agents_core.mem import SCHEMA  # noqa: E402


ROWS = [
    # (key, content, tags, source, created_at, updated_at)
    ("elevator/proposals/p1", "proposal one", "elevator,proposal", "elevator_scheduler",
     "2026-08-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00"),
    ("elevator/proposals/p2", "proposal two", "elevator,proposal", "elevator_scheduler",
     "2026-08-02T00:00:00+00:00", "2026-08-02T00:00:00+00:00"),
    ("weather/2026-08-01/task-a", "weather payload a", "operational", "ops-primitives",
     "2026-08-01T01:00:00+00:00", "2026-08-01T01:00:00+00:00"),
    ("router/gw-review-divergence/r1", '{"run_id": "r1"}', "", "gw-review",
     "2026-08-03T00:00:00+00:00", "2026-08-03T00:00:00+00:00"),
    # Not routed — must survive untouched.
    ("router/lapis-pm/decisions/keep-me", '{"verdict": "keep"}', "router-portfolio", "lapis-pm",
     "2026-08-04T00:00:00+00:00", "2026-08-04T00:00:00+00:00"),
    ("review/finding/1", "unrelated finding", "review-finding", "pytest",
     "2026-08-05T00:00:00+00:00", "2026-08-05T00:00:00+00:00"),
]


def _build_scratch_mem_db(path: Path) -> None:
    """A scratch mem.db with the real production schema (FTS5 + triggers),
    seeded via raw INSERT so rows land as if written before this deploy
    existed — MemoryStore.set() would now route three of these away, which
    is exactly the pre-migration state this tool has to clean up."""
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    for key, content, tags, source, created_at, updated_at in ROWS:
        conn.execute(
            "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, content, tags, source, created_at, updated_at),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def scratch(tmp_path):
    mem_db = tmp_path / "scratch_mem.db"
    exhaust_db = tmp_path / "scratch_exhaust.db"
    _build_scratch_mem_db(mem_db)
    return mem_db, exhaust_db


def test_dry_run_writes_nothing(scratch):
    mem_db, exhaust_db = scratch
    before = mem_db.read_bytes()

    rc = migrate(mem_db, exhaust_db, write=False, allow_count_mismatch=True)

    assert rc == 0
    assert mem_db.read_bytes() == before
    assert not exhaust_db.exists() or sqlite3.connect(str(exhaust_db)).execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()[0] == 0


def test_write_moves_exactly_the_three_prefixes_byte_for_byte(scratch):
    mem_db, exhaust_db = scratch

    rc = migrate(mem_db, exhaust_db, write=True, allow_count_mismatch=True)
    assert rc == 0

    src = sqlite3.connect(str(mem_db))
    src.row_factory = sqlite3.Row
    dst = sqlite3.connect(str(exhaust_db))
    dst.row_factory = sqlite3.Row

    moved_keys = {
        "elevator/proposals/p1", "elevator/proposals/p2",
        "weather/2026-08-01/task-a", "router/gw-review-divergence/r1",
    }
    kept_keys = {"router/lapis-pm/decisions/keep-me", "review/finding/1"}

    # Moved rows: gone from source, present in destination, byte-for-byte.
    by_key = {r[0]: r[1:] for r in ROWS}
    for key in moved_keys:
        assert src.execute("SELECT 1 FROM memories WHERE key=?", (key,)).fetchone() is None
        drow = dst.execute(
            "SELECT content, tags, source, created_at, updated_at FROM memories WHERE key=?",
            (key,),
        ).fetchone()
        assert drow is not None
        assert tuple(drow) == by_key[key]

    # Untouched rows: still in source, absent from destination.
    for key in kept_keys:
        assert src.execute("SELECT 1 FROM memories WHERE key=?", (key,)).fetchone() is not None
        assert dst.execute("SELECT 1 FROM memories WHERE key=?", (key,)).fetchone() is None

    # memories / memories_fts row counts on the source still match.
    mem_count = src.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    fts_count = src.execute("SELECT COUNT(*) FROM memories_fts_docsize").fetchone()[0]
    assert mem_count == fts_count == len(kept_keys)

    src.close()
    dst.close()


def test_second_write_run_is_idempotent(scratch):
    mem_db, exhaust_db = scratch

    first_rc = migrate(mem_db, exhaust_db, write=True, allow_count_mismatch=True)
    assert first_rc == 0

    mem_db_before = mem_db.read_bytes()
    exhaust_db_before = exhaust_db.read_bytes()

    second_rc = migrate(mem_db, exhaust_db, write=True, allow_count_mismatch=True)

    assert second_rc == 0
    # A second run over already-migrated data touches nothing further.
    src = sqlite3.connect(str(mem_db))
    dst = sqlite3.connect(str(exhaust_db))
    assert src.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2  # the two kept rows
    assert dst.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 4  # the four moved rows
    src.close()
    dst.close()


def test_count_mismatch_is_a_stop_condition_without_override(scratch):
    mem_db, exhaust_db = scratch
    # Scratch data (6 rows) deviates wildly from the documented ~14,816 —
    # without the override flag this must refuse to write.
    rc = migrate(mem_db, exhaust_db, write=True, allow_count_mismatch=False)
    assert rc == 1

    src = sqlite3.connect(str(mem_db))
    # Nothing moved — the stop condition fired before any row was touched.
    assert src.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == len(ROWS)
    src.close()
