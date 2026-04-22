"""Smoke test for agents_core.mem — MemoryStore sqlite roundtrip."""
from __future__ import annotations

from pathlib import Path

from agents_core.mem import MemoryStore


def test_set_get_search(tmp_path: Path):
    db_path = tmp_path / "smoke.db"
    store = MemoryStore(db_path=db_path)

    created = store.set("pattern/test-key", "The quick brown fox jumps",
                        tags=["test", "smoke"])
    assert created is True

    # Update returns False
    updated = store.set("pattern/test-key", "The quick brown fox jumps again",
                        tags=["test", "smoke"])
    assert updated is False

    got = store.get("pattern/test-key")
    assert got is not None
    assert got["content"] == "The quick brown fox jumps again"
    assert "test" in got["tags"]

    hits = store.search("brown fox")
    assert len(hits) == 1
    assert hits[0]["key"] == "pattern/test-key"

    store.close()


def test_delete_and_stats(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "smoke2.db")
    store.set("a", "apple")
    store.set("b", "banana")
    assert store.stats()["total_memories"] == 2
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert store.stats()["total_memories"] == 1
    store.close()
