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


def test_list_all_multi_tag_intersection(tmp_path: Path):
    store = MemoryStore(db_path=tmp_path / "multitag.db")
    store.set("review/finding/1", "x", tags=["review-finding", "repo-a"])
    store.set("review/finding/2", "y", tags=["review-finding", "repo-b"])
    store.set("review/debt/a/1", "z", tags=["review-debt", "repo-a", "debt-open"])

    # single-tag form still works
    assert len(store.list_all(tag="review-finding")) == 2

    # intersection via tags list
    hits = store.list_all(tags=["review-finding", "repo-a"])
    assert [h["key"] for h in hits] == ["review/finding/1"]

    # tag + tags combined
    hits = store.list_all(tag="review-debt", tags=["repo-a", "debt-open"])
    assert [h["key"] for h in hits] == ["review/debt/a/1"]

    # no match on bogus intersection
    assert store.list_all(tags=["review-finding", "repo-z"]) == []
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
