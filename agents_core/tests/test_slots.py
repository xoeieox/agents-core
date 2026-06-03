"""Tests for agents_core.slots — SlotStore project-slot blackboard.

Covers:
  - slot lifecycle (create -> status transitions -> terminal)
  - contributor-of-record single-writer guard (non-owner writes rejected)
  - observer writes land in the separate weaver_* namespace, not contributor fields
  - checkpoint append + domain_touch publish + escalate
  - adjacency by file / mem-key overlap (active-only by default)
  - expiration age-out (parked 30d, abandoned 7d)
  - status validation
  - JSON round-trip on read
  - concurrent writes under WAL don't corrupt or lose updates (in-process)
  - cross-process WAL + busy_timeout: two processes write concurrently without lock errors
"""

from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents_core.slots import (
    SCHEMA,
    OffMasterWriteError,
    SlotNotFoundError,
    SlotOwnershipError,
    SlotStore,
)


@pytest.fixture
def store(tmp_path: Path) -> SlotStore:
    s = SlotStore(db_path=tmp_path / "slots.db")
    yield s
    s.close()


CONTRIB = {"type": "fixer", "id": "agent-1"}


# --- lifecycle -------------------------------------------------------------

def test_create_and_get(store: SlotStore):
    sid = store.create_slot(
        project_id="proj-A",
        contributor=CONTRIB,
        horizon={"project_summary": "ctx-mgmt tail", "immediate_goal": "split retrieval"},
    )
    assert isinstance(sid, str) and len(sid) == 12
    slot = store.get(sid)
    assert slot["project_id"] == "proj-A"
    assert slot["status"] == "dispatched"
    assert slot["contributor_type"] == "fixer"
    assert slot["contributor_id"] == "agent-1"
    # JSON fields parsed back to objects
    assert slot["horizon"]["immediate_goal"] == "split retrieval"
    assert slot["checkpoints"] == []
    assert slot["started_at"]  # defaulted to now

def test_status_transitions(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.update_status(sid, "in-progress", by="agent-1")
    assert store.get(sid)["status"] == "in-progress"
    store.update_status(sid, "landed", by="agent-1")
    assert store.get(sid)["status"] == "landed"

def test_explicit_slot_id_and_duplicate_rejected(store: SlotStore):
    store.create_slot("proj-A", CONTRIB, slot_id="fixed-id")
    with pytest.raises(ValueError):
        store.create_slot("proj-A", CONTRIB, slot_id="fixed-id")

def test_invalid_status_rejected(store: SlotStore):
    with pytest.raises(ValueError):
        store.create_slot("proj-A", CONTRIB, status="bogus")
    sid = store.create_slot("proj-A", CONTRIB)
    with pytest.raises(ValueError):
        store.update_status(sid, "not-a-status", by="agent-1")


# --- single-writer guard ---------------------------------------------------

def test_non_owner_status_write_rejected(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    with pytest.raises(SlotOwnershipError):
        store.update_status(sid, "landed", by="intruder")
    # unchanged
    assert store.get(sid)["status"] == "dispatched"

def test_non_owner_checkpoint_and_domain_rejected(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    with pytest.raises(SlotOwnershipError):
        store.append_checkpoint(sid, "self-report", "x", by="intruder")
    with pytest.raises(SlotOwnershipError):
        store.set_domain_touch(sid, files=["a.py"], by="intruder")

def test_write_to_missing_slot_raises(store: SlotStore):
    with pytest.raises(SlotNotFoundError):
        store.update_status("nope", "landed", by="agent-1")
    with pytest.raises(SlotNotFoundError):
        store.observer_update("nope", "stuck")


# --- observer namespace separation -----------------------------------------

def test_observer_writes_separate_namespace(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.update_status(sid, "in-progress", by="agent-1")
    # Observer reports divergent state — recorded, NOT merged into contributor status
    store.observer_update(sid, "stuck", by="weaver")
    slot = store.get(sid)
    assert slot["status"] == "in-progress"        # contributor-of-record unchanged
    assert slot["weaver_status"] == "stuck"       # observer signal recorded separately
    assert slot["weaver_last_update"]


# --- Facets ratification namespace (blackboard step 4) ---------------------

def test_facets_ratify_separate_namespace(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.escalate(sid, to="facets", reason="authority undeclared", by="agent-1")
    verdict = {"deliberation_id": "d1", "council_status": "resolved",
               "council_landing": "proceed with changes", "confidence": "medium"}
    store.facets_ratify(sid, verdict, by="facets")
    slot = store.get(sid)
    # Facets verdict lands in its own namespace; status stays escalated (clearing the
    # escalation is the contributor-of-record's owner-guarded call, not Facets').
    assert slot["status"] == "escalated"
    assert slot["facets_verdict"]["council_status"] == "resolved"
    assert slot["facets_verdict"]["council_landing"] == "proceed with changes"
    assert slot["facets_verdict"]["by"] == "facets"
    assert slot["facets_verdict"]["ratified_at"]
    assert slot["facets_last_update"]
    # contributor + weaver namespaces untouched
    assert slot["weaver_status"] is None

def test_facets_ratify_missing_slot_raises(store: SlotStore):
    with pytest.raises(SlotNotFoundError):
        store.facets_ratify("nope", {"council_status": "resolved"})

def test_migration_idempotent_on_reopen(tmp_path: Path):
    # Opening an existing DB twice must not fail on ADD COLUMN (idempotent _migrate).
    db = tmp_path / "slots.db"
    s1 = SlotStore(db_path=db)
    sid = s1.create_slot("proj-A", CONTRIB)
    s1.close()
    s2 = SlotStore(db_path=db)          # re-open: _migrate runs again, must be a no-op
    assert s2.get(sid)["slot_id"] == sid
    s2.close()


def test_migration_adds_facets_columns_to_legacy_db(tmp_path: Path):
    """The real upgrade path: a pre-existing #52-shaped slots table (no facets_*
    columns) must be upgraded by _migrate's ADD COLUMN on open. A fresh CREATE TABLE
    already has the columns, so only a legacy DB exercises the migration itself."""
    db = tmp_path / "legacy.db"
    # A #52-era schema = the current SCHEMA with the facets_* lines removed.
    legacy_schema = "\n".join(
        ln for ln in SCHEMA.splitlines() if "facets" not in ln.lower()
    )
    assert "facets" not in legacy_schema.lower()  # guard: the fixture is truly #52-shaped
    conn = sqlite3.connect(db)
    conn.executescript(legacy_schema)
    # A row written under the #52 schema, before the facets columns existed.
    conn.execute(
        "INSERT INTO slots (slot_id, project_id, status, last_update, created_at) "
        "VALUES ('s1', 'proj-A', 'escalated', ?, ?)",
        ("2026-06-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
    )
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slots)").fetchall()}
    conn.close()
    assert "facets_verdict" not in cols  # confirm the ADD COLUMN path is actually untrodden

    # Opening via SlotStore runs _migrate -> the facets_* columns are added in place.
    store = SlotStore(db_path=db)
    migrated = {r[1] for r in store._conn.execute("PRAGMA table_info(slots)").fetchall()}
    assert {"facets_verdict", "facets_last_update"} <= migrated
    # The pre-existing row reads back with facets_verdict None-guarded (never ratified).
    assert store.get("s1")["facets_verdict"] is None
    # And ratification now works against the upgraded legacy table.
    store.facets_ratify("s1", {"council_status": "resolved"})
    assert store.get("s1")["facets_verdict"]["council_status"] == "resolved"
    store.close()


# --- checkpoints / domain_touch / escalate ---------------------------------

def test_append_checkpoint(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.append_checkpoint(sid, "self-report", "touched app.py", by="agent-1")
    store.append_checkpoint(sid, "reality-snap", "no overlap", by="agent-1")
    cps = store.get(sid)["checkpoints"]
    assert len(cps) == 2
    assert cps[0]["kind"] == "self-report"
    assert cps[1]["note"] == "no overlap"
    assert cps[0]["at"] <= cps[1]["at"]

def test_set_domain_touch_dedups_and_sorts(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(sid, files=["b.py", "a.py", "a.py"],
                           mem_keys=["k2", "k1"], by="agent-1")
    dt = store.get(sid)["domain_touch"]
    assert dt["files"] == ["a.py", "b.py"]
    assert dt["mem_keys"] == ["k1", "k2"]

def test_escalate(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.escalate(sid, to="facets", reason="authority undeclared", by="agent-1")
    slot = store.get(sid)
    assert slot["status"] == "escalated"
    assert slot["escalation"] == {"to": "facets", "reason": "authority undeclared"}


# --- query -----------------------------------------------------------------

def test_query_filters(store: SlotStore):
    store.create_slot("proj-A", {"type": "fixer", "id": "agent-1"}, slot_id="s1")
    store.create_slot("proj-A", {"type": "reviewer", "id": "agent-2"}, slot_id="s2")
    store.create_slot("proj-B", {"type": "fixer", "id": "agent-1"}, slot_id="s3")
    store.update_status("s2", "landed", by="agent-2")

    assert {s["slot_id"] for s in store.query(project_id="proj-A")} == {"s1", "s2"}
    assert {s["slot_id"] for s in store.query(status="dispatched")} == {"s1", "s3"}
    assert {s["slot_id"] for s in store.query(contributor_id="agent-1")} == {"s1", "s3"}


# --- adjacency -------------------------------------------------------------

def test_adjacent_file_and_memkey_overlap(store: SlotStore):
    a = store.create_slot("proj-A", {"type": "fixer", "id": "agent-1"})
    b = store.create_slot("proj-A", {"type": "fixer", "id": "agent-2"})
    store.set_domain_touch(a, files=["app.py", "fusion.py"], mem_keys=["mem/x"], by="agent-1")
    store.set_domain_touch(b, files=["render.py"], mem_keys=["mem/x"], by="agent-2")

    # query by a file only b doesn't touch -> only a
    hits = store.adjacent(files=["app.py"])
    assert {h["slot_id"] for h in hits} == {a}
    # query by shared mem-key -> both
    hits = store.adjacent(mem_keys=["mem/x"])
    assert {h["slot_id"] for h in hits} == {a, b}
    # exclude self
    hits = store.adjacent(mem_keys=["mem/x"], exclude_slot_id=a)
    assert {h["slot_id"] for h in hits} == {b}
    # overlap detail surfaced
    hits = store.adjacent(files=["fusion.py"], mem_keys=["mem/x"])
    by_id = {h["slot_id"]: h["_overlap"] for h in hits}
    assert by_id[a]["files"] == ["fusion.py"]
    assert by_id[a]["mem_keys"] == ["mem/x"]

def test_adjacent_excludes_inactive(store: SlotStore):
    a = store.create_slot("proj-A", {"type": "fixer", "id": "agent-1"})
    store.set_domain_touch(a, files=["app.py"], by="agent-1")
    store.update_status(a, "landed", by="agent-1")
    assert store.adjacent(files=["app.py"]) == []
    assert {h["slot_id"] for h in store.adjacent(files=["app.py"], include_inactive=True)} == {a}

def test_adjacent_empty_query_returns_nothing(store: SlotStore):
    a = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(a, files=["app.py"], by="agent-1")
    assert store.adjacent() == []


# --- expiration ------------------------------------------------------------

def test_expire_ages_out_parked_and_abandoned(store: SlotStore):
    parked_old = store.create_slot("p", CONTRIB, slot_id="parked-old")
    parked_new = store.create_slot("p", CONTRIB, slot_id="parked-new")
    aband_old = store.create_slot("p", CONTRIB, slot_id="aband-old")
    store.update_status("parked-old", "parked", by="agent-1")
    store.update_status("parked-new", "parked", by="agent-1")
    store.update_status("aband-old", "abandoned", by="agent-1")

    # Backdate last_update directly for the "old" rows.
    old = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    aband_old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    with store._lock:
        store._conn.execute("UPDATE slots SET last_update=? WHERE slot_id='parked-old'", (old,))
        store._conn.execute("UPDATE slots SET last_update=? WHERE slot_id='aband-old'", (aband_old_ts,))
        store._conn.commit()

    counts = store.expire()
    assert counts == {"parked_expired": 1, "abandoned_expired": 1}
    assert store.get("parked-old") is None
    assert store.get("aband-old") is None
    assert store.get("parked-new") is not None  # within 30d, survives


# --- stats -----------------------------------------------------------------

def test_stats(store: SlotStore):
    store.create_slot("p", CONTRIB, slot_id="s1")
    store.create_slot("p", CONTRIB, slot_id="s2")
    store.update_status("s2", "landed", by="agent-1")
    st = store.stats()
    assert st["total_slots"] == 2
    assert st["by_status"]["dispatched"] == 1
    assert st["by_status"]["landed"] == 1


# --- concurrency (WAL + lock) ----------------------------------------------

def test_concurrent_checkpoints_no_loss(store: SlotStore):
    sid = store.create_slot("p", CONTRIB)
    n = 50

    def worker(i: int):
        store.append_checkpoint(sid, "self-report", f"note-{i}", by="agent-1")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cps = store.get(sid)["checkpoints"]
    assert len(cps) == n  # no lost updates under the lock
    assert {c["note"] for c in cps} == {f"note-{i}" for i in range(n)}


# --- cross-process lock safety (D4: busy_timeout + WAL) -------------------

def _cross_proc_create_slots(db_path: str, prefix: str, n: int, result_queue) -> None:
    """Subprocess target: create n slots and report success count.

    create_slot is a pure INSERT (no read-modify-write), so two processes can
    race on it safely under WAL + busy_timeout without logical corruption.
    """
    try:
        store = SlotStore(db_path=db_path)
        for i in range(n):
            store.create_slot(
                f"{prefix}-proj-{i}",
                {"type": "fixer", "id": "agent-1"},
            )
        store.close()
        result_queue.put(n)
    except Exception as exc:
        result_queue.put(exc)


def test_cross_process_busy_timeout(tmp_path: Path):
    """Two separate processes INSERT slots concurrently.

    Without PRAGMA busy_timeout, the second writer raises OperationalError
    ("database is locked") immediately when the WAL write lock is held. With
    busy_timeout=5000 both processes complete without raising. Verifies D4.
    """
    db_path = str(tmp_path / "slots.db")
    # Initialize DB schema in-process before forking.
    store = SlotStore(db_path=db_path)
    store.close()

    n_per_proc = 20
    ctx = multiprocessing.get_context("fork")
    q: multiprocessing.Queue = ctx.Queue()
    p1 = ctx.Process(target=_cross_proc_create_slots, args=(db_path, "proc1", n_per_proc, q))
    p2 = ctx.Process(target=_cross_proc_create_slots, args=(db_path, "proc2", n_per_proc, q))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)

    assert p1.exitcode == 0, f"process 1 exited {p1.exitcode}"
    assert p2.exitcode == 0, f"process 2 exited {p2.exitcode}"

    results = [q.get_nowait() for _ in range(2)]
    for r in results:
        if isinstance(r, Exception):
            raise r
    assert sum(results) == n_per_proc * 2

    # Verify all slots were committed — no silent loss under concurrent INSERTs.
    store2 = SlotStore(db_path=db_path)
    total = store2.stats()["total_slots"]
    store2.close()
    assert total == n_per_proc * 2


# --- off-master write guard (D1) ------------------------------------------

def test_off_master_write_raises(tmp_path: Path, monkeypatch):
    """A SlotStore on a non-master host must refuse mutating writes."""
    monkeypatch.setattr("agents_core.slots.IS_MASTER", False)
    store = SlotStore(db_path=tmp_path / "off-master.db")
    with pytest.raises(OffMasterWriteError):
        store.create_slot("p", {"type": "fixer", "id": "a"})
    store.close()


def test_off_master_reads_still_work(tmp_path: Path, monkeypatch):
    """Reads (get, query, adjacent, stats) must work regardless of IS_MASTER."""
    # Seed on master first, then flip IS_MASTER.
    db = tmp_path / "slots.db"
    store = SlotStore(db_path=db)
    sid = store.create_slot("p", {"type": "fixer", "id": "a"})
    store.close()

    monkeypatch.setattr("agents_core.slots.IS_MASTER", False)
    store2 = SlotStore(db_path=db)
    assert store2.get(sid) is not None
    assert store2.query() != []
    assert store2.stats()["total_slots"] == 1
    store2.close()
