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

import hashlib
import multiprocessing
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents_core.notify import Priority
from agents_core.slots import (
    ARTIFACT_KINDS,
    NEXT_KINDS,
    SCHEMA,
    WITHHELD_AGE_DAYS,
    OffMasterWriteError,
    SlotNotFoundError,
    SlotOwnershipError,
    SlotStore,
    board_bucket,
    resolve_build_state,
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


# --- handoff / next --------------------------------------------------------

def test_set_next_happy_path(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(
        sid,
        by="agent-1",
        kind="review-pr",
        ref="pr-123",
        blocked_on=["slot-x"],
        proposal="await reviewer feedback",
        actuated=False,
    )
    slot = store.get(sid)
    assert slot["next"] == {
        "kind": "review-pr",
        "ref": "pr-123",
        "blocked_on": ["slot-x"],
        "proposal": "await reviewer feedback",
        "actuated": False,
    }
    assert slot["last_update"]  # updated

def test_set_next_defaults(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(sid, by="agent-1", kind="done")
    slot = store.get(sid)
    assert slot["next"] == {
        "kind": "done",
        "ref": None,
        "blocked_on": [],
        "proposal": "",
        "actuated": False,
    }

def test_set_next_blocked_on_dedups_and_sorts(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(
        sid,
        by="agent-1",
        kind="await-human",
        blocked_on=["c", "a", "b", "a"],  # duplicates, unsorted
    )
    slot = store.get(sid)
    assert slot["next"]["blocked_on"] == ["a", "b", "c"]  # deduplicated and sorted

def test_set_next_kind_validation(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    # Invalid kind raises ValueError
    with pytest.raises(ValueError):
        store.set_next(sid, by="agent-1", kind="invalid-kind")
    # Each valid kind is accepted
    for kind in NEXT_KINDS:
        store.set_next(sid, by="agent-1", kind=kind)
        assert store.get(sid)["next"]["kind"] == kind

def test_set_next_non_owner_rejected(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    with pytest.raises(SlotOwnershipError):
        store.set_next(sid, by="intruder", kind="done")
    # Unchanged
    assert store.get(sid)["next"] == {}

def test_set_next_off_master_raises(tmp_path: Path, monkeypatch):
    # Create slot when IS_MASTER is True (default fixture state)
    db = tmp_path / "off-master.db"
    store = SlotStore(db_path=db)
    sid = store.create_slot("p", {"type": "fixer", "id": "a"})
    store.close()
    # Now reopen with IS_MASTER False
    monkeypatch.setattr("agents_core.slots.IS_MASTER", False)
    store2 = SlotStore(db_path=db)
    with pytest.raises(OffMasterWriteError):
        store2.set_next(sid, by="a", kind="done")
    store2.close()

def test_migration_adds_next_column_to_legacy_db(tmp_path: Path):
    """A pre-existing DB without the next column must be upgraded by _migrate's
    ADD COLUMN on open, and the column must default to '{}'."""
    db = tmp_path / "legacy.db"
    # A schema without the next column (simulate pre-v0 state).
    legacy_schema = "\n".join(
        ln for ln in SCHEMA.splitlines() if "next" not in ln.lower()
    )
    assert "next" not in legacy_schema.lower()  # guard: next is truly absent
    conn = sqlite3.connect(db)
    conn.executescript(legacy_schema)
    # A row written before next existed. Set contributor_id so we can write to it later.
    conn.execute(
        "INSERT INTO slots (slot_id, project_id, contributor_id, status, last_update, created_at) "
        "VALUES ('s1', 'proj-A', 'legacy-agent', 'dispatched', ?, ?)",
        ("2026-06-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
    )
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slots)").fetchall()}
    conn.close()
    assert "next" not in cols  # confirm the ADD COLUMN path is untrodden

    # Opening via SlotStore runs _migrate -> the next column is added.
    store = SlotStore(db_path=db)
    migrated = {r[1] for r in store._conn.execute("PRAGMA table_info(slots)").fetchall()}
    assert "next" in migrated
    # The pre-existing row reads back with next defaulted to {}.
    assert store.get("s1")["next"] == {}
    # And set_next now works against the upgraded legacy table.
    store.set_next("s1", by="legacy-agent", kind="done")
    assert store.get("s1")["next"]["kind"] == "done"
    store.close()


# --- set_actuated (non-owner-guarded baton actuation) ----------------------

def test_set_actuated_marks_next_baton(store: SlotStore):
    """set_actuated on a slot with an un-acted next baton -> True; read shows actuated."""
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(
        sid,
        by="agent-1",
        kind="review-pr",
        ref="pr-123",
        actuated=False,
    )
    # Before actuating
    assert store.get(sid)["next"]["actuated"] is False
    # Morph actuates the baton (non-owner, no error)
    result = store.set_actuated(sid, by="morph")
    assert result is True
    # After actuating, the read projection shows actuated=True
    slot = store.get(sid)
    assert slot["next"]["actuated"] is True
    assert slot["next_actuated"] == 1
    assert slot["next_actuated_at"] is not None

def test_set_actuated_idempotent_second_call(store: SlotStore):
    """A second set_actuated on the same baton -> False (idempotent, no error)."""
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(sid, by="agent-1", kind="done")
    assert store.set_actuated(sid, by="morph") is True
    # Second call returns False (already actuated)
    assert store.set_actuated(sid, by="morph") is False
    # State unchanged
    assert store.get(sid)["next"]["actuated"] is True

def test_set_actuated_no_next_baton_returns_false(store: SlotStore):
    """set_actuated on a slot with no next baton -> False (idempotent)."""
    sid = store.create_slot("proj-A", CONTRIB)
    # No next baton set yet
    assert store.get(sid)["next"] == {}
    result = store.set_actuated(sid, by="morph")
    assert result is False
    # Still no next baton
    assert store.get(sid)["next"] == {}

def test_set_actuated_non_owner_succeeds(store: SlotStore):
    """set_actuated by non-owner (morph) succeeds — no SlotOwnershipError.

    This is the regression the unit exists to prevent: Morph must be able to
    mark a baton actuated despite not being the slot's contributor-of-record.
    """
    sid = store.create_slot("proj-A", {"type": "fixer", "id": "agent-1"})
    store.set_next(sid, by="agent-1", kind="done")
    # Morph (different contributor) actuates without error
    result = store.set_actuated(sid, by="morph")
    assert result is True
    assert store.get(sid)["next"]["actuated"] is True

def test_set_actuated_no_clobber_baton_fields(store: SlotStore):
    """After set_actuated, baton's kind/ref/blocked_on/proposal and status unchanged."""
    sid = store.create_slot("proj-A", CONTRIB)
    store.set_next(
        sid,
        by="agent-1",
        kind="review-pr",
        ref="pr-999",
        blocked_on=["slot-dep"],
        proposal="wait for reviewer",
        actuated=False,
    )
    store.update_status(sid, "in-progress", by="agent-1")
    # Snapshot before actuating
    before = store.get(sid)
    assert before["next"]["kind"] == "review-pr"
    assert before["next"]["ref"] == "pr-999"
    assert before["next"]["blocked_on"] == ["slot-dep"]
    assert before["next"]["proposal"] == "wait for reviewer"
    assert before["status"] == "in-progress"
    # Actuate
    store.set_actuated(sid, by="morph")
    # After actuating, contributor fields unchanged
    after = store.get(sid)
    assert after["next"]["kind"] == "review-pr"
    assert after["next"]["ref"] == "pr-999"
    assert after["next"]["blocked_on"] == ["slot-dep"]
    assert after["next"]["proposal"] == "wait for reviewer"
    assert after["status"] == "in-progress"
    # Only actuated flag changed
    assert after["next"]["actuated"] is True

def test_set_actuated_missing_slot_raises(store: SlotStore):
    """set_actuated on missing slot -> SlotNotFoundError."""
    with pytest.raises(SlotNotFoundError):
        store.set_actuated("nope", by="morph")

def test_set_next_resets_actuated_on_new_baton(store: SlotStore):
    """set_next publishing a NEW baton resets actuated to False (fresh baton un-acted)."""
    sid = store.create_slot("proj-A", CONTRIB)
    # First baton
    store.set_next(sid, by="agent-1", kind="review-pr", ref="pr-1")
    store.set_actuated(sid, by="morph")
    assert store.get(sid)["next"]["actuated"] is True
    # Contributor publishes a NEW baton — actuated resets to False
    store.set_next(sid, by="agent-1", kind="deploy", ref="v2", actuated=False)
    slot = store.get(sid)
    assert slot["next"]["kind"] == "deploy"
    assert slot["next"]["ref"] == "v2"
    assert slot["next"]["actuated"] is False  # Fresh baton is un-acted
    assert slot["next_actuated"] == 0
    assert slot["next_actuated_at"] is None

def test_set_actuated_read_only_mode_raises(tmp_path: Path, monkeypatch):
    """set_actuated in read-only mode raises OffMasterWriteError."""
    db = tmp_path / "off-master.db"
    store = SlotStore(db_path=db)
    sid = store.create_slot("p", {"type": "fixer", "id": "a"})
    store.set_next(sid, by="a", kind="done")
    store.close()
    # Reopen in read-only mode
    monkeypatch.setattr("agents_core.slots.IS_MASTER", False)
    store2 = SlotStore(db_path=db)
    with pytest.raises(OffMasterWriteError):
        store2.set_actuated(sid, by="morph")
    store2.close()


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
    assert counts == {"parked_expired": 1, "abandoned_expired": 1, "withheld_retired": 0}
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


# --- ETag conditional reads (AC1-AC4) ----------------------------------------

def test_read_version_etag_stable_across_calls(store: SlotStore):
    """AC1: ETag is stable across two identical requests with no intervening writes."""
    sid = store.create_slot("proj-A", CONTRIB)
    # Compute ETag twice, should be identical
    etag1 = store.read_version()
    etag2 = store.read_version()
    assert etag1 == etag2
    assert etag1.startswith('"') and etag1.endswith('"')


def test_read_version_changes_on_insert(store: SlotStore):
    """AC3: ETag changes after an insert."""
    etag_before = store.read_version()
    store.create_slot("proj-A", CONTRIB)
    etag_after = store.read_version()
    assert etag_before != etag_after


def test_read_version_changes_on_update(store: SlotStore):
    """AC3: ETag changes after an update (last_update bumped)."""
    sid = store.create_slot("proj-A", CONTRIB)
    etag_before = store.read_version()
    store.update_status(sid, "in-progress", by="agent-1")
    etag_after = store.read_version()
    assert etag_before != etag_after


def test_read_version_changes_on_delete(store: SlotStore):
    """AC3: ETag changes after a delete/expire."""
    store.create_slot("proj-A", CONTRIB, slot_id="s1")
    etag_before = store.read_version()
    # Expire the slot by forcing its status and age
    store.update_status("s1", "abandoned", by="agent-1")
    with store._lock:
        old_time = "2025-01-01T00:00:00+00:00"
        store._conn.execute("UPDATE slots SET last_update=? WHERE slot_id=?", (old_time, "s1"))
        store._conn.commit()
    store.expire()
    etag_after = store.read_version()
    assert etag_before != etag_after


def test_read_version_filter_project_id(store: SlotStore):
    """AC4: ETag is filter-correct — different filters produce different ETags."""
    store.create_slot("proj-A", CONTRIB, slot_id="s1")
    store.create_slot("proj-B", CONTRIB, slot_id="s2")
    etag_a = store.read_version(project_id="proj-A")
    etag_b = store.read_version(project_id="proj-B")
    etag_all = store.read_version()
    assert etag_a != etag_b != etag_all


def test_read_version_filter_status(store: SlotStore):
    """AC4: ETag differentiates by status filter."""
    sid = store.create_slot("proj-A", CONTRIB)
    etag_dispatched = store.read_version(status="dispatched")
    store.update_status(sid, "in-progress", by="agent-1")
    etag_after = store.read_version(status="dispatched")
    assert etag_dispatched != etag_after


def test_read_version_slot_id_specific(store: SlotStore):
    """AC4: ETag can be computed for a specific slot_id."""
    sid = store.create_slot("proj-A", CONTRIB)
    etag_slot = store.read_version(slot_id=sid)
    etag_all = store.read_version()
    assert etag_slot != etag_all
    # Update that slot, its ETag changes
    store.update_status(sid, "landed", by="agent-1")
    etag_slot_after = store.read_version(slot_id=sid)
    assert etag_slot != etag_slot_after


def test_read_version_filter_contributor_id(store: SlotStore):
    """AC4: ETag differentiates by contributor_id filter."""
    store.create_slot("proj-A", {"type": "fixer", "id": "agent-1"}, slot_id="s1")
    store.create_slot("proj-A", {"type": "fixer", "id": "agent-2"}, slot_id="s2")
    etag_a1 = store.read_version(contributor_id="agent-1")
    etag_a2 = store.read_version(contributor_id="agent-2")
    assert etag_a1 != etag_a2


# --- Adjacent cache: single-flight + TTL (AC5-AC6) ---------------------------

def test_adjacent_cache_single_flight_dedup(store: SlotStore, monkeypatch):
    """AC5: Concurrent identical adjacent() calls trigger exactly one full-table scan."""
    a = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(a, files=["app.py"], by="agent-1")

    call_count = [0]
    original_impl = store._adjacent_impl

    def counted_impl(*args, **kwargs):
        call_count[0] += 1
        return original_impl(*args, **kwargs)

    monkeypatch.setattr(store, "_adjacent_impl", counted_impl)

    # Two concurrent threads with the same key
    result1, result2 = [None], [None]

    def thread1():
        result1[0] = store.adjacent(files=["app.py"])

    def thread2():
        result2[0] = store.adjacent(files=["app.py"])

    t1 = threading.Thread(target=thread1)
    t2 = threading.Thread(target=thread2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both got the same result
    assert result1[0] == result2[0]
    assert len(result1[0]) == 1
    # But the implementation was called only once (single-flight dedup)
    assert call_count[0] == 1


def test_adjacent_cache_ttl_zero_disables_cache(store: SlotStore, tmp_path: Path, monkeypatch):
    """AC5: With TTL=0, every adjacent() call triggers a fresh scan."""
    monkeypatch.setenv("SLOT_ADJACENT_CACHE_TTL_SEC", "0")
    store2 = SlotStore(db_path=tmp_path / "slots2.db")
    a = store2.create_slot("proj-A", CONTRIB)
    store2.set_domain_touch(a, files=["app.py"], by="agent-1")

    call_count = [0]
    original_impl = store2._adjacent_impl

    def counted_impl(*args, **kwargs):
        call_count[0] += 1
        return original_impl(*args, **kwargs)

    monkeypatch.setattr(store2, "_adjacent_impl", counted_impl)

    # Call adjacent() twice with same params
    store2.adjacent(files=["app.py"])
    store2.adjacent(files=["app.py"])

    # With TTL=0, cache is disabled, so both calls hit the implementation
    assert call_count[0] == 2
    store2.close()


def test_adjacent_cache_ttl_respects_freshness(store: SlotStore, tmp_path: Path, monkeypatch):
    """AC5: With TTL>0, old cache entries are refreshed on the next call."""
    monkeypatch.setenv("SLOT_ADJACENT_CACHE_TTL_SEC", "0.1")  # 100ms TTL
    store2 = SlotStore(db_path=tmp_path / "slots3.db")
    a = store2.create_slot("proj-A", CONTRIB)
    store2.set_domain_touch(a, files=["app.py"], by="agent-1")

    call_count = [0]
    original_impl = store2._adjacent_impl

    def counted_impl(*args, **kwargs):
        call_count[0] += 1
        return original_impl(*args, **kwargs)

    monkeypatch.setattr(store2, "_adjacent_impl", counted_impl)

    # First call: cache miss, call the impl
    store2.adjacent(files=["app.py"])
    assert call_count[0] == 1

    # Second call immediately after: cache hit
    store2.adjacent(files=["app.py"])
    assert call_count[0] == 1

    # Wait for TTL to expire
    import time
    time.sleep(0.15)

    # Third call after TTL: cache miss again, call the impl
    store2.adjacent(files=["app.py"])
    assert call_count[0] == 2

    store2.close()


def test_adjacent_cache_serve_stale_on_error(store: SlotStore, tmp_path: Path, monkeypatch):
    """AC6: With a warm cache, OperationalError returns the last-good result."""
    a = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(a, files=["app.py"], by="agent-1")

    # Warm the cache
    result = store.adjacent(files=["app.py"])
    assert len(result) == 1

    # Inject an error on the next compute
    original_impl = store._adjacent_impl
    def error_impl(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_adjacent_impl", error_impl)

    # Call should not raise, but return the cached result
    result_stale = store.adjacent(files=["app.py"])
    assert result_stale == result


def test_adjacent_cache_serve_stale_no_cache_raises(store: SlotStore, monkeypatch):
    """AC6: With a cold cache, OperationalError propagates."""
    a = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(a, files=["app.py"], by="agent-1")

    # Inject an error WITHOUT warming the cache first
    original_impl = store._adjacent_impl
    def error_impl(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_adjacent_impl", error_impl)

    # Call should raise since there's no cached result
    with pytest.raises(sqlite3.OperationalError):
        store.adjacent(files=["app.py"])


def test_adjacent_cache_multi_waiter_error(store: SlotStore, monkeypatch):
    """Multi-waiter error race fix: all waiters must see the error, not just the first.

    When compute fails with a cold cache, multiple concurrent waiters should all
    re-raise the error, not silently return [] for later waiters.
    """
    a = store.create_slot("proj-A", CONTRIB)
    store.set_domain_touch(a, files=["app.py"], by="agent-1")

    # Inject an error WITHOUT warming the cache first
    def error_impl(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_adjacent_impl", error_impl)

    # Multiple concurrent threads with same params
    errors_caught = []
    results = [None] * 3

    def thread_worker(idx: int):
        try:
            results[idx] = store.adjacent(files=["app.py"])
            errors_caught.append(None)  # No error raised
        except sqlite3.OperationalError as e:
            errors_caught.append(str(e))

    threads = [threading.Thread(target=thread_worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should have caught the error, not returned [] or cached result
    assert len(errors_caught) == 3
    # All three should have caught the OperationalError
    assert all("database is locked" in e for e in errors_caught if e is not None)
    # None should have silently returned a result
    assert all(r is None for r in results)


# --- baton-lineage-verified-handoff-v0 --------------------------------------
# lineage / artifact columns, content-hash verification, withheld status,
# new-slot-per-hop handoff, escalate-to-flame wiring, withheld fate, and the
# Board read-view precedence helpers.

def test_migration_adds_lineage_and_artifact_columns_to_legacy_db(tmp_path: Path):
    """AC1: a pre-existing DB without lineage/artifact must be upgraded by
    _migrate's ADD COLUMN on open, existing rows unaffected."""
    db = tmp_path / "legacy.db"
    legacy_schema = "\n".join(
        ln for ln in SCHEMA.splitlines()
        if "lineage" not in ln.lower() and "artifact" not in ln.lower()
    )
    assert "lineage" not in legacy_schema.lower()
    assert "artifact" not in legacy_schema.lower()
    conn = sqlite3.connect(db)
    conn.executescript(legacy_schema)
    conn.execute(
        "INSERT INTO slots (slot_id, project_id, contributor_id, status, last_update, created_at) "
        "VALUES ('s1', 'proj-A', 'legacy-agent', 'dispatched', ?, ?)",
        ("2026-06-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
    )
    conn.commit()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slots)").fetchall()}
    conn.close()
    assert "lineage" not in cols and "artifact" not in cols

    store = SlotStore(db_path=db)
    migrated = {r[1] for r in store._conn.execute("PRAGMA table_info(slots)").fetchall()}
    assert {"lineage", "artifact"} <= migrated
    # Pre-existing row survives, no loss.
    slot = store.get("s1")
    assert slot["project_id"] == "proj-A"
    assert slot["lineage"] == {}  # legacy row: schema DEFAULT '{}', not normalized
    store.close()


def test_create_slot_normalizes_lineage_origin_intent_null_by_default(store: SlotStore):
    """AC2: freshly authored molecule with no lineage passed -> origin_intent is
    None, never a fabricated string."""
    sid = store.create_slot("proj-A", CONTRIB)
    lineage = store.get(sid)["lineage"]
    assert lineage["origin_intent"] is None
    assert lineage["authors"] == []
    assert lineage["atoms"] == []


def test_create_slot_accepts_real_origin_intent_pointer(store: SlotStore):
    """AC2: a real pointer to the human utterance is preserved verbatim."""
    origin = {"author": "Erah", "intent": "split retrieval", "source_pointer": "spec.md:12"}
    sid = store.create_slot("proj-A", CONTRIB, lineage={"origin_intent": origin})
    assert store.get(sid)["lineage"]["origin_intent"] == origin


def test_set_down_computes_whole_body_sha256_for_draft_file(store: SlotStore, tmp_path: Path):
    sid = store.create_slot("proj-A", CONTRIB)
    f = tmp_path / "draft.md"
    f.write_text("hello world")
    store.set_down(
        sid, {"kind": "draft-file", "ref": str(f)}, by="agent-1", next_kind="review-pr",
    )
    slot = store.get(sid)
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert slot["artifact"] == {"kind": "draft-file", "ref": str(f), "content_hash": expected}
    assert slot["next"]["kind"] == "review-pr"  # wraps set_next unchanged
    assert slot["lineage"]["authors"][-1]["id"] == "agent-1"
    assert slot["lineage"]["authors"][-1]["role"] == "set-down"


def test_set_down_computes_head_sha_for_pr(store: SlotStore, monkeypatch):
    sid = store.create_slot("proj-A", CONTRIB)
    monkeypatch.setattr(
        "agents_core.forgejo.get_pr",
        lambda repo, number, owner=None: {"head": {"sha": "deadbeef"}},
    )
    store.set_down(sid, {"kind": "pr", "ref": "agents-core#141"}, by="agent-1", next_kind="done")
    assert store.get(sid)["artifact"]["content_hash"] == "deadbeef"


def test_set_down_invalid_artifact_kind_rejected(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    with pytest.raises(ValueError):
        store.set_down(sid, {"kind": "bogus", "ref": "x"}, by="agent-1", next_kind="done")


# --- pick_up: Reality Snap success -> new-slot-per-hop handoff -------------

def test_pick_up_success_mints_new_slot_and_chains_lineage(store: SlotStore, tmp_path: Path):
    """AC6 (single hop): successful pick_up mints a new slot with its own owner,
    predecessor_slot_id set, origin_intent carried forward, author appended."""
    origin = {"author": "Erah", "intent": "ship the baton spec", "source_pointer": "spec:1"}
    old = store.create_slot(
        "proj-A", CONTRIB, horizon={"immediate_goal": "land the spec"},
        lineage={"origin_intent": origin},
    )
    f = tmp_path / "draft.md"
    f.write_text("v1 content")
    store.set_down(old, {"kind": "draft-file", "ref": str(f)}, by="agent-1", next_kind="bind-next")

    new_sid = store.pick_up(old, contributor={"type": "fixer", "id": "agent-2"})
    assert new_sid is not None and new_sid != old

    old_slot = store.get(old)
    assert old_slot["status"] == "landed"
    assert old_slot["next"]["actuated"] is True  # set_actuated acknowledged the baton

    new_slot = store.get(new_sid)
    assert new_slot["contributor_id"] == "agent-2"  # new owner, ownership never transferred
    assert new_slot["horizon"]["immediate_goal"] == "land the spec"  # horizon carried forward
    assert new_slot["artifact"]["content_hash"] == old_slot["artifact"]["content_hash"]
    assert new_slot["lineage"]["origin_intent"] == origin  # never dropped/invented
    assert new_slot["lineage"]["predecessor_slot_id"] == old
    author_ids = [a["id"] for a in new_slot["lineage"]["authors"]]
    assert author_ids == ["agent-1", "agent-2"]  # appended, prior author not dropped
    # The new owner can write to its own slot (no SlotOwnershipError).
    store.update_status(new_sid, "in-progress", by="agent-2")


def test_pick_up_three_hop_chain(store: SlotStore, tmp_path: Path):
    """AC6: after a 3-hop machine-to-machine handoff there are 3 linked slots,
    each hop's contributor owns its own slot, origin_intent + full author chain
    queryable from the head slot."""
    origin = {"author": "Erah", "intent": "three-hop relay", "source_pointer": "spec:2"}
    f = tmp_path / "molecule.txt"
    f.write_text("hop-0")

    s0 = store.create_slot("proj-A", {"type": "fixer", "id": "hop-0"}, lineage={"origin_intent": origin})
    store.set_down(s0, {"kind": "draft-file", "ref": str(f)}, by="hop-0", next_kind="bind-next")
    s1 = store.pick_up(s0, contributor={"type": "fixer", "id": "hop-1"})
    assert s1 is not None

    store.set_down(s1, {"kind": "draft-file", "ref": str(f)}, by="hop-1", next_kind="bind-next")
    s2 = store.pick_up(s1, contributor={"type": "fixer", "id": "hop-2"})
    assert s2 is not None

    store.set_down(s2, {"kind": "draft-file", "ref": str(f)}, by="hop-2", next_kind="done")
    s3 = store.pick_up(s2, contributor={"type": "fixer", "id": "hop-3"})
    assert s3 is not None

    # 3 distinct new slots minted (s1, s2, s3), each chained to its predecessor.
    assert len({s0, s1, s2, s3}) == 4
    assert store.get(s1)["lineage"]["predecessor_slot_id"] == s0
    assert store.get(s2)["lineage"]["predecessor_slot_id"] == s1
    assert store.get(s3)["lineage"]["predecessor_slot_id"] == s2

    head = store.get(s3)
    assert head["lineage"]["origin_intent"] == origin  # null-or-real, never invented, never dropped
    author_ids = [a["id"] for a in head["lineage"]["authors"]]
    # set_down appends the sender, pick_up appends the receiver, at every hop —
    # no earlier author dropped.
    assert author_ids == ["hop-0", "hop-1", "hop-1", "hop-2", "hop-2", "hop-3"]

    # Each hop's contributor owns only its own slot.
    for sid, owner in ((s0, "hop-0"), (s1, "hop-1"), (s2, "hop-2"), (s3, "hop-3")):
        assert store.get(sid)["contributor_id"] == owner
    # Prior-hop slots are all landed (handed off), never mutated by later owners.
    assert store.get(s0)["status"] == "landed"
    assert store.get(s1)["status"] == "landed"
    assert store.get(s2)["status"] == "landed"


# --- pick_up: Reality Snap failure -> withheld + active flame escalation ---

def test_pick_up_hash_mismatch_withholds_and_escalates(store: SlotStore, tmp_path: Path, monkeypatch):
    """AC3/AC4: a force-pushed/edited artifact fails the Snap; pick_up withholds
    the OLD slot, fires a flame notification, and does NOT resume (returns None)."""
    f = tmp_path / "draft.md"
    f.write_text("original")
    old = store.create_slot("proj-A", CONTRIB)
    store.set_down(old, {"kind": "draft-file", "ref": str(f)}, by="agent-1", next_kind="bind-next")

    # Simulate a silent edit after set_down.
    f.write_text("tampered")

    sent = []
    monkeypatch.setattr(
        "agents_core.slots.send_notification",
        lambda message, **kw: sent.append((message, kw)) or True,
    )

    result = store.pick_up(old, contributor={"type": "fixer", "id": "agent-2"})
    assert result is None  # never resumes

    slot = store.get(old)
    assert slot["status"] == "withheld"  # NOT clobbered to "escalated"
    assert slot["escalation"]["to"] == "flame"
    assert "content_hash mismatch" in slot["escalation"]["reason"]
    assert any(cp["kind"] == "reality-snap" and "FAILED" in cp["note"] for cp in slot["checkpoints"])

    # Notification fired on the flame channel.
    assert len(sent) == 1
    message, kwargs = sent[0]
    assert "withheld" in message.lower()
    assert kwargs["priority"] == Priority.HIGH

    # Old slot stays owned by its current contributor-of-record (no handoff).
    assert slot["contributor_id"] == "agent-1"


def test_pick_up_missing_artifact_withholds(store: SlotStore, monkeypatch):
    """No artifact on record at all -> Snap fails, withheld (not a crash)."""
    old = store.create_slot("proj-A", CONTRIB)
    monkeypatch.setattr("agents_core.slots.send_notification", lambda *a, **k: True)
    result = store.pick_up(old, contributor={"type": "fixer", "id": "agent-2"})
    assert result is None
    assert store.get(old)["status"] == "withheld"


def test_pick_up_join_check_failure_withholds(store: SlotStore, tmp_path: Path, monkeypatch):
    """Hash matches but the live-join leg fails -> still withheld."""
    f = tmp_path / "draft.md"
    f.write_text("stable content")
    old = store.create_slot("proj-A", CONTRIB)
    store.set_down(old, {"kind": "draft-file", "ref": str(f)}, by="agent-1", next_kind="bind-next")
    monkeypatch.setattr("agents_core.slots.send_notification", lambda *a, **k: True)

    result = store.pick_up(old, contributor={"type": "fixer", "id": "agent-2"}, join_check=lambda: False)
    assert result is None
    slot = store.get(old)
    assert slot["status"] == "withheld"
    assert "live-join" in slot["escalation"]["reason"]


def test_pick_up_withheld_leaves_other_slots_untouched(store: SlotStore, monkeypatch):
    """A withheld pick_up on one slot must not affect unrelated slots."""
    other = store.create_slot("proj-B", {"type": "fixer", "id": "bystander"})
    old = store.create_slot("proj-A", CONTRIB)
    monkeypatch.setattr("agents_core.slots.send_notification", lambda *a, **k: True)
    store.pick_up(old, contributor={"type": "fixer", "id": "agent-2"})
    assert store.get(other)["status"] == "dispatched"  # unrelated slot untouched


def test_pick_up_missing_slot_raises(store: SlotStore):
    with pytest.raises(SlotNotFoundError):
        store.pick_up("nope", contributor={"type": "fixer", "id": "a"})


# --- escalate() preserve_status -----------------------------------------

def test_escalate_preserve_status_does_not_clobber(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.update_status(sid, "withheld", by="agent-1")
    store.escalate(sid, to="flame", reason="test", by="agent-1", preserve_status=True)
    slot = store.get(sid)
    assert slot["status"] == "withheld"  # unchanged
    assert slot["escalation"] == {"to": "flame", "reason": "test"}


def test_escalate_default_still_forces_escalated(store: SlotStore):
    """Backward-compat: default behavior (preserve_status=False) is unchanged."""
    sid = store.create_slot("proj-A", CONTRIB)
    store.escalate(sid, to="facets", reason="x", by="agent-1")
    assert store.get(sid)["status"] == "escalated"


# --- withheld fate: expire() retires withheld -> abandoned, scar preserved -

def test_expire_retires_withheld_to_abandoned_preserving_lineage(store: SlotStore):
    """AC5: an unanswered withheld slot past threshold retires to abandoned via
    the extended expire(); it's a status transition, not a delete — checkpoints
    and lineage remain queryable (scar preserved)."""
    origin = {"author": "Erah", "intent": "x", "source_pointer": "spec:3"}
    sid = store.create_slot("proj-A", CONTRIB, lineage={"origin_intent": origin})
    store.update_status(sid, "withheld", by="agent-1")
    store.append_checkpoint(sid, "reality-snap", "pick_up FAILED: hash mismatch", by="agent-1")

    old = (datetime.now(timezone.utc) - timedelta(days=WITHHELD_AGE_DAYS + 1)).isoformat()
    with store._lock:
        store._conn.execute("UPDATE slots SET last_update=? WHERE slot_id=?", (old, sid))
        store._conn.commit()

    counts = store.expire()
    assert counts["withheld_retired"] == 1

    slot = store.get(sid)
    assert slot is not None  # retired, NOT deleted
    assert slot["status"] == "abandoned"
    assert slot["lineage"]["origin_intent"] == origin  # scar preserved
    assert any("FAILED" in cp["note"] for cp in slot["checkpoints"])  # failure legible


def test_expire_withheld_within_threshold_survives(store: SlotStore):
    sid = store.create_slot("proj-A", CONTRIB)
    store.update_status(sid, "withheld", by="agent-1")
    counts = store.expire()
    assert counts["withheld_retired"] == 0
    assert store.get(sid)["status"] == "withheld"


# --- Board read-view precedence (AC7) ---------------------------------------

def test_board_bucket_mapping():
    assert board_bucket("landed") == "built"
    for s in ("dispatched", "in-progress", "escalated", "awaiting-input", "withheld"):
        assert board_bucket(s) == "in-flight"
    assert board_bucket("parked") == "parked"  # passthrough, not remapped


def test_resolve_build_state_artifact_wins_over_lagging_status():
    """AC7: a merged PR (artifact present) reports built even though the slot's
    own status lags behind — the bakeoff mis-scrape case."""
    artifact = {"kind": "pr", "ref": "agents-core#1", "content_hash": "abc"}
    assert resolve_build_state(artifact, "in-progress") == "built"


def test_resolve_build_state_unmerged_pr_falls_through_to_status():
    artifact = {"kind": "pr", "ref": "agents-core#1", "content_hash": "abc"}
    assert resolve_build_state(artifact, "in-progress", pr_merged=False) == "in-flight"


def test_resolve_build_state_no_artifact_uses_status_bucket():
    assert resolve_build_state(None, "landed") == "built"
    assert resolve_build_state({}, "dispatched") == "in-flight"


def test_artifact_kinds_frozenset_matches_spec():
    assert ARTIFACT_KINDS == {"pr", "spec", "atom-output", "draft-file"}
