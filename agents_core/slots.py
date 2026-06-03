"""SlotStore — SQLite-backed project-slot blackboard (bMAS coordination layer).

The blackboard from `Lapis/Architecture-Project-Slots-As-Blackboard.md` (2026-05-28):
the shared, single-source-of-truth store that holds live operational state across all
active Lapis work, mapped onto the Zephyr protocol's project / slot / contributor model.

- **Project** = a target (existing `/srv/lapis/targets/<tid>.yaml`, owned by `targets.TargetStore`).
- **Slot**    = a unit of work within a project (a dispatch, a review cycle, a sub-task).
- **Contributor** = the agent (or human) currently assigned to a slot.

This is the storage half (bootstrap step 1). Self-report wiring (step 2) and the
Layer-5 read API (step 3) consume it. Steps 4-5 (Facets, Composer) need GravityWell.

Ownership (decision/brix-coordinates-gw-opines-ownership-2026-06-02): BRIX owns the
canonical store, single-writer. The **contributor-of-record** writes its own slot's
state; **observers** (Weaver) write only to the separate ``weaver_*`` columns. This is
the concurrency discipline from the design doc's open-questions section.

Usage:
    from agents_core.slots import SlotStore
    store = SlotStore()
    sid = store.create_slot(
        project_id="synapse-retrieval-scope-split-v0",
        contributor={"type": "fixer", "id": "claude_1717360000_gpu0"},
        horizon={"project_summary": "...", "immediate_goal": "split episodic/semantic"},
    )
    store.append_checkpoint(sid, kind="self-report", note="touched app.py", by="claude_1717360000_gpu0")
    store.update_status(sid, "landed", by="claude_1717360000_gpu0")
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --- Configuration ---

DB_DIR = Path("/data/slots")
DB_PATH = Path(os.environ.get("SLOTS_DB_PATH", DB_DIR / "slots.db"))
# socket.gethostname() is cross-platform; os.uname() is Unix-only.
HOSTNAME = socket.gethostname()
# Master (read-write) host for the slot substrate. Single deliberate value, NOT an env
# toggle — mirrors mem.py's MEM_MASTER_HOST so the designation cannot drift per-host into
# a dual-master split-brain. The blackboard is the single source of truth; BRIX owns it.
SLOTS_MASTER_HOST = "brix"
IS_MASTER = HOSTNAME == SLOTS_MASTER_HOST
# HTTP base URL of the slot master's slot-server. Off-master writers POST here via the
# slot client instead of writing a divergent local sqlite. Env-overridable.
SLOTS_MASTER_URL = os.environ.get("SLOTS_MASTER_URL", "http://203.0.113.10:8405")

# Slot lifecycle vocabulary (design doc "Proposed minimum slot state schema").
STATUSES = frozenset({
    "dispatched", "in-progress", "awaiting-input",
    "escalated", "parked", "landed", "abandoned",
})
# Slots no longer actively contributing — excluded from adjacency by default.
TERMINAL_STATUSES = frozenset({"landed", "abandoned"})
INACTIVE_STATUSES = frozenset({"landed", "abandoned", "parked"})

# Contributor types (design doc). Not enforced — permissive — but documented.
CONTRIBUTOR_TYPES = frozenset({
    "fixer", "reviewer", "council", "facets", "expert", "human", "gardener",
})

# Expiration defaults (design doc open-questions): parked age out after 30d,
# abandoned after 7d. Tunable.
PARKED_AGE_DAYS = 30
ABANDONED_AGE_DAYS = 7


# --- Database Schema ---

SCHEMA = """\
CREATE TABLE IF NOT EXISTS slots (
    slot_id            TEXT PRIMARY KEY,
    project_id         TEXT NOT NULL,
    contributor_type   TEXT,
    contributor_id     TEXT,
    started_at         TEXT,
    status             TEXT NOT NULL,
    last_update        TEXT NOT NULL,
    domain_touch       TEXT NOT NULL DEFAULT '{}',
    horizon            TEXT NOT NULL DEFAULT '{}',
    checkpoints        TEXT NOT NULL DEFAULT '[]',
    escalation         TEXT NOT NULL DEFAULT '{}',
    -- observer (Weaver-derived) fields: written by observers, NEVER by the
    -- contributor-of-record. Kept in a separate column namespace so the two
    -- signals never collide (design doc: "observers write to separate fields").
    weaver_status      TEXT,
    weaver_last_update TEXT,
    -- ratification (Facets-derived) fields: the opinion-authority namespace
    -- (blackboard bootstrap step 4). Facets reads an escalated slot, runs the
    -- three-voice persona consult, and writes its ratification verdict HERE —
    -- never to contributor or weaver columns. Same separate-namespace discipline:
    -- the store records the verdict, the contributor decides whether to act on it.
    facets_verdict      TEXT,
    facets_last_update  TEXT,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS slots_project ON slots(project_id);
CREATE INDEX IF NOT EXISTS slots_status  ON slots(status);
CREATE INDEX IF NOT EXISTS slots_contrib ON slots(contributor_id);
"""

# JSON-encoded columns, parsed back to objects on read.
_JSON_FIELDS = ("domain_touch", "horizon", "checkpoints", "escalation", "facets_verdict")

# Observer-namespace columns added after the initial gate-5 schema. CREATE TABLE
# IF NOT EXISTS won't add columns to a pre-existing table, so they are applied as
# idempotent ADD COLUMN migrations against live DBs (see SlotStore._migrate).
_ADDED_COLUMNS = (
    ("facets_verdict", "TEXT"),
    ("facets_last_update", "TEXT"),
)


class SlotOwnershipError(PermissionError):
    """Raised when a writer that is not the contributor-of-record attempts a
    contributor write (single-writer discipline)."""


class SlotNotFoundError(KeyError):
    """Raised when a write targets a slot_id that does not exist."""


class OffMasterWriteError(RuntimeError):
    """Raised when a non-master node attempts a local mutating write to SlotStore.

    Off-master writers must POST to SLOTS_MASTER_URL instead of writing a
    divergent local sqlite — the docstring contract enforced loudly.
    """


class SlotStore:
    """SQLite-backed project-slot blackboard. Thread-safe; one connection + lock."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Let cross-process writers wait up to 5 s for a WAL write lock instead
        # of raising OperationalError("database is locked") immediately. The
        # in-process RLock below serializes threads within one process; this
        # covers a separate shaper writer + slot-server concurrently on-disk.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._migrate()
        # Serializes concurrent access from the FastAPI threadpool. RLock because
        # escalate() reuses update_status() internally.
        self._lock = threading.RLock()

    def _migrate(self):
        """Apply idempotent ADD COLUMN migrations for columns introduced after the
        initial gate-5 schema. Safe to run on every open: ALTER TABLE ADD COLUMN
        on an already-present column is a no-op here because we check pragma first."""
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(slots)").fetchall()
        }
        for name, decl in _ADDED_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE slots ADD COLUMN {name} {decl}")
        self._conn.commit()

    def close(self):
        self._conn.close()

    # -- create -------------------------------------------------------------

    def create_slot(
        self,
        project_id: str,
        contributor: dict,
        horizon: dict | None = None,
        *,
        slot_id: str | None = None,
        status: str = "dispatched",
        domain_touch: dict | None = None,
    ) -> str:
        """Create a slot for ``project_id`` assigned to ``contributor``.

        ``contributor`` is ``{"type": <contributor-type>, "id": <agent run/session id>}``.
        ``horizon`` is the vision-propagation field (``project_summary``,
        ``immediate_goal``, ``adjacent_slots``). Returns the slot_id.
        """
        self._check_writable()
        self._validate_status(status)
        sid = slot_id or uuid.uuid4().hex[:12]
        now = _now()
        ctype = contributor.get("type")
        cid = contributor.get("id")
        started = contributor.get("started_at") or now
        horizon = horizon or {}
        domain_touch = domain_touch or {}
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM slots WHERE slot_id = ?", (sid,)
            ).fetchone()
            if exists:
                raise ValueError(f"slot_id already exists: {sid}")
            self._conn.execute(
                "INSERT INTO slots (slot_id, project_id, contributor_type, contributor_id, "
                " started_at, status, last_update, domain_touch, horizon, checkpoints, "
                " escalation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid, project_id, ctype, cid, started, status, now,
                    json.dumps(domain_touch, sort_keys=True),
                    json.dumps(horizon, sort_keys=True),
                    "[]", "{}", now,
                ),
            )
            self._conn.commit()
        return sid

    # -- contributor-of-record writes (single-writer guarded) ---------------

    def update_status(self, slot_id: str, status: str, *, by: str) -> bool:
        """Transition a slot's status. Only the contributor-of-record may write.

        Returns True. Raises SlotNotFoundError / SlotOwnershipError / ValueError.
        """
        self._check_writable()
        self._validate_status(status)
        now = _now()
        with self._lock:
            self._require_owner(slot_id, by)
            self._conn.execute(
                "UPDATE slots SET status=?, last_update=? WHERE slot_id=?",
                (status, now, slot_id),
            )
            self._conn.commit()
        return True

    def append_checkpoint(self, slot_id: str, kind: str, note: str, *, by: str) -> bool:
        """Append a Reality-Snap / self-report / external-event checkpoint.

        ``kind`` is one of reality-snap | self-report | external-event (free string).
        Only the contributor-of-record may write.
        """
        self._check_writable()
        now = _now()
        entry = {"at": now, "kind": kind, "note": note}
        with self._lock:
            row = self._require_owner(slot_id, by)
            checkpoints = json.loads(row["checkpoints"])
            checkpoints.append(entry)
            self._conn.execute(
                "UPDATE slots SET checkpoints=?, last_update=? WHERE slot_id=?",
                (json.dumps(checkpoints), now, slot_id),
            )
            self._conn.commit()
        return True

    def set_domain_touch(
        self,
        slot_id: str,
        files: list[str] | None = None,
        mem_keys: list[str] | None = None,
        scopes: list[str] | None = None,
        *,
        by: str,
    ) -> bool:
        """Publish what files / mem-keys / scopes this slot is touching — the field
        that makes cross-slot proximity detection possible. Contributor-of-record only.
        """
        self._check_writable()
        now = _now()
        dt = {
            "files": sorted(set(files or [])),
            "mem_keys": sorted(set(mem_keys or [])),
            "scopes": sorted(set(scopes or [])),
        }
        with self._lock:
            self._require_owner(slot_id, by)
            self._conn.execute(
                "UPDATE slots SET domain_touch=?, last_update=? WHERE slot_id=?",
                (json.dumps(dt, sort_keys=True), now, slot_id),
            )
            self._conn.commit()
        return True

    def update_horizon(self, slot_id: str, *, by: str, **fields) -> bool:
        """Merge fields into the slot's horizon (e.g. adjacent_slots). Owner only."""
        self._check_writable()
        now = _now()
        with self._lock:
            row = self._require_owner(slot_id, by)
            horizon = json.loads(row["horizon"])
            horizon.update(fields)
            self._conn.execute(
                "UPDATE slots SET horizon=?, last_update=? WHERE slot_id=?",
                (json.dumps(horizon, sort_keys=True), now, slot_id),
            )
            self._conn.commit()
        return True

    def escalate(self, slot_id: str, to: str, reason: str, *, by: str) -> bool:
        """Escalate the slot (status -> escalated, populate escalation field).

        ``to`` is facets | flame | none. Contributor-of-record only.
        """
        self._check_writable()
        now = _now()
        with self._lock:
            self._require_owner(slot_id, by)
            self._conn.execute(
                "UPDATE slots SET status=?, escalation=?, last_update=? WHERE slot_id=?",
                ("escalated", json.dumps({"to": to, "reason": reason}, sort_keys=True),
                 now, slot_id),
            )
            self._conn.commit()
        return True

    # -- observer writes (separate column namespace; not owner-guarded) ------

    def observer_update(self, slot_id: str, weaver_status: str, *, by: str = "weaver") -> bool:
        """Write the Weaver-derived corroboration signal. Observers write ONLY here,
        never to contributor fields — so the two signals never collide. When this
        diverges from the contributor's ``status``, the divergence is itself a percept
        (surfaced to Facets later); this store records both, it does not resolve them.
        """
        self._check_writable()
        now = _now()
        with self._lock:
            if not self._conn.execute(
                "SELECT 1 FROM slots WHERE slot_id=?", (slot_id,)
            ).fetchone():
                raise SlotNotFoundError(slot_id)
            self._conn.execute(
                "UPDATE slots SET weaver_status=?, weaver_last_update=? WHERE slot_id=?",
                (weaver_status, now, slot_id),
            )
            self._conn.commit()
        return True

    def facets_ratify(self, slot_id: str, verdict: dict, *, by: str = "facets") -> bool:
        """Write a Facets ratification verdict (blackboard bootstrap step 4).

        Facets is the opinion-authority observer: it reads a slot escalated ``to:
        facets``, runs the three-voice persona consult, and records its verdict in
        the separate ``facets_*`` namespace — like the Weaver's ``observer_update``,
        it NEVER writes contributor-of-record or weaver fields. The store records
        the verdict; the contributor-of-record decides whether to act on it. This
        write does not change ``status`` — clearing the escalation (escalated ->
        in-progress/landed/abandoned) stays the contributor's owner-guarded call.

        ``verdict`` is the consult outcome, e.g. ``{"deliberation_id", "council_status",
        "recommendation", "confidence", ...}``. Raises SlotNotFoundError.
        """
        self._check_writable()
        now = _now()
        payload = dict(verdict)
        payload.setdefault("ratified_at", now)
        payload.setdefault("by", by)
        with self._lock:
            if not self._conn.execute(
                "SELECT 1 FROM slots WHERE slot_id=?", (slot_id,)
            ).fetchone():
                raise SlotNotFoundError(slot_id)
            self._conn.execute(
                "UPDATE slots SET facets_verdict=?, facets_last_update=? WHERE slot_id=?",
                (json.dumps(payload, sort_keys=True), now, slot_id),
            )
            self._conn.commit()
        return True

    # -- reads --------------------------------------------------------------

    def get(self, slot_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM slots WHERE slot_id=?", (slot_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def query(
        self,
        project_id: str | None = None,
        status: str | None = None,
        contributor_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query the blackboard by project / status / contributor."""
        sql = "SELECT * FROM slots WHERE 1=1 "
        params: list = []
        if project_id:
            sql += "AND project_id=? "
            params.append(project_id)
        if status:
            sql += "AND status=? "
            params.append(status)
        if contributor_id:
            sql += "AND contributor_id=? "
            params.append(contributor_id)
        sql += "ORDER BY last_update DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def adjacent(
        self,
        files: list[str] | None = None,
        mem_keys: list[str] | None = None,
        *,
        exclude_slot_id: str | None = None,
        include_inactive: bool = False,
    ) -> list[dict]:
        """Return active slots whose domain_touch overlaps the given files / mem-keys.

        This is the cross-slot proximity primitive: an agent at a Reality-Snap
        checkpoint calls this to detect "another slot is touching what I'm about to."
        Proximity is **file + mem-key overlap only** (design doc default; scope-overlap
        detection deferred until a scope taxonomy exists).
        """
        want_files = set(files or [])
        want_keys = set(mem_keys or [])
        if not want_files and not want_keys:
            return []
        out: list[dict] = []
        with self._lock:
            rows = self._conn.execute("SELECT * FROM slots").fetchall()
        for row in rows:
            d = self._row_to_dict(row)
            if d["slot_id"] == exclude_slot_id:
                continue
            if not include_inactive and d["status"] in INACTIVE_STATUSES:
                continue
            dt = d["domain_touch"] or {}
            overlap_files = want_files & set(dt.get("files", []))
            overlap_keys = want_keys & set(dt.get("mem_keys", []))
            if overlap_files or overlap_keys:
                d["_overlap"] = {
                    "files": sorted(overlap_files),
                    "mem_keys": sorted(overlap_keys),
                }
                out.append(d)
        return out

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS n FROM slots").fetchone()["n"]
            by_status = {
                r["status"]: r["n"]
                for r in self._conn.execute(
                    "SELECT status, COUNT(*) AS n FROM slots GROUP BY status"
                ).fetchall()
            }
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_slots": total,
            "by_status": by_status,
            "db_size_bytes": size_bytes,
            "db_path": str(self.db_path),
            "hostname": HOSTNAME,
            "mode": "read-write" if IS_MASTER else f"read-only (master is {SLOTS_MASTER_HOST})",
        }

    # -- maintenance --------------------------------------------------------

    def expire(self, now: datetime | None = None) -> dict:
        """Age out stale slots: parked > 30d, abandoned > 7d (design doc defaults).

        Blackboards accumulate cruft fast — cleanup discipline matters. Returns counts.
        """
        self._check_writable()
        now = now or datetime.now(timezone.utc)
        parked_cut = (now - timedelta(days=PARKED_AGE_DAYS)).isoformat()
        abandoned_cut = (now - timedelta(days=ABANDONED_AGE_DAYS)).isoformat()
        with self._lock:
            p = self._conn.execute(
                "DELETE FROM slots WHERE status='parked' AND last_update < ?",
                (parked_cut,),
            ).rowcount
            a = self._conn.execute(
                "DELETE FROM slots WHERE status='abandoned' AND last_update < ?",
                (abandoned_cut,),
            ).rowcount
            self._conn.commit()
        return {"parked_expired": p, "abandoned_expired": a}

    def checkpoint_wal(self):
        """Force WAL checkpoint for a clean sync copy."""
        self._check_writable()
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # -- internal -----------------------------------------------------------

    def _check_writable(self) -> None:
        """Refuse mutating writes on non-master nodes.

        Off-master nodes must not create a divergent local sqlite — they must
        POST to SLOTS_MASTER_URL instead. Fail loud so the error is not silent.
        """
        if not IS_MASTER:
            raise OffMasterWriteError(
                f"off-master write refused: this node is {HOSTNAME!r}, "
                f"the canonical store is on {SLOTS_MASTER_HOST!r}. "
                f"POST to {SLOTS_MASTER_URL!r} instead of writing a local sqlite."
            )

    def _require_owner(self, slot_id: str, by: str) -> sqlite3.Row:
        """Fetch a slot's row, asserting ``by`` is the contributor-of-record.

        Must be called inside the lock. Returns the row for further reads.
        """
        row = self._conn.execute(
            "SELECT * FROM slots WHERE slot_id=?", (slot_id,)
        ).fetchone()
        if row is None:
            raise SlotNotFoundError(slot_id)
        if row["contributor_id"] != by:
            raise SlotOwnershipError(
                f"writer {by!r} is not the contributor-of-record "
                f"({row['contributor_id']!r}) for slot {slot_id}"
            )
        return row

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in STATUSES:
            raise ValueError(
                f"invalid status {status!r}; must be one of {sorted(STATUSES)}"
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in _JSON_FIELDS:
            try:
                d[field] = json.loads(d[field]) if d.get(field) else None
            except (json.JSONDecodeError, TypeError):
                d[field] = None
        return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
