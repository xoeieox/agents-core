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

import hashlib
import json
import logging
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

# Handoff-baton kinds that a controller (Morph) can dispatch on without parsing prose.
# Closed set: exactly these dispatch verbs; anything else is a ValueError.
NEXT_KINDS = frozenset({
    "review-pr", "bind-next", "deploy", "await-human", "done", "blocked", "exploring",
})

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
    next               TEXT NOT NULL DEFAULT '{}',
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
_JSON_FIELDS = ("domain_touch", "horizon", "checkpoints", "escalation", "facets_verdict", "next")

# Observer-namespace columns added after the initial gate-5 schema. CREATE TABLE
# IF NOT EXISTS won't add columns to a pre-existing table, so they are applied as
# idempotent ADD COLUMN migrations against live DBs (see SlotStore._migrate).
_ADDED_COLUMNS = (
    ("facets_verdict", "TEXT"),
    ("facets_last_update", "TEXT"),
    ("next", "TEXT NOT NULL DEFAULT '{}'"),
    ("next_actuated", "INTEGER DEFAULT 0"),
    ("next_actuated_at", "TEXT"),
)


class _AdjacentCache:
    """Single-flight + short-TTL micro-cache for adjacent() calls.

    Pattern adapted (clean-room, no code copied) from the Fast Gemma Challenge dashboard:
    huggingface.co/spaces/gemma-challenge/gemma-dashboard — single-flight+serve-stale.

    Designed for **sync-threaded** execution (FastAPI threadpool + threading primitives).
    A future migration to asyncio handlers would silently break this cache.
    """

    def __init__(self, ttl_sec: float = 2.0):
        """ttl_sec: cache lifetime (0 disables cache)."""
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}  # key -> (computed_at, result)
        self._inflight: dict[tuple, threading.Event] = {}  # key -> Event (per-key single-flight)
        self._inflight_errors: dict[tuple, Exception] = {}  # key -> Exception on compute failure
        self._lock = threading.Lock()
        self._ttl_sec = ttl_sec

    def get(self, key: tuple, compute_fn, store_ref: "SlotStore") -> list[dict]:
        """Get cached result or compute once (single-flight dedup) per key.

        Args:
            key: (tuple(sorted(files)), tuple(sorted(mem_keys)), exclude_slot_id, include_inactive)
            compute_fn: callable() -> list[dict] that runs the actual scan
            store_ref: the SlotStore instance, for serve-stale error handling

        Returns: list[dict] result from cache, in-flight, or fresh compute.
        Raises: sqlite3.OperationalError if compute fails and no cache exists.
        On compute error with warm cache, logs a warning and returns last-good result.
        """
        now = datetime.now(timezone.utc).timestamp()
        should_compute = False

        with self._lock:
            # Check cache validity (only if TTL > 0, i.e., caching is enabled).
            if self._ttl_sec > 0 and key in self._cache:
                computed_at, cached_result = self._cache[key]
                age = now - computed_at
                if age < self._ttl_sec:
                    return cached_result
                # else: cache expired, fall through to compute (don't return yet)

            # Check if another thread is computing this key — wait for it (only if caching enabled).
            if self._ttl_sec > 0 and key in self._inflight:
                event = self._inflight[key]
            else:
                # First caller for this key (or caching disabled) — compute fresh.
                if self._ttl_sec > 0:
                    event = threading.Event()
                    self._inflight[key] = event
                else:
                    event = None
                should_compute = True

        if should_compute:
            try:
                result = compute_fn()
                if self._ttl_sec > 0:
                    with self._lock:
                        self._cache[key] = (now, result)
                        self._inflight.pop(key, None)
                        self._inflight_errors.pop(key, None)
                        event.set()
                return result
            except sqlite3.OperationalError as e:
                # Serve-stale on DB contention/lock.
                with self._lock:
                    if self._ttl_sec > 0 and key in self._cache:
                        _, cached = self._cache[key]
                        logging.warning(
                            f"adjacent() scan failed (DB busy), serving stale cache: {e}"
                        )
                        self._inflight.pop(key, None)
                        self._inflight_errors.pop(key, None)
                        event.set()
                        return cached
                    if event is not None:
                        self._inflight.pop(key, None)
                        self._inflight_errors[key] = e
                        event.set()
                raise
        else:
            # Wait for the in-flight computation to finish.
            event.wait()
            with self._lock:
                # Check if the in-flight compute failed.
                if key in self._inflight_errors:
                    error = self._inflight_errors.get(key, None)
                    raise error
                if key in self._cache:
                    _, result = self._cache[key]
                    return result
            # Should not reach here, but fallback to empty list if cache somehow vanished.
            return []


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
        # Single-flight + short-TTL cache for adjacent() scans (env-configurable).
        ttl_sec = float(os.environ.get("SLOT_ADJACENT_CACHE_TTL_SEC", "2.0"))
        self._adjacent_cache = _AdjacentCache(ttl_sec)

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

    def set_next(
        self,
        slot_id: str,
        *,
        by: str,
        kind: str,
        ref: str | None = None,
        blocked_on: list[str] | None = None,
        proposal: str = "",
        actuated: bool = False,
    ) -> bool:
        """Publish the next handoff baton (what should happen next).

        ``kind`` must be in NEXT_KINDS — the dispatch verbs Morph can act on.
        The ``exploring`` kind is the quiet-pause state: Morph halts auto-advance
        but does NOT notify (unlike ``await-human`` or ``blocked`` which trigger
        notifications). All other kinds represent active transitions.
        ``blocked_on`` is a list of slot_ids this slot depends on (deduplicated,
        sorted). ``proposal`` is optional human-readable context. ``actuated``
        indicates whether this has been acted on yet (defaults False, UI-honesty).
        Contributor-of-record only.

        A fresh baton starts un-actuated (next_actuated=0); when this method publishes
        a new baton, it resets the actuation state.
        """
        self._check_writable()
        if kind not in NEXT_KINDS:
            raise ValueError(
                f"invalid kind {kind!r}; must be one of {sorted(NEXT_KINDS)}"
            )
        now = _now()
        next_record = {
            "kind": kind,
            "ref": ref,
            "blocked_on": sorted(set(blocked_on or [])),
            "proposal": proposal,
            "actuated": bool(actuated),
        }
        with self._lock:
            self._require_owner(slot_id, by)
            self._conn.execute(
                "UPDATE slots SET next=?, next_actuated=0, next_actuated_at=NULL, last_update=? WHERE slot_id=?",
                (json.dumps(next_record, sort_keys=True), now, slot_id),
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

    def set_actuated(self, slot_id: str, *, by: str = "morph") -> bool:
        """Mark a slot's ``next`` baton as actuated (acknowledged/acted-upon).

        This is a non-owner-guarded observer-pattern write (separate column namespace,
        like ``observer_update`` and ``facets_ratify``). Morph (Unit 3) uses this
        after acting on a baton: it marks the baton acknowledged but does NOT change
        the contributor's owner-guarded fields (status, kind, ref, blocked_on, proposal).

        Idempotent: if there is no ``next`` baton, or it is already actuated, returns
        False without error (safe for at-least-once retry). On successful actuation of
        a fresh baton, returns True.

        Raises SlotNotFoundError if the slot does not exist.
        """
        self._check_writable()
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM slots WHERE slot_id=?", (slot_id,)
            ).fetchone()
            if row is None:
                raise SlotNotFoundError(slot_id)
            # Idempotent: if no next baton or already actuated, return False.
            next_data = json.loads(row["next"]) if row["next"] else {}
            if not next_data:
                return False
            # Check if already actuated (via the separate next_actuated column).
            if row["next_actuated"]:
                return False
            # Mark as actuated in the separate column namespace.
            self._conn.execute(
                "UPDATE slots SET next_actuated=1, next_actuated_at=? WHERE slot_id=?",
                (now, slot_id),
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

        Cached with single-flight dedup (per-key) and serve-stale-on-error for
        robustness under DB contention.
        """
        if not (files or mem_keys):
            return []
        # Normalize the cache key: lists are unhashable, so use sorted tuples.
        key = (
            tuple(sorted(files or [])),
            tuple(sorted(mem_keys or [])),
            exclude_slot_id,
            include_inactive,
        )
        return self._adjacent_cache.get(
            key,
            lambda: self._adjacent_impl(files, mem_keys, exclude_slot_id, include_inactive),
            self,
        )

    def _adjacent_impl(
        self,
        files: list[str] | None,
        mem_keys: list[str] | None,
        exclude_slot_id: str | None,
        include_inactive: bool,
    ) -> list[dict]:
        """Internal implementation of adjacent() — the actual full-table scan."""
        want_files = set(files or [])
        want_keys = set(mem_keys or [])
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

    def read_version(
        self,
        project_id: str | None = None,
        status: str | None = None,
        contributor_id: str | None = None,
        slot_id: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Compute an ETag token from COUNT + MAX(last_update) for conditional reads.

        Pattern adapted (clean-room, no code copied) from the Fast Gemma Challenge dashboard:
        huggingface.co/spaces/gemma-challenge/gemma-dashboard — ETag conditional-read validator.

        The token is stable across calls with the same filters and changes only when the
        filtered result set changes (insert/update/delete). Used by handlers to emit an ETag
        header and respond with 304 Not Modified if the client's If-None-Match matches.
        """
        sql = "SELECT COUNT(*) AS cnt, MAX(last_update) AS max_lu FROM slots WHERE 1=1 "
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
        if slot_id:
            sql += "AND slot_id=? "
            params.append(slot_id)
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        count = row["cnt"] or 0
        max_lu = row["max_lu"] or ""
        # Normalize filters for the token salt.
        norm = json.dumps({k: v for k, v in {
            "project_id": project_id,
            "status": status,
            "contributor_id": contributor_id,
            "slot_id": slot_id,
            "limit": limit,
        }.items() if v is not None}, sort_keys=True)
        token = f"{count}-{max_lu}"
        etag = '"' + hashlib.sha256((norm + token).encode()).hexdigest()[:16] + '"'
        return etag

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
        # Sync the separate next_actuated column back into the next JSON blob for
        # back-compat: readers see actuated as bool(next_actuated) in the next dict.
        if d.get("next") and isinstance(d["next"], dict):
            d["next"]["actuated"] = bool(d.get("next_actuated"))
        return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
