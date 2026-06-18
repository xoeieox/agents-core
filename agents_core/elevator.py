"""ElevatorStore — SQLite-backed work queue for GravityWell-lane tasks.

The elevator queue holds and orders three classes of work (interactive, deliberation, execution)
for the morph read-loop. Producers enqueue work; the scheduler (future unit) claims → serves → acks
items as a non-owner broker. The queue exposes node-state (queue depth, GW status) via a read-only
floor-indicator endpoint.

- **Store location:** SQLite at `/data/elevator/queue.db` (single-writer on BRIX)
- **Routes:** Served by slot_server at `/v0/elevator/*` (no new port)
- **Auth:** Bearer token (legacy-shared mode per spec "Why not SlotStore")
- **Lanes:** 'interactive' | 'deliberation' | 'execution' (priority-ordered)
- **Reaper:** Wired as background thread + inline sweep (expires aged pending, reclaims stale claims)

Usage:
    from agents_core.elevator import ElevatorStore
    store = ElevatorStore()
    item_id = store.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "...", "context": {...}},
        principal="session-123",
        latency_class="interactive",
    )
    item = store.claim(lanes=["interactive", "deliberation"], owner="broker-1", claim_ttl_sec=30)
    store.ack(item["item_id"], result_ref="result://...", provenance={...})
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
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None


# --- Configuration ---

DB_DIR = Path("/data/elevator")
DB_PATH = Path(os.environ.get("ELEVATOR_DB_PATH", DB_DIR / "queue.db"))
HOSTNAME = socket.gethostname()
ELEVATOR_MASTER_HOST = "brix"
IS_MASTER = HOSTNAME == ELEVATOR_MASTER_HOST
ELEVATOR_MASTER_URL = os.environ.get("ELEVATOR_MASTER_URL", "http://203.0.113.10:8405")

# Reaper tuning: pending items expire after this many seconds (U1 conservative default; tune with data).
ELEVATOR_PENDING_MAX_AGE_SEC = 3600

# Lane order (priority for claim): interactive first, then deliberation, then execution.
LANES = ("interactive", "deliberation", "execution")
VALID_LANES = frozenset(LANES)

# Item statuses through lifecycle.
STATUSES = frozenset({"pending", "claimed", "served", "failed", "expired"})


# --- Database Schema ---

SCHEMA = """\
CREATE TABLE IF NOT EXISTS queue_items (
    item_id            TEXT PRIMARY KEY,
    lane               TEXT NOT NULL,
    kind               TEXT NOT NULL,
    principal          TEXT NOT NULL,
    payload            TEXT NOT NULL,
    latency_class      TEXT NOT NULL,
    status             TEXT NOT NULL,
    attempts           INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    claimed_at         TEXT,
    claim_owner        TEXT,
    claim_ttl_sec      INTEGER,
    served_at          TEXT,
    result_ref         TEXT,
    slot_ref           TEXT,
    depends_on         TEXT,
    provenance         TEXT,
    result             TEXT
);

CREATE INDEX IF NOT EXISTS queue_items_lane_status_created
    ON queue_items(lane, status, created_at);
CREATE INDEX IF NOT EXISTS queue_items_status_claimed
    ON queue_items(status, claimed_at);
"""


def _now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class QueueError(Exception):
    """Base exception for queue operations."""


class QueueNotFoundError(KeyError):
    """Raised when an operation targets an item_id that does not exist."""


class OffMasterWriteError(RuntimeError):
    """Raised when a non-master node attempts a local mutating write to ElevatorStore."""


class ElevatorStore:
    """SQLite-backed work queue for GravityWell lanes. Thread-safe; one connection + lock."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()
        self._migrate_add_result_column()
        # Start reaper thread (master only).
        self._reaper_stop = threading.Event()
        self._reaper_thread = None
        if IS_MASTER:
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop, daemon=True, name="elevator-reaper"
            )
            self._reaper_thread.start()

    def _check_writable(self):
        """Ensure this node is the master before allowing writes."""
        if not IS_MASTER:
            raise OffMasterWriteError(
                f"off-master write attempt on {HOSTNAME}; "
                f"POST to {ELEVATOR_MASTER_URL} instead"
            )

    def _migrate_add_result_column(self):
        """Idempotent migration: add result column if it doesn't exist."""
        with self._lock:
            # Check if result column already exists.
            cursor = self._conn.execute("PRAGMA table_info(queue_items)")
            columns = {row[1] for row in cursor.fetchall()}
            if "result" not in columns:
                self._conn.execute("ALTER TABLE queue_items ADD COLUMN result TEXT")
                self._conn.commit()

    def close(self):
        """Shut down the reaper and close the database."""
        self._reaper_stop.set()
        if self._reaper_thread:
            self._reaper_thread.join(timeout=5)
        self._conn.close()

    # -- enqueue (producer) -----------------------------------------------

    def enqueue(
        self,
        lane: str,
        kind: str,
        payload: dict,
        principal: str,
        latency_class: str,
        slot_ref: str | None = None,
        item_id: str | None = None,
    ) -> str:
        """Enqueue a work item. Returns item_id."""
        self._check_writable()
        if lane not in VALID_LANES:
            raise ValueError(f"invalid lane {lane!r}; must be one of {sorted(VALID_LANES)}")
        if latency_class not in ("interactive", "batch"):
            raise ValueError(f"invalid latency_class {latency_class!r}")

        iid = item_id or uuid.uuid4().hex[:16]
        now = _now()
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM queue_items WHERE item_id = ?", (iid,)
            ).fetchone()
            if exists:
                raise ValueError(f"item_id already exists: {iid}")
            self._conn.execute(
                "INSERT INTO queue_items "
                "(item_id, lane, kind, principal, payload, latency_class, status, "
                " attempts, created_at, slot_ref) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    iid, lane, kind, principal, json.dumps(payload),
                    latency_class, "pending", 0, now, slot_ref,
                ),
            )
            self._conn.commit()
        return iid

    # -- broker operations ------------------------------------------------

    def claim(
        self, lanes: list[str], owner: str, claim_ttl_sec: int
    ) -> dict[str, Any] | None:
        """Atomically claim the head-of-line pending item across ordered lanes.

        Returns the claimed item (as a dict) with status='claimed', or None if no
        pending items exist in the given lanes. Inline reaps stale claims first."""
        self._check_writable()
        self._reap_inline()

        now = _now()
        with self._lock:
            # Claim head-of-line across lanes in order.
            for lane in lanes:
                row = self._conn.execute(
                    "SELECT * FROM queue_items WHERE lane=? AND status='pending' "
                    "ORDER BY created_at ASC LIMIT 1",
                    (lane,),
                ).fetchone()
                if row:
                    iid = row["item_id"]
                    self._conn.execute(
                        "UPDATE queue_items SET status='claimed', claim_owner=?, "
                        "claim_ttl_sec=?, claimed_at=? WHERE item_id=?",
                        (owner, claim_ttl_sec, now, iid),
                    )
                    self._conn.commit()
                    # Fetch the updated row.
                    updated_row = self._conn.execute(
                        "SELECT * FROM queue_items WHERE item_id=?", (iid,)
                    ).fetchone()
                    return self._row_to_dict(updated_row)
        return None

    def ack(
        self,
        item_id: str,
        result_ref: str | None = None,
        provenance: dict | None = None,
        result: str | None = None,
    ) -> bool:
        """Mark a claimed item as served. Idempotent on an already-served item.

        Returns True. Raises QueueNotFoundError if item does not exist."""
        self._check_writable()
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not row:
                raise QueueNotFoundError(item_id)

            # Idempotent: if already served, just return True.
            if row["status"] == "served":
                return True

            # claimed -> served.
            provenance_json = json.dumps(provenance) if provenance else None
            self._conn.execute(
                "UPDATE queue_items SET status='served', served_at=?, "
                "result_ref=?, provenance=?, result=? WHERE item_id=?",
                (now, result_ref, provenance_json, result, item_id),
            )
            self._conn.commit()
        return True

    def requeue(self, item_id: str) -> bool:
        """Requeue a claimed item back to pending (retry). Increments attempts.

        Returns True. Raises QueueNotFoundError if item does not exist."""
        self._check_writable()
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts FROM queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not row:
                raise QueueNotFoundError(item_id)

            attempts = (row["attempts"] or 0) + 1
            self._conn.execute(
                "UPDATE queue_items SET status='pending', attempts=?, "
                "claimed_at=NULL, claim_owner=NULL, claim_ttl_sec=NULL "
                "WHERE item_id=?",
                (attempts, item_id),
            )
            self._conn.commit()
        return True

    def fail(self, item_id: str) -> bool:
        """Mark a claimed item as permanently failed (terminal).

        Returns True. Raises QueueNotFoundError if item does not exist."""
        self._check_writable()
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            if not exists:
                raise QueueNotFoundError(item_id)

            self._conn.execute(
                "UPDATE queue_items SET status='failed' WHERE item_id=?", (item_id,)
            )
            self._conn.commit()
        return True

    # -- reader (HTTP) ----------------------------------------------------

    def get(self, item_id: str) -> dict[str, Any] | None:
        """Fetch an item by ID. Returns dict or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM queue_items WHERE item_id=?", (item_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def state(self) -> dict[str, Any]:
        """Return node-state / floor-indicator: queue depth by lane + GW status.

        The gw block is best-effort (composes flip-controller + doorman); returns
        queue block always, gw block with unknown/null on any control-plane error."""
        with self._lock:
            queue_state = {}
            for lane in LANES:
                pending = self._conn.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE lane=? AND status='pending'",
                    (lane,),
                ).fetchone()[0]
                claimed = self._conn.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE lane=? AND status='claimed'",
                    (lane,),
                ).fetchone()[0]
                oldest = self._conn.execute(
                    "SELECT created_at FROM queue_items WHERE lane=? AND status='pending' "
                    "ORDER BY created_at ASC LIMIT 1",
                    (lane,),
                ).fetchone()
                oldest_age = None
                if oldest:
                    created = datetime.fromisoformat(oldest[0])
                    now = datetime.now(timezone.utc)
                    oldest_age = int((now - created).total_seconds())

                queue_state[lane] = {
                    "pending": pending,
                    "claimed": claimed,
                    "oldest_age_sec": oldest_age,
                }

        # Best-effort GW status from flip-controller + doorman (reads only, no error).
        gw_state = self._compose_gw_status()

        return {
            "queue": queue_state,
            "gw": gw_state,
            "as_of": _now(),
        }

    def _compose_gw_status(self) -> dict[str, Any]:
        """Best-effort read of GW state from control-plane services.

        Returns a dict with mode/serving_ready/healthy/unknown fields.
        If either service is unreachable, returns unknown/null gracefully."""
        mode = "unknown"
        serving_ready = None
        healthy = None

        # Attempt to fetch flip-controller mode (best-effort, no error on failure).
        if httpx:
            try:
                resp = httpx.get("http://203.0.113.10:8408/v0/status", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    flip_mode = data.get("mode")
                    if flip_mode in ("big", "swarm", "offline", "transitioning"):
                        mode = flip_mode
            except Exception:
                pass

            # Attempt to fetch doorman serving_ready (best-effort, no error on failure).
            try:
                resp = httpx.get("http://127.0.0.1:8407/status", timeout=2.0)
                if resp.status_code == 200:
                    data = resp.json()
                    gw_data = data.get("nodes", {}).get("gravitywell", {})
                    # Map nodes.gravitywell.serving → gw.serving_ready.
                    if "serving" in gw_data:
                        serving_ready = gw_data["serving"]
            except Exception:
                pass

        return {
            "mode": mode,
            "serving_ready": serving_ready,
            "healthy": healthy,
        }

    # -- reaper (maintenance) ---------------------------------------------

    def _reaper_loop(self):
        """Background thread: reaps expired pending and stale claims."""
        while not self._reaper_stop.wait(timeout=60):  # reap every 60 seconds
            try:
                self.reap()
            except Exception:
                # Swallow exceptions to keep reaper alive.
                pass

    def reap(self) -> dict[str, int]:
        """Expire aged pending items and reclaim stale claims.

        Returns {"expired": count, "reclaimed": count}."""
        self._check_writable()
        return self._reap_inline()

    def _reap_inline(self) -> dict[str, int]:
        """Inline reap: expire pending items and reclaim stale claims.

        Returns {"expired": count, "reclaimed": count}."""
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        max_age = timedelta(seconds=ELEVATOR_PENDING_MAX_AGE_SEC)

        with self._lock:
            # Expire pending items older than max_age.
            cutoff = (now - max_age).isoformat()
            cursor = self._conn.execute(
                "UPDATE queue_items SET status='expired' "
                "WHERE status='pending' AND created_at < ?",
                (cutoff,),
            )
            expired = cursor.rowcount

            # Reclaim stale claims: claimed items past their claim_ttl_sec.
            cursor = self._conn.execute(
                "UPDATE queue_items SET status='pending', attempts=attempts+1, "
                "claimed_at=NULL, claim_owner=NULL, claim_ttl_sec=NULL "
                "WHERE status='claimed' AND claim_ttl_sec IS NOT NULL "
                "AND claimed_at IS NOT NULL "
                "AND datetime(claimed_at) < datetime(?, '-' || claim_ttl_sec || ' seconds')",
                (now_str,),
            )
            reclaimed = cursor.rowcount
            self._conn.commit()

        return {"expired": expired, "reclaimed": reclaimed}

    # -- helpers ----------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any] | None:
        """Convert a sqlite3.Row to a dict with parsed JSON fields."""
        if not row:
            return None
        d = dict(row)
        # Parse JSON fields.
        if d.get("payload"):
            d["payload"] = json.loads(d["payload"])
        if d.get("provenance"):
            d["provenance"] = json.loads(d["provenance"])
        return d
