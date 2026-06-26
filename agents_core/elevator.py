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

# Pending-orphan reaper tuning (gw-admission-pending-orphan-reclaim-v0).
# Grace period before a freshly-enqueued ticket can be reaped (enqueuer may be mid-startup).
GW_ADMISSION_ORPHAN_GRACE_SEC = int(os.environ.get("GW_ADMISSION_ORPHAN_GRACE_SEC", "60"))
# Presumed-dead backstop: any pending gw-admission ticket older than this is reaped regardless
# of liveness probe result (any live enqueuer would have timed out itself by then).
GW_ADMISSION_MAX_WAIT_SEC = int(os.environ.get("GW_ADMISSION_MAX_WAIT_SEC", "900"))

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


def _get_process_start_time(pid: int) -> float | None:
    """Return psutil create_time() for pid, or None on any error (including NoSuchProcess)."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


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
        depends_on: str | None = None,
    ) -> str:
        """Enqueue a work item. Returns item_id."""
        self._check_writable()
        if lane not in VALID_LANES:
            raise ValueError(f"invalid lane {lane!r}; must be one of {sorted(VALID_LANES)}")
        if latency_class not in ("interactive", "batch"):
            raise ValueError(f"invalid latency_class {latency_class!r}")

        iid = item_id or uuid.uuid4().hex[:16]
        now = _now()

        # Validate depends_on atomically with insertion.
        if depends_on is not None:
            if depends_on == iid:
                raise ValueError(f"self-reference: item cannot depend on itself")

        # Stamp enqueuer identity (host + pid + process-start-time triple) for the
        # pending-orphan reaper. Stored in payload to avoid a schema migration.
        # Backward-compatible: existing code that reads payload ignores unknown keys.
        enqueuer_pid = os.getpid()
        enqueuer_id: dict[str, Any] = {"host": HOSTNAME, "pid": enqueuer_pid}
        st = _get_process_start_time(enqueuer_pid)
        if st is not None:
            enqueuer_id["start_time"] = st
        payload = dict(payload)
        payload["_enqueuer_id"] = enqueuer_id

        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM queue_items WHERE item_id = ?", (iid,)
            ).fetchone()
            if exists:
                raise ValueError(f"item_id already exists: {iid}")

            # If depends_on is set, verify precursor exists in this same transaction.
            if depends_on is not None:
                precursor = self._conn.execute(
                    "SELECT 1 FROM queue_items WHERE item_id = ?", (depends_on,)
                ).fetchone()
                if not precursor:
                    raise ValueError(f"depends_on_not_found: precursor item {depends_on!r} does not exist")

            self._conn.execute(
                "INSERT INTO queue_items "
                "(item_id, lane, kind, principal, payload, latency_class, status, "
                " attempts, created_at, slot_ref, depends_on) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    iid, lane, kind, principal, json.dumps(payload),
                    latency_class, "pending", 0, now, slot_ref, depends_on,
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
        pending items exist in the given lanes. Skips rows with unserved dependencies.
        Inline reaps stale claims and cascade-fails dependents of terminal precursors first."""
        self._check_writable()
        self._reap_inline()
        self._cascade_fail_dependents()

        now = _now()
        with self._lock:
            # Claim head-of-line across lanes in order.
            for lane in lanes:
                row = self._conn.execute(
                    "SELECT * FROM queue_items WHERE lane=? AND status='pending' "
                    "AND (depends_on IS NULL "
                    "  OR EXISTS (SELECT 1 FROM queue_items p "
                    "             WHERE p.item_id = queue_items.depends_on AND p.status='served')) "
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

    # -- admission (self-serve) -------------------------------------------

    def _has_claimed_on_lane(self, lane: str, principal: str) -> bool:
        """Return True if any claimed item with this principal exists on lane (read-only)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM queue_items WHERE lane=? AND status='claimed' AND principal=? LIMIT 1",
                (lane, principal),
            ).fetchone()
            return row is not None

    def _claimed_principals_on_lane(self, lane: str) -> set:
        """Return the set of distinct principals with claimed items on lane (read-only)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT principal FROM queue_items WHERE lane=? AND status='claimed'",
                (lane,),
            ).fetchall()
            return {row[0] for row in rows}

    def try_admit(
        self,
        item_id: str,
        lane: str,
        principal: str,
        max_groups: int = 1,
        claim_ttl_sec: int = 960,
    ) -> tuple[bool, bool]:
        """Atomically admit item_id on lane per principal-group concurrency policy.

        Admits iff:
        - Same-principal item already claimed (ride-along): admit immediately, skip FIFO.
        - No other-principal group holds >= max_groups slots: this item is the
          oldest pending head-of-line on lane (FIFO fairness for fresh groups).

        Sets status='claimed', claim_owner=principal, claimed_at, claim_ttl_sec on admit.
        Returns (admitted, is_ride_along). Both determinations are atomic within the
        store lock so callers don't need a separate _has_claimed_on_lane check. Requires
        IS_MASTER.
        """
        self._check_writable()
        now = _now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT principal FROM queue_items WHERE lane=? AND status='claimed'",
                (lane,),
            ).fetchall()
            claimed_principals = {row[0] for row in rows}
            other_principals = claimed_principals - {principal}

            if len(other_principals) >= max_groups:
                return False, False  # Different group(s) hold all available slots.

            if principal in claimed_principals:
                # Ride-along: same-group item already in-flight; admit immediately.
                cursor = self._conn.execute(
                    "UPDATE queue_items SET status='claimed', claim_owner=?, "
                    "claim_ttl_sec=?, claimed_at=? WHERE item_id=? AND status='pending'",
                    (principal, claim_ttl_sec, now, item_id),
                )
                if cursor.rowcount == 0:
                    return False, False
                self._conn.commit()
                return True, True  # admitted as ride-along

            # Fresh group or lane idle: squeeze past any dead pending heads (D2),
            # then run the FIFO head check against the updated state.
            now_dt = datetime.now(timezone.utc)
            self._squeeze_past_dead_pending_locked(lane, now_dt)
            self._conn.commit()  # commit dead-head failures (may be empty commit)

            head = self._conn.execute(
                "SELECT item_id, principal FROM queue_items "
                "WHERE lane=? AND status='pending' "
                "AND (depends_on IS NULL "
                "  OR EXISTS (SELECT 1 FROM queue_items p "
                "             WHERE p.item_id = queue_items.depends_on AND p.status='served')) "
                "ORDER BY created_at ASC LIMIT 1",
                (lane,),
            ).fetchone()
            if not head or head["item_id"] != item_id or head["principal"] != principal:
                return False, False

            cursor = self._conn.execute(
                "UPDATE queue_items SET status='claimed', claim_owner=?, "
                "claim_ttl_sec=?, claimed_at=? WHERE item_id=? AND status='pending'",
                (principal, claim_ttl_sec, now, item_id),
            )
            if cursor.rowcount == 0:
                return False, False
            self._conn.commit()
            return True, False  # admitted as fresh group

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

    def _reclaim_stale_locked(self, lane: str, now_str: str) -> int:
        """Run stale-claim reclaim. Caller must hold self._lock; does not commit.

        D3: gw-admission tickets past TTL branch on owner liveness (FOLD 1):
          - 'dead' (past grace): fail (do not requeue — dead rider is not requeued).
          - 'alive' or 'unknown': requeue to BACK with fresh created_at.
            No reclaim path retains created_at (prevents re-heading the line).
        Non-gw-admission tickets: bulk requeue to pending (existing behaviour)."""
        # Non-gw-admission: bulk requeue (unchanged behaviour).
        cursor = self._conn.execute(
            "UPDATE queue_items SET status='pending', attempts=attempts+1, "
            "claimed_at=NULL, claim_owner=NULL, claim_ttl_sec=NULL "
            "WHERE lane=? AND status='claimed' AND kind != 'gw-admission' "
            "AND claim_ttl_sec IS NOT NULL AND claimed_at IS NOT NULL "
            "AND datetime(claimed_at) < datetime(?, '-' || claim_ttl_sec || ' seconds')",
            (lane, now_str),
        )
        reclaimed = cursor.rowcount

        # gw-admission: per-row liveness branching (D3).
        try:
            now_dt = datetime.fromisoformat(now_str)
        except Exception:
            now_dt = datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        gw_stale = self._conn.execute(
            "SELECT item_id, payload, claimed_at FROM queue_items "
            "WHERE lane=? AND status='claimed' AND kind='gw-admission' "
            "AND claim_ttl_sec IS NOT NULL AND claimed_at IS NOT NULL "
            "AND datetime(claimed_at) < datetime(?, '-' || claim_ttl_sec || ' seconds')",
            (lane, now_str),
        ).fetchall()

        for row in gw_stale:
            item_id = row["item_id"]
            try:
                claimed_at = datetime.fromisoformat(row["claimed_at"])
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=timezone.utc)
                age_sec = (now_dt - claimed_at).total_seconds()
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except Exception:
                age_sec = float("inf")
                payload = {}

            verdict = self._enqueuer_liveness(payload, age_sec)
            if verdict == "dead":
                # Dead owner past grace: fail, do not requeue (FOLD 1).
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (json.dumps({"reclaim_stale": "owner_dead", "liveness": verdict}), item_id),
                )
            else:
                # Alive or unknown: requeue to BACK with fresh created_at (FOLD 1).
                fresh_ts = _now()
                self._conn.execute(
                    "UPDATE queue_items SET status='pending', attempts=attempts+1, "
                    "claimed_at=NULL, claim_owner=NULL, claim_ttl_sec=NULL, "
                    "created_at=? WHERE item_id=?",
                    (fresh_ts, item_id),
                )
            reclaimed += 1

        return reclaimed

    def reclaim_stale(self, lane: str) -> int:
        """Reclaim stale claimed items on lane past their claim_ttl_sec.

        Atomic under the store lock. Only touches items genuinely past their
        claim_ttl_sec. Returns the count of reclaimed rows."""
        self._check_writable()
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock:
            reclaimed = self._reclaim_stale_locked(lane, now_str)
            self._conn.commit()
        return reclaimed

    def reap(self) -> dict[str, int]:
        """Expire aged pending items and reclaim stale claims.

        Returns {"expired": count, "reclaimed": count}."""
        self._check_writable()
        return self._reap_inline()

    def _reap_inline(self) -> dict[str, int]:
        """Inline reap: expire pending items, reclaim stale claims, and reap dead-enqueuer orphans.

        All operations run as a single atomic transaction under the store lock.
        Returns {"expired": count, "reclaimed": count, "orphan_reaped": count}."""
        now = datetime.now(timezone.utc)
        max_age = timedelta(seconds=ELEVATOR_PENDING_MAX_AGE_SEC)
        now_str = now.isoformat()

        with self._lock:
            cutoff = (now - max_age).isoformat()
            cursor = self._conn.execute(
                "UPDATE queue_items SET status='expired' "
                "WHERE status='pending' AND created_at < ?",
                (cutoff,),
            )
            expired = cursor.rowcount
            # Reap dead-enqueuer pending orphans (fast path; before absolute TTL).
            orphan_reaped = self._reap_pending_orphans_locked(now)
            # D4: fast-reclaim claimed slots whose owner is provably dead.
            dead_claimed_reaped = self._reap_dead_claimed_locked(now)
            # Reclaim stale claims on all lanes in the same transaction.
            reclaimed = sum(self._reclaim_stale_locked(lane, now_str) for lane in LANES)
            self._conn.commit()

        return {
            "expired": expired,
            "reclaimed": reclaimed,
            "orphan_reaped": orphan_reaped,
            "dead_claimed_reaped": dead_claimed_reaped,
        }

    def _enqueuer_liveness(self, payload: dict, age_sec: float) -> str:
        """Compute host+pid+start_time liveness verdict for a ticket's _enqueuer_id stamp.

        Returns 'alive', 'dead', or 'unknown'.
        - 'dead': pid is ESRCH or start_time mismatches, AND age_sec is past the grace
          window. A dead pid within the grace window is treated as 'unknown' (FOLD 2:
          process may be mid-handoff/restart).
        - 'alive': same-host, pid alive, start_time matches (or absent).
        - 'unknown': cross-host, missing stamp/pid, or within grace window for a dead pid.

        Caller does NOT need to hold self._lock.
        """
        enqueuer_id = (payload or {}).get("_enqueuer_id")
        if not enqueuer_id:
            return "unknown"

        enqueuer_host = enqueuer_id.get("host")
        enqueuer_pid = enqueuer_id.get("pid")
        enqueuer_start_time = enqueuer_id.get("start_time")

        if enqueuer_host != HOSTNAME or enqueuer_pid is None:
            return "unknown"  # cross-host or missing pid: unverifiable

        # Probe pid liveness.
        try:
            os.kill(enqueuer_pid, 0)
            pid_alive = True
        except OSError:
            pid_alive = False

        if not pid_alive:
            # Grace window (FOLD 2): dead pid within grace treated as transitioning.
            if age_sec < GW_ADMISSION_ORPHAN_GRACE_SEC:
                return "unknown"
            return "dead"

        # PID is alive — check start_time to guard against PID recycling.
        if enqueuer_start_time is not None:
            current_start = _get_process_start_time(enqueuer_pid)
            if current_start is not None and abs(current_start - enqueuer_start_time) > 1.0:
                # Start-time mismatch: PID was recycled; original enqueuer is dead.
                if age_sec < GW_ADMISSION_ORPHAN_GRACE_SEC:
                    return "unknown"
                return "dead"

        return "alive"

    def _reap_pending_orphans_locked(self, now: datetime) -> int:
        """Reap pending gw-admission tickets whose enqueuer process is provably dead.

        Caller MUST hold self._lock and must call self._conn.commit() after.

        Liveness ladder (per spec gw-admission-pending-orphan-reclaim-v0 D2):
        1. Grace period: tickets younger than GW_ADMISSION_ORPHAN_GRACE_SEC are spared.
        2. Precise liveness (primary): calls _enqueuer_liveness. Reaps on 'dead' verdict.
        3. Presumed-dead backstop: any pending gw-admission ticket older than
           GW_ADMISSION_MAX_WAIT_SEC is reaped regardless of liveness result.
        4. Absolute TTL: handled by the caller's cutoff sweep (ELEVATOR_PENDING_MAX_AGE_SEC).

        Returns count of tickets transitioned to failed."""
        rows = self._conn.execute(
            "SELECT item_id, payload, created_at FROM queue_items "
            "WHERE status='pending' AND kind='gw-admission'"
        ).fetchall()

        reaped = 0
        for row in rows:
            item_id = row["item_id"]
            try:
                created = datetime.fromisoformat(row["created_at"])
                # Ensure both datetimes are tz-aware for comparison.
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_sec = (now - created).total_seconds()
            except Exception:
                continue

            # 1. Grace period: never reap a freshly-enqueued ticket.
            if age_sec < GW_ADMISSION_ORPHAN_GRACE_SEC:
                continue

            # 3. Presumed-dead backstop: any live enqueuer would have timed out by now.
            if age_sec >= GW_ADMISSION_MAX_WAIT_SEC:
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (json.dumps({"orphan_reap": "presumed_dead_backstop",
                                 "age_sec": int(age_sec)}), item_id),
                )
                reaped += 1
                continue

            # 2. Precise liveness via shared helper.
            try:
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except Exception:
                payload = {}

            verdict = self._enqueuer_liveness(payload, age_sec)
            if verdict == "dead":
                enqueuer_id = payload.get("_enqueuer_id", {})
                enqueuer_pid = enqueuer_id.get("pid")
                # Distinguish pid_dead vs pid_recycled for provenance compatibility.
                try:
                    os.kill(enqueuer_pid, 0)
                    reap_reason = "pid_recycled"
                except OSError:
                    reap_reason = "pid_dead"
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (json.dumps({"orphan_reap": reap_reason,
                                 "enqueuer_pid": enqueuer_pid}), item_id),
                )
                reaped += 1
            # 'alive' → spare; 'unknown' → fall through to absolute TTL

        return reaped

    def _squeeze_past_dead_pending_locked(self, lane: str, now_dt: datetime) -> int:
        """Inline-fail any pending gw-admission ticket on lane whose owner is provably dead.

        D2: called in try_admit before the FIFO head check so a live waiter behind a
        dead pending head is not forced to wait one full reaper cycle.

        Caller MUST hold self._lock. Does NOT commit.
        Returns count of tickets transitioned to failed."""
        rows = self._conn.execute(
            "SELECT item_id, payload, created_at FROM queue_items "
            "WHERE lane=? AND status='pending' AND kind='gw-admission'",
            (lane,),
        ).fetchall()

        failed = 0
        for row in rows:
            try:
                created = datetime.fromisoformat(row["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_sec = (now_dt - created).total_seconds()
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except Exception:
                continue

            if self._enqueuer_liveness(payload, age_sec) == "dead":
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (json.dumps({"squeeze_past": "owner_dead"}), row["item_id"]),
                )
                failed += 1

        return failed

    def _reap_dead_claimed_locked(self, now: datetime) -> int:
        """D4: fail claimed gw-admission tickets whose owner pid is provably dead.

        Fast-reclaim of the slot from a dead claimant — does not wait for claim_ttl_sec
        to elapse. Caller MUST hold self._lock and must call self._conn.commit() after.

        Returns count of tickets transitioned to failed."""
        rows = self._conn.execute(
            "SELECT item_id, payload, claimed_at FROM queue_items "
            "WHERE status='claimed' AND kind='gw-admission'"
        ).fetchall()

        reaped = 0
        for row in rows:
            try:
                claimed_at = datetime.fromisoformat(row["claimed_at"])
                if claimed_at.tzinfo is None:
                    claimed_at = claimed_at.replace(tzinfo=timezone.utc)
                age_sec = (now - claimed_at).total_seconds()
                payload = json.loads(row["payload"]) if row["payload"] else {}
            except Exception:
                continue

            if self._enqueuer_liveness(payload, age_sec) == "dead":
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (json.dumps({"orphan_reap": "claim_owner_dead",
                                 "age_since_claimed_sec": int(age_sec)}), row["item_id"]),
                )
                reaped += 1

        return reaped

    def fail_pending_by_pid(self, host: str, pid: int) -> int:
        """Best-effort: fail all pending/claimed gw-admission tickets stamped with host+pid.

        Used by parent-kill cleanup (D4) after proc.kill() to immediately unblock the
        FIFO lane without waiting for the next reaper cycle. Never raises.
        Returns count of tickets failed."""
        try:
            self._check_writable()
            with self._lock:
                rows = self._conn.execute(
                    "SELECT item_id, payload FROM queue_items "
                    "WHERE status IN ('pending', 'claimed') AND kind='gw-admission'"
                ).fetchall()
                count = 0
                for row in rows:
                    try:
                        payload = json.loads(row["payload"]) if row["payload"] else {}
                    except Exception:
                        continue
                    eid = payload.get("_enqueuer_id", {})
                    if eid.get("host") == host and eid.get("pid") == pid:
                        self._conn.execute(
                            "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                            (json.dumps({"orphan_reap": "parent_kill",
                                         "enqueuer_pid": pid}), row["item_id"]),
                        )
                        count += 1
                if count:
                    self._conn.commit()
                return count
        except Exception:
            return 0

    def _cascade_fail_dependents(self) -> None:
        """Cascade-fail pending items whose precursors are in terminal states.

        A pending item with a non-null depends_on is cascade-failed if:
        - Its precursor is 'failed' or 'expired'
        - Its precursor row does not exist (missing)

        Each cascade-fail sets provenance with 'depends_on_failed' key.
        This is called before claim() and after reap() to ensure wedged dependents are cleaned.
        """
        with self._lock:
            # Find all pending dependents whose precursors are failed, expired, or missing.
            # Use a cursor loop to handle per-row provenance JSON.
            cursor = self._conn.execute(
                "SELECT q.item_id, q.depends_on, "
                "       COALESCE(p.status, 'missing') as precursor_status "
                "FROM queue_items q "
                "LEFT JOIN queue_items p ON q.depends_on = p.item_id "
                "WHERE q.status='pending' AND q.depends_on IS NOT NULL "
                "AND (p.status IN ('failed', 'expired') OR p.status IS NULL)"
            )
            rows = cursor.fetchall()

            for row in rows:
                item_id = row["item_id"]
                precursor_id = row["depends_on"]
                precursor_status = row["precursor_status"]

                provenance_dict = {
                    "depends_on_failed": {
                        "precursor_id": precursor_id,
                        "precursor_status": precursor_status,
                    }
                }
                provenance_json = json.dumps(provenance_dict)
                self._conn.execute(
                    "UPDATE queue_items SET status='failed', provenance=? WHERE item_id=?",
                    (provenance_json, item_id),
                )

            if rows:
                self._conn.commit()

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
