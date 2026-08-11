"""ExhaustStore — purpose-built sibling SQLite store for machine-exhaust
mem.db writes that no code anywhere reads.

Tier 1 of `agents-core-mem-exhaust-sibling-store-v0`
(decision/mem-exhaust-split-scoped-by-reader-census-2026-08-11, ruling THREE
of decision/five-rulings-compose-pin-mem-split-lying-reviewer-translation-layer-2026-08-08).
Moves the `elevator/`, `weather/`, and `router/gw-review-divergence/` families
out of the searched, backed-up, FTS5-indexed mem.db into a second file that
has neither the search cost nor the FTS write cost, without touching any of
the three writers (conductor's elevator_scheduler.py, ops-layer's
ops_primitives.py, lapis-pm's spec_review.py via the `mem` CLI/HTTP) — they
all funnel through `agents_core.mem.MemoryStore.set()`, the one chokepoint
every exhaust writer already shares. See mem.py's `set()`/`get()`/
`list_by_prefix()` for how this is wired in.

Does NOT move `router/lapis-pm/*` (ratified, follow-on target, two live
readers) or `pm/*` (not ratified to move at all) — see EXHAUST_PREFIXES.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---

# /data/memory, NOT /srv/fast — despite the BRIX rule that bulk data belongs
# on /srv/* and root (where /data lives) is for the OS. This is a deliberate,
# ratified exception (agents-core-mem-exhaust-sibling-store-v0, leg 1): the
# nightly backup (/usr/local/bin/brix-backup-substrate.sh:40-41) copies only
# /data and /room, and Syncthing's `memory-store` folder covers only
# /data/memory — /srv/fast would leave this store unbacked-up and unsynced.
# At ~30-100MB/year this is not bulk data. Do not "correct" this to /srv/fast
# later; it must stay a sibling of mem.db.
EXHAUST_DB_DIR = Path("/data/memory")
EXHAUST_DB_PATH = Path(os.environ.get("MEM_EXHAUST_DB_PATH", EXHAUST_DB_DIR / "exhaust.db"))


def default_exhaust_path(db_path: Path) -> Path:
    """Resolve the sibling exhaust.db path for a given mem.db `db_path`.

    Priority: an explicit `MEM_EXHAUST_DB_PATH` env var (mirrors `MEM_DB_PATH`'s
    own override convention) wins; otherwise the sibling is a file named
    `exhaust.db` next to `db_path`. The second branch is what keeps every
    caller that points MemoryStore at a tmp_path db_path (i.e. every existing
    test in this repo) automatically isolated from the real
    `/data/memory/exhaust.db` — a routed write or a fall-through-read miss
    in a test never touches production, with no per-test plumbing required.
    In production `db_path` defaults to DB_PATH (`/data/memory/mem.db`), so
    this resolves to the real sibling, `/data/memory/exhaust.db`.
    """
    override = os.environ.get("MEM_EXHAUST_DB_PATH")
    if override:
        return Path(override)
    return Path(db_path).parent / "exhaust.db"

# Narrow, explicit routing table for agents-core-mem-exhaust-sibling-store-v0
# tier 1. Matched by exact `key.startswith(prefix)` against the FULL LITERAL
# prefix string below — never a substring test, a regex, or a split on the
# first path segment. A loose `router/` match would also catch
# `router/lapis-pm/*` (8,067 rows, two live readers, ratified to move
# separately and only after this store is proven live) and silently break
# PM-state grounding. Adding a prefix here must be a deliberate one-line
# spec-level edit, not a pattern that grows by accident — an unrouted new
# exhaust family staying in mem.db is the *safe* failure direction.
EXHAUST_PREFIXES: tuple[str, ...] = (
    "elevator/",
    "weather/",
    "router/gw-review-divergence/",
)

# `memories` table shape only, copied from agents_core.mem.SCHEMA. No FTS5,
# no triggers — FTS is 31% of mem.db's on-disk size (15.48MB of 49.31MB) and
# adds trigger writes per insert; nothing searches exhaust, which is the
# entire premise of the split.
EXHAUST_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    tags        TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

HOSTNAME = os.uname().nodename

_cold_path_logger = logging.getLogger("agents_core.mem.cold_path")


def route_to_exhaust(key: str) -> bool:
    """True if `key` belongs to one of the tier-1 exhaust prefixes.

    `str.startswith()` given a tuple checks each element with exact
    startswith semantics — equivalent to `any(key.startswith(p) for p in
    EXHAUST_PREFIXES)`, never a substring or regex test.
    """
    return key.startswith(EXHAUST_PREFIXES)


def could_overlap_exhaust(prefix: str) -> bool:
    """Cheap pre-filter for MemoryStore.list_by_prefix()'s fall-through.

    True if a `list_by_prefix(prefix)` query could possibly select any key
    covered by EXHAUST_PREFIXES — in either direction: a broader query like
    "elevator" as well as a narrower one like "elevator/proposals/". Lets
    list_by_prefix() skip the sibling store entirely for the vast majority of
    prefixes that share no root with the three routed families, instead of
    adding a query (and a lazily-created exhaust.db) to every call site.
    """
    return any(p.startswith(prefix) or prefix.startswith(p) for p in EXHAUST_PREFIXES)


def log_cold_path_access(key: str) -> None:
    """Emit a structured cold-path-access event: a read of `key` fell
    through from mem.db to the sibling exhaust store and found it there.

    This is the falsification mechanism for the zero-reader census
    (finding/mem-exhaust-45pct-has-zero-readers-2026-08-11): a grep census
    proves nobody *appears* to read these keys, not that nobody *does*. If
    the census holds, this log stays empty forever and costs nothing. If it
    doesn't, an operator finds out from a query instead of a silent
    degradation.

    Must be called directly from the MemoryStore method that hit the
    fall-through (get() / list_by_prefix()) — this walks the stack assuming
    exactly one frame between here and the original external caller.
    """
    frame = inspect.stack()[2]
    caller = f"{Path(frame.filename).name}:{frame.function}:{frame.lineno}"
    _cold_path_logger.warning(json.dumps({
        "event": "cold_path_access",
        "key": key,
        "caller": caller,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))


class ExhaustStore:
    """SQLite-backed sibling store for the three routed exhaust prefixes.

    Open sequence copied from agents_core.slots.SlotStore (PRAGMA
    busy_timeout=5000 + a `_migrate()` hook), NOT from MemoryStore's — per
    spec, because a second cross-process writer already exists on mem.db
    (mem-checkpoint.timer, every 5 minutes) and exhaust.db will pick up the
    same contention once the checkpoint covers both files (leg 4).
    """

    def __init__(self, db_path: Path = EXHAUST_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Let a second cross-process writer wait up to 5s for the WAL lock
        # instead of raising immediately.
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(EXHAUST_SCHEMA)
        self._migrate()
        # Serializes concurrent access from threads within one process, same
        # role as MemoryStore._lock.
        self._lock = threading.RLock()

    def _migrate(self):
        """Idempotent ADD COLUMN hook, empty today. Kept so a future schema
        change to this store doesn't have to introduce the pattern from
        scratch — shape copied from SlotStore._migrate(), not its content."""
        return

    def close(self):
        self._conn.close()

    def set(self, key: str, content: str, tags: list[str] | None = None,
            source: str = "") -> bool:
        """Upsert. Same contract as MemoryStore.set(): returns True if created,
        False if updated."""
        now = datetime.now(timezone.utc).isoformat()
        tag_str = ",".join(sorted(tags)) if tags else ""
        source = source or HOSTNAME

        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE memories SET content=?, tags=?, source=?, updated_at=? WHERE key=?",
                    (content, tag_str, source, now, key),
                )
            else:
                self._conn.execute(
                    "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key, content, tag_str, source, now, now),
                )
            self._conn.commit()
        return not existing

    def get(self, key: str) -> dict | None:
        """Exact key lookup."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
        return dict(row) if row else None

    def list_by_prefix(self, prefix: str, limit: int = 50) -> list[dict]:
        """Same LIKE-escape contract as MemoryStore.list_by_prefix()."""
        if not prefix:
            raise ValueError(
                "list_by_prefix() requires a non-empty prefix"
            )
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"{escaped}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE key LIKE ? ESCAPE '\\' ORDER BY key LIMIT ?",
                (pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def checkpoint_wal(self):
        """Force WAL checkpoint — the other half of mem.py's checkpoint_wal()
        single pass (leg 4)."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
