"""MemoryStore — SQLite + FTS5 cross-instance memory library.

This is the library half of `mem`. The `mem` CLI lives in
`/srv/agents/scripts/mem.py`, which imports MemoryStore from here.

StarHouse is the read/write source of truth. MacBook reads a synced
copy and proxies writes to StarHouse via SSH (handled in the CLI).

Usage:
    from agents_core.mem import MemoryStore
    store = MemoryStore()
    store.set("pattern/docker-bind", "Must explicitly bind ports in compose",
              tags=["docker", "networking"])
    results = store.search("docker bind")
"""

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from agents_core import mem_exhaust

# --- Configuration ---

DB_DIR = Path("/data/memory")
DB_PATH = Path(os.environ.get("MEM_DB_PATH", DB_DIR / "mem.db"))
HOSTNAME = os.uname().nodename
# Master (read-write) host for the mem substrate. Single deliberate value, NOT an
# env toggle — so the designation cannot drift per-host into a dual-master
# split-brain (the failure the BRIX-canonical cutover exists to escape). Flipping
# the master is a reviewed code change, deployed old-master-first. Was implicitly
# "starhouse" via the former IS_STARHOUSE; promoted to "brix" 2026-05-29 (substrate
# cutover step B). `IS_STARHOUSE` kept as a back-compat alias for any external ref.
MEM_MASTER_HOST = "brix"
IS_MASTER = HOSTNAME == MEM_MASTER_HOST
IS_STARHOUSE = IS_MASTER  # back-compat alias (no in-tree consumers as of 2026-05-29)
STARHOUSE_SSH = "user@203.0.113.12"
# HTTP base URL of the mem master's mem-server. Off-master writers (e.g.
# host-fault-recorder on StarHouse) POST here via MemClient instead of writing a
# divergent local sqlite. Env-overridable; default is the BRIX tailscale address.
MEM_MASTER_URL = os.environ.get("MEM_MASTER_URL", "http://203.0.113.10:8404")

# --- D4 write-path guard (mem-hygiene-automation-v0) ---
#
# The guard sits at this library chokepoint (MemoryStore.set), NOT at the
# HTTP layer: the break-glass scar writes are in-process
# (MemoryStore().set() from conductor tests / gw_topology reach()), never
# reach mem_server.py, and a source-pattern filter cannot distinguish test
# from production — the production gw_topology path's own source is
# "gw_topology", the same value its test scars carry. The writer-side scar
# fix (finding/gw-topology-tests-write-breakglass-scars-to-prod-mem-2026-08-11)
# is what lets the guard default ON: once the scar writer stops writing
# through the chokepoint, the pattern below matches nothing in production.
#
# MEM_ALLOW_TEST_WRITE=1 disables the guard (the conductor test suite sets
# it; the maintenance path never needs it — restore/purge bypass set()
# entirely, the named F6 exception).
#
# Pattern strictness (reviewer low, PR #331 cycle 1): the pattern matches
# the test-provenance SEGMENTS (test|tests|_test|mock|fake|fixture)
# delimited by -, _, / or a string boundary — a source like
# "production_test_data" matches (the "test" segment), which is the
# intended behavior: a source that names test provenance anywhere in its
# value is rejected unless MEM_ALLOW_TEST_WRITE=1. A production source
# that happens to embed a test segment and must write is the operator's
# choice to flag via the env var, not a pattern hole.
_TEST_SOURCE_PATTERN = re.compile(r"(^|[-_/])(test|tests|_test|mock|fake|fixture)([-_/]|$)")


class TestWriteRejected(Exception):
    """D4: a write with a test-provenance source was rejected by the
    MemoryStore.set() guard. mem_server maps this to a 4xx (not a bare
    500); the CLI exits non-zero."""


# --- Database Schema ---

SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    key         TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    tags        TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    key,
    content,
    tags,
    content='memories',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, key, content, tags)
    VALUES (new.rowid, new.key, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
    VALUES ('delete', old.rowid, old.key, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, content, tags)
    VALUES ('delete', old.rowid, old.key, old.content, old.tags);
    INSERT INTO memories_fts(rowid, key, content, tags)
    VALUES (new.rowid, new.key, new.content, new.tags);
END;
"""


class MemoryStore:
    """SQLite-backed memory store with FTS5 full-text search."""

    def __init__(self, db_path: Path = DB_PATH, exhaust_db_path: Path | None = None):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        # Serializes concurrent access from FastAPI threadpool. check_same_thread=False
        # removes the ownership guard but does not make the connection object safe for
        # simultaneous calls from different threads — this lock does. RLock because
        # stats() calls all_tags() internally.
        self._lock = threading.RLock()
        # Sibling exhaust store (agents-core-mem-exhaust-sibling-store-v0) —
        # opened lazily on first routed write or fall-through read, so the vast
        # majority of MemoryStore() instances that never touch a routed prefix
        # never create exhaust.db as a side effect. Defaults to a file next to
        # `db_path` (see default_exhaust_path()), which is what keeps every
        # test that points MemoryStore at a tmp_path automatically isolated
        # from the real /data/memory/exhaust.db.
        self._exhaust_db_path = exhaust_db_path or mem_exhaust.default_exhaust_path(self.db_path)
        self._exhaust: mem_exhaust.ExhaustStore | None = None

    def _exhaust_store(self) -> "mem_exhaust.ExhaustStore":
        if self._exhaust is None:
            with self._lock:
                if self._exhaust is None:
                    self._exhaust = mem_exhaust.ExhaustStore(db_path=self._exhaust_db_path)
        return self._exhaust

    def close(self):
        self._conn.close()
        if self._exhaust is not None:
            self._exhaust.close()

    def set(self, key: str, content: str, tags: list[str] | None = None,
            source: str = "") -> bool:
        """Upsert a memory. Returns True if created, False if updated.

        Keys matching a tier-1 exhaust prefix (agents-core-mem-exhaust-sibling-
        store-v0; see mem_exhaust.EXHAUST_PREFIXES) route to the sibling
        exhaust store instead of mem.db. This is the sole write chokepoint —
        every exhaust writer (elevator_scheduler, ops_primitives, lapis-pm's
        spec_review via the mem CLI) already funnels through here, so no
        caller needs to change.

        D4 write-path guard (mem-hygiene-automation-v0): a `source` that
        matches the test-provenance pattern is rejected with
        TestWriteRejected unless MEM_ALLOW_TEST_WRITE=1. The guard is on
        the EXPLICIT source argument only — the empty-source hostname
        default (source or HOSTNAME) is never pattern-matched, so a
        production writer that omits source on a test host is not
        silently locked out.
        """
        if source and _TEST_SOURCE_PATTERN.search(source):
            if os.environ.get("MEM_ALLOW_TEST_WRITE") != "1":
                raise TestWriteRejected(
                    f"test-provenance source {source!r} rejected at the "
                    f"MemoryStore.set() chokepoint (D4 write-path guard); "
                    f"set MEM_ALLOW_TEST_WRITE=1 to allow (test harnesses "
                    f"only)"
                )
        if mem_exhaust.route_to_exhaust(key):
            return self._exhaust_store().set(key, content, tags=tags, source=source)

        with self._lock:
            return self._set_unlocked(key, content, tags, source)

    def upsert_line(self, key: str, line: str, header: str, tags: list[str] | None = None,
                    source: str = "") -> bool:
        """Atomic read-modify-write append of one line to a row's content.

        Holds ``self._lock`` ONCE across the whole get-then-set sequence.
        The inner write goes through ``_set_unlocked()`` — the same body as
        ``set()`` minus the lock — so the batch-key upsert in
        ``mem_server.promote`` never re-acquires the lock (no nested
        acquisition, independent of the lock being re-entrant; reviewer
        PR #337 cycle 1 [med]: the previous shape held ``store._lock`` and
        called ``store.set()`` inside, which only worked because the lock
        is an RLock).

        Semantics: if the row does not exist, its content is
        ``header + "\\n" + line + "\\n"``; if it exists, ``line`` is
        appended on its own line unless it is already present (exact
        line, newline-terminated — not a substring test, so a key that is
        a substring of another line does not mis-dedup). Returns True if
        the row was created, False if it already existed (updated or
        deduped — the dedup case is an idempotent no-op write, which is
        the correct semantics for a retrying curation run).
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                base = ""
                created = True
            else:
                base = row["content"]
                created = False
            existing_lines = base.splitlines() if base else []
            if line not in existing_lines:
                if not base:
                    content = f"{header}\n{line}\n"
                else:
                    content = f"{base}\n{line}\n"
                self._set_unlocked(key, content, tags, source)
        return created

    def _set_unlocked(self, key: str, content: str, tags: list[str] | None,
                      source: str) -> bool:
        """The body of ``set()`` WITHOUT the lock acquisition.

        Caller MUST hold ``self._lock``. Kept as the single write path so
        ``set()`` and ``upsert_line()`` cannot drift apart.
        """
        now = datetime.now(timezone.utc).isoformat()
        tag_str = ",".join(sorted(tags)) if tags else ""
        source = source or HOSTNAME

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
        """Exact key lookup.

        Falls through to the sibling exhaust store on a miss so reads never
        develop a hole for a routed key (agents-core-mem-exhaust-sibling-
        store-v0). A fall-through hit logs exactly one cold-path-access event
        — see mem_exhaust.log_cold_path_access().
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
        if row:
            return self._row_to_dict(row)

        exhaust_row = self._exhaust_store().get(key)
        if exhaust_row is not None:
            mem_exhaust.log_cold_path_access(key)
        return exhaust_row

    def search(self, query: str, tag: str = "", limit: int = 20) -> list[dict]:
        """FTS5 ranked search. Returns results sorted by relevance."""
        safe_query = self._fts_escape(query)
        sql = (
            "SELECT m.*, rank FROM memories_fts f "
            "JOIN memories m ON m.rowid = f.rowid "
            "WHERE memories_fts MATCH ? "
        )
        params: list = [safe_query]

        if tag:
            sql += "AND (',' || m.tags || ',') LIKE ? "
            params.append(f"%,{tag},%")

        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_all(self, tag: str = "", tags: list[str] | None = None,
                 since: str = "", limit: int = 50) -> list[dict]:
        """List memories with optional tag/date filters.

        `tag` and `tags` may both be supplied and are combined (all required,
        AND intersection). Each tag is matched exactly against the normalized
        CSV tag column.
        """
        sql = "SELECT * FROM memories WHERE 1=1 "
        params: list = []

        required_tags: list[str] = []
        if tag:
            required_tags.append(tag)
        if tags:
            required_tags.extend(tags)
        for t in required_tags:
            sql += "AND (',' || tags || ',') LIKE ? "
            params.append(f"%,{t},%")
        if since:
            sql += "AND updated_at >= ? "
            params.append(since)

        sql += "ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_by_prefix(self, prefix: str, limit: int = 50) -> list[dict]:
        """Return entries whose key starts with `prefix`, ordered by key ascending.

        Empty prefix raises ValueError (use list_all() for unfiltered listing).
        Escapes SQL-LIKE wildcards in prefix so callers with literal dots, slashes,
        underscores, or percent signs get exact prefix semantics.
        Returns the same dict shape as list_all() / search() / get().
        """
        if not prefix:
            raise ValueError(
                "list_by_prefix() requires a non-empty prefix; "
                "use list_all() for unfiltered listing"
            )

        # Escape SQL-LIKE special chars before appending %
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"{escaped}%"

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE key LIKE ? ESCAPE '\\' ORDER BY key LIMIT ?",
                (pattern, limit),
            ).fetchall()
        results = [self._row_to_dict(row) for row in rows]

        # Fall through to the sibling exhaust store (agents-core-mem-exhaust-
        # sibling-store-v0) whenever `prefix` could possibly select a routed
        # key — always merged, not gated on mem.db returning zero rows.
        # During the leg-3 observation window mem.db still holds the
        # pre-migration rows for a routed prefix while new writes for that
        # same prefix land only in exhaust.db; gating on "mem.db was empty"
        # would silently hide those new rows from any caller — the exact
        # hole this fall-through exists to prevent. Cheap for every other
        # prefix: could_overlap_exhaust() is a pure string check, so the
        # sibling store is never even opened for an unrelated prefix.
        if mem_exhaust.could_overlap_exhaust(prefix):
            seen_keys = {row["key"] for row in results}
            for erow in self._exhaust_store().list_by_prefix(prefix, limit=limit):
                if erow["key"] in seen_keys:
                    continue
                results.append(erow)
                seen_keys.add(erow["key"])
                mem_exhaust.log_cold_path_access(erow["key"])
            results.sort(key=lambda r: r["key"])
            results = results[:limit]

        return results

    def delete(self, key: str) -> bool:
        """Delete a memory by key. Returns True if deleted."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE key = ?", (key,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def all_tags(self) -> list[tuple[str, int]]:
        """Return all unique tags with counts, sorted by count desc."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT tags FROM memories WHERE tags != ''"
            ).fetchall()

        tag_counts: dict[str, int] = {}
        for row in rows:
            for tag in row["tags"].split(","):
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return sorted(tag_counts.items(), key=lambda x: -x[1])

    def stats(self) -> dict:
        """Return store statistics."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) as n FROM memories"
            ).fetchone()["n"]
        tags = self.all_tags()
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_memories": count,
            "unique_tags": len(tags),
            "db_size_bytes": size_bytes,
            "db_path": str(self.db_path),
            "hostname": HOSTNAME,
            "mode": "read-write" if IS_MASTER else f"read-only (master is {MEM_MASTER_HOST})",
        }

    def dump(self, fmt: str = "md") -> str:
        """Export all memories."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY key"
            ).fetchall()
        memories = [self._row_to_dict(row) for row in rows]

        if fmt == "json":
            return json.dumps(memories, indent=2)

        lines = [f"# Memory Store Dump ({len(memories)} entries)\n"]
        for m in memories:
            tags = f" `[{m['tags']}]`" if m["tags"] else ""
            lines.append(f"## {m['key']}{tags}")
            lines.append(f"*Updated: {m['updated_at'][:10]} | Source: {m['source']}*\n")
            lines.append(m["content"])
            lines.append("")
        return "\n".join(lines)

    def checkpoint_wal(self):
        """Force WAL checkpoint for clean sync copy.

        Also checkpoints the sibling exhaust store in the same pass
        (agents-core-mem-exhaust-sibling-store-v0, leg 4) — two
        independently-scheduled checkpoints could leave the pair diverged at
        the moment Syncthing or the nightly backup reads them.

        Gated on whether the sibling file exists ON DISK, not on whether
        *this* instance ever opened it (`self._exhaust is not None`). The
        latter was tried in PR #224 and is inert in production: the real
        checkpoint mechanism, mem-checkpoint.service, constructs a fresh
        MemoryStore() every 5 minutes (confirmed via journalctl — new PID
        each firing) with no MEM_SERVER env set, so it always takes the
        direct `store = MemoryStore(); store.checkpoint_wal()` path on an
        instance that has never called set()/get()/list_by_prefix() — the
        per-instance gate is never true there, so the sibling WAL is never
        truncated by the mechanism this leg exists to fix (CORRECTION
        2026-08-11 on agents-core-mem-exhaust-sibling-store-v0's Leg 4).
        Checking the file's existence instead means any process that
        happens to run the checkpoint still covers the sibling as long as
        *some* process has ever routed a write there — while a store that
        has never been written anywhere still has no exhaust.db and nothing
        to truncate, so this still doesn't conjure an empty file as a side
        effect.
        """
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if self._exhaust is not None or self._exhaust_db_path.exists():
            self._exhaust_store().checkpoint_wal()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    @staticmethod
    def _fts_escape(query: str) -> str:
        """Escape special FTS5 characters, wrap terms for prefix matching."""
        cleaned = re.sub(r'["\(\)\*\-\+]', " ", query)
        terms = cleaned.split()
        if not terms:
            return '""'
        return " ".join(f'"{t}"' for t in terms)
