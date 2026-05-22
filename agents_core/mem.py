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

# --- Configuration ---

DB_DIR = Path("/data/memory")
DB_PATH = Path(os.environ.get("MEM_DB_PATH", DB_DIR / "mem.db"))
HOSTNAME = os.uname().nodename
IS_STARHOUSE = HOSTNAME == "starhouse"
STARHOUSE_SSH = "user@203.0.113.12"


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

    def __init__(self, db_path: Path = DB_PATH):
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

    def close(self):
        self._conn.close()

    def set(self, key: str, content: str, tags: list[str] | None = None,
            source: str = "") -> bool:
        """Upsert a memory. Returns True if created, False if updated."""
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
        return self._row_to_dict(row) if row else None

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
        return [self._row_to_dict(row) for row in rows]

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
            "mode": "read-write" if IS_STARHOUSE else "read-only (writes proxy to StarHouse)",
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
        """Force WAL checkpoint for clean sync copy."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

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
