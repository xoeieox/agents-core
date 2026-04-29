"""agents_core.vault_audit — SQLite-backed append-only audit log.

Database path: /data/vault-audit.db (override via VAULT_AUDIT_DB env var).

Schema:
    id             INTEGER PRIMARY KEY AUTOINCREMENT
    ts             TEXT     ISO-8601 timestamp with timezone
    path           TEXT     vault-relative path of the written file
    content_hash   TEXT     "sha256:<hex>" of content after write
    prev_hash      TEXT     "sha256:<hex>" of content before write, or NULL
    agent_id       TEXT     agent identifier
    intent         TEXT     human-readable write intent
    citations_json TEXT     JSON-serialized list[Citation], or "[]"

All columns are NOT NULL except prev_hash (NULL on first write).
The table is append-only; no UPDATE or DELETE ever runs on it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_DB = Path("/data/vault-audit.db")


def _db_path() -> Path:
    return Path(os.environ.get("VAULT_AUDIT_DB", str(_DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             TEXT    NOT NULL,
            path           TEXT    NOT NULL,
            content_hash   TEXT    NOT NULL,
            prev_hash      TEXT,
            agent_id       TEXT    NOT NULL,
            intent         TEXT    NOT NULL,
            citations_json TEXT    NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_path    ON audit_log(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_agent   ON audit_log(agent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit_log(ts)")
    conn.commit()
    return conn


# Module-level connection (lazy singleton, re-created on first use per process)
_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _connect()
    return _conn


def reset_connection() -> None:
    """Force reconnect — used in tests to pick up a new VAULT_AUDIT_DB value."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None


def append(
    *,
    ts: datetime,
    path: str,
    content_hash: str,
    prev_hash: str | None,
    agent_id: str,
    intent: str,
    citations: list[dict[str, Any]] | None = None,
) -> int:
    """Append a row to the audit log. Returns the new row id."""
    conn = _get_conn()
    citations_json = json.dumps(citations or [])
    ts_str = ts.isoformat()
    cur = conn.execute(
        """
        INSERT INTO audit_log (ts, path, content_hash, prev_hash, agent_id, intent, citations_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts_str, path, content_hash, prev_hash, agent_id, intent, citations_json),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def query_by_path(path: str) -> list[dict[str, Any]]:
    """Return all audit rows for a given path, oldest first."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id,ts,path,content_hash,prev_hash,agent_id,intent,citations_json "
        "FROM audit_log WHERE path=? ORDER BY id",
        (path,),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def query_by_agent(agent_id: str) -> list[dict[str, Any]]:
    """Return all audit rows for a given agent_id, oldest first."""
    conn = _get_conn()
    cur = conn.execute(
        "SELECT id,ts,path,content_hash,prev_hash,agent_id,intent,citations_json "
        "FROM audit_log WHERE agent_id=? ORDER BY id",
        (agent_id,),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def query_by_time_range(
    since: datetime,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return audit rows with ts >= since (and <= until if given), oldest first."""
    conn = _get_conn()
    if until is None:
        cur = conn.execute(
            "SELECT id,ts,path,content_hash,prev_hash,agent_id,intent,citations_json "
            "FROM audit_log WHERE ts >= ? ORDER BY id",
            (since.isoformat(),),
        )
    else:
        cur = conn.execute(
            "SELECT id,ts,path,content_hash,prev_hash,agent_id,intent,citations_json "
            "FROM audit_log WHERE ts >= ? AND ts <= ? ORDER BY id",
            (since.isoformat(), until.isoformat()),
        )
    return [_row_to_dict(r) for r in cur.fetchall()]


def _row_to_dict(row: tuple) -> dict[str, Any]:
    id_, ts, path, content_hash, prev_hash, agent_id, intent, citations_json = row
    return {
        "id": id_,
        "ts": ts,
        "path": path,
        "content_hash": content_hash,
        "prev_hash": prev_hash,
        "agent_id": agent_id,
        "intent": intent,
        "citations": json.loads(citations_json),
    }
