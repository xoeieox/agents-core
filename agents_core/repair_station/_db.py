"""SQLite persistence for repair-station registry and incident intake.

Schema:
  station_registry — self-registered stations (one row per station_id).
  station_fires    — individual fire events for n_within window checks.
  incidents        — escalated incidents with status lifecycle for Leg 2 Expert.

DB path: /srv/lapis/repair-station/repair_station.db (overridable via REPAIR_STATION_DB env).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from agents_core.room_paths import room_path

_DEFAULT_DB = room_path("repair_station")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS station_registry (
    station_id      TEXT PRIMARY KEY,
    owning_module   TEXT NOT NULL DEFAULT '',
    stable_pointer  TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    author_intent   TEXT NOT NULL,
    first_fire      TEXT NOT NULL,
    last_fire       TEXT NOT NULL,
    fire_count      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS station_fires (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id  TEXT NOT NULL,
    fired_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fires_station_time ON station_fires (station_id, fired_at);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT PRIMARY KEY,
    station_id      TEXT NOT NULL,
    stable_pointer  TEXT NOT NULL,
    error_signal    TEXT NOT NULL,
    author_intent   TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    error_signature TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',
    back_ref        TEXT DEFAULT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_station_sig
    ON incidents (station_id, error_signature, status);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status);
"""


class _DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def execute(self, sql: str, params=()):
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, params):
        with self._lock:
            return self._conn.executemany(sql, params)

    def commit(self):
        with self._lock:
            self._conn.commit()


_instances: dict[Path, _DB] = {}
_instances_lock = threading.Lock()


def get_db(path: Path | None = None) -> _DB:
    """Return a module-level singleton DB for the given path."""
    resolved = path or _DEFAULT_DB
    with _instances_lock:
        if resolved not in _instances:
            _instances[resolved] = _DB(resolved)
        return _instances[resolved]
