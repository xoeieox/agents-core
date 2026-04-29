"""Tests for agents_core.process_health — registry, inventory, and detect helpers."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents_core.process_health import (
    KNOWN_PROCESSES,
    ProcessSpec,
    ProcessState,
    inventory,
    log_mtime,
    mem_key_fresh,
    pgrep_match,
    port_listening,
)


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def test_registry_loads():
    """KNOWN_PROCESSES is non-empty and all entries are valid ProcessSpec."""
    assert len(KNOWN_PROCESSES) > 0
    for spec in KNOWN_PROCESSES:
        assert isinstance(spec, ProcessSpec)
        assert spec.name
        assert spec.description
        assert callable(spec.detect)
        assert isinstance(spec.expected, bool)


def test_registry_names_unique():
    names = [s.name for s in KNOWN_PROCESSES]
    assert len(names) == len(set(names)), "Duplicate process names in KNOWN_PROCESSES"


def test_registry_has_required_v0_processes():
    names = {s.name for s in KNOWN_PROCESSES}
    required = {
        "lapis-pm-tick",
        "model-server",
        "claude-view",
        "claude-queue-runner",
        "kami_batch",
        "roomrag-indexer",
        "ops-supervisor",
    }
    assert required <= names, f"Missing processes: {required - names}"


# ---------------------------------------------------------------------------
# inventory() — failure isolation
# ---------------------------------------------------------------------------

def test_inventory_returns_one_state_per_spec():
    """inventory() always returns one ProcessState per ProcessSpec, even on failures."""
    states = inventory()
    assert len(states) == len(KNOWN_PROCESSES)
    for state in states:
        assert isinstance(state, ProcessState)


def test_inventory_isolates_failing_probe():
    """A probe that raises must not prevent other probes from running."""
    # Inject a probe that always raises into a local list
    bomb_spec = ProcessSpec(
        name="__test_bomb__",
        description="always raises",
        detect=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        expected=False,
    )

    original = list(KNOWN_PROCESSES)
    KNOWN_PROCESSES.append(bomb_spec)
    try:
        states = inventory()
    finally:
        KNOWN_PROCESSES.remove(bomb_spec)

    assert len(states) == len(original) + 1
    bomb_state = next(s for s in states if s.name == "__test_bomb__")
    assert bomb_state.running is False
    assert "probe failed" in bomb_state.notes


def test_inventory_shape_conformant():
    """Each ProcessState has all required fields with correct types."""
    states = inventory()
    for s in states:
        assert isinstance(s.name, str) and s.name
        assert isinstance(s.running, bool)
        assert s.pid is None or isinstance(s.pid, int)
        assert s.uptime_seconds is None or isinstance(s.uptime_seconds, int)
        assert s.last_advanced is None or isinstance(s.last_advanced, datetime)
        assert isinstance(s.last_advanced_source, str)
        assert isinstance(s.notes, str)


# ---------------------------------------------------------------------------
# pgrep_match
# ---------------------------------------------------------------------------

def test_pgrep_match_absent_pattern():
    """A clearly-absent pgrep pattern returns running=False."""
    running, pid, uptime = pgrep_match("__no_such_process_xyz_lapis_test__")
    assert running is False
    assert pid is None
    assert uptime is None


def test_pgrep_match_present_pattern():
    """pgrep_match finds a process we know is running (pytest itself)."""
    running, pid, uptime = pgrep_match(r"pytest")
    assert running is True
    assert isinstance(pid, int) and pid > 0
    assert uptime is None or (isinstance(uptime, int) and uptime >= 0)


# ---------------------------------------------------------------------------
# log_mtime
# ---------------------------------------------------------------------------

def test_log_mtime_missing_file():
    """Missing file returns None without raising."""
    result = log_mtime("/tmp/__no_such_file_lapis_test_xyz__.log")
    assert result is None


def test_log_mtime_existing_file(tmp_path: Path):
    """Existing file returns a timezone-aware UTC datetime."""
    f = tmp_path / "test.log"
    f.write_text("hello")
    result = log_mtime(f)
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    # Should be recent
    age = (datetime.now(tz=timezone.utc) - result).total_seconds()
    assert abs(age) < 60


# ---------------------------------------------------------------------------
# port_listening
# ---------------------------------------------------------------------------

def test_port_listening_unused_port():
    """A port with nothing listening returns False."""
    # Port 19999 is unlikely to be in use in test environments
    result = port_listening(19999)
    assert result is False


def test_port_listening_open_socket(tmp_path):
    """port_listening returns True for a port we open ourselves."""
    import socket
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    actual_port = server.getsockname()[1]

    try:
        result = port_listening(actual_port)
        assert result is True
    finally:
        server.close()


# ---------------------------------------------------------------------------
# mem_key_fresh
# ---------------------------------------------------------------------------

def _make_mem_db(tmp_path: Path) -> Path:
    """Create a minimal mem.db with the memories table."""
    db = tmp_path / "mem.db"
    con = sqlite3.connect(str(db))
    con.execute("""
        CREATE TABLE memories (
            key TEXT PRIMARY KEY,
            content TEXT,
            tags TEXT,
            updated_at TEXT
        )
    """)
    con.commit()
    con.close()
    return db


def test_mem_key_fresh_missing_db(tmp_path):
    """Missing DB path returns (False, None) without raising."""
    missing = tmp_path / "nonexistent.db"
    is_fresh, dt = mem_key_fresh("any/key", max_age_seconds=60, _db_path=missing)
    assert is_fresh is False
    assert dt is None


def test_mem_key_fresh_missing_key(tmp_path):
    """Key not in DB returns (False, None)."""
    db = _make_mem_db(tmp_path)
    is_fresh, dt = mem_key_fresh("no/such/key", max_age_seconds=60, _db_path=db)
    assert is_fresh is False
    assert dt is None


def test_mem_key_fresh_fresh_key(tmp_path):
    """A key updated just now is fresh."""
    db = _make_mem_db(tmp_path)
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    con = sqlite3.connect(str(db))
    con.execute(
        "INSERT INTO memories (key, content, tags, updated_at) VALUES (?, ?, ?, ?)",
        ("pm/last-tick", "ok", "[]", now_iso),
    )
    con.commit()
    con.close()

    is_fresh, dt = mem_key_fresh("pm/last-tick", max_age_seconds=720, _db_path=db)
    assert is_fresh is True
    assert isinstance(dt, datetime)


def test_mem_key_fresh_stale_key(tmp_path):
    """A key updated 30 minutes ago with a 10-minute window is stale."""
    db = _make_mem_db(tmp_path)
    stale_dt = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
    con = sqlite3.connect(str(db))
    con.execute(
        "INSERT INTO memories (key, content, tags, updated_at) VALUES (?, ?, ?, ?)",
        ("pm/last-tick", "ok", "[]", stale_dt.isoformat()),
    )
    con.commit()
    con.close()

    is_fresh, dt = mem_key_fresh("pm/last-tick", max_age_seconds=600, _db_path=db)
    assert is_fresh is False
    assert isinstance(dt, datetime)
