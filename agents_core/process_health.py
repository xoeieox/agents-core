"""StarHouse process-health inventory — passive read-only probes.

Provides a declarative registry of known long-running StarHouse processes
and an inventory() function that returns live state for each one. All probes
are independently failure-isolated: a probe that raises only affects its own
ProcessState entry (running=False, notes describe the error).

Usage:
    from agents_core.process_health import inventory, KNOWN_PROCESSES
    states = inventory()
    for s in states:
        print(s.name, s.running, s.notes)

This module is read-only. No auto-restart, no kill, no write actions.
"""

from __future__ import annotations

import socket
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ProcessState:
    name: str
    running: bool
    pid: int | None = None
    uptime_seconds: int | None = None
    last_advanced: datetime | None = None
    last_advanced_source: str = ""
    notes: str = ""


@dataclass
class ProcessSpec:
    name: str
    description: str
    detect: Callable[[], ProcessState]
    expected: bool = True


# Tailscale-assigned IP for the StarHouse host. Update here if Tailscale reassigns.
_STARHOUSE_TAILSCALE_IP = "203.0.113.12"


# ---------------------------------------------------------------------------
# Detect helpers — composable primitives
# ---------------------------------------------------------------------------

def pgrep_match(pattern: str) -> tuple[bool, int | None, int | None]:
    """Return (running, pid, uptime_seconds) by pattern-matching running processes.

    Uses ``pgrep -f`` so the pattern can match anywhere in the command line.
    Returns uptime_seconds via ``ps -o etimes=``.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return False, None, None

        pids = result.stdout.strip().splitlines()
        pid = int(pids[0])

        # Elapsed time in seconds for the first matched pid
        ps_result = subprocess.run(
            ["ps", "-o", "etimes=", "-p", str(pid)],
            capture_output=True,
            text=True,
        )
        uptime: int | None = None
        if ps_result.returncode == 0 and ps_result.stdout.strip():
            uptime = int(ps_result.stdout.strip())

        return True, pid, uptime
    except Exception:
        return False, None, None


def log_mtime(path: str | Path) -> datetime | None:
    """Return the mtime of a log file as a timezone-aware UTC datetime.

    Returns None if the file does not exist or is not accessible; does not raise.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        return None


def port_listening(port: int) -> bool:
    """Return True if any process is listening on the given TCP port.

    Tries the Tailscale-bound interface first, then localhost. Does not raise.
    """
    try:
        for host in (_STARHOUSE_TAILSCALE_IP, "127.0.0.1"):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((host, port)) == 0:
                    return True
        return False
    except Exception:
        return False


def mem_key_fresh(
    key: str,
    max_age_seconds: int,
    _db_path: Path | None = None,
) -> tuple[bool, datetime | None]:
    """Return (is_fresh, last_updated_datetime) for a mem.db key.

    is_fresh is True if the key exists and was updated within max_age_seconds.
    last_updated_datetime is the recorded updated_at timestamp (UTC), or None.
    Does not raise.

    _db_path overrides the default /data/memory/mem.db (used in tests).
    """
    db_path = _db_path if _db_path is not None else Path("/data/memory/mem.db")
    try:
        if not db_path.exists():
            return False, None
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # Queries the memories table directly rather than through
            # agents_core.mem.MemoryStore to keep probe isolation clean and
            # avoid circular-import risk. Schema assumption: memories(updated_at TEXT
            # ISO-8601). If MemoryStore's table shape changes, update this query.
            cur = con.execute(
                "SELECT updated_at FROM memories WHERE key = ?", (key,)
            )
            row = cur.fetchone()
        finally:
            con.close()

        if row is None:
            return False, None

        updated_at_str: str = row[0]
        # updated_at is stored as ISO-8601 string; parse it
        try:
            dt = datetime.fromisoformat(updated_at_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return False, None

        now = datetime.now(tz=timezone.utc)
        age_seconds = (now - dt).total_seconds()
        return age_seconds <= max_age_seconds, dt
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Per-process detect functions
# ---------------------------------------------------------------------------

def _detect_lapis_pm_tick() -> ProcessState:
    is_fresh, last_dt = mem_key_fresh("pm/last-tick", max_age_seconds=720)
    if last_dt is None:
        return ProcessState(
            name="lapis-pm-tick",
            running=False,
            notes="pm/last-tick key not found in mem.db",
        )
    age_min = int((datetime.now(tz=timezone.utc) - last_dt).total_seconds() / 60)
    return ProcessState(
        name="lapis-pm-tick",
        running=is_fresh,
        last_advanced=last_dt,
        last_advanced_source="mem key: pm/last-tick",
        notes=f"pm/last-tick {'fresh' if is_fresh else 'stale'} ({age_min}min ago)",
    )


def _detect_model_server() -> ProcessState:
    running, pid, uptime = pgrep_match(r"llama-server")
    listening = port_listening(8081)
    # Both pgrep and port must agree on running; port is authoritative for "can serve"
    is_running = running and listening
    notes_parts = []
    if listening:
        notes_parts.append("port 8081 listening")
    else:
        notes_parts.append("port 8081 not listening")
    if not running:
        notes_parts.append("pgrep: no llama-server process")

    # Use pgrep for pid/uptime; no file log (journald-only)
    return ProcessState(
        name="model-server",
        running=is_running,
        pid=pid if running else None,
        uptime_seconds=uptime,
        notes="; ".join(notes_parts),
    )


def _detect_claude_view() -> ProcessState:
    listening = port_listening(8410)
    return ProcessState(
        name="claude-view",
        running=listening,
        notes="port 8410 listening" if listening else "port 8410 not listening",
    )


def _detect_claude_queue_runner() -> ProcessState:
    running, pid, uptime = pgrep_match(r"agents_core\.claude_queue_runner")
    mtime = log_mtime("/srv/lapis/claude-queue/history.jsonl")
    notes_parts = []
    if running:
        notes_parts.append(f"pgrep matched pid {pid}")
    else:
        notes_parts.append("pgrep: no claude_queue_runner process")
    return ProcessState(
        name="claude-queue-runner",
        running=running,
        pid=pid,
        uptime_seconds=uptime,
        last_advanced=mtime,
        last_advanced_source="log mtime: /srv/lapis/claude-queue/history.jsonl",
        notes="; ".join(notes_parts),
    )


def _detect_ops_supervisor() -> ProcessState:
    # The ops-layer supervisor is not yet deployed as a persistent daemon (post-lift).
    # When deployed, it will match this pgrep pattern.
    running, pid, uptime = pgrep_match(r"ops_layer_supervisor|haiku_supervisor")
    return ProcessState(
        name="ops-supervisor",
        running=running,
        pid=pid,
        uptime_seconds=uptime,
        notes=(
            f"pgrep matched pid {pid}"
            if running
            else "pgrep: no ops-supervisor process (not yet deployed as daemon)"
        ),
    )


def _detect_kami_batch() -> ProcessState:
    running, pid, uptime = pgrep_match(r"kami_batch")
    return ProcessState(
        name="kami_batch",
        running=running,
        pid=pid,
        uptime_seconds=uptime,
        notes=(
            f"pgrep matched pid {pid}"
            if running
            else "not running (expected only during nightly window or manual invoke)"
        ),
    )


def _detect_roomrag_indexer() -> ProcessState:
    running, pid, uptime = pgrep_match(r"roomrag serve")
    mtime = log_mtime("/srv/agents/logs/room-rag-reindex.log")
    notes_parts = []
    if running:
        notes_parts.append(f"pgrep matched pid {pid}")
    else:
        notes_parts.append("pgrep: no roomrag process")
    return ProcessState(
        name="roomrag-indexer",
        running=running,
        pid=pid,
        uptime_seconds=uptime,
        last_advanced=mtime,
        last_advanced_source="log mtime: /srv/agents/logs/room-rag-reindex.log",
        notes="; ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KNOWN_PROCESSES: list[ProcessSpec] = [
    ProcessSpec(
        name="lapis-pm-tick",
        description="Lapis PM tick orchestration — perceive/score/decide loop",
        detect=_detect_lapis_pm_tick,
        expected=True,
    ),
    ProcessSpec(
        name="model-server",
        description="qwen3.6-35b-a3b llama-server on port 8081",
        detect=_detect_model_server,
        expected=True,
    ),
    ProcessSpec(
        name="claude-view",
        description="claude-view session dashboard (docker, port 8410)",
        detect=_detect_claude_view,
        expected=True,
    ),
    ProcessSpec(
        name="claude-queue-runner",
        description="ClaudeQueue bounded-concurrency claude -p executor",
        detect=_detect_claude_queue_runner,
        expected=True,
    ),
    ProcessSpec(
        name="ops-supervisor",
        description="Ops-layer supervisor daemon (post-lift)",
        detect=_detect_ops_supervisor,
        expected=False,  # not yet deployed as persistent daemon
    ),
    ProcessSpec(
        name="kami_batch",
        description="Kami batch ingestion (nightly window or manual)",
        detect=_detect_kami_batch,
        expected=False,  # only expected during nightly window or manual invocation
    ),
    ProcessSpec(
        name="roomrag-indexer",
        description="RoomRAG indexer server (scheduled at 22:45 + 08:00 PT)",
        detect=_detect_roomrag_indexer,
        expected=False,  # scheduled, not always running — no schedule-awareness in v0
    ),
]


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def inventory(
    process_list: list[ProcessSpec] | None = None,
) -> list[ProcessState]:
    """Run all detect callables and return one ProcessState per ProcessSpec.

    Per-probe failures are caught and returned as running=False with a
    descriptive notes field. One probe raising never affects others.

    Args:
        process_list: Registry to use. Defaults to KNOWN_PROCESSES. Pass an
            explicit list in tests to avoid mutating the module-level registry.
    """
    specs = process_list if process_list is not None else KNOWN_PROCESSES
    results: list[ProcessState] = []
    for spec in specs:
        try:
            state = spec.detect()
        except Exception as exc:
            state = ProcessState(
                name=spec.name,
                running=False,
                notes=f"probe failed: {exc}",
            )
        results.append(state)
    return results
