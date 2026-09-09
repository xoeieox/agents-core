"""Host Fault Recorder — capture per-event kernel fault context and detect bursts.

Tails `journalctl -fk` for crash events (traps:, general protection fault,
segfault at, Out of memory / oom-kill). On each match:
  1. Writes a JSON snapshot to /srv/agents/logs/fault-events/<utc>-<pid>.json
  2. Appends a summary line to /srv/lapis/memory/host-fault-events.jsonl (rolls at 10k lines)
  3. Checks burst window; sends Pushover + emits mem.db entry if threshold crossed

Read-only: no auto-drop_caches, no service restart, no host-state writes.

Schema v2 (2026-06-12):
  - process.cmdline: list[str] | null — victim process argv (from /proc/<pid>/cmdline).
    null if process was reaped (common in OOM) or unreachable. Best-effort.
  - process.cmdline_unavailable_reason: str | null — when cmdline is null, explains why:
    "no_such_pid", "permission", "empty", or other reason. Absent if cmdline is available.
  - top_rss_processes[].cmdline: list[str] | null — cmdline per top-RSS process.
    Reliable path since processes listed are still alive. null if unreadable.
  - top_rss_processes[].cmdline_unavailable_reason: str | null — reason if cmdline is null.
    Absent if cmdline is available.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

from agents_core.mem import HOSTNAME, IS_MASTER, MEM_MASTER_URL, MemoryStore
from agents_core.notify import Priority, send_notification
from agents_core.room_paths import room_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FAULT_EVENT_DIR = Path("/srv/agents/logs/fault-events")
SUMMARY_JSONL = room_path("memory.host_fault_events")
PACIFIC_OFFSET = timedelta(hours=-7)  # PDT; adjust for PST (-8) when in effect

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FaultEvent:
    captured_at_utc: datetime
    journal_line: str
    fault_kind: Literal["general_protection_fault", "segfault", "trap", "oom_kill"]
    comm: str
    pid: int
    ip: str | None
    sp: str | None
    error_code: str | None
    faulting_module: str | None
    faulting_offset: str | None
    cpu: int | None


class BurstLevel(Enum):
    NONE = 0
    NORMAL = 1   # >= 3 events in trailing 5 min
    HIGH = 2     # >= 10 events in trailing 5 min


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Patterns compiled once at module load.

# General protection fault:
#   process[pid]: traps: NAME[pid] trap general protection fault[...]
#   OR kernel log:  traps: NAME[pid] general protection fault ...
_RE_TRAP = re.compile(
    r"traps:\s+(?P<comm>\S+)\[(?P<pid>\d+)\].*?general protection fault.*?"
    r"(?:ip:(?P<ip>[0-9a-f]+)\s+sp:(?P<sp>[0-9a-f]+)\s+error:(?P<error>[0-9a-f]+))?",
    re.IGNORECASE,
)

# Segfault:
#   NAME[pid]: segfault at ADDR ip 0xADDR sp 0xADDR error N in MODULE[OFFSET]
_RE_SEGFAULT = re.compile(
    r"(?P<comm>\S+)\[(?P<pid>\d+)\].*?segfault at\s+\S+\s+"
    r"(?:ip\s+(?P<ip>[0-9a-fx]+)\s+sp\s+(?P<sp>[0-9a-fx]+)\s+error\s+(?P<error>\d+))?"
    r"(?:\s+in\s+(?P<module>\S+))?",
    re.IGNORECASE,
)

# OOM killer:
#   Out of memory: Killed process PID (COMM)
_RE_OOM_KILL = re.compile(
    r"Out of memory:\s+Killed process\s+(?P<pid>\d+)\s+\((?P<comm>[^)]+)\)",
    re.IGNORECASE,
)

# oom-kill event:
#   oom-kill:constraint=MEMCG,nodemask=...,task=NAME,pid=PID,...
_RE_OOM_EVENT = re.compile(
    r"oom-kill:.*?task=(?P<comm>[^,]+).*?pid=(?P<pid>\d+)",
    re.IGNORECASE,
)

# Module+offset extractor from segfault lines:
#   in libfoo.so[0xADDR+0xSIZE]  or  /path/to/libfoo.so[0x...]
_RE_MODULE_OFFSET = re.compile(
    r"in\s+(?P<path>\S+?)\[(?P<offset>[^\]]+)\]"
)

# CPU field from trap lines:  CPU: N
_RE_CPU = re.compile(r"\bCPU:\s*(?P<cpu>\d+)", re.IGNORECASE)

# IP/SP from trap lines (alternate format):  RIP: 0010:addr  RSP: addr
_RE_RIP = re.compile(r"RIP:\s*[0-9a-f]+:(?P<ip>[0-9a-f]+)", re.IGNORECASE)
_RE_RSP = re.compile(r"RSP:\s*(?P<sp>[0-9a-f]+)", re.IGNORECASE)

# Inline ip:X sp:X error:X — appears in compact traps/segfault lines
_RE_IP_SP_INLINE = re.compile(
    r"\bip:(?P<ip>[0-9a-f]+)\s+sp:(?P<sp>[0-9a-f]+)\s+error:(?P<error>[0-9a-f]+)",
    re.IGNORECASE,
)


def _extract_module_offset(line: str) -> tuple[str | None, str | None]:
    m = _RE_MODULE_OFFSET.search(line)
    if not m:
        return None, None
    path = m.group("path")
    module = Path(path).name
    offset = m.group("offset")
    return module, offset


def _extract_cpu(line: str) -> int | None:
    m = _RE_CPU.search(line)
    return int(m.group("cpu")) if m else None


def parse_fault_line(line: str) -> FaultEvent | None:
    """Parse a single kernel journal line into a FaultEvent, or return None."""
    now = datetime.now(timezone.utc)

    # --- General protection fault (traps:) ---
    if "general protection fault" in line and "traps:" in line:
        m = _RE_TRAP.search(line)
        if m:
            module, offset = _extract_module_offset(line)
            # Try inline ip:X sp:X error:X format first, then RIP/RSP alternate
            ip = m.group("ip")
            sp = m.group("sp")
            error_code = m.group("error")
            inline = _RE_IP_SP_INLINE.search(line)
            if inline:
                ip = ip or inline.group("ip")
                sp = sp or inline.group("sp")
                error_code = error_code or inline.group("error")
            if not ip:
                rim = _RE_RIP.search(line)
                if rim:
                    ip = rim.group("ip")
            if not sp:
                rspm = _RE_RSP.search(line)
                if rspm:
                    sp = rspm.group("sp")
            return FaultEvent(
                captured_at_utc=now,
                journal_line=line,
                fault_kind="general_protection_fault",
                comm=m.group("comm"),
                pid=int(m.group("pid")),
                ip=ip,
                sp=sp,
                error_code=error_code,
                faulting_module=module,
                faulting_offset=offset,
                cpu=_extract_cpu(line),
            )

    # --- Trap (without "general protection fault" keyword, just "traps:") ---
    if "traps:" in line and "general protection fault" not in line:
        m = _RE_TRAP.search(line)
        if not m:
            # Try a simpler pattern: "traps: NAME[PID]"
            m2 = re.search(r"traps:\s+(?P<comm>\S+)\[(?P<pid>\d+)\]", line)
            if m2:
                module, offset = _extract_module_offset(line)
                return FaultEvent(
                    captured_at_utc=now,
                    journal_line=line,
                    fault_kind="trap",
                    comm=m2.group("comm"),
                    pid=int(m2.group("pid")),
                    ip=None,
                    sp=None,
                    error_code=None,
                    faulting_module=module,
                    faulting_offset=offset,
                    cpu=_extract_cpu(line),
                )

    # --- Segfault ---
    if "segfault at" in line:
        m = _RE_SEGFAULT.search(line)
        if m:
            module, offset = _extract_module_offset(line)
            return FaultEvent(
                captured_at_utc=now,
                journal_line=line,
                fault_kind="segfault",
                comm=m.group("comm"),
                pid=int(m.group("pid")),
                ip=m.group("ip"),
                sp=m.group("sp"),
                error_code=m.group("error"),
                faulting_module=module or m.group("module"),
                faulting_offset=offset,
                cpu=_extract_cpu(line),
            )

    # --- OOM Kill (verbose form) ---
    if "Out of memory" in line and "Killed process" in line:
        m = _RE_OOM_KILL.search(line)
        if m:
            return FaultEvent(
                captured_at_utc=now,
                journal_line=line,
                fault_kind="oom_kill",
                comm=m.group("comm"),
                pid=int(m.group("pid")),
                ip=None,
                sp=None,
                error_code=None,
                faulting_module=None,
                faulting_offset=None,
                cpu=None,
            )

    # --- oom-kill event (compact form) ---
    if "oom-kill:" in line:
        m = _RE_OOM_EVENT.search(line)
        if m:
            return FaultEvent(
                captured_at_utc=now,
                journal_line=line,
                fault_kind="oom_kill",
                comm=m.group("comm"),
                pid=int(m.group("pid")),
                ip=None,
                sp=None,
                error_code=None,
                faulting_module=None,
                faulting_offset=None,
                cpu=None,
            )

    return None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def _read_cmdline(pid: int) -> tuple[list[str] | None, str | None]:
    """Read /proc/<pid>/cmdline (NUL-separated args).

    Returns (cmdline, unavailable_reason) where:
    - cmdline is list[str] if readable, None otherwise
    - unavailable_reason is None if cmdline available, else a reason string

    Never raises; always best-effort.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read()
            if not data:
                return None, "empty"
            # NUL-separated; strip final NUL and split
            cmdline = data.rstrip(b'\x00').split(b'\x00')
            return [arg.decode('utf-8', errors='replace') for arg in cmdline], None
    except FileNotFoundError:
        return None, "no_such_pid"
    except PermissionError:
        return None, "permission"
    except Exception as e:
        return None, f"read_error:{type(e).__name__}"


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        result[key] = int(parts[1])
                    except ValueError:
                        pass
    except OSError:
        pass
    return result


def _read_loadavg() -> tuple[float, float, float]:
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0, 0.0


def _count_processes(names: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        out = subprocess.run(
            ["ps", "-eo", "comm="],
            capture_output=True, text=True, timeout=5,
        )
        comms = out.stdout.splitlines()
        for name in names:
            if name == "python3_total":
                result[name] = sum(1 for c in comms if c.startswith("python3"))
            else:
                result[name] = sum(1 for c in comms if c == name)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def _top_rss_processes(n: int = 10) -> list[dict]:
    results = []
    try:
        out = subprocess.run(
            ["ps", "-eo", "comm=,pid=,rss=", "--sort=-rss"],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.splitlines()[:n]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pid = int(parts[1])
                    cmdline, unavailable_reason = _read_cmdline(pid)
                    entry = {
                        "comm": parts[0],
                        "pid": pid,
                        "rss_kb": int(parts[2]),
                        "cmdline": cmdline,
                    }
                    if unavailable_reason is not None:
                        entry["cmdline_unavailable_reason"] = unavailable_reason
                    results.append(entry)
                except ValueError:
                    pass
    except (OSError, subprocess.TimeoutExpired):
        pass
    return results


def _recent_dmesg(since_seconds: int = 60) -> list[str]:
    try:
        out = subprocess.run(
            ["dmesg", "--since", f"-{since_seconds}s"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Fallback: journalctl -k
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", f"{since_seconds} seconds ago", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return []


def _mce_state() -> dict:
    mce_base = Path("/sys/devices/system/machinecheck")
    if mce_base.exists():
        return {"available": True, "path": str(mce_base)}
    return {"available": False, "note": "no /sys/devices/system/machinecheck/* on this host"}


def capture_snapshot(event: FaultEvent, output_dir: Path | None = None) -> Path:
    """Write per-event JSON snapshot atomically. Returns the snapshot path.

    `output_dir` overrides the module-level FAULT_EVENT_DIR constant (default).
    Callers — and tests — can pass an explicit directory so I/O is redirected
    without monkeypatching module globals (overridable-path convention, cf.
    GPUQueue/TargetStore/MemoryStore).
    """
    out_dir = Path(output_dir) if output_dir is not None else FAULT_EVENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    utc_str = event.captured_at_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
    pacific_dt = event.captured_at_utc + PACIFIC_OFFSET
    pacific_str = pacific_dt.strftime("%Y-%m-%dT%H:%M:%S") + "-07:00"
    utc_iso = event.captured_at_utc.strftime("%Y-%m-%dT%H:%M:%S.") + \
              f"{event.captured_at_utc.microsecond // 1000:03d}Z"

    target = out_dir / f"{utc_str}-{event.pid}.json"

    load1, load5, load15 = _read_loadavg()
    meminfo = _read_meminfo()
    mem_subset = {
        k: meminfo.get(k, 0) for k in
        ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapFree", "Dirty")
    }

    cmdline, cmdline_unavailable_reason = _read_cmdline(event.pid)

    process_block = {
        "comm": event.comm,
        "pid": event.pid,
        "ip": event.ip,
        "sp": event.sp,
        "error_code": event.error_code,
        "faulting_module": event.faulting_module,
        "faulting_offset": event.faulting_offset,
        "cpu": event.cpu,
        "cmdline": cmdline,
    }
    if cmdline_unavailable_reason is not None:
        process_block["cmdline_unavailable_reason"] = cmdline_unavailable_reason

    snapshot = {
        "schema_version": 2,
        "captured_at_utc": utc_iso,
        "captured_at_pacific": pacific_str,
        "journal_line": event.journal_line,
        "fault_kind": event.fault_kind,
        "process": process_block,
        "system": {
            "loadavg_1": load1,
            "loadavg_5": load5,
            "loadavg_15": load15,
            "mem_kb": mem_subset,
            "concurrent_processes": _count_processes(
                ["claude", "shaped_runner", "python3_total"]
            ),
        },
        "top_rss_processes": _top_rss_processes(10),
        "recent_dmesg": _recent_dmesg(60),
        "mce_state": _mce_state(),
    }

    # Atomic write: tmpfile in same directory + rename
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snapshot, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return target


# ---------------------------------------------------------------------------
# Summary JSONL
# ---------------------------------------------------------------------------

SUMMARY_MAX_LINES = 10_000


def append_summary_line(event: FaultEvent, snapshot_path: Path,
                        jsonl_path: Path | None = None) -> None:
    """Append one line to the rolling JSONL summary. Rolls at SUMMARY_MAX_LINES.

    `jsonl_path` overrides the module-level SUMMARY_JSONL constant (default).
    Callers — and tests — can pass an explicit path so I/O is redirected
    without monkeypatching module globals (overridable-path convention, cf.
    GPUQueue/TargetStore/MemoryStore).
    """
    summary_path = Path(jsonl_path) if jsonl_path is not None else SUMMARY_JSONL
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Roll if over limit
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                count = sum(1 for _ in f)
            if count >= SUMMARY_MAX_LINES:
                date_tag = event.captured_at_utc.strftime("%Y-%m-%d")
                rolled = summary_path.with_suffix(f".{date_tag}.jsonl")
                summary_path.rename(rolled)
        except OSError as exc:
            logger.warning("summary roll check failed: %s", exc)

    line = json.dumps({
        "ts": event.captured_at_utc.isoformat(),
        "fault_kind": event.fault_kind,
        "comm": event.comm,
        "pid": event.pid,
        "faulting_module": event.faulting_module,
        "snapshot": str(snapshot_path),
    })
    try:
        with open(summary_path, "a") as f:
            f.write(line + "\n")
    except OSError as exc:
        logger.error("failed to append summary line: %s", exc)


# ---------------------------------------------------------------------------
# Burst detection
# ---------------------------------------------------------------------------


class BurstDetector:
    def __init__(
        self,
        window_seconds: int = 300,
        normal_threshold: int = 3,
        high_threshold: int = 10,
    ):
        self.window_seconds = window_seconds
        self.normal_threshold = normal_threshold
        self.high_threshold = high_threshold
        # Store FaultEvent objects so callers can retrieve them for Pushover messages
        self._window: deque[FaultEvent] = deque()

    def add(self, event: FaultEvent) -> BurstLevel:
        """Add an event and return the current burst level."""
        now_ts = event.captured_at_utc.timestamp()
        self._window.append(event)
        # Trim expired entries
        cutoff = now_ts - self.window_seconds
        while self._window and self._window[0].captured_at_utc.timestamp() < cutoff:
            self._window.popleft()

        count = len(self._window)
        if count >= self.high_threshold:
            return BurstLevel.HIGH
        if count >= self.normal_threshold:
            return BurstLevel.NORMAL
        return BurstLevel.NONE

    @property
    def recent_events(self) -> list[FaultEvent]:
        return list(self._window)

    @property
    def recent_events_count(self) -> int:
        return len(self._window)


# ---------------------------------------------------------------------------
# Pushover notification
# ---------------------------------------------------------------------------

_LAST_PUSHOVER_TS: float = 0.0
_PUSHOVER_COOLDOWN = 30 * 60  # 30 minutes
_PUSHOVER_LOCK = threading.Lock()


def _do_notify_burst(level: BurstLevel, recent_events: list[FaultEvent],
                     latest_snapshot: Path) -> None:
    """Fire-and-forget: send Pushover + write mem entry. Not called if in cooldown."""
    count = len(recent_events)
    modules = list(dict.fromkeys(
        e.faulting_module or e.comm for e in recent_events
        if e.faulting_module or e.comm
    ))[:5]
    modules_str = ", ".join(modules)
    snap_name = latest_snapshot.name

    if level == BurstLevel.HIGH:
        title = "StarHouse fault STORM"
        body = (
            f"StarHouse fault STORM (HIGH): {count} events in 5 min. "
            f"Modules: {modules_str}. "
            f"`sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` is the documented first-aid. "
            f"`fault-report --burst` for full context."
        )
        priority = Priority.NORMAL  # Per spec: HIGH storms still use NORMAL priority
    else:
        title = "StarHouse fault burst"
        body = (
            f"StarHouse fault burst (NORMAL): {count} events in 5 min. "
            f"Modules: {modules_str}. "
            f"Latest snapshot: {snap_name}. "
            f"`fault-report --burst` for details."
        )
        priority = Priority.NORMAL

    # Fire-and-forget — never block on network I/O
    try:
        send_notification(body, title=title, priority=priority)
    except Exception as exc:
        logger.warning("Pushover send failed (non-fatal): %s", exc)

    # Emit incident entry to the mem MASTER. host-fault-recorder records the LOCAL
    # host's faults but the substrate is BRIX-canonical, so off-master hosts (e.g.
    # StarHouse) must POST to the BRIX mem-server rather than write a divergent
    # local sqlite. On the master itself, write locally (avoids HTTP-to-self and a
    # mem-server-up dependency). Tag with the actual host so incidents are attributable.
    ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    mem_key = f"incident/host-fault-burst-{ts_str}"
    summary = (
        f"{level.name} burst: {count} events in 5 min. "
        f"Modules: {modules_str}. "
        f"Latest snapshot: {latest_snapshot}."
    )
    try:
        if IS_MASTER:
            store = MemoryStore()
            store.set(mem_key, summary, tags=["host-fault", HOSTNAME])
            store.close()
        else:
            # MemClient.set takes tags as a comma-separated string.
            from agents_core.mem_client import MemClient

            base_url = os.environ.get("MEM_SERVER") or MEM_MASTER_URL
            with MemClient(base_url=base_url) as client:
                client.set(mem_key, summary, tags=f"host-fault,{HOSTNAME}")
    except Exception as exc:
        logger.warning("mem incident emit failed (non-fatal): %s", exc)


def notify_burst(
    level: BurstLevel,
    recent_events: list[FaultEvent],
    latest_snapshot: Path,
) -> None:
    """Send Pushover + mem.db entry with 30-min cooldown between repeat alerts."""
    global _LAST_PUSHOVER_TS
    now = time.monotonic()
    with _PUSHOVER_LOCK:
        if now - _LAST_PUSHOVER_TS < _PUSHOVER_COOLDOWN:
            logger.info("Pushover cooldown active, suppressing burst notification")
            return
        _LAST_PUSHOVER_TS = now

    # Run in background thread so network I/O never blocks event capture
    t = threading.Thread(
        target=_do_notify_burst,
        args=(level, list(recent_events), latest_snapshot),
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> None:
    """Main loop. Tails journalctl -fk, processes events, never returns."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("host-fault-recorder starting")

    FAULT_EVENT_DIR.mkdir(parents=True, exist_ok=True)
    burst = BurstDetector()

    try:
        proc = subprocess.Popen(
            ["journalctl", "-fk", "-o", "cat", "--since", "now"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        logger.error("failed to start journalctl: %s", exc)
        raise

    logger.info("tailing journalctl -fk")

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            try:
                event = parse_fault_line(line)
            except Exception as exc:
                logger.warning("parse error (non-fatal): %s | line: %.200s", exc, line)
                continue

            if event is None:
                continue

            logger.info(
                "fault event: %s pid=%d comm=%s module=%s",
                event.fault_kind, event.pid, event.comm, event.faulting_module,
            )

            snap_path: Path | None = None
            try:
                snap_path = capture_snapshot(event)
                logger.info("snapshot: %s", snap_path)
            except Exception as exc:
                logger.error("snapshot write failed (non-fatal): %s", exc)

            try:
                if snap_path is not None:
                    append_summary_line(event, snap_path)
            except Exception as exc:
                logger.error("summary append failed (non-fatal): %s", exc)

            try:
                level = burst.add(event)
                if level != BurstLevel.NONE and snap_path is not None:
                    notify_burst(level, burst.recent_events, snap_path)
            except Exception as exc:
                logger.error("burst detection failed (non-fatal): %s", exc)

    except Exception as exc:
        logger.exception("main loop crashed: %s", exc)
        raise
    finally:
        proc.terminate()
        logger.info("host-fault-recorder stopped")
