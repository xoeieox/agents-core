"""Tests for agents_core.host_fault_recorder.

All tests are pure unit tests using fixtures (no live daemon, no real journalctl).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agents_core.host_fault_recorder import (
    BurstDetector,
    BurstLevel,
    FaultEvent,
    append_summary_line,
    capture_snapshot,
    notify_burst,
    parse_fault_line,
)


# ---------------------------------------------------------------------------
# Fixtures — real kernel journal line samples
# ---------------------------------------------------------------------------

# General protection fault (traps:) — from 2026-05-02 investigation
GP_FAULT_LINE = (
    "traps: python3.12[3686883] general protection fault ip:750e1fb51587 "
    "sp:750e0d3673f0 error:0 in libpython3.12.so.1.0[750e1dc00000+2bcd000]"
)

# Segfault at — typical kernel segfault log
SEGFAULT_LINE = (
    "rustc[3686884]: segfault at 0 ip 0000750e1fb51587 sp 0000750e0d367310 "
    "error 6 in librustc_driver-83018425804cb0fc.so[750e1dc00000+2bcd000]"
)

# OOM killer — verbose form
OOM_KILL_LINE = (
    "Out of memory: Killed process 3686885 (python3) "
    "total-vm:2097152kB, anon-rss:1843200kB, file-rss:0kB"
)

# oom-kill event — compact form
OOM_EVENT_LINE = (
    "oom-kill:constraint=MEMCG,nodemask=(null),cpuset=system.slice,"
    "mems_allowed=0,global_oom,task=python3,pid=3686886,uid=1000"
)

# Unrelated kernel log lines that must not match
UNRELATED_LINES = [
    "EXT4-fs (sda1): mounted filesystem",
    "NET: Registered PF_INET6 protocol family",
    "audit: type=1400 audit(1714639654.321:123): apparmor=\"ALLOWED\"",
    "systemd[1]: Started Session 42 of user user.",
    "kauditd: hold queue overflow",
]


# ---------------------------------------------------------------------------
# parse_fault_line — correct extraction
# ---------------------------------------------------------------------------

def test_parse_gp_fault_extracts_ip_sp_error():
    ev = parse_fault_line(GP_FAULT_LINE)
    assert ev is not None
    assert ev.fault_kind == "general_protection_fault"
    assert ev.comm == "python3.12"
    assert ev.pid == 3686883
    assert ev.ip == "750e1fb51587"
    assert ev.sp == "750e0d3673f0"
    assert ev.error_code == "0"


def test_parse_gp_fault_extracts_module_offset():
    ev = parse_fault_line(GP_FAULT_LINE)
    assert ev is not None
    assert ev.faulting_module == "libpython3.12.so.1.0"
    assert "750e1dc00000" in ev.faulting_offset
    assert "2bcd000" in ev.faulting_offset


def test_parse_segfault_extracts_fields():
    ev = parse_fault_line(SEGFAULT_LINE)
    assert ev is not None
    assert ev.fault_kind == "segfault"
    assert ev.comm == "rustc"
    assert ev.pid == 3686884
    assert ev.ip is not None and "750e1fb51587" in ev.ip
    assert ev.error_code == "6"


def test_parse_segfault_extracts_module():
    ev = parse_fault_line(SEGFAULT_LINE)
    assert ev is not None
    assert ev.faulting_module is not None
    assert "librustc_driver" in ev.faulting_module


def test_parse_oom_kill_verbose():
    ev = parse_fault_line(OOM_KILL_LINE)
    assert ev is not None
    assert ev.fault_kind == "oom_kill"
    assert ev.comm == "python3"
    assert ev.pid == 3686885


def test_parse_oom_kill_compact():
    ev = parse_fault_line(OOM_EVENT_LINE)
    assert ev is not None
    assert ev.fault_kind == "oom_kill"
    assert ev.comm == "python3"
    assert ev.pid == 3686886


def test_parse_fault_line_returns_none_for_unrelated(line: str = ""):
    for line in UNRELATED_LINES:
        result = parse_fault_line(line)
        assert result is None, f"Expected None for: {line!r}, got: {result}"


def test_parse_fault_line_timestamp_is_utc():
    before = datetime.now(timezone.utc)
    ev = parse_fault_line(GP_FAULT_LINE)
    after = datetime.now(timezone.utc)
    assert ev is not None
    assert ev.captured_at_utc.tzinfo is not None
    assert before <= ev.captured_at_utc <= after


def test_parse_fault_line_journal_line_preserved():
    ev = parse_fault_line(GP_FAULT_LINE)
    assert ev is not None
    assert ev.journal_line == GP_FAULT_LINE


# ---------------------------------------------------------------------------
# capture_snapshot — atomic write + schema validation
# ---------------------------------------------------------------------------

def _make_event(comm: str = "python3", pid: int = 1234,
                fault_kind: str = "general_protection_fault") -> FaultEvent:
    return FaultEvent(
        captured_at_utc=datetime.now(timezone.utc),
        journal_line="test line",
        fault_kind=fault_kind,
        comm=comm,
        pid=pid,
        ip="0xdeadbeef",
        sp="0xcafebabe",
        error_code="0",
        faulting_module="libtest.so",
        faulting_offset="0x1000+0x200",
        cpu=0,
    )


def test_capture_snapshot_creates_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents_core.host_fault_recorder.FAULT_EVENT_DIR", tmp_path
    )
    ev = _make_event()
    snap = capture_snapshot(ev)

    assert snap.exists()
    data = json.loads(snap.read_text())
    assert data["schema_version"] == 1
    assert data["fault_kind"] == "general_protection_fault"
    assert data["process"]["comm"] == "python3"
    assert data["process"]["pid"] == 1234
    assert "system" in data
    assert "mem_kb" in data["system"]
    assert "top_rss_processes" in data
    assert "recent_dmesg" in data
    assert "mce_state" in data


def test_capture_snapshot_atomic_write(tmp_path, monkeypatch):
    """Verify no half-written files: target either doesn't exist or is fully valid JSON."""
    monkeypatch.setattr(
        "agents_core.host_fault_recorder.FAULT_EVENT_DIR", tmp_path
    )
    ev = _make_event(pid=9999)

    # Check target doesn't exist before
    utc_str = ev.captured_at_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
    expected = tmp_path / f"{utc_str}-9999.json"
    assert not expected.exists()

    snap = capture_snapshot(ev)

    # After write: target exists and is valid JSON (no partial file)
    assert snap.exists()
    data = json.loads(snap.read_text())
    assert isinstance(data, dict)
    # No .tmp- files should remain
    tmp_files = list(tmp_path.glob(".tmp-*.json"))
    assert tmp_files == [], f"Orphaned tmp files: {tmp_files}"


def test_capture_snapshot_schema_fields_present(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents_core.host_fault_recorder.FAULT_EVENT_DIR", tmp_path
    )
    ev = _make_event()
    snap = capture_snapshot(ev)
    data = json.loads(snap.read_text())

    required_top = {
        "schema_version", "captured_at_utc", "captured_at_pacific",
        "journal_line", "fault_kind", "process", "system",
        "top_rss_processes", "recent_dmesg", "mce_state",
    }
    assert required_top <= set(data.keys())

    required_proc = {"comm", "pid", "ip", "sp", "error_code",
                     "faulting_module", "faulting_offset", "cpu"}
    assert required_proc <= set(data["process"].keys())

    required_sys = {"loadavg_1", "loadavg_5", "loadavg_15", "mem_kb", "concurrent_processes"}
    assert required_sys <= set(data["system"].keys())


# ---------------------------------------------------------------------------
# append_summary_line — rolling at 10k lines
# ---------------------------------------------------------------------------

def test_append_summary_line_writes_valid_json(tmp_path, monkeypatch):
    jsonl = tmp_path / "test-events.jsonl"
    monkeypatch.setattr("agents_core.host_fault_recorder.SUMMARY_JSONL", jsonl)
    ev = _make_event()
    snap = tmp_path / "snap.json"
    snap.write_text("{}")

    append_summary_line(ev, snap)

    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["fault_kind"] == "general_protection_fault"
    assert data["comm"] == "python3"
    assert data["pid"] == 1234


def test_append_summary_line_rolls_at_10k(tmp_path, monkeypatch):
    jsonl = tmp_path / "test-events.jsonl"
    monkeypatch.setattr("agents_core.host_fault_recorder.SUMMARY_JSONL", jsonl)
    monkeypatch.setattr("agents_core.host_fault_recorder.SUMMARY_MAX_LINES", 5)

    snap = tmp_path / "snap.json"
    snap.write_text("{}")

    # Write 5 lines to hit the limit
    for i in range(5):
        ev = _make_event(pid=i + 100)
        append_summary_line(ev, snap)

    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 5

    # One more should trigger a roll
    ev = _make_event(pid=999)
    append_summary_line(ev, snap)

    # Old file should be renamed to a dated suffix
    dated_files = list(tmp_path.glob("*.jsonl"))
    assert len(dated_files) == 2, f"Expected 2 .jsonl files after roll, got: {dated_files}"

    # New file should have 1 line
    new_content = jsonl.read_text().strip().splitlines()
    assert len(new_content) == 1


# ---------------------------------------------------------------------------
# BurstDetector — threshold transitions
# ---------------------------------------------------------------------------

def _ts(offset_seconds: float = 0.0) -> FaultEvent:
    """Create a FaultEvent with a specific UTC timestamp offset from now."""
    t = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return FaultEvent(
        captured_at_utc=t,
        journal_line="x",
        fault_kind="trap",
        comm="test",
        pid=1,
        ip=None, sp=None, error_code=None,
        faulting_module=None, faulting_offset=None, cpu=None,
    )


def test_burst_none_below_threshold():
    bd = BurstDetector(window_seconds=300, normal_threshold=3, high_threshold=10)
    assert bd.add(_ts()) == BurstLevel.NONE
    assert bd.add(_ts()) == BurstLevel.NONE


def test_burst_normal_at_threshold():
    bd = BurstDetector(window_seconds=300, normal_threshold=3, high_threshold=10)
    bd.add(_ts())
    bd.add(_ts())
    level = bd.add(_ts())
    assert level == BurstLevel.NORMAL


def test_burst_high_at_threshold():
    bd = BurstDetector(window_seconds=300, normal_threshold=3, high_threshold=10)
    for _ in range(9):
        bd.add(_ts())
    level = bd.add(_ts())
    assert level == BurstLevel.HIGH


def test_burst_trims_expired_events():
    bd = BurstDetector(window_seconds=5, normal_threshold=3, high_threshold=10)
    # Add events 10 seconds in the past (outside 5-second window)
    old_event = FaultEvent(
        captured_at_utc=datetime.now(timezone.utc) - timedelta(seconds=10),
        journal_line="old",
        fault_kind="trap",
        comm="old",
        pid=1,
        ip=None, sp=None, error_code=None,
        faulting_module=None, faulting_offset=None, cpu=None,
    )
    bd.add(old_event)
    bd.add(old_event)
    bd.add(old_event)

    # These old events should be trimmed when a new recent event is added
    level = bd.add(_ts())  # only 1 event in the window now
    assert level == BurstLevel.NONE
    assert bd.recent_events_count == 1


def test_burst_none_to_normal_transition():
    """NONE → NORMAL transition happens exactly at normal_threshold."""
    bd = BurstDetector(window_seconds=300, normal_threshold=3, high_threshold=10)
    assert bd.add(_ts()) == BurstLevel.NONE
    assert bd.add(_ts()) == BurstLevel.NONE
    assert bd.add(_ts()) == BurstLevel.NORMAL
    assert bd.add(_ts()) == BurstLevel.NORMAL


# ---------------------------------------------------------------------------
# notify_burst — cooldown: two bursts within 30 min → one Pushover
# ---------------------------------------------------------------------------

def test_notify_burst_cooldown_suppresses_second(monkeypatch):
    """Two burst notifications within cooldown → only one send_notification call."""
    sent = []

    def fake_send(msg, **kwargs):
        sent.append(msg)
        return True

    def fake_store_set(*args, **kwargs):
        pass

    fake_store = MagicMock()
    fake_store.set = fake_store_set
    fake_store.close = MagicMock()

    monkeypatch.setattr("agents_core.host_fault_recorder.send_notification", fake_send)
    monkeypatch.setattr(
        "agents_core.host_fault_recorder.MemoryStore",
        lambda: fake_store,
    )
    # Reset cooldown
    import agents_core.host_fault_recorder as hfr
    hfr._LAST_PUSHOVER_TS = 0.0

    snap = Path("/tmp/fake-snap.json")
    events = [_make_event()]

    notify_burst(BurstLevel.NORMAL, events, snap)
    # Wait briefly for background thread
    time.sleep(0.1)

    notify_burst(BurstLevel.NORMAL, events, snap)
    time.sleep(0.1)

    # Only one Pushover should have been sent
    assert len(sent) == 1


def test_notify_burst_after_cooldown_sends_again(monkeypatch):
    """Third burst after 30 min cooldown window → second Pushover."""
    sent = []

    def fake_send(msg, **kwargs):
        sent.append(msg)
        return True

    fake_store = MagicMock()
    fake_store.set = MagicMock()
    fake_store.close = MagicMock()

    monkeypatch.setattr("agents_core.host_fault_recorder.send_notification", fake_send)
    monkeypatch.setattr(
        "agents_core.host_fault_recorder.MemoryStore",
        lambda: fake_store,
    )
    import agents_core.host_fault_recorder as hfr

    # Simulate last pushover was 31 minutes ago
    hfr._LAST_PUSHOVER_TS = time.monotonic() - (31 * 60)

    snap = Path("/tmp/fake-snap.json")
    events = [_make_event()]

    notify_burst(BurstLevel.HIGH, events, snap)
    time.sleep(0.1)

    assert len(sent) == 1


# ---------------------------------------------------------------------------
# notify_burst — mem.db emission and message format
# ---------------------------------------------------------------------------


class _FakeMemoryStore:
    """Minimal MemoryStore stand-in that records set() calls."""
    def __init__(self):
        self.calls: list[tuple[str, str, list]] = []

    def set(self, key: str, value: str, tags=None):
        self.calls.append((key, value, tags or []))

    def close(self):
        pass


def _patch_notify(monkeypatch):
    """Patch send_notification + MemoryStore; return (sent_msgs, store_instance)."""
    sent: list[str] = []
    store = _FakeMemoryStore()

    monkeypatch.setattr("agents_core.host_fault_recorder.send_notification",
                        lambda msg, **kw: sent.append(msg) or True)
    monkeypatch.setattr("agents_core.host_fault_recorder.MemoryStore",
                        lambda: store)
    return sent, store


def test_notify_burst_emits_mem_db_entry(monkeypatch):
    """notify_burst writes incident key to mem.db with correct prefix and tags."""
    sent, store = _patch_notify(monkeypatch)

    import agents_core.host_fault_recorder as hfr
    hfr._LAST_PUSHOVER_TS = 0.0

    notify_burst(BurstLevel.NORMAL, [_make_event()], Path("/tmp/fake-snap.json"))
    time.sleep(0.2)

    assert len(store.calls) == 1, f"Expected 1 mem.db set call, got {store.calls}"
    key, value, tags = store.calls[0]
    assert key.startswith("incident/host-fault-burst-"), (
        f"mem key should start with 'incident/host-fault-burst-', got: {key!r}"
    )
    assert "host-fault" in tags, f"Expected 'host-fault' tag, got: {tags}"
    # Incidents are tagged with the actual host they occurred on (was hardcoded
    # "starhouse"; now host-attributed since the recorder may run on any host).
    assert hfr.HOSTNAME in tags, f"Expected host tag {hfr.HOSTNAME!r}, got: {tags}"


def test_notify_burst_message_format_normal(monkeypatch):
    """NORMAL burst Pushover message contains '(NORMAL)' and 'fault-report --burst'."""
    sent, store = _patch_notify(monkeypatch)

    import agents_core.host_fault_recorder as hfr
    hfr._LAST_PUSHOVER_TS = 0.0

    notify_burst(BurstLevel.NORMAL, [_make_event()], Path("/tmp/snap.json"))
    time.sleep(0.2)

    assert len(sent) == 1
    assert "(NORMAL)" in sent[0], f"Expected '(NORMAL)' in message: {sent[0]!r}"
    assert "fault-report --burst" in sent[0], f"Expected CLI hint in message: {sent[0]!r}"


def test_notify_burst_message_format_high(monkeypatch):
    """HIGH burst Pushover message contains '(HIGH)' and drop_caches first-aid hint."""
    sent, store = _patch_notify(monkeypatch)

    import agents_core.host_fault_recorder as hfr
    hfr._LAST_PUSHOVER_TS = 0.0

    notify_burst(BurstLevel.HIGH, [_make_event() for _ in range(10)],
                 Path("/tmp/snap.json"))
    time.sleep(0.2)

    assert len(sent) == 1
    assert "(HIGH)" in sent[0], f"Expected '(HIGH)' in message: {sent[0]!r}"
    assert "drop_caches" in sent[0], f"Expected drop_caches hint in message: {sent[0]!r}"


# ---------------------------------------------------------------------------
# Smoke test: import + parse on fixture
# ---------------------------------------------------------------------------

def test_smoke_import_and_parse():
    """Basic smoke: import works, parse returns non-None for known fixture."""
    from agents_core import host_fault_recorder
    result = host_fault_recorder.parse_fault_line(GP_FAULT_LINE)
    assert result is not None
    assert result.fault_kind == "general_protection_fault"


# ---------------------------------------------------------------------------
# Incident emit routes to the mem MASTER (substrate cutover)
# ---------------------------------------------------------------------------

def test_incident_emit_off_master_posts_to_brix(monkeypatch):
    """Off-master (e.g. StarHouse): incident POSTs to the mem master via MemClient,
    NOT a divergent local sqlite. Tags are passed as a comma-separated string."""
    import agents_core.host_fault_recorder as hfr
    import agents_core.mem_client as mem_client_mod

    monkeypatch.setattr("agents_core.host_fault_recorder.send_notification", lambda *a, **k: True)
    monkeypatch.setattr(hfr, "IS_MASTER", False)
    # MemoryStore must NOT be used off-master.
    monkeypatch.setattr(hfr, "MemoryStore", lambda *a, **k: (_ for _ in ()).throw(AssertionError("local write off-master")))

    calls = []

    class FakeClient:
        def __init__(self, base_url=None, **kw):
            calls.append(("init", base_url))
        def set(self, key, content, tags="", source=""):
            calls.append(("set", key, tags))
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mem_client_mod, "MemClient", FakeClient)

    hfr._do_notify_burst(BurstLevel.NORMAL, [_make_event()], Path("/tmp/snap.json"))

    set_calls = [c for c in calls if c[0] == "set"]
    assert len(set_calls) == 1
    _, key, tags = set_calls[0]
    assert key.startswith("incident/host-fault-burst-")
    assert tags == f"host-fault,{hfr.HOSTNAME}"  # string form, host-attributed
    # MemClient was pointed at the master URL.
    assert any(c[0] == "init" and c[1] for c in calls)


def test_incident_emit_on_master_writes_local(monkeypatch):
    """On the master itself: write locally via MemoryStore (no HTTP-to-self)."""
    import agents_core.host_fault_recorder as hfr

    monkeypatch.setattr("agents_core.host_fault_recorder.send_notification", lambda *a, **k: True)
    monkeypatch.setattr(hfr, "IS_MASTER", True)

    store_calls = []
    fake_store = MagicMock()
    fake_store.set = lambda key, content, tags=None: store_calls.append((key, tags))
    monkeypatch.setattr(hfr, "MemoryStore", lambda *a, **k: fake_store)

    hfr._do_notify_burst(BurstLevel.NORMAL, [_make_event()], Path("/tmp/snap.json"))

    assert len(store_calls) == 1
    key, tags = store_calls[0]
    assert key.startswith("incident/host-fault-burst-")
    assert tags == ["host-fault", hfr.HOSTNAME]  # list form for the local store
