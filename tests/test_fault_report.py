"""Tests for scripts/fault_report.py CLI.

Uses fixture JSONL data; no live journalctl or daemon needed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(offset_minutes: int = 0) -> str:
    t = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
    return t.isoformat()


def _make_event_line(
    offset_minutes: int = 0,
    comm: str = "python3",
    fault_kind: str = "general_protection_fault",
    faulting_module: str = "libpython3.12.so.1.0",
    snapshot: str = "/srv/agents/logs/fault-events/2026-05-02T15-47-34Z-3686883.json",
) -> str:
    return json.dumps({
        "ts": _iso(offset_minutes),
        "fault_kind": fault_kind,
        "comm": comm,
        "pid": 3686883,
        "faulting_module": faulting_module,
        "snapshot": snapshot,
    })


def _write_fixture_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Import the CLI module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def inject_scripts_path():
    """Make scripts/ importable as fault_report."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    yield


def _import_fault_report():
    import importlib
    import fault_report as fr
    importlib.reload(fr)
    return fr


# ---------------------------------------------------------------------------
# generate_report — histogram shape
# ---------------------------------------------------------------------------

def test_generate_report_histogram_shape(tmp_path, monkeypatch):
    """generate_report returns correct histogram structure from fixture JSONL."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    lines = [
        _make_event_line(5, "python3", "general_protection_fault", "libpython3.12.so.1.0"),
        _make_event_line(10, "python3", "general_protection_fault", "libpython3.12.so.1.0"),
        _make_event_line(15, "rustc", "general_protection_fault", "libLLVM.so.21"),
        _make_event_line(20, "rustc", "segfault", "libLLVM.so.21"),
        _make_event_line(25, "systemd", "trap", "libsystemd-core-255.so"),
        _make_event_line(30, "python3", "oom_kill", None),
    ]
    _write_fixture_jsonl(jsonl, lines)

    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    data = fr.generate_report("2h")

    assert data["total_events"] == 6
    module_dict = dict(data["top_modules"])
    assert module_dict.get("libpython3.12.so.1.0", 0) == 2
    assert module_dict.get("libLLVM.so.21", 0) == 2

    proc_dict = dict(data["top_processes"])
    assert proc_dict.get("python3", 0) == 3
    assert proc_dict.get("rustc", 0) == 2
    assert proc_dict.get("systemd", 0) == 1


def test_generate_report_json_output(tmp_path, monkeypatch, capsys):
    """fault-report --since "24h" --json returns parseable JSON."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    lines = [_make_event_line(i * 5) for i in range(4)]
    _write_fixture_jsonl(jsonl, lines)

    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    rc = fr.main(["--since", "24h", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "total_events" in data
    assert "top_modules" in data
    assert "top_processes" in data


def test_generate_report_missing_file(tmp_path, monkeypatch):
    """Returns exit code 1 when JSONL file does not exist."""
    fr = _import_fault_report()

    missing = tmp_path / "nonexistent.jsonl"
    monkeypatch.setattr(fr, "SUMMARY_JSONL", missing)

    rc = fr.main(["--since", "24h", "--json"])
    assert rc == 1


def test_generate_report_filters_by_since(tmp_path, monkeypatch):
    """Events older than --since window are excluded."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    lines = [
        _make_event_line(10),   # 10 min ago — inside 30m window
        _make_event_line(20),   # 20 min ago — inside 30m window
        _make_event_line(90),   # 90 min ago — outside 30m window
        _make_event_line(120),  # 2h ago — outside 30m window
    ]
    _write_fixture_jsonl(jsonl, lines)
    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    data = fr.generate_report("30m")
    assert data["total_events"] == 2


# ---------------------------------------------------------------------------
# --burst — most recent burst
# ---------------------------------------------------------------------------

def test_burst_report_finds_recent_burst(tmp_path, monkeypatch):
    """--burst returns the most recent burst when threshold is met."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    # 4 events within 2 minutes — should constitute a burst (>= 3 threshold)
    lines = [
        _make_event_line(2, "python3"),
        _make_event_line(3, "python3"),
        _make_event_line(4, "rustc"),
        _make_event_line(5, "systemd"),
        # One isolated event an hour ago — not part of burst
        _make_event_line(60, "other"),
    ]
    _write_fixture_jsonl(jsonl, lines)
    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    data = fr.generate_burst_report()
    assert data["burst_found"] is True
    assert data["event_count"] >= 3


def test_burst_report_no_burst(tmp_path, monkeypatch):
    """--burst returns burst_found=False when no threshold is crossed."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    # Only 2 events total — below threshold
    lines = [
        _make_event_line(60),
        _make_event_line(120),
    ]
    _write_fixture_jsonl(jsonl, lines)
    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    data = fr.generate_burst_report()
    assert data["burst_found"] is False


def test_burst_report_json_output(tmp_path, monkeypatch, capsys):
    """--burst --json returns parseable JSON."""
    fr = _import_fault_report()

    jsonl = tmp_path / "events.jsonl"
    lines = [_make_event_line(i) for i in range(5)]
    _write_fixture_jsonl(jsonl, lines)
    monkeypatch.setattr(fr, "SUMMARY_JSONL", jsonl)

    rc = fr.main(["--burst", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "burst_found" in data
