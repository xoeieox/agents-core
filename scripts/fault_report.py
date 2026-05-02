#!/usr/bin/env python3
"""fault-report — summarize host fault events from the rolling JSONL log.

Usage:
    fault-report                      # last 24h, human table
    fault-report --since "1h"
    fault-report --since "7d"
    fault-report --burst              # most recent burst summary + snapshot path
    fault-report --json               # machine-readable output

Exit codes:
    0 — report generated successfully
    1 — log file missing or unreadable
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

SUMMARY_JSONL = Path("/srv/lapis/memory/host-fault-events.jsonl")
FAULT_EVENT_DIR = Path("/srv/agents/logs/fault-events")

BURST_WINDOW = timedelta(minutes=5)
NORMAL_THRESHOLD = 3
HIGH_THRESHOLD = 10


def parse_since(since_str: str) -> datetime:
    """Parse '24h', '1h', '7d', '30m' etc. into a UTC cutoff datetime."""
    since_str = since_str.strip()
    multipliers = {"m": 60, "h": 3600, "d": 86400}
    if since_str[-1].lower() in multipliers:
        try:
            value = float(since_str[:-1])
            seconds = value * multipliers[since_str[-1].lower()]
            return datetime.now(timezone.utc) - timedelta(seconds=seconds)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse --since value: {since_str!r}. Use e.g. '24h', '1h', '7d'.")


def load_events(since: datetime) -> list[dict]:
    """Load events from JSONL that fall after `since`."""
    if not SUMMARY_JSONL.exists():
        return []
    events = []
    try:
        with open(SUMMARY_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    ts_str = ev.get("ts", "")
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= since:
                        ev["_ts"] = ts
                        events.append(ev)
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return []
    return events


def compute_burst_windows(events: list[dict]) -> list[tuple[datetime, int, dict | None]]:
    """Compute sliding 5-min windows. Returns list of (window_start, count, best_event)."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda e: e["_ts"])
    windows = []
    for i, ev in enumerate(sorted_events):
        window_start = ev["_ts"]
        window_end = window_start + BURST_WINDOW
        count = sum(1 for e in sorted_events if window_start <= e["_ts"] < window_end)
        windows.append((window_start, count, ev))
    return windows


def find_peak_burst(events: list[dict]) -> tuple[int, datetime | None, dict | None]:
    """Return (peak_count, peak_ts, peak_event)."""
    windows = compute_burst_windows(events)
    if not windows:
        return 0, None, None
    peak_ts, peak_count, peak_ev = max(windows, key=lambda w: w[1])
    return peak_count, peak_ts, peak_ev


def find_most_recent_burst(events: list[dict]) -> tuple[list[dict], datetime | None]:
    """Return the most recent burst window (>= NORMAL_THRESHOLD) events + its start time."""
    if not events:
        return [], None
    windows = compute_burst_windows(events)
    bursts = [(ts, count, ev) for ts, count, ev in windows if count >= NORMAL_THRESHOLD]
    if not bursts:
        return [], None
    # Most recent burst start
    latest_ts, _, _ = max(bursts, key=lambda w: w[0])
    window_end = latest_ts + BURST_WINDOW
    burst_events = [e for e in events if latest_ts <= e["_ts"] < window_end]
    return burst_events, latest_ts


def get_snapshot_mem_available(snapshot_path: str) -> int | None:
    """Read MemAvailable from a snapshot JSON file. Returns KB or None."""
    try:
        data = json.loads(Path(snapshot_path).read_text())
        return data.get("system", {}).get("mem_kb", {}).get("MemAvailable")
    except (OSError, json.JSONDecodeError):
        return None


def generate_report(since_str: str = "24h") -> dict:
    """Generate full report data as a dict."""
    since = parse_since(since_str)
    events = load_events(since)

    total = len(events)
    module_counts: Counter[str] = Counter()
    process_counts: Counter[str] = Counter()
    for ev in events:
        mod = ev.get("faulting_module") or "(unknown)"
        module_counts[mod] += 1
        comm = ev.get("comm") or "(unknown)"
        process_counts[comm] += 1

    peak_count, peak_ts, peak_ev = find_peak_burst(events)
    peak_snapshot = peak_ev.get("snapshot") if peak_ev else None

    # Peak memory pressure: min MemAvailable across snapshots in peak window
    peak_mem_available_kb: int | None = None
    if peak_ts and events:
        window_end = peak_ts + BURST_WINDOW
        window_events = [e for e in events if peak_ts <= e["_ts"] < window_end]
        for ev in window_events:
            snap = ev.get("snapshot")
            if snap:
                avail = get_snapshot_mem_available(snap)
                if avail is not None:
                    if peak_mem_available_kb is None or avail < peak_mem_available_kb:
                        peak_mem_available_kb = avail
                        peak_ts = ev["_ts"]  # time of the lowest mem event

    return {
        "since": since.isoformat(),
        "total_events": total,
        "peak_burst_count": peak_count,
        "peak_burst_ts": peak_ts.isoformat() if peak_ts else None,
        "peak_burst_snapshot": peak_snapshot,
        "peak_mem_available_kb": peak_mem_available_kb,
        "top_modules": module_counts.most_common(10),
        "top_processes": process_counts.most_common(10),
    }


def generate_burst_report() -> dict:
    """Report on the most recent burst."""
    since = datetime.now(timezone.utc) - timedelta(days=7)
    events = load_events(since)
    burst_events, burst_ts = find_most_recent_burst(events)
    if not burst_events:
        return {
            "burst_found": False,
            "message": "No burst detected in the last 7 days",
        }
    latest_snap = None
    for ev in reversed(burst_events):
        if ev.get("snapshot"):
            latest_snap = ev["snapshot"]
            break
    return {
        "burst_found": True,
        "burst_start_ts": burst_ts.isoformat() if burst_ts else None,
        "event_count": len(burst_events),
        "latest_snapshot": latest_snap,
        "events": [
            {
                "ts": e["_ts"].isoformat(),
                "comm": e.get("comm"),
                "faulting_module": e.get("faulting_module"),
                "snapshot": e.get("snapshot"),
            }
            for e in burst_events
        ],
    }


def print_human_report(data: dict) -> None:
    since_dt = datetime.fromisoformat(data["since"])
    # Determine since label
    age_h = (datetime.now(timezone.utc) - since_dt).total_seconds() / 3600
    if age_h < 2:
        label = f"last {int(age_h * 60)} min"
    elif age_h < 48:
        label = f"last {int(age_h)}h"
    else:
        label = f"last {int(age_h / 24)}d"

    print(f"Host fault report — {label}")
    print("=" * 44)
    print(f"Total events: {data['total_events']}")

    if data["peak_burst_count"] >= NORMAL_THRESHOLD:
        peak_ts_str = data["peak_burst_ts"][:16] + "Z" if data["peak_burst_ts"] else "?"
        snap = data.get("peak_burst_snapshot") or ""
        snap_name = Path(snap).name if snap else "(no snapshot)"
        print(f"Peak burst:   {data['peak_burst_count']} events in 5 min "
              f"at {peak_ts_str} (snapshot: {snap_name})")
    else:
        print("Peak burst:   below threshold")

    if data.get("peak_mem_available_kb") is not None:
        mb = data["peak_mem_available_kb"] // 1024
        print(f"Peak pressure during faults: MemAvailable {mb} MB")

    print()
    print("Top faulting modules:")
    for module, count in data["top_modules"]:
        print(f"  {module:<44} {count}")

    print()
    print("Top faulting processes:")
    for proc, count in data["top_processes"]:
        print(f"  {proc:<20} {count}")


def print_human_burst(data: dict) -> None:
    if not data.get("burst_found"):
        print(data.get("message", "No burst found."))
        return
    print(f"Most recent burst: {data['event_count']} events starting {data['burst_start_ts']}")
    if data.get("latest_snapshot"):
        print(f"Latest snapshot:   {data['latest_snapshot']}")
    print()
    print(f"{'Time':<26}  {'Comm':<20}  {'Module'}")
    print("-" * 80)
    for ev in data.get("events", []):
        ts = ev.get("ts", "?")[:19]
        comm = (ev.get("comm") or "?")[:20]
        mod = ev.get("faulting_module") or "(unknown)"
        print(f"  {ts:<24}  {comm:<20}  {mod}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize host fault events")
    parser.add_argument("--since", default="24h",
                        help="Time window (e.g. 24h, 1h, 7d). Default: 24h")
    parser.add_argument("--burst", action="store_true",
                        help="Show most recent burst summary")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    if not SUMMARY_JSONL.exists():
        if args.as_json:
            print(json.dumps({"error": f"log file not found: {SUMMARY_JSONL}"}))
        else:
            print(f"Error: log file not found: {SUMMARY_JSONL}", file=sys.stderr)
        return 1

    try:
        if args.burst:
            data = generate_burst_report()
        else:
            data = generate_report(args.since)
    except Exception as exc:
        if args.as_json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(data, default=str))
    else:
        if args.burst:
            print_human_burst(data)
        else:
            print_human_report(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
