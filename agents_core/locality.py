"""Locality ledger — per-call provenance for every LLM call (Gate 0 of the
independence blueprint: "the fallback is an event, not a default; every cloud
call is chosen, visible, budgeted and counted").

Storage: /srv/lapis/locality/<YYYY-MM-DD>.jsonl (env override LOCALITY_LEDGER_ROOT).
One write point per seam (call_operator, call_claude_cli, call_gw_agent) in
llm.py / gw_agent.py appends one record per call. record() never raises and
never affects a call's return value — a ledger failure is a WARNING, not an
error.

Durability: plain flock'd append, no fsync — survives a process crash, not a
power loss. That matches every other flock-based writer in this ecosystem
(claude_queue.py, gpu.py); inventing stronger guarantees here would be new
behaviour, not a shim over existing conventions.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents_core.room_paths import room_path

logger = logging.getLogger(__name__)

DEFAULT_ROOT = room_path("locality")

# Cost-class vocabulary — extensible (append-only), never repurposed. A record
# written today must stay unambiguously interpretable after new operators are
# added: do not narrow or reassign an existing value's meaning, only append
# new ones. "unknown" is the sink for anything unrecognised at write or read
# time and must never be silently reclassified later.
COST_CLASSES = ("local-gw", "local-sh", "paid-anthropic", "unknown")

_DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MiB per day-file before rotation


def _max_bytes() -> int:
    """Read at call time (not module load) so tests can monkeypatch.setenv."""
    return int(os.environ.get("LOCALITY_LEDGER_MAX_BYTES", str(_DEFAULT_MAX_BYTES)))


def root() -> Path:
    """Return the resolved ledger root, honoring LOCALITY_LEDGER_ROOT override.
    Default: /srv/lapis/locality/"""
    override = os.environ.get("LOCALITY_LEDGER_ROOT")
    return Path(override) if override else DEFAULT_ROOT


def _rotated_path(path: Path) -> Path:
    n = 1
    while True:
        candidate = path.with_name(f"{path.name}.{n}")
        if not candidate.exists():
            return candidate
        n += 1


def _append_with_rotation(path: Path, line: str, max_bytes: int) -> None:
    """Append `line` under an exclusive flock, rotating the file first if it has
    grown past `max_bytes`. Rotation and append run under the same lock (mirrors
    /srv/agents/scripts/activity_log.py's rotate-then-write pattern, generalized
    from a date trigger to a size trigger) so a runaway day never silently fills
    the disk and never interleaves with a concurrent writer's rotation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with open(path, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() >= max_bytes:
                f.seek(0)
                old_content = f.read()
                try:
                    _rotated_path(path).write_text(old_content, encoding="utf-8")
                except OSError:
                    pass
                f.seek(0)
                f.truncate()
            f.seek(0, os.SEEK_END)
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def record(
    *,
    requested_operator,
    served_model,
    host,
    cost_class,
    fallback_fired: bool = False,
    fallback_reason=None,
    seam,
    cost_usd=None,
    duration_ms=None,
    ok: bool = True,
    extra=None,
) -> None:
    """Append one provenance record for a single LLM call. Never raises.

    seam: one of "call_operator", "call_claude_cli", "call_gw_agent" — the
    chokepoint that wrote this record, so double-counting across seams (e.g. a
    call_operator fallback that itself recurses through another call_operator
    invocation) is detectable and attributable on read, not silently merged.

    This function must complete in well under 100ms and must never raise or
    change the caller's return value — the whole body is wrapped in
    try/except so a full disk or a permissions error degrades to a WARNING,
    never a broken call.
    """
    try:
        if cost_class not in COST_CLASSES:
            cost_class = "unknown"

        now = datetime.now(timezone.utc)
        entry = {
            "ts": now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+00:00",
            "seam": seam,
            "requested_operator": requested_operator,
            "served_model": served_model,
            "host": host,
            "cost_class": cost_class,
            "fallback_fired": bool(fallback_fired),
            "fallback_reason": fallback_reason,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "ok": bool(ok),
        }
        if extra:
            entry["extra"] = extra

        day = now.strftime("%Y-%m-%d")
        path = root() / f"{day}.jsonl"
        line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n"
        _append_with_rotation(path, line, _max_bytes())
    except Exception as e:
        logger.warning("[locality] ledger write failed: %s", e)


def _to_aware_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_entry_ts(entry: dict):
    ts_str = entry.get("ts")
    if not ts_str:
        return None
    try:
        parsed = datetime.fromisoformat(ts_str)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_entries(*, since=None, until=None) -> list[dict]:
    """Read every ledger entry whose timestamp falls in [since, until].
    Iterates files/lines rather than slurping (agents_core.observations
    precedent); malformed lines are skipped with a warning, not raised."""
    ledger_root = root()
    if not ledger_root.exists():
        return []

    since_aware = _to_aware_utc(since)
    until_aware = _to_aware_utc(until)

    results: list[dict] = []
    try:
        paths = sorted(p for p in ledger_root.glob("*.jsonl"))
    except OSError:
        return []

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "skipping malformed locality ledger line in %s", path
                        )
                        continue
                    if not isinstance(entry, dict):
                        continue

                    entry_ts = _parse_entry_ts(entry)
                    if since_aware is not None and (
                        entry_ts is None or entry_ts < since_aware
                    ):
                        continue
                    if until_aware is not None and (
                        entry_ts is None or entry_ts > until_aware
                    ):
                        continue

                    results.append(entry)
        except OSError:
            continue

    results.sort(key=lambda e: e.get("ts", ""))
    return results


def summarize(*, since=None, until=None) -> dict:
    """Aggregate ledger entries in [since, until] into a locality/cost summary.

    pct_local is on a 0-100 scale (42.5 means 42.5% of calls served locally)
    and is computed over *successful* calls only (local_ok / total_ok * 100)
    — a refused call (ok=false) never reached a model and must not inflate
    the locality number just because it was routed at a local cost class.
    total and by_cost_class still count every record, successful or not, so
    the raw census stays auditable; failed_count and by_cost_class_failed
    make the denominator shift visible instead of silent.
    paid_cost_usd is None when no entry in the window carried a cost_usd value
    (the claude -p envelope's total_cost_usd is optional and unverified) —
    callers must not treat None as zero.
    """
    entries = _read_entries(since=since, until=until)

    total = len(entries)
    by_cost_class: dict[str, int] = {}
    by_cost_class_failed: dict[str, int] = {}
    local_ok = 0
    total_ok = 0
    failed_count = 0
    fallback_count = 0
    by_fallback_reason: dict[str, int] = {}
    paid_cost_usd = None
    by_requested_operator: dict[str, dict] = {}

    for entry in entries:
        cost_class = entry.get("cost_class") or "unknown"
        if cost_class not in COST_CLASSES:
            cost_class = "unknown"
        by_cost_class[cost_class] = by_cost_class.get(cost_class, 0) + 1

        ok = entry.get("ok", True)
        if ok:
            total_ok += 1
            if cost_class in ("local-gw", "local-sh"):
                local_ok += 1
        else:
            failed_count += 1
            by_cost_class_failed[cost_class] = by_cost_class_failed.get(cost_class, 0) + 1

        if entry.get("fallback_fired"):
            fallback_count += 1
            reason = entry.get("fallback_reason") or "unknown"
            by_fallback_reason[reason] = by_fallback_reason.get(reason, 0) + 1

        cost_usd = entry.get("cost_usd")
        if isinstance(cost_usd, (int, float)):
            paid_cost_usd = (paid_cost_usd or 0.0) + cost_usd

        requested_operator = entry.get("requested_operator") or "unknown"
        served_model = entry.get("served_model") or "unknown"
        op_bucket = by_requested_operator.setdefault(requested_operator, {"served": {}})
        op_bucket["served"][served_model] = op_bucket["served"].get(served_model, 0) + 1

    pct_local = (local_ok / total_ok * 100.0) if total_ok else 0.0

    since_aware = _to_aware_utc(since)
    until_aware = _to_aware_utc(until)

    return {
        "total": total,
        "pct_local": pct_local,
        "by_cost_class": by_cost_class,
        "failed_count": failed_count,
        "by_cost_class_failed": by_cost_class_failed,
        "fallback_count": fallback_count,
        "by_fallback_reason": by_fallback_reason,
        "paid_cost_usd": paid_cost_usd,
        "by_requested_operator": by_requested_operator,
        "window": {
            "since": since_aware.isoformat() if since_aware else None,
            "until": until_aware.isoformat() if until_aware else None,
        },
    }


def is_ledger_healthy(*, max_silence_hours: float = 24.0) -> tuple[bool, str]:
    """Return (False, reason) when no record has been written in
    max_silence_hours. An empty/stale ledger reads identically to a perfect
    (all-local) week if this isn't checked explicitly — silence must present
    as friction, not as a good result. Never raises.
    """
    try:
        ledger_root = root()
        if not ledger_root.exists():
            return False, "no ledger directory found — no records ever written"

        try:
            paths = sorted(p for p in ledger_root.glob("*.jsonl"))
        except OSError as e:
            return False, f"could not list ledger directory: {e}"

        if not paths:
            return False, "ledger directory exists but contains no record files"

        last_ts = None
        for path in reversed(paths):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                last_ts = _parse_entry_ts(entry)
                if last_ts is not None:
                    break
            if last_ts is not None:
                break

        if last_ts is None:
            return False, "ledger files present but no parseable records found"

        now = datetime.now(timezone.utc)
        silence_hours = (now - last_ts).total_seconds() / 3600.0
        if silence_hours > max_silence_hours:
            return False, (
                f"no ledger record in {silence_hours:.1f}h "
                f"(threshold {max_silence_hours}h) — last record at {last_ts.isoformat()}"
            )
        return True, f"last record {silence_hours:.2f}h ago"
    except Exception as e:
        return False, f"health check failed: {e}"


def _parse_duration(spec: str) -> timedelta:
    match = re.match(r"^(\d+(?:\.\d+)?)\s*([dhm])$", spec.strip())
    if not match:
        raise ValueError(
            f"invalid duration {spec!r}; expected e.g. '7d', '24h', '30m'"
        )
    value, unit = float(match.group(1)), match.group(2)
    if unit == "d":
        return timedelta(days=value)
    if unit == "h":
        return timedelta(hours=value)
    return timedelta(minutes=value)


def _cli_summarize(argv: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m agents_core.locality summarize")
    parser.add_argument("--since", default="7d", help="lookback duration, e.g. 7d, 24h, 30m")
    parser.add_argument("--until", default=None, help="ISO8601 timestamp (default: now)")
    parser.add_argument("--format", dest="fmt", choices=["json", "text"], default="text")
    ns = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    since_dt = now - _parse_duration(ns.since)
    until_dt = datetime.fromisoformat(ns.until) if ns.until else now

    healthy, health_message = is_ledger_healthy()
    result = summarize(since=since_dt, until=until_dt)

    if ns.fmt == "json":
        print(json.dumps(
            {"healthy": healthy, "health_message": health_message, **result},
            ensure_ascii=False,
        ))
        return

    print(f"Locality ledger — window {result['window']['since']} .. {result['window']['until']}")
    if not healthy:
        print(f"** LEDGER SILENCE: {health_message} **")
    print(f"total calls:        {result['total']}")
    print(f"pct local:          {result['pct_local']:.1f}%")
    print(f"by cost class:      {result['by_cost_class']}")
    print(f"fallback_count:     {result['fallback_count']}")
    print(f"by fallback reason: {result['by_fallback_reason']}")
    if result["paid_cost_usd"] is not None:
        print(f"paid cost usd:      ${result['paid_cost_usd']:.4f}")
    else:
        print("paid cost usd:      unknown (no envelope carried cost)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] != "summarize":
        print(
            "usage: python -m agents_core.locality summarize "
            "[--since 7d] [--until ISO8601] [--format json|text]",
            file=sys.stderr,
        )
        sys.exit(1)

    _cli_summarize(sys.argv[2:])
