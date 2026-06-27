"""Queue-runner transition-trace recorder (agents-core-queue-trace-recorder-v0).

Passive, read-only replay of /srv/lapis/claude-queue/history.jsonl into a
calibration corpus of (state_before, event, state_after) transitions at
/srv/lapis/calibration/queue-runner/transitions.jsonl.

The recorder is a SEPARATE, READ-ONLY consumer.  It MUST NOT edit
claude_queue_runner.py or the queue hot path, and MUST NOT hold any lock
the runner uses.

CLI:
  python -m agents_core.calibration.queue_trace --backfill
  python -m agents_core.calibration.queue_trace --follow [--poll-s N]

The shared interface (state → predict → score) is documented in SCHEMA.md,
written to /srv/lapis/calibration/queue-runner/SCHEMA.md on first --backfill run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from agents_core.room_paths import room_path

log = logging.getLogger("queue-trace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLAUDE_QUEUE_WORKERS: int = int(os.environ.get("CLAUDE_QUEUE_WORKERS", "2"))
DRIFT_ABORT_THRESHOLD: int = int(os.environ.get("QUEUE_TRACE_DRIFT_THRESHOLD", "3"))
FIDELITY_CHECK_INTERVAL: int = int(os.environ.get("QUEUE_TRACE_FIDELITY_INTERVAL", "100"))
FOLLOW_POLL_INTERVAL_S: float = float(os.environ.get("QUEUE_TRACE_POLL_S", "2.0"))

HISTORY_PATH: Path = room_path("claude_queue.history")
STATE_JSON_PATH: Path = room_path("claude_queue") / "state.json"
CALIBRATION_DIR: Path = room_path("calibration.queue_runner")
TRANSITIONS_PATH: Path = CALIBRATION_DIR / "transitions.jsonl"
SCHEMA_PATH: Path = CALIBRATION_DIR / "SCHEMA.md"
RECORDER_STATE_PATH: Path = CALIBRATION_DIR / "recorder_state.json"

# ---------------------------------------------------------------------------
# Schema document (written to SCHEMA_PATH on first backfill)
# ---------------------------------------------------------------------------

_SCHEMA_MD = """\
# Queue-Runner Calibration Schema

Shared interface for the real recorder, AgentWorld predictor, and scorer.
Each line of `transitions.jsonl` is one JSON object with this shape:

## Transition record

```json
{
  "ts":            "<ISO8601 wall clock at recording time>",
  "raw_event_ts":  "<ISO8601 timestamp from history.jsonl — use for stasis_velocity recomputation>",
  "state_before":  <State>,
  "event":         <Event>,
  "state_after":   <State>,
  "consistency":   "ok" | "drift" | "unknown"
}
```

## State (S)

```json
{
  "counts": {
    "pending":   <int>,
    "active":    <int>,
    "completed": <int>,
    "failed":    <int>
  },
  "in_flight": [
    {
      "id":              "<task_id>",
      "task_type":       "<str>",
      "model":           "<str | null>",
      "claimed_at":      "<ISO8601>",
      "stasis_duration": <float, seconds since claimed_at at snapshot>
    }
  ],
  "workers": {
    "capacity":    <int, CLAUDE_QUEUE_WORKERS env var — static in v0>,
    "utilization": <float, len(in_flight)/capacity>
  },
  "stasis_duration": <float, seconds since previous lifecycle event (queue-level)>,
  "stasis_velocity": <float, stasis_duration_this - stasis_duration_prev (seconds)>
}
```

### Stasis measures (raw coordinates — NO classification)

- `stasis_duration`: time the queue has been silent before this transition.
  Positive = queue was idle. 0 = an event just happened.
- `stasis_velocity`: derivative of stasis_duration across successive transitions.
  Positive = inter-event gaps growing (stall building).
  Negative = queue accelerating (gaps shrinking).
  Recomputable from `raw_event_ts`: `velocity[n] = gap[n] - gap[n-1]`
  where `gap[n] = ts[n] - ts[n-1]`.

**No classification rule**: the recorder emits raw coordinates only.
Normal-vs-suspicious boundaries are for AgentWorld to learn; the recorder
MUST NOT encode `mood`, narrative, or threshold judgments.

### Workers (static in v0)

`capacity` = `CLAUDE_QUEUE_WORKERS` env var (default 2). Dynamic capacity
inference is deferred to v1 (changing worker count breaks deterministic replay).

## Event (A)

```json
{
  "event":            "submitted" | "claimed" | "completed" | "failed",
  "id":               "<task_id>",
  "task_type":        "<str>",
  "model":            "<str | null>",
  "priority":         <int | null>,
  "duration_seconds": <float | null>,
  "intention_id":     "<str | null>",
  "submitted_by":     "<str | null>",
  "error":            "<str | null>"
}
```

Only lifecycle events (submitted/claimed/completed/failed) produce transition
records.  `cancelled` and `intention_*` events update replay state silently.

## Consistency flag

Set by cross-checking replayed `in_flight` IDs against live `state.json`:

- `"ok"`:      replay in_flight matches live state (within QUEUE_TRACE_DRIFT_THRESHOLD)
- `"drift"`:   real divergence detected (reliable fields differ beyond threshold);
              only reachable in `--follow` periodic checks (`abort_on_drift=False`).
              In `--backfill` (`abort_on_drift=True`), threshold-exceeding drift raises
              DriftAbortError before any record can carry this value.
- `"unknown"`: no fidelity check has been run yet (initial transitions)

`state.json.queue_depth` is EXCLUDED from fidelity checks — it globs `*.yaml`
but tasks are `*.json` in older schema versions; any delta there is a known
quirk, not drift.

## Idempotency

Re-running `--backfill` produces the same corpus (replay is a pure function
of history.jsonl).  `--follow` resumes from the stored byte offset in
`recorder_state.json` without duplicating entries.
"""

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_ts(raw: str | None) -> float | None:
    """Parse an ISO timestamp to a UTC unix float. Returns None on failure."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------------------------------------------------------------------------
# Replay accumulator (internal mutable state)
# ---------------------------------------------------------------------------

class _ReplayAccumulator:
    """Mutable state driven by replaying history.jsonl events in order.

    Not thread-safe — single-threaded replay only.
    """

    def __init__(self, workers: int = CLAUDE_QUEUE_WORKERS):
        self.counts: dict[str, int] = {
            "pending": 0, "active": 0, "completed": 0, "failed": 0
        }
        self.in_flight: dict[str, dict] = {}  # id → {id, task_type, model, claimed_at}
        self.workers_capacity: int = workers

        # Stasis tracking
        self._last_event_ts: float | None = None
        self._prev_stasis_duration: float = 0.0

        # Consistency state (set by fidelity_check)
        self.consistency: str = "unknown"

    def in_flight_ids(self) -> set[str]:
        return set(self.in_flight.keys())

    def _make_state_snapshot(
        self,
        event_ts: float,
        stasis_duration: float,
        stasis_velocity: float,
    ) -> dict:
        in_flight_list = []
        for job in self.in_flight.values():
            claimed_ts = _parse_ts(job.get("claimed_at"))
            job_stasis = (event_ts - claimed_ts) if claimed_ts is not None else 0.0
            in_flight_list.append({
                "id": job["id"],
                "task_type": job["task_type"],
                "model": job["model"],
                "claimed_at": job["claimed_at"],
                "stasis_duration": round(max(0.0, job_stasis), 3),
            })

        cap = self.workers_capacity
        utilization = len(self.in_flight) / cap if cap > 0 else 0.0
        return {
            "counts": dict(self.counts),
            "in_flight": in_flight_list,
            "workers": {"capacity": cap, "utilization": round(utilization, 4)},
            "stasis_duration": round(max(0.0, stasis_duration), 3),
            "stasis_velocity": round(stasis_velocity, 3),
        }

    def snapshot_before(self, event_ts: float) -> dict:
        """State snapshot BEFORE applying the event at event_ts."""
        stasis = (
            event_ts - self._last_event_ts
            if self._last_event_ts is not None
            else 0.0
        )
        velocity = stasis - self._prev_stasis_duration
        return self._make_state_snapshot(event_ts, stasis, velocity)

    def snapshot_after(self, event_ts: float) -> dict:
        """State snapshot AFTER applying the event — stasis_duration reset to 0."""
        return self._make_state_snapshot(event_ts, 0.0, 0.0)

    def apply(self, event: dict, event_ts: float) -> bool:
        """Apply one event to the accumulator.  Returns True if it's a
        lifecycle event (submitted/claimed/completed/failed) that should
        produce a transition record; False for silent-update events.
        """
        ev = event.get("event", "")
        eid = event.get("id", "")

        if ev == "submitted":
            self.counts["pending"] = max(0, self.counts["pending"] + 1)

        elif ev == "claimed":
            self.counts["pending"] = max(0, self.counts["pending"] - 1)
            self.counts["active"] += 1
            self.in_flight[eid] = {
                "id": eid,
                "task_type": event.get("task_type", "unknown"),
                "model": event.get("model"),
                "claimed_at": event.get("timestamp") or "",
            }

        elif ev == "completed":
            self.counts["active"] = max(0, self.counts["active"] - 1)
            self.counts["completed"] += 1
            self.in_flight.pop(eid, None)

        elif ev == "failed":
            self.counts["active"] = max(0, self.counts["active"] - 1)
            self.counts["failed"] += 1
            self.in_flight.pop(eid, None)

        elif ev == "cancelled":
            self.counts["pending"] = max(0, self.counts["pending"] - 1)
            return False  # state updated but no transition record

        else:
            return False  # intention_* and unknown events: skip entirely

        # Update inter-event gap tracking (only for lifecycle events)
        if self._last_event_ts is not None:
            self._prev_stasis_duration = event_ts - self._last_event_ts
        self._last_event_ts = event_ts
        return True

# ---------------------------------------------------------------------------
# Fidelity cross-check
# ---------------------------------------------------------------------------

class DriftAbortError(RuntimeError):
    """Raised when replayed state diverges from live state beyond threshold."""


def _read_live_state(state_path: Path) -> dict | None:
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def fidelity_check(
    acc: _ReplayAccumulator,
    state_path: Path = STATE_JSON_PATH,
    abort_on_drift: bool = True,
) -> str:
    """Compare replayed in_flight IDs to live state.json.

    Returns "ok", "drift", or "unknown".
    Raises DriftAbortError if real drift exceeds DRIFT_ABORT_THRESHOLD and
    abort_on_drift=True.

    queue_depth is EXCLUDED — it uses *.yaml glob in older schema; treat any
    delta there as a known quirk.
    """
    live = _read_live_state(state_path)
    if live is None:
        return "unknown"

    live_ids: set[str] = set(live.get("in_flight") or [])
    replay_ids: set[str] = acc.in_flight_ids()

    diff = live_ids.symmetric_difference(replay_ids)
    diff_count = len(diff)

    if diff_count == 0:
        acc.consistency = "ok"
        return "ok"

    if diff_count > DRIFT_ABORT_THRESHOLD:
        msg = (
            f"fidelity drift EXCEEDED threshold ({diff_count} ID mismatches, "
            f"threshold={DRIFT_ABORT_THRESHOLD}); "
            f"live_only={live_ids - replay_ids}, "
            f"replay_only={replay_ids - live_ids}"
        )
        log.error(msg)
        acc.consistency = "drift"
        if abort_on_drift:
            raise DriftAbortError(msg)
        return "drift"

    # Within threshold — flag as drift but don't abort
    log.warning(
        "fidelity: %d ID mismatch(es) within threshold; "
        "live_only=%s replay_only=%s",
        diff_count,
        live_ids - replay_ids,
        replay_ids - live_ids,
    )
    acc.consistency = "ok"  # within threshold = acceptable
    return "ok"

# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def _parse_event_line(line: str) -> dict | None:
    """Parse one JSONL line from history.jsonl.  Returns None on failure."""
    line = line.strip()
    if not line:
        return None
    try:
        ev = json.loads(line)
        if not isinstance(ev, dict) or "event" not in ev:
            return None
        return ev
    except json.JSONDecodeError:
        return None


def _event_to_record(event: dict) -> dict:
    """Extract the Event (A) portion of a history.jsonl entry."""
    return {
        "event": event.get("event"),
        "id": event.get("id"),
        "task_type": event.get("task_type"),
        "model": event.get("model"),
        "priority": event.get("priority"),
        "duration_seconds": event.get("duration_seconds"),
        "intention_id": event.get("intention_id"),
        "submitted_by": event.get("submitted_by"),
        "error": event.get("error"),
    }


def replay_events(
    events: list[dict],
    acc: _ReplayAccumulator,
    fidelity_check_interval: int = FIDELITY_CHECK_INTERVAL,
    state_path: Path = STATE_JSON_PATH,
) -> Iterator[dict]:
    """Replay a sequence of history events, yielding transition records.

    Does NOT perform start/end fidelity checks — caller owns those.
    Periodically checks fidelity every fidelity_check_interval transitions.
    """
    transition_count = 0
    for event in events:
        raw_ts = event.get("timestamp")
        event_ts = _parse_ts(raw_ts)
        if event_ts is None:
            log.debug("skipping event with missing/unparseable timestamp: %s", event)
            continue

        state_before = acc.snapshot_before(event_ts)
        is_lifecycle = acc.apply(event, event_ts)
        if not is_lifecycle:
            continue

        state_after = acc.snapshot_after(event_ts)
        transition_count += 1

        # Periodic fidelity check
        if fidelity_check_interval > 0 and transition_count % fidelity_check_interval == 0:
            try:
                fidelity_check(acc, state_path=state_path, abort_on_drift=True)
            except DriftAbortError:
                raise
            except Exception as exc:
                log.warning("fidelity check failed: %s", exc)

        yield {
            "ts": _now_iso(),
            "raw_event_ts": raw_ts,
            "state_before": state_before,
            "event": _event_to_record(event),
            "state_after": state_after,
            "consistency": acc.consistency,
        }

# ---------------------------------------------------------------------------
# History reader with offset tracking
# ---------------------------------------------------------------------------

def _read_events_from(path: Path, start_offset: int = 0) -> tuple[list[dict], int]:
    """Read all events from path starting at start_offset.

    Returns (events, new_offset).  new_offset is the byte position after
    the last line read, suitable for resuming with --follow.
    """
    events: list[dict] = []
    try:
        with open(path, "rb") as fh:
            fh.seek(start_offset)
            while True:
                line_bytes = fh.readline()
                if not line_bytes:
                    break
                ev = _parse_event_line(line_bytes.decode(errors="replace"))
                if ev is not None:
                    events.append(ev)
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            new_offset = fh.tell()
    except OSError as exc:
        log.error("cannot read %s: %s", path, exc)
        new_offset = start_offset
    return events, new_offset


def _write_transition(path: Path, record: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")

# ---------------------------------------------------------------------------
# Recorder state (offset persistence)
# ---------------------------------------------------------------------------

def _load_recorder_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"offset": 0, "events_processed": 0}


def _save_recorder_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n")

# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def backfill(
    history_path: Path = HISTORY_PATH,
    transitions_path: Path = TRANSITIONS_PATH,
    state_path: Path = STATE_JSON_PATH,
    schema_path: Path = SCHEMA_PATH,
    recorder_state_path: Path = RECORDER_STATE_PATH,
    workers: int = CLAUDE_QUEUE_WORKERS,
) -> int:
    """Replay the full history.jsonl from offset 0 and write transitions.

    Idempotent: re-running overwrites transitions.jsonl with the same content.
    Returns the number of transitions written.
    """
    transitions_path.parent.mkdir(parents=True, exist_ok=True)

    # Write schema doc if missing
    if not schema_path.exists():
        schema_path.write_text(_SCHEMA_MD)
        log.info("wrote SCHEMA.md to %s", schema_path)

    acc = _ReplayAccumulator(workers=workers)

    # Pre-backfill: snapshot live state for comparison reference
    live_before = _read_live_state(state_path)
    if live_before is None:
        log.warning("state.json not readable before backfill; fidelity unknown")

    log.info("reading history from %s", history_path)
    events, end_offset = _read_events_from(history_path, start_offset=0)
    log.info("%d raw history lines read", len(events))

    # Replay with periodic fidelity checks
    count = 0
    with open(transitions_path, "w") as out:
        for record in replay_events(events, acc, state_path=state_path):
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1

    log.info("wrote %d transitions to %s", count, transitions_path)

    # End-of-backfill fidelity check (post-replay state vs live state.json)
    try:
        result = fidelity_check(acc, state_path=state_path, abort_on_drift=True)
        log.info("end-of-backfill fidelity: %s", result)
    except DriftAbortError as exc:
        log.error("BACKFILL ABORTED: %s", exc)
        raise
    except Exception as exc:
        log.warning("end-of-backfill fidelity check failed: %s", exc)

    _save_recorder_state(recorder_state_path, {
        "offset": end_offset,
        "events_processed": len(events),
    })

    return count

# ---------------------------------------------------------------------------
# Follow (live tail mode)
# ---------------------------------------------------------------------------

def follow(
    history_path: Path = HISTORY_PATH,
    transitions_path: Path = TRANSITIONS_PATH,
    state_path: Path = STATE_JSON_PATH,
    recorder_state_path: Path = RECORDER_STATE_PATH,
    workers: int = CLAUDE_QUEUE_WORKERS,
    poll_interval_s: float = FOLLOW_POLL_INTERVAL_S,
) -> None:
    """Tail history.jsonl and append new transitions indefinitely.

    Resumes from the stored offset so no transitions are duplicated.
    Performs periodic fidelity checks.  Runs until SIGINT/SIGTERM.
    """
    transitions_path.parent.mkdir(parents=True, exist_ok=True)

    rec_state = _load_recorder_state(recorder_state_path)
    offset = rec_state.get("offset", 0)
    events_processed = rec_state.get("events_processed", 0)

    log.info("follow: resuming from offset=%d", offset)

    # Replay from beginning to reconstruct accumulator state up to current offset
    acc = _ReplayAccumulator(workers=workers)
    if offset > 0:
        try:
            with open(history_path, "rb") as fh:
                fh.seek(0)
                raw = fh.read(offset).decode(errors="replace")
            for line in raw.splitlines():
                ev = _parse_event_line(line)
                if ev is not None:
                    event_ts = _parse_ts(ev.get("timestamp"))
                    if event_ts is not None:
                        acc.apply(ev, event_ts)
        except OSError as exc:
            log.warning("could not bootstrap accumulator state: %s", exc)

    check_counter = 0
    try:
        while True:
            new_events, new_offset = _read_events_from(history_path, start_offset=offset)
            if new_events:
                for record in replay_events(new_events, acc, state_path=state_path):
                    _write_transition(transitions_path, record)
                    events_processed += 1

                offset = new_offset
                _save_recorder_state(recorder_state_path, {
                    "offset": offset,
                    "events_processed": events_processed,
                })

            check_counter += 1
            if check_counter % 30 == 0:
                try:
                    fidelity_check(acc, state_path=state_path, abort_on_drift=False)
                except Exception as exc:
                    log.warning("follow fidelity check: %s", exc)

            time.sleep(poll_interval_s)
    except (KeyboardInterrupt, SystemExit):
        log.info("follow: stopping at offset=%d events=%d", offset, events_processed)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Queue-runner transition-trace recorder",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--backfill", action="store_true",
        help="Replay full history.jsonl and write transitions.jsonl from scratch",
    )
    mode.add_argument(
        "--follow", action="store_true",
        help="Tail history.jsonl and append new transitions (resumes from last offset)",
    )
    parser.add_argument(
        "--history", type=Path, default=HISTORY_PATH,
        help="Path to history.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=TRANSITIONS_PATH,
        help="Path to transitions.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--state-json", type=Path, default=STATE_JSON_PATH,
        help="Path to queue state.json for fidelity checks (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-s", type=float, default=FOLLOW_POLL_INTERVAL_S,
        help="Polling interval for --follow mode in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=CLAUDE_QUEUE_WORKERS,
        help="Static worker capacity (default: CLAUDE_QUEUE_WORKERS env or %(default)s)",
    )
    args = parser.parse_args()

    recorder_state = args.output.parent / "recorder_state.json"

    if args.backfill:
        try:
            n = backfill(
                history_path=args.history,
                transitions_path=args.output,
                state_path=args.state_json,
                schema_path=args.output.parent / "SCHEMA.md",
                recorder_state_path=recorder_state,
                workers=args.workers,
            )
            print(f"backfill complete: {n} transitions written to {args.output}")
        except DriftAbortError as exc:
            print(f"ERROR: drift abort — {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        follow(
            history_path=args.history,
            transitions_path=args.output,
            state_path=args.state_json,
            recorder_state_path=recorder_state,
            workers=args.workers,
            poll_interval_s=args.poll_s,
        )


if __name__ == "__main__":
    main()
