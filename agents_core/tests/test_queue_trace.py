"""Unit tests for agents_core.calibration.queue_trace.

Coverage:
  1. Replay correctness on a synthetic event log
  2. Idempotent backfill (re-running produces identical output)
  3. Resume offset (--follow resumes without duplicating)
  4. active-never-negative invariant (bad log can't drive active < 0)
  5. Fidelity-drift abort fires on corrupted-log fixture
  6. stasis_velocity is recomputable from raw_event_ts
  7. Read-only: backfill asserts no mutation of /srv/lapis/claude-queue
"""

import json
from pathlib import Path

import pytest

from agents_core.calibration.queue_trace import (
    DriftAbortError,
    _ReplayAccumulator,
    _parse_event_line,
    _read_events_from,
    backfill,
    fidelity_check,
    replay_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_s: float = 0.0) -> str:
    """Build a deterministic ISO timestamp at 2026-01-01T00:00:00Z + offset_s."""
    from datetime import datetime, timezone, timedelta
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_s)).isoformat()


def _make_history(events: list[dict], tmp_path: Path) -> Path:
    p = tmp_path / "history.jsonl"
    with open(p, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    return p


def _make_state_json(in_flight: list[str], tmp_path: Path) -> Path:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"in_flight": in_flight, "queue_depth": 0}))
    return p


# ---------------------------------------------------------------------------
# 1. Replay correctness
# ---------------------------------------------------------------------------

def test_replay_basic_lifecycle(tmp_path):
    """submitted → claimed → completed produces 3 transition records
    with internally consistent counts."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(0), "model": "sonnet", "priority": 10,
         "submitted_by": "pm"},
        {"event": "claimed",   "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(5), "model": "sonnet", "priority": 10},
        {"event": "completed", "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(15), "duration_seconds": 10.0},
    ]
    history = _make_history(events, tmp_path)
    state_json = _make_state_json([], tmp_path)

    acc = _ReplayAccumulator(workers=2)
    records = list(replay_events(events, acc, state_path=state_json))

    assert len(records) == 3

    # After submitted: pending=1, active=0
    r0 = records[0]
    assert r0["event"]["event"] == "submitted"
    assert r0["state_before"]["counts"]["pending"] == 0
    assert r0["state_after"]["counts"]["pending"] == 1
    assert r0["state_after"]["counts"]["active"] == 0

    # After claimed: pending=0, active=1, t1 in in_flight
    r1 = records[1]
    assert r1["event"]["event"] == "claimed"
    assert r1["state_after"]["counts"]["active"] == 1
    assert r1["state_after"]["counts"]["pending"] == 0
    assert any(j["id"] == "t1" for j in r1["state_after"]["in_flight"])

    # After completed: completed=1, active=0
    r2 = records[2]
    assert r2["event"]["event"] == "completed"
    assert r2["state_after"]["counts"]["completed"] == 1
    assert r2["state_after"]["counts"]["active"] == 0
    assert r2["state_after"]["in_flight"] == []

    # Counts are monotonic where expected
    assert acc.counts["completed"] == 1
    assert acc.counts["failed"] == 0


def test_replay_multiple_jobs(tmp_path):
    """Two concurrent jobs produce the correct active count at peak."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "submitted", "id": "t2", "task_type": "x",
         "timestamp": _ts(1), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(2), "model": None, "priority": 10},
        {"event": "claimed", "id": "t2", "task_type": "x",
         "timestamp": _ts(3), "model": None, "priority": 10},
        {"event": "completed", "id": "t1", "task_type": "x",
         "timestamp": _ts(10)},
        {"event": "completed", "id": "t2", "task_type": "x",
         "timestamp": _ts(11)},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    assert len(records) == 6

    # After both claimed: active == 2
    r3 = records[3]  # second claimed
    assert r3["state_after"]["counts"]["active"] == 2
    assert len(r3["state_after"]["in_flight"]) == 2

    # Workers utilization at peak
    assert r3["state_after"]["workers"]["utilization"] == 1.0


def test_cancelled_event_silent(tmp_path):
    """cancelled events update pending count but produce no transition."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "cancelled", "id": "t1", "task_type": "x",
         "timestamp": _ts(2), "reason": "user"},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    # Only submitted produces a transition; cancelled is silent
    assert len(records) == 1
    assert records[0]["event"]["event"] == "submitted"
    # Pending count was incremented then decremented
    assert acc.counts["pending"] == 0


def test_intention_events_skipped(tmp_path):
    """intention_match_* events produce no transitions and don't corrupt state."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "intention_match_reinforce", "id": "t2", "timestamp": _ts(1),
         "task_type": "x", "intention_id": "i-1", "shared_task_id": "t1"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(2), "model": None, "priority": 10},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json(["t1"], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    assert len(records) == 2
    assert records[0]["event"]["event"] == "submitted"
    assert records[1]["event"]["event"] == "claimed"


def test_raw_event_ts_present(tmp_path):
    """Each transition record carries raw_event_ts from history.jsonl."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": "2026-01-01T00:00:00+00:00", "model": None,
         "priority": 5, "submitted_by": "x"},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    assert records[0]["raw_event_ts"] == "2026-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# 2. Idempotent backfill
# ---------------------------------------------------------------------------

def test_backfill_idempotent(tmp_path):
    """Re-running backfill produces byte-for-byte identical transitions."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(0), "model": "sonnet", "priority": 10, "submitted_by": "pm"},
        {"event": "claimed", "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(5), "model": "sonnet", "priority": 10},
        {"event": "failed", "id": "t1", "task_type": "llm_call",
         "timestamp": _ts(30), "error": "timeout"},
    ]
    history = _make_history(events, tmp_path)
    out = tmp_path / "transitions.jsonl"
    state_json = _make_state_json([], tmp_path)
    rec_state = tmp_path / "recorder_state.json"
    schema = tmp_path / "SCHEMA.md"

    n1 = backfill(history, out, state_json, schema, rec_state, workers=2)
    first_run = out.read_text()

    n2 = backfill(history, out, state_json, schema, rec_state, workers=2)
    second_run = out.read_text()

    assert n1 == n2 == 3
    # ts field differs between runs but the structural content is the same.
    # Compare state_before/event/state_after ignoring ts field.
    lines1 = [json.loads(l) for l in first_run.splitlines() if l.strip()]
    lines2 = [json.loads(l) for l in second_run.splitlines() if l.strip()]
    for a, b in zip(lines1, lines2):
        assert a["state_before"] == b["state_before"]
        assert a["event"] == b["event"]
        assert a["state_after"] == b["state_after"]


# ---------------------------------------------------------------------------
# 3. Resume offset — follow mode
# ---------------------------------------------------------------------------

def test_resume_offset_no_duplication(tmp_path):
    """follow resumes from stored offset; events before offset not duplicated."""
    # First batch of events
    batch1 = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(2), "model": None, "priority": 10},
    ]
    history = _make_history(batch1, tmp_path)
    out = tmp_path / "transitions.jsonl"
    state_json = _make_state_json(["t1"], tmp_path)
    rec_state = tmp_path / "recorder_state.json"

    # Backfill to set the offset
    backfill(history, out, state_json, tmp_path / "schema.md", rec_state, workers=2)
    offset_after_batch1 = json.loads(rec_state.read_text())["offset"]
    assert offset_after_batch1 > 0

    # Append a second batch
    with open(history, "a") as fh:
        fh.write(json.dumps({
            "event": "completed", "id": "t1", "task_type": "x",
            "timestamp": _ts(10), "duration_seconds": 8.0,
        }) + "\n")

    # Read only new events (simulate what follow does incrementally)
    new_events, new_offset = _read_events_from(history, start_offset=offset_after_batch1)
    assert len(new_events) == 1
    assert new_events[0]["event"] == "completed"
    assert new_offset > offset_after_batch1


def test_read_events_from_start(tmp_path):
    """_read_events_from(offset=0) reads all events."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 1, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(1), "model": None, "priority": 1},
    ]
    history = _make_history(events, tmp_path)
    read_events, offset = _read_events_from(history, start_offset=0)
    assert len(read_events) == 2
    assert offset > 0


def test_read_events_from_partial(tmp_path):
    """_read_events_from resumes correctly mid-file."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 1, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(1), "model": None, "priority": 1},
    ]
    history = _make_history(events, tmp_path)
    # Get offset after first event
    _, offset_after_first = _read_events_from(history, start_offset=0)

    # Re-read from start to find first-event end
    with open(history, "rb") as fh:
        first_line = fh.readline()
    first_offset = len(first_line)

    remaining, _ = _read_events_from(history, start_offset=first_offset)
    assert len(remaining) == 1
    assert remaining[0]["event"] == "claimed"


# ---------------------------------------------------------------------------
# 4. active-never-negative invariant
# ---------------------------------------------------------------------------

def test_active_never_negative(tmp_path):
    """Even if completed fires without a prior claimed, active stays >= 0."""
    events = [
        # completed without prior submitted/claimed (corrupted log)
        {"event": "completed", "id": "t_orphan", "task_type": "x",
         "timestamp": _ts(0), "duration_seconds": 5.0},
        # Normal sequence
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(1), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(2), "model": None, "priority": 10},
        {"event": "completed", "id": "t1", "task_type": "x",
         "timestamp": _ts(10)},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    for r in records:
        assert r["state_before"]["counts"]["active"] >= 0
        assert r["state_after"]["counts"]["active"] >= 0
        assert r["state_before"]["counts"]["pending"] >= 0
        assert r["state_after"]["counts"]["pending"] >= 0


def test_pending_never_negative(tmp_path):
    """claimed without prior submitted never drives pending below 0."""
    events = [
        {"event": "claimed", "id": "t_orphan", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10},
        {"event": "completed", "id": "t_orphan", "task_type": "x",
         "timestamp": _ts(5)},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    for r in records:
        assert r["state_before"]["counts"]["pending"] >= 0
        assert r["state_after"]["counts"]["pending"] >= 0


# ---------------------------------------------------------------------------
# 5. Fidelity-drift abort
# ---------------------------------------------------------------------------

def test_fidelity_check_ok_when_matching(tmp_path):
    """fidelity_check returns 'ok' when replay in_flight == live in_flight."""
    acc = _ReplayAccumulator(workers=2)
    acc.in_flight["t1"] = {"id": "t1", "task_type": "x", "model": None, "claimed_at": _ts(0)}
    acc.in_flight["t2"] = {"id": "t2", "task_type": "x", "model": None, "claimed_at": _ts(1)}

    state_json = _make_state_json(["t1", "t2"], tmp_path)
    result = fidelity_check(acc, state_path=state_json, abort_on_drift=True)
    assert result == "ok"


def test_fidelity_check_drift_abort(tmp_path):
    """DriftAbortError fires when mismatch exceeds DRIFT_ABORT_THRESHOLD."""
    import agents_core.calibration.queue_trace as qt
    orig = qt.DRIFT_ABORT_THRESHOLD

    try:
        qt.DRIFT_ABORT_THRESHOLD = 1  # low threshold for test

        acc = _ReplayAccumulator(workers=2)
        # replay thinks: t1, t2, t3 in flight
        for tid in ("t1", "t2", "t3"):
            acc.in_flight[tid] = {
                "id": tid, "task_type": "x", "model": None, "claimed_at": _ts(0)
            }
        # live state: only t1 (2 ID mismatches: t2,t3 replay-only)
        state_json = _make_state_json(["t1"], tmp_path)

        with pytest.raises(DriftAbortError):
            fidelity_check(acc, state_path=state_json, abort_on_drift=True)
    finally:
        qt.DRIFT_ABORT_THRESHOLD = orig


def test_fidelity_check_within_threshold_no_abort(tmp_path):
    """Small drift within threshold does not raise."""
    import agents_core.calibration.queue_trace as qt
    orig = qt.DRIFT_ABORT_THRESHOLD

    try:
        qt.DRIFT_ABORT_THRESHOLD = 3
        acc = _ReplayAccumulator(workers=2)
        acc.in_flight["t1"] = {
            "id": "t1", "task_type": "x", "model": None, "claimed_at": _ts(0)
        }
        # live has t1 + t2 (1 mismatch ≤ threshold)
        state_json = _make_state_json(["t1", "t2"], tmp_path)
        result = fidelity_check(acc, state_path=state_json, abort_on_drift=True)
        assert result == "ok"
    finally:
        qt.DRIFT_ABORT_THRESHOLD = orig


def test_backfill_aborts_on_drift(tmp_path):
    """backfill raises DriftAbortError when end-of-backfill check exceeds threshold."""
    import agents_core.calibration.queue_trace as qt
    orig = qt.DRIFT_ABORT_THRESHOLD

    try:
        qt.DRIFT_ABORT_THRESHOLD = 0  # any mismatch aborts

        events = [
            {"event": "submitted", "id": "t1", "task_type": "x",
             "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
            {"event": "claimed", "id": "t1", "task_type": "x",
             "timestamp": _ts(2), "model": None, "priority": 10},
            # no completed — replay says t1 in flight, but live says nothing
        ]
        history = _make_history(events, tmp_path)
        out = tmp_path / "transitions.jsonl"
        # live state claims no in_flight (corrupted/diverged)
        state_json = _make_state_json([], tmp_path)

        with pytest.raises(DriftAbortError):
            backfill(
                history, out, state_json,
                tmp_path / "schema.md",
                tmp_path / "rec_state.json",
                workers=2,
            )
    finally:
        qt.DRIFT_ABORT_THRESHOLD = orig


def test_fidelity_check_unknown_when_no_state_json(tmp_path):
    """fidelity_check returns 'unknown' when state.json is missing."""
    acc = _ReplayAccumulator(workers=2)
    result = fidelity_check(
        acc,
        state_path=tmp_path / "nonexistent_state.json",
        abort_on_drift=False,
    )
    assert result == "unknown"


# ---------------------------------------------------------------------------
# 6. stasis_velocity recomputable from raw_event_ts
# ---------------------------------------------------------------------------

def test_stasis_velocity_recomputable(tmp_path):
    """velocity[n] == gap[n] - gap[n-1] where gap[n] = ts[n] - ts[n-1]."""
    # Events at t=0, t=10, t=15, t=30 (gaps: -, 10, 5, 15)
    ts_vals = [0.0, 10.0, 15.0, 30.0]
    events = [
        {"event": "submitted", "id": f"t{i}", "task_type": "x",
         "timestamp": _ts(ts_vals[i]), "model": None, "priority": 10, "submitted_by": "x"}
        for i in range(4)
    ]
    acc = _ReplayAccumulator(workers=4)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    assert len(records) == 4

    # Extract gaps from raw_event_ts
    from datetime import datetime, timezone

    def _unix(iso: str) -> float:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()

    raw_ts = [_unix(r["raw_event_ts"]) for r in records]
    # gaps[n] = inter-event gap before event n (0.0 for first event: unknown)
    gaps = [0.0] + [raw_ts[i] - raw_ts[i - 1] for i in range(1, len(raw_ts))]
    # velocity[0] = 0 (no previous gap); velocity[n] = gaps[n] - gaps[n-1] for n > 0
    expected_velocities = [0.0] + [gaps[i] - gaps[i - 1] for i in range(1, len(gaps))]

    for i, r in enumerate(records):
        computed = r["state_before"]["stasis_velocity"]
        assert abs(computed - expected_velocities[i]) < 0.01, (
            f"record {i}: got velocity {computed}, expected {expected_velocities[i]}"
        )


def test_stasis_duration_matches_gap(tmp_path):
    """state_before.stasis_duration == seconds since previous event."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "submitted", "id": "t2", "task_type": "x",
         "timestamp": _ts(100), "model": None, "priority": 10, "submitted_by": "x"},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json([], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    # First record: no previous event, stasis_duration = 0
    assert records[0]["state_before"]["stasis_duration"] == 0.0
    # Second record: 100s since first event
    assert abs(records[1]["state_before"]["stasis_duration"] - 100.0) < 0.01


def test_in_flight_stasis_duration(tmp_path):
    """In-flight job's stasis_duration = seconds since it was claimed."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": "sonnet", "priority": 10, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(10), "model": "sonnet", "priority": 10},
        # Next event happens 30s after claim
        {"event": "submitted", "id": "t2", "task_type": "x",
         "timestamp": _ts(40), "model": None, "priority": 10, "submitted_by": "x"},
    ]
    acc = _ReplayAccumulator(workers=2)
    state_json = _make_state_json(["t1"], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    # In the state_before of t2's submission, t1 should have been in-flight for 30s
    r2 = records[2]
    in_flight = r2["state_before"]["in_flight"]
    assert len(in_flight) == 1
    assert in_flight[0]["id"] == "t1"
    assert abs(in_flight[0]["stasis_duration"] - 30.0) < 0.01


# ---------------------------------------------------------------------------
# 7. Read-only: backfill does not write to /srv/lapis/claude-queue
# ---------------------------------------------------------------------------

def test_backfill_readonly(tmp_path, monkeypatch):
    """backfill writes only to calibration dir, never to the queue dir."""
    queue_dir = tmp_path / "claude-queue"
    queue_dir.mkdir()
    history = queue_dir / "history.jsonl"
    history.write_text(json.dumps({
        "event": "submitted", "id": "t1", "task_type": "x",
        "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x",
    }) + "\n")

    calib_dir = tmp_path / "calibration"
    out = calib_dir / "transitions.jsonl"
    state_json = queue_dir / "state.json"
    state_json.write_text(json.dumps({"in_flight": [], "queue_depth": 0}))

    # Monkeypatch the queue-dir subpaths to track if anything is written there
    written_paths: list[Path] = []
    _orig_write = Path.write_text

    def _spy_write(self, *args, **kwargs):
        if str(self).startswith(str(queue_dir)):
            written_paths.append(self)
        return _orig_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _spy_write)

    backfill(
        history, out, state_json,
        calib_dir / "SCHEMA.md",
        calib_dir / "recorder_state.json",
        workers=2,
    )

    assert written_paths == [], (
        f"backfill wrote to queue dir: {written_paths}"
    )
    # Output was written to calibration dir
    assert out.exists()


# ---------------------------------------------------------------------------
# 8. SCHEMA.md is written by backfill
# ---------------------------------------------------------------------------

def test_backfill_writes_schema(tmp_path):
    history = _make_history([], tmp_path)
    out = tmp_path / "transitions.jsonl"
    state_json = _make_state_json([], tmp_path)
    schema = tmp_path / "SCHEMA.md"
    rec_state = tmp_path / "recorder_state.json"

    backfill(history, out, state_json, schema, rec_state, workers=2)

    assert schema.exists()
    content = schema.read_text()
    assert "stasis_duration" in content
    assert "stasis_velocity" in content
    assert "no classification" in content.lower() or "No classification" in content


# ---------------------------------------------------------------------------
# 9. workers field is static capacity only
# ---------------------------------------------------------------------------

def test_workers_static_capacity(tmp_path):
    """workers.capacity is always CLAUDE_QUEUE_WORKERS, utilization derived."""
    events = [
        {"event": "submitted", "id": "t1", "task_type": "x",
         "timestamp": _ts(0), "model": None, "priority": 10, "submitted_by": "x"},
        {"event": "claimed", "id": "t1", "task_type": "x",
         "timestamp": _ts(1), "model": None, "priority": 10},
    ]
    acc = _ReplayAccumulator(workers=4)
    state_json = _make_state_json(["t1"], tmp_path)
    records = list(replay_events(events, acc, state_path=state_json))

    for r in records:
        assert r["state_before"]["workers"]["capacity"] == 4
        assert r["state_after"]["workers"]["capacity"] == 4

    # After claimed: 1 job / 4 capacity = 0.25
    assert abs(records[1]["state_after"]["workers"]["utilization"] - 0.25) < 0.001


# ---------------------------------------------------------------------------
# 10. _parse_event_line handles malformed input
# ---------------------------------------------------------------------------

def test_parse_event_line_valid():
    line = '{"event":"submitted","id":"t1"}'
    result = _parse_event_line(line)
    assert result is not None
    assert result["event"] == "submitted"


def test_parse_event_line_empty():
    assert _parse_event_line("") is None
    assert _parse_event_line("   ") is None


def test_parse_event_line_invalid_json():
    assert _parse_event_line("{not json}") is None


def test_parse_event_line_no_event_key():
    assert _parse_event_line('{"id":"t1"}') is None
