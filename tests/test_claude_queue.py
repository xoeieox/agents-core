"""Unit tests for agents_core.claude_queue.ClaudeQueue.

Mirrors the shape of test_gpu_coordinator.py — uses tmp_path for isolation
so tests never touch the real /srv/lapis/claude-queue/ on disk.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agents_core import claude_queue as cq_mod
from agents_core.claude_queue import ClaudeQueue, Priority
from agents_core.gpu import PACIFIC


@pytest.fixture(autouse=True)
def _no_coordinator_during_base_tests():
    """The module-load autoload may bind intention_registry (if installed);
    base ClaudeQueue tests assert queue plumbing in isolation, so we clear
    the slot per-test and restore afterwards. Coordinator-wired behaviour
    is covered in tests/test_claude_queue_coordinator.py.
    """
    prior = cq_mod.get_coordinator()
    cq_mod.register_coordinator(None)
    yield
    cq_mod.register_coordinator(prior)


@pytest.fixture
def queue(tmp_path: Path) -> ClaudeQueue:
    return ClaudeQueue(queue_dir=tmp_path / "claude-queue")


def _basic_task(**over) -> dict:
    base = {
        "task_type": "subprocess",
        "priority": Priority.NORMAL,
        "timeout_seconds": 60,
        "submitted_by": "test",
        "model": "sonnet",
        "payload": {"command": "echo hi", "spec_path": "/tmp/spec.json"},
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def test_submit_writes_yaml_and_returns_id(queue):
    task_id = queue.submit(_basic_task())
    assert task_id.startswith("claude_")
    yaml_path = queue.pending_dir / f"{task_id}.yaml"
    assert yaml_path.exists()
    assert queue.status()["depth"] == 1


def test_submit_respects_caller_supplied_task_id(queue):
    task_id = queue._generate_id(slug="fixer-t1")
    returned = queue.submit(_basic_task(), task_id=task_id)
    assert returned == task_id
    assert (queue.pending_dir / f"{task_id}.yaml").exists()


def test_submit_missing_required_field_raises(queue):
    with pytest.raises(ValueError):
        queue.submit({"priority": Priority.NORMAL})  # no task_type


def test_submit_appends_history_event(queue):
    task_id = queue.submit(_basic_task())
    events = queue.get_history()
    assert any(e["event"] == "submitted" and e["id"] == task_id
               for e in events)


def test_submit_preserves_claude_specific_fields(queue):
    task_id = queue.submit(_basic_task(
        worktree_required=True, base_branch="main",
        description="fixer:target_x", notify=True,
    ))
    task = queue._read_task(queue.pending_dir / f"{task_id}.yaml")
    assert task["worktree_required"] is True
    assert task["base_branch"] == "main"
    assert task["description"] == "fixer:target_x"
    assert task["notify"] is True


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def test_claim_returns_none_when_empty(queue):
    assert queue.claim() is None


def test_claim_orders_by_priority(queue):
    low = queue.submit(_basic_task(priority=Priority.LOW))
    high = queue.submit(_basic_task(priority=Priority.HIGH))
    normal = queue.submit(_basic_task(priority=Priority.NORMAL))

    first = queue.claim()
    assert first["id"] == high
    second = queue.claim()
    assert second["id"] == normal
    third = queue.claim()
    assert third["id"] == low


def test_claim_breaks_ties_by_submitted_at(queue):
    ids = [queue.submit(_basic_task(priority=Priority.NORMAL)) for _ in range(3)]
    claimed = [queue.claim()["id"] for _ in range(3)]
    assert claimed == ids  # FIFO at equal priority


def test_claim_moves_task_from_pending_to_active(queue):
    task_id = queue.submit(_basic_task())
    queue.claim()
    assert not (queue.pending_dir / f"{task_id}.yaml").exists()
    assert (queue.active_dir / f"{task_id}.yaml").exists()


def test_concurrent_claims_return_disjoint_tasks(queue):
    ids = {queue.submit(_basic_task()) for _ in range(8)}
    claimed = []
    lock = threading.Lock()

    def worker():
        while True:
            t = queue.claim()
            if t is None:
                return
            with lock:
                claimed.append(t["id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == len(set(claimed)) == 8
    assert set(claimed) == ids


# ---------------------------------------------------------------------------
# Complete / Fail / Cancel
# ---------------------------------------------------------------------------

def test_complete_moves_task_to_completed(queue):
    task_id = queue.submit(_basic_task())
    queue.claim()
    queue.complete(task_id, output_path="/tmp/out.md", result_summary="ok")
    assert not (queue.active_dir / f"{task_id}.yaml").exists()
    assert (queue.completed_dir / f"{task_id}.yaml").exists()
    task = queue._read_task(queue.completed_dir / f"{task_id}.yaml")
    assert task["status"] == "completed"
    assert task["output_path"] == "/tmp/out.md"
    assert task["result_summary"] == "ok"


def test_fail_moves_task_to_failed(queue):
    task_id = queue.submit(_basic_task())
    queue.claim()
    queue.fail(task_id, error="boom")
    assert not (queue.active_dir / f"{task_id}.yaml").exists()
    assert (queue.failed_dir / f"{task_id}.yaml").exists()
    task = queue._read_task(queue.failed_dir / f"{task_id}.yaml")
    assert task["status"] == "failed"
    assert task["error"] == "boom"


def test_fail_on_missing_active_file_reconciles_state(queue):
    state = queue._read_state()
    state["in_flight"] = ["ghost-id"]
    queue._write_state(state)

    queue.fail("ghost-id", error="hand-killed and rm'd out of band")

    import json
    persisted = json.loads(queue.state_path.read_text())
    assert "ghost-id" not in persisted["in_flight"]


def test_fail_on_missing_active_file_does_not_write_failed_or_history(queue):
    state = queue._read_state()
    state["in_flight"] = ["ghost-id"]
    queue._write_state(state)

    queue.fail("ghost-id", error="hand-killed and rm'd out of band")

    assert list(queue.failed_dir.glob("*.yaml")) == []
    assert not queue.history_path.exists() or queue.history_path.read_text() == ""


def test_complete_computes_duration_seconds(queue):
    task_id = queue.submit(_basic_task())
    queue.claim()
    queue.complete(task_id)
    task = queue._read_task(queue.completed_dir / f"{task_id}.yaml")
    assert "duration_seconds" in task
    assert isinstance(task["duration_seconds"], int)


def test_cancel_removes_pending_task(queue):
    task_id = queue.submit(_basic_task())
    assert queue.cancel(task_id, reason="changed my mind") is True
    assert not (queue.pending_dir / f"{task_id}.yaml").exists()
    assert queue.status()["depth"] == 0


def test_cancel_returns_false_for_missing(queue):
    assert queue.cancel("claude_nonexistent") is False


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_status_reports_depth_and_in_flight(queue):
    a = queue.submit(_basic_task())
    b = queue.submit(_basic_task())
    assert queue.status() == {"depth": 2, "in_flight": []}

    first = queue.claim()
    status = queue.status()
    assert status["depth"] == 1
    assert status["in_flight"] == [first["id"]]

    queue.complete(first["id"])
    assert queue.status()["in_flight"] == []


# ---------------------------------------------------------------------------
# Slot independence — ClaudeQueue keeps its own coordinator slot, separate
# from agents_core.gpu's. A coordinator registered only with gpu's slot must
# not receive ClaudeQueue lifecycle events.
# ---------------------------------------------------------------------------

def test_claude_queue_slot_independent_from_gpu_slot(queue):
    """Registering a coordinator ONLY on agents_core.gpu must not cause
    ClaudeQueue to call it. Full coordinator-call coverage lives in
    tests/test_claude_queue_coordinator.py — this is the anti-coupling
    guard that used to forbid the integration entirely; it now asserts
    the two coordinator slots are distinct.
    """
    calls: list[str] = []

    class _Spy:
        def project_from_task(self, *a, **k):
            calls.append("project")
            return None

        def manifest(self, *a, **k):
            calls.append("manifest")

        def compost(self, *a, **k):
            calls.append("compost")

    from agents_core import gpu as gpu_mod
    prior_gpu = gpu_mod.get_coordinator()
    prior_cq = cq_mod.get_coordinator()
    gpu_mod.register_coordinator(_Spy())
    cq_mod.register_coordinator(None)  # claude slot explicitly empty
    try:
        tid = queue.submit(_basic_task())
        queue.claim()
        queue.complete(tid)
        tid2 = queue.submit(_basic_task())
        queue.claim()
        queue.fail(tid2, "x")
    finally:
        gpu_mod.register_coordinator(prior_gpu)
        cq_mod.register_coordinator(prior_cq)

    assert calls == []  # claude slot empty → no coordinator traffic


# ---------------------------------------------------------------------------
# _generate_id — single _now_pacific() call
# ---------------------------------------------------------------------------

def test_generate_id_format(queue):
    tid = queue._generate_id(slug="fixer-foo")
    parts = tid.split("_")
    # claude_YYYYMMDD_HHMMSS_ffff_<slug>
    assert parts[0] == "claude"
    assert len(parts[1]) == 8   # YYYYMMDD
    assert len(parts[2]) == 6   # HHMMSS
    assert len(parts[3]) == 4   # microseconds
    assert parts[4] == "fixerfoo"


# ---------------------------------------------------------------------------
# AC3: gravitywell-122b serialization — at most one active claim at a time
# ---------------------------------------------------------------------------

def test_gravitywell_122b_second_claim_defers_while_first_active(queue):
    """AC3: second gravitywell-122b task is not claimed while one is already active."""
    t1 = queue.submit(_basic_task(model="gravitywell-122b"))
    t2 = queue.submit(_basic_task(model="gravitywell-122b"))

    # Claim the first — should succeed.
    first = queue.claim()
    assert first is not None
    assert first["model"] == "gravitywell-122b"

    # First is now active; claiming again should defer the second gravitywell-122b.
    second = queue.claim()
    assert second is None, "second gravitywell-122b should be deferred while first is active"


def test_gravitywell_122b_can_claim_after_first_completes(queue):
    """AC3: once the first gravitywell-122b task completes, the second can be claimed."""
    t1 = queue.submit(_basic_task(model="gravitywell-122b"))
    t2 = queue.submit(_basic_task(model="gravitywell-122b"))

    first = queue.claim()
    assert first is not None
    # Verify deferred while first active
    assert queue.claim() is None

    queue.complete(first["id"])

    # Now the second should be claimable.
    second = queue.claim()
    assert second is not None
    assert second["model"] == "gravitywell-122b"


def test_non_gravitywell_model_not_blocked_by_serialization(queue):
    """AC3: sonnet/haiku tasks are not affected by the gravitywell-122b serialization gate."""
    t1 = queue.submit(_basic_task(model="gravitywell-122b"))
    t2 = queue.submit(_basic_task(model="sonnet"))

    # Claim gravitywell first.
    first = queue.claim()
    assert first is not None
    assert first["model"] == "gravitywell-122b"

    # Sonnet task should still be claimable despite gravitywell being active.
    second = queue.claim()
    assert second is not None
    assert second["model"] == "sonnet"


def test_gravitywell_122b_mixed_queue_claims_non_gw_when_gw_active(queue):
    """AC3: with a gravitywell-122b active, other model tasks can still be claimed."""
    queue.submit(_basic_task(model="gravitywell-122b"))
    queue.submit(_basic_task(model="gravitywell-122b"))
    queue.submit(_basic_task(model="haiku"))

    gw_task = queue.claim()
    assert gw_task["model"] == "gravitywell-122b"

    # With gw active: second gw deferred but haiku available.
    next_task = queue.claim()
    assert next_task is not None
    assert next_task["model"] == "haiku"


def test_generate_id_uses_single_clock_read(queue, monkeypatch):
    """Regression guard: GPUQueue._generate_id calls _now_pacific() twice
    which can straddle a second boundary. ClaudeQueue calls it once."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    calls = []

    def fake_now():
        # Alternate seconds on each call — if the implementation reads
        # _now_pacific() twice, it will pick up a different second and the
        # resulting id will be internally inconsistent.
        calls.append(1)
        sec = len(calls)  # 1, 2, 3 ...
        return datetime(2026, 4, 23, 12, 0, sec, 123400,
                        tzinfo=ZoneInfo("America/Los_Angeles"))

    monkeypatch.setattr(cq_mod, "_now_pacific", fake_now)
    queue._generate_id(slug="x")
    assert len(calls) == 1, (
        f"_generate_id called _now_pacific() {len(calls)} times; "
        "must be 1 to avoid the second-boundary bug."
    )


def _write_spec(path: Path, **fields) -> Path:
    data = {"backend_url": "http://203.0.113.11:8082"}
    data.update(fields)
    path.write_text(json.dumps(data))
    return path

def _berth_task(tmp_path: Path, n: int, url: str = "http://203.0.113.11:8082") -> dict:
    return _basic_task(
        model="ninfer-27b",
        payload={"command": "echo hi", "spec_path": str(_write_spec(tmp_path / f"spec{n}.json", backend_url=url))},
    )

def _rewrite_active(queue, task_id: str, **field_over) -> dict:
    apath = queue.active_dir / f"{task_id}.yaml"
    task = queue._read_task(apath)
    task.update(field_over)
    if "started_at_unset" in field_over:
        task.pop("started_at", None)
    queue._write_task(queue.active_dir, task)
    return task


# ---------------------------------------------------------------------------
# AC4: per-seat serialization (default bare set includes ninfer-27b)
# ---------------------------------------------------------------------------

def test_serial_seat_second_claim_defers_while_first_active(queue, tmp_path):
    """Second ninfer-27b task defers while the first is active; claims after fail()."""
    t1 = queue.submit(_berth_task(tmp_path, 1))
    t2 = queue.submit(_berth_task(tmp_path, 2))

    first = queue.claim()
    assert first is not None and first["id"] == t1

    second = queue.claim()
    assert second is None, "second ninfer-27b must defer while the first is active"

    queue.fail(t1, error="done")
    third = queue.claim()
    assert third is not None and third["id"] == t2
def test_serial_fail_open_on_malformed_spec(queue, tmp_path, caplog):
    """Malformed specs (JSON list / non-string backend_url) fail open:
    (None, model) key, no exception escapes, bare-model deferral still applies."""
    import logging
    caplog.set_level(logging.DEBUG)

    bad1 = tmp_path / "bad1.json"
    bad1.write_text("[1, 2, 3]")  # JSON list: not a dict
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({"backend_url": 42}))  # non-string backend_url

    t1 = queue.submit(_basic_task(
        model="ninfer-27b",
        payload={"command": "x", "spec_path": str(bad1)}))
    t2 = queue.submit(_basic_task(
        model="ninfer-27b",
        payload={"command": "x", "spec_path": str(bad2)}))

    # Direct _seat_key: (None, model), no exception.
    task1 = queue._read_task(queue.pending_dir / f"{t1}.yaml")
    task2 = queue._read_task(queue.pending_dir / f"{t2}.yaml")
    assert queue._seat_key(task1, need_url=True) == (None, "ninfer-27b")
    assert queue._seat_key(task2, need_url=True) == (None, "ninfer-27b")

    # One WARN per task id (deduped), never repeated across calls.
    for _ in range(3):
        queue._seat_key(task1, need_url=True)
    warns1 = [r for r in caplog.records
              if r.levelno >= logging.WARNING
              and "unreadable spec" in r.getMessage() and t1 in r.getMessage()]
    warns2 = [r for r in caplog.records
              if r.levelno >= logging.WARNING
              and "unreadable spec" in r.getMessage() and t2 in r.getMessage()]
    assert len(warns1) == 1, f"task1 must WARN exactly once, got {len(warns1)}"
    assert len(warns2) == 1, f"task2 must WARN exactly once, got {len(warns2)}"

    # Bare-model deferral still applies despite the malformed specs.
    first = queue.claim()
    assert first is not None
    assert queue.claim() is None, "bare-model entry must still defer"


def test_serial_seat_can_claim_after_first_completes(queue, tmp_path):
    """Same pair via complete(): the seat frees the instant the YAML leaves active/."""
    t1 = queue.submit(_berth_task(tmp_path, 1))
    t2 = queue.submit(_berth_task(tmp_path, 2))

    first = queue.claim()
    assert first is not None and first["id"] == t1
    assert queue.claim() is None  # deferred while active

    queue.complete(t1)

    second = queue.claim()
    assert second is not None and second["id"] == t2


def test_serial_deferral_does_not_block_other_seats(queue):
    """A busy serialized seat never starves other seats (candidate-skip)."""
    t1 = queue.submit(_basic_task(model="ninfer-27b"))
    t2 = queue.submit(_basic_task(model="gravitywell-slot1"))

    first = queue.claim()
    assert first is not None and first["id"] == t1

    second = queue.claim()
    assert second is not None and second["id"] == t2, \
        "other-seat task must claim normally while the berth is busy"


def test_serial_entry_url_scoped(queue, tmp_path, monkeypatch):
    """An exact url|model entry scopes deferral to that seat (per-call env read)."""
    monkeypatch.setenv("CLAUDE_QUEUE_SERIAL_SEATS", "http://203.0.113.11:8082|ninfer-27b")

    t1 = queue.submit(_berth_task(tmp_path, 1, url="http://203.0.113.11:8082"))
    t2 = queue.submit(_berth_task(tmp_path, 2, url="http://other:9999"))

    c1 = queue.claim()
    assert c1 is not None and c1["id"] == t1
    c2 = queue.claim()
    assert c2 is not None and c2["id"] == t2, \
        "different backend_url = different seat = no deferral"

    t3 = queue.submit(_berth_task(tmp_path, 3, url="http://203.0.113.11:8082"))
    c3 = queue.claim()
    assert c3 is None, "same exact seat as an active task must defer"


def test_serial_seats_env_parse(queue, monkeypatch):
    """_serial_seat_entries: parse rules + safe default on unset/empty."""
    # Unset -> default set.
    monkeypatch.delenv("CLAUDE_QUEUE_SERIAL_SEATS", raising=False)
    assert cq_mod._serial_seat_entries() == [
        ("model", None, "gravitywell-122b"),
        ("model", None, "ninfer-27b"),
    ]

    # Empty string -> default set (a stray env edit cannot opt out).
    monkeypatch.setenv("CLAUDE_QUEUE_SERIAL_SEATS", "")
    assert cq_mod._serial_seat_entries() == [
        ("model", None, "gravitywell-122b"),
        ("model", None, "ninfer-27b"),
    ]

    # Bare vs url|model; whitespace; duplicate collapse; last-pipe split
    # (a pipe inside the URL stays on the URL side).
    monkeypatch.setenv(
        "CLAUDE_QUEUE_SERIAL_SEATS",
        " ninfer-27b , http://a:1|model-a , ninfer-27b , http://x|y|model-b ",
    )
    assert cq_mod._serial_seat_entries() == [
        ("model", None, "ninfer-27b"),
        ("exact", "http://a:1", "model-a"),
        ("exact", "http://x|y", "model-b"),
    ]


def test_serial_default_set_no_spec_read_for_missing_spec(queue, tmp_path, caplog):
    """Lazy-resolution discriminator: with the DEFAULT bare set, a task whose
    spec file is missing defers by model key and logs NO unreadable-spec WARN
    (the WARN only fires when an exact entry forces a spec read)."""
    t1 = queue.submit(_basic_task(model="ninfer-27b"))  # spec_path = /tmp/spec.json (absent)
    t2 = queue.submit(_basic_task(model="ninfer-27b"))

    first = queue.claim()
    assert first is not None and first["id"] == t1
    with caplog.at_level("WARNING", logger="claude_queue.coordinator"):
        second = queue.claim()
    assert second is None, "bare-model deferral must apply (model key only)"
    warns = [r for r in caplog.records
             if r.levelno >= 30 and "unreadable spec" in r.getMessage()]
    assert warns == [], f"no WARN expected for bare entries, got {warns}"
    assert queue._warned_spec_ids == set()


def test_serial_stale_active_yaml_does_not_occupy(queue, tmp_path):
    """An active YAML older than timeout_seconds + SEAT_STALE_GRACE_S is a
    self-clearing orphan: it stops pinning the seat."""
    t1 = queue.submit(_berth_task(tmp_path, 1))
    t2 = queue.submit(_berth_task(tmp_path, 2))

    first = queue.claim()
    assert first is not None and first["id"] == t1
    assert queue.claim() is None  # fresh -> defers

    # Age the active YAML to 2h ago (timeout 60s + grace 120s << 2h).
    old = (datetime.now(PACIFIC) - timedelta(hours=2)).isoformat(timespec="seconds")
    _rewrite_active(queue, t1, started_at=old)

    second = queue.claim()
    assert second is not None and second["id"] == t2, \
        "stale orphan must not pin the seat (staleness rule self-clears)"


def test_serial_missing_started_at_treated_fresh(queue, tmp_path):
    """An active task with NO started_at is treated as FRESH (occupying) -
    deliberately the OPPOSITE of startup_sweep's eviction direction."""
    t1 = queue.submit(_berth_task(tmp_path, 1))
    t2 = queue.submit(_berth_task(tmp_path, 2))

    first = queue.claim()
    assert first is not None and first["id"] == t1

    _rewrite_active(queue, t1, started_at_unset=True)

    assert queue.claim() is None, \
        "missing started_at must default to FRESH (still occupying)"
