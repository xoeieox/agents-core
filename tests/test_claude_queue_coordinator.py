"""Coordinator-hook tests for agents_core.claude_queue.ClaudeQueue.

Mirrors tests/test_gpu_coordinator.py. Verifies that a registered
coordinator receives lifecycle events (submit/complete/fail/cancel) and
that the orphan-spec unlink fires on match_reinforce / match_manifested
(spec §1 resolution (b)).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from agents_core import claude_queue as cq_mod
from agents_core.claude_queue import (
    ClaudeQueue,
    Priority,
    get_coordinator,
    register_coordinator,
)


@dataclass
class _FakeIntention:
    intention_id: str
    reinforces: str | None = None


@dataclass
class _FakeProjection:
    decision: str
    shared_task_id: str | None
    intention: _FakeIntention


class _RecordingCoordinator:
    """Minimal coordinator that records calls without any policy."""

    def __init__(self):
        self.projected: list[dict] = []
        self.manifested: list[tuple[str, str | None]] = []
        self.composted: list[tuple[str, str]] = []
        self.next_decision: str = "projected"
        self.next_intention_id: str = "intent-1"
        self.next_shared_task_id: str | None = None

    def project_from_task(self, task, *, projected_by, proposed_change,
                          target_heading, task_id):
        self.projected.append({
            "task_id": task_id,
            "task_type": task.get("task_type"),
            "projected_by": projected_by,
            "proposed_change": proposed_change,
            "target_heading": target_heading,
        })
        return _FakeProjection(
            decision=self.next_decision,
            shared_task_id=self.next_shared_task_id,
            intention=_FakeIntention(
                intention_id=self.next_intention_id,
                reinforces=None,
            ),
        )

    def manifest(self, intention_id, *, linked_task_id):
        self.manifested.append((intention_id, linked_task_id))

    def compost(self, intention_id, *, reason):
        self.composted.append((intention_id, reason))


@pytest.fixture
def _restore_coordinator():
    prior = get_coordinator()
    yield
    register_coordinator(prior)


@pytest.fixture
def queue(tmp_path: Path) -> ClaudeQueue:
    return ClaudeQueue(queue_dir=tmp_path / "claude-queue")


def _basic_task(spec_path: str | None = None, **over) -> dict:
    base = {
        "task_type": "subprocess",
        "priority": Priority.NORMAL,
        "timeout_seconds": 60,
        "submitted_by": "unit-test",
        "model": "sonnet",
        "description": "fixer:target_x",
        "payload": {
            "command": "echo hi",
            "spec_path": spec_path or "/tmp/spec-none.json",
        },
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_and_get_coordinator(_restore_coordinator):
    obj = object()
    register_coordinator(obj)
    assert get_coordinator() is obj
    register_coordinator(None)
    assert get_coordinator() is None


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def test_submit_without_coordinator_queues_task(queue, _restore_coordinator):
    register_coordinator(None)
    tid = queue.submit(_basic_task())
    assert tid.startswith("claude_")
    assert (queue.pending_dir / f"{tid}.yaml").exists()


def test_submit_projects_when_coordinator_registered(queue, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-projected"
    register_coordinator(coord)

    task_in = _basic_task()
    task_in["payload"]["target_heading"] = "engine/"
    tid = queue.submit(task_in)

    assert len(coord.projected) == 1
    assert coord.projected[0]["task_id"] == tid
    assert coord.projected[0]["projected_by"] == "unit-test"
    assert coord.projected[0]["proposed_change"] == "fixer:target_x"
    assert coord.projected[0]["target_heading"] == "engine/"
    assert (queue.pending_dir / f"{tid}.yaml").exists()

    task = queue._read_task(queue.pending_dir / f"{tid}.yaml")
    assert task["intention_id"] == "int-projected"


def test_match_reinforce_returns_shared_task_id_and_unlinks_orphan_spec(
    queue, tmp_path, _restore_coordinator
):
    """§1 resolution (b): match_reinforce unlinks the orphaned pre-written
    spec file, returns the shared task_id, and does not queue a new YAML.
    """
    coord = _RecordingCoordinator()
    coord.next_decision = "match_reinforce"
    coord.next_shared_task_id = "claude_prior_shared"
    coord.next_intention_id = "int-reinforcer"
    register_coordinator(coord)

    orphan_spec = tmp_path / "orphan-spec.json"
    orphan_spec.write_text(json.dumps({"task_id": "claude_second_submitter"}))
    assert orphan_spec.exists()

    tid = queue.submit(_basic_task(spec_path=str(orphan_spec)))

    assert tid == "claude_prior_shared"
    assert list(queue.pending_dir.glob("*.yaml")) == []
    assert not orphan_spec.exists(), "pre-written spec should be unlinked"

    events = queue.get_history()
    assert any(e.get("event") == "intention_match_reinforce" for e in events)


def test_match_manifested_returns_shared_task_id_and_unlinks_orphan_spec(
    queue, tmp_path, _restore_coordinator
):
    coord = _RecordingCoordinator()
    coord.next_decision = "match_manifested"
    coord.next_shared_task_id = "claude_old_done"
    coord.next_intention_id = "int-manifested-match"
    register_coordinator(coord)

    orphan_spec = tmp_path / "orphan-spec.json"
    orphan_spec.write_text("{}")

    tid = queue.submit(_basic_task(spec_path=str(orphan_spec)))

    assert tid == "claude_old_done"
    assert list(queue.pending_dir.glob("*.yaml")) == []
    assert not orphan_spec.exists()


def test_match_tolerates_missing_spec_path(queue, _restore_coordinator):
    """If the caller never wrote a spec (e.g., ad-hoc dispatch or GPU-style
    submit), unlink should be a no-op."""
    coord = _RecordingCoordinator()
    coord.next_decision = "match_reinforce"
    coord.next_shared_task_id = "claude_prior"
    coord.next_intention_id = "int-x"
    register_coordinator(coord)

    task = _basic_task()
    task["payload"].pop("spec_path", None)
    tid = queue.submit(task)

    assert tid == "claude_prior"
    assert list(queue.pending_dir.glob("*.yaml")) == []


def test_ignore_intention_registry_opt_out(queue, _restore_coordinator):
    coord = _RecordingCoordinator()
    register_coordinator(coord)

    task = _basic_task()
    task["payload"]["_ignore_intention_registry"] = True
    tid = queue.submit(task)

    assert coord.projected == []
    assert (queue.pending_dir / f"{tid}.yaml").exists()


# ---------------------------------------------------------------------------
# Complete / Fail / Cancel
# ---------------------------------------------------------------------------

def test_complete_calls_manifest(queue, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-manifest"
    register_coordinator(coord)

    tid = queue.submit(_basic_task())
    queue.claim()
    queue.complete(tid, result_summary="ok")

    assert coord.manifested == [("int-to-manifest", tid)]
    assert coord.composted == []


def test_fail_calls_compost(queue, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-compost"
    register_coordinator(coord)

    tid = queue.submit(_basic_task())
    queue.claim()
    queue.fail(tid, error="simulated failure")

    assert coord.composted and coord.composted[0][0] == "int-to-compost"
    assert "simulated failure" in coord.composted[0][1]
    assert coord.manifested == []


def test_cancel_calls_compost(queue, _restore_coordinator):
    """Regression companion to gpu.py's 2026-04-23 zombie fix: cancel on a
    pending task must cascade compost so the intention isn't orphaned.
    """
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-cancel"
    register_coordinator(coord)

    tid = queue.submit(_basic_task())
    assert queue.cancel(tid, reason="no longer needed")

    assert coord.composted and coord.composted[0][0] == "int-to-cancel"
    assert "no longer needed" in coord.composted[0][1]
    assert coord.manifested == []


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_coordinator_exception_does_not_break_submit(queue, _restore_coordinator):
    class Boom:
        def project_from_task(self, *a, **kw):
            raise RuntimeError("boom")

        def manifest(self, *a, **kw):
            raise RuntimeError("boom")

        def compost(self, *a, **kw):
            raise RuntimeError("boom")

    register_coordinator(Boom())
    tid = queue.submit(_basic_task())
    assert tid.startswith("claude_")
    assert (queue.pending_dir / f"{tid}.yaml").exists()


def test_no_coordinator_is_noop(queue, _restore_coordinator):
    register_coordinator(None)
    tid = queue.submit(_basic_task())
    queue.claim()
    queue.complete(tid)
    tid2 = queue.submit(_basic_task())
    queue.claim()
    queue.fail(tid2, error="x")
    # No crash — coordinator slot empty, all paths no-op cleanly.
    assert (queue.completed_dir / f"{tid}.yaml").exists()
    assert (queue.failed_dir / f"{tid2}.yaml").exists()
