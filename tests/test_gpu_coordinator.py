"""Smoke tests for agents_core.gpu coordinator hook.

Verifies the dependency-inversion contract: a registered coordinator receives
lifecycle events (submit/complete/fail); without a coordinator, the queue is
a pure priority queue with no coordination side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agents_core import gpu as gpu_mod
from agents_core.gpu import (
    GPUQueue,
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
        # Scripted responses so tests can drive decision branches.
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
    """Save and restore the module-level _coordinator so tests are isolated."""
    prior = get_coordinator()
    yield
    register_coordinator(prior)


def test_register_and_get_coordinator(_restore_coordinator):
    assert get_coordinator() is None or get_coordinator() is not None  # idempotent get
    obj = object()
    register_coordinator(obj)
    assert get_coordinator() is obj
    register_coordinator(None)
    assert get_coordinator() is None


def test_submit_without_coordinator_queues_task(tmp_path, _restore_coordinator):
    register_coordinator(None)
    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest", "payload": {"x": 1}})
    assert tid
    assert (tmp_path / "pending" / f"{tid}.yaml").exists()


def test_submit_projects_when_coordinator_registered(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-projected"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({
        "task_type": "pytest",
        "submitted_by": "unit-test",
        "description": "run tests",
        "payload": {"target_heading": "tests/"},
    })

    assert len(coord.projected) == 1
    assert coord.projected[0]["task_id"] == tid
    assert coord.projected[0]["projected_by"] == "unit-test"
    assert coord.projected[0]["proposed_change"] == "run tests"
    assert coord.projected[0]["target_heading"] == "tests/"
    # Task was queued (projected decision → not a match)
    assert (tmp_path / "pending" / f"{tid}.yaml").exists()


def test_match_reinforce_returns_shared_task_id(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "match_reinforce"
    coord.next_shared_task_id = "gpu_prior_task"
    coord.next_intention_id = "int-reinforcer"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    # Match → caller gets the prior task_id, no new task queued.
    assert tid == "gpu_prior_task"
    assert list((tmp_path / "pending").glob("*.yaml")) == []


def test_match_manifested_returns_shared_task_id(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "match_manifested"
    coord.next_shared_task_id = "gpu_old_done"
    coord.next_intention_id = "int-manifested-match"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    assert tid == "gpu_old_done"
    assert list((tmp_path / "pending").glob("*.yaml")) == []


def test_complete_calls_manifest(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-manifest"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    q.claim()  # move pending -> active
    q.complete(tid, result_summary="ok")

    assert coord.manifested == [("int-to-manifest", tid)]
    assert coord.composted == []


def test_fail_calls_compost(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-compost"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    q.claim()
    q.fail(tid, error="simulated failure")

    assert coord.composted and coord.composted[0][0] == "int-to-compost"
    assert "simulated failure" in coord.composted[0][1]
    assert coord.manifested == []


def test_cancel_calls_compost(tmp_path, _restore_coordinator):
    # Regression: cancel() used to skip the compost cascade, leaving
    # intentions stuck in-flight forever. See 2026-04-23 zombie report.
    coord = _RecordingCoordinator()
    coord.next_decision = "projected"
    coord.next_intention_id = "int-to-cancel"
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    assert q.cancel(tid, reason="no longer needed")

    assert coord.composted and coord.composted[0][0] == "int-to-cancel"
    assert "no longer needed" in coord.composted[0][1]
    assert coord.manifested == []


def test_ignore_intention_registry_opt_out(tmp_path, _restore_coordinator):
    coord = _RecordingCoordinator()
    register_coordinator(coord)

    q = GPUQueue(queue_dir=tmp_path)
    q.submit({
        "task_type": "pytest",
        "payload": {"_ignore_intention_registry": True},
    })
    # Opt-out: coordinator must not be called.
    assert coord.projected == []


def test_coordinator_exception_does_not_break_submit(tmp_path, _restore_coordinator):
    """A broken coordinator must not break the queue — degrade, don't crash."""

    class Boom:
        def project_from_task(self, *a, **kw):
            raise RuntimeError("boom")

        def manifest(self, *a, **kw):
            raise RuntimeError("boom")

        def compost(self, *a, **kw):
            raise RuntimeError("boom")

    register_coordinator(Boom())
    q = GPUQueue(queue_dir=tmp_path)
    tid = q.submit({"task_type": "pytest"})
    assert tid  # still queued
    assert (tmp_path / "pending" / f"{tid}.yaml").exists()


def test_no_sys_path_probe_left_behind():
    """Regression guard: the old /srv/agents/scripts sys.path hack is gone."""
    import inspect
    src = inspect.getsource(gpu_mod)
    assert "/srv/agents/scripts" not in src, (
        "agents_core.gpu still contains the legacy sys.path append"
    )
    assert "_load_intention_registry" not in src, (
        "agents_core.gpu still has the legacy lazy-loader helper"
    )
