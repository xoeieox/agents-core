"""Endpoint-level tests for gpu_server using FastAPI TestClient."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents_core.gpu import GPUQueue
from agents_core.gpu_server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_queue(tmp_path):
    return tmp_path / "gpu-queue"


@pytest.fixture
def app(tmp_queue, monkeypatch):
    monkeypatch.delenv("GPU_QUEUE_BEARER_TOKEN", raising=False)
    return create_app(tmp_queue)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------

def test_healthz_empty_queue(client, tmp_queue):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert str(tmp_queue) in body["queue_dir"]
    assert body["queue_depth"] == 0
    assert body["mode"] in ("idle", "processing")
    assert body["paused"] is False
    assert body["active_task_age_seconds"] is None
    assert body["counts"]["pending"] == 0
    assert body["counts"]["active"] == 0


def test_healthz_reflects_queue_depth(client):
    client.post("/v0/tasks", json={"task_type": "test"})
    client.post("/v0/tasks", json={"task_type": "test"})
    resp = client.get("/healthz")
    body = resp.json()
    assert body["queue_depth"] == 2
    assert body["counts"]["pending"] == 2


# ---------------------------------------------------------------------------
# REQUIRED_FIELDS validation
# ---------------------------------------------------------------------------

def test_submit_missing_task_type_returns_400(client):
    resp = client.post("/v0/tasks", json={"priority": 50})
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "bad_request"


def test_submit_with_task_type_succeeds(client):
    resp = client.post("/v0/tasks", json={"task_type": "pytest"})
    assert resp.status_code == 200
    body = resp.json()
    assert "id" in body
    assert body["id"].startswith("gpu_")


# ---------------------------------------------------------------------------
# Submit → claim round-trip
# ---------------------------------------------------------------------------

def test_submit_then_claim_returns_task(client):
    r = client.post("/v0/tasks", json={"task_type": "round_trip"})
    task_id = r.json()["id"]

    claim = client.post("/v0/claim", json={"current_model": None})
    assert claim.status_code == 200
    task = claim.json()
    assert task["id"] == task_id
    assert task["task_type"] == "round_trip"
    assert task["status"] == "running"

    # Pending now empty, task in active
    assert client.get("/v0/pending").json() == []
    active = client.get("/v0/active").json()
    assert active is not None
    assert active["id"] == task_id


# ---------------------------------------------------------------------------
# Claim → complete
# ---------------------------------------------------------------------------

def test_claim_then_complete(client):
    r = client.post("/v0/tasks", json={"task_type": "to_complete"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})

    resp = client.post(f"/v0/tasks/{task_id}/complete", json={
        "output_path": "/tmp/out.txt",
        "result_summary": "done",
    })
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    # Task in completed, not active
    assert client.get("/v0/active").json() is None
    completed = client.get("/v0/completed").json()
    assert any(t["id"] == task_id for t in completed)


# ---------------------------------------------------------------------------
# Claim → fail
# ---------------------------------------------------------------------------

def test_claim_then_fail(client):
    r = client.post("/v0/tasks", json={"task_type": "to_fail"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})

    resp = client.post(f"/v0/tasks/{task_id}/fail", json={"error": "kaboom"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    assert client.get("/v0/active").json() is None
    failed = client.get("/v0/failed").json()
    assert any(t["id"] == task_id for t in failed)
    assert failed[0]["error"] == "kaboom"


# ---------------------------------------------------------------------------
# Empty claim → 204
# ---------------------------------------------------------------------------

def test_claim_empty_queue_returns_204(client):
    resp = client.post("/v0/claim", json={"current_model": None})
    assert resp.status_code == 204
    assert resp.content == b""


# ---------------------------------------------------------------------------
# Complete / fail / preempt 404 (H1 fold)
# ---------------------------------------------------------------------------

def test_complete_bogus_id_returns_404(client):
    resp = client.post("/v0/tasks/gpu_bogus_id/complete", json={})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_fail_bogus_id_returns_404(client):
    resp = client.post("/v0/tasks/gpu_bogus_id/fail", json={"error": "test"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_preempt_bogus_id_returns_404(client):
    resp = client.post("/v0/tasks/gpu_bogus_id/preempt")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_complete_active_task_succeeds(client):
    r = client.post("/v0/tasks", json={"task_type": "test_complete_ok"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})

    resp = client.post(f"/v0/tasks/{task_id}/complete", json={"output_path": None})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    completed = client.get("/v0/completed").json()
    assert any(t["id"] == task_id for t in completed)


def test_preempt_active_task_requeues(client):
    r = client.post("/v0/tasks", json={"task_type": "preemptible_task"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})

    resp = client.post(f"/v0/tasks/{task_id}/preempt")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    # Task should be back in pending
    pending = client.get("/v0/pending").json()
    assert any(t["id"] == task_id for t in pending)
    assert client.get("/v0/active").json() is None


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def test_cancel_pending_task(client):
    r = client.post("/v0/tasks", json={"task_type": "to_cancel"})
    task_id = r.json()["id"]

    resp = client.post(f"/v0/tasks/{task_id}/cancel", json={"reason": "test cancel"})
    assert resp.status_code == 200
    assert resp.json() == {"cancelled": True}

    pending = client.get("/v0/pending").json()
    assert not any(t["id"] == task_id for t in pending)


def test_cancel_non_pending_returns_404(client):
    resp = client.post("/v0/tasks/gpu_not_pending/cancel", json={"reason": ""})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# Caller-supplied id idempotency (M3 fold)
# ---------------------------------------------------------------------------

def test_caller_supplied_id_dedups(client):
    """Two submits with the same id → one task in queue (second overwrites first)."""
    body = {"task_type": "dedup_test", "id": "gpu_custom_id_test_123"}
    r1 = client.post("/v0/tasks", json=body)
    assert r1.status_code == 200
    assert r1.json()["id"] == "gpu_custom_id_test_123"

    r2 = client.post("/v0/tasks", json=body)
    assert r2.status_code == 200
    assert r2.json()["id"] == "gpu_custom_id_test_123"

    pending = client.get("/v0/pending").json()
    ids = [t["id"] for t in pending]
    assert ids.count("gpu_custom_id_test_123") == 1


# ---------------------------------------------------------------------------
# Concurrent-claim no-double-issue (Invariant 3 acceptance)
# ---------------------------------------------------------------------------

def test_concurrent_claim_no_double_issue(tmp_queue, monkeypatch):
    """N=20 tasks, M=8 concurrent claims — no task claimed twice."""
    monkeypatch.delenv("GPU_QUEUE_BEARER_TOKEN", raising=False)
    app = create_app(tmp_queue)

    N = 20
    M = 8

    with TestClient(app) as c:
        for i in range(N):
            r = c.post("/v0/tasks", json={"task_type": f"concurrent_{i}", "priority": 50})
            assert r.status_code == 200

    def do_claim():
        with TestClient(app) as c:
            r = c.post("/v0/claim", json={"current_model": None})
            if r.status_code == 200:
                return r.json()["id"]
            return None  # 204 — empty queue

    with concurrent.futures.ThreadPoolExecutor(max_workers=M) as executor:
        results = list(executor.map(lambda _: do_claim(), range(M)))

    claimed = [r for r in results if r is not None]
    assert len(claimed) == len(set(claimed)), (
        f"Double-issue detected! {len(claimed)} claims but only {len(set(claimed))} unique IDs"
    )
    assert len(claimed) <= N
    assert len(claimed) >= 1  # at least one task was claimed


# ---------------------------------------------------------------------------
# Priority + model-affinity
# ---------------------------------------------------------------------------

def test_priority_ordering(client):
    """High priority task is claimed before normal priority."""
    client.post("/v0/tasks", json={"task_type": "low", "priority": 80})
    client.post("/v0/tasks", json={"task_type": "high", "priority": 10})
    client.post("/v0/tasks", json={"task_type": "normal", "priority": 50})

    first = client.post("/v0/claim", json={"current_model": None}).json()
    assert first["task_type"] == "high"

    second = client.post("/v0/claim", json={"current_model": None}).json()
    assert second["task_type"] == "normal"


def test_model_affinity_tiebreak(client):
    """Same-model task wins tiebreak at equal priority."""
    client.post("/v0/tasks", json={"task_type": "other_model", "priority": 50, "model": "modelB"})
    client.post("/v0/tasks", json={"task_type": "matching_model", "priority": 50, "model": "modelA"})

    # Claim with current_model=modelA — matching_model should win
    first = client.post("/v0/claim", json={"current_model": "modelA"}).json()
    assert first["task_type"] == "matching_model"


# ---------------------------------------------------------------------------
# Stale-active visibility (M2 fold / Invariant 9)
# ---------------------------------------------------------------------------

def test_stale_active_age_in_healthz(client):
    """With an active task, healthz reports non-null active_task_age_seconds."""
    assert client.get("/healthz").json()["active_task_age_seconds"] is None

    client.post("/v0/tasks", json={"task_type": "age_test"})
    client.post("/v0/claim", json={"current_model": None})

    h = client.get("/healthz").json()
    assert h["active_task_age_seconds"] is not None
    assert h["active_task_age_seconds"] >= 0.0


def test_stale_active_age_in_state(client):
    """With an active task, /v0/state reports non-null active_task_age_seconds."""
    assert client.get("/v0/state").json().get("active_task_age_seconds") is None

    client.post("/v0/tasks", json={"task_type": "age_test_state"})
    client.post("/v0/claim", json={"current_model": None})

    s = client.get("/v0/state").json()
    assert s["active_task_age_seconds"] is not None
    assert s["active_task_age_seconds"] >= 0.0


def test_active_age_null_after_complete(client):
    """After completing a task, active_task_age_seconds returns to null."""
    r = client.post("/v0/tasks", json={"task_type": "age_reset"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})
    client.post(f"/v0/tasks/{task_id}/complete", json={})

    assert client.get("/healthz").json()["active_task_age_seconds"] is None


# ---------------------------------------------------------------------------
# Pause / resume (L2 fold)
# ---------------------------------------------------------------------------

def test_pause_resume_cycle(client):
    assert client.get("/v0/paused").json()["paused"] is False
    assert client.get("/healthz").json()["paused"] is False

    resp = client.post("/v0/pause")
    assert resp.status_code == 200
    assert resp.json() == {"paused": True}
    assert client.get("/v0/paused").json()["paused"] is True
    assert client.get("/healthz").json()["paused"] is True

    resp = client.post("/v0/resume")
    assert resp.status_code == 200
    assert resp.json() == {"paused": False}
    assert client.get("/v0/paused").json()["paused"] is False


# ---------------------------------------------------------------------------
# State readers shape
# ---------------------------------------------------------------------------

def test_state_shape(client):
    resp = client.get("/v0/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "mode" in body
    assert "queue_depth" in body
    assert "active_task_age_seconds" in body


def test_pending_shape(client):
    client.post("/v0/tasks", json={"task_type": "shape_test", "priority": 50})
    resp = client.get("/v0/pending")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "shape_test"


def test_completed_with_limit(client):
    for i in range(5):
        r = client.post("/v0/tasks", json={"task_type": f"t{i}"})
        task_id = r.json()["id"]
        client.post("/v0/claim", json={"current_model": None})
        client.post(f"/v0/tasks/{task_id}/complete", json={})

    resp = client.get("/v0/completed?limit=3")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_failed_endpoint(client):
    r = client.post("/v0/tasks", json={"task_type": "fail_test"})
    task_id = r.json()["id"]
    client.post("/v0/claim", json={"current_model": None})
    client.post(f"/v0/tasks/{task_id}/fail", json={"error": "test error"})

    resp = client.get("/v0/failed")
    assert resp.status_code == 200
    assert any(t["id"] == task_id for t in resp.json())


def test_history_endpoint(client):
    client.post("/v0/tasks", json={"task_type": "history_test"})
    resp = client.get("/v0/history")
    assert resp.status_code == 200
    events = resp.json()
    assert any(e.get("event") == "submitted" for e in events)


def test_history_with_limit(client):
    for i in range(10):
        client.post("/v0/tasks", json={"task_type": f"hist_{i}"})
    resp = client.get("/v0/history?limit=5")
    assert resp.status_code == 200
    assert len(resp.json()) <= 5


# ---------------------------------------------------------------------------
# Runner state
# ---------------------------------------------------------------------------

def test_runner_state_update(client):
    resp = client.post("/v0/runner-state", json={"mode": "processing", "current_model": "Qwen2.5-72B"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    state = client.get("/v0/state").json()
    assert state.get("mode") == "processing"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def test_cleanup_endpoint(client):
    resp = client.post("/v0/cleanup?max_age_hours=0")
    assert resp.status_code == 200
    assert "removed" in resp.json()


# ---------------------------------------------------------------------------
# Bearer-token middleware
# ---------------------------------------------------------------------------

def test_bearer_token_valid(tmp_queue, monkeypatch):
    monkeypatch.setenv("GPU_QUEUE_BEARER_TOKEN", "gpu_secret_123")
    app = create_app(tmp_queue)
    with TestClient(app) as c:
        resp = c.get("/healthz", headers={"Authorization": "Bearer gpu_secret_123"})
        assert resp.status_code == 200


def test_bearer_token_missing_returns_401(tmp_queue, monkeypatch):
    monkeypatch.setenv("GPU_QUEUE_BEARER_TOKEN", "gpu_secret_123")
    app = create_app(tmp_queue)
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 401


def test_bearer_token_wrong_returns_401(tmp_queue, monkeypatch):
    monkeypatch.setenv("GPU_QUEUE_BEARER_TOKEN", "gpu_secret_123")
    app = create_app(tmp_queue)
    with TestClient(app) as c:
        resp = c.get("/healthz", headers={"Authorization": "Bearer wrongtoken"})
        assert resp.status_code == 401


def test_no_bearer_token_env_no_auth_required(tmp_queue, monkeypatch):
    monkeypatch.delenv("GPU_QUEUE_BEARER_TOKEN", raising=False)
    app = create_app(tmp_queue)
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 200
