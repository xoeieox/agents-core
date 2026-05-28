"""End-to-end tests: real temp queue dir + local uvicorn subprocess + GPUClient."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agents_core.gpu import GPUQueue
from agents_core.gpu_client import GPUClient, GPUQueueHTTPError


E2E_PORT = 18405


@pytest.fixture(scope="module")
def e2e_env(tmp_path_factory):
    """Spin up gpu-queue-server subprocess against a temp queue dir. Yield (client, queue)."""
    queue_dir = tmp_path_factory.mktemp("e2e_gpu") / "gpu-queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "agents_core.gpu_server"],
        env={
            "GPU_QUEUE_DIR": str(queue_dir),
            "GPU_QUEUE_BIND_HOST": "127.0.0.1",
            "GPU_QUEUE_BIND_PORT": str(E2E_PORT),
            "GPU_QUEUE_LOG_LEVEL": "error",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{E2E_PORT}"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/healthz", timeout=1.0)
            if resp.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(
            f"Server did not start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    client = GPUClient(base_url=base_url, timeout=10.0)
    queue = GPUQueue(queue_dir)
    yield client, queue, queue_dir

    proc.kill()
    proc.wait()
    client.close()


def test_e2e_healthz(e2e_env):
    client, queue, queue_dir = e2e_env
    h = client.healthz()
    assert h["status"] == "ok"
    assert str(queue_dir) in h["queue_dir"]
    assert h["queue_depth"] == 0


def test_e2e_submit_claim_complete(e2e_env):
    """Full submit → claim → complete round-trip via HTTP, verified against direct GPUQueue."""
    client, queue, queue_dir = e2e_env

    # Submit via HTTP
    task_id = client.submit({"task_type": "e2e_complete_test", "submitted_by": "pytest-e2e"})
    assert task_id.startswith("gpu_")

    # Verify via direct queue that it's in pending
    pending = queue.get_pending()
    assert any(t["id"] == task_id for t in pending)

    # Claim via HTTP
    task = client.claim(current_model=None)
    assert task is not None
    assert task["id"] == task_id
    assert task["status"] == "running"

    # Active via HTTP and directly
    active_http = client.active()
    assert active_http is not None
    assert active_http["id"] == task_id

    active_direct = queue.get_active()
    assert active_direct is not None
    assert active_direct["id"] == task_id

    # Complete via HTTP
    result = client.complete(task_id, output_path="/tmp/e2e_out.txt", result_summary="e2e done")
    assert result == {"ok": True}

    # Verify completed via direct queue
    completed = queue.get_recent_completed(limit=5)
    assert any(t["id"] == task_id for t in completed)

    # Active should now be null
    assert client.active() is None


def test_e2e_submit_claim_fail(e2e_env):
    """Full submit → claim → fail round-trip via HTTP."""
    client, queue, queue_dir = e2e_env

    task_id = client.submit({"task_type": "e2e_fail_test", "submitted_by": "pytest-e2e"})

    task = client.claim(current_model=None)
    assert task is not None
    assert task["id"] == task_id

    result = client.fail(task_id, error="e2e test failure")
    assert result == {"ok": True}

    failed = queue.get_recent_failed(limit=5)
    assert any(t["id"] == task_id for t in failed)
    matching = next(t for t in failed if t["id"] == task_id)
    assert matching["error"] == "e2e test failure"

    assert client.active() is None


def test_e2e_state_equivalence(e2e_env):
    """HTTP state matches direct GPUQueue.get_state() for key fields."""
    client, queue, queue_dir = e2e_env

    # Submit a task to give the queue some depth
    task_id = client.submit({"task_type": "e2e_state_check"})

    state_http = client.state()
    state_direct = queue.get_state()

    assert state_http["queue_depth"] == state_direct["queue_depth"]
    assert "active_task_age_seconds" in state_http  # server enriches state

    # Clean up
    client.cancel(task_id, reason="state equivalence cleanup")


def test_e2e_empty_claim_returns_none(e2e_env):
    """Claiming from an empty queue returns None (204)."""
    client, queue, queue_dir = e2e_env

    # Drain any remaining pending tasks
    while True:
        t = client.claim()
        if t is None:
            break
        client.complete(t["id"])

    result = client.claim()
    assert result is None


def test_e2e_pause_resume(e2e_env):
    """Pause and resume reflected on server state."""
    client, queue, queue_dir = e2e_env

    assert client.is_paused()["paused"] is False

    client.pause()
    assert client.is_paused()["paused"] is True
    assert client.healthz()["paused"] is True

    client.resume()
    assert client.is_paused()["paused"] is False


def test_e2e_cancel(e2e_env):
    """Cancel removes task from pending."""
    client, queue, queue_dir = e2e_env

    task_id = client.submit({"task_type": "e2e_cancel_test"})
    result = client.cancel(task_id, reason="e2e cancel test")
    assert result == {"cancelled": True}

    pending = queue.get_pending()
    assert not any(t["id"] == task_id for t in pending)


def test_e2e_complete_not_active_raises_404(e2e_env):
    """Completing a non-active task raises GPUQueueHTTPError(404)."""
    client, queue, queue_dir = e2e_env

    with pytest.raises(GPUQueueHTTPError) as exc:
        client.complete("gpu_bogus_e2e_id")
    assert exc.value.status_code == 404


def test_e2e_healthz_counts(e2e_env):
    """Healthz counts reflect the actual filesystem state."""
    client, queue, queue_dir = e2e_env

    # Drain the queue first
    while True:
        t = client.claim()
        if t is None:
            break
        client.complete(t["id"])

    h_before = client.healthz()
    pending_before = h_before["counts"]["pending"]

    client.submit({"task_type": "e2e_count_test_1"})
    client.submit({"task_type": "e2e_count_test_2"})

    h_after = client.healthz()
    assert h_after["counts"]["pending"] == pending_before + 2

    # Clean up
    while True:
        t = client.claim()
        if t is None:
            break
        client.complete(t["id"])
