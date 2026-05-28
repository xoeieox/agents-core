"""GPUClient tests using respx-mocked httpx layer."""

from __future__ import annotations

import pytest
import httpx
import respx

from agents_core.gpu_client import GPUClient, GPUQueueHTTPError


BASE = "http://gpu-queue:8405"


def make_client(**kwargs) -> GPUClient:
    return GPUClient(base_url=BASE, **kwargs)


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------

@respx.mock
def test_healthz_routes_get():
    route = respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok", "queue_depth": 0})
    )
    client = make_client()
    result = client.healthz()
    assert result["status"] == "ok"
    assert route.called


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

@respx.mock
def test_submit_routes_post_tasks():
    route = respx.post(f"{BASE}/v0/tasks").mock(
        return_value=httpx.Response(200, json={"id": "gpu_20260527_001234_pytest"})
    )
    client = make_client()
    task_id = client.submit({"task_type": "pytest"})
    assert task_id == "gpu_20260527_001234_pytest"
    assert route.called


@respx.mock
def test_submit_raises_on_400():
    respx.post(f"{BASE}/v0/tasks").mock(
        return_value=httpx.Response(400, json={"error": {"code": "bad_request", "message": "task_type is required"}})
    )
    client = make_client()
    with pytest.raises(GPUQueueHTTPError) as exc:
        client.submit({"priority": 50})
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Claim — 200 returns task, 204 returns None
# ---------------------------------------------------------------------------

@respx.mock
def test_claim_200_returns_task():
    task = {"id": "gpu_test_abc", "task_type": "render", "status": "running"}
    respx.post(f"{BASE}/v0/claim").mock(
        return_value=httpx.Response(200, json=task)
    )
    client = make_client()
    result = client.claim(current_model="Qwen2.5-72B")
    assert result is not None
    assert result["id"] == "gpu_test_abc"


@respx.mock
def test_claim_204_returns_none():
    respx.post(f"{BASE}/v0/claim").mock(
        return_value=httpx.Response(204)
    )
    client = make_client()
    result = client.claim()
    assert result is None


@respx.mock
def test_claim_sends_current_model():
    route = respx.post(f"{BASE}/v0/claim").mock(
        return_value=httpx.Response(204)
    )
    client = make_client()
    client.claim(current_model="mymodel")
    import json
    body = json.loads(route.calls[0].request.content)
    assert body["current_model"] == "mymodel"


# ---------------------------------------------------------------------------
# Complete / fail / preempt
# ---------------------------------------------------------------------------

@respx.mock
def test_complete_routes_post():
    task_id = "gpu_20260527_complete"
    route = respx.post(f"{BASE}/v0/tasks/{task_id}/complete").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = make_client()
    result = client.complete(task_id, output_path="/tmp/out.txt", result_summary="done")
    assert result == {"ok": True}
    assert route.called


@respx.mock
def test_complete_404_raises():
    task_id = "gpu_bogus"
    respx.post(f"{BASE}/v0/tasks/{task_id}/complete").mock(
        return_value=httpx.Response(404, json={"error": {"code": "not_found", "message": "not in active"}})
    )
    client = make_client()
    with pytest.raises(GPUQueueHTTPError) as exc:
        client.complete(task_id)
    assert exc.value.status_code == 404


@respx.mock
def test_fail_routes_post():
    task_id = "gpu_20260527_fail"
    route = respx.post(f"{BASE}/v0/tasks/{task_id}/fail").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = make_client()
    result = client.fail(task_id, error="exploded")
    assert result == {"ok": True}
    assert route.called


@respx.mock
def test_preempt_routes_post():
    task_id = "gpu_20260527_preempt"
    route = respx.post(f"{BASE}/v0/tasks/{task_id}/preempt").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = make_client()
    result = client.preempt(task_id)
    assert result == {"ok": True}
    assert route.called


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@respx.mock
def test_cancel_routes_post():
    task_id = "gpu_20260527_cancel"
    route = respx.post(f"{BASE}/v0/tasks/{task_id}/cancel").mock(
        return_value=httpx.Response(200, json={"cancelled": True})
    )
    client = make_client()
    result = client.cancel(task_id, reason="test")
    assert result == {"cancelled": True}
    assert route.called


# ---------------------------------------------------------------------------
# State readers
# ---------------------------------------------------------------------------

@respx.mock
def test_state_routes_get():
    route = respx.get(f"{BASE}/v0/state").mock(
        return_value=httpx.Response(200, json={"mode": "idle", "queue_depth": 0, "active_task_age_seconds": None})
    )
    client = make_client()
    result = client.state()
    assert result["mode"] == "idle"
    assert route.called


@respx.mock
def test_pending_routes_get():
    route = respx.get(f"{BASE}/v0/pending").mock(
        return_value=httpx.Response(200, json=[{"id": "gpu_p1", "task_type": "test"}])
    )
    client = make_client()
    result = client.pending()
    assert len(result) == 1
    assert route.called


@respx.mock
def test_active_routes_get():
    route = respx.get(f"{BASE}/v0/active").mock(
        return_value=httpx.Response(200, json={"id": "gpu_a1", "task_type": "active"})
    )
    client = make_client()
    result = client.active()
    assert result["id"] == "gpu_a1"
    assert route.called


@respx.mock
def test_active_none_on_null_response():
    # FastAPI serializes Python None → JSON "null"; simulate with text="null"
    respx.get(f"{BASE}/v0/active").mock(
        return_value=httpx.Response(
            200, text="null", headers={"content-type": "application/json"}
        )
    )
    client = make_client()
    result = client.active()
    assert result is None


@respx.mock
def test_recent_completed_with_limit():
    route = respx.get(f"{BASE}/v0/completed").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = make_client()
    client.recent_completed(limit=5)
    assert "limit=5" in str(route.calls[0].request.url)


@respx.mock
def test_recent_failed_routes_get():
    route = respx.get(f"{BASE}/v0/failed").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = make_client()
    client.recent_failed()
    assert route.called


@respx.mock
def test_history_routes_get():
    route = respx.get(f"{BASE}/v0/history").mock(
        return_value=httpx.Response(200, json=[{"event": "submitted"}])
    )
    client = make_client()
    result = client.history()
    assert result[0]["event"] == "submitted"
    assert route.called


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@respx.mock
def test_pause_routes_post():
    route = respx.post(f"{BASE}/v0/pause").mock(
        return_value=httpx.Response(200, json={"paused": True})
    )
    client = make_client()
    result = client.pause()
    assert result == {"paused": True}
    assert route.called


@respx.mock
def test_resume_routes_post():
    route = respx.post(f"{BASE}/v0/resume").mock(
        return_value=httpx.Response(200, json={"paused": False})
    )
    client = make_client()
    result = client.resume()
    assert result == {"paused": False}
    assert route.called


@respx.mock
def test_is_paused_routes_get():
    route = respx.get(f"{BASE}/v0/paused").mock(
        return_value=httpx.Response(200, json={"paused": False})
    )
    client = make_client()
    result = client.is_paused()
    assert result == {"paused": False}
    assert route.called


@respx.mock
def test_update_runner_state_routes_post():
    route = respx.post(f"{BASE}/v0/runner-state").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    client = make_client()
    result = client.update_runner_state(mode="processing", current_model="Qwen")
    assert result == {"ok": True}
    assert route.called


@respx.mock
def test_cleanup_routes_post():
    route = respx.post(f"{BASE}/v0/cleanup").mock(
        return_value=httpx.Response(200, json={"removed": 3})
    )
    client = make_client()
    result = client.cleanup(max_age_hours=24)
    assert result == {"removed": 3}
    assert "max_age_hours=24" in str(route.calls[0].request.url)


# ---------------------------------------------------------------------------
# Bearer-token header construction
# ---------------------------------------------------------------------------

@respx.mock
def test_bearer_token_sent_in_header():
    route = respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = GPUClient(base_url=BASE, token="my_secret_token")
    client.healthz()
    auth_header = route.calls[0].request.headers.get("Authorization", "")
    assert auth_header == "Bearer my_secret_token"


@respx.mock
def test_no_token_no_auth_header():
    route = respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = GPUClient(base_url=BASE, token="")
    client.healthz()
    auth_header = route.calls[0].request.headers.get("Authorization", "")
    assert auth_header == ""


# ---------------------------------------------------------------------------
# Error raising
# ---------------------------------------------------------------------------

@respx.mock
def test_error_raises_gpu_queue_http_error():
    respx.get(f"{BASE}/v0/state").mock(
        return_value=httpx.Response(500, json={"error": {"code": "internal", "message": "boom"}})
    )
    client = make_client()
    with pytest.raises(GPUQueueHTTPError) as exc:
        client.state()
    assert exc.value.status_code == 500


@respx.mock
def test_401_raises_gpu_queue_http_error():
    respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(401, json={"error": {"code": "unauthorized", "message": "invalid token"}})
    )
    client = make_client()
    with pytest.raises(GPUQueueHTTPError) as exc:
        client.healthz()
    assert exc.value.status_code == 401
