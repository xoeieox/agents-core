"""gpu-queue HTTP client — thin httpx wrapper for gpu-queue-server.

Reads configuration from environment:
  GPU_QUEUE_SERVER         — base URL, e.g. http://203.0.113.10:8405
  GPU_QUEUE_BEARER_TOKEN   — optional bearer token (must match server)
  GPU_QUEUE_CLIENT_TIMEOUT — per-request timeout in seconds (default 10.0)

Raises GPUQueueHTTPError on non-2xx responses.

Invariant 11: GPUClient.submit() MUST NOT auto-retry on timeout.
A server-success/client-timeout would double-queue the task. If submit()
times out, httpx.TimeoutException propagates — the caller must handle it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class GPUQueueHTTPError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class GPUClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ):
        self._base_url = (base_url or os.environ.get("GPU_QUEUE_SERVER", "")).rstrip("/")
        _token = token if token is not None else os.environ.get("GPU_QUEUE_BEARER_TOKEN", "")
        _timeout = timeout if timeout is not None else float(
            os.environ.get("GPU_QUEUE_CLIENT_TIMEOUT", "10.0")
        )
        headers = {}
        if _token:
            headers["Authorization"] = f"Bearer {_token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=_timeout,
        )

    def _check(self, resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise GPUQueueHTTPError(resp.status_code, body)
        return resp

    def healthz(self) -> dict:
        return self._check(self._client.get("/healthz")).json()

    # ------------------------------------------------------------------
    # Submit (Invariant 11: no auto-retry — TimeoutException propagates)
    # ------------------------------------------------------------------

    def submit(self, task: dict[str, Any]) -> str:
        """Submit a task. Returns the task_id (str).

        Does NOT auto-retry on timeout. A server-success/client-timeout
        would double-queue if retried (Invariant 11).
        """
        resp = self._check(self._client.post("/v0/tasks", json=task))
        return resp.json()["id"]

    # ------------------------------------------------------------------
    # Consumer surface
    # ------------------------------------------------------------------

    def claim(self, current_model: str | None = None) -> dict | None:
        """Claim the highest-priority pending task. Returns task dict or None (204)."""
        resp = self._client.post("/v0/claim", json={"current_model": current_model})
        if resp.status_code == 204:
            return None
        self._check(resp)
        return resp.json()

    def complete(
        self,
        task_id: str,
        output_path: str | None = None,
        result_summary: str | None = None,
    ) -> dict:
        body = {"output_path": output_path, "result_summary": result_summary}
        return self._check(
            self._client.post(f"/v0/tasks/{task_id}/complete", json=body)
        ).json()

    def fail(self, task_id: str, error: str = "Unknown error") -> dict:
        return self._check(
            self._client.post(f"/v0/tasks/{task_id}/fail", json={"error": error})
        ).json()

    def preempt(self, task_id: str) -> dict:
        return self._check(
            self._client.post(f"/v0/tasks/{task_id}/preempt")
        ).json()

    def cancel(self, task_id: str, reason: str = "") -> dict:
        return self._check(
            self._client.post(f"/v0/tasks/{task_id}/cancel", json={"reason": reason})
        ).json()

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    def state(self) -> dict:
        return self._check(self._client.get("/v0/state")).json()

    def pending(self) -> list[dict]:
        return self._check(self._client.get("/v0/pending")).json()

    def active(self) -> dict | None:
        return self._check(self._client.get("/v0/active")).json()

    def recent_completed(self, limit: int = 10) -> list[dict]:
        return self._check(
            self._client.get("/v0/completed", params={"limit": limit})
        ).json()

    def recent_failed(self, limit: int = 10) -> list[dict]:
        return self._check(
            self._client.get("/v0/failed", params={"limit": limit})
        ).json()

    def history(self, limit: int = 50) -> list[dict]:
        return self._check(
            self._client.get("/v0/history", params={"limit": limit})
        ).json()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def pause(self) -> dict:
        return self._check(self._client.post("/v0/pause")).json()

    def resume(self) -> dict:
        return self._check(self._client.post("/v0/resume")).json()

    def is_paused(self) -> dict:
        return self._check(self._client.get("/v0/paused")).json()

    def update_runner_state(self, **kw) -> dict:
        return self._check(
            self._client.post("/v0/runner-state", json=kw)
        ).json()

    def cleanup(self, max_age_hours: float = 48) -> dict:
        return self._check(
            self._client.post("/v0/cleanup", params={"max_age_hours": max_age_hours})
        ).json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
