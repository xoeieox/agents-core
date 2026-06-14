"""doorman HTTP client — thin httpx wrapper for doorman-server.

Reads configuration from environment:
  DOORMAN_SERVER         — base URL (default http://127.0.0.1:8407)
  DOORMAN_BEARER_TOKEN   — optional bearer token (must match server)
  DOORMAN_CLIENT_TIMEOUT — per-request timeout in seconds (default 30.0)

Raises DoormanUnreachable when the HTTP layer itself fails (connection error,
timeout). The operator treats DoormanUnreachable exactly like status:"wake_failed":
it cannot guarantee GW is awake, so it applies the per-surface on_wake_fail policy.

Acquire statuses:
  "serving" — GW is serving; lease registered and keepawake hold placed
  "deferred" — GW is serving a controller-owned non-big mode; caller cannot use big
  "wake_failed" — GW failed to wake or serve; big endpoint unavailable for non-controller reason
"""

from __future__ import annotations

import os

import httpx


class DoormanUnreachable(Exception):
    """HTTP transport failure reaching the doorman service."""


class DoormanClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ):
        self._base_url = (
            base_url or os.environ.get("DOORMAN_SERVER", "http://127.0.0.1:8407")
        ).rstrip("/")
        _token = token if token is not None else os.environ.get("DOORMAN_BEARER_TOKEN", "")
        _timeout = timeout if timeout is not None else float(
            os.environ.get("DOORMAN_CLIENT_TIMEOUT", "30.0")
        )
        headers = {}
        if _token:
            headers["Authorization"] = f"Bearer {_token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=_timeout,
        )

    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = self._client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as e:
            raise DoormanUnreachable(f"doorman unreachable at {self._base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise DoormanUnreachable(f"doorman timeout at {self._base_url}: {e}") from e

    def _get(self, path: str) -> dict:
        try:
            resp = self._client.get(path)
            resp.raise_for_status()
            return resp.json()
        except httpx.TransportError as e:
            raise DoormanUnreachable(f"doorman unreachable at {self._base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise DoormanUnreachable(f"doorman timeout at {self._base_url}: {e}") from e

    def acquire(self, node: str, work_id: str, ttl_sec: int, reason: str, role: str = "worker") -> dict:
        """Acquire a lease for node.

        Args:
          role: optional role descriptor (default "worker"). Use "mode-controller"
                if acquiring as the flip-controller so the doorman recognizes
                controller ownership and defers to it.

        Returns dict with status field:
          "serving" — GW is serving; lease registered and keepawake hold placed
          "deferred" — GW is serving a controller-owned non-big mode; no lease registered
          "wake_failed" — GW failed to wake
        """
        return self._post("/lease/acquire", {
            "node": node,
            "work_id": work_id,
            "ttl_sec": ttl_sec,
            "reason": reason,
            "role": role,
        })

    def release(self, node: str, work_id: str) -> None:
        """Release a lease. Idempotent — unknown work_id is a no-op."""
        self._post("/lease/release", {"node": node, "work_id": work_id})

    def status(self) -> dict:
        return self._get("/status")

    def healthz(self) -> dict:
        return self._get("/healthz")

    def mode_owner(self, node: str = "gravitywell") -> dict | None:
        """Get the /v0/mode-owner deference-liveness probe.

        Args:
          node: node name (default "gravitywell")

        Returns dict with keys: node, controller, active, owner_lease_held,
        owner_lease_age_sec, owner_lease_stale. Returns None if the doorman
        does not support this endpoint (404 — pre-1b doorman) or is unreachable.
        """
        try:
            return self._get(f"/v0/mode-owner?node={node}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except DoormanUnreachable:
            return None

    @staticmethod
    def is_deferred(resp: dict) -> bool:
        """Convenience predicate: is this acquire response a deferred outcome?

        Returns True iff resp["status"] == "deferred", False otherwise.
        """
        return resp.get("status") == "deferred"

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
