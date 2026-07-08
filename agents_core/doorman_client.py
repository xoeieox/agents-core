"""doorman HTTP client — thin httpx wrapper for doorman-server.

Reads configuration from environment:
  DOORMAN_SERVER         — base URL (default http://127.0.0.1:8407)
  DOORMAN_BEARER_TOKEN   — optional bearer token (must match server)
  DOORMAN_CLIENT_TIMEOUT — per-request timeout in seconds (default 30.0)
  GW_WAKE_DEADLINE_SEC   — GravityWell big-mode wake deadline in seconds, used by
                           doorman-server (default 180; also drives acquire timeout
                           coupling)
  DOORMAN_DEFAULT_SERVE_MODE — must match doorman-server's setting ("dual" default,
                           or "big"). When "dual", the acquire timeout also accounts
                           for GW_DUAL_WAKE_DEADLINE_SEC (mode-aware coupling,
                           gw-doorman-wake-to-default-mode-v0) — a real ~488s dual
                           cold-wake must not trip DoormanUnreachable client-side
                           before the server-side wake deadline.
  GW_DUAL_WAKE_DEADLINE_SEC — GravityWell dual-mode wake deadline in seconds, used by
                           doorman-server (default 720). Only consulted here when
                           DOORMAN_DEFAULT_SERVE_MODE is "dual".
  GW_ACQUIRE_MARGIN_SEC  — margin for acquire timeout above wake deadline (default 30)
  GW_ACQUIRE_TIMEOUT_SEC — (optional) override acquire timeout; if set below the
                           effective wake deadline, a warning is emitted

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
import warnings

import httpx


def _gw_acquire_timeout() -> float:
    """Derive the GW acquire timeout from the server's (mode-aware) wake deadline.

    Mode-aware (gw-doorman-wake-to-default-mode-v0): the doorman's default cold-wake
    target is "dual" (both vLLM slots; Devstral's cold-init measures ~488s), bounded
    server-side by GW_DUAL_WAKE_DEADLINE_SEC (default 720s), not the big-mode
    GW_WAKE_DEADLINE_SEC (default 180s). When DOORMAN_DEFAULT_SERVE_MODE is "dual"
    (the default — must match doorman-server's setting), the effective deadline is
    max(GW_WAKE_DEADLINE_SEC, GW_DUAL_WAKE_DEADLINE_SEC); when "big", it is
    GW_WAKE_DEADLINE_SEC alone (byte-identical to pre-dual-default behavior).

    Returns effective_deadline_sec + GW_ACQUIRE_MARGIN_SEC (defaults to 720 + 30 = 750s
    under the dual default; 180 + 30 = 210s with DOORMAN_DEFAULT_SERVE_MODE=big).

    The acquire HTTP request must outlive the doorman's wake deadline so that
    successful cold wakes are never misread as DoormanUnreachable timeouts. This
    function reads the same env vars doorman_server.py reads, ensuring coupling.

    If an explicit GW_ACQUIRE_TIMEOUT_SEC override is set below the effective
    deadline, emits a loud RuntimeWarning (not an error) so that mis-configurations
    are observable but not service-breaking.
    """
    gw_wake_deadline_sec = int(os.environ.get("GW_WAKE_DEADLINE_SEC", "180"))
    default_serve_mode = os.environ.get("DOORMAN_DEFAULT_SERVE_MODE", "dual").strip().lower()
    if default_serve_mode == "big":
        effective_deadline_sec = gw_wake_deadline_sec
    else:
        gw_dual_wake_deadline_sec = int(os.environ.get("GW_DUAL_WAKE_DEADLINE_SEC", "720"))
        effective_deadline_sec = max(gw_wake_deadline_sec, gw_dual_wake_deadline_sec)
    gw_acquire_margin_sec = int(os.environ.get("GW_ACQUIRE_MARGIN_SEC", "30"))
    derived_timeout = effective_deadline_sec + gw_acquire_margin_sec

    # Check for explicit override
    explicit_override = os.environ.get("GW_ACQUIRE_TIMEOUT_SEC")
    if explicit_override is not None:
        override_value = float(explicit_override)
        if override_value < effective_deadline_sec:
            warnings.warn(
                f"GW_ACQUIRE_TIMEOUT_SEC={override_value} is below the effective wake "
                f"deadline={effective_deadline_sec} (DOORMAN_DEFAULT_SERVE_MODE="
                f"{default_serve_mode!r}); acquire HTTP calls may timeout before the "
                f"doorman completes the wake, causing silent fallback to on_wake_fail "
                f"policy. Set GW_ACQUIRE_TIMEOUT_SEC >= {effective_deadline_sec} to fix.",
                RuntimeWarning,
                stacklevel=2,
            )
        return override_value

    return float(derived_timeout)


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

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        try:
            kwargs = {"json": body}
            if timeout is not None:
                kwargs["timeout"] = timeout
            resp = self._client.post(path, **kwargs)
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

    def acquire(self, node: str, work_id: str, ttl_sec: int, reason: str, role: str = "worker", timeout: float | None = None, principal: str | None = None, require_drain_clear: bool = False, lease_kind: str = "inference") -> dict:
        """Acquire a lease for node.

        Args:
          role: optional role descriptor (default "worker"). Use "mode-controller"
                if acquiring as the flip-controller so the doorman recognizes
                controller ownership and defers to it.
          timeout: optional per-request timeout override (default uses client timeout).
                   For GW acquire, pass _gw_acquire_timeout() to ensure the HTTP
                   timeout outlives the server's GW_WAKE_DEADLINE_SEC.
          principal: logical admission group for drain-gate exclusion. Worker leases
                     without a principal are stamped __GHOST_LEASE__ on the server —
                     always counted, never excluded. Pass the group's shared identifier
                     (e.g. council-delib-<run_id>) so sibling calls can self-exclude.
          require_drain_clear: when True, requests an atomic drain-gate check — the
                               server counts cross-group worker leases and registers
                               this lease only if none exist, all within one lock.
                               Returns {"ok": False, "contended": True} if gated.
                               Use is_contended() to detect this outcome.
                               Defaults False — all existing callers unchanged.
          lease_kind: "inference" (default) or "coordination". Coordination leases are
                      excluded from the drain-gate contention count (they hold no GPU
                      inference) but still counted by /v0/drain-count for flip-protection.
                      Omitting is byte-identical to "inference".

        Returns dict with status field (or contended/creative_occupied sentinel):
          "serving" — GW is serving; lease registered and keepawake hold placed
          "deferred" — GW is serving a controller-owned non-big mode; no lease registered
          "wake_failed" — GW failed to wake
          {"ok": False, "contended": True} — drain gate active; another group holds a lease
          {"ok": False, "creative_occupied": True} — Llama-3.3-70B holds the GPU; check is_creative_occupied()
        """
        body: dict = {
            "node": node,
            "work_id": work_id,
            "ttl_sec": ttl_sec,
            "reason": reason,
            "role": role,
        }
        if principal is not None:
            body["principal"] = principal
        if require_drain_clear:
            body["require_drain_clear"] = True
        if lease_kind != "inference":
            body["lease_kind"] = lease_kind
        return self._post("/lease/acquire", body, timeout=timeout)

    def release(self, node: str, work_id: str) -> None:
        """Release a lease. Idempotent — unknown work_id is a no-op."""
        self._post("/lease/release", {"node": node, "work_id": work_id})

    def status(self) -> dict:
        return self._get("/status")

    def healthz(self) -> dict:
        return self._get("/healthz")

    def drain_count(self, node: str = "gravitywell", exclude_principal: str | None = None) -> int | None:
        """Get the in-flight worker-lease count (drain-count) for node.

        Args:
          exclude_principal: when set, leases whose principal equals this value are
                             excluded from the count (same-group self-exclusion).
                             Ghost leases (__GHOST_LEASE__) are never excluded.
                             Without this param the endpoint behaves as before
                             (counts all workers) — flip-controller unaffected.

        Returns int on success, None on 404 (pre-this-unit doormen) or unreachable.
        """
        url = f"/v0/drain-count?node={node}"
        if exclude_principal is not None:
            url += f"&exclude_principal={exclude_principal}"
        try:
            data = self._get(url)
            return data.get("drain_count")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except DoormanUnreachable:
            return None

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

    @staticmethod
    def is_creative_occupied(resp: dict) -> bool:
        """Return True if the acquire was refused because the creative 70B holds the GPU."""
        return bool(resp.get("creative_occupied"))

    @staticmethod
    def is_contended(resp: dict) -> bool:
        """Convenience predicate: is this acquire response a CONTENDED outcome?

        Returns True iff the server's atomic drain-gate check found another-principal
        worker lease active. Distinct from deferred (controller owns mode) and
        wake_failed (GW not serving). Caller should retry within its deadline.
        """
        return bool(resp.get("contended"))

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
