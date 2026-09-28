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
  GW_ACQUIRE_MARGIN_SEC  — margin for acquire timeout above wake deadline (default 30);
                           also the margin used by _defer_wait_timeout()
  GW_ACQUIRE_TIMEOUT_SEC — (optional) override acquire timeout; if set below the
                           effective wake deadline, a warning is emitted
  DOORMAN_MAX_HOLD_TIMEOUT_SEC — must match doorman-server's setting (default 900).
                           Read by _defer_wait_timeout() to size a retry-loop budget
                           for a `deferrable` acquire waiting on the foreground-priority
                           gate's pending-defer wait-list (gw-router-phase1-foreground-gate).
  GW_SERVE_STOP_TIMEOUT_SEC — must match doorman-server's setting (default 120). Read
                           by _force_stop_timeout() to size force_stop()'s default HTTP
                           timeout above the server's own ssh-subprocess budget for
                           `gw-serve stop`, so a slow-but-normal unload is never
                           misread as DoormanUnreachable before the server even answers
                           (agents-core-doorman-force-stop-timeout-truthfulness-v0).

Raises DoormanUnreachable when the HTTP layer itself fails (connection error,
timeout). The operator treats DoormanUnreachable exactly like status:"wake_failed":
it cannot guarantee GW is awake, so it applies the per-surface on_wake_fail policy.

Acquire statuses:
  "serving" — GW is serving; lease registered and keepawake hold placed
  "deferred" — GW is serving a controller-owned non-big mode; caller cannot use big
  "wake_failed" — GW failed to wake or serve; big endpoint unavailable for non-controller reason
  "pending_defer" — foreground-priority gate (gw-router-phase1-foreground-gate): a
                     `deferrable`-class acquire is queued on the doorman's pending-defer
                     wait-list because a `protected` lease (or the brake) is active; no
                     lease was registered. Retry the acquire call (same work_id — the
                     wait-list anchors enqueued_at to the first call) to check for release;
                     see DEFER_WAIT_TIMEOUT_SEC / _defer_wait_timeout() for how long a
                     retry loop may need to keep polling before max-hold-timeout releases
                     it unconditionally.
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


def _force_stop_timeout() -> float:
    """Derive the force_stop() HTTP timeout from the server's own stop budget.

    Same spirit as _gw_acquire_timeout(): the client must outlive the
    server's GW_SERVE_STOP_TIMEOUT_SEC (default 120s) budget for the ssh
    subprocess, or a stop the server is about to correctly resolve as
    stop_in_progress gets misread as DoormanUnreachable before the server
    even answers (agents-core-doorman-force-stop-timeout-truthfulness-v0).

    Reads the same GW_SERVE_STOP_TIMEOUT_SEC env var doorman_server.py reads,
    plus GW_ACQUIRE_MARGIN_SEC (shared margin convention with
    _gw_acquire_timeout()) to cover the server's own post-subprocess
    lease-recheck and serving-probe overhead on top of the subprocess budget
    itself.
    """
    gw_serve_stop_timeout_sec = float(os.environ.get("GW_SERVE_STOP_TIMEOUT_SEC", "120"))
    gw_acquire_margin_sec = float(os.environ.get("GW_ACQUIRE_MARGIN_SEC", "30"))
    return gw_serve_stop_timeout_sec + gw_acquire_margin_sec


def _defer_wait_timeout() -> float:
    """Derive a request timeout long enough to cover one defer-gated acquire poll.

    Foreground-priority gate (gw-router-phase1-foreground-gate): a `deferrable`
    acquire against a `protected` lease (or the brake) does not block server-side —
    it returns "pending_defer" immediately (see DoormanClient.acquire docstring) — so
    the DEFAULT client timeout is fine for a single call. This helper exists for
    callers that want to size a retry-loop budget against the server's own
    non-resettable max-hold-timeout (DOORMAN_MAX_HOLD_TIMEOUT_SEC, default 900s):
    a `deferrable` job is guaranteed to be releasable within this many seconds of
    its first enqueue, regardless of how many new `protected` leases arrive meanwhile.

    Reads the same DOORMAN_MAX_HOLD_TIMEOUT_SEC env var doorman_server.py reads,
    plus GW_ACQUIRE_MARGIN_SEC (shared margin convention with _gw_acquire_timeout()).
    """
    max_hold_sec = int(os.environ.get("DOORMAN_MAX_HOLD_TIMEOUT_SEC", "900"))
    margin_sec = int(os.environ.get("GW_ACQUIRE_MARGIN_SEC", "30"))
    return float(max_hold_sec + margin_sec)


class DoormanUnreachable(Exception):
    """HTTP transport failure reaching the doorman service."""


# Named /lease/acquire refusals (doorman-flashnext-serving-admission-v0, S1).
# These are the body flags the server already returns on a 409 — the 409 is an
# EXPECTED outcome for a caller that needs no wake, not a transport failure, so
# the acquire path answers with the parsed dict instead of letting
# httpx.HTTPStatusError escape. A body that is unparseable or carries none of
# these flags takes TODAY's error path (raise) — never a named skip (I4: the
# preflight/lease layer must not become an availability SPOF, and an unknown
# body is never read as a permission to skip).
_ACQUIRE_REFUSAL_FLAGS = ("creative_occupied", "flashnext_occupied", "contended")


def _parse_acquire_refusal(response: "httpx.Response") -> dict | None:
    """Parse a 409 /lease/acquire body into the documented refusal dict.

    Returns the body verbatim when it is a dict carrying at least one of the
    server's named refusal flags with a truthy value; None otherwise (caller
    re-raises — I4 fail-open to today's behavior).

    Trust note: the body flags are ADVISORY within the ratified loopback trust
    model (bearer auth unset, live-verified) — they classify a refusal the
    server already made; they never grant anything.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    if not any(body.get(flag) is True for flag in _ACQUIRE_REFUSAL_FLAGS):
        return None
    return body


def is_flashnext_occupied(resp: dict) -> bool:
    """Return True if the acquire was refused because the flash-next seat
    (:30000) holds GPU 0 whole-card during an active handover window.

    Module-level so consumers can call it without going through the class
    (mock-safety: tests patch the DoormanClient class, and a MagicMock
    attribute would read truthy for every response).
    """
    return bool(isinstance(resp, dict) and resp.get("flashnext_occupied"))


def is_creative_occupied(resp: dict) -> bool:
    """Return True if the acquire was refused because the creative 70B holds the GPU."""
    return bool(isinstance(resp, dict) and resp.get("creative_occupied"))


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

    def _post(self, path: str, body: dict, timeout: float | None = None,
              named_refusals: bool = False) -> dict:
        """POST a JSON body and return the parsed response dict.

        named_refusals=True (the /lease/acquire path, S1) turns a 409 whose body
        carries one of the server's named refusal flags into a returned dict
        instead of an escaping httpx.HTTPStatusError. Any other non-2xx — and a
        409 with an unparseable / flag-less body — keeps raise_for_status()'s
        today's-error-path behavior (I4 fail-open; no new silent skip).
        named_refusals defaults False: every other endpoint is byte-identical.
        Transport failures keep raising DoormanUnreachable (unchanged).
        """
        try:
            kwargs = {"json": body}
            if timeout is not None:
                kwargs["timeout"] = timeout
            resp = self._client.post(path, **kwargs)
            if named_refusals and resp.status_code == 409:
                refusal = _parse_acquire_refusal(resp)
                if refusal is not None:
                    return refusal
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

    def acquire(self, node: str, work_id: str, ttl_sec: int, reason: str, role: str = "worker", timeout: float | None = None, principal: str | None = None, require_drain_clear: bool = False, lease_kind: str = "inference", lease_class: str | None = None, mode: str | None = None, accept_flashnext_seat: bool = False) -> dict:
        """Acquire a lease for node.

        Args:
          role: optional role descriptor (default "worker"). Use "mode-controller"
                if acquiring as the flip-controller so the doorman recognizes
                controller ownership and defers to it.
          mode: optional requested serve mode ("big" or "dual"). Only honored when
                role=="mode-controller" and no foreign controller lease is active —
                then ensure_serving issues gw-serve <mode> instead of deferring,
                overriding DOORMAN_DEFAULT_SERVE_MODE for this acquire
                (agents-core-doorman-mode-bearing-acquire-v0). Omitted or empty
                string is treated as no mode supplied — byte-identical to omission.
                Ignored (not an error) when role != "mode-controller".
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
          lease_class: foreground-priority gate class (gw-router-phase1-foreground-gate):
                      "protected" (never deferred — interactive PM session, measured
                      gates) or "deferrable" (yields to an active protected lease/brake;
                      fixers, code-review, subagents, Hermes). Omitting sends no `class`
                      field — the server defaults missing class to "deferrable" (safe).
                      Invalid values are rejected 400 by the server.
          accept_flashnext_seat: OPT-IN to being served by an ALREADY-SERVING flash-next
                      seat without a wake (doorman-flashnext-serving-admission-v0, S2).
                      When True AND no `mode` was supplied AND role != "mode-controller"
                      AND the doorman's own guard-computed seat probe reads
                      up_registered, the doorman SKIPS the 27B wake, registers the lease
                      normally and answers status "serving" with serve_axis="flashnext"
                      on the lease dict. Default False is byte-identical to today: the
                      flash-next window guard refuses with the flashnext_occupied 409.
                      Set it only when the caller can actually dial the flash-next seat
                      — a day-seat-only caller that opts in would consume a lease and
                      then fail against :8081. `up_unverified` (identity unverified, e.g.
                      mid-load) keeps today's refusal: a refused acquire is cheap, a
                      grant to a squatter is not.

        Returns dict with status field (or contended/creative_occupied/flashnext_occupied
        sentinel). Post-S1 a /lease/acquire refusal is a NAMED OUTCOME in the dict, not
        an exception — httpx.HTTPStatusError no longer escapes this path for the
        server's named 409 bodies (transport failures still raise DoormanUnreachable):
          "serving" — GW is serving; lease registered and keepawake hold placed
          "deferred" — GW is serving a controller-owned non-big mode; no lease registered
          "pending_defer" — a `deferrable` acquire is queued behind an active `protected`
                            lease/brake; no lease registered — retry to check for release
          "wake_failed" — GW failed to wake
          "invalid_mode" — mode was supplied but is not in the accepted set (only
                            possible when a mode was supplied); no subprocess was invoked
          {"ok": False, "contended": True} — drain gate active; another group holds a lease
          {"ok": False, "creative_occupied": True} — Llama-3.3-70B holds the GPU; check is_creative_occupied()
          {"ok": False, "flashnext_occupied": True} — the flash-next seat holds GPU 0 (active
                            handover window) and this acquire needed a wake; check
                            is_flashnext_occupied()
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
        if lease_class is not None:
            body["class"] = lease_class
        if mode:
            body["mode"] = mode
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

    def force_stop(
        self, node: str = "gravitywell", exclude_principal: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """Manually unload the GW model now, bypassing GW_STOP_GRACE_SEC.

        Args:
          exclude_principal: when set, a lease held by this principal does not
                             block the stop (self-exclusion, same convention as
                             drain_count()). Ghost leases (__GHOST_LEASE__) and
                             any other worker's lease still block.
          timeout: optional per-request timeout override. Defaults to
                   _force_stop_timeout(), sized above the server's own
                   GW_SERVE_STOP_TIMEOUT_SEC ssh-subprocess budget so a slow-
                   but-normal unload is never misread as DoormanUnreachable
                   before the server answers. An explicit value here always
                   wins over the derived default.

        Returns dict with a status field:
          "stopped" / "already_stopped" — model unloaded (or already was)
          "blocked" — refused; includes active_leases: [{work_id, principal}, ...]
          "stop_in_progress" — the ssh subprocess exceeded the server's
                                GW_SERVE_STOP_TIMEOUT_SEC but GW is still
                                reachable and shutting down; includes
                                waited_seconds. Not a failure — the unload is
                                proceeding; poll again or check /status. The
                                server's own background reconciler resolves a
                                genuinely wedged stop on a bounded budget.
          "error" — gw-serve stop failed; includes error/exit_code
        """
        body: dict = {"node": node}
        if exclude_principal is not None:
            body["exclude_principal"] = exclude_principal
        _timeout = timeout if timeout is not None else _force_stop_timeout()
        return self._post("/v0/force-stop", body, timeout=_timeout)

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

    def brake(self, reason: str, ttl_s: int | None = None, node: str = "gravitywell") -> dict:
        """Hold the emergency defer-only brake (gw-router-phase1-foreground-gate).

        Defers new `deferrable` dispatch (same wait-list as an active `protected`
        lease) without clearing or killing any existing lease. Bounded TTL — the
        brake always auto-expires server-side (default DOORMAN_BRAKE_TTL_SEC=900s
        if ttl_s is omitted); a job already waiting still releases at its own
        enqueue + max-hold-timeout even while the brake is held.

        Returns {"braked": True, "expires_at": <epoch>}.
        """
        body: dict = {"reason": reason, "node": node}
        if ttl_s is not None:
            body["ttl_s"] = ttl_s
        return self._post("/v0/brake", body)

    def brake_release(self, node: str = "gravitywell") -> dict:
        """Release the emergency brake early. Returns {"braked": False}."""
        return self._post("/v0/brake/release", {"node": node})

    @staticmethod
    def is_deferred(resp: dict) -> bool:
        """Convenience predicate: is this acquire response a deferred outcome?

        Returns True iff resp["status"] == "deferred", False otherwise.
        """
        return resp.get("status") == "deferred"

    @staticmethod
    def is_pending_defer(resp: dict) -> bool:
        """Convenience predicate: is this acquire response queued on the
        foreground-priority gate's pending-defer wait-list?

        Returns True iff resp["status"] == "pending_defer", False otherwise.
        """
        return resp.get("status") == "pending_defer"

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
