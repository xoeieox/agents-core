"""doorman HTTP service — GravityWell power/lease lifecycle manager.

Runs on BRIX (always-on). Owns wake/suspend management for GravityWell so
individual callers never need to shell wake-gravitywell or gw-keepawake.

Entry point:  doorman-server  (console_scripts in pyproject.toml)
Port:         8407  (DOORMAN_BIND_PORT env var — live-verified free 2026-06-09;
                     8400/8401/8403/8404/8405/8406 are all occupied)
Bind:         127.0.0.1 by default  (BRIX-local; Unit 1 has no off-box clients)

Environment variables:
  DOORMAN_BIND_HOST      — uvicorn bind host (default 127.0.0.1)
  DOORMAN_BIND_PORT      — uvicorn bind port (default 8407)
  DOORMAN_BEARER_TOKEN   — optional shared bearer token
  DOORMAN_IDLE_LOG       — path for structured idle-lifecycle JSONL log
                           (default /var/log/doorman-idle.jsonl)
  DOORMAN_DEFER_TO_CONTROLLER — enable deference to flip-controller (default true);
                                 also kill-switch for non-big flips (REQUIRE_DOORMAN_DEFERENCE gate)
  DOORMAN_CONTROLLER_NAME     — identity of the mode-controller (default flip-controller);
                                reported by /v0/mode-owner
  GW_URL                 — GravityWell base URL (default http://203.0.113.11:8081)
                           NOTE: must match the GW_URL configured for agents_core.llm
                           (the operator reads the same env var for inference POSTs).
  GW_WAKE_DEADLINE_SEC   — max seconds to wait for GW to serve (default 180;
                           cold 77GB model load backstop — typical warm wake is ~10s)
  GW_HOLD_TTL_SEC        — keepawake hold TTL in seconds (default 120)
  GW_HOLD_REFRESH_SEC    — refresh interval for the keepawake hold (default 45)
  GW_STOP_GRACE_SEC      — seconds after last-release before the refresh thread
                           issues gw-serve stop (default 600; machine-economics
                           boundary that amortizes the ~25s cold-load against burst
                           gaps — not a human-rhythm value)
  DOORMAN_MODE_AWARE_ADMISSION — enable mode-aware deference before the _is_serving()
                                  fast path (HOLE 1), controller-lease-aware serving_mode
                                  (HOLE 2), and the three-state /v1/models big-model probe.
                                  Default false (lands dark). Set "true" or "1" to activate.

Safety properties (gravitywell-doorman-clean-stop-v0):
  - Doorman crash → GW stays POWERED, not suspended. The host-side guard
    (gw-idle-suspend.sh) blocks suspend while llama-server.service is active.
    A crashed doorman leaves the service running, so the guard keeps GW powered
    (safe, but no power saving). The doorman is a power-saving optimizer layered
    on the guard's safety floor — if the doorman never stops the service, the
    node degrades to "always powered," not to "unsafe suspend."
  - The doorman never causes an unsafe suspend: it can only *enable* suspend by
    first issuing gw-serve stop. A stop failure leaves GW powered (guard holds).
  - Leaked client lease → GC: stale leases (acquired_at + ttl_sec < now) are
    auto-released by the background refresh loop — a crashed operator cannot pin
    GW forever.
  - SSH-refresh failure → logged loudly, last_error set, retried on next tick;
    does NOT crash the thread and does NOT drop live leases.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import threading
import time
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("doorman-server")

GW_URL_DEFAULT = "http://203.0.113.11:8081"
GW_CREATIVE_URL = os.getenv("GW_CREATIVE_URL", "http://203.0.113.11:8093")
# Also defined in llm.py; intentionally not imported to avoid a doorman_server → llm dep.
# GW_WAKE_DEADLINE_SEC coupling: this deadline (default 180s) must be kept in sync
# with the client-side acquire timeout in agents_core.doorman_client._gw_acquire_timeout(),
# which derives the HTTP acquire timeout as GW_WAKE_DEADLINE_SEC + GW_ACQUIRE_MARGIN_SEC.
# The client timeout must be >= this deadline so successful cold wakes (which can take
# up to GW_WAKE_DEADLINE_SEC) are never misread as DoormanUnreachable timeouts.
GW_WAKE_DEADLINE_SEC = int(os.environ.get("GW_WAKE_DEADLINE_SEC", "180"))
GW_HOLD_TTL_SEC = int(os.environ.get("GW_HOLD_TTL_SEC", "120"))
GW_HOLD_REFRESH_SEC = int(os.environ.get("GW_HOLD_REFRESH_SEC", "45"))
# Machine-economics boundary: amortizes the ~25s cold-load against burst gaps.
# Calibrate from /var/log/doorman-idle.jsonl observations — never auto-tuned.
GW_STOP_GRACE_SEC = int(os.environ.get("GW_STOP_GRACE_SEC", "600"))

DOORMAN_DEFER_TO_CONTROLLER = os.environ.get("DOORMAN_DEFER_TO_CONTROLLER", "true").lower() == "true"
DOORMAN_CONTROLLER_NAME = os.environ.get("DOORMAN_CONTROLLER_NAME", "flip-controller")

# Mode-aware admission guard — dark / default-OFF. When True:
#   ensure_serving checks deference BEFORE _is_serving() (HOLE 1 fix);
#   status_snapshot.serving_mode is controller-lease-aware (HOLE 2 fix);
#   _refresh_serving_cache probes /v1/models for the three-state serving_is_big predicate.
DOORMAN_MODE_AWARE_ADMISSION = os.environ.get(
    "DOORMAN_MODE_AWARE_ADMISSION", ""
).lower() in ("1", "true", "yes")

# Must match OPERATOR_DEFAULTS['gravitywell'] in agents_core.llm (verified: llm.py:58).
GW_BIG_MODEL_ID = "gravitywell-122b"

HOLD_NAME = "doorman"
DOORMAN_IDLE_LOG = os.environ.get("DOORMAN_IDLE_LOG", "/var/log/doorman-idle.jsonl")

# Sentinel for deferred acquire (controller owns the mode)
DEFERRED = object()

# Sentinel for contended acquire (require_drain_clear=True failed: another-principal worker active)
CONTENDED = object()

# Sentinel for creative-occupied acquire (Llama-3.3-70B on :8093 holds the GPU)
CREATIVE_OCCUPIED = object()

# Sentinel principal for worker leases acquired without an explicit principal.
# Never excluded from drain_count — makes a forgotten-principal diagnosable instead of invisible.
GHOST_PRINCIPAL = "__GHOST_LEASE__"


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _write_idle_log(
    node: str, event: str, lease_count: int, idle_secs: float | None = None, **extra_fields
) -> None:
    """Append one structured entry to the idle-lifecycle observation log.

    Best-effort: a write failure must never crash the caller or block the stop.
    This is an observation substrate for human calibration — not operational
    alerting and not consumed internally for auto-tuning.
    """
    entry: dict[str, Any] = {
        "ts": time.time(),
        "node": node,
        "event": event,
        "lease_count": lease_count,
    }
    if idle_secs is not None:
        entry["idle_secs"] = round(idle_secs, 2)
    entry.update(extra_fields)
    try:
        with open(DOORMAN_IDLE_LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning(f"idle log write failed ({DOORMAN_IDLE_LOG}): {exc}")


# ---------------------------------------------------------------------------
# Node state (per-node; Unit 1 only handles "gravitywell")
# ---------------------------------------------------------------------------

class _NodeState:
    """All mutable state for one node, guarded by a single threading.Lock.

    The lock serializes:
    - every lease-registry mutation (acquire, release, GC)
    - ensure_serving calls (prevents parallel wake-gravitywell subprocesses)
    - background refresh-thread reads and SSH hold re-issues
    """

    def __init__(self, gw_url: str, node_name: str = "gravitywell"):
        self.lock = threading.Lock()
        self.gw_url = gw_url
        self.node_name = node_name
        # keyed by work_id → {acquired_at: float, ttl_sec: int, reason: str, role: str}
        self.leases: dict[str, dict] = {}
        self.last_wake_at: float | None = None
        self.last_error: str | None = None
        # Service-lifecycle fields (gravitywell-doorman-clean-stop-v0)
        # Seeded at construction (doorman-seed-idle-since-on-startup-v0): leases
        # starts empty, so idle-tracking must begin now, not only on a later
        # empty-transition that may never occur if the process starts at zero leases.
        self.idle_since: float | None = time.time()
        self.service_stopped: bool = False     # True after gw-serve stop confirmed
        # Cached serving state (doorman-status-cached-serving-v0)
        self._cached_serving: bool | None = None   # None until first refresh
        self._serving_checked_at: float = 0.0      # walltime of last successful probe
        self._cached_creative_serving: bool = False
        # Mode-aware big predicate (populated only when DOORMAN_MODE_AWARE_ADMISSION is True)
        self._serving_is_big: bool | None = None   # None until first refresh with flag ON
        self._big_probe_state: str | None = None   # 'confirmed'|'refuted'|'inconclusive'

    # ------------------------------------------------------------------
    # Health poll (lock-free — read-only HTTP, safe to call outside lock)
    # ------------------------------------------------------------------

    def _is_serving(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(f"{self.gw_url}/health", timeout=timeout)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception:
            return False

    def _is_creative_serving(self) -> bool:
        """Return True if the Llama-3.3-70B creative server is up on :8093.

        Lock-free HTTP - safe to call outside lock; also called under lock in
        ensure_serving(). Returns False on any error - if :8093 is unreachable,
        the 70B is not actively serving.
        """
        try:
            r = requests.get(f"{GW_CREATIVE_URL}/health", timeout=2.5)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def _probe_big_model(self) -> tuple[bool | None, str]:
        """Probe /v1/models to determine whether the big model is resident.

        Returns (big_probe_raw, big_probe_state):
          confirmed    — GW_BIG_MODEL_ID in model list
          refuted      — a different model id is served (fail-closed: real split-brain)
          inconclusive — timeout / network error (caller degrades to legacy judgment)

        Must be called OUTSIDE self.lock (blocking HTTP, ~2.5s timeout).
        """
        try:
            resp = requests.get(f"{self.gw_url}/v1/models", timeout=2.5)
            if resp.status_code == 200:
                model_ids = [m.get("id", "") for m in resp.json().get("data", [])]
                if GW_BIG_MODEL_ID in model_ids:
                    return True, "confirmed"
                if model_ids:
                    # non-empty list without our model — a competing model is resident
                    return False, "refuted"
                # empty list — registry not yet populated during startup
                return None, "inconclusive"
            return None, "inconclusive"
        except Exception as exc:
            log.debug(f"[{self.node_name}] big-model probe inconclusive: {exc}")
            return None, "inconclusive"

    def _refresh_serving_cache(self) -> None:
        """Refresh the serving cache by probing _is_serving outside the lock.

        This method MUST be called when the lock is NOT held, as it performs
        a blocking network call. It then takes the lock briefly to update the
        cached fields.

        WARNING: This method is non-reentrant — it MUST NOT be called from
        within an already-held self.lock context or it will deadlock
        (threading.Lock is non-reentrant).

        When DOORMAN_MODE_AWARE_ADMISSION is True, also probes /v1/models for
        the three-state serving_is_big predicate (outside the lock, AC11).
        """
        serving = self._is_serving(timeout=2.0)
        creative_serving = self._is_creative_serving()

        # Optional big-model probe — outside the lock (blocking HTTP, AC11)
        big_probe_state: str | None = None
        if DOORMAN_MODE_AWARE_ADMISSION:
            _, big_probe_state = self._probe_big_model()

        with self.lock:
            self._cached_serving = serving
            self._cached_creative_serving = creative_serving
            self._serving_checked_at = time.time()
            if DOORMAN_MODE_AWARE_ADMISSION:
                self._big_probe_state = big_probe_state
                controller_owns = self._controller_lease_active()
                if big_probe_state == "refuted":
                    self._serving_is_big = False
                elif big_probe_state == "confirmed":
                    # Controller win takes precedence over probe confirmation (AC6-D)
                    self._serving_is_big = bool(serving and not controller_owns)
                else:  # inconclusive — fall back to legacy controller-lease judgment
                    self._serving_is_big = bool(serving and not controller_owns)
                    log.warning(
                        f"[{self.node_name}] big-model probe inconclusive — "
                        f"falling back to legacy controller-lease judgment "
                        f"(serving_is_big={self._serving_is_big}); probe_inconclusive"
                    )

    def _controller_lease_active(self) -> bool:
        """Check if a mode-controller lease is currently active (non-expired).

        Must be called under self.lock. Returns True iff some non-expired lease
        has role == "mode-controller".
        """
        now = time.time()
        for lease_info in self.leases.values():
            if (lease_info.get("role") == "mode-controller"
                and now <= lease_info["acquired_at"] + lease_info["ttl_sec"]):
                return True
        return False

    # ------------------------------------------------------------------
    # ensure_serving — must be called under lock
    # ------------------------------------------------------------------

    def ensure_serving(self, role: str | None = None) -> bool | object:
        """Wake GW if needed, start the serving unit, and wait until it serves.

        Returns True on success, DEFERRED if controller owns the mode, False on failure.
        Called under self.lock — serializes concurrent wake attempts so only one
        wake-gravitywell subprocess runs at a time.

        Args:
          role: optional role of the caller (e.g., "mode-controller" for flip-controller).
                If role=="mode-controller", this is the controller's own acquire and
                short-circuits to DEFERRED without needing a pre-registered lease.

        Flow (gravitywell-doorman-clean-stop-v0 + doorman-mode-deference-v0):
          0. Mode-aware deference (HOLE 1 fix, flag ON only): if controller owns the
             mode, return DEFERRED immediately — before _is_serving() or wake-gravitywell.
             This prevents wrong-model leases when a controller-owned swarm is up.
             Mode-controller's own acquire skips this check and always proceeds.
          1. Fast-path: _is_serving() → return True (service already up).
          2. wake-gravitywell: idempotent host-wake (no-op if already up).
          3. Check deference: if DOORMAN_DEFER_TO_CONTROLLER and (role=="mode-controller"
             or an active mode-controller lease exists), return DEFERRED (no gw-serve big).
          4. gw-serve big: start llama-server.service if stopped (idempotent).
          5. Poll /health until serving or GW_WAKE_DEADLINE_SEC (covers ~25s
             cold-load after gw-serve big).
        """
        # Block co-load if creative 70B holds the GPU lane
        if self._is_creative_serving():
            return CREATIVE_OCCUPIED

        # HOLE 1 fix (AC2): mode-aware deference before _is_serving() fast path.
        # Worker acquires return DEFERRED immediately when the controller owns the mode,
        # even when _is_serving() would return True (avoids wrong-model leases on a live swarm).
        # Mode-controller's own acquire (role='mode-controller') skips this and always proceeds.
        if DOORMAN_MODE_AWARE_ADMISSION and DOORMAN_DEFER_TO_CONTROLLER:
            if role != "mode-controller" and self._controller_lease_active():
                log.info(
                    f"[{self.node_name}] mode-aware: controller owns mode — "
                    f"deferring before is_serving check (role={role!r})"
                )
                return DEFERRED

        # Fast path: already awake and serving
        if self._is_serving():
            self.last_error = None
            self.service_stopped = False
            return True

        log.info(f"[{self.node_name}] GW not serving — running wake-gravitywell")
        try:
            proc = subprocess.run(
                ["wake-gravitywell", "doorman-acquire"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                err = f"wake-gravitywell failed rc={proc.returncode}: {proc.stderr[:300]}"
                log.error(f"[{self.node_name}] {err}")
                self.last_error = err
                return False
        except Exception as e:
            err = f"wake-gravitywell subprocess error: {e}"
            log.error(f"[{self.node_name}] {err}")
            self.last_error = err
            return False

        # Deference guard: if controller owns the mode, don't issue gw-serve big
        if DOORMAN_DEFER_TO_CONTROLLER:
            if role == "mode-controller" or self._controller_lease_active():
                log.info(
                    f"[{self.node_name}] GW not serving but controller owns mode — "
                    f"deferring (no gw-serve big)"
                )
                return DEFERRED

        # Ensure the serving unit is up (idempotent — fast no-op if already active)
        log.info(f"[{self.node_name}] running gw-serve big to ensure llama-server.service is up")
        try:
            proc = subprocess.run(
                ["ssh", "gravitywell", "gw-serve big"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                err = (
                    f"gw-serve big failed rc={proc.returncode}: {proc.stderr[:300]}"
                )
                log.error(f"[{self.node_name}] {err}")
                self.last_error = err
                return False
        except Exception as e:
            err = f"gw-serve big subprocess error: {e}"
            log.error(f"[{self.node_name}] {err}")
            self.last_error = err
            return False

        # Poll /health until serving or deadline (covers ~25s cold-load)
        deadline = time.time() + GW_WAKE_DEADLINE_SEC
        poll_interval = 3.0
        while time.time() < deadline:
            if self._is_serving():
                elapsed = GW_WAKE_DEADLINE_SEC - (deadline - time.time())
                log.info(f"[{self.node_name}] GW serving after ~{elapsed:.0f}s")
                self.last_wake_at = time.time()
                self.last_error = None
                self.service_stopped = False
                self._cached_serving = True
                self._serving_checked_at = time.time()
                self._place_hold()
                return True
            time.sleep(poll_interval)

        err = f"GW did not serve within {GW_WAKE_DEADLINE_SEC}s after wake"
        log.error(f"[{self.node_name}] {err}")
        self.last_error = err
        return False

    # ------------------------------------------------------------------
    # Keepawake hold helpers — must be called under lock
    # ------------------------------------------------------------------

    def _place_hold(self) -> None:
        try:
            subprocess.run(
                ["ssh", "gravitywell",
                 f"gw-keepawake hold {HOLD_NAME} {GW_HOLD_TTL_SEC} doorman-active"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            log.warning(f"[{self.node_name}] gw-keepawake hold failed: {e}")

    def _release_hold(self) -> None:
        try:
            subprocess.run(
                ["ssh", "gravitywell", f"gw-keepawake release {HOLD_NAME}"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            log.warning(f"[{self.node_name}] gw-keepawake release failed: {e}")

    # ------------------------------------------------------------------
    # Lease operations — must be called under lock
    # ------------------------------------------------------------------

    def _gc_stale(self) -> list[str]:
        """Remove expired leases. Returns list of GC'd work_ids.

        Emits an orphan-reclaim scar event if a mode-controller lease is evicted.
        If GC empties the lease set, records idle_since the same way release_lease()
        does on an explicit release -- otherwise the dwell-stop timer never starts
        for leases that expire via TTL rather than an explicit /lease/release call.
        """
        now = time.time()
        expired = [
            wid for wid, info in self.leases.items()
            if now > info["acquired_at"] + info["ttl_sec"]
        ]
        for wid in expired:
            info = self.leases[wid]
            log.info(f"[{self.node_name}] GC stale lease work_id={wid}")
            # Emit scar if a mode-controller lease is being evicted
            if info.get("role") == "mode-controller":
                log.warning(
                    f"[{self.node_name}] mode-controller lease TTL-expired (not released); "
                    f"deference lapsed; legacy wake/serve will auto-recover"
                )
                acquired_at = info.get("acquired_at")
                last_renewed_iso = (
                    datetime.datetime.utcfromtimestamp(acquired_at).isoformat() + "Z"
                    if acquired_at else None
                )
                _write_idle_log(
                    self.node_name,
                    "controller-orphan-reclaim",
                    len(self.leases) - 1,  # count before deletion
                    evicted_lease=wid,
                    evicted_role="mode-controller",
                    last_renewed=last_renewed_iso,
                    ttl_sec=info.get("ttl_sec"),
                    detail="mode-controller lease TTL-expired (not released); deference lapsed; legacy wake/serve",
                )
            del self.leases[wid]
        if expired and not self.leases and self.idle_since is None:
            self.idle_since = time.time()
            _write_idle_log(self.node_name, "idle_start", 0)
        return expired

    def acquire_lease(self, work_id: str, ttl_sec: int, reason: str, role: str = "worker", principal: str | None = None, require_drain_clear: bool = False, lease_kind: str = "inference") -> bool | object:
        """Try to ensure GW is serving, then register the lease.

        Returns True on success, DEFERRED if a foreign caller acquires during controller
        ownership (no lease registered), CONTENDED if require_drain_clear=True and another
        principal's worker lease is active (lease not registered), False on failure.

        Args:
          role: optional role descriptor (default "worker"). E.g., "mode-controller"
                for the flip-controller's keepawake lease. Stored on the lease dict
                for later ownership checks.
          principal: logical admission group for drain-gate exclusion. Worker leases
                     without a principal are stamped GHOST_PRINCIPAL — always counted,
                     never excluded, emits critical log when counted in a drain decision.
                     Non-worker leases are drain-gate-exempt; principal is ignored.
          require_drain_clear: when True and role=="worker", atomically checks for
                               other-principal worker leases before registering this one.
                               Returns CONTENDED without registering if any exist.
                               Defaults False — all existing callers are unchanged.
          lease_kind: discriminator for admission-contention counting. "inference" (default)
                      — the lease holds GPU inference and serializes via the drain-gate.
                      "coordination" — the lease holds no inference (span/keepawake); excluded
                      from drain-gate contention count but still counted by /v0/drain-count
                      for flip-protection. Omitting is byte-identical to "inference".
        """
        # Clear idle tracking: an arriving lease means the node is no longer idle
        was_idle = self.idle_since is not None
        self.idle_since = None
        if was_idle:
            _write_idle_log(self.node_name, "resumed", len(self.leases))

        # ensure_serving serializes concurrent wakes under the same lock
        ok = self.ensure_serving(role=role)
        if ok is CREATIVE_OCCUPIED:
            return CREATIVE_OCCUPIED
        if ok is DEFERRED:
            # Controller's own acquire (role="mode-controller") registers the lease and hold
            # even though ensure_serving returns DEFERRED (no gw-serve big was issued).
            # Foreign acquires during controller ownership don't register a lease.
            if role == "mode-controller":
                self.leases[work_id] = {
                    "acquired_at": time.time(),
                    "ttl_sec": ttl_sec,
                    "reason": reason,
                    "role": role,
                }
                self._place_hold()
                return DEFERRED  # still return DEFERRED so endpoint knows not to issue gw-serve big
            else:
                # Foreign caller during controller ownership — return deferred, no lease
                return DEFERRED
        if not ok:
            return False

        # Atomic drain-gate check (AC3): count cross-group worker leases and register
        # the new lease in one critical section — check-and-register atomic; closes the
        # drain-gate TOCTOU where separate drain_count + acquire calls let multiple
        # distinct-principal workers all observe drain=0 before any registers.
        if require_drain_clear and role == "worker":
            effective_principal = principal if principal is not None else GHOST_PRINCIPAL
            for _wid, _info in self.leases.items():
                if _info.get("role") != "worker":
                    continue
                # Coordination leases hold no inference — not a drain-gate contender (AC2).
                # /v0/drain-count still counts them for flip-protection (unchanged, AC4).
                if _info.get("lease_kind", "inference") == "coordination":
                    continue
                _p = _info.get("principal", GHOST_PRINCIPAL)
                if _p == GHOST_PRINCIPAL:
                    # Ghost leases always count as contending; never silently excluded (AC7).
                    log.critical(
                        "[doorman] drain_count ghost_lease_counted work_id=%s - "
                        "role=worker lease has no principal; add principal= to "
                        "acquire() call to prevent drain-gate freeze",
                        _wid,
                    )
                    return CONTENDED
                if _p != effective_principal:
                    return CONTENDED

        lease_entry: dict = {
            "acquired_at": time.time(),
            "ttl_sec": ttl_sec,
            "reason": reason,
            "role": role,
            "lease_kind": lease_kind,
        }
        if role == "worker":
            lease_entry["principal"] = principal if principal is not None else GHOST_PRINCIPAL
        self.leases[work_id] = lease_entry
        self._place_hold()
        return True

    def release_lease(self, work_id: str) -> None:
        """Drop a lease. If it was the last, record idle_since and release the hold."""
        self.leases.pop(work_id, None)
        self._gc_stale()
        if not self.leases:
            self.idle_since = time.time()
            self._release_hold()
            _write_idle_log(self.node_name, "idle_start", 0)

    # ------------------------------------------------------------------
    # Status snapshot (for /status endpoint)
    # ------------------------------------------------------------------

    def status_snapshot(self) -> dict:
        with self.lock:
            self._gc_stale()
            serving = self._cached_serving
            # Check if controller owns the mode
            controller_owns = self._controller_lease_active()
            # HOLE 2 fix (AC5): when flag ON, controller-lease check precedes serving-wins.
            # When flag OFF, preserve today's order (serving wins) for byte-identical behavior.
            if DOORMAN_MODE_AWARE_ADMISSION:
                if controller_owns:
                    serving_mode = "deferred"
                elif serving:
                    serving_mode = "big"
                elif self.service_stopped:
                    serving_mode = "stopped"
                else:
                    serving_mode = "unknown"
            else:
                if serving:
                    serving_mode = "big"
                elif controller_owns:
                    serving_mode = "deferred"
                elif self.service_stopped:
                    serving_mode = "stopped"
                else:
                    serving_mode = "unknown"
            # drain_count: worker leases only (mode-controller excluded), AC8
            drain_count = sum(
                1 for info in self.leases.values() if info.get("role") == "worker"
            )
            return {
                "serving": serving,
                "serving_mode": serving_mode,
                "serving_checked_at": self._serving_checked_at,
                "snapshot_mode": "cached",
                "service_stopped": self.service_stopped,
                "idle_since": self.idle_since,
                "lease_count": len(self.leases),
                "leases": [
                    {"work_id": wid, **info}
                    for wid, info in self.leases.items()
                ],
                "last_wake_at": self.last_wake_at,
                "last_error": self.last_error,
                "mode_owner": DOORMAN_CONTROLLER_NAME if controller_owns else None,
                "drain_count": drain_count,
                "worker_lease_count": drain_count,
                "serving_is_big": self._serving_is_big,
                "big_probe_state": self._big_probe_state,
                "creative_serving": self._cached_creative_serving,
            }


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------

def _start_refresh_thread(nodes: dict[str, _NodeState]) -> threading.Thread:
    """Start the keepawake refresh + stale-lease GC + deferred stop background thread."""

    def _loop():
        backoff = 0.0
        first_iteration = True
        while True:
            if not first_iteration:
                time.sleep(max(GW_HOLD_REFRESH_SEC - backoff, GW_HOLD_REFRESH_SEC // 2))
            first_iteration = False
            backoff = 0.0
            for node_name, state in nodes.items():
                # Refresh serving cache OUTSIDE the lock (probing is a blocking network call)
                state._refresh_serving_cache()
                with state.lock:
                    state._gc_stale()
                    if not state.leases:
                        # No active leases: check if deferred service stop is due
                        if (
                            state.idle_since is not None
                            and not state.service_stopped
                        ):
                            idle_elapsed = time.time() - state.idle_since
                            if idle_elapsed >= GW_STOP_GRACE_SEC:
                                log.warning(
                                    f"[{node_name}] idle {idle_elapsed:.0f}s >= grace "
                                    f"{GW_STOP_GRACE_SEC}s — issuing gw-serve stop. "
                                    f"Safety: guard blocks suspend while service active; "
                                    f"doorman stop enables suspend, never forces it."
                                )
                                try:
                                    stop_proc = subprocess.run(
                                        ["ssh", "gravitywell", "gw-serve stop"],
                                        capture_output=True, text=True, timeout=60,
                                    )
                                    if stop_proc.returncode == 0:
                                        state.service_stopped = True
                                        state.idle_since = None
                                        state._cached_serving = False
                                        state._serving_checked_at = time.time()
                                        log.warning(
                                            f"[{node_name}] gw-serve stop succeeded — "
                                            f"llama-server.service stopped, host now "
                                            f"suspend-eligible via guard"
                                        )
                                        _write_idle_log(
                                            node_name, "stopped", 0,
                                            idle_secs=idle_elapsed,
                                        )
                                    else:
                                        # rc != 0: idempotency guard — check if already down
                                        if not state._is_serving():
                                            # Already stopped — treat as success
                                            state.service_stopped = True
                                            state.idle_since = None
                                            state._cached_serving = False
                                            state._serving_checked_at = time.time()
                                            log.warning(
                                                f"[{node_name}] gw-serve stop "
                                                f"rc={stop_proc.returncode} but service "
                                                f"already down — treating as success"
                                            )
                                            _write_idle_log(
                                                node_name, "stopped", 0,
                                                idle_secs=idle_elapsed,
                                            )
                                        else:
                                            # Real failure: still serving
                                            err = (
                                                f"gw-serve stop failed "
                                                f"rc={stop_proc.returncode}: "
                                                f"{stop_proc.stderr[:200]}"
                                            )
                                            log.error(f"[{node_name}] {err}")
                                            state.last_error = err
                                            backoff = min(
                                                backoff + 15, GW_HOLD_REFRESH_SEC
                                            )
                                            _write_idle_log(
                                                node_name, "stop_failed", 0,
                                                idle_secs=idle_elapsed,
                                            )
                                except Exception as exc:
                                    err = f"gw-serve stop exception: {exc}"
                                    log.error(f"[{node_name}] {err}")
                                    state.last_error = err
                                    backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)
                                    _write_idle_log(node_name, "stop_failed", 0)
                        continue  # no hold refresh needed for idle node

                    # Leases are active: re-issue the keepawake hold to refresh its TTL
                    try:
                        proc = subprocess.run(
                            ["ssh", "gravitywell",
                             f"gw-keepawake hold {HOLD_NAME} {GW_HOLD_TTL_SEC} doorman-refresh"],
                            capture_output=True, text=True, timeout=15,
                        )
                        if proc.returncode != 0:
                            err = (f"keepawake refresh failed rc={proc.returncode}: "
                                   f"{proc.stderr[:200]}")
                            log.error(f"[{node_name}] {err}")
                            state.last_error = err
                            backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)
                        else:
                            log.debug(f"[{node_name}] keepawake hold refreshed")
                    except Exception as e:
                        err = f"keepawake refresh exception: {e}"
                        log.error(f"[{node_name}] {err}")
                        state.last_error = err
                        backoff = min(backoff + 15, GW_HOLD_REFRESH_SEC)

    t = threading.Thread(target=_loop, daemon=True, name="doorman-refresh")
    t.start()
    return t


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(gw_url: str | None = None) -> FastAPI:
    _gw_url = gw_url or os.environ.get("GW_URL", GW_URL_DEFAULT)

    nodes: dict[str, _NodeState] = {
        "gravitywell": _NodeState(_gw_url, node_name="gravitywell"),
    }

    # Start background refresh thread
    _start_refresh_thread(nodes)

    app = FastAPI(title="doorman-server", version="0")

    # ------------------------------------------------------------------
    # Bearer-token auth middleware
    # ------------------------------------------------------------------
    _token = os.environ.get("DOORMAN_BEARER_TOKEN", "")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if _token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[len("Bearer "):] != _token:
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
        return await call_next(request)

    # ------------------------------------------------------------------
    # /healthz — doorman's own liveness
    # ------------------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------------
    # /status — node state snapshot (future Unit 2 UI read surface)
    # ------------------------------------------------------------------

    @app.get("/status")
    def status():
        return {
            "nodes": {
                name: state.status_snapshot()
                for name, state in nodes.items()
            }
        }

    # ------------------------------------------------------------------
    # GET /v0/mode-owner — deference-liveness probe (doorman-mode-deference-v0)
    # ------------------------------------------------------------------

    @app.get("/v0/mode-owner")
    def mode_owner(node: str = "gravitywell"):
        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state._gc_stale()
            owner_lease_held = state._controller_lease_active()
            # Compute owner_lease_age_sec and stale flag
            owner_lease_age_sec = None
            owner_lease_stale = False
            if owner_lease_held:
                now = time.time()
                for lease_info in state.leases.values():
                    if lease_info.get("role") == "mode-controller":
                        age_sec = now - lease_info["acquired_at"]
                        owner_lease_age_sec = age_sec
                        # Past renewal point (60% of TTL)
                        owner_lease_stale = age_sec > lease_info["ttl_sec"] * 0.6
                        break

        return {
            "node": node,
            "controller": DOORMAN_CONTROLLER_NAME,
            "active": DOORMAN_DEFER_TO_CONTROLLER,
            "owner_lease_held": owner_lease_held,
            "owner_lease_age_sec": owner_lease_age_sec,
            "owner_lease_stale": owner_lease_stale,
        }

    # ------------------------------------------------------------------
    # GET /v0/drain-count — in-flight worker-lease count for drain-gate (AC9)
    # ------------------------------------------------------------------

    @app.get("/v0/drain-count")
    def drain_count_endpoint(node: str = "gravitywell", exclude_principal: str | None = None):
        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        count = 0
        with state.lock:
            state._gc_stale()
            for wid, info in state.leases.items():
                if info.get("role") != "worker":
                    continue
                p = info.get("principal", GHOST_PRINCIPAL)
                if p == GHOST_PRINCIPAL:
                    # Ghost leases are always counted; emit critical log when counted in a drain decision
                    if exclude_principal is not None:
                        log.critical(
                            "[doorman] drain_count ghost_lease_counted work_id=%s - "
                            "role=worker lease has no principal; add principal= to "
                            "acquire() call to prevent drain-gate freeze",
                            wid,
                        )
                    count += 1
                elif exclude_principal is not None and p == exclude_principal:
                    continue  # same admission group — exclude from drain count
                else:
                    count += 1
        return {"node": node, "drain_count": count}

    # ------------------------------------------------------------------
    # POST /lease/acquire
    # ------------------------------------------------------------------

    @app.post("/lease/acquire")
    def lease_acquire(body: dict[str, Any]):
        node = body.get("node", "")
        work_id = body.get("work_id", "")
        ttl_sec = int(body.get("ttl_sec", 300))
        reason = body.get("reason", "")
        role = body.get("role", "worker")
        principal = body.get("principal") or None  # empty string → None → ghost
        require_drain_clear = bool(body.get("require_drain_clear", False))
        lease_kind = body.get("lease_kind", "inference")

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )
        if not work_id:
            return JSONResponse(
                status_code=400,
                content=_error("bad_request", "work_id is required"),
            )

        state = nodes[node]
        with state.lock:
            ok = state.acquire_lease(
                work_id, ttl_sec, reason, role=role, principal=principal,
                require_drain_clear=require_drain_clear, lease_kind=lease_kind,
            )

        if ok is CREATIVE_OCCUPIED:
            return JSONResponse(
                {"ok": False, "creative_occupied": True,
                 "reason": "creative-collider-holding-gpu"},
                status_code=409,
            )
        if ok is CONTENDED:
            return {"ok": False, "contended": True, "node": node}
        if ok is DEFERRED:
            return {
                "status": "deferred",
                "node": node,
                "mode_owner": DOORMAN_CONTROLLER_NAME,
                "detail": "GW serving controller-owned non-big mode; big endpoint unavailable",
            }
        if not ok:
            return {"status": "wake_failed", "detail": state.last_error or "wake failed"}

        resp: dict = {"status": "serving", "node": node, "work_id": work_id}
        if require_drain_clear:
            resp["drain_cleared"] = True  # signals to client that drain check was honored (AC5a)
        return resp

    # ------------------------------------------------------------------
    # POST /lease/release
    # ------------------------------------------------------------------

    @app.post("/lease/release")
    def lease_release(body: dict[str, Any]):
        node = body.get("node", "")
        work_id = body.get("work_id", "")

        if node not in nodes:
            return JSONResponse(
                status_code=400,
                content=_error("bad_node", f"unknown node {node!r}"),
            )

        state = nodes[node]
        with state.lock:
            state.release_lease(work_id)

        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    host = os.environ.get("DOORMAN_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("DOORMAN_BIND_PORT", "8407"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
