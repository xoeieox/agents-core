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
  GW_URL                 — GravityWell base URL (default http://203.0.113.11:8081)
                           NOTE: must match the GW_URL configured for agents_core.llm
                           (the operator reads the same env var for inference POSTs).
  GW_WAKE_DEADLINE_SEC   — max seconds to wait for GW to serve (default 180;
                           cold 77GB model load backstop — typical warm wake is ~10s)
  GW_HOLD_TTL_SEC        — keepawake hold TTL in seconds (default 120)
  GW_HOLD_REFRESH_SEC    — refresh interval for the keepawake hold (default 45)

Safety properties:
  - Doorman crash → GW frees itself: hold TTL lapses within ~2min, GW idle-suspends.
  - Leaked client lease → GC: stale leases (acquired_at + ttl_sec < now) are auto-
    released by the background refresh loop — a crashed operator cannot pin GW forever.
  - SSH-refresh failure → logged loudly, last_error set, retried on next tick; does NOT
    crash the thread and does NOT drop live leases.
"""

from __future__ import annotations

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
GW_WAKE_DEADLINE_SEC = int(os.environ.get("GW_WAKE_DEADLINE_SEC", "180"))
GW_HOLD_TTL_SEC = int(os.environ.get("GW_HOLD_TTL_SEC", "120"))
GW_HOLD_REFRESH_SEC = int(os.environ.get("GW_HOLD_REFRESH_SEC", "45"))

HOLD_NAME = "doorman"


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


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

    def __init__(self, gw_url: str):
        self.lock = threading.Lock()
        self.gw_url = gw_url
        # keyed by work_id → {acquired_at: float, ttl_sec: int, reason: str}
        self.leases: dict[str, dict] = {}
        self.last_wake_at: float | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # Health poll (lock-free — read-only HTTP, safe to call outside lock)
    # ------------------------------------------------------------------

    def _is_serving(self, timeout: float = 3.0) -> bool:
        try:
            resp = requests.get(f"{self.gw_url}/health", timeout=timeout)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # ensure_serving — must be called under lock
    # ------------------------------------------------------------------

    def ensure_serving(self) -> bool:
        """Wake GW if needed and wait until it serves. Returns True on success.

        Called under self.lock — serializes concurrent wake attempts so only
        one wake-gravitywell subprocess runs at a time.
        """
        # Fast path: already awake
        if self._is_serving():
            self.last_error = None
            return True

        log.info("GW not serving — running wake-gravitywell")
        try:
            proc = subprocess.run(
                ["wake-gravitywell", "doorman-acquire"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                err = f"wake-gravitywell failed rc={proc.returncode}: {proc.stderr[:300]}"
                log.error(err)
                self.last_error = err
                return False
        except Exception as e:
            err = f"wake-gravitywell subprocess error: {e}"
            log.error(err)
            self.last_error = err
            return False

        # Poll /health until serving or deadline
        deadline = time.time() + GW_WAKE_DEADLINE_SEC
        poll_interval = 3.0
        while time.time() < deadline:
            if self._is_serving():
                elapsed = GW_WAKE_DEADLINE_SEC - (deadline - time.time())
                log.info(f"GW serving after ~{elapsed:.0f}s")
                self.last_wake_at = time.time()
                self.last_error = None
                self._place_hold()
                return True
            time.sleep(poll_interval)

        err = f"GW did not serve within {GW_WAKE_DEADLINE_SEC}s after wake"
        log.error(err)
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
            log.warning(f"gw-keepawake hold failed: {e}")

    def _release_hold(self) -> None:
        try:
            subprocess.run(
                ["ssh", "gravitywell", f"gw-keepawake release {HOLD_NAME}"],
                capture_output=True, text=True, timeout=15,
            )
        except Exception as e:
            log.warning(f"gw-keepawake release failed: {e}")

    # ------------------------------------------------------------------
    # Lease operations — must be called under lock
    # ------------------------------------------------------------------

    def _gc_stale(self) -> list[str]:
        """Remove expired leases. Returns list of GC'd work_ids."""
        now = time.time()
        expired = [
            wid for wid, info in self.leases.items()
            if now > info["acquired_at"] + info["ttl_sec"]
        ]
        for wid in expired:
            log.info(f"GC stale lease work_id={wid}")
            del self.leases[wid]
        return expired

    def acquire_lease(self, work_id: str, ttl_sec: int, reason: str) -> bool:
        """Try to ensure GW is serving, then register the lease. Returns True on success."""
        # ensure_serving serializes concurrent wakes under the same lock
        ok = self.ensure_serving()
        if not ok:
            return False
        self.leases[work_id] = {
            "acquired_at": time.time(),
            "ttl_sec": ttl_sec,
            "reason": reason,
        }
        self._place_hold()
        return True

    def release_lease(self, work_id: str) -> None:
        """Drop a lease. If it was the last, release the keepawake hold."""
        self.leases.pop(work_id, None)
        self._gc_stale()
        if not self.leases:
            self._release_hold()

    # ------------------------------------------------------------------
    # Status snapshot (for /status endpoint)
    # ------------------------------------------------------------------

    def status_snapshot(self) -> dict:
        with self.lock:
            self._gc_stale()
            return {
                "serving": self._is_serving(timeout=2.0),
                "lease_count": len(self.leases),
                "leases": [
                    {"work_id": wid, **info}
                    for wid, info in self.leases.items()
                ],
                "last_wake_at": self.last_wake_at,
                "last_error": self.last_error,
            }


# ---------------------------------------------------------------------------
# Background refresh thread
# ---------------------------------------------------------------------------

def _start_refresh_thread(nodes: dict[str, _NodeState]) -> threading.Thread:
    """Start the keepawake refresh + stale-lease GC background thread."""

    def _loop():
        backoff = 0.0
        while True:
            time.sleep(max(GW_HOLD_REFRESH_SEC - backoff, GW_HOLD_REFRESH_SEC // 2))
            backoff = 0.0
            for node_name, state in nodes.items():
                with state.lock:
                    state._gc_stale()
                    if not state.leases:
                        continue
                    # Re-issue the keepawake hold to refresh its TTL
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
        "gravitywell": _NodeState(_gw_url),
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
    # POST /lease/acquire
    # ------------------------------------------------------------------

    @app.post("/lease/acquire")
    def lease_acquire(body: dict[str, Any]):
        node = body.get("node", "")
        work_id = body.get("work_id", "")
        ttl_sec = int(body.get("ttl_sec", 300))
        reason = body.get("reason", "")

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
            ok = state.acquire_lease(work_id, ttl_sec, reason)

        if not ok:
            return {"status": "wake_failed", "detail": state.last_error or "wake failed"}

        return {"status": "serving", "node": node, "work_id": work_id}

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
