"""gpu-queue HTTP service — FastAPI wrapper around GPUQueue.

Entry point:  gpu-queue-server  (console_scripts in pyproject.toml)
Port:         8405  (GPU_QUEUE_BIND_PORT env var)
Bind:         127.0.0.1 by default; production sets GPU_QUEUE_BIND_HOST=<tailscale-ip>

Environment variables (server side):
  GPU_QUEUE_DIR         — queue directory (default /srv/lapis/gpu-queue)
  GPU_QUEUE_BIND_HOST   — uvicorn bind host (default 127.0.0.1)
  GPU_QUEUE_BIND_PORT   — uvicorn bind port (default 8405)
  GPU_QUEUE_BEARER_TOKEN — optional shared bearer token; omit to disable auth
  GPU_QUEUE_LOG_LEVEL   — uvicorn log level (default info)
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agents_core.room_paths import room_path

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import JSONResponse

from agents_core.gpu import GPUQueue

PACIFIC = ZoneInfo("America/Los_Angeles")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _active_task_age(queue: GPUQueue) -> float | None:
    """Return seconds since active task's started_at, or None if no active task."""
    task = queue.get_active()
    if task is None:
        return None
    started_at = task.get("started_at")
    if not started_at:
        return None
    try:
        dt = datetime.fromisoformat(started_at)
        now = datetime.now(PACIFIC)
        return max(0.0, (now - dt).total_seconds())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(queue_dir: Path) -> FastAPI:
    app = FastAPI(title="gpu-queue-server", version="0")
    queue = GPUQueue(queue_dir)

    # Process-wide mutation lock (Invariant 3).
    # All mutating operations acquire this lock so claim() never double-issues
    # a task under FastAPI threadpool concurrency.
    lock = threading.Lock()

    # ------------------------------------------------------------------
    # Custom exception handler — return {error: {code, message}} directly
    # rather than FastAPI's default {detail: ...} wrapper.
    # ------------------------------------------------------------------

    @app.exception_handler(FastAPIHTTPException)
    async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    # ------------------------------------------------------------------
    # Bearer-token middleware (only active when GPU_QUEUE_BEARER_TOKEN is set)
    # ------------------------------------------------------------------
    _token = os.environ.get("GPU_QUEUE_BEARER_TOKEN", "")

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
    # Health
    # ------------------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        state = queue.get_state()
        pending_count = len(list(queue.pending_dir.glob("*.yaml")))
        active_count = len(list(queue.active_dir.glob("*.yaml")))
        completed_count = len(list(queue.completed_dir.glob("*.yaml")))
        failed_count = len(list(queue.failed_dir.glob("*.yaml")))
        return {
            "status": "ok",
            "queue_dir": str(queue_dir),
            "queue_depth": pending_count,
            "mode": state.get("mode", "idle"),
            "paused": state.get("paused", False),
            "active_task_age_seconds": _active_task_age(queue),
            "counts": {
                "pending": pending_count,
                "active": active_count,
                "completed": completed_count,
                "failed": failed_count,
            },
        }

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    @app.post("/v0/tasks")
    def submit_task(body: dict[str, Any]):
        if "task_type" not in body:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "task_type is required"),
            )
        with lock:
            task_id = queue.submit(body)
        return {"id": task_id}

    # ------------------------------------------------------------------
    # State readers (no lock — pure reads)
    # ------------------------------------------------------------------

    @app.get("/v0/state")
    def get_state():
        state = queue.get_state()
        state["active_task_age_seconds"] = _active_task_age(queue)
        return state

    @app.get("/v0/pending")
    def get_pending():
        return queue.get_pending()

    @app.get("/v0/active")
    def get_active():
        return queue.get_active()

    @app.get("/v0/completed")
    def get_completed(limit: int = 10):
        return queue.get_recent_completed(limit=limit)

    @app.get("/v0/failed")
    def get_failed(limit: int = 10):
        return queue.get_recent_failed(limit=limit)

    @app.get("/v0/history")
    def get_history(limit: int = 50):
        return queue.get_history(limit=limit)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    @app.post("/v0/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, body: dict[str, Any] | None = Body(default=None)):
        reason = (body or {}).get("reason", "")
        with lock:
            cancelled = queue.cancel(task_id, reason=reason)
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Task '{task_id}' not found in pending"),
            )
        return {"cancelled": True}

    # ------------------------------------------------------------------
    # Claim (consumer / runner surface)
    # ------------------------------------------------------------------

    @app.post("/v0/claim")
    def claim_task(body: dict[str, Any] | None = Body(default=None)):
        current_model = (body or {}).get("current_model")
        with lock:
            task = queue.claim(current_model=current_model)
        if task is None:
            return Response(status_code=204)
        return task

    # ------------------------------------------------------------------
    # Complete (H1 fold: pre-check active_dir/{id}.yaml before calling library)
    # ------------------------------------------------------------------

    @app.post("/v0/tasks/{task_id}/complete")
    def complete_task(task_id: str, body: dict[str, Any] | None = Body(default=None)):
        active_path = queue.active_dir / f"{task_id}.yaml"
        if not active_path.exists():
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Task '{task_id}' not found in active"),
            )
        b = body or {}
        output_path = b.get("output_path")
        result_summary = b.get("result_summary")
        with lock:
            queue.complete(task_id, output_path=output_path, result_summary=result_summary)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Fail (H1 fold: pre-check active_dir/{id}.yaml before calling library)
    # ------------------------------------------------------------------

    @app.post("/v0/tasks/{task_id}/fail")
    def fail_task(task_id: str, body: dict[str, Any] | None = Body(default=None)):
        active_path = queue.active_dir / f"{task_id}.yaml"
        if not active_path.exists():
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Task '{task_id}' not found in active"),
            )
        error = (body or {}).get("error", "Unknown error")
        with lock:
            queue.fail(task_id, error=error)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Preempt (H1 fold: pre-check active_dir/{id}.yaml before calling library)
    # ------------------------------------------------------------------

    @app.post("/v0/tasks/{task_id}/preempt")
    def preempt_task(task_id: str):
        active_path = queue.active_dir / f"{task_id}.yaml"
        if not active_path.exists():
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Task '{task_id}' not found in active"),
            )
        with lock:
            queue.preempt(task_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Runner state
    # ------------------------------------------------------------------

    @app.post("/v0/runner-state")
    def update_runner_state(body: dict[str, Any] | None = Body(default=None)):
        with lock:
            queue.update_runner_state(**(body or {}))
        return {"ok": True}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @app.post("/v0/cleanup")
    def cleanup(max_age_hours: float = 48):
        with lock:
            removed = queue.cleanup(max_age_hours=max_age_hours)
        return {"removed": removed}

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    @app.post("/v0/pause")
    def pause():
        with lock:
            queue.pause()
        return {"paused": True}

    @app.post("/v0/resume")
    def resume():
        with lock:
            queue.resume()
        return {"paused": False}

    @app.get("/v0/paused")
    def is_paused():
        return {"paused": queue.is_paused()}

    return app


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    queue_dir = room_path("gpu_queue")
    host = os.environ.get("GPU_QUEUE_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("GPU_QUEUE_BIND_PORT", "8405"))
    log_level = os.environ.get("GPU_QUEUE_LOG_LEVEL", "info")

    app = create_app(queue_dir)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
