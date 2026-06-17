"""Shared deliberation HTTP service — FastAPI wrapper.

Entry point:  shared-deliberation-server  (console_scripts in pyproject.toml)
Port:         8409  (SHARED_DELIBERATION_BIND_PORT env var; 8408 is occupied by flip-controller)
Bind:         127.0.0.1 by default  (SHARED_DELIBERATION_BIND_HOST env var)

Environment variables:
  SHARED_DELIBERATION_BIND_HOST      — uvicorn bind host (default 127.0.0.1)
  SHARED_DELIBERATION_BIND_PORT      — uvicorn bind port (default 8409)
  SHARED_DELIBERATION_BEARER_TOKEN   — optional shared bearer token
  SHARED_DELIBERATION_MAX_CONCURRENT — max concurrent Facets subprocesses (default 2)
  SHARED_DELIBERATION_COUNCIL_TIMEOUT_S — council poll timeout in seconds (default 1800)
  SHARED_DELIBERATION_FACETS_STUB    — set to 1 to stub Facets (testing)
  SHARED_DELIBERATION_COUNCIL_STUB   — set to 1 to stub Council (testing)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from agents_core.shared_deliberation.envelope import DeliberationRequest, DeliberationEnvelope
from agents_core.shared_deliberation.orchestrator import run_deliberation, init_facets_semaphore

log = logging.getLogger("shared-deliberation-server")

BIND_HOST = os.environ.get("SHARED_DELIBERATION_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("SHARED_DELIBERATION_BIND_PORT", "8409"))
BEARER_TOKEN = os.environ.get("SHARED_DELIBERATION_BEARER_TOKEN")
MAX_CONCURRENT = int(os.environ.get("SHARED_DELIBERATION_MAX_CONCURRENT", "2"))


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def _bearer_auth(headers: dict) -> None:
    """Verify bearer token if configured."""
    if not BEARER_TOKEN:
        return
    auth_header = headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header[7:]
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token")


def create_app() -> FastAPI:
    app = FastAPI(title="shared-deliberation-server", version="0")

    @app.on_event("startup")
    async def startup():
        init_facets_semaphore(MAX_CONCURRENT)
        log.info(f"Shared deliberation server starting on {BIND_HOST}:{BIND_PORT}")
        log.info(f"Max concurrent Facets: {MAX_CONCURRENT}")

    @app.get("/v0/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v0/deliberate")
    async def deliberate(request_data: dict, request: Request) -> dict:
        """Submit a deliberation request. Returns DeliberationEnvelope."""
        try:
            await _bearer_auth(dict(request.headers))
        except HTTPException:
            raise

        try:
            req = DeliberationRequest(**request_data)
        except TypeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        try:
            envelope = await run_deliberation(req)
            return envelope.to_dict()
        except Exception as e:
            log.exception("Deliberation error")
            # Partial failure still returns 200; only return 5xx on total orchestration failure
            raise HTTPException(status_code=500, detail=f"Orchestration error: {e}")

    return app


def main():
    app = create_app()
    uvicorn.run(
        app,
        host=BIND_HOST,
        port=BIND_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
