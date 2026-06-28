"""Dowser HTTP service — FastAPI wrapper around the dowser funnel.

Entry point:  dowser-server  (console_scripts in pyproject.toml)
Port:         8412  (DOWSER_BIND_PORT env var)
Bind:         127.0.0.1 by default

Environment variables (server side):
  DOWSER_BIND_HOST  — uvicorn bind host (default 127.0.0.1)
  DOWSER_BIND_PORT  — uvicorn bind port (default 8412)
  DOWSER_LOG_LEVEL  — uvicorn log level (default info)
  SEARXNG_URL       — SearXNG endpoint (default http://203.0.113.10:8888)
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from agents_core.dowser import read_batch, critique_batch


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def create_app() -> FastAPI:
    app = FastAPI(title="dowser-server", version="0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "port": int(os.environ.get("DOWSER_BIND_PORT", "8412"))}

    @app.post("/research/read-batch")
    def research_read_batch(body: dict[str, Any]):
        requests_list = body.get("requests")
        if not isinstance(requests_list, list):
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "'requests' must be a list"),
            )
        read_operator = body.get("read_operator", "quest")
        if read_operator not in ("quest", "sonnet", "haiku", "opus"):
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", f"Unknown read_operator {read_operator!r}"),
            )
        budget = body.get("budget") or {}
        try:
            result = read_batch(
                requests_list=requests_list,
                read_operator=read_operator,
                budget=budget,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=_error("internal", str(e)))
        return JSONResponse(content=result)

    @app.post("/research/critique-batch")
    def research_critique_batch(body: dict[str, Any]):
        drafts = body.get("drafts")
        if not isinstance(drafts, list):
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "'drafts' must be a list"),
            )
        critic_operator = body.get("critic_operator", "gravitywell")
        if critic_operator not in ("gravitywell",):
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", f"Unknown critic_operator {critic_operator!r}"),
            )
        try:
            result = critique_batch(drafts=drafts, critic_operator=critic_operator)
        except Exception as e:
            raise HTTPException(status_code=500, detail=_error("internal", str(e)))
        return JSONResponse(content=result)

    return app


def main():
    import uvicorn

    host = os.environ.get("DOWSER_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("DOWSER_BIND_PORT", "8412"))
    log_level = os.environ.get("DOWSER_LOG_LEVEL", "info")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
