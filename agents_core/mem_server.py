"""mem HTTP service — FastAPI wrapper around MemoryStore.

Entry point:  mem-server  (console_scripts in pyproject.toml)
Port:         8403  (MEM_BIND_PORT env var)
Bind:         127.0.0.1 by default; production sets MEM_BIND_HOST=<tailscale-ip>

Environment variables (server side):
  MEM_DB_PATH       — SQLite DB file (default /data/memory/mem.db)
  MEM_BIND_HOST     — uvicorn bind host (default 127.0.0.1)
  MEM_BIND_PORT     — uvicorn bind port (default 8403)
  MEM_BEARER_TOKEN  — optional shared bearer token; omit to disable auth
  MEM_LOG_LEVEL     — uvicorn log level (default info)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from agents_core.mem import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _row_response(row: dict) -> dict:
    """Ensure all memory-shaped responses carry timestamps."""
    return {
        "key": row["key"],
        "content": row["content"],
        "tags": row["tags"],
        "source": row["source"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _search_row_response(row: dict) -> dict:
    r = _row_response(row)
    r["rank"] = row.get("rank")
    return r


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="mem-server", version="0")
    store = MemoryStore(db_path)

    # ------------------------------------------------------------------
    # Bearer-token middleware (only active when MEM_BEARER_TOKEN is set)
    # ------------------------------------------------------------------
    _token = os.environ.get("MEM_BEARER_TOKEN", "")

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
        mem_count = store._conn.execute(
            "SELECT COUNT(*) AS n FROM memories"
        ).fetchone()["n"]
        # memories_fts_docsize is the FTS5 shadow table that tracks one row per
        # indexed document. With content='memories', COUNT(*) FROM memories_fts
        # delegates to the backing table and never diverges — docsize is the
        # reliable proxy for the actual FTS index entry count.
        fts_count = store._conn.execute(
            "SELECT COUNT(*) AS n FROM memories_fts_docsize"
        ).fetchone()["n"]
        in_sync = mem_count == fts_count
        return {
            "status": "ok",
            "db_path": str(store.db_path),
            "row_counts": {"memories": mem_count, "memories_fts": fts_count},
            "fts_integrity": {
                "in_sync": in_sync,
                "divergence": mem_count - fts_count,
            },
        }

    # ------------------------------------------------------------------
    # List memories
    # ------------------------------------------------------------------

    @app.get("/v0/memories")
    def list_memories(
        tag: str = "",
        since: str = "",
        limit: int = 50,
    ):
        rows = store.list_all(tag=tag, since=since, limit=limit)
        return [_row_response(r) for r in rows]

    # ------------------------------------------------------------------
    # Get single memory
    # ------------------------------------------------------------------

    @app.get("/v0/memories/{key:path}")
    def get_memory(key: str):
        row = store.get(key)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Memory '{key}' not found"),
            )
        return _row_response(row)

    # ------------------------------------------------------------------
    # Upsert memory
    # ------------------------------------------------------------------

    @app.put("/v0/memories/{key:path}")
    def put_memory(key: str, request_data: dict[str, Any]):
        content = request_data.get("content", "")
        tags_raw = request_data.get("tags", "")
        source = request_data.get("source", "")

        # tags arrives as a comma-separated string; split for MemoryStore.set()
        if isinstance(tags_raw, str) and tags_raw:
            tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags_list = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tags_list = None

        store.set(key, content, tags=tags_list, source=source)
        row = store.get(key)
        return _row_response(row)

    # ------------------------------------------------------------------
    # Delete memory
    # ------------------------------------------------------------------

    @app.delete("/v0/memories/{key:path}", status_code=204)
    def delete_memory(key: str):
        deleted = store.delete(key)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Memory '{key}' not found"),
            )
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @app.get("/v0/search")
    def search_memories(q: str = "", tag: str = "", limit: int = 20):
        if not q:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "Query parameter 'q' is required and must not be empty"),
            )
        rows = store.search(q, tag=tag, limit=limit)
        return [_search_row_response(r) for r in rows]

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    @app.get("/v0/tags")
    def list_tags():
        pairs = store.all_tags()
        return [{"tag": t, "count": c} for t, c in pairs]

    # ------------------------------------------------------------------
    # Dump
    # ------------------------------------------------------------------

    @app.get("/v0/dump")
    def dump(format: str = "md"):
        result = store.dump(fmt=format)
        if format == "json":
            import json
            return JSONResponse(content=json.loads(result))
        return PlainTextResponse(content=result)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @app.get("/v0/stats")
    def stats():
        return store.stats()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    @app.post("/v0/checkpoint")
    def checkpoint():
        store.checkpoint_wal()
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    db_path = Path(os.environ.get("MEM_DB_PATH", "/data/memory/mem.db"))
    host = os.environ.get("MEM_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("MEM_BIND_PORT", "8403"))
    log_level = os.environ.get("MEM_LOG_LEVEL", "info")

    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
