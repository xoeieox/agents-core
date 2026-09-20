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
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from agents_core.mem import MemoryStore, TestWriteRejected
from agents_core.mem_hygiene import (
    HygieneAborted,
    HygieneConfig,
    MemHygieneRunner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


@runtime_checkable
class DepositRecorder(Protocol):
    """Attribution-log sink injected at boot (zephyr provides the impl).

    agents-core declares this interface and NEVER imports zephyr; the wiring is
    config-driven via MEM_DEPOSIT_RECORDER=<module>:<callable> (dynamic import in
    main()). Keeps the dependency arrow agents-core -> (interface) <- zephyr, per
    the substrate deposit endpoint spec (HIGH-2 layering)."""

    def already_recorded(self, manifest_hash: str) -> bool: ...

    def record(self, provenance: dict, *, store_kind: str, key: str | None) -> bool: ...


def _normalize_tags(tags_raw: Any) -> list[str] | None:
    """Accept tags as a comma-separated string or a list; normalize for set()."""
    if isinstance(tags_raw, str) and tags_raw:
        return [t.strip() for t in tags_raw.split(",") if t.strip()]
    if isinstance(tags_raw, list):
        return [str(t).strip() for t in tags_raw if str(t).strip()]
    return None


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

def create_app(db_path: Path, deposit_recorder: "DepositRecorder | None" = None) -> FastAPI:
    app = FastAPI(title="mem-server", version="0")
    store = MemoryStore(db_path)

    # ------------------------------------------------------------------
    # Error envelope (mem-hygiene-automation-v0 directive 5f61cad3, fix 1):
    # routes raise HTTPException(detail=_error(code, message)); the default
    # FastAPI handler would wrap that dict in {"detail": {...}}. Unwrap it so
    # every error response is the top-level {"error": {"code", "message"}}
    # envelope the _error() helper and the test suite contract on.
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
            content = detail
        else:
            content = _error("error", str(detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=getattr(exc, "headers", None),
        )

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
            "deposit": {"configured": deposit_recorder is not None},
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

        tags_list = _normalize_tags(tags_raw)

        try:
            created = store.set(key, content, tags=tags_list, source=source)
        except TestWriteRejected as exc:
            # D4 (mem-hygiene-automation-v0): a dedicated 4xx, not a bare
            # 500 — the write was refused by the guard, not a server fault.
            raise HTTPException(
                status_code=409,
                detail=_error("test_write_rejected", str(exc)),
            )
        row = store.get(key)
        # `created` is PUT-only: the store.set() return is the in-lock
        # pre-existence check (True = just created, False = updated). A read
        # cannot know the create/update status of the last write, so the field
        # is confined to this response and never added to _row_response.
        return {**_row_response(row), "created": created}

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

    # ------------------------------------------------------------------
    # Hygiene (mem-hygiene-automation-v0) — server-side run surface.
    #
    # The CLI (conductor scripts/mem.py `mem hygiene ...`) is a thin HTTP
    # client over these endpoints; the server holds the store. The
    # MEM_SERVER-unset direct-sqlite fallback path in the conductor CLI is
    # a NAMED NON-PATH for hygiene (the guard + quarantine are
    # server-side). Config comes from MEM_HYGIENE_CONFIG (named config
    # file) — never auto-inferred.
    # ------------------------------------------------------------------

    def _hygiene_runner(run_id: str | None = None) -> MemHygieneRunner:
        # MEM_HYGIENE_CONFIG is read per-call (not at boot) so a config
        # rotation does not require a mem-server restart.
        try:
            config = HygieneConfig.from_env()
        except Exception as exc:  # noqa: BLE001 - any config failure -> 503
            raise HTTPException(
                status_code=503,
                detail=_error("hygiene_unconfigured",
                              f"MEM_HYGIENE_CONFIG not loadable: {exc}"),
            )
        return MemHygieneRunner(store, config, run_id=run_id)

    @app.post("/v0/hygiene/run")
    def hygiene_run(request_data: dict[str, Any]):
        """One bounded hygiene run (D1 + D3). dry_run=true writes the
        candidate artifact and returns the verdict without mutating."""
        runner = _hygiene_runner(request_data.get("run_id"))
        try:
            verdict = runner.run_quarantine(
                dry_run=bool(request_data.get("dry_run", False)),
                allow_over_cap=bool(request_data.get("allow_over_cap", False)),
            )
        except HygieneAborted as exc:
            raise HTTPException(
                status_code=409,
                detail=_error("hygiene_aborted", str(exc)),
            )
        return verdict.to_dict()

    @app.post("/v0/hygiene/ageout")
    def hygiene_ageout(request_data: dict[str, Any]):
        """Purge quarantined rows past the rollback window (D1)."""
        runner = _hygiene_runner()
        window = request_data.get("window")
        purged = runner.ageout(window_days=int(window) if window else None)
        return {"purged": purged, "run_id": runner.run_id}

    @app.get("/v0/hygiene/list")
    def hygiene_list():
        """Quarantine census (D-2 `mem hygiene list` surface)."""
        runner = _hygiene_runner()
        return runner.quarantine_stats()

    @app.post("/v0/hygiene/restore")
    def hygiene_restore(request_data: dict[str, Any]):
        """Restore a quarantined prefix back into memories (D-2:
        INSERT..SELECT + DELETE pair; the memories_ai trigger re-indexes
        FTS). Named F6 exception: a direct store write from the
        maintenance path, bypassing the D4 set-guard by design."""
        prefix = (request_data.get("prefix") or "").strip()
        if not prefix:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "'prefix' is required"),
            )
        runner = _hygiene_runner()
        try:
            restored = runner.restore_prefix(prefix)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", str(exc)),
            )
        return {"restored": restored, "prefix": prefix, "run_id": runner.run_id}

    # ------------------------------------------------------------------
    # Deposit (Zephyr work-record envelope) - rides LapisToolReturn
    # ------------------------------------------------------------------

    @app.post("/v0/deposit")
    def deposit(envelope: dict[str, Any]):
        """Accept a LapisToolReturn deposit: persist payload to the mem store and
        append its provenance to the (injected) attribution log. Idempotent on
        provenance.manifest_hash (HIGH-1 construct-once / retry-identical-bytes)."""
        if deposit_recorder is None:
            raise HTTPException(
                status_code=503,
                detail=_error(
                    "deposit_unconfigured",
                    "No attribution recorder injected (set MEM_DEPOSIT_RECORDER)",
                ),
            )
        # Validate the envelope via the canonical dataclass (lazy import keeps boot
        # free of a hard archetypes_core dependency).
        try:
            from archetypes_core.provenance import LapisToolReturn

            ltr = LapisToolReturn.from_dict(envelope)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_envelope", f"Invalid LapisToolReturn: {e}"),
            )

        prov = ltr.provenance
        mh = prov.manifest_hash
        if not mh:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_envelope", "provenance.manifest_hash is required"),
            )

        payload = ltr.payload
        if not isinstance(payload, dict) or "key" not in payload:
            raise HTTPException(
                status_code=400,
                detail=_error(
                    "bad_payload", "mem deposit payload must be {key, value, tags?}"
                ),
            )
        key = payload["key"]

        # Dedup on manifest_hash. mem set() is an idempotent upsert, but we honor
        # the duplicate contract so append-style sinks (weaver) share this shape.
        if deposit_recorder.already_recorded(mh):
            return {"status": "duplicate", "key": key, "manifest_hash": mh}

        value = payload.get("value", payload.get("content", ""))
        tags_list = _normalize_tags(payload.get("tags"))
        store.set(key, value, tags=tags_list, source=prov.agent_id or "")
        deposit_recorder.record(prov.to_dict(), store_kind="mem", key=key)
        return {"status": "accepted", "key": key, "manifest_hash": mh}

    return app


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def _load_deposit_recorder() -> "DepositRecorder | None":
    """Load the attribution recorder from MEM_DEPOSIT_RECORDER=<module>:<callable>.

    Dynamic import (no static zephyr dependency). ANY failure is swallowed with a
    warning and returns None so the server still boots and serves every existing
    route; only /v0/deposit degrades to 503. This protects the live master."""
    import logging

    log = logging.getLogger("mem-server")
    spec = os.environ.get("MEM_DEPOSIT_RECORDER", "").strip()
    if not spec:
        return None
    try:
        import importlib

        mod_name, _, attr = spec.partition(":")
        factory = getattr(importlib.import_module(mod_name), attr)
        recorder = factory()
        log.info("deposit recorder loaded: %s", spec)
        return recorder
    except Exception as e:  # noqa: BLE001 - boot must never fail over this
        log.warning("deposit recorder load FAILED (%s): %s; /v0/deposit will 503", spec, e)
        return None


def main():
    import uvicorn

    db_path = Path(os.environ.get("MEM_DB_PATH", "/data/memory/mem.db"))
    host = os.environ.get("MEM_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("MEM_BIND_PORT", "8403"))
    log_level = os.environ.get("MEM_LOG_LEVEL", "info")

    app = create_app(db_path, deposit_recorder=_load_deposit_recorder())
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
