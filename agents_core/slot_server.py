"""slot HTTP service — FastAPI wrapper around SlotStore (the project-slot blackboard).

This is the **actor-read API** ("Weaver Layer 5" in the design doc): the read-only
surface agents query to discover live slot/blackboard state before contributing — plus
the single-writer write surface that off-master contributors (shaped runners on other
nodes) use to self-report against the BRIX-canonical store.

Entry point:  slot-server  (console_scripts in pyproject.toml)
Port:         8405  (SLOTS_BIND_PORT env var)
Bind:         127.0.0.1 by default; production sets SLOTS_BIND_HOST=<tailscale-ip>

Environment variables (server side):
  SLOTS_DB_PATH      — SQLite DB file (default /data/slots/slots.db)
  SLOTS_BIND_HOST    — uvicorn bind host (default 127.0.0.1)
  SLOTS_BIND_PORT    — uvicorn bind port (default 8405)
  SLOTS_BEARER_TOKEN — optional shared bearer token; omit to disable auth
  SLOTS_LOG_LEVEL    — uvicorn log level (default info)

Agent-operable, NOT agent-as-destination: every write carries a contributor-of-record
(`by`), the single-writer guard rejects impostor writes, and the blackboard feeds the
human-readable Composer surface (post-GravityWell). Reads are advisory discovery.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from agents_core.slots import (
    SlotNotFoundError,
    SlotOwnershipError,
    SlotStore,
)


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _csv(value: str | None) -> list[str]:
    """Parse a comma-separated query param into a list (empty -> [])."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(title="slot-server", version="0")
    store = SlotStore(db_path)

    _token = os.environ.get("SLOTS_BEARER_TOKEN", "")

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
        st = store.stats()
        return {"status": "ok", **st}

    # ------------------------------------------------------------------
    # Reads — the actor-read blackboard surface (Layer 5)
    # ------------------------------------------------------------------

    @app.get("/v0/slots")
    def list_slots(
        project_id: str = "",
        status: str = "",
        contributor_id: str = "",
        limit: int = 100,
    ):
        return store.query(
            project_id=project_id or None,
            status=status or None,
            contributor_id=contributor_id or None,
            limit=limit,
        )

    @app.get("/v0/slots/adjacent")
    def adjacent(
        files: str = "",
        mem_keys: str = "",
        exclude_slot_id: str = "",
        include_inactive: bool = False,
    ):
        """Cross-slot proximity: active slots whose domain_touch overlaps the given
        files / mem-keys. The Reality-Snap primitive — read this before contributing."""
        return store.adjacent(
            files=_csv(files),
            mem_keys=_csv(mem_keys),
            exclude_slot_id=exclude_slot_id or None,
            include_inactive=include_inactive,
        )

    @app.get("/v0/slots/{slot_id}")
    def get_slot(slot_id: str):
        slot = store.get(slot_id)
        if slot is None:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Slot '{slot_id}' not found"),
            )
        return slot

    @app.get("/v0/stats")
    def stats():
        return store.stats()

    # ------------------------------------------------------------------
    # Writes — contributor-of-record (single-writer guarded)
    # ------------------------------------------------------------------

    @app.post("/v0/slots", status_code=201)
    def create_slot(body: dict[str, Any]):
        try:
            sid = store.create_slot(
                project_id=body["project_id"],
                contributor=body["contributor"],
                horizon=body.get("horizon"),
                slot_id=body.get("slot_id"),
                status=body.get("status", "dispatched"),
                domain_touch=body.get("domain_touch"),
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", f"missing field {e}"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", str(e)))
        return store.get(sid)

    @app.post("/v0/slots/{slot_id}/status")
    def update_status(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.update_status(
            slot_id, body["status"], by=body["by"]), slot_id)

    @app.post("/v0/slots/{slot_id}/checkpoint")
    def append_checkpoint(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.append_checkpoint(
            slot_id, body.get("kind", "self-report"), body.get("note", ""), by=body["by"]), slot_id)

    @app.post("/v0/slots/{slot_id}/domain")
    def set_domain(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.set_domain_touch(
            slot_id, files=body.get("files"), mem_keys=body.get("mem_keys"),
            scopes=body.get("scopes"), by=body["by"]), slot_id)

    @app.post("/v0/slots/{slot_id}/escalate")
    def escalate(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.escalate(
            slot_id, to=body.get("to", "facets"), reason=body.get("reason", ""), by=body["by"]), slot_id)

    @app.post("/v0/slots/{slot_id}/observer")
    def observer_update(slot_id: str, body: dict[str, Any]):
        """Observer (Weaver-derived) corroboration write — lands in the separate
        weaver_* namespace, never touches contributor-of-record fields."""
        try:
            store.observer_update(slot_id, body["weaver_status"], by=body.get("by", "weaver"))
        except SlotNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", slot_id))
        return store.get(slot_id)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @app.post("/v0/checkpoint")
    def checkpoint():
        store.checkpoint_wal()
        return {"ok": True}

    @app.post("/v0/expire")
    def expire():
        return store.expire()

    # ------------------------------------------------------------------
    # Shared write-guard error mapping
    # ------------------------------------------------------------------

    def _guarded(fn, slot_id: str):
        try:
            fn()
        except SlotNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", f"Slot '{slot_id}' not found"))
        except SlotOwnershipError as e:
            raise HTTPException(status_code=403, detail=_error("not_owner", str(e)))
        except KeyError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", f"missing field {e}"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", str(e)))
        return store.get(slot_id)

    return app


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    db_path = Path(os.environ.get("SLOTS_DB_PATH", "/data/slots/slots.db"))
    host = os.environ.get("SLOTS_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("SLOTS_BIND_PORT", "8405"))
    log_level = os.environ.get("SLOTS_LOG_LEVEL", "info")

    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
