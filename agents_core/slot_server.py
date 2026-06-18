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
  SLOTS_LOG_LEVEL    — uvicorn log level (default info)
  SLOT_ADJACENT_CACHE_TTL_SEC — TTL for the single-flight adjacent() cache in seconds
                                (default 2.0; set to 0 to disable caching)

  SLOTS_BEARER_TOKEN — auth token in one of two forms:
    - "secret"                  legacy shared mode: any authenticated caller may write
                                as any contributor (weaker ownership; backward-compat).
    - "contributor_id:secret"   per-principal mode: only writes where body["by"] ==
                                contributor_id are permitted. Recommended for production.
    Omit SLOTS_BEARER_TOKEN only when the server is bound to 127.0.0.1 (loopback).
    Binding to a non-loopback host without a token is refused at startup (fail-closed).

Auth implementation:
  Bearer comparison uses hmac.compare_digest (constant-time) to prevent timing attacks.

Execution-model contract (correctness precondition):
  The read cache (ETag token computation and the single-flight adjacent() cache)
  assumes **sync-threaded execution** — FastAPI on a threadpool + threading primitives
  (threading.RLock, threading.Event). This is how slot-server runs today.

  If slot-server is ever migrated to asyncio-native handlers (async def, asyncio.Lock),
  the threading-based single-flight cache will silently break and serve stale/incorrect
  data. Before migrating to asyncio, replace the threading.Event single-flight with an
  asyncio.Event-based implementation.

Agent-operable, NOT agent-as-destination: every write carries a contributor-of-record
(`by`), the single-writer guard rejects impostor writes, and the blackboard feeds the
human-readable Composer surface (post-GravityWell). Reads are advisory discovery.

Deployment:
  The slot_server binary is installed as a console_script via agents_core/pyproject.toml.
  It may be managed as:
  1. Systemd service (slot-server.service at /srv/agents/systemd/) — enable with:
       sudo systemctl enable /srv/agents/systemd/slot-server.service
       sudo systemctl start slot-server
  2. Unmanaged bare PID (current state as of 2026-06-17) — restart manually:
       pkill -f "^/.*slot-server$"
       # New instance will start on next scheduled tick or manual invocation
  The elevator queue routes (/v0/elevator/*) added in gw-elevator-queue-substrate-v0
  are served by this same process (port 8405); no separate queue-server is needed.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from agents_core.elevator import (
    ElevatorStore,
    OffMasterWriteError,
    QueueNotFoundError,
)
from agents_core.interactive_submit import submit
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


def create_app(db_path: Path, elevator_db_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="slot-server", version="0")
    store = SlotStore(db_path)
    elevator = ElevatorStore(elevator_db_path or Path("/data/elevator/queue.db"))

    _raw_token = os.environ.get("SLOTS_BEARER_TOKEN", "")
    # Per-principal mode: "contributor_id:secret" — the part before the first colon
    # is the authorized writer; writes where body["by"] != contributor_id are rejected.
    # Legacy shared mode: no colon — any authenticated caller may write as any principal.
    if ":" in _raw_token:
        _colon = _raw_token.index(":")
        _token_principal: str | None = _raw_token[:_colon]
        _token_secret: str = _raw_token[_colon + 1:]
    else:
        _token_principal = None
        _token_secret = _raw_token

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # Propagate the authenticated principal to write endpoints. None means no
        # per-principal binding (loopback or legacy shared-token mode).
        request.state.principal = _token_principal
        if _token_secret:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
            presented = auth[len("Bearer "):]
            # Constant-time comparison — prevents timing-oracle token enumeration.
            if not hmac.compare_digest(
                presented.encode("utf-8"), _token_secret.encode("utf-8")
            ):
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
        return await call_next(request)

    @app.exception_handler(OffMasterWriteError)
    async def off_master_write_error_handler(request: Request, exc: OffMasterWriteError):
        """Handle off-master write attempts with a 403 error."""
        return JSONResponse(
            status_code=403,
            content=_error("forbidden", str(exc)),
        )

    def _check_by_principal(by: str) -> None:
        """Reject writes where the caller's authenticated principal doesn't match `by`.

        No-op in legacy shared-token mode (_token_principal is None) and when the
        server is bound to loopback without a token (trusted local callers).
        """
        if _token_principal and by != _token_principal:
            raise HTTPException(
                status_code=403,
                detail=_error(
                    "not_authorized",
                    f"token not authorized to write as '{by}'",
                ),
            )

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
        request: Request = None,
    ):
        # Compute ETag for conditional reads (same filters as query).
        etag = store.read_version(
            project_id=project_id or None,
            status=status or None,
            contributor_id=contributor_id or None,
            limit=limit,
        )
        # Check If-None-Match: if client's ETag matches, return 304 Not Modified.
        if_none_match = request.headers.get("If-None-Match", "").strip()
        if if_none_match == etag:
            return Response(status_code=304, headers={"ETag": etag})
        # Else run query and return 200 with headers.
        data = store.query(
            project_id=project_id or None,
            status=status or None,
            contributor_id=contributor_id or None,
            limit=limit,
        )
        return JSONResponse(
            content=data,
            headers={"ETag": etag, "Cache-Control": "no-cache"},
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
    def get_slot(slot_id: str, request: Request = None):
        # Compute ETag specific to this slot_id.
        etag = store.read_version(slot_id=slot_id)
        # Check If-None-Match.
        if_none_match = request.headers.get("If-None-Match", "").strip()
        if if_none_match == etag:
            return Response(status_code=304, headers={"ETag": etag})
        # Else fetch and return 200 with headers.
        slot = store.get(slot_id)
        if slot is None:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Slot '{slot_id}' not found"),
            )
        return JSONResponse(
            content=slot,
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    @app.get("/v0/stats")
    def stats():
        return store.stats()

    # ------------------------------------------------------------------
    # Writes — contributor-of-record (single-writer guarded)
    # ------------------------------------------------------------------

    @app.post("/v0/slots", status_code=201)
    def create_slot(body: dict[str, Any]):
        # Principal binding: the contributor.id must match the authenticated principal.
        by = (body.get("contributor") or {}).get("id")
        if by is not None:
            _check_by_principal(by)
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
            slot_id, body["status"], by=body["by"]), slot_id,
            by=body.get("by"))

    @app.post("/v0/slots/{slot_id}/checkpoint")
    def append_checkpoint(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.append_checkpoint(
            slot_id, body.get("kind", "self-report"), body.get("note", ""), by=body["by"]), slot_id,
            by=body.get("by"))

    @app.post("/v0/slots/{slot_id}/domain")
    def set_domain(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.set_domain_touch(
            slot_id, files=body.get("files"), mem_keys=body.get("mem_keys"),
            scopes=body.get("scopes"), by=body["by"]), slot_id,
            by=body.get("by"))

    @app.post("/v0/slots/{slot_id}/escalate")
    def escalate(slot_id: str, body: dict[str, Any]):
        return _guarded(lambda: store.escalate(
            slot_id, to=body.get("to", "facets"), reason=body.get("reason", ""), by=body["by"]), slot_id,
            by=body.get("by"))

    @app.post("/v0/slots/{slot_id}/observer")
    def observer_update(slot_id: str, body: dict[str, Any]):
        """Observer (Weaver-derived) corroboration write — lands in the separate
        weaver_* namespace, never touches contributor-of-record fields.
        Observer writes are not principal-bound (Weaver is a distinct actor)."""
        try:
            store.observer_update(slot_id, body["weaver_status"], by=body.get("by", "weaver"))
        except SlotNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", slot_id))
        return store.get(slot_id)

    @app.post("/v0/slots/{slot_id}/ratify")
    def facets_ratify(slot_id: str, body: dict[str, Any]):
        """Facets ratification write (blackboard bootstrap step 4) — the three-voice
        consult verdict for an escalated slot lands in the separate facets_* namespace,
        never touching contributor-of-record or weaver fields. Body: {"verdict": {...},
        "by": "facets"}."""
        try:
            store.facets_ratify(slot_id, body["verdict"], by=body.get("by", "facets"))
        except SlotNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", slot_id))
        except KeyError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", f"missing field {e}"))
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

    def _guarded(fn, slot_id: str, *, by: str | None = None):
        # Principal binding check fires before the store layer so a spoofed `by`
        # gets 403 (not_authorized) rather than 403 (not_owner) from SlotStore.
        if by is not None:
            _check_by_principal(by)
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

    # ------------------------------------------------------------------
    # Elevator queue routes
    # ------------------------------------------------------------------

    @app.post("/v0/elevator/enqueue", status_code=201)
    def enqueue_item(body: dict[str, Any]):
        """Enqueue a work item to the elevator queue."""
        try:
            item_id = elevator.enqueue(
                lane=body["lane"],
                kind=body["kind"],
                payload=body["payload"],
                principal=body["principal"],
                latency_class=body["latency_class"],
                slot_ref=body.get("slot_ref"),
            )
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", str(e)))
        return elevator.get(item_id)

    @app.post("/v0/elevator/claim")
    def claim_item(body: dict[str, Any]):
        """Claim the next pending item from the given lanes."""
        try:
            lanes = body.get("lanes", ["interactive", "deliberation", "execution"])
            owner = body["owner"]
            claim_ttl_sec = body.get("claim_ttl_sec", 30)
            item = elevator.claim(lanes=lanes, owner=owner, claim_ttl_sec=claim_ttl_sec)
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", str(e)))
        if item is None:
            return None  # No pending items.
        return item

    @app.post("/v0/elevator/ack")
    def ack_item(body: dict[str, Any]):
        """Acknowledge a served item."""
        try:
            item_id = body["item_id"]
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        try:
            elevator.ack(
                item_id,
                result_ref=body.get("result_ref"),
                provenance=body.get("provenance"),
                result=body.get("result"),
            )
        except QueueNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", item_id))
        return elevator.get(item_id)

    @app.post("/v0/elevator/requeue")
    def requeue_item(body: dict[str, Any]):
        """Requeue a claimed item back to pending."""
        try:
            item_id = body["item_id"]
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        try:
            elevator.requeue(item_id)
        except QueueNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", item_id))
        return elevator.get(item_id)

    @app.post("/v0/elevator/fail")
    def fail_item(body: dict[str, Any]):
        """Mark a claimed item as permanently failed."""
        try:
            item_id = body["item_id"]
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        try:
            elevator.fail(item_id)
        except QueueNotFoundError:
            raise HTTPException(status_code=404, detail=_error("not_found", item_id))
        return elevator.get(item_id)

    @app.get("/v0/elevator/item/{item_id}")
    def get_item(item_id: str):
        """Fetch an elevator queue item by ID."""
        item = elevator.get(item_id)
        if item is None:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Item '{item_id}' not found"),
            )
        return item

    @app.get("/v0/elevator/state")
    def get_state():
        """Fetch queue state (node-state / floor-indicator)."""
        return elevator.state()

    @app.post("/v0/elevator/submit")
    def submit_turn(body: dict[str, Any]):
        """Submit a turn (async queue or direct routing).

        Body:
            turn: str - the prompt/message
            context: dict (optional) - system context
            destination: str - "queue-gw-interactive" or "route-to-fast"
            operator: str (optional) - operator for route-to-fast (default: "qwen")
            principal: str (optional) - submitter identity (default: "default")
        """
        try:
            turn = body["turn"]
            destination = body.get("destination", "queue-gw-interactive")
        except KeyError as e:
            raise HTTPException(
                status_code=400, detail=_error("bad_request", f"missing field {e}")
            )
        try:
            result = submit(
                turn=turn,
                context=body.get("context"),
                destination=destination,
                operator=body.get("operator"),
                principal=body.get("principal", "default"),
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=_error("bad_request", str(e)))

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
    token = os.environ.get("SLOTS_BEARER_TOKEN", "")

    # Fail-closed: refuse to start if no bearer token and not loopback-bound.
    # Loopback-only binding (127.0.0.1 / ::1) is safe without a token because
    # only local processes can reach it. Any Tailscale or routable bind without
    # a token exposes an unauthenticated write surface — refuse, don't warn.
    if not token:
        try:
            addr = ipaddress.ip_address(host)
            if not addr.is_loopback:
                print(
                    f"FATAL: SLOTS_BEARER_TOKEN is unset but SLOTS_BIND_HOST={host!r} "
                    f"is non-loopback. Set SLOTS_BEARER_TOKEN or bind to 127.0.0.1.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except ValueError:
            # Host is a hostname string, not a bare IP — can't check loopback
            # status at startup. The operator is responsible for token config.
            pass

    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
