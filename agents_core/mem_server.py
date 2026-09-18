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
  MEM_PRINCIPALS    — JSON registration of principals (default: the built-in
                      brix-pm / zephyr-deposit table). Shape:
                      {"<name>": {"role": "curator"|"reader",
                                   "machine_state_prefixes": ["..."]}}
  MEM_ENFORCE_PRINCIPALS — "1"/"true" to ENFORCE the principal model (reject
                      reader writes with 403 principal_reader). Default OFF
                      (observe-only): the write class is LOGGED with what
                      WOULD be rejected, but not rejected. The enforcement
                      flip is a follow-up dispatch after the observe week
                      (openclaw-memdb-influx-reader-v0, D1).
  MEM_MACHINE_STATE_PREFIXES_PATH — the shared machine-state allowlist JSON
                      (default /srv/agents/config/mem-machine-state-prefixes.json).
                      The server REFUSES TO START if it is missing/malformed/
                      unreadable (fail-closed; no fail-open bypass).
  MEM_OBSERVE_LOG   — path for the observe-only faucet log (default
                      /srv/lapis/planning/reports/memdb-influx-observe-<YYYYMMDD>.log).

Principal model (openclaw-memdb-influx-reader-v0, D1):
  READ is open, WRITE is curated, enforced server-side. The principal travels
  in the X-Mem-Principal header (sent by MemClient from MEM_PRINCIPAL).
  `source` is NOT the principal (backfilled hostname, client-spoofable).
  Unknown/absent principal = reader; a reader's set/delete/deposit is
  REJECTED (403 principal_reader) under enforcement. The reader secret is
  honored ONLY for the read verb-set (secret-to-verb binding, panel F2).
  Fail-closed: unknown principal = reader; today's unauthenticated tailnet
  writers collapse to reader by default.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from agents_core import mem_machinery
from agents_core.mem import MemoryStore
from agents_core.mem_client import build_promoted_content

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Principal model (D1)
# ---------------------------------------------------------------------------

# The built-in principal registration. `brix-pm` covers the mem CLI +
# conductor node scripts + PM machinery (BRIX-resident). `zephyr-deposit` is
# the /v0/deposit route's principal (BRIX-resident). Everything else —
# including an absent or unrecognized principal — is a READER (fail-closed).
DEFAULT_PRINCIPALS: dict[str, dict] = {
    "brix-pm": {
        "role": "curator",
        "machine_state_prefixes": [],  # filled from the allowlist at boot
    },
    "zephyr-deposit": {
        "role": "curator",
        "machine_state_prefixes": [],
    },
}

# The write class: PUT / DELETE / POST /v0/deposit. Everything else is read.
WRITE_METHODS = {"PUT", "DELETE"}
DEPOSIT_PATH = "/v0/deposit"


def _load_principal_registration() -> dict[str, dict]:
    """Load the principal registration from MEM_PRINCIPALS (JSON) or fall back
    to the built-in table. Unknown/malformed JSON falls back to the built-in
    table (the principal model degrades to the known-good default; the
    allowlist guard is the hard fail-closed surface)."""
    raw = os.environ.get("MEM_PRINCIPALS", "").strip()
    if not raw:
        return json.loads(json.dumps(DEFAULT_PRINCIPALS))  # deep copy
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("MEM_PRINCIPALS must be a JSON object")
        return data
    except (json.JSONDecodeError, ValueError):
        logging.getLogger("mem-server").warning(
            "MEM_PRINCIPALS is malformed; falling back to the built-in table"
        )
        return json.loads(json.dumps(DEFAULT_PRINCIPALS))


def _is_enforce() -> bool:
    return os.environ.get("MEM_ENFORCE_PRINCIPALS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _principal_of(request: Request) -> str:
    """The caller's principal from the X-Mem-Principal header ("" = absent)."""
    return request.headers.get("X-Mem-Principal", "").strip()


def _principal_role(request: Request, principals: dict[str, dict]) -> str:
    """The role for the request's asserted principal. Unknown/absent = reader."""
    name = _principal_of(request)
    if not name:
        return "reader"
    reg = principals.get(name)
    if not reg:
        return "reader"
    return reg.get("role", "reader")


def _principal_prefixes(request: Request, principals: dict[str, dict]) -> set[str]:
    """The machine-state prefixes the asserted principal is registered for."""
    name = _principal_of(request)
    reg = principals.get(name)
    if not reg:
        return set()
    return set(reg.get("machine_state_prefixes", []))


# ---------------------------------------------------------------------------
# Observe-only faucet log (D1 / D3)
# ---------------------------------------------------------------------------

def _default_observe_log_path() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = os.environ.get("MEM_OBSERVE_LOG", "")
    if base:
        return Path(base)
    return Path(f"/srv/lapis/planning/reports/memdb-influx-observe-{day}.log")


class FaucetObserver:
    """Observe-only logger for the write class (D1 observe week).

    Records principal + source IP + key + HTTP verb on every write attempt,
    partitioning write counts by verb (PUT / DELETE / POST /v0/deposit) so a
    dormant or unregistered deposit writer cannot hide behind the PUT/DELETE
    counts (gate trickster). Also records per-prefix would-reject
    machine-state counts and the break list (today's writers that enforcement
    would reject).

    Observe-only: it LOGS what would be rejected; it never rejects. Enabling
    enforcement (MEM_ENFORCE_PRINCIPALS) is a separate, follow-up flip.
    """

    def __init__(self, log_path: Path | str, allowlist: "mem_machinery.MachineStateAllowlist",
                 principals: dict[str, dict]):
        self.log_path = Path(log_path)
        self.allowlist = allowlist
        self.principals = principals
        self._counts: dict[tuple[str, str], int] = {}  # (principal, verb) -> n
        self._would_reject_prefixes: dict[str, int] = {}  # prefix -> n
        self._break_list: dict[str, dict] = {}  # principal -> {verbs, prefixes}

    def _log_line(self, record: dict) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            # The observe log must never take the server down.
            pass

    def observe_write(self, request: Request, verb: str, key: str) -> dict:
        """Log a write attempt. Returns the verdict record (for logging/audit).

        verb is the normalized verb: "PUT", "DELETE", or "POST_DEPOSIT".
        """
        principal = _principal_of(request) or "none"
        role = _principal_role(request, self.principals)
        ip = request.client.host if request.client else "unknown"
        is_machine_state = self.allowlist.is_machine_state(key)
        producer = self.allowlist.producer_for(key) if is_machine_state else None

        # Verdict: would enforcement reject this write?
        would_reject = False
        if role != "curator":
            would_reject = True
        elif is_machine_state and producer is not None and producer != principal:
            # A curator writing a machine-state prefix it does not own.
            would_reject = True

        # Partitioned write counts by (principal, verb).
        ck = (principal, verb)
        self._counts[ck] = self._counts.get(ck, 0) + 1

        # Per-prefix would-reject machine-state counts.
        if is_machine_state and would_reject:
            prefix = self._matching_prefix(key)
            if prefix:
                self._would_reject_prefixes[prefix] = (
                    self._would_reject_prefixes.get(prefix, 0) + 1
                )

        # Break list: today's writers that enforcement would reject.
        if would_reject:
            entry = self._break_list.setdefault(
                principal, {"verbs": set(), "prefixes": set()}
            )
            entry["verbs"].add(verb)
            if is_machine_state:
                prefix = self._matching_prefix(key)
                if prefix:
                    entry["prefixes"].add(prefix)

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "write_attempt",
            "verb": verb,
            "key": key,
            "principal": principal,
            "role": role,
            "source_ip": ip,
            "machine_state": is_machine_state,
            "producer": producer,
            "would_reject": would_reject,
            "enforce": _is_enforce(),
        }
        self._log_line(record)
        return record

    def _matching_prefix(self, key: str) -> str | None:
        for e in self.allowlist.entries:
            if key.startswith(e.prefix):
                return e.prefix
        return None

    def report(self) -> dict:
        """The per-observed-writer table for the observe-week report."""
        # The write-class verbs partitioned in the report. POST_CHECKPOINT is
        # a maintenance verb (not the write class), but it is still observed —
        # include it in the verbs dict so total == sum(verbs) always holds
        # (a checkpoint writer must not show total > sum(verbs)).
        known_verbs = ("PUT", "DELETE", "POST_DEPOSIT", "POST_CHECKPOINT")
        writers: dict[str, dict] = {}
        for (principal, verb), n in self._counts.items():
            w = writers.setdefault(
                principal,
                {
                    "principal": principal,
                    "verbs": {v: 0 for v in known_verbs},
                    "total": 0,
                },
            )
            # Any observed verb is counted (defensive: a new verb never drops
            # out of the report silently).
            w["verbs"][verb] = w["verbs"].get(verb, 0) + n
            w["total"] += n
        break_list = {
            p: {"verbs": sorted(v["verbs"]), "prefixes": sorted(v["prefixes"])}
            for p, v in self._break_list.items()
        }
        return {
            "writers": writers,
            "would_reject_prefixes": self._would_reject_prefixes,
            "break_list": break_list,
        }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    db_path: Path,
    deposit_recorder: "DepositRecorder | None" = None,
    allowlist_path: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(title="mem-server", version="0")

    # ------------------------------------------------------------------
    # Startup-validation guard (gate technical-integrity + transmuter,
    # hard requirement): REFUSE TO START if the shared machine-state
    # allowlist is missing/malformed/unreadable. A fail-open bypass where a
    # missing allowlist silently disables the prefix-reject is the silent
    # fail-open the gate names. This runs BEFORE the store is even opened,
    # so the server never boots with a disabled guard.
    # ------------------------------------------------------------------
    _allowlist_path = (
        Path(allowlist_path)
        if allowlist_path is not None
        else mem_machinery.default_allowlist_path()
    )
    try:
        allowlist = mem_machinery.load_allowlist(_allowlist_path)
    except mem_machinery.AllowlistError as e:
        # Refuse to start: raise so the console-script entry point (main())
        # and any test harness see a hard failure, not a silently-open server.
        raise RuntimeError(
            f"mem-server REFUSES TO START: machine-state allowlist guard "
            f"failed (fail-closed, no fail-open bypass): {e}"
        ) from e

    store = MemoryStore(db_path)

    # ------------------------------------------------------------------
    # Principal model (D1)
    # ------------------------------------------------------------------
    principals = _load_principal_registration()
    # The registered producer principals inherit the machine-state prefixes
    # the allowlist assigns to them (faucet: producer principal writes its
    # own machine-state prefix into the machinery store).
    for entry in allowlist.entries:
        reg = principals.get(entry.producer_principal)
        if reg is not None:
            reg.setdefault("machine_state_prefixes", []).append(entry.prefix)

    enforce = _is_enforce()
    observer = FaucetObserver(
        _default_observe_log_path(), allowlist, principals
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
    # Write-class guard (D1 principal model + D3 faucet)
    # ------------------------------------------------------------------
    def _write_guard(request: Request, verb: str, key: str) -> JSONResponse | None:
        """Return a 403/400 JSONResponse to reject the write, or None to allow.

        Observe-only (default): logs the write attempt + what would be
        rejected, and returns None (allow). Under MEM_ENFORCE_PRINCIPALS:
        rejects a reader's write (403 principal_reader) and a non-owning
        curator's machine-state write (403 machine_state_prefix).
        """
        # Observe-only logging happens on EVERY write attempt, enforce or not.
        observer.observe_write(request, verb, key)

        if not enforce:
            return None  # observe-only: log, do not reject

        role = _principal_role(request, principals)
        if role != "curator":
            return JSONResponse(
                status_code=403,
                content=_error(
                    "principal_reader",
                    f"principal {_principal_of(request) or '<absent>'!r} is a "
                    f"reader; the write class (PUT/DELETE/deposit) is "
                    f"curator-only",
                ),
            )

        # D3 faucet: a curator may write a machine-state prefix only if it is
        # the owning registered producer principal for that prefix.
        if allowlist.is_machine_state(key):
            producer = allowlist.producer_for(key)
            if producer != _principal_of(request):
                return JSONResponse(
                    status_code=403,
                    content=_error(
                        "machine_state_prefix",
                        f"principal {_principal_of(request)!r} is not the "
                        f"registered producer for machine-state key {key!r} "
                        f"(owner: {producer!r})",
                    ),
                )
        return None

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
            "principal_model": {
                "enforce": enforce,
                "principals": sorted(principals.keys()),
            },
            "machinery_allowlist": {
                "path": str(_allowlist_path),
                "prefixes": list(allowlist.prefixes),
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
    def put_memory(key: str, request_data: dict[str, Any], request: Request):
        reject = _write_guard(request, "PUT", key)
        if reject is not None:
            return reject

        content = request_data.get("content", "")
        tags_raw = request_data.get("tags", "")
        source = request_data.get("source", "")

        tags_list = _normalize_tags(tags_raw)

        created = store.set(key, content, tags=tags_list, source=source)
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
    def delete_memory(key: str, request: Request):
        reject = _write_guard(request, "DELETE", key)
        if reject is not None:
            return reject
        deleted = store.delete(key)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=_error("not_found", f"Memory '{key}' not found"),
            )
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Promote (D2 — explicit verb, server-side provenance shape)
    # ------------------------------------------------------------------

    @app.post("/v0/promote")
    def promote(body: dict[str, Any], request: Request):
        """Promote one row from an agent store into mem.db (D2).

        Server-side endpoint: the client (MemClient.promote) may pre-validate,
        but the SERVER repeats the --from shape check (loud 400, panel
        security F6) and builds the exact provenance header + batch decision
        key. The server is the source of truth for the provenance shape —
        an auditor who cannot parse the shape cannot audit.

        Body: {key, from, principal, content, tags?}
          key       — the mem.db key to write
          from      — the --from agent-store ref, '<agent>/<store>'
          principal — the curator principal (X-Mem-Principal header is the
                      identity; this is the provenance header's 'by')
          content   — the promoted body (the header line is prepended)
          tags      — extra tags (the 'promoted' tag is always added)

        Writes with source="promoted:<agent>/<store>", tags 'promoted' +
        curator-chosen tags, and the exact provenance header as the first
        line of the content. Also upserts the batch decision key
        decision/memdb-promotion-<YYYYMMDD>-<principal> listing the promoted
        key + its --from ref (the auditable-to-a-decision artifact).
        """
        key = body.get("key", "")
        ref = body.get("from", "")
        principal = body.get("principal", "") or _principal_of(request)
        content = body.get("content", "")

        # Server-side --from shape validation (loud 400). The client validates
        # too, but the server repeats the check — a malformed ref must be
        # rejected at the HTTP edge regardless of the client. Validate the
        # key/principal FIRST so the write-class guard below runs on the
        # validated key (a malformed key must not reach the guard's
        # observe/prefix checks — reviewer PR #328 cycle 2 [med]).
        if not key:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_request", "'key' is required"),
            )
        if not principal:
            raise HTTPException(
                status_code=400,
                detail=_error(
                    "bad_request",
                    "a curator principal is required (X-Mem-Principal header "
                    "or body 'principal')",
                ),
            )
        try:
            promoted_content = build_promoted_content(ref, principal, content)
        except ValueError as e:
            # Newline/control-char injection or a malformed <agent>/<store>
            # ref — loud 400 (panel security F6).
            raise HTTPException(
                status_code=400,
                detail=_error("bad_from_ref", str(e)),
            )

        agent, store_name = ref.split("/", 1)
        source = f"promoted:{agent}/{store_name}"
        tag_list = ["promoted"]
        extra = body.get("tags", "")
        if extra:
            tag_list.extend(_normalize_tags(extra) or [])
        # De-dup while preserving the 'promoted' tag first.
        seen = set()
        tags_final = [t for t in tag_list if not (t in seen or seen.add(t))]

        store.set(key, promoted_content, tags=tags_final, source=source)
        row = store.get(key)

        # Batch decision key (D2 named decision artifact):
        # decision/memdb-promotion-<YYYYMMDD>-<curator> lists promoted keys +
        # their --from refs. Upserted (append-if-new) so a curation run
        # accumulates its batch in one key.
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        batch_key = f"decision/memdb-promotion-{day}-{principal}"
        line = f"- {key} --from {ref}"
        existing = store.get(batch_key)
        if existing and line in existing["content"]:
            batch_content = existing["content"]
        else:
            base = existing["content"] if existing else ""
            header = f"Promotion batch {day} by {principal}"
            batch_content = (
                f"{header}\n{line}\n" if not base else f"{base}\n{line}"
            )
        store.set(batch_key, batch_content, tags=["promoted-batch"],
                  source=source)

        return {**_row_response(row), "batch_key": batch_key}

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
            return JSONResponse(content=json.loads(result))
        return PlainTextResponse(content=result)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @app.get("/v0/stats")
    def stats():
        return store.stats()

    # ------------------------------------------------------------------
    # Checkpoint (maintenance — brix-pm)
    # ------------------------------------------------------------------

    @app.post("/v0/checkpoint")
    def checkpoint(request: Request):
        # Maintenance verb: brix-pm only. Observed like the write class;
        # enforced under MEM_ENFORCE_PRINCIPALS.
        reject = _write_guard(request, "POST_CHECKPOINT", "checkpoint")
        if reject is not None:
            return reject
        store.checkpoint_wal()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Deposit (Zephyr work-record envelope) - rides LapisToolReturn
    # ------------------------------------------------------------------

    @app.post("/v0/deposit")
    def deposit(envelope: dict[str, Any], request: Request):
        """Accept a LapisToolReturn deposit: persist payload to the mem store and
        append its provenance to the (injected) attribution log. Idempotent on
        provenance.manifest_hash (HIGH-1 construct-once / retry-identical-bytes).

        The route's principal is `zephyr-deposit` (BRIX-resident). Observed
        (and under enforcement, gated) like the write class — the deposit verb
        is partitioned as POST_DEPOSIT so a dormant/unregistered deposit writer
        cannot hide behind the PUT/DELETE counts (gate trickster)."""
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
        payload = ltr.payload
        if not isinstance(payload, dict) or "key" not in payload:
            raise HTTPException(
                status_code=400,
                detail=_error(
                    "bad_payload", "mem deposit payload must be {key, value, tags?}"
                ),
            )
        key = payload["key"]

        # Write-class guard (observed always; enforced under the flag).
        reject = _write_guard(request, "POST_DEPOSIT", key)
        if reject is not None:
            return reject

        if deposit_recorder is None:
            raise HTTPException(
                status_code=503,
                detail=_error(
                    "deposit_unconfigured",
                    "No attribution recorder injected (set MEM_DEPOSIT_RECORDER)",
                ),
            )

        mh = prov.manifest_hash
        if not mh:
            raise HTTPException(
                status_code=400,
                detail=_error("bad_envelope", "provenance.manifest_hash is required"),
            )

        # Dedup on manifest_hash. mem set() is an idempotent upsert, but we honor
        # the duplicate contract so append-style sinks (weaver) share this shape.
        if deposit_recorder.already_recorded(mh):
            return {"status": "duplicate", "key": key, "manifest_hash": mh}

        value = payload.get("value", payload.get("content", ""))
        tags_list = _normalize_tags(payload.get("tags"))
        store.set(key, value, tags=tags_list, source=prov.agent_id or "")
        deposit_recorder.record(prov.to_dict(), store_kind="mem", key=key)
        return {"status": "accepted", "key": key, "manifest_hash": mh}

    # ------------------------------------------------------------------
    # Observe-week report (D1 deliverable)
    # ------------------------------------------------------------------

    @app.get("/v0/observe-report")
    def observe_report():
        """The per-observed-writer table + would-reject counts + break list.

        Read-only surface for the observe-week report (panel H5). The report
        body lands at mem key state/memdb-influx-observe-week-<YYYYMMDD> + raw
        log /srv/lapis/planning/reports/memdb-influx-observe-<YYYYMMDD>.log; this
        endpoint exposes the in-process aggregation for the gate."""
        return observer.report()

    # Stash references for tests / the report.
    app.state.allowlist = allowlist
    app.state.principals = principals
    app.state.observer = observer
    app.state.enforce = enforce

    return app


# ---------------------------------------------------------------------------
# Row response helpers (kept after the factory so the routes above can use
# them — module-level defs are resolved at call time).
# ---------------------------------------------------------------------------

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


@runtime_checkable
class DepositRecorder(Protocol):
    """Attribution-log sink injected at boot (zephyr provides the impl).

    agents-core declares this interface and NEVER imports zephyr; the wiring is
    config-driven via MEM_DEPOSIT_RECORDER=<module>:<callable> (dynamic import in
    main()). Keeps the dependency arrow agents-core -> (interface) <- zephyr, per
    the substrate deposit endpoint spec (HIGH-2 layering)."""

    def already_recorded(self, manifest_hash: str) -> bool: ...

    def record(self, provenance: dict, *, store_kind: str, key: str | None) -> bool: ...


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

    # create_app() raises RuntimeError on a missing/malformed/unreadable
    # machine-state allowlist (fail-closed). Let it propagate: the server
    # must NOT start with the prefix-reject silently disabled.
    app = create_app(db_path, deposit_recorder=_load_deposit_recorder())
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
