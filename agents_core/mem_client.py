"""mem HTTP client — thin httpx wrapper for mem-server.

Reads configuration from environment:
  MEM_SERVER          — base URL, e.g. http://100.x.y.z:8403
  MEM_BEARER_TOKEN    — optional bearer token (must match server)
  MEM_CLIENT_TIMEOUT  — per-request timeout in seconds (default 5.0)
  MEM_PRINCIPAL       — the caller's principal name, sent as the X-Mem-Principal
                        header on every request (openclaw-memdb-influx-reader-v0,
                        D1). `source` is NOT the principal (backfilled hostname,
                        client-spoofable) — the principal is the identity the
                        server's principal model + observe-only faucet log key on.
                        Unset = no header = the server treats the caller as a
                        reader (fail-closed).

Raises MemHTTPError on non-2xx responses.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Promotion provenance (openclaw-memdb-influx-reader-v0, D2)
# ---------------------------------------------------------------------------

# The first line of a promoted row's content is EXACTLY this shape:
#   [promoted from <agent>/<store> by <principal> at <YYYY-MM-DDTHH:MM:SSZ>]
# An auditor who cannot parse the shape cannot audit.
PROMOTED_HEADER_RE = re.compile(
    r"^\[promoted from ([^/ \[\]]+)/([^\] \[\]]+) by ([^\] \[\]]+) at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]$"
)

# --from must be a path-like/ref-like token: no newlines, no control chars.
# One `if` in the promote handler (panel security F6) — newline injection into
# the provenance header is the attack this shape exists to block.
# Also reused by the server's /v0/promote handler for the one-line rationale
# check (a multi-line/control-char rationale would corrupt the batch decision
# key's line-based listing the same way).
INVALID_REF_CHARS_RE = re.compile(r"[\n\r\x00-\x1f\x7f]")
# Back-compat alias (pre-rename name; kept so existing imports/tests still work).
_FROM_INVALID_RE = INVALID_REF_CHARS_RE


def validate_promote_source_ref(ref: str) -> str:
    """Validate a `mem promote --from <agent-store-ref>` token.

    Returns the ref unchanged when valid; raises ValueError (loud 400 at the
    HTTP edge) when it contains a newline or any control character, or is
    empty. The shape is a path-like/ref-like token: no newlines or control
    chars, ever.
    """
    if not ref or INVALID_REF_CHARS_RE.search(ref):
        raise ValueError(
            f"--from ref is not a valid path-like/ref-like token "
            f"(no newlines or control chars): {ref!r}"
        )
    return ref


def format_promoted_header(agent: str, store: str, principal: str,
                           at: datetime | None = None) -> str:
    """Render the exact promoted-content header line (D2 provenance shape)."""
    ts = (at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"[promoted from {agent}/{store} by {principal} at {ts}]"


def parse_promoted_header(content: str) -> dict[str, str] | None:
    """Parse the D2 provenance header off a promoted row's content.

    Returns {"agent", "store", "principal", "at"} when the FIRST line of
    `content` matches the exact shape, else None.
    """
    if not content:
        return None
    m = PROMOTED_HEADER_RE.match(content.split("\n", 1)[0])
    if not m:
        return None
    return {
        "agent": m.group(1),
        "store": m.group(2),
        "principal": m.group(3),
        "at": m.group(4),
    }


def build_promoted_content(ref: str, principal: str, content: str,
                           at: datetime | None = None) -> str:
    """Build the full promoted row content: header line + blank line + content.

    `ref` is the `--from` agent-store ref, shape-validated as
    `<agent>/<store>`. Raises ValueError on a malformed ref (no newlines or
    control chars, must contain exactly one `/` separating agent from store).
    """
    validate_promote_source_ref(ref)
    if ref.count("/") != 1 or not ref[0] or ref[-1] == "/":
        raise ValueError(
            f"--from ref must be '<agent>/<store>' (path-like, one slash): {ref!r}"
        )
    agent, store = ref.split("/", 1)
    if not agent or not store:
        raise ValueError(f"--from ref must be '<agent>/<store>': {ref!r}")
    header = format_promoted_header(agent, store, principal, at=at)
    return f"{header}\n\n{content}"


class MemHTTPError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


# ---------------------------------------------------------------------------
# --store plumbing (openclaw-memdb-influx-reader-v0, D2 / Files-changed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --store plumbing (openclaw-memdb-influx-reader-v0, D2 / Files-changed)
# ---------------------------------------------------------------------------

# RESCOPED (rev-2, 2026-09-14 gate proceed-to-bind): the machinery store is
# the EXISTING exhaust store (the route_to_exhaust mechanism in
# mem_exhaust.py), not a second sqlite (no mem_machinery.db). 'machinery'
# == 'exhaust' for every allowlist entry.
#
# The --store flag is therefore MOOT at the HTTP layer: the mem-server routes
# machine-state keys to the machinery (exhaust) store transparently at the
# set() chokepoint (MemoryStore.set -> mem_exhaust.route_to_exhaust), so the
# client never selects a store — the server does. This plumbing exists so the
# spec's Files-changed line is honored and so a future non-rescoped
# second-sqlite deployment has a named flag ready. The flag is validated
# (loud ValueError on a typo) so a caller who passes it gets a clear error
# rather than a silent no-op.
STORE_ATOMS = "atoms"
STORE_MACHINERY = "machinery"
# Accepted values for the --store flag. 'exhaust' is an alias for 'machinery'
# (the rescoped store name) so a caller who knows the underlying store can
# use either.
KNOWN_STORES = (STORE_ATOMS, STORE_MACHINERY, "exhaust")


def validate_store(store: str) -> str:
    """Validate a --store value. Returns the normalized store name
    ('exhaust' -> 'machinery'). Raises ValueError on an unknown store so a
    typo is loud, not silently ignored (fail-closed at the client edge)."""
    if store in KNOWN_STORES:
        return STORE_MACHINERY if store == "exhaust" else store
    raise ValueError(
        f"unknown --store {store!r} (known: {', '.join(KNOWN_STORES)})"
    )


class MemClient:
    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
        principal: str | None = None,
    ):
        self._base_url = (base_url or os.environ.get("MEM_SERVER", "")).rstrip("/")
        _token = token if token is not None else os.environ.get("MEM_BEARER_TOKEN", "")
        _timeout = timeout if timeout is not None else float(
            os.environ.get("MEM_CLIENT_TIMEOUT", "5.0")
        )
        _principal = (
            principal if principal is not None
            else os.environ.get("MEM_PRINCIPAL", "")
        )
        headers = {}
        if _token:
            headers["Authorization"] = f"Bearer {_token}"
        if _principal:
            # The principal travels in a dedicated header (D1) — never in
            # `source`, which the server backfills to its hostname and which
            # any client can spoof.
            headers["X-Mem-Principal"] = _principal
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=_timeout,
        )

    def _check(self, resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise MemHTTPError(resp.status_code, body)
        return resp

    def healthz(self) -> dict:
        return self._check(self._client.get("/healthz")).json()

    def list(
        self,
        tag: str = "",
        since: str = "",
        limit: int = 50,
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if tag:
            params["tag"] = tag
        if since:
            params["since"] = since
        return self._check(self._client.get("/v0/memories", params=params)).json()

    def get(self, key: str) -> dict:
        return self._check(self._client.get(f"/v0/memories/{key}")).json()

    def set(
        self,
        key: str,
        content: str,
        tags: str = "",
        source: str = "",
        store: str = STORE_ATOMS,
    ) -> dict:
        """Upsert a memory.

        `store` is the --store flag (openclaw-memdb-influx-reader-v0, D2 /
        Files-changed). RESCOPED (rev-2): the machinery store is the EXISTING
        exhaust store, so the flag is MOOT at the HTTP layer — the server
        routes machine-state keys to the machinery (exhaust) store
        transparently at the set() chokepoint. The flag is validated (loud
        ValueError on a typo) but does NOT change routing: the server's
        route_to_exhaust is the single source of truth for where a key lands.
        Passing store='machinery' for a non-machine-state key is a no-op (the
        key still lands in mem.db) — the server does not honor a
        client-selected store for a key that is not machine-state, because
        that would let a client move an arbitrary key out of the ledger of
        record (influx, not merge: mem.db is the single ledger of record)."""
        validate_store(store)
        body: dict[str, Any] = {"content": content, "tags": tags, "source": source}
        return self._check(self._client.put(f"/v0/memories/{key}", json=body)).json()

    def delete(self, key: str) -> None:
        self._check(self._client.delete(f"/v0/memories/{key}"))

    def search(
        self,
        query: str,
        tag: str = "",
        limit: int = 20,
    ) -> list[dict]:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if tag:
            params["tag"] = tag
        return self._check(self._client.get("/v0/search", params=params)).json()

    def tags(self) -> list[dict]:
        return self._check(self._client.get("/v0/tags")).json()

    def dump(self, format: str = "md") -> str:
        resp = self._check(self._client.get("/v0/dump", params={"format": format}))
        if format == "json":
            return resp.text
        return resp.text

    def stats(self) -> dict:
        return self._check(self._client.get("/v0/stats")).json()

    def promote(
        self,
        key: str,
        ref: str,
        principal: str,
        content: str,
        tags: str = "",
        rationale: str = "",
    ) -> dict:
        """Promote one row from an agent store into mem.db (D2).

        Writes with `source="promoted:<agent>/<store>"`, tags `promoted` +
        curator-chosen tags, and the exact provenance header as the first line
        of the content. One-way: mem.db never writes back.

        `ref` is the `--from` agent-store ref (`<agent>/<store>`); a ref with
        a newline or control char is rejected client-side (loud ValueError)
        before any request is sent — the server repeats the check (loud 400)
        at the /v0/promote edge (panel security F6).

        Uses the server-side POST /v0/promote endpoint: the server builds the
        exact provenance header, sets source="promoted:<agent>/<store>", adds
        the 'promoted' tag, and upserts the batch decision key. The client
        still validates the ref up front so a bad ref never leaves the process.

        `rationale` is the one-line rationale the D2 named decision artifact
        requires per promoted key: the batch decision key
        decision/memdb-promotion-<YYYYMMDD>-<curator> lists promoted keys +
        their --from refs + one-line rationale. It is audit metadata only —
        it is NOT part of the provenance header shape (the header stays
        exactly `[promoted from <agent>/<store> by <principal> at <ts>]`).
        """
        # Client-side shape validation (loud ValueError) before any request.
        validate_promote_source_ref(ref)
        if ref.count("/") != 1 or not ref[0] or ref[-1] == "/":
            raise ValueError(
                f"--from ref must be '<agent>/<store>' (path-like, one slash): {ref!r}"
            )
        body: dict[str, Any] = {
            "key": key,
            "from": ref,
            "principal": principal,
            "content": content,
        }
        if tags:
            body["tags"] = tags
        if rationale:
            body["rationale"] = rationale
        return self._check(
            self._client.post("/v0/promote", json=body)
        ).json()

    def checkpoint(self) -> dict:
        return self._check(self._client.post("/v0/checkpoint")).json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
