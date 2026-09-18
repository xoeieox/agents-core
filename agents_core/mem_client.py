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
_FROM_INVALID_RE = re.compile(r"[\n\r\x00-\x1f\x7f]")


def validate_promote_source_ref(ref: str) -> str:
    """Validate a `mem promote --from <agent-store-ref>` token.

    Returns the ref unchanged when valid; raises ValueError (loud 400 at the
    HTTP edge) when it contains a newline or any control character, or is
    empty. The shape is a path-like/ref-like token: no newlines or control
    chars, ever.
    """
    if not ref or _FROM_INVALID_RE.search(ref):
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
    ) -> dict:
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

    def checkpoint(self) -> dict:
        return self._check(self._client.post("/v0/checkpoint")).json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
