"""mem HTTP client — thin httpx wrapper for mem-server.

Reads configuration from environment:
  MEM_SERVER          — base URL, e.g. http://100.x.y.z:8403
  MEM_BEARER_TOKEN    — optional bearer token (must match server)
  MEM_CLIENT_TIMEOUT  — per-request timeout in seconds (default 5.0)

Raises MemHTTPError on non-2xx responses.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


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
    ):
        self._base_url = (base_url or os.environ.get("MEM_SERVER", "")).rstrip("/")
        _token = token if token is not None else os.environ.get("MEM_BEARER_TOKEN", "")
        _timeout = timeout if timeout is not None else float(
            os.environ.get("MEM_CLIENT_TIMEOUT", "5.0")
        )
        headers = {}
        if _token:
            headers["Authorization"] = f"Bearer {_token}"
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

    # --- Hygiene (mem-hygiene-automation-v0) ---
    # The server holds the store; these are the thin-client half of the
    # `mem hygiene run|ageout|list|restore` subcommands. The
    # MEM_SERVER-unset direct-sqlite fallback in the conductor CLI is a
    # named NON-PATH for hygiene.

    def hygiene_run(
        self,
        *,
        dry_run: bool = False,
        allow_over_cap: bool = False,
        run_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"dry_run": dry_run, "allow_over_cap": allow_over_cap}
        if run_id:
            body["run_id"] = run_id
        return self._check(
            self._client.post("/v0/hygiene/run", json=body)
        ).json()

    def hygiene_ageout(self, window: int | None = None) -> dict:
        body: dict[str, Any] = {}
        if window is not None:
            body["window"] = window
        return self._check(
            self._client.post("/v0/hygiene/ageout", json=body)
        ).json()

    def hygiene_list(self) -> dict:
        return self._check(self._client.get("/v0/hygiene/list")).json()

    def hygiene_restore(self, prefix: str) -> dict:
        return self._check(
            self._client.post("/v0/hygiene/restore", json={"prefix": prefix})
        ).json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
