"""Dowser HTTP client — thin httpx wrapper for dowser-server.

Reads configuration from environment:
  DOWSER_SERVER         — base URL, e.g. http://127.0.0.1:8412
  DOWSER_CLIENT_TIMEOUT — per-request timeout in seconds (default 120.0)
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class DowserHTTPError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class DowserClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        self._base_url = (
            base_url or os.environ.get("DOWSER_SERVER", "http://127.0.0.1:8412")
        ).rstrip("/")
        _timeout = timeout if timeout is not None else float(
            os.environ.get("DOWSER_CLIENT_TIMEOUT", "120.0")
        )
        self._client = httpx.Client(base_url=self._base_url, timeout=_timeout)

    def _check(self, resp: httpx.Response) -> httpx.Response:
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise DowserHTTPError(resp.status_code, body)
        return resp

    def healthz(self) -> dict:
        return self._check(self._client.get("/healthz")).json()

    def read_batch(
        self,
        requests_list: list[dict],
        read_operator: str = "quest",
        budget: dict | None = None,
    ) -> dict:
        """POST /research/read-batch — run the read funnel for a list of intents.

        Returns {drafts: [{intent, findings, citations, outcome, provenance}]}.
        """
        body: dict[str, Any] = {
            "requests": requests_list,
            "read_operator": read_operator,
        }
        if budget is not None:
            body["budget"] = budget
        return self._check(
            self._client.post("/research/read-batch", json=body)
        ).json()

    def critique_batch(
        self,
        drafts: list[dict],
        critic_operator: str = "gravitywell",
    ) -> dict:
        """POST /research/critique-batch — score a list of read-batch drafts.

        Returns {verdicts: [{intent, status, verdict, diagnosis}]}.
        """
        body: dict[str, Any] = {
            "drafts": drafts,
            "critic_operator": critic_operator,
        }
        return self._check(
            self._client.post("/research/critique-batch", json=body)
        ).json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
