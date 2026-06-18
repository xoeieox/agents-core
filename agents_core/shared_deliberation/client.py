"""Async HTTP client for the shared deliberation service."""

from __future__ import annotations

import os
from typing import Optional
import httpx

from agents_core.shared_deliberation.envelope import DeliberationRequest, DeliberationEnvelope


class SharedDeliberationClient:
    """Ship-and-return async client for remote callers."""

    def __init__(self, base_url: str = "http://127.0.0.1:8409", bearer_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def deliberate(self, request: DeliberationRequest) -> DeliberationEnvelope:
        """Submit a deliberation request. Returns DeliberationEnvelope."""
        if self._client is None:
            self._client = httpx.AsyncClient()

        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        timeout = float(os.environ.get("SHARED_DELIBERATION_CLIENT_TIMEOUT_S", "2400.0"))
        response = await self._client.post(
            f"{self.base_url}/v0/deliberate",
            json=request.to_dict(),
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return DeliberationEnvelope(**data)

    async def health(self) -> dict:
        """Check service health."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        response = await self._client.get(f"{self.base_url}/v0/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
