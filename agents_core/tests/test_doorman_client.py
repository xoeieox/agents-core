"""Tests for agents_core.doorman_client — acquire/release/status happy paths
and DoormanUnreachable on connection error. Uses httpx mock transport."""

from __future__ import annotations

import json

import httpx
import pytest

from agents_core.doorman_client import DoormanClient, DoormanUnreachable


# ---------------------------------------------------------------------------
# Mock transport helper
# ---------------------------------------------------------------------------

class _MockTransport(httpx.BaseTransport):
    def __init__(self, responses: list[tuple[int, dict]]):
        self._responses = iter(responses)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        status, body = next(self._responses)
        return httpx.Response(status, json=body)


def _client_with(responses: list[tuple[int, dict]]) -> DoormanClient:
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_MockTransport(responses),
    )
    return c


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------

def test_acquire_serving():
    c = _client_with([(200, {"status": "serving", "node": "gravitywell", "work_id": "w1"})])
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test")
    assert result["status"] == "serving"
    assert result["node"] == "gravitywell"


def test_acquire_wake_failed():
    c = _client_with([(200, {"status": "wake_failed", "detail": "GW unreachable"})])
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test")
    assert result["status"] == "wake_failed"


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

def test_release_ok():
    c = _client_with([(200, {"ok": True})])
    c.release("gravitywell", "w1")  # should not raise


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_happy_path():
    body = {
        "nodes": {
            "gravitywell": {
                "serving": True,
                "lease_count": 1,
                "leases": [{"work_id": "w1", "acquired_at": 1000.0, "ttl_sec": 120, "reason": "t"}],
                "last_wake_at": 999.0,
                "last_error": None,
            }
        }
    }
    c = _client_with([(200, body)])
    result = c.status()
    assert "nodes" in result
    assert result["nodes"]["gravitywell"]["serving"] is True


# ---------------------------------------------------------------------------
# healthz
# ---------------------------------------------------------------------------

def test_healthz_ok():
    c = _client_with([(200, {"ok": True})])
    result = c.healthz()
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# DoormanUnreachable on connection error
# ---------------------------------------------------------------------------

class _ErrorTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")


def test_acquire_unreachable_raises():
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_ErrorTransport(),
    )
    with pytest.raises(DoormanUnreachable):
        c.acquire("gravitywell", "w1", ttl_sec=120, reason="test")


def test_release_unreachable_raises():
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_ErrorTransport(),
    )
    with pytest.raises(DoormanUnreachable):
        c.release("gravitywell", "w1")


def test_status_unreachable_raises():
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_ErrorTransport(),
    )
    with pytest.raises(DoormanUnreachable):
        c.status()
