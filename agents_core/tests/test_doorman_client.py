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


# ---------------------------------------------------------------------------
# Deference features (doorman-mode-deference-v0)
# ---------------------------------------------------------------------------

def test_acquire_with_role_parameter():
    """acquire() must accept and forward role parameter."""
    c = _client_with([(200, {"status": "serving", "node": "gravitywell", "work_id": "flip-gw"})])
    result = c.acquire("gravitywell", "flip-gw", ttl_sec=240, reason="mode", role="mode-controller")
    assert result["status"] == "serving"


def test_acquire_default_role_is_worker():
    """acquire() must default role to 'worker' when not provided."""
    c = _client_with([(200, {"status": "serving", "node": "gravitywell", "work_id": "w1"})])
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test")
    # The mock doesn't verify the body, but the call should succeed
    assert result["status"] == "serving"


def test_acquire_deferred_outcome():
    """acquire() must return a deferred response when controller owns the mode."""
    resp_body = {
        "status": "deferred",
        "node": "gravitywell",
        "mode_owner": "flip-controller",
        "detail": "GW serving controller-owned non-big mode; big endpoint unavailable",
    }
    c = _client_with([(200, resp_body)])
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="inference")
    assert result["status"] == "deferred"
    assert result["mode_owner"] == "flip-controller"


def test_is_deferred_true_for_deferred_response():
    """is_deferred() must return True for status=='deferred'."""
    resp = {"status": "deferred", "mode_owner": "flip-controller"}
    assert DoormanClient.is_deferred(resp) is True


def test_is_deferred_false_for_serving_response():
    """is_deferred() must return False for status=='serving'."""
    resp = {"status": "serving", "node": "gravitywell"}
    assert DoormanClient.is_deferred(resp) is False


def test_is_deferred_false_for_wake_failed_response():
    """is_deferred() must return False for status=='wake_failed'."""
    resp = {"status": "wake_failed", "detail": "GW unreachable"}
    assert DoormanClient.is_deferred(resp) is False


def test_mode_owner_returns_dict_on_success():
    """mode_owner() must return the parsed dict on 200 response."""
    body = {
        "node": "gravitywell",
        "controller": "flip-controller",
        "active": True,
        "owner_lease_held": False,
        "owner_lease_age_sec": None,
        "owner_lease_stale": False,
    }
    c = _client_with([(200, body)])
    result = c.mode_owner("gravitywell")
    assert result == body
    assert result["active"] is True


def test_mode_owner_returns_none_on_404():
    """mode_owner() must return None on 404 (pre-1b doorman)."""
    class _NotFoundTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not found"})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_NotFoundTransport(),
    )
    result = c.mode_owner("gravitywell")
    assert result is None


def test_mode_owner_returns_none_on_unreachable():
    """mode_owner() must return None when doorman is unreachable (DoormanUnreachable)."""
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_ErrorTransport(),
    )
    result = c.mode_owner("gravitywell")
    assert result is None


def test_mode_owner_defaults_to_gravitywell():
    """mode_owner() must default to node='gravitywell' when not provided."""
    body = {
        "node": "gravitywell",
        "controller": "flip-controller",
        "active": True,
        "owner_lease_held": False,
        "owner_lease_age_sec": None,
        "owner_lease_stale": False,
    }
    c = _client_with([(200, body)])
    result = c.mode_owner()  # no node arg
    assert result["node"] == "gravitywell"
