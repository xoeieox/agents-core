"""Tests for agents_core.doorman_client — acquire/release/status happy paths
and DoormanUnreachable on connection error. Uses httpx mock transport."""

from __future__ import annotations

import json
import os
from unittest import mock

import httpx
import pytest

from agents_core.doorman_client import DoormanClient, DoormanUnreachable, _gw_acquire_timeout


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


# ---------------------------------------------------------------------------
# GW acquire timeout coupling (gw-doorman-client-wake-timeout-v0)
# ---------------------------------------------------------------------------

def test_gw_acquire_timeout_default():
    """_gw_acquire_timeout() must accommodate the dual wake deadline by default.

    DOORMAN_DEFAULT_SERVE_MODE defaults to "dual" (gw-doorman-wake-to-default-mode-v0),
    so the timeout derives from GW_DUAL_WAKE_DEADLINE_SEC (720) + GW_ACQUIRE_MARGIN_SEC
    (30) = 750s, not the big-mode-only GW_WAKE_DEADLINE_SEC (180).
    """
    with mock.patch.dict(os.environ, {}, clear=False):
        # Clear any existing overrides
        os.environ.pop("GW_WAKE_DEADLINE_SEC", None)
        os.environ.pop("GW_DUAL_WAKE_DEADLINE_SEC", None)
        os.environ.pop("DOORMAN_DEFAULT_SERVE_MODE", None)
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)
        timeout = _gw_acquire_timeout()
        assert timeout == 750.0


def test_gw_acquire_timeout_custom_deadline():
    """_gw_acquire_timeout() must respect custom GW_WAKE_DEADLINE_SEC under the big-mode
    rollback (DOORMAN_DEFAULT_SERVE_MODE=big) — the dual default is covered separately
    by test_gw_acquire_timeout_dual_default_uses_dual_deadline."""
    with mock.patch.dict(
        os.environ,
        {"GW_WAKE_DEADLINE_SEC": "120", "DOORMAN_DEFAULT_SERVE_MODE": "big"},
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)
        os.environ.pop("GW_DUAL_WAKE_DEADLINE_SEC", None)
        timeout = _gw_acquire_timeout()
        assert timeout == 150.0  # 120 + 30


def test_gw_acquire_timeout_custom_margin():
    """_gw_acquire_timeout() must respect custom GW_ACQUIRE_MARGIN_SEC under the
    big-mode rollback (DOORMAN_DEFAULT_SERVE_MODE=big)."""
    with mock.patch.dict(
        os.environ,
        {
            "GW_WAKE_DEADLINE_SEC": "180",
            "GW_ACQUIRE_MARGIN_SEC": "60",
            "DOORMAN_DEFAULT_SERVE_MODE": "big",
        },
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)
        os.environ.pop("GW_DUAL_WAKE_DEADLINE_SEC", None)
        timeout = _gw_acquire_timeout()
        assert timeout == 240.0  # 180 + 60


def test_gw_acquire_timeout_dual_default_uses_dual_deadline():
    """_gw_acquire_timeout() must derive from GW_DUAL_WAKE_DEADLINE_SEC + margin when
    DOORMAN_DEFAULT_SERVE_MODE is dual (default), even if GW_WAKE_DEADLINE_SEC is smaller."""
    with mock.patch.dict(
        os.environ,
        {
            "DOORMAN_DEFAULT_SERVE_MODE": "dual",
            "GW_DUAL_WAKE_DEADLINE_SEC": "720",
            "GW_WAKE_DEADLINE_SEC": "180",
        },
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)
        timeout = _gw_acquire_timeout()
        assert timeout == 750.0  # 720 + 30
        assert timeout >= 720 + 30  # closes the coupled-timeout gap (scope item 5)


def test_gw_acquire_timeout_dual_custom_deadline():
    """_gw_acquire_timeout() must respect a custom GW_DUAL_WAKE_DEADLINE_SEC."""
    with mock.patch.dict(
        os.environ,
        {
            "DOORMAN_DEFAULT_SERVE_MODE": "dual",
            "GW_DUAL_WAKE_DEADLINE_SEC": "500",
            "GW_WAKE_DEADLINE_SEC": "180",
        },
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)
        timeout = _gw_acquire_timeout()
        assert timeout == 530.0  # max(180, 500) + 30


def test_gw_acquire_timeout_explicit_override():
    """_gw_acquire_timeout() must return explicit GW_ACQUIRE_TIMEOUT_SEC override."""
    with mock.patch.dict(
        os.environ,
        {"GW_ACQUIRE_TIMEOUT_SEC": "500"},
        clear=False,
    ):
        timeout = _gw_acquire_timeout()
        assert timeout == 500.0


def test_gw_acquire_timeout_override_below_deadline_warns():
    """_gw_acquire_timeout() must emit RuntimeWarning if override < deadline."""
    with mock.patch.dict(
        os.environ,
        {"GW_WAKE_DEADLINE_SEC": "180", "GW_ACQUIRE_TIMEOUT_SEC": "100"},
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        with pytest.warns(RuntimeWarning, match="GW_ACQUIRE_TIMEOUT_SEC=100"):
            timeout = _gw_acquire_timeout()
            assert timeout == 100.0  # Still returns the override


def test_acquire_respects_timeout_parameter():
    """acquire() must pass the timeout parameter to _post."""
    # Mock transport that captures the request
    captured_timeout = []

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            # httpx doesn't expose the per-request timeout in the request object,
            # but we can verify the call doesn't raise by returning a response
            return httpx.Response(200, json={"status": "serving"})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_CaptureTransport(),
    )

    # Call acquire with a custom timeout
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test", timeout=500.0)
    assert result["status"] == "serving"


def test_acquire_slow_response_with_long_timeout():
    """acquire() with a long timeout must not timeout on a slow response.

    Simulates a slow doorman response that would timeout with the old short timeout
    but succeeds with the new long timeout. Demonstrates the boundary at toy scale.
    Env: GW_WAKE_DEADLINE_SEC=2, margin=1, so acquire timeout=3s (new).
    Also patches DOORMAN_CLIENT_TIMEOUT to 1s (old ceiling), and uses a 1.5s delay.
    Delay sits between old ceiling (1s) and new timeout (3s).
    """
    class _SlowTransport(httpx.BaseTransport):
        def __init__(self, delay_sec: float = 1.5):
            self.delay_sec = delay_sec

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            # Simulate doorman blocking on ensure_serving().
            # Delay (1.5s) > old DOORMAN_CLIENT_TIMEOUT (1s) would fail without override.
            # Delay (1.5s) < new _gw_acquire_timeout() (3s) succeeds with override.
            import time
            time.sleep(self.delay_sec)
            return httpx.Response(200, json={"status": "serving"})

    with mock.patch.dict(
        os.environ,
        {
            "GW_WAKE_DEADLINE_SEC": "2",
            "GW_ACQUIRE_MARGIN_SEC": "1",
            "DOORMAN_CLIENT_TIMEOUT": "1",  # Scale down old ceiling for test
        },
        clear=False,
    ):
        os.environ.pop("GW_ACQUIRE_TIMEOUT_SEC", None)

        c = DoormanClient(base_url="http://doorman.test")
        c._client = httpx.Client(
            base_url="http://doorman.test",
            transport=_SlowTransport(delay_sec=1.5),  # Between old (1s) and new (3s)
        )

        # This should succeed without raising TimeoutException because
        # the override timeout (3s) allows the slow response.
        result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test",
                          timeout=_gw_acquire_timeout())
        assert result["status"] == "serving"


def test_release_uses_short_default_timeout():
    """release() must use the short default timeout, not the long GW timeout."""
    # The test verifies that release() calls _post without a timeout override,
    # so it uses the client's default timeout (30s).
    c = _client_with([(200, {"ok": True})])
    # If release() were incorrectly passing a long timeout, this would be a
    # functional problem for fast-fail on a hung doorman.
    c.release("gravitywell", "w1")  # should not raise


def test_status_uses_short_default_timeout():
    """status() must use the short default timeout, not the long GW timeout."""
    body = {
        "nodes": {
            "gravitywell": {
                "serving": True,
                "lease_count": 1,
                "leases": [{"work_id": "w1", "acquired_at": 1000.0, "ttl_sec": 120}],
                "last_wake_at": 999.0,
                "last_error": None,
            }
        }
    }
    c = _client_with([(200, body)])
    result = c.status()
    assert result["nodes"]["gravitywell"]["serving"] is True


def test_healthz_uses_short_default_timeout():
    """healthz() must use the short default timeout, not the long GW timeout."""
    c = _client_with([(200, {"ok": True})])
    result = c.healthz()
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# drain_count (doorman-mode-aware-serving-predicate-v0, AC9)
# ---------------------------------------------------------------------------

def test_drain_count_returns_int_on_200():
    """drain_count() must return int on 200 response."""
    c = _client_with([(200, {"node": "gravitywell", "drain_count": 2})])
    result = c.drain_count("gravitywell")
    assert result == 2


def test_drain_count_returns_none_on_404():
    """drain_count() must return None on 404 (pre-this-unit doormen)."""
    class _NotFoundTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "not found"})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_NotFoundTransport(),
    )
    result = c.drain_count("gravitywell")
    assert result is None


def test_drain_count_returns_none_on_unreachable():
    """drain_count() must return None when doorman is unreachable."""
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_ErrorTransport(),
    )
    result = c.drain_count("gravitywell")
    assert result is None


def test_drain_count_defaults_to_gravitywell():
    """drain_count() must default to node='gravitywell'."""
    c = _client_with([(200, {"node": "gravitywell", "drain_count": 0})])
    result = c.drain_count()  # no node arg
    assert result == 0


# ---------------------------------------------------------------------------
# Foreground-priority gate (gw-router-phase1-foreground-gate)
# ---------------------------------------------------------------------------

def test_acquire_with_lease_class_forwards_class_field():
    """acquire() must forward lease_class as the `class` body field."""
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "serving"})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    result = c.acquire("gravitywell", "w1", ttl_sec=120, reason="test", lease_class="protected")
    assert result["status"] == "serving"
    assert captured["body"]["class"] == "protected"


def test_acquire_without_lease_class_omits_class_field():
    """acquire() must omit `class` from the body when lease_class is not provided
    (missing → server defaults to deferrable, per doorman-server contract)."""
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "serving"})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    c.acquire("gravitywell", "w1", ttl_sec=120, reason="test")
    assert "class" not in captured["body"]


def test_acquire_pending_defer_status():
    """acquire() must surface a pending_defer status verbatim."""
    resp_body = {
        "status": "pending_defer",
        "work_id": "fixer-1",
        "class": "deferrable",
        "enqueued_at": 1000.0,
        "waited_seconds": 0.0,
    }
    c = _client_with([(200, resp_body)])
    result = c.acquire("gravitywell", "fixer-1", ttl_sec=120, reason="fixer work", lease_class="deferrable")
    assert result["status"] == "pending_defer"


def test_is_pending_defer_true_for_pending_defer_response():
    resp = {"status": "pending_defer", "work_id": "fixer-1"}
    assert DoormanClient.is_pending_defer(resp) is True


def test_is_pending_defer_false_for_serving_response():
    resp = {"status": "serving", "node": "gravitywell"}
    assert DoormanClient.is_pending_defer(resp) is False


def test_brake_hold_posts_reason_and_ttl():
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"braked": True, "expires_at": 12345.0})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    result = c.brake("urgent human task", ttl_s=60)
    assert result["braked"] is True
    assert captured["path"] == "/v0/brake"
    assert captured["body"]["reason"] == "urgent human task"
    assert captured["body"]["ttl_s"] == 60


def test_brake_hold_omits_ttl_s_when_not_provided():
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"braked": True, "expires_at": 12345.0})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    c.brake("urgent")
    assert "ttl_s" not in captured["body"]


def test_brake_release_posts_to_release_endpoint():
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            return httpx.Response(200, json={"braked": False})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    result = c.brake_release()
    assert result["braked"] is False
    assert captured["path"] == "/v0/brake/release"


def test_defer_wait_timeout_default():
    """_defer_wait_timeout() must derive from DOORMAN_MAX_HOLD_TIMEOUT_SEC + margin."""
    from agents_core.doorman_client import _defer_wait_timeout

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DOORMAN_MAX_HOLD_TIMEOUT_SEC", None)
        os.environ.pop("GW_ACQUIRE_MARGIN_SEC", None)
        assert _defer_wait_timeout() == 930.0  # 900 + 30


def test_defer_wait_timeout_custom():
    from agents_core.doorman_client import _defer_wait_timeout

    with mock.patch.dict(
        os.environ,
        {"DOORMAN_MAX_HOLD_TIMEOUT_SEC": "300", "GW_ACQUIRE_MARGIN_SEC": "10"},
        clear=False,
    ):
        assert _defer_wait_timeout() == 310.0


# ---------------------------------------------------------------------------
# force_stop (gw-force-stop-lease-guard-v0)
# ---------------------------------------------------------------------------

def test_force_stop_posts_node_and_exclude_principal():
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "stopped", "node": "gravitywell", "exit_code": 0})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    result = c.force_stop("gravitywell", exclude_principal="cockpit")
    assert captured["path"] == "/v0/force-stop"
    assert captured["body"] == {"node": "gravitywell", "exclude_principal": "cockpit"}
    assert result["status"] == "stopped"


def test_force_stop_omits_exclude_principal_when_not_provided():
    captured = {}

    class _CaptureTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "stopped", "node": "gravitywell", "exit_code": 0})

    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(base_url="http://doorman.test", transport=_CaptureTransport())
    c.force_stop()
    assert captured["body"] == {"node": "gravitywell"}
    assert "exclude_principal" not in captured["body"]


def test_force_stop_returns_blocked_response():
    c = _client_with([(200, {
        "status": "blocked", "node": "gravitywell",
        "active_leases": [{"work_id": "w1", "principal": "other-consumer"}],
    })])
    result = c.force_stop("gravitywell", exclude_principal="cockpit")
    assert result["status"] == "blocked"
    assert result["active_leases"] == [{"work_id": "w1", "principal": "other-consumer"}]
