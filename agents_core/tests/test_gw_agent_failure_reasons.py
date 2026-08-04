"""Unit tests for gw_agent's per-step POST failure classification + bounded retry.

Covers agents-core-gw-agent-failure-reasons-bounded-retry-v0 (D1-D3):
- distinct, typed reason strings at the per-step POST site (D1)
- bounded in-step retry for transient classes only, governed by the
  _per_step_timeout envelope (D2)
- provenance: retries/final classification are logged, reason_out unchanged (D3)

All fixtured/mocked - no live GW traffic.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from agents_core.gw_agent import (
    call_gw_agent,
    GW_REASON_RATE_LIMITED,
    GW_REASON_BACKEND_UNREACHABLE,
    GW_REASON_REQUEST_TIMEOUT,
    GW_REASON_SERVER_ERROR,
    GW_REASON_REQUEST_FAILED,
    GW_REASON_NO_CHOICES,
    GW_TRANSIENT_REASONS,
    GW_STEP_MAX_RETRIES,
    _classify_response,
    _classify_exception,
    _post_step_with_bounded_retry,
)


def _http_error_resp(status_code: int, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(status_code=status_code, headers=headers or {})
    return resp


def _mock_success_resp(data: dict) -> MagicMock:
    r = MagicMock(status_code=200)
    r.json = MagicMock(return_value=data)
    r.raise_for_status = MagicMock()
    return r


class TestClassifyResponse:
    def test_429_with_retry_after_captured(self):
        resp = _http_error_resp(429, {"Retry-After": "5"})
        reason, retry_after = _classify_response(resp)
        assert reason == GW_REASON_RATE_LIMITED
        assert retry_after == 5.0

    def test_429_without_retry_after_header(self):
        resp = _http_error_resp(429, {})
        reason, retry_after = _classify_response(resp)
        assert reason == GW_REASON_RATE_LIMITED
        assert retry_after is None

    def test_429_with_malformed_retry_after_ignored(self):
        resp = _http_error_resp(429, {"Retry-After": "not-a-number"})
        reason, retry_after = _classify_response(resp)
        assert reason == GW_REASON_RATE_LIMITED
        assert retry_after is None

    def test_5xx_is_server_error(self):
        for status in (500, 502, 503):
            reason, retry_after = _classify_response(_http_error_resp(status))
            assert reason == GW_REASON_SERVER_ERROR
            assert retry_after is None

    def test_other_4xx_falls_to_request_failed(self):
        reason, retry_after = _classify_response(_http_error_resp(404))
        assert reason == GW_REASON_REQUEST_FAILED
        assert retry_after is None


class TestClassifyException:
    def test_connection_error_is_backend_unreachable(self):
        reason, retry_after = _classify_exception(requests.exceptions.ConnectionError("boom"))
        assert reason == GW_REASON_BACKEND_UNREACHABLE
        assert retry_after is None

    def test_timeout_is_request_timeout(self):
        reason, retry_after = _classify_exception(requests.exceptions.Timeout("slow"))
        assert reason == GW_REASON_REQUEST_TIMEOUT
        assert retry_after is None

    def test_connect_timeout_classifies_as_timeout_not_unreachable(self):
        # ConnectTimeout subclasses both ConnectionError and Timeout - Timeout wins.
        reason, retry_after = _classify_exception(requests.exceptions.ConnectTimeout("slow connect"))
        assert reason == GW_REASON_REQUEST_TIMEOUT

    def test_http_error_delegates_to_classify_response(self):
        exc = requests.exceptions.HTTPError(response=_http_error_resp(503))
        reason, retry_after = _classify_exception(exc)
        assert reason == GW_REASON_SERVER_ERROR

    def test_unclassified_exception_falls_to_request_failed(self):
        reason, retry_after = _classify_exception(RemoteDisconnectedLike("connection reset"))
        assert reason == GW_REASON_REQUEST_FAILED


class RemoteDisconnectedLike(Exception):
    """A generic exception with no requests.exceptions lineage - unclassified fallback."""


class TestPostStepWithBoundedRetry:
    """Direct unit tests of the isolated retry helper."""

    def test_transient_class_retries_at_most_twice_then_returns_classified_reason(self):
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            mock_post.side_effect = requests.exceptions.ConnectionError("down")

            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert data is None
            assert reason == GW_REASON_BACKEND_UNREACHABLE
            assert mock_post.call_count == GW_STEP_MAX_RETRIES + 1  # 1 initial + 2 retries
            assert mock_sleep.call_count == GW_STEP_MAX_RETRIES

    def test_non_transient_returns_immediately_no_retry(self):
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            resp = _http_error_resp(404)
            mock_post.return_value = resp
            resp.raise_for_status = MagicMock(
                side_effect=requests.exceptions.HTTPError(response=resp)
            )

            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert data is None
            assert reason == GW_REASON_REQUEST_FAILED
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()

    def test_unclassified_exception_returns_request_failed_no_retry(self):
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            mock_post.side_effect = RemoteDisconnectedLike("reset")

            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert data is None
            assert reason == GW_REASON_REQUEST_FAILED
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()

    def test_succeeds_after_transient_retries(self):
        success_data = {"choices": [{"message": {"content": "ok"}}]}
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            mock_post.side_effect = [
                requests.exceptions.ConnectionError("down"),
                _mock_success_resp(success_data),
            ]

            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert reason is None
            assert data == success_data
            assert mock_post.call_count == 2
            assert mock_sleep.call_count == 1

    def test_429_honors_retry_after_capped_at_30s(self):
        resp = _http_error_resp(429, {"Retry-After": "9999"})
        resp.raise_for_status = MagicMock(side_effect=requests.exceptions.HTTPError(response=resp))
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            mock_post.return_value = resp

            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert reason == GW_REASON_RATE_LIMITED
            # Every sleep call must be capped at 30s despite the 9999s header.
            for call in mock_sleep.call_args_list:
                assert call.args[0] <= 30.0

    def test_retry_never_busy_waits_past_per_step_timeout_envelope(self):
        """A wait that would exceed the remaining budget aborts instead of sleeping."""
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep, \
             patch("agents_core.gw_agent.time.monotonic", return_value=999.5):
            mock_post.side_effect = requests.exceptions.ConnectionError("down")

            # deadline is 1000.0, "now" is already 999.5 -> only 0.5s of budget left,
            # far less than the 1s+ backoff the first retry would need.
            data, reason = _post_step_with_bounded_retry(
                "http://gw-test", {}, now=999.5, deadline=1000.0,
                conclusion_reserve_s=0.0, log=None, step_num=0,
            )

            assert data is None
            assert reason == GW_REASON_BACKEND_UNREACHABLE
            # Aborted on the first failure without ever sleeping past the envelope.
            mock_sleep.assert_not_called()
            assert mock_post.call_count == 1

    def test_provenance_logged_for_each_retry_and_final_reason(self):
        logged = []
        with patch("agents_core.gw_agent.requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep"), \
             patch("agents_core.gw_agent.time.monotonic", return_value=0.0):
            mock_post.side_effect = requests.exceptions.ConnectionError("down")

            _post_step_with_bounded_retry(
                "http://gw-test", {}, now=0.0, deadline=1000.0,
                conclusion_reserve_s=0.0, log=logged.append, step_num=0,
            )

            joined = "\n".join(logged)
            assert "backend_unreachable" in joined
            assert any("retrying" in line for line in logged)


class TestCallGwAgentIntegrationReasonOut:
    """End-to-end through call_gw_agent(reason_out=...) - backward-compat fence."""

    def test_unclassified_exception_still_reports_request_failed(self):
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.side_effect = RemoteDisconnectedLike("reset")

            reason_out = []
            result = call_gw_agent(prompt="Review.", reason_out=reason_out, timeout=10)

            assert result is None or result == ""
            assert reason_out == [GW_REASON_REQUEST_FAILED]
            mock_post.assert_called_once()

    def test_no_choices_reason_unchanged(self):
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = {"choices": []}
            mock_post.return_value.raise_for_status = MagicMock()

            reason_out = []
            call_gw_agent(prompt="Review.", reason_out=reason_out, timeout=10)

            assert reason_out == [GW_REASON_NO_CHOICES]

    def test_connection_error_classified_and_retried_then_reported(self):
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post, \
             patch("agents_core.gw_agent.time.sleep") as mock_sleep:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            mock_post.side_effect = requests.exceptions.ConnectionError("down")

            reason_out = []
            call_gw_agent(prompt="Review.", reason_out=reason_out, timeout=60)

            assert reason_out == [GW_REASON_BACKEND_UNREACHABLE]
            # 1 initial attempt + GW_STEP_MAX_RETRIES retries.
            assert mock_post.call_count == GW_STEP_MAX_RETRIES + 1
            assert mock_sleep.call_count == GW_STEP_MAX_RETRIES
