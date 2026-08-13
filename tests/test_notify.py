"""Tests for agents_core.notify — CAPTURE_LOG tap on send_notification()."""
from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agents_core import notify as notify_mod


class TestCaptureEvent:
    """Tests for _capture_event — structured capture logging."""

    def test_writes_json_line_with_expected_fields(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)

        notify_mod._capture_event(
            source="host_health_watchdog",
            message="disk usage high",
            title="Alert",
            priority=notify_mod.Priority.HIGH,
            delivered=True,
            extra={"foo": "bar"},
        )

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["source"] == "host_health_watchdog"
        assert entry["title"] == "Alert"
        assert entry["message_head"] == "disk usage high"
        assert entry["priority"] == "HIGH"
        assert entry["delivered"] is True
        assert entry["extra"] == {"foo": "bar"}
        assert "ts" in entry

    def test_message_head_truncated_to_300_chars(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)

        notify_mod._capture_event(
            source="test",
            message="X" * 500,
            title="t",
            priority=notify_mod.Priority.NORMAL,
            delivered=None,
        )

        entry = json.loads(log_path.read_text())
        assert len(entry["message_head"]) == 300

    def test_extra_defaults_to_empty_dict(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)

        notify_mod._capture_event(
            source="test", message="m", title="t",
            priority=notify_mod.Priority.LOW, delivered=None,
        )

        entry = json.loads(log_path.read_text())
        assert entry["extra"] == {}

    def test_never_raises_on_write_failure(self, tmp_path, monkeypatch, caplog):
        bad_path = tmp_path / "is_a_directory"
        bad_path.mkdir()
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", bad_path)

        notify_mod._capture_event(
            source="test", message="m", title="t",
            priority=notify_mod.Priority.NORMAL, delivered=None,
        )

        assert "failed to log captured event" in caplog.text.lower()


class TestSendNotificationCapture:
    """Tests for send_notification()'s tap into CAPTURE_LOG."""

    def test_delivered_true_on_success(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "t")

        class FakeResp:
            status_code = 200

        monkeypatch.setattr(notify_mod.requests, "post", lambda *a, **k: FakeResp())

        result = notify_mod.send_notification("hello", source="my_source")

        assert result is True
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is True
        assert entry["source"] == "my_source"

    def test_delivered_false_on_missing_creds(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

        result = notify_mod.send_notification("hello")

        assert result is False
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is False
        assert entry["source"] == "unknown"

    def test_delivered_false_on_request_exception(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "t")

        def raise_exc(*a, **k):
            raise Exception("boom")

        monkeypatch.setattr(notify_mod.requests, "post", raise_exc)

        result = notify_mod.send_notification("hello")

        assert result is False
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is False

    def test_source_defaults_to_unknown(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

        notify_mod.send_notification("hello", title="t")

        entry = json.loads(log_path.read_text())
        assert entry["source"] == "unknown"

    def test_exactly_one_capture_write_per_call(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

        notify_mod.send_notification("hello")

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_capture_failure_does_not_prevent_delivery_result(self, tmp_path, monkeypatch):
        """A capture write failure must never raise out of send_notification,
        and must not affect the actual Pushover call outcome."""
        bad_path = tmp_path / "is_a_directory"
        bad_path.mkdir()
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", bad_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "t")

        class FakeResp:
            status_code = 200

        monkeypatch.setattr(notify_mod.requests, "post", lambda *a, **k: FakeResp())

        result = notify_mod.send_notification("hello")

        assert result is True

    def test_message_and_title_reflect_truncated_values(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

        long_message = "M" * (notify_mod.MAX_MESSAGE_LENGTH + 50)
        notify_mod.send_notification(long_message)

        entry = json.loads(log_path.read_text())
        assert entry["message_head"] == ("M" * (notify_mod.MAX_MESSAGE_LENGTH - 3) + "...")[:300]


class TestPytestDeliveryGuard:
    """Guard: send_notification must never deliver a real page from inside a
    pytest run, and the suppression must be visible in the audit log.

    Covers all four combinations of (PYTEST_CURRENT_TEST present/absent) x
    (credentials present/absent). requests.post is always patched with a
    Mock so the suite can assert whether it was called at all, and never
    reaches the network either way.
    """

    def test_pytest_present_creds_present_suppresses_post(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_notify.py::fake")
        monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "t")
        post = Mock()
        monkeypatch.setattr(notify_mod.requests, "post", post)

        result = notify_mod.send_notification("hello", source="my_source")

        assert result is False
        post.assert_not_called()
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is False
        assert entry["extra"] == {"suppressed": "pytest"}

    def test_pytest_present_creds_absent_suppresses_post(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_notify.py::fake")
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)
        post = Mock()
        monkeypatch.setattr(notify_mod.requests, "post", post)

        result = notify_mod.send_notification("hello")

        assert result is False
        post.assert_not_called()
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is False
        assert entry["extra"] == {"suppressed": "pytest"}

    def test_pytest_absent_creds_present_posts_as_before(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
        monkeypatch.setenv("PUSHOVER_APP_TOKEN", "t")
        post = Mock(return_value=Mock(status_code=200))
        monkeypatch.setattr(notify_mod.requests, "post", post)

        result = notify_mod.send_notification("hello", source="my_source")

        assert result is True
        post.assert_called_once()
        entry = json.loads(log_path.read_text())
        assert entry["delivered"] is True
        assert entry["extra"] == {}

    def test_pytest_absent_creds_absent_unchanged(self, tmp_path, monkeypatch):
        log_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", log_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
        monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)
        post = Mock()
        monkeypatch.setattr(notify_mod.requests, "post", post)

        result = notify_mod.send_notification("hello")

        assert result is False
        post.assert_not_called()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["delivered"] is False
        assert entry["extra"] == {}
