# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D4 (attestation-contract-v0, leg 1): the doorman's restore-failure page.

The 2026-09-08 incident: the doorman's WAKE_REFUSED loop ran ~40s then went
quiet while the seat stayed down (the loop's last_error was recorded, nothing
was paged). D4 adds a consecutive-restore-failure counter (per seat, in the
refresh-thread state, reset on any successful serve or restore): after 3
consecutive failures within 10 minutes, page HIGH via
agents_core.notify.send_notification - one page per failure episode (no
repetition while the streak persists).

The counter is the testable unit (the 09-08 loop shape: the same
WAKE_REFUSED reason repeated). The page body composition is tested
directly. Page hygiene (I5): the notify audit line is what the test
asserts on - under pytest send_notification suppresses the HTTP POST but
still writes the audit line with source + extra, so no test page leaves
the house.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.doorman_server import (
    _build_restore_failure_page,
    _maybe_page_restore_failure,
    _RestoreFailureStreak,
)


class TestStreakCounter:
    """The consecutive-restore-failure counter (the 09-08 loop shape)."""

    def test_streak_reaches_threshold(self):
        s = _RestoreFailureStreak()
        assert s.record_failure("drift", "render mismatch", 0.0) is False
        assert s.record_failure("drift", "render mismatch", 10.0) is False
        assert s.record_failure("drift", "render mismatch", 20.0) is True

    def test_success_resets_counter(self):
        s = _RestoreFailureStreak()
        s.record_failure("drift", "x", 0.0)
        s.record_failure("drift", "x", 10.0)
        s.record_success()
        s.record_failure("drift", "x", 100.0)
        s.record_failure("drift", "x", 110.0)
        assert s.consecutive == 2  # reset worked

    def test_streak_resets_when_window_lapses(self):
        s = _RestoreFailureStreak()
        s.record_failure("drift", "x", 0.0)
        s.record_failure("drift", "x", 10.0)
        # >10 min later: the streak does not continue (the episode is over)
        s.record_failure("drift", "x", 700.0)
        assert s.consecutive == 1

    def test_streak_resets_on_reason_change(self):
        s = _RestoreFailureStreak()
        s.record_failure("drift", "a", 0.0)
        s.record_failure("timeout", "b", 10.0)
        assert s.consecutive == 1  # new failure class, new episode

    def test_reason_string_truncated(self):
        s = _RestoreFailureStreak()
        s.record_failure("drift", "x" * 500, 0.0)
        assert len(s.last_reason) <= 80

    def test_page_fires_exactly_once_per_episode(self):
        s = _RestoreFailureStreak()
        sent = []
        s.record_failure("drift", "x", 0.0, _should_page=lambda: True,
                         _on_page=lambda: sent.append(1))
        s.record_failure("drift", "x", 10.0, _should_page=lambda: True,
                         _on_page=lambda: sent.append(2))
        assert s.record_failure("drift", "x", 20.0, _should_page=lambda: True,
                                _on_page=lambda: sent.append(3)) is True
        # streak persists: no second page
        s.record_failure("drift", "x", 30.0, _should_page=lambda: True,
                         _on_page=lambda: sent.append(4))
        assert sent == [3]

    def test_success_resets_page_dedup(self):
        s = _RestoreFailureStreak()
        sent = []
        for t in (0.0, 10.0, 20.0):
            s.record_failure("drift", "x", t, _should_page=lambda: True,
                             _on_page=lambda: sent.append(1))
        s.record_success()
        sent.clear()
        # a fresh episode pages again
        for t in (100.0, 110.0, 120.0):
            s.record_failure("drift", "x", t, _should_page=lambda: True,
                             _on_page=lambda: sent.append(1))
        assert sent == [1]

    def test_no_page_when_should_page_false(self):
        s = _RestoreFailureStreak()
        sent = []
        for t in (0.0, 10.0, 20.0):
            s.record_failure("drift", "x", t, _should_page=lambda: False,
                             _on_page=lambda: sent.append(1))
        assert sent == []
        # the episode is still paged-once: recovery resets, the next
        # episode pages
        s.record_success()
        sent.clear()
        for t in (100.0, 110.0, 120.0):
            s.record_failure("drift", "x", t, _should_page=lambda: False,
                             _on_page=lambda: sent.append(1))
        assert sent == []


class TestPageBody:
    """The page content: reason + streak + manual-restore command (I1)."""

    def test_page_body_names_reason_streak_and_command(self):
        body = _build_restore_failure_page(
            seat_id="gw-slot1", reason="drift", detail="render mismatch",
            streak=3, last_ts=1700000000.0,
        )
        assert "gw-slot1" in body
        assert "drift" in body
        assert "render mismatch" in body
        assert "3" in body  # streak length
        # the one-line manual-restore command
        assert "gw-topology" in body

    def test_page_body_truncates_long_detail(self):
        body = _build_restore_failure_page(
            seat_id="gw-slot1", reason="drift", detail="x" * 500,
            streak=3, last_ts=0.0,
        )
        # the detail line is capped (the page is a pushover, not a log)
        assert "x" * 300 not in body


class TestMaybePageWiring:
    """_maybe_page_restore_failure: the seam the wake-failure path calls."""

    def test_pages_via_notify_audit_line(self, tmp_path: Path, monkeypatch):
        """The page lands on the named human-visible surface (I1): the
        notify audit line carries source + the reason (the test asserts on
        the audit line, so no test page leaves the house)."""
        import agents_core.notify as notify_mod

        audit_log = tmp_path / "audit.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", audit_log)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        sent = []

        def _fake_send(message, *, source="", title="", priority=None,
                       extra=None, **kw):
            sent.append(message)
            # write the audit line (the real send_notification does this via _capture_event)
            notify_mod._capture_event(
                source=source,
                message=message,
                title=title,
                priority=notify_mod.Priority.HIGH if priority is None else priority,
                delivered=True,
                extra=extra,
            )
            return True

        monkeypatch.setattr(notify_mod, "send_notification", _fake_send)

        paged = _maybe_page_restore_failure(
            reason="drift", detail="render mismatch",
            streak=3, last_ts=0.0, seat_id="gw-slot1",
        )
        assert paged is True
        assert len(sent) == 1
        assert "gw-slot1" in sent[0]
        assert "drift" in sent[0]

        # the audit line: source pinned (test 14, I5 hygiene)
        lines = audit_log.read_text().strip().split("\n")
        entry = json.loads(lines[-1])
        assert entry["source"] == "lapis-pm-doorman"
        assert "drift" in entry["message_head"]

    def test_never_raises(self, monkeypatch):
        import agents_core.notify as notify_mod

        def _boom(*a, **k):
            raise RuntimeError("pushover down")

        monkeypatch.setattr(notify_mod, "send_notification", _boom)
        # page-only, never raises into the wake path
        assert _maybe_page_restore_failure(
            reason="drift", detail="x", streak=3, last_ts=0.0,
            seat_id="gw-slot1",
        ) is False
