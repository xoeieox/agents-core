"""Tests for notify_policy — per-shape silencing of routine execution telemetry."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from agents_core import claude_queue_runner as runner_mod


class TestFailureClass:
    """Tests for _failure_class — failure classification."""

    def test_failure_class_error_prefix(self):
        """ERROR: on first line → infra."""
        result = "ERROR: worktree_setup: some error\nmore lines"
        assert runner_mod._failure_class(result) == "infra"

    def test_failure_class_exit_prefix(self):
        """EXIT on first line → execution."""
        result = "EXIT 1:\nstderr output"
        assert runner_mod._failure_class(result) == "execution"

    def test_failure_class_timeout_prefix(self):
        """TIMEOUT: on first line → execution."""
        result = "TIMEOUT: exceeded 300s\nmore"
        assert runner_mod._failure_class(result) == "execution"

    def test_failure_class_interrupted_prefix(self):
        """INTERRUPTED on first line → execution."""
        result = "INTERRUPTED by signal 15:\nstuff"
        assert runner_mod._failure_class(result) == "execution"

    def test_failure_class_unknown_prefix(self):
        """Unknown first line → infra (conservative)."""
        result = "UNKNOWN: something\nmore"
        assert runner_mod._failure_class(result) == "infra"

    def test_failure_class_empty_string(self):
        """Empty string → infra (no IndexError)."""
        result = ""
        assert runner_mod._failure_class(result) == "infra"

    def test_failure_class_whitespace_only(self):
        """Whitespace-only string → infra."""
        result = "   \n  \n"
        assert runner_mod._failure_class(result) == "infra"

    def test_failure_class_first_line_only(self):
        """ERROR on line 2 must NOT match (first line only discipline)."""
        result = "EXIT 1:\nERROR: this should not match"
        assert runner_mod._failure_class(result) == "execution"

    def test_failure_class_timeout_on_second_line_ignored(self):
        """TIMEOUT on line 2 must NOT match."""
        result = "EXIT 5:\nTIMEOUT: this is line 2"
        assert runner_mod._failure_class(result) == "execution"


class TestLogSilenced:
    """Tests for _log_silenced — silenced-event logging."""

    def test_log_silenced_writes_json_line(self, tmp_path, monkeypatch):
        """A silenced failure event appends one JSON line with expected fields."""
        log_path = tmp_path / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        task = {"id": "task123", "description": "fixer:target_x", "task_type": "subprocess"}
        runner_mod._log_silenced(
            "failure",
            task,
            failure_class="execution",
            demoted_from="HIGH",
            result="EXIT 1:\nsome stderr",
        )

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["task_id"] == "task123"
        assert entry["description"] == "fixer:target_x"
        assert entry["event"] == "failure"
        assert entry["failure_class"] == "execution"
        assert entry["demoted_from"] == "HIGH"
        assert entry["result_head"] == "EXIT 1:"
        assert entry["source"] == "claude_queue_runner"
        assert "ts" in entry

    def test_log_silenced_completion_no_failure_class(self, tmp_path, monkeypatch):
        """A silenced completion has failure_class=None."""
        log_path = tmp_path / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        task = {"id": "task456", "description": "reviewer:pr123"}
        runner_mod._log_silenced(
            "completion",
            task,
            failure_class=None,
            demoted_from="NORMAL",
            result="",
        )

        entry = json.loads(log_path.read_text())
        assert entry["event"] == "completion"
        assert entry["failure_class"] is None
        assert entry["demoted_from"] == "NORMAL"

    def test_log_silenced_truncates_result_head(self, tmp_path, monkeypatch):
        """result_head is truncated to 300 chars."""
        log_path = tmp_path / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        task = {"id": "task789", "description": "test"}
        long_result = "X" * 500 + "\nmore"
        runner_mod._log_silenced(
            "failure",
            task,
            failure_class="execution",
            demoted_from="HIGH",
            result=long_result,
        )

        entry = json.loads(log_path.read_text())
        assert len(entry["result_head"]) == 300
        assert entry["result_head"] == "X" * 300

    def test_log_silenced_creates_parent_dir(self, tmp_path, monkeypatch):
        """Parent directory is created on first write."""
        log_dir = tmp_path / "nested" / "dir"
        log_path = log_dir / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        assert not log_dir.exists()

        task = {"id": "task_new_dir"}
        runner_mod._log_silenced(
            "failure", task, failure_class="infra",
            demoted_from="HIGH", result="ERROR: test"
        )

        assert log_path.exists()

    def test_log_silenced_never_raises_on_error(self, tmp_path, monkeypatch, caplog):
        """Logging failures are swallowed — never raise, never block push."""
        # Make SILENCED_LOG a path we can't write to (a directory).
        bad_path = tmp_path / "is_a_directory"
        bad_path.mkdir()
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", bad_path)

        task = {"id": "task_fail_write"}
        # Should not raise.
        runner_mod._log_silenced(
            "failure", task, failure_class="execution",
            demoted_from="HIGH", result="test"
        )

        # A warning should be logged.
        assert "failed to log silenced event" in caplog.text.lower()

    def test_log_silenced_also_writes_capture_log(self, tmp_path, monkeypatch):
        """_log_silenced additionally writes a CAPTURE_LOG entry (AC3)."""
        silenced_path = tmp_path / "silenced.jsonl"
        captured_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", silenced_path)
        import agents_core.notify as notify_mod
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", captured_path)

        task = {"id": "task_both_logs", "description": "fixer:target_x"}
        runner_mod._log_silenced(
            "failure",
            task,
            failure_class="execution",
            demoted_from="HIGH",
            result="EXIT 1:\nsome stderr",
        )

        # Existing SILENCED_LOG write is unaffected.
        silenced_entry = json.loads(silenced_path.read_text())
        assert silenced_entry["task_id"] == "task_both_logs"

        # New CAPTURE_LOG write mirrors the same task_id/event/failure_class.
        captured_entry = json.loads(captured_path.read_text())
        assert captured_entry["source"] == "claude_queue_runner"
        assert captured_entry["extra"]["task_id"] == "task_both_logs"
        assert captured_entry["extra"]["event"] == "failure"
        assert captured_entry["extra"]["failure_class"] == "execution"
        assert captured_entry["extra"]["demoted_from"] == "HIGH"
        assert captured_entry["delivered"] is None
        assert captured_entry["priority"] == "HIGH"


class TestNotifyFailure:
    """Tests for notify_failure with notify_policy."""

    def test_notify_failure_always_policy_sends_high(self, monkeypatch):
        """Policy 'always' → send HIGH for any failure."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"message": message, "title": title, "priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": True,
            "notify_policy": "always",
            "description": "fixer:test",
        }
        runner_mod.notify_failure(task, "EXIT 1:\nstderr")

        assert len(sent) == 1
        assert sent[0]["priority"].name == "HIGH"

    def test_notify_failure_infra_only_infra_failure_sends_high(self, monkeypatch):
        """Policy 'infra-only' + infra failure → send HIGH."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": True,
            "notify_policy": "infra-only",
            "description": "fixer:test",
        }
        runner_mod.notify_failure(task, "ERROR: worktree_setup: something")

        assert len(sent) == 1
        assert sent[0]["priority"].name == "HIGH"

    def test_notify_failure_infra_only_execution_failure_no_send(
        self, monkeypatch, tmp_path
    ):
        """Policy 'infra-only' + execution failure → no send, log silenced."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        log_path = tmp_path / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        task = {
            "notify": True,
            "notify_policy": "infra-only",
            "id": "task_exec_fail",
            "description": "reviewer:test",
        }
        runner_mod.notify_failure(task, "EXIT 1:\nstderr")

        assert len(sent) == 0
        entry = json.loads(log_path.read_text())
        assert entry["event"] == "failure"
        assert entry["failure_class"] == "execution"
        assert entry["demoted_from"] == "HIGH"

    def test_notify_failure_missing_notify_policy_defaults_always(self, monkeypatch):
        """Missing notify_policy key → defaults to 'always' → HIGH."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": True,
            # no notify_policy key
            "description": "test",
        }
        runner_mod.notify_failure(task, "EXIT 1:\ntest")

        assert len(sent) == 1
        assert sent[0]["priority"].name == "HIGH"

    def test_notify_failure_disabled_respects_notify_false(self, monkeypatch):
        """If notify=False, neither send nor log (regardless of policy)."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": False,
            "notify_policy": "infra-only",
            "description": "test",
        }
        runner_mod.notify_failure(task, "EXIT 1:\ntest")

        assert len(sent) == 0

    def test_notify_failure_disabled_still_writes_capture_log(self, tmp_path, monkeypatch):
        """notify=False produces a CAPTURE_LOG entry with delivered=null (AC2)."""
        import agents_core.notify as notify_mod
        captured_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", captured_path)

        task = {
            "notify": False,
            "id": "task_scout",
            "description": "scout:target_y",
        }
        runner_mod.notify_failure(task, "EXIT 1:\nstderr")

        entry = json.loads(captured_path.read_text())
        assert entry["source"] == "claude_queue_runner"
        assert entry["delivered"] is None
        assert entry["extra"]["task_id"] == "task_scout"
        assert entry["extra"]["notify_flag"] is False


class TestNotifyCompletion:
    """Tests for notify_completion with notify_policy."""

    def test_notify_completion_always_policy_sends_normal(self, monkeypatch):
        """Policy 'always' → send NORMAL for completion."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": True,
            "notify_policy": "always",
            "description": "fixer:test",
        }
        runner_mod.notify_completion(task, "/path/to/output.md")

        assert len(sent) == 1
        assert sent[0]["priority"].name == "NORMAL"

    def test_notify_completion_infra_only_no_send(
        self, monkeypatch, tmp_path
    ):
        """Policy 'infra-only' → no send, log silenced."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        log_path = tmp_path / "silenced.jsonl"
        monkeypatch.setattr(runner_mod, "SILENCED_LOG", log_path)

        task = {
            "notify": True,
            "notify_policy": "infra-only",
            "id": "task_complete",
            "description": "reviewer:test",
        }
        runner_mod.notify_completion(task, "/path/to/output.md")

        assert len(sent) == 0
        entry = json.loads(log_path.read_text())
        assert entry["event"] == "completion"
        assert entry["failure_class"] is None
        assert entry["demoted_from"] == "NORMAL"

    def test_notify_completion_missing_notify_policy_defaults_always(self, monkeypatch):
        """Missing notify_policy key → defaults to 'always' → NORMAL."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": True,
            # no notify_policy key
            "description": "test",
        }
        runner_mod.notify_completion(task, "/path/to/output.md")

        assert len(sent) == 1
        assert sent[0]["priority"].name == "NORMAL"

    def test_notify_completion_disabled_respects_notify_false(self, monkeypatch):
        """If notify=False, neither send nor log (regardless of policy)."""
        sent: list = []

        def fake_send(*, message, title, priority):
            sent.append({"priority": priority})

        monkeypatch.setattr(runner_mod, "send_notification", fake_send)

        task = {
            "notify": False,
            "notify_policy": "infra-only",
            "description": "test",
        }
        runner_mod.notify_completion(task, "/path/to/output.md")

        assert len(sent) == 0

    def test_notify_completion_disabled_still_writes_capture_log(self, tmp_path, monkeypatch):
        """notify=False produces a CAPTURE_LOG entry with delivered=null (AC2)."""
        import agents_core.notify as notify_mod
        captured_path = tmp_path / "captured.jsonl"
        monkeypatch.setattr(notify_mod, "CAPTURE_LOG", captured_path)

        task = {
            "notify": False,
            "id": "task_scout2",
            "description": "scout:target_z",
        }
        runner_mod.notify_completion(task, "/path/to/output.md")

        entry = json.loads(captured_path.read_text())
        assert entry["source"] == "claude_queue_runner"
        assert entry["delivered"] is None
        assert entry["extra"]["task_id"] == "task_scout2"
        assert entry["extra"]["notify_flag"] is False


class TestShaperNotifyPolicy:
    """Tests for Shaper — notify_policy field parsing."""

    def test_shaper_parses_notify_policy_from_registry(self, tmp_path):
        """A registry entry with notify_policy: infra-only is parsed correctly."""
        from agents_core.shaper import Shaper

        registry_yaml = tmp_path / "registry.yaml"
        registry_yaml.write_text(
            """
agents:
  test_agent:
    model: haiku
    system_template: test prompt
    notify: true
    notify_policy: infra-only
"""
        )

        shaper = Shaper(registry_yaml)
        agent = shaper.get_agent("test_agent")

        assert agent.notify_policy == "infra-only"

    def test_shaper_defaults_notify_policy_to_always(self, tmp_path):
        """A registry entry without notify_policy defaults to 'always'."""
        from agents_core.shaper import Shaper

        registry_yaml = tmp_path / "registry.yaml"
        registry_yaml.write_text(
            """
agents:
  test_agent:
    model: haiku
    system_template: test prompt
    notify: true
"""
        )

        shaper = Shaper(registry_yaml)
        agent = shaper.get_agent("test_agent")

        assert agent.notify_policy == "always"

    def test_shaper_agent_has_notify_policy_field(self, tmp_path):
        """The ShapedAgent object carries notify_policy from the registry."""
        from agents_core.shaper import Shaper

        registry_yaml = tmp_path / "registry.yaml"
        registry_yaml.write_text(
            """
agents:
  test_agent_always:
    model: haiku
    system_template: test prompt
    notify: true
    notify_policy: always
  test_agent_infra_only:
    model: sonnet
    system_template: test prompt 2
    notify: true
    notify_policy: infra-only
shared_preamble: ""
"""
        )

        shaper = Shaper(registry_yaml)

        # Verify both agents have the correct notify_policy.
        agent_always = shaper.get_agent("test_agent_always")
        assert agent_always.notify_policy == "always"

        agent_infra = shaper.get_agent("test_agent_infra_only")
        assert agent_infra.notify_policy == "infra-only"
