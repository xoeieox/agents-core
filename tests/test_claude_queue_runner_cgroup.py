"""Tests for cgroup isolation: user-manager scope + slice (Layer 1 + 2).

Covers:
  - _build_scope_argv (pure function, regression guard)
  - _cage_buildable (binary + socket checks)
  - _check_slice_has_cpu_quota (transient/uncapped detection)
  - _run_shaped_task: scope-wrapped launch, fail-closed hold, ALLOW_UNBOUNDED
  - _reap_orphan_scopes: reaper reconciliation + error swallowing

Zero network; no real systemd spawn; no Docker.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from agents_core import claude_queue_runner as runner_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_cage_alert_ts():
    """Reset cage-alert cooldown state before/after each test."""
    runner_mod._cage_alert_last_ts = 0.0
    yield
    runner_mod._cage_alert_last_ts = 0.0


def _fake_queue(tmp_path):
    """Return a minimal queue-like object with real active/pending dirs."""
    active = tmp_path / "active"
    pending = tmp_path / "pending"
    active.mkdir(exist_ok=True)
    pending.mkdir(exist_ok=True)

    class FakeQueue:
        active_dir = active
        pending_dir = pending

        def fail(self, task_id, error=""):
            pass

        def complete(self, task_id, **kwargs):
            pass

    return FakeQueue()


def _write_active_task(queue, task_id, spec_path="/fake/spec.md"):
    data = {"id": task_id, "status": "running", "payload": {"spec_path": spec_path}}
    (queue.active_dir / f"{task_id}.yaml").write_text(yaml.safe_dump(data))


class _FakeProcess:
    returncode = 0

    async def communicate(self):
        return b"agent output " * 10, b""


# ---------------------------------------------------------------------------
# _build_scope_argv — pure function tests
# ---------------------------------------------------------------------------

def test_build_scope_argv_shape(monkeypatch):
    """Scope wrapper includes all required flags in correct positions."""
    monkeypatch.setenv("CLAUDE_QUEUE_JOB_CPUQUOTA", "300%")
    monkeypatch.setenv("CLAUDE_QUEUE_JOB_MEMMAX", "6G")
    orig = [sys.executable, "-m", "agents_core.shaped_runner", "/spec.md"]

    argv = runner_mod._build_scope_argv("task-abc", orig)

    assert argv[0] == "systemd-run"
    assert "--user" in argv
    assert "--scope" in argv
    assert "--collect" in argv
    assert "--unit=lapis-fixer-task-abc.scope" in argv
    assert "--slice=lapis-fixer.slice" in argv
    p_vals = [argv[i + 1] for i, a in enumerate(argv) if a == "-p" and i + 1 < len(argv)]
    assert "CPUQuota=300%" in p_vals
    assert "MemoryMax=6G" in p_vals
    # Original argv is preserved verbatim after --
    assert "--" in argv
    sep = argv.index("--")
    assert argv[sep + 1:] == orig


def test_build_scope_argv_regression_slice_present():
    """Regression: --slice= must never be dropped — it is the aggregate isolation boundary."""
    argv = runner_mod._build_scope_argv("xyz", ["cmd"])
    assert any(a == "--slice=lapis-fixer.slice" for a in argv), (
        "slice flag missing — aggregate isolation is lost without it"
    )


def test_build_scope_argv_unit_name_encodes_task_id():
    """--unit embeds the task_id so the reaper can map scope → task."""
    argv = runner_mod._build_scope_argv("abc-def-123", [])
    assert "--unit=lapis-fixer-abc-def-123.scope" in argv


def test_build_scope_argv_env_override(monkeypatch):
    """CLAUDE_QUEUE_JOB_CPUQUOTA / CLAUDE_QUEUE_JOB_MEMMAX are honoured."""
    monkeypatch.setenv("CLAUDE_QUEUE_JOB_CPUQUOTA", "500%")
    monkeypatch.setenv("CLAUDE_QUEUE_JOB_MEMMAX", "10G")
    argv = runner_mod._build_scope_argv("t1", [])
    p_vals = [argv[i + 1] for i, a in enumerate(argv) if a == "-p" and i + 1 < len(argv)]
    assert "CPUQuota=500%" in p_vals
    assert "MemoryMax=10G" in p_vals


# ---------------------------------------------------------------------------
# _cage_buildable
# ---------------------------------------------------------------------------

def test_cage_buildable_no_systemd_run(monkeypatch):
    """(False, reason) when systemd-run is absent from PATH."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    ok, reason = runner_mod._cage_buildable()
    assert not ok
    assert "systemd-run" in reason


def test_cage_buildable_no_bus(monkeypatch):
    """(False, reason) when the user bus socket is missing."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemd-run")
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    ok, reason = runner_mod._cage_buildable()
    assert not ok
    assert "bus" in reason.lower()


def test_cage_buildable_ok(monkeypatch):
    """(True, '') when binary + socket are present."""
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemd-run")
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    ok, reason = runner_mod._cage_buildable()
    assert ok
    assert reason == ""


# ---------------------------------------------------------------------------
# _check_slice_has_cpu_quota (aggregate detection)
# ---------------------------------------------------------------------------

def _fake_run_slice(stdout_text, rc=0):
    """Return a fake subprocess.run result for systemctl show."""
    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, rc)
        result.stdout = stdout_text
        result.stderr = ""
        return result
    return fake_run


def test_check_slice_not_found_returns_false(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        _fake_run_slice("LoadState=not-found\nCPUQuota=infinity\n"))
    assert runner_mod._check_slice_has_cpu_quota() is False


def test_check_slice_infinity_returns_false(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        _fake_run_slice("LoadState=loaded\nCPUQuota=infinity\n"))
    assert runner_mod._check_slice_has_cpu_quota() is False


def test_check_slice_quota_set_returns_true(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        _fake_run_slice("LoadState=loaded\nCPUQuota=10000ms 10s\n"))
    assert runner_mod._check_slice_has_cpu_quota() is True


def test_check_slice_error_fails_open(monkeypatch):
    def boom(*a, **k):
        raise OSError("no systemctl")
    monkeypatch.setattr(subprocess, "run", boom)
    # Fail open: returns True so no spurious warning is emitted.
    assert runner_mod._check_slice_has_cpu_quota() is True


# ---------------------------------------------------------------------------
# _run_shaped_task: launch-command construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_shaped_task_uses_scope_wrapper(monkeypatch, tmp_path):
    """With a healthy cage, _run_shaped_task wraps argv in systemd-run scope."""
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_cage_buildable", lambda: (True, ""))
    monkeypatch.setattr(runner_mod, "_check_slice_has_cpu_quota", lambda: True)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)

    captured = []

    async def fake_spawn(*args, **kwargs):
        captured.extend(args)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    q = _fake_queue(tmp_path)
    task = {"id": "fixer-001", "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}

    await runner_mod._run_shaped_task(q, task)

    assert captured, "subprocess should have been spawned"
    assert captured[0] == "systemd-run", "first arg must be systemd-run"
    assert "--user" in captured
    assert "--scope" in captured
    assert "--slice=lapis-fixer.slice" in captured
    assert "--unit=lapis-fixer-fixer-001.scope" in captured
    assert "--" in captured
    sep = list(captured).index("--")
    tail = list(captured)[sep + 1:]
    assert sys.executable in tail
    assert "-m" in tail


# ---------------------------------------------------------------------------
# _run_shaped_task: fail-closed hold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_shaped_task_fail_closed_holds_and_alerts(monkeypatch, tmp_path):
    """Cage build failure: task requeued to pending, exactly one alert sent."""
    monkeypatch.setattr(runner_mod, "_cage_buildable",
                        lambda: (False, "no systemd-run"))
    monkeypatch.delenv("CLAUDE_QUEUE_ALLOW_UNBOUNDED", raising=False)

    alerts = []
    monkeypatch.setattr(runner_mod, "send_notification",
                        lambda message, title, priority: alerts.append(message))

    spawned = []

    async def fail_if_spawned(*a, **kw):
        spawned.append(a)
        raise AssertionError("subprocess must NOT be spawned in fail-closed mode")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_if_spawned)

    q = _fake_queue(tmp_path)
    task_id = "hold-001"
    task = {"id": task_id, "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}
    _write_active_task(q, task_id)

    # First call: cage fails → requeue + alert.
    await runner_mod._run_shaped_task(q, task)

    assert not spawned, "no subprocess should have been spawned"
    assert (q.pending_dir / f"{task_id}.yaml").exists(), "task must be in pending"
    assert not (q.active_dir / f"{task_id}.yaml").exists(), "task must leave active"
    assert len(alerts) == 1, "exactly one alert on first failure"

    # Second call within cooldown: re-create the active file, call again.
    _write_active_task(q, task_id)
    (q.pending_dir / f"{task_id}.yaml").unlink(missing_ok=True)

    await runner_mod._run_shaped_task(q, task)

    assert len(alerts) == 1, "no second alert within cooldown window"


@pytest.mark.asyncio
async def test_run_shaped_task_fail_closed_alert_resends_after_cooldown(monkeypatch, tmp_path):
    """Alert re-fires once the cooldown expires."""
    monkeypatch.setattr(runner_mod, "_cage_buildable",
                        lambda: (False, "no bus"))
    monkeypatch.delenv("CLAUDE_QUEUE_ALLOW_UNBOUNDED", raising=False)

    alerts = []
    monkeypatch.setattr(runner_mod, "send_notification",
                        lambda message, title, priority: alerts.append(message))
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))

    q = _fake_queue(tmp_path)
    task = {"id": "hold-002", "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}

    _write_active_task(q, "hold-002")
    await runner_mod._run_shaped_task(q, task)
    assert len(alerts) == 1

    # Simulate cooldown expiry by resetting timestamp to the past.
    runner_mod._cage_alert_last_ts -= runner_mod._CAGE_ALERT_COOLDOWN_S + 1

    _write_active_task(q, "hold-002")
    (q.pending_dir / "hold-002.yaml").unlink(missing_ok=True)
    await runner_mod._run_shaped_task(q, task)
    assert len(alerts) == 2, "alert should re-fire after cooldown"


@pytest.mark.asyncio
async def test_run_shaped_task_allow_unbounded_launches_without_cage(monkeypatch, tmp_path):
    """CLAUDE_QUEUE_ALLOW_UNBOUNDED=1: falls back to plain argv + WARNING."""
    monkeypatch.setenv("CLAUDE_QUEUE_ALLOW_UNBOUNDED", "1")
    monkeypatch.setattr(runner_mod, "_cage_buildable",
                        lambda: (False, "no bus"))
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)

    alerts = []
    monkeypatch.setattr(runner_mod, "send_notification",
                        lambda message, title, priority: alerts.append(message))

    captured = []

    async def fake_spawn(*args, **kwargs):
        captured.extend(args)
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    q = _fake_queue(tmp_path)
    task = {"id": "unbound-001", "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}

    await runner_mod._run_shaped_task(q, task)

    # Subprocess was spawned.
    assert captured, "subprocess must be spawned with ALLOW_UNBOUNDED=1"
    # NOT via systemd-run.
    assert captured[0] != "systemd-run", "unbounded launch must not use systemd-run"
    assert captured[0] == sys.executable
    # No cage-build alert.
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_run_shaped_task_allow_unbounded_logs_warning(monkeypatch, tmp_path, caplog):
    """CLAUDE_QUEUE_ALLOW_UNBOUNDED=1 emits a WARNING (not silent)."""
    monkeypatch.setenv("CLAUDE_QUEUE_ALLOW_UNBOUNDED", "1")
    monkeypatch.setattr(runner_mod, "_cage_buildable",
                        lambda: (False, "no bus"))
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)

    async def fake_spawn(*a, **kw):
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(runner_mod, "send_notification", lambda **kw: None)

    q = _fake_queue(tmp_path)
    task = {"id": "unbound-002", "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}

    with caplog.at_level(logging.WARNING, logger="claude-queue-runner"):
        await runner_mod._run_shaped_task(q, task)

    assert any("unbounded" in r.message.lower() for r in caplog.records), (
        "a WARNING mentioning 'unbounded' must be logged for ALLOW_UNBOUNDED runs"
    )


# ---------------------------------------------------------------------------
# _run_shaped_task: unbounded-aggregate detection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_shaped_task_warns_unbounded_aggregate(monkeypatch, tmp_path, caplog):
    """When slice has no CPUQuota, a distinct 'unbounded-aggregate' WARNING fires."""
    monkeypatch.setattr(runner_mod, "_cage_buildable", lambda: (True, ""))
    monkeypatch.setattr(runner_mod, "_check_slice_has_cpu_quota", lambda: False)
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_extract_ops_primitives", lambda *a, **kw: None)

    async def fake_spawn(*a, **kw):
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    q = _fake_queue(tmp_path)
    task = {"id": "agg-001", "payload": {"spec_path": "/spec.md"},
            "timeout_seconds": 30, "notify": False}

    with caplog.at_level(logging.WARNING, logger="claude-queue-runner"):
        await runner_mod._run_shaped_task(q, task)

    assert any("unbounded-aggregate" in r.message for r in caplog.records), (
        "unbounded-aggregate WARNING must be emitted when slice has no CPUQuota"
    )


# ---------------------------------------------------------------------------
# _reap_orphan_scopes — reaper reconciliation
# ---------------------------------------------------------------------------

def _scope_list_stdout(*scope_names):
    """Format fake systemctl list-units --plain output."""
    lines = []
    for name in scope_names:
        lines.append(f"{name}  loaded active running  -")
    return "\n".join(lines) + "\n"


def test_reap_orphan_scopes_stops_non_matching(monkeypatch, tmp_path):
    """Scopes with no matching active task are stopped; matching ones are left alone."""
    stop_calls = []

    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        if "list-units" in cmd:
            result.stdout = _scope_list_stdout(
                "lapis-fixer-active-task.scope",   # has a matching active task
                "lapis-fixer-orphan-1.scope",       # no matching task → stop
                "lapis-fixer-orphan-2.scope",       # no matching task → stop
            )
            result.stderr = ""
        elif "stop" in cmd:
            stop_calls.append(cmd[-1])  # capture the unit name
            result.stdout = ""
            result.stderr = ""
        else:
            result.stdout = ""
            result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemctl")

    q = _fake_queue(tmp_path)
    # "active-task" has a live active YAML; orphans do not.
    _write_active_task(q, "active-task")

    runner_mod._reap_orphan_scopes(q)

    assert "lapis-fixer-orphan-1.scope" in stop_calls
    assert "lapis-fixer-orphan-2.scope" in stop_calls
    assert "lapis-fixer-active-task.scope" not in stop_calls


def test_reap_orphan_scopes_swallows_stop_error(monkeypatch, tmp_path):
    """A 'systemctl stop' failure is swallowed; startup_sweep continues."""
    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        if "list-units" in cmd:
            result.stdout = _scope_list_stdout("lapis-fixer-orphan-bad.scope")
            result.stderr = ""
        elif "stop" in cmd:
            raise OSError("systemctl unavailable")
        else:
            result.stdout = ""
            result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemctl")

    q = _fake_queue(tmp_path)

    # Must not raise.
    runner_mod._reap_orphan_scopes(q)


def test_reap_orphan_scopes_list_units_error_swallowed(monkeypatch, tmp_path):
    """list-units subprocess error is swallowed; reaper exits cleanly."""
    def fake_run(cmd, **kwargs):
        raise OSError("no systemd")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemctl")

    q = _fake_queue(tmp_path)

    runner_mod._reap_orphan_scopes(q)  # must not raise


def test_reap_orphan_scopes_no_systemctl_is_noop(monkeypatch, tmp_path):
    """On hosts without systemctl, reaper does nothing silently."""
    called = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: called.append(a))
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    runner_mod._reap_orphan_scopes(_fake_queue(tmp_path))

    assert not called


def test_reap_orphan_scopes_list_units_nonzero_rc_skips(monkeypatch, tmp_path):
    """Non-zero rc from list-units is logged and reaper returns without stopping."""
    stop_calls = []

    def fake_run(cmd, **kwargs):
        result = subprocess.CompletedProcess(cmd, 0)
        if "list-units" in cmd:
            result.returncode = 5
            result.stdout = ""
            result.stderr = ""
        elif "stop" in cmd:
            stop_calls.append(cmd)
            result.stdout = ""
            result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/systemctl")

    runner_mod._reap_orphan_scopes(_fake_queue(tmp_path))

    assert not stop_calls, "no stop calls when list-units fails"
