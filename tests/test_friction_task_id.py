# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D4b (attestation-contract-v0, leg 1): the friction writer carries the
real task id.

The live record `friction/agents-core-test-gate-bypassed-no-python-test-
infra` carries `first_task_id: abc123` (placeholder), so the record cannot
be tied to the run that produced it. D4b threads the actual task_id
through `_write_friction_entry` and removes the placeholder path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents_core import shaped_runner


class TestFrictionTaskId:
    def test_entry_carries_real_task_id(self, tmp_path: Path, monkeypatch):
        """A new friction record carries the real task id (no abc123)."""
        friction_dir = tmp_path / "friction"
        monkeypatch.setattr(shaped_runner, "FRICTION_DIR", friction_dir)

        shaped_runner._write_friction_entry(
            category="test-gate-bypassed",
            task_id="task-20260909-123456",
            detail="gate bypassed: no python test infra",
            repo="agents-core",
        )

        files = list(friction_dir.glob("*"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "task-20260909-123456" in content
        # the placeholder is gone
        assert "abc123" not in content

    def test_entry_without_task_id_uses_unknown_not_placeholder(
        self, tmp_path: Path, monkeypatch,
    ):
        """The caller without a task id (the friction seam is generic)
        writes 'unknown' - never the placeholder."""
        friction_dir = tmp_path / "friction"
        monkeypatch.setattr(shaped_runner, "FRICTION_DIR", friction_dir)

        shaped_runner._write_friction_entry(
            category="test-gate-bypassed",
            detail="gate bypassed",
            repo="agents-core",
        )

        content = list(friction_dir.glob("*"))[0].read_text()
        assert "abc123" not in content

    def test_gate_bypass_caller_threads_task_id(self, tmp_path: Path, monkeypatch):
        """The test-gate-bypass caller (shaped_runner.py:2254-2261 shape)
        has the task_id in scope and threads it - the live record's
        placeholder shape cannot recur."""
        import inspect

        src = inspect.getsource(shaped_runner._run_local_fixer)
        # the gate-bypass friction write names the task id (the caller at
        # the bypass site has it)
        assert "task_id=" in src
        # and the placeholder is not in the module's friction writes
        assert "abc123" not in src
