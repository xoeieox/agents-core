"""Tests for agents_core.claude_queue_runner — primarily the ops-primitive
extraction hook. The full `_run_task` coroutine spawns a subprocess, so we
don't end-to-end it here — we exercise the hook in isolation.
"""
from __future__ import annotations

import sys
import types

import pytest

from agents_core import claude_queue_runner as runner_mod


def _install_fake_ops_primitives(monkeypatch, recorder: list) -> None:
    """Inject a fake `ops_primitives` module so the hook is driven without
    hitting mem.db. Records the (task_id, task_type, text) triple."""
    fake = types.ModuleType("ops_primitives")

    def extract_and_store(task_id, task_type, text):
        recorder.append((task_id, task_type, text[:60]))
        return [{"type": "recorded"}]

    fake.extract_and_store = extract_and_store
    monkeypatch.setitem(sys.modules, "ops_primitives", fake)


def test_extract_ops_primitives_forwards_text(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    text = "A" * 200  # exceeds the >=80 "worth tagging" gate
    runner_mod._extract_ops_primitives(
        "claude_test_1", "subprocess", text, "/tmp/out.md"
    )

    assert calls == [("claude_test_1", "subprocess", text[:60])]


def test_extract_ops_primitives_falls_back_to_output_file(monkeypatch, tmp_path):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    output_path = tmp_path / "out.md"
    output_path.write_text("B" * 300)

    runner_mod._extract_ops_primitives(
        "claude_test_2", "subprocess",
        result_text="",           # empty — forces fallback read
        output_path=str(output_path),
    )

    assert calls
    assert calls[0][0] == "claude_test_2"
    assert calls[0][2].startswith("B")


def test_extract_ops_primitives_skips_trivial_text(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)

    runner_mod._extract_ops_primitives(
        "claude_test_3", "subprocess", "tiny", None
    )

    assert calls == []


def test_extract_ops_primitives_honours_env_gate(monkeypatch):
    calls: list = []
    _install_fake_ops_primitives(monkeypatch, calls)
    monkeypatch.setenv("LAPIS_OPS_PRIMITIVES", "0")

    runner_mod._extract_ops_primitives(
        "claude_test_4", "subprocess", "A" * 300, "/tmp/out.md"
    )

    assert calls == []


def test_extract_ops_primitives_swallows_failure(monkeypatch, caplog):
    """Missing weather data is acceptable at the aggregate — a broken
    ops_primitives must not blow up the runner."""
    fake = types.ModuleType("ops_primitives")

    def extract_and_store(*a, **k):
        raise RuntimeError("fake extractor error")

    fake.extract_and_store = extract_and_store
    monkeypatch.setitem(sys.modules, "ops_primitives", fake)

    # Should not raise.
    runner_mod._extract_ops_primitives(
        "claude_test_5", "subprocess", "A" * 300, None
    )
