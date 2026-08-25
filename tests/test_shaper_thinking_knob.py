"""Shaper registry -> spec ENCODE link for the `think` field
(agents-core-fixer-thinking-off-knob-v0). Same presence-assertion pattern
as test_shaper_swarm_payload_passthrough.py: a missing key in the encoded
spec would be silently filled by shaped_runner's .get(...) default while
flipping the payload behavior - the exact mutation class this pins.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from agents_core.shaper import Shaper


def _write_registry(path: Path, agents: dict) -> Path:
    reg = path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": agents}))
    return reg


def _berth_agent_def(**extra) -> dict:
    d = {
        "chub_bundles": [],
        "system_template": "test for {repo}",
        "model": "ninfer-27b",
        "timeout_s": 60,
        "engine": "local-fixer",
        "backend_url": "http://203.0.113.11:8082",
        "acquire_lease": True,
        "swarm_payload": True,
    }
    d.update(extra)
    return d


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    import agents_core.shaper as shaper_mod

    reg = _write_registry(tmp_path, {"fixer_berth": _berth_agent_def(think=True)})
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")
    claude_q = MagicMock()
    claude_q._generate_id.return_value = "claude_task_id"
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"
    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda: gpu_q)
    monkeypatch.delenv("AGENTS_CORE_FORCE_GPU_QUEUE", raising=False)
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)
    s = Shaper(reg)
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))
    return s


def test_shaped_agent_default_think_false(tmp_path):
    reg = _write_registry(tmp_path, {"fixer": _berth_agent_def()})
    s = Shaper(reg)
    assert s.get_agent("fixer").think is False


def test_shaped_agent_loads_explicit_think_true(tmp_path):
    reg = _write_registry(tmp_path, {"fixer": _berth_agent_def(think=True)})
    s = Shaper(reg)
    assert s.get_agent("fixer").think is True


def test_shaped_agent_parses_string_think_false(tmp_path):
    reg = _write_registry(tmp_path, {"fixer": _berth_agent_def(think="false")})
    s = Shaper(reg)
    assert s.get_agent("fixer").think is False


def test_dispatched_spec_carries_think_key_presence(shaper_mocks, tmp_path):
    shaper = shaper_mocks
    spec_dir = tmp_path / "shaped"
    shaper.dispatch("fixer_berth", "t-think", "fix it", vars_={"repo": "agents-core"})
    specs = sorted(spec_dir.glob("*.json"))
    assert len(specs) >= 1
    spec = json.loads(specs[-1].read_text())
    assert "think" in spec
    assert spec["think"] is True