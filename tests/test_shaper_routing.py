"""Tests for Shaper's queue-routing decision.

Opus/Sonnet/Haiku → ClaudeQueue (Anthropic API, no local-GPU gating).
Qwen → GPUQueue (local GPU, TOU-paused 4-9 PM).
AGENTS_CORE_FORCE_GPU_QUEUE=1 forces everything to GPUQueue (rollback knob).
LAPIS_PM_FORCE_GPU_QUEUE=1 is the deprecated alias (one merge cycle).

Moved from lapis-pm tests/ on 2026-04-28 as part of the shaper promotion
to agents-core (agents-core-promote-shaper). Rewritten to use Shaper class API.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agents_core.shaper as shaper_mod
from agents_core.shaper import Shaper, ShapedAgent


def _write_registry(path: Path, agents: dict) -> Path:
    reg = path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": agents}))
    return reg


def _agent_def(model: str) -> dict:
    return {
        "chub_bundles": [],
        "system_template": "test for {repo}",
        "model": model,
        "timeout_s": 60,
    }


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    reg = _write_registry(tmp_path, {
        "fixer": _agent_def("sonnet"),
        "reviewer": _agent_def("opus"),
        "haiku_agent": _agent_def("haiku"),
        "qwen_agent": _agent_def("qwen3.6-35b-a3b"),
    })
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")

    claude_q = MagicMock()
    claude_q._generate_id.return_value = "claude_task_id"
    claude_q.submit.return_value = None
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"

    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda: gpu_q)
    monkeypatch.delenv("AGENTS_CORE_FORCE_GPU_QUEUE", raising=False)
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)

    s = Shaper(reg)
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))

    return s, claude_q, gpu_q


def _dispatch(shaper_mocks, agent_name: str) -> None:
    s, _, _ = shaper_mocks
    s.dispatch(
        agent_type=agent_name,
        target_id="t-1",
        user_prompt="do the thing",
        vars_={"repo": "test-repo"},
    )


@pytest.mark.parametrize("agent_name", ["reviewer", "fixer", "haiku_agent"])
def test_anthropic_models_route_to_claude_queue(shaper_mocks, agent_name):
    s, claude_q, gpu_q = shaper_mocks
    _dispatch(shaper_mocks, agent_name)
    assert claude_q.submit.called, f"{agent_name} should route to ClaudeQueue"
    assert not gpu_q.submit.called, f"{agent_name} must not touch GPUQueue"


def test_qwen_routes_to_gpu_queue(shaper_mocks):
    s, claude_q, gpu_q = shaper_mocks
    _dispatch(shaper_mocks, "qwen_agent")
    assert gpu_q.submit.called, "qwen should route to GPUQueue"
    assert not claude_q.submit.called, "qwen must not touch ClaudeQueue"


@pytest.mark.parametrize("agent_name", ["reviewer", "fixer", "haiku_agent"])
def test_force_gpu_queue_env_overrides(shaper_mocks, monkeypatch, agent_name):
    s, claude_q, gpu_q = shaper_mocks
    monkeypatch.setenv("AGENTS_CORE_FORCE_GPU_QUEUE", "1")
    _dispatch(shaper_mocks, agent_name)
    assert gpu_q.submit.called, f"{agent_name} should fall back to GPUQueue when forced"
    assert not claude_q.submit.called


@pytest.mark.parametrize("agent_name", ["reviewer", "fixer", "haiku_agent"])
def test_deprecated_lapis_pm_force_gpu_queue_overrides(shaper_mocks, monkeypatch, agent_name):
    s, claude_q, gpu_q = shaper_mocks
    monkeypatch.setenv("LAPIS_PM_FORCE_GPU_QUEUE", "1")
    _dispatch(shaper_mocks, agent_name)
    assert gpu_q.submit.called, f"{agent_name} should fall back to GPUQueue via deprecated alias"
    assert not claude_q.submit.called
