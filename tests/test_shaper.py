"""Unit tests for agents_core.shaper.Shaper.

Covers:
  - Registry load: valid, missing, malformed, missing agents: key, defaults.
  - Dispatch routing: sonnet/haiku/opus → ClaudeQueue; qwen → GPUQueue;
    AGENTS_CORE_FORCE_GPU_QUEUE=1; LAPIS_PM_FORCE_GPU_QUEUE=1 (deprecated alias).
  - Spec generation: capture_meta from agent, permission_mode, worktree_required.
  - DispatchResult fields: task_id, agent_type, spec_path, output_path.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agents_core.shaper as shaper_mod
from agents_core.shaper import DispatchResult, Shaper, ShapedAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(path: Path, agents: dict | None = None, shared_preamble: str = "") -> Path:
    reg = path / "registry.yaml"
    data: dict = {}
    if shared_preamble:
        data["shared_preamble"] = shared_preamble
    if agents is not None:
        data["agents"] = agents
    reg.write_text(yaml.dump(data) if data else "")
    return reg


def _minimal_agent_def(model: str = "sonnet", capture_meta: bool = False, notify: bool = False) -> dict:
    return {
        "chub_bundles": [],
        "system_template": "test agent for {repo}",
        "model": model,
        "timeout_s": 60,
        "capture_meta": capture_meta,
        "notify": notify,
    }


@pytest.fixture
def dispatch_mocks(tmp_path, monkeypatch):
    """Fixture: temp registry + patched SPEC_DIR + mock queues."""
    reg = _write_registry(tmp_path, agents={
        "fixer": _minimal_agent_def("sonnet", capture_meta=True, notify=True),
        "reviewer": _minimal_agent_def("opus", capture_meta=False, notify=True),
        "scout": _minimal_agent_def("haiku"),
        "qwen_agent": _minimal_agent_def("qwen3.6-35b-a3b"),
    })
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
    # Avoid filesystem stat for cwd resolution in tests.
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))

    return s, claude_q, gpu_q


# ---------------------------------------------------------------------------
# Registry load
# ---------------------------------------------------------------------------

def test_registry_load_valid(tmp_path):
    reg = _write_registry(tmp_path, agents={"fixer": _minimal_agent_def()})
    s = Shaper(reg)
    assert "fixer" in s.list_agents()
    agent = s.get_agent("fixer")
    assert isinstance(agent, ShapedAgent)
    assert agent.model == "sonnet"


def test_registry_load_missing_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Shaper registry missing"):
        Shaper(tmp_path / "nonexistent.yaml")


def test_registry_load_malformed_raises(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text(": bad: yaml: [unclosed")
    with pytest.raises(RuntimeError, match="Shaper registry malformed"):
        Shaper(reg)


def test_registry_load_missing_agents_key_empty_registry(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text("shared_preamble: 'hello'\n")
    s = Shaper(reg)
    assert s.list_agents() == []


def test_registry_defaults_capture_meta_and_notify(tmp_path):
    """Agents without capture_meta/notify in registry get False defaults."""
    reg = _write_registry(tmp_path, agents={
        "agent_no_flags": {
            "chub_bundles": [],
            "system_template": "hi",
            "model": "haiku",
            "timeout_s": 60,
        }
    })
    s = Shaper(reg)
    a = s.get_agent("agent_no_flags")
    assert a.capture_meta is False
    assert a.notify is False


def test_registry_capture_meta_and_notify_loaded(tmp_path):
    reg = _write_registry(tmp_path, agents={
        "fixer": _minimal_agent_def("sonnet", capture_meta=True, notify=True),
        "reviewer": _minimal_agent_def("opus", capture_meta=False, notify=True),
        "scout": _minimal_agent_def("haiku", capture_meta=False, notify=False),
    })
    s = Shaper(reg)
    assert s.get_agent("fixer").capture_meta is True
    assert s.get_agent("fixer").notify is True
    assert s.get_agent("reviewer").capture_meta is False
    assert s.get_agent("reviewer").notify is True
    assert s.get_agent("scout").capture_meta is False
    assert s.get_agent("scout").notify is False


def test_get_agent_unknown_raises(tmp_path):
    reg = _write_registry(tmp_path, agents={"fixer": _minimal_agent_def()})
    s = Shaper(reg)
    with pytest.raises(KeyError, match="Unknown shaped agent"):
        s.get_agent("nonexistent")


def test_reload_registry(tmp_path):
    reg = _write_registry(tmp_path, agents={"fixer": _minimal_agent_def()})
    s = Shaper(reg)
    assert s.list_agents() == ["fixer"]
    # Add an agent and reload.
    reg.write_text(yaml.dump({"agents": {
        "fixer": _minimal_agent_def(),
        "scout": _minimal_agent_def("haiku"),
    }}))
    s.reload_registry()
    assert sorted(s.list_agents()) == ["fixer", "scout"]


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model,agent_name", [
    ("sonnet", "fixer"),
    ("opus", "reviewer"),
    ("haiku", "scout"),
])
def test_anthropic_models_route_to_claude_queue(dispatch_mocks, model, agent_name):
    s, claude_q, gpu_q = dispatch_mocks
    s.dispatch(agent_name, "t-1", "do thing", vars_={"repo": "test-repo"})
    assert claude_q.submit.called, f"{model} should route to ClaudeQueue"
    assert not gpu_q.submit.called, f"{model} must not touch GPUQueue"


def test_qwen_routes_to_gpu_queue(dispatch_mocks):
    s, claude_q, gpu_q = dispatch_mocks
    s.dispatch("qwen_agent", "t-1", "do thing", vars_={"repo": "test-repo"})
    assert gpu_q.submit.called
    assert not claude_q.submit.called


def test_force_gpu_queue_env_overrides_claude(dispatch_mocks, monkeypatch):
    s, claude_q, gpu_q = dispatch_mocks
    monkeypatch.setenv("AGENTS_CORE_FORCE_GPU_QUEUE", "1")
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    assert gpu_q.submit.called
    assert not claude_q.submit.called


def test_deprecated_lapis_pm_force_gpu_queue_routes_to_gpu(dispatch_mocks, monkeypatch, capsys):
    s, claude_q, gpu_q = dispatch_mocks
    monkeypatch.setenv("LAPIS_PM_FORCE_GPU_QUEUE", "1")
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    assert gpu_q.submit.called
    assert not claude_q.submit.called
    captured = capsys.readouterr()
    assert "DeprecationWarning" in captured.err
    assert "LAPIS_PM_FORCE_GPU_QUEUE" in captured.err
    assert "AGENTS_CORE_FORCE_GPU_QUEUE" in captured.err


def test_deprecated_warning_emitted_once_per_instance(dispatch_mocks, monkeypatch, capsys):
    s, claude_q, gpu_q = dispatch_mocks
    monkeypatch.setenv("LAPIS_PM_FORCE_GPU_QUEUE", "1")
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    s.dispatch("fixer", "t-2", "do thing", vars_={"repo": "test-repo"})
    captured = capsys.readouterr()
    assert captured.err.count("DeprecationWarning") == 1


# ---------------------------------------------------------------------------
# Spec generation
# ---------------------------------------------------------------------------

def test_spec_capture_meta_matches_agent(dispatch_mocks, tmp_path):
    s, claude_q, gpu_q = dispatch_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    written = list(spec_dir.glob("*.json"))
    assert len(written) == 1
    spec = json.loads(written[0].read_text())
    assert spec["capture_meta"] is True, "fixer has capture_meta=true"
    assert spec["permission_mode"] == "bypassPermissions"
    assert spec["worktree_required"] is True  # ClaudeQueue route


def test_spec_worktree_required_false_for_gpu_route(dispatch_mocks, tmp_path):
    s, claude_q, gpu_q = dispatch_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("qwen_agent", "t-1", "do thing", vars_={"repo": "test-repo"})
    written = list(spec_dir.glob("*.json"))
    assert len(written) == 1
    spec = json.loads(written[0].read_text())
    assert "worktree_required" not in spec or not spec.get("worktree_required")


def test_dispatch_result_fields_claude_route(dispatch_mocks):
    s, claude_q, gpu_q = dispatch_mocks
    result = s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    assert isinstance(result, DispatchResult)
    assert result.task_id == "claude_task_id"
    assert result.agent_type == "fixer"
    assert "fixer" in result.spec_path
    assert result.output_path.startswith("/srv/lapis/claude-queue/completed/")


def test_dispatch_result_fields_gpu_route(dispatch_mocks):
    s, claude_q, gpu_q = dispatch_mocks
    result = s.dispatch("qwen_agent", "t-1", "do thing", vars_={"repo": "test-repo"})
    assert isinstance(result, DispatchResult)
    assert result.task_id == "gpu_task_id"
    assert result.agent_type == "qwen_agent"
    assert result.output_path.startswith("/srv/lapis/gpu-queue/completed/")


def test_dispatch_uses_runner_module_invocation(dispatch_mocks, tmp_path):
    """Command in spec payload uses `python3 -m agents_core.shaped_runner`, not a path."""
    s, claude_q, gpu_q = dispatch_mocks
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    call_kwargs = claude_q.submit.call_args[0][0]
    cmd = call_kwargs["payload"]["command"]
    assert "-m agents_core.shaped_runner" in cmd
    assert "_runner.py" not in cmd


def test_notify_set_from_agent_registry(dispatch_mocks):
    s, claude_q, gpu_q = dispatch_mocks
    s.dispatch("fixer", "t-1", "do thing", vars_={"repo": "test-repo"})
    submitted = claude_q.submit.call_args[0][0]
    assert submitted["notify"] is True

    claude_q.reset_mock()
    s.dispatch("reviewer", "t-2", "do review", vars_={"repo": "test-repo"})
    submitted = claude_q.submit.call_args[0][0]
    assert submitted["notify"] is True

    claude_q.reset_mock()
    s.dispatch("scout", "t-3", "scout thing", vars_={"repo": "test-repo"})
    submitted = claude_q.submit.call_args[0][0]
    assert submitted["notify"] is False
