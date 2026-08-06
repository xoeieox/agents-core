"""Tests for shaped-agent per-agent backend_url / acquire_lease / model pass-through
(agents-core-shaped-agent-backend-url-passthrough-v0).

Covers:
  - ShapedAgent.backend_url / acquire_lease load from registry YAML with a proper
    boolean parser (not bool(...)) and empty/whitespace backend_url normalized to None.
  - Phantom-swarm guardrail: acquire_lease: false with no backend_url raises at
    _load_registry time (fail-fast); acquire_lease: false with an explicit backend_url
    loads fine.
  - spec dict carries backend_url/acquire_lease through to _run_local_fixer and
    _run_local_reviewer, which pass them (plus model) to call_gw_agent.
  - Byte-identical-default regression: a registry entry with no backend_url/
    acquire_lease keys (today's registry.yaml shape) produces
    call_gw_agent(backend_url=None, acquire_lease=True) for both runners.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agents_core.shaped_runner as sr
from agents_core.shaper import Shaper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(path: Path, agents: dict) -> Path:
    reg = path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": agents}))
    return reg


def _agent_def(model: str = "gravitywell-122b", engine: str = "local-fixer", **extra) -> dict:
    d = {
        "chub_bundles": [],
        "system_template": "test for {repo}",
        "model": model,
        "timeout_s": 60,
        "engine": engine,
    }
    d.update(extra)
    return d


def _make_fixer_spec(tmp_path: Path, **overrides) -> dict:
    spec = {
        "model": "gravitywell-122b",
        "engine": "local-fixer",
        "system": "you are a fixer",
        "prompt": "fix the bug",
        "timeout_s": 60,
        "capture_meta": False,
        "target_id": "my-target-v0",
        "repo": "agents-core",
        "task_id": "abc123",
        "base_branch": "main",
        "slot_id": "abc123",
    }
    spec.update(overrides)
    return spec


def _make_reviewer_spec(**overrides) -> dict:
    spec = {
        "model": "gravitywell-122b",
        "engine": "local-reviewer",
        "system": "you are a reviewer",
        "prompt": "review this",
        "timeout_s": 60,
        "task_id": "rev123",
        "slot_id": "rev123",
    }
    spec.update(overrides)
    return spec


def _fake_handle(worktree: Path) -> MagicMock:
    h = MagicMock()
    h.path = worktree
    h.env = {}
    return h


def _good_fixer_result(diff: str = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n") -> dict:
    return {
        "final_diff": diff,
        "concluded": True,
        "last_test_outcome": {"passed": 3, "failed": 0},
        "steps": [{"tool": "read_file"}, {"tool": "write_file"}],
    }


# ---------------------------------------------------------------------------
# ShapedAgent / registry load
# ---------------------------------------------------------------------------

def test_shaped_agent_defaults_backend_url_none_acquire_lease_true(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def()})
    s = Shaper(reg)
    agent = s.get_agent("fixer_local")
    assert agent.backend_url is None
    assert agent.acquire_lease is True


def test_shaped_agent_loads_explicit_backend_url_and_acquire_lease_false(tmp_path):
    reg = _write_registry(tmp_path, {
        "fixer_local": _agent_def(backend_url="http://localhost:1234/v1", acquire_lease=False),
    })
    s = Shaper(reg)
    agent = s.get_agent("fixer_local")
    assert agent.backend_url == "http://localhost:1234/v1"
    assert agent.acquire_lease is False


def test_quoted_acquire_lease_false_string_resolves_to_false(tmp_path):
    reg = _write_registry(tmp_path, {
        "fixer_local": _agent_def(backend_url="http://localhost:1234/v1", acquire_lease="false"),
    })
    s = Shaper(reg)
    assert s.get_agent("fixer_local").acquire_lease is False


def test_empty_string_backend_url_normalizes_to_none(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def(backend_url="")})
    s = Shaper(reg)
    assert s.get_agent("fixer_local").backend_url is None


def test_whitespace_backend_url_normalizes_to_none(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def(backend_url="   ")})
    s = Shaper(reg)
    assert s.get_agent("fixer_local").backend_url is None


# ---------------------------------------------------------------------------
# Phantom-swarm guardrail
# ---------------------------------------------------------------------------

def test_acquire_lease_false_without_backend_url_raises(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def(acquire_lease=False)})
    with pytest.raises(RuntimeError, match="acquire_lease"):
        Shaper(reg)


def test_acquire_lease_false_with_explicit_backend_url_loads_fine(tmp_path):
    reg = _write_registry(tmp_path, {
        "fixer_local": _agent_def(backend_url="http://localhost:1234/v1", acquire_lease=False),
    })
    s = Shaper(reg)  # must not raise
    assert s.get_agent("fixer_local").acquire_lease is False


def test_acquire_lease_false_string_without_backend_url_raises(tmp_path):
    """Quoted 'false' must also trip the guardrail, not just Python bool False."""
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def(acquire_lease="false")})
    with pytest.raises(RuntimeError, match="acquire_lease"):
        Shaper(reg)


def test_acquire_lease_false_with_empty_backend_url_raises(tmp_path):
    """Empty-string backend_url normalizes to None before the guardrail check runs."""
    reg = _write_registry(tmp_path, {
        "fixer_local": _agent_def(backend_url="", acquire_lease=False),
    })
    with pytest.raises(RuntimeError, match="acquire_lease"):
        Shaper(reg)


# ---------------------------------------------------------------------------
# spec dict assembly
# ---------------------------------------------------------------------------

@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    import agents_core.shaper as shaper_mod

    reg = _write_registry(tmp_path, {
        "fixer_local": _agent_def(),
        "fixer_local_override": _agent_def(backend_url="http://localhost:1234/v1", acquire_lease=False),
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
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))
    return s, claude_q, gpu_q


def test_spec_carries_default_backend_url_none_and_acquire_lease_true(shaper_mocks, tmp_path):
    s, _, _ = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer_local", "t-1", "fix it", vars_={"repo": "agents-core"})
    spec = json.loads(list(spec_dir.glob("*.json"))[0].read_text())
    assert spec["backend_url"] is None
    assert spec["acquire_lease"] is True


def test_spec_carries_explicit_backend_url_and_acquire_lease_false(shaper_mocks, tmp_path):
    s, _, _ = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer_local_override", "t-2", "fix it", vars_={"repo": "agents-core"})
    spec = json.loads(list(spec_dir.glob("*.json"))[0].read_text())
    assert spec["backend_url"] == "http://localhost:1234/v1"
    assert spec["acquire_lease"] is False


# ---------------------------------------------------------------------------
# _run_local_fixer / _run_local_reviewer pass-through to call_gw_agent
# ---------------------------------------------------------------------------

def test_run_local_fixer_byte_identical_default_passthrough(tmp_path):
    """No backend_url/acquire_lease in spec (today's registry.yaml shape) -> call_gw_agent
    receives backend_url=None, acquire_lease=True, model=spec['model'] -- argument-identical
    to the pre-change call for every other kwarg."""
    spec = _make_fixer_spec(tmp_path)
    worktree = tmp_path / "wt"
    worktree.mkdir()

    captured = {}

    def fake_gw_agent(**kwargs):
        captured.update(kwargs)
        return _good_fixer_result(), []

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/pulls/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured["backend_url"] is None
    assert captured["acquire_lease"] is True
    assert captured["model"] == "gravitywell-122b"
    assert captured["prompt"] == "fix the bug"
    assert captured["system"] == "you are a fixer"
    assert captured["cwd"] == str(worktree)
    assert captured["writeable"] is True
    assert captured["timeout"] == 60
    assert captured["think"] is False
    assert captured["on_wake_fail"] == "skip"
    assert captured["work_id"] == "abc123"


def test_run_local_fixer_explicit_backend_url_and_acquire_lease_passthrough(tmp_path):
    spec = _make_fixer_spec(tmp_path, backend_url="http://localhost:1234/v1", acquire_lease=False)
    worktree = tmp_path / "wt"
    worktree.mkdir()

    captured = {}

    def fake_gw_agent(**kwargs):
        captured.update(kwargs)
        return _good_fixer_result(), []

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/pulls/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured["backend_url"] == "http://localhost:1234/v1"
    assert captured["acquire_lease"] is False


def test_run_local_reviewer_byte_identical_default_passthrough(tmp_path):
    spec = _make_reviewer_spec()

    captured = {}

    def fake_gw_agent(**kwargs):
        captured.update(kwargs)
        return "review verdict"

    with patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent):
        result = sr._run_local_reviewer(spec, str(tmp_path))

    assert result == "review verdict"
    assert captured["backend_url"] is None
    assert captured["acquire_lease"] is True
    assert captured["model"] == "gravitywell-122b"
    assert captured["prompt"] == "review this"
    assert captured["system"] == "you are a reviewer"
    assert captured["writeable"] is False
    assert captured["json_mode"] is True
    assert captured["timeout"] == 60
    assert captured["work_id"] == "rev123"


def test_run_local_reviewer_explicit_backend_url_and_acquire_lease_passthrough(tmp_path):
    spec = _make_reviewer_spec(backend_url="http://localhost:1234/v1", acquire_lease=False)

    captured = {}

    def fake_gw_agent(**kwargs):
        captured.update(kwargs)
        return "review verdict"

    with patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent):
        sr._run_local_reviewer(spec, str(tmp_path))

    assert captured["backend_url"] == "http://localhost:1234/v1"
    assert captured["acquire_lease"] is False


def test_run_local_fixer_model_passthrough_matches_no_model_prior_call(tmp_path):
    """DoD: model=spec.get('model') must not change the resolved (endpoint, model) for
    today's gravitywell-122b entries -- call_gw_agent(model='gravitywell-122b') resolves
    identically to the prior no-model call, exactly as _run_local_reviewer already does."""
    spec = _make_fixer_spec(tmp_path, model="gravitywell-122b")
    worktree = tmp_path / "wt"
    worktree.mkdir()

    captured = {}

    def fake_gw_agent(**kwargs):
        captured.update(kwargs)
        return _good_fixer_result(), []

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/pulls/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured["model"] == "gravitywell-122b"
