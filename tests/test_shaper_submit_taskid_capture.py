"""Tests for capturing ClaudeQueue.submit()'s return value in Shaper.dispatch.

Coverage per spec:
  - AC1: submit() returns a different id than generated -> DispatchResult.task_id
    equals the returned id
  - AC2: submit() returns the same id it was handed (novel-signature path) ->
    DispatchResult.task_id is unchanged from today's behaviour (regression guard)
  - AC3: output_path is derived from the final task_id (the returned/joined id,
    not the locally generated one)
  - AC4: _record_dispatch_slot is called with the final task_id as
    contributor-of-record
  - AC5: GPUQueue routing branch is behaviourally unchanged (it already
    captures the return)
  - AC6: DispatchResult exposes the dedup-join indicator, set only when the
    returned id differs from the generated one, and it retains the generated id
  - AC7.1 (poison pill): a submit() double that ignores task_id= and returns a
    different, recognisable id must flip DispatchResult.task_id to that id;
    this test fails against pre-Leg-1 shaper.py and passes only after it.

agents_core.shaper.ClaudeQueue and GPUQueue are patched throughout; no live
queue, no live mem, no filesystem queue writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agents_core.shaper as shaper_mod
from agents_core.shaper import DispatchResult, Shaper


def _write_registry(path: Path, agents: dict) -> Path:
    reg = path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": agents}))
    return reg


def _agent_def(model: str = "sonnet") -> dict:
    return {
        "chub_bundles": [],
        "system_template": "test agent for {repo}",
        "model": model,
        "timeout_s": 60,
    }


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    reg = _write_registry(tmp_path, agents={
        "fixer": _agent_def("sonnet"),
        "qwen_agent": _agent_def("qwen3.6-35b-a3b"),
    })
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")

    claude_q = MagicMock()
    claude_q._generate_id.return_value = "generated_id"
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"

    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda: gpu_q)
    monkeypatch.delenv("AGENTS_CORE_FORCE_GPU_QUEUE", raising=False)
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)

    s = Shaper(reg)
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))

    return s, claude_q, gpu_q


def _dispatch(s):
    return s.dispatch(
        agent_type="fixer",
        target_id="t-1",
        user_prompt="do the thing",
        vars_={"repo": "test-repo"},
    )


# ---------------------------------------------------------------------------
# AC1 / AC7.1 — dedup path: submit() returns a different id
# ---------------------------------------------------------------------------

def test_dedup_join_returns_submit_id_not_generated_id(shaper_mocks):
    """AC1 / AC7.1 poison pill: submit() ignores task_id= and returns a
    different, recognisable id. DispatchResult.task_id must equal that
    returned id, not the locally generated one. Fails against pre-Leg-1
    shaper.py (which discards submit()'s return and keeps the generated id)."""
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: "shared_task_id_from_other_dispatch"

    result = _dispatch(s)

    assert result.task_id == "shared_task_id_from_other_dispatch"
    assert result.task_id != "generated_id"


# ---------------------------------------------------------------------------
# AC2 — regression guard: novel-signature path unchanged
# ---------------------------------------------------------------------------

def test_novel_signature_path_task_id_unchanged(shaper_mocks):
    """AC2: when submit() returns the same id it was handed, DispatchResult.task_id
    is exactly that id (today's byte-identical common-path behaviour)."""
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id

    result = _dispatch(s)

    assert result.task_id == "generated_id"


# ---------------------------------------------------------------------------
# AC3 — output_path derived from the final task_id
# ---------------------------------------------------------------------------

def test_output_path_uses_final_task_id_on_dedup(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: "shared_task_id_from_other_dispatch"

    result = _dispatch(s)

    assert "shared_task_id_from_other_dispatch" in result.output_path
    assert "generated_id" not in result.output_path


def test_output_path_uses_generated_task_id_on_novel_path(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id

    result = _dispatch(s)

    assert "generated_id" in result.output_path


# ---------------------------------------------------------------------------
# AC4 — _record_dispatch_slot receives the final task_id
# ---------------------------------------------------------------------------

def test_record_dispatch_slot_receives_final_task_id_on_dedup(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: "shared_task_id_from_other_dispatch"

    with patch.object(shaper_mod, "_record_dispatch_slot") as mock_slot:
        _dispatch(s)

    assert mock_slot.called
    assert mock_slot.call_args.kwargs["task_id"] == "shared_task_id_from_other_dispatch"


# ---------------------------------------------------------------------------
# AC5 — GPUQueue routing branch unchanged
# ---------------------------------------------------------------------------

def test_gpu_route_task_id_unchanged(shaper_mocks):
    s, _claude_q, gpu_q = shaper_mocks
    result = s.dispatch(
        agent_type="qwen_agent",
        target_id="t-1",
        user_prompt="do the thing",
        vars_={"repo": "test-repo"},
    )

    assert gpu_q.submit.called
    assert result.task_id == "gpu_task_id"
    assert result.joined_task_id is None


# ---------------------------------------------------------------------------
# AC6 — dedup-join indicator on DispatchResult
# ---------------------------------------------------------------------------

def test_joined_task_id_set_and_retains_generated_id_on_dedup(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: "shared_task_id_from_other_dispatch"

    result = _dispatch(s)

    assert result.joined_task_id == "generated_id"
    assert result.task_id == "shared_task_id_from_other_dispatch"


def test_joined_task_id_absent_on_novel_path(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id

    result = _dispatch(s)

    assert result.joined_task_id is None


def test_dispatch_result_is_dataclass_instance(shaper_mocks):
    s, claude_q, _gpu_q = shaper_mocks
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id

    result = _dispatch(s)

    assert isinstance(result, DispatchResult)
