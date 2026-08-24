"""Tests for the shaper registry->spec ENCODE link for swarm_payload
(shaper-swarm-payload-carry-v0).

The leg-2 parity test (agents-core PR #254) validated the gw_agent._is_swarm
FORMULA against registry fields but never tested that the shaper actually
carries the field into the spec dict. This is the encode-link test:

  - A berth-shaped registry entry (backend_url + swarm_payload: true +
    acquire_lease: true) encodes a spec dict with swarm_payload: True and
    backend_url set.
  - A stopgap-shaped entry (no swarm_payload key) encodes swarm_payload: False.
  - PRESENCE assertions ("swarm_payload" in spec) are mandatory in both
    cases: a value-only check would pass if a future edit deleted the key
    (the value would then come from shaped_runner's .get(..., False) default)
    while silently flipping _is_swarm behavior — the exact mutation class
    this spec closes.
  - The encoded spec fields make the gw_agent.py:1877 formula
    (swarm_payload or (backend_url is not None and not acquire_lease))
    compute True for the berth shape and False for the stopgap shape. The
    formula is asserted against the spec fields directly — gw_agent is NOT
    imported; this test is about the encode link only.

Plus the doorman enqueue-decision log line (item 2): unit tests asserting
the enqueue path of acquire_or_defer emits the new warning and the grant
path does not.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.doorman_server import GHOST_PRINCIPAL, _NodeState
from agents_core.shaper import Shaper

# The gw_agent.py:1877 _is_swarm formula, replicated verbatim against the
# encoded spec fields. Deliberately NOT imported from gw_agent — this test
# pins the encode link (registry -> spec dict), not the formula's consumer.
def _is_swarm_from_spec(spec: dict) -> bool:
    return spec.get("swarm_payload", False) or (
        (spec.get("backend_url") is not None) and (not spec.get("acquire_lease", True))
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(path: Path, agents: dict) -> Path:
    reg = path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": agents}))
    return reg


def _agent_def(**extra) -> dict:
    d = {
        "chub_bundles": [],
        "system_template": "test for {repo}",
        "model": "haiku",
        "timeout_s": 60,
        "engine": "local-fixer",
    }
    d.update(extra)
    return d


def _berth_agent_def() -> dict:
    """Berth-shaped entry: backend_url + swarm_payload: true +
    acquire_lease: true (the lapis-pm registry shape post-#302)."""
    return _agent_def(
        backend_url="http://203.0.113.11:8082/v1",
        acquire_lease=True,
        swarm_payload=True,
    )


def _stopgap_agent_def() -> dict:
    """Stopgap-shaped entry: no backend_url, no swarm_payload key (the
    pre-#302 / reverted registry shape)."""
    return _agent_def()


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    import agents_core.shaper as shaper_mod

    reg = _write_registry(tmp_path, {
        "fixer_berth": _berth_agent_def(),
        "fixer_stopgap": _stopgap_agent_def(),
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
    return s


def _dispatched_spec(shaper: Shaper, agent_type: str, tmp_path: Path) -> dict:
    """Dispatch one spec and read it back from the spec dir (fresh per call)."""
    spec_dir = tmp_path / "shaped"
    shaper.dispatch(agent_type, f"t-{agent_type}", "fix it", vars_={"repo": "agents-core"})
    specs = sorted(spec_dir.glob("*.json"))
    assert len(specs) >= 1
    return json.loads(specs[-1].read_text())


# ---------------------------------------------------------------------------
# ShapedAgent / registry load
# ---------------------------------------------------------------------------

def test_shaped_agent_default_swarm_payload_false(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_stopgap": _stopgap_agent_def()})
    s = Shaper(reg)
    agent = s.get_agent("fixer_stopgap")
    assert agent.swarm_payload is False


def test_shaped_agent_loads_explicit_swarm_payload_true(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_berth": _berth_agent_def()})
    s = Shaper(reg)
    agent = s.get_agent("fixer_berth")
    assert agent.swarm_payload is True


def test_quoted_swarm_payload_true_string_resolves_to_true(tmp_path):
    reg = _write_registry(tmp_path, {
        "fixer_berth": _agent_def(swarm_payload="true"),
    })
    s = Shaper(reg)
    assert s.get_agent("fixer_berth").swarm_payload is True


def test_quoted_swarm_payload_false_string_resolves_to_false(tmp_path):
    reg = _write_registry(tmp_path, {
        "fixer_stopgap": _agent_def(swarm_payload="false"),
    })
    s = Shaper(reg)
    assert s.get_agent("fixer_stopgap").swarm_payload is False


# ---------------------------------------------------------------------------
# spec dict assembly — the encode link
# ---------------------------------------------------------------------------

def test_berth_spec_carries_swarm_payload_true_and_backend_url(shaper_mocks, tmp_path):
    spec = _dispatched_spec(shaper_mocks, "fixer_berth", tmp_path)
    # PRESENCE (mandatory — gate correction 2026-08-24): the key must exist
    # in the encoded spec dict, not merely resolve to a value via the
    # runner's .get(..., False) default.
    assert "swarm_payload" in spec
    assert spec["swarm_payload"] is True
    assert "backend_url" in spec
    assert spec["backend_url"] == "http://203.0.113.11:8082/v1"
    assert spec["acquire_lease"] is True
    # The encoded fields make the gw_agent.py:1877 formula compute True.
    assert _is_swarm_from_spec(spec) is True


def test_stopgap_spec_carries_swarm_payload_false_key_present(shaper_mocks, tmp_path):
    spec = _dispatched_spec(shaper_mocks, "fixer_stopgap", tmp_path)
    # PRESENCE (mandatory): the key is present with value False for entries
    # without a swarm_payload registry key — default-False is load-bearing.
    assert "swarm_payload" in spec
    assert spec["swarm_payload"] is False
    assert "backend_url" in spec
    assert spec["backend_url"] is None
    assert spec["acquire_lease"] is True
    # The encoded fields make the gw_agent.py:1877 formula compute False.
    assert _is_swarm_from_spec(spec) is False


def test_stopgap_spec_other_fields_byte_identical_to_today(shaper_mocks, tmp_path):
    """Invariant: no spec change for registry entries without swarm_payload —
    every pre-existing field is present and unchanged; only the new key is
    added (value False)."""
    spec = _dispatched_spec(shaper_mocks, "fixer_stopgap", tmp_path)
    for key in (
        "agent_type", "target_id", "repo", "engine", "model",
        "backend_url", "acquire_lease", "timeout_s", "system", "prompt",
        "cwd", "task_id", "base_branch", "worktree_required",
    ):
        assert key in spec
    assert spec["agent_type"] == "fixer_stopgap"
    assert spec["model"] == "haiku"
    assert spec["timeout_s"] == 60
    assert spec["engine"] == "local-fixer"
    assert spec["backend_url"] is None
    assert spec["acquire_lease"] is True


# ---------------------------------------------------------------------------
# Doorman enqueue-decision log line (item 2)
# ---------------------------------------------------------------------------

def _make_state() -> _NodeState:
    return _NodeState("http://203.0.113.11:8081")


def _grant_lease(state: _NodeState, work_id: str, principal: str, ttl_sec: int = 600) -> None:
    state.leases[work_id] = {
        "acquired_at": time.time(),
        "ttl_sec": ttl_sec,
        "reason": "test",
        "role": "test",
        "class": "protected",
        "principal": principal,
    }


def test_acquire_or_defer_enqueue_emits_warning(caplog):
    """The defer/wait-list (enqueue) path emits exactly one warning-level
    line carrying work_id, principal, gating-lease identity, and wait-list
    depth."""
    state = _make_state()
    _grant_lease(state, "gating-job", "flip-controller")

    with caplog.at_level(logging.WARNING, logger="doorman-server"):
        resp, release = state.acquire_or_defer(
            work_id="deferred-job", reason="r", role="fixer",
            lease_class="deferrable", principal="lapis-pm",
        )

    assert resp is not None and resp["status"] == "pending_defer"
    assert release is None
    assert len(state.wait_list) == 1

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "deferred-job" in msg            # work_id
    assert "lapis-pm" in msg                # requesting principal
    assert "gating-job" in msg              # gating lease id
    assert "flip-controller" in msg         # gating lease principal
    assert "wait_list_depth=1" in msg       # wait-list depth


def test_acquire_or_defer_grant_path_emits_no_enqueue_warning(caplog):
    """The grant path (no gating lease, fresh request) logs nothing new —
    grants stay visible via the idle-log, not this line."""
    state = _make_state()

    with caplog.at_level(logging.WARNING, logger="doorman-server"):
        resp, release = state.acquire_or_defer(
            work_id="free-job", reason="r", role="fixer",
            lease_class="deferrable", principal="lapis-pm",
        )

    assert resp is None and release is None
    assert state.wait_list == {}
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == []


def test_acquire_or_defer_brake_gate_enqueues_with_brake_identity(caplog):
    """Brake-gated enqueue names the brake as the gating identity (no
    protected lease present) and still logs the one warning line."""
    state = _make_state()
    state.brake_expires_at = time.time() + 300
    state.brake_reason = "test-brake"

    with caplog.at_level(logging.WARNING, logger="doorman-server"):
        resp, _ = state.acquire_or_defer(
            work_id="braked-job", reason="r", role="fixer",
            lease_class="deferrable", principal=None,
        )

    assert resp is not None and resp["status"] == "pending_defer"
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "braked-job" in msg
    assert "brake" in msg
    assert GHOST_PRINCIPAL in msg   # principal=None -> ghost principal