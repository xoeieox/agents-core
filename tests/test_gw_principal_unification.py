"""Tests for gw-gate-principal-unification-agents-core-v0.

Covers AC1-AC7: shared gw_principal threading through the DeliberationRequest,
call_gw_agent reference leg, orchestrator span-hold, Facets env, Council hold +
adapter, and drain-gate integration (same-principal admit vs split/ghost CONTENDED).

All tests are DARK — they verify behaviour without requiring gw_principal to be
set by the caller. With gw_principal=None every code path is byte-identical to
today; the new field / param simply does nothing.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agents_core.shared_deliberation.envelope import DeliberationRequest
from agents_core.doorman_server import (
    CONTENDED,
    GHOST_PRINCIPAL,
    _NodeState,
)


# ---------------------------------------------------------------------------
# AC1 — DeliberationRequest.gw_principal field round-trip
# ---------------------------------------------------------------------------

class TestDeliberationRequestGwPrincipal:
    def test_field_defaults_to_none(self):
        req = DeliberationRequest(text="t", context={})
        assert req.gw_principal is None

    def test_field_round_trips_via_to_dict(self):
        req = DeliberationRequest(text="t", context={}, gw_principal="gw-delib-abc123")
        d = req.to_dict()
        assert "gw_principal" in d
        assert d["gw_principal"] == "gw-delib-abc123"

    def test_none_round_trips_via_to_dict(self):
        req = DeliberationRequest(text="t", context={})
        d = req.to_dict()
        assert "gw_principal" in d
        assert d["gw_principal"] is None

    def test_existing_constructors_unchanged(self):
        # All previously valid constructor forms still compile and yield gw_principal=None
        req = DeliberationRequest(
            text="review spec",
            context={"repo": "agents-core"},
            triage="full",
            caller="spec-review",
            council_voicing="gravitywell",
            facets_operator="gravitywell",
        )
        assert req.gw_principal is None

    def test_asdict_includes_gw_principal(self):
        from dataclasses import asdict
        req = DeliberationRequest(text="t", context={}, gw_principal="P")
        d = asdict(req)
        assert d["gw_principal"] == "P"


# ---------------------------------------------------------------------------
# AC2 — call_gw_agent(principal=...) sends it to doorman acquire
# ---------------------------------------------------------------------------

class TestCallGwAgentPrincipal:
    def _make_gw_response(self, content="done"):
        return {
            "choices": [{
                "message": {"content": content, "tool_calls": []},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 10},
        }

    def test_principal_forwarded_to_acquire(self):
        """call_gw_agent(principal='P') passes principal='P' to doorman.acquire."""
        from agents_core.gw_agent import call_gw_agent

        with patch("agents_core.doorman_client.DoormanClient") as mock_cls, \
             patch("requests.post") as mock_post:
            mock_dc = MagicMock()
            mock_cls.return_value = mock_dc
            mock_dc.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = self._make_gw_response()

            call_gw_agent(prompt="review", principal="gw-delib-xyz", timeout=10)

            kw = mock_dc.acquire.call_args.kwargs
            assert kw.get("principal") == "gw-delib-xyz", (
                f"expected principal='gw-delib-xyz' in acquire kwargs, got {kw}"
            )

    def test_no_principal_sends_none_ghost(self):
        """call_gw_agent() without principal sends principal=None (ghost) — byte-identical to today."""
        from agents_core.gw_agent import call_gw_agent

        with patch("agents_core.doorman_client.DoormanClient") as mock_cls, \
             patch("requests.post") as mock_post:
            mock_dc = MagicMock()
            mock_cls.return_value = mock_dc
            mock_dc.acquire.return_value = {"status": "serving"}
            mock_post.return_value.json.return_value = self._make_gw_response()

            call_gw_agent(prompt="review", timeout=10)

            kw = mock_dc.acquire.call_args.kwargs
            # principal should be None (ghost) — the default
            assert kw.get("principal") is None, (
                f"expected principal=None (ghost) when not set; got {kw}"
            )


# ---------------------------------------------------------------------------
# AC3 — Orchestrator span-hold uses shared principal
# ---------------------------------------------------------------------------

class TestOrchestratorSpanHoldPrincipal:
    def _make_mock_doorman(self, acquire_returns=None):
        m = MagicMock()
        m.acquire = MagicMock(return_value=acquire_returns or {"status": "serving"})
        m.release = MagicMock()
        m.close = MagicMock()
        return m

    @pytest.mark.asyncio
    async def test_span_hold_uses_gw_principal_when_set(self, monkeypatch):
        """With gw_principal='P', span-hold initial acquire and refresh both send principal='P'."""
        from agents_core.shared_deliberation.orchestrator import run_deliberation, init_facets_semaphore
        import agents_core.shared_deliberation.orchestrator as orch

        monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")
        monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")
        monkeypatch.setenv("SHARED_DELIB_SPAN_REFRESH_S", "999")  # suppress refresh during test
        init_facets_semaphore(2)

        mock_dc = self._make_mock_doorman()
        with patch("agents_core.doorman_client.DoormanClient", return_value=mock_dc), \
             patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
            req = DeliberationRequest(
                text="t", context={}, council_voicing="gravitywell", gw_principal="gw-delib-P"
            )
            await run_deliberation(req)

        # Initial acquire call: principal must be 'gw-delib-P'
        first_call_kw = mock_dc.acquire.call_args_list[0].kwargs
        assert first_call_kw.get("principal") == "gw-delib-P", (
            f"span-hold initial acquire must use gw_principal; got {first_call_kw}"
        )
        assert first_call_kw.get("lease_kind") == "coordination", (
            f"span-hold must keep lease_kind='coordination'; got {first_call_kw}"
        )

    @pytest.mark.asyncio
    async def test_span_hold_uses_span_work_id_when_no_gw_principal(self, monkeypatch):
        """With gw_principal=None, span-hold sends principal=span_work_id (byte-identical to today)."""
        from agents_core.shared_deliberation.orchestrator import run_deliberation, init_facets_semaphore

        monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")
        monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")
        monkeypatch.setenv("SHARED_DELIB_SPAN_REFRESH_S", "999")
        init_facets_semaphore(2)

        mock_dc = self._make_mock_doorman()
        with patch("agents_core.doorman_client.DoormanClient", return_value=mock_dc), \
             patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
            req = DeliberationRequest(
                text="t", context={}, council_voicing="gravitywell"
                # gw_principal absent → None
            )
            await run_deliberation(req)

        first_call_kw = mock_dc.acquire.call_args_list[0].kwargs
        principal = first_call_kw.get("principal", "")
        assert principal.startswith("shared-delib-"), (
            f"without gw_principal, span-hold must use shared-delib-<id>; got {principal!r}"
        )
        assert first_call_kw.get("lease_kind") == "coordination"


# ---------------------------------------------------------------------------
# AC4 — GW_GATE_PRINCIPAL injected into Facets env
# ---------------------------------------------------------------------------

class TestFacetsEnvGwGatePrincipal:
    def test_gw_gate_principal_set_in_env_when_gw_principal_present(self, tmp_path):
        """_run_facets_subprocess includes GW_GATE_PRINCIPAL=P in facets_env when gw_principal='P'."""
        from agents_core.shared_deliberation.orchestrator import _run_facets_subprocess

        captured_envs = []

        def fake_run(argv, **kwargs):
            captured_envs.append(kwargs.get("env", {}))
            m = MagicMock()
            m.returncode = 0
            import json
            m.stdout = json.dumps({"deliberation_id": "test-id"})
            m.stderr = ""
            return m

        facets_repo = tmp_path / "facets"
        facets_repo.mkdir()

        with patch("subprocess.run", side_effect=fake_run):
            _run_facets_subprocess(
                "text", {}, "gravitywell", facets_repo,
                grounding_result_file=None, gw_principal="gw-delib-P"
            )

        assert len(captured_envs) == 1
        env = captured_envs[0]
        assert "GW_GATE_PRINCIPAL" in env, "GW_GATE_PRINCIPAL must be in facets_env when gw_principal is set"
        assert env["GW_GATE_PRINCIPAL"] == "gw-delib-P"

    def test_gw_gate_principal_absent_when_gw_principal_none(self, tmp_path):
        """_run_facets_subprocess does NOT set GW_GATE_PRINCIPAL when gw_principal=None."""
        from agents_core.shared_deliberation.orchestrator import _run_facets_subprocess

        captured_envs = []

        def fake_run(argv, **kwargs):
            captured_envs.append(kwargs.get("env", {}))
            m = MagicMock()
            m.returncode = 0
            import json
            m.stdout = json.dumps({"deliberation_id": "test-id"})
            m.stderr = ""
            return m

        facets_repo = tmp_path / "facets"
        facets_repo.mkdir()

        with patch("subprocess.run", side_effect=fake_run):
            _run_facets_subprocess(
                "text", {}, "gravitywell", facets_repo,
                grounding_result_file=None, gw_principal=None
            )

        assert len(captured_envs) == 1
        env = captured_envs[0]
        assert "GW_GATE_PRINCIPAL" not in env, (
            "GW_GATE_PRINCIPAL must NOT be in facets_env when gw_principal is None"
        )


# ---------------------------------------------------------------------------
# AC5 — Council: cmd_submit persists gw_principal; hold + adapter use it
# ---------------------------------------------------------------------------

class TestCouncilGwPrincipal:
    def _make_submit_args(self, gw_principal=None, **kwargs):
        defaults = {
            "decision": "Should we unify principals?",
            "mode": "deliberation",
            "n": None,
            "turns": 8,
            "voicing": "gravitywell",
            "with_entity": None,
            "narrator": False,
            "narrator_voice": None,
            "no_queue": False,
            "notify": False,
            "gw_principal": gw_principal,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_cmd_submit_persists_gw_principal(self, tmp_path, monkeypatch):
        """cmd_submit with args.gw_principal='P' writes run['gw_principal']='P' to YAML."""
        import yaml
        from agents_core.council import cli as council_cli
        import agents_core.claude_queue as cq_mod

        monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
        monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
        monkeypatch.setattr(council_cli, "build_roster", lambda: [])
        monkeypatch.setattr(
            council_cli, "select_entities",
            lambda **kw: {"selected": ["ada-lovelace-canonical", "benjamin-franklin-canonical"], "reasoning": "test"}
        )

        class _FakeQueue:
            def submit(self, task, task_id=None):
                return task_id or "fake-id"

        monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

        args = self._make_submit_args(gw_principal="gw-delib-P")
        rc = council_cli.cmd_submit(args)

        assert rc == 0
        yamls = list(tmp_path.glob("*.yaml"))
        assert len(yamls) == 1
        run_data = yaml.safe_load(yamls[0].read_text())
        assert run_data.get("gw_principal") == "gw-delib-P", (
            f"run['gw_principal'] must be 'gw-delib-P'; got {run_data.get('gw_principal')!r}"
        )

    def test_cmd_submit_persists_gw_principal_none_when_absent(self, tmp_path, monkeypatch):
        """cmd_submit without args.gw_principal persists None (byte-identical to today)."""
        import yaml
        from agents_core.council import cli as council_cli
        import agents_core.claude_queue as cq_mod

        monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
        monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
        monkeypatch.setattr(council_cli, "build_roster", lambda: [])
        monkeypatch.setattr(
            council_cli, "select_entities",
            lambda **kw: {"selected": ["ada-lovelace-canonical", "benjamin-franklin-canonical"], "reasoning": "test"}
        )

        class _FakeQueue:
            def submit(self, task, task_id=None):
                return task_id or "fake-id"

        monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

        # args WITHOUT gw_principal attribute (mimics old call sites)
        args = argparse.Namespace(
            decision="test", mode="deliberation", n=None, turns=8,
            voicing="gravitywell", with_entity=None, narrator=False,
            narrator_voice=None, no_queue=False, notify=False,
        )
        rc = council_cli.cmd_submit(args)

        assert rc == 0
        yamls = list(tmp_path.glob("*.yaml"))
        run_data = yaml.safe_load(yamls[0].read_text())
        assert run_data.get("gw_principal") is None

    def test_submit_council_sets_gw_principal_on_namespace(self, monkeypatch):
        """_submit_council(text, voicing, gw_principal='P') sets args.gw_principal='P' before cmd_submit."""
        from agents_core.shared_deliberation.orchestrator import _submit_council

        captured_args = []

        def fake_cmd_submit(args):
            captured_args.append(args)
            # Print what cmd_submit would print so the regex finds task_id
            print("task_id=test-run-001")

        monkeypatch.setattr(
            "agents_core.council.cli.cmd_submit",
            fake_cmd_submit,
        )
        monkeypatch.setattr(
            "agents_core.council.cli.DEFAULT_TURNS",
            8,
        )

        _submit_council("decide this", "gravitywell", "gw-delib-P")

        assert len(captured_args) == 1
        args = captured_args[0]
        assert args.gw_principal == "gw-delib-P", (
            f"args.gw_principal must be 'gw-delib-P'; got {args.gw_principal!r}"
        )

    def test_submit_council_none_gw_principal_passes_none(self, monkeypatch):
        """_submit_council without gw_principal sets args.gw_principal=None."""
        from agents_core.shared_deliberation.orchestrator import _submit_council

        captured_args = []

        def fake_cmd_submit(args):
            captured_args.append(args)
            print("task_id=test-run-002")

        monkeypatch.setattr("agents_core.council.cli.cmd_submit", fake_cmd_submit)
        monkeypatch.setattr("agents_core.council.cli.DEFAULT_TURNS", 8)

        _submit_council("decide this", "gravitywell")

        args = captured_args[0]
        assert args.gw_principal is None

    def test_build_adapter_uses_gw_principal(self):
        """_build_adapter with gw_principal='P' yields GravityWellAdapter(principal='P')."""
        from agents_core.council.cli import _build_adapter

        captured = []

        class FakeGWAdapter:
            def __init__(self, temperature=None, principal=None):
                captured.append({"temperature": temperature, "principal": principal})

        class FakeLlamaAdapter:
            pass

        class FakeClaudeAdapter:
            pass

        with patch("agents_core.council.cli.GravityWellAdapter", FakeGWAdapter):
            _build_adapter("gravitywell", FakeClaudeAdapter, FakeLlamaAdapter,
                           run_id="run-001", gw_principal="gw-delib-P")

        assert len(captured) == 1
        assert captured[0]["principal"] == "gw-delib-P", (
            f"GravityWellAdapter must be built with principal='gw-delib-P'; got {captured[0]['principal']!r}"
        )

    def test_build_adapter_falls_back_to_council_delib_run_id_when_no_gw_principal(self):
        """_build_adapter without gw_principal yields principal='council-delib-<run_id>' (byte-identical)."""
        from agents_core.council.cli import _build_adapter

        captured = []

        class FakeGWAdapter:
            def __init__(self, temperature=None, principal=None):
                captured.append(principal)

        with patch("agents_core.council.cli.GravityWellAdapter", FakeGWAdapter):
            _build_adapter("gravitywell", object, object, run_id="run-001")

        assert captured[0] == "council-delib-run-001"

    def test_council_hold_principal_uses_gw_principal_from_run(self, tmp_path, monkeypatch):
        """Council deliberation hold acquires with gw_principal from run dict when set."""
        import yaml
        from agents_core.council import cli as council_cli

        # Write a minimal run YAML with gw_principal set
        run_id = "test-run-principal"
        run = {
            "run_id": run_id,
            "status": "deliberating",
            "mode": "deliberation",
            "decision": "test",
            "voicing": "gravitywell",
            "selected_entities": [],
            "turns": [],
            "turns_cap": 2,
            "context_gathered": {"terms": [], "hits": []},
            "gw_principal": "gw-delib-P",
            "selection_reasoning": "",
            "selection_voicing": "gravitywell",
            "selection_degraded": False,
            "paid_spend": False,
        }
        (tmp_path / f"{run_id}.yaml").write_text(yaml.dump(run))

        monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
        monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
        monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

        acquired_principals = []

        class FakeDoorman:
            def acquire(self, node, work_id, ttl_sec, reason, timeout=None, principal=None, **kwargs):
                acquired_principals.append(principal)
                return {"status": "serving"}

            def release(self, node, work_id):
                pass

            def close(self):
                pass

        with patch("agents_core.doorman_client.DoormanClient", return_value=FakeDoorman()), \
             patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0), \
             patch("agents_core.council.cli._apply_position_cast_tail"), \
             patch("agents_core.council.cli._calculate_paid_spend", return_value=False):
            council_cli.run_deliberation(run_id)

        # The initial hold acquire must use gw_principal='gw-delib-P'
        assert "gw-delib-P" in acquired_principals, (
            f"council hold must acquire with principal='gw-delib-P'; got {acquired_principals}"
        )

    def test_council_hold_falls_back_to_council_delib_run_id_when_no_gw_principal(self, tmp_path, monkeypatch):
        """Without gw_principal in run dict, council hold uses 'council-delib-<run_id>' (byte-identical)."""
        import yaml
        from agents_core.council import cli as council_cli

        run_id = "test-run-no-principal"
        run = {
            "run_id": run_id,
            "status": "deliberating",
            "mode": "deliberation",
            "decision": "test",
            "voicing": "gravitywell",
            "selected_entities": [],
            "turns": [],
            "turns_cap": 2,
            "context_gathered": {"terms": [], "hits": []},
            "selection_reasoning": "",
            "selection_voicing": "gravitywell",
            "selection_degraded": False,
            "paid_spend": False,
            # gw_principal absent → None
        }
        (tmp_path / f"{run_id}.yaml").write_text(yaml.dump(run))

        monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
        monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")
        monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
        monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

        acquired_principals = []

        class FakeDoorman:
            def acquire(self, node, work_id, ttl_sec, reason, timeout=None, principal=None, **kwargs):
                acquired_principals.append(principal)
                return {"status": "serving"}

            def release(self, node, work_id):
                pass

            def close(self):
                pass

        with patch("agents_core.doorman_client.DoormanClient", return_value=FakeDoorman()), \
             patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0), \
             patch("agents_core.council.cli._apply_position_cast_tail"), \
             patch("agents_core.council.cli._calculate_paid_spend", return_value=False):
            council_cli.run_deliberation(run_id)

        assert f"council-delib-{run_id}" in acquired_principals, (
            f"without gw_principal, hold must use 'council-delib-{run_id}'; got {acquired_principals}"
        )


# ---------------------------------------------------------------------------
# AC6 — Drain-gate integration: same-principal admits; split/ghost CONTENDED
# ---------------------------------------------------------------------------

@pytest.fixture
def node():
    """In-process _NodeState with ensure_serving always returning True."""
    state = _NodeState(gw_url="http://mock.internal/", node_name="gravitywell")
    with patch.object(state, "_is_serving", return_value=True), \
         patch("subprocess.run"):
        yield state


class TestDrainGatePrincipalUnification:
    """AC6: shared-principal admit vs split/ghost CONTENDED — no lease_kind change does the work."""

    def test_same_principal_voice_admitted_when_reference_leg_holds(self, node):
        """Reference leg (inference, principal=P) + voice (require_drain_clear, principal=P) → GRANTED."""
        P = f"gw-delib-{uuid.uuid4().hex[:8]}"
        ref_wid = f"ref-{uuid.uuid4().hex[:6]}"
        voice_wid = f"voice-{uuid.uuid4().hex[:6]}"

        # Reference leg acquires (inference, no drain constraint — it's the first)
        with node.lock:
            ok_ref = node.acquire_lease(
                ref_wid, 60, "gw_agent", role="worker",
                principal=P, lease_kind="inference",
            )
        assert ok_ref is True

        # Voice/persona tries to ride along under the same principal
        with node.lock:
            ok_voice = node.acquire_lease(
                voice_wid, 60, "gw_voice", role="worker",
                principal=P, lease_kind="inference",
                require_drain_clear=True,
            )
        assert ok_voice is True, (
            f"same-principal voice must be GRANTED (ride-along); got {ok_voice!r}"
        )

        with node.lock:
            node.leases.pop(ref_wid, None)
            node.leases.pop(voice_wid, None)

    def test_same_principal_with_coordination_span_still_admitted(self, node):
        """Span (coordination, principal=P) + reference (inference, P) + voice (require_drain_clear, P) → GRANTED."""
        P = f"gw-delib-{uuid.uuid4().hex[:8]}"
        span_wid = f"span-{uuid.uuid4().hex[:6]}"
        ref_wid = f"ref-{uuid.uuid4().hex[:6]}"
        voice_wid = f"voice-{uuid.uuid4().hex[:6]}"

        # Span hold (coordination — excluded from drain-gate count)
        with node.lock:
            ok_span = node.acquire_lease(
                span_wid, 60, "span-hold", role="worker",
                principal=P, lease_kind="coordination",
            )
        assert ok_span is True

        # Reference leg (inference)
        with node.lock:
            ok_ref = node.acquire_lease(
                ref_wid, 60, "gw_agent", role="worker",
                principal=P, lease_kind="inference",
            )
        assert ok_ref is True

        # Voice (require_drain_clear, same principal)
        with node.lock:
            ok_voice = node.acquire_lease(
                voice_wid, 60, "gw_voice", role="worker",
                principal=P, lease_kind="inference",
                require_drain_clear=True,
            )
        assert ok_voice is True, (
            f"voice must be GRANTED under shared principal even with span+ref leases; got {ok_voice!r}"
        )

        with node.lock:
            for wid in (span_wid, ref_wid, voice_wid):
                node.leases.pop(wid, None)

    def test_different_principal_voice_contended(self, node):
        """Reference leg (inference, principal=P) blocks voice with different principal → CONTENDED."""
        P = f"gw-delib-{uuid.uuid4().hex[:8]}"
        Q = f"gw-delib-{uuid.uuid4().hex[:8]}"
        ref_wid = f"ref-{uuid.uuid4().hex[:6]}"
        voice_wid = f"voice-{uuid.uuid4().hex[:6]}"

        with node.lock:
            node.acquire_lease(
                ref_wid, 60, "gw_agent", role="worker",
                principal=P, lease_kind="inference",
            )

        with node.lock:
            ok = node.acquire_lease(
                voice_wid, 60, "gw_voice", role="worker",
                principal=Q, lease_kind="inference",
                require_drain_clear=True,
            )
        assert ok is CONTENDED, (
            f"different-principal voice must be CONTENDED; got {ok!r}"
        )

        with node.lock:
            node.leases.pop(ref_wid, None)

    def test_ghost_reference_leg_contends_named_principal_voice(self, node, caplog):
        """Ghost reference leg (no principal) causes CONTENDED for a named voice — the 5th-wedge scenario."""
        import logging

        ghost_wid = f"ghost-ref-{uuid.uuid4().hex[:6]}"
        voice_wid = f"voice-{uuid.uuid4().hex[:6]}"
        P = f"gw-delib-{uuid.uuid4().hex[:8]}"

        # Register ghost reference leg directly (no principal → GHOST_PRINCIPAL)
        with node.lock:
            node.leases[ghost_wid] = {
                "acquired_at": time.time(),
                "ttl_sec": 60,
                "reason": "gw_agent",
                "role": "worker",
                "principal": GHOST_PRINCIPAL,
                "lease_kind": "inference",
            }

        # Voice tries to acquire with a named principal → ghost blocks it
        with caplog.at_level(logging.CRITICAL, logger="doorman-server"):
            with node.lock:
                ok = node.acquire_lease(
                    voice_wid, 60, "gw_voice", role="worker",
                    principal=P, lease_kind="inference",
                    require_drain_clear=True,
                )

        assert ok is CONTENDED, (
            f"ghost reference leg must cause CONTENDED for named-principal voice; got {ok!r}"
        )
        assert any(
            "ghost_lease_counted" in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.CRITICAL
        ), "ghost_lease_counted CRITICAL log must fire"

        with node.lock:
            node.leases.pop(ghost_wid, None)

    def test_coordination_span_alone_does_not_block_voice(self, node):
        """Coordination span alone (no inference lease) does NOT block a voice — coordination excluded."""
        P = f"gw-delib-{uuid.uuid4().hex[:8]}"
        span_wid = f"span-{uuid.uuid4().hex[:6]}"
        voice_wid = f"voice-{uuid.uuid4().hex[:6]}"

        with node.lock:
            node.acquire_lease(
                span_wid, 60, "span-hold", role="worker",
                principal=P, lease_kind="coordination",
            )

        # Voice from a DIFFERENT principal — coordination span is excluded from drain-gate
        with node.lock:
            ok = node.acquire_lease(
                voice_wid, 60, "gw_voice", role="worker",
                principal="other-delib", lease_kind="inference",
                require_drain_clear=True,
            )
        assert ok is True, (
            f"coordination span alone must not block voice (excluded from drain-gate); got {ok!r}"
        )

        with node.lock:
            node.leases.pop(span_wid, None)
            node.leases.pop(voice_wid, None)


# ---------------------------------------------------------------------------
# AC7 — drain-count unchanged: inference leases counted, coordination excluded
# ---------------------------------------------------------------------------

class TestDrainCountInvariant:
    """Council hold stays inference (counted); span stays coordination (uncounted for drain-gate)."""

    def test_council_hold_inference_counted_in_drain_count(self):
        """Council deliberation hold (inference) is still counted by /v0/drain-count."""
        from agents_core.doorman_server import create_app
        from fastapi.testclient import TestClient

        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch("subprocess.run"):
            app = create_app(gw_url="http://mock.internal/")
            http = TestClient(app, raise_server_exceptions=True)

            # Acquire a council-hold style lease (inference, named principal)
            r1 = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "council-delib-testrun",
                "ttl_sec": 60,
                "reason": "council-deliberation-hold",
                "role": "worker",
                "principal": "council-delib-testrun",
                "lease_kind": "inference",
            })
            assert r1.status_code == 200

            drain = http.get("/v0/drain-count?node=gravitywell")
            assert drain.status_code == 200
            count = drain.json()["drain_count"]
            assert count >= 1, (
                f"council inference hold must be counted in drain_count; got {count}"
            )

            http.post("/lease/release", json={"node": "gravitywell", "work_id": "council-delib-testrun"})

    def test_span_coordination_excluded_from_drain_gate_but_counted_in_drain_count(self):
        """Span hold (coordination) is excluded from drain-gate check but still counted by /v0/drain-count."""
        from agents_core.doorman_server import create_app
        from fastapi.testclient import TestClient

        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch("subprocess.run"):
            app = create_app(gw_url="http://mock.internal/")
            http = TestClient(app, raise_server_exceptions=True)

            P = f"gw-delib-{uuid.uuid4().hex[:8]}"

            # Span hold (coordination)
            r_span = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": f"shared-delib-{P}",
                "ttl_sec": 60,
                "reason": "shared-deliberation-span-hold",
                "role": "worker",
                "principal": P,
                "lease_kind": "coordination",
            })
            assert r_span.status_code == 200

            # drain-count endpoint includes coordination per #117 (flip-protection)
            drain = http.get("/v0/drain-count?node=gravitywell")
            assert drain.status_code == 200
            count = drain.json()["drain_count"]
            assert count >= 1, (
                f"coordination span must be included in drain_count (flip-protection); got {count}"
            )

            # But a new voice with require_drain_clear and DIFFERENT principal still gets through
            # because coordination is excluded from drain-gate (AC2 of #117)
            r_other = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "other-voice-wid",
                "ttl_sec": 60,
                "reason": "test",
                "role": "worker",
                "principal": "other-group",
                "lease_kind": "inference",
                "require_drain_clear": True,
            })
            # coordination span alone does not gate a different-principal voice
            assert r_other.status_code == 200
            body = r_other.json()
            assert body.get("status") == "serving", (
                f"coordination-only span must not block a named-principal voice; got {body}"
            )

            http.post("/lease/release", json={"node": "gravitywell", "work_id": f"shared-delib-{P}"})
            http.post("/lease/release", json={"node": "gravitywell", "work_id": "other-voice-wid"})
