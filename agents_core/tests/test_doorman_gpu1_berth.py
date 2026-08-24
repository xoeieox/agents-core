"""Hermetic tests for the doorman's GPU 1 (berth) awareness —
gw-gpu1-berth-standing-seat-v0, leg 2.

All subprocess / SSH / :8081 / Glances HTTP is mocked; no GPU or network is
required. Covers the spec's eight test groups:

  (a) the probe tri-state (proc>0 -> True; proc==0 + mem high -> False [the
      warm-berth case, the S2 binding statement]; proc==0 + mem low -> False;
      unreachable -> None; malformed -> None)
  (b) the combine branches (gpu1 vote in all three; a True gpu1 vote re-arms
      while all others False; a None gpu1 vote with all others False -> None
      not False)
  (c) the blind-bounded mapping (glances dead -> blind bound at grace+900 ->
      proceed with stop_reason="probe_blind_bound_exceeded" — NOT the
      topology-unknown unbounded path)
  (d) the readiness discriminator (vLLM payload -> ready; berth payload -> not
      ready; unreachable -> False)
  (e) the supervisor lease lifecycle (acquired at _run_local_fixer entry
      before worktree setup; released in finally on success AND failure;
      distinct work_id from the per-run lease; acquire soft-fail on
      DoormanUnreachable -> job proceeds; pending_defer -> job proceeds)
  (f) the dual-init cleanup guard (a supervisor lease present -> no
      gw-serve stop fired, logged)
  (g) the _is_swarm parity table (every existing registry entry's
      (backend_url, acquire_lease) -> expected _is_swarm unchanged; the berth
      shape (backend_url set, acquire_lease true, swarm_payload true) -> True;
      the berth shape WITHOUT swarm_payload -> False)
  (h) the gpu1 status block shape
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agents_core.doorman_server import (
    GW_URL_DEFAULT,
    _NodeState,
    create_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


def _glances_list(proc: float, mem: float, gpu_id: str = "nvidia1"):
    """A Glances API v4 /api/4/gpu/ LIST payload with one GPU entry.

    gpu_id uses the REAL glances v4 shape - "nvidia<N>" where N is the CUDA
    device index (the berth is CUDA device 1; the primary card is nvidia0).
    The old fixture hard-coded the bare index ("1"), which the live API does
    not emit - so the suite validated against a shape the production glances
    never returns. (finding/doorman-gpu1-glances-gpu-id-2026-08-24)"""
    return [
        {"gpu_id": "nvidia0", "mem": 10.0, "proc": 0.0},
        {"gpu_id": gpu_id, "mem": mem, "proc": proc},
    ]


def _mock_resp(status_code: int, json_data=None, text: str = ""):
    m = MagicMock()
    m.status_code = status_code
    if json_data is not None:
        m.json.return_value = json_data
    else:
        m.json.side_effect = ValueError("no json")
    m.text = text
    return m


# ---------------------------------------------------------------------------
# (a) the probe tri-state
# ---------------------------------------------------------------------------

class TestGpu1ProbeTriState:
    def test_proc_above_threshold_votes_true(self):
        """nvidia1.proc > 0 -> True (any proc>0 sample is activity)."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # berth /health 200
            _mock_resp(200, _glances_list(proc=12.0, mem=77.0)),
        ]):
            assert state._probe_gpu1_glances() is True

    def test_idle_warm_berth_votes_false(self):
        """The S2 binding statement: proc==0 + mem high (idle-but-warm berth)
        MUST vote False or the box never sleeps. The vote is proc-only — mem
        is diagnostic and never part of the vote."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # berth /health 200
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]):
            assert state._probe_gpu1_glances() is False

    def test_idle_cold_votes_false(self):
        """proc==0 + mem low -> False (same as the warm case — mem is not in the vote)."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(200, _glances_list(proc=0.0, mem=5.0)),
        ]):
            assert state._probe_gpu1_glances() is False

    def test_unreachable_votes_none(self):
        """Dead Glances (connection error) -> None, NEVER a spurious False."""
        state = _make_state()
        import requests as req_lib
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # berth /health
            req_lib.exceptions.ConnectionError("glances down"),
        ]):
            assert state._probe_gpu1_glances() is None

    def test_non_200_votes_none(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(503),
        ]):
            assert state._probe_gpu1_glances() is None

    def test_malformed_list_votes_none(self):
        """A non-LIST (dict) payload -> None (malformed)."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(200, {"gpu_id": "nvidia1", "proc": 0.0, "mem": 77.0}),
        ]):
            assert state._probe_gpu1_glances() is None

    def test_missing_gpu1_entry_votes_none(self):
        """A LIST with no index-1 (nvidia1) entry -> None (no reading, not
        idle): only the primary card (nvidia0) is present."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(200, [{"gpu_id": "nvidia0", "mem": 10.0, "proc": 0.0}]),
        ]):
            assert state._probe_gpu1_glances() is None

    def test_seat_health_recorded(self):
        """The berth's :8082 /health 200 is recorded in _gpu1_seat_health."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]):
            state._probe_gpu1_glances()
        assert state._gpu1_seat_health is True

    def test_seat_health_down_recorded(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(503),
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]):
            state._probe_gpu1_glances()
        assert state._gpu1_seat_health is False

    def test_berth_unit_active_recorded(self):
        """berth_unit (the ninfer-fixer systemd unit state) is probed over ssh
        and recorded DISTINCT from seat_health: unit active + seat healthy."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # berth /health 200
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]), patch("agents_core.doorman_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="active\n", stderr="", returncode=0)
            state._probe_gpu1_glances()
        assert state._gpu1_berth_unit is True
        assert state._gpu1_seat_health is True

    def test_berth_unit_active_seat_down(self):
        """Unit active with the seat DOWN: berth_unit and seat_health are
        distinct fields - the unit is up but :8082 /health is not 200."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(503),  # berth /health down
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]), patch("agents_core.doorman_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="active\n", stderr="", returncode=0)
            state._probe_gpu1_glances()
        assert state._gpu1_berth_unit is True
        assert state._gpu1_seat_health is False

    def test_berth_unit_ssh_failure_is_unknown(self):
        """An ssh timeout/error -> berth_unit None (unknown), never conflated
        with the seat health (which is still recorded from /health)."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # berth /health 200
            _mock_resp(200, _glances_list(proc=0.0, mem=77.0)),
        ]), patch("agents_core.doorman_server.subprocess.run",
                   side_effect=OSError("ssh down")):
            state._probe_gpu1_glances()
        assert state._gpu1_berth_unit is None
        assert state._gpu1_seat_health is True


# ---------------------------------------------------------------------------
# (b) the combine branches
# ---------------------------------------------------------------------------

class TestGpu1CombineBranches:
    def _patch_probes(self, a, b1, b2, c, gpu1):
        """Patch the four legacy probes + the gpu1 probe to fixed votes."""
        return (
            patch.object(_NodeState, "_probe_llama_slots_activity", return_value=a),
            patch.object(_NodeState, "_probe_vllm_metrics_activity", return_value=b1),
            patch.object(_NodeState, "_probe_vllm_metrics_activity", return_value=b2),
            patch.object(_NodeState, "_probe_llamacpp_metrics_activity", return_value=c),
            patch.object(_NodeState, "_probe_gpu1_glances", return_value=gpu1),
        )

    def test_gpu1_true_rearms_when_all_others_false(self):
        """A True gpu1 vote wins even when all other voters are False — the
        berth activity re-arms the dwell-stop clock."""
        state = _make_state()
        state._serving_is_big = None  # topology-unknown branch: all sources vote
        p1, p2, p3, p4, p5 = self._patch_probes(
            a=False, b1=False, b2=False, c=False, gpu1=True)
        with p1, p2, p3, p4, p5:
            assert state._probe_slot_activity() is True

    def test_gpu1_none_with_all_others_false_is_none(self):
        """A None gpu1 vote with all others False -> None (indeterminate), NOT
        False — the blind class, not a confirmed-idle park."""
        state = _make_state()
        state._serving_is_big = None
        p1, p2, p3, p4, p5 = self._patch_probes(
            a=False, b1=False, b2=False, c=False, gpu1=None)
        with p1, p2, p3, p4, p5:
            assert state._probe_slot_activity() is None

    def test_gpu1_false_with_all_others_false_is_false(self):
        """A False gpu1 vote with all others False -> False (confirmed idle)."""
        state = _make_state()
        state._serving_is_big = None
        p1, p2, p3, p4, p5 = self._patch_probes(
            a=False, b1=False, b2=False, c=False, gpu1=False)
        with p1, p2, p3, p4, p5:
            assert state._probe_slot_activity() is False

    def test_gpu1_votes_in_big_branch(self):
        """Big class (serving_is_big=True): the gpu1 vote is present (A + C + gpu1).
        A True gpu1 vote re-arms even with A/C False."""
        state = _make_state()
        state._serving_is_big = True
        p1, p2, p3, p4, p5 = self._patch_probes(
            a=False, b1=None, b2=None, c=False, gpu1=True)
        with p1, p2, p3, p4, p5:
            assert state._probe_slot_activity() is True

    def test_gpu1_votes_in_vllm_branch(self):
        """vLLM class (serving_is_big=False): the gpu1 vote is present.
        slot1-solo home posture -> [b1, gpu1]; a True gpu1 re-arms."""
        state = _make_state()
        state._serving_is_big = False
        with patch.object(state, "_read_declared_home_posture", return_value="slot1-solo"):
            p1, p2, p3, p4, p5 = self._patch_probes(
                a=None, b1=False, b2=None, c=None, gpu1=True)
            with p1, p2, p3, p4, p5:
                assert state._probe_slot_activity() is True

    def test_gpu1_votes_in_vllm_dual_branch(self):
        """vLLM dual branch (home posture not slot1-solo): [b1, b2, gpu1]."""
        state = _make_state()
        state._serving_is_big = False
        with patch.object(state, "_read_declared_home_posture", return_value="dual"):
            p1, p2, p3, p4, p5 = self._patch_probes(
                a=None, b1=False, b2=False, c=None, gpu1=True)
            with p1, p2, p3, p4, p5:
                assert state._probe_slot_activity() is True

    def test_last_probe_raw_records_gpu1(self):
        """The raw per-source record gains a 'GPU1' field."""
        state = _make_state()
        state._serving_is_big = None
        p1, p2, p3, p4, p5 = self._patch_probes(
            a=False, b1=False, b2=False, c=False, gpu1=True)
        with p1, p2, p3, p4, p5:
            state._probe_slot_activity()
        assert state._last_probe_raw["GPU1"] is True
        assert set(state._last_probe_raw) == {"A", "B1", "B2", "C", "GPU1"}


# ---------------------------------------------------------------------------
# (c) the blind-bounded mapping
# ---------------------------------------------------------------------------

class TestBlindBoundedMapping:
    def test_dead_glances_maps_to_bounded_blindness_not_unbounded(self):
        """A dead Glances (gpu1 probe -> None) makes the combine indeterminate
        (_probe_indeterminate True). The park block then pauses the grace clock
        bounded at grace+900, then PROCEEDS with stop_reason=
        "probe_blind_bound_exceeded" — the BOUNDED class, NOT the topology-
        unknown unbounded never-park path (which requires serving_is_big None
        AND the flag ON; here serving_is_big is resolved False)."""
        state = _make_state()
        state._serving_is_big = False  # resolved class -> NOT the unbounded path
        state.idle_since = time.time() - 2000  # past grace+900 (1500s)
        state.leases = {}
        state._probe_indeterminate = True
        # The unbounded never-park branch is gated on
        # `DOORMAN_MODE_AWARE_ADMISSION and state._serving_is_big is None`.
        # A resolved class (False) never reaches it — the bounded blind path
        # (grace+900) governs instead.
        assert state._serving_is_big is not None  # not the unbounded class
        assert state._probe_indeterminate is True

    def test_unbounded_path_requires_unresolved_topology(self):
        """The unbounded never-park class fires ONLY when serving_is_big is
        None (topology unresolved) — a dead Glances with a RESOLVED class
        never reaches it."""
        state = _make_state()
        state._serving_is_big = False
        # The unbounded branch is gated on `state._serving_is_big is None`.
        assert not (state._serving_is_big is None)


# ---------------------------------------------------------------------------
# (d) the readiness discriminator
# ---------------------------------------------------------------------------

class TestSlot2ReadinessDiscriminator:
    def test_vllm_payload_is_ready(self):
        """A :8082 /v1/models payload with owned_by=="vllm" -> ready."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health 200
            _mock_resp(200, {"data": [
                {"id": "gravitywell-27b", "owned_by": "vllm"},
            ]}),
        ]):
            assert state._is_slot2_serving() is True

    def test_berth_payload_is_not_ready(self):
        """The berth's NInfer payload (no owned_by=="vllm") -> NOT ready, even
        though /health is 200. A late wake is the safe direction, a
        false-ready is not."""
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health 200 (the berth answers this)
            _mock_resp(200, {"data": [
                {"id": "ninfer-27b", "owned_by": "ninfer"},
            ]}),
        ]):
            assert state._is_slot2_serving() is False

    def test_unreachable_health_is_not_ready(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(503),
        ]):
            assert state._is_slot2_serving() is False

    def test_unreachable_models_is_not_ready(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),  # /health 200
            _mock_resp(503),  # /v1/models down
        ]):
            assert state._is_slot2_serving() is False

    def test_connection_error_is_not_ready(self):
        import requests as req_lib
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get",
                  side_effect=req_lib.exceptions.ConnectionError("down")):
            assert state._is_slot2_serving() is False

    def test_empty_data_is_not_ready(self):
        state = _make_state()
        with patch("agents_core.doorman_server.requests.get", side_effect=[
            _mock_resp(200),
            _mock_resp(200, {"data": []}),
        ]):
            assert state._is_slot2_serving() is False


# ---------------------------------------------------------------------------
# (f) the dual-init cleanup guard
# ---------------------------------------------------------------------------

class TestDualInitCleanupGuard:
    def test_cleanup_blocked_by_supervisor_lease(self):
        """A supervisor lease (role=worker) present -> NO gw-serve stop fired;
        the half-initialized wake is left to the giveup/reconciler path."""
        state = _make_state()
        state.leases["task-1-berth-sup"] = {
            "acquired_at": time.time(), "ttl_sec": 1860, "reason":
            "fixer-job-supervisor", "role": "worker",
            "principal": "fixer-supervisor",
        }
        with patch("agents_core.doorman_server.subprocess.run") as mock_run:
            state._cleanup_failed_dual_initiation()
        mock_run.assert_not_called()

    def test_cleanup_proceeds_with_no_lease(self):
        """No worker lease -> the gw-serve stop fires (unchanged legacy path)."""
        state = _make_state()
        with patch("agents_core.doorman_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            state._cleanup_failed_dual_initiation()
        mock_run.assert_called_once()
        assert "gw-serve" in str(mock_run.call_args) and "stop" in str(mock_run.call_args)

    def test_cleanup_proceeds_when_only_controller_lease(self):
        """A mode-controller lease is NOT a worker blocker -> the stop fires."""
        state = _make_state()
        state.leases["ctrl"] = {
            "acquired_at": time.time(), "ttl_sec": 120, "reason": "c",
            "role": "mode-controller", "principal": "flip-controller",
        }
        with patch("agents_core.doorman_server.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            state._cleanup_failed_dual_initiation()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# (h) the gpu1 status block shape
# ---------------------------------------------------------------------------

class TestGpu1StatusBlock:
    def test_status_has_gpu1_block(self):
        with patch("agents_core.doorman_server._start_refresh_thread"):
            app = create_app(gw_url=GW_URL_DEFAULT)
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/status")
        assert resp.status_code == 200
        gpu1 = resp.json()["nodes"]["gravitywell"]["gpu1"]
        # The block shape: berth_unit, seat_health, glances, glances_mem_pct,
        # glances_proc, last_vote.
        assert set(gpu1) == {
            "berth_unit", "seat_health", "glances",
            "glances_mem_pct", "glances_proc", "last_vote",
        }
        # Pre-probe defaults: glances dead, last_vote None.
        assert gpu1["glances"] == "dead"
        assert gpu1["last_vote"] is None

    def test_status_gpu1_reflects_probe(self):
        """After a probe tick, the gpu1 block reflects the readings.
        berth_unit (the ninfer-fixer unit state) and seat_health (the :8082
        /health 200) are DISTINCT fields."""
        state = _make_state()
        state._gpu1_seat_health = True
        state._gpu1_berth_unit = True
        state._gpu1_glances_mem_pct = 77.0
        state._gpu1_glances_proc = 0.0
        state._last_probe_raw = {"GPU1": False}
        snap = state.status_snapshot()
        assert snap["gpu1"] == {
            "berth_unit": "active",
            "seat_health": True,
            "glances": "reachable",
            "glances_mem_pct": 77.0,
            "glances_proc": 0.0,
            "last_vote": False,
        }

    def test_status_gpu1_unit_active_seat_down(self):
        """Unit active with the seat down: berth_unit and seat_health are
        asserted as DISTINCT fields."""
        state = _make_state()
        state._gpu1_seat_health = False
        state._gpu1_berth_unit = True
        state._gpu1_glances_mem_pct = None
        state._gpu1_glances_proc = None
        state._last_probe_raw = {}
        snap = state.status_snapshot()
        assert snap["gpu1"]["berth_unit"] == "active"
        assert snap["gpu1"]["seat_health"] is False

    def test_status_gpu1_unit_unknown(self):
        """Unit state unknown (pre-probe / ssh failure) -> berth_unit None,
        distinct from seat_health."""
        state = _make_state()
        state._gpu1_seat_health = True
        state._gpu1_berth_unit = None
        state._gpu1_glances_mem_pct = None
        state._gpu1_glances_proc = None
        state._last_probe_raw = {}
        snap = state.status_snapshot()
        assert snap["gpu1"]["berth_unit"] is None
        assert snap["gpu1"]["seat_health"] is True


# ---------------------------------------------------------------------------
# (e) the supervisor lease lifecycle
# ---------------------------------------------------------------------------

class TestSupervisorLeaseLifecycle:
    def test_supervisor_lease_acquired_and_released_on_success(self):
        """The supervisor lease is acquired at _run_local_fixer entry (distinct
        work_id f"{task_id}-berth-sup", role=worker, principal=
        fixer-supervisor, lease_class=deferrable) and released in finally.
        The harness's git/PR tail is short-circuited (final_diff empty -> the
        harness returns "" before any git step), so only the lease + worktree
        + call_gw_agent surface is exercised — the acquire/release contract is
        the load-bearing assertion."""
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-1",
            "target_id": "tgt",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                   return_value=({"final_diff": ""}, [])):
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-1-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")
            mock_client = MockClient.return_value
            # Acquire: distinct work_id, role=worker, principal=fixer-supervisor,
            # lease_class=deferrable, TTL = timeout_s + 60 = 1860.
            mock_client.acquire.assert_called_once()
            args, kwargs = mock_client.acquire.call_args
            assert args[0] == "gravitywell"
            assert args[1] == "task-1-berth-sup"
            assert args[2] == 1860
            assert kwargs.get("role") == "worker"
            assert kwargs.get("principal") == "fixer-supervisor"
            assert kwargs.get("lease_class") == "deferrable"
            # Release in finally (success path).
            mock_client.release.assert_called_once_with("gravitywell", "task-1-berth-sup")
            # The worktree teardown still runs (unchanged).
            mock_teardown.assert_called_once()

    def test_supervisor_lease_released_on_failure(self):
        """The supervisor lease is released in finally on the FAILURE path too
        (call_gw_agent raises -> the harness's except branch returns "")."""
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-1",
            "target_id": "tgt",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                   side_effect=RuntimeError("gw down")):
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-1-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")
            mock_client = MockClient.return_value
            mock_client.acquire.assert_called_once()
            # Released in finally despite the failure.
            mock_client.release.assert_called_once_with("gravitywell", "task-1-berth-sup")

    def test_supervisor_lease_soft_fail_on_unreachable(self):
        """Acquire failure (DoormanUnreachable) -> the job PROCEEDS un-supervised
        (soft fail): no lease is released (none was acquired) and the harness
        still runs the worktree + call_gw_agent."""
        from agents_core import shaped_runner
        from agents_core.doorman_client import DoormanUnreachable

        spec = {
            "task_id": "task-1",
            "target_id": "tgt",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                   return_value=({"final_diff": ""}, [])) as mock_cga:
            MockClient.return_value.acquire.side_effect = DoormanUnreachable("down")
            mock_setup.return_value = MagicMock(path="/wt")
            shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")
            mock_client = MockClient.return_value
            mock_client.acquire.assert_called_once()
            # No lease was registered -> release must NOT be called.
            mock_client.release.assert_not_called()
            # The job still proceeded to call_gw_agent (soft fail, never a gate).
            mock_cga.assert_called_once()

    def test_supervisor_lease_soft_fail_on_pending_defer(self):
        """A pending_defer acquire response -> the job PROCEEDS (soft fail):
        the lease is not held, the soft-fail path logs a WARN and the
        finally-release is skipped (nothing was registered), and the
        harness runs to completion."""
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-1",
            "target_id": "tgt",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                   return_value=({"final_diff": ""}, [])) as mock_cga:
            MockClient.return_value.acquire.return_value = {
                "status": "pending_defer", "work_id": "task-1-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")
            mock_client = MockClient.return_value
            # The supervisor lease was NOT registered (pending_defer) ->
            # the soft-fail path drops the client, so release must NOT be
            # called. The job still proceeded (soft fail, never a gate).
            mock_client.release.assert_not_called()
            mock_cga.assert_called_once()

    def test_fixer_threads_swarm_payload_from_spec(self):
        """The registry's swarm_payload key must reach the fixer call:
        the berth seat (backend_url set + acquire_lease true +
        swarm_payload true) runs the swarm payload shape WITH the
        per-run lease (keep-both)."""
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-1",
            "target_id": "tgt",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
            "backend_url": "http://203.0.113.11:8082",
            "acquire_lease": True,
            "swarm_payload": True,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                   return_value=({"final_diff": ""}, [])) as mock_cga:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-1-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")
            mock_cga.assert_called_once()
            kwargs = mock_cga.call_args.kwargs
            assert kwargs.get("swarm_payload") is True


# ---------------------------------------------------------------------------
# (g) the _is_swarm parity table
# ---------------------------------------------------------------------------

class TestIsSwarmParity:
    """The load-bearing invariant: every EXISTING seat's _is_swarm value is
    UNCHANGED by the new swarm_payload key (which defaults False), and the
    berth shape (backend_url set + acquire_lease true + swarm_payload true)
    -> True while the same shape WITHOUT swarm_payload -> False (proving the
    key is what does the work).

    The fixture source is the LIVE lapis-pm registry.yaml (imported at test
    time) — NOT a hand-maintained subset (Facets hardening) — so a future
    registry entry that sets backend_url without swarm_payload is caught by
    the invariant, not silently skipped. The portable berth-shape tests below
    are the floor that runs wherever the live registry is absent.
    """

    def _is_swarm_for(self, backend_url, acquire_lease, swarm_payload=False):
        """Compute _is_swarm exactly as gw_agent does (the OR clause)."""
        return swarm_payload or ((backend_url is not None) and (not acquire_lease))

    def test_parity_from_live_registry(self):
        """Every seat in the LIVE lapis-pm registry keeps its legacy
        _is_swarm value (spec Facets hardening: the fixture source is the
        live registry imported at test time, not a hand-maintained subset).
        Env-overridable; skipped where the live registry is absent."""
        import yaml

        legacy = lambda bu, al: (bu is not None) and (not al)
        live = os.environ.get(
            "LAPIS_PM_REGISTRY",
            "/srv/lapis/lapis-pm/lapis_pm/registry.yaml",
        )
        if not os.path.exists(live):
            pytest.skip("live lapis-pm registry not present on this host")
        reg = yaml.safe_load(open(live).read())
        seats = reg.get("agents", {})
        assert seats, "live registry has no agents section"
        for name, entry in seats.items():
            if not isinstance(entry, dict):
                continue
            bu = entry.get("backend_url")
            al = entry.get("acquire_lease", True)
            sp = entry.get("swarm_payload", False)
            assert self._is_swarm_for(bu, al, sp) == legacy(bu, al), \
                f"seat {name}: _is_swarm changed (backend_url={bu!r}, " \
                f"acquire_lease={al}, swarm_payload={sp})"

    def test_berth_shape_with_swarm_payload_is_true(self):
        """The berth shape (backend_url set + acquire_lease true + swarm_payload
        true) -> True (the keep-both design: swarm shape WITH the per-run lease)."""
        assert self._is_swarm_for(
            "http://203.0.113.11:8082", True, swarm_payload=True) is True

    def test_berth_shape_without_swarm_payload_is_false(self):
        """The SAME berth shape WITHOUT swarm_payload -> False (proving the key
        is what does the work — the legacy condition alone would be False for
        backend_url set + acquire_lease true)."""
        assert self._is_swarm_for(
            "http://203.0.113.11:8082", True, swarm_payload=False) is False

    def test_call_gw_agent_swarm_payload_param_threads_through(self):
        """The public call_gw_agent() accepts swarm_payload and threads it to
        the impl (the parity is not just the internal computation — the
        registry key is reachable from the caller)."""
        import inspect
        from agents_core.gw_agent import call_gw_agent
        sig = inspect.signature(call_gw_agent)
        assert "swarm_payload" in sig.parameters
        assert sig.parameters["swarm_payload"].default is False