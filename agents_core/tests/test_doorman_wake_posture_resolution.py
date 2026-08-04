"""Tests for agents-core-doorman-wake-honors-declared-posture-v0.

The doorman's cold-wake fallback (agents_core.doorman_server) used to branch
on the static DOORMAN_DEFAULT_SERVE_MODE constant alone, fighting any
declared home posture that wasn't that literal value. This unit replaces the
branch with a resolution ladder (Part 1/2) and a check-then-act generic
dispatch path for any posture other than "big"/"dual" (Part 1b/A1/A2), built
on top of `scripts.gw_topology.reach()`'s typed refusals.

`scripts.gw_topology` is a conductor-repo module deployed to
/srv/agents/scripts on the live host; it does not exist in this repo, so
these tests inject a fake `scripts` / `scripts.gw_topology` module into
sys.modules for the duration of each test that needs one (mirroring the
lazy `from scripts import gw_topology` the wake path itself performs).

Covers:
  - resolution precedence: mode-bearing acquire > declared posture > the
    DOORMAN_DEFAULT_SERVE_MODE literal > unreadable-and-unset WARN degrade
  - generic-posture dispatch calls gw_topology.reach(), never re-implements
    actuability checking
  - refused-before-acting (WAKE_REFUSED: ACTUATOR_UNAVAILABLE / POSTURE_INVALID)
    vs attempted-then-failed (WAKE_FAILED: REACH_BUSY / REACH_FAILED) —
    distinct log/last_error shapes, never confused
  - the readiness gate is posture-driven: a single-slot posture never polls
    a port it doesn't declare
  - the import-form poison pill (A3): bare `import gw_topology` fails under
    the live service's PYTHONPATH=/srv/agents; `from scripts import
    gw_topology` is the only form that works, and doorman_server.py must
    use it
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.doorman_server import GW_URL_DEFAULT, _NodeState


def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


# ---------------------------------------------------------------------------
# Fake scripts.gw_topology module — injected into sys.modules
# ---------------------------------------------------------------------------

class _FakeTopologyErrors:
    class TopologyUnknown(Exception):
        pass

    class TopologyNotProven(Exception):
        pass

    class TopologyOverCeiling(Exception):
        pass

    class ForceUnprovenReasonRequired(Exception):
        pass

    class TopologyReachBusy(Exception):
        pass

    class TopologyReachFailed(Exception):
        pass


@pytest.fixture
def fake_gw_topology(monkeypatch):
    """Install a fake `scripts.gw_topology` module in sys.modules with the
    typed exception classes and MagicMock `load_topology`/`reach`, so
    `from scripts import gw_topology` inside doorman_server resolves to it.
    Yields the fake module for the test to configure/assert against.
    Removes both `scripts` and `scripts.gw_topology` from sys.modules on
    teardown so this never leaks into other tests."""
    fake_module = types.ModuleType("scripts.gw_topology")
    for name in (
        "TopologyUnknown", "TopologyNotProven", "TopologyOverCeiling",
        "ForceUnprovenReasonRequired", "TopologyReachBusy", "TopologyReachFailed",
    ):
        setattr(fake_module, name, getattr(_FakeTopologyErrors, name))
    fake_module.load_topology = MagicMock(return_value=MagicMock())
    fake_module.reach = MagicMock(return_value={"lease": "acquired"})

    fake_scripts_pkg = sys.modules.get("scripts")
    installed_scripts_pkg = False
    if fake_scripts_pkg is None:
        fake_scripts_pkg = types.ModuleType("scripts")
        fake_scripts_pkg.__path__ = []  # mark as a package
        sys.modules["scripts"] = fake_scripts_pkg
        installed_scripts_pkg = True
    monkeypatch.setattr(fake_scripts_pkg, "gw_topology", fake_module, raising=False)
    monkeypatch.setitem(sys.modules, "scripts.gw_topology", fake_module)

    yield fake_module

    if installed_scripts_pkg:
        sys.modules.pop("scripts", None)


def _topology_entry(kind: str, slots: list[dict]) -> MagicMock:
    """A minimal fake Topology object satisfying _posture_slot_ports'
    reads: topology.topologies[name] -> entry, then either
    topology.modes[entry['ref']]['slots'] (kind=mode) or
    topology.pairings[entry['ref']]['slot1'/'slot2'] (kind=pairing)."""
    topology = MagicMock()
    if kind == "mode":
        entry = {"kind": "mode", "ref": "some-mode"}
        topology.topologies = {"some-posture": entry}
        topology.modes = {"some-mode": {"slots": slots}}
    else:
        entry = {"kind": "pairing", "ref": "some-pairing"}
        topology.topologies = {"some-posture": entry}
        topology.pairings = {"some-pairing": {"slot1": slots[0], "slot2": slots[1]}}
    return topology


# ---------------------------------------------------------------------------
# Part 1 — resolution precedence
# ---------------------------------------------------------------------------

class TestResolutionPrecedence:
    """DoD 1: mode-bearing acquire wins over declared posture; declared
    posture wins over the env literal; env literal used when no declaration
    is readable; unreadable-and-unset degrades to today's behaviour with a
    WARN emitted."""

    def test_declared_posture_wins_over_env_literal(self):
        state = _make_state()
        with patch.object(state, "_read_declared_home_posture", return_value="slot1-solo"), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "big"):
            assert state._resolve_cold_wake_posture() == "slot1-solo"

    def test_env_literal_used_when_declaration_unreadable(self):
        state = _make_state()
        with patch.object(state, "_read_declared_home_posture", return_value=None), \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            assert state._resolve_cold_wake_posture() == "dual"

    def test_unreadable_declaration_emits_warn_and_degrades(self, caplog):
        """Part 2: an unreadable file (or absent GW_HOME_MODE key) is a
        WAKE_DEGRADED WARN, not silence, and never a refusal."""
        state = _make_state()
        with patch("agents_core.doorman_server.GW_HOME_MODE_ENV_PATH", "/nonexistent/conductor.env"), \
             caplog.at_level("WARNING"):
            result = state._read_declared_home_posture()
        assert result is None
        assert any("WAKE_DEGRADED" in r.message and "POSTURE_UNDECLARED" in r.message
                    for r in caplog.records)

    def test_declaration_present_but_no_gw_home_mode_key_degrades(self, tmp_path, caplog):
        env_file = tmp_path / "conductor.env"
        env_file.write_text("SOME_OTHER_KEY=value\n")
        state = _make_state()
        with patch("agents_core.doorman_server.GW_HOME_MODE_ENV_PATH", str(env_file)), \
             caplog.at_level("WARNING"):
            result = state._read_declared_home_posture()
        assert result is None
        assert any("WAKE_DEGRADED" in r.message and "POSTURE_UNDECLARED" in r.message
                    for r in caplog.records)

    def test_declaration_present_reads_gw_home_mode(self, tmp_path):
        env_file = tmp_path / "conductor.env"
        env_file.write_text("# comment\nOTHER=1\nGW_HOME_MODE=slot1-solo\n")
        state = _make_state()
        with patch("agents_core.doorman_server.GW_HOME_MODE_ENV_PATH", str(env_file)):
            assert state._read_declared_home_posture() == "slot1-solo"

    def test_mode_bearing_acquire_wins_over_declared_posture(self):
        """DoD 1 case 1: the mode-bearing controller acquire at ensure_serving's
        line ~1076 must still win over anything this unit adds — cold-box path,
        exercised end to end through ensure_serving()."""
        state = _make_state()
        with patch.object(state, "_is_serving", return_value=False), \
             patch.object(state, "_read_declared_home_posture", return_value="slot1-solo"), \
             patch("subprocess.run") as mock_sub, \
             patch.object(state, "_wake_big", return_value=True) as mock_big, \
             patch.object(state, "_wake_dual") as mock_dual, \
             patch.object(state, "_wake_generic_posture") as mock_generic, \
             patch("agents_core.doorman_server.DOORMAN_DEFAULT_SERVE_MODE", "dual"):
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            result = state.ensure_serving(role="mode-controller", mode="big", work_id="w1")

        assert result is True
        mock_big.assert_called_once()
        mock_dual.assert_not_called()
        mock_generic.assert_not_called()


# ---------------------------------------------------------------------------
# Part 1 / Part 1b — generic dispatch, check-then-act
# ---------------------------------------------------------------------------

class TestGenericPostureDispatch:
    def test_big_and_dual_still_dispatch_to_unchanged_wake_functions(self):
        """DoD 3: "big"/"dual" resolved postures must still hit the exact
        unchanged _wake_big/_wake_dual methods, never the generic path."""
        state = _make_state()
        with patch.object(state, "_resolve_cold_wake_posture", return_value="big"), \
             patch.object(state, "_is_serving", return_value=False), \
             patch("subprocess.run") as mock_sub, \
             patch.object(state, "_wake_generic_posture") as mock_generic:
            mock_sub.return_value = MagicMock(returncode=0, stderr="")
            with patch.object(state, "_wake_big", return_value=True) as mock_big:
                state.ensure_serving()
            mock_generic.assert_not_called()
            mock_big.assert_called_once()

    def test_generic_posture_calls_reach_not_gw_serve_subprocess(self, fake_gw_topology):
        """A3/A1: the generic path calls gw_topology.reach() — it must not
        shell out to gw-serve directly for a non-big/dual posture."""
        state = _make_state()
        topology = _topology_entry("mode", [{"model": "x", "port": 8081}])
        fake_gw_topology.load_topology.return_value = topology

        with patch.object(state, "_poll_posture_ready", return_value=True) as mock_poll:
            result = state._wake_generic_posture("some-posture")

        assert result is True
        fake_gw_topology.reach.assert_called_once()
        args, kwargs = fake_gw_topology.reach.call_args
        assert args[0] == "some-posture"
        assert "force_unproven" not in kwargs  # A2: never set from cold-wake
        mock_poll.assert_called_once_with([8081])

    def test_actuator_unavailable_refuses_before_any_wake(self):
        """DoD 4a: an unimportable gw_topology refuses BEFORE any wake
        command is issued — assert on the mock, not the log line."""
        state = _make_state()
        with patch.dict(sys.modules, {"scripts.gw_topology": None, "scripts": None}), \
             patch.object(state, "_poll_posture_ready") as mock_poll, \
             patch.object(state, "_is_port_serving") as mock_port:
            result = state._wake_generic_posture("slot1-solo")

        assert result is False
        mock_poll.assert_not_called()
        mock_port.assert_not_called()
        assert state.last_error is not None
        assert state.last_error.startswith("WAKE_REFUSED reason=ACTUATOR_UNAVAILABLE")

    def test_posture_invalid_refuses_before_any_wake(self, fake_gw_topology):
        """DoD 4a/4b: TopologyUnknown (etc.) refuses before any wake command
        — reach() answered, this specific posture is refused."""
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        fake_gw_topology.reach.side_effect = fake_gw_topology.TopologyUnknown("nope")

        with patch.object(state, "_poll_posture_ready") as mock_poll, \
             patch.object(state, "_is_port_serving") as mock_port:
            result = state._wake_generic_posture("nope")

        assert result is False
        mock_poll.assert_not_called()
        mock_port.assert_not_called()
        assert state.last_error.startswith("WAKE_REFUSED reason=POSTURE_INVALID")

    @pytest.mark.parametrize("exc_name", [
        "TopologyNotProven", "TopologyOverCeiling", "ForceUnprovenReasonRequired",
    ])
    def test_all_refusal_categories_map_to_posture_invalid(self, fake_gw_topology, exc_name):
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        fake_gw_topology.reach.side_effect = getattr(fake_gw_topology, exc_name)("refused")

        with patch.object(state, "_poll_posture_ready") as mock_poll:
            result = state._wake_generic_posture("some-posture")

        assert result is False
        mock_poll.assert_not_called()
        assert state.last_error.startswith("WAKE_REFUSED reason=POSTURE_INVALID")

    def test_reach_busy_is_attempted_not_refused(self, fake_gw_topology):
        """DoD 4b: TopologyReachBusy/TopologyReachFailed are attempted-then-
        failed, a DIFFERENT record shape (WAKE_FAILED) from a refusal —
        a command WAS issued (the host helper), just contended/failed."""
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        fake_gw_topology.reach.side_effect = fake_gw_topology.TopologyReachBusy("busy")

        result = state._wake_generic_posture("some-posture")

        assert result is False
        assert state.last_error.startswith("WAKE_FAILED reason=REACH_BUSY")
        assert not state.last_error.startswith("WAKE_REFUSED")

    def test_reach_failed_is_attempted_not_refused(self, fake_gw_topology):
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        fake_gw_topology.reach.side_effect = fake_gw_topology.TopologyReachFailed(
            "some-posture", 1, "rolled back"
        )

        result = state._wake_generic_posture("some-posture")

        assert result is False
        assert state.last_error.startswith("WAKE_FAILED reason=REACH_FAILED")
        assert not state.last_error.startswith("WAKE_REFUSED")


# ---------------------------------------------------------------------------
# Part 1 — posture-driven readiness gate (DoD 5)
# ---------------------------------------------------------------------------

class TestPostureDrivenReadinessGate:
    def test_single_slot_posture_never_polls_slot2_port(self):
        """The remaining live false-negative: a one-slot posture must reach
        ready without :8082 (GW_SLOT2_PORT) ever being polled."""
        state = _make_state()
        polled_ports = []

        def fake_is_port_serving(port, timeout=3.0):
            polled_ports.append(port)
            return True

        with patch.object(state, "_is_port_serving", side_effect=fake_is_port_serving), \
             patch("time.sleep"), \
             patch.object(state, "_place_hold"):
            result = state._poll_posture_ready([8081])

        assert result is True
        assert 8082 not in polled_ports
        assert set(polled_ports) == {8081}

    def test_two_slot_posture_polls_both_declared_ports(self):
        state = _make_state()
        polled_ports = set()

        def fake_is_port_serving(port, timeout=3.0):
            polled_ports.add(port)
            return True

        with patch.object(state, "_is_port_serving", side_effect=fake_is_port_serving), \
             patch("time.sleep"), \
             patch.object(state, "_place_hold"):
            result = state._poll_posture_ready([8081, 8082])

        assert result is True
        assert polled_ports == {8081, 8082}

    def test_posture_slot_ports_mode_kind(self):
        state = _make_state()
        topology = _topology_entry("mode", [
            {"model": "x", "port": 8081},
        ])
        assert state._posture_slot_ports(topology, "some-posture") == [8081]

    def test_posture_slot_ports_pairing_kind(self):
        state = _make_state()
        topology = _topology_entry("pairing", [
            {"model": "x", "port": 8081},
            {"model": "y", "port": 8082},
        ])
        assert state._posture_slot_ports(topology, "some-posture") == [8081, 8082]

    def test_readiness_timeout_produces_wake_failed_style_last_error(self):
        """A single-slot posture that never comes healthy still degrades to
        the normal wake-timeout shape (transient, caller retries) — not a
        refusal, since a command WAS issued."""
        state = _make_state()
        with patch.object(state, "_is_port_serving", return_value=False), \
             patch("time.sleep"), \
             patch("agents_core.doorman_server.GW_DUAL_WAKE_DEADLINE_SEC", 0):
            result = state._poll_posture_ready([8081])

        assert result is False
        assert state.last_error is not None
        assert "8081" in state.last_error


# ---------------------------------------------------------------------------
# A3 — the import-form poison pill
# ---------------------------------------------------------------------------

class TestImportFormPoisonPill:
    """Unit tests import from the repo tree and cannot reproduce the deploy-
    tree trap on their own — this class runs real subprocesses under a
    PYTHONPATH shaped like the live service's (a `scripts/` dir with no
    `__init__.py`, resolved via PEP 420 namespace packages) to prove the
    bare `import gw_topology` form fails there while `from scripts import
    gw_topology` succeeds, then statically locks the form doorman_server.py
    actually uses."""

    @pytest.fixture
    def deploy_tree_fixture(self, tmp_path):
        """A minimal stand-in for /srv/agents: a `scripts/` directory
        (no __init__.py — PEP 420 namespace package, matching the real
        deploy tree) containing a stub gw_topology.py."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "gw_topology.py").write_text(
            textwrap.dedent(
                """
                class TopologyUnknown(Exception):
                    pass
                def load_topology():
                    return object()
                def reach(name, **kwargs):
                    return {"lease": "acquired"}
                """
            )
        )
        return tmp_path

    def test_bare_import_form_fails_under_deploy_pythonpath(self, deploy_tree_fixture):
        proc = subprocess.run(
            [sys.executable, "-c", "import gw_topology"],
            env={"PYTHONPATH": str(deploy_tree_fixture)},
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert "ModuleNotFoundError" in proc.stderr

    def test_from_scripts_import_form_succeeds_under_deploy_pythonpath(self, deploy_tree_fixture):
        proc = subprocess.run(
            [sys.executable, "-c", "from scripts import gw_topology; print('ok')"],
            env={"PYTHONPATH": str(deploy_tree_fixture)},
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout

    def test_doorman_server_source_uses_the_safe_import_form(self):
        """Static lock on the exact import form (A3): doorman_server.py must
        use `from scripts import gw_topology`, and must never contain a bare
        `import gw_topology` statement — copying gw_actuator.py:15's form
        would ship a runtime ModuleNotFoundError no other test catches."""
        source_path = (
            Path(__file__).resolve().parents[1] / "doorman_server.py"
        )
        source = source_path.read_text(encoding="utf-8")
        assert "from scripts import gw_topology" in source
        for line in source.splitlines():
            stripped = line.strip()
            assert stripped != "import gw_topology", (
                "doorman_server.py must not use the bare `import gw_topology` "
                "form — it raises ModuleNotFoundError under the live service's "
                "PYTHONPATH=/srv/agents (see conductor gw_actuator.py:15, which "
                "only works because it executes from inside scripts/)."
            )
