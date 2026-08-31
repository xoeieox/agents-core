"""Tests for agents-core-doorman-drift-refusal-500-v0.

The doorman's cold-wake path (agents_core.doorman_server._wake_generic_posture)
previously let two typed `scripts.gw_topology.reach()` refusal classes
propagate out of the `/lease/acquire` handler as an unhandled HTTP 500:

  - `PairingApplyError` - the confirmed 2026-08-30 incident
    (finding/night-dag-20260830-partial-source-gw-drift-doorman-500-2026-08-30):
    a hand-edit drift of the live /etc/default/gw-dual made reach() refuse
    pre-apply; the uncaught exception 500'd the acquire, the night-plan
    runner read the 500 as reason class UNHANDLED (not in the wake-retry
    allowlist), and the night's GW work was silently skipped with no
    operator-facing reason.
  - `TopologyHelperIncompatible` - the deployed convergence helper's
    pre-host-touch self-hash check failing; verified reachable through the
    same reach() call, same 500 class.

The fix catches both in _wake_generic_posture's except chain and returns the
doorman's existing clean refusal shape (WAKE_REFUSED reason=<TOKEN>
detail="...") with distinct tokens DRIFT_REFUSED / HELPER_INCOMPATIBLE, so
the runner's existing detail-grammar parse surfaces a clean hard hold with a
diagnosable reason instead of UNHANDLED.

This file is scoped to the two NEW tokens only. The pre-existing clauses
(ACTUATOR_UNAVAILABLE / POSTURE_INVALID / REACH_BUSY / REACH_FAILED) are
pinned where they already are:
agents_core/tests/test_doorman_wake_posture_resolution.py.

Hermetic: mirrors the existing fake_gw_topology fixture pattern (fake
`scripts` / `scripts.gw_topology` modules injected into sys.modules, MagicMock
load_topology/reach, save/restore teardown). No SSH, no network, no
/etc/default/gw-dual read, no live GW required - passes on a bare checkout.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from agents_core.doorman_server import GW_URL_DEFAULT, _NodeState


def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


# ---------------------------------------------------------------------------
# Fake scripts.gw_topology module - injected into sys.modules
#
# Mirrors the fake_gw_topology fixture in
# agents_core/tests/test_doorman_wake_posture_resolution.py, EXTENDED with the
# two new refusal classes. The post-fix except chain evaluates all EIGHT
# gw_topology names, so the fake module must expose all eight - a missing
# name would fail with AttributeError, never silently pass (self-checking
# completeness).
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

    class PairingApplyError(Exception):
        pass

    class TopologyHelperIncompatible(Exception):
        pass


@pytest.fixture
def fake_gw_topology(monkeypatch):
    """Install a fake `scripts.gw_topology` module in sys.modules with the
    typed exception classes (all eight the post-fix except chain evaluates)
    and MagicMock `load_topology`/`reach`, so `from scripts import
    gw_topology` inside doorman_server resolves to it. Yields the fake
    module for the test to configure/assert against.

    Teardown saves/restores pre-existing sys.modules entries: this repo
    carries its own importable `scripts/` package (root conftest puts the
    repo root on the test sys.path), so a bare delete could evict the real
    module test-order dependently. Only what this fixture installed is
    popped."""
    fake_module = types.ModuleType("scripts.gw_topology")
    for name in (
        "TopologyUnknown", "TopologyNotProven", "TopologyOverCeiling",
        "ForceUnprovenReasonRequired", "TopologyReachBusy", "TopologyReachFailed",
        "PairingApplyError", "TopologyHelperIncompatible",
    ):
        setattr(fake_module, name, getattr(_FakeTopologyErrors, name))
    fake_module.load_topology = MagicMock(return_value=MagicMock())
    fake_module.reach = MagicMock(return_value={"lease": "acquired"})

    pre_existing_scripts = sys.modules.get("scripts")
    installed_scripts_pkg = False
    if pre_existing_scripts is None:
        pre_existing_scripts = types.ModuleType("scripts")
        pre_existing_scripts.__path__ = []  # mark as a package
        sys.modules["scripts"] = pre_existing_scripts
        installed_scripts_pkg = True
    monkeypatch.setattr(pre_existing_scripts, "gw_topology", fake_module, raising=False)
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
# AC1/AC2/AC3 - the two new refusal clauses: clean hold, never a 500
# ---------------------------------------------------------------------------

class TestDriftRefusalClauses:
    """reach() raises PairingApplyError / TopologyHelperIncompatible ->
    _wake_generic_posture returns False (no exception propagates to the
    handler, so /lease/acquire takes its existing wake_failed path instead
    of 500) and last_error carries the structured WAKE_REFUSED grammar with
    the raised message verbatim in the detail= field. The runner's
    WAKE_(REFUSED|FAILED) reason=(\\S+) parse then yields a clean hard hold
    with a diagnosable reason - neither token is in the runner's retry
    allowlist, so the operator reads the remediation, not a retry."""

    def test_pairing_apply_error_refuses_with_drift_refused_token(
        self, fake_gw_topology
    ):
        """The confirmed 2026-08-30 incident: reach() refuses pre-apply on a
        live-config drift. The method must NOT let the exception escape -
        the handler takes its existing {"status": "wake_failed", ...} path,
        and last_error names the drift, not UNHANDLED."""
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        drift_msg = (
            "pre-apply drift check failed: live /etc/default/gw-dual SLOT "
            "lines match no registered topology - refusing to clobber an "
            "untracked hand-edit"
        )
        fake_gw_topology.reach.side_effect = (
            fake_gw_topology.PairingApplyError(drift_msg)
        )

        with patch.object(state, "_poll_posture_ready") as mock_poll, \
             patch.object(state, "_is_port_serving") as mock_port:
            result = state._wake_generic_posture("slot1-solo")

        assert result is False
        mock_poll.assert_not_called()
        mock_port.assert_not_called()
        assert state.last_error is not None
        assert state.last_error.startswith("WAKE_REFUSED reason=DRIFT_REFUSED")
        assert not state.last_error.startswith("WAKE_FAILED")
        # The raised message is embedded verbatim in the detail= field -
        # the operator reads the exact cause from /status last_error.
        assert f'detail="{drift_msg}"' in state.last_error

    def test_topology_helper_incompatible_refuses_with_helper_token(
        self, fake_gw_topology
    ):
        """The verified latent case: the deployed convergence helper's
        self-hash check fails pre-host-touch. Same 500 class if uncaught;
        the token names the remediation (re-deploy the helper)."""
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        helper_msg = (
            "deployed convergence helper hash mismatch on gravitywell: "
            "local abc123 != remote def456 - re-deploy the helper"
        )
        fake_gw_topology.reach.side_effect = (
            fake_gw_topology.TopologyHelperIncompatible(helper_msg)
        )

        with patch.object(state, "_poll_posture_ready") as mock_poll, \
             patch.object(state, "_is_port_serving") as mock_port:
            result = state._wake_generic_posture("slot1-solo")

        assert result is False
        mock_poll.assert_not_called()
        mock_port.assert_not_called()
        assert state.last_error is not None
        assert state.last_error.startswith(
            "WAKE_REFUSED reason=HELPER_INCOMPATIBLE"
        )
        assert not state.last_error.startswith("WAKE_FAILED")
        assert f'detail="{helper_msg}"' in state.last_error

    @pytest.mark.parametrize(
        "exc_name, token",
        [
            ("PairingApplyError", "DRIFT_REFUSED"),
            ("TopologyHelperIncompatible", "HELPER_INCOMPATIBLE"),
        ],
    )
    def test_new_tokens_are_wake_refused_not_wake_failed(
        self, fake_gw_topology, exc_name, token
    ):
        """AC3: both new clauses use _refuse_wake - the class contracts
        guarantee zero side effects on the managed live config in every
        raise case, so REFUSED ("refused, nothing was done") is the honest
        shape. The runner's regex accepts WAKE_REFUSED and WAKE_FAILED
        alike and neither token is in the retry allowlist, so the
        operator-facing outcome (clean hard hold) does not depend on the
        prefix - but the record must say REFUSED."""
        state = _make_state()
        fake_gw_topology.load_topology.return_value = _topology_entry(
            "mode", [{"model": "x", "port": 8081}]
        )
        fake_gw_topology.reach.side_effect = getattr(
            fake_gw_topology, exc_name
        )("some diagnostic sentence")

        result = state._wake_generic_posture("slot1-solo")

        assert result is False
        assert state.last_error is not None
        assert state.last_error.startswith(f"WAKE_REFUSED reason={token}")
        assert not state.last_error.startswith("WAKE_FAILED")
