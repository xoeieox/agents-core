"""Tests for friction-test CLI argument parsing and exit codes.

The CLI itself lives OUTSIDE the package (Library Purity invariant):
/srv/agents/scripts/friction_test.py is the argparse entry point and
agents_core.friction_test is library-only. The shim is a cross-repo
artifact: it is committed/pushed in the /srv/agents deploy clone
(separate from this PR's diff), so tests that depend on it are skipped
on checkouts/CI runners where the shim file is not present. The
in-package library-purity check (no agents_core/friction_test/cli.py)
always runs.
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.friction_test.observe import Observation
from agents_core.friction_test.report import FrictionReport
from agents_core.friction_test.scenario import Scenario, _make_scenario_id
from agents_core.friction_test.orchestrator import run as orch_run

_SHIM_PATH = Path("/srv/agents/scripts/friction_test.py")


@pytest.fixture
def shim(monkeypatch):
    """Load the /srv/agents/scripts/friction_test.py shim as a module.

    Skips when the shim is not present (fresh checkout / CI runner): the
    shim is a cross-repo artifact committed in the /srv/agents deploy
    clone, not part of this repo's diff.
    """
    if not _SHIM_PATH.exists():
        pytest.skip(
            "friction_test.py shim not present at "
            f"{_SHIM_PATH} (cross-repo artifact, deployed in /srv/agents)"
        )
    spec = importlib.util.spec_from_file_location("friction_test_shim", _SHIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_mock_report(with_harness_error: bool = False) -> FrictionReport:
    sid = _make_scenario_id("radio-op", "test", {})
    obs = Observation(
        scenario_id=sid,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        harness_error=with_harness_error,
        errors=[{"stage": "setup", "error": "down"}] if with_harness_error else [],
    )
    return FrictionReport(
        target="radio-op",
        scenario_set="smoke",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        n_scenarios=1,
        n_invariants_declared=6,
        n_invariants_inferred=0,
        n_dissonances={"system_likely": 0, "model_likely": 0, "total": 0},
        scenarios=[],
        observations=[obs],
        invariant_results=[],
        inferred_invariants=[],
        md_path="/tmp/test.md",
        json_path="/tmp/test.json",
    )


def test_no_command_exits_1(shim):
    assert shim.main([]) == 1


def test_run_exits_0_on_success(shim, tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "run", lambda **kw: _make_mock_report())
    rc = shim.main(["run", "--target", "radio-op", "--scenario-set", "smoke",
                    "--out-dir", str(tmp_path)])
    assert rc == 0


def test_run_with_invariant_mode_declared(shim, tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return _make_mock_report()

    monkeypatch.setattr(shim, "run", fake_run)
    rc = shim.main(["run", "--target", "radio-op", "--invariant-mode", "declared",
                    "--out-dir", str(tmp_path)])

    assert rc == 0
    assert captured_kwargs["invariant_mode"] == "declared"


def test_run_with_invariant_mode_both(shim, tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return _make_mock_report()

    monkeypatch.setattr(shim, "run", fake_run)
    rc = shim.main(["run", "--target", "cockpit", "--invariant-mode", "both",
                    "--out-dir", str(tmp_path)])

    assert rc == 0
    assert captured_kwargs["invariant_mode"] == "both"


def test_run_exits_0_even_with_dissonances(shim, tmp_path, monkeypatch):
    """Dissonant invariants are information, not CLI failures."""
    report = _make_mock_report()
    report.n_dissonances = {"system_likely": 2, "model_likely": 1, "total": 3}
    monkeypatch.setattr(shim, "run", lambda **kw: report)
    rc = shim.main(["run", "--target", "radio-op", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_run_exits_1_on_harness_error_with_strict(shim, tmp_path, monkeypatch):
    """With --strict, harness errors cause exit 1."""
    monkeypatch.setattr(shim, "run", lambda **kw: _make_mock_report(with_harness_error=True))
    rc = shim.main(["run", "--target", "radio-op", "--strict",
                    "--out-dir", str(tmp_path)])
    assert rc == 1


def test_run_exits_0_on_harness_error_without_strict(shim, tmp_path, monkeypatch):
    """Without --strict, harness errors are warnings, not failures."""
    monkeypatch.setattr(shim, "run", lambda **kw: _make_mock_report(with_harness_error=True))
    rc = shim.main(["run", "--target", "radio-op", "--out-dir", str(tmp_path)])
    assert rc == 0


def test_strict_exits_1_when_run_raises(shim, tmp_path, monkeypatch):
    """If run() raises with --strict, exit 1."""

    def _raise(**kw):
        raise RuntimeError("down")

    monkeypatch.setattr(shim, "run", _raise)
    rc = shim.main(["run", "--target", "radio-op", "--strict",
                    "--out-dir", str(tmp_path)])
    assert rc == 1


def test_cli_lives_outside_package():
    """Library Purity: the friction-test CLI must not live inside agents_core.

    Only the in-repo half of the check runs unconditionally: the
    in-package cli.py must not exist. The shim's existence at
    /srv/agents/scripts/friction_test.py is a cross-repo artifact
    (committed in the /srv/agents deploy clone), so that half is
    skipped on checkouts without the deploy clone.
    """
    assert not (Path("agents_core") / "friction_test" / "cli.py").exists()
    if not _SHIM_PATH.exists():
        pytest.skip(
            "friction_test.py shim not present at "
            f"{_SHIM_PATH} (cross-repo artifact, deployed in /srv/agents)"
        )
    assert _SHIM_PATH.exists()


def test_no_anthropic_api_key_reference():
    """Module must not reference ANTHROPIC_API_KEY."""
    import agents_core.friction_test.orchestrator as orch_mod
    import inspect
    source = inspect.getsource(orch_mod)
    assert "ANTHROPIC_API_KEY" not in source

    import agents_core.friction_test.critique as crit_mod
    source2 = inspect.getsource(crit_mod)
    assert "ANTHROPIC_API_KEY" not in source2


def test_orchestrator_run_accepts_cli_kwargs(tmp_path):
    """The in-repo library entry point accepts the same kwargs the CLI shim
    passes (target, scenario_set, n_max, invariant_mode, target_base_url,
    out_dir, qwen_endpoint, strict). Guards the shim/library contract
    without requiring the cross-repo shim file to be present."""
    sig_kwargs = {
        "target": "radio-op",
        "scenario_set": "smoke",
        "n_max": None,
        "invariant_mode": "declared",
        "target_base_url": None,
        "out_dir": tmp_path,
        "qwen_endpoint": "http://localhost:1/v1/chat/completions",
        "strict": False,
    }
    import inspect
    params = inspect.signature(orch_run).parameters
    for name in sig_kwargs:
        assert name in params, f"orchestrator.run() missing kwarg {name!r}"
