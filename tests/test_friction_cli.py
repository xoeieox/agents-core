"""Tests for friction-test CLI argument parsing and exit codes."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.friction_test.cli import main
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.report import FrictionReport
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


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


def test_no_command_exits_1():
    rc = main([])
    assert rc == 1


def test_run_exits_0_on_success(tmp_path):
    mock_report = _make_mock_report()
    with patch("agents_core.friction_test.cli.run", return_value=mock_report):
        rc = main(["run", "--target", "radio-op", "--scenario-set", "smoke",
                   "--out-dir", str(tmp_path)])
    assert rc == 0


def test_run_with_invariant_mode_declared(tmp_path):
    mock_report = _make_mock_report()
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_report

    with patch("agents_core.friction_test.cli.run", side_effect=fake_run):
        rc = main(["run", "--target", "radio-op", "--invariant-mode", "declared",
                   "--out-dir", str(tmp_path)])

    assert rc == 0
    assert captured_kwargs["invariant_mode"] == "declared"


def test_run_with_invariant_mode_both(tmp_path):
    mock_report = _make_mock_report()
    captured_kwargs = {}

    def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_report

    with patch("agents_core.friction_test.cli.run", side_effect=fake_run):
        rc = main(["run", "--target", "cockpit", "--invariant-mode", "both",
                   "--out-dir", str(tmp_path)])

    assert captured_kwargs["invariant_mode"] == "both"


def test_run_exits_0_even_with_dissonances(tmp_path):
    """Dissonant invariants are information, not CLI failures."""
    mock_report = _make_mock_report()
    mock_report.n_dissonances = {"system_likely": 2, "model_likely": 1, "total": 3}

    with patch("agents_core.friction_test.cli.run", return_value=mock_report):
        rc = main(["run", "--target", "radio-op", "--out-dir", str(tmp_path)])

    assert rc == 0


def test_run_exits_1_on_harness_error_with_strict(tmp_path):
    """With --strict, harness errors cause exit 1."""
    mock_report = _make_mock_report(with_harness_error=True)

    with patch("agents_core.friction_test.cli.run", return_value=mock_report):
        rc = main(["run", "--target", "radio-op", "--strict",
                   "--out-dir", str(tmp_path)])

    assert rc == 1


def test_run_exits_0_on_harness_error_without_strict(tmp_path):
    """Without --strict, harness errors are warnings, not failures."""
    mock_report = _make_mock_report(with_harness_error=True)

    with patch("agents_core.friction_test.cli.run", return_value=mock_report):
        rc = main(["run", "--target", "radio-op", "--out-dir", str(tmp_path)])

    assert rc == 0


def test_strict_exits_1_when_run_raises(tmp_path):
    """If run() raises with --strict, exit 1."""
    with patch("agents_core.friction_test.cli.run", side_effect=RuntimeError("down")):
        rc = main(["run", "--target", "radio-op", "--strict",
                   "--out-dir", str(tmp_path)])
    assert rc == 1


def test_no_anthropic_api_key_reference():
    """Module must not reference ANTHROPIC_API_KEY."""
    import agents_core.friction_test.orchestrator as orch_mod
    import inspect
    source = inspect.getsource(orch_mod)
    assert "ANTHROPIC_API_KEY" not in source

    import agents_core.friction_test.critique as crit_mod
    source2 = inspect.getsource(crit_mod)
    assert "ANTHROPIC_API_KEY" not in source2
