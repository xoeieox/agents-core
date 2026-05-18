"""Tests for FrictionReport Markdown + JSON rendering and vault_writer call shape."""
import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from agents_core.friction_test.critique import Invariant, InvariantResult
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.report import FrictionReport, write
from agents_core.friction_test.report_render import render_json, render_markdown
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


def _make_minimal_report(tmp_path: Path) -> FrictionReport:
    sid = _make_scenario_id("radio-op", "test", {})
    scenario = Scenario(
        scenario_id=sid, target="radio-op", family_id="test",
        kind="happy", inputs={}, expected_class="test",
    )
    obs = Observation(
        scenario_id=sid,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
    )
    ir = InvariantResult(
        scenario_id=sid,
        invariant_id="i01_consult_log_row_per_judgment",
        dimension="row_count_delta",
        expected=1,
        observed=1,
        distance=0.0,
        tolerance=0,
        status="held",
        classification="held",
        proposed_by="Erah",
        provisional=False,
        justification="Within tolerance.",
        evidence_refs=["observation.log_appends[0]"],
    )
    return FrictionReport(
        target="radio-op",
        scenario_set="smoke",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        n_scenarios=1,
        n_invariants_declared=1,
        n_invariants_inferred=0,
        n_dissonances={"system_likely": 0, "model_likely": 0, "total": 0},
        scenarios=[scenario],
        observations=[obs],
        invariant_results=[ir],
        inferred_invariants=[],
    )


def test_render_markdown_contains_target(tmp_path):
    report = _make_minimal_report(tmp_path)
    md = render_markdown(report)
    assert "radio-op" in md
    assert "smoke" in md


def test_render_markdown_no_dissonances_section(tmp_path):
    report = _make_minimal_report(tmp_path)
    md = render_markdown(report)
    assert "None detected" in md


def test_render_json_parses_correctly(tmp_path):
    report = _make_minimal_report(tmp_path)
    json_str = render_json(report)
    data = json.loads(json_str)
    assert data["target"] == "radio-op"
    assert data["scenario_set"] == "smoke"
    assert data["n_scenarios"] == 1
    assert "scenarios" in data
    assert "observations" in data
    assert "invariant_results" in data
    assert "inferred_invariants" in data


def test_render_json_invariant_result_shape(tmp_path):
    report = _make_minimal_report(tmp_path)
    data = json.loads(render_json(report))
    ir = data["invariant_results"][0]
    for key in ("scenario_id", "invariant_id", "dimension", "expected", "observed",
                "distance", "tolerance", "status", "classification",
                "proposed_by", "provisional", "justification", "evidence_refs"):
        assert key in ir


def test_write_calls_vault_writer_twice(tmp_path):
    report = _make_minimal_report(tmp_path)

    mock_record = MagicMock()
    mock_record.path = tmp_path / "test.md"

    calls_received = []

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        calls_received.append({"path": path, "agent_id": agent_id, "intent": intent})
        # Write the file so it exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content if isinstance(content, str) else content.decode())
        return mock_record

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write):
        md_path, json_path = write(report, out_dir=tmp_path)

    assert len(calls_received) == 2
    agent_ids = {c["agent_id"] for c in calls_received}
    assert "friction-tester-v0" in agent_ids
    intents = {c["intent"] for c in calls_received}
    assert any("friction-report:radio-op:smoke" in i for i in intents)


def test_write_agent_id_is_friction_tester_v0(tmp_path):
    report = _make_minimal_report(tmp_path)
    agent_ids = []

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        agent_ids.append(agent_id)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return MagicMock()

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write):
        write(report, out_dir=tmp_path)

    assert all(a == "friction-tester-v0" for a in agent_ids)


def test_render_markdown_dissonance_section(tmp_path):
    """Report with a dissonance should render the dissonance section."""
    report = _make_minimal_report(tmp_path)
    # Add a dissonant result
    report.invariant_results[0] = InvariantResult(
        scenario_id=report.invariant_results[0].scenario_id,
        invariant_id="i01_consult_log_row_per_judgment",
        dimension="row_count_delta",
        expected=1,
        observed=0,
        distance=1.0,
        tolerance=0,
        status="dissonant",
        classification="dissonant_system_likely",
        proposed_by="Erah",
        provisional=False,
        justification="Second row missing.",
    )
    report.n_dissonances = {"system_likely": 1, "model_likely": 0, "total": 1}

    md = render_markdown(report)
    assert "System-likely dissonance" in md
    assert "i01_consult_log_row_per_judgment" in md
