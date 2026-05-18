"""Tests for vault_writer audit chain emission from friction report writes."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.friction_test.critique import InvariantResult
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.report import FrictionReport, write
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


def _minimal_report() -> FrictionReport:
    sid = _make_scenario_id("radio-op", "audit-test", {})
    return FrictionReport(
        target="radio-op",
        scenario_set="smoke",
        started_at="2026-05-18T00:00:00Z",
        finished_at="2026-05-18T00:01:00Z",
        n_scenarios=0,
        n_invariants_declared=0,
        n_invariants_inferred=0,
        n_dissonances={"system_likely": 0, "model_likely": 0, "total": 0},
        scenarios=[],
        observations=[],
        invariant_results=[],
        inferred_invariants=[],
    )


def test_vault_write_intent_matches_target_and_scenario_set(tmp_path):
    """vault_writer.write() is called with intent containing target:scenario_set."""
    report = _minimal_report()
    intents: list[str] = []

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        intents.append(intent)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return MagicMock()

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write):
        write(report, out_dir=tmp_path)

    assert any("radio-op" in i and "smoke" in i for i in intents)


def test_vault_write_called_with_md_and_json(tmp_path):
    """One call for .md, one call for .json sidecar."""
    report = _minimal_report()
    paths_written: list[str] = []

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        paths_written.append(str(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return MagicMock()

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write):
        write(report, out_dir=tmp_path)

    assert any(p.endswith(".md") for p in paths_written)
    assert any(p.endswith(".json") for p in paths_written)


def test_vault_event_emission_via_vault_writer(tmp_path):
    """Verify vault_writer.write() is called (not direct Path.write_text).

    The actual event emission to /data/vault-events.jsonl is handled by
    vault_writer internally; this test verifies our write path flows
    through vault_writer, not direct IO.
    """
    report = _minimal_report()
    direct_writes: list[str] = []

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        # Simulate actual write
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return MagicMock()

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write) as mock_vw:
        write(report, out_dir=tmp_path)

    # vault_write should have been called exactly twice
    assert mock_vw.call_count == 2


def test_md_path_and_json_path_set_on_report(tmp_path):
    """write() sets md_path and json_path on the report object."""
    report = _minimal_report()

    def fake_vault_write(path, content, *, agent_id, intent, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(str(content))
        return MagicMock()

    with patch("agents_core.friction_test.report.vault_write", side_effect=fake_vault_write):
        md_path, json_path = write(report, out_dir=tmp_path)

    assert report.md_path == str(md_path)
    assert report.json_path == str(json_path)
    assert str(md_path).endswith(".md")
    assert str(json_path).endswith(".json")
