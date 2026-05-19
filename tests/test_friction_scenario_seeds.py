"""Tests for scenario seed loading and expansion."""
import copy

import pytest

from agents_core.friction_test.scenario import (
    Scenario,
    _make_scenario_id,
    generate_smoke_scenarios,
)


def test_radio_op_seeds_load_at_least_8():
    scenarios = generate_smoke_scenarios("radio-op")
    assert len(scenarios) >= 8, f"Expected >=8 radio-op scenarios, got {len(scenarios)}"


def test_cockpit_seeds_load_at_least_8():
    scenarios = generate_smoke_scenarios("cockpit")
    assert len(scenarios) >= 8, f"Expected >=8 cockpit scenarios, got {len(scenarios)}"


def test_scenario_id_stable_under_key_reordering():
    inputs_a = {"segments": [{"segment": "hello", "ts": "2026-01-01"}], "harvest": False}
    inputs_b = {"harvest": False, "segments": [{"segment": "hello", "ts": "2026-01-01"}]}
    id_a = _make_scenario_id("radio-op", "test_family", inputs_a)
    id_b = _make_scenario_id("radio-op", "test_family", inputs_b)
    assert id_a == id_b, "scenario_id should be stable under key reordering"


def test_scenario_id_differs_by_target():
    inputs = {"segments": [{"segment": "hello"}]}
    id_radio = _make_scenario_id("radio-op", "test_family", inputs)
    id_cockpit = _make_scenario_id("cockpit", "test_family", inputs)
    assert id_radio != id_cockpit


def test_scenario_id_differs_by_family():
    inputs = {"segments": [{"segment": "hello"}]}
    id_a = _make_scenario_id("radio-op", "family_a", inputs)
    id_b = _make_scenario_id("radio-op", "family_b", inputs)
    assert id_a != id_b


def test_scenario_dataclass_fields():
    scenarios = generate_smoke_scenarios("radio-op")
    for s in scenarios:
        assert isinstance(s, Scenario)
        assert len(s.scenario_id) == 12
        assert s.target == "radio-op"
        assert s.family_id
        assert s.kind in ("happy", "obvious_break")
        assert isinstance(s.inputs, dict)
        assert isinstance(s.expected_class, str)


def test_n_max_caps_per_family():
    scenarios = generate_smoke_scenarios("radio-op", n_max=1)
    # With n_max=1, no family should produce more than 1 scenario
    family_counts: dict[str, int] = {}
    for s in scenarios:
        family_counts[s.family_id] = family_counts.get(s.family_id, 0) + 1
    for fid, count in family_counts.items():
        assert count <= 1, f"Family {fid} produced {count} scenarios with n_max=1"


def test_scenario_to_dict():
    scenarios = generate_smoke_scenarios("cockpit")
    s = scenarios[0]
    d = s.to_dict()
    assert d["scenario_id"] == s.scenario_id
    assert d["target"] == s.target
    assert d["family_id"] == s.family_id
    assert "inputs" in d


def test_multi_scenario_family_ids_are_unique():
    """Families with n_scenarios>1 produce unique scenario IDs."""
    scenarios = generate_smoke_scenarios("radio-op")
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs detected"


def test_unknown_target_raises():
    with pytest.raises(FileNotFoundError):
        generate_smoke_scenarios("nonexistent-target-xyz")
