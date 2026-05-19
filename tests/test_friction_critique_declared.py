"""Tests for declared invariant loading and critique."""
import pytest

from agents_core.friction_test.critique import (
    InvariantResult,
    critique,
    load_declared_invariants,
)
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


def _make_obs(scenario_id: str, **kwargs) -> Observation:
    return Observation(
        scenario_id=scenario_id,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        **kwargs,
    )


def _make_scenario(target: str, family_id: str = "test", **inputs) -> Scenario:
    sid = _make_scenario_id(target, family_id, inputs or {})
    return Scenario(
        scenario_id=sid,
        target=target,
        family_id=family_id,
        kind="happy",
        inputs=inputs or {},
        expected_class="test",
    )


def test_radio_op_invariants_load():
    invs = load_declared_invariants("radio-op")
    assert len(invs) >= 6
    ids = {inv.invariant_id for inv in invs}
    for expected_id in [
        "i01_consult_log_row_per_judgment",
        "i02_authority_tier_host_determined",
        "i03_should_surface_default_false",
        "i04_harvest_marker_iff_explicit_gesture",
        "i05_synapse_degradation_handled",
        "i06_label_isolation",
    ]:
        assert expected_id in ids


def test_cockpit_invariants_load():
    invs = load_declared_invariants("cockpit")
    assert len(invs) >= 4
    ids = {inv.invariant_id for inv in invs}
    for expected_id in [
        "c01_cockpit_provenance_present",
        "c02a_directive_writes_to_commentstore",
        "c02b_directive_write_latency",
        "c03_unknown_tid_returns_4xx",
        "c04_path_traversal_rejected",
    ]:
        assert expected_id in ids


def test_unknown_target_raises():
    with pytest.raises(FileNotFoundError):
        load_declared_invariants("nonexistent-target")


def test_critique_returns_invariant_results():
    invs = load_declared_invariants("radio-op")
    s = _make_scenario("radio-op", "test", segments=[], harvest=False)
    obs = _make_obs(s.scenario_id)
    results = critique([s], [obs], invs, qwen_endpoint="http://127.0.0.1:19997")
    assert len(results) > 0
    for r in results:
        assert isinstance(r, InvariantResult)


def test_invariant_result_has_valid_shape():
    invs = load_declared_invariants("cockpit")
    s = _make_scenario("cockpit", "test")
    obs = _make_obs(s.scenario_id)
    results = critique([s], [obs], invs, qwen_endpoint="http://127.0.0.1:19997")
    for r in results:
        assert r.status in ("held", "dissonant", "inapplicable")
        assert r.classification in (
            "held", "dissonant_model_likely", "dissonant_system_likely", "inapplicable"
        )
        assert isinstance(r.justification, str)
        assert isinstance(r.evidence_refs, list)
        d = r.to_dict()
        for key in ("scenario_id", "invariant_id", "dimension", "expected", "observed",
                    "distance", "tolerance", "status", "classification",
                    "proposed_by", "provisional", "justification", "evidence_refs"):
            assert key in d


def test_provisional_flag_preserved():
    """c02b should be provisional=True, others provisional=False."""
    invs = load_declared_invariants("cockpit")
    by_id = {inv.invariant_id: inv for inv in invs}
    assert by_id["c02b_directive_write_latency"].provisional is True
    assert by_id["c01_cockpit_provenance_present"].provisional is False
    assert by_id["c02a_directive_writes_to_commentstore"].provisional is False
