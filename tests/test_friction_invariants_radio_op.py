"""Tests for radio-op programmatic invariant check functions."""
import pytest

from agents_core.friction_test.invariants_impl.radio_op import (
    i01_consult_log_row_per_judgment,
    i02_authority_tier_host_determined,
    i03_should_surface_default_false,
    i04_harvest_marker_iff_explicit_gesture,
    i05_synapse_degradation_handled,
    i06_label_isolation,
)
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


def _obs(scenario_id: str, **kwargs) -> Observation:
    return Observation(
        scenario_id=scenario_id,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        **kwargs,
    )


def _sc(family="test", segments=None, harvest=False, **extra) -> Scenario:
    inputs = {"segments": segments or [], "harvest": harvest, **extra}
    sid = _make_scenario_id("radio-op", family, inputs)
    return Scenario(
        scenario_id=sid, target="radio-op", family_id=family,
        kind="happy", inputs=inputs, expected_class="test",
    )


# ---------------------------------------------------------------------------
# i01_consult_log_row_per_judgment
# ---------------------------------------------------------------------------

def test_i01_held_one_segment_one_row():
    s = _sc(segments=[{"segment": "hello world", "ts": "t1"}])
    obs = _obs(s.scenario_id, log_appends=[
        {"file": f"/srv/lapis/radio/consults/{s.scenario_id}.jsonl", "line": {"verdict": "judge"}}
    ])
    observed, distance = i01_consult_log_row_per_judgment(s, obs)
    assert observed == 1
    assert distance == 0.0


def test_i01_dissonant_missing_row():
    s = _sc(segments=[{"segment": "hello world", "ts": "t1"}])
    obs = _obs(s.scenario_id, log_appends=[])  # no rows
    observed, distance = i01_consult_log_row_per_judgment(s, obs)
    assert observed == 0
    assert distance == 1.0


def test_i01_inapplicable_empty_segment():
    s = _sc(segments=[{"segment": "", "ts": "t1"}])
    obs = _obs(s.scenario_id)
    observed, distance = i01_consult_log_row_per_judgment(s, obs)
    assert observed is None


def test_i01_inapplicable_no_segments():
    s = _sc(segments=[])
    obs = _obs(s.scenario_id)
    observed, distance = i01_consult_log_row_per_judgment(s, obs)
    assert observed is None


# ---------------------------------------------------------------------------
# i02_authority_tier_host_determined
# ---------------------------------------------------------------------------

def test_i02_held_no_overrides():
    s = _sc()
    obs = _obs(s.scenario_id)
    observed, distance = i02_authority_tier_host_determined(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_i02_dissonant_tier_override_in_log():
    s = _sc()
    obs = _obs(s.scenario_id, log_appends=[{
        "file": "/srv/lapis/radio/consults/x.jsonl",
        "line": {"fragments": [
            {"id": "decision/foo", "authority_tier": "low"}  # should be "high"
        ]},
    }])
    observed, distance = i02_authority_tier_host_determined(s, obs)
    assert observed == 1
    assert distance == 1.0


# ---------------------------------------------------------------------------
# i03_should_surface_default_false
# ---------------------------------------------------------------------------

def test_i03_held_empty_segment():
    s = _sc(segments=[{"segment": "", "ts": "t1"}])
    obs = _obs(s.scenario_id)
    observed, distance = i03_should_surface_default_false(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_i03_dissonant_empty_segment_surfaces():
    s = _sc(segments=[{"segment": "", "ts": "t1"}])
    obs = _obs(s.scenario_id, sse_events=[{"fragments": [{"id": "f1"}]}])
    observed, distance = i03_should_surface_default_false(s, obs)
    assert observed == 1
    assert distance == 1.0


def test_i03_inapplicable_nontrivial_segment():
    s = _sc(segments=[{"segment": "This is a real message with substance", "ts": "t1"}])
    obs = _obs(s.scenario_id)
    observed, distance = i03_should_surface_default_false(s, obs)
    assert observed is None


# ---------------------------------------------------------------------------
# i04_harvest_marker_iff_explicit_gesture
# ---------------------------------------------------------------------------

def test_i04_held_gesture_and_marker():
    s = _sc(harvest=True)
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": True, "session_id": "x", "label": f"friction-test-{s.scenario_id}"}
    ])
    observed, distance = i04_harvest_marker_iff_explicit_gesture(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_i04_held_no_gesture_no_marker():
    s = _sc(harvest=False)
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": False, "session_id": "x", "label": f"friction-test-{s.scenario_id}"}
    ])
    observed, distance = i04_harvest_marker_iff_explicit_gesture(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_i04_dissonant_marker_without_gesture():
    s = _sc(harvest=False)
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": True, "session_id": "x", "label": f"friction-test-{s.scenario_id}"}
    ])
    observed, distance = i04_harvest_marker_iff_explicit_gesture(s, obs)
    assert observed == 1
    assert distance == 1.0


def test_i04_dissonant_system_likely():
    """i04 is non-provisional; dissonance should classify as system_likely."""
    from agents_core.friction_test.critique import critique, load_declared_invariants
    invs = load_declared_invariants("radio-op")
    i04 = next(inv for inv in invs if inv.invariant_id == "i04_harvest_marker_iff_explicit_gesture")
    assert i04.provisional is False

    s = _sc(harvest=False)
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": True, "session_id": "x", "label": f"friction-test-{s.scenario_id}"}
    ])
    results = critique([s], [obs], [i04], qwen_endpoint="http://127.0.0.1:19996")
    r = next(r for r in results if r.invariant_id == "i04_harvest_marker_iff_explicit_gesture")
    assert r.status == "dissonant"
    assert r.classification == "dissonant_system_likely"


# ---------------------------------------------------------------------------
# i05_synapse_degradation_handled
# ---------------------------------------------------------------------------

def test_i05_inapplicable_normal_scenario():
    s = _sc()
    obs = _obs(s.scenario_id)
    observed, distance = i05_synapse_degradation_handled(s, obs)
    assert observed is None


def test_i05_held_degraded_methodology():
    s = _sc(synapse_down=True)
    obs = _obs(s.scenario_id, log_appends=[{
        "file": "/srv/lapis/radio/consults/x.jsonl",
        "line": {"methodology": {"retrieve_pass": "degraded"}},
    }])
    observed, distance = i05_synapse_degradation_handled(s, obs)
    assert observed == 1
    assert distance == 0.0


def test_i05_dissonant_unhandled_exception():
    s = _sc(synapse_down=True)
    obs = _obs(s.scenario_id, errors=[{"stage": "ingest", "error": "unhandled exception: 500"}])
    observed, distance = i05_synapse_degradation_handled(s, obs)
    assert observed == 0
    assert distance == 1.0


# ---------------------------------------------------------------------------
# i06_label_isolation
# ---------------------------------------------------------------------------

def test_i06_held_correct_label():
    s = _sc()
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": False, "session_id": "foyer-xyz", "label": f"friction-test-{s.scenario_id}"}
    ])
    observed, distance = i06_label_isolation(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_i06_dissonant_wrong_label():
    s = _sc()
    obs = _obs(s.scenario_id, log_appends=[
        {"harvest_marker_exists": False, "session_id": "foyer-xyz", "label": "production-session-001"}
    ])
    observed, distance = i06_label_isolation(s, obs)
    assert observed == 1
    assert distance == 1.0
