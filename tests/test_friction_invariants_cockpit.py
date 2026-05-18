"""Tests for cockpit programmatic invariant check functions."""
import pytest

from agents_core.friction_test.invariants_impl.cockpit import (
    c01_cockpit_provenance_present,
    c02a_directive_writes_to_commentstore,
    c02b_directive_write_latency,
    c03_unknown_tid_returns_4xx,
    c04_path_traversal_rejected,
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


def _sc(family="test", method="GET", path="/api/status", **inputs) -> Scenario:
    all_inputs = {"method": method, "path": path, **inputs}
    sid = _make_scenario_id("cockpit", family, all_inputs)
    return Scenario(
        scenario_id=sid, target="cockpit", family_id=family,
        kind="happy", inputs=all_inputs, expected_class="test",
    )


# ---------------------------------------------------------------------------
# c01_cockpit_provenance_present
# ---------------------------------------------------------------------------

def test_c01_held_provenance_present():
    s = _sc(method="GET", path="/cockpit/api/workday")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/cockpit/api/workday",
        "status": 200,
        "body": {"data": {}, "provenance": {"agent_id": "cockpit", "signature": "abc", "mode": "live"}},
        "latency_ms": 45.0,
    }])
    observed, distance = c01_cockpit_provenance_present(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_c01_dissonant_missing_provenance():
    s = _sc(method="GET", path="/cockpit/api/workday")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/cockpit/api/workday",
        "status": 200,
        "body": {"data": {}},  # no provenance field
        "latency_ms": 45.0,
    }])
    observed, distance = c01_cockpit_provenance_present(s, obs)
    assert observed == 1
    assert distance == 1.0


def test_c01_inapplicable_non_provenance_route():
    s = _sc(method="GET", path="/api/status")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/api/status",
        "status": 200,
        "body": {"ok": True},
        "latency_ms": 10.0,
    }])
    observed, distance = c01_cockpit_provenance_present(s, obs)
    assert observed is None


# ---------------------------------------------------------------------------
# c02a_directive_writes_to_commentstore
# ---------------------------------------------------------------------------

def test_c02a_held_one_row():
    s = _sc(method="POST", path="/api/thread/test-tid/directive", tid="test-tid")
    obs = _obs(s.scenario_id,
        http_calls=[{
            "method": "POST",
            "url": "http://localhost:8400/api/thread/test-tid/directive",
            "status": 200,
            "body": {"ok": True},
            "latency_ms": 80.0,
        }],
        log_appends=[{
            "file": "/srv/lapis/targets/comments/test-tid.jsonl",
            "lines_before": 0,
            "lines_after": 1,
            "new_count": 1,
        }],
    )
    observed, distance = c02a_directive_writes_to_commentstore(s, obs)
    assert observed == 1
    assert distance == 0.0


def test_c02a_dissonant_no_rows():
    s = _sc(method="POST", path="/api/thread/test-tid/directive", tid="test-tid")
    obs = _obs(s.scenario_id,
        http_calls=[{
            "method": "POST",
            "url": "http://localhost:8400/api/thread/test-tid/directive",
            "status": 200,
            "body": {"ok": True},
            "latency_ms": 80.0,
        }],
        log_appends=[{
            "file": "/srv/lapis/targets/comments/test-tid.jsonl",
            "lines_before": 0,
            "lines_after": 0,
            "new_count": 0,
        }],
    )
    observed, distance = c02a_directive_writes_to_commentstore(s, obs)
    assert observed == 0
    assert distance == 1.0


def test_c02a_inapplicable_no_directive_call():
    s = _sc(method="GET", path="/api/status")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET", "url": "/api/status", "status": 200, "body": {}, "latency_ms": 10.0,
    }])
    observed, distance = c02a_directive_writes_to_commentstore(s, obs)
    assert observed is None


# ---------------------------------------------------------------------------
# c02b_directive_write_latency
# ---------------------------------------------------------------------------

def test_c02b_held_within_tolerance():
    s = _sc(method="POST", path="/api/thread/test-tid/directive", tid="test-tid")
    obs = _obs(s.scenario_id,
        http_calls=[{
            "method": "POST",
            "url": "http://localhost:8400/api/thread/test-tid/directive",
            "status": 200, "body": {"ok": True}, "latency_ms": 150.0,
        }],
        log_appends=[{
            "file": "/srv/lapis/targets/comments/test-tid.jsonl",
            "write_latency_ms": 150.0,
            "new_count": 1,
        }],
    )
    observed, distance = c02b_directive_write_latency(s, obs)
    assert observed == 150.0
    assert distance == 0.0  # 150 <= 200+50


def test_c02b_dissonant_exceeds_tolerance():
    s = _sc(method="POST", path="/api/thread/test-tid/directive", tid="test-tid")
    obs = _obs(s.scenario_id,
        http_calls=[{
            "method": "POST",
            "url": "http://localhost:8400/api/thread/test-tid/directive",
            "status": 200, "body": {"ok": True}, "latency_ms": 400.0,
        }],
        log_appends=[{
            "file": "/srv/lapis/targets/comments/test-tid.jsonl",
            "write_latency_ms": 400.0,
            "new_count": 1,
        }],
    )
    observed, distance = c02b_directive_write_latency(s, obs)
    assert observed == 400.0
    assert distance > 0.0  # 400 > 200+50


def test_c02b_dissonant_classifies_as_model_likely():
    """c02b is provisional; dissonance => dissonant_model_likely."""
    from agents_core.friction_test.critique import critique, load_declared_invariants
    invs = load_declared_invariants("cockpit")
    c02b = next(inv for inv in invs if inv.invariant_id == "c02b_directive_write_latency")
    assert c02b.provisional is True

    s = _sc(method="POST", path="/api/thread/test-tid/directive", tid="test-tid")
    obs = _obs(s.scenario_id,
        http_calls=[{
            "method": "POST",
            "url": "http://localhost:8400/api/thread/test-tid/directive",
            "status": 200, "body": {"ok": True}, "latency_ms": 500.0,
        }],
        log_appends=[{
            "file": "/srv/lapis/targets/comments/test-tid.jsonl",
            "write_latency_ms": 500.0,
            "new_count": 1,
        }],
    )
    results = critique([s], [obs], [c02b], qwen_endpoint="http://127.0.0.1:19996")
    r = next(r for r in results if r.invariant_id == "c02b_directive_write_latency")
    assert r.status == "dissonant"
    assert r.classification == "dissonant_model_likely"


# ---------------------------------------------------------------------------
# c03_unknown_tid_returns_4xx
# ---------------------------------------------------------------------------

def test_c03_held_404_returned():
    s = _sc(method="POST", path="/api/thread/unknown-tid/directive", unknown_tid=True)
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "POST",
        "url": "http://localhost:8400/api/thread/unknown-tid/directive",
        "status": 404, "body": {"error": "not found"}, "latency_ms": 20.0,
    }])
    observed, distance = c03_unknown_tid_returns_4xx(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_c03_dissonant_500_returned():
    s = _sc(method="POST", path="/api/thread/unknown-tid/directive", unknown_tid=True)
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "POST",
        "url": "http://localhost:8400/api/thread/unknown-tid/directive",
        "status": 500, "body": {"error": "crash"}, "latency_ms": 20.0,
    }])
    observed, distance = c03_unknown_tid_returns_4xx(s, obs)
    assert observed == 1
    assert distance == 1.0


# ---------------------------------------------------------------------------
# c04_path_traversal_rejected
# ---------------------------------------------------------------------------

def test_c04_held_4xx_no_leak():
    s = _sc(method="GET", path="/api/documents/targets/../../etc/passwd")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/api/documents/targets/../../etc/passwd",
        "status": 400,
        "body": {"error": "invalid path"},
        "latency_ms": 5.0,
    }])
    observed, distance = c04_path_traversal_rejected(s, obs)
    assert observed == 0
    assert distance == 0.0


def test_c04_dissonant_path_leaked():
    s = _sc(method="GET", path="/api/documents/targets/../../etc/passwd")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/api/documents/targets/../../etc/passwd",
        "status": 400,
        "body": {"error": "file not found: /srv/agents/dashboard/targets/../../etc/passwd"},
        "latency_ms": 5.0,
    }])
    observed, distance = c04_path_traversal_rejected(s, obs)
    assert observed == 1
    assert distance == 1.0


def test_c04_inapplicable_normal_path():
    s = _sc(method="GET", path="/api/documents/targets/foo.yaml")
    obs = _obs(s.scenario_id, http_calls=[{
        "method": "GET",
        "url": "http://localhost:8400/api/documents/targets/foo.yaml",
        "status": 200, "body": {}, "latency_ms": 10.0,
    }])
    observed, distance = c04_path_traversal_rejected(s, obs)
    assert observed is None
