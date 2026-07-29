"""Unit and integration tests for council voicing provenance (effective-voicing tracking).

Covers:
  - GravityWellAdapter.voicing_events list is populated on chat() calls
  - Call to call_operator('gravitywell') with fallback records both calls
  - Council run record includes effective_voicing and voicing_degraded fields
  - Per-turn effective_voicing keys are added to turn records
  - Distinct failure reasons are recorded (doorman_unreachable, serving_http_error, gw_not_serving)
"""

import json

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
import yaml


# ---------------------------------------------------------------------------
# GravityWellAdapter voicing_events tracking tests
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_tracks_voicing_events():
    """GravityWellAdapter.chat() populates voicing_events list on successful call."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter
    from agents_core.doorman_client import DoormanClient

    adapter = GravityWellAdapter(temperature=0.8, timeout=300, on_wake_fail="sonnet")

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="test response"):

        @dataclass
        class Message:
            role: str
            content: str

        messages = [Message(role="user", content="test")]
        response = adapter.chat(system="test system", messages=messages)

    assert response == "test response"
    assert len(adapter.voicing_events) == 1
    assert adapter.voicing_events[0]["effective_operator"] == "gravitywell"
    assert adapter.voicing_events[0]["reason"] == "success"


def test_gravitywell_adapter_tracks_fallback_events():
    """GravityWellAdapter.chat() records fallback to Sonnet on GW failure with specific reason."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    adapter = GravityWellAdapter(temperature=0.8, timeout=300, on_wake_fail="sonnet")

    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="sonnet response"):

        @dataclass
        class Message:
            role: str
            content: str

        messages = [Message(role="user", content="test")]
        response = adapter.chat(system="test system", messages=messages)

    assert response == "sonnet response"
    assert len(adapter.voicing_events) == 1
    assert adapter.voicing_events[0]["effective_operator"] == "sonnet"
    # The actual failure reason (doorman_unreachable) is recorded, not generic "fallback"
    assert adapter.voicing_events[0]["reason"] == "doorman_unreachable"


def test_gravitywell_adapter_multiple_calls_accumulate_events():
    """GravityWellAdapter.chat() accumulates multiple voicing_events across turns."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter
    from agents_core.doorman_client import DoormanClient

    adapter = GravityWellAdapter(temperature=0.8, timeout=300, on_wake_fail="sonnet")

    @dataclass
    class Message:
        role: str
        content: str

    # First call - success
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw response 1"):
        messages = [Message(role="user", content="turn 1")]
        response = adapter.chat(system="sys", messages=messages)

    assert response == "gw response 1"
    assert len(adapter.voicing_events) == 1

    # Second call - also success
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw response 2"):
        messages = [Message(role="user", content="turn 2")]
        response = adapter.chat(system="sys", messages=messages)

    assert response == "gw response 2"
    assert len(adapter.voicing_events) == 2
    assert all(e["effective_operator"] == "gravitywell" for e in adapter.voicing_events)


def test_gravitywell_adapter_mixed_events():
    """GravityWellAdapter.chat() records mixed success/fallback across multiple calls."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    adapter = GravityWellAdapter(temperature=0.8, timeout=300, on_wake_fail="sonnet")

    @dataclass
    class Message:
        role: str
        content: str

    # First call - success
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw response 1"):
        messages = [Message(role="user", content="turn 1")]
        adapter.chat(system="sys", messages=messages)

    # Second call - fallback
    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="sonnet response 2"):
        messages = [Message(role="user", content="turn 2")]
        adapter.chat(system="sys", messages=messages)

    assert len(adapter.voicing_events) == 2
    assert adapter.voicing_events[0]["effective_operator"] == "gravitywell"
    assert adapter.voicing_events[1]["effective_operator"] == "sonnet"


# ---------------------------------------------------------------------------
# park-not-degrade tests (agents-core-council-park-not-degrade-v0)
# ---------------------------------------------------------------------------

def test_gravitywell_adapter_default_on_wake_fail_is_park():
    """GravityWellAdapter's default on_wake_fail is 'park' (fail-closed by default per
    decision/independence-blueprint-ratified-2026-07-28), not 'sonnet'. On GW unreachable,
    chat() raises GWParkedError uncaught rather than silently spending on Sonnet."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable
    from agents_core.llm import GWParkedError

    adapter = GravityWellAdapter(temperature=0.8, timeout=300)
    assert adapter.on_wake_fail == "park"

    @dataclass
    class Message:
        role: str
        content: str

    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait") as mock_submit:

        with pytest.raises(GWParkedError):
            adapter.chat(system="sys", messages=[Message(role="user", content="test")])

    mock_submit.assert_not_called()
    assert adapter.voicing_events == []


def _sample_roster():
    return [
        {
            "character_id": "alpha", "character_name": "Alpha", "pool": "reviewer",
            "cultural_context": "test context alpha.",
        },
        {
            "character_id": "beta", "character_name": "Beta", "pool": "reviewer",
            "cultural_context": "test context beta.",
        },
    ]


def test_select_entities_gw_unreachable_raises_gwparked_error(monkeypatch):
    """select_entities() propagates GWParkedError (not a RuntimeWarning-and-continue)
    when GravityWell is unreachable, per the park-not-degrade default."""
    from agents_core.council import cli
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable
    from agents_core.llm import GWParkedError

    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)

    with patch.object(cli, "find_card_path", return_value=object()), \
         patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait") as mock_submit:

        with pytest.raises(GWParkedError):
            cli.select_entities(
                decision="test decision",
                roster=_sample_roster(),
                context={"hits": [], "cohesion_findings": []},
                n=2,
                mode="deliberation",
            )

    mock_submit.assert_not_called()


def test_select_entities_poison_pill_sonnet_regression_is_detected(monkeypatch):
    """Poison-pill test (spec-review Facets synthesis, 2026-07-29): if a future edit
    reintroduces on_wake_fail='sonnet' at the select_entities() call site, this test
    harness must be able to catch it — proving the AC2 grep-is-clean check is backed by
    a runtime invariant, not just a static snapshot.

    Simulates the regression by forcing call_operator's on_wake_fail kwarg to 'sonnet'
    at the exact seam cli.select_entities() calls through (agents_core.council.cli.call_operator),
    then asserts the paid-Sonnet fallback actually fires when GW is unreachable.
    """
    from agents_core.council import cli
    from agents_core.llm import call_operator as real_call_operator
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)

    def _regressed_call_operator(*args, **kwargs):
        kwargs["on_wake_fail"] = "sonnet"
        return real_call_operator(*args, **kwargs)

    sonnet_json = json.dumps({"selected": ["alpha", "beta"], "reasoning": "poison-pill probe"})

    with patch.object(cli, "find_card_path", return_value=object()), \
         patch.object(cli, "call_operator", side_effect=_regressed_call_operator), \
         patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value=sonnet_json) as mock_submit:

        result = cli.select_entities(
            decision="test decision",
            roster=_sample_roster(),
            context={"hits": [], "cohesion_findings": []},
            n=2,
            mode="deliberation",
        )

    mock_submit.assert_called()
    assert result["selection_operator"] == "sonnet"
    assert sorted(result["selected"]) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Council run record voicing provenance tests
# ---------------------------------------------------------------------------

def test_apply_voicing_provenance_gravitywell_clean():
    """_apply_voicing_provenance() sets effective_voicing=gravitywell on clean run."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
        {"effective_operator": "gravitywell", "reason": "success"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}, {"step": 2}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "gravitywell"
    assert run["voicing_degraded"] is False
    assert "voicing_degraded_reason" not in run
    assert run["turns"][0]["effective_voicing"] == "gravitywell"
    assert run["turns"][1]["effective_voicing"] == "gravitywell"


def test_apply_voicing_provenance_gravitywell_degraded():
    """_apply_voicing_provenance() sets effective_voicing=sonnet on degraded run."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
        {"effective_operator": "sonnet", "reason": "doorman_unreachable"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}, {"step": 2}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "doorman_unreachable"
    assert run["turns"][0]["effective_voicing"] == "gravitywell"
    assert run["turns"][1]["effective_voicing"] == "sonnet"


def test_apply_voicing_provenance_stream_culled_is_degraded():
    """AC6 (spec-review-gw-generation-guards-v0): a gravitywell turn that culled-and-
    salvaged must NOT be reported as clean, even though gravitywell itself answered
    (no fallback operator involved) - a truncated deliberation is never honest as
    voicing_degraded=False."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
        {"effective_operator": "gravitywell", "reason": "stream_culled"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}, {"step": 2}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "stream_culled"
    # gravitywell did answer (just degraded) - not "unknown".
    assert run["effective_voicing"] == "gravitywell"
    assert run["turns"][1]["effective_voicing"] == "gravitywell"


def test_apply_voicing_provenance_doorman_unreachable_reason():
    """_apply_voicing_provenance() records doorman_unreachable as reason."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "doorman_unreachable"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "doorman_unreachable"


def test_apply_voicing_provenance_non_gravitywell_adapter():
    """_apply_voicing_provenance() with non-GW adapter sets effective=requested."""
    from agents_core.council.cli import _apply_voicing_provenance

    # Mock a non-GravityWell adapter (e.g., ClaudeAdapter)
    adapter = MagicMock()

    run = {
        "voicing": "sonnet",
        "turns": [],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is False
    assert "voicing_degraded_reason" not in run


def test_apply_voicing_provenance_none_adapter():
    """_apply_voicing_provenance() with None adapter (stub mode) preserves existing fields."""
    from agents_core.council.cli import _apply_voicing_provenance

    run = {
        "voicing": "sonnet",
        "effective_voicing": "sonnet",
        "voicing_degraded": False,
        "turns": [],
    }

    _apply_voicing_provenance(run, None)

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is False


def test_voicing_record_matches_requested_voicing():
    """Run record keeps requested voicing field unchanged (not overwritten by effective)."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "fallback"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    _apply_voicing_provenance(run, adapter)

    # voicing (requested) must remain unchanged
    assert run["voicing"] == "gravitywell"
    # effective_voicing (actual) is different
    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is True


# ---------------------------------------------------------------------------
# Failure reason prioritization tests
# ---------------------------------------------------------------------------

def test_voicing_degraded_reason_prioritizes_specific_errors():
    """_apply_voicing_provenance() records specific failure reasons."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    # Specific failure reason is recorded
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "serving_http_error"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    _apply_voicing_provenance(run, adapter)

    # Should record the specific failure reason
    assert run["voicing_degraded_reason"] == "serving_http_error"


def test_voicing_degraded_reason_all_gravitywell_clean():
    """_apply_voicing_provenance() with all gravitywell events doesn't set degraded_reason."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
        {"effective_operator": "gravitywell", "reason": "success"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}, {"step": 2}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["voicing_degraded"] is False
    assert "voicing_degraded_reason" not in run


# ---------------------------------------------------------------------------
# Integration tests (full run flow)
# ---------------------------------------------------------------------------

def test_council_run_with_gw_failure_records_provenance_in_yaml():
    """Full integration test: council run with GW fallback → run yaml has effective_voicing + degraded_reason."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    # Build a minimal run dict that would have been created by cmd_submit
    run = {
        "id": "test-council-run",
        "voicing": "gravitywell",  # Requested voicing
        "mode": "deliberation",
        "decision": "Test decision",
        "turns_cap": "2",
        "selected_entities": [],
        "turns": [
            {"step": 1, "content": "Turn 1 response"},
        ],
        "synthesis": {},
    }

    # Manually create the adapter with fallback history (simulating a failed GW call)
    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "doorman_unreachable"},
    ]

    # Apply the voicing provenance - this is what run_deliberation does after Engine().run()
    _apply_voicing_provenance(run, adapter)

    # Verify the run record has the correct fields
    assert run["voicing"] == "gravitywell", "Requested voicing must be unchanged"
    assert run["effective_voicing"] == "sonnet", "Effective voicing should be sonnet"
    assert run["voicing_degraded"] is True, "Should be marked as degraded"
    assert run["voicing_degraded_reason"] == "doorman_unreachable", "Reason should be specific"
    assert run["turns"][0]["effective_voicing"] == "sonnet", "Turn should have effective_voicing"


def test_council_run_with_gw_success_records_clean_provenance():
    """Integration test: council run with GW success records clean voicing (no degradation)."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
    ]

    _apply_voicing_provenance(run, adapter)

    # Verify clean run has correct fields
    assert run["voicing"] == "gravitywell", "Requested voicing must be unchanged"
    assert run["effective_voicing"] == "gravitywell", "Effective voicing should be gravitywell"
    assert run["voicing_degraded"] is False, "Should not be marked as degraded"
    assert "voicing_degraded_reason" not in run, "No reason field on clean run"
    assert run["turns"][0]["effective_voicing"] == "gravitywell", "Turn should have effective_voicing"


# ---------------------------------------------------------------------------
# Voicing signal emission tests (Leg A — stdout signal)
# ---------------------------------------------------------------------------

def test_voicing_signal_degraded_line_emitted(capsys):
    """Voicing signal: DEGRADED line is emitted when voicing_degraded is True."""
    from agents_core.council.cli import _apply_voicing_provenance, _emit_voicing_signal
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    run = {
        "run_id": "test-run-123",
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "doorman_unreachable"},
    ]

    _apply_voicing_provenance(run, adapter)
    _emit_voicing_signal(run, adapter, "test-run-123")

    captured = capsys.readouterr()
    assert "[council] VOICING DEGRADED" in captured.out
    assert "run_id=test-run-123" in captured.out
    assert "requested=gravitywell" in captured.out
    assert "effective=sonnet" in captured.out
    assert "reason=doorman_unreachable" in captured.out


def test_voicing_signal_clean_line_emitted(capsys):
    """Voicing signal: positive 'voicing ok' line is emitted when clean GW run."""
    from agents_core.council.cli import _apply_voicing_provenance, _emit_voicing_signal
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    run = {
        "run_id": "test-run-456",
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    adapter = GravityWellAdapter()
    adapter.voicing_events = [
        {"effective_operator": "gravitywell", "reason": "success"},
    ]

    _apply_voicing_provenance(run, adapter)
    _emit_voicing_signal(run, adapter, "test-run-456")

    captured = capsys.readouterr()
    assert "[council] voicing ok" in captured.out
    assert "run_id=test-run-456" in captured.out
    assert "effective=gravitywell" in captured.out
    # Should NOT contain DEGRADED
    assert "VOICING DEGRADED" not in captured.out


def test_voicing_signal_no_emission_for_non_gw_run(capsys):
    """Voicing signal: no signal emitted for non-GW runs (noise reduction)."""
    from agents_core.council.cli import _apply_voicing_provenance, _emit_voicing_signal
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    run = {
        "run_id": "test-run-789",
        "voicing": "sonnet",  # Not gravitywell
        "turns": [{"step": 1}],
    }

    adapter = GravityWellAdapter()
    adapter.voicing_events = []

    _apply_voicing_provenance(run, adapter)
    _emit_voicing_signal(run, adapter, "test-run-789")

    captured = capsys.readouterr()
    # No voicing signal should be emitted for non-GW runs
    assert "[council]" not in captured.out or "voicing" not in captured.out


def test_voicing_signal_sentinel_guard_no_events(capsys):
    """Voicing signal: SENTINEL guard prevents spurious 'voicing ok' when no voicing_events."""
    from agents_core.council.cli import _apply_voicing_provenance, _emit_voicing_signal
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    # This simulates the no-events path (line 975-976 in _apply_voicing_provenance)
    # where effective_voicing=gravitywell and voicing_degraded=False despite no events
    run = {
        "run_id": "test-run-noevents",
        "voicing": "gravitywell",
        "turns": [],
    }

    adapter = GravityWellAdapter()
    # No voicing_events - simulates the empty path

    _apply_voicing_provenance(run, adapter)

    # Verify that the function set the fields as expected for no-events path
    assert run["voicing_degraded"] is False
    assert run["effective_voicing"] == "gravitywell"
    assert len(adapter.voicing_events) == 0

    # Emit the signal with SENTINEL guard
    _emit_voicing_signal(run, adapter, "test-run-noevents")

    captured = capsys.readouterr()
    # Should NOT emit positive line (sentinel guard blocks it)
    assert "[council] voicing ok" not in captured.out
    assert "[council] VOICING DEGRADED" not in captured.out
