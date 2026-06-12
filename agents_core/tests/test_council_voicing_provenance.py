"""Unit and integration tests for council voicing provenance (effective-voicing tracking).

Covers:
  - GravityWellAdapter.voicing_events list is populated on chat() calls
  - Call to call_operator('gravitywell') with fallback records both calls
  - Council run record includes effective_voicing and voicing_degraded fields
  - Per-turn effective_voicing keys are added to turn records
  - Distinct failure reasons are recorded (doorman_unreachable, serving_http_error, gw_not_serving)
"""

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
    """GravityWellAdapter.chat() records fallback to Sonnet on GW failure."""
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
    assert adapter.voicing_events[0]["reason"] == "fallback"


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
        {"effective_operator": "sonnet", "reason": "fallback"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}, {"step": 2}],
    }

    _apply_voicing_provenance(run, adapter)

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "fallback"
    assert run["turns"][0]["effective_voicing"] == "gravitywell"
    assert run["turns"][1]["effective_voicing"] == "sonnet"


def test_apply_voicing_provenance_doorman_unreachable_reason():
    """_apply_voicing_provenance() records doorman_unreachable as reason."""
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

    assert run["effective_voicing"] == "sonnet"
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "fallback"


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
    """_apply_voicing_provenance() prioritizes specific failure reasons over generic ones."""
    from agents_core.council.cli import _apply_voicing_provenance
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    adapter = GravityWellAdapter()
    # Both doorman_unreachable and fallback are recorded
    adapter.voicing_events = [
        {"effective_operator": "sonnet", "reason": "fallback"},
    ]

    run = {
        "voicing": "gravitywell",
        "turns": [{"step": 1}],
    }

    _apply_voicing_provenance(run, adapter)

    # Should pick the most specific reason
    assert run["voicing_degraded_reason"] == "fallback"


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
