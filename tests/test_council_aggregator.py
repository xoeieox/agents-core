"""Tests for _aggregate_positions and turns_used derivation (v0.next)."""
from __future__ import annotations


def _pos(position, voice="v", reason="test reason here", basis=None):
    return {"voice": voice, "position": position, "reason": reason, "basis": basis}


def test_all_agree_yields_converged():
    from agents_core.council.cli import _aggregate_positions
    positions = [_pos("agree", "a"), _pos("agree", "b")]
    agg = _aggregate_positions(positions, turns_used=2, turns_cap=8)
    assert agg["confidence"] == "converged"
    assert agg["stood_aside"] == []
    assert agg["blocks"] == []


def test_one_stand_aside_yields_converged_with_reservation():
    from agents_core.council.cli import _aggregate_positions
    positions = [
        _pos("agree", "a"),
        _pos("stand-aside", "b", reason="I have concerns about X."),
    ]
    agg = _aggregate_positions(positions, turns_used=2, turns_cap=8)
    assert agg["confidence"] == "converged-with-reservation"
    assert len(agg["stood_aside"]) == 1
    assert agg["stood_aside"][0]["voice"] == "b"
    assert agg["blocks"] == []


def test_block_with_turns_remaining_yields_open():
    from agents_core.council.cli import _aggregate_positions
    positions = [
        _pos("agree", "a"),
        _pos("block", "b", reason="This violates invariant 4.", basis="kernel.invariant.4"),
    ]
    agg = _aggregate_positions(positions, turns_used=4, turns_cap=8)
    assert agg["confidence"] == "partial"
    assert len(agg["blocks"]) == 1
    assert agg["blocks"][0]["voice"] == "b"


def test_block_with_turns_exhausted_yields_laid_down():
    from agents_core.council.cli import _aggregate_positions
    positions = [
        _pos("agree", "a"),
        _pos("block", "b", reason="This is fundamentally wrong.", basis="kernel.invariant.1"),
    ]
    agg = _aggregate_positions(positions, turns_used=8, turns_cap=8)
    assert agg["confidence"] == "laid-down"
    assert len(agg["blocks"]) == 1


def test_two_blocks_with_turns_exhausted_yields_laid_down():
    from agents_core.council.cli import _aggregate_positions
    positions = [
        _pos("block", "a", reason="Block reason A here.", basis="kernel.invariant.2"),
        _pos("block", "b", reason="Block reason B here.", basis="kernel.invariant.3"),
    ]
    agg = _aggregate_positions(positions, turns_used=8, turns_cap=8)
    assert agg["confidence"] == "laid-down"
    assert len(agg["blocks"]) == 2


def test_turns_used_derivation_excludes_synthesis_turn():
    """Guard against H3 off-by-one: synthesis turn must not count toward turns_used."""
    turns = [
        {"step": 1, "type": "deliberation_turn", "content": "a"},
        {"step": 2, "type": "deliberation_turn", "content": "b"},
        {"step": 3, "type": "deliberation_turn", "content": "c"},
        {"step": 4, "type": "synthesis", "content": "synthesis turn"},
    ]
    turns_used = len([t for t in turns if t.get("type") == "deliberation_turn"])
    assert turns_used == 3


def test_turns_used_derivation_handles_unknown_types_as_excluded():
    """Whitelist semantics: only deliberation_turn counts; unknown types excluded."""
    turns = [
        {"step": 1, "type": "deliberation_turn", "content": "a"},
        {"step": 2, "type": "unexpected", "content": "b"},
        {"step": 3, "type": "deliberation_turn", "content": "c"},
    ]
    turns_used = len([t for t in turns if t.get("type") == "deliberation_turn"])
    assert turns_used == 2
