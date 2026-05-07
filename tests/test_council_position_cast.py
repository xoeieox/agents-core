"""Tests for _cast_positions — per-voice position casting (v0.next)."""
from __future__ import annotations

import os
import pytest
from datetime import datetime


def _make_run(turns=None, turns_cap=8):
    return {
        "run_id": "2026-05-07-cast-test",
        "mode": "deliberation",
        "decision": "Test decision for position casting.",
        "selected_entities": [
            {"id": "entity-a", "role": "first_voice"},
            {"id": "entity-b", "role": "second_voice"},
        ],
        "turns_cap": turns_cap,
        "turns": turns if turns is not None else [
            {
                "step": 1,
                "speaker": "entity-a",
                "type": "deliberation_turn",
                "content": "I think the core issue is reversibility.",
                "timestamp": "2026-05-07T12:00:00",
            }
        ],
        "synthesis": {
            "landing": "We agree on the core approach.",
            "open_questions": [],
            "confidence": "converged",
        },
    }


class _MockEntity:
    """Mock entity whose act() returns a fixed JSON string."""
    def __init__(self, response: str, entity_id: str = "mock-entity"):
        self.id = entity_id
        self._response = response
        self.last_prompt = None

    def act(self, prompt, ctx=None):
        self.last_prompt = prompt
        return self._response


def test_cast_positions_returns_one_per_voice(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-a"),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    positions = _cast_positions(_make_run(), entities, adapter=None)
    assert len(positions) == 2


def test_cast_positions_parses_agree(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-a"),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    positions = _cast_positions(_make_run(), entities, adapter=None)
    assert positions[0]["position"] == "agree"
    assert positions[0]["basis"] is None


def test_cast_positions_parses_stand_aside_with_reason(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity(
            '{"position": "stand-aside", "reason": "I have reservations about timeline.", "basis": null}',
            "entity-a",
        ),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    positions = _cast_positions(_make_run(), entities, adapter=None)
    assert positions[0]["position"] == "stand-aside"
    assert "reservations" in positions[0]["reason"]


def test_cast_positions_parses_block_with_basis(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity(
            '{"position": "block", "reason": "This violates the core principle.", "basis": "kernel.invariant.4"}',
            "entity-a",
        ),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    positions = _cast_positions(_make_run(), entities, adapter=None)
    assert positions[0]["position"] == "block"
    assert positions[0]["basis"] == "kernel.invariant.4"


def test_cast_positions_rejects_unknown_position(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity('{"position": "abstain", "reason": "Not sure.", "basis": null}', "entity-a"),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    with pytest.raises(RuntimeError, match="abstain"):
        _cast_positions(_make_run(), entities, adapter=None)


def test_cast_positions_rejects_stand_aside_without_reason(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity('{"position": "stand-aside", "reason": "", "basis": null}', "entity-a"),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    with pytest.raises(RuntimeError):
        _cast_positions(_make_run(), entities, adapter=None)


def test_cast_positions_rejects_block_without_basis(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entities = [
        _MockEntity(
            '{"position": "block", "reason": "Strong objection here.", "basis": null}',
            "entity-a",
        ),
        _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b"),
    ]
    with pytest.raises(RuntimeError):
        _cast_positions(_make_run(), entities, adapter=None)


def test_cast_positions_stub_mode_uses_env_var(monkeypatch):
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,block")
    from agents_core.council.cli import _cast_positions
    positions = _cast_positions(_make_run(), entities=None, adapter=None)
    assert len(positions) == 2
    assert positions[0]["position"] == "agree"
    assert positions[1]["position"] == "block"
    assert positions[1]["basis"] == "kernel.invariant.1"
    assert positions[1]["reason"] == "[stub] block reason"


def test_cast_positions_includes_transcript_in_prompt(monkeypatch):
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)
    from agents_core.council.cli import _cast_positions
    entity_a = _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-a")
    entity_b = _MockEntity('{"position": "agree", "reason": "", "basis": null}', "entity-b")
    run = _make_run(turns=[
        {
            "step": 1,
            "speaker": "entity-a",
            "type": "deliberation_turn",
            "content": "The core issue is reversibility of the decision.",
            "timestamp": "2026-05-07T12:00:00",
        }
    ])
    _cast_positions(run, [entity_a, entity_b], adapter=None)
    assert "TRANSCRIPT" in entity_a.last_prompt
    assert "reversibility" in entity_a.last_prompt
