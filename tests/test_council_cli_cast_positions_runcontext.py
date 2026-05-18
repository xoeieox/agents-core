"""Test that _cast_positions passes a fully-populated RunContext to entity.act.

Skipped when lapis_engine is not importable (stub-mode environments).
"""
import json
import pytest

try:
    from lapis_engine import RunContext  # type: ignore

    HAS_LAPIS_ENGINE = True
except ImportError:
    HAS_LAPIS_ENGINE = False

pytestmark = pytest.mark.skipif(
    not HAS_LAPIS_ENGINE, reason="lapis_engine not installed"
)


class _RecordingEntity:
    """Stub entity that records the ctx passed to act() and returns 'agree'."""

    def __init__(self):
        self.captured_ctx = None

    def act(self, prompt: str, ctx):
        self.captured_ctx = ctx
        return json.dumps({"position": "agree", "reason": "", "basis": None})


def _make_run(n_turns: int, entity_ids: list) -> dict:
    return {
        "turns": [{"type": "deliberation_turn", "voice": entity_ids[i % len(entity_ids)]} for i in range(n_turns)],
        "selected_entities": [{"id": eid, "role": "character"} for eid in entity_ids],
        "synthesis": {
            "landing": "Test landing.",
            "open_questions": [],
        },
    }


def test_run_context_step_and_entity_ids():
    from agents_core.council.cli import _cast_positions

    entity_ids = ["alice", "bob"]
    n_turns = 3
    run = _make_run(n_turns, entity_ids)

    recorders = [_RecordingEntity(), _RecordingEntity()]

    _cast_positions(run, recorders, adapter=None)

    for recorder in recorders:
        ctx = recorder.captured_ctx
        assert isinstance(ctx, RunContext), f"expected RunContext, got {type(ctx)}"
        assert ctx.step == n_turns, f"expected step={n_turns}, got {ctx.step}"
        assert ctx.entity_ids == entity_ids, (
            f"expected entity_ids={entity_ids}, got {ctx.entity_ids}"
        )


def test_run_context_step_zero_turns():
    from agents_core.council.cli import _cast_positions

    entity_ids = ["carol"]
    run = _make_run(0, entity_ids)
    recorder = _RecordingEntity()

    _cast_positions(run, [recorder], adapter=None)

    ctx = recorder.captured_ctx
    assert ctx.step == 0
    assert ctx.entity_ids == entity_ids
