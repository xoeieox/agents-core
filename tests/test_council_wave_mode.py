"""Tests for council-wave-mode-v0: wave mode CLI wiring, fan-out executor,
provenance fix, timeouts, lease-refresh cadence, digest fallback, and the
122B refusal.

Regression fence: deliberation and scene modes are exercised elsewhere
(test_council_module.py, test_gw_principal_unification.py, ...) and are
intentionally left untouched by this file.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("lapis_engine")


# ---------------------------------------------------------------------------
# D1 — the five CLI enforcement sites
# ---------------------------------------------------------------------------


def test_wave_n_range_matches_lapis_engine():
    """Two drifting seat ranges is a latent bug (D1.2) — assert parity."""
    from lapis_engine.directors import WAVE_N_RANGE as engine_range
    from agents_core.council.cli import WAVE_N_RANGE as cli_range

    assert cli_range == engine_range == (5, 7)


def test_valid_modes_includes_wave():
    from agents_core.council.cli import VALID_MODES

    assert "wave" in VALID_MODES


@pytest.mark.parametrize("n", [5, 6, 7])
def test_validate_mode_n_wave_accepts_in_range(n):
    from agents_core.council.cli import _validate_mode_n

    _validate_mode_n("wave", n)  # must not raise


@pytest.mark.parametrize("n", [4, 8])
def test_validate_mode_n_wave_rejects_out_of_range(n):
    from agents_core.council.cli import _validate_mode_n

    with pytest.raises(ValueError):
        _validate_mode_n("wave", n)


def test_role_assignments_wave_emits_indexed_slots():
    """No fourth_voice/fifth_voice invention — indexed roles per the scene
    precedent (D1.3, the 'silent' site)."""
    from agents_core.council.cli import _role_assignments

    ids = [f"voice-{i}" for i in range(6)]
    result = _role_assignments(ids, mode="wave")
    assert [r["role"] for r in result] == [f"wave_slot_{i}" for i in range(6)]
    assert [r["id"] for r in result] == ids


def test_build_director_dispatches_wave():
    from lapis_engine.directors import WaveDirector
    from agents_core.council.cli import _build_director

    director = _build_director(
        mode="wave", prompt="should we do X?", turns=3,
        DeliberationDirector=None, SceneDirector=None,
        WaveDirector=WaveDirector,
    )
    assert isinstance(director, WaveDirector)
    assert director.decision == "should we do X?"
    assert director.rounds == 3


def test_build_director_wave_without_wavedirector_raises():
    from agents_core.council.cli import _build_director

    with pytest.raises(ValueError):
        _build_director(
            mode="wave", prompt="x", turns=3,
            DeliberationDirector=None, SceneDirector=None,
        )


def test_build_director_unknown_mode_still_raises_valueerror():
    """D1.4 — the fall-through ValueError must survive the wave dispatch add."""
    from agents_core.council.cli import _build_director

    with pytest.raises(ValueError, match="Unknown mode"):
        _build_director(
            mode="not-a-mode", prompt="x", turns=3,
            DeliberationDirector=None, SceneDirector=None,
        )


# ---------------------------------------------------------------------------
# D2 — the injected fan-out executor
# ---------------------------------------------------------------------------


class _StubEntity:
    def __init__(self, id_, delay=0.0, raises=False):
        self.id = id_
        self.delay = delay
        self.raises = raises
        self.observed = []

    def act(self, prompt, ctx):
        time.sleep(self.delay)
        if self.raises:
            raise RuntimeError(f"{self.id} blew up")
        return f"{self.id} said: {prompt}"

    def observe(self, event):
        self.observed.append(event)


def test_concurrent_wave_executor_preserves_seat_order_despite_completion_order():
    from agents_core.council.wave_executor import concurrent_wave_executor

    # seat 0 is slowest, seat 2 completes first — completion order != seat order.
    entities = [_StubEntity("seat-0", delay=0.15), _StubEntity("seat-1", delay=0.05), _StubEntity("seat-2", delay=0.0)]
    pairs = [(e, f"prompt-for-{e.id}") for e in entities]
    results = concurrent_wave_executor(pairs, ctx=None, max_concurrent=3)

    assert len(results) == 3
    assert results[0] == "seat-0 said: prompt-for-seat-0"
    assert results[1] == "seat-1 said: prompt-for-seat-1"
    assert results[2] == "seat-2 said: prompt-for-seat-2"


def test_concurrent_wave_executor_isolates_seat_failure():
    from lapis_engine.types import SeatError
    from agents_core.council.wave_executor import concurrent_wave_executor

    entities = [_StubEntity("ok-0"), _StubEntity("boom-1", raises=True), _StubEntity("ok-2")]
    pairs = [(e, "p") for e in entities]
    results = concurrent_wave_executor(pairs, ctx=None, max_concurrent=3)

    assert len(results) == 3
    assert results[0] == "ok-0 said: p"
    assert isinstance(results[1], SeatError)
    assert results[1].seat_id == "boom-1"
    assert results[1].error_class == "RuntimeError"
    assert results[2] == "ok-2 said: p"


def test_concurrent_wave_executor_never_shortens_the_round():
    """A short round corrupts the transcript every downstream consumer
    re-parses — length-preservation is a hard invariant, not a nicety."""
    from agents_core.council.wave_executor import concurrent_wave_executor

    entities = [_StubEntity(f"seat-{i}", raises=(i % 2 == 0)) for i in range(5)]
    pairs = [(e, "p") for e in entities]
    results = concurrent_wave_executor(pairs, ctx=None, max_concurrent=4)
    assert len(results) == len(pairs) == 5


def test_concurrent_wave_executor_fires_on_seat_complete_before_all_done():
    """H6's whole premise: a per-seat-completion hook must fire as each seat
    actually finishes, not only once the round has fully drained."""
    from agents_core.council.wave_executor import concurrent_wave_executor

    entities = [_StubEntity("slow", delay=0.3), _StubEntity("fast", delay=0.0)]
    pairs = [(e, "p") for e in entities]
    fired = []
    lock = threading.Lock()

    def _on_complete(entity, response):
        with lock:
            fired.append((entity.id, time.monotonic()))

    start = time.monotonic()
    concurrent_wave_executor(pairs, ctx=None, max_concurrent=2, on_seat_complete=_on_complete)

    assert [f[0] for f in fired] == ["fast", "slow"], "fast seat must complete (and fire) first"
    # The fast seat's callback must fire well before the slow seat's delay elapses —
    # i.e. before the whole round drains — not bunched at the end.
    assert (fired[0][1] - start) < 0.2


# ---------------------------------------------------------------------------
# D3/H1/H4 — per-seat provenance, aggregated without positional slicing
# ---------------------------------------------------------------------------


class _FakeSeatAdapter:
    def __init__(self, events):
        self.voicing_events = events


def test_apply_wave_voicing_provenance_clean():
    from agents_core.council.cli import _apply_wave_voicing_provenance

    seat_adapters = [
        _FakeSeatAdapter([{"effective_operator": "gravitywell", "reason": "success"}]),
        _FakeSeatAdapter([{"effective_operator": "gravitywell", "reason": "success"}]),
    ]
    run = {}
    _apply_wave_voicing_provenance(run, seat_adapters)
    assert run["effective_voicing"] == "gravitywell"
    assert run["voicing_degraded"] is False


def test_apply_wave_voicing_provenance_degraded_by_one_seat():
    from agents_core.council.cli import _apply_wave_voicing_provenance

    seat_adapters = [
        _FakeSeatAdapter([{"effective_operator": "gravitywell", "reason": "success"}]),
        _FakeSeatAdapter([{"effective_operator": "sonnet", "reason": "gw_not_serving"}]),
    ]
    run = {}
    _apply_wave_voicing_provenance(run, seat_adapters)
    assert run["voicing_degraded"] is True
    assert run["voicing_degraded_reason"] == "gw_not_serving"


def test_wave_uses_per_seat_adapters_sharing_one_principal(tmp_path, monkeypatch):
    """H4: the per-adapter route is a trap unless every seat shares one
    principal — splitting into per-seat adapters without threading the same
    principal converts a wave into N separate GW admission groups."""
    from agents_core.council import cli as council_cli

    captured_principals = []
    orig_init = council_cli.GravityWellAdapter.__init__

    def _spy_init(self, *a, **kw):
        captured_principals.append(kw.get("principal"))
        orig_init(self, *a, **kw)

    entities = [_StubEntity(f"seat-{i}") for i in range(5)]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    run = {
        "run_id": "test-wave-principal",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
        "gw_principal": "shared-P",
    }

    with patch.object(council_cli.GravityWellAdapter, "__init__", _spy_init), \
         patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        from lapis_engine.engine import Engine
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-principal", Engine=Engine,
            hold_active=False, doorman=None, hold_work_id="w", hold_principal="p",
            refresh_threads=[],
            CharacterEntity=object, NarratorEntity=object,
        )

    assert captured_principals, "no seat adapters were constructed"
    assert all(p == "shared-P" for p in captured_principals), captured_principals


# ---------------------------------------------------------------------------
# D5 — run-YAML shape: consecutive turns[] entries, deliberation_turn type
# ---------------------------------------------------------------------------


def test_wave_run_seat_order_preserved_in_turns_yaml(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli
    from lapis_engine.engine import Engine

    # seat 0 is slowest so completion order != seat order; turns[] must still
    # come out in seat order (D2 contract + D5 shape preservation).
    entities = [
        _StubEntity("seat-0", delay=0.1),
        _StubEntity("seat-1", delay=0.05),
        _StubEntity("seat-2", delay=0.0),
        _StubEntity("seat-3", delay=0.0),
        _StubEntity("seat-4", delay=0.0),
    ]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    run = {
        "run_id": "test-wave-order",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
    }

    with patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-order", Engine=Engine,
            hold_active=False, doorman=None, hold_work_id="w", hold_principal="p",
            refresh_threads=[],
            CharacterEntity=object, NarratorEntity=object,
        )

    wave_turns = [t for t in run["turns"] if t["type"] == "deliberation_turn"]
    assert [t["speaker"] for t in wave_turns] == [f"seat-{i}" for i in range(5)]
    assert all(t["wave_round"] == 0 for t in wave_turns)
    # Synthesis is present and distinct from deliberation_turn.
    assert any(t["type"] == "synthesis" for t in run["turns"])


def test_wave_seat_failure_never_typed_deliberation_turn(tmp_path):
    """A failed seat must surface as a failed seat, never a silently short
    round, and never pollute _render_transcript / turns_used, which filter
    strictly on type == 'deliberation_turn'."""
    from agents_core.council import cli as council_cli
    from lapis_engine.engine import Engine

    entities = [
        _StubEntity("seat-0"),
        _StubEntity("seat-1", raises=True),
        _StubEntity("seat-2"),
        _StubEntity("seat-3"),
        _StubEntity("seat-4"),
    ]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    run = {
        "run_id": "test-wave-failure",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
    }

    with patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-failure", Engine=Engine,
            hold_active=False, doorman=None, hold_work_id="w", hold_principal="p",
            refresh_threads=[],
            CharacterEntity=object, NarratorEntity=object,
        )

    failure_turns = [t for t in run["turns"] if t["speaker"] == "seat-1"]
    assert len(failure_turns) == 1
    assert failure_turns[0]["type"] == "wave_seat_failure"
    assert failure_turns[0]["type"] != "deliberation_turn"

    turns_used = len([t for t in run["turns"] if t["type"] == "deliberation_turn"])
    assert turns_used == 4  # the 4 seats that answered, not the 5th (failed) seat


def test_wave_seat_failure_marks_run_degraded_distinct_from_voicing_degraded(tmp_path):
    """D6b: run-level degraded flag distinct from voicing_degraded."""
    from agents_core.council import cli as council_cli
    from lapis_engine.engine import Engine

    entities = [_StubEntity(f"seat-{i}", raises=(i == 2)) for i in range(5)]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    run = {
        "run_id": "test-wave-degraded",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
    }

    with patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-degraded", Engine=Engine,
            hold_active=False, doorman=None, hold_work_id="w", hold_principal="p",
            refresh_threads=[],
            CharacterEntity=object, NarratorEntity=object,
        )

    assert run["seats_degraded"] is True
    assert 0 in run["failed_seats"] and "seat-2" in run["failed_seats"][0]
    # voicing_degraded tracks a different fault (adapter fell back off GW) —
    # here every successful seat answered via a real GravityWellAdapter with
    # no voicing_events recorded (no real .chat() call happened), so it
    # defaults to the no-events "assume success" path. The two flags must
    # remain independently settable, not conflated into one boolean.
    assert "seats_degraded" in run and "voicing_degraded" in run
    assert run["seats_degraded"] != run.get("voicing_degraded")


# ---------------------------------------------------------------------------
# D6 — digest failure posture
# ---------------------------------------------------------------------------


def test_digest_failure_falls_back_to_verbatim_prior_round():
    from agents_core.council.wave_digest import build_round_digest

    with patch("agents_core.council.wave_digest.call_operator", side_effect=RuntimeError("gw down")):
        digest = build_round_digest(0, [("seat-a", "hello"), ("seat-b", "world")], principal="P")

    assert "seat-a: hello" in digest
    assert "seat-b: world" in digest


def test_digest_empty_result_falls_back_to_verbatim():
    from agents_core.council.wave_digest import build_round_digest

    with patch("agents_core.council.wave_digest.call_operator", return_value=None):
        digest = build_round_digest(0, [("seat-a", "hello")], principal="P")

    assert digest == "seat-a: hello"


def test_digest_no_seats_returns_empty():
    from agents_core.council.wave_digest import build_round_digest

    assert build_round_digest(0, [], principal="P") == ""


# ---------------------------------------------------------------------------
# D7 — refuse to run against gravitywell-122b
# ---------------------------------------------------------------------------


def test_refuse_wave_against_122b_raises():
    from agents_core.council.cli import _refuse_wave_against_122b, GW_122B_MODEL

    with patch("agents_core.llm._gw_default_model", return_value=GW_122B_MODEL):
        with pytest.raises(ValueError, match="gravitywell-122b"):
            _refuse_wave_against_122b("gravitywell")


def test_refuse_wave_against_v4flash_raises():
    """agents-core-doorman-big-seat-membership-v0 DoD 8: the guard covers the
    whole registry-declared big seat, not only the 122B by name — V4-Flash
    (gw_models.yaml mode_alias: big) must refuse identically."""
    from agents_core.council.cli import _refuse_wave_against_122b

    with patch("agents_core.llm._gw_default_model", return_value="gravitywell-v4flash"):
        with pytest.raises(ValueError, match="gravitywell-v4flash"):
            _refuse_wave_against_122b("gravitywell")


def test_refuse_wave_against_non_122b_model_allows():
    from agents_core.council.cli import _refuse_wave_against_122b

    with patch("agents_core.llm._gw_default_model", return_value="gravitywell-27b"):
        _refuse_wave_against_122b("gravitywell")  # must not raise


def test_refuse_wave_against_122b_skips_non_gravitywell_voicing():
    from agents_core.council.cli import _refuse_wave_against_122b

    with patch("agents_core.llm._gw_default_model", side_effect=AssertionError("must not be called")):
        _refuse_wave_against_122b("local")  # must not raise, must not even resolve the model


def test_wave_run_refuses_against_122b_end_to_end(tmp_path):
    """The refusal must actually fire from inside the real run path (D7),
    not just from the standalone helper."""
    from agents_core.council import cli as council_cli
    from agents_core.council.cli import GW_122B_MODEL
    from lapis_engine.engine import Engine

    run = {
        "run_id": "test-wave-122b",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": f"seat-{i}", "role": f"wave_slot_{i}"} for i in range(5)],
    }

    with patch("agents_core.llm._gw_default_model", return_value=GW_122B_MODEL):
        with pytest.raises(ValueError, match="gravitywell-122b"):
            council_cli._run_wave_deliberation(
                run=run, run_id="test-wave-122b", Engine=Engine,
                hold_active=False, doorman=None, hold_work_id="w", hold_principal="p",
                refresh_threads=[],
                CharacterEntity=object, NarratorEntity=object,
            )


# ---------------------------------------------------------------------------
# H6 / H6b — the lease chaos test, both windows
# ---------------------------------------------------------------------------


class _FakeDoorman:
    def __init__(self):
        self.acquisitions = []
        self.lock = threading.Lock()

    def acquire(self, node, work_id, ttl_sec, reason, timeout=None, principal=None, **kwargs):
        with self.lock:
            self.acquisitions.append({"reason": reason, "at": time.monotonic(), "ttl_sec": ttl_sec})
        return {"status": "serving"}

    def release(self, node, work_id):
        pass

    def close(self):
        pass


def test_lease_refresh_timer_floor_covers_slow_first_seat(monkeypatch):
    """H6b, window (a): a round that outlives COUNCIL_STALL_S before ANY
    seat has completed must still refresh, via the timer-based floor — only
    the timer can save the lease here, since completion-coupled refresh
    does nothing until the first seat returns."""
    from agents_core.council import cli as council_cli
    from lapis_engine.engine import Engine

    monkeypatch.setattr(council_cli, "COUNCIL_STALL_S", 0.2)

    # Every seat is slow — nothing completes until well after the floor's
    # first tick (COUNCIL_STALL_S / 2 = 0.1s).
    entities = [_StubEntity(f"seat-{i}", delay=0.5) for i in range(5)]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    doorman = _FakeDoorman()
    run = {
        "run_id": "test-wave-floor",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
    }

    with patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-floor", Engine=Engine,
            hold_active=True, doorman=doorman, hold_work_id="w", hold_principal="p",
            refresh_threads=[],
            CharacterEntity=object, NarratorEntity=object,
        )

    floor_refreshes = [a for a in doorman.acquisitions if "floor" in a["reason"]]
    assert floor_refreshes, "timer-based floor refresh never fired before the first seat completed"


def test_lease_refresh_fires_per_seat_completion_not_per_round(monkeypatch):
    """H6, window (b): completion-coupled refresh must fire once per seat as
    it actually completes, not once the whole round has drained."""
    from agents_core.council import cli as council_cli
    from lapis_engine.engine import Engine

    monkeypatch.setattr(council_cli, "COUNCIL_STALL_S", 60)  # floor won't fire in this short test

    entities = [_StubEntity(f"seat-{i}", delay=0.02 * i) for i in range(5)]

    def _fake_build_entity(sel, adapter, CE, NE):
        return next(e for e in entities if e.id == sel["id"])

    doorman = _FakeDoorman()
    refresh_threads = []
    run = {
        "run_id": "test-wave-seatcomplete",
        "decision": "x",
        "voicing": "gravitywell",
        "turns_cap": 1,
        "turns": [],
        "selected_entities": [{"id": e.id, "role": f"wave_slot_{i}"} for i, e in enumerate(entities)],
    }

    with patch("agents_core.council.cli._build_entity", _fake_build_entity), \
         patch("agents_core.council.cli._refuse_wave_against_122b"), \
         patch("agents_core.council.wave_digest.call_operator", return_value="digest"):
        council_cli._run_wave_deliberation(
            run=run, run_id="test-wave-seatcomplete", Engine=Engine,
            hold_active=True, doorman=doorman, hold_work_id="w", hold_principal="p",
            refresh_threads=refresh_threads,
            CharacterEntity=object, NarratorEntity=object,
        )

    for t in refresh_threads:
        t.join(timeout=2.0)

    seat_refreshes = [a for a in doorman.acquisitions if "seat-complete" in a["reason"]]
    # 5 wave-round seat completions + 1 synthesis-round completion (seat-0,
    # the synthesis speaker) — every act() completion gets its own refresh.
    assert len(seat_refreshes) == 6, (
        f"expected one seat-complete refresh per completed seat (5 wave + 1 synthesis), "
        f"got {len(seat_refreshes)}: {[a['reason'] for a in seat_refreshes]}"
    )


# ---------------------------------------------------------------------------
# Regression fence — deliberation/scene VALID_MODES membership unaffected
# ---------------------------------------------------------------------------


def test_deliberation_and_scene_still_valid_modes():
    from agents_core.council.cli import VALID_MODES

    assert "deliberation" in VALID_MODES
    assert "scene" in VALID_MODES


def test_validate_mode_n_deliberation_and_scene_unchanged():
    from agents_core.council.cli import _validate_mode_n

    _validate_mode_n("deliberation", 2)
    _validate_mode_n("deliberation", 3)
    with pytest.raises(ValueError):
        _validate_mode_n("deliberation", 4)
    _validate_mode_n("scene", 2)
    _validate_mode_n("scene", 3)
    with pytest.raises(ValueError):
        _validate_mode_n("scene", 5)


def test_role_assignments_deliberation_and_scene_unchanged():
    from agents_core.council.cli import _role_assignments

    delib = _role_assignments(["a", "b"], mode="deliberation")
    assert [r["role"] for r in delib] == ["first_voice", "second_voice"]

    scene = _role_assignments(["a", "b"], mode="scene")
    assert [r["role"] for r in scene] == ["scene_slot_0", "scene_slot_1"]
