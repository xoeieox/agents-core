"""Tests for agents_core.council.ensemble — the in-process persona-ensemble
fast-path coordinator (council-inprocess-ensemble-v0).

Covers:
  - run_ensemble against the real seeded reviewer deck (1 card), voicing
    mocked at the GravityWellAdapter's call_operator seam.
  - the N>1 assembly path with 2 fixture cards under a temp
    ARCHETYPAL_CARDS_PATH (decks/reviewer + matching primitives/ stubs, so
    validate_deck_card resolves offline).
  - open_slot_for's best-effort BRIX-only slot-open seam: exactly one slot
    created on a simulated-master test path, and no-op-that-never-raises on
    off-master / generic slot-store failure.

Module-level importorskip: agents_core.council.ensemble imports
PersonaCardEntity at module scope (same as persona_card_entity.py itself),
which is a hard import of lapis_engine — mirrors the skip convention already
used for those tests in tests/test_council_reviewer_pool.py.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytest.importorskip("lapis_engine")

from agents_core.council import ensemble  # noqa: E402

SEED_CARD = (
    Path(__file__).resolve().parents[1]
    / "agents_core" / "council" / "seed_decks" / "decks" / "reviewer" / "technical-integrity.yaml"
)
SEED_PRIMITIVES = {
    "cognitive": ["systematic-analysis", "pattern-recognition", "concrete-pragmatism"],
    "behavioral": ["methodical-approach", "cautious-conservatism"],
    "relational": ["sustained-moral-demand"],
}

FIXTURE_CARDS = [
    {
        "slug": "fixture-reviewer-alpha",
        "composition": {"primitives": {"alpha-primitive": 1.0}},
        "voice_exemplars": [
            "This is the first alpha exemplar line.",
            "This is the second alpha exemplar line.",
            "This is the third alpha exemplar line.",
        ],
        "domains": ["alpha-domain"],
        "kernel_invariants": ["Truth integrity"],
    },
    {
        "slug": "fixture-reviewer-beta",
        "composition": {"primitives": {"beta-primitive-one": 0.6, "beta-primitive-two": 0.4}},
        "voice_exemplars": [
            "This is the first beta exemplar line.",
            "This is the second beta exemplar line.",
            "This is the third beta exemplar line.",
        ],
        "domains": ["beta-domain"],
        "kernel_invariants": ["Provenance"],
    },
]


def _wire_seeded_reviewer_deck(tmp_path: Path, monkeypatch) -> Path:
    """Portable cards root with the ONE real seeded reviewer card + its primitives."""
    root = tmp_path / "cards"
    for family, prim_ids in SEED_PRIMITIVES.items():
        family_dir = root / "primitives" / family
        family_dir.mkdir(parents=True)
        for prim_id in prim_ids:
            (family_dir / f"{prim_id}.yaml").write_text("id: " + prim_id + "\n")
    reviewer_dir = root / "decks" / "reviewer"
    reviewer_dir.mkdir(parents=True)
    shutil.copy(SEED_CARD, reviewer_dir / SEED_CARD.name)

    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    monkeypatch.setattr(ensemble, "DECKS_ROOT", root / "decks")
    return root


def _fake_operator(responses: dict, *, effective_operator: str = "gravitywell"):
    """Build a call_operator side_effect that also fills `_provenance_out`
    (kwargs["_provenance_out"]) the way the real call_operator does, so
    GravityWellAdapter.chat()'s voicing_events bookkeeping fires for real."""
    def _side_effect(operator_class, prompt, **kwargs):
        provenance_out = kwargs.get("_provenance_out")
        if provenance_out is not None:
            provenance_out.append(("success", effective_operator))
        system = kwargs.get("system", "")
        for slug, text in responses.items():
            if slug in system:
                return text
        raise AssertionError(f"unrecognized system prompt: {system!r}")
    return _side_effect


def _wire_fixture_reviewer_deck(tmp_path: Path, monkeypatch) -> Path:
    """Portable cards root with 2 minimal fixture reviewer cards + matching
    primitive stubs, per the DoD-2 offline-validation contract."""
    root = tmp_path / "cards"
    primitives_dir = root / "primitives"
    primitives_dir.mkdir(parents=True)
    reviewer_dir = root / "decks" / "reviewer"
    reviewer_dir.mkdir(parents=True)

    for card in FIXTURE_CARDS:
        (reviewer_dir / f"{card['slug']}.yaml").write_text(yaml.safe_dump(card))
        for prim_id in card["composition"]["primitives"]:
            (primitives_dir / f"{prim_id}.yaml").write_text(f"id: {prim_id}\n")

    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    monkeypatch.setattr(ensemble, "DECKS_ROOT", root / "decks")
    return root


def _wire_n_fixture_reviewer_cards(tmp_path: Path, monkeypatch, n: int) -> list[str]:
    """Portable cards root with `n` minimal fixture reviewer cards
    (fixture-reviewer-a, -b, -c, ... — alphabetical == deck order, since
    load_deck_cards sorts by filename). Returns the slugs in deck order."""
    root = tmp_path / "cards"
    primitives_dir = root / "primitives"
    primitives_dir.mkdir(parents=True)
    reviewer_dir = root / "decks" / "reviewer"
    reviewer_dir.mkdir(parents=True)

    slugs = []
    for i in range(n):
        letter = chr(ord("a") + i)
        slug = f"fixture-reviewer-{letter}"
        slugs.append(slug)
        prim_id = f"{letter}-primitive"
        card = {
            "slug": slug,
            "composition": {"primitives": {prim_id: 1.0}},
            "voice_exemplars": [
                f"{letter} exemplar one.",
                f"{letter} exemplar two.",
                f"{letter} exemplar three.",
            ],
            "domains": [f"{letter}-domain"],
            "kernel_invariants": ["Truth integrity"],
        }
        (reviewer_dir / f"{slug}.yaml").write_text(yaml.safe_dump(card))
        (primitives_dir / f"{prim_id}.yaml").write_text(f"id: {prim_id}\n")

    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    monkeypatch.setattr(ensemble, "DECKS_ROOT", root / "decks")
    return slugs


# ---------------------------------------------------------------------------
# run_ensemble — real seeded deck (1 card)
# ---------------------------------------------------------------------------

def test_run_ensemble_loads_seeded_reviewer_deck(tmp_path, monkeypatch):
    _wire_seeded_reviewer_deck(tmp_path, monkeypatch)

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=_fake_operator({"technical-integrity": "voiced by technical-integrity"}),
    ) as mock_op:
        result = ensemble.run_ensemble("Review this diff.")

    assert isinstance(result, ensemble.EnsembleResult)
    assert result.deck == "reviewer"
    assert len(result.roster) == 1
    entry = result.roster[0]
    assert entry.slug == "technical-integrity"
    assert entry.output == "voiced by technical-integrity"
    assert len(entry.voicing_provenance) == 1
    assert entry.voicing_provenance[0]["effective_operator"] is not None
    mock_op.assert_called_once()

    assert result.synthesized is False
    assert result.roster_semantics == "raw-divergent-voices-no-synthesis"
    assert result.prompt_hash.startswith("sha256:")
    assert result.run_id


def test_run_ensemble_empty_pool_fails_loud(tmp_path, monkeypatch):
    from agents_core.cards import CardsRootError

    root = tmp_path / "cards"
    (root / "decks" / "reviewer").mkdir(parents=True)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    monkeypatch.setattr(ensemble, "DECKS_ROOT", root / "decks")

    with pytest.raises(CardsRootError, match="deck pool is empty"):
        ensemble.run_ensemble("Review this diff.")


# ---------------------------------------------------------------------------
# run_ensemble — N>1 assembly path with fixture cards
# ---------------------------------------------------------------------------

def test_run_ensemble_two_fixture_cards_each_voiced_once(tmp_path, monkeypatch):
    _wire_fixture_reviewer_deck(tmp_path, monkeypatch)

    responses = {
        "fixture-reviewer-alpha": "alpha take",
        "fixture-reviewer-beta": "beta take",
    }

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=_fake_operator(responses),
    ) as mock_op:
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert len(result.roster) == 2
    assert mock_op.call_count == 2

    by_slug = {entry.slug: entry for entry in result.roster}
    assert set(by_slug) == {"fixture-reviewer-alpha", "fixture-reviewer-beta"}
    assert by_slug["fixture-reviewer-alpha"].output == "alpha take"
    assert by_slug["fixture-reviewer-beta"].output == "beta take"
    # Each voice's provenance is its own single call, not the other's.
    assert len(by_slug["fixture-reviewer-alpha"].voicing_provenance) == 1
    assert len(by_slug["fixture-reviewer-beta"].voicing_provenance) == 1

    assert result.synthesized is False


def test_run_ensemble_shares_one_principal_across_voices(tmp_path, monkeypatch):
    _wire_fixture_reviewer_deck(tmp_path, monkeypatch)

    captured_principals = []

    def fake_call_operator(operator_class, prompt, **kwargs):
        captured_principals.append(kwargs.get("principal"))
        return "voiced"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        ensemble.run_ensemble("Review this diff.", principal="shared-principal-x")

    assert len(captured_principals) == 2
    assert captured_principals[0] == captured_principals[1] == "shared-principal-x"


# ---------------------------------------------------------------------------
# run_ensemble — concurrent voicing (council-concurrent-voicing-and-governor-caps-v0)
# ---------------------------------------------------------------------------

def test_run_ensemble_provenance_isolated_under_concurrent_interleaving(tmp_path, monkeypatch):
    """Deliberately interleave completion order (later-deck-index cards finish
    first) and assert every VoiceEntry carries exactly its own event — no
    cross-voice bleed from concurrent-append on a shared adapter."""
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 4)
    monkeypatch.setenv("COUNCIL_VOICING_MAX_CONCURRENT", "4")

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        # Reverse-order sleep: the LAST deck card finishes FIRST, forcing
        # real out-of-order completion under the thread pool.
        idx = slugs.index(matched)
        time.sleep(0.03 * (len(slugs) - idx))
        provenance_out = kwargs.get("_provenance_out")
        if provenance_out is not None:
            provenance_out.append(("success", matched))
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert len(result.roster) == 4
    for entry in result.roster:
        assert len(entry.voicing_provenance) == 1
        assert entry.voicing_provenance[0]["effective_operator"] == entry.slug
        assert entry.output == f"voiced by {entry.slug}"


def test_run_ensemble_roster_order_is_deck_order_despite_completion_order(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 4)
    monkeypatch.setenv("COUNCIL_VOICING_MAX_CONCURRENT", "4")

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        idx = slugs.index(matched)
        time.sleep(0.03 * (len(slugs) - idx))  # first card finishes LAST
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert [entry.slug for entry in result.roster] == slugs


def test_run_ensemble_voicing_cap_1_matches_sequential_behavior(tmp_path, monkeypatch):
    """COUNCIL_VOICING_MAX_CONCURRENT=1 must reproduce today's sequential
    behavior exactly — one call in flight at a time, in deck order."""
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 3)
    monkeypatch.setenv("COUNCIL_VOICING_MAX_CONCURRENT", "1")

    call_order = []
    in_flight = []
    max_in_flight = [0]

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        in_flight.append(matched)
        max_in_flight[0] = max(max_in_flight[0], len(in_flight))
        call_order.append(matched)
        time.sleep(0.01)
        in_flight.remove(matched)
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert max_in_flight[0] == 1  # never more than one call in flight
    assert call_order == slugs   # and it happened in deck order
    assert [entry.slug for entry in result.roster] == slugs


def test_council_voicing_cap_invalid_env_falls_back_to_1_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("COUNCIL_VOICING_MAX_CONCURRENT", "not-a-number")
    with caplog.at_level("WARNING", logger="agents_core.council.ensemble"):
        cap = ensemble._council_voicing_cap()
    assert cap == 1
    assert any(
        record.levelname == "WARNING" and "COUNCIL_VOICING_MAX_CONCURRENT" in record.message
        for record in caplog.records
    )


def test_council_voicing_cap_zero_falls_back_to_1_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("COUNCIL_VOICING_MAX_CONCURRENT", "0")
    with caplog.at_level("WARNING", logger="agents_core.council.ensemble"):
        cap = ensemble._council_voicing_cap()
    assert cap == 1
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_council_voicing_cap_default_is_4(monkeypatch):
    monkeypatch.delenv("COUNCIL_VOICING_MAX_CONCURRENT", raising=False)
    assert ensemble._council_voicing_cap() == 4


def test_run_ensemble_partial_failure_yields_error_entry_others_succeed(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 3)
    failing_slug = slugs[1]

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        if matched == failing_slug:
            raise RuntimeError("simulated voice failure")
        provenance_out = kwargs.get("_provenance_out")
        if provenance_out is not None:
            provenance_out.append(("success", matched))
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert [entry.slug for entry in result.roster] == slugs  # position preserved
    failed = result.roster[1]
    assert failed.error is not None
    assert failed.output == ""
    assert failed.voicing_provenance == []
    for i in (0, 2):
        assert result.roster[i].error is None
        assert result.roster[i].output == f"voiced by {slugs[i]}"


def test_run_ensemble_all_voices_fail_raises(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 2)

    def fake_call_operator(operator_class, prompt, **kwargs):
        raise RuntimeError("simulated total outage")

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        with pytest.raises(RuntimeError):
            ensemble.run_ensemble("Review this diff.", deck="reviewer")


# ---------------------------------------------------------------------------
# run_ensemble — roster rotation seam
# ---------------------------------------------------------------------------

def test_run_ensemble_rotation_same_key_same_subset(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 6)

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result_a = ensemble.run_ensemble(
            "Review this diff.", deck="reviewer", voices=3, rotation_key="spec-a"
        )
        result_b = ensemble.run_ensemble(
            "Review this diff.", deck="reviewer", voices=3, rotation_key="spec-a"
        )

    assert len(result_a.roster) == 3
    assert [e.slug for e in result_a.roster] == [e.slug for e in result_b.roster]


def test_run_ensemble_rotation_different_key_different_subset(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 6)

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result_a = ensemble.run_ensemble(
            "Review this diff.", deck="reviewer", voices=3, rotation_key="spec-a"
        )
        result_b = ensemble.run_ensemble(
            "Review this diff.", deck="reviewer", voices=3, rotation_key="spec-b"
        )

    slugs_a = {e.slug for e in result_a.roster}
    slugs_b = {e.slug for e in result_b.roster}
    assert slugs_a != slugs_b


def test_run_ensemble_rotation_selection_preserves_deck_order(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 6)

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble(
            "Review this diff.", deck="reviewer", voices=3, rotation_key="spec-order-check"
        )

    result_slugs = [e.slug for e in result.roster]
    # Selected subset must appear in deck (alphabetical) order, not rotation-key order.
    assert result_slugs == sorted(result_slugs)


def test_run_ensemble_voices_none_is_full_deck_unchanged(tmp_path, monkeypatch):
    slugs = _wire_n_fixture_reviewer_cards(tmp_path, monkeypatch, 3)

    def fake_call_operator(operator_class, prompt, **kwargs):
        system = kwargs.get("system", "")
        matched = next(s for s in slugs if s in system)
        return f"voiced by {matched}"

    with patch(
        "agents_core.council.gravitywell_adapter.call_operator",
        side_effect=fake_call_operator,
    ):
        result = ensemble.run_ensemble("Review this diff.", deck="reviewer")

    assert [e.slug for e in result.roster] == slugs


# ---------------------------------------------------------------------------
# open_slot_for — governance boundary / slot-crossing seam
# ---------------------------------------------------------------------------

def _fake_result(n_voices=1) -> "ensemble.EnsembleResult":
    roster = [
        ensemble.VoiceEntry(slug=f"voice-{i}", output=f"output-{i}", voicing_provenance=[])
        for i in range(n_voices)
    ]
    return ensemble.EnsembleResult(
        roster=roster,
        deck="reviewer",
        prompt_hash="sha256:deadbeef",
        run_id="ensemble-test-run",
    )


def test_open_slot_for_opens_exactly_one_slot_on_master(tmp_path, monkeypatch):
    from agents_core.slots import SlotStore as RealSlotStore

    db_path = tmp_path / "slots.db"

    def factory(*args, **kwargs):
        return RealSlotStore(db_path=db_path)

    monkeypatch.setattr("agents_core.slots.SlotStore", factory)
    monkeypatch.setattr("agents_core.slots.IS_MASTER", True)

    result = _fake_result(n_voices=2)
    slot_id = ensemble.open_slot_for(result, project_id="proj-ensemble-test")

    assert slot_id is not None

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT slot_id, project_id, contributor_id FROM slots").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == slot_id
    assert rows[0][1] == "proj-ensemble-test"
    assert rows[0][2] == result.run_id


def test_open_slot_for_never_edits_repo_or_dispatches(tmp_path, monkeypatch):
    """The slot-open seam is coordination metadata only — no filesystem repo
    mutation, no subprocess/dispatch side effect."""
    from agents_core.slots import SlotStore as RealSlotStore

    db_path = tmp_path / "slots.db"
    monkeypatch.setattr("agents_core.slots.SlotStore", lambda *a, **kw: RealSlotStore(db_path=db_path))
    monkeypatch.setattr("agents_core.slots.IS_MASTER", True)

    repo_marker = tmp_path / "repo-untouched.txt"
    repo_marker.write_text("original")

    with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
        result = ensemble.open_slot_for(_fake_result(), project_id="proj-no-side-effects")

    assert result is not None
    assert repo_marker.read_text() == "original"
    mock_run.assert_not_called()
    mock_popen.assert_not_called()


def test_open_slot_for_off_master_returns_none_never_raises(monkeypatch, capsys):
    from agents_core.slots import OffMasterWriteError

    class _OffMasterStore:
        def __init__(self, *args, **kwargs):
            pass

        def create_slot(self, **kwargs):
            raise OffMasterWriteError("off-master write refused (test)")

    monkeypatch.setattr("agents_core.slots.SlotStore", _OffMasterStore)

    slot_id = ensemble.open_slot_for(_fake_result(), project_id="proj-off-master")

    assert slot_id is None
    captured = capsys.readouterr()
    assert "off-master slot-open not supported in v0" in captured.err


def test_open_slot_for_generic_failure_returns_none_never_raises(monkeypatch, capsys):
    class _RaisingStore:
        def __init__(self, *args, **kwargs):
            pass

        def create_slot(self, **kwargs):
            raise RuntimeError("slot store unavailable (test)")

    monkeypatch.setattr("agents_core.slots.SlotStore", _RaisingStore)

    slot_id = ensemble.open_slot_for(_fake_result(), project_id="proj-generic-failure")

    assert slot_id is None
    captured = capsys.readouterr()
    assert "slot-open-failed" in captured.err
