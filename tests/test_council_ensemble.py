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
