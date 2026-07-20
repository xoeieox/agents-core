"""Tests for the seeded `reviewer` card pool wiring in agents_core.council.cli.

Covers §1 item 4 of knowledge-mesh-persona-seeder-v0: build_roster's
reviewer-pool fail-loud semantics (scoped to that pool only — the legacy
personal/historical/fiction pools keep their skip-empty behavior), and
_build_entity dispatching a reviewer-pool card to PersonaCardEntity.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from agents_core.cards import CardsRootError

SEED_CARD = (
    Path(__file__).resolve().parents[1]
    / "agents_core" / "council" / "seed_decks" / "decks" / "reviewer" / "technical-integrity.yaml"
)

SEED_PRIMITIVES = {
    "cognitive": ["systematic-analysis", "pattern-recognition", "concrete-pragmatism"],
    "behavioral": ["methodical-approach", "cautious-conservatism"],
    "relational": ["sustained-moral-demand"],
}


def _make_portable_cards_root(tmp_path: Path, with_reviewer_card: bool = True) -> Path:
    root = tmp_path / "cards"
    for family, prim_ids in SEED_PRIMITIVES.items():
        family_dir = root / "primitives" / family
        family_dir.mkdir(parents=True)
        for prim_id in prim_ids:
            (family_dir / f"{prim_id}.yaml").write_text("id: " + prim_id + "\n")
    reviewer_dir = root / "decks" / "reviewer"
    reviewer_dir.mkdir(parents=True)
    if with_reviewer_card:
        shutil.copy(SEED_CARD, reviewer_dir / SEED_CARD.name)
    return root


def _wire_cards_root(tmp_path, monkeypatch, with_reviewer_card=True):
    """Point ARCHETYPAL_CARDS_PATH + council.cli.CARDS_ROOT/DECKS_ROOT at a matching portable root."""
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=with_reviewer_card)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "CARDS_ROOT", root / "characters")
    monkeypatch.setattr(council_cli, "DECKS_ROOT", root / "decks")
    return root, council_cli


def test_build_roster_reviewer_pool_loads_seeded_card(tmp_path, monkeypatch):
    _root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    roster = council_cli.build_roster([council_cli.REVIEWER_POOL])
    assert len(roster) == 1
    assert roster[0]["character_id"] == "technical-integrity"
    assert roster[0]["pool"] == "reviewer"


def test_build_roster_reviewer_pool_empty_fails_loud(tmp_path, monkeypatch):
    _root, council_cli = _wire_cards_root(tmp_path, monkeypatch, with_reviewer_card=False)
    with pytest.raises(CardsRootError, match="deck pool is empty"):
        council_cli.build_roster([council_cli.REVIEWER_POOL])


def test_build_roster_legacy_pool_missing_dir_still_skips_silently(tmp_path, monkeypatch):
    """personal/historical/fiction keep build_roster's legacy skip-empty behavior —
    only the reviewer pool is fail-loud (§1 item 4)."""
    root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    assert not (root / "characters" / "personal").exists()
    roster = council_cli.build_roster(["personal"])
    assert roster == []


def test_build_roster_reviewer_and_legacy_pool_together(tmp_path, monkeypatch):
    root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    (root / "characters" / "historical").mkdir(parents=True)
    roster = council_cli.build_roster(["historical", council_cli.REVIEWER_POOL])
    assert len(roster) == 1
    assert roster[0]["pool"] == "reviewer"


def test_resolve_requested_pools_selects_non_default_pool_set(tmp_path, monkeypatch):
    root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    personal_dir = root / "characters" / "personal"
    personal_dir.mkdir(parents=True)
    (personal_dir / "ada-lovelace.yaml").write_text(
        "character_id: ada-lovelace\ncharacter_name: Ada Lovelace\ncultural_context: test\n"
    )
    pools = council_cli._resolve_requested_pools("reviewer, personal")
    assert pools == ["reviewer", "personal"]
    roster = council_cli.build_roster(pools=pools)
    assert {r["pool"] for r in roster} == {"reviewer", "personal"}


def test_resolve_requested_pools_unknown_pool_names_it(tmp_path, monkeypatch):
    _root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="bogus"):
        council_cli._resolve_requested_pools("bogus")


def test_resolve_requested_pools_empty_after_split_raises(tmp_path, monkeypatch):
    _root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="empty"):
        council_cli._resolve_requested_pools("")
    with pytest.raises(ValueError, match="empty"):
        council_cli._resolve_requested_pools("  , ,")


def test_find_card_path_resolves_reviewer_card(tmp_path, monkeypatch):
    root, council_cli = _wire_cards_root(tmp_path, monkeypatch)
    path = council_cli.find_card_path("technical-integrity")
    assert path == (root / "decks" / "reviewer" / "technical-integrity.yaml").resolve()


def test_build_entity_dispatches_reviewer_card_to_persona_card_entity(tmp_path, monkeypatch):
    pytest.importorskip("lapis_engine")
    from agents_core.council.persona_card_entity import PersonaCardEntity

    _root, council_cli = _wire_cards_root(tmp_path, monkeypatch)

    class _FakeCharacterEntity:
        @classmethod
        def load(cls, path, adapter):
            raise AssertionError("should not load CharacterEntity for a reviewer-pool card")

    class _StubLLM:
        def chat(self, system, messages):
            return "voiced"

    entity = council_cli._build_entity(
        {"id": "technical-integrity", "role": "first_voice"},
        _StubLLM(),
        _FakeCharacterEntity,
        None,
    )
    assert isinstance(entity, PersonaCardEntity)
    assert entity.id == "technical-integrity"


def test_persona_card_entity_load_and_act():
    pytest.importorskip("lapis_engine")
    from agents_core.council.persona_card_entity import PersonaCardEntity

    class _StubLLM:
        def __init__(self):
            self.last_system = None
        def chat(self, system, messages):
            self.last_system = system
            return "voiced response"

    llm = _StubLLM()
    entity = PersonaCardEntity.load(SEED_CARD, llm)
    assert entity.id == "technical-integrity"
    assert entity.kernel_invariants == ["Truth integrity", "Provenance", "Possibility, not prescription"]

    reply = entity.act("Review this diff.", None)
    assert reply == "voiced response"
    assert "technical-integrity" in llm.last_system
    assert "Every interface decision propagates through the coupling chain." in llm.last_system


def test_persona_card_entity_load_raises_on_missing_file(tmp_path):
    pytest.importorskip("lapis_engine")
    from agents_core.council.persona_card_entity import PersonaCardEntity

    with pytest.raises(FileNotFoundError):
        PersonaCardEntity.load(tmp_path / "nope.yaml", llm=None)


def test_seed_card_matches_documented_spec_fields():
    """The shipped seed asset (agents_core/council/seed_decks/decks/reviewer/
    technical-integrity.yaml) matches the worked card in the spec verbatim
    on its structural fields."""
    data = yaml.safe_load(SEED_CARD.read_text())
    assert data["slug"] == "technical-integrity"
    assert set(data["composition"]["primitives"]) == {
        "systematic-analysis", "methodical-approach", "cautious-conservatism",
        "pattern-recognition", "sustained-moral-demand", "concrete-pragmatism",
    }
    assert abs(sum(data["composition"]["primitives"].values()) - 1.0) < 1e-9
    assert len(data["voice_exemplars"]) >= 3
    assert data["domains"]
    assert data["kernel_invariants"] == [
        "Truth integrity", "Provenance", "Possibility, not prescription",
    ]
