"""Tests for agents_core.cards — the node-portable archetypal cards root resolver.

DoD-1 (regression): with ARCHETYPAL_CARDS_PATH unset, cards_root() resolves to
the retired hardcode unchanged (see also tests/test_council_module.py /
tests/test_council_cli.py, which stay green with the env var unset).

DoD-2 (portability): with ARCHETYPAL_CARDS_PATH pointed at a node-owned copy,
the seeded reviewer deck loads from that location. A missing/empty configured
deck fails loudly (negative test).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from agents_core.cards import (
    CANONICAL_KERNEL_INVARIANTS,
    CardsRootError,
    DEFAULT_CARDS_ROOT,
    cards_root,
    load_deck_cards,
    resolve_under_cards_root,
    validate_deck_card,
)

SEED_CARD = (
    Path(__file__).resolve().parents[1]
    / "agents_core" / "council" / "seed_decks" / "reviewer" / "technical-integrity.yaml"
)

# The six primitive ids the seed card references (agents_core/council/seed_decks/
# reviewer/technical-integrity.yaml), grouped by their primitive family dir.
SEED_PRIMITIVES = {
    "cognitive": ["systematic-analysis", "pattern-recognition", "concrete-pragmatism"],
    "behavioral": ["methodical-approach", "cautious-conservatism"],
    "relational": ["sustained-moral-demand"],
}


def _make_portable_cards_root(tmp_path: Path, with_reviewer_card: bool = True) -> Path:
    """Build a self-contained cards root: primitives/ stubs + characters/reviewer/."""
    root = tmp_path / "cards"
    for family, prim_ids in SEED_PRIMITIVES.items():
        family_dir = root / "primitives" / family
        family_dir.mkdir(parents=True)
        for prim_id in prim_ids:
            (family_dir / f"{prim_id}.yaml").write_text("id: " + prim_id + "\n")
    reviewer_dir = root / "characters" / "reviewer"
    reviewer_dir.mkdir(parents=True)
    if with_reviewer_card:
        shutil.copy(SEED_CARD, reviewer_dir / SEED_CARD.name)
    return root


# ---------------------------------------------------------------------------
# cards_root()
# ---------------------------------------------------------------------------


def test_cards_root_default_unchanged_when_env_unset(monkeypatch):
    monkeypatch.delenv("ARCHETYPAL_CARDS_PATH", raising=False)
    if not DEFAULT_CARDS_ROOT.is_dir():
        pytest.skip("archetypal-intelligence cards dir not present on this node")
    assert cards_root() == DEFAULT_CARDS_ROOT.resolve()


def test_cards_root_env_override_resolves_to_node_owned_copy(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    assert cards_root() == root.resolve()


def test_cards_root_raises_on_missing_directory(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(missing))
    with pytest.raises(CardsRootError):
        cards_root()


def test_cards_root_raises_when_path_is_a_file(tmp_path, monkeypatch):
    f = tmp_path / "not-a-dir"
    f.write_text("x")
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(f))
    with pytest.raises(CardsRootError):
        cards_root()


# ---------------------------------------------------------------------------
# resolve_under_cards_root() — symlink-escape containment
# ---------------------------------------------------------------------------


def test_resolve_under_cards_root_accepts_contained_path(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    card = root / "characters" / "reviewer" / "technical-integrity.yaml"
    assert resolve_under_cards_root(card) == card.resolve()


def test_resolve_under_cards_root_rejects_symlink_escape(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text("slug: escaped\n")
    escape_link = root / "characters" / "reviewer" / "escaped.yaml"
    escape_link.symlink_to(outside)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    with pytest.raises(CardsRootError, match="escapes cards root"):
        resolve_under_cards_root(escape_link)


# ---------------------------------------------------------------------------
# validate_deck_card()
# ---------------------------------------------------------------------------


def test_validate_deck_card_never_raises_on_parse_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("{not: valid: yaml: [")
    errors = validate_deck_card(bad)
    assert errors and "parse" in errors[0].lower()


def test_validate_deck_card_accepts_seed_card_against_portable_root(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    card = root / "characters" / "reviewer" / "technical-integrity.yaml"
    assert validate_deck_card(card) == []


def test_validate_deck_card_missing_slug(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    del data["slug"]
    card = root / "characters" / "reviewer" / "no-slug.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("slug" in e for e in errors)


def test_validate_deck_card_weight_sum_not_one(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    data["composition"]["primitives"]["systematic-analysis"] = 0.99
    card = root / "characters" / "reviewer" / "bad-weight.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("weight" in e for e in errors)


def test_validate_deck_card_unresolved_primitive(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    primitives = data["composition"]["primitives"]
    del primitives["systematic-analysis"]
    primitives["not-a-real-primitive"] = 0.25
    card = root / "characters" / "reviewer" / "bad-primitive.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("not-a-real-primitive" in e for e in errors)


def test_validate_deck_card_too_few_voice_exemplars(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    data["voice_exemplars"] = data["voice_exemplars"][:2]
    card = root / "characters" / "reviewer" / "few-exemplars.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("voice_exemplars" in e for e in errors)


def test_validate_deck_card_empty_domains(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    data["domains"] = []
    card = root / "characters" / "reviewer" / "no-domains.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("domains" in e for e in errors)


def test_validate_deck_card_invalid_kernel_invariant(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    data["kernel_invariants"] = ["Not a real invariant"]
    card = root / "characters" / "reviewer" / "bad-invariant.yaml"
    card.write_text(yaml.safe_dump(data))
    errors = validate_deck_card(card)
    assert any("kernel_invariant" in e for e in errors)


def test_canonical_kernel_invariants_has_eight_entries():
    assert len(CANONICAL_KERNEL_INVARIANTS) == 8


# ---------------------------------------------------------------------------
# load_deck_cards() — fail-closed deck loader
# ---------------------------------------------------------------------------


def test_load_deck_cards_portability_proof(tmp_path, monkeypatch):
    """DoD-2: with ARCHETYPAL_CARDS_PATH pointed at a node-owned copy, the
    seeded reviewer deck loads from that location — the deck travels."""
    root = _make_portable_cards_root(tmp_path)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    cards = load_deck_cards(root / "characters" / "reviewer")
    assert len(cards) == 1
    assert cards[0]["data"]["slug"] == "technical-integrity"


def test_load_deck_cards_raises_on_missing_directory(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    with pytest.raises(CardsRootError, match="deck pool is empty"):
        load_deck_cards(root / "characters" / "does-not-exist")


def test_load_deck_cards_raises_on_empty_directory(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    with pytest.raises(CardsRootError, match="deck pool is empty"):
        load_deck_cards(root / "characters" / "reviewer")


def test_load_deck_cards_raises_on_invalid_card_naming_the_error(tmp_path, monkeypatch):
    root = _make_portable_cards_root(tmp_path, with_reviewer_card=False)
    monkeypatch.setenv("ARCHETYPAL_CARDS_PATH", str(root))
    data = yaml.safe_load(SEED_CARD.read_text())
    primitives = data["composition"]["primitives"]
    del primitives["systematic-analysis"]
    primitives["not-a-real-primitive"] = 0.25
    (root / "characters" / "reviewer" / "bad-primitive.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(CardsRootError, match="not-a-real-primitive"):
        load_deck_cards(root / "characters" / "reviewer")
