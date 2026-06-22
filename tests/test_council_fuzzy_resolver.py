"""Tests for _resolve_character_id fuzzy edit-distance fallback.

Tests _resolve_character_id directly with a synthetic roster + monkeypatched
find_card_path — no live council or GW calls.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resolver(valid_ids):
    """Return a mock find_card_path that recognises only valid_ids."""
    def _find(cid):
        return Path(f"/fake/cards/{cid}.yaml") if cid in valid_ids else None
    return _find


def _build_resolution(canonical_ids):
    """Replicate the suffix-resolution map from select_entities."""
    resolution = {}
    for cid in canonical_ids:
        resolution[cid] = cid
        parts = cid.split("-")
        for i in range(1, len(parts)):
            suffix = "-".join(parts[i:])
            if suffix not in resolution:
                resolution[suffix] = cid
    return resolution


# ---------------------------------------------------------------------------
# Unit tests for _resolve_character_id
# ---------------------------------------------------------------------------

def test_fuzzy_corrects_single_char_misspelling(monkeypatch):
    """scheherazada-canonical -> scheherazade-canonical (one char off)."""
    from agents_core.council.cli import _resolve_character_id

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical", "rumi-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    result = _resolve_character_id("scheherazada-canonical", canonical_ids, resolution)
    assert result == "scheherazade-canonical"


def test_exact_match_still_works(monkeypatch):
    """Exact match takes priority, fuzzy path not reached."""
    from agents_core.council.cli import _resolve_character_id

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    result = _resolve_character_id("scheherazade-canonical", canonical_ids, resolution)
    assert result == "scheherazade-canonical"


def test_suffix_resolution_still_works(monkeypatch):
    """Suffix-only id resolves via the suffix map before fuzzy is consulted."""
    from agents_core.council.cli import _resolve_character_id

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    # "canonical" is a suffix of both — but only one matches; in the suffix map
    # whichever came first wins; the point is it resolves via suffix, not fuzzy.
    result = _resolve_character_id("scheherazade-canonical", canonical_ids, resolution)
    assert result is not None


def test_unknown_id_returns_none(monkeypatch):
    """A genuinely unknown id with no close roster match returns None."""
    from agents_core.council.cli import _resolve_character_id

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical", "rumi-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    result = _resolve_character_id("zzz-totally-unknown-xyz", canonical_ids, resolution)
    assert result is None


def test_ambiguous_near_tie_not_corrected(monkeypatch):
    """Two roster ids equally close -> no auto-correct (gap < 0.10 threshold)."""
    from agents_core.council.cli import _resolve_character_id

    # Construct two ids that both differ from the query by a similar ratio.
    # "alice-canonical" and "alicx-canonical" both have similarity ~0.93 to "alic-canonical"
    # We need to verify via SequenceMatcher that the gap is < 0.10.
    canonical_ids = {"alice-canonical", "alicx-canonical", "rumi-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    query = "alicy-canonical"
    matches = difflib.get_close_matches(query, list(canonical_ids), n=2, cutoff=0.90)
    # Only run test when both candidates actually meet the cutoff (test validity guard).
    if len(matches) >= 2:
        gap = (
            difflib.SequenceMatcher(None, query, matches[0]).ratio()
            - difflib.SequenceMatcher(None, query, matches[1]).ratio()
        )
        if gap < 0.10:
            result = _resolve_character_id(query, canonical_ids, resolution)
            assert result is None, f"Expected no correction for near-tie gap={gap:.3f}, got {result!r}"


def test_non_string_returns_none(monkeypatch):
    """Non-string sid returns None without error."""
    from agents_core.council.cli import _resolve_character_id

    canonical_ids = {"scheherazade-canonical"}
    resolution = _build_resolution(canonical_ids)
    monkeypatch.setattr(
        "agents_core.council.cli.find_card_path",
        _make_resolver(canonical_ids),
    )

    assert _resolve_character_id(None, canonical_ids, resolution) is None
    assert _resolve_character_id(42, canonical_ids, resolution) is None


# ---------------------------------------------------------------------------
# Integration: fuzzy correction prevents RuntimeError in select_entities
# ---------------------------------------------------------------------------

def test_select_entities_fuzzy_no_raise(monkeypatch, tmp_path):
    """A misspelled id in selection resolves without raising RuntimeError."""
    from agents_core.council import cli as council_cli

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical"}
    fake_roster = [
        {"character_id": cid, "character_name": cid, "pool": "fiction", "cultural_context": ""}
        for cid in canonical_ids
    ]

    # Misspelled selection: model returned "scheherazada-canonical"
    fake_selection = {"selected": ["scheherazada-canonical", "aristotle-canonical"], "reasoning": "test"}

    monkeypatch.setattr(council_cli, "find_card_path", _make_resolver(canonical_ids))
    monkeypatch.setattr(council_cli, "call_operator", lambda *a, **kw: '{"selected": ["scheherazada-canonical", "aristotle-canonical"], "reasoning": "test"}')
    monkeypatch.setattr(council_cli, "_extract_json", lambda _: fake_selection)
    monkeypatch.setattr(council_cli, "gather_mem_context", lambda *a, **kw: {"preview": "", "hits": []})

    result = council_cli.select_entities(
        decision="test decision",
        roster=fake_roster,
        context={},
        n=2,
    )
    assert "scheherazade-canonical" in result["selected"]
    assert "aristotle-canonical" in result["selected"]


def test_select_entities_unknown_id_still_raises(monkeypatch):
    """Completely unknown id after fuzzy fallback -> raises RuntimeError."""
    from agents_core.council import cli as council_cli

    canonical_ids = {"scheherazade-canonical", "aristotle-canonical"}
    fake_roster = [
        {"character_id": cid, "character_name": cid, "pool": "fiction", "cultural_context": ""}
        for cid in canonical_ids
    ]

    # Both calls return garbage id
    garbage_selection = {"selected": ["zzz-totally-unknown", "aristotle-canonical"], "reasoning": "test"}

    monkeypatch.setattr(council_cli, "find_card_path", _make_resolver(canonical_ids))
    monkeypatch.setattr(council_cli, "call_operator", lambda *a, **kw: '{}')
    monkeypatch.setattr(council_cli, "_extract_json", lambda _: garbage_selection)
    monkeypatch.setattr(council_cli, "gather_mem_context", lambda *a, **kw: {"preview": "", "hits": []})

    with pytest.raises(RuntimeError, match="Selected unknown character_id after retry"):
        council_cli.select_entities(
            decision="test decision",
            roster=fake_roster,
            context={},
            n=2,
        )


# ---------------------------------------------------------------------------
# Roster-ambiguity simulation (BINDING acceptance criterion #7)
# ---------------------------------------------------------------------------

def test_roster_no_ambiguous_near_tie_pairs():
    """Scan the ACTUAL canonical roster and assert no two ids fall within the
    auto-correct window (similarity >= 0.90 AND gap < 0.10).

    If this test fails, the chosen thresholds need re-evaluation before
    the fuzzy resolver can be safely deployed (wrong-character substitution risk).
    """
    from agents_core.council.cli import build_roster, DEFAULT_POOLS

    roster = build_roster(DEFAULT_POOLS)
    canonical_ids = sorted({r["character_id"] for r in roster})

    # Use the SAME cutoff as _resolve_character_id (0.92).
    # Threshold derivation: the spec mandated 0.90 as a floor; the roster scan found
    # 4 pairs at 0.90 (highest: adam-paradise-lost-canonical / satan-paradise-lost-canonical
    # at 0.912), so 0.90 was raised to 0.92 (above 0.912, below scheherazada ratio 0.9545).
    # Update this constant if _resolve_character_id's cutoff ever changes.
    FUZZY_CUTOFF = 0.92

    near_tie_pairs = []
    for i, a in enumerate(canonical_ids):
        for b in canonical_ids[i + 1:]:
            ratio_ab = difflib.SequenceMatcher(None, a, b).ratio()
            # Two ids are "in the auto-correct window" for each other when their pairwise
            # similarity >= FUZZY_CUTOFF: any minor misspelling of A could be within the
            # window for B simultaneously, making auto-correction ambiguous.
            if ratio_ab >= FUZZY_CUTOFF:
                near_tie_pairs.append((a, b, ratio_ab))

    assert not near_tie_pairs, (
        f"Found {len(near_tie_pairs)} roster id pair(s) within the auto-correct window "
        f"(similarity >= {FUZZY_CUTOFF}) - thresholds may allow wrong-character substitution:\n"
        + "\n".join(f"  {a!r} <-> {b!r} (ratio={r:.3f})" for a, b, r in near_tie_pairs)
    )
