"""agents_core.cards — node-portable archetypal card root resolver.

The archetypal card decks (council characters, primitives) were pinned to a
BRIX-absolute filesystem path. This module retires that hardcoding through a
single resolver so a node's Persona layer travels with it, paired with its
owned Keeper store (see architecture/keeper-plus-persona-equals-expert).

Precedence: explicit env ARCHETYPAL_CARDS_PATH -> default BRIX path. The
default preserves today's behavior exactly (zero change when unset).

Fail-closed, per the Council/trickster containment finding: a card is fed to
the model as a verbatim system prompt, so the supplying root must be a
contained coordinate, not an open door. cards_root() raises CardsRootError if
the resolved root does not exist; resolve_under_cards_root() raises if a
joined card/pool path resolves outside that root (symlink-escape guard).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

DEFAULT_CARDS_ROOT = Path("/srv/git/archetypal-intelligence-working/cards")

# Mirrors facets' VALID_KERNEL_INVARIANTS (facets/persona_validation.py) — the
# canonical 8-invariant set. Duplicated here per the dependency-layering
# invariant: agents_core cannot import facets (separate repo, would cycle).
CANONICAL_KERNEL_INVARIANTS: list[str] = [
    "Sovereignty of the user",
    "Truth integrity",
    "Possibility, not prescription",
    "Friction as signal",
    "Flame preservation",
    "Provenance",
    "Humanitarian over economic",
    "Antivenin / open behavioral literacy",
]

_VOICE_EXEMPLARS_FLOOR = 3
_KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class CardsRootError(Exception):
    """Cards root missing/not-a-directory, a card/pool path escaping it, or an
    invalid/empty deck — always a legible, named failure, never a silent
    degrade (fail-closed per the Council containment finding)."""


def cards_root() -> Path:
    """Resolve the archetypal cards root. Env override -> default BRIX path.

    Raises CardsRootError if the resolved root is not an existing directory.
    """
    raw = os.environ.get("ARCHETYPAL_CARDS_PATH") or str(DEFAULT_CARDS_ROOT)
    root = Path(raw).resolve()
    if not root.is_dir():
        raise CardsRootError(
            f"Archetypal cards root not found or not a directory: {root} "
            "(set ARCHETYPAL_CARDS_PATH to override)"
        )
    return root


def resolve_under_cards_root(path: Path) -> Path:
    """Resolve `path` and require it stay under cards_root() (symlink-escape guard).

    Raises CardsRootError if the resolved path is not relative to cards_root().
    """
    root = cards_root()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise CardsRootError(f"Path escapes cards root: {resolved} not under {root}")
    return resolved


def validate_deck_card(path: str | Path) -> list[str]:
    """Validate a reviewer-deck Composition card. Never raises; returns error strings.

    An agents-core-local implementation of the same documented Composition
    schema facets' validate_definition() enforces (facets/personas/README.md
    + facets-adapter-v0 spec) — not a shared import (cross-repo cycle).
    Enforces: slug present; composition.primitives a dict of kebab-case ids
    summing to 1.0, each resolvable under cards_root()/primitives/;
    voice_exemplars >= 3; domains non-empty; kernel_invariants drawn from the
    canonical 8.
    """
    errors: list[str] = []
    try:
        data = yaml.safe_load(Path(path).read_text())
    except Exception as exc:
        return [f"failed to parse YAML: {exc}"]

    if not isinstance(data, dict):
        return ["top-level YAML value is not a mapping"]

    slug = data.get("slug")
    if not slug or not isinstance(slug, str):
        errors.append("slug is missing or not a string")

    primitives = None
    try:
        primitives = data["composition"]["primitives"]
    except (KeyError, TypeError):
        errors.append("composition.primitives missing or malformed")

    if isinstance(primitives, dict):
        if not primitives:
            errors.append("composition.primitives is empty")
        else:
            for prim_id in primitives:
                if not isinstance(prim_id, str) or not _KEBAB_RE.match(prim_id):
                    errors.append(f"primitive id {prim_id!r} is not kebab-case")
            weight_sum = sum(
                v for v in primitives.values() if isinstance(v, (int, float))
            )
            if abs(weight_sum - 1.0) > 1e-6:
                errors.append(
                    f"composition.primitives weights sum to {weight_sum}, expected 1.0"
                )
            try:
                primitives_root = cards_root() / "primitives"
                for prim_id in primitives:
                    if not isinstance(prim_id, str):
                        continue
                    matches = list(primitives_root.rglob(f"{prim_id}.yaml"))
                    if not matches:
                        errors.append(
                            f"primitive {prim_id!r} does not resolve under {primitives_root}"
                        )
            except CardsRootError as exc:
                errors.append(f"cards root unresolvable while checking primitives: {exc}")
    elif primitives is not None:
        errors.append("composition.primitives is not a mapping")

    voice_exemplars = data.get("voice_exemplars")
    if not isinstance(voice_exemplars, list) or len(voice_exemplars) < _VOICE_EXEMPLARS_FLOOR:
        got = len(voice_exemplars) if isinstance(voice_exemplars, list) else 0
        errors.append(f"voice_exemplars has {got} entries; minimum is {_VOICE_EXEMPLARS_FLOOR}")

    domains = data.get("domains")
    if not isinstance(domains, list) or not domains:
        errors.append("domains must be a non-empty list")

    kernel_invariants = data.get("kernel_invariants")
    if not isinstance(kernel_invariants, list):
        errors.append("kernel_invariants must be a list")
    else:
        for inv in kernel_invariants:
            if inv not in CANONICAL_KERNEL_INVARIANTS:
                errors.append(
                    f"kernel_invariant {inv!r} is not in the canonical 8-invariant set"
                )

    return errors


def load_deck_cards(pool_dir: Path) -> list[dict]:
    """Load and validate every card in a fail-closed deck pool (e.g. `reviewer`).

    Never silently degrades (§Invariants, Path containment is fail-closed):
    raises CardsRootError naming exactly what is missing — the empty pool
    path if the directory has no cards, or the first validation error
    (e.g. an unresolved primitive id) if any card is invalid.

    Returns a list of {"path": Path, "data": dict} for every valid card.
    """
    if not pool_dir.is_dir():
        raise CardsRootError(f"deck pool is empty: no such directory {pool_dir}")

    card_paths = sorted(pool_dir.glob("*.yaml"))
    if not card_paths:
        raise CardsRootError(f"deck pool is empty: no cards found under {pool_dir}")

    cards: list[dict] = []
    for card_path in card_paths:
        resolved = resolve_under_cards_root(card_path)
        errors = validate_deck_card(resolved)
        if errors:
            raise CardsRootError(
                f"invalid deck card {resolved.name}: {'; '.join(errors)}"
            )
        cards.append({"path": resolved, "data": yaml.safe_load(resolved.read_text())})
    return cards
