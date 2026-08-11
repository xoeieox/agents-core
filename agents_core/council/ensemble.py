"""agents_core.council.ensemble — in-process persona-ensemble fast-path coordinator.

Single-turn, in-process coordinator that assembles N persona-card entities,
voices each once, and returns one composed result. Sibling of
persona_card_entity.py / narrator_entity.py. Borrows the *shape* of NPC
Worldwide's Team in-process coordination (`convene` fan-out) — a code-shape
borrow, not a code copy; attribution deposit is a separate PM step, out of
scope here.

Distinct from agents_core.council.cli.run_deliberation: that path is the
governed, durable, multi-hour Engine.run dialectic (DEFAULT_TURNS=8 +
synthesis + position cast), spawned as a queue worker and polled. This module
is the lightweight in-process counterpart — one turn, N sequential GW calls,
structured aggregation, no synthesis LLM call. See the
council-inprocess-ensemble-v0 spec for the full design rationale, including
why synthesized is always False (a structural property, not an omission).

Import discipline: only PersonaCardEntity + RunContext from lapis_engine
(via persona_card_entity's own import). Engine / DeliberationDirector are
never imported here — those stay behind run_deliberation's function-local
import in cli.py.

Governance boundary: this module never edits a repo or dispatches a fixer.
open_slot_for() is the ONLY way an ensemble result crosses into the governed
async lane, and it is best-effort — mirrors shaper.py::_record_dispatch_slot
(the work is the real work; the slot is coordination metadata).
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from lapis_engine.types import RunContext

from agents_core.council import DECKS_ROOT
from agents_core.council.gravitywell_adapter import GravityWellAdapter
from agents_core.council.persona_card_entity import PersonaCardEntity
from agents_core.cards import load_deck_cards

REVIEWER_POOL = "reviewer"

log = logging.getLogger("agents_core.council.ensemble")

# Measured-free band per the 2026-08-09 GravityWell concurrency sweep
# (finding/gw-concurrency-sweep-measured-2026-08-09).
_DEFAULT_VOICING_MAX_CONCURRENT = 4


def _new_run_id() -> str:
    return f"ensemble-{uuid.uuid4().hex[:12]}"


def _council_voicing_cap() -> int:
    """Parse COUNCIL_VOICING_MAX_CONCURRENT from the env.

    Invalid or <1 values fall back to 1 and log a WARNING (never crash on a
    bad env var). Missing env var uses the measured-free-band default.
    Read fresh on every call — never cached at import time — so tests (and
    live config reloads) can change it without a process restart.
    """
    raw = os.environ.get("COUNCIL_VOICING_MAX_CONCURRENT")
    if raw is None:
        return _DEFAULT_VOICING_MAX_CONCURRENT
    try:
        value = int(raw)
        if value < 1:
            raise ValueError(f"COUNCIL_VOICING_MAX_CONCURRENT={raw!r} must be >= 1")
        return value
    except ValueError:
        log.warning(
            f"COUNCIL_VOICING_MAX_CONCURRENT={raw!r} is invalid — falling back to 1"
        )
        return 1


def _select_rotation(cards: list[dict], voices: int | None, rotation_key: str | None) -> list[dict]:
    """Deterministically select `voices` cards from `cards`, keyed on
    `rotation_key`, preserving deck order among the selected cards.

    `voices=None` (default) returns the full deck unchanged. Selection is a
    pure function of (cards, voices, rotation_key) — no wall-clock or RNG
    dependence — so the same (deck, voices, rotation_key) always draws the
    same subset, and a different rotation_key draws a different one.
    """
    if voices is None or voices >= len(cards):
        return cards
    key_material = rotation_key or ""
    ranked = sorted(
        cards,
        key=lambda c: hashlib.sha256(
            f"{key_material}:{c['data']['slug']}".encode("utf-8")
        ).hexdigest(),
    )
    selected_slugs = {c["data"]["slug"] for c in ranked[:voices]}
    return [c for c in cards if c["data"]["slug"] in selected_slugs]


@dataclass
class VoiceEntry:
    slug: str
    output: str
    voicing_provenance: list = field(default_factory=list)
    error: str | None = None
    """Set (non-None) when this voice failed. `output` is empty and
    `voicing_provenance` is empty in that case. Failed voices are never
    dropped from the roster — they occupy their original deck index — so a
    single dead voice cannot reorder or silently shrink the panel."""


@dataclass
class EnsembleResult:
    roster: list  # list[VoiceEntry]
    deck: str
    prompt_hash: str
    run_id: str
    synthesized: bool = False
    roster_semantics: str = "raw-divergent-voices-no-synthesis"


def _voice_card(card: dict, prompt: str, ctx: RunContext | None, shared_principal: str) -> VoiceEntry:
    """Voice one deck card on its own adapter instance.

    Per-voice adapter instance (not a shared one) is the provenance-isolation
    seam: GravityWellAdapter.voicing_events defaults to a fresh list per
    instance (see __post_init__), so concurrent voices can never interleave
    each other's events — no event tagging on a shared adapter, which is a
    concurrent-append race surface. `shared_principal` is still passed
    explicitly to every instance so all voices in this run ride one GW
    admission group; admission identity is decoupled from data isolation,
    never inherited implicitly.

    Never raises: a single voice's failure is caught and returned as a
    VoiceEntry carrying `error` and empty provenance, so run_ensemble can
    place it at its original deck index without collapsing the panel.
    """
    slug = card["data"].get("slug") or card["path"].stem
    adapter = GravityWellAdapter(principal=shared_principal, on_wake_fail="park")
    try:
        entity = PersonaCardEntity.load(card["path"], adapter)
        output = entity.act(prompt, ctx)
        return VoiceEntry(
            slug=entity.slug,
            output=output,
            voicing_provenance=list(adapter.voicing_events),
        )
    except Exception as exc:  # partial-success: one dead voice must not sink the panel
        return VoiceEntry(slug=slug, output="", voicing_provenance=[], error=str(exc))


def run_ensemble(
    prompt: str,
    *,
    deck: str = REVIEWER_POOL,
    ctx: RunContext | None = None,
    principal: str | None = None,
    voices: int | None = None,
    rotation_key: str | None = None,
) -> EnsembleResult:
    """Voice every card in `deck` once, in-process, and return one composed result.

    Fan-out-and-compose (the NPC-Worldwide `Team.convene` shape, not
    route-to-one `orchestrate`): cards are voiced concurrently, under a
    per-ensemble cap (env COUNCIL_VOICING_MAX_CONCURRENT, default 4 — the
    measured-free band; see _council_voicing_cap), and the outputs are
    collected raw, with synthesized always False (see module docstring).
    COUNCIL_VOICING_MAX_CONCURRENT=1 reproduces the historical sequential
    behavior exactly (fallback and A/B surface).

    All voices share one GW admission group via `shared_principal`, passed
    explicitly to each voice's own adapter instance (see _voice_card) — data
    isolation (per-voice adapter) and admission identity (shared principal)
    are independent knobs.

    `voices` / `rotation_key` (both optional, default None = full deck) draw
    a deterministic subset of the deck for this run — see _select_rotation.

    Roster order in the result is deck order, independent of voicing
    completion order. A single voice's failure never drops it from the
    roster or reorders the others (see _voice_card); run_ensemble raises
    only when every voice fails.
    """
    run_id = _new_run_id()
    cards = load_deck_cards(DECKS_ROOT / deck)
    cards = _select_rotation(cards, voices, rotation_key)

    shared_principal = principal or run_id
    cap = _council_voicing_cap()

    roster: list[VoiceEntry | None] = [None] * len(cards)
    with ThreadPoolExecutor(max_workers=cap) as pool:
        future_to_idx = {
            pool.submit(_voice_card, card, prompt, ctx, shared_principal): idx
            for idx, card in enumerate(cards)
        }
        for future in as_completed(future_to_idx):
            roster[future_to_idx[future]] = future.result()

    if roster and all(entry.error is not None for entry in roster):
        raise RuntimeError(
            f"council ensemble {run_id}: all {len(roster)} voices failed — "
            + "; ".join(f"{e.slug}: {e.error}" for e in roster)
        )

    return EnsembleResult(
        roster=roster,
        deck=deck,
        prompt_hash="sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        run_id=run_id,
    )


def open_slot_for(
    result: EnsembleResult,
    *,
    project_id: str,
    contributor_type: str = "reviewer",
) -> str | None:
    """Best-effort slot-open — the ONLY seam through which an ensemble result
    can cross into the governed async lane (opening a slot, never editing a
    repo or dispatching a fixer directly).

    Mirrors shaper.py::_record_dispatch_slot: never raises. v0 is BRIX-only —
    SlotStore().create_slot() is called directly; an OffMasterWriteError
    (running off the slots master) is caught and logged, not retried via the
    :8405 HTTP path (that path is a named follow, not built in v0).
    """
    try:
        from agents_core.slots import SlotStore

        horizon = {
            "project_summary": (
                f"council ensemble run {result.run_id} over deck={result.deck!r} "
                f"({len(result.roster)} voices, {result.roster_semantics})"
            ),
            "immediate_goal": (
                ", ".join(v.slug for v in result.roster)
            )[:280],
            "adjacent_slots": [],
        }
        return SlotStore().create_slot(
            project_id=project_id,
            contributor={"type": contributor_type, "id": result.run_id},
            horizon=horizon,
            slot_id=f"ensemble-{result.run_id}",
        )
    except Exception as exc:  # never block the caller — slot is coordination metadata
        # Matched by class name (not isinstance) so a failed `agents_core.slots`
        # import above can never mask itself behind a NameError on the type name.
        if type(exc).__name__ == "OffMasterWriteError":
            print(
                f"[ensemble:slot-open-skipped] off-master slot-open not supported "
                f"in v0 — run on BRIX: {exc}",
                file=sys.stderr,
            )
        else:
            print(f"[ensemble:slot-open-failed] {exc}", file=sys.stderr)
        return None
