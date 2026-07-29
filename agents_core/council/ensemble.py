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
import sys
import uuid
from dataclasses import dataclass, field

from lapis_engine.types import RunContext

from agents_core.council import DECKS_ROOT
from agents_core.council.gravitywell_adapter import GravityWellAdapter
from agents_core.council.persona_card_entity import PersonaCardEntity
from agents_core.cards import load_deck_cards

REVIEWER_POOL = "reviewer"


def _new_run_id() -> str:
    return f"ensemble-{uuid.uuid4().hex[:12]}"


@dataclass
class VoiceEntry:
    slug: str
    output: str
    voicing_provenance: list = field(default_factory=list)


@dataclass
class EnsembleResult:
    roster: list  # list[VoiceEntry]
    deck: str
    prompt_hash: str
    run_id: str
    synthesized: bool = False
    roster_semantics: str = "raw-divergent-voices-no-synthesis"


def run_ensemble(
    prompt: str,
    *,
    deck: str = REVIEWER_POOL,
    ctx: RunContext | None = None,
    principal: str | None = None,
) -> EnsembleResult:
    """Voice every card in `deck` once, in-process, and return one composed result.

    Fan-out-and-compose (the NPC-Worldwide `Team.convene` shape, not
    route-to-one `orchestrate`): each card is voiced sequentially — N
    GravityWell calls, no inference batching — and the outputs are collected
    raw, with synthesized always False (see module docstring). All voices
    share one GravityWellAdapter/principal so they ride one GW admission
    group; per-voice provenance is sliced from the adapter's voicing_events
    by call order.
    """
    run_id = _new_run_id()
    cards = load_deck_cards(DECKS_ROOT / deck)

    shared_principal = principal or run_id
    adapter = GravityWellAdapter(principal=shared_principal, on_wake_fail="park")

    roster: list[VoiceEntry] = []
    for card in cards:
        entity = PersonaCardEntity.load(card["path"], adapter)
        start = len(adapter.voicing_events)
        output = entity.act(prompt, ctx)
        roster.append(VoiceEntry(
            slug=entity.slug,
            output=output,
            voicing_provenance=list(adapter.voicing_events[start:]),
        ))

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
