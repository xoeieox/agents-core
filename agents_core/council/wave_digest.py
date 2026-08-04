"""agents_core.council.wave_digest — per-round condensation for wave mode.

D6 of council-wave-mode-v0. Ruling 1 makes the digest a *condition* of the
5-7 seat width, not an optional extra: with no KV reuse available (prefix
caching is off and architecturally unavailable on GravityWell), transcript
growth without a digest is ~N^2*R^2/2 — this is what makes 5-7 seats
affordable at all.

Mechanics:
  - One call at the END of each round, after all seats resolve, before the
    next round's prompts are built. One per round, not one per seat.
  - Input is that round's seat outputs only — the previous round's digest is
    already carried forward, so digesting is incremental and cost stays flat
    per round instead of growing with round count.
  - Output is a short condensation, capped in the same order of magnitude as
    the seat cap (WAVE_SEAT_MAX_TOKENS in cli.py).
  - Later rounds (and the synthesis speaker) see the digests of all prior
    rounds plus the verbatim current round — never the raw prior rounds.
    The synthesis speaker seeing digests-not-transcript is a deliberate,
    visible fidelity trade, not an emergent one.
  - Failure posture: if the digest call fails, fall back to the verbatim
    prior round rather than dropping context. Slower is acceptable; amnesia
    is not.

Hard invariant this module does NOT own, and must never erode: every seat's
full, uncondensed output is still written to turns[] as its own
deliberation_turn entry (see cli.py's wave on_step). The digest governs only
what later rounds / synthesis are *shown* — never what is *recorded*. This
module has no "digest-only" storage mode and must never grow one.

The digest is a condensation, not a voice: it carries no persona and is
never emitted as a deliberation_turn — see cli.py's wave on_step, which
keeps digests out of turns[] entirely so they cannot pollute
_render_transcript or the turns_used count.
"""
from __future__ import annotations

from agents_core.llm import call_operator

DIGEST_MAX_TOKENS = 500

_DIGEST_SYSTEM = (
    "You condense one round of a multi-voice deliberation into a short, "
    "neutral summary for the next round's participants. Preserve the "
    "substance of disagreement — do not manufacture consensus. Do not "
    "adopt any speaker's persona; write as a neutral condenser. Be terse."
)


def build_round_digest(
    round_idx: int,
    seat_texts: list[tuple[str, str]],
    *,
    principal: str | None = None,
    timeout: int = 120,
) -> str:
    """Condense one round's seat outputs into a short digest.

    Args:
        round_idx: the round number (for the prompt only; not load-bearing).
        seat_texts: list of (speaker_id, content) for every seat that
            answered this round (failed seats are already excluded by the
            caller).
        principal: GW admission-group principal — threaded through so the
            digest call rides the same admission group as the seats it
            summarizes (H4's shared-principal requirement applies here too).
        timeout: per-call timeout in seconds.

    Returns the digest text. On any failure (GW unreachable, empty result,
    exception), returns the verbatim joined seat_texts instead — the
    caller never receives an empty/lost round (D6 failure posture).
    """
    verbatim = "\n\n".join(f"{speaker}: {content}" for speaker, content in seat_texts)
    if not verbatim:
        return ""

    prompt = (
        f"Round {round_idx} of a wave deliberation. Condense the following "
        f"{len(seat_texts)} voices into a short digest a later round can "
        "read instead of the full transcript:\n\n"
        f"{verbatim}"
    )
    try:
        result = call_operator(
            "gravitywell",
            prompt,
            system=_DIGEST_SYSTEM,
            temperature=0.3,
            timeout=timeout,
            on_wake_fail="park",
            principal=principal,
            lease_class="protected",
            max_tokens=DIGEST_MAX_TOKENS,
        )
    except Exception:
        result = None

    if not result:
        # Failure posture (D6): fall back to the verbatim round rather than
        # dropping context. Slower downstream, never amnesiac.
        return verbatim
    return result
