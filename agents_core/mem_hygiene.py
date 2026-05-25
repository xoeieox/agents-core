"""mem_hygiene — Recency guard for mem.db hygiene passes.

Provides RecencyGuard (probe-with-latch) and HygieneVerdict for use by
hygiene automation that proposes pruning mem.db artifacts.

USAGE CONVENTION:

Any caller that runs a hygiene pass and calls RecencyGuard.evaluate() MUST
pass the resulting HygieneVerdict to weaver.hygiene_audit.AuditLog.record().
The guard is only as useful as the chassis that records its decisions.
A guard that stays a prune and has no log entry was never there.

The guard does not import from weaver. The convention is enforced by the
caller, not at runtime here.

LATCH LIFECYCLE:

The guard is in PROBE state until a future Facets-ratified, comprehensive
de-crystallization session writes latch_key to mem.db with one of:
  - "absorbed"  — Q2-full subsumed the guard into a broader policy
  - "replaced"  — Q2-full superseded with a different mechanism
  - "forgotten" — Q2-full landed without addressing the guard (process gap)

When latch_state != PROBE the guard stops downgrading. The caller must record
a final latch-state entry to the chassis.
"""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from agents_core.mem import MemoryStore


class LatchState(str, Enum):
    PROBE     = "probe"      # guard is active; no Q2-full ratification seen
    ABSORBED  = "absorbed"   # Q2-full subsumed the guard into a broader policy
    REPLACED  = "replaced"   # Q2-full superseded with a different mechanism
    FORGOTTEN = "forgotten"  # Q2-full landed without addressing the guard (process gap)


@dataclass
class HygieneVerdict:
    artifact_key:      str
    artifact_age_days: float | None    # None if artifact not found in store
    proposed_action:   str             # 'deprecate' | 'merge' | 'keep_both' | 'delete'
    final_action:      str             # same enum; equals proposed_action if fired
    fired:             bool
    staying_reason:    str | None      # populated when fired=False
    latch_state:       LatchState
    sweep_id:          str | None = None


class RecencyGuard:
    """Probe-with-latch recency guard for mem.db hygiene passes.

    Reads artifact age from the store and downgrades destructive proposed
    actions (deprecate, delete) to keep_both when the artifact is younger
    than recency_threshold_days. Safe actions (merge, keep_both) always fire.

    The latch is per-instance cached — read once on first access and stored
    for the lifetime of the instance. One RecencyGuard per sweep pass.
    """

    def __init__(
        self,
        store: MemoryStore,
        recency_threshold_days: float = 30,
        latch_key: str = "decision/decrystallization-full-ratified",
    ):
        self._store = store
        self._recency_threshold_days = recency_threshold_days
        self._latch_key = latch_key
        self._cached_latch_state: LatchState | None = None

    def evaluate(
        self,
        artifact_key: str,
        proposed_action: Literal["deprecate", "merge", "keep_both", "delete"],
        sweep_id: str | None = None,
    ) -> HygieneVerdict:
        """Snap to actual artifact age and decide whether to fire or stay.

        If proposed_action would delete a newer record, downgrades to
        keep_both. Returns HygieneVerdict for the chassis to record.

        Does NOT execute the action — action execution is the caller's
        responsibility.
        """
        record = self._store.get(artifact_key)

        if record is None:
            # Cannot measure age — fire through with artifact_age_days=None.
            return HygieneVerdict(
                artifact_key=artifact_key,
                artifact_age_days=None,
                proposed_action=proposed_action,
                final_action=proposed_action,
                fired=True,
                staying_reason=None,
                latch_state=self.latch_state,
                sweep_id=sweep_id,
            )

        created_at_str = record["created_at"]
        created_at = datetime.fromisoformat(created_at_str)
        # Strip timezone info if present so arithmetic works with utcnow()
        if created_at.tzinfo is not None:
            created_at = created_at.replace(tzinfo=None)
        artifact_age_days = (
            datetime.utcnow() - created_at
        ).total_seconds() / 86400

        current_latch = self.latch_state
        destructive = proposed_action in ("deprecate", "delete")

        if (
            current_latch == LatchState.PROBE
            and destructive
            and artifact_age_days < self._recency_threshold_days
        ):
            return HygieneVerdict(
                artifact_key=artifact_key,
                artifact_age_days=artifact_age_days,
                proposed_action=proposed_action,
                final_action="keep_both",
                fired=False,
                staying_reason="recency_guard",
                latch_state=current_latch,
                sweep_id=sweep_id,
            )

        return HygieneVerdict(
            artifact_key=artifact_key,
            artifact_age_days=artifact_age_days,
            proposed_action=proposed_action,
            final_action=proposed_action,
            fired=True,
            staying_reason=None,
            latch_state=current_latch,
            sweep_id=sweep_id,
        )

    @property
    def latch_state(self) -> LatchState:
        """Check mem.db for latch key. Per-instance cached after first read."""
        if self._cached_latch_state is not None:
            return self._cached_latch_state

        record = self._store.get(self._latch_key)
        if record is None:
            state = LatchState.PROBE
        else:
            content = (record.get("content") or "").strip().lower()
            try:
                state = LatchState(content)
            except ValueError:
                # Key exists but content is unrecognized — FORGOTTEN (process gap).
                state = LatchState.FORGOTTEN

        self._cached_latch_state = state
        return state
