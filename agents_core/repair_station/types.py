"""Repair station contract types — Tier, EscalationPolicy, Incident.

Tier is defined here (the contract owns the vocabulary). It drives three downstream
behaviors in Leg 2 / the queue — NOT acted on in this leg:
  - elevator-queue priority / preemption
  - degradation tolerance (HIGH/CRITICAL stations degrade gates when they fail)
  - model-escalation aggressiveness (higher tier → more capable model)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Tier(enum.IntEnum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class EscalationPolicy:
    """Governs when a station fire produces an incident.

    kind="first"    — escalate on first qualifying fire (use for definitive
                      infra-break signals that are never transient).
    kind="n_within" — escalate after `count` fires within `window_s` seconds.
    """
    kind: str
    count: int = 1
    window_s: float = 0.0


def first() -> EscalationPolicy:
    """Escalate on the first fire."""
    return EscalationPolicy(kind="first", count=1)


def n_within(count: int, window_s: float) -> EscalationPolicy:
    """Escalate after `count` fires within `window_s` seconds."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    return EscalationPolicy(kind="n_within", count=count, window_s=window_s)


@dataclass
class Incident:
    """A well-formed incident as it lands in the intake.

    Consumers (Leg 2 Expert) read these from the intake; this dataclass is the
    deserialization view, not the write path (use escalate() to create incidents).
    """
    incident_id: str
    station_id: str
    stable_pointer: str
    error_signal: dict
    author_intent: str
    tier: Tier
    error_signature: str
    status: str        # "open" | "closed" | "dismissed"
    back_ref: str | None  # close-the-loop reference; Expert sets this on resolution
    created_at: str
    updated_at: str
    closed_at: str | None = None  # UTC ISO; set by close_incident()/dismiss_incident()
    prose: str | None = None      # close's `resolution` or dismiss's `reason` — one column,
                                   # the close/dismiss distinction lives in `status` alone.


# Per-run-unique field names: values that make an otherwise-identical failure look
# novel under whole-payload signature hashing (run ids, deliberation ids, timestamps,
# resolved commit SHAs, per-run counts). This is the exclusion vocabulary shared by
# Leg 2's `signature_fields` amendments in orchestrator.py (which whitelist the fields
# to KEEP per station) and Leg 3's triage collapse (which blacklists these fields
# across all stations to regroup the existing backlog). Keep the two in sync: a field
# a station excludes via signature_fields should appear here.
PER_RUN_UNIQUE_KEYS: frozenset[str] = frozenset({
    "run_id",
    "deliberation_id",
    "last_heartbeat",
    "resolved_sha",
    "rounds_affected",
    "timestamp",
    "created_at",
    "count",
})
