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
    status: str        # "open" | "superseded" | "resolved"
    back_ref: str | None  # close-the-loop reference; Expert sets this on resolution
    created_at: str
    updated_at: str
