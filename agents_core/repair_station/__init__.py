"""Repair Station Contract v0 — alert-station nervous wiring.

Exposes the thin escalation client, registry, tier/policy types, and incident intake.
This is Leg 1 of repair-expert-v0: pure wiring, no brain, no model calls.

Usage:
    from agents_core.repair_station import escalate, Tier, first, n_within

    escalate(
        station_id="my-module/my-check",
        stable_pointer="agents_core/my_module.py",
        error_signal={"code": "ECONNREFUSED", "host": "localhost:5432"},
        author_intent="postgres connection failed at startup",
        escalation_policy=first(),
        tier=Tier.HIGH,
    )
"""

from .types import EscalationPolicy, Incident, Tier, first, n_within
from .escalate import (
    escalate,
    get_incident,
    list_open_incidents,
    close_incident,
    dismiss_incident,
)

__all__ = [
    "Tier",
    "EscalationPolicy",
    "first",
    "n_within",
    "Incident",
    "escalate",
    "get_incident",
    "list_open_incidents",
    "close_incident",
    "dismiss_incident",
]
