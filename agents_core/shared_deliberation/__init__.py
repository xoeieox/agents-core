"""Shared deliberation service — reusable orchestration layer for Facets + Mirror Council.

This module provides a single async/HTTP service that wraps Facets and Mirror Council,
tapped by spec-review, Gardener (compost-recovery), and Sessions (send-to-deliberation).

Exports:
  - SharedDeliberationClient: httpx async client for remote callers
  - DeliberationRequest: request dataclass
  - DeliberationEnvelope: response dataclass
  - start_server: helper to start the service
"""

from agents_core.shared_deliberation.envelope import (
    DeliberationEnvelope,
    DeliberationRequest,
)
from agents_core.shared_deliberation.client import SharedDeliberationClient

__all__ = [
    "DeliberationEnvelope",
    "DeliberationRequest",
    "SharedDeliberationClient",
]
