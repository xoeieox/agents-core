"""Request and response schemas for the shared deliberation service."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DeliberationRequest:
    """Request shape for POST /v0/deliberate."""

    text: str
    context: dict
    triage: str = "full"  # "lightweight" | "full"
    caller: str = "unknown"  # "spec-review" | "gardener" | "sessions"
    council_voicing: str = "gravitywell"
    facets_operator: str = "gravitywell"
    seam: Optional[dict] = None  # reserved for jagged-seam tap (v0: unused)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeliberationEnvelope:
    """Response shape for POST /v0/deliberate."""

    # Core provenance
    deliberation_request_id: str
    triage: str  # "lightweight" | "full" (echo of request)
    triage_reason: Optional[str] = None  # e.g. "caller-requested" or "escalated: conflict-signal"
    triage_escalated: bool = False  # True if lightweight was bumped to full

    # Facets leg
    facets_ok: bool = False
    facets: Optional[dict] = None  # FacetsDeliberation.to_dict(); None if facets_ok is False
    facets_deliberation_id: Optional[str] = None
    operator_requested: Optional[str] = None  # facets methodology.operator_requested
    operator_effective: Optional[str] = None  # facets methodology.synthesis_operator

    # Council leg (absent when triage=="lightweight")
    council_ok: bool = False
    council_run_id: Optional[str] = None
    council_status: Optional[str] = None  # resolved | open | laid-down | failed | timeout | ...
    council_landing: Optional[str] = None
    council_confidence: Optional[str] = None
    council_open_questions: list[str] = field(default_factory=list)
    council_positions: list[dict] = field(default_factory=list)  # synthesis.positions from council YAML
    council_voicing_requested: Optional[str] = None
    council_voicing_effective: Optional[str] = None
    council_voicing_degraded: bool = False
    council_voicing_degraded_reason: Optional[str] = None

    # Extensions and errors
    extra_modes: list[dict] = field(default_factory=list)  # reserved typed seam output (empty in v0)
    errors: dict = field(default_factory=dict)  # per-leg error detail, {} on full success

    def to_dict(self) -> dict:
        return asdict(self)
