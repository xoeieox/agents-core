"""Compatibility adapter — derive SpecReviewBrief from DeliberationEnvelope.

This module exists to prove that the shared deliberation service can be
consumed by spec-review; the actual migration of spec_review.py to use
this service is a follow-on unit (shared-deliberation-spec-review-migration-v0),
bound to the H4 cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

if TYPE_CHECKING:
    from agents_core.shared_deliberation.envelope import DeliberationEnvelope


@dataclass
class SpecReviewBriefCompat:
    """Minimal SpecReviewBrief-like shape derived from DeliberationEnvelope.

    For compatibility testing only; the real SpecReviewBrief in lapis-pm is the
    canonical schema. This adapter proves that the fields mapping is straightforward:
    envelope fields map 1:1 to brief fields.
    """

    # Council fields (straight from envelope)
    council_status: Literal[
        "resolved", "open", "laid-down", "failed", "timeout", "error", "closed"
    ]
    council_landing: str
    council_open_questions: list[str]
    council_confidence: str
    council_positions: list[dict]
    council_run_id: str

    # Operator/voicing degradation (from envelope)
    council_voicing_requested: str = "gravitywell"
    council_voicing_effective: str = "unknown"
    council_voicing_degraded: bool = False
    council_voicing_degraded_reason: str = ""
    facets_operator_requested: str = ""
    facets_operator_effective: str = "unknown"
    facets_operator_degraded: bool = False

    # Facets envelope (for reference)
    facets_deliberation: dict | None = None

    # Caller context (optional, provided at adapter callsite)
    spec_path: Path = field(default_factory=lambda: Path(""))
    target_id: str = ""
    repo: str = ""
    elapsed_s: float = 0.0


def derive_spec_review_brief_compat(
    envelope: DeliberationEnvelope,
    spec_path: Path,
    target_id: str,
    repo: str,
    elapsed_s: float = 0.0,
) -> SpecReviewBriefCompat:
    """Derive a SpecReviewBrief-like shape from a DeliberationEnvelope.

    Maps envelope fields directly to brief fields. Caller must provide
    spec_path, target_id, repo, and elapsed_s (not available in the envelope).

    Returns a SpecReviewBriefCompat suitable for testing the integration.
    """
    status = envelope.council_status or "error"
    if status not in ("resolved", "open", "laid-down", "failed", "timeout", "closed"):
        status = "error"

    return SpecReviewBriefCompat(
        council_status=status,
        council_landing=envelope.council_landing or "",
        council_open_questions=envelope.council_open_questions or [],
        council_confidence=envelope.council_confidence or "",
        council_positions=envelope.council_positions or [],
        council_run_id=envelope.council_run_id or "",
        council_voicing_requested=envelope.council_voicing_requested or "gravitywell",
        council_voicing_effective=envelope.council_voicing_effective or "unknown",
        council_voicing_degraded=envelope.council_voicing_degraded,
        council_voicing_degraded_reason=envelope.council_voicing_degraded_reason or "",
        facets_operator_requested=envelope.operator_requested or "",
        facets_operator_effective=envelope.operator_effective or "unknown",
        facets_operator_degraded=envelope.operator_requested != envelope.operator_effective if envelope.operator_requested else False,
        facets_deliberation=envelope.facets,
        spec_path=spec_path,
        target_id=target_id,
        repo=repo,
        elapsed_s=elapsed_s,
    )
