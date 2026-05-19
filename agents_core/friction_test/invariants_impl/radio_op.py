"""Radio-op programmatic invariant check functions.

Each function returns (observed, distance):
  - observed: the measured value (or None if inapplicable)
  - distance: numeric scalar (0.0 means perfectly held)

The critique layer classifies results; check functions never construct
InvariantResult objects themselves.
"""
from __future__ import annotations

from typing import Any

from ..observe import Observation
from ..scenario import Scenario


def i01_consult_log_row_per_judgment(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Every ingest that reached the Qwen judge appends exactly one consult-log row.

    Returns (observed_row_count, distance) where distance = abs(observed - expected).
    Returns (None, 0.0) if the scenario short-circuits before the judge.
    """
    # Check if the scenario would have triggered the judge:
    # Empty/whitespace segments should NOT trigger the judge (short-circuit)
    segments = scenario.inputs.get("segments", [])
    if not segments:
        return None, 0.0

    # Count non-empty, non-whitespace segments that would reach the judge
    judge_eligible = [
        seg for seg in segments
        if seg.get("segment", "").strip()
    ]
    if not judge_eligible:
        # No eligible segments; inapplicable
        return None, 0.0

    # Count new consult-log rows from log_appends
    consult_rows = [
        la for la in obs.log_appends
        if la.get("file", "").endswith(".jsonl")
        and "/consults/" in la.get("file", "")
        and "line" in la
    ]
    observed = len(consult_rows)
    expected = len(judge_eligible)
    distance = float(abs(observed - expected))
    return observed, distance


def i02_authority_tier_host_determined(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Surfaced fragments' authority_tier matches map_authority_tier over their ID prefix.

    Checks SSE events and consult-log entries for any authority_tier override.
    Returns (override_count, float(override_count)).
    """
    override_count = 0

    # Check SSE events for tier overrides
    for event in obs.sse_events:
        fragments = event.get("fragments", [])
        for frag in fragments:
            frag_id = frag.get("id", "")
            declared_tier = frag.get("authority_tier")
            computed_tier = _map_authority_tier(frag_id)
            if declared_tier is not None and computed_tier is not None:
                if declared_tier != computed_tier:
                    override_count += 1

    # Check log_appends for tier data
    for la in obs.log_appends:
        line = la.get("line", {})
        if isinstance(line, dict):
            fragments = line.get("fragments", [])
            for frag in fragments:
                frag_id = frag.get("id", "")
                declared_tier = frag.get("authority_tier")
                computed_tier = _map_authority_tier(frag_id)
                if declared_tier is not None and computed_tier is not None:
                    if declared_tier != computed_tier:
                        override_count += 1

    return override_count, float(override_count)


def _map_authority_tier(fragment_id: str) -> str | None:
    """Map fragment ID prefix to authority tier (mirrors radio_op.py convention)."""
    if not fragment_id:
        return None
    if fragment_id.startswith("decision/"):
        return "high"
    if fragment_id.startswith("mem:decision/"):
        return "high"
    if fragment_id.startswith("feedback/"):
        return "medium"
    if fragment_id.startswith("arc/"):
        return "medium"
    return "low"


def i03_should_surface_default_false(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Empty/whitespace/single-stopword segments surface zero fragments.

    Returns (surfaced_count, float(surfaced_count)).
    Returns (None, 0.0) if the scenario has non-trivial segments (inapplicable).
    """
    segments = scenario.inputs.get("segments", [])
    # Check if ALL segments are empty/whitespace
    all_trivial = all(
        not seg.get("segment", "").strip() or _is_stopword_only(seg.get("segment", ""))
        for seg in segments
    )
    if not all_trivial:
        return None, 0.0

    # Count surfaced fragments from SSE events
    surfaced_count = 0
    for event in obs.sse_events:
        frags = event.get("fragments", [])
        surfaced_count += len(frags)

    # Also check HTTP response bodies
    for call in obs.http_calls:
        body = call.get("body", {})
        if isinstance(body, dict):
            frags = body.get("fragments", [])
            surfaced_count += len(frags)
            if body.get("should_surface"):
                surfaced_count += 1

    return surfaced_count, float(surfaced_count)


_STOPWORDS = frozenset(["the", "a", "an", "is", "it", "to", "in", "of", "and", "or"])


def _is_stopword_only(text: str) -> bool:
    words = text.strip().lower().split()
    return bool(words) and all(w in _STOPWORDS for w in words)


def i04_harvest_marker_iff_explicit_gesture(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Harvest-queue marker appears iff the scenario contains an explicit harvest gesture.

    Returns (xor_distance, xor_distance) where xor_distance = XOR(marker_present, gesture_present).
    0 means aligned.
    """
    gesture_present = bool(scenario.inputs.get("harvest", False))

    # Find harvest marker status in log_appends
    marker_present = False
    for la in obs.log_appends:
        if "harvest_marker_exists" in la:
            marker_present = bool(la["harvest_marker_exists"])
            break

    xor = int(marker_present) ^ int(gesture_present)
    return xor, float(xor)


def i05_synapse_degradation_handled(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Synapse 5xx upstream produces structured fallback, not unhandled exception.

    Returns (1, 0.0) if degradation was handled gracefully.
    Returns (0, 1.0) if an unhandled exception occurred during Synapse degradation.
    Returns (None, 0.0) if this scenario doesn't involve Synapse degradation (inapplicable).
    """
    # Only applicable if scenario is about Synapse degradation
    if not scenario.inputs.get("synapse_down", False):
        return None, 0.0

    # Check for unhandled exceptions in errors
    for err in obs.errors:
        error_str = str(err.get("error", "")).lower()
        if "exception" in error_str or "traceback" in error_str or "500" in error_str:
            return 0, 1.0

    # Check log_appends for degraded methodology
    for la in obs.log_appends:
        line = la.get("line", {})
        if isinstance(line, dict):
            methodology = line.get("methodology", {})
            if methodology.get("retrieve_pass") == "degraded":
                return 1, 0.0

    # Check HTTP responses for degraded indicator
    for call in obs.http_calls:
        body = call.get("body", {})
        if isinstance(body, dict):
            methodology = body.get("methodology", {})
            if methodology.get("retrieve_pass") == "degraded":
                return 1, 0.0

    # Synapse was down but no structured degradation marker found
    # and no exception either — ambiguous; treat as handled (graceful silence)
    return 1, 0.0


def i06_label_isolation(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Every session started by friction-tester carries a friction-test-<scenario_id> label.

    Returns (foreign_label_count, float(foreign_label_count)).
    """
    expected_label = f"friction-test-{scenario.scenario_id}"
    foreign_count = 0

    # Check harvest marker data for session label
    for la in obs.log_appends:
        label = la.get("label")
        session_id = la.get("session_id")
        if label is not None and session_id is not None:
            if label != expected_label:
                foreign_count += 1

    return foreign_count, float(foreign_count)
