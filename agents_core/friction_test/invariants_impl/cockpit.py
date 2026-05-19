"""Cockpit programmatic invariant check functions.

Each function returns (observed, distance). See radio_op.py for convention.
"""
from __future__ import annotations

from typing import Any

from ..observe import Observation
from ..scenario import Scenario

# Routes that must carry provenance per cockpit spec
PROVENANCE_ROUTES = {"/cockpit/api/workday", "/cockpit/api/substrate", "/cockpit/api/needs-call"}


def c01_cockpit_provenance_present(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Provenance-bearing cockpit routes return a top-level 'provenance' field.

    Only checks GET calls to routes in PROVENANCE_ROUTES.
    Returns (routes_missing_count, float(routes_missing_count)).
    Returns (None, 0.0) if no provenance-bearing route was called.
    """
    relevant_calls = [
        c for c in obs.http_calls
        if c.get("method") == "GET"
        and any(r in c.get("url", "") for r in PROVENANCE_ROUTES)
    ]
    if not relevant_calls:
        return None, 0.0

    missing = 0
    for call in relevant_calls:
        body = call.get("body", {})
        if not isinstance(body, dict):
            missing += 1
            continue
        provenance = body.get("provenance")
        if not provenance:
            missing += 1
            continue
        # Minimal shape check: must have agent_id, signature, mode
        if not all(k in provenance for k in ("agent_id", "signature", "mode")):
            missing += 1

    return missing, float(missing)


def c02a_directive_writes_to_commentstore(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """POST directive results in exactly 1 new line in /srv/lapis/targets/comments/<tid>.jsonl.

    Returns (new_row_count, abs(new_row_count - 1)).
    Returns (None, 0.0) if no directive call was made.
    """
    # Find POST directive call
    directive_calls = [
        c for c in obs.http_calls
        if c.get("method") == "POST"
        and "/directive" in c.get("url", "")
        and c.get("status", 500) < 500
    ]
    if not directive_calls:
        return None, 0.0

    # Find matching log_appends for the comment store
    for la in obs.log_appends:
        if la.get("file", "").endswith(".jsonl") and "/comments/" in la.get("file", ""):
            new_count = la.get("new_count", 0)
            return new_count, float(abs(new_count - 1))

    # No log_append found for comment store — means 0 rows written
    return 0, 1.0


def c02b_directive_write_latency(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """POST directive CommentStore append completes within 200ms of HTTP response.

    This is a provisional invariant (200ms is a guess).
    Returns (latency_ms, max(0, latency_ms - expected)).
    The critique layer applies tolerance; the check function must not.
    Returns (None, 0.0) if no directive call was made.
    """
    # Find POST directive call latency
    directive_calls = [
        c for c in obs.http_calls
        if c.get("method") == "POST"
        and "/directive" in c.get("url", "")
        and c.get("status", 500) < 500
    ]
    if not directive_calls:
        return None, 0.0

    # Use the HTTP response latency as a proxy (write_latency_ms from log_appends if available)
    for la in obs.log_appends:
        if la.get("file", "").endswith(".jsonl") and "/comments/" in la.get("file", ""):
            write_latency = la.get("write_latency_ms")
            if write_latency is not None:
                expected = 200
                distance = max(0.0, float(write_latency) - expected)
                return float(write_latency), distance

    # Fall back to HTTP call latency
    latency = directive_calls[0].get("latency_ms", 0)
    expected = 200
    distance = max(0.0, float(latency) - expected)
    return float(latency), distance


def c03_unknown_tid_returns_4xx(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """Directive/comment to unknown tid returns 4xx, not 5xx.

    Returns (0, 0.0) if 4xx returned; (1, 1.0) if 5xx or 2xx returned.
    Returns (None, 0.0) if scenario doesn't involve an unknown tid.
    """
    if not scenario.inputs.get("unknown_tid", False):
        return None, 0.0

    for call in obs.http_calls:
        method = call.get("method", "")
        status = call.get("status", 0)
        url = call.get("url", "")
        if method == "POST" and ("/directive" in url or "/comment" in url):
            if 400 <= status < 500:
                return 0, 0.0
            # 5xx or unexpected
            return 1, 1.0

    # No POST call found
    return None, 0.0


def c04_path_traversal_rejected(
    scenario: Scenario, obs: Observation
) -> tuple[Any, float]:
    """GET /api/documents with '..' in filename returns 4xx, no path leak in body.

    Returns (leak_count, float(leak_count)).
    Returns (None, 0.0) if scenario doesn't involve path traversal.
    """
    # Only applicable when scenario has path traversal filename
    path = scenario.inputs.get("path", "")
    if ".." not in path:
        return None, 0.0

    leak_count = 0

    for call in obs.http_calls:
        if call.get("method") == "GET" and ".." in call.get("url", ""):
            status = call.get("status", 0)
            body = call.get("body", "")

            # Must return 4xx
            if not (400 <= status < 500):
                leak_count += 1
                continue

            # Body must not leak absolute paths
            body_str = str(body)
            if _has_path_leak(body_str):
                leak_count += 1

    return leak_count, float(leak_count)


def _has_path_leak(text: str) -> bool:
    """Return True if text contains an absolute path that could be a leak."""
    import re
    # Absolute paths starting with / followed by a real dir name
    return bool(re.search(r"/(?:data|home|opt|etc|var|usr|root|tmp)/[^\s\"']+", text))
