"""agents_core.friction_test.critique — Dissonance Engine.

Invariants are hypotheses, not propositions. Observations are continuous
signals. Reports surface gaps for human judgment.

Classification heuristic (v0 — single-shot, no historical patterns):
  - inapplicable: observed is None (preconditions not reached)
  - held: distance <= tolerance
  - dissonant_model_likely: dissonant + provisional
  - dissonant_system_likely: dissonant + non-provisional
"""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml

from .observe import Observation
from .scenario import Scenario

logger = logging.getLogger(__name__)

QWEN_ENDPOINT = "http://203.0.113.12:8081/v1/chat/completions"


@dataclass
class Invariant:
    """One invariant — a hypothesis about target behavior."""

    invariant_id: str
    description: str
    dimension: str
    expected: Any
    tolerance: Any
    proposed_by: str  # "Erah" | "qwen" | "council"
    provisional: bool
    check: Callable[[Scenario, Observation], tuple[Any, float]] | None = None
    qwen_check: str | None = None


@dataclass
class InvariantResult:
    """Result of running one invariant against one observation."""

    scenario_id: str
    invariant_id: str
    dimension: str
    expected: Any
    observed: Any
    distance: float
    tolerance: Any
    status: str  # "held" | "dissonant" | "inapplicable"
    classification: str  # "held" | "dissonant_model_likely" | "dissonant_system_likely" | "inapplicable"
    proposed_by: str
    provisional: bool
    justification: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "invariant_id": self.invariant_id,
            "dimension": self.dimension,
            "expected": self.expected,
            "observed": self.observed,
            "distance": self.distance,
            "tolerance": self.tolerance,
            "status": self.status,
            "classification": self.classification,
            "proposed_by": self.proposed_by,
            "provisional": self.provisional,
            "justification": self.justification,
            "evidence_refs": self.evidence_refs,
        }


def _invariants_path(target: str) -> Path:
    here = Path(__file__).parent
    return here / "invariants" / f"{target}.yaml"


def load_declared_invariants(target: str) -> list[Invariant]:
    """Load invariants from ``invariants/<target>.yaml`` and resolve check callables."""
    inv_path = _invariants_path(target)
    if not inv_path.exists():
        raise FileNotFoundError(f"No invariants file for target '{target}': {inv_path}")

    with inv_path.open() as fh:
        data = yaml.safe_load(fh)

    result: list[Invariant] = []
    for item in data.get("invariants", []):
        check_fn = None
        check_dotpath = item.get("check")
        if check_dotpath:
            check_fn = _import_check(check_dotpath)

        result.append(Invariant(
            invariant_id=item["id"],
            description=item.get("description", ""),
            dimension=item["dimension"],
            expected=item["expected"],
            tolerance=item["tolerance"],
            proposed_by=item.get("proposed_by", "Erah"),
            provisional=bool(item.get("provisional", False)),
            check=check_fn,
            qwen_check=item.get("qwen_check"),
        ))
    return result


def _import_check(dotpath: str) -> Callable[[Scenario, Observation], tuple[Any, float]]:
    """Import a dotted-path check function."""
    parts = dotpath.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Cannot parse check path: {dotpath!r}")
    module_path, func_name = parts
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    return fn


def infer_invariants(
    observations: list[Observation],
    qwen_endpoint: str = QWEN_ENDPOINT,
) -> list[Invariant]:
    """Single Qwen pass to surface implicit invariants from observations.

    Returns [] (with warning logged) if the endpoint is unreachable.
    Each returned Invariant has check=None, proposed_by="qwen", provisional=True.
    """
    if not observations:
        return []

    obs_summary = [
        {
            "scenario_id": o.scenario_id,
            "http_calls": [
                {"method": c.get("method"), "url": c.get("url"),
                 "status": c.get("status"), "latency_ms": c.get("latency_ms")}
                for c in o.http_calls
            ],
            "log_appends": len(o.log_appends),
            "sse_events": len(o.sse_events),
            "errors": len(o.errors),
        }
        for o in observations
    ]

    prompt = (
        "You are observing a set of agent-driven test observations. "
        "Surface 2-5 implicit invariants you notice from the data below. "
        "For each, provide: id (short slug), description, dimension (latency_ms|row_count|presence|etc), "
        "expected (value or condition), tolerance (numeric or 0), and a qwen_check prompt template. "
        "Return a JSON array of objects with keys: id, description, dimension, expected, tolerance, qwen_check. "
        "Do not include invariants already obvious from the test setup. Focus on latency patterns, "
        "error rates, and structural consistency.\n\n"
        f"Observations:\n{json.dumps(obs_summary, indent=2)}"
    )

    try:
        resp = httpx.post(
            qwen_endpoint,
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Extract JSON array from response
        raw_list = _extract_json_list(content)
        if raw_list is None:
            logger.warning("Qwen infer_invariants: could not parse JSON from response")
            return []

        result: list[Invariant] = []
        for i, item in enumerate(raw_list):
            result.append(Invariant(
                invariant_id=item.get("id", f"inferred_{i+1:02d}"),
                description=item.get("description", ""),
                dimension=item.get("dimension", "unknown"),
                expected=item.get("expected", None),
                tolerance=item.get("tolerance", 0),
                proposed_by="qwen",
                provisional=True,
                check=None,
                qwen_check=item.get("qwen_check"),
            ))
        return result

    except Exception as exc:
        logger.warning("Qwen unreachable; skipped inferred-invariants pass: %s", exc)
        return []


def _extract_json_list(text: str) -> list | None:
    """Extract the first JSON array from text."""
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _classify(
    invariant: Invariant,
    observed: Any,
    distance: float,
) -> tuple[str, str]:
    """Return (status, classification) from observed + distance."""
    if observed is None:
        return "inapplicable", "inapplicable"

    try:
        tol = float(invariant.tolerance)
    except (TypeError, ValueError):
        tol = 0.0

    if distance <= tol:
        return "held", "held"

    if invariant.provisional:
        return "dissonant", "dissonant_model_likely"
    return "dissonant", "dissonant_system_likely"


def _run_qwen_check(
    inv: Invariant,
    scenario: Scenario,
    obs: Observation,
    observed: Any,
    qwen_endpoint: str,
) -> tuple[str, float | None]:
    """Run a Qwen check prompt for an invariant.

    Returns (justification, distance_estimate).
    distance_estimate is None when Qwen did not return a parseable numeric estimate.
    Returns ("qwen_check unreachable: ...", None) on network failure.
    """
    if not inv.qwen_check:
        return "", None
    prompt = (
        f"{inv.qwen_check}\n\n"
        f"Scenario: {json.dumps(scenario.to_dict())}\n"
        f"Observation summary: http_calls={len(obs.http_calls)}, "
        f"log_appends={len(obs.log_appends)}, errors={len(obs.errors)}\n"
        f"Expected: {inv.expected}\nObserved: {observed}\n\n"
        "Respond with a JSON object with exactly these keys:\n"
        "  verdict: \"held\" | \"dissonant\"\n"
        "  distance_estimate: <non-negative number, 0 if held>\n"
        "  justification: <one or two sentence explanation>\n"
        "Return only the JSON object, no markdown fences."
    )
    try:
        resp = httpx.post(
            qwen_endpoint,
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300,
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _extract_json_object(raw)
        if parsed is not None:
            justification = str(parsed.get("justification", raw))
            try:
                distance_estimate = float(parsed["distance_estimate"])
            except (KeyError, TypeError, ValueError):
                distance_estimate = None
            return justification, distance_estimate
        # Fallback: treat full text as justification, no distance
        return raw, None
    except Exception as exc:
        return f"qwen_check unreachable: {exc}", None


def _extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from text."""
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def critique(
    scenarios: list[Scenario],
    observations: list[Observation],
    invariants: list[Invariant],
    qwen_endpoint: str = QWEN_ENDPOINT,
) -> list[InvariantResult]:
    """Run every (invariant, observation) pair and return InvariantResult list."""
    obs_by_sid = {o.scenario_id: o for o in observations}
    sc_by_sid = {s.scenario_id: s for s in scenarios}

    results: list[InvariantResult] = []
    for inv in invariants:
        for scenario in scenarios:
            obs = obs_by_sid.get(scenario.scenario_id)
            if obs is None:
                continue

            observed: Any = None
            distance: float = 0.0
            evidence_refs: list[str] = []

            if inv.check is not None:
                try:
                    observed, distance = inv.check(scenario, obs)
                except Exception as exc:
                    results.append(InvariantResult(
                        scenario_id=scenario.scenario_id,
                        invariant_id=inv.invariant_id,
                        dimension=inv.dimension,
                        expected=inv.expected,
                        observed=None,
                        distance=0.0,
                        tolerance=inv.tolerance,
                        status="inapplicable",
                        classification="inapplicable",
                        proposed_by=inv.proposed_by,
                        provisional=inv.provisional,
                        justification=f"check() raised: {exc}",
                        evidence_refs=[],
                    ))
                    continue
            elif inv.qwen_check:
                # Qwen-only invariant (inferred); run qwen_check
                qwen_justification, qwen_distance = _run_qwen_check(
                    inv, scenario, obs, observed, qwen_endpoint
                )
                if "unreachable" in qwen_justification:
                    results.append(InvariantResult(
                        scenario_id=scenario.scenario_id,
                        invariant_id=inv.invariant_id,
                        dimension=inv.dimension,
                        expected=inv.expected,
                        observed=None,
                        distance=0.0,
                        tolerance=inv.tolerance,
                        status="inapplicable",
                        classification="inapplicable",
                        proposed_by=inv.proposed_by,
                        provisional=inv.provisional,
                        justification=qwen_justification,
                        evidence_refs=[],
                    ))
                    continue
                # Use Qwen's distance_estimate if provided; fall back to 0.0 only
                # when Qwen explicitly said "held" or gave no numeric estimate.
                observed = "qwen_evaluated"
                distance = qwen_distance if qwen_distance is not None else 0.0
                status, classification = _classify(inv, observed, distance)
                results.append(InvariantResult(
                    scenario_id=scenario.scenario_id,
                    invariant_id=inv.invariant_id,
                    dimension=inv.dimension,
                    expected=inv.expected,
                    observed=observed,
                    distance=distance,
                    tolerance=inv.tolerance,
                    status=status,
                    classification=classification,
                    proposed_by=inv.proposed_by,
                    provisional=inv.provisional,
                    justification=qwen_justification,
                    evidence_refs=evidence_refs,
                ))
                continue
            else:
                # No check available at all
                results.append(InvariantResult(
                    scenario_id=scenario.scenario_id,
                    invariant_id=inv.invariant_id,
                    dimension=inv.dimension,
                    expected=inv.expected,
                    observed=None,
                    distance=0.0,
                    tolerance=inv.tolerance,
                    status="inapplicable",
                    classification="inapplicable",
                    proposed_by=inv.proposed_by,
                    provisional=inv.provisional,
                    justification="No check function and no qwen_check.",
                    evidence_refs=[],
                ))
                continue

            status, classification = _classify(inv, observed, distance)

            # Build justification
            if status == "inapplicable":
                justification = "Scenario did not reach the invariant's preconditions."
            elif status == "held":
                justification = (
                    f"Within tolerance (distance={distance} ≤ tolerance={inv.tolerance})."
                )
            else:
                if inv.provisional:
                    justification = (
                        f"Distance {distance} exceeds tolerance {inv.tolerance}. "
                        "Invariant is provisional — this may be a wrong threshold rather than "
                        "a system regression. Recommend review and either widen tolerance, "
                        "mark non-provisional, or open a system investigation."
                    )
                else:
                    justification = (
                        f"Distance {distance} exceeds tolerance {inv.tolerance}. "
                        "Non-provisional invariant — the system is the more likely source of the gap."
                    )

            # Optional Qwen secondary pass for qwen_check
            if inv.qwen_check and status != "inapplicable":
                extra, _ = _run_qwen_check(inv, scenario, obs, observed, qwen_endpoint)
                if extra:
                    justification = f"{justification} Qwen: {extra}"

            results.append(InvariantResult(
                scenario_id=scenario.scenario_id,
                invariant_id=inv.invariant_id,
                dimension=inv.dimension,
                expected=inv.expected,
                observed=observed,
                distance=distance,
                tolerance=inv.tolerance,
                status=status,
                classification=classification,
                proposed_by=inv.proposed_by,
                provisional=inv.provisional,
                justification=justification,
                evidence_refs=evidence_refs,
            ))

    return results
