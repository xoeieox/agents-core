"""AgentWorld fidelity scorer for queue-runner calibration (agents-core-agentworld-scorer-v0).

CLI:
  python -m agents_core.calibration.scorer \\
    --transitions <path> [--endpoint <url>] [--model agentworld] \\
    [--samples N] [--limit N] [--few-shot K] [--sample-stride M]

Two mandates drive the design:
- Parse failures are a distinct zero-valued category (never excluded, never smoothed).
- The consistency flag is an analysis dimension, never a pre-filter.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any

from agents_core.calibration.predictor import FakeAgentWorldClient, Predictor
from agents_core.room_paths import room_path

log = logging.getLogger("calibration.scorer")

SCORES_DIR_KEY = "calibration.queue_runner"

# Tolerance bands for continuous fields
UTILIZATION_ABS_TOLERANCE = 0.05   # ±5pp is a pass
STASIS_DURATION_REL_TOLERANCE = 0.10  # ±10% relative error is a pass
# Minimum ratio of noise-floor to be considered "resolvable"
NOISE_FLOOR_SNR_MIN = 2.0  # prediction error must be >= 2x self-consistency variance


# ---------------------------------------------------------------------------
# Field-typed scoring helpers
# ---------------------------------------------------------------------------

def score_counts(predicted: dict, real: dict, event_type: str) -> dict:
    """Exact-match each count field and check delta direction."""
    fields = ["pending", "active", "completed", "failed"]
    scores: dict[str, Any] = {}
    for f in fields:
        p_val = predicted.get("counts", {}).get(f, None)
        r_val = real.get("counts", {}).get(f, None)
        if p_val is None or r_val is None:
            scores[f"counts.{f}"] = {"exact": False, "delta_correct": False, "predicted": p_val, "real": r_val}
            continue
        exact = p_val == r_val
        scores[f"counts.{f}"] = {
            "exact": exact,
            "predicted": p_val,
            "real": r_val,
        }
    return scores


def score_in_flight(predicted: dict, real: dict) -> dict:
    """Precision/recall of predicted in_flight IDs + per-job field match."""
    p_jobs = {j["id"]: j for j in predicted.get("in_flight", []) if "id" in j}
    r_jobs = {j["id"]: j for j in real.get("in_flight", []) if "id" in j}

    p_ids = set(p_jobs)
    r_ids = set(r_jobs)

    tp = p_ids & r_ids
    fp = p_ids - r_ids
    fn = r_ids - p_ids

    precision = len(tp) / len(p_ids) if p_ids else (1.0 if not r_ids else 0.0)
    recall = len(tp) / len(r_ids) if r_ids else (1.0 if not p_ids else 0.0)

    # Per-job field match on correctly-predicted IDs
    job_field_scores: list[dict] = []
    for jid in sorted(tp):
        pj = p_jobs[jid]
        rj = r_jobs[jid]
        job_field_scores.append({
            "id": jid,
            "task_type_match": pj.get("task_type") == rj.get("task_type"),
            "model_match": pj.get("model") == rj.get("model"),
        })

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positive_ids": sorted(tp),
        "false_positive_ids": sorted(fp),
        "false_negative_ids": sorted(fn),
        "job_field_scores": job_field_scores,
    }


def score_capacity(predicted: dict, real: dict) -> dict:
    p = predicted.get("workers", {}).get("capacity")
    r = real.get("workers", {}).get("capacity")
    return {"exact": p == r, "predicted": p, "real": r}


def score_utilization(predicted: dict, real: dict) -> dict:
    p = predicted.get("workers", {}).get("utilization")
    r = real.get("workers", {}).get("utilization")
    if p is None or r is None:
        return {"pass": False, "error": None, "predicted": p, "real": r}
    err = abs(float(p) - float(r))
    return {"pass": err <= UTILIZATION_ABS_TOLERANCE, "abs_error": round(err, 4), "predicted": p, "real": r}


def score_stasis_duration(predicted: dict, real: dict) -> dict:
    p = predicted.get("stasis_duration")
    r = real.get("stasis_duration")
    if p is None or r is None:
        return {"pass": False, "rel_error": None, "predicted": p, "real": r}
    p, r = float(p), float(r)
    if r == 0.0:
        # After an event, stasis_duration should be 0; exact match is the only pass
        exact = abs(p) < 0.001
        return {"pass": exact, "rel_error": abs(p) if not exact else 0.0, "predicted": p, "real": r}
    rel_err = abs(p - r) / abs(r)
    return {
        "pass": rel_err <= STASIS_DURATION_REL_TOLERANCE,
        "rel_error": round(rel_err, 4),
        "predicted": p,
        "real": r,
    }


def score_stasis_velocity(predicted: dict, real: dict) -> dict:
    p = predicted.get("stasis_velocity")
    r = real.get("stasis_velocity")
    if p is None or r is None:
        return {
            "direction_match": False,
            "magnitude_error": None,
            "rel_error": None,
            "predicted": p,
            "real": r,
        }
    p, r = float(p), float(r)
    # Direction match: both zero, same sign, or zero+near-zero
    if r == 0.0 and abs(p) < 0.001:
        direction_match = True
    elif r == 0.0:
        direction_match = False
    elif p == 0.0:
        direction_match = False
    else:
        direction_match = (p > 0) == (r > 0)
    mag_err = abs(p - r)
    rel_err = mag_err / abs(r) if r != 0.0 else (0.0 if p == 0.0 else float("inf"))
    return {
        "direction_match": direction_match,
        "magnitude_error": round(mag_err, 4),
        "rel_error": round(rel_err, 4) if math.isfinite(rel_err) else None,
        "predicted": p,
        "real": r,
    }


def score_fields(predicted: dict, real: dict, event_type: str) -> dict:
    """Full field-typed diff of predicted vs real state."""
    return {
        "counts": score_counts(predicted, real, event_type),
        "in_flight": score_in_flight(predicted, real),
        "capacity": score_capacity(predicted, real),
        "utilization": score_utilization(predicted, real),
        "stasis_duration": score_stasis_duration(predicted, real),
        "stasis_velocity": score_stasis_velocity(predicted, real),
    }


def transition_fidelity_score(field_scores: dict, parse_ok: bool) -> float:
    """Overall [0,1] fidelity score for one transition.

    Discrete deterministic fields are weighted highest (pure state-machine).
    Continuous fields contribute less because exact match is unrealistic.
    Parse failures → 0.0 (distinct zero-valued category, mandate 2).
    """
    if not parse_ok:
        return 0.0

    # Discrete deterministic fields (weight 3 total)
    count_fields = ["counts.pending", "counts.active", "counts.completed", "counts.failed"]
    counts_score = field_scores.get("counts", {})
    count_pass = sum(
        1 for f in count_fields if counts_score.get(f, {}).get("exact", False)
    )
    count_frac = count_pass / len(count_fields)

    inf = field_scores.get("in_flight", {})
    precision = inf.get("precision", 0.0)
    recall = inf.get("recall", 0.0)
    in_flight_frac = (precision + recall) / 2.0

    cap = field_scores.get("capacity", {})
    capacity_frac = 1.0 if cap.get("exact", False) else 0.0

    # Continuous fields (weight 1 total)
    util = field_scores.get("utilization", {})
    util_frac = 1.0 if util.get("pass", False) else 0.0

    stasis_d = field_scores.get("stasis_duration", {})
    stasis_d_frac = 1.0 if stasis_d.get("pass", False) else 0.0

    stasis_v = field_scores.get("stasis_velocity", {})
    stasis_v_frac = 1.0 if stasis_v.get("direction_match", False) else 0.0

    # Weighted sum: discrete fields = 3x weight each, continuous = 1x
    total_weight = 3 * 3 + 1 * 3  # 9 + 3 = 12
    weighted = (
        3 * count_frac
        + 3 * in_flight_frac
        + 3 * capacity_frac
        + 1 * util_frac
        + 1 * stasis_d_frac
        + 1 * stasis_v_frac
    )
    return round(weighted / total_weight, 4)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class Aggregator:
    """Accumulates per-transition scores into aggregate statistics."""

    def __init__(self) -> None:
        self.total = 0
        self.parse_failures = 0
        self.by_event: dict[str, list[float]] = {}  # event_type → [fidelity]
        self.by_consistency: dict[str, list[float]] = {}  # consistency_flag → [fidelity]

        # Per-field accumulators
        self.count_exact: dict[str, list[bool]] = {
            f: [] for f in ["pending", "active", "completed", "failed"]
        }
        self.in_flight_precision: list[float] = []
        self.in_flight_recall: list[float] = []
        self.capacity_exact: list[bool] = []
        self.util_pass: list[bool] = []
        self.stasis_d_pass: list[bool] = []
        self.stasis_v_direction: list[bool] = []
        self.stasis_v_rel_errors: list[float] = []  # for noise-floor analysis

        # Self-consistency (multi-sample): per-transition variance per field
        # Stored as lists of per-sample values indexed by (transition_ref, sample_idx)
        # Aggregated into variance across samples per transition, then averaged
        self._self_consistency_fidelity: dict[str, list[float]] = {}  # ref → [fidelity per sample]
        self._self_consistency_stasis_v: dict[str, list[float]] = {}  # ref → [stasis_v per sample]

        self.all_fidelity: list[float] = []

    def add(
        self,
        transition_ref: str,
        sample_idx: int,
        fidelity: float,
        field_scores: dict,
        parse_ok: bool,
        event_type: str,
        consistency_flag: str,
    ) -> None:
        self.total += 1

        if not parse_ok:
            self.parse_failures += 1

        self.all_fidelity.append(fidelity)

        # By event type
        self.by_event.setdefault(event_type, []).append(fidelity)

        # By consistency flag (NEVER pre-filter — mandate 3)
        self.by_consistency.setdefault(consistency_flag, []).append(fidelity)

        if not parse_ok:
            return

        # Count exact-match
        counts = field_scores.get("counts", {})
        for f in ["pending", "active", "completed", "failed"]:
            val = counts.get(f, {}).get("exact", False)
            self.count_exact[f].append(val)

        # in_flight
        inf = field_scores.get("in_flight", {})
        self.in_flight_precision.append(inf.get("precision", 0.0))
        self.in_flight_recall.append(inf.get("recall", 0.0))

        # Capacity
        cap = field_scores.get("capacity", {})
        self.capacity_exact.append(cap.get("exact", False))

        # Utilization
        util = field_scores.get("utilization", {})
        self.util_pass.append(util.get("pass", False))

        # Stasis duration
        sd = field_scores.get("stasis_duration", {})
        self.stasis_d_pass.append(sd.get("pass", False))

        # Stasis velocity
        sv = field_scores.get("stasis_velocity", {})
        self.stasis_v_direction.append(sv.get("direction_match", False))
        rel_err = sv.get("rel_error")
        if rel_err is not None:
            self.stasis_v_rel_errors.append(float(rel_err))

        # Self-consistency accumulators
        self._self_consistency_fidelity.setdefault(transition_ref, []).append(fidelity)
        sv_pred = sv.get("predicted")
        if sv_pred is not None:
            self._self_consistency_stasis_v.setdefault(transition_ref, []).append(float(sv_pred))

    def noise_floor(self) -> dict:
        """Per-field self-consistency (predictor's intrinsic noise floor)."""
        def _variance_of_lists(groups: dict[str, list[float]]) -> float:
            variances = []
            for vals in groups.values():
                if len(vals) >= 2:
                    variances.append(statistics.variance(vals))
            return statistics.mean(variances) if variances else 0.0

        fidelity_var = _variance_of_lists(self._self_consistency_fidelity)
        # Noise floor for stasis_velocity: std-dev of per-transition relative errors.
        # Using rel-error space keeps units compatible with mean_err in the SNR gate.
        stasis_v_rel_std = (
            statistics.stdev(self.stasis_v_rel_errors)
            if len(self.stasis_v_rel_errors) >= 2
            else 0.0
        )
        return {
            "fidelity_variance": round(fidelity_var, 6),
            "stasis_velocity_rel_error_std": round(stasis_v_rel_std, 6),
        }

    def velocity_distribution(self) -> dict:
        """Percentile distribution of stasis_velocity relative errors."""
        errs = sorted(self.stasis_v_rel_errors)
        if not errs:
            return {"n": 0}
        n = len(errs)

        def _pct(p: float) -> float:
            idx = max(0, min(n - 1, int(p * n / 100)))
            return round(errs[idx], 4)

        return {
            "n": n,
            "p10": _pct(10),
            "p25": _pct(25),
            "p50": _pct(50),
            "p75": _pct(75),
            "p90": _pct(90),
            "p95": _pct(95),
            "p99": _pct(99),
            "mean": round(statistics.mean(errs), 4),
        }

    def stasis_velocity_threshold_analysis(self) -> dict:
        """Q1: derive a candidate threshold or report 'unknowable-noise-dominated'."""
        nf = self.noise_floor()
        stasis_v_noise = nf["stasis_velocity_rel_error_std"]
        dist = self.velocity_distribution()

        if dist["n"] == 0:
            return {
                "status": "signal-absent",
                "reason": "no stasis_velocity observations",
                "noise_floor_rel_error_std": stasis_v_noise,
            }

        # Prediction error signal: mean relative error
        mean_err = dist.get("mean", 0.0)

        # Gate: prediction error must exceed noise floor by SNR_MIN (mandate 1).
        # Both mean_err and stasis_v_noise are dimensionless relative errors, so
        # SNR = mean_err / std(rel_errors) is a valid signal-to-noise ratio.
        if stasis_v_noise == 0.0 and mean_err == 0.0:
            # Perfect predictions — threshold derivable but trivially
            return {
                "status": "threshold-derivable",
                "candidate_threshold_rel_error": 0.0,
                "noise_floor_rel_error_std": stasis_v_noise,
                "snr_ratio": float("inf"),
                "note": "perfect predictions (likely fake client)",
            }

        snr = mean_err / stasis_v_noise if stasis_v_noise > 0 else float("inf")
        if snr < NOISE_FLOOR_SNR_MIN:
            return {
                "status": "unknowable-noise-dominated",
                "reason": (
                    f"prediction error (mean_rel_err={mean_err:.4f}) does not clear "
                    f"noise floor (rel_error_std={stasis_v_noise:.6f}, SNR={snr:.2f} < {NOISE_FLOOR_SNR_MIN})"
                ),
                "noise_floor_rel_error_std": stasis_v_noise,
                "snr_ratio": round(snr, 3),
                "distribution": dist,
            }

        # Candidate: p75 relative error as a reasonable operating threshold
        candidate = dist.get("p75", mean_err)
        return {
            "status": "threshold-derivable",
            "candidate_threshold_rel_error": candidate,
            "noise_floor_rel_error_std": stasis_v_noise,
            "snr_ratio": round(snr, 3),
            "distribution": dist,
            "note": (
                "candidate is the p75 rel-error — a starting point. "
                "Operating 'Point of Retreat' is PM-domain (mandate 5a)."
            ),
        }

    def consistency_flag_verdict(self) -> dict:
        """Q2: is the consistency flag signal or noise?"""
        ok_scores = self.by_consistency.get("ok", [])
        drift_scores = self.by_consistency.get("drift", [])
        unknown_scores = self.by_consistency.get("unknown", [])

        ok_mean = statistics.mean(ok_scores) if ok_scores else None
        drift_mean = statistics.mean(drift_scores) if drift_scores else None

        if ok_mean is None or drift_mean is None:
            return {
                "verdict": "insufficient-data",
                "ok_n": len(ok_scores),
                "drift_n": len(drift_scores),
                "unknown_n": len(unknown_scores),
                "ok_mean_fidelity": ok_mean,
                "drift_mean_fidelity": drift_mean,
            }

        delta = ok_mean - drift_mean
        # Heuristic: if drift fidelity is more than 5pp lower than ok, flag has signal
        if delta > 0.05:
            verdict = "flag-carries-signal"
            explanation = (
                f"drift-flagged transitions score {delta:.3f} lower fidelity than ok "
                f"(ok_mean={ok_mean:.3f}, drift_mean={drift_mean:.3f}); "
                "the flag is informative."
            )
        else:
            verdict = "flag-is-noise"
            explanation = (
                f"drift-flagged transitions score no meaningfully lower fidelity "
                f"(ok_mean={ok_mean:.3f}, drift_mean={drift_mean:.3f}, delta={delta:.3f}); "
                "the queue_depth yaml/json quirk likely poisoned it. "
                "Acting on this finding (deprecate vs ignore) is recorder-v1 follow-on."
            )

        return {
            "verdict": verdict,
            "explanation": explanation,
            "ok_n": len(ok_scores),
            "drift_n": len(drift_scores),
            "unknown_n": len(unknown_scores),
            "ok_mean_fidelity": round(ok_mean, 4),
            "drift_mean_fidelity": round(drift_mean, 4),
        }

    def resolution_map(self) -> dict:
        """Mandate 4: per-field noise-floor vs prediction error, labeled."""
        nf = self.noise_floor()
        stasis_v_analysis = self.stasis_velocity_threshold_analysis()

        # Discrete fields: always deterministic, signal always present + derivable
        # Continuous fields: gate on noise floor
        sv_status = stasis_v_analysis.get("status", "signal-absent")
        stasis_v_label = (
            "threshold-derivable"
            if sv_status == "threshold-derivable"
            else ("unknowable-noise-dominated" if sv_status == "unknowable-noise-dominated" else "signal-absent")
        )

        return {
            "counts.pending": "threshold-derivable",
            "counts.active": "threshold-derivable",
            "counts.completed": "threshold-derivable",
            "counts.failed": "threshold-derivable",
            "in_flight": "threshold-derivable",
            "workers.capacity": "threshold-derivable",
            "workers.utilization": "threshold-derivable" if self.util_pass else "signal-absent",
            "stasis_duration": (
                "threshold-derivable"
                if self.stasis_d_pass and len(self.stasis_d_pass) > 0
                else "signal-absent"
            ),
            "stasis_velocity": stasis_v_label,
        }

    def summary(self, scored_count: int, total_corpus: int) -> dict:
        """Build the full aggregate report dict."""
        def _rate(vals: list[bool]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        def _mean(vals: list[float]) -> float | None:
            return round(statistics.mean(vals), 4) if vals else None

        parse_failure_rate = round(self.parse_failures / self.total, 4) if self.total else 0.0
        overall_fidelity = _mean(self.all_fidelity)

        per_field = {
            "counts.pending": _rate(self.count_exact.get("pending", [])),
            "counts.active": _rate(self.count_exact.get("active", [])),
            "counts.completed": _rate(self.count_exact.get("completed", [])),
            "counts.failed": _rate(self.count_exact.get("failed", [])),
            "in_flight.precision": _mean(self.in_flight_precision),
            "in_flight.recall": _mean(self.in_flight_recall),
            "workers.capacity": _rate(self.capacity_exact),
            "workers.utilization": _rate(self.util_pass),
            "stasis_duration": _rate(self.stasis_d_pass),
            "stasis_velocity.direction": _rate(self.stasis_v_direction),
        }

        per_event = {
            ev: {"mean_fidelity": round(statistics.mean(scores), 4), "n": len(scores)}
            for ev, scores in self.by_event.items()
        }

        per_consistency = {
            flag: {"mean_fidelity": round(statistics.mean(scores), 4), "n": len(scores)}
            for flag, scores in self.by_consistency.items()
        }

        return {
            "total_scored": self.total,
            "corpus_total": total_corpus,
            "scored_fraction": round(scored_count / total_corpus, 4) if total_corpus else 0.0,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": parse_failure_rate,
            "overall_fidelity": overall_fidelity,
            "per_field_fidelity": per_field,
            "per_event_type": per_event,
            "per_consistency_flag": per_consistency,
            "noise_floor": self.noise_floor(),
            "stasis_velocity_analysis": self.stasis_velocity_threshold_analysis(),
            "consistency_flag_verdict": self.consistency_flag_verdict(),
            "resolution_map": self.resolution_map(),
        }


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def _pct_bar(rate: float, width: int = 20) -> str:
    filled = int(rate * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {rate:.1%}"


def _safe_pct(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:.1%}"


def render_report_md(agg: dict, run_args: dict) -> str:
    """Render the human-readable report.md (flame-shaped: written for Erah/PM reading)."""
    lines = ["# AgentWorld Queue-Runner Fidelity Report", ""]

    # Run metadata
    lines += [
        "## Run",
        f"- Corpus: {agg['corpus_total']} transitions",
        f"- Scored: {agg['total_scored']} ({agg['scored_fraction']:.1%} of corpus)",
        f"- Samples per transition: {run_args.get('samples', 1)}",
        f"- Endpoint: {run_args.get('endpoint', 'n/a')}",
        f"- Model: {run_args.get('model', 'n/a')}",
        "",
    ]

    # Parse failure rate
    pfr = agg["parse_failure_rate"]
    lines += [
        "## Parse Failures",
        f"- Rate: {pfr:.1%} ({agg['parse_failures']} / {agg['total_scored']})",
        "- Parse failures count as fidelity=0 in all aggregates (distinct zero-valued category).",
        "",
    ]

    # Overall fidelity
    of = agg.get("overall_fidelity")
    lines += [
        "## Overall Fidelity",
        f"- Score: {_safe_pct(of)} (weighted: discrete fields 3x, continuous 1x)",
        "",
    ]

    # Per-field fidelity
    lines += ["## Per-Field Fidelity", ""]
    pf = agg.get("per_field_fidelity", {})
    field_order = [
        ("counts.pending", "Count: pending (exact)"),
        ("counts.active", "Count: active (exact)"),
        ("counts.completed", "Count: completed (exact)"),
        ("counts.failed", "Count: failed (exact)"),
        ("in_flight.precision", "in_flight precision"),
        ("in_flight.recall", "in_flight recall"),
        ("workers.capacity", "workers.capacity (exact)"),
        ("workers.utilization", "workers.utilization (±5pp)"),
        ("stasis_duration", "stasis_duration (±10% rel)"),
        ("stasis_velocity.direction", "stasis_velocity direction"),
    ]
    for key, label in field_order:
        val = pf.get(key)
        bar = _pct_bar(val or 0.0) if val is not None else "n/a"
        lines.append(f"- **{label}**: {bar}")
    lines.append("")

    # Per event type
    lines += ["## Fidelity by Event Type", ""]
    for ev, data in sorted(agg.get("per_event_type", {}).items()):
        lines.append(f"- **{ev}**: {_safe_pct(data['mean_fidelity'])} (n={data['n']})")
    lines.append("")

    # Consistency flag breakdown (Q2)
    lines += ["## Q2 - Consistency Flag Analysis", ""]
    cfv = agg.get("consistency_flag_verdict", {})
    for flag, data in sorted(agg.get("per_consistency_flag", {}).items()):
        lines.append(f"- **{flag}**: {_safe_pct(data['mean_fidelity'])} (n={data['n']})")
    lines.append("")
    lines += [
        f"**Verdict:** {cfv.get('verdict', 'n/a')}",
        "",
        cfv.get("explanation", ""),
        "",
    ]

    # Noise floor (self-consistency)
    nf = agg.get("noise_floor", {})
    lines += [
        "## Noise Floor (Predictor Self-Consistency)",
        f"- Fidelity variance across samples: {nf.get('fidelity_variance', 'n/a')}",
        f"- stasis_velocity rel-error std across samples: {nf.get('stasis_velocity_rel_error_std', 'n/a')}",
        "",
    ]

    # Q1 - stasis_velocity threshold
    sva = agg.get("stasis_velocity_analysis", {})
    lines += ["## Q1 - stasis_velocity Threshold Analysis", ""]
    sv_status = sva.get("status", "n/a")
    lines.append(f"**Status:** {sv_status}")
    lines.append("")
    if sv_status == "unknowable-noise-dominated":
        lines += [
            f"> {sva.get('reason', '')}",
            "",
            "This is a first-class finding, not a failure.",
            "The model does not have resolution to derive a meaningful threshold.",
            "",
        ]
    elif sv_status == "threshold-derivable":
        lines += [
            f"- Candidate threshold (p75 rel-error): {sva.get('candidate_threshold_rel_error', 'n/a')}",
            f"- SNR ratio: {sva.get('snr_ratio', 'n/a')}",
            f"- Noise floor (rel-error std): {sva.get('noise_floor_rel_error_std', 'n/a')}",
            "",
            f"_{sva.get('note', '')}_",
            "",
        ]
        dist = sva.get("distribution", {})
        if dist.get("n", 0) > 0:
            lines += [
                "Relative-error distribution:",
                f"- p10={dist.get('p10')} p25={dist.get('p25')} p50={dist.get('p50')}",
                f"  p75={dist.get('p75')} p90={dist.get('p90')} p95={dist.get('p95')} p99={dist.get('p99')}",
                f"  mean={dist.get('mean')} n={dist.get('n')}",
                "",
            ]
    else:
        lines.append(f"- Reason: {sva.get('reason', 'no data')}")
        lines.append("")

    # Resolution map (mandate 4)
    rm = agg.get("resolution_map", {})
    lines += ["## Resolution Map (per-field)", ""]
    lines.append("| Field | Status |")
    lines.append("|---|---|")
    for field, label in sorted(rm.items()):
        lines.append(f"| {field} | {label} |")
    lines.append("")
    lines += [
        "Labels:",
        "- `threshold-derivable`: prediction error clears noise floor; a threshold can be stated.",
        "- `unknowable-noise-dominated`: signal lost to predictor variance; threshold not derivable.",
        "- `signal-absent`: no observations or consistent zero-error.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def load_transitions(
    transitions_path: Path,
    limit: int | None,
    sample_stride: int,
    skip_tail: int = 0,
) -> tuple[list[dict], int]:
    """Load transitions.jsonl, applying stride and limit.

    skip_tail removes the last N records before stride/limit, used to exclude
    few-shot examples that were sourced from the corpus tail.

    Returns (sampled_transitions, total_corpus_size).
    """
    all_records: list[dict] = []
    try:
        with open(transitions_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("skipping malformed line in transitions.jsonl")
    except OSError as exc:
        log.error("cannot read transitions.jsonl: %s", exc)
        raise

    total = len(all_records)
    if skip_tail > 0 and len(all_records) > skip_tail:
        all_records = all_records[:-skip_tail]
    # Apply stride
    sampled = all_records[::sample_stride] if sample_stride > 1 else all_records
    # Apply limit
    if limit is not None:
        sampled = sampled[:limit]
    return sampled, total


def run_scorer(
    transitions_path: Path,
    output_dir: Path,
    predictor: Predictor,
    samples: int = 1,
    limit: int | None = None,
    sample_stride: int = 1,
    few_shot_examples: list[dict] | None = None,
    few_shot_skip_tail: int = 0,
) -> dict:
    """Core scoring loop. Returns the aggregate summary dict.

    Writes scores.jsonl under output_dir.
    Never mutates transitions_path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "scores.jsonl"

    transitions, total_corpus = load_transitions(transitions_path, limit, sample_stride, skip_tail=few_shot_skip_tail)
    n_scored = len(transitions)
    log.info(
        "scoring %d transitions (stride=%d, limit=%s) out of %d corpus total (%.1f%%)",
        n_scored, sample_stride, limit, total_corpus,
        100.0 * n_scored / total_corpus if total_corpus else 0.0,
    )

    agg = Aggregator()

    with open(scores_path, "w") as scores_fh:
        for i, tr in enumerate(transitions):
            ref = tr.get("ts", str(i))
            state_before = tr.get("state_before", {})
            event = tr.get("event", {})
            real_state_after = tr.get("state_after", {})
            consistency_flag = tr.get("consistency", "unknown")
            event_type = event.get("event", "unknown")

            for sample_idx in range(samples):
                result = predictor.predict(state_before, event, few_shot_examples)
                parse_ok = result["parse_ok"]
                predicted_state = result.get("predicted_state") or {}

                if parse_ok:
                    field_scores = score_fields(predicted_state, real_state_after, event_type)
                else:
                    field_scores = {}

                fidelity = transition_fidelity_score(field_scores, parse_ok)

                agg.add(
                    transition_ref=ref,
                    sample_idx=sample_idx,
                    fidelity=fidelity,
                    field_scores=field_scores,
                    parse_ok=parse_ok,
                    event_type=event_type,
                    consistency_flag=consistency_flag,
                )

                record = {
                    "transition_ref": ref,
                    "sample_idx": sample_idx,
                    "predicted_state": predicted_state if parse_ok else None,
                    "field_scores": field_scores,
                    "fidelity": fidelity,
                    "parse_ok": parse_ok,
                    "consistency_flag": consistency_flag,
                    "event_type": event_type,
                }
                scores_fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    summary = agg.summary(scored_count=n_scored, total_corpus=total_corpus)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="AgentWorld queue-runner fidelity scorer",
    )
    parser.add_argument(
        "--transitions", type=Path,
        default=room_path(SCORES_DIR_KEY).parent / "transitions.jsonl",
        help="Path to transitions.jsonl corpus (default: %(default)s)",
    )
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8090/v1",
        help="AgentWorld OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--model", default="agentworld",
        help="Model name for AgentWorld (default: %(default)s)",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="Predictions per transition for self-consistency noise-floor (default: %(default)s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max transitions to score (default: all)",
    )
    parser.add_argument(
        "--few-shot", type=int, default=0, dest="few_shot",
        help="K-shot examples prepended to each prompt (default: 0)",
    )
    parser.add_argument(
        "--sample-stride", type=int, default=1, dest="sample_stride",
        help="Stride over corpus (every Nth transition; default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=room_path(SCORES_DIR_KEY) / "scores",
        help="Output directory for scores.jsonl, report.md, report.json (default: %(default)s)",
    )
    parser.add_argument(
        "--fake", action="store_true",
        help="Use the deterministic fake predictor (offline mode — no AgentWorld endpoint needed)",
    )
    args = parser.parse_args(argv)

    endpoint = os.environ.get("AGENTWORLD_ENDPOINT", args.endpoint)

    if args.fake:
        client = FakeAgentWorldClient()
        predictor = Predictor(client=client, model=args.model, endpoint=endpoint)
        log.info("using deterministic fake predictor (offline mode)")
    else:
        predictor = Predictor(model=args.model, endpoint=endpoint)
        log.info("using HTTP AgentWorld client at %s", endpoint)

    # Load few-shot examples from the corpus TAIL to avoid contaminating the scored set.
    # The tail items are then excluded from scoring via few_shot_skip_tail.
    few_shot_examples: list[dict] | None = None
    few_shot_skip_tail = 0
    if args.few_shot > 0:
        all_transitions, _ = load_transitions(args.transitions, limit=None, sample_stride=1)
        few_shot_examples = all_transitions[-args.few_shot:] if len(all_transitions) >= args.few_shot else all_transitions
        few_shot_skip_tail = len(few_shot_examples)
        log.info("using %d few-shot examples from corpus tail (last %d records excluded from scoring)",
                 len(few_shot_examples), few_shot_skip_tail)

    run_args = {
        "samples": args.samples,
        "endpoint": endpoint,
        "model": args.model,
        "few_shot": args.few_shot,
        "limit": args.limit,
        "sample_stride": args.sample_stride,
    }

    summary = run_scorer(
        transitions_path=args.transitions,
        output_dir=args.output_dir,
        predictor=predictor,
        samples=args.samples,
        limit=args.limit,
        sample_stride=args.sample_stride,
        few_shot_examples=few_shot_examples,
        few_shot_skip_tail=few_shot_skip_tail,
    )

    # Write report.json
    report_json_path = args.output_dir / "report.json"
    report_json_path.write_text(json.dumps({"run_args": run_args, **summary}, indent=2) + "\n")
    log.info("wrote report.json to %s", report_json_path)

    # Write report.md
    report_md_path = args.output_dir / "report.md"
    report_md_path.write_text(render_report_md(summary, run_args))
    log.info("wrote report.md to %s", report_md_path)

    print(f"Scored {summary['total_scored']} transitions")
    print(f"  corpus fraction: {summary['scored_fraction']:.1%}")
    print(f"  parse failures: {summary['parse_failures']} ({summary['parse_failure_rate']:.1%})")
    print(f"  overall fidelity: {summary.get('overall_fidelity', 'n/a')}")
    print(f"  stasis_velocity: {summary.get('stasis_velocity_analysis', {}).get('status', 'n/a')}")
    print(f"  Q2 verdict: {summary.get('consistency_flag_verdict', {}).get('verdict', 'n/a')}")
    print(f"  output: {args.output_dir}")


if __name__ == "__main__":
    main()
