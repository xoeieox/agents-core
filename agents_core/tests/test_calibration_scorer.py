"""Tests for agents_core.calibration.scorer + predictor.

All tests are offline: no AgentWorld endpoint, no GPU.
Fixtures use synthetic corpus data + FakeAgentWorldClient.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from agents_core.calibration.predictor import (
    FakeAgentWorldClient,
    Predictor,
    _SYSTEM_PROMPT_NO_RULES,
    _SYSTEM_PROMPT_WITH_RULES,
    _apply_rules,
    _extract_json,
    build_prompt,
)
from agents_core.calibration.scorer import (
    Aggregator,
    load_transitions,
    render_report_md,
    run_scorer,
    score_counts,
    score_fields,
    score_in_flight,
    score_stasis_duration,
    score_stasis_velocity,
    transition_fidelity_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_s: float = 0.0) -> str:
    from datetime import datetime, timezone, timedelta
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset_s)).isoformat()


def _make_state(
    pending=0, active=0, completed=0, failed=0,
    in_flight=None, capacity=2,
    stasis_duration=0.0, stasis_velocity=0.0,
) -> dict:
    inf = in_flight or []
    util = len(inf) / capacity if capacity > 0 else 0.0
    return {
        "counts": {"pending": pending, "active": active, "completed": completed, "failed": failed},
        "in_flight": inf,
        "workers": {"capacity": capacity, "utilization": round(util, 4)},
        "stasis_duration": stasis_duration,
        "stasis_velocity": stasis_velocity,
    }


def _make_event(event_type: str, job_id: str = "j1", task_type: str = "test", model: str = "m1") -> dict:
    return {
        "event": event_type,
        "id": job_id,
        "task_type": task_type,
        "model": model,
        "priority": None,
        "duration_seconds": None,
        "intention_id": None,
        "submitted_by": None,
        "error": None,
    }


def _make_transition(
    event_type: str,
    state_before: dict,
    state_after: dict,
    consistency: str = "ok",
    ts_offset: float = 0.0,
) -> dict:
    return {
        "ts": _ts(ts_offset),
        "raw_event_ts": _ts(ts_offset),
        "state_before": state_before,
        "event": _make_event(event_type),
        "state_after": state_after,
        "consistency": consistency,
    }


def _write_transitions(path: Path, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# 1. Field-typed scoring: exact-match for discrete deterministic fields
# ---------------------------------------------------------------------------

class TestScoreCounts:
    def test_exact_match_all(self):
        real = pred = _make_state(pending=1, active=2, completed=3, failed=0)
        scores = score_counts(pred, real, "submitted")
        for field in ["pending", "active", "completed", "failed"]:
            assert scores[f"counts.{field}"]["exact"] is True

    def test_mismatch_pending(self):
        real = _make_state(pending=1)
        pred = _make_state(pending=2)
        scores = score_counts(pred, real, "submitted")
        assert scores["counts.pending"]["exact"] is False
        # Active still correct
        assert scores["counts.active"]["exact"] is True

    def test_missing_field_does_not_crash(self):
        real = {"counts": {"pending": 1}}
        pred = {}
        scores = score_counts(pred, real, "submitted")
        # Should return False, not raise
        assert scores["counts.pending"]["exact"] is False


class TestScoreInFlight:
    def test_perfect_match(self):
        job = {"id": "j1", "task_type": "test", "model": "m", "claimed_at": _ts(), "stasis_duration": 1.0}
        real = _make_state(in_flight=[job])
        pred = _make_state(in_flight=[job])
        s = score_in_flight(pred, real)
        assert s["precision"] == 1.0
        assert s["recall"] == 1.0
        assert len(s["true_positive_ids"]) == 1
        assert len(s["false_positive_ids"]) == 0

    def test_no_jobs_both_empty(self):
        s = score_in_flight(_make_state(), _make_state())
        assert s["precision"] == 1.0
        assert s["recall"] == 1.0

    def test_missed_job(self):
        job = {"id": "j1", "task_type": "test", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        real = _make_state(in_flight=[job])
        pred = _make_state(in_flight=[])
        s = score_in_flight(pred, real)
        assert s["precision"] == 0.0  # no TP, real has jobs → precision 0
        assert s["recall"] == 0.0     # missed j1
        assert "j1" in s["false_negative_ids"]

    def test_hallucinated_job(self):
        job = {"id": "j1", "task_type": "test", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        real = _make_state(in_flight=[])
        pred = _make_state(in_flight=[job])
        s = score_in_flight(pred, real)
        assert s["precision"] == 0.0  # hallucination → no TP → precision 0
        assert s["recall"] == 0.0     # nothing real to recall, hallucinations present → 0
        assert "j1" in s["false_positive_ids"]

    def test_job_field_scores_on_tp(self):
        job_real = {"id": "j1", "task_type": "typeA", "model": "mA", "claimed_at": _ts(), "stasis_duration": 0.0}
        job_pred = {"id": "j1", "task_type": "typeB", "model": "mA", "claimed_at": _ts(), "stasis_duration": 0.0}
        s = score_in_flight(_make_state(in_flight=[job_pred]), _make_state(in_flight=[job_real]))
        assert len(s["job_field_scores"]) == 1
        jfs = s["job_field_scores"][0]
        assert jfs["task_type_match"] is False
        assert jfs["model_match"] is True


class TestScoreContinuousFields:
    def test_stasis_duration_zero_exact(self):
        pred = _make_state(stasis_duration=0.0)
        real = _make_state(stasis_duration=0.0)
        s = score_stasis_duration(pred, real)
        assert s["pass"] is True

    def test_stasis_duration_within_tolerance(self):
        pred = _make_state(stasis_duration=10.0)
        real = _make_state(stasis_duration=10.5)
        s = score_stasis_duration(pred, real)
        assert s["pass"] is True  # 0.5/10.5 = 4.8% < 10%

    def test_stasis_duration_exceeds_tolerance(self):
        pred = _make_state(stasis_duration=5.0)
        real = _make_state(stasis_duration=10.0)
        s = score_stasis_duration(pred, real)
        assert s["pass"] is False  # 50% error

    def test_stasis_velocity_direction_match(self):
        pred = _make_state(stasis_velocity=3.0)
        real = _make_state(stasis_velocity=5.0)
        s = score_stasis_velocity(pred, real)
        assert s["direction_match"] is True

    def test_stasis_velocity_direction_mismatch(self):
        pred = _make_state(stasis_velocity=-1.0)
        real = _make_state(stasis_velocity=5.0)
        s = score_stasis_velocity(pred, real)
        assert s["direction_match"] is False

    def test_stasis_velocity_both_zero(self):
        pred = _make_state(stasis_velocity=0.0)
        real = _make_state(stasis_velocity=0.0)
        s = score_stasis_velocity(pred, real)
        assert s["direction_match"] is True


# ---------------------------------------------------------------------------
# 2. Parse failures as distinct zero-valued category (mandate 2)
# ---------------------------------------------------------------------------

class TestParseFailureAsZeroCategory:
    def test_parse_failure_scores_zero(self):
        fidelity = transition_fidelity_score({}, parse_ok=False)
        assert fidelity == 0.0

    def test_parse_failure_counted_not_excluded(self, tmp_path):
        """Aggregate INCLUDES parse failures as fidelity=0, not omitted."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        tr = _make_transition("submitted", state_before, state_after)
        transitions_path = tmp_path / "transitions.jsonl"
        _write_transitions(transitions_path, [tr])

        class _AlwaysFailClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return "NOT JSON AT ALL", "stop"

        predictor = Predictor(client=_AlwaysFailClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        assert summary["total_scored"] == 1
        assert summary["parse_failures"] == 1
        assert summary["parse_failure_rate"] == 1.0
        # Overall fidelity must be 0 (not None or 1.0)
        assert summary["overall_fidelity"] == 0.0

    def test_parse_failure_rate_reported_separately(self, tmp_path):
        """parse_failure_rate is a distinct field in the aggregate."""
        state = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [
            _make_transition("submitted", state, state_after),
            _make_transition("submitted", state, state_after),
        ])

        class _HalfFail:
            _call = 0
            def predict(self, **kwargs) -> tuple[str, str]:
                self._call += 1
                if self._call == 1:
                    return "not json", "stop"
                return json.dumps(_make_state(pending=2)), "stop"

        predictor = Predictor(client=_HalfFail())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        assert summary["parse_failures"] == 1
        assert summary["parse_failure_rate"] == 0.5


# ---------------------------------------------------------------------------
# 3. Consistency flag: dimension not pre-filter (mandate 3)
# ---------------------------------------------------------------------------

class TestConsistencyFlagDimension:
    def test_drift_transitions_are_scored_not_dropped(self, tmp_path):
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions = [
            _make_transition("submitted", state_before, state_after, consistency="ok"),
            _make_transition("submitted", state_before, state_after, consistency="drift"),
            _make_transition("submitted", state_before, state_after, consistency="unknown"),
        ]
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, transitions)

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        # All three scored
        assert summary["total_scored"] == 3
        # All three consistency flags present
        per_flag = summary["per_consistency_flag"]
        assert "ok" in per_flag
        assert "drift" in per_flag
        assert "unknown" in per_flag

    def test_consistency_flag_breakdown_in_report(self, tmp_path):
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [
            _make_transition("submitted", state_before, state_after, consistency="ok"),
            _make_transition("submitted", state_before, state_after, consistency="drift"),
        ])
        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {"samples": 1, "endpoint": "fake", "model": "fake"})
        assert "ok" in md
        assert "drift" in md
        assert "Q2" in md


# ---------------------------------------------------------------------------
# 4. Velocity distribution + threshold gated on noise floor (Q1, mandate 1)
# ---------------------------------------------------------------------------

class TestVelocityThresholdAnalysis:
    def _run_with_velocity(self, tmp_path, sv_before, sv_after, n_samples=1):
        state_before = _make_state(stasis_velocity=sv_before)
        state_after = _make_state(stasis_velocity=sv_after)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])
        predictor = Predictor(client=FakeAgentWorldClient())
        return run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=n_samples,
        )

    def test_perfect_velocity_prediction_is_derivable(self, tmp_path):
        """Fake client predicts velocity=0 in state_after; real is also 0 → clean derivable."""
        summary = self._run_with_velocity(tmp_path, sv_before=0.0, sv_after=0.0)
        analysis = summary["stasis_velocity_analysis"]
        # With zero rel-error, status is threshold-derivable or signal-absent; not unknowable
        assert analysis["status"] != "unknowable-noise-dominated"

    def test_unknowable_when_noise_dominates(self, tmp_path):
        """When mean_err < noise_floor_std * SNR_MIN, report 'unknowable-noise-dominated'.

        The SNR gate operates in rel-error space: mean(rel_errors) / stdev(rel_errors).
        Noise dominates when the spread of rel_errors is large relative to their mean.
        """
        from agents_core.calibration.scorer import NOISE_FLOOR_SNR_MIN

        agg = Aggregator()
        # Two transitions with very different rel_errors: mean=0.5, stdev≈0.636 → SNR≈0.79 < 2.0
        for i, (sv_pred, rel_err) in enumerate([(1.0, 0.01), (-1.0, 0.99)]):
            field_scores = {
                "counts": {},
                "in_flight": {"precision": 1.0, "recall": 1.0, "true_positive_ids": [], "false_positive_ids": [], "false_negative_ids": [], "job_field_scores": []},
                "capacity": {"exact": True},
                "utilization": {"pass": True},
                "stasis_duration": {"pass": True},
                "stasis_velocity": {
                    "direction_match": True,
                    "magnitude_error": abs(rel_err),
                    "rel_error": rel_err,
                    "predicted": sv_pred,
                    "real": 1.0,
                },
            }
            agg.add(
                transition_ref=f"tr-{i}",
                sample_idx=0,
                fidelity=0.9,
                field_scores=field_scores,
                parse_ok=True,
                event_type="submitted",
                consistency_flag="ok",
            )

        nf = agg.noise_floor()
        # Noise floor is now std-dev of rel-errors: stdev([0.01, 0.99]) ≈ 0.693
        assert nf["stasis_velocity_rel_error_std"] > 0.0

        analysis = agg.stasis_velocity_threshold_analysis()
        # SNR = mean([0.01, 0.99]) / stdev([0.01, 0.99]) = 0.5 / 0.693 ≈ 0.72 < 2.0
        assert analysis["status"] == "unknowable-noise-dominated"

    def test_threshold_candidate_emitted_when_clear(self, tmp_path):
        """With consistent predictions far above noise floor, threshold is derivable."""
        agg = Aggregator()
        # Identical predictions (variance=0) but nonzero rel_error > noise floor
        for i in range(5):
            field_scores = {
                "counts": {},
                "in_flight": {"precision": 1.0, "recall": 1.0, "true_positive_ids": [], "false_positive_ids": [], "false_negative_ids": [], "job_field_scores": []},
                "capacity": {"exact": True},
                "utilization": {"pass": True},
                "stasis_duration": {"pass": True},
                "stasis_velocity": {
                    "direction_match": True,
                    "magnitude_error": 5.0,
                    "rel_error": 0.5,  # 50% error = strong signal
                    "predicted": 10.0,
                    "real": 5.0,
                },
            }
            agg.add(
                transition_ref="tr-1",
                sample_idx=i,
                fidelity=0.5,
                field_scores=field_scores,
                parse_ok=True,
                event_type="submitted",
                consistency_flag="ok",
            )

        nf = agg.noise_floor()
        analysis = agg.stasis_velocity_threshold_analysis()
        assert analysis["status"] == "threshold-derivable"
        assert "candidate_threshold_rel_error" in analysis


# ---------------------------------------------------------------------------
# 5. Resolution map labeling (mandate 4)
# ---------------------------------------------------------------------------

class TestResolutionMap:
    def test_discrete_fields_always_threshold_derivable(self, tmp_path):
        """Discrete deterministic fields are always threshold-derivable."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        rm = summary["resolution_map"]
        for field in ["counts.pending", "counts.active", "counts.completed", "counts.failed",
                      "in_flight", "workers.capacity"]:
            assert rm[field] == "threshold-derivable", f"{field} should be threshold-derivable"

    def test_resolution_map_in_report_md(self, tmp_path):
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {"samples": 1, "endpoint": "fake", "model": "fake"})
        assert "Resolution Map" in md
        assert "threshold-derivable" in md


# ---------------------------------------------------------------------------
# 6. Noise floor aggregation (self-consistency from --samples N)
# ---------------------------------------------------------------------------

class TestNoiseFloorAggregation:
    def test_single_sample_has_zero_variance(self, tmp_path):
        """With samples=1 there is no intra-transition variance."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        nf = summary["noise_floor"]
        # With 1 sample per transition, variance is 0 (only one data point)
        assert nf["fidelity_variance"] == 0.0

    def test_multi_sample_noise_floor_present(self, tmp_path):
        """With samples>1 and a stochastic client, noise floor is nonzero."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        call_count = {"n": 0}

        class _StochasticClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                call_count["n"] += 1
                # Alternate between correct and wrong prediction
                if call_count["n"] % 2 == 0:
                    return json.dumps(_make_state(pending=2)), "stop"
                else:
                    return json.dumps(_make_state(pending=99)), "stop"

        predictor = Predictor(client=_StochasticClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=2,
        )
        nf = summary["noise_floor"]
        assert nf["fidelity_variance"] > 0.0


# ---------------------------------------------------------------------------
# 7. Scored-fraction logging (no silent truncation)
# ---------------------------------------------------------------------------

class TestScoredFraction:
    def test_fraction_reported_with_limit(self, tmp_path):
        """With --limit, scored_fraction reflects partial coverage."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [
            _make_transition("submitted", state_before, state_after, ts_offset=float(i))
            for i in range(10)
        ])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
            limit=3,
        )
        assert summary["corpus_total"] == 10
        assert summary["total_scored"] == 3
        assert abs(summary["scored_fraction"] - 0.3) < 0.01

    def test_fraction_reported_with_stride(self, tmp_path):
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [
            _make_transition("submitted", state_before, state_after, ts_offset=float(i))
            for i in range(10)
        ])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
            sample_stride=2,
        )
        assert summary["corpus_total"] == 10
        assert summary["total_scored"] == 5  # every 2nd


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_scores(self, tmp_path):
        """FakeAgentWorldClient is deterministic: two runs produce identical scores."""
        state_before = _make_state(pending=1, active=1, in_flight=[
            {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(), "stasis_duration": 1.0}
        ])
        state_after = _make_state(pending=2, active=1, in_flight=[
            {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        ])
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        def _run():
            predictor = Predictor(client=FakeAgentWorldClient())
            return run_scorer(
                transitions_path=transitions_path,
                output_dir=tmp_path / f"scores_{id(predictor)}",
                predictor=predictor,
                samples=1,
            )

        s1 = _run()
        s2 = _run()
        assert s1["overall_fidelity"] == s2["overall_fidelity"]
        assert s1["parse_failures"] == s2["parse_failures"]


# ---------------------------------------------------------------------------
# 9. Predictor JSON extraction (resilient parsing)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        raw = '{"a": 1}'
        assert _extract_json(raw) == {"a": 1}

    def test_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _extract_json(raw) == {"a": 1}

    def test_prose_with_json(self):
        raw = 'Here is the answer: {"a": 1} as you can see.'
        assert _extract_json(raw) == {"a": 1}

    def test_no_json(self):
        assert _extract_json("sorry I cannot answer") is None

    def test_code_fence_no_lang(self):
        raw = '```\n{"b": 2}\n```'
        assert _extract_json(raw) == {"b": 2}


# ---------------------------------------------------------------------------
# 10. Read-only: scorer never mutates transitions.jsonl
# ---------------------------------------------------------------------------

class TestReadOnly:
    def test_transitions_not_mutated(self, tmp_path):
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        original_mtime = transitions_path.stat().st_mtime
        original_content = transitions_path.read_text()

        predictor = Predictor(client=FakeAgentWorldClient())
        run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )

        assert transitions_path.stat().st_mtime == original_mtime
        assert transitions_path.read_text() == original_content


# ---------------------------------------------------------------------------
# 11. Scores written to output dir (not corpus dir)
# ---------------------------------------------------------------------------

class TestOutputPaths:
    def test_scores_jsonl_written(self, tmp_path):
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        output_dir = tmp_path / "scores"
        predictor = Predictor(client=FakeAgentWorldClient())
        run_scorer(
            transitions_path=transitions_path,
            output_dir=output_dir,
            predictor=predictor,
            samples=1,
        )

        assert (output_dir / "scores.jsonl").exists()
        lines = (output_dir / "scores.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "transition_ref" in record
        assert "predicted_state" in record
        assert "field_scores" in record
        assert "parse_ok" in record
        assert "fidelity" in record
        assert "consistency_flag" in record
        assert "sample_idx" in record

    def test_report_files_written(self, tmp_path):
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        output_dir = tmp_path / "scores"
        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=output_dir,
            predictor=predictor,
            samples=1,
        )

        # report.json and report.md are written by the CLI; check summary structure
        assert "total_scored" in summary
        assert "parse_failures" in summary
        assert "overall_fidelity" in summary
        assert "per_field_fidelity" in summary
        assert "per_event_type" in summary
        assert "per_consistency_flag" in summary
        assert "noise_floor" in summary
        assert "stasis_velocity_analysis" in summary
        assert "consistency_flag_verdict" in summary
        assert "resolution_map" in summary


# ---------------------------------------------------------------------------
# 12. Fake predictor applies exact state-machine rules
# ---------------------------------------------------------------------------

class TestFakePredictor:
    def test_submitted_increments_pending(self):
        sb = _make_state(pending=0)
        ev = _make_event("submitted")
        result = _apply_rules(sb, ev)
        assert result["counts"]["pending"] == 1

    def test_claimed_moves_to_active(self):
        job = {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        sb = _make_state(pending=1, active=0)
        ev = _make_event("claimed", job_id="j1")
        ev["timestamp"] = _ts()
        result = _apply_rules(sb, ev)
        assert result["counts"]["pending"] == 0
        assert result["counts"]["active"] == 1
        assert any(j["id"] == "j1" for j in result["in_flight"])

    def test_completed_removes_from_in_flight(self):
        job = {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        sb = _make_state(active=1, in_flight=[job])
        ev = _make_event("completed", job_id="j1")
        result = _apply_rules(sb, ev)
        assert result["counts"]["active"] == 0
        assert result["counts"]["completed"] == 1
        assert len(result["in_flight"]) == 0

    def test_failed_removes_from_in_flight(self):
        job = {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(), "stasis_duration": 0.0}
        sb = _make_state(active=1, in_flight=[job])
        ev = _make_event("failed", job_id="j1")
        result = _apply_rules(sb, ev)
        assert result["counts"]["active"] == 0
        assert result["counts"]["failed"] == 1
        assert len(result["in_flight"]) == 0

    def test_stasis_after_event_is_zero(self):
        sb = _make_state(pending=0, stasis_duration=10.0, stasis_velocity=5.0)
        ev = _make_event("submitted")
        result = _apply_rules(sb, ev)
        assert result["stasis_duration"] == 0.0
        assert result["stasis_velocity"] == 0.0


# ---------------------------------------------------------------------------
# 13. End-to-end: multi-event run with FakeAgentWorldClient
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_lifecycle_scored(self, tmp_path):
        """submitted → claimed → completed: 3 transitions, all pass with fake client."""
        job = {"id": "j1", "task_type": "t", "model": "m", "claimed_at": _ts(1.0), "stasis_duration": 0.0}
        transitions = [
            _make_transition("submitted", _make_state(pending=0), _make_state(pending=1), ts_offset=0.0),
            _make_transition("claimed", _make_state(pending=1), _make_state(active=1, in_flight=[job]), ts_offset=1.0),
            _make_transition("completed", _make_state(active=1, in_flight=[job]), _make_state(completed=1), ts_offset=2.0),
        ]
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, transitions)

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )

        assert summary["total_scored"] == 3
        assert summary["parse_failures"] == 0
        assert summary["corpus_total"] == 3
        assert summary["scored_fraction"] == 1.0
        # Event types present
        assert set(summary["per_event_type"].keys()) == {"submitted", "claimed", "completed"}

    def test_report_md_rendered(self, tmp_path):
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {"samples": 1, "endpoint": "fake", "model": "fake"})
        assert "AgentWorld Queue-Runner Fidelity Report" in md
        assert "Parse Failures" in md
        assert "Q1" in md
        assert "Q2" in md
        assert "Resolution Map" in md


# ---------------------------------------------------------------------------
# 14. v0.1 features: prompt-mode, D1 resolution-map, D2 counts fix, D4 truncation
# ---------------------------------------------------------------------------

class TestV01PromptMode:
    def test_no_rules_omits_state_machine_section(self):
        """Spec 1: no-rules prompt must not contain the lifecycle rule list."""
        messages = build_prompt({}, {}, prompt_mode="no-rules")
        sys_content = messages[0]["content"]
        assert "submitted  →" not in sys_content
        assert "claimed    →" not in sys_content
        assert "Queue-runner state machine" not in sys_content

    def test_no_rules_retains_role_schema_task(self):
        """Spec 1: no-rules prompt keeps role framing, State schema, and task instruction."""
        messages = build_prompt({}, {}, prompt_mode="no-rules")
        sys_content = messages[0]["content"]
        assert "AgentWorld" in sys_content
        assert "State schema" in sys_content
        assert "state_after" in sys_content

    def test_with_rules_includes_lifecycle_rules(self):
        """Spec 1: with-rules prompt retains the v0 state-machine rule list."""
        messages = build_prompt({}, {}, prompt_mode="with-rules")
        sys_content = messages[0]["content"]
        assert "submitted  →" in sys_content
        assert "claimed    →" in sys_content
        assert "Queue-runner state machine" in sys_content

    def test_default_is_no_rules(self):
        """Spec 1: default prompt_mode is no-rules."""
        messages_default = build_prompt({}, {})
        messages_no_rules = build_prompt({}, {}, prompt_mode="no-rules")
        assert messages_default[0]["content"] == messages_no_rules[0]["content"]

    def test_with_rules_report_carries_banner(self, tmp_path):
        """Spec 1 / mandate 2: with-rules report.md carries the instruction-following banner."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient(), prompt_mode="with-rules")
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {
            "samples": 1, "endpoint": "fake", "model": "fake", "prompt_mode": "with-rules"
        })
        assert "Instruction-following benchmark" in md
        assert "NOT world-model fidelity" in md

    def test_no_rules_report_has_no_banner(self, tmp_path):
        """Spec 1: no-rules report must NOT carry the instruction-following banner."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient(), prompt_mode="no-rules")
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {
            "samples": 1, "endpoint": "fake", "model": "fake", "prompt_mode": "no-rules"
        })
        assert "Instruction-following benchmark" not in md

    def test_prompt_mode_recorded_in_run_args_report(self, tmp_path):
        """Spec 1: prompt_mode appears in the Run section of the report."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        md = render_report_md(summary, {
            "samples": 1, "endpoint": "fake", "model": "fake", "prompt_mode": "no-rules"
        })
        assert "no-rules" in md


class TestV01D2CountsFieldFix:
    """D2: per-field counts fidelity must reflect actual exact-match, not always-0."""

    def test_correct_prediction_yields_100pct_counts(self, tmp_path):
        """D2 regression: FakeAgentWorldClient gives correct predictions → all counts 100%."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)  # submitted increments pending
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        pf = summary["per_field_fidelity"]
        assert pf["counts.pending"] == 1.0, "correct pending count must yield 100%"
        assert pf["counts.active"] == 1.0
        assert pf["counts.completed"] == 1.0
        assert pf["counts.failed"] == 1.0

    def test_wrong_counts_yield_0pct(self, tmp_path):
        """D2 regression: a wrong count prediction must yield 0% for that field."""
        state_before = _make_state(pending=1)
        state_after = _make_state(pending=2)

        class _WrongPendingClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return json.dumps(_make_state(pending=99)), "stop"

        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=_WrongPendingClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        pf = summary["per_field_fidelity"]
        assert pf["counts.pending"] == 0.0, "wrong pending count must yield 0%"


class TestV01D1ResolutionMapDerived:
    """D1: resolution_map must derive every label from measured rates."""

    def test_signal_absent_when_all_counts_wrong(self, tmp_path):
        """D1: a field is signal-absent when the predictor always gets it wrong."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)

        class _AlwaysWrongCountsClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return json.dumps(_make_state(pending=999)), "stop"

        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=_AlwaysWrongCountsClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        rm = summary["resolution_map"]
        assert rm["counts.pending"] == "signal-absent"

    def test_threshold_derivable_when_correct(self, tmp_path):
        """D1: a correctly-predicted discrete field is threshold-derivable."""
        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=FakeAgentWorldClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        rm = summary["resolution_map"]
        assert rm["counts.pending"] == "threshold-derivable"
        assert rm["workers.capacity"] == "threshold-derivable"
        assert rm["in_flight"] == "threshold-derivable"


class TestV01D4TruncationDetection:
    """D4: truncation must be detected as a distinct parse-failure subtype."""

    def test_max_tokens_default_is_4096(self):
        """D4: Predictor default max_tokens >= 4096."""
        p = Predictor(client=FakeAgentWorldClient())
        assert p.max_tokens >= 4096

    def test_finish_reason_length_recorded_as_truncated(self, tmp_path):
        """D4: finish_reason=length → parse_failure_type=truncated in summary."""
        class _TruncatingClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return '{"counts": {"pending": 1', "length"

        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=_TruncatingClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        assert summary["parse_failures"] == 1
        types = summary.get("parse_failure_types", {})
        assert types.get("truncated", 0) == 1, f"expected truncated=1, got {types}"

    def test_malformed_response_not_labeled_truncated(self, tmp_path):
        """D4: malformed JSON with finish_reason=stop is labeled malformed, not truncated."""
        class _MalformedClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return "sorry, cannot answer", "stop"

        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=_MalformedClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        types = summary.get("parse_failure_types", {})
        assert types.get("malformed", 0) == 1
        assert types.get("truncated", 0) == 0

    def test_truncation_still_scores_fidelity_zero(self, tmp_path):
        """D4 / mandate from v0: parse failures (including truncated) score fidelity=0."""
        class _TruncatingClient:
            def predict(self, **kwargs) -> tuple[str, str]:
                return '{"counts":', "length"

        state_before = _make_state(pending=0)
        state_after = _make_state(pending=1)
        transitions_path = tmp_path / "t.jsonl"
        _write_transitions(transitions_path, [_make_transition("submitted", state_before, state_after)])

        predictor = Predictor(client=_TruncatingClient())
        summary = run_scorer(
            transitions_path=transitions_path,
            output_dir=tmp_path / "scores",
            predictor=predictor,
            samples=1,
        )
        assert summary["overall_fidelity"] == 0.0
