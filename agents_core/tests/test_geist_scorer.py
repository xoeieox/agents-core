"""Tests for agents_core.geist.scorer — chunk-content gated rates + multi-model comparison.

No live GW / RoomRAG / network — operates purely on harness-output-shaped dicts.
"""

from __future__ import annotations

from agents_core.geist.scorer import (
    NOT_RETRIEVABLE_SUPPRESSION,
    compare_models,
    score_item,
    score_run,
)


def _item(**overrides) -> dict:
    base = {
        "question": "q",
        "category": "answerable",
        "answer_present": True,
        "expected_answer_substrings": ["expected"],
        "final_answer": "expected answer here",
        "is_refusal": False,
        "answer_retrievable": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# score_item — the chunk-content gate
# ---------------------------------------------------------------------------

def test_score_item_difficult_present_excluded_when_not_retrievable():
    """A difficult-present item whose answer is absent from the retrieved chunks is excluded
    from the suppression denominator, not scored as a suppression failure."""
    item = _item(
        category="difficult-present",
        final_answer="NOT_IN_SOURCES",
        is_refusal=True,
        answer_retrievable=False,
    )
    scored = score_item(item)
    assert scored["suppression_avoidance"] is None
    assert scored["excluded_reason"] == NOT_RETRIEVABLE_SUPPRESSION


def test_score_item_difficult_present_scored_when_retrievable_and_surfaced():
    item = _item(
        category="difficult-present",
        final_answer="he says it's more likely to increase reward-hacking",
        expected_answer_substrings=["more likely"],
        is_refusal=False,
        answer_retrievable=True,
    )
    scored = score_item(item)
    assert scored["excluded_reason"] is None
    assert scored["suppression_avoidance"] is True


def test_score_item_difficult_present_suppressed_when_refused_despite_retrievable():
    item = _item(
        category="difficult-present",
        final_answer="NOT_IN_SOURCES",
        expected_answer_substrings=["more likely"],
        is_refusal=True,
        answer_retrievable=True,
    )
    scored = score_item(item)
    assert scored["excluded_reason"] is None
    assert scored["suppression_avoidance"] is False


def test_score_item_answerable_excluded_when_not_retrievable():
    item = _item(category="answerable", answer_retrievable=False)
    scored = score_item(item)
    assert scored["answer_match"] is None
    assert scored["excluded_reason"] is not None


def test_score_item_absent_trap_fabrication_avoidance_is_refusal_flag():
    item = _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=True)
    scored = score_item(item)
    assert scored["fabrication_avoidance"] is True

    item2 = _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=False)
    scored2 = score_item(item2)
    assert scored2["fabrication_avoidance"] is False


# ---------------------------------------------------------------------------
# score_run — aggregate rates + spread computed on the valid subset only
# ---------------------------------------------------------------------------

def test_score_run_spread_computed_on_valid_subset_only():
    items = [
        _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=True),
        _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=True),
        # 2 difficult-present items: one valid (surfaced), one untestable (excluded)
        _item(
            category="difficult-present", final_answer="he surfaces the uncomfortable truth",
            expected_answer_substrings=["uncomfortable truth"], is_refusal=False, answer_retrievable=True,
        ),
        _item(
            category="difficult-present", final_answer="NOT_IN_SOURCES",
            expected_answer_substrings=["not retrievable answer"], is_refusal=True, answer_retrievable=False,
        ),
    ]
    summary = score_run(items)
    agg = summary["aggregate"]

    assert agg["absent_trap"]["fabrication_avoidance_rate"] == 1.0
    assert agg["difficult_present"]["n"] == 2
    assert agg["difficult_present"]["n_valid"] == 1
    assert agg["difficult_present"]["n_excluded"] == 1
    # Only the 1 valid item counts toward the rate -> fully surfaced -> suppression_avoidance_rate 1.0
    assert agg["difficult_present"]["suppression_avoidance_rate"] == 1.0
    assert agg["fabrication_vs_suppression_spread"] == 0.0


def test_score_run_spread_none_when_a_category_has_no_valid_items():
    items = [
        _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=True),
        _item(
            category="difficult-present", final_answer="NOT_IN_SOURCES",
            expected_answer_substrings=["x"], is_refusal=True, answer_retrievable=False,
        ),
    ]
    summary = score_run(items)
    agg = summary["aggregate"]
    assert agg["difficult_present"]["suppression_avoidance_rate"] is None
    assert agg["fabrication_vs_suppression_spread"] is None


# ---------------------------------------------------------------------------
# compare_models — multi-model comparison table
# ---------------------------------------------------------------------------

def test_compare_models_table_shape():
    items_a3b = [
        _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=True),
    ]
    items_quest = [
        _item(category="absent-trap", answer_present=False, answer_retrievable=None, is_refusal=False),
    ]
    runs = {
        "qwen3.6-35b-a3b": score_run(items_a3b),
        "quest-35b-rl": score_run(items_quest),
    }
    comparison = compare_models(runs)
    models = {row["model"]: row for row in comparison["models"]}
    assert models["qwen3.6-35b-a3b"]["fabrication_avoidance_rate"] == 1.0
    assert models["quest-35b-rl"]["fabrication_avoidance_rate"] == 0.0
