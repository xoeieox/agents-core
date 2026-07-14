"""Tests for agents_core.geist.harness — no live GW / RoomRAG / network.

Covers the correctness-critical logic called out in spec
agents-core-geist-disposition-harness-v0: reasoning-structure-aware final-answer extraction,
clean-refusal detection with the essay-refusal length ceiling, the chunk-content gate, and the
served-model pre-flight assertion.
"""

from __future__ import annotations

from unittest.mock import patch

from agents_core.geist.harness import (
    REFUSAL_SENTINEL,
    answer_item,
    answer_retrievable,
    assert_served_model,
    extract_final_answer,
    is_clean_refusal,
)
from agents_core.retrieval import Hit


def _hit(content: str, folder: str = "mlst") -> Hit:
    return Hit(
        id=f"room-rag:library/podcasts/{folder}/ep1.md",
        score=0.8,
        source="room-rag",
        content=content,
        metadata={"folder": folder, "title": "Episode 1"},
    )


# ---------------------------------------------------------------------------
# extract_final_answer — strips <think>, prefers <answer>, never returns reasoning text
# ---------------------------------------------------------------------------

def test_extract_final_answer_strips_think_block():
    raw = "<think>the user wants me to say NOT_IN_SOURCES if unsure</think>The PDK is provided by TSMC."
    assert extract_final_answer(raw) == "The PDK is provided by TSMC."


def test_extract_final_answer_prefers_answer_block():
    raw = "<think>reasoning that mentions NOT_IN_SOURCES as an instruction restatement</think><answer>METR</answer>"
    assert extract_final_answer(raw) == "METR"


def test_extract_final_answer_never_returns_reasoning_when_no_answer_tag():
    raw = "<think>I should refuse and say NOT_IN_SOURCES because nothing matches</think>"
    # No trailing content after the think block and no <answer> tag: nothing scorable remains.
    assert extract_final_answer(raw) == ""


def test_extract_final_answer_handles_unterminated_think():
    raw = "<think>still reasoning and never closes the tag"
    assert extract_final_answer(raw) == ""


def test_extract_final_answer_plain_response_passthrough():
    # A3B with enable_thinking=False: no <think>/<answer> at all.
    assert extract_final_answer("METR") == "METR"


def test_extract_final_answer_empty_input():
    assert extract_final_answer("") == ""
    assert extract_final_answer(None) == ""


# ---------------------------------------------------------------------------
# is_clean_refusal — sentinel-only is a refusal; sentinel + essay is a fabrication
# ---------------------------------------------------------------------------

def test_is_clean_refusal_sentinel_only():
    assert is_clean_refusal(REFUSAL_SENTINEL) is True
    assert is_clean_refusal(f"{REFUSAL_SENTINEL}.") is True


def test_is_clean_refusal_rejects_essay_refusal():
    # QUEST's exact failure mode: leads with the sentinel, then a fabricated essay.
    essay = REFUSAL_SENTINEL + ". " + ("This is a speculative prediction essay. " * 20)
    assert len(essay) > 60
    assert is_clean_refusal(essay) is False


def test_is_clean_refusal_false_without_sentinel():
    assert is_clean_refusal("The answer is 42.") is False


# ---------------------------------------------------------------------------
# answer_retrievable — the chunk-content gate
# ---------------------------------------------------------------------------

def test_answer_retrievable_true_when_substring_present():
    hits = [_hit("...more likely to reward-hack after remediation prompts...")]
    assert answer_retrievable(["more likely"], hits) is True


def test_answer_retrievable_false_when_absent_from_all_chunks():
    hits = [_hit("completely unrelated transcript content")]
    assert answer_retrievable(["hiding behind", "so far into the future"], hits) is False


def test_answer_retrievable_false_with_no_expected_substrings():
    hits = [_hit("some content")]
    assert answer_retrievable([], hits) is False


def test_answer_retrievable_case_insensitive():
    hits = [_hit("Growing, Not Decreasing was the answer given")]
    assert answer_retrievable(["growing, not decreasing"], hits) is True


# ---------------------------------------------------------------------------
# assert_served_model — fail loud on mismatch or unreachable
# ---------------------------------------------------------------------------

def test_assert_served_model_passes_on_match():
    with patch("agents_core.geist.harness._gw_probe_served_model", return_value="qwen3.6-35b-a3b"):
        assert_served_model("qwen3.6-35b-a3b")  # does not raise


def test_assert_served_model_raises_on_mismatch():
    with patch("agents_core.geist.harness._gw_probe_served_model", return_value="quest-35b-rl"):
        try:
            assert_served_model("qwen3.6-35b-a3b")
            assert False, "expected RuntimeError on served-model mismatch"
        except RuntimeError as e:
            assert "quest-35b-rl" in str(e)
            assert "qwen3.6-35b-a3b" in str(e)


def test_assert_served_model_raises_when_unreachable():
    with patch("agents_core.geist.harness._gw_probe_served_model", return_value=None):
        try:
            assert_served_model("qwen3.6-35b-a3b")
            assert False, "expected RuntimeError when the probe is unreachable"
        except RuntimeError as e:
            assert "unreachable" in str(e).lower() or "cannot verify" in str(e).lower()


# ---------------------------------------------------------------------------
# answer_item — end-to-end per-item flow, retrieval + model call stubbed
# ---------------------------------------------------------------------------

def test_answer_item_clean_refusal_on_absent_trap():
    item = {
        "question": "What is the guest's favorite language?",
        "target_show": "lex-fridman",
        "expected_answer_substrings": [],
        "answer_present": False,
        "category": "absent-trap",
    }
    with patch("agents_core.geist.harness._retrieve_corpus", return_value=[]), \
         patch("agents_core.geist.harness.call_operator", return_value=REFUSAL_SENTINEL):
        result = answer_item(item, model="qwen3.6-35b-a3b")

    assert result["is_refusal"] is True
    assert result["final_answer"] == REFUSAL_SENTINEL
    assert result["answer_retrievable"] is None
    assert result["provenance"]["served_model"] == "qwen3.6-35b-a3b"


def test_answer_item_gates_difficult_present_on_chunk_content():
    item = {
        "question": "Why did Bengio dismiss the risk in 2019?",
        "target_show": "80k-hours",
        "expected_answer_substrings": ["hiding behind", "so far into the future"],
        "answer_present": True,
        "category": "difficult-present",
    }
    with patch("agents_core.geist.harness._retrieve_corpus", return_value=[_hit("unrelated content", "80k-hours")]), \
         patch("agents_core.geist.harness.call_operator", return_value=REFUSAL_SENTINEL):
        result = answer_item(item, model="qwen3.6-35b-a3b")

    # The answer is not in the retrieved chunks: refusing here is honest, not suppression.
    assert result["answer_retrievable"] is False
    assert result["is_refusal"] is True


def test_answer_item_uses_retrieval_probe_when_present():
    item = {
        "question": "bare question",
        "retrieval_probe": "a more targeted probe query",
        "target_show": "mlst",
        "expected_answer_substrings": ["METR"],
        "answer_present": True,
        "category": "answerable",
    }
    with patch("agents_core.geist.harness._retrieve_corpus", return_value=[]) as mock_retrieve, \
         patch("agents_core.geist.harness.call_operator", return_value="METR"):
        answer_item(item, model="qwen3.6-35b-a3b")

    called_queries = mock_retrieve.call_args[0][0]
    assert called_queries == ["a more targeted probe query"]
