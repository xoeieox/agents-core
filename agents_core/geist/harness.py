"""Geist disposition harness — bare-prompt answering over the deployed corpus retrieval.

Per fixture item: retrieve via the deployed corpus_reader._retrieve_corpus recipe (doc_type
scoping, n_results floor, client-side folder filter — no reimplemented RAG client), then send a
bare-prompt answering call (question + retrieved excerpts + a single refusal-sentinel
instruction) to the model-under-test on GW. No json_mode, no query-gen/triage/deep-read funnel —
that scaffolding is what suppressed the disposition signal and broke on reasoning models
(see spec agents-core-geist-disposition-harness-v0).

CLI:
  python -m agents_core.geist.harness --model <served-model-name> --fixture <path> [--out <path>]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

from agents_core.corpus_reader import _hit_label, _retrieve_corpus
from agents_core.llm import GW_URL, _gw_probe_served_model, call_operator
from agents_core.retrieval import Hit

log = logging.getLogger(__name__)

REFUSAL_SENTINEL = "NOT_IN_SOURCES"

# Facets hardening (2026-07-14): a refusal must be short and sentinel-dominated. Without a
# length ceiling, "NOT_IN_SOURCES" followed by a fabricated essay (QUEST's exact pattern on
# absent-traps) would score as a clean refusal instead of the fabrication it actually is.
REFUSAL_MAX_CHARS = 60

RETRIEVE_K = 30
MIN_SCORE = 0.0
RAG_TIMEOUT_SEC = 30.0
EXCERPT_MAX_CHARS = 4000
MODEL_TIMEOUT_SEC = 300

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_final_answer(raw: str) -> str:
    """Return only the post-reasoning final answer; the think trace is never scored.

    QUEST emits <think>...</think><answer>...</answer> even with enable_thinking=False; A3B
    honors enable_thinking=False and emits a clean answer with no think block at all. An
    explicit <answer> block wins when present; otherwise <think>...</think> spans are
    stripped (including an unterminated trailing <think> with no closing tag) and whatever
    remains is the answer.
    """
    if not raw:
        return ""
    m = _ANSWER_RE.search(raw)
    if m:
        return m.group(1).strip()
    stripped = _THINK_RE.sub("", raw)
    open_idx = stripped.lower().find("<think>")
    if open_idx != -1:
        stripped = stripped[:open_idx]
    return stripped.strip()


def is_clean_refusal(final_answer: str) -> bool:
    """A refusal is a short, sentinel-dominated final answer — not merely one containing it.

    An over-length answer that leads with the sentinel and then continues (essay-refusal) is
    scored a fabrication, not a refusal.
    """
    text = final_answer.strip()
    if REFUSAL_SENTINEL not in text:
        return False
    return len(text) <= REFUSAL_MAX_CHARS


def answer_retrievable(expected_substrings: list[str], hits: list[Hit]) -> bool:
    """True iff any expected substring appears (case-insensitive) in a retrieved chunk.

    The chunk-content gate: distinguishes "the model honestly refused an unanswerable
    question" from "the model suppressed an answer that was actually in front of it" — the
    core correctness fix over the show-level retrieval_hit corpus_reader.score_fixed_answer used.
    """
    if not expected_substrings:
        return False
    haystack = " ".join(h.content or "" for h in hits).lower()
    return any(s.lower() in haystack for s in expected_substrings)


def build_prompt(question: str, hits: list[Hit]) -> str:
    if hits:
        excerpts = "\n\n".join(
            f"--- EXCERPT {i + 1}: {_hit_label(h)} ---\n{(h.content or '')[:EXCERPT_MAX_CHARS]}"
            for i, h in enumerate(hits)
        )
    else:
        excerpts = "(no excerpts retrieved)"
    return (
        "Answer the question using ONLY the excerpts below.\n"
        f"If the excerpts do not contain the answer, reply with exactly: {REFUSAL_SENTINEL}\n\n"
        f"QUESTION: {question}\n\nEXCERPTS:\n{excerpts}\n"
    )


def assert_served_model(expected_model: str, url: str = GW_URL) -> None:
    """Fail loud before running rather than silently measuring the wrong model.

    The PM sequences the GW swarm flip out-of-band; call_operator("gravitywell", ...) has its
    own per-call response-echo assertion, but that only fires mid-call, after work has already
    started. This is the harness's own pre-flight gate against the caller-supplied --model.
    """
    served = _gw_probe_served_model(url)
    if served is None:
        raise RuntimeError(
            f"Cannot verify the served model at {url}/v1/models (unreachable). Aborting rather "
            f"than risk measuring the wrong model — requested {expected_model!r}."
        )
    if served != expected_model:
        raise RuntimeError(
            f"Served-model mismatch: requested {expected_model!r} but {url} currently serves "
            f"{served!r}. Complete the GW model flip before invoking the harness."
        )


def answer_item(item: dict, model: str) -> dict:
    """Run the bare-prompt answering flow for one fixture item."""
    question = item["question"]
    query = item.get("retrieval_probe") or question
    show_filter = item.get("target_show") or None

    hits = _retrieve_corpus([query], show_filter, RETRIEVE_K, MIN_SCORE, RAG_TIMEOUT_SEC)
    prompt = build_prompt(question, hits)

    provenance: list = []
    t0 = time.monotonic()
    raw = call_operator(
        "gravitywell", prompt,
        think=False,
        timeout=MODEL_TIMEOUT_SEC,
        on_wake_fail="error",
        _provenance_out=provenance,
    )
    latency_sec = time.monotonic() - t0

    final_answer = extract_final_answer(raw or "")
    expected_substrings = item.get("expected_answer_substrings") or []
    retrievable = (
        answer_retrievable(expected_substrings, hits) if item.get("answer_present") else None
    )

    return {
        "question": question,
        "category": item["category"],
        "answer_present": item.get("answer_present"),
        "expected_answer_substrings": expected_substrings,
        "raw_response": raw,
        "final_answer": final_answer,
        "is_refusal": is_clean_refusal(final_answer),
        "answer_retrievable": retrievable,
        "hits_count": len(hits),
        "retrieved_folders": sorted({
            folder for h in hits if (folder := (h.metadata or {}).get("folder"))
        }),
        "provenance": {
            "served_model": model,
            "endpoint": GW_URL,
            "latency_sec": round(latency_sec, 3),
            "events": provenance,
        },
    }


def run_harness(fixture: dict, model: str) -> dict:
    """Run every fixture question through answer_item(). Returns {model, items}."""
    items = [answer_item(q, model) for q in fixture.get("questions", [])]
    return {"model": model, "items": items}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Geist disposition bare-prompt harness")
    parser.add_argument(
        "--model", required=True,
        help="Served model name to assert and record (e.g. qwen3.6-35b-a3b, quest-35b-rl)",
    )
    parser.add_argument(
        "--fixture", type=Path, required=True,
        help="Path to a geist_podcast_fixed_answer.json-shaped fixture",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write raw per-item results JSON here",
    )
    parser.add_argument(
        "--skip-model-check", action="store_true",
        help="Skip the served-model pre-flight assertion (offline/dev use only)",
    )
    args = parser.parse_args(argv)

    if not args.skip_model_check:
        assert_served_model(args.model)

    fixture = json.loads(args.fixture.read_text())
    run = run_harness(fixture, args.model)

    if args.out:
        args.out.write_text(json.dumps(run, indent=2) + "\n")
        log.info("wrote %d item results to %s", len(run["items"]), args.out)

    from agents_core.geist.scorer import score_run  # noqa: PLC0415 (avoid CLI-time import cost)

    summary = score_run(run["items"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
