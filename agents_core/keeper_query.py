"""agents_core.keeper_query — headless CLI for Knowledge-Keeper queries.

Entry points
------------
  python3 -m agents_core.keeper_query "<question>"
  keeper-query "<question>"

The CLI is a thin, read-only wrapper over ``agents_core.librarian.corroborate()``.
All synthesis, citation verification, signing, and caching are delegated to the
librarian substrate.  This module adds only:

  - Lazy stakes routing (``--stakes low|high|auto``).
  - Empty-result fallback (retry with full corpus when a scoped query returns zero hits).
  - Graceful degradation surface (``[DEGRADED …]`` banner on LibrarianUnavailable).
  - Human-readable output + ``--json`` machine mode.
  - ``--dry-run`` that resolves scope/stakes without calling the LLM.

Invariants
----------
- No direct Anthropic API.  No ``import anthropic``.  No ``ANTHROPIC_API_KEY``.
- Read-only: writes nothing to the vault, corpus, mem.db, or attribution.db.
- Never crash on substrate failure.  Exit codes: 0 ok, 1 bad args, 2 degraded.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents_core.retrieval import Hit

log = logging.getLogger(__name__)

_DEFAULT_CORPUS = "vault-rag"
_DEFAULT_TOP_K = 8

# ---------------------------------------------------------------------------
# D2 — Stakes heuristic
# ---------------------------------------------------------------------------

# Markers that indicate the user wants synthesis / comparison / decision support.
_HIGH_STAKES_MARKERS = frozenset({
    "why", "should", "compare", "vs", "versus", "trade-off", "tradeoff",
    "trade off", "difference", "differences", "explain", "how does", "how do",
    "recommend", "better", "worse", "pros", "cons",
})

# Score spread threshold: if top hit score minus second hit score is below this
# value the results are "too close to call" and we route high.
_AMBIGUOUS_SPREAD_THRESHOLD = 0.15


def decide_stakes(question: str, hits: list) -> tuple[str, str]:
    """Return ``(stakes, reason)`` for the given question and retrieval hits.

    PROVISIONAL — v0 placeholder heuristic. This function's routing logic is a
    best-guess calibrated against early dogfooding. It is explicitly NOT a settled
    contract: real usage will reveal mis-routes that should be reported and used to
    replace or tune these rules. See the ``--dry-run`` output for what the heuristic
    could not see.

    Allowed signals:
    - Question text: presence of synthesis/comparison markers → high.
    - Hit count and score spread: zero hits or ambiguous top scores → high.
      A clear top hit with a lookup-shaped question → low.
    """
    q_lower = question.lower()

    # Synthesis/comparison markers — explicit question words that signal the user
    # wants more than a document retrieval.
    for marker in _HIGH_STAKES_MARKERS:
        if marker in q_lower:
            return (
                "high",
                f"routed high: question contains a synthesis marker (\"{marker}\"); "
                "could not see prior corrections, correction history, or your intent",
            )

    # A question mark with a clause (i.e. a genuine question rather than a keyword lookup)
    # is a signal that the user is asking for an explanation, not just a document.
    if "?" in question and len(question.split()) > 4:
        return (
            "high",
            "routed high: question ends with '?' and has multiple terms, suggesting a "
            "genuine question rather than a keyword lookup; "
            "could not see whether you wanted synthesis or just retrieval",
        )

    # No hits or ambiguous top-hit spread — synthesis may be necessary.
    if not hits:
        return (
            "high",
            "routed high: retrieval returned no hits; could not see whether corpus "
            "contains relevant material",
        )

    if len(hits) >= 2:
        spread = hits[0].score - hits[1].score
        if spread < _AMBIGUOUS_SPREAD_THRESHOLD:
            return (
                "high",
                f"routed high: top two hit scores are close (spread={spread:.2f} < "
                f"{_AMBIGUOUS_SPREAD_THRESHOLD}), suggesting ambiguous evidence; "
                "could not see which source is authoritative for this question",
            )

    # Short keyword/lookup-shaped query with a clear top hit → low.
    return (
        "low",
        "routed low: short keyword-shaped query with a clear top retrieval hit; "
        "could not see whether you expected a synthesized answer",
    )


# ---------------------------------------------------------------------------
# D3 — Empty-result fallback
# ---------------------------------------------------------------------------

def _retrieve_with_fallback(
    question: str,
    retrieval_scopes: list[str],
    top_k: int,
    orig_corpus: list[str],
) -> tuple[list, bool]:
    """Retrieve hits, falling back to full corpus on zero results.

    Returns ``(hits, fell_back)`` where ``fell_back`` is True when the fallback
    was triggered.
    """
    from agents_core import retrieval

    try:
        hits = retrieval.retrieve(question, retrieval_scopes, top_k=top_k)
    except Exception as exc:
        log.warning("keeper-query: retrieval error: %s", exc)
        hits = []

    if hits:
        return hits, False

    # Zero hits — retry with full corpus (vault-rag) if current scope differs.
    full_scopes = ["vault-rag"]
    if sorted(retrieval_scopes) == sorted(full_scopes):
        # Already at full corpus; no fallback possible.
        return [], False

    log.info("keeper-query: empty-result fallback to full corpus")
    try:
        fallback_hits = retrieval.retrieve(question, full_scopes, top_k=top_k)
    except Exception as exc:
        log.warning("keeper-query: fallback retrieval error: %s", exc)
        fallback_hits = []

    return fallback_hits, True


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _format_hits_human(hits: list, fell_back: bool, orig_corpus: list[str]) -> str:
    """Format retrieval-only hits for human consumption."""
    lines: list[str] = []
    if fell_back:
        lines.append(
            f"[scope: searched {orig_corpus}, none found — fell back to full corpus]"
        )
    if not hits:
        lines.append("No results found.")
        return "\n".join(lines)
    lines.append(f"Top {len(hits)} retrieval hit(s):\n")
    for i, hit in enumerate(hits, 1):
        path = hit.metadata.get("file_path") or hit.metadata.get("path") or hit.id
        snippet = (hit.content or "")[:160].replace("\n", " ")
        lines.append(f"  {i}. [{hit.score:.2f}] {path}")
        if snippet:
            lines.append(f"       {snippet!r}")
    return "\n".join(lines)


def _format_artifact_human(artifact, fell_back: bool, orig_corpus: list[str]) -> str:
    """Format a SynthesisArtifact for human consumption."""
    from agents_core.librarian import SynthesisArtifact
    lines: list[str] = []
    if fell_back:
        lines.append(
            f"[scope: searched {orig_corpus}, none found — fell back to full corpus]"
        )
    answer_text = artifact.answer.get("text") or artifact.answer.get("summary") or (
        artifact.answer.get("raw") or json.dumps(artifact.answer)
    )
    lines.append(answer_text)
    if artifact.citations:
        lines.append("\nCitations:")
        for cit in artifact.citations:
            path = cit.get("path", "")
            snippet = cit.get("quoted_snippet", "")
            lines.append(f'  {path} — "{snippet}"')
    lines.append(f"\nVerification: {artifact.verification}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _build_scope(corpus_tokens: list[str]) -> dict:
    return {"corpus": corpus_tokens}


def _resolve_retrieval_scopes(corpus: list[str]) -> list[str]:
    """Map corpus tokens to retrieval scope tokens via the librarian helper."""
    from agents_core.librarian import _corpus_to_retrieval_scopes
    return _corpus_to_retrieval_scopes(corpus)


def run(
    question: str,
    *,
    stakes: str = "auto",
    corpus: list[str] | None = None,
    freshness: int | None = None,
    no_cache: bool = False,
    top_k: int = _DEFAULT_TOP_K,
    emit_json: bool = False,
    dry_run: bool = False,
) -> int:
    """Execute a keeper query and print to stdout. Returns exit code."""
    corpus = corpus or [_DEFAULT_CORPUS]
    scope = _build_scope(corpus)
    retrieval_scopes = _resolve_retrieval_scopes(corpus)
    full_retrieval_scopes = _resolve_retrieval_scopes([_DEFAULT_CORPUS])

    # Resolve effective freshness
    effective_freshness: int
    if no_cache:
        effective_freshness = 0
    elif freshness is not None:
        effective_freshness = freshness
    else:
        effective_freshness = 60  # corroborate()'s default

    # For auto routing we need a quick retrieval pass first.
    # For explicit low/high we can skip that pre-flight.
    hits: list = []
    fell_back = False
    stakes_resolved: str
    stakes_reason: str

    if stakes == "low" or (stakes == "auto" and not dry_run):
        # Retrieve now for either: (a) low stakes output, or (b) auto routing signal.
        hits, fell_back = _retrieve_with_fallback(
            question, retrieval_scopes, top_k, orig_corpus=corpus
        )

    if stakes == "auto":
        stakes_resolved, stakes_reason = decide_stakes(question, hits)
    else:
        stakes_resolved = stakes
        stakes_reason = f"explicit --stakes {stakes}"

    # --dry-run: print resolved info and exit
    if dry_run:
        info = {
            "corpus": corpus,
            "retrieval_scopes": retrieval_scopes,
            "stakes_decision": stakes_resolved,
            "reason": stakes_reason,
        }
        if emit_json:
            print(json.dumps(info, indent=2))
        else:
            print(f"corpus:           {corpus}")
            print(f"retrieval_scopes: {retrieval_scopes}")
            print(f"stakes:           {stakes_resolved}")
            print(f"reason:           {stakes_reason}")
        return 0

    # -----------------------------------------------------------------------
    # Low stakes: retrieval only
    # -----------------------------------------------------------------------
    if stakes_resolved == "low":
        # hits already populated above for auto; for explicit low, retrieve now.
        if stakes == "low" and not hits:
            # retrieve was already called above for explicit low; this handles
            # the case where hits is empty but we haven't retrieved yet (shouldn't
            # happen but be safe).
            pass

        if stakes == "auto":
            print(f"[auto → low: {stakes_reason.split(': ', 1)[-1]}]")

        if emit_json:
            output = [
                {
                    "id": h.id,
                    "score": h.score,
                    "source": h.source,
                    "content": h.content,
                    "metadata": h.metadata,
                }
                for h in hits
            ]
            if fell_back:
                print(json.dumps({"fell_back": True, "orig_corpus": corpus, "hits": output}))
            else:
                print(json.dumps(output))
        else:
            print(_format_hits_human(hits, fell_back, corpus))

        return 0

    # -----------------------------------------------------------------------
    # High stakes: full corroborate()
    # -----------------------------------------------------------------------
    from agents_core.librarian import LibrarianUnavailable, SynthesisArtifact, corroborate

    if stakes == "auto":
        print(f"[auto → high: {stakes_reason.split(': ', 1)[-1]}]")

    try:
        result = corroborate(
            question,
            scope,
            freshness=effective_freshness,
            policy="auto-update",
        )
    except Exception as exc:
        # corroborate() should never raise, but mirror the guarantee.
        log.error("keeper-query: unexpected error from corroborate: %s", exc)
        print(f"[ERROR] Unexpected failure: {exc}", file=sys.stderr)
        return 2

    if isinstance(result, LibrarianUnavailable):
        cached = result.most_recent_cached
        if cached is not None:
            print("[DEGRADED — LLM unavailable, served from cache]")
            if emit_json:
                print(json.dumps(cached.to_dict(), indent=2))
            else:
                print(_format_artifact_human(cached, fell_back=False, orig_corpus=corpus))
        else:
            print("[DEGRADED — LLM unavailable, no cached answer available]")
            print("Nothing found.")
        return 2

    # Check for empty retrieval in the artifact — trigger fallback and re-corroborate.
    if isinstance(result, SynthesisArtifact) and not result.citations:
        # Heuristic: if we got a "none" verification and the question seems non-trivial,
        # check whether a full-corpus retry could do better.
        if sorted(retrieval_scopes) != sorted(full_retrieval_scopes):
            log.info("keeper-query: empty-result fallback to full corpus")
            fell_back = True
            print(f"[scope: searched {corpus}, none found — fell back to full corpus]")
            full_scope = _build_scope([_DEFAULT_CORPUS])
            try:
                result = corroborate(
                    question,
                    full_scope,
                    freshness=effective_freshness,
                    policy="auto-update",
                )
            except Exception as exc:
                log.error("keeper-query: fallback corroborate failed: %s", exc)
                result = None

            if result is None or isinstance(result, LibrarianUnavailable):
                print("[DEGRADED — fallback corroborate failed]")
                return 2

    if emit_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_format_artifact_human(result, fell_back, corpus))

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="keeper-query",
        description="Query the Knowledge-Keeper corpus and get a synthesized answer.",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--stakes",
        choices=["low", "high", "auto"],
        default="auto",
        help="Routing mode: low=retrieval-only, high=full synthesis, auto=heuristic (default).",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        dest="corpus",
        default=None,
        metavar="TOKEN",
        help="Corpus token(s) to query (default: vault-rag). Repeatable.",
    )
    parser.add_argument(
        "--freshness",
        type=int,
        default=None,
        metavar="N",
        help="Max cache age in seconds before re-synthesis.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Force re-synthesis (equivalent to --freshness 0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        metavar="N",
        help=f"Number of retrieval hits for low-stakes mode (default: {_DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        default=False,
        help="Emit machine-readable JSON output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print resolved scope and stakes without calling the LLM.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        code = run(
            args.question,
            stakes=args.stakes,
            corpus=args.corpus,
            freshness=args.freshness,
            no_cache=args.no_cache,
            top_k=args.top_k,
            emit_json=args.emit_json,
            dry_run=args.dry_run,
        )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(2)

    sys.exit(code)


if __name__ == "__main__":
    main()
