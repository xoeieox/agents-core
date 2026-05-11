#!/usr/bin/env python3
"""
Kami Selector — build the explicit docs list for tonight's kami-batch.

Dual-mode:
  • Mode A (changes):  read /srv/lapis/kami/flagged-docs.yaml, take flagged paths.
  • Mode B (backfill): fill remaining slots with unprocessed corpus docs
    using a weighted policy — ~50% vault, ~50% rotating deficit-first
    through anthro-corpus → dplace → invariant → psych-mining.

Returns a list of dicts: {path, base_path, corpus, reason}.

Called by night_manifest._seed_kami_batch() to seed a single kami-batch
ManifestTask each night. Empty list → no task scheduled.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import yaml

log = logging.getLogger("kami-selector")

FLAGGED_DOCS = Path("/srv/lapis/kami/flagged-docs.yaml")
PROCESSED_DIR = Path("/srv/lapis/kami/processed")

VAULT_PATH = Path("/srv/git/inertia-vault-working")
RESEARCH_BASE = Path("/srv/lapis/research")
FORMING_BASE = Path("/srv/lapis/forming")

# Corpus configuration: (corpus_label, base_path, glob_patterns)
CORPORA = {
    "vault":                   (VAULT_PATH,                     ["**/*.md"]),
    "research/anthro-corpus":  (RESEARCH_BASE / "anthro-corpus", ["**/*.yaml", "**/*.md"]),
    "research/dplace":         (RESEARCH_BASE / "dplace",        ["**/*.yaml", "**/*.md"]),
    "research/invariant":      (RESEARCH_BASE / "invariant",     ["**/*.yaml", "**/*.md"]),
    "forming/psych-mining":    (FORMING_BASE / "psych-mining",   ["**/*.md", "**/*.yaml"]),
    "library/podcasts":        (Path("/srv/lapis/library/podcasts"),  ["**/*.md"]),
    "library/civic-theory":    (Path("/srv/lapis/library/civic-theory"), ["**/*.md"]),
}

# Deficit-first rotation order for the "other half" of Mode B
ROTATION_ORDER = [
    "research/anthro-corpus",
    "research/dplace",
    "research/invariant",
    "forming/psych-mining",
    "library/podcasts",
    "library/civic-theory",
]

VAULT_SKIP_DIRS = {
    ".git", ".obsidian", "node_modules", ".trash",
    "Daily-Notes", "Templates", "visual-toolkit", "Transcripts",
}

# Root-level vault files that are system/meta, not knowledge docs
VAULT_SKIP_FILES = {"CLAUDE.md", "Bases Hub.md", "Active Work.md"}


def _load_processed_paths() -> set[str]:
    """Return the set of paths that already have a processed/ record."""
    if not PROCESSED_DIR.exists():
        return set()
    paths: set[str] = set()
    for f in PROCESSED_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(f.read_text())
            if isinstance(data, dict) and data.get("path"):
                paths.add(data["path"])
        except Exception:
            continue
    return paths


def _load_flagged(stale_after: datetime | None = None) -> list[dict]:
    """Read flagged-docs.yaml; reject if stale or malformed. Returns list of entries."""
    if not FLAGGED_DOCS.exists():
        return []
    try:
        data = yaml.safe_load(FLAGGED_DOCS.read_text())
    except Exception as e:
        log.warning(f"flagged-docs.yaml unreadable: {e}")
        return []
    if not isinstance(data, dict):
        return []
    flagged = data.get("flagged") or []
    if not isinstance(flagged, list):
        return []
    # Staleness check
    if stale_after is not None:
        generated_at = data.get("generated_at", "")
        try:
            gen = datetime.fromisoformat(generated_at)
            # Normalize: strip tzinfo for comparison if stale_after is naive
            if gen.tzinfo and stale_after.tzinfo is None:
                gen = gen.replace(tzinfo=None)
            if gen < stale_after:
                log.info(f"flagged-docs.yaml is stale (generated_at={generated_at}); ignoring")
                return []
        except Exception:
            log.warning("flagged-docs.yaml has no parseable generated_at; ignoring")
            return []
    return flagged


def _list_corpus_docs(corpus: str) -> list[tuple[Path, str]]:
    """Return [(abs_path, rel_path)] for every doc in the corpus."""
    if corpus not in CORPORA:
        return []
    base, patterns = CORPORA[corpus]
    if not base.exists():
        return []
    out: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for pat in patterns:
        for abs_p in base.rglob(pat):
            if not abs_p.is_file():
                continue
            # Skip certain vault subdirs, hidden paths, and root-level system files
            parts = abs_p.relative_to(base).parts
            if parts and parts[0] in VAULT_SKIP_DIRS:
                continue
            if any(p.startswith(".") for p in parts):
                continue
            if corpus == "vault" and len(parts) == 1 and parts[0] in VAULT_SKIP_FILES:
                continue
            rel = str(abs_p.relative_to(base))
            if rel in seen:
                continue
            seen.add(rel)
            out.append((abs_p, rel))
    return out


def _corpus_full_paths(corpus: str) -> list[str]:
    """Return corpus-qualified paths used as keys in the processed/ tracker.

    We store `path` in processed/ as it was written by kami_small
    (vault docs: relative vault path; research/forming: relative to their base).
    The matching is path-string equality against that stored value.
    """
    base, _ = CORPORA[corpus]
    return [rel for _, rel in _list_corpus_docs(corpus)]


def _unprocessed_docs(corpus: str, processed: set[str]) -> list[dict]:
    """Return unprocessed docs for a corpus, oldest-mtime first."""
    entries = _list_corpus_docs(corpus)
    base, _ = CORPORA[corpus]
    result = []
    for abs_p, rel in entries:
        if rel in processed:
            continue
        try:
            mtime = abs_p.stat().st_mtime
        except OSError:
            mtime = 0
        result.append({
            "path": rel,
            "base_path": str(base),
            "corpus": corpus,
            "mtime": mtime,
        })
    result.sort(key=lambda d: d["mtime"])
    return result


def _take_weighted_backfill(
    slots: int, processed: set[str], exclude_paths: set[str]
) -> list[dict]:
    """Fill N backfill slots: ~50% vault, ~50% rotating deficit-first."""
    if slots <= 0:
        return []

    vault_slots = slots // 2
    rotation_slots = slots - vault_slots

    out: list[dict] = []

    # Half for vault
    vault_unprocessed = _unprocessed_docs("vault", processed)
    for d in vault_unprocessed:
        if len(out) >= vault_slots:
            break
        if d["path"] in exclude_paths:
            continue
        d.pop("mtime", None)
        d["reason"] = "backfill:vault"
        out.append(d)

    # Half rotating deficit-first (loop through rotation order, one doc at a time,
    # so smallest corpora finish evenly instead of anthro dominating the first 5 nights)
    rotation_pools = {c: _unprocessed_docs(c, processed) for c in ROTATION_ORDER}
    # Sort each pool's docs within corpus by mtime (already done) — we'll pop from head

    taken = 0
    while taken < rotation_slots:
        took_any_this_loop = False
        for corpus in ROTATION_ORDER:
            if taken >= rotation_slots:
                break
            pool = rotation_pools[corpus]
            while pool:
                d = pool.pop(0)
                if d["path"] in exclude_paths or any(
                    o["path"] == d["path"] and o["corpus"] == d["corpus"] for o in out
                ):
                    continue
                d.pop("mtime", None)
                d["reason"] = f"backfill:{corpus}"
                out.append(d)
                taken += 1
                took_any_this_loop = True
                break
        if not took_any_this_loop:
            # All rotation pools exhausted. Fall back to more vault docs to fill.
            break

    # If rotation ran dry and we still have slots, top up with vault.
    if len(out) < slots:
        for d in vault_unprocessed[vault_slots:]:
            if len(out) >= slots:
                break
            if d["path"] in exclude_paths or any(
                o["path"] == d["path"] for o in out
            ):
                continue
            d.pop("mtime", None)
            d["reason"] = "backfill:vault-rollover"
            out.append(d)

    return out


def _corpus_to_base(corpus: str) -> str | None:
    cfg = CORPORA.get(corpus)
    return str(cfg[0]) if cfg else None


def select_kami_docs(
    limit: int = 80,
    weighted_backfill: bool = True,
    flagged_stale_after: datetime | None = None,
) -> list[dict]:
    """Return the ordered docs list for tonight's kami-batch.

    Args:
        limit: max docs total.
        weighted_backfill: whether Mode B uses 50/50 vault + rotation.
                           Currently always True in production.
        flagged_stale_after: reject flagged-docs.yaml if generated_at is older.
                             Defaults to today at 18:00 local.

    Returns:
        list of {path, base_path, corpus, reason} dicts.
    """
    if flagged_stale_after is None:
        today_18 = datetime.now().replace(hour=18, minute=0, second=0, microsecond=0)
        # If current time is before 18:00, accept yesterday's (cycle hasn't started)
        if datetime.now() < today_18:
            today_18 = today_18 - timedelta(days=1)
        flagged_stale_after = today_18

    # Mode A — changes
    flagged_entries = _load_flagged(stale_after=flagged_stale_after)
    changes: list[dict] = []
    seen_paths: set[str] = set()
    for entry in flagged_entries:
        if not isinstance(entry, dict):
            continue
        p = entry.get("path")
        corpus = entry.get("corpus", "vault")
        if not p or p in seen_paths:
            continue
        if p == "__push_truncated__":
            continue
        seen_paths.add(p)
        changes.append({
            "path": p,
            "base_path": _corpus_to_base(corpus),
            "corpus": corpus,
            "reason": f"change:{entry.get('reason', 'flagged')}"[:120],
        })
        if len(changes) >= limit:
            break

    # Mode B — backfill
    remaining = max(0, limit - len(changes))
    processed = _load_processed_paths()
    backfill = _take_weighted_backfill(
        slots=remaining,
        processed=processed,
        exclude_paths=seen_paths,
    ) if weighted_backfill else []

    combined = changes + backfill
    log.info(
        f"kami-selector: {len(changes)} changes + {len(backfill)} backfill "
        f"= {len(combined)} docs (limit={limit})"
    )
    return combined[:limit]


if __name__ == "__main__":
    import argparse
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Dry-run the Kami docs selector")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--json", action="store_true", help="Dump full docs list as JSON")
    args = ap.parse_args()
    docs = select_kami_docs(limit=args.limit)
    if args.json:
        print(json.dumps(docs, indent=2))
    else:
        from collections import Counter
        c = Counter(d["corpus"] for d in docs)
        print(f"Total: {len(docs)}")
        for k, n in c.most_common():
            print(f"  {k:<30}  {n}")
