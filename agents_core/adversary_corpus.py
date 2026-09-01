"""Adversary corpus assembler for Expert mini-vaults (v0: Tech-Kami only).

v0 invariants:
  Zero LLM calls — pure CPU. If someone reaches for agents_core.llm the
  design has drifted.
  Read-only from sources — never mutates mem; never touches /srv/lapis/lapis-state/.
  Stable fragment ids — derived from source key, not invocation time. Re-runs
  produce identical filenames.
  Mini-vault expansion is additive — corpora/<mode>/ lives alongside existing
  seeds/, post-mortems/, dispatches/. Existing Expert layout consumers are
  unaffected.
  No abstraction extraction — Tech-Kami source-kind list lives as module-level
  constants. A second consumer triggers the refactor, not v0.
  Both modes recognized in corpus_root(); only adversary is assembled here.
  Removal scoped to enumerated source-kinds — if a source-kind is absent from
  the current run's constant list, its subdirectory is left untouched.
  Mem enumeration uses MemoryStore.list_all() (same code path as `mem list
  --tag <tag>`). Raw SQLite against mem.db is not used (WAL + immutable trap).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents_core.expert_layout import corpus_root as _default_corpus_root
from agents_core.mem import MemoryStore
from agents_core.room_paths import room_path

# ---------------------------------------------------------------------------
# Module-level constants — Tech-Kami source set (v0).
# Adding a second Expert triggers extraction; do not generalise yet.
# ---------------------------------------------------------------------------

_TECH_KAMI_EXPERT_ID = "tech-kami"

# Each tuple: (source_kind_dir_name, mem_tag).
# Order determines write order but not fragment ids.
_MEM_SOURCES: tuple[tuple[str, str], ...] = (
    ("feedback", "feedback"),
    ("ratify-correct", "ratify:correct"),
    ("ratify-override", "ratify:override"),
)

_ARC_DOC_SOURCE_KIND = "arc-doc"
_ARC_DOC_DIR = room_path("lapis_state")

# Fail loud if any single mem source reaches this count (silently truncating).
_MEM_LIMIT = 1000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stable_id(source_key: str) -> str:
    """Return a 16-char hex id derived deterministically from source_key."""
    return hashlib.sha256(source_key.encode()).hexdigest()[:16]


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _render_fragment(source_key: str, source_kind: str, body: str, captured_at: str) -> str:
    """Render frontmatter + body as a single fragment string."""
    fm = {
        "captured_at": captured_at,
        "source_key": source_key,
        "source_kind": source_kind,
        "source_sha256": _sha256_of(body),
    }
    return f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=True)}---\n{body}"


def _parse_source_sha256(fragment_text: str) -> str | None:
    """Extract source_sha256 from an existing fragment's YAML front matter."""
    if not fragment_text.startswith("---\n"):
        return None
    end = fragment_text.find("\n---\n", 4)
    if end == -1:
        return None
    try:
        fm = yaml.safe_load(fragment_text[4:end])
        if isinstance(fm, dict):
            return fm.get("source_sha256")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Source enumerators
# ---------------------------------------------------------------------------

def _mem_source_total_count(
    mem_tag: str,
    tags: list[str] | None,
    store: MemoryStore,
) -> int | None:
    """Best-effort true total count of matching mem entries.

    Prefers an explicit count API on the store (count/has_more signal);
    falls back to a single unbounded list_all() probe. Returns None when the
    store exposes no usable count signal (defensive; a real MemoryStore
    always supports the probe).
    """
    count = getattr(store, "count", None)
    if callable(count):
        try:
            return int(count(tag=mem_tag, tags=tags))
        except TypeError:
            pass
        try:
            return int(count(mem_tag))
        except Exception:
            pass
    try:
        return len(store.list_all(tag=mem_tag, tags=tags))
    except Exception:
        return None


def _enumerate_mem_source(
    source_kind: str,
    mem_tag: str,
    store: MemoryStore,
) -> list[tuple[str, str]]:
    """Enumerate mem entries for a tag. Raises RuntimeError if the TOTAL
    number of matching entries exceeds _MEM_LIMIT (not just when one paged
    result happens to equal it)."""
    total = _mem_source_total_count(mem_tag, None, store)
    if total is not None and total > _MEM_LIMIT:
        raise RuntimeError(
            f"adversary_corpus: mem source '{source_kind}' (tag={mem_tag!r}) "
            f"has {total} entries — exceeds _MEM_LIMIT ({_MEM_LIMIT}), corpus "
            f"would silently truncate. Raise _MEM_LIMIT or prune the source tag."
        )
    rows = store.list_all(tag=mem_tag, limit=_MEM_LIMIT)
    if len(rows) == _MEM_LIMIT:
        raise RuntimeError(
            f"adversary_corpus: mem source '{source_kind}' (tag={mem_tag!r}) "
            f"returned {_MEM_LIMIT} entries — limit reached, corpus is silently "
            f"truncating. Raise _MEM_LIMIT or prune the source tag."
        )
    return [(row["key"], row["content"]) for row in rows]


def _enumerate_arc_docs(arc_dir: Path) -> list[tuple[str, str]]:
    """Enumerate *.md files in arc_dir as (source_key, body) pairs."""
    if not arc_dir.is_dir():
        return []
    result = []
    for md_file in sorted(arc_dir.glob("*.md")):
        source_key = f"{_ARC_DOC_SOURCE_KIND}/{md_file.name}"
        body = md_file.read_text(encoding="utf-8", errors="replace")
        result.append((source_key, body))
    return result


# ---------------------------------------------------------------------------
# Per-source-kind processor
# ---------------------------------------------------------------------------

def _process_source_kind(
    source_kind: str,
    items: list[tuple[str, str]],
    subdir: Path,
    dry_run: bool,
) -> tuple[int, int, int, list[str]]:
    """Write/update/remove fragments for one source-kind.

    Returns (written, unchanged, removed, errors).
    Removal is scoped to this subdir only.
    """
    written = unchanged = removed = 0
    errors: list[str] = []

    # desired: stable_id -> (source_key, body)
    desired: dict[str, tuple[str, str]] = {}
    for source_key, body in items:
        sid = _stable_id(source_key)
        desired[sid] = (source_key, body)

    # existing fragments on disk (keyed by stem = stable_id)
    existing: dict[str, Path] = {}
    if subdir.exists():
        for f in subdir.glob("*.md"):
            existing[f.stem] = f

    if not dry_run:
        subdir.mkdir(parents=True, exist_ok=True)

    captured_at = datetime.now(timezone.utc).isoformat()

    # Write or mark unchanged
    for sid, (source_key, body) in desired.items():
        out_path = subdir / f"{sid}.md"
        new_sha = _sha256_of(body)

        if out_path.exists():
            try:
                existing_text = out_path.read_text(encoding="utf-8", errors="replace")
                existing_sha = _parse_source_sha256(existing_text)
                if existing_sha == new_sha:
                    unchanged += 1
                    continue
            except Exception:
                pass  # fall through to (re)write

        if not dry_run:
            try:
                fragment = _render_fragment(source_key, source_kind, body, captured_at)
                # Atomic write (same pattern as target YAML / GPU queue task
                # persistence): mkstemp in the destination dir + os.replace,
                # with best-effort temp cleanup on any failure.
                fd, tmp = tempfile.mkstemp(
                    dir=str(out_path.parent),
                    prefix="." + out_path.name + ".",
                    suffix=".tmp",
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(fragment)
                    os.replace(tmp, out_path)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
            except Exception as exc:
                errors.append(f"{source_kind}/{sid}: {exc}")
                continue
        written += 1

    # Remove stale fragments (source disappeared)
    for sid, fpath in existing.items():
        if sid not in desired:
            if not dry_run:
                try:
                    fpath.unlink()
                except Exception as exc:
                    errors.append(f"remove {source_kind}/{sid}: {exc}")
                    continue
            removed += 1

    return written, unchanged, removed, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_adversary_corpus(
    expert_id: str,
    *,
    dry_run: bool = False,
    _corpus_root: Path | None = None,
    _arc_dir: Path | None = None,
    _mem_store: Any | None = None,
) -> dict:
    """Assemble or refresh the adversary corpus for the named Expert.

    Returns {"written": N, "unchanged": M, "removed": K, "errors": [...]}.
    Idempotent — fragment ids are derived from source keys, not invocation time.
    Removes fragments whose source has disappeared, scoped to enumerated
    source-kinds only.

    Raises ValueError for unknown expert_id.
    Raises RuntimeError if any mem source hits the _MEM_LIMIT ceiling.
    """
    if expert_id != _TECH_KAMI_EXPERT_ID:
        raise ValueError(
            f"build_adversary_corpus: unknown expert_id {expert_id!r}; "
            f"only {_TECH_KAMI_EXPERT_ID!r} is wired in v0."
        )

    corpus_dir = (
        _corpus_root
        if _corpus_root is not None
        else _default_corpus_root(expert_id, "adversary")
    )
    arc_dir = _arc_dir if _arc_dir is not None else _ARC_DOC_DIR

    total_written = total_unchanged = total_removed = 0
    all_errors: list[str] = []

    own_store = _mem_store is None
    store: MemoryStore = MemoryStore() if own_store else _mem_store

    try:
        # Mem-tagged sources
        for source_kind, mem_tag in _MEM_SOURCES:
            items = _enumerate_mem_source(source_kind, mem_tag, store)
            subdir = corpus_dir / source_kind
            w, u, r, errs = _process_source_kind(source_kind, items, subdir, dry_run)
            total_written += w
            total_unchanged += u
            total_removed += r
            all_errors.extend(errs)

        # Arc-doc source
        arc_items = _enumerate_arc_docs(arc_dir)
        subdir = corpus_dir / _ARC_DOC_SOURCE_KIND
        w, u, r, errs = _process_source_kind(_ARC_DOC_SOURCE_KIND, arc_items, subdir, dry_run)
        total_written += w
        total_unchanged += u
        total_removed += r
        all_errors.extend(errs)

    finally:
        if own_store:
            store.close()

    return {
        "written": total_written,
        "unchanged": total_unchanged,
        "removed": total_removed,
        "errors": all_errors,
    }
