"""agents_core.librarian.shift — degree-of-shift computation.

Given a synthesis cache entry's cited content hashes and the current corpus,
classifies how much the cited sources have changed since synthesis.

Shift levels (ordered by severity)::

    no_shift      — essentially identical (ratio >= 0.98)
    word_line     — word/line-level changes (ratio >= 0.85, no structural diff)
    paragraph     — paragraph-level changes (ratio >= 0.50 or frontmatter changed)
    section       — section-level changes (ratio < 0.50 or heading structure changed)
    source_broken — cited source no longer exists at that path in the corpus

Classification uses ``difflib.SequenceMatcher`` ratios plus structural marker
detection (heading lines, YAML frontmatter blocks).

For an entire artifact, the shift level is the *maximum* across all cited sources.
"""
from __future__ import annotations

import difflib
import re
from enum import IntEnum
from typing import Callable


class ShiftLevel(IntEnum):
    """Ordered levels of content change severity."""
    no_shift = 0
    word_line = 1
    paragraph = 2
    section = 3
    source_broken = 4

    def __str__(self) -> str:
        return self.name.replace("_", "-")  # "no-shift", "word-line", etc.


# ---------------------------------------------------------------------------
# Structural marker helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_FM_RE = re.compile(r"^---\r?\n(.*?\r?\n)---\r?\n", re.DOTALL)


def _heading_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if _HEADING_RE.match(ln)]


def _frontmatter(text: str) -> str | None:
    m = _FM_RE.match(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Single-source comparison
# ---------------------------------------------------------------------------

def compute_shift(old_content: str, new_content: str) -> ShiftLevel:
    """Classify the shift between old_content and new_content.

    Parameters
    ----------
    old_content:
        Content of the source at synthesis time.
    new_content:
        Current content of the source (pass empty string if source is missing).

    Returns
    -------
    ShiftLevel
    """
    if not new_content or not new_content.strip():
        return ShiftLevel.source_broken

    # Fast path: identical
    if old_content == new_content:
        return ShiftLevel.no_shift

    ratio = difflib.SequenceMatcher(None, old_content, new_content, autojunk=False).ratio()

    if ratio >= 0.98:
        return ShiftLevel.no_shift

    # Check structural markers
    old_headings = _heading_lines(old_content)
    new_headings = _heading_lines(new_content)
    heading_ratio = (
        difflib.SequenceMatcher(None, old_headings, new_headings).ratio()
        if (old_headings or new_headings)
        else 1.0
    )

    old_fm = _frontmatter(old_content)
    new_fm = _frontmatter(new_content)
    fm_changed = old_fm != new_fm

    # Section-level: very low ratio or significant heading structural change
    if ratio < 0.50 or (len(old_headings) != len(new_headings) and heading_ratio < 0.50):
        return ShiftLevel.section

    # Paragraph-level: moderate ratio, or frontmatter changed, or notable heading drift
    if ratio < 0.85 or fm_changed or heading_ratio < 0.80:
        return ShiftLevel.paragraph

    return ShiftLevel.word_line


# ---------------------------------------------------------------------------
# Artifact-level shift (max across cited sources)
# ---------------------------------------------------------------------------

def compute_shift_from_rows(
    source_rows: list[dict],
    read_old: Callable[[str, str], str | None],
    read_new: Callable[[str], str | None],
) -> ShiftLevel:
    """Full shift computation when both old and new content are available.

    Parameters
    ----------
    source_rows:
        From ``synthesis_cache.get_source_rows(artifact_id)``.
    read_old:
        Callable(source_path, content_hash) → old content or None.
        In practice, old content is retrieved from the RAG chunk store.
    read_new:
        Callable(source_path) → current content or None.

    Returns
    -------
    ShiftLevel — maximum shift across all cited sources.
    """
    if not source_rows:
        return ShiftLevel.no_shift

    max_shift = ShiftLevel.no_shift

    for row in source_rows:
        if row["source_content_hash"] == row["latest_content_hash"]:
            continue  # no change recorded in tracking

        old_content = read_old(row["source_path"], row["source_content_hash"]) or ""
        new_content = read_new(row["source_path"]) or ""

        level = compute_shift(old_content, new_content)
        if level > max_shift:
            max_shift = level
            if max_shift == ShiftLevel.source_broken:
                break

    return max_shift
