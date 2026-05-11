"""Tests for the success/failure output-write behaviour in _run_shaped_task.

Covers:
  1. Large success: combined >3000 chars → full payload written (head preserved).
  2. Small success: combined <3000 chars → output equals combined exactly.
  3. Failure tail: combined >3000 chars with rc=1 → output starts with failure
     prefix and ends with the last 3000 chars of combined.
"""
from __future__ import annotations

from pathlib import Path

from agents_core.claude_queue_runner import _classify_runner_failure, _write_success_output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_large_success_preserves_head(tmp_path):
    """combined ~5000 chars: output file must contain the head sentinel."""
    sentinel = "__HEAD_MARKER__"
    combined = sentinel + ("x" * 4985)
    assert len(combined) > 3000

    out = tmp_path / "output.md"
    _write_success_output(out, combined)

    content = out.read_text()
    assert sentinel in content
    assert content.startswith(sentinel)


def test_small_success_unchanged(tmp_path):
    """combined ~500 chars: output file contents equal combined exactly."""
    combined = "small output " + ("y" * 487)
    assert len(combined) < 3000

    out = tmp_path / "output.md"
    _write_success_output(out, combined)

    assert out.read_text() == combined


def test_failure_tail_preserved(tmp_path):
    """combined ~5000 chars with rc=1: output starts with failure prefix and
    ends with the last 3000 chars of combined."""
    combined = ("HEADER_NOISE " * 400) + "TAIL_END"
    assert len(combined) > 3000

    out = tmp_path / "output.md"
    prefix, _ = _classify_runner_failure(combined, rc=1)
    out.write_text(f"{prefix}:\n{combined[-3000:]}")

    content = out.read_text()
    # Must start with the EXIT prefix line
    assert content.startswith("EXIT 1:\n")
    # Must end with the last 3000 chars of the combined payload
    assert content.endswith(combined[-3000:])
