"""Regression tests for _classify_runner_failure prefix selection.

Guard against the 2026-04-27 misclassification where a benign WARN line
mentioning 'worktree_setup' caused the runner to label a call_claude_cli
failure as 'ERROR: worktree_setup', masking the real outage for 48 hours.

Contract source for ERROR: line prefixes: lapis-pm's _runner.py at
/srv/lapis/lapis-pm/lapis_pm/_runner.py.
"""
from __future__ import annotations

import pytest

from agents_core.claude_queue_runner import _classify_runner_failure


# ---------------------------------------------------------------------------
# (1) Negative regression — the literal 2026-04-27 fixture
#
# combined contains a benign WARN mentioning 'worktree_setup' but no
# ERROR: worktree_setup line.  The real failure was call_claude_cli returning
# None.  Prefix must be EXIT 1, not ERROR: worktree_setup.
# ---------------------------------------------------------------------------

def test_benign_warn_picks_exit_prefix():
    combined = (
        "ERROR: shaped agent call returned None (timeout or invocation failure)\n"
        "WARN: worktree_setup: .claude/settings.json missing at /tmp/lapis-pm-worktrees/claude_20260425_164727_3432\n"
    )
    prefix, error_str = _classify_runner_failure(combined, rc=1)
    assert prefix == "EXIT 1", (
        f"Expected 'EXIT 1' for benign WARN, got {prefix!r}. "
        "Substring 'worktree_setup' in a WARN line must not trigger the worktree prefix."
    )
    assert error_str == "EXIT 1"


# ---------------------------------------------------------------------------
# (2) Positive case — actual ERROR: worktree_setup: line from _runner.py:198
# ---------------------------------------------------------------------------

def test_actual_worktree_setup_error_picks_worktree_prefix():
    combined = (
        "ERROR: worktree_setup: git worktree add failed: fatal: 'refs/heads/mybranch' is not a valid branch name\n"
        "Traceback (most recent call last):\n"
        "  File '_runner.py', line 198, in ...\n"
    )
    prefix, error_str = _classify_runner_failure(combined, rc=1)
    assert prefix == "ERROR: worktree_setup"
    assert error_str == "ERROR: worktree_setup"


# ---------------------------------------------------------------------------
# (3) Embedded-mention case — 'worktree_setup' appears only inside a
#     non-ERROR log line (e.g. a completion notice).  Must pick EXIT {rc}.
# ---------------------------------------------------------------------------

def test_embedded_substring_in_non_error_line_picks_exit_prefix():
    combined = (
        "INFO: worktree_setup: completed successfully\n"
        "INFO: running shaped agent\n"
        "EXIT 2: subprocess exited with rc 2\n"
    )
    prefix, error_str = _classify_runner_failure(combined, rc=2)
    assert prefix == "EXIT 2", (
        f"Expected 'EXIT 2' for embedded substring, got {prefix!r}."
    )
    assert error_str == "EXIT 2"


# ---------------------------------------------------------------------------
# Edge: rc != 1, error line present
# ---------------------------------------------------------------------------

def test_worktree_error_with_nonzero_rc():
    combined = "ERROR: worktree_setup: permission denied\n"
    prefix, error_str = _classify_runner_failure(combined, rc=3)
    assert prefix == "ERROR: worktree_setup"


# ---------------------------------------------------------------------------
# Edge: empty combined string
# ---------------------------------------------------------------------------

def test_empty_combined_picks_exit_prefix():
    prefix, error_str = _classify_runner_failure("", rc=1)
    assert prefix == "EXIT 1"
