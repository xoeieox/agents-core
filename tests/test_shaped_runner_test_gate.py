"""Tests for the D1 positive-only test gate + D6 friction logging.

agents-core-local-fixer-harness-fix-v0: the legacy gate used the model's
LAST run_tests outcome; a pre-existing failure (or a non-existent test file)
as the last outcome discarded a 207-passing-test diff. The positive-only
gate instead checks that every test the model CREATED OR EDITED in this run
passes; pre-existing failures are logged (and witnessed via a friction mem
entry, D6) but do NOT block.

Fail-closed: when the model touched no tests at all (production-code-only
fix), the gate falls back to the legacy _tests_passed(last_test_outcome)
behavior so a fixer cannot merge untested production code by simply refusing
to write tests.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from agents_core import shaped_runner as sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transcript_with_test_write(path: str, cwd: str) -> list[dict]:
    """Build a transcript with a single successful write_file to a test file."""
    return [
        {
            "tool_name": "write_file",
            "arguments": {"path": path, "content": "def test_foo():\n    assert True\n"},
            "error": None,
        }
    ]


def _outcome(passed: int, failed: int, errors: int = 0, output_tail: str = "") -> dict:
    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "returncode": 1 if (failed or errors) else 0,
        "output_tail": output_tail,
    }


# ---------------------------------------------------------------------------
# D1: _collect_model_touched_tests — path resolution
# ---------------------------------------------------------------------------


def test_collect_model_touched_tests_relative_path(tmp_path):
    """A relative write_file target is resolved against cwd."""
    cwd = str(tmp_path)
    transcript = _transcript_with_test_write("tests/test_foo.py", cwd)
    touched = sr._collect_model_touched_tests(transcript, cwd)
    assert "tests/test_foo.py" in touched


def test_collect_model_touched_tests_absolute_path(tmp_path):
    """An absolute write_file target resolves to the same CWD-relative form."""
    cwd = str(tmp_path)
    abs_path = str(Path(cwd) / "tests" / "test_foo.py")
    transcript = _transcript_with_test_write(abs_path, cwd)
    touched = sr._collect_model_touched_tests(transcript, cwd)
    assert "tests/test_foo.py" in touched


def test_collect_model_touched_tests_apply_edit(tmp_path):
    """apply_edit calls to test files are also collected."""
    cwd = str(tmp_path)
    transcript = [
        {
            "tool_name": "apply_edit",
            "arguments": {
                "path": "tests/test_bar.py",
                "old_string": "x",
                "new_string": "y",
            },
            "error": None,
        }
    ]
    touched = sr._collect_model_touched_tests(transcript, cwd)
    assert "tests/test_bar.py" in touched


def test_collect_model_touched_tests_failed_write_excluded(tmp_path):
    """A write_file that errored did NOT actually touch the file."""
    cwd = str(tmp_path)
    transcript = [
        {
            "tool_name": "write_file",
            "arguments": {"path": "tests/test_foo.py", "content": "x"},
            "error": "permission denied",
        }
    ]
    touched = sr._collect_model_touched_tests(transcript, cwd)
    assert "tests/test_foo.py" not in touched


def test_collect_model_touched_tests_non_test_file_excluded(tmp_path):
    """A write to a non-test file is not collected."""
    cwd = str(tmp_path)
    transcript = [
        {
            "tool_name": "write_file",
            "arguments": {"path": "src/foo.py", "content": "x"},
            "error": None,
        }
    ]
    touched = sr._collect_model_touched_tests(transcript, cwd)
    assert not touched


def test_collect_model_touched_tests_empty_transcript(tmp_path):
    """An empty transcript yields an empty set (fail-closed fallback)."""
    cwd = str(tmp_path)
    touched = sr._collect_model_touched_tests([], cwd)
    assert not touched


# ---------------------------------------------------------------------------
# D1: _extract_failed_node_ids
# ---------------------------------------------------------------------------


def test_extract_failed_node_ids_basic():
    outcome = _outcome(
        passed=5,
        failed=1,
        output_tail="FAILED tests/test_foo.py::TestX::test_y\n",
    )
    nodes = sr._extract_failed_node_ids(outcome)
    assert "tests/test_foo.py::TestX::test_y" in nodes


def test_extract_failed_node_ids_error():
    outcome = _outcome(
        passed=0,
        failed=0,
        errors=1,
        output_tail="ERROR tests/test_bar.py\n",
    )
    nodes = sr._extract_failed_node_ids(outcome)
    assert "tests/test_bar.py" in nodes


def test_extract_failed_node_ids_error_at_setup():
    outcome = _outcome(
        passed=0,
        failed=0,
        errors=1,
        output_tail="ERROR at setup of tests/test_baz.py::test_setup\n",
    )
    nodes = sr._extract_failed_node_ids(outcome)
    assert "tests/test_baz.py::test_setup" in nodes


def test_extract_failed_node_ids_none_outcome():
    assert sr._extract_failed_node_ids(None) == []


def test_extract_failed_node_ids_no_failures():
    outcome = _outcome(passed=10, failed=0, output_tail="10 passed in 0.5s\n")
    assert sr._extract_failed_node_ids(outcome) == []


# ---------------------------------------------------------------------------
# D1: _resolve_test_path
# ---------------------------------------------------------------------------


def test_resolve_test_path_relative(tmp_path):
    cwd = str(tmp_path)
    assert sr._resolve_test_path(cwd, "tests/test_foo.py") == "tests/test_foo.py"


def test_resolve_test_path_absolute(tmp_path):
    cwd = str(tmp_path)
    abs_path = str(Path(cwd) / "tests" / "test_foo.py")
    assert sr._resolve_test_path(cwd, abs_path) == "tests/test_foo.py"


def test_resolve_test_path_outside_cwd(tmp_path):
    """A path outside cwd degrades to the raw string (exact-match)."""
    cwd = str(tmp_path)
    outside = "/somewhere/else/tests/test_foo.py"
    assert sr._resolve_test_path(cwd, outside) == outside


def test_resolve_test_path_empty(tmp_path):
    assert sr._resolve_test_path(str(tmp_path), "") == ""


# ---------------------------------------------------------------------------
# D1: gate logic (positive-only + fail-closed fallback)
#
# The gate logic lives inline in _run_local_fixer. We test the constituent
# helpers (which the gate composes) and the overall gate decision via a
# small re-implementation that mirrors the production logic exactly.
# ---------------------------------------------------------------------------


def _gate_decision(
    model_touched_tests: set[str],
    last_test_outcome: dict | None,
    cwd: str,
) -> bool:
    """Mirror of the inline gate logic in _run_local_fixer (D1).

    Positive-only gate: every test the model touched must pass.
    A touched test "fails" if a FAILED/ERROR node ID refers to it
    (file-level or node-level). Fail-closed: empty model_touched_tests
    -> legacy _tests_passed.
    """
    def _tests_passed(outcome):
        if not outcome:
            return False
        return (
            int(outcome.get("passed") or 0) > 0
            and int(outcome.get("failed") or 0) == 0
            and int(outcome.get("errors") or 0) == 0
        )

    if model_touched_tests:
        failed_node_ids = sr._extract_failed_node_ids(last_test_outcome)
        touched_failures = [
            t for t in model_touched_tests
            if any(
                n.split("::")[0] == t or n == t
                for n in failed_node_ids
            )
        ]
        if last_test_outcome is not None:
            passed_c = int(last_test_outcome.get("passed") or 0)
            if passed_c > 0 and not touched_failures:
                return True
        return False
    else:
        return _tests_passed(last_test_outcome)


def test_gate_positive_only_model_test_passes_preexisting_fails(tmp_path):
    """D1 core scenario: model's new test passes, pre-existing test fails.

    The gate MUST pass (vs. the legacy gate which would fail because the
    last outcome has a failure).
    """
    cwd = str(tmp_path)
    model_touched = {"tests/test_new.py"}
    # Last outcome: the model's test passed, but a pre-existing test failed.
    outcome = _outcome(
        passed=207,
        failed=1,
        output_tail="FAILED tests/test_preexisting.py::TestX::test_y - AssertionError\n",
    )
    assert _gate_decision(model_touched, outcome, cwd) is True


def test_gate_positive_only_model_test_fails(tmp_path):
    """If the model's own test fails, the gate must fail."""
    cwd = str(tmp_path)
    model_touched = {"tests/test_new.py"}
    outcome = _outcome(
        passed=206,
        failed=1,
        output_tail="FAILED tests/test_new.py::TestNew::test_new - AssertionError\n",
    )
    assert _gate_decision(model_touched, outcome, cwd) is False


def test_gate_fail_closed_empty_touched_legacy_pass(tmp_path):
    """Fail-closed: no tests touched + all tests pass -> legacy gate passes."""
    cwd = str(tmp_path)
    model_touched: set[str] = set()
    outcome = _outcome(passed=207, failed=0, output_tail="207 passed\n")
    assert _gate_decision(model_touched, outcome, cwd) is True


def test_gate_fail_closed_empty_touched_legacy_fail(tmp_path):
    """Fail-closed: no tests touched + a test fails -> legacy gate fails."""
    cwd = str(tmp_path)
    model_touched: set[str] = set()
    outcome = _outcome(
        passed=206,
        failed=1,
        output_tail="FAILED tests/test_foo.py::test_bar\n",
    )
    assert _gate_decision(model_touched, outcome, cwd) is False


def test_gate_fail_closed_empty_touched_no_outcome(tmp_path):
    """Fail-closed: no tests touched + no outcome -> legacy gate fails."""
    cwd = str(tmp_path)
    model_touched: set[str] = set()
    assert _gate_decision(model_touched, None, cwd) is False


def test_gate_positive_only_model_test_fails(tmp_path):
    """If the model's own test fails, the gate must fail."""
    cwd = str(tmp_path)
    model_touched = {"tests/test_new.py"}
    outcome = _outcome(
        passed=206,
        failed=1,
        output_tail="FAILED tests/test_new.py::TestNew::test_new\n",
    )
    assert _gate_decision(model_touched, outcome, cwd) is False


def test_gate_positive_only_zero_passed_fails(tmp_path):
    """A 0-passed run is never a pass, even if the model's tests weren't the ones that failed."""
    cwd = str(tmp_path)
    model_touched = {"tests/test_new.py"}
    outcome = _outcome(
        passed=0,
        failed=1,
        output_tail="FAILED tests/test_preexisting.py::TestX::test_y\n",
    )
    assert _gate_decision(model_touched, outcome, cwd) is False


def test_gate_positive_only_no_outcome_fails(tmp_path):
    """Model touched tests but no run_tests outcome -> gate fails."""
    cwd = str(tmp_path)
    model_touched = {"tests/test_new.py"}
    assert _gate_decision(model_touched, None, cwd) is False


# ---------------------------------------------------------------------------
# D6: _slugify, _error_signature, _write_friction_entry
# ---------------------------------------------------------------------------


def test_slugify_basic():
    assert (
        sr._slugify("tests/test_foo.py::TestX::test_y")
        == "tests-test-foo-py-testx-test-y"
    )


def test_slugify_stable():
    """Same node ID always produces the same slug (no date, no raw error)."""
    a = sr._slugify("tests/test_foo.py::TestX::test_y")
    b = sr._slugify("tests/test_foo.py::TestX::test_y")
    assert a == b


def test_error_signature_stable():
    """The error signature is stable across runs (exception type + node ID)."""
    node = "tests/test_foo.py::TestX::test_y"
    outcome = _outcome(
        passed=0,
        failed=1,
        output_tail="E   AssertionError: assert 1 == 2\nFAILED tests/test_foo.py::TestX::test_y\n",
    )
    sig = sr._error_signature(node, outcome)
    assert sig == "AssertionError@tests/test_foo.py::TestX::test_y"


def test_error_signature_unknown_when_no_exception():
    """When no exception type is visible, 'unknown' is used (still stable)."""
    node = "tests/test_foo.py::TestX::test_y"
    outcome = _outcome(passed=0, failed=1, output_tail="FAILED tests/test_foo.py::TestX::test_y\n")
    sig = sr._error_signature(node, outcome)
    assert sig == "unknown@tests/test_foo.py::TestX::test_y"


def test_error_signature_none_outcome():
    node = "tests/test_foo.py::TestX::test_y"
    sig = sr._error_signature(node, None)
    assert sig == "unknown@tests/test_foo.py::TestX::test_y"


def test_write_friction_entry_creates_new_entry(tmp_path, monkeypatch):
    """A pre-existing failure writes a friction mem entry with the correct key."""
    # Build a fake MemoryStore that records set() calls.
    set_calls: list[tuple[str, str, list[str]]] = []
    get_results: dict[str, dict | None] = {}

    class FakeMemoryStore:
        def __init__(self):
            pass

        def get(self, key):
            return get_results.get(key)

        def set(self, key, content, tags=None):
            set_calls.append((key, content, tags or []))

        def close(self):
            pass

    monkeypatch.setattr(
        "agents_core.mem.MemoryStore",
        FakeMemoryStore,
    )

    node = "tests/test_foo.py::TestX::test_y"
    sig = "AssertionError@tests/test_foo.py::TestX::test_y"
    sr._write_friction_entry(
        repo="agents-core",
        node_id=node,
        error_signature=sig,
        task_id="task-123",
        today="2026-08-23",
        log=lambda m: None,
    )

    assert len(set_calls) == 1
    key, content, tags = set_calls[0]
    # Key: friction/<repo>-<node-slug> (NO date in the key)
    expected_key = f"friction/agents-core-{sr._slugify(node)}"
    assert key == expected_key
    assert "2026" not in key  # no date in the key
    entry = json.loads(content)
    assert entry["status"] == "open"
    assert entry["test_node_id"] == node
    assert entry["error_signature"] == sig
    assert entry["first_seen"] == "2026-08-23"
    assert entry["last_seen"] == "2026-08-23"
    assert entry["first_task_id"] == "task-123"
    assert entry["last_task_id"] == "task-123"
    assert "friction" in tags


def test_write_friction_entry_dedup_open(tmp_path, monkeypatch):
    """A repeat of the same friction (status: open) does NOT create a duplicate.

    It updates last_seen/last_task_id on the existing entry.
    """
    set_calls: list[tuple[str, str, list[str]]] = []

    existing_content = json.dumps(
        {
            "status": "open",
            "test_node_id": "tests/test_foo.py::TestX::test_y",
            "error_signature": "AssertionError@tests/test_foo.py::TestX::test_y",
            "first_seen": "2026-08-20",
            "last_seen": "2026-08-20",
            "first_task_id": "task-100",
            "last_task_id": "task-100",
        }
    )
    get_results: dict[str, dict | None] = {}

    class FakeMemoryStore:
        def __init__(self):
            pass

        def get(self, key):
            if key in get_results:
                return get_results[key]
            return None

        def set(self, key, content, tags=None):
            set_calls.append((key, content, tags or []))
            get_results[key] = {"content": content}

        def close(self):
            pass

    # Pre-populate the existing entry.
    node = "tests/test_foo.py::TestX::test_y"
    key = f"friction/agents-core-{sr._slugify(node)}"
    get_results[key] = {"content": existing_content}

    monkeypatch.setattr(
        "agents_core.mem.MemoryStore",
        FakeMemoryStore,
    )

    sr._write_friction_entry(
        repo="agents-core",
        node_id=node,
        error_signature="AssertionError@tests/test_foo.py::TestX::test_y",
        task_id="task-200",
        today="2026-08-23",
        log=lambda m: None,
    )

    # Exactly one set() call (the dedup update, not a new entry).
    assert len(set_calls) == 1
    updated_key, content, _ = set_calls[0]
    assert updated_key == key
    entry = json.loads(content)
    # Status stays open, last_seen/last_task_id updated, first_seen/first_task_id preserved.
    assert entry["status"] == "open"
    assert entry["first_seen"] == "2026-08-20"
    assert entry["last_seen"] == "2026-08-23"
    assert entry["first_task_id"] == "task-100"
    assert entry["last_task_id"] == "task-200"


def test_write_friction_entry_recurred_resolved_to_open(tmp_path, monkeypatch):
    """A friction that was resolved but recurred flips back to status: open."""
    set_calls: list[tuple[str, str, list[str]]] = []

    existing_content = json.dumps(
        {
            "status": "resolved",
            "test_node_id": "tests/test_foo.py::TestX::test_y",
            "error_signature": "AssertionError@tests/test_foo.py::TestX::test_y",
            "first_seen": "2026-08-20",
            "last_seen": "2026-08-21",
            "first_task_id": "task-100",
            "last_task_id": "task-101",
        }
    )
    get_results: dict[str, dict | None] = {}

    class FakeMemoryStore:
        def __init__(self):
            pass

        def get(self, key):
            return get_results.get(key)

        def set(self, key, content, tags=None):
            set_calls.append((key, content, tags or []))
            get_results[key] = {"content": content}

        def close(self):
            pass

    node = "tests/test_foo.py::TestX::test_y"
    key = f"friction/agents-core-{sr._slugify(node)}"
    get_results[key] = {"content": existing_content}

    monkeypatch.setattr(
        "agents_core.mem.MemoryStore",
        FakeMemoryStore,
    )

    sr._write_friction_entry(
        repo="agents-core",
        node_id=node,
        error_signature="AssertionError@tests/test_foo.py::TestX::test_y",
        task_id="task-300",
        today="2026-08-23",
        log=lambda m: None,
    )

    assert len(set_calls) == 1
    _, content, _ = set_calls[0]
    entry = json.loads(content)
    # Status flipped back to open (recurred).
    assert entry["status"] == "open"
    assert entry["last_seen"] == "2026-08-23"
    assert entry["last_task_id"] == "task-300"


def test_write_friction_entry_never_raises(tmp_path, monkeypatch):
    """A friction-write failure is swallowed (never blocks the gate)."""

    class BrokenMemoryStore:
        def __init__(self):
            raise RuntimeError("mem store unavailable")

    monkeypatch.setattr(
        "agents_core.mem.MemoryStore",
        BrokenMemoryStore,
    )

    # Must not raise.
    sr._write_friction_entry(
        repo="agents-core",
        node_id="tests/test_foo.py::TestX::test_y",
        error_signature="AssertionError@tests/test_foo.py::TestX::test_y",
        task_id="task-123",
        today="2026-08-23",
        log=lambda m: None,
    )