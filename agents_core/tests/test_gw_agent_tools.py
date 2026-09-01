"""Unit tests for gw_agent tool executors.

Tests the security hardening: path confinement, metacharacter rejection,
output capping, and read-only allowlists.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from agents_core.gw_agent import (
    ReadFileExecutor,
    GrepExecutor,
    GitExecutor,
    MemExecutor,
    GW_AGENT_TOOL_OUTPUT_CAP,
)


def _has_ripgrep():
    """Check if ripgrep (rg) is installed."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, timeout=1, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


class TestReadFileExecutor:
    def test_read_file_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "test.txt"
            test_file.write_text("line 1\nline 2\nline 3\n")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "test.txt"})
            assert result == "line 1\nline 2\nline 3"

    def test_read_file_with_line_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "test.txt"
            test_file.write_text("line 1\nline 2\nline 3\nline 4\n")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "test.txt", "start_line": 2, "end_line": 3})
            assert result == "line 2\nline 3"

    def test_read_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "nonexistent.txt"})
            assert isinstance(result, dict)
            assert "error" in result
            assert "not found" in result["error"]

    def test_read_file_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "../../etc/passwd"})
            assert isinstance(result, dict)
            assert "error" in result
            assert "outside cwd" in result["error"]

    def test_read_file_absolute_path_outside_cwd_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "/etc/passwd"})
            assert isinstance(result, dict)
            assert "error" in result
            assert "outside cwd" in result["error"]

    def test_read_file_output_capped(self):
        # fixers-harness-staged-v0 (S4): the truncation marker is enriched to
        # `…[truncated at {cap} chars; file has {N} lines, showing lines
        # {start}-{end}]` so a model can page to the end in one jump.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "large.txt"
            # One long line (no embedded newlines) so the slice is a single
            # line and the marker's "showing lines 1-1" range is deterministic.
            large_content = "x" * (GW_AGENT_TOOL_OUTPUT_CAP + 1000)
            test_file.write_text(large_content)

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "large.txt"})
            assert isinstance(result, str)
            assert "…[truncated at " in result
            assert f"{GW_AGENT_TOOL_OUTPUT_CAP} chars" in result
            assert "file has 1 lines" in result
            assert "showing lines 1-1" in result
            # The body is capped at the tool output cap (the marker is the only
            # excess).
            assert len(result) <= GW_AGENT_TOOL_OUTPUT_CAP + 200

    def test_empty_slice_past_eof(self):
        # fixers-harness-staged-v0 (S4): reading past EOF returns an explicit
        # `(empty slice: ...)` marker naming the file's line count, not "".
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "small.txt"
            test_file.write_text("l1\nl2\nl3\nl4\nl5\n")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute(
                {"path": "small.txt", "start_line": 100, "end_line": 200}
            )
            assert isinstance(result, str)
            assert result.startswith("(empty slice:")
            assert "file has 5 lines" in result
            assert "is outside it" in result

    def test_zero_byte_file_marker(self):
        # fixers-harness-staged-v0 (S4): a zero-byte file returns a distinct
        # `(file is empty: ...)` marker.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "zero.txt").write_text("")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "zero.txt"})
            assert isinstance(result, str)
            assert result.startswith("(file is empty")

    def test_empty_slice_marker_no_path(self):
        # fixers-harness-staged-v0 (S4): the empty-slice marker interpolates
        # INTEGERS ONLY - never the file path (or file content).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "small.txt"
            test_file.write_text("l1\nl2\nl3\nl4\nl5\n")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute(
                {"path": "small.txt", "start_line": 100, "end_line": 200}
            )
            assert isinstance(result, str)
            assert str(tmpdir) not in result

    def test_start_gt_end_inverted(self):
        # fixers-harness-staged-v0 (S4): a `start > end` range is inverted
        # (single-line-read contract), not treated as an empty slice.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "small.txt"
            test_file.write_text("l1\nl2\nl3\nl4\nl5\n")

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute(
                {"path": "small.txt", "start_line": 5, "end_line": 2}
            )
            assert isinstance(result, str)
            assert not result.startswith("(empty slice:")
            # Inverted range 5..2 -> 2..5: the file's lines 2 through 5.
            for line in ("l2", "l3", "l4", "l5"):
                assert line in result
            assert "l1" not in result


class TestGitExecutor:
    def test_git_allowed_subcommand_log(self):
        # Use the current repo as a known git repository
        executor = GitExecutor("/srv/git/agents-core-working")
        # log should be allowed; will succeed if in a git repo
        result = executor.execute({"args": "log --oneline -1"})
        # Either a string result or error dict, but NOT "not allowed"
        if isinstance(result, dict):
            assert "not allowed" not in result.get("error", "")
        else:
            assert isinstance(result, str)

    def test_git_disallowed_subcommand_commit_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "commit -m test"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_git_disallowed_subcommand_push_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "push origin main"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_git_disallowed_subcommand_checkout_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "checkout main"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_git_metacharacter_semicolon_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "log; cat /etc/passwd"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "shell metacharacters" in result["error"]

    def test_git_metacharacter_pipe_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "log | grep foo"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "shell metacharacters" in result["error"]

    def test_git_metacharacter_dollar_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": "log $(cat /etc/passwd)"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "shell metacharacters" in result["error"]

    def test_git_no_subcommand_rejected(self):
        executor = GitExecutor("/tmp")
        result = executor.execute({"args": ""})
        assert isinstance(result, dict)
        assert "error" in result
        assert "no git subcommand" in result["error"]


class TestMemExecutor:
    def test_mem_search_allowed(self):
        executor = MemExecutor()
        # This will likely fail (mem CLI may not be available), but it proves the action is allowed
        result = executor.execute({"action": "search", "query": "test"})
        # Either a valid result or an error dict, but NOT "action not allowed"
        if isinstance(result, dict):
            assert "action not allowed" not in result.get("error", "")

    def test_mem_get_allowed(self):
        executor = MemExecutor()
        result = executor.execute({"action": "get", "query": "test_key"})
        if isinstance(result, dict):
            assert "action not allowed" not in result.get("error", "")

    def test_mem_set_disallowed(self):
        executor = MemExecutor()
        result = executor.execute({"action": "set", "query": "key=value"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not allowed" in result["error"]

    def test_mem_delete_disallowed(self):
        executor = MemExecutor()
        result = executor.execute({"action": "delete", "query": "key"})
        assert isinstance(result, dict)
        assert "error" in result
        assert "not allowed" in result["error"]


class TestGrepExecutor:
    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_finds_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file1.txt").write_text("hello world")
            (tmpdir_path / "file2.txt").write_text("goodbye")

            executor = GrepExecutor(tmpdir)
            result = executor.execute({"pattern": "hello"})
            assert isinstance(result, str)
            assert "file1.txt" in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_no_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file1.txt").write_text("hello world")

            executor = GrepExecutor(tmpdir)
            result = executor.execute({"pattern": "nomatch"})
            assert result == "(no matches)"

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_output_capped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Create many files with the pattern
            for i in range(200):
                (tmpdir_path / f"file{i}.txt").write_text("pattern here")

            executor = GrepExecutor(tmpdir)
            result = executor.execute({"pattern": "pattern"})
            if isinstance(result, str) and len(result) > GW_AGENT_TOOL_OUTPUT_CAP:
                assert "…[truncated]" in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_path_glob_scoped_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "config").mkdir()
            (tmpdir_path / "docs").mkdir()
            (tmpdir_path / "config" / "gw-topology.yaml").write_text("slot1-laguna: true\n")
            (tmpdir_path / "docs" / "other.md").write_text("slot1-laguna\n")

            executor = GrepExecutor(tmpdir)
            result = executor.execute(
                {"pattern": "slot1-laguna", "path_glob": "config/*"}
            )
            assert isinstance(result, str)
            assert result != "(no matches)"
            assert "config/gw-topology.yaml" in result
            assert "docs/other.md" not in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_path_glob_nested_finds_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "config" / "nested").mkdir(parents=True)
            (tmpdir_path / "config" / "nested" / "deep.yaml").write_text("needle-xyz\n")
            (tmpdir_path / "config" / "sibling.txt").write_text("needle-xyz\n")

            executor = GrepExecutor(tmpdir)
            result = executor.execute(
                {"pattern": "needle-xyz", "path_glob": "**/*.yaml"}
            )
            assert isinstance(result, str)
            assert result != "(no matches)"
            assert "config/nested/deep.yaml" in result
            assert "sibling.txt" not in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_line_numbers_emits_file_line_content(self):
        # fixers-harness-staged-v0 (S3): line_numbers=True carries `-n` -
        # output is `file:line:content` so a model can jump to a function.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file.txt").write_text("a\nb\nneedle\n")

            executor = GrepExecutor(tmpdir, line_numbers=True)
            result = executor.execute({"pattern": "needle"})
            assert isinstance(result, str)
            assert "file.txt:3:" in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_legacy_path_no_line_numbers(self):
        # fixers-harness-staged-v0 (S3): the legacy default (`line_numbers=False`)
        # stays the exact `rg -l` path - file names only, no line numbers.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file.txt").write_text("a\nb\nneedle\n")

            executor = GrepExecutor(tmpdir)
            result = executor.execute({"pattern": "needle"})
            assert isinstance(result, str)
            assert "file.txt" in result
            assert ":3:" not in result

    @pytest.mark.skipif(not _has_ripgrep(), reason="ripgrep not installed")
    def test_grep_pattern_leading_dash_after_e(self):
        # fixers-harness-staged-v0 (S3): the pattern is pinned behind `-e`, so
        # a pattern starting with `-` is never parsed as a flag.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "file.txt").write_text("x\n-leading-dash\n")

            executor = GrepExecutor(tmpdir)
            result = executor.execute({"pattern": "-leading-dash"})
            assert isinstance(result, str)
            assert result != "(no matches)"
            assert "file.txt" in result


def test_fixer_result_result_text_default():
    # fixers-harness-staged-v0 (S0): the additive `result_text` key on the
    # FixerResult dict defaults to "" and is ignored by existing consumers.
    from agents_core.gw_agent import _build_fixer_result

    r = _build_fixer_result(cwd=".", transcript=[], concluded=True)
    assert r["result_text"] == ""
