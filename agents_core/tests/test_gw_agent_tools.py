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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / "large.txt"
            large_content = "x" * (GW_AGENT_TOOL_OUTPUT_CAP + 1000)
            test_file.write_text(large_content)

            executor = ReadFileExecutor(tmpdir)
            result = executor.execute({"path": "large.txt"})
            assert isinstance(result, str)
            assert len(result) <= GW_AGENT_TOOL_OUTPUT_CAP + len("\n…[truncated]")
            assert "…[truncated]" in result


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
