"""Unit tests for OpenPrsExecutor in gw_agent.

Tests cover basic PR listing, file inclusion, diff parsing,
error handling, and signal preservation (no silent truncation).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import OpenPrsExecutor


@pytest.fixture
def executor():
    """Create an OpenPrsExecutor instance."""
    return OpenPrsExecutor()


def test_open_prs_executor_requires_repo_parameter(executor):
    """Test that repo parameter is required."""
    result = executor.execute({})
    assert isinstance(result, dict)
    assert "error" in result
    assert "repo parameter is required" in result["error"]


def test_open_prs_executor_basic_pr_listing(executor):
    """Test basic open PR listing without file details."""
    mock_prs = [
        {
            "number": 1,
            "title": "Fix bug in feature A",
            "head": {"ref": "feature/bugfix"},
            "base": {"ref": "main"},
            "body": "This PR fixes the bug in feature A",
            "updated_at": "2026-06-19T10:00:00Z",
        },
        {
            "number": 2,
            "title": "Add new feature B",
            "head": {"ref": "feature/new-b"},
            "base": {"ref": "main"},
            "body": "This PR adds feature B",
            "updated_at": "2026-06-19T11:00:00Z",
        },
    ]

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        result = executor.execute({"repo": "agents-core"})

    assert isinstance(result, str)
    prs = json.loads(result)
    assert len(prs) == 2
    assert prs[0]["number"] == 1
    assert prs[0]["title"] == "Fix bug in feature A"
    assert prs[0]["head"] == "feature/bugfix"
    assert prs[0]["base"] == "main"
    assert prs[0]["body"] == "This PR fixes the bug in feature A"
    assert prs[1]["number"] == 2


def test_open_prs_executor_body_truncation_signaling(executor):
    """Test that long body is snipped with … marker."""
    long_body = "x" * 300  # Longer than 200 char limit
    mock_prs = [
        {
            "number": 1,
            "title": "Test",
            "head": {"ref": "branch"},
            "base": {"ref": "main"},
            "body": long_body,
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        result = executor.execute({"repo": "agents-core"})

    prs = json.loads(result)
    assert len(prs[0]["body"]) <= 201  # 200 + "…"
    assert prs[0]["body"].endswith("…")


def test_open_prs_executor_with_files_using_files_endpoint(executor):
    """Test that when with_files=True, changed_files are included."""
    mock_prs = [
        {
            "number": 1,
            "title": "Fix bug",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "body": "Description",
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    mock_files = [
        {"filename": "agents_core/gw_agent.py"},
        {"filename": "tests/test_gw_agent.py"},
    ]

    def mock_get(url, headers=None, timeout=None, **kwargs):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = mock_files
        m.raise_for_status = MagicMock()
        return m

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("httpx.get", side_effect=mock_get):
            result = executor.execute({"repo": "agents-core", "with_files": True})

    prs = json.loads(result)
    assert "changed_files" in prs[0]
    assert prs[0]["changed_files"] == ["agents_core/gw_agent.py", "tests/test_gw_agent.py"]


def test_open_prs_executor_diff_parsing_with_spaces(executor):
    """Test that file paths with spaces are preserved in diff parsing."""
    mock_prs = [
        {
            "number": 1,
            "title": "Fix bug",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "body": "Description",
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    # Simulate diff with file containing spaces
    mock_diff = """diff --git a/my file.txt b/my file.txt
index abc123..def456 100644
--- a/my file.txt
+++ b/my file.txt
@@ -1 +1 @@
-old content
+new content
"""

    def mock_get(url, headers=None, timeout=None, **kwargs):
        # Return 404 for files endpoint (fall back to diff)
        m = MagicMock()
        m.status_code = 404
        m.raise_for_status.side_effect = Exception("404")
        return m

    def mock_get_pr_diff(repo, pr_number, owner=None):
        return mock_diff

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("httpx.get", side_effect=mock_get):
            with patch("agents_core.forgejo.get_pr_diff", side_effect=mock_get_pr_diff):
                result = executor.execute({"repo": "agents-core", "with_files": True})

    prs = json.loads(result)
    assert "changed_files" in prs[0]
    # The file path with spaces should be intact
    assert "my file.txt" in prs[0]["changed_files"]


def test_open_prs_executor_changed_files_capped_at_50(executor):
    """Test that changed_files list is capped at 50 entries with truncation marker."""
    mock_prs = [
        {
            "number": 1,
            "title": "Large PR",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "body": "Description",
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    # Create a mock files list with 60 files
    mock_files = [{"filename": f"file_{i}.txt"} for i in range(60)]

    def mock_get(url, headers=None, timeout=None, **kwargs):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = mock_files
        m.raise_for_status = MagicMock()
        return m

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("httpx.get", side_effect=mock_get):
            result = executor.execute({"repo": "agents-core", "with_files": True})

    prs = json.loads(result)
    changed_files = prs[0]["changed_files"]
    assert len(changed_files) <= 51  # 50 files + truncation marker
    assert any("…" in f for f in changed_files[-1:])  # Last entry marked as truncated


def test_open_prs_executor_no_silent_signal_loss(executor):
    """Test that all PRs are returned with complete number/title/head/base fields."""
    mock_prs = [
        {
            "number": 1,
            "title": "PR 1",
            "head": {"ref": "branch1"},
            "base": {"ref": "main"},
            "body": "x" * 300,  # Long body that gets snipped
            "updated_at": "2026-06-19T10:00:00Z",
        },
        {
            "number": 2,
            "title": "PR 2",
            "head": {"ref": "branch2"},
            "base": {"ref": "main"},
            "body": "Short",
            "updated_at": "2026-06-19T11:00:00Z",
        },
    ]

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        result = executor.execute({"repo": "agents-core"})

    prs = json.loads(result)
    # All PRs present
    assert len(prs) == 2
    # All key fields present and unchanged
    assert prs[0]["number"] == 1
    assert prs[0]["title"] == "PR 1"
    assert prs[0]["head"] == "branch1"
    assert prs[0]["base"] == "main"
    assert prs[1]["number"] == 2
    assert prs[1]["title"] == "PR 2"


def test_open_prs_executor_forgejo_error_handling(executor):
    """Test that Forgejo errors return error dict, not exception."""
    with patch("agents_core.forgejo.get_open_prs", side_effect=Exception("Connection timeout")):
        result = executor.execute({"repo": "agents-core"})

    assert isinstance(result, dict)
    assert "error" in result
    assert "Connection timeout" in result["error"]


def test_open_prs_executor_diff_parsing_multiple_files(executor):
    """Test diff parsing with multiple files."""
    mock_prs = [
        {
            "number": 1,
            "title": "Multi-file change",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "body": "Description",
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    mock_diff = """diff --git a/file1.py b/file1.py
index abc..def 100644
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new
diff --git a/file2.txt b/file2.txt
index 123..456 100644
--- a/file2.txt
+++ b/file2.txt
@@ -1 +1 @@
-old
+new
"""

    def mock_get(url, headers=None, timeout=None, **kwargs):
        m = MagicMock()
        m.status_code = 404
        m.raise_for_status.side_effect = Exception("404")
        return m

    def mock_get_pr_diff(repo, pr_number, owner=None):
        return mock_diff

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("httpx.get", side_effect=mock_get):
            with patch("agents_core.forgejo.get_pr_diff", side_effect=mock_get_pr_diff):
                result = executor.execute({"repo": "agents-core", "with_files": True})

    prs = json.loads(result)
    assert "changed_files" in prs[0]
    assert "file1.py" in prs[0]["changed_files"]
    assert "file2.txt" in prs[0]["changed_files"]


def test_open_prs_executor_empty_pr_list(executor):
    """Test handling of empty PR list."""
    with patch("agents_core.forgejo.get_open_prs", return_value=[]):
        result = executor.execute({"repo": "agents-core"})

    assert isinstance(result, str)
    prs = json.loads(result)
    assert prs == []


def test_open_prs_executor_read_only_semantics(executor):
    """Verify that executor never calls forgejo mutating functions."""
    mock_prs = [
        {
            "number": 1,
            "title": "Test",
            "head": {"ref": "branch"},
            "base": {"ref": "main"},
            "body": "Test",
            "updated_at": "2026-06-19T10:00:00Z",
        },
    ]

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("agents_core.forgejo.merge_pr") as mock_merge:
            with patch("agents_core.forgejo.create_pr") as mock_create:
                with patch("agents_core.forgejo.add_comment") as mock_comment:
                    with patch("agents_core.forgejo.close_pr") as mock_close:
                        result = executor.execute({"repo": "agents-core"})

                        # Verify no mutating operations were called
                        mock_merge.assert_not_called()
                        mock_create.assert_not_called()
                        mock_comment.assert_not_called()
                        mock_close.assert_not_called()
                        # Verify we got a result
                        assert result is not None


def test_open_prs_executor_large_pr_list_no_output_cap_corruption(executor):
    """Test that large PR lists don't corrupt JSON via invalid truncation.

    This test simulates 20 PRs each with 50 changed_files, which would exceed
    the 8192 byte output cap if the post-serialization string truncation were
    still in place. Verifies that:
    1. The result is valid JSON (json.loads() succeeds)
    2. All PRs are present in the output
    3. No PR is silently dropped from the list
    4. No signal fields (number, title, head, base) are truncated
    """
    # Create 20 mock PRs, each with 50 changed files
    mock_prs = []
    for pr_num in range(1, 21):
        mock_prs.append(
            {
                "number": pr_num,
                "title": f"Large PR {pr_num}",
                "head": {"ref": f"feature/large-{pr_num}"},
                "base": {"ref": "main"},
                "body": f"This is a large PR that touches many files. PR number {pr_num}.",
                "updated_at": f"2026-06-19T{10 + pr_num:02d}:00:00Z",
            }
        )

    # Create 50+ file entries per PR for the mock files endpoint
    mock_files = [{"filename": f"file_{i:03d}.py"} for i in range(55)]

    def mock_get(url, headers=None, timeout=None, **kwargs):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = mock_files
        m.raise_for_status = MagicMock()
        return m

    with patch("agents_core.forgejo.get_open_prs", return_value=mock_prs):
        with patch("httpx.get", side_effect=mock_get):
            result = executor.execute({"repo": "agents-core", "with_files": True})

    # Verify result is valid JSON (this would fail if truncation corrupts it)
    assert isinstance(result, str)
    prs = json.loads(result)

    # Verify all 20 PRs are present (no silent PR drops)
    assert len(prs) == 20

    # Verify each PR has complete signal fields and files
    for i, pr in enumerate(prs):
        assert pr["number"] == i + 1
        assert pr["title"] == f"Large PR {i + 1}"
        assert pr["head"] == f"feature/large-{i + 1}"
        assert pr["base"] == "main"
        assert "changed_files" in pr
        # Each PR should have 50 files + truncation marker (since we provide 55 files)
        assert len(pr["changed_files"]) == 51
        # Last entry should be the truncation marker
        assert "…(+5 more)" in pr["changed_files"][-1]
