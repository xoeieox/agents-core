"""mcp_mem tests: each MCP tool must call-through to the right MemClient
method + args, and return exactly what MemClient returns."""

from unittest.mock import MagicMock, patch

from agents_core import mcp_mem


def _mock_client():
    client = MagicMock()
    client.__enter__.return_value = client
    client.search.return_value = [{"key": "a"}]
    client.get.return_value = {"key": "a", "content": "b"}
    client.set.return_value = {"key": "a", "content": "b"}
    client.list.return_value = [{"key": "a"}]
    client.tags.return_value = [{"tag": "t", "count": 1}]
    return client


def test_mem_search_calls_through():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        result = mcp_mem.mem_search("query text", tag="foo", limit=5)
    client.search.assert_called_once_with("query text", tag="foo", limit=5)
    assert result == [{"key": "a"}]


def test_mem_search_defaults():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        mcp_mem.mem_search("q")
    client.search.assert_called_once_with("q", tag="", limit=20)


def test_mem_get_calls_through():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        result = mcp_mem.mem_get("foo/bar")
    client.get.assert_called_once_with("foo/bar")
    assert result == {"key": "a", "content": "b"}


def test_mem_set_calls_through_with_opencode_source_default():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        result = mcp_mem.mem_set("foo/bar", "hello", tags="x,y")
    client.set.assert_called_once_with("foo/bar", "hello", tags="x,y", source="opencode")
    assert result == {"key": "a", "content": "b"}


def test_mem_set_explicit_source_overrides_default():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        mcp_mem.mem_set("foo/bar", "hello", source="claude-code")
    client.set.assert_called_once_with("foo/bar", "hello", tags="", source="claude-code")


def test_mem_list_calls_through():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        result = mcp_mem.mem_list(tag="t", since="2026-01-01", limit=10)
    client.list.assert_called_once_with(tag="t", since="2026-01-01", limit=10)
    assert result == [{"key": "a"}]


def test_mem_tags_calls_through():
    client = _mock_client()
    with patch.object(mcp_mem, "_client", return_value=client):
        result = mcp_mem.mem_tags()
    client.tags.assert_called_once_with()
    assert result == [{"tag": "t", "count": 1}]
