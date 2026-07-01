"""MCP stdio server exposing `mem` (MemClient) as typed tools for opencode.

Thin proxy only: each tool call-throughs to the corresponding MemClient
method. No new business logic, no re-implementation of search/ranking.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agents_core.mem_client import MemClient

mcp = FastMCP("lapis-mem")


def _client() -> MemClient:
    return MemClient()


@mcp.tool()
def mem_search(query: str, tag: str = "", limit: int = 20) -> list[dict]:
    """Search mem.db for memories matching a query, optionally filtered by tag."""
    with _client() as client:
        return client.search(query, tag=tag, limit=limit)


@mcp.tool()
def mem_get(key: str) -> dict:
    """Fetch a single memory by its exact key."""
    with _client() as client:
        return client.get(key)


@mcp.tool()
def mem_set(key: str, value: str, tags: str = "", source: str = "opencode") -> dict:
    """Write (create or update) a memory. Defaults source to 'opencode' for provenance."""
    with _client() as client:
        return client.set(key, value, tags=tags, source=source)


@mcp.tool()
def mem_list(tag: str = "", since: str = "", limit: int = 50) -> list[dict]:
    """List memories, optionally filtered by tag and/or a since timestamp."""
    with _client() as client:
        return client.list(tag=tag, since=since, limit=limit)


@mcp.tool()
def mem_tags() -> list[dict]:
    """List all known tags with their memory counts."""
    with _client() as client:
        return client.tags()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
