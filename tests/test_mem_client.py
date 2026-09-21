"""MemClient tests against a respx-mocked HTTP layer."""

import pytest
import respx
import httpx

from agents_core.mem_client import (
    MemClient,
    MemHTTPError,
    STORE_ATOMS,
    STORE_MACHINERY,
    validate_store,
)


BASE = "http://test-mem-server:8403"


@pytest.fixture
def client():
    return MemClient(base_url=BASE)


@respx.mock
def test_healthz(client):
    respx.get(f"{BASE}/healthz").mock(return_value=httpx.Response(
        200, json={"status": "ok", "db_path": "/data/memory/mem.db",
                   "row_counts": {"memories": 5, "memories_fts": 5},
                   "fts_integrity": {"in_sync": True, "divergence": 0}}
    ))
    result = client.healthz()
    assert result["status"] == "ok"
    assert result["row_counts"]["memories"] == 5


@respx.mock
def test_list(client):
    respx.get(f"{BASE}/v0/memories").mock(return_value=httpx.Response(
        200, json=[{"key": "a", "content": "b", "tags": "", "source": "", "created_at": "x", "updated_at": "x"}]
    ))
    rows = client.list()
    assert len(rows) == 1
    assert rows[0]["key"] == "a"


@respx.mock
def test_get(client):
    respx.get(f"{BASE}/v0/memories/foo/bar").mock(return_value=httpx.Response(
        200, json={"key": "foo/bar", "content": "c", "tags": "", "source": "", "created_at": "x", "updated_at": "x"}
    ))
    row = client.get("foo/bar")
    assert row["key"] == "foo/bar"


@respx.mock
def test_get_404_raises(client):
    respx.get(f"{BASE}/v0/memories/missing").mock(return_value=httpx.Response(
        404, json={"error": {"code": "not_found", "message": "not found"}}
    ))
    with pytest.raises(MemHTTPError) as exc:
        client.get("missing")
    assert exc.value.status_code == 404


@respx.mock
def test_set(client):
    respx.put(f"{BASE}/v0/memories/new/key").mock(return_value=httpx.Response(
        200, json={"key": "new/key", "content": "val", "tags": "a,b", "source": "s", "created_at": "x", "updated_at": "y"}
    ))
    row = client.set("new/key", "val", tags="a,b", source="s")
    assert row["key"] == "new/key"


@respx.mock
def test_delete(client):
    respx.delete(f"{BASE}/v0/memories/del/key").mock(return_value=httpx.Response(204))
    client.delete("del/key")  # should not raise


@respx.mock
def test_delete_404_raises(client):
    respx.delete(f"{BASE}/v0/memories/missing").mock(return_value=httpx.Response(
        404, json={"error": {"code": "not_found", "message": "not found"}}
    ))
    with pytest.raises(MemHTTPError) as exc:
        client.delete("missing")
    assert exc.value.status_code == 404


@respx.mock
def test_search(client):
    respx.get(f"{BASE}/v0/search").mock(return_value=httpx.Response(
        200, json=[{"key": "k", "content": "c", "tags": "", "source": "", "created_at": "x", "updated_at": "x", "rank": -1.0}]
    ))
    rows = client.search("docker")
    assert len(rows) == 1
    assert rows[0]["key"] == "k"


@respx.mock
def test_tags(client):
    respx.get(f"{BASE}/v0/tags").mock(return_value=httpx.Response(
        200, json=[{"tag": "alpha", "count": 3}]
    ))
    tags = client.tags()
    assert tags[0]["tag"] == "alpha"


@respx.mock
def test_stats(client):
    respx.get(f"{BASE}/v0/stats").mock(return_value=httpx.Response(
        200, json={"total_memories": 10, "unique_tags": 4, "db_size_bytes": 1024,
                   "db_path": "/d", "hostname": "h", "mode": "read-write"}
    ))
    s = client.stats()
    assert s["total_memories"] == 10


@respx.mock
def test_checkpoint(client):
    respx.post(f"{BASE}/v0/checkpoint").mock(return_value=httpx.Response(200, json={"ok": True}))
    result = client.checkpoint()
    assert result == {"ok": True}


@respx.mock
def test_bearer_header_sent():
    c = MemClient(base_url=BASE, token="mytoken")
    route = respx.get(f"{BASE}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    c.healthz()
    assert route.calls[0].request.headers["Authorization"] == "Bearer mytoken"


@respx.mock
def test_no_token_no_auth_header():
    c = MemClient(base_url=BASE, token="")
    route = respx.get(f"{BASE}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    c.healthz()
    assert "Authorization" not in route.calls[0].request.headers


@respx.mock
def test_error_raises_mem_http_error(client):
    respx.get(f"{BASE}/v0/stats").mock(return_value=httpx.Response(
        500, json={"error": {"code": "server_error", "message": "boom"}}
    ))
    with pytest.raises(MemHTTPError) as exc:
        client.stats()
    assert exc.value.status_code == 500


# ---------------------------------------------------------------------------
# --store plumbing (openclaw-memdb-influx-reader-v0, D2 / Files-changed)
# ---------------------------------------------------------------------------

def test_validate_store_known_values():
    assert validate_store(STORE_ATOMS) == STORE_ATOMS
    assert validate_store(STORE_MACHINERY) == STORE_MACHINERY
    # 'exhaust' is an alias for 'machinery' (the rescoped store name).
    assert validate_store("exhaust") == STORE_MACHINERY


def test_validate_store_rejects_typo():
    with pytest.raises(ValueError, match="unknown --store"):
        validate_store("machinery-typo")
    with pytest.raises(ValueError, match="unknown --store"):
        validate_store("")


@respx.mock
def test_set_accepts_store_machinery():
    """The --store machinery flag is accepted (rescoped MOOT at the HTTP
    layer — the server routes machine-state keys transparently). The flag
    is validated (loud ValueError on a typo) but does NOT change routing."""
    c = MemClient(base_url=BASE)
    route = respx.put(f"{BASE}/v0/memories/elevator/proposals/p1").mock(
        return_value=httpx.Response(200, json={
            "key": "elevator/proposals/p1", "content": "v", "tags": "",
            "source": "s", "created_at": "x", "updated_at": "y", "created": True,
        })
    )
    row = c.set("elevator/proposals/p1", "v", store=STORE_MACHINERY)
    assert row["key"] == "elevator/proposals/p1"


@respx.mock
def test_set_rejects_unknown_store():
    """An unknown --store value is rejected client-side (loud ValueError):
    a typo is loud, not silently ignored (fail-closed at the client edge)."""
    c = MemClient(base_url=BASE)
    with pytest.raises(ValueError, match="unknown --store"):
        c.set("foo/bar", "v", store="machinery-typo")
