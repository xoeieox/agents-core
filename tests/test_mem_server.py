"""Endpoint-level tests for mem_server using FastAPI TestClient."""

import concurrent.futures
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents_core.mem_server import create_app
from agents_core.mem import MemoryStore


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_mem.db"


@pytest.fixture
def client(tmp_db):
    app = create_app(tmp_db)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def store(tmp_db):
    # used for direct DB assertions
    return MemoryStore(tmp_db)


# ---------------------------------------------------------------------------
# Healthz
# ---------------------------------------------------------------------------

def test_healthz_empty_db(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "db_path" in body
    assert body["row_counts"]["memories"] == 0
    assert body["row_counts"]["memories_fts"] == 0
    assert body["fts_integrity"]["in_sync"] is True
    assert body["fts_integrity"]["divergence"] == 0


def test_healthz_fts_divergence(tmp_db):
    """Manually delete from memories_fts to simulate trigger failure."""
    app = create_app(tmp_db)
    with TestClient(app) as c:
        c.put("/v0/memories/test/key", json={"content": "hello", "tags": "", "source": ""})
        c.put("/v0/memories/test/key2", json={"content": "world", "tags": "", "source": ""})

    # bypass triggers: directly delete from memories_fts
    conn = sqlite3.connect(str(tmp_db))
    conn.execute("DELETE FROM memories_fts WHERE key='test/key'")
    conn.commit()
    conn.close()

    with TestClient(create_app(tmp_db)) as c:
        resp = c.get("/healthz")
        body = resp.json()
        assert body["fts_integrity"]["in_sync"] is False
        assert body["fts_integrity"]["divergence"] != 0


# ---------------------------------------------------------------------------
# Slash-bearing keys (HIGH fold acceptance test)
# ---------------------------------------------------------------------------

def test_slash_bearing_key_round_trip(client):
    key = "pattern/docker-bind"
    put = client.put(f"/v0/memories/{key}", json={"content": "explicit port bind", "tags": "docker", "source": "test"})
    assert put.status_code == 200
    assert put.json()["key"] == key

    get = client.get(f"/v0/memories/{key}")
    assert get.status_code == 200
    assert get.json()["content"] == "explicit port bind"

    delete = client.delete(f"/v0/memories/{key}")
    assert delete.status_code == 204

    get2 = client.get(f"/v0/memories/{key}")
    assert get2.status_code == 404


def test_deep_slash_key(client):
    key = "decision/infra/docker/bind-ports"
    client.put(f"/v0/memories/{key}", json={"content": "always bind", "tags": "", "source": ""})
    resp = client.get(f"/v0/memories/{key}")
    assert resp.status_code == 200
    assert resp.json()["key"] == key


# ---------------------------------------------------------------------------
# Threadpool safety (check_same_thread=False fold)
# ---------------------------------------------------------------------------

def test_threadpool_safety(tmp_db):
    """N=8 concurrent requests must all succeed without sqlite3.ProgrammingError."""
    app = create_app(tmp_db)

    def make_request(i):
        with TestClient(app) as c:
            r = c.put(
                f"/v0/memories/thread/key{i}",
                json={"content": f"content {i}", "tags": "", "source": ""},
            )
            assert r.status_code == 200
            return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(make_request, range(8)))
    assert all(s == 200 for s in results)


# ---------------------------------------------------------------------------
# PUT → GET round-trip
# ---------------------------------------------------------------------------

def test_put_get_round_trip(client):
    client.put("/v0/memories/foo/bar", json={"content": "hello world", "tags": "a,b", "source": "pytest"})
    resp = client.get("/v0/memories/foo/bar")
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "hello world"
    assert body["source"] == "pytest"
    assert body["created_at"]
    assert body["updated_at"]


# ---------------------------------------------------------------------------
# FTS search round-trip
# ---------------------------------------------------------------------------

def test_put_then_search_finds_memory(client):
    client.put("/v0/memories/search/test", json={"content": "unique_fts_term_xyz", "tags": "", "source": ""})
    resp = client.get("/v0/search?q=unique_fts_term_xyz")
    assert resp.status_code == 200
    keys = [r["key"] for r in resp.json()]
    assert "search/test" in keys


def test_delete_then_search_not_found(client):
    client.put("/v0/memories/delete/me", json={"content": "removable_term_abc", "tags": "", "source": ""})
    client.delete("/v0/memories/delete/me")
    resp = client.get("/v0/search?q=removable_term_abc")
    assert resp.status_code == 200
    keys = [r["key"] for r in resp.json()]
    assert "delete/me" not in keys


# ---------------------------------------------------------------------------
# Upsert semantics
# ---------------------------------------------------------------------------

def test_upsert_preserves_created_at(client):
    client.put("/v0/memories/upsert/key", json={"content": "v1", "tags": "", "source": ""})
    r1 = client.get("/v0/memories/upsert/key").json()
    created_at_1 = r1["created_at"]

    client.put("/v0/memories/upsert/key", json={"content": "v2", "tags": "", "source": ""})
    r2 = client.get("/v0/memories/upsert/key").json()
    assert r2["created_at"] == created_at_1
    assert r2["content"] == "v2"


# ---------------------------------------------------------------------------
# tags string-to-list conversion
# ---------------------------------------------------------------------------

def test_tags_comma_string_stored_correctly(client, tmp_db):
    client.put("/v0/memories/foo", json={"content": "test", "tags": "alpha,beta", "source": ""})
    resp = client.get("/v0/memories/foo")
    assert resp.status_code == 200
    # the stored tags column should contain "alpha,beta" (sorted by MemoryStore.set)
    stored_tags = resp.json()["tags"]
    assert "alpha" in stored_tags
    assert "beta" in stored_tags


# ---------------------------------------------------------------------------
# Empty search query → 400
# ---------------------------------------------------------------------------

def test_search_empty_q_returns_400(client):
    resp = client.get("/v0/search?q=")
    assert resp.status_code == 400

def test_search_missing_q_returns_400(client):
    resp = client.get("/v0/search")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 404 cases
# ---------------------------------------------------------------------------

def test_get_unknown_returns_404(client):
    resp = client.get("/v0/memories/no/such/key")
    assert resp.status_code == 404

def test_delete_unknown_returns_404(client):
    resp = client.delete("/v0/memories/no/such/key")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Bearer-token middleware
# ---------------------------------------------------------------------------

def test_bearer_token_valid(tmp_db, monkeypatch):
    monkeypatch.setenv("MEM_BEARER_TOKEN", "secret123")
    app = create_app(tmp_db)
    with TestClient(app) as c:
        resp = c.get("/healthz", headers={"Authorization": "Bearer secret123"})
        assert resp.status_code == 200


def test_bearer_token_missing_returns_401(tmp_db, monkeypatch):
    monkeypatch.setenv("MEM_BEARER_TOKEN", "secret123")
    app = create_app(tmp_db)
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 401


def test_bearer_token_wrong_returns_401(tmp_db, monkeypatch):
    monkeypatch.setenv("MEM_BEARER_TOKEN", "secret123")
    app = create_app(tmp_db)
    with TestClient(app) as c:
        resp = c.get("/healthz", headers={"Authorization": "Bearer wrongtoken"})
        assert resp.status_code == 401


def test_no_bearer_token_env_no_auth_required(tmp_db, monkeypatch):
    monkeypatch.delenv("MEM_BEARER_TOKEN", raising=False)
    app = create_app(tmp_db)
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stats passthrough
# ---------------------------------------------------------------------------

def test_stats_shape(client):
    resp = client.get("/v0/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_memories" in body
    assert "unique_tags" in body
    assert "db_size_bytes" in body
    assert "db_path" in body
    assert "hostname" in body
    assert "mode" in body


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint(client):
    resp = client.post("/v0/checkpoint")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# ---------------------------------------------------------------------------
# Tags endpoint
# ---------------------------------------------------------------------------

def test_tags_endpoint(client):
    client.put("/v0/memories/t1", json={"content": "c", "tags": "alpha,beta", "source": ""})
    client.put("/v0/memories/t2", json={"content": "c", "tags": "alpha", "source": ""})
    resp = client.get("/v0/tags")
    assert resp.status_code == 200
    tags = {item["tag"]: item["count"] for item in resp.json()}
    assert tags.get("alpha") == 2
    assert tags.get("beta") == 1


# ---------------------------------------------------------------------------
# List endpoint
# ---------------------------------------------------------------------------

def test_list_all(client):
    client.put("/v0/memories/list/a", json={"content": "a", "tags": "", "source": ""})
    client.put("/v0/memories/list/b", json={"content": "b", "tags": "", "source": ""})
    resp = client.get("/v0/memories")
    assert resp.status_code == 200
    keys = [r["key"] for r in resp.json()]
    assert "list/a" in keys
    assert "list/b" in keys


def test_list_tag_filter(client):
    client.put("/v0/memories/tag/yes", json={"content": "y", "tags": "filtered", "source": ""})
    client.put("/v0/memories/tag/no", json={"content": "n", "tags": "other", "source": ""})
    resp = client.get("/v0/memories?tag=filtered")
    keys = [r["key"] for r in resp.json()]
    assert "tag/yes" in keys
    assert "tag/no" not in keys
