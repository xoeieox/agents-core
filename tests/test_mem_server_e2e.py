"""End-to-end tests: real temp SQLite DB + local uvicorn subprocess + MemClient."""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from agents_core.mem import MemoryStore
from agents_core.mem_client import MemClient, MemHTTPError


E2E_PORT = 18403


@pytest.fixture(scope="module")
def e2e_env(tmp_path_factory):
    """Spin up mem-server subprocess against a temp DB. Yield (client, store)."""
    db_path = tmp_path_factory.mktemp("e2e") / "mem_e2e.db"

    # D4 write-guard (mem-hygiene-automation-v0) is default-ON: the e2e
    # writes carry test-provenance sources (pytest-e2e), so the server
    # subprocess runs with MEM_ALLOW_TEST_WRITE=1. The guard mapping itself
    # is covered by test_mem_hygiene_server.py.
    proc = subprocess.Popen(
        [sys.executable, "-m", "agents_core.mem_server"],
        env={
            "MEM_DB_PATH": str(db_path),
            "MEM_BIND_HOST": "127.0.0.1",
            "MEM_BIND_PORT": str(E2E_PORT),
            "MEM_LOG_LEVEL": "error",
            "MEM_ALLOW_TEST_WRITE": "1",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # wait for server to start
    base_url = f"http://127.0.0.1:{E2E_PORT}"
    import httpx
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/healthz", timeout=1.0)
            if resp.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"Server did not start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}")

    client = MemClient(base_url=base_url, timeout=5.0)
    store = MemoryStore(db_path)
    yield client, store, db_path

    proc.kill()
    proc.wait()
    client.close()
    store.close()


def test_e2e_healthz(e2e_env):
    client, store, db_path = e2e_env
    h = client.healthz()
    assert h["status"] == "ok"
    assert str(db_path) in h["db_path"]


def test_e2e_set_get_equivalence(e2e_env):
    client, store, db_path = e2e_env
    client.set("e2e/key1", "hello e2e", tags="e2e,test", source="pytest-e2e")

    via_http = client.get("e2e/key1")
    via_direct = store.get("e2e/key1")

    assert via_http["content"] == via_direct["content"]
    assert via_http["key"] == via_direct["key"]
    assert via_http["tags"] == via_direct["tags"]


def test_e2e_search_finds_memory(e2e_env):
    client, store, db_path = e2e_env
    client.set("e2e/searchable", "unique_e2e_search_term_qwerty", tags="e2e", source="pytest-e2e")
    results = client.search("unique_e2e_search_term_qwerty")
    keys = [r["key"] for r in results]
    assert "e2e/searchable" in keys


def test_e2e_delete_not_in_search(e2e_env):
    client, store, db_path = e2e_env
    client.set("e2e/deletable", "to_be_deleted_e2e_term", tags="", source="")
    client.delete("e2e/deletable")
    results = client.search("to_be_deleted_e2e_term")
    keys = [r["key"] for r in results]
    assert "e2e/deletable" not in keys


def test_e2e_slash_key(e2e_env):
    client, store, db_path = e2e_env
    key = "pattern/docker-bind/e2e"
    client.set(key, "explicit bind", tags="docker", source="pytest-e2e")
    row = client.get(key)
    assert row["key"] == key
    client.delete(key)
    with pytest.raises(MemHTTPError) as exc:
        client.get(key)
    assert exc.value.status_code == 404


def test_e2e_set_then_direct_read(e2e_env):
    client, store, db_path = e2e_env
    client.set("e2e/direct", "direct check content", tags="", source="pytest-e2e")
    direct = store.get("e2e/direct")
    assert direct is not None
    assert direct["content"] == "direct check content"


def test_e2e_stats(e2e_env):
    client, store, db_path = e2e_env
    s = client.stats()
    assert s["total_memories"] > 0
    assert "db_path" in s
