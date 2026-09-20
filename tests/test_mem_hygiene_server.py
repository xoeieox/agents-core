"""Endpoint-level tests for the mem hygiene surface
(mem-hygiene-automation-v0) — the server-side run/ageout/list/restore
endpoints and the D4 write-guard 4xx mapping.

Hermetic: tmp_path fixture dbs + FastAPI TestClient. The config is a
tmp_path JSON pair pointed at via MEM_HYGIENE_CONFIG (read per-call).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents_core.mem import MemoryStore
from agents_core.mem_server import create_app

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_pair(tmp_path: Path):
    """Write the config pair into tmp_path and return the main config path."""
    (tmp_path / "dead_producers.json").write_text(json.dumps({
        "version": 1,
        "dead_sources": {"elevator/": ["elevator_scheduler"]},
    }))
    cfg = tmp_path / "mem_hygiene.json"
    cfg.write_text(json.dumps({
        "registry": "dead_producers.json",
        "allowlist": ["elevator/"],
        "dead_stream_age_days": 30,
        "batch_cap": 5000,
        "rollback_window_days": 14,
    }))
    return cfg


def _seed_dead_stream(tmp_path: Path, n: int = 4) -> None:
    """Seed dead-stream rows into mem.db directly (bypassing store.set()
    which would route elevator/ to the exhaust twin). The quarantine
    runner classifies the mem.db `memories` table, so the fixture must
    land there — the spec's own counts (elevator/ 10,975) are mem.db rows.
    """
    from agents_core import mem_exhaust
    store = MemoryStore(db_path=tmp_path / "mem.db")
    ts = (NOW - timedelta(days=45)).isoformat()
    for i in range(n):
        key = f"elevator/row-{i}"
        if mem_exhaust.route_to_exhaust(key):
            store._conn.execute(
                "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, f"machine state {i}", "machine", "elevator_scheduler", ts, ts),
            )
        else:
            store.set(key, f"machine state {i}", tags=["machine"],
                      source="elevator_scheduler")
            store._conn.execute(
                "UPDATE memories SET created_at=?, updated_at=? WHERE key=?",
                (ts, ts, key),
            )
    store._conn.commit()
    store.close()


@pytest.fixture
def client(tmp_path, cfg_pair, monkeypatch):
    monkeypatch.setenv("MEM_HYGIENE_CONFIG", str(cfg_pair))
    app = create_app(tmp_path / "mem.db")
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /v0/hygiene/run
# ---------------------------------------------------------------------------

def test_hygiene_run_dry_run(client, tmp_path):
    _seed_dead_stream(tmp_path)
    resp = client.post("/v0/hygiene/run", json={"dry_run": True, "run_id": "t1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "dry-run"
    assert body["candidate_count"] == 4
    assert body["quarantined"] == 0
    assert body["candidate_artifact"] is not None
    assert Path(body["candidate_artifact"]).exists()


def test_hygiene_run_mutates_and_reports(client, tmp_path):
    _seed_dead_stream(tmp_path)
    resp = client.post("/v0/hygiene/run", json={"dry_run": False, "run_id": "t2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["quarantined"] == 4
    assert body["fts_integrity_ok"] is True
    assert body["db_row_count_after"] == 0

    # The rows are gone from the read path.
    assert client.get("/v0/memories/elevator/row-0").status_code == 404
    assert client.get("/v0/search", params={"q": "machine state"}).json() == []


def test_hygiene_run_cap_aborts_409(client, tmp_path, monkeypatch):
    _seed_dead_stream(tmp_path, n=6)
    # Tighten the cap below the candidate count.
    cfg = json.loads((tmp_path / "mem_hygiene.json").read_text())
    cfg["batch_cap"] = 3
    (tmp_path / "mem_hygiene.json").write_text(json.dumps(cfg))
    resp = client.post("/v0/hygiene/run", json={"dry_run": False})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "hygiene_aborted"
    # Nothing mutated.
    assert client.get("/v0/memories/elevator/row-0").status_code == 200


def test_hygiene_run_over_cap_allowed(client, tmp_path, monkeypatch):
    _seed_dead_stream(tmp_path, n=6)
    cfg = json.loads((tmp_path / "mem_hygiene.json").read_text())
    cfg["batch_cap"] = 3
    (tmp_path / "mem_hygiene.json").write_text(json.dumps(cfg))
    resp = client.post("/v0/hygiene/run",
                       json={"dry_run": False, "allow_over_cap": True})
    assert resp.status_code == 200
    assert resp.json()["quarantined"] == 6


def test_hygiene_run_unconfigured_503(tmp_path, monkeypatch):
    monkeypatch.delenv("MEM_HYGIENE_CONFIG", raising=False)
    app = create_app(tmp_path / "mem.db")
    with TestClient(app) as c:
        resp = c.post("/v0/hygiene/run", json={"dry_run": True})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "hygiene_unconfigured"


# ---------------------------------------------------------------------------
# /v0/hygiene/list + /v0/hygiene/restore
# ---------------------------------------------------------------------------

def test_hygiene_list_empty_then_after_run(client, tmp_path):
    assert client.get("/v0/hygiene/list").json() == {
        "total": 0, "by_run": [], "by_prefix": []
    }
    _seed_dead_stream(tmp_path)
    client.post("/v0/hygiene/run", json={"dry_run": False, "run_id": "t3"})
    body = client.get("/v0/hygiene/list").json()
    assert body["total"] == 4
    assert body["by_prefix"][0]["prefix"] == "elevator"
    assert body["by_prefix"][0]["rows"] == 4
    assert body["by_run"][0]["run_id"] == "t3"


def test_hygiene_restore_round_trip(client, tmp_path):
    _seed_dead_stream(tmp_path)
    client.post("/v0/hygiene/run", json={"dry_run": False, "run_id": "t4"})
    resp = client.post("/v0/hygiene/restore", json={"prefix": "elevator/"})
    assert resp.status_code == 200
    assert resp.json()["restored"] == 4
    # Rows are back and searchable (memories_ai re-indexed on restore).
    assert client.get("/v0/memories/elevator/row-0").status_code == 200
    hits = client.get("/v0/search", params={"q": "machine state"}).json()
    assert len(hits) == 4
    assert client.get("/v0/hygiene/list").json()["total"] == 0


def test_hygiene_restore_missing_prefix_400(client):
    resp = client.post("/v0/hygiene/restore", json={})
    assert resp.status_code == 400


def test_hygiene_restore_atom_class_400(client):
    resp = client.post("/v0/hygiene/restore", json={"prefix": "decision/"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request"


# ---------------------------------------------------------------------------
# /v0/hygiene/ageout
# ---------------------------------------------------------------------------

def test_hygiene_ageout(client, tmp_path):
    _seed_dead_stream(tmp_path)
    client.post("/v0/hygiene/run", json={"dry_run": False, "run_id": "t5"})
    # Freshly quarantined: within the 14-day window -> 0 purged.
    resp = client.post("/v0/hygiene/ageout", json={})
    assert resp.status_code == 200
    assert resp.json()["purged"] == 0
    # A 0-day window purges everything.
    resp = client.post("/v0/hygiene/ageout", json={"window": 0})
    assert resp.json()["purged"] == 4
    assert client.get("/v0/hygiene/list").json()["total"] == 0


# ---------------------------------------------------------------------------
# D4 — write guard maps to a 4xx at the HTTP layer
# ---------------------------------------------------------------------------

def test_put_test_source_maps_to_4xx(client, monkeypatch):
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    resp = client.put(
        "/v0/memories/pattern/x",
        json={"content": "x", "tags": "", "source": "test-harness"},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "test_write_rejected"
    assert client.get("/v0/memories/pattern/x").status_code == 404


def test_put_production_source_still_works(client, monkeypatch):
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)
    resp = client.put(
        "/v0/memories/pattern/x",
        json={"content": "x", "tags": "", "source": "gw_topology"},
    )
    assert resp.status_code == 200
    assert client.get("/v0/memories/pattern/x").status_code == 200
