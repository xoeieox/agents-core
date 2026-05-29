"""Tests for the /v0/deposit endpoint (LapisToolReturn -> mem store + attribution).

Uses a self-contained fake DepositRecorder so agents-core tests stay free of any
zephyr import (the real recorder is injected at boot via MEM_DEPOSIT_RECORDER)."""

import json

import pytest
from fastapi.testclient import TestClient

from agents_core.mem_server import create_app


class _FakeRecorder:
    """In-memory DepositRecorder: dedups on manifest_hash like the real one."""

    def __init__(self):
        self.seen: dict[str, dict] = {}

    def already_recorded(self, manifest_hash: str) -> bool:
        return manifest_hash in self.seen

    def record(self, provenance: dict, *, store_kind: str, key) -> bool:
        mh = provenance["manifest_hash"]
        if mh in self.seen:
            return False
        self.seen[mh] = {"provenance": provenance, "store_kind": store_kind, "key": key}
        return True


def _envelope(key="pm/test", value="v", tags=None):
    """A LapisToolReturn dict with a real manifest_hash (construct-once)."""
    from archetypes_core.provenance import to_lapis_return

    ltr = to_lapis_return(
        {"key": key, "value": value, "tags": tags or []},
        agent_id="test-agent",
        tool="pytest",
        summary="deposit test",
    )
    return json.loads(ltr.to_json())


@pytest.fixture
def client(tmp_path):
    rec = _FakeRecorder()
    app = create_app(tmp_path / "mem.db", deposit_recorder=rec)
    c = TestClient(app)
    c._rec = rec  # type: ignore[attr-defined]
    return c


def test_deposit_accept_then_dedup(client):
    body = _envelope(key="pm/test-x", value="hello", tags=["t"])
    r1 = client.post("/v0/deposit", json=body)
    assert r1.status_code == 200 and r1.json()["status"] == "accepted"
    # identical retry -> duplicate, no double write
    r2 = client.post("/v0/deposit", json=body)
    assert r2.json()["status"] == "duplicate"
    assert len(client._rec.seen) == 1
    # payload persisted to the mem store
    g = client.get("/v0/memories/pm/test-x")
    assert g.status_code == 200 and g.json()["content"] == "hello"


def test_deposit_healthz_reports_configured(client):
    assert client.get("/healthz").json()["deposit"]["configured"] is True


def test_deposit_bad_payload_400(client):
    body = _envelope()
    body["payload"] = {"no_key": 1}
    assert client.post("/v0/deposit", json=body).status_code == 400


def test_deposit_unconfigured_503(tmp_path):
    app = create_app(tmp_path / "mem.db", deposit_recorder=None)
    c = TestClient(app)
    assert c.post("/v0/deposit", json=_envelope()).status_code == 503
    assert c.get("/healthz").json()["deposit"]["configured"] is False
