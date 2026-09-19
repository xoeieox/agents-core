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


def test_deposit_guard_ordering_403_before_503(tmp_path, monkeypatch):
    """Guard ordering is by design (spec D-1: the principal model is the
    OUTER gate): under MEM_ENFORCE_PRINCIPALS a reader/unknown-principal
    deposit attempt gets 403 principal_reader EVEN WHEN the recorder is
    unconfigured (which would otherwise 503) or the envelope is invalid
    (which would otherwise 400 bad_envelope). This is a deliberate behavior
    change vs the pre-PR ordering where 503/400 surfaced first — the
    reviewer (PR #337 cycle 1 [med]) confirmed it against the spec's
    intent: a reader must be rejected before any deposit-specific 503/400
    can mask the principal failure. Under observe-only (the default) the
    guard never rejects, so the pre-PR 503/400 precedence is preserved
    (test_deposit_unconfigured_503 above pins that)."""
    monkeypatch.setenv("MEM_ENFORCE_PRINCIPALS", "1")
    try:
        # Recorder unconfigured -> would be 503 pre-PR; the guard's 403
        # surfaces first for a reader (no X-Mem-Principal header).
        app = create_app(tmp_path / "mem.db", deposit_recorder=None)
        c = TestClient(app)
        resp = c.post("/v0/deposit", json=_envelope())
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"

        # A VALID envelope with a missing manifest_hash (would be 400
        # bad_envelope pre-PR) also surfaces the guard's 403 first for a
        # reader. (A structurally-invalid envelope — e.g. not a
        # LapisToolReturn — is rejected by FastAPI's body parser with a
        # 400 BEFORE the handler runs, so it cannot reach the guard at
        # all; the manifest_hash case is the in-handler 400 the ordering
        # note names.)
        bad = _envelope()
        bad["provenance"]["manifest_hash"] = ""
        resp_bad = c.post("/v0/deposit", json=bad)
        assert resp_bad.status_code == 403
        assert resp_bad.json()["error"]["code"] == "principal_reader"

        # The registered curator principal passes the guard and reaches the
        # deposit-specific 503 (recorder unconfigured) — the 503 is NOT
        # masked for an authorized writer.
        resp_curator = c.post(
            "/v0/deposit",
            json=_envelope(),
            headers={"X-Mem-Principal": "zephyr-deposit"},
        )
        assert resp_curator.status_code == 503
        assert resp_curator.json()["error"]["code"] == "deposit_unconfigured"
    finally:
        monkeypatch.delenv("MEM_ENFORCE_PRINCIPALS", raising=False)
