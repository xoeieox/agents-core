"""Tests for the /v0/deposit endpoint (LapisToolReturn -> mem store + attribution).

Uses a self-contained fake DepositRecorder so agents-core tests stay free of any
zephyr import (the real recorder is injected at boot via MEM_DEPOSIT_RECORDER).

The envelope is ALSO self-contained (no archetypes_core import): archetypes_core
is NOT a declared dependency of agents-core (see pyproject.toml), so importing
it here made this file fail in any clean CI/clone environment where
archetypes_core is not installed (the gate's concluded_gate_rejected: the
reviewer's most-likely-failure hypothesis, PR #345 cycle 1 [med]). The server's
LapisToolReturn.from_dict only requires the provenance fields it reads
(schema_version / agent_id / tool / timestamp / manifest_hash) — the
manifest_hash is an opaque dedup token to the server (it never recomputes it),
so a hand-built envelope with a fixed manifest_hash exercises the exact same
from_dict -> payload/provenance path without the import.
"""

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
    """A LapisToolReturn-shaped dict with a fixed manifest_hash.

    Hand-built (no archetypes_core import): the server's
    LapisToolReturn.from_dict reads only the provenance fields below and treats
    manifest_hash as an opaque dedup token (it never recomputes it), so this
    exercises the same from_dict -> payload/provenance path the real
    to_lapis_return() factory would produce, without a dependency agents-core
    does not declare. The manifest_hash is FIXED (not computed) so a
    construct-once / retry-identical-bytes dedup test (test_deposit_accept_
    then_dedup) is deterministic across calls.
    """
    return {
        "payload": {"key": key, "value": value, "tags": tags or []},
        "summary": "deposit test",
        "provenance": {
            "schema_version": "lapis-provenance-v0",
            "agent_id": "test-agent",
            "tool": "pytest",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "manifest_hash": f"sha256:{key}-{value}",
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    # D4 write-path guard (mem-hygiene-automation-v0, merged on main):
    # the deposit route persists with source=prov.agent_id, and this
    # fixture's self-contained envelope uses agent_id "test-agent" — a
    # test-provenance source the guard rejects at the MemoryStore.set()
    # chokepoint unless MEM_ALLOW_TEST_WRITE=1. The test suite sets it
    # (same convention as tests/test_mem_server.py).
    monkeypatch.setenv("MEM_ALLOW_TEST_WRITE", "1")
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

        # A MALFORMED envelope (structurally a dict, but not a valid
        # LapisToolReturn — LapisToolReturn.from_dict raises, so the
        # handler's own 400 bad_envelope path is the pre-PR behavior) also
        # surfaces the guard's 403 first for a reader: the write-class
        # guard is the outer gate and runs before envelope validation
        # (reviewer PR #344 cycle 1 [med]: this case was the one not
        # pinned by a test).
        malformed = {"payload": {"key": "pm/test-x"}, "summary": "s"}
        resp_malformed = c.post("/v0/deposit", json=malformed)
        assert resp_malformed.status_code == 403
        assert resp_malformed.json()["error"]["code"] == "principal_reader"

        # The registered curator principal passes the guard and reaches the
        # deposit-specific 503 (recorder unconfigured) — the 503 is NOT
        # masked for an authorized writer.
        resp_curator = c.post(
            "/v0/deposit",
            json=_envelope(),
            headers={"X-Mem-Principal": "zephyr-deposit"},
        )
        assert resp_curator.status_code == 503
        assert "recorder" in str(resp_curator.json())

        # ...and reaches the deposit-specific 400 bad_envelope for the
        # MALFORMED envelope (the guard does not mask the envelope's own
        # error for an authorized writer either — the guard is the outer
        # gate, not a replacement for validation).
        resp_curator_malformed = c.post(
            "/v0/deposit",
            json=malformed,
            headers={"X-Mem-Principal": "zephyr-deposit"},
        )
        assert resp_curator_malformed.status_code == 400
        assert resp_curator_malformed.json()["error"]["code"] == "bad_envelope"
    finally:
        monkeypatch.delenv("MEM_ENFORCE_PRINCIPALS", raising=False)


def test_deposit_guard_ordering_403_before_bad_envelope(tmp_path, monkeypatch):
    """The guard's 403 beats the handler's 400 bad_envelope for a reader
    (reviewer PR #344 cycle 1 [med]): under MEM_ENFORCE_PRINCIPALS a
    malformed-envelope deposit from a reader gets 403 principal_reader,
    not the pre-PR 400 bad_envelope. The principal model is the outer
    gate (spec D-1); the malformed-envelope case was the one the
    503-before-403 test did not pin."""
    monkeypatch.setenv("MEM_ENFORCE_PRINCIPALS", "1")
    try:
        # Recorder configured: the only pre-PR outcome for a malformed
        # envelope is the handler's 400 bad_envelope (no 503 in play).
        app = create_app(tmp_path / "mem.db", deposit_recorder=_FakeRecorder())
        c = TestClient(app)
        malformed = {"payload": {"key": "pm/test-x"}, "summary": "s"}
        resp = c.post("/v0/deposit", json=malformed)
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"
    finally:
        monkeypatch.delenv("MEM_ENFORCE_PRINCIPALS", raising=False)


def test_deposit_bad_envelope_400_observe_only(client):
    """Under observe-only (the default) the pre-PR precedence is
    preserved for the malformed-envelope case: a reader's malformed
    envelope gets the handler's 400 bad_envelope (the guard logs the
    attempt but never rejects), and a curator's malformed envelope gets
    the same 400 — the guard is a no-op under observe-only."""
    malformed = {"payload": {"key": "pm/test-x"}, "summary": "s"}
    resp_reader = client.post("/v0/deposit", json=malformed)
    assert resp_reader.status_code == 400
    assert resp_reader.json()["error"]["code"] == "bad_envelope"
    resp_curator = client.post(
        "/v0/deposit", json=malformed, headers={"X-Mem-Principal": "zephyr-deposit"}
    )
    assert resp_curator.status_code == 400
    assert resp_curator.json()["error"]["code"] == "bad_envelope"
