"""HTTP tests for agents_core.slot_server (FastAPI / TestClient).

Covers:
  - auth middleware: constant-time reject, fail-closed on unset token
  - no-token loopback mode (reads + writes pass without auth header)
  - per-principal token: write with wrong "by" → 403 not_authorized
  - per-principal token: write with correct "by" → OK
  - legacy shared token: any "by" accepted after auth
  - create_slot 201, missing field → 400, duplicate slot → 400
  - get_slot 200 / 404
  - update_status 200, wrong owner → 403, slot not found → 404
  - escalate via HTTP
  - observer_update via HTTP
  - input validation errors (400)
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents_core.slot_server import create_app
from agents_core.slots import SlotStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "slots.db"


def _client(db: Path, token: str = "") -> TestClient:
    """Build a TestClient with SLOTS_BEARER_TOKEN set to `token`."""
    old = os.environ.get("SLOTS_BEARER_TOKEN")
    os.environ["SLOTS_BEARER_TOKEN"] = token
    try:
        app = create_app(db)
    finally:
        if old is None:
            os.environ.pop("SLOTS_BEARER_TOKEN", None)
        else:
            os.environ["SLOTS_BEARER_TOKEN"] = old
    return TestClient(app, raise_server_exceptions=True)


def _auth(token_secret: str) -> dict:
    return {"Authorization": f"Bearer {token_secret}"}


# ---------------------------------------------------------------------------
# Auth middleware — shared-token mode
# ---------------------------------------------------------------------------

class TestAuthSharedToken:
    TOKEN = "shared-secret"

    @pytest.fixture
    def client(self, db):
        return _client(db, self.TOKEN)

    def test_valid_token_passes(self, client):
        r = client.get("/healthz", headers=_auth(self.TOKEN))
        assert r.status_code == 200

    def test_wrong_token_rejected(self, client):
        r = client.get("/healthz", headers=_auth("wrong"))
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "unauthorized"

    def test_missing_auth_header_rejected(self, client):
        r = client.get("/healthz")
        assert r.status_code == 401

    def test_bearer_prefix_required(self, client):
        r = client.get("/healthz", headers={"Authorization": self.TOKEN})
        assert r.status_code == 401

    def test_legacy_any_by_accepted(self, client, db):
        """Shared token allows writing as any contributor (legacy mode)."""
        store = SlotStore(db_path=db)
        sid = store.create_slot("p", {"type": "fixer", "id": "agent-1"})
        store.close()
        r = client.post(
            f"/v0/slots/{sid}/status",
            json={"status": "in-progress", "by": "agent-1"},
            headers=_auth(self.TOKEN),
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Auth middleware — no-token (loopback-only) mode
# ---------------------------------------------------------------------------

class TestNoTokenLoopback:
    @pytest.fixture
    def client(self, db):
        return _client(db, "")  # no token = loopback-only

    def test_reads_pass_without_header(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200

    def test_writes_pass_without_header(self, client):
        r = client.post(
            "/v0/slots",
            json={"project_id": "proj", "contributor": {"type": "fixer", "id": "a"}},
        )
        assert r.status_code == 201


# ---------------------------------------------------------------------------
# Auth middleware — constant-time comparison
# ---------------------------------------------------------------------------

class TestConstantTimeAuth:
    """Verify hmac.compare_digest is used (observable via behaviour, not timing)."""

    def test_empty_string_token_not_accepted_as_any_bearer(self, db):
        """When SLOTS_BEARER_TOKEN is non-empty, an empty Authorization value
        must be rejected even if the internal token happens to have length 0 after
        parsing — i.e. hmac.compare_digest("", "") is not a backdoor."""
        # Set a real non-empty token; empty auth must still fail.
        client = _client(db, "realtoken")
        r = client.get("/healthz", headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_prefix_of_token_not_accepted(self, db):
        client = _client(db, "longsecret")
        r = client.get("/healthz", headers=_auth("long"))
        assert r.status_code == 401

    def test_token_with_extra_chars_not_accepted(self, db):
        client = _client(db, "secret")
        r = client.get("/healthz", headers=_auth("secret_extra"))
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Per-principal token — ownership binding (D3)
# ---------------------------------------------------------------------------

class TestPrincipalBinding:
    TOKEN = "agent-1:mysecret"  # principal=agent-1, secret=mysecret

    @pytest.fixture
    def client(self, db):
        return _client(db, self.TOKEN)

    @pytest.fixture
    def slot_id(self, db):
        store = SlotStore(db_path=db)
        sid = store.create_slot("p", {"type": "fixer", "id": "agent-1"})
        store.close()
        return sid

    def test_correct_by_accepted(self, client, slot_id):
        r = client.post(
            f"/v0/slots/{slot_id}/status",
            json={"status": "in-progress", "by": "agent-1"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 200
        assert r.json()["status"] == "in-progress"

    def test_impersonation_rejected(self, client, slot_id):
        """A caller with a valid token cannot write as a different contributor."""
        r = client.post(
            f"/v0/slots/{slot_id}/status",
            json={"status": "in-progress", "by": "agent-2"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403
        # FastAPI wraps HTTPException.detail under "detail"
        assert r.json()["detail"]["error"]["code"] == "not_authorized"

    def test_create_slot_impersonation_rejected(self, client):
        r = client.post(
            "/v0/slots",
            json={"project_id": "p", "contributor": {"type": "fixer", "id": "agent-2"}},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_authorized"

    def test_create_slot_correct_principal_ok(self, client):
        r = client.post(
            "/v0/slots",
            json={"project_id": "p", "contributor": {"type": "fixer", "id": "agent-1"}},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 201

    def test_checkpoint_impersonation_rejected(self, client, slot_id):
        r = client.post(
            f"/v0/slots/{slot_id}/checkpoint",
            json={"kind": "self-report", "note": "hi", "by": "agent-2"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403

    def test_escalate_impersonation_rejected(self, client, slot_id):
        r = client.post(
            f"/v0/slots/{slot_id}/escalate",
            json={"to": "facets", "reason": "blocked", "by": "agent-2"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403

    def test_observer_not_principal_bound(self, client, slot_id):
        """Observer writes (Weaver) are not subject to principal binding."""
        r = client.post(
            f"/v0/slots/{slot_id}/observer",
            json={"weaver_status": "stuck", "by": "weaver"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 200
        assert r.json()["weaver_status"] == "stuck"

    def test_ratify_not_principal_bound(self, client, slot_id):
        """Facets ratification writes (like the Weaver's) are a distinct-actor
        observer namespace — not subject to contributor-of-record principal binding."""
        r = client.post(
            f"/v0/slots/{slot_id}/ratify",
            json={"verdict": {"council_status": "resolved"}, "by": "facets"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 200
        assert r.json()["facets_verdict"]["council_status"] == "resolved"


# ---------------------------------------------------------------------------
# CRUD / input validation
# ---------------------------------------------------------------------------

class TestCRUD:
    @pytest.fixture
    def client(self, db):
        return _client(db, "")

    @pytest.fixture
    def sid(self, client):
        r = client.post(
            "/v0/slots",
            json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-1"}},
        )
        assert r.status_code == 201
        return r.json()["slot_id"]

    def test_create_returns_slot(self, client):
        r = client.post(
            "/v0/slots",
            json={
                "project_id": "proj-A",
                "contributor": {"type": "fixer", "id": "agent-1"},
                "horizon": {"project_summary": "ctx", "immediate_goal": "split"},
            },
        )
        assert r.status_code == 201
        d = r.json()
        assert d["project_id"] == "proj-A"
        assert d["contributor_id"] == "agent-1"
        assert d["status"] == "dispatched"

    def test_create_missing_project_id_400(self, client):
        r = client.post(
            "/v0/slots",
            json={"contributor": {"type": "fixer", "id": "agent-1"}},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_request"

    def test_create_invalid_status_400(self, client):
        r = client.post(
            "/v0/slots",
            json={"project_id": "p", "contributor": {"type": "fixer", "id": "a"}, "status": "bogus"},
        )
        assert r.status_code == 400

    def test_create_duplicate_slot_id_400(self, client, sid):
        r = client.post(
            "/v0/slots",
            json={"project_id": "p", "contributor": {"type": "fixer", "id": "agent-1"}, "slot_id": sid},
        )
        assert r.status_code == 400

    def test_get_slot_200(self, client, sid):
        r = client.get(f"/v0/slots/{sid}")
        assert r.status_code == 200
        assert r.json()["slot_id"] == sid

    def test_get_slot_404(self, client):
        r = client.get("/v0/slots/doesnotexist")
        assert r.status_code == 404
        assert r.json()["detail"]["error"]["code"] == "not_found"

    def test_update_status_ok(self, client, sid):
        r = client.post(f"/v0/slots/{sid}/status", json={"status": "in-progress", "by": "agent-1"})
        assert r.status_code == 200
        assert r.json()["status"] == "in-progress"

    def test_update_status_wrong_owner_403(self, client, sid):
        r = client.post(f"/v0/slots/{sid}/status", json={"status": "landed", "by": "intruder"})
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_owner"

    def test_update_status_not_found_404(self, client):
        r = client.post("/v0/slots/nope/status", json={"status": "landed", "by": "agent-1"})
        assert r.status_code == 404

    def test_update_status_invalid_status_400(self, client, sid):
        r = client.post(f"/v0/slots/{sid}/status", json={"status": "bogus", "by": "agent-1"})
        assert r.status_code == 400

    def test_escalate_via_http(self, client, sid):
        r = client.post(
            f"/v0/slots/{sid}/escalate",
            json={"to": "facets", "reason": "blocked on spec", "by": "agent-1"},
        )
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "escalated"
        assert d["escalation"]["to"] == "facets"

    def test_observer_update_via_http(self, client, sid):
        r = client.post(
            f"/v0/slots/{sid}/observer",
            json={"weaver_status": "stuck", "by": "weaver"},
        )
        assert r.status_code == 200
        assert r.json()["weaver_status"] == "stuck"

    def test_observer_update_404(self, client):
        r = client.post("/v0/slots/nope/observer", json={"weaver_status": "stuck"})
        assert r.status_code == 404

    def test_ratify_via_http(self, client, sid):
        r = client.post(
            f"/v0/slots/{sid}/ratify",
            json={"verdict": {"council_status": "resolved", "council_landing": "proceed"},
                  "by": "facets"},
        )
        assert r.status_code == 200
        d = r.json()
        # Verdict lands in the facets_* namespace; status is untouched (still dispatched).
        assert d["facets_verdict"]["council_status"] == "resolved"
        assert d["facets_verdict"]["by"] == "facets"
        assert d["status"] == "dispatched"

    def test_ratify_missing_slot_404(self, client):
        r = client.post("/v0/slots/nope/ratify", json={"verdict": {"council_status": "resolved"}})
        assert r.status_code == 404
        assert r.json()["detail"]["error"]["code"] == "not_found"

    def test_ratify_missing_verdict_400(self, client, sid):
        r = client.post(f"/v0/slots/{sid}/ratify", json={"by": "facets"})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_request"

    def test_list_slots(self, client, sid):
        r = client.get("/v0/slots")
        assert r.status_code == 200
        assert any(s["slot_id"] == sid for s in r.json())

    def test_stats(self, client, sid):
        r = client.get("/v0/stats")
        assert r.status_code == 200
        assert r.json()["total_slots"] >= 1
