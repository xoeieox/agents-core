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

import logging
import multiprocessing
import os
from pathlib import Path
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# ETag conditional reads (AC1-AC4) + Cache-Control headers
# ---------------------------------------------------------------------------

class TestETags:
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

    def test_list_slots_returns_etag_header(self, client, sid):
        """AC1: GET /v0/slots returns ETag header."""
        r = client.get("/v0/slots")
        assert r.status_code == 200
        assert "ETag" in r.headers
        assert r.headers["ETag"].startswith('"')

    def test_list_slots_returns_cache_control_header(self, client, sid):
        """AC1: GET /v0/slots returns Cache-Control: no-cache header."""
        r = client.get("/v0/slots")
        assert r.status_code == 200
        assert "Cache-Control" in r.headers
        assert "no-cache" in r.headers["Cache-Control"]

    def test_etag_stable_across_identical_requests(self, client, sid):
        """AC1: ETag is stable across two identical requests with no intervening writes."""
        r1 = client.get("/v0/slots")
        etag1 = r1.headers["ETag"]
        r2 = client.get("/v0/slots")
        etag2 = r2.headers["ETag"]
        assert etag1 == etag2

    def test_304_on_if_none_match_match(self, client, sid):
        """AC2: If-None-Match matching the ETag returns 304 Not Modified.
        Verify query() is NOT called when 304 fires (optimization validates)."""
        r1 = client.get("/v0/slots")
        etag = r1.headers["ETag"]
        # Spy on SlotStore.query to verify it is NOT called on 304.
        with patch('agents_core.slot_server.SlotStore.query') as mock_query:
            r2 = client.get("/v0/slots", headers={"If-None-Match": etag})
            assert r2.status_code == 304
            assert r2.headers["ETag"] == etag
            # 304 should have empty body
            assert r2.text == ""
            # Verify query was NOT called (only read_version should run).
            mock_query.assert_not_called()

    def test_200_on_if_none_match_mismatch(self, client, sid):
        """AC2: If-None-Match NOT matching returns 200 with content."""
        r1 = client.get("/v0/slots")
        r2 = client.get("/v0/slots", headers={"If-None-Match": '"wrong-etag"'})
        assert r2.status_code == 200
        assert "ETag" in r2.headers
        assert len(r2.json()) >= 1

    def test_etag_changes_after_create(self, client, sid):
        """AC3: ETag for a query changes after an insert."""
        r1 = client.get("/v0/slots")
        etag1 = r1.headers["ETag"]
        # Create another slot
        client.post(
            "/v0/slots",
            json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-2"}},
        )
        r2 = client.get("/v0/slots")
        etag2 = r2.headers["ETag"]
        assert etag1 != etag2

    def test_etag_changes_after_update(self, client, sid):
        """AC3: ETag changes after an update."""
        r1 = client.get("/v0/slots")
        etag1 = r1.headers["ETag"]
        # Update a slot status
        client.post(f"/v0/slots/{sid}/status", json={"status": "in-progress", "by": "agent-1"})
        r2 = client.get("/v0/slots")
        etag2 = r2.headers["ETag"]
        assert etag1 != etag2

    def test_etag_changes_after_expire(self, client, db):
        """AC3: ETag changes after a delete/expire at the HTTP layer."""
        # Create a parked slot.
        r = client.post(
            "/v0/slots",
            json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-1"}, "status": "parked"},
        )
        sid = r.json()["slot_id"]
        # Get the ETag before expiration.
        r1 = client.get("/v0/slots")
        etag1 = r1.headers["ETag"]
        # Manually update the slot's last_update to be > 30 days old so it's eligible for expiration.
        from datetime import datetime, timedelta, timezone
        # Access the store via a fresh instance to the same DB.
        store = SlotStore(db_path=db)
        old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        with store._lock:
            store._conn.execute(
                "UPDATE slots SET last_update=? WHERE slot_id=?",
                (old_date, sid),
            )
            store._conn.commit()
        store.close()
        # Call expire via HTTP.
        client.post("/v0/expire")
        r2 = client.get("/v0/slots")
        etag2 = r2.headers["ETag"]
        # ETag should change (the old parked slot was deleted, so COUNT decreased).
        assert etag1 != etag2

    def test_etag_filter_specific_project(self, client):
        """AC4: ETag changes when filter changes."""
        # Create two slots in different projects
        r1 = client.post(
            "/v0/slots",
            json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-1"}},
        )
        sid_a = r1.json()["slot_id"]
        client.post(
            "/v0/slots",
            json={"project_id": "proj-B", "contributor": {"type": "fixer", "id": "agent-1"}},
        )
        # ETags for different project filters should differ
        r_a = client.get("/v0/slots?project_id=proj-A")
        r_b = client.get("/v0/slots?project_id=proj-B")
        r_all = client.get("/v0/slots")
        assert r_a.headers["ETag"] != r_b.headers["ETag"]
        assert r_a.headers["ETag"] != r_all.headers["ETag"]
        assert r_b.headers["ETag"] != r_all.headers["ETag"]

    def test_cross_filter_if_none_match_does_not_304(self, client):
        """AC4: Cross-filter If-None-Match does NOT return 304.
        An ETag from proj-A query must not match when sent with proj-B query."""
        # Create slots in different projects.
        client.post(
            "/v0/slots",
            json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-1"}},
        )
        client.post(
            "/v0/slots",
            json={"project_id": "proj-B", "contributor": {"type": "fixer", "id": "agent-1"}},
        )
        # Get ETag for proj-A.
        r_a = client.get("/v0/slots?project_id=proj-A")
        etag_a = r_a.headers["ETag"]
        # Send proj-A's ETag as If-None-Match with proj-B query — should return 200 (mismatch).
        r_b = client.get("/v0/slots?project_id=proj-B", headers={"If-None-Match": etag_a})
        assert r_b.status_code == 200
        # Verify the response contains the proj-B slot(s).
        assert "ETag" in r_b.headers
        assert len(r_b.json()) >= 1

    def test_get_slot_returns_etag_header(self, client, sid):
        """AC1: GET /v0/slots/{slot_id} returns ETag header."""
        r = client.get(f"/v0/slots/{sid}")
        assert r.status_code == 200
        assert "ETag" in r.headers
        assert r.headers["ETag"].startswith('"')

    def test_get_slot_returns_cache_control_header(self, client, sid):
        """AC1: GET /v0/slots/{slot_id} returns Cache-Control: no-cache header."""
        r = client.get(f"/v0/slots/{sid}")
        assert r.status_code == 200
        assert "Cache-Control" in r.headers

    def test_get_slot_304_on_match(self, client, sid):
        """AC2: GET /v0/slots/{slot_id} with matching If-None-Match returns 304."""
        r1 = client.get(f"/v0/slots/{sid}")
        etag = r1.headers["ETag"]
        r2 = client.get(f"/v0/slots/{sid}", headers={"If-None-Match": etag})
        assert r2.status_code == 304
        assert r2.headers["ETag"] == etag

    def test_get_slot_etag_changes_after_update(self, client, sid):
        """AC3: ETag for a specific slot changes after updating it."""
        r1 = client.get(f"/v0/slots/{sid}")
        etag1 = r1.headers["ETag"]
        # Update the slot
        client.post(f"/v0/slots/{sid}/status", json={"status": "landed", "by": "agent-1"})
        r2 = client.get(f"/v0/slots/{sid}")
        etag2 = r2.headers["ETag"]
        assert etag1 != etag2


# ---------------------------------------------------------------------------
# Mailbox HTTP surface (agents-core-slot1-mailbox-v0)
# ---------------------------------------------------------------------------

class TestMailboxNoToken:
    """No-token (loopback) mode — mirrors TestCRUD's client setup."""

    @pytest.fixture
    def client(self, db):
        return _client(db, "")

    @pytest.fixture
    def mailbox_id(self, client):
        r = client.post("/v0/mailbox/open", json={"by": "Erah", "window_ref": "w1"})
        assert r.status_code == 201
        return r.json()["slot_id"]

    def test_open_returns_slot_id_only(self, client):
        r = client.post("/v0/mailbox/open", json={"by": "Erah"})
        assert r.status_code == 201
        assert set(r.json().keys()) == {"slot_id"}

    def test_open_missing_by_400(self, client):
        r = client.post("/v0/mailbox/open", json={})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_request"

    def test_open_window_ref_persisted(self, client, mailbox_id):
        r = client.get(f"/v0/slots/{mailbox_id}")
        assert r.status_code == 200
        assert r.json()["horizon"]["window_ref"] == "w1"

    def test_bank_happy_path(self, client, mailbox_id):
        r = client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "hello", "by": "Erah"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_bank_missing_text_400(self, client, mailbox_id):
        r = client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"by": "Erah"})
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "bad_request"

    def test_bank_not_found_404(self, client):
        r = client.post("/v0/mailbox/nope/bank", json={"text": "hi", "by": "Erah"})
        assert r.status_code == 404
        assert r.json()["detail"]["error"]["code"] == "not_found"

    def test_bank_wrong_owner_403_not_owner(self, client, mailbox_id):
        """store-layer SlotOwnershipError -> not_owner (loopback mode has no
        principal binding, so this exercises the store-layer 403 cause)."""
        r = client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "hi", "by": "intruder"})
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_owner"

    def test_drain_happy_path_and_order(self, client, mailbox_id):
        client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "one", "by": "Erah"})
        client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "two", "by": "Erah"})
        r = client.get(f"/v0/mailbox/{mailbox_id}/drain")
        assert r.status_code == 200
        prompts = r.json()["prompts"]
        assert [p["text"] for p in prompts] == ["one", "two"]

    def test_drain_not_found_404(self, client):
        r = client.get("/v0/mailbox/nope/drain")
        assert r.status_code == 404
        assert r.json()["detail"]["error"]["code"] == "not_found"

    def test_close_happy_path(self, client, mailbox_id):
        r = client.post(f"/v0/mailbox/{mailbox_id}/close", json={"by": "Erah"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        assert client.get(f"/v0/slots/{mailbox_id}").json()["status"] == "landed"

    def test_close_not_found_404(self, client):
        r = client.post("/v0/mailbox/nope/close", json={"by": "Erah"})
        assert r.status_code == 404
        assert r.json()["detail"]["error"]["code"] == "not_found"

    def test_close_wrong_owner_403_not_owner(self, client, mailbox_id):
        r = client.post(f"/v0/mailbox/{mailbox_id}/close", json={"by": "intruder"})
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_owner"

    def test_current_none_when_no_mailbox_open(self, client):
        r = client.get("/v0/mailbox/current")
        assert r.status_code == 200
        assert r.json() == {"slot_id": None, "count": 0}

    def test_current_returns_open_slot_and_count(self, client, mailbox_id):
        client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "one", "by": "Erah"})
        client.post(f"/v0/mailbox/{mailbox_id}/bank", json={"text": "two", "by": "Erah"})
        r = client.get("/v0/mailbox/current")
        assert r.status_code == 200
        assert r.json() == {"slot_id": mailbox_id, "count": 2}

    def test_current_excludes_closed_mailbox(self, client, mailbox_id):
        client.post(f"/v0/mailbox/{mailbox_id}/close", json={"by": "Erah"})
        r = client.get("/v0/mailbox/current")
        assert r.status_code == 200
        assert r.json() == {"slot_id": None, "count": 0}


class TestMailboxPrincipalBinding:
    """Per-principal token — the OTHER 403 cause on bank/close: auth-layer
    _check_by_principal rejecting an impersonated `by` before the store layer
    is even reached (not_authorized, not not_owner)."""

    TOKEN = "Erah:mysecret"

    @pytest.fixture
    def client(self, db):
        return _client(db, self.TOKEN)

    @pytest.fixture
    def mailbox_id(self, client):
        r = client.post("/v0/mailbox/open", json={"by": "Erah"}, headers=_auth("mysecret"))
        assert r.status_code == 201
        return r.json()["slot_id"]

    def test_open_impersonation_rejected(self, client):
        r = client.post("/v0/mailbox/open", json={"by": "someone-else"}, headers=_auth("mysecret"))
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_authorized"

    def test_bank_impersonation_rejected_not_authorized(self, client, mailbox_id):
        r = client.post(
            f"/v0/mailbox/{mailbox_id}/bank",
            json={"text": "hi", "by": "someone-else"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_authorized"

    def test_close_impersonation_rejected_not_authorized(self, client, mailbox_id):
        r = client.post(
            f"/v0/mailbox/{mailbox_id}/close",
            json={"by": "someone-else"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 403
        assert r.json()["detail"]["error"]["code"] == "not_authorized"

    def test_bank_correct_principal_ok(self, client, mailbox_id):
        r = client.post(
            f"/v0/mailbox/{mailbox_id}/bank",
            json={"text": "hi", "by": "Erah"},
            headers=_auth("mysecret"),
        )
        assert r.status_code == 200


class TestMailboxGenericRouteBypass:
    """§1d: the generic /status, /escalate, /domain routes still work unchanged
    against a mailbox slot_id (backward-compat), but log a bypass warning."""

    @pytest.fixture
    def client(self, db):
        return _client(db, "")

    @pytest.fixture
    def mailbox_id(self, client):
        r = client.post("/v0/mailbox/open", json={"by": "Erah"})
        assert r.status_code == 201
        return r.json()["slot_id"]

    def test_status_route_unchanged_response_and_logs_bypass(self, client, mailbox_id, caplog):
        with caplog.at_level(logging.WARNING):
            r = client.post(
                f"/v0/slots/{mailbox_id}/status", json={"status": "in-progress", "by": "Erah"}
            )
        assert r.status_code == 200
        assert r.json()["status"] == "in-progress"
        assert any("bypassing mailbox verbs" in rec.message for rec in caplog.records)
        assert any("/status" in rec.message for rec in caplog.records)

    def test_escalate_route_unchanged_response_and_logs_bypass(self, client, mailbox_id, caplog):
        with caplog.at_level(logging.WARNING):
            r = client.post(
                f"/v0/slots/{mailbox_id}/escalate",
                json={"to": "facets", "reason": "test", "by": "Erah"},
            )
        assert r.status_code == 200
        assert r.json()["status"] == "escalated"
        assert any("bypassing mailbox verbs" in rec.message for rec in caplog.records)
        assert any("/escalate" in rec.message for rec in caplog.records)

    def test_domain_route_unchanged_response_and_logs_bypass(self, client, mailbox_id, caplog):
        with caplog.at_level(logging.WARNING):
            r = client.post(
                f"/v0/slots/{mailbox_id}/domain",
                json={"files": ["x.py"], "by": "Erah"},
            )
        assert r.status_code == 200
        assert any("bypassing mailbox verbs" in rec.message for rec in caplog.records)
        assert any("/domain" in rec.message for rec in caplog.records)

    def test_non_mailbox_slot_status_route_no_bypass_log(self, client, caplog):
        """Regression guard: a normal (non-mailbox) slot must NOT trigger the
        bypass warning — only project_id=="slot1-mailbox" rows do."""
        r = client.post(
            "/v0/slots", json={"project_id": "proj-A", "contributor": {"type": "fixer", "id": "agent-1"}}
        )
        sid = r.json()["slot_id"]
        with caplog.at_level(logging.WARNING):
            r2 = client.post(f"/v0/slots/{sid}/status", json={"status": "in-progress", "by": "agent-1"})
        assert r2.status_code == 200
        assert not any("bypassing mailbox verbs" in rec.message for rec in caplog.records)
