"""Tests for elevator queue HTTP routes on slot_server."""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from agents_core.slot_server import create_app


@pytest.fixture
def client():
    """Create a test client with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slot_db = Path(tmpdir) / "slots.db"
        elevator_db = Path(tmpdir) / "elevator.db"
        app = create_app(slot_db, elevator_db_path=elevator_db)
        yield TestClient(app)


def test_enqueue_basic(client):
    """Test basic enqueue via POST /v0/elevator/enqueue."""
    response = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "session-turn",
            "payload": {"prompt": "hello"},
            "principal": "session-123",
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "pending"
    assert data["lane"] == "interactive"
    assert data["payload"]["prompt"] == "hello"


def test_enqueue_missing_field(client):
    """Test enqueue rejects missing required field."""
    response = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "session-turn",
            "payload": {},
            # missing principal
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 400
    assert "missing field" in response.json()["detail"]["error"]["message"]


def test_enqueue_invalid_lane(client):
    """Test enqueue rejects invalid lane."""
    response = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "invalid",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 400
    assert "invalid lane" in response.json()["detail"]["error"]["message"]


def test_claim_basic(client):
    """Test claim via POST /v0/elevator/claim."""
    # Enqueue an item.
    enqueue_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    item_id = enqueue_resp.json()["item_id"]

    # Claim it.
    response = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive", "deliberation", "execution"],
            "owner": "broker-1",
            "claim_ttl_sec": 30,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == item_id
    assert data["status"] == "claimed"
    assert data["claim_owner"] == "broker-1"


def test_claim_no_items(client):
    """Test claim returns null when no pending items."""
    response = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            "owner": "broker-1",
            "claim_ttl_sec": 30,
        },
    )
    assert response.status_code == 200
    assert response.json() is None


def test_claim_missing_owner(client):
    """Test claim rejects missing owner field."""
    response = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            # missing owner
            "claim_ttl_sec": 30,
        },
    )
    assert response.status_code == 400
    assert "missing field" in response.json()["detail"]["error"]["message"]


def test_ack_basic(client):
    """Test ack via POST /v0/elevator/ack."""
    # Enqueue and claim.
    enqueue_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    item_id = enqueue_resp.json()["item_id"]

    client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    )

    # Ack.
    response = client.post(
        "/v0/elevator/ack",
        json={
            "item_id": item_id,
            "result_ref": "result://abc",
            "provenance": {"phase": "big"},
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "served"
    assert data["result_ref"] == "result://abc"


def test_ack_nonexistent_item(client):
    """Test ack returns 404 for nonexistent item."""
    response = client.post(
        "/v0/elevator/ack",
        json={
            "item_id": "nonexistent",
        },
    )
    assert response.status_code == 404


def test_requeue_basic(client):
    """Test requeue via POST /v0/elevator/requeue."""
    # Enqueue and claim.
    enqueue_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    item_id = enqueue_resp.json()["item_id"]

    client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    )

    # Requeue.
    response = client.post(
        "/v0/elevator/requeue",
        json={"item_id": item_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"


def test_requeue_nonexistent_item(client):
    """Test requeue returns 404 for nonexistent item."""
    response = client.post(
        "/v0/elevator/requeue",
        json={"item_id": "nonexistent"},
    )
    assert response.status_code == 404


def test_fail_basic(client):
    """Test fail via POST /v0/elevator/fail."""
    # Enqueue and claim.
    enqueue_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    item_id = enqueue_resp.json()["item_id"]

    client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    )

    # Fail.
    response = client.post(
        "/v0/elevator/fail",
        json={"item_id": item_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"


def test_fail_nonexistent_item(client):
    """Test fail returns 404 for nonexistent item."""
    response = client.post(
        "/v0/elevator/fail",
        json={"item_id": "nonexistent"},
    )
    assert response.status_code == 404


def test_get_item_basic(client):
    """Test GET /v0/elevator/item/{item_id}."""
    # Enqueue.
    enqueue_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {"x": 1},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    item_id = enqueue_resp.json()["item_id"]

    # Get.
    response = client.get(f"/v0/elevator/item/{item_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == item_id
    assert data["payload"]["x"] == 1


def test_get_item_not_found(client):
    """Test GET /v0/elevator/item/{item_id} returns 404 for nonexistent item."""
    response = client.get("/v0/elevator/item/nonexistent")
    assert response.status_code == 404


def test_get_state_basic(client):
    """Test GET /v0/elevator/state returns queue + gw blocks."""
    # Enqueue items.
    client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "deliberation",
            "kind": "test",
            "payload": {},
            "principal": "p2",
            "latency_class": "batch",
        },
    )

    response = client.get("/v0/elevator/state")
    assert response.status_code == 200
    data = response.json()

    # Check queue block.
    assert "queue" in data
    assert "interactive" in data["queue"]
    assert "deliberation" in data["queue"]
    assert "execution" in data["queue"]
    assert data["queue"]["interactive"]["pending"] == 1
    assert data["queue"]["deliberation"]["pending"] == 1
    assert data["queue"]["execution"]["pending"] == 0

    # Check gw block (best-effort, may be unknown).
    assert "gw" in data
    assert "as_of" in data


def test_state_oldest_age_sec(client):
    """Test that state includes oldest_age_sec for pending items."""
    # Enqueue.
    client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )

    response = client.get("/v0/elevator/state")
    data = response.json()
    assert data["queue"]["interactive"]["oldest_age_sec"] is not None
    assert data["queue"]["interactive"]["oldest_age_sec"] >= 0


def test_state_no_pending_items(client):
    """Test state when no pending items exist."""
    response = client.get("/v0/elevator/state")
    data = response.json()
    assert data["queue"]["interactive"]["pending"] == 0
    assert data["queue"]["interactive"]["oldest_age_sec"] is None


def test_lane_priority_ordering(client):
    """Test that claim respects lane priority via HTTP."""
    # Enqueue in mixed order.
    exec_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "execution",
            "kind": "fixer",
            "payload": {},
            "principal": "p1",
            "latency_class": "batch",
        },
    )
    exec_id = exec_resp.json()["item_id"]

    delib_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "deliberation",
            "kind": "council",
            "payload": {},
            "principal": "p2",
            "latency_class": "batch",
        },
    )
    delib_id = delib_resp.json()["item_id"]

    inter_resp = client.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "session-turn",
            "payload": {},
            "principal": "p3",
            "latency_class": "interactive",
        },
    )
    inter_id = inter_resp.json()["item_id"]

    # Claim across all lanes; should respect priority.
    claim1 = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive", "deliberation", "execution"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    ).json()
    assert claim1["item_id"] == inter_id

    claim2 = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive", "deliberation", "execution"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    ).json()
    assert claim2["item_id"] == delib_id

    claim3 = client.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive", "deliberation", "execution"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    ).json()
    assert claim3["item_id"] == exec_id


@pytest.fixture
def client_with_auth():
    """Create a test client with bearer auth enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slot_db = Path(tmpdir) / "slots.db"
        elevator_db = Path(tmpdir) / "elevator.db"
        # Set bearer token in environment.
        os.environ["SLOTS_BEARER_TOKEN"] = "test-secret-token"
        try:
            app = create_app(slot_db, elevator_db_path=elevator_db)
            yield TestClient(app)
        finally:
            del os.environ["SLOTS_BEARER_TOKEN"]


def test_bearer_auth_required_enqueue(client_with_auth):
    """Test that enqueue requires bearer token."""
    response = client_with_auth.post(
        "/v0/elevator/enqueue",
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 401
    assert "bearer token" in response.json()["error"]["message"].lower()


def test_bearer_auth_wrong_token_enqueue(client_with_auth):
    """Test that enqueue rejects wrong bearer token."""
    response = client_with_auth.post(
        "/v0/elevator/enqueue",
        headers={"Authorization": "Bearer wrong-token"},
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 401


def test_bearer_auth_valid_token_enqueue(client_with_auth):
    """Test that enqueue accepts valid bearer token."""
    response = client_with_auth.post(
        "/v0/elevator/enqueue",
        headers={"Authorization": "Bearer test-secret-token"},
        json={
            "lane": "interactive",
            "kind": "test",
            "payload": {},
            "principal": "p1",
            "latency_class": "interactive",
        },
    )
    assert response.status_code == 201


def test_bearer_auth_required_claim(client_with_auth):
    """Test that claim requires bearer token."""
    response = client_with_auth.post(
        "/v0/elevator/claim",
        json={
            "lanes": ["interactive"],
            "owner": "broker",
            "claim_ttl_sec": 30,
        },
    )
    assert response.status_code == 401


def test_bearer_auth_required_ack(client_with_auth):
    """Test that ack requires bearer token."""
    response = client_with_auth.post(
        "/v0/elevator/ack",
        json={"item_id": "test-id"},
    )
    assert response.status_code == 401


def test_off_master_write_protection(client):
    """Test that writes from non-master nodes are rejected."""
    with mock.patch("agents_core.elevator.IS_MASTER", False):
        response = client.post(
            "/v0/elevator/enqueue",
            json={
                "lane": "interactive",
                "kind": "test",
                "payload": {},
                "principal": "p1",
                "latency_class": "interactive",
            },
        )
        # Should return 403 Forbidden (off-master write attempt).
        assert response.status_code == 403
        assert "off-master" in response.json()["error"]["message"].lower()


# --- AC4, AC5: exclude_kinds passthrough and state() gw_admission counts ---


def _enqueue_kind(client, lane, kind):
    return client.post(
        "/v0/elevator/enqueue",
        json={"lane": lane, "kind": kind, "payload": {}, "principal": "test", "latency_class": "batch"},
    )


def test_claim_no_exclude_kinds_unchanged(client):
    """AC4: POST /v0/elevator/claim with no exclude_kinds behaves exactly as today."""
    _enqueue_kind(client, "deliberation", "sample")
    resp = client.post(
        "/v0/elevator/claim",
        json={"owner": "broker", "lanes": ["deliberation"], "claim_ttl_sec": 30},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "sample"
    assert data["status"] == "claimed"


def test_claim_exclude_kinds_honored_via_http(client):
    """AC4: POST with exclude_kinds:['gw-admission'] skips gw-admission, claims sample."""
    import time

    _enqueue_kind(client, "deliberation", "gw-admission")
    time.sleep(0.01)
    _enqueue_kind(client, "deliberation", "sample")

    resp = client.post(
        "/v0/elevator/claim",
        json={
            "owner": "broker",
            "lanes": ["deliberation"],
            "claim_ttl_sec": 30,
            "exclude_kinds": ["gw-admission"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "sample"


def test_claim_exclude_kinds_only_gw_admission_returns_none(client):
    """AC4: with only gw-admission ticket and exclude_kinds=['gw-admission'], returns null."""
    _enqueue_kind(client, "deliberation", "gw-admission")
    resp = client.post(
        "/v0/elevator/claim",
        json={
            "owner": "broker",
            "lanes": ["deliberation"],
            "claim_ttl_sec": 30,
            "exclude_kinds": ["gw-admission"],
        },
    )
    assert resp.status_code == 200
    assert resp.json() is None


def test_state_gw_admission_counts_present_and_zero(client):
    """AC5: state() carries gw_admission_pending/claimed on every lane, zero when empty."""
    resp = client.get("/v0/elevator/state")
    assert resp.status_code == 200
    data = resp.json()
    for lane_data in data["queue"].values():
        assert "gw_admission_pending" in lane_data
        assert "gw_admission_claimed" in lane_data
        assert lane_data["gw_admission_pending"] == 0
        assert lane_data["gw_admission_claimed"] == 0
        # Existing keys untouched.
        assert "pending" in lane_data
        assert "claimed" in lane_data
        assert "oldest_age_sec" in lane_data


def test_state_gw_admission_counts_correct(client):
    """AC5: gw_admission_pending/claimed equal true filtered counts."""
    _enqueue_kind(client, "deliberation", "gw-admission")
    _enqueue_kind(client, "deliberation", "gw-admission")
    _enqueue_kind(client, "deliberation", "wave")

    resp = client.get("/v0/elevator/state")
    delib = resp.json()["queue"]["deliberation"]
    assert delib["gw_admission_pending"] == 2
    assert delib["gw_admission_claimed"] == 0
    assert delib["pending"] == 3
