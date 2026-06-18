"""Tests for ElevatorStore — the GW-lane work queue."""

import json
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from agents_core.elevator import (
    ELEVATOR_PENDING_MAX_AGE_SEC,
    ElevatorStore,
    OffMasterWriteError,
    QueueNotFoundError,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_elevator.db"
        store = ElevatorStore(db_path)
        yield store
        store.close()


def test_enqueue_basic(temp_db):
    """Test basic enqueue."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "hello", "context": {}},
        principal="session-123",
        latency_class="interactive",
    )
    assert item_id
    item = temp_db.get(item_id)
    assert item["status"] == "pending"
    assert item["lane"] == "interactive"
    assert item["kind"] == "session-turn"
    assert item["principal"] == "session-123"
    assert item["payload"]["prompt"] == "hello"
    assert item["attempts"] == 0


def test_enqueue_with_slot_ref(temp_db):
    """Test enqueue with optional slot_ref."""
    item_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-council-review",
        payload={"verdict": "approve"},
        principal="elevator/facets",
        latency_class="batch",
        slot_ref="slot-abc123",
    )
    item = temp_db.get(item_id)
    assert item["slot_ref"] == "slot-abc123"


def test_enqueue_invalid_lane(temp_db):
    """Test enqueue rejects invalid lane."""
    with pytest.raises(ValueError, match="invalid lane"):
        temp_db.enqueue(
            lane="invalid",
            kind="test",
            payload={},
            principal="test",
            latency_class="interactive",
        )


def test_enqueue_invalid_latency_class(temp_db):
    """Test enqueue rejects invalid latency_class."""
    with pytest.raises(ValueError, match="invalid latency_class"):
        temp_db.enqueue(
            lane="interactive",
            kind="test",
            payload={},
            principal="test",
            latency_class="fast",
        )


def test_enqueue_duplicate_item_id(temp_db):
    """Test enqueue rejects duplicate item_id."""
    iid = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="test",
        latency_class="interactive",
        item_id="custom-id",
    )
    assert iid == "custom-id"

    with pytest.raises(ValueError, match="already exists"):
        temp_db.enqueue(
            lane="interactive",
            kind="test",
            payload={},
            principal="test",
            latency_class="interactive",
            item_id="custom-id",
        )


def test_claim_single_item(temp_db):
    """Test claiming a single item."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="producer-1",
        latency_class="interactive",
    )

    item = temp_db.claim(
        lanes=["interactive", "deliberation", "execution"],
        owner="broker-1",
        claim_ttl_sec=30,
    )
    assert item is not None
    assert item["item_id"] == item_id
    assert item["status"] == "claimed"
    assert item["claim_owner"] == "broker-1"
    assert item["claim_ttl_sec"] == 30
    assert item["claimed_at"] is not None


def test_claim_fifo_within_lane(temp_db):
    """Test that claim respects FIFO ordering within a lane."""
    # Enqueue three items in order.
    id1 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )
    id2 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p2",
        latency_class="interactive",
    )
    id3 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p3",
        latency_class="interactive",
    )

    # Claim them; should come in FIFO order.
    claim1 = temp_db.claim(
        lanes=["interactive"], owner="broker", claim_ttl_sec=30
    )
    assert claim1["item_id"] == id1

    claim2 = temp_db.claim(
        lanes=["interactive"], owner="broker", claim_ttl_sec=30
    )
    assert claim2["item_id"] == id2

    claim3 = temp_db.claim(
        lanes=["interactive"], owner="broker", claim_ttl_sec=30
    )
    assert claim3["item_id"] == id3


def test_claim_lane_priority(temp_db):
    """Test that claim respects lane priority (interactive > deliberation > execution)."""
    # Enqueue in mixed order: deliberation, execution, interactive.
    exec_id = temp_db.enqueue(
        lane="execution",
        kind="fixer",
        payload={},
        principal="p1",
        latency_class="batch",
    )
    delibid = temp_db.enqueue(
        lane="deliberation",
        kind="council-review",
        payload={},
        principal="p2",
        latency_class="batch",
    )
    inter_id = temp_db.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={},
        principal="p3",
        latency_class="interactive",
    )

    # Claim across all lanes; should return interactive first (priority).
    claim1 = temp_db.claim(
        lanes=["interactive", "deliberation", "execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert claim1["item_id"] == inter_id

    # Next claim should return deliberation.
    claim2 = temp_db.claim(
        lanes=["interactive", "deliberation", "execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert claim2["item_id"] == delibid

    # Next claim should return execution.
    claim3 = temp_db.claim(
        lanes=["interactive", "deliberation", "execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert claim3["item_id"] == exec_id


def test_claim_no_pending_items(temp_db):
    """Test claim returns None when no pending items exist."""
    result = temp_db.claim(
        lanes=["interactive", "deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert result is None


def test_claim_respects_lane_filter(temp_db):
    """Test that claim only considers specified lanes."""
    exec_id = temp_db.enqueue(
        lane="execution",
        kind="fixer",
        payload={},
        principal="p1",
        latency_class="batch",
    )
    inter_id = temp_db.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={},
        principal="p2",
        latency_class="interactive",
    )

    # Claim only from execution lane.
    claim = temp_db.claim(
        lanes=["execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert claim["item_id"] == exec_id

    # Claim only from interactive lane.
    claim2 = temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert claim2["item_id"] == inter_id


def test_ack(temp_db):
    """Test acknowledging a claimed item."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )

    result = temp_db.ack(
        item_id,
        result_ref="result://abc123",
        provenance={"phase": "big", "wait_ms": 5000},
    )
    assert result is True

    item = temp_db.get(item_id)
    assert item["status"] == "served"
    assert item["result_ref"] == "result://abc123"
    assert item["served_at"] is not None
    assert item["provenance"]["phase"] == "big"


def test_ack_idempotent(temp_db):
    """Test that ack is idempotent on an already-served item."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )

    result1 = temp_db.ack(item_id, result_ref="result1")
    assert result1 is True

    # Ack again; should return True without error.
    result2 = temp_db.ack(item_id, result_ref="result2")
    assert result2 is True

    # Verify the first result_ref is preserved (idempotent).
    item = temp_db.get(item_id)
    assert item["result_ref"] == "result1"


def test_ack_nonexistent_item(temp_db):
    """Test ack raises QueueNotFoundError for nonexistent item."""
    with pytest.raises(QueueNotFoundError):
        temp_db.ack("nonexistent-item-id")


def test_requeue(temp_db):
    """Test requeuing a claimed item back to pending."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    # Claim it.
    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )
    item = temp_db.get(item_id)
    assert item["status"] == "claimed"
    assert item["attempts"] == 0

    # Requeue.
    result = temp_db.requeue(item_id)
    assert result is True

    item = temp_db.get(item_id)
    assert item["status"] == "pending"
    assert item["attempts"] == 1
    assert item["claim_owner"] is None
    assert item["claim_ttl_sec"] is None


def test_requeue_increments_attempts(temp_db):
    """Test that requeue increments attempts counter."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    for expected_attempts in [1, 2, 3]:
        temp_db.claim(
            lanes=["interactive"],
            owner="broker",
            claim_ttl_sec=30,
        )
        temp_db.requeue(item_id)
        item = temp_db.get(item_id)
        assert item["attempts"] == expected_attempts


def test_requeue_nonexistent_item(temp_db):
    """Test requeue raises QueueNotFoundError for nonexistent item."""
    with pytest.raises(QueueNotFoundError):
        temp_db.requeue("nonexistent-item-id")


def test_fail(temp_db):
    """Test failing a claimed item."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )

    result = temp_db.fail(item_id)
    assert result is True

    item = temp_db.get(item_id)
    assert item["status"] == "failed"


def test_fail_nonexistent_item(temp_db):
    """Test fail raises QueueNotFoundError for nonexistent item."""
    with pytest.raises(QueueNotFoundError):
        temp_db.fail("nonexistent-item-id")


def test_get_nonexistent_item(temp_db):
    """Test get returns None for nonexistent item."""
    result = temp_db.get("nonexistent-item-id")
    assert result is None


def test_state_returns_queue_block(temp_db):
    """Test that state() returns queue block with lane counts."""
    # Enqueue items across lanes.
    temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )
    temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p2",
        latency_class="interactive",
    )
    temp_db.enqueue(
        lane="deliberation",
        kind="test",
        payload={},
        principal="p3",
        latency_class="batch",
    )

    state = temp_db.state()
    assert "queue" in state
    assert "gw" in state
    assert "as_of" in state

    queue = state["queue"]
    assert queue["interactive"]["pending"] == 2
    assert queue["deliberation"]["pending"] == 1
    assert queue["execution"]["pending"] == 0
    assert queue["interactive"]["claimed"] == 0
    assert queue["interactive"]["oldest_age_sec"] is not None


def test_state_tracks_claimed(temp_db):
    """Test that state() tracks claimed items."""
    id1 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )
    id2 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p2",
        latency_class="interactive",
    )

    # Claim one item.
    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=30,
    )

    state = temp_db.state()
    assert state["queue"]["interactive"]["pending"] == 1
    assert state["queue"]["interactive"]["claimed"] == 1


def test_stale_claim_reclamation(temp_db):
    """Test that the reaper reclaims stale claims."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )

    # Claim with a very short TTL.
    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=1,
    )
    item = temp_db.get(item_id)
    assert item["status"] == "claimed"

    # Sleep to exceed TTL.
    time.sleep(2)

    # Trigger reap (inline).
    temp_db._reap_inline()

    # Item should be back to pending with attempts incremented.
    item = temp_db.get(item_id)
    assert item["status"] == "pending"
    assert item["attempts"] == 1
    assert item["claim_owner"] is None


def test_broker_can_claim_any_item(temp_db):
    """Test that a non-owner broker can claim items enqueued by any principal.

    This proves the non-owner broker access that SlotStore structurally cannot give.
    """
    # Producer enqueues.
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="producer-session",
        latency_class="interactive",
    )

    # Different principal (broker) claims it.
    claimed = temp_db.claim(
        lanes=["interactive"],
        owner="broker-agent",
        claim_ttl_sec=30,
    )
    assert claimed["item_id"] == item_id
    assert claimed["claim_owner"] == "broker-agent"
    assert claimed["principal"] == "producer-session"

    # Broker acks it.
    temp_db.ack(item_id, result_ref="result://xyz")
    item = temp_db.get(item_id)
    assert item["status"] == "served"


def test_synthetic_demo(temp_db):
    """Integration test: synthetic demo (no fabricated human attribution).

    - A synthetic principal enqueues an interactive session-turn
    - GET /state shows interactive.pending == 1
    - A different synthetic broker claims it (non-owner access)
    - Broker acks it with provenance
    - GET /item shows served with provenance
    - /state shows interactive.pending == 0
    """
    # Enqueue.
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "test prompt", "context": {}},
        principal="elevator/smoke-session",
        latency_class="interactive",
    )
    assert item_id

    # Check state: pending == 1.
    state = temp_db.state()
    assert state["queue"]["interactive"]["pending"] == 1

    # Broker claims (different principal).
    claimed = temp_db.claim(
        lanes=["interactive", "deliberation", "execution"],
        owner="elevator/smoke-broker",
        claim_ttl_sec=30,
    )
    assert claimed["item_id"] == item_id
    assert claimed["claim_owner"] == "elevator/smoke-broker"

    # Broker acks with provenance.
    temp_db.ack(
        item_id,
        result_ref="result://synthetic-demo-result",
        provenance={
            "phase": "big",
            "wait_ms": 5000,
            "model": "claude-sonnet-4-6",
            "served_at": "2026-06-17T12:00:00Z",
        },
    )

    # Check state: pending == 0.
    state = temp_db.state()
    assert state["queue"]["interactive"]["pending"] == 0

    # Get item: should be served with provenance.
    item = temp_db.get(item_id)
    assert item["status"] == "served"
    assert item["result_ref"] == "result://synthetic-demo-result"
    assert item["provenance"]["phase"] == "big"
    assert item["provenance"]["wait_ms"] == 5000


def test_reap_returns_counts(temp_db):
    """Test that reap() returns actual expired and reclaimed counts."""
    # Enqueue items with short TTL and aged timestamps.
    item_id1 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p1",
        latency_class="interactive",
    )
    item_id2 = temp_db.enqueue(
        lane="interactive",
        kind="test",
        payload={},
        principal="p2",
        latency_class="interactive",
    )

    # Claim one item with very short TTL.
    temp_db.claim(
        lanes=["interactive"],
        owner="broker",
        claim_ttl_sec=1,
    )

    # Sleep to exceed TTL.
    time.sleep(2)

    # Reap should return counts.
    result = temp_db.reap()
    assert isinstance(result, dict)
    assert "expired" in result
    assert "reclaimed" in result
    assert result["reclaimed"] == 1  # One stale claim reclaimed
    # The other item is still pending (not aged enough yet).


def test_compose_gw_status_with_flip_controller(temp_db):
    """Test _compose_gw_status with mocked flip-controller response."""
    with mock.patch("agents_core.elevator.httpx") as mock_httpx:
        mock_client = mock.MagicMock()
        mock_httpx.get = mock.MagicMock()

        # Mock flip-controller response.
        flip_resp = mock.MagicMock()
        flip_resp.status_code = 200
        flip_resp.json.return_value = {"mode": "big"}

        # Mock doorman response.
        doorman_resp = mock.MagicMock()
        doorman_resp.status_code = 200
        doorman_resp.json.return_value = {"nodes": {"gravitywell": {"serving": True}}}

        def mock_get(url, timeout=None):
            if "8408" in url:
                return flip_resp
            elif "8407" in url:
                return doorman_resp
            raise Exception("Unknown URL")

        mock_httpx.get = mock_get

        result = temp_db._compose_gw_status()
        assert result["mode"] == "big"
        assert result["serving_ready"] is True


def test_compose_gw_status_degraded(temp_db):
    """Test _compose_gw_status degrades gracefully when services unavailable."""
    with mock.patch("agents_core.elevator.httpx") as mock_httpx:
        # Simulate httpx not available or timeout.
        mock_httpx.get = mock.MagicMock(side_effect=Exception("Connection timeout"))

        result = temp_db._compose_gw_status()
        assert result["mode"] == "unknown"
        assert result["serving_ready"] is None


def test_off_master_enqueue_rejects(temp_db):
    """Test that enqueue rejects on non-master node."""
    with mock.patch("agents_core.elevator.IS_MASTER", False):
        store = temp_db
        with pytest.raises(OffMasterWriteError):
            store.enqueue(
                lane="interactive",
                kind="test",
                payload={},
                principal="p1",
                latency_class="interactive",
            )
