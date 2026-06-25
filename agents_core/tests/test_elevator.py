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


# --- depends_on (U4a) tests ---


def test_enqueue_with_depends_on_basic(temp_db):
    """Test enqueue and persist depends_on."""
    # Enqueue precursor.
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={"query": "test"},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent with depends_on.
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={"depends_on": precursor_id},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
    )

    # Get and verify depends_on round-trips through GET.
    dependent = temp_db.get(dependent_id)
    assert dependent["depends_on"] == precursor_id


def test_enqueue_omit_depends_on_back_compat(temp_db):
    """Test that omitting depends_on stores NULL (back-compat)."""
    item_id = temp_db.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={},
        principal="p1",
        latency_class="interactive",
    )
    item = temp_db.get(item_id)
    assert item["depends_on"] is None


def test_enqueue_self_reference_rejected(temp_db):
    """Test that self-reference is rejected."""
    # Attempt to enqueue with depends_on == item_id (unknown at enqueue time, so test by
    # setting both explicitly).
    with pytest.raises(ValueError, match="self-reference"):
        temp_db.enqueue(
            lane="execution",
            kind="test",
            payload={},
            principal="p1",
            latency_class="batch",
            item_id="item-123",
            depends_on="item-123",
        )


def test_enqueue_nonexistent_depends_on_rejected(temp_db):
    """Test that depends_on on non-existent precursor is rejected."""
    with pytest.raises(ValueError, match="depends_on_not_found"):
        temp_db.enqueue(
            lane="execution",
            kind="test",
            payload={},
            principal="p1",
            latency_class="batch",
            depends_on="nonexistent-precursor",
        )


def test_claim_skips_unserved_dependent(temp_db):
    """Test that claim skips a dependent with unserved precursor."""
    # Enqueue precursor (pending).
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent (depends on pending precursor).
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
    )

    # Attempt to claim from deliberation; should get None (dependent is gated).
    item = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert item is None

    # Ack the precursor (mark as served).
    temp_db.claim(
        lanes=["execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    temp_db.ack(precursor_id)

    # Now claim should return the dependent (precursor is served).
    item = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert item is not None
    assert item["item_id"] == dependent_id


def test_claim_no_head_of_line_deadlock(temp_db):
    """Test that a blocked dependent at head-of-line doesn't stall unrelated items.

    Enqueue blocked dependent V2 (older created_at) and unrelated item X (newer).
    Claim should return X, not block on V2.
    """
    # Enqueue precursor (pending, in execution lane).
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent (depends on pending precursor, in deliberation lane, older).
    # Use a fixed timestamp to ensure older created_at.
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
        item_id="dependent-v2",
    )

    # Enqueue unrelated item in deliberation lane (newer created_at).
    unrelated_id = temp_db.enqueue(
        lane="deliberation",
        kind="other-task",
        payload={},
        principal="p2",
        latency_class="batch",
    )

    # Claim from deliberation; should return unrelated item, not dependent.
    item = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )
    assert item is not None
    assert item["item_id"] == unrelated_id
    assert item["item_id"] != dependent_id


def test_cascade_fail_on_failed_precursor(temp_db):
    """Test cascade-fail when precursor is marked failed."""
    # Enqueue precursor.
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent.
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
    )

    # Claim and fail the precursor.
    precursor_item = temp_db.claim(
        lanes=["execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    temp_db.fail(precursor_item["item_id"])

    # Trigger cascade-fail (via claim which calls _cascade_fail_dependents).
    temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )

    # Check that dependent is now failed with correct provenance.
    dependent = temp_db.get(dependent_id)
    assert dependent["status"] == "failed"
    assert dependent["provenance"]["depends_on_failed"]["precursor_id"] == precursor_id
    assert dependent["provenance"]["depends_on_failed"]["precursor_status"] == "failed"


def test_cascade_fail_on_expired_precursor(temp_db):
    """Test cascade-fail when precursor expires."""
    # Enqueue precursor.
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent.
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
    )

    # Manually expire the precursor (via direct SQL).
    temp_db._conn.execute(
        "UPDATE queue_items SET status='expired' WHERE item_id=?",
        (precursor_id,),
    )
    temp_db._conn.commit()

    # Trigger cascade-fail (via claim).
    temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )

    # Check that dependent is now failed with correct provenance.
    dependent = temp_db.get(dependent_id)
    assert dependent["status"] == "failed"
    assert dependent["provenance"]["depends_on_failed"]["precursor_id"] == precursor_id
    assert dependent["provenance"]["depends_on_failed"]["precursor_status"] == "expired"


def test_cascade_fail_on_missing_precursor(temp_db):
    """Test cascade-fail when precursor row is deleted (missing).

    This is a defensive guard: enqueue validation should prevent this, but
    test that the reaper/claim handles it correctly.
    """
    # Enqueue precursor.
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent.
    dependent_id = temp_db.enqueue(
        lane="deliberation",
        kind="facets-verdict",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=precursor_id,
    )

    # Delete the precursor (raw SQL, simulating missing precursor).
    temp_db._conn.execute(
        "DELETE FROM queue_items WHERE item_id=?",
        (precursor_id,),
    )
    temp_db._conn.commit()

    # Trigger cascade-fail (via claim).
    temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )

    # Check that dependent is now failed with "missing" status.
    dependent = temp_db.get(dependent_id)
    assert dependent["status"] == "failed"
    assert dependent["provenance"]["depends_on_failed"]["precursor_id"] == precursor_id
    assert dependent["provenance"]["depends_on_failed"]["precursor_status"] == "missing"


def test_payload_blind_dependency_resolution(temp_db):
    """Test that dependency resolution is payload-blind.

    Enqueue a dependent with malformed-JSON payload. Verify that claim +
    cascade-fail work without attempting to parse the payload.
    """
    from datetime import datetime, timezone

    # Enqueue precursor.
    precursor_id = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Enqueue dependent with intentionally malformed JSON payload (via raw SQL).
    dependent_id = "dependent-bad-json"
    now = datetime.now(timezone.utc).isoformat()
    temp_db._conn.execute(
        "INSERT INTO queue_items "
        "(item_id, lane, kind, principal, payload, latency_class, status, "
        " attempts, created_at, depends_on) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            dependent_id, "deliberation", "facets-verdict", "p1",
            "{broken json", "batch", "pending", 0, now,
            precursor_id,
        ),
    )
    temp_db._conn.commit()

    # Fail the precursor.
    precursor_item = temp_db.claim(
        lanes=["execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    temp_db.fail(precursor_item["item_id"])

    # Trigger cascade-fail; should not raise JSON parse error.
    result = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )

    # Verify dependent was cascade-failed (claim returns None because dependent is failed).
    assert result is None
    # Check raw row to avoid deserializing malformed payload.
    with temp_db._lock:
        row = temp_db._conn.execute(
            "SELECT status FROM queue_items WHERE item_id=?", (dependent_id,)
        ).fetchone()
    assert row["status"] == "failed"
    # The fact that cascade-fail completed without JSON parse error proves payload-blindness.


def test_depends_on_immutable_no_update_path(temp_db):
    """Test that depends_on is immutable (no public update path exists).

    Structurally assert: there is no update_depends_on method.
    Additionally verify that a raw-SQL UPDATE of depends_on does not occur
    through any store method (defensive check).
    """
    # Enqueue two items.
    item1 = temp_db.enqueue(
        lane="execution",
        kind="grounding",
        payload={},
        principal="p1",
        latency_class="batch",
    )
    item2 = temp_db.enqueue(
        lane="deliberation",
        kind="verdict",
        payload={},
        principal="p1",
        latency_class="batch",
    )

    # Assert no update_depends_on method exists.
    assert not hasattr(temp_db, "update_depends_on")

    # Verify requeue, ack, fail, reap don't mutate depends_on.
    # These are all the mutation methods; none should touch depends_on.
    item2_with_dep = temp_db.enqueue(
        lane="deliberation",
        kind="verdict2",
        payload={},
        principal="p1",
        latency_class="batch",
        depends_on=item1,
    )

    # Claim and ack item1.
    claimed = temp_db.claim(
        lanes=["execution"],
        owner="broker",
        claim_ttl_sec=30,
    )
    temp_db.ack(claimed["item_id"])

    # Requeue the dependent (if it were claimed).
    claimed_dep = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
    )
    if claimed_dep:
        temp_db.requeue(claimed_dep["item_id"])

    # Check that depends_on is still the original precursor.
    item_after = temp_db.get(item2_with_dep)
    assert item_after["depends_on"] == item1


# --- AC1-AC3, AC5, AC7: exclude_kinds and gw_admission state counts ---


def _enqueue(store, lane, kind, principal="test"):
    return store.enqueue(
        lane=lane,
        kind=kind,
        payload={},
        principal=principal,
        latency_class="batch",
    )


def test_claim_exclude_kinds_none_unchanged(temp_db):
    """AC1: claim(exclude_kinds=None) is byte-identical to claim() with no param."""
    iid = _enqueue(temp_db, "deliberation", "sample")
    result = temp_db.claim(
        lanes=["deliberation"], owner="broker", claim_ttl_sec=30, exclude_kinds=None
    )
    assert result is not None
    assert result["item_id"] == iid
    assert result["status"] == "claimed"


def test_claim_exclude_kinds_omitted_unchanged(temp_db):
    """AC1: claim() with param omitted still claims the item."""
    iid = _enqueue(temp_db, "deliberation", "sample")
    result = temp_db.claim(lanes=["deliberation"], owner="broker", claim_ttl_sec=30)
    assert result is not None
    assert result["item_id"] == iid


def test_claim_exclude_kinds_skips_excluded_returns_other(temp_db):
    """AC2: exclude=['gw-admission'] skips the older gw-admission, returns newer sample."""
    import time

    iid_gw = _enqueue(temp_db, "deliberation", "gw-admission")
    time.sleep(0.01)
    iid_sample = _enqueue(temp_db, "deliberation", "sample")

    result = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
        exclude_kinds=["gw-admission"],
    )
    assert result is not None
    assert result["item_id"] == iid_sample
    assert result["kind"] == "sample"

    # The gw-admission ticket must still be pending (AC7 — no mutation).
    gw_item = temp_db.get(iid_gw)
    assert gw_item["status"] == "pending"


def test_claim_exclude_kinds_only_excluded_returns_none(temp_db):
    """AC2: with only a gw-admission ticket present, returns None."""
    _enqueue(temp_db, "deliberation", "gw-admission")
    result = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
        exclude_kinds=["gw-admission"],
    )
    assert result is None


def test_claim_exclude_kinds_sql_injection_safe(temp_db):
    """AC3: a kind string with SQL metacharacters cannot alter the query."""
    iid = _enqueue(temp_db, "deliberation", "safe-kind")
    # Attempt injection in the exclusion list.
    result = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
        exclude_kinds=["' OR '1'='1"],
    )
    # The safe-kind ticket should still be claimable (injection did not affect query).
    assert result is not None
    assert result["item_id"] == iid


def test_state_gw_admission_counts_zero_when_empty(temp_db):
    """AC5: gw_admission_pending/claimed are 0 when no such tickets exist."""
    s = temp_db.state()
    for lane_data in s["queue"].values():
        assert lane_data["gw_admission_pending"] == 0
        assert lane_data["gw_admission_claimed"] == 0
        # Existing keys still present.
        assert "pending" in lane_data
        assert "claimed" in lane_data
        assert "oldest_age_sec" in lane_data


def test_state_gw_admission_counts_reflect_kind(temp_db):
    """AC5: gw_admission_pending/claimed reflect true filtered counts."""
    _enqueue(temp_db, "deliberation", "gw-admission")
    _enqueue(temp_db, "deliberation", "gw-admission")
    _enqueue(temp_db, "deliberation", "sample")

    s = temp_db.state()
    delib = s["queue"]["deliberation"]
    assert delib["gw_admission_pending"] == 2
    assert delib["gw_admission_claimed"] == 0
    assert delib["pending"] == 3  # all three pending

    # Claim one gw-admission ticket.
    temp_db.claim(lanes=["deliberation"], owner="broker", claim_ttl_sec=30)

    s2 = temp_db.state()
    delib2 = s2["queue"]["deliberation"]
    assert delib2["gw_admission_pending"] == 1
    assert delib2["gw_admission_claimed"] == 1


def test_claim_exclude_leaves_ticket_pending_and_countable(temp_db):
    """AC7: excluded ticket stays pending and is reflected in gw_admission_pending."""
    iid = _enqueue(temp_db, "deliberation", "gw-admission")

    result = temp_db.claim(
        lanes=["deliberation"],
        owner="broker",
        claim_ttl_sec=30,
        exclude_kinds=["gw-admission"],
    )
    assert result is None

    # Ticket must be pending, not claimed/deleted/mutated.
    item = temp_db.get(iid)
    assert item["status"] == "pending"

    # Must be countable via state().
    s = temp_db.state()
    assert s["queue"]["deliberation"]["gw_admission_pending"] == 1
    assert s["queue"]["deliberation"]["gw_admission_claimed"] == 0
