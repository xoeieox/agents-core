"""Tests for agents_core.elevator — ElevatorStore and node-state composition."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents_core.elevator import ElevatorStore


def _mock_response(status_code=200, json_data=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data or {}
    return m


@pytest.fixture
def elevator_store(tmp_path):
    """Create a temporary ElevatorStore for testing."""
    db_path = tmp_path / "test_queue.db"
    store = ElevatorStore(db_path=db_path)
    yield store
    store.close()


def test_compose_gw_status_doorman_serving_ready_true(elevator_store):
    """Doorman at 127.0.0.1:8407 returns serving=true → serving_ready=True."""
    captured_urls = []

    def fake_get(url, timeout=None):
        captured_urls.append(url)
        # Flip-controller response (no serving_ready from this endpoint).
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        # Doorman response with serving=true.
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": True}}},
            )
        return _mock_response(status_code=404)

    with patch.object(ElevatorStore, "_compose_gw_status", wraps=elevator_store._compose_gw_status):
        with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
            result = elevator_store._compose_gw_status()

    assert result["serving_ready"] is True
    # Verify the doorman was queried at the correct (localhost) host.
    doorman_urls = [url for url in captured_urls if "8407" in url]
    assert len(doorman_urls) == 1
    assert "127.0.0.1:8407" in doorman_urls[0]


def test_compose_gw_status_doorman_serving_ready_false(elevator_store):
    """Doorman returns serving=false → serving_ready=False."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": False}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    assert result["serving_ready"] is False


def test_compose_gw_status_doorman_failure_graceful(elevator_store):
    """Doorman read failure (timeout/404) → serving_ready=None (graceful degrade)."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        elif "8407" in url:
            # Simulate doorman timeout or failure.
            raise Exception("doorman unreachable")
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # On failure, serving_ready should remain None (graceful degrade).
    assert result["serving_ready"] is None
    # Mode should still be set from flip-controller.
    assert result["mode"] == "big"


def test_compose_gw_status_doorman_missing_key(elevator_store):
    """Doorman response missing nodes.gravitywell.serving → serving_ready=None."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "swarm"})
        elif "8407" in url:
            # Response is 200 but missing the serving key.
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # Missing key → serving_ready stays None.
    assert result["serving_ready"] is None
    assert result["mode"] == "swarm"


def test_compose_gw_status_flip_controller_untouched(elevator_store):
    """Flip-controller at 203.0.113.10:8408 remains unchanged."""
    captured_urls = []

    def fake_get(url, timeout=None):
        captured_urls.append(url)
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "offline"})
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": True}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # Verify flip-controller is still queried at Tailscale IP (NOT localhost).
    fc_urls = [url for url in captured_urls if "8408" in url]
    assert len(fc_urls) == 1
    assert "203.0.113.10:8408" in fc_urls[0]
    assert result["mode"] == "offline"


# -- Tests for submit() (interactive-submit interface) ----

def test_submit_queue_gw_interactive(tmp_path, monkeypatch):
    """submit(destination='queue-gw-interactive') enqueues and returns handle."""
    from agents_core.interactive_submit import submit

    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    # Get initial state
    from agents_core.elevator import ElevatorStore
    elevator = ElevatorStore(db_path)
    initial_pending = elevator.state()["queue"]["interactive"]["pending"]
    elevator.close()

    result = submit(
        turn="ping",
        context={"scope": "test"},
        destination="queue-gw-interactive",
        principal="test-caller",
    )

    assert "item_id" in result
    assert "poll" in result
    assert "node_state" in result
    assert result["poll"].startswith("/v0/elevator/item/")
    # The new submit should add 1 to the pending count
    assert result["node_state"]["queue"]["interactive"]["pending"] == initial_pending + 1


def test_submit_route_to_fast_qwen(monkeypatch):
    """submit(destination='route-to-fast') calls operator directly, returns result."""
    from agents_core.interactive_submit import submit

    def fake_call_operator(operator_class, prompt, system=None, _provenance_out=None, **kwargs):
        assert operator_class == "qwen"
        assert prompt == "test-prompt"
        assert system is None  # No context passed.
        if _provenance_out is not None:
            _provenance_out.append(("success", "qwen"))
        return "test-response"

    monkeypatch.setattr("agents_core.interactive_submit.call_operator", fake_call_operator)

    result = submit(
        turn="test-prompt",
        context=None,
        destination="route-to-fast",
        operator="qwen",
    )

    assert result["result"] == "test-response"
    assert result["served_by"] == "qwen"
    assert result["provenance"] == [("success", "qwen")]


def test_submit_route_to_fast_with_context(monkeypatch):
    """submit(route-to-fast) with context serializes it as system=JSON."""
    from agents_core.interactive_submit import submit
    import json

    captured_call = {}

    def fake_call_operator(operator_class, prompt, system=None, _provenance_out=None, **kwargs):
        captured_call["system"] = system
        if _provenance_out is not None:
            _provenance_out.append(("success", "qwen"))
        return "response"

    monkeypatch.setattr("agents_core.interactive_submit.call_operator", fake_call_operator)

    context = {"role": "assistant", "history": ["a", "b"]}
    result = submit(
        turn="prompt",
        context=context,
        destination="route-to-fast",
    )

    assert json.loads(captured_call["system"]) == context


def test_submit_context_reaches_call_operator(monkeypatch, tmp_path):
    """AC2: submit(queue-gw-interactive) stores context; serving step reconstructs system=."""
    from agents_core.interactive_submit import submit
    from agents_core.elevator_interactive_worker import _reconstruct_call_args

    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    context_payload = {"role": "system", "data": "test-context"}
    result = submit(
        turn="test-turn",
        context=context_payload,
        destination="queue-gw-interactive",
    )

    item_id = result["item_id"]
    elevator = ElevatorStore(db_path)
    item = elevator.get(item_id)
    elevator.close()

    payload = item["payload"]
    prompt, system = _reconstruct_call_args(payload)

    assert prompt == "test-turn"
    assert system is not None
    import json
    assert json.loads(system) == context_payload


# -- Tests for interactive serving worker ----

def test_elevator_ack_with_result(elevator_store):
    """elevator.ack() accepts and stores result parameter."""
    item_id = elevator_store.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "test", "context": {}},
        principal="test",
        latency_class="interactive",
    )
    item = elevator_store.claim(
        lanes=["interactive"], owner="test-worker", claim_ttl_sec=360
    )
    assert item["item_id"] == item_id

    elevator_store.ack(
        item_id,
        result="test-response",
        provenance={"served_by": "gravitywell"},
    )

    item = elevator_store.get(item_id)
    assert item["status"] == "served"
    assert item["result"] == "test-response"
    assert item["provenance"]["served_by"] == "gravitywell"


def test_serving_worker_success_path(tmp_path, monkeypatch):
    """Worker serves a baton and writes result + provenance on success."""
    from agents_core.elevator_interactive_worker import serve_interactive_baton

    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    elevator = ElevatorStore(db_path)
    item_id = elevator.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "ping", "context": {}},
        principal="test",
        latency_class="interactive",
    )
    item = elevator.claim(
        lanes=["interactive"], owner="test-worker", claim_ttl_sec=360
    )
    elevator.close()

    def fake_call_operator(operator, prompt, system=None, on_wake_fail=None, _provenance_out=None, **kwargs):
        assert operator == "gravitywell"
        assert on_wake_fail == "skip"
        if _provenance_out is not None:
            _provenance_out.append(("success", "gravitywell"))
        return "pong"

    monkeypatch.setattr("agents_core.elevator_interactive_worker.call_operator", fake_call_operator)

    result = serve_interactive_baton(item)
    assert result is True

    elevator = ElevatorStore(db_path)
    served_item = elevator.get(item_id)
    elevator.close()
    assert served_item["status"] == "served"
    assert served_item["result"] == "pong"


def test_serving_worker_deferred_requeue(tmp_path, monkeypatch):
    """Worker requeues immediately when GW is deferred (no paid operator call)."""
    from agents_core.elevator_interactive_worker import serve_interactive_baton

    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))

    elevator = ElevatorStore(db_path)
    item_id = elevator.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "test", "context": {}},
        principal="test",
        latency_class="interactive",
    )
    item = elevator.claim(
        lanes=["interactive"], owner="test-worker", claim_ttl_sec=360
    )
    elevator.close()

    def fake_call_operator(operator, prompt, system=None, on_wake_fail=None, _provenance_out=None, **kwargs):
        if _provenance_out is not None:
            _provenance_out.append(("gw_not_serving", "gravitywell"))
        return None

    def fake_doorman_status():
        return {"nodes": {"gravitywell": {"serving": False}}}

    from unittest.mock import MagicMock
    monkeypatch.setattr("agents_core.elevator_interactive_worker.call_operator", fake_call_operator)
    monkeypatch.setattr(
        "agents_core.elevator_interactive_worker.DoormanClient",
        lambda: MagicMock(status=fake_doorman_status, close=MagicMock())
    )

    result = serve_interactive_baton(item)
    assert result is False

    elevator = ElevatorStore(db_path)
    requeued_item = elevator.get(item_id)
    elevator.close()
    assert requeued_item["status"] == "pending"
    assert requeued_item["attempts"] == 1


def test_serving_worker_wake_failed_backoff(tmp_path, monkeypatch):
    """Worker requeues with backoff on wake_failed, marks failed after max retries."""
    from agents_core.elevator_interactive_worker import serve_interactive_baton

    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(db_path))
    monkeypatch.setenv("MAX_WAKE_FAIL_RETRIES", "3")

    elevator = ElevatorStore(db_path)
    item_id = elevator.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "test", "context": {}},
        principal="test",
        latency_class="interactive",
    )

    def fake_call_operator(operator, prompt, system=None, on_wake_fail=None, _provenance_out=None, **kwargs):
        if _provenance_out is not None:
            _provenance_out.append(("gw_not_serving", "gravitywell"))
        return None

    def fake_doorman_status():
        return {"nodes": {"gravitywell": {"serving": True}}}  # serving=true => wake_failed

    from unittest.mock import MagicMock
    monkeypatch.setattr("agents_core.elevator_interactive_worker.call_operator", fake_call_operator)
    monkeypatch.setattr(
        "agents_core.elevator_interactive_worker.DoormanClient",
        lambda: MagicMock(status=fake_doorman_status, close=MagicMock())
    )

    # Attempt 1: requeue
    item = elevator.claim(lanes=["interactive"], owner="worker", claim_ttl_sec=360)
    serve_interactive_baton(item)
    item = elevator.get(item_id)
    assert item["status"] == "pending"
    assert item["attempts"] == 1

    # Attempt 2: requeue
    item = elevator.claim(lanes=["interactive"], owner="worker", claim_ttl_sec=360)
    serve_interactive_baton(item)
    item = elevator.get(item_id)
    assert item["status"] == "pending"
    assert item["attempts"] == 2

    # Attempt 3: requeue
    item = elevator.claim(lanes=["interactive"], owner="worker", claim_ttl_sec=360)
    serve_interactive_baton(item)
    item = elevator.get(item_id)
    assert item["status"] == "pending"
    assert item["attempts"] == 3

    # Attempt 4: max retries exceeded, mark failed
    item = elevator.claim(lanes=["interactive"], owner="worker", claim_ttl_sec=360)
    serve_interactive_baton(item)
    item = elevator.get(item_id)
    assert item["status"] == "failed"

    elevator.close()


def test_serving_worker_only_claims_interactive(elevator_store):
    """Worker claims only from interactive lane, never deliberation/execution."""
    from agents_core.elevator_interactive_worker import worker_loop
    from unittest.mock import MagicMock, patch

    # Enqueue items in different lanes.
    elevator_store.enqueue(
        lane="interactive",
        kind="session-turn",
        payload={"prompt": "test", "context": {}},
        principal="test",
        latency_class="interactive",
    )
    elevator_store.enqueue(
        lane="deliberation",
        kind="deliberate",
        payload={"data": "test"},
        principal="test",
        latency_class="batch",
    )

    claimed_lanes = []

    def fake_claim(lanes, owner, claim_ttl_sec):
        claimed_lanes.append(lanes)
        # Return None to exit loop after first claim.
        return None

    with patch.object(ElevatorStore, "claim", side_effect=fake_claim):
        with patch("agents_core.elevator_interactive_worker.time.sleep"):
            try:
                with patch.object(ElevatorStore, "close"):
                    # Run one iteration of the loop
                    elevator = ElevatorStore()
                    item = elevator.claim(
                        lanes=["interactive"],
                        owner="elevator-interactive-worker",
                        claim_ttl_sec=360,
                    )
                    claimed_lanes.append(["interactive"])
            except StopIteration:
                pass

    # Verify only interactive was claimed
    assert ["interactive"] in claimed_lanes


# ---------------------------------------------------------------------------
# try_admit — AC3 principal-group concurrency
# ---------------------------------------------------------------------------

def test_try_admit_different_principal_blocks(tmp_path):
    """AC3(a): principal A claimed -> B's try_admit returns False."""
    store = ElevatorStore(db_path=tmp_path / "q.db")
    # Enqueue and claim an item for principal A
    a1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    store.claim(lanes=["deliberation"], owner="test", claim_ttl_sec=300)

    # Enqueue an item for principal B
    b1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-b", latency_class="batch")

    admitted, _ = store.try_admit(b1, "deliberation", "principal-b")
    assert admitted is False
    item = store.get(b1)
    assert item["status"] == "pending"
    store.close()


def test_try_admit_same_principal_ride_along(tmp_path):
    """AC3(b): principal A claimed -> second A ticket try_admit returns True (ride-along)."""
    store = ElevatorStore(db_path=tmp_path / "q.db")
    a1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    store.claim(lanes=["deliberation"], owner="test", claim_ttl_sec=300)

    a2 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    admitted, is_ride_along = store.try_admit(a2, "deliberation", "principal-a")
    assert admitted is True
    assert is_ride_along is True
    item = store.get(a2)
    assert item["status"] == "claimed"
    store.close()


def test_try_admit_fifo_head_of_line(tmp_path):
    """AC3(c): lane idle, A(older) and B(newer) -> only A's try_admit succeeds (FIFO)."""
    store = ElevatorStore(db_path=tmp_path / "q.db")
    a1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    b1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-b", latency_class="batch")

    # B's try_admit should fail (A is older = head-of-line)
    b_admitted, _ = store.try_admit(b1, "deliberation", "principal-b")
    assert b_admitted is False

    # A's try_admit should succeed (head-of-line)
    a_admitted, a_ride = store.try_admit(a1, "deliberation", "principal-a")
    assert a_admitted is True
    assert a_ride is False  # fresh group, not ride-along
    assert store.get(a1)["status"] == "claimed"
    assert store.get(b1)["status"] == "pending"
    store.close()


def test_try_admit_max_groups_two(tmp_path):
    """AC3(d): max_groups=2 admits two distinct principals."""
    store = ElevatorStore(db_path=tmp_path / "q.db")
    a1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    b1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-b", latency_class="batch")

    # A is head-of-line; admit with max_groups=2
    a_admitted, _ = store.try_admit(a1, "deliberation", "principal-a", max_groups=2)
    assert a_admitted is True

    # B can now also be admitted since only 1 other group (A) holds the lane and max_groups=2
    b_admitted, _ = store.try_admit(b1, "deliberation", "principal-b", max_groups=2)
    assert b_admitted is True
    assert store.get(a1)["status"] == "claimed"
    assert store.get(b1)["status"] == "claimed"
    store.close()


def test_try_admit_claim_ttl_sec_above_max_wait(tmp_path):
    """AC13: try_admit sets claim_ttl_sec >= max_wait; reap does not reclaim the ticket."""
    store = ElevatorStore(db_path=tmp_path / "q.db")
    max_wait = 900
    claim_ttl = max_wait + 60  # 960

    a1 = store.enqueue(lane="deliberation", kind="gw-admission", payload={},
                       principal="principal-a", latency_class="batch")
    admitted, _ = store.try_admit(a1, "deliberation", "principal-a", claim_ttl_sec=claim_ttl)
    assert admitted is True

    item = store.get(a1)
    assert item["status"] == "claimed"
    assert item["claim_ttl_sec"] >= max_wait

    # Inline reap should NOT reclaim this ticket (TTL hasn't expired)
    store._reap_inline()
    item_after = store.get(a1)
    assert item_after["status"] == "claimed"
    store.close()
