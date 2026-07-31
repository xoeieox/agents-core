"""Tests for elevator_interactive_worker — mode-peek, provenance classification, requeue.

Covers:
  AC1 — not-big-window peek skips claim (baton stays pending)
  AC2 — big-mode peek claims + serves
  AC3 — hardened fail: last-known mode + bounded backoff, NOT blind-claim
  AC4 — _classify_deferred_reason reads provenance first (no /status call)
  AC5 — other provenance reasons + legacy /status fallback
  AC6 — downstream requeue/backoff unchanged (deferred/wake_failed/success)
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from agents_core.doorman_client import DoormanUnreachable
from agents_core.elevator import ElevatorStore
from agents_core.elevator_interactive_worker import (
    _ModePeekState,
    _classify_deferred_reason,
    _peek_serve_big,
    serve_interactive_baton,
    worker_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_status(serving: bool, serving_is_big: bool | None = True, serving_mode: str = "big") -> dict:
    return {
        "nodes": {
            "gravitywell": {
                "serving": serving,
                "serving_is_big": serving_is_big,
                "serving_mode": serving_mode,
            }
        }
    }


def _make_client(serving: bool = True, serving_is_big: bool | None = True, serving_mode: str = "big"):
    client = mock.MagicMock()
    client.status.return_value = _make_status(serving, serving_is_big, serving_mode)
    return client


def _make_item(item_id: str = "item-1", attempts: int = 0) -> dict:
    return {
        "item_id": item_id,
        "payload": {"prompt": "hello", "context": None},
        "attempts": attempts,
    }


@pytest.fixture
def tmp_db():
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "test.db"
        store = ElevatorStore(db_path)
        yield db_path, store
        store.close()


# ---------------------------------------------------------------------------
# AC4 + AC5 — _classify_deferred_reason
# ---------------------------------------------------------------------------

def test_classify_deferred_from_provenance_no_status_call():
    """AC4: gw_deferred_swarm in provenance → 'deferred' without calling /status."""
    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient"
    ) as mock_cls:
        result = _classify_deferred_reason([("gw_deferred_swarm", "gravitywell")])
    assert result == "deferred"
    mock_cls.assert_not_called()


def test_classify_doorman_unreachable_from_provenance():
    """AC5: doorman_unreachable in provenance → 'doorman_unreachable'."""
    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient"
    ) as mock_cls:
        result = _classify_deferred_reason([("doorman_unreachable", "gravitywell")])
    assert result == "doorman_unreachable"
    mock_cls.assert_not_called()


def test_classify_wake_failed_from_provenance():
    """AC5: gw_not_serving in provenance → 'wake_failed'."""
    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient"
    ) as mock_cls:
        result = _classify_deferred_reason([("gw_not_serving", "gravitywell")])
    assert result == "wake_failed"
    mock_cls.assert_not_called()


def test_classify_provenance_first_takes_priority():
    """AC4: when gw_deferred_swarm is present alongside other entries, provenance wins."""
    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient"
    ) as mock_cls:
        result = _classify_deferred_reason([
            ("success", "gravitywell"),
            ("gw_deferred_swarm", "gravitywell"),
        ])
    assert result == "deferred"
    mock_cls.assert_not_called()


def test_classify_legacy_fallback_calls_status_when_no_known_reason():
    """AC5: empty/legacy provenance → falls back to /status re-probe."""
    fake_client = mock.MagicMock()
    fake_client.status.return_value = _make_status(serving=False)

    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient",
        return_value=fake_client,
    ):
        result = _classify_deferred_reason([])

    assert result == "deferred"
    fake_client.status.assert_called_once()


def test_classify_legacy_fallback_wake_failed_when_serving():
    """AC5: legacy fallback, doorman says serving=True → 'wake_failed'."""
    fake_client = mock.MagicMock()
    fake_client.status.return_value = _make_status(serving=True)

    with mock.patch(
        "agents_core.elevator_interactive_worker.DoormanClient",
        return_value=fake_client,
    ):
        result = _classify_deferred_reason([("unknown_reason", "gravitywell")])

    assert result == "wake_failed"


# ---------------------------------------------------------------------------
# AC6 — serve_interactive_baton downstream requeue/backoff unchanged
# ---------------------------------------------------------------------------

def test_serve_deferred_requeues_immediately(tmp_db):
    """AC6: deferred result → elevator.requeue called, no fail."""
    db_path, store = tmp_db
    item_id = store.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hi", "context": None},
        principal="s1", latency_class="interactive",
    )
    item = store.claim(lanes=["interactive"], owner="test", claim_ttl_sec=60)

    with mock.patch("agents_core.elevator_interactive_worker.call_operator", return_value=None), \
         mock.patch(
             "agents_core.elevator_interactive_worker._classify_deferred_reason",
             return_value="deferred"
         ), \
         mock.patch.dict("os.environ", {"ELEVATOR_DB_PATH": str(db_path)}):
        result = serve_interactive_baton(item)

    assert result is False
    row = store.get(item["item_id"])
    assert row["status"] == "pending"


def test_serve_success_acks(tmp_db):
    """AC6: success result → ack."""
    db_path, store = tmp_db
    store.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hi", "context": None},
        principal="s1", latency_class="interactive",
    )
    item = store.claim(lanes=["interactive"], owner="test", claim_ttl_sec=60)

    with mock.patch(
        "agents_core.elevator_interactive_worker.call_operator",
        return_value="response text",
    ), mock.patch.dict("os.environ", {"ELEVATOR_DB_PATH": str(db_path)}):
        result = serve_interactive_baton(item)

    assert result is True
    row = store.get(item["item_id"])
    assert row["status"] == "served"


def test_serve_wake_failed_requeues_with_backoff(tmp_db):
    """AC6: wake_failed with attempts < max → requeue."""
    db_path, store = tmp_db
    store.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hi", "context": None},
        principal="s1", latency_class="interactive",
    )
    item = store.claim(lanes=["interactive"], owner="test", claim_ttl_sec=60)
    item["attempts"] = 2

    with mock.patch("agents_core.elevator_interactive_worker.call_operator", return_value=None), \
         mock.patch(
             "agents_core.elevator_interactive_worker._classify_deferred_reason",
             return_value="wake_failed"
         ), \
         mock.patch.dict("os.environ", {
             "ELEVATOR_DB_PATH": str(db_path),
             "MAX_WAKE_FAIL_RETRIES": "5",
         }):
        result = serve_interactive_baton(item)

    assert result is False
    row = store.get(item["item_id"])
    assert row["status"] == "pending"


def test_serve_wake_failed_fails_after_max_retries(tmp_db):
    """AC6: wake_failed at max retries → fail."""
    db_path, store = tmp_db
    store.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hi", "context": None},
        principal="s1", latency_class="interactive",
    )
    item = store.claim(lanes=["interactive"], owner="test", claim_ttl_sec=60)
    item["attempts"] = 5  # == MAX_WAKE_FAIL_RETRIES default

    with mock.patch("agents_core.elevator_interactive_worker.call_operator", return_value=None), \
         mock.patch(
             "agents_core.elevator_interactive_worker._classify_deferred_reason",
             return_value="wake_failed"
         ), \
         mock.patch.dict("os.environ", {
             "ELEVATOR_DB_PATH": str(db_path),
             "MAX_WAKE_FAIL_RETRIES": "5",
         }):
        result = serve_interactive_baton(item)

    assert result is False
    row = store.get(item["item_id"])
    assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# _peek_serve_big
# ---------------------------------------------------------------------------

def test_peek_serve_big_true_when_big():
    client = _make_client(serving=True, serving_is_big=True, serving_mode="big")
    assert _peek_serve_big(client) is True


def test_peek_serve_big_false_when_dual():
    """A dual snapshot (serving_is_big=False) must read as not-big."""
    client = _make_client(serving=True, serving_is_big=False, serving_mode="dual")
    assert _peek_serve_big(client) is False


def test_peek_serve_big_false_when_not_serving():
    client = _make_client(serving=False, serving_is_big=True, serving_mode="big")
    assert _peek_serve_big(client) is False


def test_peek_serve_big_false_when_serving_is_big_none():
    """serving_is_big=None (pre-first-refresh or resolver failure) must read as
    not-serving, not "uncertain, retry" — `is True` is required, not truthiness."""
    client = _make_client(serving=True, serving_is_big=None, serving_mode="unknown")
    assert _peek_serve_big(client) is False


def test_peek_serve_big_decoupled_from_serving_mode_string():
    """The predicate reads serving_is_big, not serving_mode — a stale/wrong
    serving_mode="big" string must not flip it when serving_is_big says otherwise."""
    client = _make_client(serving=True, serving_is_big=False, serving_mode="big")
    assert _peek_serve_big(client) is False


# ---------------------------------------------------------------------------
# _ModePeekState
# ---------------------------------------------------------------------------

def test_peek_state_freshness():
    state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)
    assert not state.is_fresh()
    state.record_success(serve_big=True)
    assert state.is_fresh()
    assert state.last_known_serve_big() is True


def test_peek_state_backoff_caps():
    state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)
    assert state.next_backoff_sec() == 1  # 2^0 = 1
    state.record_failure()
    assert state.next_backoff_sec() == 2  # 2^1 = 2
    state.record_failure()
    assert state.next_backoff_sec() == 4
    state.record_failure()
    assert state.next_backoff_sec() == 8
    state.record_failure()
    assert state.next_backoff_sec() == 16  # capped
    state.record_failure()
    assert state.next_backoff_sec() == 16  # stays capped


def test_peek_state_past_cap():
    state = _ModePeekState(freshness_sec=60, backoff_cap_sec=4)
    assert not state.past_cap()
    state.record_failure()  # 2^0=1, not yet
    assert not state.past_cap()
    state.record_failure()  # 2^1=2, not yet
    assert not state.past_cap()
    state.record_failure()  # 2^2=4 >= cap=4
    assert state.past_cap()


def test_peek_state_resets_on_success():
    state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)
    state.record_failure()
    state.record_failure()
    state.record_success(serve_big=False)
    assert state._fail_count == 0
    assert state.is_fresh()
    assert state.last_known_serve_big() is False


# ---------------------------------------------------------------------------
# AC1 — dual-window peek skips claim
# ---------------------------------------------------------------------------

def test_worker_loop_skips_claim_when_dual(tmp_db):
    """AC1: peek returns dual (serving_is_big=False) → elevator.claim NOT called;
    baton stays pending."""
    db_path, store = tmp_db
    item_id = store.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hi", "context": None},
        principal="s1", latency_class="interactive",
    )

    # Dual client → will trigger skip
    dual_client = _make_client(serving=True, serving_is_big=False, serving_mode="dual")

    call_count = {"n": 0}

    def fake_claim(**kwargs):
        call_count["n"] += 1
        raise KeyboardInterrupt  # stop loop after one cycle

    fake_elevator = mock.MagicMock()
    fake_elevator.claim.side_effect = fake_claim

    peek_state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)

    with mock.patch("agents_core.elevator_interactive_worker.ElevatorStore", return_value=fake_elevator), \
         mock.patch("agents_core.elevator_interactive_worker.time") as mock_time:
        # Make time.sleep raise KeyboardInterrupt to stop the loop after one idle sleep
        mock_time.sleep.side_effect = KeyboardInterrupt
        mock_time.monotonic = time.monotonic

        try:
            worker_loop(_peek_state=peek_state, _doorman_client=dual_client)
        except KeyboardInterrupt:
            pass  # expected exit

    # claim must not have been called
    fake_elevator.claim.assert_not_called()
    # the real store still has the baton as pending
    row = store.get(item_id)
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# AC2 — big-mode peek claims and serves
# ---------------------------------------------------------------------------

def test_worker_loop_claims_when_big(tmp_db):
    """AC2: peek returns big → claim + serve_interactive_baton invoked."""
    db_path, store = tmp_db

    big_client = _make_client(serving=True, serving_mode="big")

    served_items = []

    def fake_serve(item):
        served_items.append(item)
        raise KeyboardInterrupt  # stop after one serve

    fake_item = {"item_id": "item-99", "payload": {"prompt": "hi"}, "attempts": 0}
    fake_elevator = mock.MagicMock()
    fake_elevator.claim.return_value = fake_item

    peek_state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)

    with mock.patch("agents_core.elevator_interactive_worker.ElevatorStore", return_value=fake_elevator), \
         mock.patch(
             "agents_core.elevator_interactive_worker.serve_interactive_baton",
             side_effect=fake_serve,
         ), \
         mock.patch("agents_core.elevator_interactive_worker.time") as mock_time:
        mock_time.monotonic = time.monotonic
        try:
            worker_loop(_peek_state=peek_state, _doorman_client=big_client)
        except KeyboardInterrupt:
            pass

    fake_elevator.claim.assert_called_once()
    assert len(served_items) == 1
    assert served_items[0]["item_id"] == "item-99"


# ---------------------------------------------------------------------------
# AC3 — hardened fail: last-known mode + bounded backoff, NOT blind-claim
# ---------------------------------------------------------------------------

def test_worker_loop_uses_last_known_swarm_on_peek_fail():
    """AC3a: peek fails, last_known=False (swarm) → claim NOT called."""
    fail_client = mock.MagicMock()
    fail_client.status.side_effect = DoormanUnreachable("down")

    fake_elevator = mock.MagicMock()
    fake_elevator.claim.side_effect = KeyboardInterrupt

    peek_state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)
    # Pre-seed fresh last-known as swarm
    peek_state.record_success(serve_big=False)

    with mock.patch("agents_core.elevator_interactive_worker.ElevatorStore", return_value=fake_elevator), \
         mock.patch("agents_core.elevator_interactive_worker.time") as mock_time:
        mock_time.sleep.side_effect = KeyboardInterrupt
        mock_time.monotonic = time.monotonic

        try:
            worker_loop(_peek_state=peek_state, _doorman_client=fail_client)
        except KeyboardInterrupt:
            pass

    fake_elevator.claim.assert_not_called()


def test_worker_loop_uses_last_known_big_on_peek_fail():
    """AC3b: peek fails, last_known=True (big) → claim IS called."""
    fail_client = mock.MagicMock()
    fail_client.status.side_effect = DoormanUnreachable("down")

    fake_item = {"item_id": "item-x", "payload": {"prompt": "hi"}, "attempts": 0}
    fake_elevator = mock.MagicMock()
    fake_elevator.claim.return_value = fake_item

    peek_state = _ModePeekState(freshness_sec=60, backoff_cap_sec=16)
    peek_state.record_success(serve_big=True)  # fresh + big

    served = []

    def fake_serve(item):
        served.append(item)
        raise KeyboardInterrupt

    with mock.patch("agents_core.elevator_interactive_worker.ElevatorStore", return_value=fake_elevator), \
         mock.patch(
             "agents_core.elevator_interactive_worker.serve_interactive_baton",
             side_effect=fake_serve,
         ), \
         mock.patch("agents_core.elevator_interactive_worker.time") as mock_time:
        mock_time.monotonic = time.monotonic
        try:
            worker_loop(_peek_state=peek_state, _doorman_client=fail_client)
        except KeyboardInterrupt:
            pass

    fake_elevator.claim.assert_called_once()
    assert len(served) == 1


def test_worker_loop_cold_start_peek_fail_bounded_backoff():
    """AC3c: no last-known + repeated peek failures → growing backoff, no tight loop."""
    fail_client = mock.MagicMock()
    fail_client.status.side_effect = DoormanUnreachable("down")

    fake_elevator = mock.MagicMock()

    peek_state = _ModePeekState(freshness_sec=60, backoff_cap_sec=4)
    # No prior record — cold start

    sleep_calls = []
    loop_count = {"n": 0}

    def fake_sleep(secs):
        sleep_calls.append(secs)
        loop_count["n"] += 1
        if loop_count["n"] >= 6:
            raise KeyboardInterrupt

    with mock.patch("agents_core.elevator_interactive_worker.ElevatorStore", return_value=fake_elevator), \
         mock.patch(
             "agents_core.elevator_interactive_worker.serve_interactive_baton",
             side_effect=lambda item: False,
         ), \
         mock.patch("agents_core.elevator_interactive_worker.time") as mock_time:
        mock_time.sleep.side_effect = fake_sleep
        mock_time.monotonic = time.monotonic

        try:
            worker_loop(_peek_state=peek_state, _doorman_client=fail_client)
        except KeyboardInterrupt:
            pass

    # Backoffs should be growing (1, 2, 4, 4, 4...) — never 0
    assert all(s > 0 for s in sleep_calls), f"Found zero sleep: {sleep_calls}"
    # First sleep is 1s (2^0), second is 2s (2^1), then cap at 4s
    assert sleep_calls[0] == 1
    assert sleep_calls[1] == 2
    assert sleep_calls[2] == 4
    # After the cap is reached, claim MUST be attempted — this is the anti-starvation
    # guarantee. The previous assertion (call_count <= n) was trivially true and gave
    # a false green; >= 1 actually verifies the property.
    assert fake_elevator.claim.call_count >= 1
