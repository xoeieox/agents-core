"""Tests for call_operator() — qwen/gravitywell model guards and operator routing."""
import warnings
import pytest
from unittest.mock import patch, MagicMock, call

from agents_core.llm import call_operator, OPERATOR_DEFAULTS, OperatorUnreachableError


# ---------------------------------------------------------------------------
# qwen: default model (None or explicit default) must reach the backend
# ---------------------------------------------------------------------------

def test_call_operator_qwen_default_model_works():
    """model=None and model=<default> both reach _call_qwen_backend without error."""
    with patch("agents_core.llm._call_qwen_backend", return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", model=None)
        assert result == "ok"
        mock_qwen.assert_called_once_with(prompt="hi")

    with patch("agents_core.llm._call_qwen_backend", return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", model=OPERATOR_DEFAULTS["qwen"])
        assert result == "ok"
        mock_qwen.assert_called_once_with(prompt="hi")


# ---------------------------------------------------------------------------
# qwen: non-default model must raise ValueError with the directive message
# ---------------------------------------------------------------------------

def test_call_operator_qwen_non_default_model_raises():
    """Passing a non-default model to the qwen operator raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        call_operator("qwen", prompt="hi", model="qwen-other-7b")
    assert "infrastructure operation" in str(exc_info.value)
    assert "qwen-other-7b" in str(exc_info.value)
    assert OPERATOR_DEFAULTS["qwen"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# sonnet: non-default model must NOT be rejected — passes through to ClaudeQueue
# ---------------------------------------------------------------------------

def test_call_operator_sonnet_model_override_passes_through():
    """A non-default model on sonnet is not rejected; it propagates into the task dict."""
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="ok") as mock_saw:
        result = call_operator("sonnet", prompt="hi", model="claude-sonnet-4-5")

    assert result == "ok"
    mock_saw.assert_called_once()
    task_dict = mock_saw.call_args[0][0]
    assert task_dict["model"] == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# gravitywell: in OPERATOR_DEFAULTS
# ---------------------------------------------------------------------------

def test_gravitywell_in_operator_defaults():
    assert "gravitywell" in OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["gravitywell"] == "gravitywell-122b"


# ---------------------------------------------------------------------------
# gravitywell: non-default model must raise ValueError
# ---------------------------------------------------------------------------

def test_call_operator_gravitywell_non_default_model_raises():
    """Passing a non-default model to gravitywell raises ValueError."""
    mock_client = MagicMock()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_client):
        with pytest.raises(ValueError) as exc_info:
            call_operator("gravitywell", prompt="hi", model="gw-other")
    assert "infrastructure operation" in str(exc_info.value)
    assert "gw-other" in str(exc_info.value)


# ---------------------------------------------------------------------------
# gravitywell: default payload has enable_thinking=False
# ---------------------------------------------------------------------------

def _gw_mock_client(status="serving"):
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    mock.acquire.return_value = {"status": status, "node": "gravitywell", "work_id": "w1"}
    return mock


def _gw_dc(status="serving"):
    """Return (DoormanClient class mock, instance mock).

    Use as: dc, mock_client = _gw_dc(status); patch("...DoormanClient", dc)

    The class mock has is_deferred wired to the real static logic so that
    patch("...DoormanClient", dc) doesn't make every status look deferred.
    """
    instance = _gw_mock_client(status)
    dc = MagicMock(return_value=instance)
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    return dc, instance


def test_gravitywell_default_think_off():
    """Default call injects enable_thinking=False into the payload."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}', "reasoning_content": None}}]
        }
        return resp

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell", prompt="test", json_mode=True)

    assert result is not None
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False


def test_gravitywell_think_true():
    """think=True injects enable_thinking=True."""
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "answer", "reasoning_content": "trace"}}]
        }
        return resp

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell", prompt="test", think=True)

    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is True


# ---------------------------------------------------------------------------
# gravitywell: bundle_ids passed through call_operator does NOT raise TypeError
# ---------------------------------------------------------------------------

def test_gravitywell_bundle_ids_discarded_no_typeerror():
    """bundle_ids kwarg is discarded before reaching _call_gravitywell_backend."""
    dc, _mock_client = _gw_dc()

    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok", "reasoning_content": None}}]
        }
        return resp

    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=fake_post):
        # Must not raise TypeError
        result = call_operator("gravitywell", prompt="test", bundle_ids=["some/bundle"])
    assert result == "ok"


# ---------------------------------------------------------------------------
# gravitywell: on_wake_fail policies
# ---------------------------------------------------------------------------

def test_gravitywell_wake_failed_skip_returns_none():
    dc, _mock_client = _gw_dc(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", dc):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None


def test_gravitywell_wake_failed_error_raises():
    dc, _mock_client = _gw_dc(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", dc):
        with pytest.raises(OperatorUnreachableError):
            call_operator("gravitywell", prompt="test", on_wake_fail="error")


def test_gravitywell_wake_failed_haiku_logs_and_falls_back():
    """on_wake_fail='haiku' emits a loud warning then falls back to haiku operator."""
    dc, _mock_client = _gw_dc(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="haiku-reply") as mock_saw, \
         warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = call_operator("gravitywell", prompt="test", on_wake_fail="haiku")

    assert result == "haiku-reply"
    assert len(w) >= 1
    assert any("haiku" in str(warning.message).lower() or "paid" in str(warning.message).lower()
               for warning in w)


def test_gravitywell_doorman_unreachable_applies_wake_fail():
    """DoormanUnreachable is treated like wake_failed."""
    from agents_core.doorman_client import DoormanUnreachable
    instance = MagicMock()
    instance.acquire.side_effect = DoormanUnreachable("can't reach doorman")
    instance.__enter__ = MagicMock(return_value=instance)
    instance.__exit__ = MagicMock(return_value=False)
    dc = MagicMock(return_value=instance)
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    with patch("agents_core.doorman_client.DoormanClient", dc):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None


def test_gravitywell_haiku_fallback_raises_propagates():
    """If the fallback call_operator raises, it propagates (not suppressed to None)."""
    dc, _mock_client = _gw_dc(status="wake_failed")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.claude_queue_sync.submit_and_wait",
               side_effect=RuntimeError("ClaudeQueue down")):
        with pytest.raises(RuntimeError, match="ClaudeQueue down"):
            call_operator("gravitywell", prompt="test", on_wake_fail="haiku")


# ---------------------------------------------------------------------------
# gravitywell: release is called even when backend raises
# ---------------------------------------------------------------------------

def test_gravitywell_release_called_on_backend_error():
    """client.release is called in the finally block even when the backend raises."""
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=Exception("backend exploded")):
        result = call_operator("gravitywell", prompt="test")
    assert mock_client.release.call_count == 1
    release_args = mock_client.release.call_args[0]
    assert release_args[0] == "gravitywell"


# ---------------------------------------------------------------------------
# gravitywell: Edit 4 — OperatorUnreachableError caught and routed to wake_fail
# ---------------------------------------------------------------------------

def test_gravitywell_operator_unreachable_fallback_skip():
    """When GW is serving then HTTP exhausts retries (OperatorUnreachableError),
    on_wake_fail='skip' returns None (no crash)."""
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw", Exception("retries exhausted"))):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")

    assert result is None
    # Verify the finally-block release was called
    assert mock_client.release.call_count == 1


def test_gravitywell_operator_unreachable_fallback_sonnet():
    """When GW is serving then HTTP exhausts retries (OperatorUnreachableError),
    on_wake_fail='sonnet' falls back to paid Sonnet (never raises)."""
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw", Exception("retries exhausted"))), \
         patch("agents_core.claude_queue_sync.submit_and_wait",
               return_value="sonnet-fallback") as mock_saw, \
         warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = call_operator("gravitywell", prompt="test", on_wake_fail="sonnet")

    assert result == "sonnet-fallback"
    # Verify fallback was invoked
    assert mock_saw.call_count >= 1
    # Verify loud warning was emitted
    assert any("sonnet" in str(warning.message).lower() or "paid" in str(warning.message).lower()
               for warning in w)
    # Verify release was called
    assert mock_client.release.call_count == 1


# ---------------------------------------------------------------------------
# gravitywell: deferred (mode-miss) — AC1/AC2/AC3 from gw-waist-deferred-provenance-v0
# ---------------------------------------------------------------------------

def test_gravitywell_deferred_appends_gw_deferred_swarm():
    """AC1/AC2: status='deferred' appends gw_deferred_swarm, not gw_not_serving."""
    dc, _mock_client = _gw_dc(status="deferred")
    prov = []
    with patch("agents_core.doorman_client.DoormanClient", dc):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip",
                               _provenance_out=prov)
    assert result is None
    assert ("gw_deferred_swarm", "gravitywell") in prov
    assert all(r != "gw_not_serving" for r, _ in prov)


def test_gravitywell_deferred_never_mislabeled():
    """AC2: deferred result never produces a gw_not_serving tuple."""
    dc, _mock_client = _gw_dc(status="deferred")
    prov = []
    with patch("agents_core.doorman_client.DoormanClient", dc):
        call_operator("gravitywell", prompt="test", on_wake_fail="skip", _provenance_out=prov)
    assert not any(r == "gw_not_serving" for r, _ in prov)


def test_gravitywell_deferred_skip_returns_none():
    """AC3: deferred + on_wake_fail='skip' returns None (identical behavior to today)."""
    dc, _mock_client = _gw_dc(status="deferred")
    with patch("agents_core.doorman_client.DoormanClient", dc):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None


def test_gravitywell_deferred_fallback_runs():
    """AC3: deferred + on_wake_fail='haiku' executes the paid fallback."""
    dc, _mock_client = _gw_dc(status="deferred")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="haiku-reply") as mock_saw, \
         warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = call_operator("gravitywell", prompt="test", on_wake_fail="haiku")
    assert result == "haiku-reply"
    assert mock_saw.call_count >= 1


def test_gravitywell_wake_failed_appends_gw_not_serving():
    """AC4: explicit wake_failed status still yields gw_not_serving (regression guard)."""
    dc, _mock_client = _gw_dc(status="wake_failed")
    prov = []
    with patch("agents_core.doorman_client.DoormanClient", dc):
        call_operator("gravitywell", prompt="test", on_wake_fail="skip", _provenance_out=prov)
    assert ("gw_not_serving", "gravitywell") in prov
    assert not any(r == "gw_deferred_swarm" for r, _ in prov)


# ---------------------------------------------------------------------------
# AC1: off mode (and unset) — byte-identical to existing direct path
# ---------------------------------------------------------------------------

def test_ac1_off_mode_no_enqueue(monkeypatch):
    """AC1: GW_ADMISSION_MODE=off takes the direct path (no elevator interaction)."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "off")
    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = 0
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok") as mock_be, \
         patch("agents_core.elevator.ElevatorStore") as mock_es:
        result = call_operator("gravitywell", prompt="test")
    assert result == "ok"
    mock_es.assert_not_called()  # No elevator interaction.


def test_ac1_unset_mode_no_enqueue(monkeypatch):
    """AC1: unset GW_ADMISSION_MODE defaults to off; no elevator interaction."""
    monkeypatch.delenv("GW_ADMISSION_MODE", raising=False)
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"), \
         patch("agents_core.elevator.ElevatorStore") as mock_es:
        result = call_operator("gravitywell", prompt="test")
    assert result == "ok"
    mock_es.assert_not_called()


# ---------------------------------------------------------------------------
# AC2: enforce — producer enqueues, serves, acks
# ---------------------------------------------------------------------------

def test_ac2_enforce_enqueues_and_serves(tmp_path, monkeypatch):
    """AC2: enforce mode enqueues a deliberation ticket, serves, acks it."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = 0

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="result"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="gate-1")

    assert result == "result"
    assert ("success", "gravitywell") in prov

    from agents_core.elevator import ElevatorStore
    store = ElevatorStore(tmp_path / "q.db")
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM queue_items WHERE lane='deliberation' AND kind='gw-admission'"
        ).fetchall()
    store.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "served"


# ---------------------------------------------------------------------------
# AC4: drain_count gates fresh group; ride-along skips drain_count
# ---------------------------------------------------------------------------

def test_ac4_fresh_group_waits_for_drain_count(tmp_path, monkeypatch):
    """AC4: fresh principal-group holds until drain_count==0; dispatches once it clears."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    dc = MagicMock()
    instance = MagicMock()
    # drain_count returns 1 on first call, then 0 (clears)
    instance.drain_count.side_effect = [1, 0]
    instance.acquire.return_value = {"status": "serving"}
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="fresh-gate")

    assert result == "ok"
    assert ("success", "gravitywell") in prov
    # drain_count must have been polled at least twice (1 then 0)
    assert instance.drain_count.call_count >= 2


def test_ac4_ride_along_skips_drain_count(tmp_path, monkeypatch):
    """AC4: same-principal ride-along dispatches immediately, drain_count not checked."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    # Pre-plant a claimed ticket for the same principal so the second call is a ride-along.
    from agents_core.elevator import ElevatorStore
    store = ElevatorStore(tmp_path / "q.db")
    anchor = store.enqueue(
        lane="deliberation", kind="gw-admission", payload={},
        principal="shared-gate", latency_class="batch",
    )
    store.claim(lanes=["deliberation"], owner="anchor", claim_ttl_sec=999)
    store.close()

    dc = MagicMock()
    instance = MagicMock()
    # drain_count returns non-zero; ride-along must NOT block on it
    instance.drain_count.return_value = 5
    instance.acquire.return_value = {"status": "serving"}
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="shared-gate")

    assert result == "ok"
    assert ("success", "gravitywell") in prov
    # drain_count must NOT have been called (ride-along skips the gate)
    instance.drain_count.assert_not_called()


# ---------------------------------------------------------------------------
# AC5: bypass — no enqueue even under enforce
# ---------------------------------------------------------------------------

def test_ac5_bypass_skips_enqueue(tmp_path, monkeypatch):
    """AC5(a): _admission_bypass=True enqueues zero tickets even under enforce."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))

    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = 0

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        result = call_operator("gravitywell", prompt="test", _admission_bypass=True)

    assert result == "ok"

    from agents_core.elevator import ElevatorStore
    store = ElevatorStore(tmp_path / "q.db")
    with store._lock:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM queue_items WHERE lane='deliberation'"
        ).fetchone()[0]
    store.close()
    assert count == 0


def test_ac5_interactive_worker_bypass_no_deadlock(tmp_path, monkeypatch):
    """AC5(b): interactive worker's call_operator passes _admission_bypass; no deadlock."""
    from agents_core.elevator_interactive_worker import serve_interactive_baton

    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))

    from agents_core.elevator import ElevatorStore
    elevator = ElevatorStore(tmp_path / "q.db")
    item_id = elevator.enqueue(
        lane="interactive", kind="session-turn",
        payload={"prompt": "hello", "context": {}},
        principal="sess-1", latency_class="interactive",
    )
    item = elevator.claim(lanes=["interactive"], owner="worker", claim_ttl_sec=360)
    elevator.close()

    captured_bypass = []

    def fake_call_operator(op, prompt, _admission_bypass=False, **kwargs):
        captured_bypass.append(_admission_bypass)
        prov = kwargs.get("_provenance_out")
        if prov is not None:
            prov.append(("success", "gravitywell"))
        return "pong"

    monkeypatch.setattr("agents_core.elevator_interactive_worker.call_operator", fake_call_operator)
    result = serve_interactive_baton(item)
    assert result is True
    assert True in captured_bypass  # bypass was passed


# ---------------------------------------------------------------------------
# AC6: deferred -> requeue, then serve
# ---------------------------------------------------------------------------

def test_ac6_deferred_requeues_then_serves(tmp_path, monkeypatch):
    """AC6: acquire returns deferred then serving; ticket requeues once, final prov has both."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    acquire_responses = [
        {"status": "deferred"},
        {"status": "serving"},
    ]

    dc = MagicMock()
    instance = MagicMock()
    instance.acquire.side_effect = acquire_responses
    instance.drain_count.return_value = 0
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"

    apply_wake_fail_calls = []

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="done"), \
         patch("agents_core.llm._apply_wake_fail", side_effect=lambda *a, **kw: apply_wake_fail_calls.append(1)):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="gate-deferred")

    assert result == "done"
    assert apply_wake_fail_calls == []  # Never called
    reasons = [r for r, _ in prov]
    assert "gw_deferred_swarm" in reasons
    assert reasons[-1] == "success"


# ---------------------------------------------------------------------------
# AC7: overflow — bounded timeout degrades loud
# ---------------------------------------------------------------------------

def test_ac7_slot_queued_timeout(tmp_path, monkeypatch):
    """AC7: timeout -> slot_queued_timeout provenance, ticket failed, degrade returns None."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_MAX_WAIT_SEC", "0")  # instant timeout
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.001")

    # Plant a competing claim so try_admit always fails
    from agents_core.elevator import ElevatorStore
    blocker = ElevatorStore(tmp_path / "q.db")
    blocker.enqueue(lane="deliberation", kind="gw-admission", payload={},
                    principal="blocker", latency_class="batch")
    blocker.claim(lanes=["deliberation"], owner="other", claim_ttl_sec=999)
    blocker.close()

    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = 0

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                on_wake_fail="skip", principal="waiter")

    assert result is None  # on_wake_fail="skip"
    assert any(r == "slot_queued_timeout" for r, _ in prov)

    store = ElevatorStore(tmp_path / "q.db")
    with store._lock:
        rows = store._conn.execute(
            "SELECT status FROM queue_items WHERE principal='waiter'"
        ).fetchall()
    store.close()
    assert any(row["status"] == "failed" for row in rows)


# ---------------------------------------------------------------------------
# AC8: wake_failed bounded backoff
# ---------------------------------------------------------------------------

def test_ac8_wake_failed_bounded_backoff(tmp_path, monkeypatch):
    """AC8: repeated wake_failed -> bounded retries then _apply_wake_fail with gw_not_serving."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.001")
    monkeypatch.setenv("MAX_WAKE_FAIL_RETRIES", "2")

    dc = MagicMock()
    instance = MagicMock()
    instance.acquire.return_value = {"status": "wake_failed"}
    instance.drain_count.return_value = 0
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm.time.sleep"):  # skip actual sleep
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                on_wake_fail="skip", principal="wf-test")

    assert result is None
    assert any(r == "gw_not_serving" for r, _ in prov)


# ---------------------------------------------------------------------------
# AC9: off-master guard — passthrough, no enqueue
# ---------------------------------------------------------------------------

def test_ac9_off_master_passthrough(monkeypatch):
    """AC9: enforce + not IS_MASTER -> passthrough provenance, direct dispatch, no enqueue."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")

    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = 0

    with patch("agents_core.elevator.IS_MASTER", False), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="p1")

    assert result == "ok"
    assert ("admission_off_master_passthrough", "gravitywell") in prov


# ---------------------------------------------------------------------------
# AC10: shadow mode
# ---------------------------------------------------------------------------

def test_ac10_shadow_no_enqueue_direct_dispatch(tmp_path, monkeypatch):
    """AC10: shadow dispatches directly, zero tickets enqueued."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "shadow")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))

    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="my-gate")

    assert result == "ok"
    assert any("admission_shadow:" in r for r, _ in prov)
    assert not any("principal_group_collision_risk" in r for r, _ in prov)

    from agents_core.elevator import ElevatorStore
    store = ElevatorStore(tmp_path / "q.db")
    with store._lock:
        count = store._conn.execute(
            "SELECT COUNT(*) FROM queue_items WHERE lane='deliberation'"
        ).fetchone()[0]
    store.close()
    assert count == 0


def test_ac10_shadow_collision_risk_on_unique_work_id(tmp_path, monkeypatch):
    """AC10: unique-work_id principal (principal=None) emits collision_risk in shadow."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "shadow")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))

    dc, _ = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        # No principal= passed -> unique-per-call work_id
        call_operator("gravitywell", prompt="test", _provenance_out=prov)

    assert any("principal_group_collision_risk" in r for r, _ in prov)


def test_ac10_shadow_no_collision_risk_on_shared_principal(tmp_path, monkeypatch):
    """AC10: shared (non-work_id) principal does NOT emit collision_risk."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "shadow")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))

    dc, _ = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        call_operator("gravitywell", prompt="test", _provenance_out=prov,
                       principal="shared-gate-abc123")

    assert not any("principal_group_collision_risk" in r for r, _ in prov)


# ---------------------------------------------------------------------------
# AC11: provenance ladder
# ---------------------------------------------------------------------------

def test_ac11_provenance_precedence():
    """AC11: gw_highest_precedence_reason returns highest-precedence reason."""
    from agents_core.llm import gw_highest_precedence_reason, GW_PROVENANCE_PRECEDENCE

    assert gw_highest_precedence_reason([
        ("gw_not_serving", "gravitywell"),
        ("gw_deferred_swarm", "gravitywell"),
    ]) == "gw_deferred_swarm"

    assert gw_highest_precedence_reason([
        ("slot_queued_timeout", "gravitywell"),
        ("gw_not_serving", "gravitywell"),
    ]) == "slot_queued_timeout"

    assert gw_highest_precedence_reason([
        ("gw_not_serving", "gravitywell"),
    ]) == "gw_not_serving"

    assert gw_highest_precedence_reason([]) is None

    # GW_PROVENANCE_PRECEDENCE is exposed for downstream consumers.
    assert "gw_deferred_swarm" in GW_PROVENANCE_PRECEDENCE
    assert GW_PROVENANCE_PRECEDENCE.index("gw_deferred_swarm") < \
           GW_PROVENANCE_PRECEDENCE.index("gw_not_serving")


# ---------------------------------------------------------------------------
# AC12: drain_count unavailable -> proceed loud, no deadlock
# ---------------------------------------------------------------------------

def test_ac12_drain_count_unavailable_proceeds(tmp_path, monkeypatch):
    """AC12: drain_count=None -> proceed loud with elevator gate alone."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    dc, mock_client = _gw_dc(status="serving")
    mock_client.drain_count.return_value = None  # unavailable

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="gate-dc-unavail")

    assert result == "ok"
    assert ("success", "gravitywell") in prov
    assert any(r == "drain_count_unavailable" for r, _ in prov)


# ---------------------------------------------------------------------------
# AC14: no swarm/qwen/anthropic regression under enforce
# ---------------------------------------------------------------------------

def test_ac14_call_swarm_untouched_under_enforce(monkeypatch):
    """AC14: call_swarm is untouched by GW_ADMISSION_MODE=enforce."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")

    from agents_core.llm import call_swarm

    def fake_post(base_url, model, messages, **kw):
        return None  # swarm_model returns None -> call_swarm returns [None]

    with patch("agents_core.llm.swarm_model", return_value=None):
        result = call_swarm(["hello"])
    assert result == [None]  # graceful None, no crash


def test_ac14_qwen_untouched_under_enforce(monkeypatch):
    """AC14: qwen call_operator is untouched by GW_ADMISSION_MODE=enforce."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    with patch("agents_core.llm._call_qwen_backend", return_value="qok") as mock_q:
        result = call_operator("qwen", prompt="hi")
    assert result == "qok"
    mock_q.assert_called_once()
