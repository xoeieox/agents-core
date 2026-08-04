"""Tests for call_operator() — qwen/gravitywell model guards and operator routing."""
import warnings
import pytest
from unittest.mock import patch, MagicMock, call

from agents_core.llm import (
    call_operator,
    OPERATOR_DEFAULTS,
    OperatorUnreachableError,
    CreativeOperatorUnavailable,
    _forward_supported_kwargs,
)


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
# qwen: the old single-fixed-model guard is gone (agents-core-local-llm-gw-repoint-v0).
# The backend is GravityWell/vLLM now, not the retired llama.cpp box, and
# OPERATOR_DEFAULTS["qwen"] is a name GW does not serve - asserting a caller's
# model= against it would just be asserting against a dead constant. A
# non-default model= no longer raises; it's simply not forwarded (the qwen
# backend never sends a `model` field at all - AC4).
# ---------------------------------------------------------------------------

def test_call_operator_qwen_non_default_model_no_longer_raises():
    """Passing a non-default model to the qwen operator no longer raises; it's ignored."""
    with patch("agents_core.llm._call_qwen_backend", return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", model="qwen-other-7b")
    assert result == "ok"
    # model= is not a _call_qwen_backend parameter, so it's dropped, not forwarded.
    mock_qwen.assert_called_once_with(prompt="hi")


# ---------------------------------------------------------------------------
# qwen: on_wake_fail (and other kwargs _call_qwen_backend doesn't accept) must be
# filtered at the dispatcher, not raise TypeError (agents-core-geist-disposition-harness-v0,
# Blocker A2: corpus_reader.read_operator="qwen" was broken by exactly this).
# ---------------------------------------------------------------------------

def test_call_operator_qwen_on_wake_fail_no_longer_raises_typeerror():
    """on_wake_fail is a call_operator-level kwarg that _call_qwen_backend never accepted;
    the dispatcher must drop it before forwarding, not let it raise TypeError.

    autospec=True so the mock's introspected signature is _call_qwen_backend's real one
    (a plain non-autospec Mock advertises a generic (*args, **kwargs) signature, which
    would make _forward_supported_kwargs treat it as accepting everything)."""
    with patch("agents_core.llm._call_qwen_backend", autospec=True, return_value="ok") as mock_qwen:
        result = call_operator("qwen", prompt="hi", on_wake_fail="skip")
        assert result == "ok"
        # on_wake_fail must not reach the backend call.
        mock_qwen.assert_called_once_with(prompt="hi")


def test_call_operator_qwen_on_wake_fail_reaches_real_backend_signature():
    """Without any patching of _call_qwen_backend itself, the dispatcher-filtered call must
    not raise — proves the fix is at call_operator, not a backend-side kwarg swallow."""
    with patch("agents_core.llm.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        # system="" avoids the chub_broker default-bundle import path (unrelated to this fix).
        result = call_operator("qwen", prompt="hi", system="s", on_wake_fail="skip", timeout=5)
    assert result == "ok"


def test_forward_supported_kwargs_drops_unsupported_and_keeps_supported():
    def backend(prompt, system=None, timeout=600):
        return prompt

    forwarded = _forward_supported_kwargs(
        backend, {"system": "sys", "on_wake_fail": "skip", "think": False}
    )
    assert forwarded == {"system": "sys"}


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
    resp = {"status": status, "node": "gravitywell", "work_id": "w1"}
    if status == "serving":
        resp["drain_cleared"] = True  # new atomic-acquire field; absence triggers AC5a warning
    mock.acquire.return_value = resp
    return mock


def _gw_dc(status="serving"):
    """Return (DoormanClient class mock, instance mock).

    Use as: dc, mock_client = _gw_dc(status); patch("...DoormanClient", dc)

    The class mock has is_deferred and is_contended wired to the real static logic so that
    patch("...DoormanClient", dc) doesn't make every status look deferred or contended.
    """
    instance = _gw_mock_client(status)
    dc = MagicMock(return_value=instance)
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))
    return dc, instance


def _make_gw_sse_resp(content_text, captured_dict=None):
    """Build a streaming SSE mock response for GW tests.

    Returns a requests.Response mock that:
    - Accepts payload via json= kwarg and optionally records it in captured_dict["payload"]
    - Yields one SSE data line with content then a [DONE] line via iter_lines()
    """
    import json as _json

    def fake_post(url, json=None, timeout=None, stream=None):
        if captured_dict is not None:
            captured_dict["payload"] = json

        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': content_text}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    return fake_post


def test_gravitywell_default_think_off(monkeypatch):
    """Default call injects enable_thinking=False into the payload.

    Pins GW_BACKEND=llamacpp explicitly (DoD: non-AC gravitywell tests that exercise the
    real _call_gravitywell_backend must not rely on ambient env / conftest precache for
    hermeticity - an ambient GW_BACKEND=vllm shell would otherwise resolve gravitywell-27b
    here and miss the conftest's 122b-precached handshake cache, firing a live /v1/models
    probe).
    """
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    captured = {}

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp('{"ok": true}', captured)):
        result = call_operator("gravitywell", prompt="test", json_mode=True)

    assert result is not None
    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is False


def test_gravitywell_think_true(monkeypatch):
    """think=True injects enable_thinking=True.

    Pins GW_BACKEND=llamacpp for the same hermeticity reason as
    test_gravitywell_default_think_off.
    """
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    captured = {}

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("answer", captured)):
        result = call_operator("gravitywell", prompt="test", think=True)

    assert captured["payload"]["chat_template_kwargs"]["enable_thinking"] is True


# ---------------------------------------------------------------------------
# gravitywell: bundle_ids passed through call_operator does NOT raise TypeError
# ---------------------------------------------------------------------------

def test_gravitywell_bundle_ids_discarded_no_typeerror(monkeypatch):
    """bundle_ids kwarg is discarded before reaching _call_gravitywell_backend.

    Pins GW_BACKEND=llamacpp for the same hermeticity reason as
    test_gravitywell_default_think_off.
    """
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    dc, _mock_client = _gw_dc()

    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
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
    """client.release is called in the finally block even when the backend raises.

    With streaming, a generic Exception from requests.post propagates through
    _gw_stream_attempt up to call_operator's except-Exception handler (gw_member_error),
    which re-raises. The release must still fire in the finally block.
    """
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=Exception("backend exploded")):
        with pytest.raises(Exception, match="backend exploded"):
            call_operator("gravitywell", prompt="test")
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


def test_direct_dispatch_acquire_carries_work_id_principal(monkeypatch):
    """Direct dispatch (off mode) with no explicit principal must still pass an
    attributable principal=<work_id> to acquire(), not omit it (which would get
    ghost-stamped server-side)."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "off")
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        result = call_operator("gravitywell", prompt="test")
    assert result == "ok"
    _, acquire_kwargs = mock_client.acquire.call_args
    assert acquire_kwargs.get("principal") is not None
    work_id_arg = mock_client.acquire.call_args[0][1]
    assert acquire_kwargs["principal"] == work_id_arg


def test_direct_dispatch_acquire_carries_explicit_principal(monkeypatch):
    """Direct dispatch (off mode) with an explicit principal must pass that
    principal through unchanged, not override it with work_id."""
    monkeypatch.setenv("GW_ADMISSION_MODE", "off")
    dc, mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        result = call_operator("gravitywell", prompt="test", principal="explicit-caller")
    assert result == "ok"
    _, acquire_kwargs = mock_client.acquire.call_args
    assert acquire_kwargs.get("principal") == "explicit-caller"


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
    """AC4: fresh principal-group waits when CONTENDED; dispatches once drain clears.

    Updated for atomic drain-gate (gw-admission-drain-gate-atomic-acquire-v0):
    require_drain_clear=True is now passed to acquire; CONTENDED response → retry,
    serving response → proceed. The old two-step drain_count + acquire is gone.
    """
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    dc = MagicMock()
    instance = MagicMock()
    # First acquire returns CONTENDED (another group active); second returns serving.
    acquire_responses = [
        {"ok": False, "contended": True},
        {"status": "serving", "drain_cleared": True},
    ]
    instance.acquire.side_effect = acquire_responses
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="fresh-gate")

    assert result == "ok"
    assert ("success", "gravitywell") in prov
    # acquire must have been called at least twice (CONTENDED → retry → serving)
    assert instance.acquire.call_count >= 2
    # drain_count must NOT be called (replaced by atomic acquire)
    instance.drain_count.assert_not_called()


def test_ac4_ride_along_skips_drain_count(tmp_path, monkeypatch):
    """AC4: same-principal ride-along dispatches immediately, drain check skipped.

    Updated for atomic drain-gate: ride-alongs call acquire with require_drain_clear=False
    (unconditional). drain_count is never called (replaced by atomic acquire).
    """
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
    # drain_count is irrelevant; ride-along must NOT block even if contended drain
    instance.acquire.return_value = {"status": "serving", "drain_cleared": True}
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))

    with patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok"):
        prov = []
        result = call_operator("gravitywell", prompt="test", _provenance_out=prov,
                                principal="shared-gate")

    assert result == "ok"
    assert ("success", "gravitywell") in prov
    # drain_count must NOT be called (atomic acquire replaced two-step for all paths)
    instance.drain_count.assert_not_called()
    # Ride-along: acquire must have been called with require_drain_clear=False (or omitted)
    for call_args in instance.acquire.call_args_list:
        assert not call_args.kwargs.get("require_drain_clear", False), \
            "ride-along must not pass require_drain_clear=True"


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
        {"status": "serving", "drain_cleared": True},
    ]

    dc = MagicMock()
    instance = MagicMock()
    instance.acquire.side_effect = acquire_responses
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))

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
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))

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
    """AC12 (AC5a): older doorman returns serving without drain_cleared → proceed loud with provenance.

    Updated for atomic drain-gate: drain_count is no longer called. The AC5a forward-compat
    path triggers when the doorman returns {"status": "serving"} without drain_cleared=True,
    signaling it ignored require_drain_clear (pre-atomic doorman). Client loud-proceeds with
    drain_count_unavailable provenance (same semantics, new detection mechanism).
    """
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")
    monkeypatch.setenv("ELEVATOR_DB_PATH", str(tmp_path / "q.db"))
    monkeypatch.setenv("GW_ADMISSION_POLL_INTERVAL_SEC", "0.01")

    dc = MagicMock()
    instance = MagicMock()
    # Old doorman: returns serving WITHOUT drain_cleared (ignored require_drain_clear)
    instance.acquire.return_value = {"status": "serving"}
    dc.return_value = instance
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))

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


# ---------------------------------------------------------------------------
# gravitywell-creative: operator class registration and model guard
# ---------------------------------------------------------------------------

def test_creative_in_operator_defaults():
    """gravitywell-creative is registered with gravitywell-llama-70b default."""
    assert "gravitywell-creative" in OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["gravitywell-creative"] == "gravitywell-llama-70b"


def test_creative_unknown_operator_rejected():
    """Unknown operator still raises ValueError."""
    with pytest.raises(ValueError, match="Unknown operator_class"):
        call_operator("gravitywell-creative-typo", prompt="hi")


def test_creative_non_default_model_raises():
    """Passing a non-default model to gravitywell-creative raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        call_operator("gravitywell-creative", prompt="hi", model="some-other-model")
    assert "infrastructure operation" in str(exc_info.value)
    assert "some-other-model" in str(exc_info.value)
    assert OPERATOR_DEFAULTS["gravitywell-creative"] in str(exc_info.value)


# ---------------------------------------------------------------------------
# gravitywell-creative: routes to :8093 with llama-70b, no thinking knob
# ---------------------------------------------------------------------------

def test_creative_routes_to_creative_url():
    """gravitywell-creative passes GW_CREATIVE_URL and llama-70b model to backend."""
    from agents_core.llm import GW_CREATIVE_URL
    import json as _json

    captured = {}

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        captured["url"] = url
        captured["payload"] = json

        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'creative reply'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    with patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell-creative", prompt="hello")

    assert result == "creative reply"
    assert captured["url"].startswith(GW_CREATIVE_URL)
    assert captured["payload"]["model"] == "gravitywell-llama-70b"


def test_creative_no_thinking_knob_in_payload():
    """gravitywell-creative never adds chat_template_kwargs to the payload."""
    import json as _json

    captured = {}

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        captured["payload"] = json

        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    with patch("requests.post", side_effect=fake_post):
        call_operator("gravitywell-creative", prompt="test")

    assert "chat_template_kwargs" not in captured["payload"]


# ---------------------------------------------------------------------------
# gravitywell-creative: env override for GW_CREATIVE_URL
# ---------------------------------------------------------------------------

def test_creative_env_url_override(monkeypatch):
    """GW_CREATIVE_URL env var is honored; stub at that URL returns its response."""
    import json as _json

    monkeypatch.setenv("GW_CREATIVE_URL", "http://stub-creative:9999")
    # Reload the module-level constant by patching it directly.
    with patch("agents_core.llm.GW_CREATIVE_URL", "http://stub-creative:9999"):
        captured = {}

        def fake_post(url, json=None, timeout=None, stream=None, **kw):
            captured["url"] = url

            def lines():
                yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'stub'}, 'finish_reason': None}]})}"
                yield "data: [DONE]"

            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.iter_lines = MagicMock(return_value=lines())
            resp.close = MagicMock()
            return resp

        with patch("requests.post", side_effect=fake_post):
            result = call_operator("gravitywell-creative", prompt="hi")

    assert result == "stub"
    assert captured["url"].startswith("http://stub-creative:9999")


# ---------------------------------------------------------------------------
# gravitywell-creative: honest failure — CreativeOperatorUnavailable, no fallback
# ---------------------------------------------------------------------------

def test_creative_unreachable_raises_distinct_exception():
    """Dead endpoint raises CreativeOperatorUnavailable, not OperatorUnreachableError."""
    from requests.exceptions import ConnectionError as ReqConnError

    with patch("requests.post", side_effect=ReqConnError("connection refused")), \
         patch("time.sleep"):  # skip retry backoff
        with pytest.raises(CreativeOperatorUnavailable) as exc_info:
            call_operator("gravitywell-creative", prompt="hi")

    assert "Llama-70B" in str(exc_info.value) or "8093" in str(exc_info.value)
    # Must NOT be a bare OperatorUnreachableError
    assert not isinstance(exc_info.value, OperatorUnreachableError)


def test_creative_unreachable_never_returns_none():
    """On network failure, gravitywell-creative raises rather than returning None."""
    from requests.exceptions import ConnectionError as ReqConnError

    with patch("requests.post", side_effect=ReqConnError("connection refused")), \
         patch("time.sleep"):
        with pytest.raises(CreativeOperatorUnavailable):
            call_operator("gravitywell-creative", prompt="hi")


# ---------------------------------------------------------------------------
# gravitywell-creative: provenance carries explicit model key
# ---------------------------------------------------------------------------

def test_creative_provenance_carries_model_key():
    """Successful creative call appends ("success", "gravitywell-creative") 2-tuple."""
    import json as _json

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'response'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    prov = []
    with patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell-creative", prompt="hi", _provenance_out=prov)

    assert result == "response"
    assert ("success", "gravitywell-creative") in prov


def test_creative_think_true_raises_value_error():
    """think=True must raise ValueError - Llama-3.3-70B-Instruct is not a reasoning model."""
    with pytest.raises(ValueError, match="not a reasoning model"):
        call_operator("gravitywell-creative", prompt="hi", think=True)


# ---------------------------------------------------------------------------
# gravitywell-creative: 122B default path is byte-identical (regression guard)
# ---------------------------------------------------------------------------

def test_gravitywell_122b_default_path_unchanged(monkeypatch):
    """122B path still uses GW_URL (:8081) and gravitywell-122b; no url/model override.

    Pins GW_BACKEND=llamacpp explicitly rather than relying on ambient env / the conftest's
    122b-precached discovery cache - under an ambient GW_BACKEND=vllm shell (this arc's own
    documented workaround env) the unpinned test would resolve gravitywell-27b, fail this
    test's gravitywell-122b assertion, and additionally miss the conftest precache (keyed to
    122b) and fire a live /v1/models probe. Pinning makes this explicit-mode regression guard
    correct under any ambient env.
    """
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    captured = {}

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("122b-reply", captured)):
        result = call_operator("gravitywell", prompt="test")

    assert result == "122b-reply"
    from agents_core.llm import GW_URL
    assert captured["payload"]["model"] == "gravitywell-122b"
    assert captured["payload"].get("chat_template_kwargs") is not None  # thinking knob present


def test_gravitywell_122b_default_url_is_8081(monkeypatch):
    """122B operator hits :8081, not :8093.

    Pins GW_BACKEND=llamacpp for the same hermeticity reason as
    test_gravitywell_122b_default_path_unchanged.
    """
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    import json as _json

    captured = {}

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        captured["url"] = url

        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=fake_post):
        call_operator("gravitywell", prompt="test")

    assert ":8081" in captured["url"]
    assert ":8093" not in captured["url"]


# ---------------------------------------------------------------------------
# agents-core-gw-voicing-vllm-repoint-v0: GW_BACKEND / GW_MODEL vLLM repoint
# ---------------------------------------------------------------------------
#
# Helpers below pre-seed agents_core.llm._gw_handshake_cache directly for the
# (url, resolved-model) tuple under test, sidestepping the pre-flight handshake's
# live GET {url}/v1/models probe for tests that only care about payload shape /
# delegation. Tests that exercise the handshake itself (AC6) clear the cache and
# mock requests.get explicitly.

def _gw_precache(model, url=None):
    """Mark (url or GW_URL, model) as handshake-verified so the real call under test
    doesn't trigger a live /v1/models probe."""
    from agents_core.llm import _gw_handshake_cache, _gw_handshake_lock, GW_URL
    with _gw_handshake_lock:
        _gw_handshake_cache[(url or GW_URL, model)] = True


def _gw_clear_auto_detect_caches():
    """Clear both the handshake cache and the TTL-bounded discovery cache, overriding the
    repo-root conftest's precache so a test can exercise its own discovery mock instead."""
    from agents_core.llm import _gw_handshake_cache, _gw_handshake_lock, _gw_discovery_cache
    with _gw_handshake_lock:
        _gw_handshake_cache.clear()
        _gw_discovery_cache.clear()


def _fake_gw_models_get(served_model):
    """A requests.get side_effect returning `served_model` from /v1/models."""
    def fake_get(url, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [{"id": served_model}]}
        return resp
    return fake_get


# --- unset GW_BACKEND/GW_MODEL: auto-detect (agents-core-gw-backend-auto-detect-when-unset-v0) --

def test_default_unset_auto_detects_vllm_when_dual_serving(monkeypatch):
    """No env set -> _gw_default_model() asks GW_URL what it is currently serving instead
    of assuming the legacy 122b default. When GW is resting dual (27B, today's real
    boot-default posture per infra/gw-dual-boot-default-promoted-2026-07-08), the payload
    uses the vllm dialect (no cache_prompt)."""
    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-27b")), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        result = call_operator("gravitywell", prompt="hi")

    assert result == "ok"
    assert captured["payload"] == {
        "model": "gravitywell-27b",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_default_unset_auto_detects_llamacpp_when_big_serving(monkeypatch):
    """The other direction: discovery reporting gravitywell-122b (GW resting big) resolves
    the llamacpp dialect (cache_prompt present) - proves genuine mode-agnosticism, not a
    hardcoded flip from one fixed default to another."""
    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-122b")), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        result = call_operator("gravitywell", prompt="hi")

    assert result == "ok"
    assert captured["payload"]["cache_prompt"] is True
    assert captured["payload"]["model"] == "gravitywell-122b"


def test_ac1_gw_backend_llamacpp_explicit_same_as_unset(monkeypatch):
    """Explicit GW_BACKEND=llamacpp resolves the same payload shape as an unset env would
    have resolved before this seam existed (gravitywell-122b, cache_prompt: True) - but
    unlike unset, it is pinned and never auto-detects, so it stays correct even if GW is
    actually resting dual at call time."""
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_precache("gravitywell-122b")

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        call_operator("gravitywell", prompt="hi")

    assert captured["payload"]["cache_prompt"] is True
    assert captured["payload"]["model"] == "gravitywell-122b"


# --- AC2: vLLM dialect -------------------------------------------------------

def test_ac2_gw_backend_vllm_omits_cache_prompt_default_model(monkeypatch):
    """GW_BACKEND=vllm + no GW_MODEL -> omits cache_prompt, model=gravitywell-27b."""
    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_precache("gravitywell-27b")

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        result = call_operator("gravitywell", prompt="hi", json_mode=True)

    assert result == "ok"
    assert "cache_prompt" not in captured["payload"]
    assert captured["payload"]["model"] == "gravitywell-27b"
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_ac2_gw_backend_vllm_with_explicit_gw_model_override(monkeypatch):
    """GW_BACKEND=vllm + explicit GW_MODEL overrides the gravitywell-27b convenience default."""
    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.setenv("GW_MODEL", "gravitywell-27b-quant")
    _gw_precache("gravitywell-27b-quant")

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        call_operator("gravitywell", prompt="hi")

    assert "cache_prompt" not in captured["payload"]
    assert captured["payload"]["model"] == "gravitywell-27b-quant"


# --- AC3: model override + guard consistency --------------------------------

def test_ac3_gw_model_override_honored_under_llamacpp_backend(monkeypatch):
    """GW_MODEL is honored under the llamacpp (default) backend too — cache_prompt stays."""
    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.setenv("GW_MODEL", "gravitywell-custom")
    _gw_precache("gravitywell-custom")

    captured = {}
    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok", captured)):
        result = call_operator("gravitywell", prompt="hi", model="gravitywell-custom")

    assert result == "ok"
    assert captured["payload"]["model"] == "gravitywell-custom"
    assert captured["payload"]["cache_prompt"] is True


def test_ac3_model_override_accepted_when_matches_resolved_default(monkeypatch):
    """An explicit model= matching the GW_BACKEND/GW_MODEL-resolved default is accepted."""
    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_precache("gravitywell-27b")

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result = call_operator("gravitywell", prompt="hi", model="gravitywell-27b")

    assert result == "ok"


def test_ac3_stale_model_override_raises_under_vllm_backend(monkeypatch):
    """A stale model= ("gravitywell-122b") raises once GW_BACKEND=vllm changes the default."""
    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.delenv("GW_MODEL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        call_operator("gravitywell", prompt="hi", model="gravitywell-122b")
    assert "gravitywell-27b" in str(exc_info.value)


# --- AC4: delegation preserved (Council adapter, zero adapter changes) -----

def test_ac4_council_adapter_delegates_model_to_seam_under_vllm(monkeypatch):
    """GravityWellAdapter (unmodified) sends whatever GW_BACKEND/GW_MODEL resolve to —
    proves Council's adapter needs zero changes to inherit the repoint."""
    from agents_core.council.gravitywell_adapter import GravityWellAdapter

    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_precache("gravitywell-27b")

    class _Msg:
        role = "user"
        content = "hello"

    captured = {}
    dc, _mock_client = _gw_dc()
    adapter = GravityWellAdapter(on_wake_fail="skip")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("voiced", captured)):
        result = adapter.chat(system="persona", messages=[_Msg()])

    assert result == "voiced"
    assert captured["payload"]["model"] == "gravitywell-27b"
    assert "cache_prompt" not in captured["payload"]
    assert adapter.voicing_events[-1]["effective_operator"] == "gravitywell"


# --- AC5: true-mirror provenance --------------------------------------------

def test_ac5_served_model_recorded_from_response_not_request():
    """The recorded served-model comes from the response's "model" field, not the request's —
    proven by sending a different requested model than the response echoes back."""
    from agents_core.llm import _call_gravitywell_backend
    import json as _json

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        assert json["model"] == "requested-model-name"

        def lines():
            chunk = {
                "model": "actually-served-model",
                "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
            }
            yield f"data: {_json.dumps(chunk)}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    out = []
    with patch("requests.post", side_effect=fake_post):
        result = _call_gravitywell_backend(
            prompt="hi", _url="http://alt-endpoint:9999", _model="requested-model-name",
            _served_model_out=out,
        )

    assert result == "hi"
    assert out == ["actually-served-model"]


# --- AC6: serving-mode drift hard-fails, distinctly from unavailability ----

def test_ac6a_handshake_wrong_model_raises_mismatch_no_fallback(monkeypatch):
    """Reachable /v1/models reporting the wrong name raises GWServingModeMismatchError and
    never reaches the paid on_wake_fail fallback."""
    from agents_core.llm import GWServingModeMismatchError, _gw_handshake_cache, _gw_handshake_lock

    monkeypatch.setenv("GW_BACKEND", "vllm")
    monkeypatch.delenv("GW_MODEL", raising=False)
    with _gw_handshake_lock:
        _gw_handshake_cache.clear()

    def fake_get(url, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": [{"id": "gravitywell-122b"}]}
        return resp

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=fake_get), \
         patch("agents_core.claude_queue_sync.submit_and_wait") as mock_saw:
        with pytest.raises(GWServingModeMismatchError) as exc_info:
            call_operator("gravitywell", prompt="test", on_wake_fail="sonnet")

    assert "gravitywell-27b" in str(exc_info.value)
    assert "gravitywell-122b" in str(exc_info.value)
    mock_saw.assert_not_called()


def test_ac6b_response_echo_mismatch_raises_mid_call(monkeypatch):
    """A response whose echoed model != the auto-detected model raises mid-call - the
    per-call response-echo check fires for the auto-detected case too (agents-core-gw-
    backend-auto-detect-when-unset-v0), catching a flip between the discovery probe and
    this call's actual streamed response, exactly as it already did for the explicit-
    handshake case."""
    from agents_core.llm import GWServingModeMismatchError
    import json as _json

    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        def lines():
            chunk = {
                "model": "gravitywell-27b",
                "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
            }
            yield f"data: {_json.dumps(chunk)}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-122b")) as mock_get, \
         patch("requests.post", side_effect=fake_post):
        with pytest.raises(GWServingModeMismatchError):
            call_operator("gravitywell", prompt="test")

    # Auto-detect discovery probe does fire (there is no pre-flight verify to elide it
    # when GW_BACKEND/GW_MODEL are unset), and only once (TTL-cached across the two
    # _gw_default_model() call sites within this single call_operator() invocation).
    assert mock_get.call_count == 1


def test_discovery_cached_second_call_within_ttl_does_not_reprobe(monkeypatch):
    """Auto-detect discovery probes /v1/models once per url within the TTL window
    (default 30s, GW_DISCOVERY_TTL_S); a second call_operator() invocation shortly after
    the first reuses the cached discovery result instead of re-probing. Supersedes the
    old per-process (url, model) handshake-cache framing - discovery is now per-url with
    a TTL, not per-(url, model) for the process lifetime."""
    _gw_clear_auto_detect_caches()

    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-122b")) as mock_get, \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        call_operator("gravitywell", prompt="one")
        call_operator("gravitywell", prompt="two")

    assert mock_get.call_count == 1


def test_discovery_ttl_reprobes_after_expiry(monkeypatch):
    """Two calls to _gw_discover_serving for the same url more than GW_DISCOVERY_TTL_S
    apart result in two separate requests.get calls - proves the TTL actually re-probes,
    not just that it caches (test_discovery_cached_second_call_within_ttl_does_not_reprobe
    only proves the caching half). GW_DISCOVERY_TTL_S is read at call time (not a
    module-load constant), so monkeypatch.setenv works here."""
    from agents_core.llm import _gw_discover_serving, GW_URL

    monkeypatch.setenv("GW_DISCOVERY_TTL_S", "0")
    _gw_clear_auto_detect_caches()

    with patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-122b")) as mock_get:
        _gw_discover_serving(GW_URL)
        _gw_discover_serving(GW_URL)

    assert mock_get.call_count == 2


def test_ac6d_handshake_connect_failure_is_unavailability_not_drift(monkeypatch):
    """A connect failure on the /v1/models probe (both the explicit-mode handshake and
    the auto-detect discovery probe route through the same shared transport helper) is
    treated as unavailability (not drift): it does not raise GWServingModeMismatchError,
    and the real call proceeds normally."""
    import requests as req

    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    dc, _mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=req.exceptions.ConnectionError("no route to host")), \
         patch("requests.post", side_effect=_make_gw_sse_resp("recovered")):
        result = call_operator("gravitywell", prompt="test")

    assert result == "recovered"


def test_ac6d_genuine_unavailability_still_routes_on_wake_fail(monkeypatch):
    """A genuine wake-fail (real call unreachable), distinct from served-wrong-model drift,
    keeps the existing on_wake_fail behavior unchanged - even with a discovery probe
    connect-failure in the mix."""
    import requests as req

    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    dc, _mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=req.exceptions.ConnectionError("no route to host")), \
         patch("requests.post", side_effect=req.exceptions.ConnectionError("gw down")), \
         patch("time.sleep"):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")

    assert result is None


def test_discovery_probe_failure_falls_back_then_operator_unreachable(monkeypatch):
    """A discovery probe failure (requests.get raises ConnectionError) with GW_BACKEND/
    GW_MODEL unset makes _gw_default_model() fall back to OPERATOR_DEFAULTS["gravitywell"]
    rather than raising or hanging. When the subsequent real call_operator("gravitywell",
    ...) call's requests.post also fails to connect, it raises OperatorUnreachableError via
    the existing connect-retry path - never GWServingModeMismatchError, never a hang."""
    from agents_core.llm import (
        _gw_default_model, _gw_handshake_lock, _gw_discovery_cache,
        OPERATOR_DEFAULTS, OperatorUnreachableError,
    )
    import requests as req

    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    with patch("requests.get", side_effect=req.exceptions.ConnectionError("no route to host")):
        assert _gw_default_model() == OPERATOR_DEFAULTS["gravitywell"]

    # A failed probe is never cached - clear defensively so the assertion above's probe
    # attempt doesn't leak into the real call below (it shouldn't have cached anything,
    # but this keeps the test's intent explicit).
    with _gw_handshake_lock:
        _gw_discovery_cache.clear()

    dc, _mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=req.exceptions.ConnectionError("no route to host")), \
         patch("requests.post", side_effect=req.exceptions.ConnectionError("gw down")), \
         patch("time.sleep"):
        with pytest.raises(OperatorUnreachableError):
            call_operator("gravitywell", prompt="test", on_wake_fail="error")


def test_ac3_model_override_accepted_when_matches_auto_detected_default(monkeypatch):
    """The override-validation call site (call_operator's own _gw_default_model() call,
    used to validate a caller-supplied model= against the resolved default) composes
    correctly with auto-detect, not just the explicit GW_BACKEND path: an explicit model=
    matching the discovered model is accepted."""
    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    dc, _mock_client = _gw_dc()
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-27b")), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result = call_operator("gravitywell", prompt="hi", model="gravitywell-27b")

    assert result == "ok"


def test_ac3_stale_model_override_raises_under_auto_detect(monkeypatch):
    """A stale model= override that doesn't match the auto-detected served model raises
    the same ValueError this call site already raises today under the explicit path."""
    monkeypatch.delenv("GW_BACKEND", raising=False)
    monkeypatch.delenv("GW_MODEL", raising=False)
    _gw_clear_auto_detect_caches()

    with patch("requests.get", side_effect=_fake_gw_models_get("gravitywell-27b")):
        with pytest.raises(ValueError) as exc_info:
            call_operator("gravitywell", prompt="hi", model="gravitywell-122b")
    assert "gravitywell-27b" in str(exc_info.value)


# --- AC7: unknown-backend hard-fails ----------------------------------------

def test_ac7_gw_backend_unset_resolves_llamacpp(monkeypatch):
    """_gw_backend() called with no discovered_model context (no real call site does this
    once auto-detecting - they all pass discovered_model) falls back to "llamacpp". This
    is the ultimate fallback value, not the value real call sites resolve to once
    auto-detecting; see test_default_unset_auto_detects_vllm_when_dual_serving for the
    discovered_model-aware resolution real calls actually use."""
    from agents_core.llm import _gw_backend

    monkeypatch.delenv("GW_BACKEND", raising=False)
    assert _gw_backend() == "llamacpp"


def test_ac7_gw_backend_unrecognized_raises_value_error(monkeypatch):
    from agents_core.llm import _gw_backend

    monkeypatch.setenv("GW_BACKEND", "vllm2")
    with pytest.raises(ValueError, match="Unknown GW_BACKEND"):
        _gw_backend()


# --- AC8: creative-path isolation (scope guard) -----------------------------

def test_ac8_creative_path_unaffected_by_gw_backend_vllm(monkeypatch):
    """GW_BACKEND=vllm set process-wide leaves the gravitywell-creative (:8093) path
    unchanged: cache_prompt retained, no handshake probe, no GWServingModeMismatchError."""
    monkeypatch.setenv("GW_BACKEND", "vllm")
    import json as _json

    def fake_get(url, timeout=None):
        raise AssertionError("creative path must never probe /v1/models")

    captured = {}

    def fake_post(url, json=None, timeout=None, stream=None, **kw):
        captured["payload"] = json

        def lines():
            yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    with patch("requests.get", side_effect=fake_get), \
         patch("requests.post", side_effect=fake_post):
        result = call_operator("gravitywell-creative", prompt="test")

    assert result == "ok"
    assert captured["payload"]["cache_prompt"] is True
    assert captured["payload"]["model"] == "gravitywell-llama-70b"
    assert "chat_template_kwargs" not in captured["payload"]


def test_ac8_creative_path_unaffected_by_invalid_gw_backend(monkeypatch):
    """An invalid GW_BACKEND value never reaches the creative path's _gw_backend() call at
    all — no ValueError, no coupling — because the scope guard checks _url/_model first."""
    monkeypatch.setenv("GW_BACKEND", "not-a-real-backend")

    with patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result = call_operator("gravitywell-creative", prompt="test")

    assert result == "ok"


def test_ac8_creative_payload_identical_with_and_without_gw_backend_vllm():
    """Extends test_creative_no_thinking_knob_in_payload: the creative payload is byte-
    identical whether or not GW_BACKEND=vllm is set process-wide."""
    import json as _json

    def _capture_creative_payload():
        captured = {}

        def fake_post(url, json=None, timeout=None, stream=None, **kw):
            captured["payload"] = json

            def lines():
                yield f"data: {_json.dumps({'choices': [{'delta': {'content': 'ok'}, 'finish_reason': None}]})}"
                yield "data: [DONE]"

            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.iter_lines = MagicMock(return_value=lines())
            resp.close = MagicMock()
            return resp

        with patch("requests.post", side_effect=fake_post):
            call_operator("gravitywell-creative", prompt="test")
        return captured["payload"]

    import os as _os
    prior = _os.environ.pop("GW_BACKEND", None)
    try:
        without_env = _capture_creative_payload()
        _os.environ["GW_BACKEND"] = "vllm"
        with_env = _capture_creative_payload()
    finally:
        if prior is None:
            _os.environ.pop("GW_BACKEND", None)
        else:
            _os.environ["GW_BACKEND"] = prior

    assert without_env == with_env


# ---------------------------------------------------------------------------
# gravitywell: acquire_lease bypass (call-operator-gravitywell-lease-bypass-v0)
#
# A caller that already holds a GravityWell mode-controller lease self-deadlocks
# against its own lease when call_operator("gravitywell") does its normal worker
# acquire. acquire_lease=False (kwarg, default True) skips the doorman entirely
# and dispatches straight to _call_gravitywell_backend. See gw_agent.py:953 for
# the sibling call_gw_agent(acquire_lease=...) precedent this mirrors.
# ---------------------------------------------------------------------------

def test_gravitywell_acquire_lease_false_never_touches_doorman(monkeypatch):
    """R4(a): acquire_lease=False performs no DoormanClient construction/acquire/release
    and still returns the backend result."""
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    dc, mock_client = _gw_dc(status="serving")

    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result = call_operator("gravitywell", prompt="test", acquire_lease=False)

    assert result == "ok"
    dc.assert_not_called()
    mock_client.acquire.assert_not_called()
    mock_client.release.assert_not_called()


def test_gravitywell_acquire_lease_false_bypasses_deferring_doorman():
    """R4(b): the incident regression. Even when a stubbed doorman would defer
    (mode-controller lease held elsewhere), acquire_lease=False must still reach
    _call_gravitywell_backend rather than short-circuiting to on_wake_fail."""
    dc, _mock_client = _gw_dc(status="deferred")
    prov = []

    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="bypassed-ok") as mock_backend:
        result = call_operator(
            "gravitywell", prompt="test", acquire_lease=False,
            on_wake_fail="skip", _provenance_out=prov,
        )

    assert result == "bypassed-ok"
    mock_backend.assert_called_once()
    dc.assert_not_called()
    assert ("success", "gravitywell") in prov
    assert not any(r in ("gw_deferred_swarm", "gw_not_serving") for r, _ in prov)


def test_gravitywell_acquire_lease_false_bypasses_enforce_and_shadow_mode(monkeypatch):
    """R4(b) extension: the bypass short-circuits regardless of GW_ADMISSION_MODE, not
    just the off/direct-dispatch path — a lease-holder bypasses ALL doorman admission."""
    dc, _mock_client = _gw_dc(status="deferred")

    for mode in ("enforce", "shadow"):
        monkeypatch.setenv("GW_ADMISSION_MODE", mode)
        with patch("agents_core.doorman_client.DoormanClient", dc), \
             patch("agents_core.llm._call_gravitywell_backend", return_value="bypassed-ok") as mock_backend:
            result = call_operator("gravitywell", prompt="test", acquire_lease=False)
        assert result == "bypassed-ok", f"mode={mode}"
        mock_backend.assert_called_once()
        dc.assert_not_called()


def test_gravitywell_acquire_lease_true_default_unchanged():
    """R4(c): the default (acquire_lease omitted, i.e. True) path is byte-identical —
    still acquires, still honors is_deferred, still releases."""
    dc, mock_client = _gw_dc(status="deferred")
    with patch("agents_core.doorman_client.DoormanClient", dc):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")
    assert result is None
    mock_client.acquire.assert_called_once()

    dc2, mock_client2 = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc2), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result2 = call_operator("gravitywell", prompt="test", acquire_lease=True)
    assert result2 == "ok"
    mock_client2.acquire.assert_called_once()
    mock_client2.release.assert_called_once()


def test_gravitywell_acquire_lease_not_leaked_to_backend(monkeypatch):
    """acquire_lease must be stripped before forwarding — never reaches
    _call_gravitywell_backend as a stray kwarg (would TypeError otherwise)."""
    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_MODEL", raising=False)
    dc, _mock_client = _gw_dc(status="serving")

    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp("ok")):
        result = call_operator("gravitywell", prompt="test", acquire_lease=False)
    assert result == "ok"


def test_gravitywell_acquire_lease_false_is_a_trust_contract_not_enforced():
    """R4(d) — contract-witness test. acquire_lease=False is honored purely on the
    caller's word; call_operator performs NO lease-ownership check before bypassing
    the doorman. This is deliberate (Erah-ratified 2026-07-24, Scope boundary section
    of call-operator-gravitywell-lease-bypass-v0): the bypass is an opt-in trust
    contract identical to call_gw_agent(acquire_lease=False), not a runtime-enforced
    guard. A caller with NO lease at all still bypasses successfully — proving there
    is no ownership check to defeat."""
    with patch("agents_core.doorman_client.DoormanClient") as dc, \
         patch("agents_core.llm._call_gravitywell_backend", return_value="ok") as mock_backend:
        result = call_operator("gravitywell", prompt="test", acquire_lease=False)

    assert result == "ok"
    mock_backend.assert_called_once()
    dc.assert_not_called()
