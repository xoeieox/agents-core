"""Tests for dual-timer streaming liveness in GW backend (gw-liveness-idle-timeout-v0).

AC1: steady stream completes normally.
AC2: first chunk then idle silence culled with idle_gap_exceeded.
AC3: phase-1 silence behaviour — idle_gap NOT used pre-first-token; first_token_gap used.
AC4: reasoning_content-only chunks reset idle timer.
AC4b: long pre-first-token silence (>IDLE_GAP, <FIRST_TOKEN_GAP) no cull; phase-2 applies after.
AC5: clean stream completion returns assembled text; empty/whitespace returns None.
AC6: connection refused → 3 retries → OperatorUnreachableError; on_wake_fail applied.
AC7: orchestrator Popen-based liveness — live child survives; silent child is killed.
AC8: stall-retry transparent to admission layer.
"""

import io
import json
import logging
import os
import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from agents_core.llm import _gw_stream_attempt, _call_gravitywell_backend, OperatorUnreachableError


# ---------------------------------------------------------------------------
# Helpers: build mock SSE responses
# ---------------------------------------------------------------------------

def make_sse_resp(chunks_before_stall, stall_after=True, add_done=False):
    """Build a mock requests.Response that streams SSE lines.

    chunks_before_stall: list of (content_str_or_None, reasoning_str_or_None).
    stall_after: if True, iter_lines blocks on close_evt before returning.
    add_done: if True and stall_after=False, append a [DONE] line.
    """
    close_evt = threading.Event()

    def lines():
        for c, r in chunks_before_stall:
            delta = {}
            if c:
                delta["content"] = c
            if r:
                delta["reasoning_content"] = r
            yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': None}]})}"
        if stall_after:
            close_evt.wait(timeout=10)
        elif add_done:
            yield "data: [DONE]"

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines = MagicMock(return_value=lines())

    def do_close():
        close_evt.set()

    resp.close = do_close
    return resp


# ---------------------------------------------------------------------------
# AC1: steady stream runs to completion, returns assembled text
# ---------------------------------------------------------------------------

def test_ac1_steady_stream_completes(monkeypatch):
    """Chunks arriving at IDLE_GAP/2 cadence complete without cull."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "1")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "10")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    # 3 chunks arriving quickly, then [DONE]
    def make_resp():
        close_evt = threading.Event()
        chunk_delay = 0.4  # IDLE_GAP/2 = 0.5s → 0.4s is safe

        def lines():
            for i in range(3):
                time.sleep(chunk_delay)
                c = f"chunk{i}"
                yield f"data: {json.dumps({'choices': [{'delta': {'content': c}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = lambda: close_evt.set()
        return resp

    resp = make_resp()

    with patch("requests.post", return_value=resp):
        idle_gap = 1.0
        first_token_gap = 10.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert cull is None
    assert text == "chunk0chunk1chunk2"


# ---------------------------------------------------------------------------
# AC2: first chunk then silence > IDLE_GAP → culled with idle_gap_exceeded
# ---------------------------------------------------------------------------

def test_ac2_idle_silence_culled(monkeypatch):
    """After first token, silence > IDLE_GAP triggers idle_gap_exceeded cull."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "1")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "10")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    # One chunk then stall
    resp = make_sse_resp([("hello", None)], stall_after=True)

    with patch("requests.post", return_value=resp):
        idle_gap = 1.0
        first_token_gap = 10.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert text is None
    assert cull is not None
    assert cull[0] == "idle_gap_exceeded"


def test_ac2_idle_silence_culled_logs(monkeypatch, caplog):
    """idle_gap_exceeded is logged as error in _call_gravitywell_backend."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "1")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "10")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    resp = make_sse_resp([("hello", None)], stall_after=True)

    with patch("requests.post", return_value=resp), \
         caplog.at_level(logging.ERROR, logger="agents_core.llm"):
        result = _call_gravitywell_backend("test prompt")

    assert result is None
    # After stall retry, logged at ERROR
    assert any("idle_gap_exceeded" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AC3: phase-1 silence — IDLE_GAP not active pre-first-token; FIRST_TOKEN_GAP is
# ---------------------------------------------------------------------------

def test_ac3_phase1_silence_longer_than_idle_gap_not_culled(monkeypatch):
    """Pre-first-token silence > IDLE_GAP but < FIRST_TOKEN_GAP: NOT culled."""
    # IDLE_GAP = 0.5s, FIRST_TOKEN_GAP = 5s; sleep 0.7s then deliver a token
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "0.5")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "5")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    def make_resp():
        def lines():
            time.sleep(0.7)  # > IDLE_GAP, < FIRST_TOKEN_GAP
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    resp = make_resp()
    with patch("requests.post", return_value=resp):
        idle_gap = 0.5
        first_token_gap = 5.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert cull is None
    assert text == "hi"


def test_ac3_phase1_silence_exceeds_first_token_gap_culled(monkeypatch):
    """Pre-first-token silence > FIRST_TOKEN_GAP triggers first_token_grace_exceeded."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "1")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    resp = make_sse_resp([], stall_after=True)  # No chunks at all — stall from start

    with patch("requests.post", return_value=resp):
        idle_gap = 10.0
        first_token_gap = 1.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert text is None
    assert cull is not None
    assert cull[0] == "first_token_grace_exceeded"


# ---------------------------------------------------------------------------
# AC4: reasoning_content-only chunks reset idle timer
# ---------------------------------------------------------------------------

def test_ac4_reasoning_content_resets_idle(monkeypatch):
    """reasoning_content-only chunks (no content) prevent idle_gap cull."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "1")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "10")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    def make_resp():
        def lines():
            for i in range(3):
                time.sleep(0.3)  # < IDLE_GAP
                yield f"data: {json.dumps({'choices': [{'delta': {'reasoning_content': f'think{i}'}, 'finish_reason': None}]})}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    resp = make_resp()
    with patch("requests.post", return_value=resp):
        idle_gap = 1.0
        first_token_gap = 10.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert cull is None
    # reasoning_content fallback when content is empty
    assert text == "think0think1think2"


# ---------------------------------------------------------------------------
# AC4b: long pre-first-token silence (>IDLE_GAP, <FIRST_TOKEN_GAP), then first
#        token arrives — no phase-1 cull; phase-2 begins after first token
# ---------------------------------------------------------------------------

def test_ac4b_pre_first_token_silence_then_token_then_phase2(monkeypatch):
    """Pre-first-token silence > IDLE_GAP but < FIRST_TOKEN_GAP → token arrives → no cull.
    After token, another silence > IDLE_GAP triggers phase-2 cull."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "0.5")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "5")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")

    def make_resp():
        close_evt = threading.Event()

        def lines():
            time.sleep(0.7)  # > IDLE_GAP, < FIRST_TOKEN_GAP: phase-1 safe
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'first'}, 'finish_reason': None}]})}"
            # Now in phase-2; stall > IDLE_GAP
            close_evt.wait(timeout=10)

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = lambda: close_evt.set()
        return resp

    resp = make_resp()
    with patch("requests.post", return_value=resp):
        idle_gap = 0.5
        first_token_gap = 5.0
        hard_ceiling = 30.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    # Phase-1 silence didn't cull; phase-2 did
    assert cull is not None
    assert cull[0] == "idle_gap_exceeded"


# ---------------------------------------------------------------------------
# AC5: clean stream returns assembled text; empty/whitespace returns None
# ---------------------------------------------------------------------------

def test_ac5_clean_stream_returns_text(monkeypatch):
    """Clean [DONE] stream returns assembled content."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "60")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "1800")

    chunks = [("Hello ", None), ("world", None)]
    resp = make_sse_resp(chunks, stall_after=False, add_done=True)

    with patch("requests.post", return_value=resp):
        idle_gap = 10.0
        first_token_gap = 60.0
        hard_ceiling = 1800.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert cull is None
    assert text == "Hello world"


def test_ac5_reasoning_content_fallback(monkeypatch):
    """If content is empty but reasoning_content present, returns reasoning_content."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "60")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "1800")

    chunks = [(None, "thinking deeply"), (None, " more")]
    resp = make_sse_resp(chunks, stall_after=False, add_done=True)

    with patch("requests.post", return_value=resp):
        idle_gap = 10.0
        first_token_gap = 60.0
        hard_ceiling = 1800.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert cull is None
    assert text == "thinking deeply more"


def test_ac5_empty_content_returns_none(monkeypatch):
    """Whitespace-only content returns None."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "60")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "1800")

    chunks = [("   ", None)]
    resp = make_sse_resp(chunks, stall_after=False, add_done=True)

    with patch("requests.post", return_value=resp):
        idle_gap = 10.0
        first_token_gap = 60.0
        hard_ceiling = 1800.0
        call_start = time.monotonic()
        payload = {"model": "gravitywell-122b", "messages": [], "stream": True}
        text, cull = _gw_stream_attempt(
            "http://gw", "gravitywell-122b", payload,
            idle_gap, first_token_gap, hard_ceiling, call_start, None,
        )

    assert text is None


# ---------------------------------------------------------------------------
# AC6: connection refused → 3 retries → OperatorUnreachableError
# ---------------------------------------------------------------------------

def test_ac6_connection_refused_raises_operator_unreachable(monkeypatch):
    """ConnectionError on every attempt → OperatorUnreachableError after 3 retries."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "60")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "1800")

    import requests as _requests

    call_count = [0]

    def fail_post(*a, **kw):
        call_count[0] += 1
        raise _requests.exceptions.ConnectionError("connection refused")

    with patch("requests.post", side_effect=fail_post), \
         patch("time.sleep"):  # skip backoff delays
        with pytest.raises(OperatorUnreachableError):
            _call_gravitywell_backend("test")

    # 3 connect attempts per call, 2 total calls (first + stall retry would not trigger
    # since connection never succeeds to produce idle_gap_exceeded — so only 3 attempts)
    assert call_count[0] == 3


def test_ac6_on_wake_fail_applied(monkeypatch):
    """After OperatorUnreachableError, on_wake_fail policy is applied by caller (call_operator)."""
    from agents_core.llm import call_operator
    from agents_core.doorman_client import DoormanClient

    monkeypatch.setenv("GW_IDLE_GAP_SECS", "10")

    import requests as _requests

    def fail_post(*a, **kw):
        raise _requests.exceptions.ConnectionError("connection refused")

    mock_client = MagicMock()
    mock_client.acquire.return_value = {"status": "serving", "drain_cleared": True}
    mock_class = MagicMock(return_value=mock_client)
    mock_class.is_deferred = lambda r: r.get("status") == "deferred"
    mock_class.is_contended = lambda r: bool(r.get("contended"))

    with patch("agents_core.doorman_client.DoormanClient", mock_class), \
         patch("requests.post", side_effect=fail_post), \
         patch("time.sleep"):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")

    assert result is None


# ---------------------------------------------------------------------------
# AC7: orchestrator Popen-based liveness
# ---------------------------------------------------------------------------

def test_ac7_live_child_survives(monkeypatch):
    """Live child emitting stderr output every 0.2s survives FACETS_ORCH_IDLE_KILL_SECS=1."""
    monkeypatch.setenv("FACETS_ORCH_IDLE_KILL_SECS", "1")
    monkeypatch.setenv("FACETS_ORCH_HARD_CEILING_SECS", "30")

    terminate_called = threading.Event()

    class FakeProc:
        def __init__(self, argv, **kwargs):
            self.returncode = None
            self._done = threading.Event()
            self.stdout = self._make_stdout()
            self.stderr = self._make_stderr()

        def _make_stdout(self):
            q = queue.Queue()

            def fill():
                self._done.wait(timeout=15)
                q.put(json.dumps({"deliberation_id": "live-test-id"}))
                q.put(None)

            threading.Thread(target=fill, daemon=True).start()

            class QueueIter:
                def __iter__(self_inner):
                    while True:
                        item = q.get()
                        if item is None:
                            return
                        yield item

            return QueueIter()

        def _make_stderr(self):
            q = queue.Queue()

            def fill():
                for i in range(5):
                    time.sleep(0.2)
                    q.put(f"[facets] round {i}\n")
                self._done.set()
                q.put(None)

            threading.Thread(target=fill, daemon=True).start()

            class QueueIter:
                def __iter__(self_inner):
                    while True:
                        item = q.get()
                        if item is None:
                            return
                        yield item

            return QueueIter()

        def poll(self):
            if self._done.is_set():
                self.returncode = 0
                return 0
            return None

        def terminate(self):
            terminate_called.set()

        def wait(self, timeout=None):
            pass

    fake_proc = FakeProc([])

    with patch("subprocess.Popen", return_value=fake_proc), \
         patch("os.path.exists", return_value=True):
        from agents_core.shared_deliberation.orchestrator import _run_facets_subprocess
        from pathlib import Path
        result = _run_facets_subprocess("text", {}, "gravitywell", Path("/fake/facets"), None, None)

    ok, deliberation_json, deliberation_id, error = result
    assert ok is True
    assert deliberation_id == "live-test-id"
    assert not terminate_called.is_set(), "Live child should NOT be terminated"


def test_ac7_silent_child_killed(monkeypatch):
    """Silent child (no stderr) is killed after FACETS_ORCH_IDLE_KILL_SECS."""
    monkeypatch.setenv("FACETS_ORCH_IDLE_KILL_SECS", "0.5")
    monkeypatch.setenv("FACETS_ORCH_HARD_CEILING_SECS", "30")

    terminate_called = threading.Event()

    class FakeProc:
        def __init__(self, argv, **kwargs):
            self.returncode = None
            self._terminated = threading.Event()
            self.stdout = self._make_stdout()
            self.stderr = self._make_stderr()

        def _make_stdout(self):
            # Will block until terminated
            q = queue.Queue()

            def fill():
                self._terminated.wait(timeout=15)
                q.put(None)

            threading.Thread(target=fill, daemon=True).start()

            class QueueIter:
                def __iter__(self_inner):
                    while True:
                        item = q.get()
                        if item is None:
                            return
                        yield item

            return QueueIter()

        def _make_stderr(self):
            # Emit nothing (silence)
            q = queue.Queue()

            def fill():
                self._terminated.wait(timeout=15)
                q.put(None)

            threading.Thread(target=fill, daemon=True).start()

            class QueueIter:
                def __iter__(self_inner):
                    while True:
                        item = q.get()
                        if item is None:
                            return
                        yield item

            return QueueIter()

        def poll(self):
            if self._terminated.is_set():
                self.returncode = -15
                return -15
            return None

        def terminate(self):
            terminate_called.set()
            self._terminated.set()

        def wait(self, timeout=None):
            pass

    fake_proc = FakeProc([])

    with patch("subprocess.Popen", return_value=fake_proc), \
         patch("os.path.exists", return_value=True):
        from agents_core.shared_deliberation.orchestrator import _run_facets_subprocess
        from pathlib import Path
        result = _run_facets_subprocess("text", {}, "gravitywell", Path("/fake/facets"), None, None)

    ok, deliberation_json, deliberation_id, error = result
    assert ok is False
    assert error == "Facets timeout (silence)"
    assert terminate_called.is_set(), "Silent child MUST be terminated"


# ---------------------------------------------------------------------------
# AC8: stall-retry transparent to admission layer
# ---------------------------------------------------------------------------

def test_ac8_stall_retry_transparent_to_admission(monkeypatch):
    """First stream → idle_gap_exceeded; second → success. DoormanClient.acquire called once."""
    monkeypatch.setenv("GW_IDLE_GAP_SECS", "1")
    monkeypatch.setenv("GW_FIRST_TOKEN_GAP_SECS", "10")
    monkeypatch.setenv("GW_LIVENESS_HARD_CEILING_SECS", "30")
    monkeypatch.setenv("GW_ADMISSION_MODE", "enforce")

    from agents_core.llm import call_operator

    # Mock elevator: immediate admit
    mock_elevator = MagicMock()
    mock_ticket = "ticket-123"
    mock_elevator.enqueue.return_value = mock_ticket
    mock_elevator.try_admit.return_value = (True, False)  # admitted, not ride-along
    mock_elevator.reclaim_stale = MagicMock()
    mock_elevator.ack = MagicMock()
    mock_elevator.fail = MagicMock()
    mock_elevator.close = MagicMock()

    # Mock doorman: serving
    mock_dc_instance = MagicMock()
    mock_dc_instance.acquire.return_value = {"status": "serving", "drain_cleared": True}
    mock_dc_instance.release = MagicMock()
    mock_dc_instance.close = MagicMock()
    mock_dc_class = MagicMock(return_value=mock_dc_instance)
    mock_dc_class.is_deferred = lambda r: r.get("status") == "deferred"
    mock_dc_class.is_contended = lambda r: bool(r.get("contended"))

    # First call: idle_gap_exceeded stall; second call: success
    attempt = [0]

    def mock_gw_stream_attempt(*args, **kwargs):
        attempt[0] += 1
        if attempt[0] == 1:
            return (None, ("idle_gap_exceeded", 2.0, 1.0))
        else:
            return ("final answer", None)

    with patch("agents_core.doorman_client.DoormanClient", mock_dc_class), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0), \
         patch("agents_core.elevator.ElevatorStore", return_value=mock_elevator), \
         patch("agents_core.elevator.IS_MASTER", True), \
         patch("agents_core.elevator.DB_DIR", __import__("pathlib").Path("/tmp")), \
         patch("agents_core.llm._gw_stream_attempt", side_effect=mock_gw_stream_attempt):
        result = call_operator("gravitywell", prompt="test", on_wake_fail="skip")

    assert result == "final answer"
    # Doorman acquire called exactly once (admission is single-entry, not per-attempt)
    assert mock_dc_instance.acquire.call_count == 1
    # Both stream attempts were made
    assert attempt[0] == 2
