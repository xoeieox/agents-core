"""Hermetic tests for doorman-flashnext-serving-admission-v0, leg 1 (agents-core).

S1 — /lease/acquire refusals become NAMED outcomes on the client:
  * a 409 whose body carries a named refusal flag is returned as a dict, never
    an escaping httpx.HTTPStatusError;
  * is_flashnext_occupied / is_creative_occupied both work (module-level and as
    staticmethods) and are the predicates consumers branch on;
  * I4 fail-open: an unparseable or flag-less 409 body takes TODAY's error path
    (raise), never a named skip; a non-409 status still raises; transport
    failures still raise DoormanUnreachable;
  * accept_flashnext_seat is sent only when True (default-false byte-identical).

S2 — the server's opt-in already-serving grant:
  * grant fires ONLY for (field true AND mode is None AND role !=
    "mode-controller" AND guard-computed seat_state == "up_registered");
  * every I1 pin: ANY supplied real mode ("big"/"dual") keeps the refusal,
    role="mode-controller" keeps the uniform refusal, the canned up_unverified
    mid-load pair refuses;
  * the grant registers a lease with the additive serve_axis="flashnext"
    (observable in /status via the **info spread) and surfaces it on the
    acquire response, WITHOUT a wake and WITHOUT a second probe pair;
  * fast-path write-set: no _cached_serving write, no last_wake_at; _place_hold
    DOES apply;
  * non-opt-in acquire is byte-identical to today (409 flashnext_occupied).

S3 (agents-core side) — named branches on the dict side:
  * gw_agent names flashnext-window / creative / contended instead of the flat
    gw_seat_occupied / gw_not_serving buckets, and keeps the legacy exception
    branch for pre-S1 doormen;
  * llm.py admission + direct-dispatch name the state from the DICT and do not
    re-pay the soft-retry loop; direct-dispatch stays NON-opt-in (no
    lease-then-fail degradation for a day-seat caller).

All HTTP is mocked (canned probe payloads, monkeypatched httpx) — no live GW
probes, no GPU, no network (I5).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from agents_core.doorman_client import (
    DoormanClient,
    DoormanUnreachable,
    is_creative_occupied,
    is_flashnext_occupied,
)
from agents_core.doorman_server import (
    FLASHNEXT_OCCUPIED,
    GW_FLASHNEXT_MODEL_ID,
    GW_URL_DEFAULT,
    _NodeState,
    create_app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTransport(httpx.BaseTransport):
    """Replays canned (status, body) pairs; records the last request body."""

    def __init__(self, responses: list[tuple[int, object]], capture: dict | None = None):
        self._responses = iter(responses)
        self._capture = capture

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        status, body = next(self._responses)
        if self._capture is not None:
            self._capture["body"] = request.content
        if isinstance(body, bytes):
            return httpx.Response(status, content=body,
                                  headers={"content-type": "text/plain"})
        return httpx.Response(status, json=body)


def _client_with(responses: list[tuple[int, object]], capture: dict | None = None) -> DoormanClient:
    c = DoormanClient(base_url="http://doorman.test")
    c._client = httpx.Client(
        base_url="http://doorman.test",
        transport=_MockTransport(responses, capture),
    )
    return c


def _make_state(gw_url: str = GW_URL_DEFAULT) -> _NodeState:
    return _NodeState(gw_url)


FLASHNEXT_GRANT = ("up_registered", GW_FLASHNEXT_MODEL_ID, True, None)
UP_UNVERIFIED = ("up_unverified", None, None, None)


class _Guard:
    """Context manager installing the canned-guard patches and exposing the
    wake-subprocess mock + probe call count."""

    def __init__(self, state: _NodeState, seat_result):
        self.state = state
        self.seat_result = seat_result
        self.mock_run = patch("agents_core.doorman_server.subprocess.run")
        self.mock_probe = patch.object(
            state, "_probe_flashnext_seat", return_value=seat_result
        )
        self.mock_serving = patch.object(state, "_is_serving", return_value=False)
        self.mock_creative = patch.object(
            state, "_is_creative_serving", return_value=False
        )

    def __enter__(self):
        self._serving = self.mock_serving.start()
        self.mock_creative.start()
        self._probe = self.mock_probe.start()
        self._run = self.mock_run.start()
        return self

    def __exit__(self, *_):
        for m in (self.mock_run, self.mock_probe, self.mock_creative, self.mock_serving):
            m.stop()

    @property
    def probe_calls(self) -> int:
        return self._probe.call_count

    @property
    def serving_calls(self) -> int:
        return self._serving.call_count

    @property
    def wake_issued(self) -> bool:
        """True iff the wake-gravitywell subprocess was actually invoked."""
        return self._run.call_count > 0


# ===========================================================================
# S1 — client: named outcomes
# ===========================================================================

class TestS1NamedRefusals:
    def test_409_flashnext_returns_dict_not_exception(self):
        c = _client_with([(409, {"ok": False, "flashnext_occupied": True,
                                 "reason": "flashnext-window-holding-gpu0"})])
        res = c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")
        assert isinstance(res, dict)
        assert is_flashnext_occupied(res) is True
        assert is_creative_occupied(res) is False
        assert DoormanClient.is_flashnext_occupied(res) is True
        assert DoormanClient.is_creative_occupied(res) is False

    def test_409_creative_returns_dict_and_predicate_is_used(self):
        c = _client_with([(409, {"ok": False, "creative_occupied": True,
                                 "reason": "creative-collider-holding-gpu"})])
        res = c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")
        assert is_creative_occupied(res) is True
        assert is_flashnext_occupied(res) is False
        assert DoormanClient.is_creative_occupied(res) is True

    def test_409_contended_returns_dict(self):
        c = _client_with([(409, {"ok": False, "contended": True, "node": "gravitywell"})])
        res = c.acquire("gravitywell", "w1", ttl_sec=120, reason="t",
                        require_drain_clear=True)
        assert DoormanClient.is_contended(res) is True

    def test_409_unknown_body_takes_todays_error_path(self):
        """I4: an unknown flag set is never read as a named skip."""
        c = _client_with([(409, {"ok": False, "reason": "something-new-inventee"})])
        with pytest.raises(httpx.HTTPStatusError):
            c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")

    def test_409_unparseable_body_takes_todays_error_path(self):
        c = _client_with([(409, b"<html>gateway</html>")])
        with pytest.raises(httpx.HTTPStatusError):
            c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")

    def test_409_flag_present_but_falsy_is_not_a_refusal(self):
        c = _client_with([(409, {"ok": False, "flashnext_occupied": False})])
        with pytest.raises(httpx.HTTPStatusError):
            c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")

    def test_non_409_still_raises(self):
        c = _client_with([(500, {"ok": False, "flashnext_occupied": True})])
        with pytest.raises(httpx.HTTPStatusError):
            c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")

    def test_serving_still_returns_dict(self):
        c = _client_with([(200, {"status": "serving", "node": "gravitywell",
                                "work_id": "w1", "serve_axis": "flashnext"})])
        res = c.acquire("gravitywell", "w1", ttl_sec=120, reason="t",
                        accept_flashnext_seat=True)
        assert res["status"] == "serving"
        assert res["serve_axis"] == "flashnext"

    def test_transport_failure_still_raises_unreachable(self):
        class _Boom(httpx.BaseTransport):
            def handle_request(self, request):
                raise httpx.ConnectError("nope")

        c = DoormanClient(base_url="http://doorman.test")
        c._client = httpx.Client(base_url="http://doorman.test", transport=_Boom())
        with pytest.raises(DoormanUnreachable):
            c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")

    def test_other_endpoints_keep_raise_for_status(self):
        """named_refusals is acquire-only: a 409 on release() keeps raising."""
        c = _client_with([(409, {"ok": False, "flashnext_occupied": True})])
        with pytest.raises(httpx.HTTPStatusError):
            c.release("gravitywell", "w1")

    def test_accept_field_sent_only_when_true(self):
        cap: dict = {}
        c = _client_with([(200, {"status": "serving"})], capture=cap)
        c.acquire("gravitywell", "w1", ttl_sec=120, reason="t")
        import json as _json
        assert "accept_flashnext_seat" not in _json.loads(cap["body"])

        cap2: dict = {}
        c2 = _client_with([(200, {"status": "serving"})], capture=cap2)
        c2.acquire("gravitywell", "w1", ttl_sec=120, reason="t",
                   accept_flashnext_seat=True)
        assert _json.loads(cap2["body"])["accept_flashnext_seat"] is True

    def test_predicates_tolerate_non_dict(self):
        assert is_flashnext_occupied(None) is False
        assert is_creative_occupied("nope") is False


# ===========================================================================
# S2 — server: opt-in already-serving grant
# ===========================================================================

class TestS2Grant:
    def test_grant_registers_lease_without_a_wake(self):
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            ok = state.acquire_lease("w-grant", 300, "t", role="worker",
                                     accept_flashnext_seat=True)
        assert ok is True
        assert not g.wake_issued, "S2 must never issue wake-gravitywell"
        lease = state.leases["w-grant"]
        assert lease["serve_axis"] == "flashnext"
        assert lease["served_id"] == GW_FLASHNEXT_MODEL_ID
        # additive-only: the pre-existing keys are still there (I3)
        for k in ("acquired_at", "ttl_sec", "reason", "role", "lease_kind", "class"):
            assert k in lease

    def test_grant_places_a_hold(self):
        """_place_hold DOES apply — real work is in flight on the seat."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT), \
             patch.object(state, "_place_hold") as hold:
            state.acquire_lease("w-hold", 300, "t", role="worker",
                                accept_flashnext_seat=True)
        hold.assert_called_once()

    def test_fast_path_write_set_no_cached_serving_no_last_wake_at(self):
        """It was not a wake: no _cached_serving write, no last_wake_at — and the
        day-seat `serving` axis is untouched (I2)."""
        state = _make_state()
        state.last_wake_at = 111.0
        state._cached_serving = False
        with _Guard(state, FLASHNEXT_GRANT):
            state.acquire_lease("w-axis", 300, "t", role="worker",
                                accept_flashnext_seat=True)
        assert state.last_wake_at == 111.0
        assert state._cached_serving is False

    def test_exactly_one_probe_pair_per_grant(self):
        """DoD-1 probe economy: the grant consumes the guard's single fresh probe
        pair — one :30000 probe call, one :8081 read (the fast-path fallthrough)."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            state.acquire_lease("w-probe", 300, "t", role="worker",
                                accept_flashnext_seat=True)
        assert g.probe_calls == 1
        assert g.serving_calls == 1

    def test_non_opt_in_is_byte_identical_refusal(self):
        """Default false: the same canned pair still yields FLASHNEXT_OCCUPIED and
        registers no lease (today's behavior)."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            ok = state.acquire_lease("w-noopt", 300, "t", role="worker")
        assert ok is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert not g.wake_issued

    def test_supplied_dual_mode_keeps_the_refusal(self):
        """I1: `dual` is the GPU-0 27B wake the guard exists to block."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            ok = state.acquire_lease("w-dual", 300, "t", role="worker",
                                     accept_flashnext_seat=True, mode="dual")
        assert ok is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert not g.wake_issued

    def test_supplied_big_mode_keeps_the_refusal(self):
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            ok = state.acquire_lease("w-big", 300, "t", role="worker",
                                     accept_flashnext_seat=True, mode="big")
        assert ok is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert not g.wake_issued

    def test_mode_controller_keeps_the_uniform_refusal(self):
        """H-controller: the controller lease is a lever; no-mode controller grant
        would reinstate the window state D4 rules out."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT) as g:
            ok = state.acquire_lease("w-ctl", 300, "t", role="mode-controller",
                                     accept_flashnext_seat=True)
        assert ok is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert not g.wake_issued

    def test_up_unverified_canned_pair_refuses(self):
        """H-unverified: identity unverified (health-before-models mid-load) is
        never a grant — refusal is cheap, a grant to a squatter is not."""
        state = _make_state()
        with _Guard(state, UP_UNVERIFIED) as g:
            ok = state.acquire_lease("w-unver", 300, "t", role="worker",
                                     accept_flashnext_seat=True)
        assert ok is FLASHNEXT_OCCUPIED
        assert state.leases == {}
        assert not g.wake_issued

    def test_down_seat_still_wakes(self):
        """Regression: the opt-in field must not suppress a legitimate wake when
        no seat is resident."""
        state = _make_state()
        with _Guard(state, ("down", None, None, "ConnectionError")), \
             patch("agents_core.doorman_server.DOORMAN_DEFER_TO_CONTROLLER", False), \
             patch.object(state, "_resolve_cold_wake_posture", return_value="dual"), \
             patch.object(state, "_wake_dual", return_value=True) as wake:
            ok = state.acquire_lease("w-wake", 300, "t", role="worker",
                                     accept_flashnext_seat=True)
        assert ok is True
        wake.assert_called_once()
        assert "serve_axis" not in state.leases["w-wake"]

    def test_status_surfaces_serve_axis(self):
        """DoD-2(a) observation, hermetic: the additive field reaches /status via
        the **info spread while `serving` still reports the DAY-SEAT axis."""
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT):
            state.acquire_lease("w-status", 300, "t", role="worker",
                                principal="pm-dod", accept_flashnext_seat=True)
        snap = state.status_snapshot()
        assert snap["serving"] is False, "day-seat axis unchanged (I2)"
        leases = {l["work_id"]: l for l in snap["leases"]}
        assert leases["w-status"]["serve_axis"] == "flashnext"
        # drain-gate semantics unchanged (I3): the flashnext-axis inference lease
        # is still COUNTED — a box-level stop hurts flashnext inference too.
        assert snap["drain_count"] == 1

    def test_coordination_stay_exempt_with_serve_axis(self):
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT):
            state.acquire_lease("w-coord", 300, "t", role="worker",
                                principal="p", lease_kind="coordination",
                                accept_flashnext_seat=True)
        assert state.leases["w-coord"]["serve_axis"] == "flashnext"
        with state.lock:
            blockers = state._worker_lease_blockers(None)
        assert "w-coord" not in blockers

    def test_audit_line_emitted_at_registration(self):
        state = _make_state()
        with _Guard(state, FLASHNEXT_GRANT), \
             patch("agents_core.doorman_server.log") as mock_log:
            state.acquire_lease("w-audit", 300, "t", role="worker",
                                principal="council-delib-1",
                                accept_flashnext_seat=True)
        lines = [c for c in mock_log.info.call_args_list
                 if "flashnext-axis-lease-registered" in str(c)]
        assert len(lines) == 1
        rendered = str(lines[0])
        for needle in ("w-audit", "council-delib-1", "flashnext", GW_FLASHNEXT_MODEL_ID):
            assert needle in rendered


class TestS2Route:
    def _route(self, body: dict, seat_result):
        state_holder = {}

        real_init = _NodeState.__init__

        def _spy_init(self, *a, **kw):
            real_init(self, *a, **kw)
            state_holder["state"] = self

        with patch.object(_NodeState, "__init__", _spy_init), \
             patch.object(_NodeState, "_is_serving", return_value=False), \
             patch.object(_NodeState, "_is_creative_serving", return_value=False), \
             patch.object(_NodeState, "_probe_flashnext_seat",
                          return_value=seat_result), \
             patch("agents_core.doorman_server.subprocess.run") as mock_run:
            from fastapi.testclient import TestClient
            resp = TestClient(create_app()).post("/lease/acquire", json=body)
        return resp, mock_run

    def test_opt_in_grant_answers_serving_with_serve_axis(self):
        resp, mock_run = self._route(
            {"node": "gravitywell", "work_id": "pm-dod", "ttl_sec": 120,
             "reason": "dod", "role": "worker", "accept_flashnext_seat": True},
            FLASHNEXT_GRANT,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "serving"
        assert body["serve_axis"] == "flashnext"
        mock_run.assert_not_called()

    def test_non_opt_in_answers_the_today_409(self):
        resp, mock_run = self._route(
            {"node": "gravitywell", "work_id": "pm-dod", "ttl_sec": 120,
             "reason": "dod", "role": "worker"},
            FLASHNEXT_GRANT,
        )
        assert resp.status_code == 409
        assert resp.json() == {
            "ok": False,
            "flashnext_occupied": True,
            "reason": "flashnext-window-holding-gpu0",
        }
        mock_run.assert_not_called()

    def test_opt_in_false_is_omission(self):
        resp, _ = self._route(
            {"node": "gravitywell", "work_id": "pm-dod", "ttl_sec": 120,
             "reason": "dod", "role": "worker", "accept_flashnext_seat": False},
            FLASHNEXT_GRANT,
        )
        assert resp.status_code == 409

    def test_mode_controller_opt_in_still_409(self):
        resp, _ = self._route(
            {"node": "gravitywell", "work_id": "ctl", "ttl_sec": 120,
             "reason": "dod", "role": "mode-controller",
             "accept_flashnext_seat": True},
            FLASHNEXT_GRANT,
        )
        assert resp.status_code == 409


# ===========================================================================
# S3 (agents-core side) — named branches on the dict side
# ===========================================================================

from agents_core.gw_agent import (  # noqa: E402
    GW_REASON_CONTENDED,
    GW_REASON_FLASHNEXT_WINDOW,
    GW_REASON_SEAT_OCCUPIED,
    _acquire_refusal_reason,
    call_gw_agent,
)


class TestS3RefusalReasonMapping:
    def test_named_flags_map_to_distinct_tokens(self):
        assert _acquire_refusal_reason(
            {"ok": False, "flashnext_occupied": True,
             "reason": "flashnext-window-holding-gpu0"}
        ) == GW_REASON_FLASHNEXT_WINDOW
        assert _acquire_refusal_reason(
            {"ok": False, "creative_occupied": True}
        ) == GW_REASON_SEAT_OCCUPIED
        assert _acquire_refusal_reason({"ok": False, "contended": True}) == GW_REASON_CONTENDED

    def test_non_refusal_returns_none(self):
        assert _acquire_refusal_reason({"status": "serving"}) is None
        assert _acquire_refusal_reason({"status": "wake_failed"}) is None
        assert _acquire_refusal_reason(None) is None


class TestS3GwAgentDictSide:
    def _run(self, acquire_result, on_wake_fail="skip"):
        reason_out: list = []
        with patch("agents_core.doorman_client.DoormanClient") as dc, \
             patch("agents_core.gw_agent._acquire_with_defer_retry",
                   return_value=(acquire_result, False)), \
             patch("requests.post"):
            dc.return_value = MagicMock()
            result = call_gw_agent(
                prompt="Review.", reason_out=reason_out, timeout=10,
                on_wake_fail=on_wake_fail,
            )
        return result, reason_out

    def test_flashnext_window_named_not_flat(self):
        result, reasons = self._run(
            {"ok": False, "flashnext_occupied": True,
             "reason": "flashnext-window-holding-gpu0"})
        assert result is None
        assert reasons == [GW_REASON_FLASHNEXT_WINDOW]

    def test_creative_named_as_seat_occupied(self):
        result, reasons = self._run(
            {"ok": False, "creative_occupied": True,
             "reason": "creative-collider-holding-gpu"})
        assert result is None
        assert reasons == [GW_REASON_SEAT_OCCUPIED]

    def test_contended_named(self):
        result, reasons = self._run({"ok": False, "contended": True})
        assert result is None
        assert reasons == [GW_REASON_CONTENDED]

    def test_error_policy_raises_on_named_refusal(self):
        with pytest.raises(Exception):
            self._run({"ok": False, "flashnext_occupied": True},
                      on_wake_fail="error")

    def test_reviewer_acquire_opts_in(self):
        """The worker/reviewer acquire carries the S2 opt-in (the caller needs no
        wake when the seat is already serving)."""
        with patch("agents_core.doorman_client.DoormanClient") as dc, \
             patch("agents_core.gw_agent._acquire_with_defer_retry",
                   return_value=({"status": "serving"}, False)) as acq, \
             patch("requests.post") as post:
            dc.return_value = MagicMock()
            post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "ok"}}]},
                raise_for_status=MagicMock(),
                text="",
            )
            call_gw_agent(prompt="hi", timeout=10)
        assert acq.call_args.kwargs.get("accept_flashnext_seat") is True


class TestS3LlmDictSide:
    """llm.py admission + direct-dispatch read the DICT (not the exception) and
    NAME the state; the day-seat direct-dispatch caller stays NON-opt-in."""

    def _dc(self, res):
        instance = MagicMock()
        instance.acquire.return_value = res
        dc = MagicMock(return_value=instance)
        dc.is_deferred = lambda r: isinstance(r, dict) and r.get("status") == "deferred"
        dc.is_contended = lambda r: bool(isinstance(r, dict) and r.get("contended"))
        return dc, instance

    def test_direct_dispatch_names_flashnext_window(self):
        from agents_core.llm import call_operator

        dc, instance = self._dc(
            {"ok": False, "flashnext_occupied": True,
             "reason": "flashnext-window-holding-gpu0"})
        prov: list = []
        with patch("agents_core.doorman_client.DoormanClient", dc):
            out = call_operator(
                "gravitywell", "hello", on_wake_fail="skip", _provenance_out=prov,
            )
        assert out is None
        assert ("gw_flashnext_window", "gravitywell") in prov
        # Non-opt-in pin: a day-seat caller must not consume a lease it cannot dial.
        assert "accept_flashnext_seat" not in instance.acquire.call_args.kwargs

    def test_direct_dispatch_names_creative_seat(self):
        from agents_core.llm import call_operator

        dc, _ = self._dc({"ok": False, "creative_occupied": True})
        prov: list = []
        with patch("agents_core.doorman_client.DoormanClient", dc):
            out = call_operator(
                "gravitywell", "hello", on_wake_fail="skip", _provenance_out=prov,
            )
        assert out is None
        assert ("gw_seat_occupied", "gravitywell") in prov

    def test_admission_names_it_once_no_soft_retry_repay(self):
        """The measured win: a named refusal is not an acquire_soft_error, so the
        ticket settles on the FIRST response instead of re-paying the loop."""
        import os

        from agents_core.llm import call_operator

        dc, _ = self._dc(
            {"ok": False, "flashnext_occupied": True,
             "reason": "flashnext-window-holding-gpu0"})
        prov: list = []
        es = MagicMock()
        es.try_admit.return_value = True
        es.get.return_value = {"status": "claimed", "claim_owner": "p"}
        with patch.dict(os.environ, {"GW_ADMISSION_MODE": "enforce"}), \
             patch("agents_core.elevator.IS_MASTER", True), \
             patch("agents_core.elevator.ElevatorStore", return_value=es), \
             patch("agents_core.doorman_client.DoormanClient", dc):
            out = call_operator(
                "gravitywell", "hello", on_wake_fail="skip",
                principal="solo-group", _provenance_out=prov,
            )
        assert out is None
        assert ("gw_flashnext_window", "gravitywell") in prov
        # exactly one acquire call — no 6-probe soft-retry re-pay loop
        assert dc.return_value.acquire.call_count == 1
        es.fail.assert_called_once()
        assert es.fail.call_args.kwargs.get("reason") == GW_REASON_FLASHNEXT_WINDOW
        es.requeue.assert_not_called()
