# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D5 (attestation-contract-v0, leg 1): claim-time doorman lease.

The 2026-09-08 cohort: three claimed tasks died `gw_not_serving` - the
seat parked under claimed work because the queue run's lease is acquired
at its first LLM call (`ensure_serving`), not at claim. D5: on a
successful GW-backend claim, best-effort acquire a doorman lease for the
run (preceded by a cheap serving probe - the wake trap: a blind acquire
would cold-wake a down seat); release it on every run exit path.

The contract: while a GW-backend task is claimed-and-in-flight and the
seat is serving, the park decision sees a non-empty lease set for the
full claim -> first-call window; no lease is acquired when the seat is
down at claim (and no wake is triggered); non-GW-backend tasks acquire
no lease.

PRODUCTION SHAPE (cycle-2 reviewer finding): the task's backend_url is
the GW SEAT LLM endpoint (e.g. http://127.0.0.1:8081 - the doorman's
configured gw_url, the GW_URL env the doorman probes for its /status
"serving" view and agents_core.llm uses for its LLM calls), while the
doorman's own base (DOORMAN_SERVER, e.g. http://127.0.0.1:8407) is the
port the doorman serves the /lease/* endpoints on. The two are DIFFERENT
URLs: the scope gate compares backend_url against the gw_url anchor
(the task's own backend_url), NOT against the doorman's own port. The
rev-1 tests masked the inverted gate by passing backend_url ==
base_url (both :8407) - they must use the production shape.

These tests drive the probe + acquire/release seams against a mock
doorman (no LLM, no real doorman - the HTTP seams and the exit-path
matrix are the unit under test).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core import claude_queue
from agents_core.claude_queue import (
    _doorman_probe_serving,
    _doorman_lease_acquire,
    _doorman_lease_release,
)

# The production shape: the GW seat endpoint (the task's backend_url -
# the doorman's configured gw_url) and the doorman's own base
# (DOORMAN_SERVER - the port the doorman serves /lease/* on).
GW_SEAT_URL = "http://127.0.0.1:8081"
DOORMAN_BASE_URL = "http://127.0.0.1:8407"


def _write_spec(spec_dir: Path, task_id: str, backend_url: str | None) -> None:
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec = {"task_id": task_id, "prompt": "p"}
    if backend_url:
        spec["backend_url"] = backend_url
    (spec_dir / f"{task_id}.json").write_text(json.dumps(spec))


class TestDoormanProbe:
    """The serving probe (the wake trap): reads the doorman's view
    (liveness + seat state), NOT the GW seat directly."""

    def test_probe_unreachable_returns_none(self):
        """Doorman unreachable -> None (skip the lease; the first LLM
        call's existing ensure_serving wake path handles that case)."""
        assert _doorman_probe_serving(
            "http://127.0.0.1:1", timeout=0.5,
        ) is None

    def test_probe_serving_true(self, tmp_path: Path, monkeypatch):
        """Doorman up + seat serving -> True (acquire the lease)."""
        import agents_core.doorman_client as dc

        class FakeClient:
            def __init__(self, base_url=None, timeout=None):
                pass

            def healthz(self):
                return {"ok": True}

            def status(self):
                # the flat top-level "serving" field of the doorman's
                # /status snapshot (the production probe reads this, not
                # a nodes-dict)
                return {"ok": True, "serving": True}

            def close(self):
                pass

        monkeypatch.setattr(dc, "DoormanClient", FakeClient)
        assert _doorman_probe_serving(
            DOORMAN_BASE_URL, timeout=1.0,
        ) is True

    def test_probe_seat_down_returns_false(self, tmp_path: Path, monkeypatch):
        """Doorman up + seat down -> False (skip the lease - the wake
        trap: no cold-wake from the claim path)."""
        import agents_core.doorman_client as dc

        class FakeClient:
            def __init__(self, base_url=None, timeout=None):
                pass

            def healthz(self):
                return {"ok": True}

            def status(self):
                # the flat top-level "serving" field of the doorman's
                # /status snapshot (the production probe reads this, not
                # a nodes-dict)
                return {"ok": True, "serving": False}

            def close(self):
                pass

        monkeypatch.setattr(dc, "DoormanClient", FakeClient)
        assert _doorman_probe_serving(
            DOORMAN_BASE_URL, timeout=1.0,
        ) is False

    def test_probe_uses_doorman_client_not_requests(self, tmp_path: Path, monkeypatch):
        """The probe rides the existing DoormanClient (httpx) - no
        second HTTP library in the claim path (cycle-2 reviewer
        finding: the rev-1 probe imported requests while
        doorman_client.py uses httpx)."""
        import agents_core.doorman_client as dc

        seen = []

        class FakeClient:
            def __init__(self, base_url=None, timeout=None):
                seen.append(base_url)

            def healthz(self):
                return {"ok": True}

            def status(self):
                return {"serving": True}

            def close(self):
                pass

        monkeypatch.setattr(dc, "DoormanClient", FakeClient)
        assert _doorman_probe_serving(DOORMAN_BASE_URL, timeout=1.0) is True
        # the probe pointed at the doorman's OWN base (not the GW seat)
        assert seen == [DOORMAN_BASE_URL]


class TestDoormanLeaseSeams:
    """The acquire/release seams (best-effort - never raise into the
    claim path)."""

    def test_acquire_ok(self, monkeypatch):
        import agents_core.doorman_client as dc

        class FakeClient:
            def __init__(self, base_url=None, timeout=None):
                pass

            def acquire(self, node, work_id, ttl_sec=None, reason=None,
                        role=None, **kw):
                return {"status": "serving"}

            def close(self):
                pass

        monkeypatch.setattr(dc, "DoormanClient", FakeClient)
        assert _doorman_lease_acquire(
            DOORMAN_BASE_URL, "wid-test", ttl_sec=600,
            reason="claim-lease", timeout=1.0,
        ) is True

    def test_acquire_unreachable_returns_false(self):
        assert _doorman_lease_acquire(
            "http://127.0.0.1:1", "wid-test", ttl_sec=600,
            reason="claim-lease", timeout=0.5,
        ) is False

    def test_release_ok(self, monkeypatch):
        import agents_core.doorman_client as dc

        class FakeClient:
            def __init__(self, base_url=None, timeout=None):
                pass

            def release(self, node, work_id):
                return None

            def close(self):
                pass

        monkeypatch.setattr(dc, "DoormanClient", FakeClient)
        assert _doorman_lease_release(
            DOORMAN_BASE_URL, "wid-test", timeout=1.0,
        ) is True

    def test_release_unreachable_returns_false(self):
        assert _doorman_lease_release(
            "http://127.0.0.1:1", "wid-test", timeout=0.5,
        ) is False


class TestClaimLeaseIntegration:
    """The claim seam: _acquire_claim_lease (called from the runner's
    claim loop after a successful GW-backend claim).

    Production shape: base_url = the doorman's own port (:8407),
    backend_url = the GW seat endpoint (:8081) - the scope gate's
    gw_url anchor is the task's backend_url, NOT the doorman base.
    """

    def test_gw_backend_serving_acquires_lease(self, tmp_path: Path, monkeypatch):
        """A claimed GW-backend task with the seat serving holds a lease
        (visible to the park decision). The acquire rides the DOORMAN
        base (:8407), not the GW seat (:8081)."""
        _write_spec(tmp_path / "spec", "t1", GW_SEAT_URL)

        # the probe says serving
        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )
        released = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_release",
            lambda base_url, wid, **k: (released.append(wid), True)[1],
        )

        # the runner's claim-loop seam (the D5 caller) - the scope gate
        # consumes the backend_url the claim() call site computed
        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL,
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL,
            gw_url=GW_SEAT_URL,
        )
        assert lease is True
        # the lease was acquired against the DOORMAN base, not the GW
        # seat (the /lease/* endpoints live on the doorman's port)
        assert acquired == [(DOORMAN_BASE_URL, "wid-t1")]

    def test_gw_backend_seat_down_no_lease_no_wake(self, tmp_path: Path, monkeypatch):
        """Seat down at claim -> no lease acquired AND no wake triggered
        (the probe returned False; the acquire path was not entered)."""
        _write_spec(tmp_path / "spec", "t1", GW_SEAT_URL)

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: False,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL,
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL,
            gw_url=GW_SEAT_URL,
        )
        assert lease is False
        assert acquired == []  # the acquire path was not entered (no wake)

    def test_non_gw_backend_no_lease(self, tmp_path: Path, monkeypatch):
        """A non-GW-backend task -> no lease (the seam is scoped to
        GW-backend tasks via _task_backend_url)."""
        _write_spec(tmp_path / "spec", "t1", "http://other-seat:9999")

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL,
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url="http://other-seat:9999",
            gw_url=GW_SEAT_URL,
        )
        assert lease is False
        assert acquired == []

    def test_no_backend_url_no_lease(self, tmp_path: Path, monkeypatch):
        """No backend_url (fail-open to None) -> no lease."""
        _write_spec(tmp_path / "spec", "t1", None)

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL,
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url=None,
            gw_url=GW_SEAT_URL,
        )
        assert lease is False
        assert acquired == []

    def test_gate_is_not_the_doorman_base(self, tmp_path: Path, monkeypatch):
        """The cycle-2 regression: the scope gate must compare the task's
        backend_url (the GW seat, :8081) against the gw_url anchor, NOT
        against the doorman's own port (:8407). With the rev-1 inverted
        gate (comparing against base_url) this production-shape call
        would return False and the lease would never acquire."""
        _write_spec(tmp_path / "spec", "t1", GW_SEAT_URL)

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )

        # production shape: backend_url != base_url (the GW seat is NOT
        # the doorman's port)
        assert GW_SEAT_URL != DOORMAN_BASE_URL
        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL,
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL,
            gw_url=GW_SEAT_URL,
        )
        assert lease is True
        assert acquired == [(DOORMAN_BASE_URL, "wid-t1")]

    def test_release_on_exit_paths(self, tmp_path: Path, monkeypatch):
        """The success/failure/raise/requeue exit paths each release the
        lease (requeue re-acquires on the re-claim)."""
        _write_spec(tmp_path / "spec", "t1", GW_SEAT_URL)

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (
                acquired.append((base_url, wid)), True)[1],
        )
        released = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_release",
            lambda base_url, wid, **k: (
                released.append((base_url, wid)), True)[1],
        )

        # success path: acquire, run, release
        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        claude_queue._release_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        assert released == [(DOORMAN_BASE_URL, "wid-t1")]

        # requeue path: release on requeue, re-acquire on re-claim
        released.clear()
        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        claude_queue._release_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        # re-claim: the lease is re-acquired (the requeued task is never
        # lease-less in the park window)
        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        assert lease is True
        assert len(acquired) == 3  # initial + re-claim x2 (the requeue
        # re-acquires on the re-claim)

    def test_acquire_failure_does_not_block_claim(self, tmp_path: Path, monkeypatch):
        """The lease is best-effort: an acquire failure does not block the
        claim (the run proceeds lease-less; the death-class signals cover
        the seat-down case)."""
        _write_spec(tmp_path / "spec", "t1", GW_SEAT_URL)

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: False,
        )

        lease = claude_queue._acquire_claim_lease(
            base_url=DOORMAN_BASE_URL, work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url=GW_SEAT_URL, gw_url=GW_SEAT_URL,
        )
        assert lease is False  # best-effort: the claim is not blocked


class TestClaimLeaseClaimLoop:
    """The claim() call site: a successful GW-backend claim acquires the
    lease against the doorman base with the task's backend_url as the
    gw_url scope anchor (the production shape)."""

    def _queue(self, tmp_path: Path, task_id: str, backend_url: str | None):
        from agents_core.claude_queue import ClaudeQueue

        q = ClaudeQueue(queue_dir=tmp_path / "queue")
        _write_spec(tmp_path / "spec", task_id, backend_url)
        q.submit(
            {
                "task_type": "shaped",
                "id": task_id,
                "payload": {
                    "spec_path": str(tmp_path / "spec" / f"{task_id}.json"),
                    "_ignore_intention_registry": True,
                },
            },
        )
        return q

    def test_claim_gw_backend_acquires_lease_production_shape(
        self, tmp_path: Path, monkeypatch,
    ):
        """A successful GW-backend claim acquires the lease with the
        production shape: base_url = the doorman's own port,
        backend_url = the GW seat endpoint (the gw_url anchor)."""
        q = self._queue(tmp_path, "t1", GW_SEAT_URL)

        monkeypatch.setenv("DOORMAN_SERVER", DOORMAN_BASE_URL)
        calls = []
        monkeypatch.setattr(
            claude_queue, "_acquire_claim_lease",
            lambda **kw: (calls.append(kw), True)[1],
        )
        task = q.claim()
        assert task is not None
        assert task["id"] == "t1"
        assert len(calls) == 1
        kw = calls[0]
        # the lease rides the doorman's own port; the scope anchor is the
        # GW seat endpoint (the task's backend_url)
        assert kw["base_url"] == DOORMAN_BASE_URL
        assert kw["backend_url"] == GW_SEAT_URL
        assert kw["gw_url"] == GW_SEAT_URL
        assert kw["work_id"] == "t1"

    def test_claim_non_gw_backend_no_lease(self, tmp_path: Path, monkeypatch):
        """A non-GW-backend claim (other seat) -> the seam is CALLED but
        its scope gate refuses (no probe, no acquire). The claim() call
        site cannot know the doorman's gw_url (it is the doorman's
        configuration, not the task's), so it routes every backend_url
        through the seam and the seam's scope gate does the scoping -
        the non-GW refusal is asserted here, and the production-shape
        scope-gate refusal is asserted in
        TestClaimLeaseIntegration.test_non_gw_backend_no_lease."""
        q = self._queue(tmp_path, "t1", "http://other-seat:9999")

        monkeypatch.setenv("DOORMAN_SERVER", DOORMAN_BASE_URL)
        calls = []
        monkeypatch.setattr(
            claude_queue, "_acquire_claim_lease",
            lambda **kw: (calls.append(kw), False)[1],
        )
        task = q.claim()
        assert task is not None
        # the seam was called (the scope gate lives in the seam, not the
        # call site) and it refused (False - no probe, no acquire)
        assert len(calls) == 1
        assert calls[0]["backend_url"] == "http://other-seat:9999"

    def test_claim_no_backend_url_no_lease(self, tmp_path: Path, monkeypatch):
        """No backend_url in the spec (fail-open to None) -> no lease
        call."""
        q = self._queue(tmp_path, "t1", None)

        monkeypatch.setenv("DOORMAN_SERVER", DOORMAN_BASE_URL)
        calls = []
        monkeypatch.setattr(
            claude_queue, "_acquire_claim_lease",
            lambda **kw: (calls.append(kw), True)[1],
        )
        task = q.claim()
        assert task is not None
        assert calls == []
