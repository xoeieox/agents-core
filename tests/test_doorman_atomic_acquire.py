"""Unit tests for the atomic conditional worker-lease acquire (gw-admission-drain-gate-atomic-acquire-v0).

Verifies that acquire_lease(require_drain_clear=True) is race-free:
  - N threads with distinct principals → exactly 1 True, the rest CONTENDED
  - Same-principal ride-along is always allowed (require_drain_clear=False)
  - Ghost principal counts as contending (GHOST_PRINCIPAL never excluded)

All tests use _NodeState directly with ensure_serving mocked to True so no
subprocesses or SSH calls are made.
"""

from __future__ import annotations

import threading
import time
import uuid
from unittest.mock import patch

import pytest

from agents_core.doorman_server import (
    CONTENDED,
    DEFERRED,
    GHOST_PRINCIPAL,
    _NodeState,
)


@pytest.fixture
def node():
    """In-process _NodeState with ensure_serving always returning True."""
    state = _NodeState(gw_url="http://mock.internal/", node_name="gravitywell")
    with patch.object(state, "_is_serving", return_value=True), \
         patch("subprocess.run"):
        yield state


# ---------------------------------------------------------------------------
# Concurrency: N distinct-principal threads → exactly one succeeds
# ---------------------------------------------------------------------------

class TestAtomicAcquireConcurrency:
    """AC1 / AC3 / AC4: concurrent distinct-principal acquire under require_drain_clear."""

    @pytest.mark.parametrize("n_threads", [2, 5, 8, 20])
    def test_exactly_one_true_rest_contended(self, node, n_threads):
        """N threads with distinct principals → exactly 1 True, rest CONTENDED.

        Threads line up on a Barrier and rush acquire_lease() simultaneously (an
        "uncoordinated rush") — this is the real concurrency exercise for the
        drain-gate TOCTOU that gw-admission-drain-gate-atomic-acquire-v0 closed:
        it must be possible for one distinct-principal worker to observe drain=0
        while another concurrently registers, and exactly one may win. The n=20
        case is the Barrier-guaranteed maximum-collision cohort (AC5); the smaller
        sizes are cheap sanity checks of the same property.

        No thread holds node.lock externally — acquire_lease() takes it internally
        exactly once per call (agents-core-doorman-acquire-lease-nonreentrant-deadlock-v0);
        holding it here would self-deadlock against that internal acquisition.

        All leases are held until every thread has reported its result so that
        the test is not sensitive to release-before-acquire races in the teardown.
        """
        results = [None] * n_threads
        work_ids = [f"wid-{i}-{uuid.uuid4().hex[:4]}" for i in range(n_threads)]
        barrier = threading.Barrier(n_threads)

        def _worker(idx):
            principal = f"group-{idx}-{uuid.uuid4().hex[:4]}"
            barrier.wait()  # all threads start simultaneously
            ok = node.acquire_lease(
                work_ids[idx], 60, "test", role="worker",
                principal=principal, require_drain_clear=True,
            )
            results[idx] = ok
            # Do NOT release here — hold the lease until all threads finish
            # so later threads still see the active lease.

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Release all leases after collecting results
        with node.lock:
            for wid in work_ids:
                node.leases.pop(wid, None)

        true_count = sum(1 for r in results if r is True)
        contended_count = sum(1 for r in results if r is CONTENDED)

        assert true_count == 1, (
            f"exactly 1 thread must succeed; got true_count={true_count} "
            f"results={results}"
        )
        assert contended_count == n_threads - 1, (
            f"rest must be CONTENDED; got contended_count={contended_count} "
            f"results={results}"
        )

    def test_second_distinct_group_contended_while_first_holds(self, node):
        """While group A holds a lease, group B acquire with require_drain_clear → CONTENDED."""
        wid_a = "wid-a"
        wid_b = "wid-b"
        principal_a = "group-alpha"
        principal_b = "group-beta"

        # Group A acquires first (no drain constraint needed — it's the first)
        ok_a = node.acquire_lease(wid_a, 60, "test", role="worker", principal=principal_a)
        assert ok_a is True

        # Group B tries atomic acquire → should see group A's lease → CONTENDED
        ok_b = node.acquire_lease(
            wid_b, 60, "test", role="worker", principal=principal_b,
            require_drain_clear=True,
        )
        assert ok_b is CONTENDED, f"group B must be CONTENDED while group A holds; got {ok_b}"

        # Cleanup
        with node.lock:
            node.leases.pop(wid_a, None)

    def test_second_group_succeeds_after_first_releases(self, node):
        """Group B succeeds once group A releases its lease."""
        wid_a = "wid-a-release"
        wid_b = "wid-b-release"

        ok_a = node.acquire_lease(wid_a, 60, "test", role="worker", principal="alpha")
        assert ok_a is True

        with node.lock:
            node.leases.pop(wid_a, None)

        # Now group B acquires with require_drain_clear=True → drain is clear → True
        ok_b = node.acquire_lease(
            wid_b, 60, "test", role="worker", principal="beta",
            require_drain_clear=True,
        )
        assert ok_b is True, f"group B must succeed after group A releases; got {ok_b}"

        with node.lock:
            node.leases.pop(wid_b, None)


# ---------------------------------------------------------------------------
# Same-principal ride-alongs are unaffected
# ---------------------------------------------------------------------------

class TestRideAlongUnaffected:
    """Same-principal callers share the group slot and must not be blocked."""

    def test_same_principal_without_flag_always_allowed(self, node):
        """Ride-alongs (require_drain_clear=False) always succeed regardless of active leases."""
        P = "shared-group"
        wid1 = "wid-shared-1"
        wid2 = "wid-shared-2"

        ok1 = node.acquire_lease(wid1, 60, "test", role="worker", principal=P)
        assert ok1 is True

        # Ride-along: same principal, no drain flag
        ok2 = node.acquire_lease(wid2, 60, "test", role="worker", principal=P)
        assert ok2 is True, "same-principal ride-along must succeed without require_drain_clear"

        # Cleanup
        with node.lock:
            node.leases.pop(wid1, None)
            node.leases.pop(wid2, None)

    def test_distinct_principal_no_flag_is_not_gated(self, node):
        """Without require_drain_clear, distinct-principal acquire always proceeds (old behavior)."""
        wid_a = "wid-old-a"
        wid_b = "wid-old-b"

        ok_a = node.acquire_lease(wid_a, 60, "test", role="worker", principal="group-x")
        assert ok_a is True

        # Old-style acquire (no require_drain_clear) → not gated (byte-identical to before)
        ok_b = node.acquire_lease(wid_b, 60, "test", role="worker", principal="group-y")
        assert ok_b is True, "require_drain_clear=False must preserve old behavior"

        with node.lock:
            node.leases.pop(wid_a, None)
            node.leases.pop(wid_b, None)


# ---------------------------------------------------------------------------
# Ghost safety (AC6)
# ---------------------------------------------------------------------------

class TestGhostSafety:
    """Ghost leases always count as contending; critical log fires (AC6)."""

    def test_ghost_lease_contends_with_fresh_group(self, node, caplog):
        """A ghost worker lease causes CONTENDED for a named-principal group."""
        import logging
        ghost_wid = f"ghost-{uuid.uuid4().hex[:6]}"
        fresh_wid = f"fresh-{uuid.uuid4().hex[:6]}"
        fresh_principal = "fresh-group"

        # Register ghost lease directly (no principal)
        with node.lock:
            node.leases[ghost_wid] = {
                "acquired_at": time.time(),
                "ttl_sec": 60,
                "reason": "ghost-test",
                "role": "worker",
                "principal": GHOST_PRINCIPAL,
            }

        # Fresh group acquires with require_drain_clear=True → ghost blocks it
        with caplog.at_level(logging.CRITICAL, logger="doorman-server"):
            ok = node.acquire_lease(
                fresh_wid, 60, "test", role="worker",
                principal=fresh_principal, require_drain_clear=True,
            )

        assert ok is CONTENDED, f"ghost must cause CONTENDED; got {ok}"
        assert any(
            "ghost_lease_counted" in rec.message
            for rec in caplog.records
            if rec.levelno >= logging.CRITICAL
        ), "critical log must fire when ghost counted in drain decision"

        with node.lock:
            node.leases.pop(ghost_wid, None)

    def test_non_worker_lease_not_counted(self, node):
        """Non-worker leases (e.g. mode-controller) do not count as contending."""
        ctrl_wid = "ctrl-lease"
        worker_wid = f"worker-{uuid.uuid4().hex[:6]}"

        with node.lock:
            node.leases[ctrl_wid] = {
                "acquired_at": time.time(),
                "ttl_sec": 60,
                "reason": "keepawake",
                "role": "mode-controller",
                "principal": "flip-controller",
            }

        ok = node.acquire_lease(
            worker_wid, 60, "test", role="worker",
            principal="some-group", require_drain_clear=True,
        )

        assert ok is True, f"non-worker lease must not gate drain check; got {ok}"

        with node.lock:
            node.leases.pop(ctrl_wid, None)
            node.leases.pop(worker_wid, None)


# ---------------------------------------------------------------------------
# Endpoint integration: CONTENDED propagates through /lease/acquire
# ---------------------------------------------------------------------------

class TestEndpointContended:
    """require_drain_clear threads through the HTTP endpoint and client."""

    def test_endpoint_returns_contended_when_gated(self):
        """POST /lease/acquire with require_drain_clear=True returns {"ok":False,"contended":True}."""
        from agents_core.doorman_server import create_app
        from fastapi.testclient import TestClient

        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch("subprocess.run"):
            app = create_app(gw_url="http://mock.internal/")
            http = TestClient(app, raise_server_exceptions=True)

            # First group acquires normally
            r1 = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "wid-group-a",
                "ttl_sec": 60,
                "reason": "test",
                "role": "worker",
                "principal": "group-a",
            })
            assert r1.status_code == 200
            assert r1.json().get("status") == "serving"
            assert r1.json().get("drain_cleared") is None  # no drain flag was sent

            # Second group tries atomic acquire → should be CONTENDED
            r2 = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "wid-group-b",
                "ttl_sec": 60,
                "reason": "test",
                "role": "worker",
                "principal": "group-b",
                "require_drain_clear": True,
            })
            assert r2.status_code == 200
            body = r2.json()
            assert body.get("ok") is False, f"expected contended response; got {body}"
            assert body.get("contended") is True, f"expected contended=True; got {body}"

            # Cleanup
            http.post("/lease/release", json={"node": "gravitywell", "work_id": "wid-group-a"})

    def test_endpoint_returns_drain_cleared_on_success(self):
        """POST /lease/acquire with require_drain_clear=True and empty drain returns drain_cleared=True."""
        from agents_core.doorman_server import create_app
        from fastapi.testclient import TestClient

        with patch("agents_core.doorman_server._start_refresh_thread"), \
             patch.object(_NodeState, "ensure_serving", return_value=True), \
             patch("subprocess.run"):
            app = create_app(gw_url="http://mock.internal/")
            http = TestClient(app, raise_server_exceptions=True)

            r = http.post("/lease/acquire", json={
                "node": "gravitywell",
                "work_id": "wid-solo",
                "ttl_sec": 60,
                "reason": "test",
                "role": "worker",
                "principal": "solo-group",
                "require_drain_clear": True,
            })
            assert r.status_code == 200
            body = r.json()
            assert body.get("status") == "serving", f"expected serving; got {body}"
            assert body.get("drain_cleared") is True, \
                f"drain_cleared must be True when drain check honored; got {body}"

            http.post("/lease/release", json={"node": "gravitywell", "work_id": "wid-solo"})
