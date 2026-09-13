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
        import requests

        class FakeResp:
            status_code = 200

            def json(self):
                # the flat top-level "serving" field of the doorman's
                # /status snapshot (the production probe reads this, not
                # a nodes-dict)
                return {"ok": True, "serving": True}

        monkeypatch.setattr(
            requests, "get", lambda *a, **k: FakeResp(), raising=True,
        )
        assert _doorman_probe_serving(
            "http://127.0.0.1:8407", timeout=1.0,
        ) is True

    def test_probe_seat_down_returns_false(self, tmp_path: Path, monkeypatch):
        """Doorman up + seat down -> False (skip the lease - the wake
        trap: no cold-wake from the claim path)."""
        import requests

        class FakeResp:
            status_code = 200

            def json(self):
                # the flat top-level "serving" field of the doorman's
                # /status snapshot (the production probe reads this, not
                # a nodes-dict)
                return {"ok": True, "serving": False}

        monkeypatch.setattr(
            requests, "get", lambda *a, **k: FakeResp(), raising=True,
        )
        assert _doorman_probe_serving(
            "http://127.0.0.1:8407", timeout=1.0,
        ) is False


class TestDoormanLeaseSeams:
    """The acquire/release seams (best-effort - never raise into the
    claim path)."""

    def test_acquire_ok(self, monkeypatch):
        import requests

        class FakeResp:
            status_code = 200

            def json(self):
                return {"ok": True, "status": "serving"}

        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResp(), raising=True,
        )
        assert _doorman_lease_acquire(
            "http://127.0.0.1:8407", "wid-test", ttl_sec=600,
            reason="claim-lease", timeout=1.0,
        ) is True

    def test_acquire_unreachable_returns_false(self):
        assert _doorman_lease_acquire(
            "http://127.0.0.1:1", "wid-test", ttl_sec=600,
            reason="claim-lease", timeout=0.5,
        ) is False

    def test_release_ok(self, monkeypatch):
        import requests

        class FakeResp:
            status_code = 200

            def json(self):
                return {"ok": True}

        monkeypatch.setattr(
            requests, "post", lambda *a, **k: FakeResp(), raising=True,
        )
        assert _doorman_lease_release(
            "http://127.0.0.1:8407", "wid-test", timeout=1.0,
        ) is True

    def test_release_unreachable_returns_false(self):
        assert _doorman_lease_release(
            "http://127.0.0.1:1", "wid-test", timeout=0.5,
        ) is False


class TestClaimLeaseIntegration:
    """The claim seam: _acquire_claim_lease (called from the runner's
    claim loop after a successful GW-backend claim)."""

    def test_gw_backend_serving_acquires_lease(self, tmp_path: Path, monkeypatch):
        """A claimed GW-backend task with the seat serving holds a lease
        (visible to the park decision)."""
        import agents_core.notify as notify_mod

        _write_spec(tmp_path / "spec", "t1", "http://127.0.0.1:8407")

        # the probe says serving
        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (acquired.append(wid), True)[1],
        )
        released = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_release",
            lambda base_url, wid, **k: (released.append(wid), True)[1],
        )

        # the runner's claim-loop seam (the D5 caller) - the scope gate
        # consumes the backend_url the claim() call site computed
        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407",
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        assert lease is True
        assert acquired == ["wid-t1"]

    def test_gw_backend_seat_down_no_lease_no_wake(self, tmp_path: Path, monkeypatch):
        """Seat down at claim -> no lease acquired AND no wake triggered
        (the probe returned False; the acquire path was not entered)."""
        _write_spec(tmp_path / "spec", "t1", "http://127.0.0.1:8407")

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: False,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (acquired.append(wid), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407",
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
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
            lambda base_url, wid, **k: (acquired.append(wid), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407",
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url="http://other-seat:9999",
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
            lambda base_url, wid, **k: (acquired.append(wid), True)[1],
        )

        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407",
            work_id="wid-t1",
            task_id="t1",
            spec_dir=tmp_path / "spec",
            backend_url=None,
        )
        assert lease is False
        assert acquired == []

    def test_release_on_exit_paths(self, tmp_path: Path, monkeypatch):
        """The success/failure/raise/requeue exit paths each release the
        lease (requeue re-acquires on the re-claim)."""
        _write_spec(tmp_path / "spec", "t1", "http://127.0.0.1:8407")

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        acquired = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: (acquired.append(wid), True)[1],
        )
        released = []
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_release",
            lambda base_url, wid, **k: (released.append(wid), True)[1],
        )

        # success path: acquire, run, release
        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        claude_queue._release_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        assert released == ["wid-t1"]

        # requeue path: release on requeue, re-acquire on re-claim
        released.clear()
        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        claude_queue._release_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        # re-claim: the lease is re-acquired (the requeued task is never
        # lease-less in the park window)
        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        assert lease is True
        assert len(acquired) == 3  # initial + re-claim x2 (the requeue
        # re-acquires on the re-claim)

    def test_acquire_failure_does_not_block_claim(self, tmp_path: Path, monkeypatch):
        """The lease is best-effort: an acquire failure does not block the
        claim (the run proceeds lease-less; the death-class signals cover
        the seat-down case)."""
        _write_spec(tmp_path / "spec", "t1", "http://127.0.0.1:8407")

        monkeypatch.setattr(
            claude_queue, "_doorman_probe_serving",
            lambda *a, **k: True,
        )
        monkeypatch.setattr(
            claude_queue, "_doorman_lease_acquire",
            lambda base_url, wid, **k: False,
        )

        lease = claude_queue._acquire_claim_lease(
            base_url="http://127.0.0.1:8407", work_id="wid-t1",
            task_id="t1", spec_dir=tmp_path / "spec",
            backend_url="http://127.0.0.1:8407",
        )
        assert lease is False  # best-effort: the claim is not blocked
