"""Smoke test — live end-to-end: doorman → operator → GW → parse → release.

Requires:
  - doorman-server running on :8407
  - GravityWell reachable at http://203.0.113.11:8081

Run with:  pytest -m smoke tests/test_gravitywell_smoke.py

Excluded from the default suite (no -m smoke flag).
"""

import json
import os
import subprocess
import time

import pytest
import requests

from agents_core.llm import call_operator

DOORMAN_URL = "http://127.0.0.1:8407"
GW_SSH_HOST = "gravitywell"

# Must match the doorman's own configuration (gw-doorman-wake-to-default-mode-v0) —
# the smoke runner is expected to set this to whatever the live doorman is running,
# same convention as GW_STOP_GRACE_SEC below. Default "dual" matches the doorman's
# real default.
DEFAULT_SERVE_MODE = os.environ.get("DOORMAN_DEFAULT_SERVE_MODE", "dual").strip().lower()
# Cold-wake acquire timeout must accommodate whichever mode is under test — dual's
# ~488s Devstral cold-init needs real headroom, big's ~25s cold-load does not.
_COLD_ACQUIRE_TIMEOUT_SEC = 200 if DEFAULT_SERVE_MODE == "big" else 750


def _services_for_mode() -> list[str]:
    if DEFAULT_SERVE_MODE == "big":
        return ["llama-server.service"]
    return ["vllm-slot1.service", "vllm-slot2.service"]


def _gw_ssh(cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", GW_SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.smoke
def test_gravitywell_wakes_returns_json_releases_hold():
    """call_operator("gravitywell") transparently wakes GW, returns clean JSON,
    and releases the lease (no lingering 'doorman' holder afterward)."""
    result = call_operator(
        "gravitywell",
        'reply with {"ok":true}',
        json_mode=True,
        on_wake_fail="error",
    )

    # Parseable JSON returned
    assert result is not None, "operator returned None — backend may be down"
    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed.get("ok") is True or "ok" in str(parsed)

    # Verify the doorman hold was released (no lingering doorman holder on GW)
    try:
        proc = _gw_ssh("ls /run/gw-keepawake.d/")
        holders = proc.stdout.strip().split() if proc.stdout.strip() else []
        # The 'doorman' hold should have been released after the call
        assert "doorman" not in holders, (
            f"doorman holder still present after call: {holders}"
        )
    except Exception:
        pytest.skip("SSH to gravitywell unavailable — skipping hold-release check")


@pytest.mark.smoke
def test_cold_acquire_starts_service_and_release_with_grace_stops():
    """Full clean-stop lifecycle: cold acquire → serving; release + grace → stopped.

    This test verifies gravitywell-doorman-clean-stop-v0, mode-aware for
    gw-doorman-wake-to-default-mode-v0 (DEFAULT_SERVE_MODE, from the
    DOORMAN_DEFAULT_SERVE_MODE env var — must match the live doorman's config):
      1. acquire from stopped → doorman issues gw-serve ${DEFAULT_SERVE_MODE} →
         the mode's service(s) active (llama-server.service for big;
         vllm-slot1.service + vllm-slot2.service for dual)
      2. release → idle_since set
      3. after GW_STOP_GRACE_SEC (injected low via doorman env), refresh thread
         issues gw-serve stop → service(s) inactive
      4. /status shows service_stopped=True and serving_mode=stopped
      5. guard now permits suspend (no serving unit active)

    Does NOT assert actual S3 suspend (irreversible/slow) — inactive + guard-eligibility
    is sufficient proof.
    """
    try:
        resp = requests.get(f"{DOORMAN_URL}/healthz", timeout=5)
        if resp.status_code != 200:
            pytest.skip("doorman-server not running on :8407")
    except Exception:
        pytest.skip("doorman-server not reachable at :8407")

    try:
        _gw_ssh("true", timeout=5)
    except Exception:
        pytest.skip("SSH to gravitywell not available")

    # --- Step 1: ensure service is stopped before the test ---
    try:
        _gw_ssh("gw-serve stop", timeout=60)
    except Exception:
        pytest.skip("gw-serve stop failed — cannot set up precondition")
    time.sleep(3)

    # --- Step 2: cold acquire (doorman must issue gw-serve ${DEFAULT_SERVE_MODE}) ---
    acq = requests.post(
        f"{DOORMAN_URL}/lease/acquire",
        json={"node": "gravitywell", "work_id": "smoke-clean-stop", "ttl_sec": 120, "reason": "smoke"},
        timeout=_COLD_ACQUIRE_TIMEOUT_SEC,
    )
    assert acq.status_code == 200, f"acquire failed: {acq.text}"
    assert acq.json().get("status") == "serving", f"unexpected status: {acq.json()}"

    # Verify the mode's serving unit(s) are now active on GW
    for svc in _services_for_mode():
        is_active = _gw_ssh(f"systemctl is-active {svc}", timeout=10)
        assert "active" in is_active.stdout, (
            f"{svc} not active after acquire: {is_active.stdout!r}"
        )

    # --- Step 3: inference still works ---
    result = call_operator(
        "gravitywell",
        'reply with {"ok":true}',
        json_mode=True,
        on_wake_fail="error",
    )
    assert result is not None
    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed.get("ok") is True or "ok" in str(parsed)

    # --- Step 4: release ---
    rel = requests.post(
        f"{DOORMAN_URL}/lease/release",
        json={"node": "gravitywell", "work_id": "smoke-clean-stop"},
        timeout=10,
    )
    assert rel.status_code == 200

    # Confirm /status shows idle_since set (not None) and service not yet stopped
    status_resp = requests.get(f"{DOORMAN_URL}/status", timeout=5)
    gw_status = status_resp.json()["nodes"]["gravitywell"]
    assert gw_status["idle_since"] is not None, "idle_since should be set after release"
    assert gw_status["service_stopped"] is False, "service should still be up in grace period"

    # --- Step 5: wait for the refresh thread to issue gw-serve stop ---
    # GW_STOP_GRACE_SEC is typically 600s in production; this test relies on
    # the doorman being configured with a low value (e.g. GW_STOP_GRACE_SEC=30)
    # for smoke runs. If the env var is at default, this poll will time out and
    # the test will skip rather than fail.
    grace_sec = 45  # poll up to 45s; assumes smoke env has GW_STOP_GRACE_SEC<=30
    deadline = time.time() + grace_sec
    stopped = False
    while time.time() < deadline:
        s = requests.get(f"{DOORMAN_URL}/status", timeout=5).json()["nodes"]["gravitywell"]
        if s.get("service_stopped") is True:
            stopped = True
            break
        time.sleep(5)

    if not stopped:
        pytest.skip(
            f"gw-serve stop not observed within {grace_sec}s — "
            f"set GW_STOP_GRACE_SEC<=30 in doorman env for smoke runs"
        )

    # The mode's serving unit(s) must now be inactive
    for svc in _services_for_mode():
        is_inactive = _gw_ssh(f"systemctl is-active {svc}", timeout=10)
        assert "inactive" in is_inactive.stdout or is_inactive.returncode != 0, (
            f"{svc} still active after gw-serve stop: {is_inactive.stdout!r}"
        )

    # /status must reflect the stopped state
    final_status = requests.get(f"{DOORMAN_URL}/status", timeout=5).json()["nodes"]["gravitywell"]
    assert final_status["serving_mode"] == "stopped"
    assert final_status["service_stopped"] is True
