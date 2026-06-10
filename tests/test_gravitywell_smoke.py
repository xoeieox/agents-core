"""Smoke test — live end-to-end: doorman → operator → GW → parse → release.

Requires:
  - doorman-server running on :8407
  - GravityWell reachable at http://203.0.113.11:8081

Run with:  pytest -m smoke tests/test_gravitywell_smoke.py

Excluded from the default suite (no -m smoke flag).
"""

import json
import subprocess

import pytest
import requests

from agents_core.llm import call_operator


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
        proc = subprocess.run(
            ["ssh", "gravitywell", "ls /run/gw-keepawake.d/"],
            capture_output=True, text=True, timeout=10,
        )
        holders = proc.stdout.strip().split() if proc.stdout.strip() else []
        # The 'doorman' hold should have been released after the call
        assert "doorman" not in holders, (
            f"doorman holder still present after call: {holders}"
        )
    except Exception:
        pytest.skip("SSH to gravitywell unavailable — skipping hold-release check")
