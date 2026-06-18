"""Tests for agents_core.elevator — ElevatorStore and node-state composition."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents_core.elevator import ElevatorStore


def _mock_response(status_code=200, json_data=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data or {}
    return m


@pytest.fixture
def elevator_store(tmp_path):
    """Create a temporary ElevatorStore for testing."""
    db_path = tmp_path / "test_queue.db"
    store = ElevatorStore(db_path=db_path)
    yield store
    store.close()


def test_compose_gw_status_doorman_serving_ready_true(elevator_store):
    """Doorman at 127.0.0.1:8407 returns serving=true → serving_ready=True."""
    captured_urls = []

    def fake_get(url, timeout=None):
        captured_urls.append(url)
        # Flip-controller response (no serving_ready from this endpoint).
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        # Doorman response with serving=true.
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": True}}},
            )
        return _mock_response(status_code=404)

    with patch.object(ElevatorStore, "_compose_gw_status", wraps=elevator_store._compose_gw_status):
        with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
            result = elevator_store._compose_gw_status()

    assert result["serving_ready"] is True
    # Verify the doorman was queried at the correct (localhost) host.
    doorman_urls = [url for url in captured_urls if "8407" in url]
    assert len(doorman_urls) == 1
    assert "127.0.0.1:8407" in doorman_urls[0]


def test_compose_gw_status_doorman_serving_ready_false(elevator_store):
    """Doorman returns serving=false → serving_ready=False."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": False}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    assert result["serving_ready"] is False


def test_compose_gw_status_doorman_failure_graceful(elevator_store):
    """Doorman read failure (timeout/404) → serving_ready=None (graceful degrade)."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "big"})
        elif "8407" in url:
            # Simulate doorman timeout or failure.
            raise Exception("doorman unreachable")
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # On failure, serving_ready should remain None (graceful degrade).
    assert result["serving_ready"] is None
    # Mode should still be set from flip-controller.
    assert result["mode"] == "big"


def test_compose_gw_status_doorman_missing_key(elevator_store):
    """Doorman response missing nodes.gravitywell.serving → serving_ready=None."""
    def fake_get(url, timeout=None):
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "swarm"})
        elif "8407" in url:
            # Response is 200 but missing the serving key.
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # Missing key → serving_ready stays None.
    assert result["serving_ready"] is None
    assert result["mode"] == "swarm"


def test_compose_gw_status_flip_controller_untouched(elevator_store):
    """Flip-controller at 203.0.113.10:8408 remains unchanged."""
    captured_urls = []

    def fake_get(url, timeout=None):
        captured_urls.append(url)
        if "8408" in url:
            return _mock_response(status_code=200, json_data={"mode": "offline"})
        elif "8407" in url:
            return _mock_response(
                status_code=200,
                json_data={"nodes": {"gravitywell": {"serving": True}}},
            )
        return _mock_response(status_code=404)

    with patch("agents_core.elevator.httpx.get", side_effect=fake_get):
        result = elevator_store._compose_gw_status()

    # Verify flip-controller is still queried at Tailscale IP (NOT localhost).
    fc_urls = [url for url in captured_urls if "8408" in url]
    assert len(fc_urls) == 1
    assert "203.0.113.10:8408" in fc_urls[0]
    assert result["mode"] == "offline"
