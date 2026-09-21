"""Tests for RadioOpDriver against a fake foyer HTTP server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from agents_core.friction_test.driver import RadioOpDriver
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


# ---------------------------------------------------------------------------
# Minimal fake foyer server
# ---------------------------------------------------------------------------

class FakeFoyerHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/talk_it_out/panel":
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/talk_it_out/stream/"):
            # SSE response — send one event then close
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"type":"ping"}\n\n')
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/talk_it_out/session/start":
            self._send_json(200, {"session_id": "fake-session-abc123"})
        elif self.path == "/talk_it_out/ingest":
            self._send_json(200, {"ok": True})
        elif self.path == "/talk_it_out/session/harvest":
            self._send_json(200, {"ok": True})
        elif self.path == "/talk_it_out/session/end":
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})


def _start_fake_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), FakeFoyerHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_foyer():
    server, port = _start_fake_server()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _make_radio_scenario(family_id: str = "test", segments=None, harvest: bool = False) -> Scenario:
    inputs = {
        "segments": segments or [{"segment": "hello world", "ts": "2026-01-01T00:00:00Z"}],
        "harvest": harvest,
    }
    sid = _make_scenario_id("radio-op", family_id, inputs)
    return Scenario(
        scenario_id=sid,
        target="radio-op",
        family_id=family_id,
        kind="happy",
        inputs=inputs,
        expected_class="test",
    )


def test_setup_succeeds_with_live_server(fake_foyer):
    driver = RadioOpDriver(base_url=fake_foyer)
    driver.setup()  # should not raise
    driver.teardown()


def test_run_scenario_returns_observation(fake_foyer):
    driver = RadioOpDriver(base_url=fake_foyer)
    driver.setup()
    s = _make_radio_scenario()
    obs = driver.run_scenario(s)
    driver.teardown()

    assert isinstance(obs, Observation)
    assert obs.scenario_id == s.scenario_id
    assert obs.started_at
    assert obs.finished_at


def test_run_scenario_records_http_calls(fake_foyer):
    driver = RadioOpDriver(base_url=fake_foyer)
    driver.setup()
    s = _make_radio_scenario()
    obs = driver.run_scenario(s)
    driver.teardown()

    methods = [c["method"] for c in obs.http_calls]
    assert "POST" in methods
    urls = [c["url"] for c in obs.http_calls]
    assert any("/session/start" in u for u in urls)
    assert any("/ingest" in u for u in urls)
    assert any("/session/end" in u for u in urls)


def test_run_scenario_session_start_failure(fake_foyer):
    """If session/start returns error, observation has harness_error=True."""
    class FailStartHandler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            body = b'{"error":"fail"}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), FailStartHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        driver = RadioOpDriver(base_url=f"http://127.0.0.1:{port}")
        driver.setup()
        s = _make_radio_scenario()
        obs = driver.run_scenario(s)
        driver.teardown()
        assert obs.harness_error is True
    finally:
        server.shutdown()


def test_setup_fails_if_foyer_down():
    driver = RadioOpDriver(base_url="http://127.0.0.1:19999")
    with pytest.raises(RuntimeError):
        driver.setup()


def test_run_scenario_with_harvest(fake_foyer):
    driver = RadioOpDriver(base_url=fake_foyer)
    driver.setup()
    s = _make_radio_scenario(harvest=True)
    obs = driver.run_scenario(s)
    driver.teardown()

    urls = [c["url"] for c in obs.http_calls]
    assert any("/session/harvest" in u for u in urls)


# ---------------------------------------------------------------------------
# Path injection (reviewer debt aa9404adc1): drivers must accept tmp_path
# overrides for the production path constants so tests can be hermetic.
# ---------------------------------------------------------------------------

def test_radio_op_accepts_tmp_path_dirs(tmp_path):
    consult = tmp_path / "consults"
    harvest = tmp_path / "harvest"
    consult.mkdir()
    harvest.mkdir()
    driver = RadioOpDriver(
        base_url="http://127.0.0.1:1",  # never contacted in this test
        consult_log_dir=consult,
        harvest_queue_dir=harvest,
    )
    assert driver._consult_log_dir == consult
    assert driver._harvest_queue_dir == harvest


def test_radio_op_defaults_to_production_constants(tmp_path):
    from agents_core.friction_test import driver as drv_mod
    driver = RadioOpDriver(base_url="http://127.0.0.1:1")
    assert driver._consult_log_dir == drv_mod.CONSULT_LOG_DIR
    assert driver._harvest_queue_dir == drv_mod.HARVEST_QUEUE_DIR
