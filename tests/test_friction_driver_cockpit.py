"""Tests for CockpitDriver against a fake cockpit HTTP server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents_core.friction_test.driver import CockpitDriver
from agents_core.friction_test.observe import Observation
from agents_core.friction_test.scenario import Scenario, _make_scenario_id


class FakeCockpitHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from urllib.parse import unquote
        decoded_path = unquote(self.path)
        if self.path == "/api/status":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/api/thread-weaver":
            self._send_json(200, {"threads": []})
        elif self.path == "/api/room-state":
            self._send_json(200, {"state": "idle"})
        elif self.path == "/api/gpu-queue":
            self._send_json(200, {"queue": []})
        elif "/api/documents/" in decoded_path and ".." in decoded_path:
            self._send_json(400, {"error": "invalid path"})
        elif self.path.startswith("/cockpit/api/workday"):
            self._send_json(200, {
                "data": {},
                "provenance": {"agent_id": "cockpit-v0", "signature": "abc", "mode": "live"},
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if "/directive" in self.path and "unknown-tid" in self.path:
            self._send_json(404, {"error": "target not found"})
        elif "/directive" in self.path:
            self._send_json(200, {"ok": True})
        elif "/comment" in self.path and not body:
            self._send_json(400, {"error": "empty body"})
        elif "/comment" in self.path:
            self._send_json(200, {"ok": True})
        elif self.path == "/api/heading":
            if not body:
                self._send_json(400, {"error": "missing heading"})
            else:
                self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not found"})


@pytest.fixture
def fake_cockpit():
    server = HTTPServer(("127.0.0.1", 0), FakeCockpitHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _make_cockpit_scenario(method="GET", path="/api/status", body=None, tid=None, **extra) -> Scenario:
    inputs = {"method": method, "path": path}
    if body is not None:
        inputs["body"] = body
    if tid:
        inputs["tid"] = tid
    inputs.update(extra)
    sid = _make_scenario_id("cockpit", f"{method}_{path.replace('/','_')}", inputs)
    return Scenario(
        scenario_id=sid,
        target="cockpit",
        family_id="test",
        kind="happy",
        inputs=inputs,
        expected_class="test",
    )


def test_setup_succeeds(fake_cockpit):
    driver = CockpitDriver(base_url=fake_cockpit)
    driver.setup()
    driver.teardown()


def test_run_get_status(fake_cockpit):
    driver = CockpitDriver(base_url=fake_cockpit)
    driver.setup()
    s = _make_cockpit_scenario("GET", "/api/status")
    obs = driver.run_scenario(s)
    driver.teardown()

    assert isinstance(obs, Observation)
    assert not obs.harness_error
    assert len(obs.http_calls) == 1
    assert obs.http_calls[0]["status"] == 200


def test_run_post_unknown_tid_returns_404(fake_cockpit):
    driver = CockpitDriver(base_url=fake_cockpit)
    driver.setup()
    s = _make_cockpit_scenario(
        "POST",
        "/api/thread/unknown-tid-xyz/directive",
        body={"action": "test"},
        tid="unknown-tid-xyz",
        unknown_tid=True,
    )
    obs = driver.run_scenario(s)
    driver.teardown()

    assert obs.http_calls[0]["status"] == 404


def test_run_path_traversal(fake_cockpit):
    driver = CockpitDriver(base_url=fake_cockpit)
    driver.setup()
    s = _make_cockpit_scenario("GET", "/api/documents/targets/../../etc/passwd")
    obs = driver.run_scenario(s)
    driver.teardown()

    # httpx normalizes the path; fake server returns 4xx for traversal patterns
    assert obs.http_calls[0]["status"] < 500


def test_setup_fails_if_cockpit_down():
    driver = CockpitDriver(base_url="http://127.0.0.1:19998")
    with pytest.raises(RuntimeError):
        driver.setup()


def test_observation_has_right_shape(fake_cockpit):
    driver = CockpitDriver(base_url=fake_cockpit)
    driver.setup()
    s = _make_cockpit_scenario("GET", "/api/thread-weaver")
    obs = driver.run_scenario(s)
    driver.teardown()

    d = obs.to_dict()
    for key in ("scenario_id", "started_at", "finished_at", "http_calls",
                "sse_events", "mem_writes", "vault_writes", "log_appends", "errors"):
        assert key in d
