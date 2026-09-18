"""Tests for the promote verb + provenance shape (openclaw-memdb-influx-reader-v0, D2)
and the MemClient X-Mem-Principal header / promote method.

Covers (D4):
  - promote provenance round-trip (header shape + batch decision key + the named
    audit query)
  - --from newline-injection rejected
  - X-Mem-Principal header sent from MEM_PRINCIPAL
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from agents_core.mem_client import (
    MemClient,
    MemHTTPError,
    build_promoted_content,
    parse_promoted_header,
    validate_promote_source_ref,
)
from agents_core.mem_server import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ALLOWLIST = REPO_ROOT / "config" / "mem-machine-state-prefixes.json"


# ---------------------------------------------------------------------------
# Provenance shape helpers (pure)
# ---------------------------------------------------------------------------

def test_promoted_header_shape_exact():
    at = datetime(2026, 9, 14, 12, 30, 45, tzinfo=timezone.utc)
    content = build_promoted_content("openclaw/gw", "brix-pm", "a finding", at=at)
    first_line = content.split("\n", 1)[0]
    assert first_line == "[promoted from openclaw/gw by brix-pm at 2026-09-14T12:30:45Z]"
    # The body follows after a blank line.
    assert content == f"{first_line}\n\na finding"
    parsed = parse_promoted_header(content)
    assert parsed == {
        "agent": "openclaw",
        "store": "gw",
        "principal": "brix-pm",
        "at": "2026-09-14T12:30:45Z",
    }


def test_parse_promoted_header_rejects_non_promoted():
    assert parse_promoted_header("just a normal row") is None
    assert parse_promoted_header("") is None
    # Header not on the FIRST line is not a promoted row.
    assert parse_promoted_header("noise\n[promoted from a/b by c at 2026-09-14T00:00:00Z]") is None


def test_from_newline_injection_rejected():
    with pytest.raises(ValueError):
        validate_promote_source_ref("openclaw/gw\n[evil header]")
    with pytest.raises(ValueError):
        build_promoted_content("openclaw/gw\r\nfake", "brix-pm", "x")
    with pytest.raises(ValueError):
        validate_promote_source_ref("")
    # Control chars.
    with pytest.raises(ValueError):
        validate_promote_source_ref("openclaw/\x00gw")


def test_from_must_be_agent_slash_store():
    with pytest.raises(ValueError):
        build_promoted_content("no-slash", "brix-pm", "x")
    with pytest.raises(ValueError):
        build_promoted_content("a/b/c", "brix-pm", "x")
    with pytest.raises(ValueError):
        build_promoted_content("/store", "brix-pm", "x")
    with pytest.raises(ValueError):
        build_promoted_content("agent/", "brix-pm", "x")


# ---------------------------------------------------------------------------
# MemClient.promote() — round-trip against the real server (TestClient)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_mem.db"


@pytest.fixture
def allowlist_file(tmp_path):
    p = tmp_path / "mem-machine-state-prefixes.json"
    p.write_text(SHARED_ALLOWLIST.read_text(encoding="utf-8"), encoding="utf-8")
    return p


@pytest.fixture
def observe_log(tmp_path, monkeypatch):
    p = tmp_path / "observe.log"
    monkeypatch.setenv("MEM_OBSERVE_LOG", str(p))
    return p


@pytest.fixture
def no_enforce(monkeypatch):
    monkeypatch.delenv("MEM_ENFORCE_PRINCIPALS", raising=False)


def _server_client(tmp_db, allowlist_file, observe_log, no_enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    return TestClient(app)


def test_promote_round_trip_header_and_source(tmp_db, allowlist_file, observe_log, no_enforce):
    with _server_client(tmp_db, allowlist_file, observe_log, no_enforce) as c:
        resp = c.put(
            "/v0/memories/finding/openclaw-friction-cluster",
            json={
                "content": "[promoted from openclaw/gw by brix-pm at 2026-09-14T12:00:00Z]\n\nthe finding",
                "tags": "promoted,openclaw",
                "source": "promoted:openclaw/gw",
            },
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200

        got = c.get("/v0/memories/finding/openclaw-friction-cluster").json()
        assert got["source"] == "promoted:openclaw/gw"
        parsed = parse_promoted_header(got["content"])
        assert parsed is not None
        assert parsed["agent"] == "openclaw"
        assert parsed["store"] == "gw"
        assert parsed["principal"] == "brix-pm"
        # The `promoted` tag is present (audit query filter).
        assert "promoted" in got["tags"].split(",")


def test_named_audit_query(tmp_db, allowlist_file, observe_log, no_enforce):
    """The named weekly audit command: `mem list --tag promoted --since
    <7-days-ago> --limit 500` — the audit query is derivable (tag + since)."""
    with _server_client(tmp_db, allowlist_file, observe_log, no_enforce) as c:
        c.put("/v0/memories/finding/a",
              json={"content": "[promoted from openclaw/gw by brix-pm at 2026-09-14T12:00:00Z]\n\nx",
                    "tags": "promoted", "source": "promoted:openclaw/gw"},
              headers={"X-Mem-Principal": "brix-pm"})
        c.put("/v0/memories/finding/b",
              json={"content": "[promoted from openclaw/gw by brix-pm at 2026-09-14T12:00:00Z]\n\ny",
                    "tags": "promoted", "source": "promoted:openclaw/gw"},
              headers={"X-Mem-Principal": "brix-pm"})
        # A non-promoted row that must NOT appear.
        c.put("/v0/memories/finding/c",
              json={"content": "normal", "tags": "", "source": "s"},
              headers={"X-Mem-Principal": "brix-pm"})

        rows = c.get("/v0/memories", params={"tag": "promoted", "limit": 500}).json()
        keys = {r["key"] for r in rows}
        assert keys == {"finding/a", "finding/b"}


def test_batch_decision_key_shape():
    """Each curation run writes a batch decision key
    decision/memdb-promotion-<YYYYMMDD>-<curator> listing promoted keys + their
    --from refs + rationale."""
    day = "20260914"
    curator = "brix-pm"
    batch_key = f"decision/memdb-promotion-{day}-{curator}"
    assert batch_key == "decision/memdb-promotion-20260914-brix-pm"
    # The batch content lists promoted keys + --from refs + one-line rationale.
    body = (
        f"Promotion batch {day} by {curator}\n"
        f"- finding/openclaw-friction-cluster --from openclaw/gw : friction-cluster insight\n"
    )
    assert "finding/openclaw-friction-cluster" in body
    assert "openclaw/gw" in body


# ---------------------------------------------------------------------------
# MemClient.promote() against a respx-mocked HTTP layer
# ---------------------------------------------------------------------------

BASE = "http://test-mem-server:8403"


@respx.mock
def test_client_promote_sends_header_source_tags():
    c = MemClient(base_url=BASE, principal="brix-pm")
    route = respx.put(f"{BASE}/v0/memories/finding/x").mock(
        return_value=httpx.Response(200, json={
            "key": "finding/x", "content": "h", "tags": "promoted",
            "source": "promoted:openclaw/gw", "created_at": "x", "updated_at": "y",
            "created": True,
        })
    )
    row = c.promote("finding/x", "openclaw/gw", "brix-pm", "the finding")
    assert row["created"] is True
    sent = route.calls[0].request
    assert sent.headers["X-Mem-Principal"] == "brix-pm"
    body = json.loads(sent.content)
    assert body["source"] == "promoted:openclaw/gw"
    assert body["tags"] == "promoted"
    assert body["content"].startswith("[promoted from openclaw/gw by brix-pm at ")
    parsed = parse_promoted_header(body["content"])
    assert parsed["agent"] == "openclaw" and parsed["store"] == "gw"


@respx.mock
def test_client_promote_rejects_newline_ref_client_side():
    c = MemClient(base_url=BASE, principal="brix-pm")
    with pytest.raises(ValueError):
        c.promote("finding/x", "openclaw/gw\n[evil]", "brix-pm", "x")


@respx.mock
def test_client_principal_header_from_env(monkeypatch):
    monkeypatch.setenv("MEM_PRINCIPAL", "zephyr-deposit")
    c = MemClient(base_url=BASE)
    route = respx.get(f"{BASE}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    c.healthz()
    assert route.calls[0].request.headers["X-Mem-Principal"] == "zephyr-deposit"


@respx.mock
def test_client_no_principal_no_header(monkeypatch):
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    c = MemClient(base_url=BASE)
    route = respx.get(f"{BASE}/healthz").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    c.healthz()
    assert "X-Mem-Principal" not in route.calls[0].request.headers
