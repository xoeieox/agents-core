"""Smoke test for agents_core.forgejo — request shape via mocked httpx.

Doesn't hit the Forgejo server. Verifies URL + body shape so schema drift
in the helper gets caught before any consumer hits it.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents_core import forgejo


@pytest.fixture(autouse=True)
def _isolate_provenance_ledger(tmp_path, monkeypatch):
    """Keep mutating-op provenance out of the real audit ledger during tests."""
    monkeypatch.setattr(forgejo, "PROVENANCE_LOG", str(tmp_path / "forgejo-ops.jsonl"))


def _mock_response(status_code=200, json_data=None):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data or {}
    m.raise_for_status = MagicMock()
    m.content = b'{"ok": true}'
    return m


def test_create_pr_request_shape():
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _mock_response(json_data={"number": 42, "html_url": "ok"})

    with patch.object(forgejo.httpx, "post", side_effect=fake_post):
        result = forgejo.create_pr(
            repo="agents-core",
            title="Test",
            head="feature",
            body="Body",
        )

    assert "Erah/agents-core/pulls" in captured["url"]
    assert captured["json"]["title"] == "Test"
    assert captured["json"]["head"] == "feature"
    assert captured["json"]["base"] == "main"
    assert captured["json"]["body"] == "Body"
    assert result["number"] == 42


def test_get_pr_url():
    captured_urls = []

    def fake_get(url, headers=None, timeout=None):
        captured_urls.append(url)
        return _mock_response(json_data={"number": 7, "state": "open"})

    with patch.object(forgejo.httpx, "get", side_effect=fake_get):
        forgejo.get_pr("agents-core", 7)

    assert "/repos/Erah/agents-core/pulls/7" in captured_urls[0]
