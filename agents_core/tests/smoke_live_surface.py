#!/usr/bin/env python3
"""Smoke test for agents_core.librarian.live_surface — PR 5 gate.

Covers the three requirements from the vault-substrate-v0 spec (PR 5):

(a) Frontmatter contract: written content contains exactly the required keys
    ``type: live-surface``, ``generated: <iso-ts>``, ``freshness: weekly``
    in a YAML frontmatter block.

(b) Idempotent re-render: calling render_live_surface() twice with the same
    mocked corpus produces byte-identical content (the ``generated`` timestamp
    is stable because the render key — hash of corpus snapshots — is unchanged).

(c) Write goes through vault_writer with agent_id="librarian".

All tests are LLM-free (corroborate() and vault_writer.write() are mocked).

Run::

    python3 -m agents_core.tests.smoke_live_surface
or::
    pytest agents_core/tests/smoke_live_surface.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_paths(tmp_path, monkeypatch):
    """Redirect LIVE_SURFACE_TS_STORE and vault infrastructure to tmp dirs."""
    ts_store = tmp_path / "live-surface-ts.json"
    audit_db = tmp_path / "vault-audit.db"
    events_jsonl = tmp_path / "vault-events.jsonl"

    monkeypatch.setenv("LIVE_SURFACE_TS_STORE", str(ts_store))
    monkeypatch.setenv("VAULT_AUDIT_DB", str(audit_db))
    monkeypatch.setenv("VAULT_EVENTS_JSONL", str(events_jsonl))

    # Reset vault_audit connection so it picks up the new env var
    import agents_core.vault_audit as va
    va.reset_connection()

    yield

    va.reset_connection()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(corpus_snapshot: str = "sha256:corpusabc123", answer_text: str = "mock answer") -> MagicMock:
    """Return a mock SynthesisArtifact with a fixed corpus_snapshot."""
    from agents_core.librarian import SynthesisArtifact

    art = MagicMock(spec=SynthesisArtifact)
    art.corpus_snapshot = corpus_snapshot
    art.answer = {"text": answer_text}
    return art


def _make_write_record(path: str = "/tmp/out.md") -> MagicMock:
    """Return a mock WriteRecord."""
    from agents_core.vault_writer import WriteRecord

    record = MagicMock(spec=WriteRecord)
    record.path = path
    record.content_hash = "sha256:writerecordhash"
    record.prev_hash = None
    record.ts = datetime.now(tz=timezone.utc)
    record.agent_id = "librarian"
    record.intent = "weekly live-surface digest"
    return record


def _extract_frontmatter(content: str) -> dict:
    """Parse a simple YAML frontmatter block into a key→value dict."""
    m = re.match(r"^---\r?\n(.*?\r?\n)---\r?\n", content, re.DOTALL)
    assert m, f"No frontmatter found in:\n{content!r}"
    fm_body = m.group(1)
    result = {}
    for line in fm_body.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


# ---------------------------------------------------------------------------
# Test (a): frontmatter contract
# ---------------------------------------------------------------------------


def test_frontmatter_contract(tmp_path):
    """render_live_surface writes content with required frontmatter keys."""
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    thread_art = _make_artifact(corpus_snapshot="sha256:thread111", answer_text="thread content")
    follow_art = _make_artifact(corpus_snapshot="sha256:follow222", answer_text="follow content")
    fake_record = _make_write_record(str(out_path))

    captured: list[str] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        captured.append(content)
        return fake_record

    with patch("agents_core.librarian.corroborate", side_effect=[thread_art, follow_art]), \
         patch("agents_core.vault_writer.write", side_effect=_capture_write):
        record = render_live_surface(out_path)

    assert record is fake_record
    assert len(captured) == 1, "vault_writer.write should be called exactly once"

    content = captured[0]
    fm = _extract_frontmatter(content)

    # (a) Frontmatter contract
    assert fm.get("type") == "live-surface", f"type mismatch in frontmatter: {fm}"
    assert "generated" in fm, f"'generated' key missing from frontmatter: {fm}"
    assert fm.get("freshness") == "weekly", f"freshness mismatch in frontmatter: {fm}"

    # generated should be an ISO timestamp (non-empty, starts with digit)
    generated = fm["generated"]
    assert generated and generated[0].isdigit(), f"'generated' is not an ISO ts: {generated!r}"


def test_frontmatter_stamp_frontmatter_false(tmp_path):
    """vault_writer.write is called with stamp_frontmatter=False."""
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    thread_art = _make_artifact()
    follow_art = _make_artifact()
    fake_record = _make_write_record(str(out_path))

    write_kwargs: list[dict] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        write_kwargs.append({"agent_id": agent_id, "intent": intent, "stamp_frontmatter": stamp_frontmatter})
        return fake_record

    with patch("agents_core.librarian.corroborate", side_effect=[thread_art, follow_art]), \
         patch("agents_core.vault_writer.write", side_effect=_capture_write):
        render_live_surface(out_path)

    assert write_kwargs, "vault_writer.write was not called"
    assert write_kwargs[0]["stamp_frontmatter"] is False, (
        "stamp_frontmatter must be False (frontmatter is already in the content string)"
    )


# ---------------------------------------------------------------------------
# Test (b): idempotent re-render — byte-identical when corpus unchanged
# ---------------------------------------------------------------------------


def test_idempotent_rerender_byte_identical(tmp_path):
    """Calling render_live_surface twice with the same corpus produces identical content."""
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"

    # Both calls return artifacts with the same corpus_snapshots → same render key
    corpus_snap_thread = "sha256:thread_stable_aabbcc"
    corpus_snap_follow = "sha256:follow_stable_ddeeff"

    def _make_call_artifacts():
        return [
            _make_artifact(corpus_snapshot=corpus_snap_thread, answer_text="stable thread text"),
            _make_artifact(corpus_snapshot=corpus_snap_follow, answer_text="stable follow text"),
        ]

    fake_record = _make_write_record(str(out_path))

    contents: list[str] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        contents.append(content)
        return fake_record

    with patch("agents_core.vault_writer.write", side_effect=_capture_write):
        # First render
        with patch("agents_core.librarian.corroborate", side_effect=_make_call_artifacts()):
            render_live_surface(out_path)

        # Second render — same corpus_snapshots → must reuse stored timestamp
        with patch("agents_core.librarian.corroborate", side_effect=_make_call_artifacts()):
            render_live_surface(out_path)

    assert len(contents) == 2, "vault_writer.write should be called once per render"
    assert contents[0] == contents[1], (
        "Re-render with unchanged corpus must produce byte-identical content.\n"
        f"First:  {contents[0]!r}\nSecond: {contents[1]!r}"
    )


def test_different_corpus_snapshot_produces_different_timestamp(tmp_path):
    """Different corpus snapshots on the second render produce a different generated timestamp."""
    from unittest.mock import MagicMock
    import agents_core.librarian.live_surface as ls_mod
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    fake_record = _make_write_record(str(out_path))
    contents: list[str] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        contents.append(content)
        return fake_record

    # Mock datetime.now in live_surface to return deterministic, distinct values.
    ts_sequence = iter(["2026-01-01T00:00:00+00:00", "2026-06-15T12:00:00+00:00"])

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):
            class _Dt:
                def isoformat(self_inner):
                    return next(ts_sequence)
            return _Dt()

    with patch("agents_core.vault_writer.write", side_effect=_capture_write), \
         patch("agents_core.librarian.live_surface.datetime", _FakeDatetime):
        # First render with v1 corpus
        with patch("agents_core.librarian.corroborate", side_effect=[
            _make_artifact(corpus_snapshot="sha256:v1thread"),
            _make_artifact(corpus_snapshot="sha256:v1follow"),
        ]):
            render_live_surface(out_path)

        # Second render with v2 corpus — different render key → new datetime.now() call
        with patch("agents_core.librarian.corroborate", side_effect=[
            _make_artifact(corpus_snapshot="sha256:v2thread"),
            _make_artifact(corpus_snapshot="sha256:v2follow"),
        ]):
            render_live_surface(out_path)

    assert len(contents) == 2, "vault_writer.write should be called once per render"
    fm1 = _extract_frontmatter(contents[0])
    fm2 = _extract_frontmatter(contents[1])
    assert fm1["generated"] != fm2["generated"], (
        f"Different corpus snapshots must produce different generated timestamps.\n"
        f"First:  {fm1['generated']!r}\nSecond: {fm2['generated']!r}"
    )


# ---------------------------------------------------------------------------
# Test (c): write goes through vault_writer with agent_id="librarian"
# ---------------------------------------------------------------------------


def test_write_uses_agent_id_librarian(tmp_path):
    """render_live_surface writes via vault_writer with agent_id='librarian'."""
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    thread_art = _make_artifact()
    follow_art = _make_artifact()
    fake_record = _make_write_record(str(out_path))

    write_calls: list[dict] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        write_calls.append({"path": str(path), "agent_id": agent_id, "intent": intent})
        return fake_record

    with patch("agents_core.librarian.corroborate", side_effect=[thread_art, follow_art]), \
         patch("agents_core.vault_writer.write", side_effect=_capture_write):
        render_live_surface(out_path)

    assert write_calls, "vault_writer.write must be called"
    assert write_calls[0]["agent_id"] == "librarian", (
        f"agent_id must be 'librarian', got {write_calls[0]['agent_id']!r}"
    )
    assert write_calls[0]["path"] == str(out_path)


def test_write_uses_correct_intent(tmp_path):
    """render_live_surface passes 'weekly live-surface digest' as intent."""
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    fake_record = _make_write_record(str(out_path))
    intents: list[str] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        intents.append(intent)
        return fake_record

    with patch("agents_core.librarian.corroborate", side_effect=[
        _make_artifact(), _make_artifact(),
    ]), patch("agents_core.vault_writer.write", side_effect=_capture_write):
        render_live_surface(out_path)

    assert intents[0] == "weekly live-surface digest"


# ---------------------------------------------------------------------------
# Additional: degraded librarian — content still renders, no exception
# ---------------------------------------------------------------------------


def test_degraded_librarian_renders_unavailable_text(tmp_path):
    """If corroborate returns LibrarianUnavailable, render still writes without raising."""
    from agents_core.librarian import LibrarianUnavailable
    from agents_core.librarian.live_surface import render_live_surface

    out_path = tmp_path / "Live-Surface.md"
    fake_record = _make_write_record(str(out_path))
    degraded = LibrarianUnavailable(most_recent_cached=None, degraded=True, reason="test")

    contents: list[str] = []

    def _capture_write(path, content, *, agent_id, intent, stamp_frontmatter=True, **kw):
        contents.append(content)
        return fake_record

    with patch("agents_core.librarian.corroborate", return_value=degraded), \
         patch("agents_core.vault_writer.write", side_effect=_capture_write):
        record = render_live_surface(out_path)

    assert record is fake_record
    assert contents, "vault_writer.write must still be called when degraded"
    content = contents[0]

    # Frontmatter still present even when degraded
    fm = _extract_frontmatter(content)
    assert fm.get("type") == "live-surface"
    assert fm.get("freshness") == "weekly"
    # Degraded text appears in the body
    assert "unavailable" in content


# ---------------------------------------------------------------------------
# CLI smoke: __main__.py parses render-live-surface --out correctly
# ---------------------------------------------------------------------------


def test_cli_render_live_surface_subcommand(tmp_path):
    """librarian_cli.py render-live-surface --out <path> exits 0."""
    import importlib.util

    cli_script = Path("/srv/agents/scripts/librarian_cli.py")
    spec = importlib.util.spec_from_file_location("librarian_cli", cli_script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    out_path = tmp_path / "cli-test.md"
    fake_record = _make_write_record(str(out_path))

    with patch("agents_core.librarian.live_surface.render_live_surface", return_value=fake_record) as mock_rls:
        rc = cli.main(["render-live-surface", "--out", str(out_path)])

    assert rc == 0
    mock_rls.assert_called_once_with(out_path)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent.parent,
    )
    sys.exit(result.returncode)
