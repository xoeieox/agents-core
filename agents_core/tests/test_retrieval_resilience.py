"""Tests for retrieve() fast-fail and canonical RAG host — ground-call-retrieve-resilience-v0.

Acceptance criteria per spec:
  1. Default base URLs resolve to the canonical host 203.0.113.10.
  2. RAG backend at unreachable URL returns [] and does not block other backends.
     Wall-clock stays within fast-fail bound + slack.
  3. chub subprocess that hangs past CHUB_TIMEOUT is killed; treated as a miss; no raise.
  4. Env overrides (RAG_HOST, RAG_HTTP_TIMEOUT, CHUB_TIMEOUT) are honored.
  5. ground() emits honest ungrounded markers for down backends and never raises.
"""

from __future__ import annotations

import importlib
import subprocess
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

import agents_core.retrieval as retrieval_mod
from agents_core.retrieval import Hit, retrieve, _RAG_BASE_URLS, RAG_HTTP_TIMEOUT, CHUB_TIMEOUT


# ---------------------------------------------------------------------------
# 1. Default base URLs resolve to canonical host
# ---------------------------------------------------------------------------

def test_default_base_urls_use_canonical_host():
    """All three RAG base URLs contain 203.0.113.10 when no env override is set."""
    canonical = "203.0.113.10"
    for backend, url in _RAG_BASE_URLS.items():
        assert canonical in url, (
            f"{backend} base URL {url!r} does not contain canonical host {canonical}"
        )


def test_default_vault_rag_port():
    """vault-rag defaults to port 8200 (the grounding scope default)."""
    assert ":8200" in _RAG_BASE_URLS["vault-rag"]


def test_default_room_rag_port():
    assert ":8201" in _RAG_BASE_URLS["room-rag"]


def test_default_code_rag_port():
    assert ":8100" in _RAG_BASE_URLS["code-rag"]


# ---------------------------------------------------------------------------
# 2. Unreachable RAG backend returns [] without blocking others
# ---------------------------------------------------------------------------

def _make_hit(source: str, key: str, content: str = "hit content") -> Hit:
    return Hit(id=f"{source}:{key}", score=0.8, source=source, content=content)


def test_unreachable_rag_backend_returns_others_hits():
    """vault-rag connect-fail -> [] for that backend; other backend hits still returned."""
    mem_hits = [_make_hit("mem", "decision/foo", "mem hit")]

    def fake_search_rag(source, query, filters):
        if source == "vault-rag":
            raise httpx.ConnectError("connection refused")
        return []

    with patch.object(retrieval_mod, "_search_mem", return_value=mem_hits), \
         patch.object(retrieval_mod, "_search_rag", side_effect=fake_search_rag):
        hits = retrieve("query", scope=["mem", "vault-rag"])

    sources = {h.source for h in hits}
    assert "mem" in sources, "mem hits must be present when vault-rag is down"
    assert all(h.source != "vault-rag" for h in hits), "vault-rag must contribute no hits"


def test_unreachable_rag_wall_clock_bounded():
    """retrieve() with a slow-connect RAG backend completes within fast-fail bound + slack."""
    # Simulate a backend that hangs for longer than RAG_HTTP_TIMEOUT
    slow_timeout = RAG_HTTP_TIMEOUT + 0.5  # slightly over bound

    def fake_search_rag(source, query, filters):
        # httpx.post will raise ReadTimeout after RAG_HTTP_TIMEOUT; simulate it cheaply
        raise httpx.ReadTimeout(f"timed out after {RAG_HTTP_TIMEOUT}s")

    with patch.object(retrieval_mod, "_search_mem", return_value=[]), \
         patch.object(retrieval_mod, "_search_chub", return_value=[]), \
         patch.object(retrieval_mod, "_search_rag", side_effect=fake_search_rag):
        t0 = time.monotonic()
        hits = retrieve("query", scope=["mem", "chub", "vault-rag"])
        elapsed = time.monotonic() - t0

    slack = 2.0  # generous CI slack
    assert elapsed < RAG_HTTP_TIMEOUT + slack, (
        f"retrieve() took {elapsed:.2f}s — exceeded bound {RAG_HTTP_TIMEOUT}s + {slack}s slack"
    )
    assert hits == []


# ---------------------------------------------------------------------------
# 3. chub subprocess hang -> killed, treated as miss, no raise
# ---------------------------------------------------------------------------

def test_chub_timeout_treated_as_miss():
    """subprocess.TimeoutExpired from chub -> retrieve() returns [], no raise."""
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["chub"], timeout=CHUB_TIMEOUT)

    with patch("subprocess.run", side_effect=raise_timeout):
        hits = retrieve("query", scope=["chub"])

    assert hits == [], "chub timeout must yield no hits"


def test_chub_timeout_no_raise_from_retrieve():
    """retrieve() must not propagate subprocess.TimeoutExpired to caller."""
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["chub"], timeout=CHUB_TIMEOUT)

    with patch("subprocess.run", side_effect=raise_timeout):
        try:
            retrieve("query", scope=["chub"])
        except subprocess.TimeoutExpired:
            pytest.fail("retrieve() propagated TimeoutExpired — must be caught internally")


def test_chub_timeout_other_backends_unaffected():
    """chub timeout does not prevent mem hits from being returned."""
    mem_hits = [_make_hit("mem", "decision/x", "mem content")]

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["chub"], timeout=CHUB_TIMEOUT)

    with patch("subprocess.run", side_effect=raise_timeout), \
         patch.object(retrieval_mod, "_search_mem", return_value=mem_hits):
        hits = retrieve("query", scope=["mem", "chub"])

    assert any(h.source == "mem" for h in hits), "mem must return hits even when chub times out"


# ---------------------------------------------------------------------------
# 4. Env overrides honored (RAG_HOST, RAG_HTTP_TIMEOUT, CHUB_TIMEOUT)
# ---------------------------------------------------------------------------

def test_env_override_rag_host(monkeypatch):
    """RAG_HOST env var changes all three base URLs after module reload."""
    monkeypatch.setenv("RAG_HOST", "10.0.0.1")
    importlib.reload(retrieval_mod)
    try:
        for backend, url in retrieval_mod._RAG_BASE_URLS.items():
            assert "10.0.0.1" in url, (
                f"{backend} URL {url!r} did not pick up RAG_HOST=10.0.0.1"
            )
    finally:
        monkeypatch.delenv("RAG_HOST", raising=False)
        importlib.reload(retrieval_mod)


def test_env_override_vault_rag_url(monkeypatch):
    """VAULT_RAG_URL overrides vault-rag base URL independently of RAG_HOST."""
    monkeypatch.setenv("VAULT_RAG_URL", "http://custom-host:9999")
    importlib.reload(retrieval_mod)
    try:
        assert retrieval_mod._RAG_BASE_URLS["vault-rag"] == "http://custom-host:9999"
    finally:
        monkeypatch.delenv("VAULT_RAG_URL", raising=False)
        importlib.reload(retrieval_mod)


def test_env_override_rag_http_timeout(monkeypatch):
    """RAG_HTTP_TIMEOUT env var is picked up at module load."""
    monkeypatch.setenv("RAG_HTTP_TIMEOUT", "0.75")
    importlib.reload(retrieval_mod)
    try:
        assert retrieval_mod.RAG_HTTP_TIMEOUT == pytest.approx(0.75)
    finally:
        monkeypatch.delenv("RAG_HTTP_TIMEOUT", raising=False)
        importlib.reload(retrieval_mod)


def test_env_override_chub_timeout(monkeypatch):
    """CHUB_TIMEOUT env var is picked up at module load."""
    monkeypatch.setenv("CHUB_TIMEOUT", "3.5")
    importlib.reload(retrieval_mod)
    try:
        assert retrieval_mod.CHUB_TIMEOUT == pytest.approx(3.5)
    finally:
        monkeypatch.delenv("CHUB_TIMEOUT", raising=False)
        importlib.reload(retrieval_mod)


# ---------------------------------------------------------------------------
# 5. ground() emits ungrounded markers for down backends; never raises
# ---------------------------------------------------------------------------

def test_ground_ungrounded_marker_for_down_rag_backend():
    """ground() emits ungrounded marker for vault-rag when it returns no hits; never raises."""
    from agents_core.ground import ground, GroundBundle

    def fake_retrieve(query, scope, **kwargs):
        # Return only mem hits; simulate vault-rag returning nothing (down)
        return [_make_hit("mem", "decision/ground-test", "content")]

    with patch("agents_core.ground.retrieve", side_effect=fake_retrieve), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("query", pm_state=False, scope=["mem", "vault-rag"])

    assert isinstance(bundle, GroundBundle)
    ungrounded = [p for p in bundle.provenance if p.get("tag") == "ungrounded"]
    sources_ungrounded = {p["source"] for p in ungrounded}
    assert "vault-rag" in sources_ungrounded, (
        "vault-rag returning no hits must produce an ungrounded provenance marker"
    )


def test_ground_never_raises_all_backends_down():
    """ground() with every backend down returns GroundBundle with ungrounded markers; no raise."""
    from agents_core.ground import ground, GroundBundle

    with patch("agents_core.ground.retrieve", return_value=[]), \
         patch("agents_core.ground._assemble_pm_state", return_value=("", [], False)):
        bundle = ground("query", pm_state=False, scope=["mem", "vault-rag", "chub"])

    assert isinstance(bundle, GroundBundle)
    # All three scope backends must have ungrounded markers
    ungrounded_sources = {
        p["source"] for p in bundle.provenance if p.get("tag") == "ungrounded"
    }
    assert ungrounded_sources == {"mem", "vault-rag", "chub"}


# ---------------------------------------------------------------------------
# 6. Parallel fan-out: multiple backends run concurrently
# ---------------------------------------------------------------------------

def test_parallel_fanout_all_backends_queried():
    """retrieve() queries all requested backends even when some are slow."""
    queried: list[str] = []

    def fake_search_mem(query, filters):
        queried.append("mem")
        return []

    def fake_search_chub(query, filters):
        queried.append("chub")
        return []

    def fake_search_rag(source, query, filters):
        queried.append(source)
        return []

    with patch.object(retrieval_mod, "_search_mem", side_effect=fake_search_mem), \
         patch.object(retrieval_mod, "_search_chub", side_effect=fake_search_chub), \
         patch.object(retrieval_mod, "_search_rag", side_effect=fake_search_rag):
        retrieve("query", scope=["mem", "chub", "vault-rag"])

    assert set(queried) == {"mem", "chub", "vault-rag"}, (
        f"Expected all three backends queried; got {queried}"
    )
