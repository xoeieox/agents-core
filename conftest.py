"""Repo-wide pytest fixtures.

Autouse fixture pre-seeds the GravityWell serving-mode handshake cache
(agents_core.llm._gw_handshake_cache) and the auto-detect discovery cache
(agents_core.llm._gw_discovery_cache) so tests that exercise the default
gravitywell operator path via mocked requests.post never trigger a live
GET {GW_URL}/v1/models probe. Both caches are precached directly with the
legacy "gravitywell-122b" value rather than by calling _gw_default_model()
unmocked - with GW_BACKEND/GW_MODEL unset (the ambient default in this test
shell), _gw_default_model() now takes the auto-detect branch and would fire
a real network call otherwise. Tests that specifically exercise the
handshake or discovery probe (agents-core-gw-voicing-vllm-repoint-v0,
agents-core-gw-backend-auto-detect-when-unset-v0) clear/repopulate the
relevant cache themselves for the (url, model) tuple under test.

Also isolates the locality ledger (agents-core-locality-ledger-v0): call_operator/
call_claude_cli/call_gw_agent now side-write a ledger record on every call, so
without this every test in the suite would append real records into the live
/srv/lapis/locality on this host. Individual tests are still free to override
LOCALITY_LEDGER_ROOT themselves (e.g. to assert on the written records).
"""
import time

import pytest


@pytest.fixture(autouse=True)
def _gw_handshake_precached():
    from agents_core import llm as llm_mod

    legacy_model = llm_mod.OPERATOR_DEFAULTS["gravitywell"]
    with llm_mod._gw_handshake_lock:
        llm_mod._gw_handshake_cache.clear()
        llm_mod._gw_handshake_cache[(llm_mod.GW_URL, legacy_model)] = True
        llm_mod._gw_discovery_cache.clear()
        llm_mod._gw_discovery_cache[llm_mod.GW_URL] = (legacy_model, None, time.monotonic())
    yield
    with llm_mod._gw_handshake_lock:
        llm_mod._gw_handshake_cache.clear()
        llm_mod._gw_discovery_cache.clear()


@pytest.fixture(autouse=True)
def _locality_ledger_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path / "locality-ledger"))
