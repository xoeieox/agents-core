"""Repo-wide pytest fixtures.

Autouse fixture pre-seeds the GravityWell serving-mode handshake cache
(agents_core.llm._gw_handshake_cache) so tests that exercise the default
gravitywell operator path via mocked requests.post never trigger a live
GET {GW_URL}/v1/models handshake probe. Tests that specifically exercise the
handshake (agents-core-gw-voicing-vllm-repoint-v0) clear/repopulate the cache
themselves for the (url, model) tuple under test.
"""
import pytest


@pytest.fixture(autouse=True)
def _gw_handshake_precached():
    from agents_core import llm as llm_mod

    with llm_mod._gw_handshake_lock:
        llm_mod._gw_handshake_cache.clear()
        llm_mod._gw_handshake_cache[(llm_mod.GW_URL, llm_mod._gw_default_model())] = True
    yield
    with llm_mod._gw_handshake_lock:
        llm_mod._gw_handshake_cache.clear()
