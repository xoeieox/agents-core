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


@pytest.fixture(autouse=True)
def _mem_allowlist_isolated(tmp_path, monkeypatch):
    """Point the machine-state allowlist at the REPO-SHIPPED artifact for
    every test (openclaw-memdb-influx-reader-v0, reviewer PR #339 cycle 1
    [high]).

    mem_server.create_app() hard-requires a valid machine-state allowlist
    (REFUSES TO START on missing/malformed/unreadable — the fail-open
    bypass the gate names). Its default resolution is
    mem_machinery.default_allowlist_path(): the live path
    (MEM_MACHINE_STATE_PREFIXES_PATH, default
    /srv/agents/config/mem-machine-state-prefixes.json) if it exists, else
    the repo-shipped copy (config/mem-machine-state-prefixes.json). On this
    host the live path happens to exist, so pre-existing tests that call
    create_app() with NO explicit allowlist_path (tests/test_mem_server.py,
    tests/test_mem_deposit.py) pass — but in a clean CI/clone where only
    the repo copy exists, the resolution is environment-dependent and the
    suite is not deterministic.

    This fixture removes the ambient dependence: it points
    MEM_MACHINE_STATE_PREFIXES_PATH at a per-test copy of the repo-shipped
    artifact, so the default resolution ALWAYS finds a valid allowlist
    regardless of what (if anything) exists at the live path. It does NOT
    weaken the fail-closed guard: tests that exercise the refuse-to-start
    behavior (tests/test_mem_influx.py) pass an explicit allowlist_path
    (missing/malformed) or override the env var themselves, and an explicit
    path bypasses this default resolution by design.
    """
    from agents_core import mem_machinery

    src = mem_machinery.REPO_ALLOWLIST_PATH
    if src.exists():
        p = tmp_path / "mem-machine-state-prefixes.json"
        p.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setenv("MEM_MACHINE_STATE_PREFIXES_PATH", str(p))
    yield


@pytest.fixture(autouse=True)
def _lane_preflight_isolated(tmp_path, monkeypatch):
    """Make the lane-reality preflight (agents-core-lane-reality-preflight-v0)
    invisible to tests that are not about it.

    Two live-state dependencies would otherwise leak into every dispatch-path
    test on this host:

      * the gw-seats registry read - which seat holds the GPU at run time
        would decide whether a local-fixer / local-reviewer dispatch fires or
        parks, so pre-existing shaper/queue tests would flip green/red with
        the seat handover (exactly the failure mode
        tests/test_lane_registry_gate_lanes.py documents for OPERATOR_DEFAULTS);
      * the lane-status ledger under the live /room claude-queue dir.

    The fixture installs an ALL-SERVING canned registry payload (the
    byte-identical-to-today branch: every gateable seat is serving, so every
    dispatch fires exactly as before this target) and redirects the ledger via
    the existing CLAUDE_QUEUE_DIR room-path env. Tests that exercise the
    preflight itself pass an explicit ``fetcher=`` (which bypasses the
    override) or install their own via ``set_registry_fetcher``.
    """
    from agents_core import lane_preflight

    all_serving = {
        "reality_view": {"reality": "slot1-solo", "anchor": "gravitywell-27b",
                         "primary": {"port": 8081, "model_root": "/data/models/27b"}},
        "seats": [
            {"port": 8081, "state": "serving", "bind": None,
             "model": "gravitywell-27b", "model_root": "/data/models/27b"},
            {"port": 8082, "state": "down", "bind": None, "model": None},
            {"port": 30000, "state": "serving", "bind": None,
             "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
             "model_root": "/data/models/flash-next"},
        ],
    }
    prev = lane_preflight.set_registry_fetcher(lambda: all_serving)
    monkeypatch.setenv("CLAUDE_QUEUE_DIR", str(tmp_path / "claude-queue"))
    lane_preflight.reset_state_for_tests()
    try:
        yield
    finally:
        lane_preflight.set_registry_fetcher(prev)
        lane_preflight.reset_state_for_tests()
