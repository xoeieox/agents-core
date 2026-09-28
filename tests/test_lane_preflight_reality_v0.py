"""Hermetic tests for agents-core-lane-reality-preflight-v0.

Every test is hermetic: the gw-seats registry is reached ONLY through canned
payloads (the ``fetcher=`` seam / ``set_registry_fetcher`` / a patched
``lane_registry._fetch_payload``). ZERO live network: the module-level guard
below blocks every socket connect, and GW_SEATS_URL is pinned so a resolved
lane's host never depends on the ambient environment.

Coverage follows the spec's Tests section:

  unit   - stubbed registry responses (serving / down / absent) ->
           fire / defer(park) / fail-open;
  unit   - lane_down does not increment the attempt counter (no completed /
           failed queue record is ever produced for a parked row);
  unit   - registry-blind: exactly one probe per target per 10-min tick,
           WARNING tag lane_unknown, never a hang (Deliverable 2);
  unit   - registry root path (never a /v0/status guess), 15 s cache;
  unit   - reviewer follows the ACTIVE lane instead of the :8081 pin
           (Deliverable 3); plain fixers park (Known-deferred);
  unit   - lane tag surfaced per target through ClaudeQueue.status()
           (Deliverable 4);
  unit   - backward compat: lane up => the submitted spec is byte-identical;
  regression - today's three deaths replayed (down :8081 reviewer rows,
           plain fixer lost-dispatch, gw_seat_occupied ceiling accumulation).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from unittest.mock import MagicMock

import agents_core.lane_preflight as lp
import agents_core.shaper as shaper_mod
from agents_core.claude_queue import ClaudeQueue, Priority


# ---------------------------------------------------------------------------
# Hermeticity guards (no live network, no live registry, no live /room)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _coordinator_off():
    """The module-load autoload may bind intention_registry, whose match path
    unlinks queued spec files (the same isolation tests/test_claude_queue.py
    applies) - these tests need the spec files to survive submit()."""
    from agents_core import claude_queue as cq_mod

    prior = cq_mod.get_coordinator()
    cq_mod.register_coordinator(None)
    yield
    cq_mod.register_coordinator(prior)


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    """Any real socket connect fails loudly instead of silently passing on
    whatever the live registry says (the pattern pinned in
    tests/test_lane_registry_gate_lanes.py)."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "hermetic test attempted a live network call (pass fetcher=... or "
            "set_registry_fetcher / patch agents_core.lane_registry._fetch_payload)"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setenv("GW_SEATS_URL", "http://10.0.0.9:8408")


@pytest.fixture(autouse=True)
def _preflight_reset(monkeypatch, tmp_path):
    """Each test starts from a clean preflight: no probe throttle, no cache,
    no process-wide fetcher override (tests opt into canned payloads), and a
    tmp ledger."""
    monkeypatch.setenv("CLAUDE_QUEUE_DIR", str(tmp_path / "claude-queue"))
    lp.set_registry_fetcher(None)
    lp.reset_state_for_tests()
    yield
    lp.set_registry_fetcher(None)
    lp.reset_state_for_tests()


# ---------------------------------------------------------------------------
# Canned registry payloads (shapes verified against the live :8408 root
# payload; see agents_core/lane_registry module docstring)
# ---------------------------------------------------------------------------

FLASHNEXT_SOLO = {
    "reality_view": {
        "reality": "flashnext-solo",
        "anchor": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
        "primary": {"port": 30000, "model_root": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"},
    },
    "seats": [
        {"port": 8081, "state": "down", "bind": None, "model": None, "model_root": None},
        {"port": 8082, "state": "down", "bind": None, "model": None, "model_root": None},
        {"port": 30000, "state": "serving", "bind": "0.0.0.0",
         "model": "Qwen3.8-Flash-Next-NVFP4-SSD-Stream",
         "model_root": "/data/models/flash-next"},
    ],
}

SLOT1_UP = {
    "reality_view": {"reality": "slot1-solo", "anchor": "/data/models/27b",
                     "primary": {"port": 8081, "model_root": "/data/models/27b"}},
    "seats": [
        {"port": 8081, "state": "serving", "bind": None, "model": "gravitywell-27b",
         "model_root": "/data/models/27b"},
        {"port": 8082, "state": "down", "bind": None, "model": None, "model_root": None},
        {"port": 30000, "state": "down", "bind": None, "model": None, "model_root": None},
    ],
}

BLIND = {"error": "not found"}  # malformed / seat-less -> UNKNOWN, not DOWN


def _fetcher(payload):
    return lambda: payload


def _fixer_spec(**over) -> dict:
    spec = {
        "agent_type": "fixer",
        "target_id": "t-fix",
        "engine": "local-fixer",
        "model": "gravitywell-27b",
        "backend_url": "http://10.0.0.9:8081",
        "prompt": "fix it",
        "system": "s",
    }
    spec.update(over)
    return spec


def _reviewer_spec(**over) -> dict:
    spec = _fixer_spec(agent_type="reviewer", target_id="t-rev",
                       engine="local-reviewer", prompt="review it")
    spec.update(over)
    return spec


# ---------------------------------------------------------------------------
# Unit: stubbed registry responses -> fire / park / fail-open
# ---------------------------------------------------------------------------

def test_serving_lane_fires_with_no_tag():
    d = lp.preflight(_reviewer_spec(), fetcher=_fetcher(SLOT1_UP))
    assert d.fire and d.tag == "" and d.seat == "serving" and d.port == 8081
    assert not d.probe and not d.registry_blind


def test_down_lane_parks_with_lane_down_tag():
    d = lp.preflight(_fixer_spec(), fetcher=_fetcher(FLASHNEXT_SOLO))
    assert not d.fire
    assert d.tag == "noop:lane_down:model=gravitywell-27b"
    assert d.seat == "down" and d.port == 8081 and d.action == "park"


def test_absent_seat_row_parks_as_lane_down():
    payload = {"reality_view": {"reality": "unknown"},
               "seats": [{"port": 8082, "state": "down", "model": None}]}
    d = lp.preflight(_fixer_spec(), fetcher=_fetcher(payload))
    assert not d.fire and d.seat == "absent"
    assert d.tag.startswith("noop:lane_down:model=")


def test_non_gate_lane_port_fires():
    """:8082 / third-party endpoints are not gate lanes - never parked."""
    d = lp.preflight(_fixer_spec(backend_url="http://elsewhere:9999"),
                     fetcher=_fetcher(FLASHNEXT_SOLO))
    assert d.fire and d.seat == "not_a_gate_lane" and d.tag == ""


def test_claude_tier_engine_is_ungated():
    called = {"n": 0}

    def _f():
        called["n"] += 1
        return FLASHNEXT_SOLO

    d = lp.preflight({"engine": "claude", "model": "sonnet", "target_id": "t"},
                     fetcher=_f)
    assert d.fire and d.reason == "engine_ungated"
    assert called["n"] == 0, "a claude-tier dispatch must not pay a registry read"


def test_internal_fault_fails_open_never_blocks():
    """Deliverable 2: the preflight must not become a new SPOF."""
    import agents_core.lane_preflight as mod

    def _boom(*a, **kw):
        raise RuntimeError("preflight internals exploded")

    orig = mod.seat_state
    mod.seat_state = _boom
    try:
        d = mod.preflight(_fixer_spec(), fetcher=_fetcher(FLASHNEXT_SOLO))
    finally:
        mod.seat_state = orig
    assert d.fire and d.tag == lp.TAG_LANE_UNKNOWN and "preflight_fault" in d.reason


# ---------------------------------------------------------------------------
# Unit: registry-blind -> one probe per target per 10-min tick (Deliverable 2)
# ---------------------------------------------------------------------------

def test_blind_registry_first_dispatch_probes_with_warning_tag():
    d = lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND), now=1000.0)
    assert d.fire, "blind is UNKNOWN, not down - fail OPEN, never a silent hang"
    assert d.tag == "lane_unknown" and d.probe is True and d.registry_blind is True


def test_blind_registry_second_dispatch_same_tick_parks():
    lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND), now=1000.0)
    d = lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND), now=1001.0)
    assert not d.fire
    assert d.tag == "noop:lane_unknown:probe_in_flight"


def test_blind_registry_probe_resumes_after_the_tick_rolls_over():
    lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND), now=1000.0)
    within = lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND),
                          now=1000.0 + lp.PROBE_TICK_S - 1)
    assert not within.fire
    after = lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND),
                         now=1000.0 + lp.PROBE_TICK_S + 1)
    assert after.fire and after.tag == "lane_unknown"


def test_blind_probe_throttle_is_per_target():
    """'one probe per TARGET per tick': a different target still probes."""
    lp.preflight(_reviewer_spec(), fetcher=_fetcher(BLIND), now=1000.0)
    other = lp.preflight(_reviewer_spec(target_id="t-other"),
                         fetcher=_fetcher(BLIND), now=1001.0)
    assert other.fire and other.tag == "lane_unknown"


def test_fetcher_exception_is_blind_not_down():
    def _boom():
        raise OSError("registry unreachable")

    d = lp.preflight(_fixer_spec(), fetcher=_boom, now=1000.0)
    assert d.fire and d.tag == "lane_unknown" and d.registry_blind is True


# ---------------------------------------------------------------------------
# Unit: registry ROOT endpoint + 15 s cache (Deliverable 1, never /v0/status)
# ---------------------------------------------------------------------------

def test_registry_read_hits_the_root_path_only(monkeypatch):
    """The night2 lesson: gw-seats serves / and /health ONLY - a /v0/status
    guess 404s and reads as dead (and a 404 read must never be mistaken for
    "the lane is down"). The preflight's live read goes through the canonical
    root-path fetcher, and it is a GET (sense), never a POST/wake."""
    from agents_core import lane_registry

    seen = []

    class _Resp:
        status_code = 200

        def json(self):
            return SLOT1_UP

    def _fake_get(url, timeout=None):
        seen.append((url, timeout))
        return _Resp()

    monkeypatch.setattr("httpx.get", _fake_get)
    lp.set_registry_fetcher(None)
    lp.reset_state_for_tests()
    d = lp.preflight(_fixer_spec())
    assert d.fire and d.seat == "serving"
    assert len(seen) == 1, "one root read per tick window (the 15 s cache)"
    url, _timeout = seen[0]
    assert url == "http://10.0.0.9:8408/", f"expected the registry ROOT path, got {url}"
    assert "/v0/status" not in url and not url.rstrip("/").endswith("/status")


def test_registry_read_is_sense_only_no_wake_no_flip(monkeypatch):
    """Standing fence: the preflight READS. No POST to the registry, no
    doorman wake call, no seat flip from this code path."""
    calls = {"get": 0, "post": 0}

    class _Resp:
        status_code = 200

        def json(self):
            return SLOT1_UP

    def _fake_get(url, timeout=None):
        calls["get"] += 1
        return _Resp()

    def _fake_post(url, *a, **kw):
        calls["post"] += 1
        raise AssertionError(f"preflight must never POST (mutating call to {url})")

    monkeypatch.setattr("httpx.get", _fake_get)
    monkeypatch.setattr("httpx.post", _fake_post)
    lp.set_registry_fetcher(None)
    lp.reset_state_for_tests()
    d = lp.preflight(_fixer_spec())
    assert d.fire and calls == {"get": 1, "post": 0}


def test_registry_read_is_cached_for_15_seconds():
    calls = {"n": 0}

    def _f():
        calls["n"] += 1
        return SLOT1_UP

    lp.reset_state_for_tests()
    lp.read_registry(fetcher=_f, now=0.0)
    lp.read_registry(fetcher=_f, now=14.0)
    assert calls["n"] == 1, "the 15 s cache must absorb repeat reads in a tick"
    lp.read_registry(fetcher=_f, now=16.0)
    assert calls["n"] == 2, "the cache must expire (a seat can come up mid-day)"


def test_cache_never_serves_one_fetchers_payload_to_another():
    """A stub payload must not leak across fetchers inside the TTL window -
    otherwise a test (or a second call site) could read a stale canned seat."""
    def _a():
        return SLOT1_UP

    def _b():
        return FLASHNEXT_SOLO

    lp.reset_state_for_tests()
    assert lp.read_registry(fetcher=_a, now=0.0) is SLOT1_UP
    assert lp.read_registry(fetcher=_b, now=1.0) is FLASHNEXT_SOLO


def test_read_registry_is_sense_only_no_flip(monkeypatch):
    """Standing fence: the preflight READS. No POST, no wake, no flip call."""
    from agents_core import lane_registry

    calls = {"get": 0, "post": 0}
    real_get = lane_registry._fetch_payload

    monkeypatch.setattr(lane_registry, "_fetch_payload", lambda timeout=None: real_get(timeout))
    lp.set_registry_fetcher(None)
    lp.reset_state_for_tests()
    lp.read_registry()
    assert calls["post"] == 0
    assert callable(lane_registry._fetch_payload)


# ---------------------------------------------------------------------------
# Unit: reviewer follows the ACTIVE lane (Deliverable 3)
# ---------------------------------------------------------------------------

def test_reviewer_follows_active_lane_under_flashnext_solo():
    d = lp.preflight(_reviewer_spec(), fetcher=_fetcher(FLASHNEXT_SOLO))
    assert d.fire, "the reviewer must no longer die on a dead :8081 pin"
    assert d.reroute_base_url == "http://10.0.0.9:30000"
    assert d.reroute_model == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert d.extras.get("active_lane") == "flashnext"
    assert d.tag.startswith("INFO:noop:lane_down:model=gravitywell-27b")


def test_auditor_family_follows_active_lane_too():
    spec = _reviewer_spec(agent_type="auditor", engine="local-auditor")
    d = lp.preflight(spec, fetcher=_fetcher(FLASHNEXT_SOLO))
    assert d.fire and d.reroute_base_url == "http://10.0.0.9:30000"


def test_plain_fixer_parks_rather_than_auto_routing():
    """Known-deferred: auto-route-to-active-lane for plain fixers needs the
    active-lane policy - so a fixer PARKS (and the tag is the whole story)."""
    d = lp.preflight(_fixer_spec(), fetcher=_fetcher(FLASHNEXT_SOLO))
    assert not d.fire
    assert d.reroute_base_url is None
    assert d.tag == "noop:lane_down:model=gravitywell-27b"


def test_reviewer_with_no_active_lane_parks():
    """Readability without a serving gate lane is still DO-NOT-FIRE - the
    reviewer cannot follow a lane that does not exist."""
    payload = {"reality_view": {"reality": "both-down"},
               "seats": [
                   {"port": 8081, "state": "down", "model": None},
                   {"port": 30000, "state": "down", "model": None},
               ]}
    d = lp.preflight(_reviewer_spec(), fetcher=_fetcher(payload))
    assert not d.fire and d.tag.startswith("noop:lane_down:")


def test_model_to_seat_resolution_when_backend_url_names_no_port():
    """Deliverable 1's 'resolve the dispatching registry row's model -> seat':
    a row with no port on its endpoint is still resolved through the registry
    by served id / lane marker."""
    assert lp.model_port(FLASHNEXT_SOLO, "Qwen3.8-Flash-Next-NVFP4-SSD-Stream") == 30000
    assert lp.model_port(FLASHNEXT_SOLO, "flash-next") == 30000
    assert lp.model_port(SLOT1_UP, "gravitywell-27b") == 8081
    assert lp.model_port(BLIND, "gravitywell-27b") is None

    d = lp.preflight(_fixer_spec(backend_url=None, model="gravitywell-27b"),
                     fetcher=_fetcher(FLASHNEXT_SOLO))
    assert not d.fire and d.port == 8081 and d.seat == "down"


# ---------------------------------------------------------------------------
# Unit: lane_down does NOT increment the attempt counter (claim path)
# ---------------------------------------------------------------------------

def _task(queue: ClaudeQueue, spec: dict, task_id: str) -> str:
    spec_path = queue.queue_dir / f"{task_id}-spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec))
    return queue.submit({
        "task_type": "subprocess",
        "priority": Priority.HIGH,
        "timeout_seconds": 60,
        "submitted_by": "test",
        "model": spec.get("model"),
        "description": f"{spec.get('agent_type')}:{spec.get('target_id')}",
        "payload": {"command": "true", "spec_path": str(spec_path)},
    }, task_id=task_id)


def test_claim_parks_down_lane_row_and_writes_no_failure_record(tmp_path):
    """The attempt counter counts queue RECORDS. A parked row must stay in
    pending/ and produce NO completed/ and NO failed/ record - that is the
    mechanical reason the counter cannot increment on lane_down."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    tid = _task(q, _fixer_spec(), task_id="fix-down-1")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    assert q.claim() is None, "a down-lane row must not be claimed"
    assert (q.pending_dir / f"{tid}.yaml").exists(), "the dispatch stays pending"
    assert not list(q.failed_dir.glob("*.yaml")), "no failure record => no attempt burned"
    assert not list(q.completed_dir.glob("*.yaml"))
    assert q.status()["depth"] == 1


def test_claim_lane_down_is_repeatable_without_accumulating(tmp_path):
    """Regression for the gw_seat_occupied ceiling rows (pr=363/364/365): the
    same parked row can be polled every tick without ever producing a record
    the ceiling key could count."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _fixer_spec(), task_id="fix-down-2")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))
    for _ in range(25):
        assert q.claim() is None
    assert not list(q.failed_dir.glob("*.yaml"))
    assert not list(q.completed_dir.glob("*.yaml"))


def test_claim_reroutes_queued_reviewer_onto_active_lane(tmp_path):
    """Deliverable 3 at the CLAIM point: a reviewer row that was queued while
    :8081 was up and whose lane died afterwards follows the ACTIVE lane
    instead of dying - the spec is re-pinned to the registry's live seat."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="rev-reroute")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    claimed = q.claim()
    assert claimed is not None, "the reviewer follows the live lane, it does not die"
    spec = json.loads((q.queue_dir / "rev-reroute-spec.json").read_text())
    assert spec["backend_url"] == "http://10.0.0.9:30000"
    assert spec["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    assert claimed["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"


def test_claim_serves_serving_lane_normally(tmp_path):
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="rev-up-1")
    lp.set_registry_fetcher(_fetcher(SLOT1_UP))
    claimed = q.claim()
    assert claimed is not None and claimed["id"] == "rev-up-1"


def test_parked_row_does_not_head_of_line_block_other_work(tmp_path):
    """Candidate-skip semantics (the fixer_flash defer pattern): a parked
    high-priority row must not starve other pending rows."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="park-me")
    other = _task(q, {"agent_type": "spec_reviewer", "target_id": "t-doc",
                      "engine": "claude", "model": "sonnet",
                      "prompt": "p", "system": "s"}, task_id="claim-me")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))
    claimed = q.claim()
    assert claimed is not None and claimed["id"] == other
    assert (q.pending_dir / "park-me.yaml").exists()


def test_claim_blind_registry_fires_one_probe_then_parks(tmp_path):
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="rev-blind-1")
    lp.set_registry_fetcher(_fetcher(BLIND))
    import time as _t
    claimed = q.claim()
    assert claimed is not None, "registry-blind fails open on the first claim"
    # re-pending the claimed row: the probe is already in flight for this tick
    (q.active_dir / "rev-blind-1.yaml").rename(q.pending_dir / "rev-blind-1.yaml")
    assert q.claim() is None


def test_claim_preflight_fault_fails_open(tmp_path, monkeypatch):
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="rev-fault-1")
    monkeypatch.setattr(lp, "preflight",
                        MagicMock(side_effect=RuntimeError("gate exploded")))
    claimed = q.claim()
    assert claimed is not None and claimed["id"] == "rev-fault-1"


# ---------------------------------------------------------------------------
# Unit: Deliverable 4 - tag surfaced per target through queue status
# ---------------------------------------------------------------------------

def test_lane_tag_visible_in_queue_status_per_target(tmp_path):
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _fixer_spec(), task_id="fix-status")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))
    q.claim()

    status = q.status()
    assert status["depth"] == 1
    assert "lane_status" in status, "the PM/Erah must see WHY a target sits"
    entry = status["lane_status"]["t-fix"]
    assert entry["tag"] == "noop:lane_down:model=gravitywell-27b"
    assert entry["action"] == "park" and entry["seat"] == "down" and entry["port"] == 8081
    # The ledger line itself (JSONL is the substrate, not the only surface).
    lines = lp.ledger_path().read_text().splitlines()
    assert any(json.loads(x)["tag"].startswith("noop:lane_down") for x in lines)


def test_status_shape_unchanged_when_no_preflight_decisions(tmp_path):
    """Backward compat: with no lane decisions recorded, status() is the
    pre-preflight dict, byte-identical."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    assert q.status() == {"depth": 0, "in_flight": []}


def test_status_shape_unchanged_when_every_lane_is_up(tmp_path):
    """A lane-up queue must look EXACTLY like it did before this target -
    clean fires are recorded (so a recovered lane clears its stale tag) but
    are not surfaced as a WHY."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _reviewer_spec(), task_id="rev-up-status")
    lp.set_registry_fetcher(_fetcher(SLOT1_UP))
    assert q.claim() is not None
    st = q.status()
    assert "lane_status" not in st, "an up lane has nothing to explain"
    assert set(st) == {"depth", "in_flight"}


def test_recovered_lane_clears_its_stale_lane_down_tag(tmp_path):
    """A stale WHY is as misleading as a missing one: once the registry says
    the seat serves again, the target must stop showing noop:lane_down."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _fixer_spec(), task_id="fix-recover")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))
    assert q.claim() is None
    assert q.status()["lane_status"]["t-fix"]["tag"].startswith("noop:lane_down")

    # The lane comes back (registry now serves :8081); the row claims.
    lp.set_registry_fetcher(_fetcher(SLOT1_UP))
    assert q.claim() is not None
    assert "lane_status" not in q.status(), "the recovered target is no longer parked"


def test_ledger_write_is_deduped_per_task_and_tag(tmp_path):
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    _task(q, _fixer_spec(), task_id="fix-dedup")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))
    for _ in range(10):
        q.claim()
    lines = [json.loads(x) for x in lp.ledger_path().read_text().splitlines()]
    assert len([x for x in lines if x["task_id"] == "fix-dedup"]) == 1


# ---------------------------------------------------------------------------
# Unit: shaper dispatch point (the code path the daemon fires through)
# ---------------------------------------------------------------------------

@pytest.fixture
def shaper(tmp_path, monkeypatch):
    reg = tmp_path / "registry.yaml"
    reg.write_text(yaml.dump({"agents": {
        "fixer": {"chub_bundles": [], "system_template": "fix {repo}",
                  "model": "gravitywell-27b", "timeout_s": 60,
                  "engine": "local-fixer", "backend_url": "http://10.0.0.9:8081"},
        "reviewer": {"chub_bundles": [], "system_template": "review {repo}",
                     "model": "gravitywell-27b", "timeout_s": 60,
                     "engine": "local-reviewer", "backend_url": "http://10.0.0.9:8081"},
        "scout": {"chub_bundles": [], "system_template": "look {repo}",
                  "model": "sonnet", "timeout_s": 60},
    }}))
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")
    s = shaper_mod.Shaper(reg)
    return s


def _patch_queues(monkeypatch):
    claude_q = MagicMock()
    claude_q._generate_id.return_value = "gen-1"
    claude_q.submit.return_value = "gen-1"
    gpu_q = MagicMock()
    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda *a, **kw: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda *a, **kw: gpu_q)
    return claude_q, gpu_q


def test_dispatch_parks_plain_fixer_on_lane_down(shaper, monkeypatch, capsys):
    claude_q, gpu_q = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    res = shaper.dispatch("fixer", "flashnext-trigger-ssh-default-fix-v0", "fix it",
                          vars_={"repo": "agents-core"})

    assert res.task_id == "" and res.spec_path == "" and res.output_path == ""
    assert res.lane_tag == "noop:lane_down:model=gravitywell-27b"
    claude_q.submit.assert_not_called()
    gpu_q.submit.assert_not_called()
    err = capsys.readouterr().err
    assert "noop:lane_down" in err and "nothing submitted" in err


def test_dispatch_reroutes_reviewer_to_active_lane(shaper, monkeypatch):
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    res = shaper.dispatch("reviewer", "pr-369-review", "review it",
                          vars_={"repo": "agents-core"})

    assert res.lane_tag == "" or res.lane_tag.startswith("INFO:")
    assert res.lane_rerouted == "http://10.0.0.9:30000"
    claude_q.submit.assert_called_once()
    spec_file = Path(shaper_mod.SPEC_DIR / list(
        Path(shaper_mod.SPEC_DIR).glob("*.json"))[0].name)
    spec = json.loads(spec_file.read_text())
    assert spec["backend_url"] == "http://10.0.0.9:30000"
    assert spec["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"


def test_dispatch_reroute_queues_row_under_the_active_seat_model(shaper, monkeypatch):
    """The serialized-seat guard keys admission on the QUEUE ROW's model, so a
    re-pinned reviewer must queue under the seat it actually dials (the
    flash-next ops cap of 2), not under the dead seat's model."""
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    shaper.dispatch("reviewer", "pr-369-review", "review it",
                    vars_={"repo": "agents-core"})

    submitted = claude_q.submit.call_args[0][0]
    assert submitted["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"


def test_dispatch_lane_up_queues_row_under_registry_model_unchanged(shaper, monkeypatch):
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(SLOT1_UP))
    shaper.dispatch("fixer", "t-model", "fix it", vars_={"repo": "agents-core"})
    assert claude_q.submit.call_args[0][0]["model"] == "gravitywell-27b"


def test_dispatch_lane_up_is_byte_identical(shaper, monkeypatch):
    """Invariant: when the lane is up, the submitted spec carries exactly the
    fields it carried before this target - no preflight keys leak in."""
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(SLOT1_UP))

    res = shaper.dispatch("fixer", "t-up", "fix it", vars_={"repo": "agents-core"})

    assert res.lane_tag == "" and res.lane_rerouted is None
    spec = json.loads(next(Path(shaper_mod.SPEC_DIR).glob("*.json")).read_text())
    assert spec["backend_url"] == "http://10.0.0.9:8081"
    assert spec["model"] == "gravitywell-27b"
    assert not any(k.startswith("lane_") for k in spec), (
        f"preflight keys leaked into the spec: {[k for k in spec if k.startswith('lane_')]}"
    )


def test_dispatch_blind_registry_fires_probe_once(shaper, monkeypatch, capsys):
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(_fetcher(BLIND))

    first = shaper.dispatch("fixer", "t-blind", "fix it", vars_={"repo": "agents-core"})
    assert first.task_id == "gen-1" and first.lane_tag == "lane_unknown"
    assert "WARN: lane-preflight" in capsys.readouterr().err

    second = shaper.dispatch("fixer", "t-blind", "fix it", vars_={"repo": "agents-core"})
    assert second.task_id == "" and second.lane_tag == "noop:lane_unknown:probe_in_flight"
    assert claude_q.submit.call_count == 1, "at most one probe per target per tick"


def test_dispatch_claude_tier_never_gated(shaper, monkeypatch):
    claude_q, _ = _patch_queues(monkeypatch)
    lp.set_registry_fetcher(lambda: (_ for _ in ()).throw(
        AssertionError("claude-tier dispatch must not read the registry")))
    res = shaper.dispatch("scout", "t-scout", "look", vars_={"repo": "agents-core"})
    assert res.task_id == "gen-1" and res.lane_tag == ""


# ---------------------------------------------------------------------------
# Regression: today's three deaths replayed (2026-09-27)
# ---------------------------------------------------------------------------

def test_regression_pr369_reviewer_death_no_longer_dies():
    """09:45/09:50 PT: reviewer dispatches for PR #369 died "local reviewer
    produced no verdict" - pinned :8081 while the registry said 8081:down.
    The reviewer now follows the ACTIVE lane instead of dying."""
    spec = _reviewer_spec(target_id="pr-369-review", task_id="pr369")
    d = lp.preflight(spec, fetcher=_fetcher(FLASHNEXT_SOLO))
    assert d.fire and d.reroute_base_url == "http://10.0.0.9:30000"
    assert "lane_down" in d.tag and "lane=flashnext" in d.tag


def test_regression_plain_fixer_lost_dispatch_now_parks():
    """09:19-09:22: plain fixer for flashnext-trigger-ssh-default-fix-v0 ->
    "local seat returned no text" -> lost dispatch, no branch, no durable
    error. It now parks WITH a durable tagged record and no seat call."""
    d = lp.preflight(
        _fixer_spec(target_id="flashnext-trigger-ssh-default-fix-v0"),
        fetcher=_fetcher(FLASHNEXT_SOLO))
    assert not d.fire and d.tag == "noop:lane_down:model=gravitywell-27b"
    lp.record_lane_tag(d, target_id="flashnext-trigger-ssh-default-fix-v0",
                       task_id="t1", agent_type="fixer", engine="local-fixer")
    status = lp.read_lane_status("flashnext-trigger-ssh-default-fix-v0")
    assert status["flashnext-trigger-ssh-default-fix-v0"]["tag"] == d.tag


def test_regression_seat_occupied_ceiling_stops_accumulating(tmp_path):
    """pr=363/364/365 accumulated gw_seat_occupied ceiling rows because each
    reviewer cycle reached the (dead) seat. Now: plain-tier rows park (no
    queue record at all, so the ceiling key cannot grow) and reviewer rows are
    re-pinned OFF the dead :8081 onto the registry's live lane, so they never
    contend the dead seat again."""
    q = ClaudeQueue(queue_dir=tmp_path / "q")
    for pr in (363, 364):
        _task(q, _fixer_spec(target_id=f"pr-{pr}-fix"), task_id=f"fix-{pr}")
    for pr in (365,):
        _task(q, _reviewer_spec(target_id=f"pr-{pr}-review"), task_id=f"rev-{pr}")
    lp.set_registry_fetcher(_fetcher(FLASHNEXT_SOLO))

    claimed = q.claim()  # the reviewer follows the active lane
    assert claimed is not None and claimed["id"] == "rev-365"
    assert claimed["model"] == "Qwen3.8-Flash-Next-NVFP4-SSD-Stream"
    spec = json.loads((q.queue_dir / "rev-365-spec.json").read_text())
    assert spec["backend_url"] == "http://10.0.0.9:30000", (
        "the reviewer must never dial the dead :8081 pin again"
    )

    for _ in range(3):  # three daemon ticks for the parked fixer rows
        assert q.claim() is None

    assert not list(q.failed_dir.glob("*.yaml"))
    assert not list(q.completed_dir.glob("*.yaml"))
    assert len(list(q.pending_dir.glob("*.yaml"))) == 2
    st = q.status()
    assert st["depth"] == 2
    assert all(v["tag"].startswith("noop:lane_down")
               for v in st["lane_status"].values() if v["action"] == "park")
