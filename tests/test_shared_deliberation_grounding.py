"""Tests for shared-deliberation grounding: target resolution, argv/spec_text threading,
and the facets-grounding-denied repair-station escalation.

Target: agents-core-ground-the-deliberation-v0.
"""

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agents_core.shared_deliberation.envelope import DeliberationRequest
from agents_core.shared_deliberation.orchestrator import (
    run_deliberation,
    init_facets_semaphore,
    _resolve_grounding_target,
    _cleanup_grounding_worktree,
    _extract_denied_codebase_surfaces,
    _maybe_escalate_grounding_denial,
    _run_facets_subprocess,
)


@pytest.fixture
def init_semaphore():
    init_facets_semaphore(2)


def _make_local_clone(root: Path, repo: str) -> Path:
    """Create a local git clone with a fake origin/main remote-tracking ref.

    No real remote is configured -- refs/remotes/origin/main is written directly,
    mirroring what a real `git clone` + `git fetch` would leave behind, without
    ever touching a network.
    """
    clone_dir = root / f"{repo}-working"
    clone_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=clone_dir, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=clone_dir, check=True)
    (clone_dir / "f.txt").write_text("hi")
    subprocess.run(["git", "add", "f.txt"], cwd=clone_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=clone_dir, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=clone_dir, check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", sha], cwd=clone_dir, check=True
    )
    return clone_dir


class _FakePopen:
    """Minimal Popen stand-in that records argv and the written context file."""

    captured_argv = []
    captured_context = []

    def __init__(self, argv, **kwargs):
        _FakePopen.captured_argv.append(argv)
        idx = argv.index("--context-file")
        context_path = argv[idx + 1]
        _FakePopen.captured_context.append(json.loads(Path(context_path).read_text()))
        self.stdout = io.StringIO(json.dumps({"deliberation_id": "test-id"}))
        self.stderr = io.StringIO("")
        self.returncode = 0

    def poll(self):
        return 0

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


@pytest.fixture(autouse=True)
def _reset_fake_popen():
    _FakePopen.captured_argv = []
    _FakePopen.captured_context = []
    yield


# ---------------------------------------------------------------------------
# 1-3: _resolve_grounding_target
# ---------------------------------------------------------------------------

def test_resolve_grounding_target_yields_worktree_and_argv(tmp_path, monkeypatch):
    """DoD 2: repo='conductor' yields a worktree path, present in --target-repo."""
    import agents_core.shared_deliberation.orchestrator as orch

    monkeypatch.setattr(orch, "_GROUNDING_CLONE_ROOT", str(tmp_path))
    _make_local_clone(tmp_path, "conductor")

    path, skip_reason, provenance = _resolve_grounding_target({"repo": "conductor"})
    try:
        assert path is not None
        assert skip_reason == ""
        assert provenance["source_repo"] == "conductor"
        assert provenance["resolved_sha"]
        assert Path(path).is_dir()

        facets_repo = tmp_path / "facets"
        facets_repo.mkdir()
        with patch("subprocess.Popen", side_effect=_FakePopen):
            _run_facets_subprocess("text", {}, "gravitywell", facets_repo, target_repo=path)
        argv = _FakePopen.captured_argv[0]
        assert "--target-repo" in argv
        idx = argv.index("--target-repo")
        assert argv[idx + 1] == path
    finally:
        _cleanup_grounding_worktree(provenance)
    assert not Path(path).exists()


def test_resolve_grounding_target_normalizes_org_qualified_repo(tmp_path, monkeypatch):
    """DoD 3: repo='someorg/foo' normalises to 'foo' before resolution."""
    import agents_core.shared_deliberation.orchestrator as orch

    monkeypatch.setattr(orch, "_GROUNDING_CLONE_ROOT", str(tmp_path))
    _make_local_clone(tmp_path, "foo")

    path, skip_reason, provenance = _resolve_grounding_target({"repo": "someorg/foo"})
    try:
        assert path is not None
        assert skip_reason == ""
        assert provenance["source_repo"] == "foo"
    finally:
        _cleanup_grounding_worktree(provenance)


def test_resolve_grounding_target_no_repo_key(tmp_path, monkeypatch):
    """DoD 4a: no repo key -> (None, 'no_repo_in_context', {})."""
    import agents_core.shared_deliberation.orchestrator as orch

    monkeypatch.setattr(orch, "_GROUNDING_CLONE_ROOT", str(tmp_path))

    path, skip_reason, provenance = _resolve_grounding_target({})
    assert path is None
    assert skip_reason == "no_repo_in_context"
    assert provenance == {}


def test_resolve_grounding_target_unresolvable_repo(tmp_path, monkeypatch):
    """DoD 4b: a repo that does not resolve -> (None, 'grounding_target_unavailable', {})."""
    import agents_core.shared_deliberation.orchestrator as orch

    monkeypatch.setattr(orch, "_GROUNDING_CLONE_ROOT", str(tmp_path))

    path, skip_reason, provenance = _resolve_grounding_target({"repo": "nope"})
    assert path is None
    assert skip_reason == "grounding_target_unavailable"
    assert provenance == {}


def test_resolve_grounding_target_rejects_junk_repo_values(tmp_path, monkeypatch):
    """Junk **Repo:** frontmatter values ('n-a', '<repo>' with traversal-ish content) never resolve."""
    import agents_core.shared_deliberation.orchestrator as orch

    monkeypatch.setattr(orch, "_GROUNDING_CLONE_ROOT", str(tmp_path))

    # "../etc/passwd" normalises (rsplit) to the harmless final segment "passwd", which
    # then simply fails to resolve as a directory -- traversal never escapes the clone root.
    path, skip_reason, provenance = _resolve_grounding_target({"repo": "../etc/passwd"})
    assert path is None
    assert skip_reason == "grounding_target_unavailable"

    for junk in ["has space", ""]:
        path, skip_reason, provenance = _resolve_grounding_target({"repo": junk})
        assert path is None
        assert skip_reason == "no_repo_in_context"


def test_argv_no_target_repo_when_none(tmp_path):
    """No --target-repo when target_repo=None (DoD 4: argv has no --target-repo)."""
    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    with patch("subprocess.Popen", side_effect=_FakePopen):
        _run_facets_subprocess("text", {}, "gravitywell", facets_repo, target_repo=None)
    assert "--target-repo" not in _FakePopen.captured_argv[0]


# ---------------------------------------------------------------------------
# 5: spec_text threaded into the context file
# ---------------------------------------------------------------------------

def test_context_file_contains_spec_text(tmp_path):
    """DoD 5: the context file written to disk contains spec_text."""
    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    with patch("subprocess.Popen", side_effect=_FakePopen):
        _run_facets_subprocess("the spec body", {}, "gravitywell", facets_repo)
    written = _FakePopen.captured_context[0]
    assert written["spec_text"] == "the spec body"


def test_context_file_explicit_spec_text_not_overwritten(tmp_path):
    """DoD 5: an explicit caller-supplied spec_text is not overwritten."""
    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    with patch("subprocess.Popen", side_effect=_FakePopen):
        _run_facets_subprocess(
            "the spec body", {"spec_text": "caller-supplied"}, "gravitywell", facets_repo
        )
    written = _FakePopen.captured_context[0]
    assert written["spec_text"] == "caller-supplied"


# ---------------------------------------------------------------------------
# 6: grounding_result_file branch + GroundingHandoffError guard unchanged
# ---------------------------------------------------------------------------

def test_grounding_result_file_branch_unchanged_with_target_repo(tmp_path):
    """DoD 6: --grounding-result-file + --no-auto-ground still work; target_repo may
    coexist in argv without disturbing that branch."""
    grounding_file = tmp_path / "grounding.json"
    grounding_file.write_text('{"result": "ok"}')
    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    with patch("subprocess.Popen", side_effect=_FakePopen):
        _run_facets_subprocess(
            "text", {}, "gravitywell", facets_repo,
            grounding_result_file=str(grounding_file),
            target_repo="/some/worktree",
        )
    argv = _FakePopen.captured_argv[0]
    assert "--grounding-result-file" in argv
    assert "--no-auto-ground" in argv
    assert "--target-repo" in argv


# ---------------------------------------------------------------------------
# 7-13: escalation logic
# ---------------------------------------------------------------------------

def _patch_escalate(monkeypatch):
    captured = []

    def fake_escalate(*args, **kwargs):
        captured.append(kwargs)
        return "inc-fake"

    import agents_core.repair_station as rs_mod
    monkeypatch.setattr(rs_mod, "escalate", fake_escalate)
    return captured


def test_codebase_denial_fires_case_a_high_tier(monkeypatch):
    """DoD 7: a round with a codebase:-keyed sim_failures entry produces exactly one
    case-(a) escalate call with first(), Tier.HIGH, the station ID and stable_pointer."""
    from agents_core.repair_station import Tier

    captured = _patch_escalate(monkeypatch)
    facets_dict = {
        "rounds": [
            {"round_num": 1, "sim_failures": {"codebase:verify foo": "connection refused"}},
            {"round_num": 2, "sim_requests": [{"surface": "codebase", "query": "x"}],
             "sim_results": {}, "sim_failures": {}},
        ]
    }
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc123"},
        facets_dict=facets_dict,
    )
    assert len(captured) == 1
    kwargs = captured[0]
    assert kwargs["tier"] == Tier.HIGH
    assert kwargs["station_id"] == "shared-deliberation/facets-grounding-denied"
    assert kwargs["stable_pointer"] == "agents_core/shared_deliberation/orchestrator.py"
    assert kwargs["escalation_policy"].kind == "first"
    assert kwargs["error_signal"]["case"] == "denied"
    # repair-station-close-dedup-triage-v0 Leg 2: dedup on stable failure identity
    # only — rounds_affected/resolved_sha (per-run uniques) must not be in the list.
    assert kwargs["signature_fields"] == [
        "case", "repo", "skip_reason", "denied_surfaces", "reasons", "source_repo",
    ]


def test_non_codebase_surfaces_produce_zero_escalations(monkeypatch):
    """DoD 8: backcaster: or unregistered-surface sim_failures produce zero escalations."""
    captured = _patch_escalate(monkeypatch)
    facets_dict = {
        "rounds": [
            {"round_num": 1, "sim_failures": {
                "backcaster:q": "unavailable",
                "static-analysis:q": "Unknown sim surface",
            }},
            {"round_num": 2, "sim_requests": [], "sim_results": {}, "sim_failures": {}},
        ]
    }
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc123"},
        facets_dict=facets_dict,
    )
    assert captured == []


def test_final_round_unfulfilled_requests_not_a_denial(monkeypatch):
    """DoD 9, regression guard: a final round with non-empty sim_requests, empty
    sim_results and empty sim_failures, cap_hit_with_unfulfilled_requests True,
    produces zero escalations. Sourced from the live corpus fixture."""
    captured = _patch_escalate(monkeypatch)
    fixture_path = Path("/srv/lapis/facets/deliberations/2026-08-01-085055-0c7cae.json")
    fixture = json.loads(fixture_path.read_text())
    final_round = fixture["rounds"][-1]
    assert final_round["sim_requests"]
    assert final_round["sim_results"] == {}
    assert final_round["sim_failures"] == {}
    assert fixture["cap_hit_with_unfulfilled_requests"] is True

    facets_dict = {"rounds": [final_round], "cap_hit_with_unfulfilled_requests": True}
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc123"},
        facets_dict=facets_dict,
    )
    assert captured == []


def test_grounding_result_file_set_suppresses_absent_case(monkeypatch):
    """DoD 10: grounding_result_file set with no repo produces zero escalations."""
    captured = _patch_escalate(monkeypatch)
    _maybe_escalate_grounding_denial(
        context={},
        grounding_result_file="/tmp/grounding.json",
        skip_reason="no_repo_in_context",
        provenance={},
        facets_dict={"stub": True},
    )
    assert captured == []


def test_stub_mode_suppresses_escalation(monkeypatch):
    """DoD 11a: stub mode produces zero escalations."""
    captured = _patch_escalate(monkeypatch)
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")
    _maybe_escalate_grounding_denial(
        context={},
        grounding_result_file=None,
        skip_reason="no_repo_in_context",
        provenance={},
        facets_dict={"stub": True, "methodology": {}},
    )
    assert captured == []


def test_facets_dispatch_disabled_suppresses_escalation(monkeypatch):
    """DoD 11b: FACETS_DISPATCH_DISABLED=1 produces zero escalations."""
    captured = _patch_escalate(monkeypatch)
    monkeypatch.setenv("FACETS_DISPATCH_DISABLED", "1")
    _maybe_escalate_grounding_denial(
        context={},
        grounding_result_file=None,
        skip_reason="no_repo_in_context",
        provenance={},
        facets_dict=None,
    )
    assert captured == []


def test_missing_facets_repo_suppresses_escalation(monkeypatch):
    """DoD 11c: missing-facets-repo produces zero escalations."""
    captured = _patch_escalate(monkeypatch)
    with patch("pathlib.Path.exists", return_value=False):
        _maybe_escalate_grounding_denial(
            context={},
            grounding_result_file=None,
            skip_reason="no_repo_in_context",
            provenance={},
            facets_dict=None,
        )
    assert captured == []


def test_facets_dict_none_produces_zero_escalations(monkeypatch):
    """DoD 12a: facets_dict is None -> zero escalations, no exception."""
    captured = _patch_escalate(monkeypatch)
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc"},
        facets_dict=None,
    )
    assert captured == []


def test_facets_dict_no_rounds_key_produces_zero_escalations(monkeypatch):
    """DoD 12b: a dict with no 'rounds' key -> zero escalations, no exception."""
    captured = _patch_escalate(monkeypatch)
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc"},
        facets_dict={"methodology": {}},
    )
    assert captured == []


def test_two_denying_rounds_produce_exactly_one_escalate_call(monkeypatch):
    """DoD 13: two denying rounds produce exactly one escalate call."""
    captured = _patch_escalate(monkeypatch)
    facets_dict = {
        "rounds": [
            {"round_num": 1, "sim_failures": {"codebase:a": "denied"}},
            {"round_num": 2, "sim_failures": {"codebase:b": "denied"}},
            {"round_num": 3, "sim_requests": [], "sim_results": {}, "sim_failures": {}},
        ]
    }
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc"},
        facets_dict=facets_dict,
    )
    assert len(captured) == 1
    assert set(captured[0]["error_signal"]["rounds_affected"]) == {1, 2}


def test_error_signal_excludes_per_run_unique_values(monkeypatch):
    """DoD 14: error_signal contains no per-run unique value -- two runs differing
    only in an arbitrary per-run identifier produce the same error_signature."""
    from agents_core.repair_station.escalate import _compute_signature

    facets_dict_a = {
        "deliberation_id": "run-aaaa",
        "rounds": [
            {"round_num": 1, "sim_failures": {"codebase:q": "denied"}},
            {"round_num": 2, "sim_requests": [], "sim_results": {}, "sim_failures": {}},
        ],
    }
    facets_dict_b = {
        "deliberation_id": "run-bbbb",
        "rounds": [
            {"round_num": 1, "sim_failures": {"codebase:q": "denied"}},
            {"round_num": 2, "sim_requests": [], "sim_results": {}, "sim_failures": {}},
        ],
    }

    signals = []

    def fake_escalate(*args, **kwargs):
        signals.append(kwargs["error_signal"])
        return "inc-fake"

    import agents_core.repair_station as rs_mod
    monkeypatch.setattr(rs_mod, "escalate", fake_escalate)

    for fd in (facets_dict_a, facets_dict_b):
        _maybe_escalate_grounding_denial(
            context={"repo": "conductor"},
            grounding_result_file=None,
            skip_reason="",
            provenance={"source_repo": "conductor", "resolved_sha": "abc"},
            facets_dict=fd,
        )

    assert len(signals) == 2
    assert "deliberation_id" not in signals[0]
    assert "deliberation_id" not in signals[1]
    assert _compute_signature(signals[0]) == _compute_signature(signals[1])


def test_escalate_isolated_db_used(monkeypatch, tmp_path):
    """DoD 16: escalation tests use an isolated DB, never /srv/lapis/repair-station/repair_station.db."""
    monkeypatch.setenv("REPAIR_STATION_DB", str(tmp_path / "rs.db"))
    facets_dict = {
        "rounds": [
            {"round_num": 1, "sim_failures": {"codebase:q": "denied"}},
            {"round_num": 2, "sim_requests": [], "sim_results": {}, "sim_failures": {}},
        ]
    }
    _maybe_escalate_grounding_denial(
        context={"repo": "conductor"},
        grounding_result_file=None,
        skip_reason="",
        provenance={"source_repo": "conductor", "resolved_sha": "abc"},
        facets_dict=facets_dict,
    )
    assert (tmp_path / "rs.db").exists()


# ---------------------------------------------------------------------------
# 15: escalate() raising is caught and suppressed; envelope unaffected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalate_raising_does_not_affect_envelope(monkeypatch, init_semaphore):
    """DoD 15: escalate() raising is caught/logged; the returned envelope equals the
    envelope produced with escalate patched to a no-op."""
    import agents_core.shared_deliberation.orchestrator as orch
    import uuid as uuid_mod

    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")
    monkeypatch.setattr(orch, "_resolve_grounding_target", lambda ctx: (
        "/fake/worktree", "", {"source_repo": "conductor", "resolved_sha": "abc",
                                "clone_dir": "/fake/clone", "worktree_path": "/fake/worktree"}
    ))
    monkeypatch.setattr(orch, "_cleanup_grounding_worktree", lambda prov: None)

    async def fake_facets_subprocess(*a, **kw):
        return (True, {"rounds": [{"round_num": 1, "sim_failures": {"codebase:q": "denied"}}]},
                "deliberation-id", None)

    monkeypatch.setattr(orch, "_facets_subprocess", fake_facets_subprocess)

    fixed_uuid = uuid_mod.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(uuid_mod, "uuid4", lambda: fixed_uuid)

    request = DeliberationRequest(
        text="test", context={"repo": "conductor"},
        facets_operator="haiku", council_voicing="haiku",
    )

    import agents_core.repair_station as rs_mod

    def raising_escalate(*args, **kwargs):
        raise RuntimeError("db unwritable")

    monkeypatch.setattr(rs_mod, "escalate", raising_escalate)
    envelope_raise = await run_deliberation(request)

    monkeypatch.setattr(rs_mod, "escalate", lambda *a, **kw: None)
    envelope_noop = await run_deliberation(request)

    assert envelope_raise == envelope_noop
