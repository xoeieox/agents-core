"""Tests for local-fixer engine dispatch (local-fixer-dispatch-wiring-v0).

AC1: engine field round-trips through shaper — spec sidecar has engine/target_id/repo;
     default "claude" leaves every existing field byte-identical.
AC2: shaped_runner.main() with engine=="claude" calls call_claude_cli, never _run_local_fixer.
AC3: _run_local_fixer happy path — setup_worktree, git branch/commit/push, create_pr,
     teardown; returns PR URL; no .lapis-pm-verdict.json written.
AC4: Empty/failed guards — no PR for empty diff, concluded=False, zero passing tests.
AC5: Provenance body is factual — attribution, diffstat, test outcome, traceability markers,
     transcript artifact link present.
AC6: Git failures (branch/commit/push non-zero) → return "", no PR, worktree torn down.
AC7: Full module green; no live-network dependency.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

import agents_core.shaper as shaper_mod
from agents_core.shaper import Shaper, ShapedAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(path: Path, agents: dict, shared_preamble: str = "") -> Path:
    reg = path / "registry.yaml"
    data: dict = {}
    if shared_preamble:
        data["shared_preamble"] = shared_preamble
    data["agents"] = agents
    reg.write_text(yaml.dump(data))
    return reg


def _agent_def(model: str = "haiku", engine: str = "claude") -> dict:
    d: dict = {
        "chub_bundles": [],
        "system_template": "test for {repo}",
        "model": model,
        "timeout_s": 60,
    }
    if engine != "claude":
        d["engine"] = engine
    return d


@pytest.fixture
def shaper_mocks(tmp_path, monkeypatch):
    reg = _write_registry(tmp_path, {
        "fixer": _agent_def("sonnet"),
        "fixer_local": _agent_def("gravitywell-122b", engine="local-fixer"),
    })
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")

    claude_q = MagicMock()
    claude_q._generate_id.return_value = "claude_task_id"
    claude_q.submit.return_value = None
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"

    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda: gpu_q)
    monkeypatch.delenv("AGENTS_CORE_FORCE_GPU_QUEUE", raising=False)
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)

    s = Shaper(reg)
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))
    return s, claude_q, gpu_q


def _make_spec(tmp_path: Path, **overrides) -> Path:
    spec = {
        "model": "gravitywell-122b",
        "engine": "local-fixer",
        "system": "you are a fixer",
        "prompt": "fix the bug",
        "timeout_s": 60,
        "capture_meta": False,
        "target_id": "my-target-v0",
        "repo": "agents-core",
        "task_id": "abc123",
        "base_branch": "main",
        "slot_id": "abc123",
    }
    spec.update(overrides)
    shaped = tmp_path / "shaped"
    shaped.mkdir(exist_ok=True)
    p = shaped / f"{spec['target_id']}-fixer_local-abc123.json"
    p.write_text(json.dumps(spec))
    return p


def _fake_handle(worktree: Path) -> MagicMock:
    h = MagicMock()
    h.path = worktree
    h.env = {}
    return h


def _good_fixer_result(diff: str = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n") -> dict:
    return {
        "final_diff": diff,
        "concluded": True,
        "last_test_outcome": {"passed": 3, "failed": 0},
        "steps": [{"tool": "read_file"}, {"tool": "write_file"}],
    }


# ---------------------------------------------------------------------------
# AC1: engine field round-trips through shaper
# ---------------------------------------------------------------------------

def test_engine_defaults_to_claude_when_absent(tmp_path):
    reg = _write_registry(tmp_path, {"fixer": _agent_def("sonnet")})
    s = Shaper(reg)
    assert s.get_agent("fixer").engine == "claude"


def test_engine_read_from_registry(tmp_path):
    reg = _write_registry(tmp_path, {"fixer_local": _agent_def("gravitywell-122b", engine="local-fixer")})
    s = Shaper(reg)
    assert s.get_agent("fixer_local").engine == "local-fixer"


def test_engine_propagated_to_spec_sidecar(shaper_mocks, tmp_path):
    s, _, gpu_q = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer_local", "my-target-v0", "fix it", vars_={"repo": "agents-core"})
    written = list(spec_dir.glob("*.json"))
    assert len(written) == 1
    spec = json.loads(written[0].read_text())
    assert spec["engine"] == "local-fixer"
    assert spec["target_id"] == "my-target-v0"
    assert spec["repo"] == "agents-core"


def test_claude_engine_spec_has_no_engine_override(shaper_mocks, tmp_path):
    s, claude_q, _ = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer", "t-2", "do thing", vars_={"repo": "test-repo"})
    written = list(spec_dir.glob("*.json"))
    spec = json.loads(written[0].read_text())
    # engine=="claude" (default) must be present and all standard fields unchanged
    assert spec["engine"] == "claude"
    assert spec["target_id"] == "t-2"
    assert spec["repo"] == "test-repo"
    assert "permission_mode" in spec
    assert spec["worktree_required"] is True


def test_local_fixer_spec_has_task_id_and_base_branch(shaper_mocks, tmp_path):
    s, _, gpu_q = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer_local", "my-target-v0", "fix it", vars_={"repo": "agents-core"})
    written = list(spec_dir.glob("*.json"))
    spec = json.loads(written[0].read_text())
    # local-fixer injects task_id so the runner can call setup_worktree
    assert "task_id" in spec
    assert spec["base_branch"] == "main"


def test_existing_claude_spec_fields_unchanged(shaper_mocks, tmp_path):
    """Regression: adding engine/target_id/repo must not drop any existing field."""
    s, claude_q, _ = shaper_mocks
    spec_dir = tmp_path / "shaped"
    s.dispatch("fixer", "t-3", "go", vars_={"repo": "test-repo"})
    spec = json.loads(list(spec_dir.glob("*.json"))[0].read_text())
    for field in ("agent_type", "model", "timeout_s", "system", "prompt", "cwd",
                  "permission_mode", "capture_meta", "task_id", "base_branch",
                  "worktree_required", "slot_id"):
        assert field in spec, f"existing field {field!r} missing from spec"


# ---------------------------------------------------------------------------
# AC2: engine=="claude" → call_claude_cli called, _run_local_fixer not called
# ---------------------------------------------------------------------------

def test_claude_engine_calls_claude_cli(tmp_path):
    shaped = tmp_path / "shaped"
    shaped.mkdir()
    spec = {
        "model": "sonnet",
        "engine": "claude",
        "system": "",
        "prompt": "do it",
        "timeout_s": 30,
        "capture_meta": False,
        "task_id": "task-99",
        "base_branch": "main",
        "worktree_required": True,
        "cwd": str(tmp_path),
        "target_id": "t-1",
        "repo": "agents-core",
    }
    spec_path = shaped / "t-1-fixer-abc.json"
    spec_path.write_text(json.dumps(spec))

    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    fake_handle = MagicMock()
    fake_handle.path = worktree
    fake_handle.env = {}

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner.call_claude_cli", return_value="ok") as mock_cli,
        patch("agents_core.shaped_runner._run_local_fixer") as mock_lf,
        patch("agents_core.worktree.setup_worktree", return_value=fake_handle),
        patch("agents_core.worktree.teardown_worktree"),
    ):
        sr.main()

    mock_cli.assert_called_once()
    mock_lf.assert_not_called()


def test_local_fixer_engine_calls_run_local_fixer_not_cli(tmp_path):
    spec_path = _make_spec(tmp_path)

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner._run_local_fixer", return_value="http://1.2.3.4:3000/e/r/pulls/5") as mock_lf,
        patch("agents_core.shaped_runner.call_claude_cli") as mock_cli,
        patch("sys.stdout"),
    ):
        sr.main()

    mock_lf.assert_called_once()
    mock_cli.assert_not_called()


# ---------------------------------------------------------------------------
# AC3: _run_local_fixer happy path
# ---------------------------------------------------------------------------

def test_run_local_fixer_happy_path(tmp_path):
    spec_path = _make_spec(tmp_path)
    spec = json.loads(spec_path.read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    pr_response = {"html_url": "http://203.0.113.10:3000/Erah/agents-core/pulls/42"}

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_fixer_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.forgejo.create_pr", return_value=pr_response) as mock_pr,
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == "http://203.0.113.10:3000/Erah/agents-core/pulls/42"
    mock_pr.assert_called_once()
    pr_call = mock_pr.call_args
    assert pr_call.kwargs.get("repo") == "agents-core" or pr_call.args[0] == "agents-core"
    assert pr_call.kwargs.get("head", pr_call.args[2] if len(pr_call.args) > 2 else "") == "lapis/my-target-v0/local"
    assert pr_call.kwargs.get("base", "main") == "main"
    # No verdict file written
    assert not (worktree / ".lapis-pm-verdict.json").exists()
    # Worktree torn down in finally
    mock_teardown.assert_called_once()


def test_run_local_fixer_branch_name_uses_slug(tmp_path):
    spec_path = _make_spec(tmp_path, slug="forced")
    spec = json.loads(spec_path.read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    git_calls = []

    def fake_run(cmd, **kwargs):
        git_calls.append(cmd)
        return MagicMock(returncode=0, stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_fixer_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/pulls/1"}),
        patch("subprocess.run", side_effect=fake_run),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    # cmd = ["git", "-C", cwd, "checkout", "-b", branch]
    branch_cmds = [c for c in git_calls if len(c) > 3 and c[3] == "checkout"]
    assert any("lapis/my-target-v0/forced" in " ".join(c) for c in branch_cmds)


# ---------------------------------------------------------------------------
# AC4: Empty/failed guards
# ---------------------------------------------------------------------------

def test_no_pr_when_not_concluded(tmp_path):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    result = {"final_diff": "some diff", "concluded": False, "last_test_outcome": None, "steps": []}

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(result, [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_pr.assert_not_called()


def test_no_pr_when_diff_empty(tmp_path):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    for empty_diff in ("", "   \n  ", None):
        result = {"final_diff": empty_diff, "concluded": True, "last_test_outcome": {"passed": 3, "failed": 0}, "steps": []}

        with (
            patch("agents_core.gw_agent.call_gw_agent", return_value=(result, [])),
            patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
            patch("agents_core.worktree.teardown_worktree"),
            patch("agents_core.forgejo.create_pr") as mock_pr,
            patch.object(Path, "mkdir"),
            patch.object(Path, "write_text"),
        ):
            url = sr._run_local_fixer(spec, str(tmp_path))

        assert url == "", f"expected '' for empty_diff={empty_diff!r}"
        mock_pr.assert_not_called()


def test_no_pr_when_zero_passing_tests(tmp_path):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    result = {
        "final_diff": "diff --git a/x b/x\n+fix\n",
        "concluded": True,
        "last_test_outcome": {"passed": 0, "failed": 5},
        "steps": [],
    }

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(result, [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_pr.assert_not_called()


def test_no_is_none_check_needed_concluded_false_covers_doorman_unreachable(tmp_path):
    """writeable=True never returns None; concluded=False covers doorman-unreachable."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    # Simulate what _build_fixer_result returns when doorman is unreachable
    doorman_unreachable = {"final_diff": "", "concluded": False, "last_test_outcome": None, "steps": []}

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(doorman_unreachable, [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_pr.assert_not_called()


# ---------------------------------------------------------------------------
# AC5: Provenance body is factual
# ---------------------------------------------------------------------------

def test_pr_body_contains_required_elements(tmp_path):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    captured_body: list[str] = []

    def fake_create_pr(**kwargs):
        captured_body.append(kwargs.get("body", ""))
        return {"html_url": "http://x/pulls/7"}

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_fixer_result(), [{"a": 1}])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", side_effect=fake_create_pr),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured_body, "create_pr was not called"
    body = captured_body[0]

    assert "122B fixer harness" in body, "attribution line missing"
    assert "<!-- lapis-gpu-id: abc123 -->" in body, "lapis-gpu-id marker missing"
    assert "<!-- lapis-tid: my-target-v0 -->" in body, "lapis-tid marker missing"
    assert "passed" in body, "test outcome missing"
    assert "gw-transcript.json" in body, "transcript artifact link missing"
    assert "step" in body.lower(), "step summary missing"


# ---------------------------------------------------------------------------
# AC6: Git failures → return "", no PR, worktree torn down
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fail_on", ["checkout", "add", "commit", "push"])
def test_git_failure_returns_empty_no_pr(tmp_path, fail_on):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    def git_side_effect(cmd, **kwargs):
        # cmd = ["git", "-C", cwd, subcmd, ...]
        subcmd = cmd[3] if len(cmd) > 3 else ""
        if fail_on == subcmd:
            return MagicMock(returncode=1, stderr=f"fake {fail_on} failure")
        return MagicMock(returncode=0, stderr="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_good_fixer_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch("subprocess.run", side_effect=git_side_effect),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == "", f"expected '' on {fail_on} failure"
    mock_pr.assert_not_called()
    mock_teardown.assert_called_once()


def test_worktree_torn_down_even_on_exception(tmp_path):
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=RuntimeError("boom")),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree") as mock_teardown,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_teardown.assert_called_once()


# ---------------------------------------------------------------------------
# AC1: Salvage on max_steps_reached with passing tests
# ---------------------------------------------------------------------------

def _max_steps_fixer_result(
    diff: str = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n",
    passed: int = 5,
    failed: int = 0,
    errors: int = 0,
) -> dict:
    return {
        "final_diff": diff,
        "concluded": False,
        "max_steps_reached": True,
        "no_progress": False,
        "last_test_outcome": {"passed": passed, "failed": failed, "errors": errors},
        "steps": [],
    }


def test_salvage_max_steps_reached_with_passing_tests_opens_pr(tmp_path):
    """AC1: max_steps_reached + non-empty diff + passing tests → PR opened (harness-salvaged)."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    captured_body: list[str] = []

    def fake_create_pr(**kwargs):
        captured_body.append(kwargs.get("body", ""))
        return {"html_url": "http://203.0.113.10:3000/Erah/agents-core/pulls/99"}

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_max_steps_fixer_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", side_effect=fake_create_pr),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == "http://203.0.113.10:3000/Erah/agents-core/pulls/99"
    assert captured_body, "create_pr was not called"
    assert "harness-salvaged" in captured_body[0], "salvage marker missing from PR body"


def test_no_pr_max_steps_reached_empty_diff(tmp_path):
    """AC1: max_steps_reached + empty diff → no-op."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    result = _max_steps_fixer_result(diff="")

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(result, [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_pr.assert_not_called()


def test_no_pr_max_steps_reached_failing_tests(tmp_path):
    """AC1: max_steps_reached + failing tests → no-op."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    result = _max_steps_fixer_result(passed=3, failed=2, errors=0)

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(result, [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr") as mock_pr,
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        url = sr._run_local_fixer(spec, str(tmp_path))

    assert url == ""
    mock_pr.assert_not_called()


# ---------------------------------------------------------------------------
# AC2a: max_steps precedence (spec > env > default 60)
# ---------------------------------------------------------------------------

def test_max_steps_uses_spec_value_over_env_and_default(tmp_path, monkeypatch):
    """AC2a: spec JSON max_steps takes highest precedence."""
    spec = json.loads(_make_spec(tmp_path, max_steps=99).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.setenv("GW_AGENT_MAX_STEPS", "77")

    import agents_core.shaped_runner as sr

    captured_kwargs: list[dict] = []

    def fake_gw_agent(**kwargs):
        captured_kwargs.append(kwargs)
        return (_good_fixer_result(), [])

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/p/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured_kwargs, "call_gw_agent not called"
    assert captured_kwargs[0]["max_steps"] == 99, "spec max_steps not used"


def test_max_steps_uses_env_when_spec_absent(tmp_path, monkeypatch):
    """AC2a: env GW_AGENT_MAX_STEPS used when spec has no max_steps."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    spec.pop("max_steps", None)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.setenv("GW_AGENT_MAX_STEPS", "55")

    import agents_core.shaped_runner as sr

    captured_kwargs: list[dict] = []

    def fake_gw_agent(**kwargs):
        captured_kwargs.append(kwargs)
        return (_good_fixer_result(), [])

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/p/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured_kwargs[0]["max_steps"] == 55, "env GW_AGENT_MAX_STEPS not used"


def test_max_steps_default_60_when_spec_and_env_absent(tmp_path, monkeypatch):
    """AC2a: default 60 used when neither spec nor env supplies max_steps."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    spec.pop("max_steps", None)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.delenv("GW_AGENT_MAX_STEPS", raising=False)

    import agents_core.shaped_runner as sr

    captured_kwargs: list[dict] = []

    def fake_gw_agent(**kwargs):
        captured_kwargs.append(kwargs)
        return (_good_fixer_result(), [])

    with (
        patch("agents_core.gw_agent.call_gw_agent", side_effect=fake_gw_agent),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr", return_value={"html_url": "http://x/p/1"}),
        patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    assert captured_kwargs[0]["max_steps"] == 60, "default max_steps should be 60"


# ---------------------------------------------------------------------------
# AC4: Disambiguated warn messages
# ---------------------------------------------------------------------------

def _no_pr_result(*, max_steps_reached=False, no_progress=False, diff="", passed=0) -> dict:
    return {
        "final_diff": diff,
        "concluded": False,
        "max_steps_reached": max_steps_reached,
        "no_progress": no_progress,
        "last_test_outcome": {"passed": passed, "failed": 0, "errors": 0} if passed else None,
        "steps": [],
    }


def test_warn_message_max_steps_reached_no_diff(tmp_path, capsys):
    """AC4: max_steps_reached with empty diff emits specific message."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_no_pr_result(max_steps_reached=True), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    err = capsys.readouterr().err
    assert "max_steps" in err
    assert "doorman" not in err.lower()
    assert "spinning" not in err.lower()


def test_warn_message_no_progress(tmp_path, capsys):
    """AC4: no_progress emits specific spinning-wheels message."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_no_pr_result(no_progress=True), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    err = capsys.readouterr().err
    assert "spinning" in err or "no semantic progress" in err
    assert "doorman" not in err.lower()


def test_warn_message_doorman_unreachable(tmp_path, capsys):
    """AC4: concluded=False with no max_steps/no_progress flags → DoormanUnreachable message."""
    spec = json.loads(_make_spec(tmp_path).read_text())
    worktree = tmp_path / "wt"
    worktree.mkdir()

    import agents_core.shaped_runner as sr

    with (
        patch("agents_core.gw_agent.call_gw_agent", return_value=(_no_pr_result(), [])),
        patch("agents_core.worktree.setup_worktree", return_value=_fake_handle(worktree)),
        patch("agents_core.worktree.teardown_worktree"),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        sr._run_local_fixer(spec, str(tmp_path))

    err = capsys.readouterr().err
    assert "DoormanUnreachable" in err or "doorman" in err.lower()
