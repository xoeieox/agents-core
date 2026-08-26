"""Tests for the local-opencode engine (agents-core-opencode-fixer-engine-v0).

13 tests:
  1. test_local_opencode_routing              - shaper routes engine local-opencode to
                                                ClaudeQueue with worktree_required=False
  2. test_local_opencode_dispatch_branch      - shaped_runner.main() dispatch branch calls
                                                _run_local_opencode (never claude/local-fixer)
  3. test_unknown_engine_fails_loud           - unknown engine -> SystemExit(2) + stderr
  4. test_opencode_tail_new_file_run          - untracked new file staged by the tail, PR opens
  5. test_opencode_tail_head_moved_no_pr      - model committed (HEAD moved) -> F2 fail-closed
  6. test_opencode_tail_remote_branch_no_pr   - model pushed the branch -> F2 orphan check
  7. test_opencode_tail_concluded_clean       - sessionID JSON-stream fallback + opencode export
  8. test_opencode_tail_deadline_salvage      - deadline kill + clean green diff -> salvage PR
  9. test_opencode_tail_gate_fails_no_pr      - failing touched test -> D1 gate fail-closed
  10. test_opencode_tail_empty_diff_no_pr     - only a lapis-spec.md copy on disk -> no PR
  11. test_opencode_process_group_kill        - killpg reaches the stub's whole process tree
  12. test_opencode_sessionid_degraded        - no sessionID anywhere -> degraded PR body
  13. test_gw_agent_fixer_untouched           - _run_local_fixer contract unchanged by the
                                                opencode engine

The engine under test is _run_local_opencode (shaped_runner.py): it runs the fixer under
`opencode run` in a real scratch git worktree, then re-derives the diff/test outcome from
the uncommitted worktree state and opens the PR itself (HARNESS-OWNS-GIT). The opencode
subprocess is stubbed via OPENCODE_BIN (an executable shell script); create_pr,
DoormanClient and room_path are stubbed. The opencode.db session query is
deterministic without a HOME redirect: its key is the exact title
(fixer-<task_id>) and the unique per-test task_ids have no row in the real db.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agents_core.shaper as shaper_mod
from agents_core.shaper import Shaper


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
def shaper_mocks_loco(tmp_path, monkeypatch):
    """Shaper with a local-opencode fixer agent, queues mocked (mirrors the
    local-fixer routing fixtures in test_shaper_routing.py)."""
    reg = _write_registry(tmp_path, {
        "fixer_loco": _agent_def("gravitywell-slot1", engine="local-opencode"),
    })
    monkeypatch.setattr(shaper_mod, "SPEC_DIR", tmp_path / "shaped")

    claude_q = MagicMock()
    claude_q._generate_id.return_value = "claude_task_id"
    claude_q.submit.side_effect = lambda payload, task_id=None: task_id
    gpu_q = MagicMock()
    gpu_q.submit.return_value = "gpu_task_id"

    monkeypatch.setattr(shaper_mod, "ClaudeQueue", lambda: claude_q)
    monkeypatch.setattr(shaper_mod, "GPUQueue", lambda: gpu_q)
    monkeypatch.delenv("AGENTS_CORE_FORCE_GPU_QUEUE", raising=False)
    monkeypatch.delenv("LAPIS_PM_FORCE_GPU_QUEUE", raising=False)

    s = Shaper(reg)
    monkeypatch.setattr(Shaper, "resolve_repo_cwd", staticmethod(lambda repo: "/tmp/fake-cwd"))
    return s, claude_q, gpu_q


# ---------------------------------------------------------------------------
# Real-git scratch repo + opencode stub (shared by the tail tests 4-12)
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_repo(tmp_path):
    """Real-git scratch environment: a tmp bare repo (plays `origin`) plus a
    real clone of it on branch `main` with f.py and tests/test_fast.py.

    The engine calls the REAL setup_worktree(), which does
    `git -C <clone> fetch origin <ref>` + `git worktree add --detach
    <wtroot>/<task_id> origin/<ref>` - so the clone needs a working `origin`
    remote. WORKTREE_ROOT is monkeypatched per test to a tmp dir so these
    tests never touch /tmp/lapis-pm-worktrees.
    """
    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(["git", "init", "--bare", str(origin)],
                   check=True, capture_output=True)

    seed = tmp_path / "seed"
    seed.mkdir()
    subprocess.run(["git", "-C", str(seed), "init", "-b", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    (seed / "f.py").write_text("VALUE = 0\n")
    tests_dir = seed / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_fast.py").write_text("def test_ok():\n    assert True\n")
    subprocess.run(["git", "-C", str(seed), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "commit", "-m", "seed"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "remote", "add", "origin", str(origin)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(seed), "push", "origin", "main"],
                   check=True, capture_output=True)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(origin), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "scratch"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email",
                    "scratch@example.com"], check=True, capture_output=True)
    return clone


# opencode stub, driven by a scenario file. The engine invokes it as:
#   <stub> run --auto --format json -m <model> --dir <cwd> --title fixer-<tid> <contract>
#   <stub> export <session_id>
# The scenario (single line in the scenario file) drives the `run` behavior:
#   plain       - create a new untracked file, exit 0 (no JSON events)
#   concluded   - create a new file + emit a sessionID JSON event, exit 0
#   head_moved  - create a file and COMMIT it (HEAD moves; contract violation)
#   remote_branch - push the target branch to origin (orphan; contract violation)
#   deadline    - create a file, then sleep past the loop budget
#   spawn       - spawn a long-running child in the same process group, sleep
#   gate_fail   - append a failing test to tests/test_fast.py
#   degraded    - create a file, emit a JSON event WITHOUT sessionID
#   spec_only   - create only a lapis-spec.md copy (staged-spec guard case)
_STUB_TEMPLATE = """#!/bin/sh
cmd="$1"
scen="$(cat '%s' 2>/dev/null)"
dir=""
prev=""
for a in "$@"; do
    if [ "$prev" = "--dir" ]; then dir="$a"; fi
    prev="$a"
done
case "$cmd" in
    export)
        echo '{"session_id": "ses_EXPORTED", "ok": true}'
        exit 0
        ;;
    run)
        case "$scen" in
            plain)
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                exit 0
                ;;
            concluded)
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                echo '{"type":"step_finish","sessionID":"ses_TEST123"}'
                exit 0
                ;;
            head_moved)
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                git -C "$dir" add -A
                git -C "$dir" commit -qm "model commit (HARNESS-OWNS-GIT violation)"
                echo '{"type":"step_finish","sessionID":"ses_HEADMOVED"}'
                exit 0
                ;;
            remote_branch)
                git -C "$dir" push -q origin "HEAD:refs/heads/%s"
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                echo '{"type":"step_finish","sessionID":"ses_REMOTE"}'
                exit 0
                ;;
            deadline)
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                sleep 60
                exit 0
                ;;
            spawn)
                python3 -c 'import os, time
with open(os.environ["STUB_CHILD_FILE"], "w") as f:
    f.write(str(os.getpid()))
time.sleep(120)' &
                sleep 60
                exit 0
                ;;
            gate_fail)
                printf 'def test_broken():\\n    assert False\\n' >> "$dir/tests/test_fast.py"
                echo '{"type":"step_finish","sessionID":"ses_GATEFAIL"}'
                exit 0
                ;;
            degraded)
                printf 'VALUE = 1\\n' > "$dir/new_mod.py"
                echo '{"type":"step_start"}'
                exit 0
                ;;
            spec_only)
                echo "bound spec body" > "$dir/lapis-spec.md"
                exit 0
                ;;
        esac
        exit 0
        ;;
esac
"""


def _write_opencode_stub(stub_dir: Path, scenario_file: Path, branch: str = "") -> Path:
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "opencode"
    stub.write_text(_STUB_TEMPLATE % (str(scenario_file), branch))
    stub.chmod(0o755)
    return stub


def _loco_spec(task_id: str, target_id: str, *, timeout_s: int = 600, **overrides) -> dict:
    spec = {
        "task_id": task_id,
        "target_id": target_id,
        "repo": "agents-core",
        "base_branch": "main",
        "slug": "loco",
        "prompt": "fix the thing",
        "timeout_s": timeout_s,
        "test_command": "tests/test_fast.py",
        "opencode_model": "gravitywell/gravitywell-slot1",
    }
    spec.update(overrides)
    return spec


def _run_engine(spec: dict, clone: Path, monkeypatch, tmp_path: Path,
                *, scenario: str, branch: str = ""):
    """Run _run_local_opencode with: a real scratch worktree (real git), the
    opencode subprocess stubbed via OPENCODE_BIN, and create_pr / DoormanClient
    / room_path stubbed. Returns (pr_url, mock_create_pr).

    The S4g sessionID DB query runs against the real ~/.local/share/opencode/
    opencode.db (no test knob for the path). Its key is the exact title the
    harness passed (fixer-<task_id>); the unique per-test task_ids here
    guarantee that row is absent, so the query deterministically falls through
    to the JSON-stream fallback (tests 7/12 assert that fallback + degraded).
    """
    from agents_core import shaped_runner, worktree

    scenario_file = tmp_path / "scenario.txt"
    scenario_file.write_text(scenario + "\n")
    stub = _write_opencode_stub(tmp_path / "bin", scenario_file, branch=branch)

    monkeypatch.setenv("OPENCODE_BIN", str(stub))
    monkeypatch.setattr(worktree, "WORKTREE_ROOT", tmp_path / "wtroot")
    monkeypatch.setattr(shaped_runner, "room_path",
                        lambda key, *parts, **kw: tmp_path / "artifacts")

    with patch("agents_core.doorman_client.DoormanClient") as mock_dm, \
         patch("agents_core.forgejo.create_pr",
               return_value={"html_url":
                             "http://forgejo/Erah/agents-core/pulls/77"}) as mock_pr:
        mock_dm.return_value.acquire.return_value = {
            "status": "serving", "work_id": f"{spec['task_id']}-berth-sup",
        }
        url = shaped_runner._run_local_opencode(spec, str(clone))
    return url, mock_pr


# ---------------------------------------------------------------------------
# 1-3: routing, dispatch branch, unknown-engine fail-loud
# ---------------------------------------------------------------------------

def test_local_opencode_routing(shaper_mocks_loco, tmp_path):
    """engine local-opencode routes to ClaudeQueue (like local-fixer) and must
    NOT set worktree_required=True: the engine manages its own worktree."""
    s, claude_q, gpu_q = shaper_mocks_loco
    s.dispatch("fixer_loco", "t-loco", "fix it", vars_={"repo": "agents-core"})
    assert claude_q.submit.called, "local-opencode must route to ClaudeQueue"
    assert not gpu_q.submit.called, "local-opencode must not touch GPUQueue"
    written = list((tmp_path / "shaped").glob("*.json"))
    assert len(written) == 1
    spec = json.loads(written[0].read_text())
    assert spec["engine"] == "local-opencode"
    assert spec["worktree_required"] is False, \
        "local-opencode must not set worktree_required=True (manages its own worktree)"


def test_local_opencode_dispatch_branch(tmp_path):
    """shaped_runner.main() with engine=local-opencode calls _run_local_opencode
    (with the loaded spec + cwd) and never _run_local_fixer or call_claude_cli."""
    spec = {
        "model": "gravitywell-slot1",
        "engine": "local-opencode",
        "prompt": "fix the bug",
        "timeout_s": 60,
        "task_id": "abc-loco",
        "target_id": "my-target-v0",
        "repo": "agents-core",
        "cwd": str(tmp_path),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    import agents_core.shaped_runner as sr

    with (
        patch.object(sys, "argv", ["sr", str(spec_path)]),
        patch("agents_core.shaped_runner._run_local_opencode",
              return_value="http://1.2.3.4:3000/e/r/pulls/7") as mock_lo,
        patch("agents_core.shaped_runner._run_local_fixer") as mock_lf,
        patch("agents_core.shaped_runner.call_claude_cli") as mock_cli,
        patch("sys.stdout"),
    ):
        sr.main()

    mock_lo.assert_called_once()
    mock_lf.assert_not_called()
    mock_cli.assert_not_called()
    assert mock_lo.call_args.args[0]["engine"] == "local-opencode"
    assert mock_lo.call_args.args[1] == str(tmp_path)


def test_unknown_engine_fails_loud(tmp_path, capsys):
    """An engine that is not claude / local-fixer / local-opencode /
    local-reviewer must exit(2) with a loud stderr message (no silent
    call_claude_cli fall-through)."""
    spec = {
        "model": "sonnet",
        "engine": "bogus-engine",
        "prompt": "do it",
        "timeout_s": 30,
        "task_id": "task-bogus",
        "target_id": "t-bogus",
        "repo": "agents-core",
        "cwd": str(tmp_path),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    import agents_core.shaped_runner as sr

    with patch.object(sys, "argv", ["sr", str(spec_path)]):
        with pytest.raises(SystemExit) as excinfo:
            sr.main()

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unknown shaped-runner engine" in err
    assert "bogus-engine" in err


# ---------------------------------------------------------------------------
# 4-6: deterministic tail re-derivation (F2 integrity, F1 staging)
# ---------------------------------------------------------------------------

def test_opencode_tail_new_file_run(scratch_repo, tmp_path, monkeypatch):
    """A run whose deliverable is an UNTRACKED new file: the tail's
    `git add -A` picks it up (a bare `git diff HEAD` would not), the PR opens
    on lapis/<target>/<slug>, and the pushed commit actually carries the file."""
    spec = _loco_spec("task-newfile", "tgt-newfile")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="plain")

    assert url == "http://forgejo/Erah/agents-core/pulls/77"
    mock_pr.assert_called_once()
    kw = mock_pr.call_args.kwargs
    assert kw["repo"] == "agents-core"
    assert kw["head"] == "lapis/tgt-newfile/loco"
    assert kw["base"] == "main"
    assert kw["title"] == "fix(tgt-newfile): local-opencode"
    assert "<!-- lapis-engine: local-opencode -->" in kw["body"]
    assert "<!-- lapis-gpu-id: task-newfile -->" in kw["body"]
    assert "<!-- lapis-tid: tgt-newfile -->" in kw["body"]

    # The branch was really pushed to origin and carries the new file.
    origin = scratch_repo.parent / "origin.git"
    r = subprocess.run(
        ["git", "-C", str(origin), "show", "lapis/tgt-newfile/loco:new_mod.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert r.stdout == "VALUE = 1\n"


def test_opencode_tail_head_moved_no_pr(scratch_repo, tmp_path, monkeypatch, capsys):
    """The model committed inside the worktree (HEAD moved off base_sha): the
    F2 integrity check fails closed - no PR, distinct WARN."""
    spec = _loco_spec("task-headmoved", "tgt-headmoved")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="head_moved")

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert "HEAD moved off base_sha" in err


def test_opencode_tail_remote_branch_no_pr(scratch_repo, tmp_path, monkeypatch, capsys):
    """The model pushed the target branch to origin (orphan) without
    committing locally: HEAD is still at base_sha and the diff is clean, so
    the F2 remote-branch check (fresh-fixer case only) fails closed - no PR."""
    branch = "lapis/tgt-remote/loco"
    spec = _loco_spec("task-remote", "tgt-remote")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="remote_branch", branch=branch)

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert "remote branch" in err and "already exists on origin" in err
    assert "model-pushed orphan" in err


# ---------------------------------------------------------------------------
# 7-10: terminal states (concluded, deadline salvage, D1 gate, empty diff)
# ---------------------------------------------------------------------------

def test_opencode_tail_concluded_clean(scratch_repo, tmp_path, monkeypatch):
    """Clean concluded run (exit 0): the JSON event stream provides the
    sessionID (S4g fallback - the unique task_id has no opencode.db row),
    the engine runs `opencode export <session_id>` and persists its stdout,
    and the PR body carries the session provenance."""
    spec = _loco_spec("task-concluded", "tgt-concluded")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="concluded")

    assert url == "http://forgejo/Erah/agents-core/pulls/77"
    mock_pr.assert_called_once()
    body = mock_pr.call_args.kwargs["body"]
    assert "sessionID: ses_TEST123" in body
    assert "session record unavailable" not in body
    assert "## Terminal state\n\nconcluded" in body

    # The export subcommand ran and its stdout was persisted to the
    # artifact dir as <task_id>-opencode-session.json.
    export_file = tmp_path / "artifacts" / "task-concluded-opencode-session.json"
    assert export_file.exists()
    assert "ses_EXPORTED" in export_file.read_text()
    assert f"session export: `{export_file}`" in body


def test_opencode_tail_deadline_salvage(scratch_repo, tmp_path, monkeypatch, capsys):
    """The model loop exceeds its budget (timeout_s - TAIL_BUDGET -
    setup_lag): the harness kills the process group, and whatever is on disk
    is re-derived. A clean diff + green targeted run is SALVAGED as a PR
    with the mandatory human sign-off marker (S4e)."""
    # timeout_s=310 -> loop_budget ~ 10s (TAIL_BUDGET=300, setup_lag ~0);
    # the stub sleeps 60s so the deadline kill is guaranteed.
    spec = _loco_spec("task-deadline", "tgt-deadline", timeout_s=310)
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="deadline")

    assert url == "http://forgejo/Erah/agents-core/pulls/77"
    mock_pr.assert_called_once()
    body = mock_pr.call_args.kwargs["body"]
    assert "## Terminal state\n\nsalvaged (deadline kill)" in body
    assert "<!-- lapis-no-progress-salvage: true -->" in body
    err = capsys.readouterr().err
    assert "killing the opencode process group" in err


def test_opencode_tail_gate_fails_no_pr(scratch_repo, tmp_path, monkeypatch, capsys):
    """The diff touches a test file and that test FAILS in the deterministic
    targeted run: the D1 positive-only gate fails closed - no PR."""
    spec = _loco_spec("task-gatefail", "tgt-gatefail")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="gate_fail")

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert "test gate failed" in err
    # The touched test file was the source of the targeted run.
    assert "tests/test_fast.py" in err


def test_opencode_tail_empty_diff_no_pr(scratch_repo, tmp_path, monkeypatch, capsys):
    """The run leaves ONLY a lapis-spec.md copy on disk (untracked at HEAD):
    the staged-spec guard unstages it, the diff vs HEAD is empty -> no PR.
    This is the 'a spec hunk never pollutes final_diff' path."""
    spec = _loco_spec("task-speconly", "tgt-speconly")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="spec_only")

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert "empty diff" in err


# ---------------------------------------------------------------------------
# 11-13: process-group kill, degraded session record, local-fixer untouched
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_opencode_process_group_kill(scratch_repo, tmp_path, monkeypatch, capsys):
    """The stub spawns a long-running child in the SAME process group
    (start_new_session=True makes the stub's pid the pgid; the child
    inherits it) and outlives the loop budget. The deadline killpg must
    reach the WHOLE tree - the child must be dead after the engine
    returns, not just the stub."""
    child_file = tmp_path / "child-pid"
    monkeypatch.setenv("STUB_CHILD_FILE", str(child_file))
    # timeout_s=310 -> loop_budget ~ 10s; stub (and its child) sleep 60-120s.
    spec = _loco_spec("task-spawn", "tgt-spawn", timeout_s=310)
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="spawn")

    assert url == ""
    mock_pr.assert_not_called()
    err = capsys.readouterr().err
    assert "killing the opencode process group" in err

    # The stub's child wrote its own pid before sleeping; it must now be
    # dead (SIGKILL via the process group), even though it would have
    # outlived the engine by minutes.
    assert child_file.exists(), "stub child never ran (pid file missing)"
    child_pid = int(child_file.read_text().strip())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _pid_alive(child_pid):
        time.sleep(0.05)
    assert not _pid_alive(child_pid), \
        f"stub child {child_pid} survived the process-group kill"


def test_opencode_sessionid_degraded(scratch_repo, tmp_path, monkeypatch):
    """No sessionID in the event stream AND no opencode.db row for the
    title: the PR still opens (session provenance is never a gate), but the
    body carries the explicit 'session record unavailable' degraded text."""
    spec = _loco_spec("task-degraded", "tgt-degraded")
    url, mock_pr = _run_engine(spec, scratch_repo, monkeypatch, tmp_path,
                               scenario="degraded")

    assert url == "http://forgejo/Erah/agents-core/pulls/77"
    mock_pr.assert_called_once()
    body = mock_pr.call_args.kwargs["body"]
    assert "session record unavailable" in body
    assert "session export: unavailable" in body
    assert "sessionID: " not in body
    # The PR opens anyway - degraded provenance never blocks.
    assert "## Terminal state\n\nconcluded" in body


def test_gw_agent_fixer_untouched(tmp_path, monkeypatch):
    """Invariant: _run_local_fixer's contract is UNCHANGED by the
    local-opencode engine. Same provenance title/body, and the engine never
    reads OPENCODE_BIN even when it is set (pointed at a non-existent
    binary here to prove it is not invoked)."""
    monkeypatch.setenv("OPENCODE_BIN", str(tmp_path / "no-such-opencode"))

    from agents_core import shaped_runner

    spec = {
        "task_id": "task-lf-untouched",
        "target_id": "tgt-lf-untouched",
        "repo": "agents-core",
        "prompt": "fix it",
        "timeout_s": 600,
        "slug": "local",
    }
    worktree = tmp_path / "wt"
    worktree.mkdir()
    good_result = {
        "final_diff": "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
                      "@@ -1 +1 @@\n-old\n+new\n",
        "concluded": True,
        "last_test_outcome": {"passed": 3, "failed": 0},
        "steps": [{"tool": "read_file"}],
    }

    fake_handle = MagicMock()
    fake_handle.path = worktree
    fake_handle.env = {}

    with (
        patch("agents_core.gw_agent.call_gw_agent",
              return_value=(good_result, [])),
        patch("agents_core.doorman_client.DoormanClient") as MockClient,
        patch("agents_core.worktree.setup_worktree",
              return_value=fake_handle),
        patch("agents_core.worktree.teardown_worktree"),
        patch("agents_core.forgejo.create_pr",
              return_value={"html_url": "http://forgejo/pulls/42"}) as mock_pr,
        patch("subprocess.run",
              return_value=MagicMock(returncode=0, stderr="")),
        patch.object(Path, "mkdir"),
        patch.object(Path, "write_text"),
    ):
        MockClient.return_value.acquire.return_value = {
            "status": "serving", "work_id": "task-lf-untouched-berth-sup",
        }
        url = shaped_runner._run_local_fixer(spec, str(tmp_path))

    assert url == "http://forgejo/pulls/42"
    mock_pr.assert_called_once()
    kw = mock_pr.call_args.kwargs
    assert kw["title"] == "fix(tgt-lf-untouched): local-fixer"
    assert "local 122B fixer harness" in kw["body"]
    assert "local-opencode" not in kw["body"]
    assert "<!-- lapis-gpu-id: task-lf-untouched -->" in kw["body"]
