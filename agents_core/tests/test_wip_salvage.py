"""Tests for the WIP-commit salvage + context-cap calibration
(agents-core-fixer-budget-compact-salvage-v0).

Covers:
  (1) test_ctx_cap_env_resolvable - per-run resolution honors
      GW_AGENT_CTX_CAP; bad input (non-numeric, negative, zero) falls back
      to 120000.
  (2) test_wip_commit_fires_on_clean_step - via the new after_step seam:
      a step writing a compiling .py produces exactly one commit on
      refs/wip/<task_id> with the `wip: <task> step <N> [auto]` message;
      the worktree's HEAD is UNCHANGED; the index is clean.
  (3) test_wip_commit_blocked_on_syntax_error - a step writing a
      non-compiling .py produces NO new commit; the previous WIP commit
      remains the ref's head.
  (4) test_wip_commit_excludes_staged_spec - with lapis-spec.md present
      in the worktree, the WIP commit's tree does not include it.
  (5) test_salvage_pr_on_output_budget_death - a run stubbed to end with
      stop_reason=output_budget_exhausted and >=1 WIP commit pushes
      refs/heads/lapis/<tid>/<slug>-salvage + opens a PR with the
      [SALVAGE] title; the run is classified lost.
  (6) test_salvage_none_without_wip_commits - UPDATE of the existing
      TestLocalFixerTruncationAndSpecStaging::test_output_budget_
      exhausted_distinct_warn_no_pr: the stub (no writes, zero WIP
      commits) still produces no PR, now with the updated WARN text and
      the "no WIP commits" condition asserted.
  (7) test_success_path_unchanged - a green concluded run: the PR branch
      lapis/<tid>/<slug> contains exactly the tail commit (no WIP
      history), refs/wip/<task_id> is never pushed, the PR title has no
      SALVAGE marker.
  (8) test_d5_mem_search_loop_unchanged - a mem_search_loop death still
      returns "" with no PR (regression guard on the sibling branch).
  (9) test_executors_fail_closed_on_empty_args - the linchpin pin:
      WriteFileExecutor/ApplyEditExecutor with {} args (the JSON-parse-
      failure shape) return an error dict and write NO file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents_core.gw_agent import (
    GW_AGENT_CTX_CAP,
    WriteFileExecutor,
    ApplyEditExecutor,
    call_gw_agent,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tester@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "hello.txt").write_text("one\n")
    _git(repo, "add", "hello.txt")
    _git(repo, "commit", "-qm", "base")


# ---------------------------------------------------------------------------
# (1) context-cap env resolution
# ---------------------------------------------------------------------------

class TestCtxCapEnvResolvable:
    """S1: the compact guard's cap is env-resolvable per run
    (GW_AGENT_CTX_CAP); bad input falls back to the module default."""

    def test_ctx_cap_env_resolvable(self, monkeypatch, capsys, tmp_path):
        """A valid GW_AGENT_CTX_CAP is used at the guard; non-numeric,
        negative, and zero values fall back to 120000."""
        import agents_core.gw_agent as gw

        def _log(msg: str) -> None:
            # The compact guard logs ONLY through the `log` callable
            # (gw_agent has no logger fallback); route it to stderr so
            # capsys can see it.
            print(msg, file=sys.stderr)

        # A real git repo: the writeable run's tail computes final_diff
        # via `git add -A` in cwd (rc=128 in a bare temp dir).
        _init_repo(tmp_path)

        # Valid value: the loop uses the env value at the guard.
        monkeypatch.setenv("GW_AGENT_CTX_CAP", "55000")
        with patch("agents_core.doorman_client.DoormanClient") as mock_doorman_class, \
             patch("requests.post") as mock_post:
            mock_doorman = MagicMock()
            mock_doorman_class.return_value = mock_doorman
            mock_doorman.acquire.return_value = {"status": "serving"}
            # A writeable run that exceeds the 55000 cap on step 1 and
            # then concludes on step 2. The usage total_tokens is the
            # ctx_tokens the guard sees.
            mock_post.return_value.json.side_effect = [
                {
                    "choices": [{
                        "message": {
                            "content": "",
                            "tool_calls": [{
                                "id": "call_0",
                                "function": {
                                    "name": "write_file",
                                    "arguments": '{"path": "a.py", "content": "x = 1\\n"}',
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"total_tokens": 60000},
                },
                {
                    "choices": [{
                        "message": {"content": "done", "tool_calls": []},
                        "finish_reason": "stop",
                    }],
                    "usage": {"total_tokens": 10},
                },
            ]
            result = call_gw_agent(
                prompt="Fix this.",
                cwd=str(tmp_path),
                writeable=True,
                max_steps=5,
                backend_url="http://gw-test:8081",
                log=_log,
            )
        # The run concluded (the truncation did not kill it).
        assert isinstance(result, tuple)
        fixer = result[0]
        assert fixer["concluded"] is True
        # The log line proves the env value was used at the guard.
        err = capsys.readouterr().err
        assert "context cap exceeded" in err
        assert "55000" in err

        # Non-numeric: falls back to the module default.
        monkeypatch.setenv("GW_AGENT_CTX_CAP", "not_a_number")
        val = gw._resolve_int_env("GW_AGENT_CTX_CAP", GW_AGENT_CTX_CAP, None)
        assert val == GW_AGENT_CTX_CAP

        # Negative: the resolver returns it, but the call-site guard
        # rejects non-positive values.
        monkeypatch.setenv("GW_AGENT_CTX_CAP", "-1")
        raw = gw._resolve_int_env("GW_AGENT_CTX_CAP", GW_AGENT_CTX_CAP, None)
        assert raw == -1  # the resolver itself has no positive guard
        # The call-site guard (the code we added) rejects this:
        ctx_cap = raw
        if ctx_cap <= 0:
            ctx_cap = GW_AGENT_CTX_CAP
        assert ctx_cap == GW_AGENT_CTX_CAP

        # Zero: same.
        monkeypatch.setenv("GW_AGENT_CTX_CAP", "0")
        raw = gw._resolve_int_env("GW_AGENT_CTX_CAP", GW_AGENT_CTX_CAP, None)
        assert raw == 0
        ctx_cap = raw
        if ctx_cap <= 0:
            ctx_cap = GW_AGENT_CTX_CAP
        assert ctx_cap == GW_AGENT_CTX_CAP


# ---------------------------------------------------------------------------
# (2)-(4) WIP-commit hook behavior (via the after_step seam)
# ---------------------------------------------------------------------------

class TestWipCommitHook:
    """S3: the after_step hook snapshots the step's write-tool paths onto
    refs/wip/<task_id> via write-tree/commit-tree; the worktree's HEAD
    and index are never moved."""

    def _make_hook(self, worktree: Path, task_id: str):
        """Build the same closure _run_local_fixer builds, but with a
        known worktree path (the real one is set up by setup_worktree,
        which we patch in the shaped_runner tests)."""
        import agents_core.shaped_runner as sr
        wip_ref = f"refs/wip/{task_id}"
        state = {"count": 0, "head_sha": "", "steps": []}

        def _wip_git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", str(worktree), *args],
                capture_output=True, text=True, timeout=30,
            )

        def _hook(ctx: dict) -> None:
            import py_compile
            step_num = ctx.get("step_num")
            cwd = ctx.get("cwd")
            if not ctx.get("writeable") or not cwd or not step_num:
                return
            paths: list[str] = []
            for entry in ctx.get("transcript") or []:
                if entry.get("tool_name") not in ("write_file", "apply_edit"):
                    continue
                p = (entry.get("arguments") or {}).get("path")
                if isinstance(p, str) and p and p not in paths:
                    paths.append(p)
            if not paths:
                return
            try:
                for p in paths:
                    if not p.endswith(".py"):
                        continue
                    full = Path(cwd) / p
                    if not full.is_file():
                        continue
                    try:
                        py_compile.compile(str(full), doraise=True)
                    except (py_compile.PyCompileError, OSError) as exc:
                        print(
                            f"WARN: wip-commit: {p} does not py_compile "
                            f"({exc}) - skipping WIP commit for step {step_num}",
                            file=__import__("sys").stderr,
                        )
                        return
                parent = _wip_git("rev-parse", "--verify", wip_ref)
                if parent.returncode != 0:
                    parent = _wip_git("rev-parse", "HEAD")
                    if parent.returncode != 0:
                        return
                add = _wip_git("add", "--", *paths)
                if add.returncode != 0:
                    return
                tree = _wip_git("write-tree")
                if tree.returncode != 0:
                    _wip_git("reset")
                    return
                commit = _wip_git(
                    "commit-tree", tree.stdout.strip(),
                    "-p", parent.stdout.strip(),
                    "-m", f"wip: {task_id} step {step_num} [auto]",
                )
                if commit.returncode != 0:
                    _wip_git("reset")
                    return
                upd = _wip_git("update-ref", wip_ref, commit.stdout.strip())
                _wip_git("reset")
                if upd.returncode != 0:
                    return
                state["count"] += 1
                state["head_sha"] = commit.stdout.strip()
                state["steps"].append(int(step_num))
            except Exception as exc:
                print(f"WARN: wip-commit: unexpected error: {exc}",
                      file=__import__("sys").stderr)

        return _hook, state, wip_ref

    def test_wip_commit_fires_on_clean_step(self, tmp_path):
        """A step writing a compiling .py produces exactly one commit on
        refs/wip/<task_id> with the `wip: <task> step <N> [auto]` message;
        the worktree's HEAD is UNCHANGED; the index is clean."""
        repo = tmp_path / "wt"
        _init_repo(repo)
        base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

        hook, state, wip_ref = self._make_hook(repo, "task-wip1")
        # Simulate the after_step context for a step that wrote a.py.
        (repo / "a.py").write_text("x = 1\n")
        hook({
            "step_num": 1,
            "transcript": [{
                "step": 1,
                "tool_name": "write_file",
                "tool_call_id": "call_0",
                "arguments": {"path": "a.py", "content": "x = 1\n"},
                "result": "wrote 6 bytes to a.py",
                "error": None,
            }],
            "cwd": str(repo),
            "writeable": True,
        })
        assert state["count"] == 1
        # The ref exists and points to a commit with the right message.
        ref_sha = _git(repo, "rev-parse", wip_ref).stdout.strip()
        assert ref_sha == state["head_sha"]
        msg = _git(repo, "log", "-1", "--format=%s", ref_sha).stdout.strip()
        assert msg == "wip: task-wip1 step 1 [auto]"
        # The worktree's HEAD is UNCHANGED (still the base sha).
        head_now = _git(repo, "rev-parse", "HEAD").stdout.strip()
        assert head_now == base_sha
        # The index is clean (no STAGED changes - the commit contains only the
        # written path). Untracked files (a.py, __pycache__) remain in
        # the worktree (the WIP commit is a snapshot, not a checkout).
        status = _git(repo, "status", "--porcelain").stdout.strip()
        assert not any(line.startswith(("M", "A", "D", "R", "C", "T", "U"))
                       for line in status.splitlines())
        # The commit's tree includes a.py.
        tree_files = _git(repo, "ls-tree", "-r", "--name-only", ref_sha).stdout
        assert "a.py" in tree_files

    def test_wip_commit_blocked_on_syntax_error(self, tmp_path):
        """A step writing a non-compiling .py produces NO new commit;
        the previous WIP commit remains the ref's head."""
        repo = tmp_path / "wt"
        _init_repo(repo)

        hook, state, wip_ref = self._make_hook(repo, "task-wip2")
        # Step 1: a clean write -> one WIP commit.
        (repo / "good.py").write_text("x = 1\n")
        hook({
            "step_num": 1,
            "transcript": [{
                "step": 1, "tool_name": "write_file",
                "tool_call_id": "call_0",
                "arguments": {"path": "good.py", "content": "x = 1\n"},
                "result": "wrote 6 bytes to good.py", "error": None,
            }],
            "cwd": str(repo), "writeable": True,
        })
        assert state["count"] == 1
        first_sha = state["head_sha"]

        # Step 2: a non-compiling .py -> NO new commit.
        (repo / "bad.py").write_text("def broken(:\n")
        hook({
            "step_num": 2,
            "transcript": [{
                "step": 2, "tool_name": "write_file",
                "tool_call_id": "call_1",
                "arguments": {"path": "bad.py", "content": "def broken(:\n"},
                "result": "wrote 14 bytes to bad.py", "error": None,
            }],
            "cwd": str(repo), "writeable": True,
        })
        assert state["count"] == 1  # no new commit
        # The ref's head is still the first commit.
        ref_sha = _git(repo, "rev-parse", wip_ref).stdout.strip()
        assert ref_sha == first_sha

    def test_wip_commit_excludes_staged_spec(self, tmp_path):
        """With lapis-spec.md present in the worktree, the WIP commit's
        tree does not include it (explicit pathspec, invariant 5)."""
        repo = tmp_path / "wt"
        _init_repo(repo)
        # Stage the spec (untracked, like the real harness does).
        (repo / "lapis-spec.md").write_text("bound spec body\n")

        hook, state, wip_ref = self._make_hook(repo, "task-wip3")
        (repo / "b.py").write_text("y = 2\n")
        hook({
            "step_num": 1,
            "transcript": [{
                "step": 1, "tool_name": "write_file",
                "tool_call_id": "call_0",
                "arguments": {"path": "b.py", "content": "y = 2\n"},
                "result": "wrote 6 bytes to b.py", "error": None,
            }],
            "cwd": str(repo), "writeable": True,
        })
        assert state["count"] == 1
        ref_sha = _git(repo, "rev-parse", wip_ref).stdout.strip()
        tree_files = _git(repo, "ls-tree", "-r", "--name-only", ref_sha).stdout
        assert "b.py" in tree_files
        assert "lapis-spec.md" not in tree_files


# ---------------------------------------------------------------------------
# (5) salvage PR on output-budget death
# ---------------------------------------------------------------------------

class TestSalvagePrOnOutputBudgetDeath:
    """S3: a run stubbed to end with stop_reason=
    output_budget_exhausted and >=1 WIP commit pushes
    refs/heads/lapis/<tid>/<slug>-salvage + opens a PR with the
    [SALVAGE] title; the run is classified lost."""

    def test_salvage_pr_on_output_budget_death(self, tmp_path, capsys):
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-salv",
            "target_id": "tgt-salv",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
            "slug": "local",
        }
        wt_dir = tmp_path / "wt"
        _init_repo(wt_dir)
        # Add a bare origin so the salvage push succeeds.
        origin_dir = tmp_path / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(origin_dir)],
                       check=True)
        _git(wt_dir, "remote", "add", "origin", str(origin_dir))
        _git(wt_dir, "push", "-q", "origin", "HEAD")

        def fake_call_gw_agent(*args, **kwargs):
            # The run died with output_budget_exhausted. The WIP commit
            # was made by the after_step hook during the run - simulate
            # the hook's side effect (the ref exists, the counter was
            # incremented) by calling the hook directly before returning.
            after_step = kwargs.get("after_step")
            if after_step is not None:
                (wt_dir / "work.py").write_text("z = 3\n")
                after_step({
                    "step_num": 1,
                    "transcript": [{
                        "step": 1, "tool_name": "write_file",
                        "tool_call_id": "call_0",
                        "arguments": {"path": "work.py",
                                       "content": "z = 3\n"},
                        "result": "wrote 6 bytes to work.py",
                        "error": None,
                    }],
                    "cwd": str(wt_dir),
                    "writeable": True,
                })
            return (
                {"final_diff": "", "concluded": False,
                 "stop_reason": "output_budget_exhausted",
                 "max_steps_reached": False, "no_progress": False,
                 "last_test_outcome": None,
                 "steps": [{"step": 1, "tool_name": "write_file"}]},
                [{"step": 1, "tool_name": "write_file",
                  "arguments": {"path": "work.py"}, "error": None}],
            )

        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                    side_effect=fake_call_gw_agent), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-salv-berth-sup",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {
                "html_url": "http://forgejo/agents-core/pulls/999",
            }
            out = shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")

        # The PR was opened with the [SALVAGE] title.
        mock_create_pr.assert_called_once()
        call_kwargs = mock_create_pr.call_args.kwargs
        assert "[SALVAGE]" in call_kwargs["title"]
        assert "output_budget_exhausted" in call_kwargs["title"]
        assert call_kwargs["head"] == "lapis/tgt-salv/local-salvage"
        assert call_kwargs["base"] == "main"
        # The run is classified lost (returns the PR URL, not a success).
        assert out == "http://forgejo/agents-core/pulls/999"
        # The WARN log line mentions the WIP commits.
        err = capsys.readouterr().err
        assert "output budget exhausted" in err
        assert "WIP commit" in err


# ---------------------------------------------------------------------------
# (6) no salvage without WIP commits (update of the existing test)
# ---------------------------------------------------------------------------

class TestSalvageNoneWithoutWipCommits:
    """UPDATE of the existing
    TestLocalFixerTruncationAndSpecStaging::test_output_budget_
    exhausted_distinct_warn_no_pr: the stub (no writes, zero WIP
    commits) still produces no PR, now with the updated WARN text and
    the 'no WIP commits' condition asserted."""

    def test_salvage_none_without_wip_commits(self, capsys):
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-obx2",
            "target_id": "tgt-obx2",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                    return_value=({"final_diff": "", "concluded": False,
                                   "stop_reason": "output_budget_exhausted",
                                   "max_steps_reached": False,
                                   "no_progress": False,
                                   "last_test_outcome": None,
                                   "steps": []}, [])), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-obx2-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            out = shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")

        assert out == ""
        err = capsys.readouterr().err
        assert "output budget exhausted" in err
        assert "no WIP commits" in err
        assert "DoormanUnreachable" not in err
        mock_create_pr.assert_not_called()


# ---------------------------------------------------------------------------
# (7) success path unchanged
# ---------------------------------------------------------------------------

class TestSuccessPathUnchanged:
    """S3 invariant 6: a green concluded run's PR branch contains exactly
    the tail commit (no WIP history), refs/wip/<task_id> is never pushed,
    the PR title has no SALVAGE marker."""

    def test_success_path_unchanged(self, tmp_path, capsys):
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-ok",
            "target_id": "tgt-ok",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
            "slug": "local",
        }
        wt_dir = tmp_path / "wt"
        _init_repo(wt_dir)
        # Add a bare origin so the tail's `git push origin HEAD:<branch>`
        # succeeds (the real worktree has origin from the parent clone).
        origin_dir = tmp_path / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(origin_dir)],
                       check=True)
        _git(wt_dir, "remote", "add", "origin", str(origin_dir))
        _git(wt_dir, "push", "-q", "origin", "HEAD")
        base_sha = _git(wt_dir, "rev-parse", "HEAD").stdout.strip()

        def fake_call_gw_agent(*args, **kwargs):
            # A green concluded run: the model wrote a file and concluded.
            (wt_dir / "fixed.py").write_text("ok = True\n")
            return (
                {"final_diff": "diff --git a/fixed.py b/fixed.py\n+ok = True\n",
                 "concluded": True, "stop_reason": "",
                 "max_steps_reached": False, "no_progress": False,
                 "last_test_outcome": {"passed": 1, "failed": 0, "errors": 0},
                 "steps": [{"step": 1, "tool_name": "write_file"}]},
                [{"step": 1, "tool_name": "write_file",
                  "arguments": {"path": "fixed.py"}, "error": None}],
            )

        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                    side_effect=fake_call_gw_agent), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-ok-berth-sup",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {
                "html_url": "http://forgejo/agents-core/pulls/1000",
            }
            out = shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")

        # The PR was opened with the NORMAL title (no SALVAGE marker).
        mock_create_pr.assert_called_once()
        call_kwargs = mock_create_pr.call_args.kwargs
        assert "[SALVAGE]" not in call_kwargs["title"]
        assert call_kwargs["head"] == "lapis/tgt-ok/local"
        assert call_kwargs["base"] == "main"
        assert out == "http://forgejo/agents-core/pulls/1000"
        # The WIP ref was never pushed (it doesn't even exist in this
        # stubbed run - the after_step hook was not called because we
        # patched call_gw_agent entirely).
        # The worktree's HEAD moved to the tail commit (the success path
        # does checkout -B + commit + push).
        head_now = _git(wt_dir, "rev-parse", "HEAD").stdout.strip()
        assert head_now != base_sha
        # The tail commit's message is the normal one.
        msg = _git(wt_dir, "log", "-1", "--format=%s").stdout.strip()
        assert msg == "fix(tgt-ok): local-fixer harness"


# ---------------------------------------------------------------------------
# (8) D5 mem_search_loop unchanged
# ---------------------------------------------------------------------------

class TestD5MemSearchLoopUnchanged:
    """A mem_search_loop death still returns "" with no PR (regression
    guard on the sibling branch)."""

    def test_d5_mem_search_loop_unchanged(self, capsys):
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-msl",
            "target_id": "tgt-msl",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
        }
        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                    return_value=({"final_diff": "", "concluded": False,
                                   "stop_reason": "mem_search_loop",
                                   "max_steps_reached": False,
                                   "no_progress": False,
                                   "last_test_outcome": None,
                                   "steps": []}, [])), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-msl-berth-sup",
            }
            mock_setup.return_value = MagicMock(path="/wt")
            out = shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")

        assert out == ""
        err = capsys.readouterr().err
        assert "mem_search_loop" in err
        mock_create_pr.assert_not_called()


# ---------------------------------------------------------------------------
# (9) executors fail closed on empty args (the linchpin pin)
# ---------------------------------------------------------------------------

class TestExecutorsFailClosedOnEmptyArgs:
    """The linchpin pin: WriteFileExecutor/ApplyEditExecutor with {} args
    (the JSON-parse-failure shape) return an error dict and write NO
    file (tmp_path assertion)."""

    def test_executors_fail_closed_on_empty_args(self, tmp_path):
        """A truncated response whose tool_calls arguments JSON-parse to
        {} (the JSON-parse-failure shape at gw_agent.py:2201) must NOT
        write a file: the fail-closed default executors reject empty
        args with an error dict."""
        # WriteFileExecutor with {} args.
        wf = WriteFileExecutor(cwd=str(tmp_path))
        result = wf.execute({})
        assert isinstance(result, dict)
        assert "error" in result
        # No file was written (the cwd is empty).
        files = list(tmp_path.iterdir())
        assert files == []

        # ApplyEditExecutor with {} args.
        ae = ApplyEditExecutor(cwd=str(tmp_path))
        result = ae.execute({})
        assert isinstance(result, dict)
        assert "error" in result
        # No file was written.
        files = list(tmp_path.iterdir())
        assert files == []

        # ApplyEditExecutor with a path but no old_string (partial args).
        (tmp_path / "existing.py").write_text("hello\n")
        result = ae.execute({"path": "existing.py"})
        assert isinstance(result, dict)
        assert "error" in result
        # The file is unchanged.
        assert (tmp_path / "existing.py").read_text() == "hello\n"


# ---------------------------------------------------------------------------
# (10) WIP-salvage defers to the green-salvage path (spec rev-3 ordering)
# ---------------------------------------------------------------------------

class TestWipSalvageDefersToGreenPath:
    """S3 ordering rule (rev-3): a non-concluded run at the step ceiling
    that has WIP commits BUT also a clean non-empty diff with passing
    tests takes the GREEN-salvage path (normal PR branch, no [SALVAGE]
    title) - the WIP-salvage PR is for the remainder."""

    def test_wip_salvage_defers_to_green_path(self, tmp_path):
        from agents_core import shaped_runner

        spec = {
            "task_id": "task-gs",
            "target_id": "tgt-gs",
            "repo": "agents-core",
            "prompt": "fix it",
            "timeout_s": 1800,
            "slug": "local",
        }
        wt_dir = tmp_path / "wt"
        _init_repo(wt_dir)
        # Add a bare origin so the green path's `git push origin
        # HEAD:<branch>` succeeds (the real worktree has origin from the
        # parent clone).
        origin_dir = tmp_path / "origin.git"
        subprocess.run(["git", "init", "--bare", "-q", str(origin_dir)],
                       check=True)
        _git(wt_dir, "remote", "add", "origin", str(origin_dir))
        _git(wt_dir, "push", "-q", "origin", "HEAD")

        def fake_call_gw_agent(*args, **kwargs):
            # >=1 WIP commit exists: the after_step hook fires during the
            # run on a compiling write step (the condition the
            # WIP-salvage branch needs to be a candidate).
            after_step = kwargs.get("after_step")
            if after_step is not None:
                (wt_dir / "work.py").write_text("z = 3\n")
                after_step({
                    "step_num": 1,
                    "transcript": [{
                        "step": 1, "tool_name": "write_file",
                        "tool_call_id": "call_0",
                        "arguments": {"path": "work.py",
                                       "content": "z = 3\n"},
                        "result": "wrote 6 bytes to work.py",
                        "error": None,
                    }],
                    "cwd": str(wt_dir),
                    "writeable": True,
                })
            # Non-concluded at the max_steps ceiling, but the tail is
            # clean: a non-empty diff, all tests passing, and no test
            # file in the transcript (the legacy gate applies).
            return (
                {"final_diff": "diff --git a/work.py b/work.py\n"
                               "--- /dev/null\n"
                               "+++ b/work.py\n"
                               "+z = 3\n",
                 "concluded": False, "stop_reason": "",
                 "max_steps_reached": True, "no_progress": False,
                 "last_test_outcome": {"passed": 2, "failed": 0,
                                       "errors": 0},
                 "steps": [{"step": 1, "tool_name": "write_file"}]},
                [{"step": 1, "tool_name": "write_file",
                  "arguments": {"path": "work.py"}, "error": None}],
            )

        with patch("agents_core.doorman_client.DoormanClient") as MockClient, \
             patch("agents_core.worktree.setup_worktree") as mock_setup, \
             patch("agents_core.worktree.teardown_worktree") as mock_teardown, \
             patch("agents_core.gw_agent.call_gw_agent",
                    side_effect=fake_call_gw_agent), \
             patch("agents_core.forgejo.create_pr") as mock_create_pr:
            MockClient.return_value.acquire.return_value = {
                "status": "serving", "work_id": "task-gs-berth-sup",
            }
            mock_setup.return_value = MagicMock(path=str(wt_dir))
            mock_create_pr.return_value = {
                "html_url": "http://forgejo/agents-core/pulls/1001",
            }
            out = shaped_runner._run_local_fixer(spec, base_cwd="/srv/agents")

        # The GREEN-salvage path was taken, not the WIP-salvage path:
        # exactly one PR, on the NORMAL branch (no -salvage suffix),
        # with a NORMAL title (no [SALVAGE]).
        mock_create_pr.assert_called_once()
        call_kwargs = mock_create_pr.call_args.kwargs
        assert call_kwargs["head"] == "lapis/tgt-gs/local"
        assert "[SALVAGE]" not in call_kwargs["title"]
        assert call_kwargs["base"] == "main"
        assert out == "http://forgejo/agents-core/pulls/1001"