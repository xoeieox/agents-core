#!/usr/bin/env python3
"""Shaped-agent subprocess runner.

The GPU/Claude queue executes one of these per dispatch. It reads a JSON spec
file containing {model, system, prompt, timeout_s, capture_meta} and
invokes `claude -p` via the shared llm_client.call_claude_cli helper.
Output goes to stdout, which the queue runner captures into
/srv/lapis/gpu-queue/completed/<id>-output.md or
/srv/lapis/claude-queue/completed/<id>-output.md.

Spec files live in /srv/lapis/gpu-queue/shaped/ and are deleted after the
runner finishes.

When the spec sets capture_meta=true, the runner also writes a
{spec_id}-meta.json sidecar to the shaped/ dir containing the parsed
envelope plus a confabulation heuristic. lapis-pm reads this to detect
fixers that produced prose without using tools.

Invoked as: python3 -m agents_core.shaped_runner <spec.json>
"""

import json
import os
import re
import sys
from pathlib import Path

from agents_core.llm import call_claude_cli
from agents_core.room_paths import room_path


_FORGEJO_PR_RE = re.compile(r"http://\d+\.\d+\.\d+\.\d+:\d+/[\w\-]+/[\w\-]+/pulls?/\d+")
_GIT_EVIDENCE_RE = re.compile(
    r"(git (?:checkout|push|commit|add|fetch|branch)\b|create_pr\(|html_url)",
    re.IGNORECASE,
)
# Phrases that recurrently appear in confabulated fixer responses (planning
# prose claiming the agent lacks permissions, or describing what it "would"
# do instead of doing it). Calibrated on the flight-2 confabulation that
# claimed "claude -p runs read-only, manual approval required." Extended
# 2026-04-23 to also catch the permission-wall phrasing seen when
# `claude -p` couldn't write to an un-trusted workspace cwd — even though
# that case is structural (not strictly confabulation), the downstream
# handling is the same: the shaped agent produced no executed artifacts
# and the dispatch should be retried.
_CONFAB_PHRASES_RE = re.compile(
    r"\b("
    r"read[- ]only"
    r"|manual approval"
    r"|cannot (?:create|write|push|edit|modify)"
    r"|unable to (?:create|write|push|edit|modify|access)"
    r"|i would (?:need|then|first|now|recommend)"
    r"|i'?d (?:need|recommend|first)"
    r"|would (?:need|have) to"
    r"|don'?t have (?:access|permission|the ability)"
    r"|require[s]? (?:manual|human) (?:approval|intervention|review)"
    r"|(?:write )?permissions? need(?:s)? to be granted"
    r"|please allow writes? to"
    r"|once you grant (?:write )?access"
    r")",
    re.IGNORECASE,
)


def _spec_id_from_path(spec_path: Path) -> str:
    """Spec filename pattern is {target_id}-{agent_name}-{spec_id}.json."""
    return spec_path.stem.rsplit("-", 1)[-1]


def _build_meta(result: str | None, envelope: dict | None) -> dict:
    """Heuristic confabulation detection for fixer agents.

    Multi-signal decision tree, ordered highest-confidence first:

      1. is_error                             → not confab (model reported error)
      2. PR URL or git evidence present       → not confab (execution artifacts)
      3. confab phrases + no artifacts        → confab if >200 chars
         (catches the permission-wall case where the fixer read many files —
          multi_turn_confirmed — but wrote nothing; multi-turn alone is NOT
          a strong enough positive signal to override phrase evidence)
      4. multi_turn_confirmed (>=2)           → not confab (tools ran, no confab phrasing)
      5. single_turn_confirmed (==1)          → confab if >200 chars OR confab phrases
      6. confab phrases (turns unknown)       → confab if >200 chars
      7. uncertain (no turns, no phrases)     → confab only if >800 chars

    Cases 5-7 trade off: when num_turns is absent (e.g., older claude CLI
    versions don't emit it), positive language signals are required to
    flag — biases toward false negatives over false positives. The 800-char
    floor in the fully-uncertain case catches egregious cases where length
    alone is suspicious without over-triggering on legitimate short replies.
    """
    text = result or ""
    char_count = len(text)
    has_pr_url = bool(_FORGEJO_PR_RE.search(text))
    has_git_evidence = bool(_GIT_EVIDENCE_RE.search(text))
    has_confab_phrases = bool(_CONFAB_PHRASES_RE.search(text))

    if isinstance(envelope, dict):
        num_turns = envelope.get("num_turns")
        is_error = bool(envelope.get("is_error"))
        usage = envelope.get("usage") or {}
    else:
        num_turns = None
        is_error = False
        usage = {}

    multi_turn_confirmed = isinstance(num_turns, int) and num_turns >= 2
    single_turn_confirmed = isinstance(num_turns, int) and num_turns == 1
    has_artifacts = has_pr_url or has_git_evidence

    if is_error:
        confabulated, basis = False, "model reported is_error"
    elif has_artifacts:
        confabulated, basis = False, "execution artifacts present"
    elif has_confab_phrases and char_count > 200:
        confabulated = True
        basis = (
            f"confab phrases + no artifacts"
            f"{f' (num_turns={num_turns})' if num_turns is not None else ''}, "
            f"{char_count} chars"
        )
    elif multi_turn_confirmed:
        confabulated, basis = False, f"multi_turn confirmed (num_turns={num_turns})"
    elif single_turn_confirmed:
        confabulated = char_count > 200 or has_confab_phrases
        basis = "single_turn confirmed + " + (
            "confab phrases" if has_confab_phrases else f"prose ({char_count} chars)"
        )
    elif has_confab_phrases:
        confabulated = char_count > 200
        basis = f"confab phrases (num_turns unknown), {char_count} chars"
    else:
        confabulated = char_count > 800
        basis = (
            f"uncertain — no num_turns, no confab phrases; "
            f"{'flagged on length' if confabulated else 'below 800-char floor'}"
        )

    return {
        "confabulated": confabulated,
        "decision_basis": basis,
        "char_count": char_count,
        "has_pr_url": has_pr_url,
        "has_git_evidence": has_git_evidence,
        "has_confab_phrases": has_confab_phrases,
        "num_turns": num_turns,
        "multi_turn_confirmed": multi_turn_confirmed,
        "single_turn_confirmed": single_turn_confirmed,
        "is_error": is_error,
        "usage": usage,
    }


def _run_local_fixer(spec: dict, base_cwd: str | None) -> str:
    """Deterministic git/PR tail for the local-fixer engine.

    Manages its own worktree (shaper doesn't set worktree_required for GPU-routed
    agents). Returns a PR URL on success, "" on any failure — never raises.
    """
    import json as _json
    import subprocess

    task_id = spec.get("task_id") or spec.get("slot_id") or "lf-unknown"
    target_id = spec.get("target_id", "unknown")
    repo = spec.get("repo", "")
    base_branch = spec.get("base_branch", "main")
    slug = spec.get("slug", "local")
    branch = f"lapis/{target_id}/{slug}"
    bare_repo = repo.rsplit("/", 1)[-1] if repo else "agents-core"
    effective_cwd = base_cwd or "/srv/agents"

    _ARTIFACT_DIR = room_path("gpu_queue.shaped")
    transcript_path = _ARTIFACT_DIR / f"{task_id}-gw-transcript.json"

    worktree_path = None
    try:
        from agents_core.gw_agent import call_gw_agent
        from agents_core.worktree import setup_worktree, teardown_worktree
        import agents_core.forgejo as _forgejo

        handle = setup_worktree(task_id, effective_cwd, base_branch)
        worktree_path = handle.path
        cwd = str(worktree_path)

        # Resolve max_steps: spec JSON > env GW_AGENT_MAX_STEPS > local-fixer default 60.
        if spec.get("max_steps") is not None:
            _max_steps = int(spec["max_steps"])
        elif os.environ.get("GW_AGENT_MAX_STEPS") is not None:
            _max_steps = int(os.environ["GW_AGENT_MAX_STEPS"])
        else:
            _max_steps = 60

        fixer_result, transcript = call_gw_agent(
            prompt=spec["prompt"],
            system=spec.get("system", ""),
            cwd=cwd,
            writeable=True,
            timeout=int(spec.get("timeout_s", 1800)),
            think=False,
            on_wake_fail="skip",
            work_id=task_id,
            max_steps=_max_steps,
            backend_url=spec.get("backend_url"),
            acquire_lease=spec.get("acquire_lease", True),
            model=spec.get("model"),
        )

        # Persist transcript regardless of outcome
        try:
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(
                _json.dumps(transcript, ensure_ascii=False, default=str)
            )
        except OSError as exc:
            print(f"WARN: local-fixer: transcript write failed: {exc}", file=sys.stderr)

        final_diff = fixer_result.get("final_diff") or ""
        concluded = fixer_result.get("concluded", False)
        last_test_outcome = fixer_result.get("last_test_outcome")
        max_steps_hit = fixer_result.get("max_steps_reached", False)
        no_progress_hit = fixer_result.get("no_progress", False)

        def _tests_passed(outcome: dict | None) -> bool:
            if not outcome:
                return False
            return (
                int(outcome.get("passed") or 0) > 0
                and int(outcome.get("failed") or 0) == 0
                and int(outcome.get("errors") or 0) == 0
            )

        salvaged = False
        if not concluded:
            if (max_steps_hit or no_progress_hit) and final_diff.strip() and _tests_passed(last_test_outcome):
                # Budget ceiling OR no-progress abort, but the diff is clean and
                # tests pass — salvage the verified work as a PR rather than discard.
                salvaged = True
            elif no_progress_hit:
                print(
                    "WARN: local-fixer: run aborted - no semantic progress after consecutive idle steps (spinning wheels)",
                    file=sys.stderr,
                )
                return ""
            elif max_steps_hit:
                print(
                    "WARN: local-fixer: run not concluded - max_steps ceiling reached (no passing tests or empty diff)",
                    file=sys.stderr,
                )
                return ""
            else:
                print(
                    "WARN: local-fixer: run not concluded - DoormanUnreachable or wake timeout",
                    file=sys.stderr,
                )
                return ""

        if not salvaged:
            if not final_diff.strip():
                print("WARN: local-fixer: empty diff — no PR", file=sys.stderr)
                return ""
            if last_test_outcome is not None:
                passed = int(last_test_outcome.get("passed") or 0)
                if passed == 0:
                    print("WARN: local-fixer: zero passing tests — no PR", file=sys.stderr)
                    return ""

        # Deterministic git (model never touches git)
        def _git(*args: str) -> subprocess.CompletedProcess:
            try:
                return subprocess.run(
                    ["git", "-C", cwd, *args],
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                print(f"WARN: local-fixer: git {args[0]} timed out", file=sys.stderr)
                return subprocess.CompletedProcess(["git", "-C", cwd, *args], 1, "", "timeout")

        r = _git("checkout", "-b", branch)
        if r.returncode != 0:
            print(f"WARN: local-fixer: git checkout -b failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        r = _git("add", "-A")
        if r.returncode != 0:
            print(f"WARN: local-fixer: git add failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        r = _git("commit", "-m", f"fix({target_id}): local-fixer harness")
        if r.returncode != 0:
            print(f"WARN: local-fixer: git commit failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        r = _git("push", "origin", f"HEAD:{branch}")
        if r.returncode != 0:
            print(f"WARN: local-fixer: git push failed: {r.stderr.strip()}", file=sys.stderr)
            return ""

        # Provenance PR body — factual only
        diff_lines = [l for l in final_diff.splitlines()
                      if l.startswith(("diff --git", "---", "+++", "@@", " ")) or l[:1] in ("+", "-")]
        diffstat = "\n".join(diff_lines[:40]) or "(no changes)"

        if last_test_outcome:
            passed_c = int(last_test_outcome.get("passed") or 0)
            failed_c = int(last_test_outcome.get("failed") or 0)
            test_summary = f"{passed_c} passed, {failed_c} failed"
        else:
            test_summary = "no test outcome recorded"

        step_count = len(fixer_result.get("steps") or [])
        salvage_kind = "no_progress" if no_progress_hit else "max_steps_reached"
        salvage_note = (
            f"**harness-salvaged: {salvage_kind}** - "
            "loop aborted before an explicit conclusion but the diff and tests were clean.\n\n"
            if salvaged else ""
        )
        # Machine-readable marker so the lapis-pm daemon can refuse to auto-merge a
        # no_progress salvage (a model-admitted stall requires human sign-off). A
        # max_steps salvage carries no such marker and follows normal authority rules.
        signoff_marker = (
            "**Requires human sign-off** (model-admitted stall — not auto-merge-eligible).\n\n"
            "<!-- lapis-no-progress-salvage: true -->\n"
            if (salvaged and no_progress_hit) else ""
        )

        if salvaged:
            print(
                f"INFO: local-fixer: harness-salvaged green diff on {salvage_kind} "
                f"abort (target={target_id}, task={task_id})",
                file=sys.stderr,
            )

        pr_body = (
            f"Implemented by the local 122B fixer harness, not paid Claude.\n\n"
            f"{salvage_note}"
            f"{signoff_marker}"
            f"## Diff summary\n\n```diff\n{diffstat}\n```\n\n"
            f"## Test outcome\n\n{test_summary}\n\n"
            f"## Steps\n\n{step_count} tool-call step(s) executed.\n\n"
            f"## Transcript\n\n`{transcript_path}`\n\n"
            f"<!-- lapis-gpu-id: {task_id} -->\n"
            f"<!-- lapis-tid: {target_id} -->"
        )

        pr = _forgejo.create_pr(
            repo=bare_repo,
            title=f"fix({target_id}): local-fixer",
            head=branch,
            base="main",
            body=pr_body,
        )
        return pr.get("html_url", "")

    except Exception as exc:
        print(f"WARN: local-fixer: unexpected error: {exc}", file=sys.stderr)
        return ""

    finally:
        if worktree_path is not None:
            try:
                teardown_worktree(task_id, effective_cwd)
            except Exception as exc:
                print(f"WARN: local-fixer: worktree teardown failed: {exc}", file=sys.stderr)


def _run_local_reviewer(spec: dict, base_cwd: str | None) -> str:
    """Read-only reviewer runner for local GW agent.

    Calls call_gw_agent(writeable=False, json_mode=True) and returns
    the text result. No worktree, no git, no PR creation.
    """
    from agents_core.gw_agent import call_gw_agent

    task_id = spec.get("task_id") or spec.get("slot_id") or "lr-unknown"
    cwd = base_cwd or "/srv/agents"
    model = spec.get("model")

    result = call_gw_agent(
        prompt=spec["prompt"],
        system=spec.get("system", ""),
        cwd=cwd,
        writeable=False,
        json_mode=True,
        timeout=int(spec.get("timeout_s", 900)),
        think=False,
        on_wake_fail="skip",
        work_id=task_id,
        max_steps=int(spec.get("max_steps", 24)),
        model=model,
        backend_url=spec.get("backend_url"),
        acquire_lease=spec.get("acquire_lease", True),
    )
    return result or ""


def main():
    if len(sys.argv) != 2:
        print("ERROR: usage: python3 -m agents_core.shaped_runner <spec.json>", file=sys.stderr)
        sys.exit(2)

    spec_path = Path(sys.argv[1])
    if not spec_path.exists():
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(2)

    try:
        spec = json.loads(spec_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid spec JSON: {e}", file=sys.stderr)
        sys.exit(2)

    capture_meta = bool(spec.get("capture_meta"))
    # cwd determines which CLAUDE.md and SessionStart hooks (chub-inject.py,
    # per-project auto-memory) the subprocess picks up. Older specs without
    # the field fall back to call_claude_cli's default.
    base_cwd = spec.get("cwd") or None
    # permission_mode is "bypassPermissions" for shaped-agent dispatches (set
    # by shaper). Without it, `claude -p` cannot grant Write/Edit in a
    # non-cached-trust workspace and returns a "please allow writes" message.
    permission_mode = spec.get("permission_mode") or None

    # Engine dispatch — local-fixer bypasses the claude -p path entirely and
    # runs the deterministic git/PR tail around the GW 122B harness.
    engine = spec.get("engine", "claude")
    if engine == "local-fixer":
        pr_url = _run_local_fixer(spec, base_cwd)
        try:
            spec_path.unlink()
        except OSError:
            pass
        print(pr_url)
        return
    elif engine == "local-reviewer":
        result = _run_local_reviewer(spec, base_cwd)
        try:
            spec_path.unlink()
        except OSError:
            pass
        print(result)
        return

    # Per-task git worktree isolation for shaped agents (2026-04-23). When
    # the shaper routes to ClaudeQueue it sets worktree_required=True;
    # concurrent runners would otherwise interleave git checkout/commit/push
    # on the shared /srv/git/<repo>-working/ tree (see
    # /srv/lapis/planning/specs/agents-core-claude-queue.md).
    worktree_path = None
    try:
        if spec.get("worktree_required"):
            try:
                from agents_core.worktree import setup_worktree
                handle = setup_worktree(
                    spec["task_id"], base_cwd,
                    spec.get("base_branch", "main"),
                )
                worktree_path = handle.path
                cwd = str(worktree_path)
                # Propagate pip-isolation env into this process so the claude -p
                # subprocess (spawned by call_claude_cli) inherits them. Each
                # shaped_runner.py invocation is a dedicated subprocess per dispatch,
                # so mutating os.environ here does not leak across tasks.
                os.environ.update(handle.env)
            except Exception as e:
                print(f"ERROR: worktree_setup: {e}", file=sys.stderr)
                sys.exit(2)
        else:
            cwd = base_cwd

        if capture_meta:
            result, envelope = call_claude_cli(
                prompt=spec["prompt"],
                system=spec.get("system", ""),
                model=spec.get("model", "haiku"),
                timeout=int(spec.get("timeout_s", 300)),
                json_mode=bool(spec.get("json_mode", False)),
                return_envelope=True,
                cwd=cwd,
                permission_mode=permission_mode,
            )
            try:
                spec_id = _spec_id_from_path(spec_path)
                meta_path = spec_path.parent / f"{spec_id}-meta.json"
                meta_path.write_text(
                    json.dumps(_build_meta(result, envelope), ensure_ascii=False, default=str)
                )
            except OSError as e:
                # Sidecar is best-effort; never block the result on it.
                print(f"WARN: meta sidecar write failed: {e}", file=sys.stderr)
        else:
            result = call_claude_cli(
                prompt=spec["prompt"],
                system=spec.get("system", ""),
                model=spec.get("model", "haiku"),
                timeout=int(spec.get("timeout_s", 300)),
                json_mode=bool(spec.get("json_mode", False)),
                cwd=cwd,
                permission_mode=permission_mode,
            )
    finally:
        if worktree_path is not None:
            verdict_src = worktree_path / ".lapis-pm-verdict.json"
            if verdict_src.exists():
                try:
                    spec_id = _spec_id_from_path(spec_path)
                    verdict_dest = spec_path.parent / f"{spec_id}-verdict.json"
                    verdict_dest.write_text(verdict_src.read_text())
                except OSError as e:
                    print(f"WARN: verdict sidecar copy failed: {e}", file=sys.stderr)
            try:
                from agents_core.worktree import teardown_worktree
                teardown_worktree(spec["task_id"], base_cwd)
            except Exception as e:
                print(f"WARN: worktree teardown failed: {e}", file=sys.stderr)
        try:
            spec_path.unlink()
        except OSError:
            pass

    if result is None:
        print("ERROR: shaped agent call returned None (timeout or invocation failure)")
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
