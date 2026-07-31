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
import subprocess
import sys
from pathlib import Path

from agents_core.llm import call_claude_cli
from agents_core.room_paths import room_path


# ---------------------------------------------------------------------------
# Handler supervision (agents-core-handler-operative-live-supervision-v0)
#
# Live Handler supervision of the local-fixer Operative's step loop. Redirect-only
# v0: the Handler can nudge the Operative back on track or vouch for a strategic
# pause, but cannot halt a run (the "stop" verdict is deferred to a follow-on).
# The Handler seat routes through call_claude_cli only (no direct Anthropic API).
# ---------------------------------------------------------------------------

HANDLER_SYSTEM = """\
You are the HANDLER in a Handler+Operative pair. The Operative is a coding agent
fixing code toward a fixed objective. It has now gone several steps with no code edit.
You do NOT execute anything. You read what it just did and said, and you steer it with
one short, imperative verdict.

Objective:
{objective}

Default to the SAFE baseline. The system already has a fallback nudge and a hard stop
that will handle an ordinary stall on their own. Only override that baseline when you can
point to something SPECIFIC and concrete - if you are unsure, do NOT issue a command;
omit "decision" (or set it to null) and let the baseline handle it. A confidently-wrong
command is worse than staying quiet, because you see only a short slice of the run and a
"wait" may be a valid dependency you cannot see.

Decide ONE of:
- "redirect": ONLY when you can name the SPECIFIC drift - a concrete off-objective action
  (waiting on a named background process, editing a file outside the objective, re-reading
  without progressing). Issue a sharp imperative instruction that returns it to the
  objective (e.g. "Stop waiting on the background test run. The objective only needs <X>.
  Make your edit to <file> now."). Cite the specific drift in "note". If you cannot name a
  specific drift, do NOT redirect.
- "continue": ONLY when you can cite the SPECIFIC objective-aligned work still in progress
  (a concrete dependency the Operative is legitimately gathering before it can edit) - a
  real strategic pause, not a hopeful guess. Cite that dependency in "note". Use sparingly.

You cannot halt the run - your only levers are a redirect nudge or letting it continue.
If something odd appears that you are NOT redirecting toward, record it in "anomaly"
(capture-don't-chase) - do not chase it.

Output ONLY this JSON, no prose, no markdown fences:
{{"decision":"redirect|continue","redirect":"<imperative instruction, or null>",
 "note":"<one line of provenance>","anomaly":"<oddity to capture, or null>"}}"""


def _build_handler_prompt(ctx: dict) -> str:
    """Render the Handler's per-invocation prompt from a gw_agent handler_hook context dict."""
    return (
        f"The Operative has made no edit for {ctx.get('consecutive_no_progress')} steps "
        f"(step {ctx.get('step_num')}, {ctx.get('explore_steps')} total exploration steps "
        "so far). Here is what it just did and said.\n\n"
        f"Recent tool calls (most recent last):\n"
        f"{json.dumps(ctx.get('transcript_slice') or [], ensure_ascii=False, default=str)}\n\n"
        f"Its most recent stated plan/reasoning:\n"
        f"{ctx.get('last_assistant_message') or '(none)'}"
    )


def _build_handler_hook(objective: str, model: str, hook_timeout_s: int):
    """Build a handler_hook closure for call_gw_agent's Handler supervision (v0).

    The seat call routes through call_claude_cli only, with an explicit timeout so the
    hook is self-time-bounding (gw_agent applies no timeout of its own around the call).
    Returns None (the gw_agent fall-through case: static nudge, no budget change) on any
    unparseable/failed seat reply, so a broken Handler can never stop or extend a healthy
    Operative.
    """
    from agents_core.calibration.handler_agent_harness import _extract_json

    system = HANDLER_SYSTEM.replace("{objective}", objective)

    def _hook(ctx: dict) -> dict | None:
        reply = call_claude_cli(
            prompt=_build_handler_prompt(ctx),
            system=system,
            model=model,
            timeout=hook_timeout_s,
        )
        if not reply:
            return None
        return _extract_json(reply)

    return _hook


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

        # Handler supervision config (agents-core-handler-operative-live-supervision-v0).
        # On by default for the local-fixer engine (handler-supervision-enable-local-fixer-v0):
        # an absent key, a malformed (non-dict) value, or an empty/missing-"enabled" dict all
        # resolve to enabled with defaults. Only an explicit {"enabled": False} opts out
        # (handler_hook=None - byte-identical to the old off-by-default behavior).
        # Unknown extra keys are ignored (superset-tolerant).
        _handler_hook = None
        _handler_objective = ""
        _handler_max_interventions = 2
        _supervision_cfg = spec.get("handler_supervision")
        _supervision_enabled = (
            _supervision_cfg.get("enabled", True)
            if isinstance(_supervision_cfg, dict)
            else True
        )
        if _supervision_enabled:
            _backend = (
                _supervision_cfg.get("backend", "claude_cli")
                if isinstance(_supervision_cfg, dict)
                else "claude_cli"
            )
            if _backend != "claude_cli":
                raise ValueError(
                    f"handler_supervision backend {_backend!r} not supported in v0, "
                    "use claude_cli"
                )
            _cfg = _supervision_cfg if isinstance(_supervision_cfg, dict) else {}
            _handler_model = _cfg.get("model", "haiku")
            _hook_timeout_s = int(_cfg.get("hook_timeout_s", 60))
            _handler_max_interventions = int(_cfg.get("max_interventions", 2))
            _handler_objective = (spec.get("prompt") or "")[:1500]
            _handler_hook = _build_handler_hook(_handler_objective, _handler_model, _hook_timeout_s)

        # fixer_retry dispatches target an already-open PR — the worktree must
        # start from the PR's own branch, not base_branch (main), or the target
        # file simply won't exist in the checkout. Verify the branch is really
        # on origin first: the local-fixer GW sandbox has no git checkout tool,
        # so if this is wrong there is no way for the model to self-correct.
        existing_branch = spec.get("existing_branch") or ""
        worktree_ref = base_branch
        if spec.get("agent_type") == "fixer_retry" and existing_branch:
            try:
                verify = subprocess.run(
                    ["git", "-C", effective_cwd, "ls-remote", "--exit-code", "origin", existing_branch],
                    capture_output=True, text=True, timeout=30,
                )
                verified = verify.returncode == 0
            except subprocess.TimeoutExpired:
                verified = False
            if not verified:
                print(
                    f"ERROR: worktree_setup: existing_branch {existing_branch} not found on origin",
                    file=sys.stderr,
                )
                return ""
            worktree_ref = existing_branch

        handle = setup_worktree(task_id, effective_cwd, worktree_ref)
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
            lease_class="deferrable",
            model=spec.get("model"),
            handler_hook=_handler_hook,
            handler_objective=_handler_objective,
            handler_max_interventions=_handler_max_interventions,
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


def _run_local_reviewer(spec: dict, base_cwd: str | None) -> str | None:
    """Read-only reviewer runner for local GW agent.

    Calls call_gw_agent(writeable=False, json_mode=True) and returns the
    text result, or None if call_gw_agent produced no verdict at all. On
    a None result, the reason from reason_out (or "no_content_no_reason"
    if reason_out was never populated) is logged to stderr, and None is
    returned rather than "" — the caller in main() converts a None result
    into a non-zero exit so claude_queue_runner's failure branch records
    the real reason instead of silently reporting a completed task with
    no verdict (agents-core-local-reviewer-no-silent-empty-verdict-v0).
    No git, no PR creation — but the caller (main()) routes
    worktree_required=True dispatches (always true for this engine)
    through the same worktree-setup path as other engines, so base_cwd here
    is already a worktree checked out to existing_branch (verified against
    origin) or base_branch, not the raw shared clone.
    """
    from agents_core.gw_agent import call_gw_agent

    task_id = spec.get("task_id") or spec.get("slot_id") or "lr-unknown"
    cwd = base_cwd or "/srv/agents"
    model = spec.get("model")

    reason: list[str] = []
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
        lease_class="deferrable",
        reason_out=reason,
    )
    if result is None:
        why = reason[0] if reason else "no_content_no_reason"
        print(f"ERROR: local reviewer produced no verdict (reason={why})", file=sys.stderr)
        return None
    return result


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

    # Per-task git worktree isolation for shaped agents (2026-04-23). When
    # the shaper routes to ClaudeQueue it sets worktree_required=True;
    # concurrent runners would otherwise interleave git checkout/commit/push
    # on the shared /srv/git/<repo>-working/ tree (see
    # /srv/lapis/planning/specs/agents-core-claude-queue.md).
    #
    # local-reviewer also sets worktree_required=True
    # (agents-core-reviewer-worktree-branch-checkout-v0) so a reviewer/
    # reviewer_fresh dispatch reviews the PR's actual head branch instead of
    # whatever branch the shared clone happened to be sitting on.
    worktree_path = None
    try:
        if spec.get("worktree_required"):
            try:
                from agents_core.worktree import setup_worktree

                worktree_ref = spec.get("base_branch", "main")
                existing_branch = spec.get("existing_branch") or ""
                if existing_branch:
                    try:
                        verify = subprocess.run(
                            ["git", "-C", base_cwd, "ls-remote", "--exit-code", "origin", existing_branch],
                            capture_output=True, text=True, timeout=30,
                        )
                        verified = verify.returncode == 0
                    except subprocess.TimeoutExpired:
                        verified = False
                    if not verified:
                        print(
                            f"ERROR: worktree_setup: existing_branch {existing_branch} not found on origin",
                            file=sys.stderr,
                        )
                        sys.exit(2)
                    worktree_ref = existing_branch

                handle = setup_worktree(spec["task_id"], base_cwd, worktree_ref)
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

        if engine == "local-reviewer":
            result = _run_local_reviewer(spec, cwd)
        elif capture_meta:
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
