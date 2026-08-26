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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agents_core.llm import call_claude_cli
from agents_core.room_paths import room_path

# Opt-in stream-json log dir for spec_reviewer shaped-runner tasks
# (agents-core-shaped-runner-stream-log-v0). mkdir'd explicitly before first
# use — room_path() only resolves the Path, it does not create directories.
STREAM_LOG_DIR = room_path("claude_queue.stream_logs")


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


# ---------------------------------------------------------------------------
# D1 + D6 (agents-core-local-fixer-harness-fix-v0): positive-only test gate
# and friction logging.
#
# The legacy gate used the model's LAST run_tests outcome: if that outcome
# was a pre-existing failure (or a non-existent test file), a 207-passing-
# test diff was discarded. The positive-only gate instead checks that every
# test the model CREATED OR EDITED in this run passes; pre-existing failures
# are logged (and witnessed via a friction mem entry, D6) but do NOT block.
#
# Fail-closed: when the model touched no tests at all (production-code-only
# fix), the gate falls back to the legacy _tests_passed(last_test_outcome)
# behavior so a fixer cannot merge untested production code by simply
# refusing to write tests.
# ---------------------------------------------------------------------------


def _resolve_test_path(cwd: str, path: str) -> str:
    """Resolve a test-file path against cwd and return a CWD-relative POSIX form.

    Both the write_file/apply_edit targets and the run_tests targets are
    resolved against the same cwd before comparison, so a relative target
    ("tests/test_foo.py") and an absolute target ("/worktree/tests/test_foo.py")
    both match (agents-core-local-fixer-harness-fix-v0, D1 mandated amendment).
    """
    from pathlib import Path as _P
    raw = (path or "").strip()
    if not raw:
        return ""
    p = _P(raw)
    if not p.is_absolute():
        p = _P(cwd) / p
    try:
        rel = p.resolve().relative_to(_P(cwd).resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        # Path is outside cwd (or unresolvable) — keep the raw string so the
        # comparison degrades to exact-match rather than silently dropping
        # the entry (which would under-count model_touched_tests).
        return raw


def _collect_model_touched_tests(transcript: list[dict], cwd: str) -> set[str]:
    """Collect the CWD-relative paths of test files the model created/edited.

    Scans the run transcript for successful write_file/apply_edit calls whose
    target resolves under tests/ (or is a test_*.py file). Paths are resolved
    against cwd before comparison (D1 mandated amendment).
    """
    touched: set[str] = set()
    for entry in transcript or []:
        if entry.get("tool_name") not in ("write_file", "apply_edit"):
            continue
        if entry.get("error") is not None:
            continue  # a failed write did not actually touch the file
        args = entry.get("arguments") or {}
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue
        rel = _resolve_test_path(cwd, path)
        if not rel:
            continue
        if rel.startswith("tests/") or "/tests/" in rel or rel.startswith("test_"):
            touched.add(rel)
    return touched


def _extract_failed_node_ids(last_test_outcome: dict | None) -> list[str]:
    """Extract pytest node IDs of failed/error tests from a run_tests outcome.

    Parses the outcome's output_tail for lines like
    'FAILED tests/test_foo.py::TestX::test_y' / 'ERROR tests/test_foo.py'.
    Returns [] when the outcome is absent or no node IDs are visible.
    """
    if not last_test_outcome:
        return []
    tail = last_test_outcome.get("output_tail") or ""
    node_ids: list[str] = []
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith(("FAILED ", "ERROR ", "ERROR at setup of ")):
            # Node ID is the last whitespace-separated token (file::Class::fn).
            # 'ERROR at setup of tests/test_foo.py::test_y' -> the node is the
            # token after 'of'.
            if line.startswith("ERROR at setup of "):
                node = line[len("ERROR at setup of "):].strip()
            else:
                node = line.rsplit(" ", 1)[-1].strip()
            if node and node not in node_ids:
                node_ids.append(node)
    return node_ids


def _slugify(node_id: str) -> str:
    """Slug a pytest node ID for use in a friction mem key (stable, no date)."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", node_id).strip("-").lower()
    return slug or "unknown"


def _error_signature(node_id: str, last_test_outcome: dict | None) -> str:
    """Build a STABLE error signature: exception type + test node ID.

    The raw error string embeds episode titles/paths/counts and would break
    cross-run dedup on wording drift (Council AC7a), so the signature keys on
    the exception type (parsed from the outcome's output_tail) plus the node
    ID only. When no exception type is visible, 'unknown' is used so the
    signature is still stable for the same node ID.
    """
    exc_type = "unknown"
    tail = (last_test_outcome or {}).get("output_tail") or ""
    # Look for a 'E <ExceptionType>:' line (pytest's short failure format).
    for line in tail.splitlines():
        m = re.match(r"^\s*E\s+([A-Za-z_][A-Za-z0-9_.]*)\s*:", line)
        if m:
            exc_type = m.group(1)
            break
    return f"{exc_type}@{node_id}"


def _write_friction_entry(
    *,
    repo: str,
    node_id: str,
    error_signature: str,
    task_id: str,
    today: str,
    log: Callable[[str], None] | None = None,
) -> None:
    """Write (or dedup-update) a friction mem entry for a pre-existing test failure.

    D6 (agents-core-local-fixer-harness-fix-v0): the friction entry is the
    signal a future daemon-side follow-up spec will scan for (status: open)
    and autonomously dispatch a fixer on. Key is friction/<repo>-<node-slug>
    (NO date in the key) so the same pre-existing failure across runs maps to
    the same key. Dedup: if the entry exists and is status: open, refresh
    last_seen/last_task_id without duplicating; if status: resolved, flip
    back to open (the friction recurred).

    Never raises: a friction-write failure is logged and swallowed so it can
    never block the test gate or the PR tail.
    """
    key = f"friction/{repo}-{_slugify(node_id)}"
    try:
        from agents_core.mem import MemoryStore

        store = MemoryStore()
        try:
            existing = store.get(key)
            if existing:
                existing_json = {}
                try:
                    existing_json = json.loads(existing.get("content") or "{}")
                except (json.JSONDecodeError, TypeError):
                    existing_json = {}
                status = existing_json.get("status", "open")
                if status == "open":
                    existing_json["last_seen"] = today
                    existing_json["last_task_id"] = task_id
                    store.set(key, json.dumps(existing_json, ensure_ascii=False),
                              tags=["friction", "test-gate", repo])
                else:
                    existing_json["status"] = "open"
                    existing_json["last_seen"] = today
                    existing_json["last_task_id"] = task_id
                    store.set(key, json.dumps(existing_json, ensure_ascii=False),
                              tags=["friction", "test-gate", repo])
            else:
                entry = {
                    "status": "open",
                    "test_node_id": node_id,
                    "error_signature": error_signature,
                    "first_seen": today,
                    "last_seen": today,
                    "first_task_id": task_id,
                    "last_task_id": task_id,
                }
                store.set(key, json.dumps(entry, ensure_ascii=False),
                          tags=["friction", "test-gate", repo])
        finally:
            store.close()
    except Exception as exc:
        if log:
            log(f"WARN: friction entry write failed for {key}: {exc}")
        else:
            print(f"WARN: friction entry write failed for {key}: {exc}", file=sys.stderr)


def _open_wip_salvage_pr(
    worktree_path: str | None,
    wip_ref: str,
    wip_head_sha: str,
    wip_steps: list,
    *,
    stop_reason: str,
    task_id: str,
    target_id: str,
    bare_repo: str,
    branch: str,
    slug: str,
    step_count: int,
    transcript_path: Path,
) -> str:
    """Push refs/wip/<task_id> to a <slug>-salvage branch and open an advisory
    [SALVAGE] PR (agents-core-fixer-budget-compact-salvage-v0, S3).

    The worktree's HEAD is still at the base (detached origin/<base>); the WIP
    history lives ONLY on the separate ref. Pushing to <slug>-salvage keeps a
    later clean run's <branch> unconflicted. The PR is advisory (never
    auto-merged); the run is still LOST (this returns the PR URL or "").
    """
    import subprocess

    import agents_core.forgejo as _forgejo

    salvage_branch = f"{branch}-salvage"

    def _git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", worktree_path, *args],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            print(f"WARN: wip-salvage: git {args[0]} timed out", file=sys.stderr)
            return subprocess.CompletedProcess(["git", "-C", worktree_path, *args], 1, "", "timeout")

    if not worktree_path or not wip_head_sha:
        print("WARN: wip-salvage: no WIP head sha - no PR", file=sys.stderr)
        return ""

    # Push the WIP history to the salvage branch.
    r = _git("push", "origin", f"{wip_ref}:refs/heads/{salvage_branch}")
    if r.returncode != 0:
        print(f"WARN: wip-salvage: git push failed: {r.stderr.strip()}", file=sys.stderr)
        return ""

    # PR body: task id, stop_reason + wall time, WIP head sha, the steps
    # included, and what remains per the spec.
    steps_included = ", ".join(str(s) for s in wip_steps) if wip_steps else "n/a"
    pr_body = (
        f"**[SALVAGE] advisory PR - the run is LOST (not concluded, no gate_passed).**\n\n"
        f"## Task\n\n"
        f"- task_id: `{task_id}`\n"
        f"- target_id: `{target_id}`\n"
        f"- stop_reason: `{stop_reason}`\n"
        f"- wall time: {datetime.now(timezone.utc).isoformat()}\n"
        f"- WIP head sha: `{wip_head_sha}`\n"
        f"- steps included: {steps_included}\n"
        f"- total steps executed: {step_count}\n\n"
        f"## What remains\n\n"
        f"Per the bound spec (staged as `lapis-spec.md` during the run, removed "
        f"after): the spec's remaining work items are NOT in this salvage. The "
        f"WIP commits are a compile-gated snapshot of the whole-file writes "
        f"made up to the death - a partial implementation at best.\n\n"
        f"## Recovery\n\n"
        f"`lapis-pm rebind --force --adopt-pr <n>` + fixer_retry (proven "
        f"2026-08-25 on PR #258).\n\n"
        f"## Transcript\n\n"
        f"`{transcript_path}`\n\n"
        f"<!-- lapis-gpu-id: {task_id} -->\n"
        f"<!-- lapis-tid: {target_id} -->\n"
        f"<!-- lapis-salvage: true -->\n"
    )

    try:
        pr = _forgejo.create_pr(
            repo=bare_repo,
            title=f"[SALVAGE] fix({target_id}): {stop_reason} at step {step_count}",
            head=salvage_branch,
            base="main",
            body=pr_body,
        )
        print(
            f"INFO: wip-salvage: [SALVAGE] PR opened: {pr.get('html_url', '')} "
            f"(branch={salvage_branch}, WIP head={wip_head_sha})",
            file=sys.stderr,
        )
        return pr.get("html_url", "")
    except Exception as exc:
        print(f"WARN: wip-salvage: create_pr failed: {exc}", file=sys.stderr)
        return ""


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

    # ------------------------------------------------------------------
    # WIP-commit salvage state (agents-core-fixer-budget-compact-salvage-
    # v0, S3). The after_step hook snapshots the step's write-tool paths
    # onto a SEPARATE ref (refs/wip/<task_id>) via write-tree/commit-tree:
    # the worktree's HEAD and index are never moved, so the success path
    # (final_diff = index vs HEAD, the tail's commit/PR) is byte-identical
    # to pre-spec. On a non-concluded terminal death with >=1 WIP commit,
    # the ref is pushed to a <slug>-salvage branch and an advisory
    # [SALVAGE] PR is opened (never auto-merged; the run is still LOST).
    # ------------------------------------------------------------------
    wip_ref = f"refs/wip/{task_id}"
    wip_commit_count = 0
    wip_head_sha = ""
    wip_steps: list[int] = []

    def _wip_git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", worktree_path, *args],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(f"WARN: wip-commit: git {args[0]} timed out", file=sys.stderr)
            return subprocess.CompletedProcess(["git", "-C", worktree_path, *args], 1, "", "timeout")

    def _wip_commit_hook(ctx: dict) -> None:
        """One WIP commit per step that wrote a file. Never raises."""
        nonlocal wip_commit_count, wip_head_sha
        import py_compile

        step_num = ctx.get("step_num")
        cwd = ctx.get("cwd")
        if not ctx.get("writeable") or not cwd or not step_num:
            return
        # Paths from this step's write-tool args only - the staged
        # lapis-spec.md and any stray files are never touched (explicit
        # pathspec, invariant 5).
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
            # py_compile floor: every touched .py must compile with the
            # process interpreter, or the commit is skipped (the previous
            # WIP commit stays the safe floor).
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
                        f"({exc}) - skipping WIP commit for step {step_num} "
                        f"(previous WIP commit stays the floor)",
                        file=sys.stderr,
                    )
                    return
            parent = _wip_git("rev-parse", "--verify", wip_ref)
            if parent.returncode != 0:
                parent = _wip_git("rev-parse", "HEAD")
                if parent.returncode != 0:
                    return
            add = _wip_git("add", "--", *paths)
            if add.returncode != 0:
                print(f"WARN: wip-commit: git add failed: {add.stderr.strip()}", file=sys.stderr)
                return
            tree = _wip_git("write-tree")
            if tree.returncode != 0:
                print(f"WARN: wip-commit: git write-tree failed: {tree.stderr.strip()}", file=sys.stderr)
                _wip_git("reset")
                return
            commit = _wip_git(
                "commit-tree", tree.stdout.strip(), "-p", parent.stdout.strip(),
                "-m", f"wip: {task_id} step {step_num} [auto]",
            )
            if commit.returncode != 0:
                print(f"WARN: wip-commit: git commit-tree failed: {commit.stderr.strip()}", file=sys.stderr)
                _wip_git("reset")
                return
            upd = _wip_git("update-ref", wip_ref, commit.stdout.strip())
            # Unstage: restore the index (and the worktree's HEAD) exactly as found.
            _wip_git("reset")
            if upd.returncode != 0:
                print(f"WARN: wip-commit: git update-ref failed: {upd.stderr.strip()}", file=sys.stderr)
                return
            wip_commit_count += 1
            wip_head_sha = commit.stdout.strip()
            wip_steps.append(int(step_num))
            print(
                f"INFO: wip-commit: {wip_ref} -> {wip_head_sha} "
                f"(step {step_num}, {len(paths)} path(s))",
                file=sys.stderr,
            )
        except Exception as exc:
            # Never break the loop: a WIP-commit failure degrades to "no
            # salvage for this step", exactly like today's no-salvage
            # behavior.
            print(f"WARN: wip-commit: unexpected error: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Supervisor lease (gw-gpu1-berth-standing-seat-v0, leg 2, Doorman B).
    #
    # A doorman lease held for the WHOLE local-fixer job (entry acquire,
    # finally release) so the doorman's mid-job stop machinery (force-stop's
    # _worker_lease_blockers guard) cannot stop the seat out from under an
    # in-flight fixer job. Distinct from the per-run job lease that
    # call_gw_agent acquires internally (work_id=task_id): the lease dict is
    # keyed by work_id, so a same-key lease would be clobbered and released
    # mid-job by the inner finally. This lease uses work_id=f"{task_id}-berth-
    # sup", principal="fixer-supervisor", role="worker" (force-stop blocks
    # ONLY role="worker" leases with principal self-exclusion), and
    # lease_class="deferrable" (the fixer's existing class).
    #
    # TTL = timeout_s + 60 — the same arithmetic as the per-run lease
    # (the queue's hard cap). Acquired at entry + setup lag, it structurally
    # outlives the process killed at submit + timeout_s + 60, so NO renewal
    # is needed.
    #
    # SOFT FAIL: acquire failure (DoormanUnreachable, pending_defer, or any
    # exception) is logged and the job PROCEEDS un-supervised. The lease is a
    # safety net, never a gate on job progress.
    # ------------------------------------------------------------------
    sup_lease_client = None
    sup_lease_id = f"{task_id}-berth-sup"
    try:
        from agents_core.doorman_client import (
            DoormanClient,
            DoormanUnreachable,
        )

        sup_lease_client = DoormanClient()
        acq = sup_lease_client.acquire(
            "gravitywell",
            sup_lease_id,
            int(spec.get("timeout_s", 1800)) + 60,
            "fixer-job-supervisor",
            role="worker",
            principal="fixer-supervisor",
            lease_class="deferrable",
        )
        # acquire() RETURNS a status dict (it does not raise on
        # pending_defer/deferred): only "serving" means a lease was
        # registered. Any other status is a soft fail - log it and drop
        # the client so the finally-release is skipped (the spec's
        # soft-fail contract: log + job proceeds un-supervised).
        if not isinstance(acq, dict) or acq.get("status") != "serving":
            _status = acq.get("status") if isinstance(acq, dict) else type(acq).__name__
            print(
                f"WARN: local-fixer: supervisor lease acquire soft-failed "
                f"(work_id={sup_lease_id}, status={_status}) - job proceeds "
                f"un-supervised",
                file=sys.stderr,
            )
            sup_lease_client = None
    except Exception as exc:
        # Soft fail: the job proceeds un-supervised. The per-run job lease
        # (keep-both) still holds on a defer, and the guard 2c covers the
        # suspend layer regardless.
        print(
            f"WARN: local-fixer: supervisor lease acquire soft-failed "
            f"(work_id={sup_lease_id}): {exc}",
            file=sys.stderr,
        )
        sup_lease_client = None

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

        # Stage the bound spec into the worktree so the model can page it
        # (its file readers are cwd-confined; /srv/lapis/planning is unreadable).
        # Best-effort: any miss degrades to "run exactly as today".
        spec_src = room_path("planning.specs") / f"{target_id}.md"
        spec_dst = Path(cwd) / "lapis-spec.md"
        try:
            if not (spec_src.exists() and spec_src.stat().st_size > 0):
                print(
                    f"WARN: local-fixer: bound spec missing or empty at {spec_src}; continuing without staging",
                    file=sys.stderr,
                )
            elif spec_dst.exists():
                print(
                    f"WARN: local-fixer: refusing to clobber existing {spec_dst} (tracked-file collision guard)",
                    file=sys.stderr,
                )
            else:
                spec_dst.write_text(spec_src.read_text())
                print(
                    f"[local-fixer] spec staged into worktree: lapis-spec.md "
                    f"({spec_dst.stat().st_size} bytes) - read it in line windows before implementing",
                    file=sys.stderr,
                )
        except OSError as exc:
            print(f"WARN: local-fixer: spec staging failed: {exc}", file=sys.stderr)

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
            think=bool(spec.get("think", False)),
            on_wake_fail="skip",
            work_id=task_id,
            max_steps=_max_steps,
            backend_url=spec.get("backend_url"),
            acquire_lease=spec.get("acquire_lease", True),
            swarm_payload=spec.get("swarm_payload", False),
            lease_class="deferrable",
            model=spec.get("model"),
            handler_hook=_handler_hook,
            handler_objective=_handler_objective,
            handler_max_interventions=_handler_max_interventions,
            # WIP-commit salvage seam (agents-core-fixer-budget-compact-
            # salvage-v0, S3): one WIP commit per write step onto the
            # separate refs/wip/<task_id> ref. Local-fixer runs only.
            after_step=_wip_commit_hook,
        )

        # Remove the staged spec before the deterministic git tail so it is
        # never committed/pushed into the PR branch.
        try:
            (Path(cwd) / "lapis-spec.md").unlink(missing_ok=True)
        except OSError as exc:
            print(f"WARN: local-fixer: staged spec remove failed: {exc}", file=sys.stderr)

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
        stop_reason = fixer_result.get("stop_reason", "")

        def _tests_passed(outcome: dict | None) -> bool:
            if not outcome:
                return False
            return (
                int(outcome.get("passed") or 0) > 0
                and int(outcome.get("failed") or 0) == 0
                and int(outcome.get("errors") or 0) == 0
            )

        # D1 (agents-core-local-fixer-harness-fix-v0): positive-only test gate.
        # The legacy gate used the model's LAST run_tests outcome; a pre-existing
        # failure (or a non-existent test file) as the last outcome discarded a
        # 207-passing-test diff. The positive-only gate instead checks that
        # every test the model CREATED OR EDITED in this run passes.
        #
        # Paths are resolved against cwd before comparison (D1 mandated
        # amendment) so a relative write_file target and an absolute run_tests
        # target both match.
        #
        # Fail-closed: when the model touched no tests at all (production-code-
        # only fix), the gate falls back to the legacy _tests_passed behavior
        # so a fixer cannot merge untested production code by simply refusing
        # to write tests.
        model_touched_tests = _collect_model_touched_tests(transcript, cwd)
        gate_passed = False
        if model_touched_tests:
            # Positive-only gate: every test the model touched must pass.
            # A touched test "fails" if a FAILED/ERROR node ID refers to it.
            # A node ID "refers to" a touched test file if the node's file
            # part equals the touched path (file-level failure) OR the node
            # is a specific test within that file (node-level failure).
            # The run must also have at least one passing test (a 0-passed
            # run is never a pass, even if the model's tests were not the
            # ones that failed).
            failed_node_ids = _extract_failed_node_ids(last_test_outcome)
            touched_failures = [
                t for t in model_touched_tests
                if any(
                    n.split("::")[0] == t or n == t
                    for n in failed_node_ids
                )
            ]
            if last_test_outcome is not None:
                passed_c = int(last_test_outcome.get("passed") or 0)
                if passed_c > 0 and not touched_failures:
                    gate_passed = True
        else:
            # Fail-closed fallback: no tests touched -> legacy gate.
            gate_passed = _tests_passed(last_test_outcome)

        # D6 (agents-core-local-fixer-harness-fix-v0): witness pre-existing
        # failures via a friction mem entry so a future daemon-side follow-up
        # can pick them up autonomously. Pre-existing = a failed node ID whose
        # file the model did NOT touch. Logged but never blocks the gate.
        if last_test_outcome is not None:
            failed_node_ids = _extract_failed_node_ids(last_test_outcome)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for node in failed_node_ids:
                node_file = node.split("::")[0]
                if node_file in model_touched_tests:
                    continue  # the model's own test failed — that's the gate's concern
                sig = _error_signature(node, last_test_outcome)
                _write_friction_entry(
                    repo=bare_repo,
                    node_id=node,
                    error_signature=sig,
                    task_id=task_id,
                    today=today,
                    log=lambda m: print(m, file=sys.stderr),
                )
                print(
                    f"INFO: local-fixer: pre-existing test failure witnessed "
                    f"(friction entry written): {node}",
                    file=sys.stderr,
                )

        # A mem-search-loop abort (D5) is a distinct failure mode: the model
        # burned the budget on redundant searches and made no edits. Do not
        # salvage or PR — log the distinct stop_reason and return "".
        if stop_reason == "mem_search_loop":
            print(
                "WARN: local-fixer: run aborted - mem_search_loop "
                "(redundant mem searches; no edits made)",
                file=sys.stderr,
            )
            return ""

        # Output-budget exhaustion is a distinct failure mode: the response was
        # cut at max_tokens. Verified-state salvage (agents-core-fixer-
        # budget-compact-salvage-v0, S3 - the D5 policy flip): when >=1 WIP
        # commit exists on refs/wip/<task_id> (each a compile-gated snapshot
        # of a whole-file write - the safety premise is that a truncated
        # response can never leave a partial file: the fail-closed default
        # executors reject empty-args calls, and valid args JSON implies a
        # complete whole-file write), push the WIP history to a
        # <slug>-salvage branch and open an advisory [SALVAGE] PR. The run
        # is still LOST (this returns "" - no concluded/gate_passed).
        if stop_reason == "output_budget_exhausted":
            if wip_commit_count > 0:
                print(
                    f"WARN: local-fixer: run aborted - output budget exhausted "
                    f"(finish_reason=output_limit/length; response truncated at "
                    f"max_tokens) - WIP commits exist; "
                    f"opening advisory [SALVAGE] PR",
                    file=sys.stderr,
                )
                return _open_wip_salvage_pr(
                    worktree_path, wip_ref, wip_head_sha, wip_steps,
                    stop_reason=stop_reason, task_id=task_id,
                    target_id=target_id, bare_repo=bare_repo,
                    branch=branch, slug=slug,
                    step_count=len(fixer_result.get("steps") or []),
                    transcript_path=transcript_path,
                )
            print(
                "WARN: local-fixer: run aborted - output budget exhausted "
                "(finish_reason=output_limit/length; response truncated at "
                "max_tokens - no WIP commits, no PR)",
                file=sys.stderr,
            )
            return ""

        # WIP-commit salvage on a non-concluded terminal death (agents-core-
        # fixer-budget-compact-salvage-v0, S3): max_steps_hit / no_progress_hit
        # with >=1 WIP commit pushes the WIP history to a <slug>-salvage branch
        # and opens an advisory [SALVAGE] PR. Ordering rule: the green-salvage
        # path below takes precedence when it applies (clean diff AND passing
        # tests) - a dead run CAN have both, and its verified tail is better
        # than the WIP history. The WIP-salvage PR is for the remainder.
        # The run is still LOST.
        if (not concluded and (max_steps_hit or no_progress_hit)
                and wip_commit_count > 0
                and not (final_diff.strip() and gate_passed)):
            _wip_stop_reason = "max_steps_hit" if max_steps_hit else "no_progress_hit"
            print(
                f"WARN: local-fixer: run not concluded - {_wip_stop_reason} "
                f"- opening advisory [SALVAGE] PR",
                file=sys.stderr,
            )
            return _open_wip_salvage_pr(
                worktree_path, wip_ref, wip_head_sha, wip_steps,
                stop_reason=_wip_stop_reason, task_id=task_id,
                target_id=target_id, bare_repo=bare_repo,
                branch=branch, slug=slug,
                step_count=len(fixer_result.get("steps") or []),
                transcript_path=transcript_path,
            )

        salvaged = False
        if not concluded:
            if (max_steps_hit or no_progress_hit) and final_diff.strip() and gate_passed:
                # Budget ceiling OR no-progress abort, but the diff is clean and
                # the model's own tests pass — salvage the verified work as a
                # PR rather than discard.
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
            if not gate_passed:
                # D1 (agents-core-local-fixer-harness-fix-v0): the positive-only
                # gate (or its fail-closed legacy fallback) rejected this run —
                # do not PR. The model's own tests did not all pass (or no
                # tests were touched and the legacy gate failed).
                print(
                    "WARN: local-fixer: test gate failed "
                    f"(model_touched_tests={sorted(model_touched_tests) if model_touched_tests else '[] (legacy gate)'}; "
                    f"last_test_outcome passed={int((last_test_outcome or {}).get('passed') or 0)} "
                    f"failed={int((last_test_outcome or {}).get('failed') or 0)}) — no PR",
                    file=sys.stderr,
                )
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

        r = _git("checkout", "-B", branch)
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
        # Release the supervisor lease (gw-gpu1-berth-standing-seat-v0, leg 2,
        # Doorman B). Mirrors the per-run lease's finally-release in
        # gw_agent.py: released on BOTH the success and failure paths. Soft
        # fail: a release failure is logged and swallowed — the TTL bounds the
        # zombie window (<= TTL + one GC tick) if the release is lost.
        if sup_lease_client is not None:
            try:
                sup_lease_client.release("gravitywell", sup_lease_id)
            except Exception as exc:
                print(
                    f"WARN: local-fixer: supervisor lease release failed "
                    f"(work_id={sup_lease_id}): {exc}",
                    file=sys.stderr,
                )
            try:
                sup_lease_client.close()
            except Exception:
                pass
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
    from agents_core.gw_agent import DEFAULT_READONLY_TOOLS, call_gw_agent, probe_seat_tool_call

    task_id = spec.get("task_id") or spec.get("slot_id") or "lr-unknown"
    cwd = base_cwd or "/srv/agents"
    model = spec.get("model")
    backend_url = spec.get("backend_url")

    # Probe the seat's tool-calling ability before dispatching the real review
    # (agents-core-reviewer-seat-tool-call-probe-v0). The seat can be alive,
    # serving, and correctly configured, yet emit zero tool_calls for every
    # request — a fault that today only surfaces after a real review burns an
    # attempt on reason=grounding_failed (2-4s exit, indistinguishable from a
    # model that looked and declined to investigate). This uses the SAME
    # model, backend_url and tool surface (DEFAULT_READONLY_TOOLS — the same
    # default call_gw_agent applies below for writeable=False) the real call
    # below will use (DoD 1, 4). Logged unconditionally, pass or fail (D7).
    probe = probe_seat_tool_call(
        backend_url=backend_url,
        model=model,
        tools=DEFAULT_READONLY_TOOLS,
    )
    # Part 4 (agents-core-reviewer-seat-prefix-perturbation-retry-v0): the
    # succeeding attempt index, total attempts made, and a short stable hash
    # of the native (attempt-0) serialized tool block — a seat that needed
    # perturbation is recovering from a real fault, and that must stay
    # visible in the dispatch logs rather than being silently papered over.
    print(
        f"INFO: reviewer-seat-probe: outcome={probe['outcome']} "
        f"served_model={probe['served_model']} "
        f"succeeded_attempt={probe.get('attempt')} "
        f"attempts_made={probe.get('attempts_made')} "
        f"refused_prefix_hash={probe.get('refused_prefix_hash')}",
        file=sys.stderr,
    )
    if probe["outcome"] == "no_tool_call":
        # D5 fail CLOSED: every perturbation attempt returned a clean
        # zero-tool-call response — a positive determination the seat is
        # dead. Short-circuit — do not spend a real attempt on a seat that
        # cannot possibly ground. New reason token (distinct from a real
        # grounding_failed) so the caller can tell a dead seat apart from a
        # live one that declined to investigate. Logged explicitly so
        # seat_no_tool_calls can never again be confused with a single
        # unlucky request against one prefix.
        print(
            "ERROR: local reviewer produced no verdict (reason=seat_no_tool_calls) "
            f"all {probe.get('attempts_made')} perturbations refused",
            file=sys.stderr,
        )
        return None
    # probe["outcome"] in {"tool_call", "error"} both proceed to the real
    # call: "tool_call" because the seat proved itself, "error" because probe
    # health is unknown and a probe outage must never block a real review
    # (D5 fail OPEN).

    # Part 3: carry the EXACT variant that succeeded (or the native tools on
    # a fail-open "error" probe, where no variant was validated) into the
    # real review — a probe that only proves *some* variant works while the
    # real call still sends the poisoned native prefix fixes nothing.
    reviewer_tools = probe.get("variant") or DEFAULT_READONLY_TOOLS

    reason: list[str] = []
    result = call_gw_agent(
        prompt=spec["prompt"],
        system=spec.get("system", ""),
        cwd=cwd,
        tools=reviewer_tools,
        writeable=False,
        json_mode=True,
        timeout=int(spec.get("timeout_s", 900)),
        think=False,
        on_wake_fail="skip",
        work_id=task_id,
        max_steps=int(spec.get("max_steps", 24)),
        model=model,
        backend_url=backend_url,
        acquire_lease=spec.get("acquire_lease", True),
        lease_class="deferrable",
        reason_out=reason,
        # This function already probed the seat's tool-calling ability above
        # (probe_seat_tool_call) and carried the exact validated variant into
        # `reviewer_tools`. Stand call_gw_agent's own grounding-guard retry
        # down so the two remedies never stack on this, the hottest dispatch
        # route in the system (agents-core-gw-agent-grounding-retry-parity-v0,
        # DoD-5).
        skip_probe=True,
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
            stream_log_path = None
            if spec.get("agent_type") == "spec_reviewer":
                STREAM_LOG_DIR.mkdir(parents=True, exist_ok=True)
                stream_log_path = str(STREAM_LOG_DIR / f"{spec['task_id']}.jsonl")
            result = call_claude_cli(
                prompt=spec["prompt"],
                system=spec.get("system", ""),
                model=spec.get("model", "haiku"),
                timeout=int(spec.get("timeout_s", 300)),
                json_mode=bool(spec.get("json_mode", False)),
                cwd=cwd,
                permission_mode=permission_mode,
                stream_log_path=stream_log_path,
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
