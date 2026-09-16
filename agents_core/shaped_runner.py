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

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
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
# Served-model echo boundedness (local-reviewer-identity-and-provenance-v0,
# L1.D1/L1.D3). The served model is server-echoed (gw_agent.py appends the
# server-supplied data["model"] into served_model_out) and the PROVENANCE
# stdout line is machine-parsed on a mixed-stdout channel (the claude engine
# prints the full model result). Accept only tokens matching the bounded
# charset; anything else is VOID (never rendered / never stamped) - a
# bound-violating token is a void echo, not a value.
# ---------------------------------------------------------------------------
_SERVED_MODEL_ECHO_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
_SERVED_MODEL_ECHO_MAX_LEN = 200


def _validate_served_model_echo(value) -> str:
    """Return the served-model echo if it is a valid, in-bounds token, else "".

    The echo is server-supplied (or parsed from a mixed-stdout channel), so
    it is untrusted input: accept only `^[A-Za-z0-9._:/-]+$` tokens of at
    most 200 chars. A violating token is VOID (""), never a value - the
    caller renders the explicit void form instead.
    """
    if not isinstance(value, str):
        return ""
    if len(value) > _SERVED_MODEL_ECHO_MAX_LEN:
        return ""
    if not _SERVED_MODEL_ECHO_RE.match(value):
        return ""
    return value


# Fail-closed sentinel (agents-core-shaperunner-fail-closed-v0): the tail's
# catch-all partition for an unclassified terminal death (not concluded,
# neither WIP-salvage-eligible nor no_progress nor max_steps) returns this
# instead of "". main() maps it to exit code 3 (distinct from 1 = call
# returned None, 2 = config error) so a ghost death shows as a FAILED
# dispatch in the claude-queue-runner log (rc is the only signal the runner
# logs) instead of a success with an empty stdout line.
TAIL_UNCLASSIFIED_DEATH = "TAIL_UNCLASSIFIED_DEATH"


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


def _gate_targeted_rerun(cwd: str, model_touched_tests: set[str],
                         timeout_s: int = 180) -> dict | None:
    """Deterministic targeted re-run of model-touched tests (local-fixer gate
    perception). Mirrors the fixer's own run_tests invocation
    (gw_agent.py:1098-1116) in shape (same env - inherited, no env override,
    cwd=worktree, shell=False, -q), with a repo-env-aware interpreter prefix
    (agents-core-gate-uv-aware-v0): _test_prefix(cwd) is the uv project venv
    (`uv run --with pytest python`, bare `python` -> venv python, the
    uv.lock-present AND uv-resolvable trigger) for uv-managed worktrees and
    the host interpreter (sys.executable) otherwise - the byte-for-byte
    legacy command. For uv repos the gate's interpreter/env therefore
    DELIBERATELY diverges from the model's own run_tests tool (which stays
    host-Python and cannot import the project deps): the gate is the
    authority and the model cannot reproduce the gate's env. Returns the
    _parse_pytest_outcome dict, or None when the re-run itself is unusable
    (timeout or spawn error).

    agents-core-local-fixer-gate-perception-v0 D1: the harness re-runs the
    tests the model touched (transcript-derived, _collect_model_touched_tests)
    so the gate decides on the CURRENT worktree state instead of the model's
    last run_tests outcome (the 0/0 last-call-wins shape). The re-run
    structurally bypasses the 8192-char tool output cap: it parses subprocess
    output directly, as the opencode F4 re-run does.

    fixers-harness-staged-v0 (S2): `timeout_s` (default 180 - legacy parity,
    pinned by the existing gate tests) is the re-run wall clock. The staged
    tail passes the mission's `tests_timeout_s` (capped [60, 240] at
    parse_mission time) because the legacy 180s cap is BELOW the measured
    189s runtime of the D2a acceptance test file - the acceptance mission
    would fail deterministically at the gate otherwise.
    """
    if not Path(cwd).is_dir():
        # A missing worktree is an unusable re-run, never a crash: the gate
        # stays fail-closed (the caller keeps its verdict) with this WARN.
        print(
            f"WARN: local-fixer: gate targeted re-run unusable - cwd {cwd!r} "
            f"is not an existing directory (touched={sorted(model_touched_tests)})",
            file=sys.stderr,
        )
        return None
    # Function-level import mirrors the F4 precedent (the opencode re-run
    # imports _parse_pytest_outcome inside the tail) so requests/
    # doorman_client stay out of shaped_runner's module import graph.
    from agents_core.gw_agent import _parse_pytest_outcome

    # argparse option-injection guard: a model-authored path starting with
    # '-' (e.g. '-kfoo/tests/test_x.py') would otherwise be consumed as a
    # pytest option and its file never run, silently skipping a red touched
    # test. shell=False makes the run_tests shell-metachar guard unnecessary.
    argv_paths = [
        ("./" + p) if p.startswith("-") else p
        for p in sorted(set(model_touched_tests))
    ]
    # agents-core-gate-uv-aware-v0: repo-env-aware prefix. Non-uv (or uv
    # unresolvable): [sys.executable, "-m", "pytest", ...] - byte-identical
    # to the legacy command. Uv: [uv_bin, "run", "--with", "pytest",
    # "python", "-m", "pytest", ...] - the project venv, not host Python.
    # The dash-path guard above still applies to argv_paths; the prefix
    # sits before the interpreter token, so `uv` passes everything after
    # the command token verbatim.
    cmd = [*_test_prefix(cwd), "-m", "pytest", *argv_paths, "-q"]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout_s,  # legacy parity: run_tests' own cap (gw_agent.py:1074)
            cwd=cwd,
            shell=False,
        )
        output = r.stdout + r.stderr
        returncode = r.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        # F4's TimeoutExpired branch only tags the output; this helper ADDS
        # the WARN so an unusable re-run is visible, not a silent 0/0.
        print(
            f"WARN: local-fixer: gate targeted re-run timed out after {timeout_s}s "
            f"(touched={sorted(model_touched_tests)})",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        # Spawn error (OSError/E2BIG, ...): an unusable re-run must never
        # propagate out of the gate into _run_local_fixer (never-raises
        # contract) - WARN + None keeps the fail-closed verdict.
        print(
            f"WARN: local-fixer: gate targeted re-run spawn error "
            f"(touched={sorted(model_touched_tests)}): {exc}",
            file=sys.stderr,
        )
        return None
    # The no-"passed"-key guard below is defensive only: _parse_pytest_outcome
    # always returns the key; the error-dict class is produced by
    # RunToolsExecutor.execute, which this helper does not call.
    outcome = _parse_pytest_outcome(output, returncode, timed_out)
    if not isinstance(outcome, dict) or "passed" not in outcome:
        print(
            f"WARN: local-fixer: gate targeted re-run returned an unusable "
            f"outcome (rc={returncode}): {outcome!r}",
            file=sys.stderr,
        )
        return None
    return outcome


def _collect_diff_touched_tests(cwd: str) -> set[str]:
    """Collect the CWD-relative paths of test files in the staged diff.

    The opencode-tail counterpart of _collect_model_touched_tests: opencode
    has no gw_agent transcript, so the source is the git index
    (`git diff --name-status --cached`), NOT a tool-call transcript (a
    transcript-shaped extractor reads tool_name/arguments and would return
    empty on a diff - Lens-3 finding). Call it AFTER `git add -A` so newly
    created (untracked) test files are staged and visible. Mirrors the
    tests/ filter of _collect_model_touched_tests. A rename row
    (R100\told\tnew) contributes its NEW path (the last field).
    """
    try:
        r = subprocess.run(
            ["git", "-C", cwd, "diff", "--name-status", "--cached"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        print("WARN: _collect_diff_touched_tests: git diff --name-status timed out",
              file=sys.stderr)
        return set()
    if r.returncode != 0:
        return set()
    touched: set[str] = set()
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        path = parts[-1].strip()
        if not path:
            continue
        if path.startswith("tests/") or "/tests/" in path or path.startswith("test_"):
            touched.add(path)
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
            # Node ID is the FIRST token after the FAILED/ERROR prefix.
            # Pytest's short summary appends the failure reason after the
            # node ID ('FAILED path::node - <reason>'), so the LAST token
            # (rsplit) would grab the reason's last word and the D1
            # touched-failure match would silently miss the failure.
            # 'ERROR at setup of tests/test_foo.py::test_y' -> the node is
            # the token after 'of'.
            if line.startswith("ERROR at setup of "):
                node = line[len("ERROR at setup of "):].strip()
            else:
                _parts = line.split(" ", 2)
                node = _parts[1].strip() if len(_parts) >= 2 else ""
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


def _empty_diff_recovery_rederive(
    worktree_path: str | Path, base_sha: str,
) -> dict | None:
    """D2 (attestation-contract-v0, leg 1): the empty-diff self-commit
    recovery re-derivation.

    Re-derives the deliverable from the worktree when the in-tail diff
    (index-vs-HEAD) is empty: `git diff --name-status <base_sha> HEAD`.
    Non-empty -> the model committed its own work (clean index, HEAD past
    base) and the gate-verified worktree state IS the deliverable:
    returns {"head_sha", "diff_summary"} (the diff summary is derived
    from <base_sha> HEAD so the PR body is not empty where the in-tail
    diff would be "(no changes)"). Empty re-derivation (the no-work case)
    -> None (the bail fires, with the WARN naming the WIP ref + HEAD sha).

    Never attempts a commit (the recovery pushes HEAD as-is - a commit on
    a clean index returns rc=1, the rev-1 bail). Never raises: any git
    failure degrades to None.
    """
    def _git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", str(worktree_path), *args],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(
                f"WARN: self-commit-recovery: git {args[0]} timed out",
                file=sys.stderr,
            )
            return subprocess.CompletedProcess(
                ["git", "-C", str(worktree_path), *args], 1, "", "timeout",
            )

    if not base_sha:
        return None
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return None
    head_sha = head.stdout.strip()
    if not head_sha:
        return None
    rederive = _git("diff", "--name-status", base_sha, head_sha)
    if rederive.returncode != 0 or not rederive.stdout.strip():
        return None
    diffstat = _git("diff", base_sha, head_sha).stdout
    diff_lines = [
        l for l in diffstat.splitlines()
        if l.startswith(("diff --git", "---", "+++", "@@", " "))
        or l[:1] in ("+", "-")
    ]
    return {
        "head_sha": head_sha,
        "diff_summary": "\n".join(diff_lines[:40]) or "(no changes)",
    }


def _empty_diff_recovery_body(
    recovered: dict, head_sha: str,
) -> str:
    """D2 (attestation-contract-v0, leg 1): the self-commit-recovery PR
    body. Carries the machine-visible marker
    `<!-- lapis-self-commit-recovery: <head_sha> -->` (the brief surfaces
    the recovery) and the non-empty diff summary derived from
    <base_sha> HEAD (the in-tail diff is empty by construction here, so
    the body must not read "(no changes)").
    """
    return (
        f"Implemented by the local fixer harness. "
        f"Self-commit recovery: the model committed its "
        f"own work (clean index, HEAD past base) and the "
        f"gate-verified worktree state was pushed as-is.\n\n"
        f"## Diff summary\n\n```diff\n"
        f"{recovered.get('diff_summary', '')}\n```\n\n"
        f"<!-- lapis-self-commit-recovery: {head_sha} -->"
    )


# D4b (attestation-contract-v0, leg 1): the friction record's on-disk home.
# The friction writer mirrors the mem entry to a JSON file here (the mem
# store is the primary record; the file is the repo-visible witness - the
# live record that carried `first_task_id: abc123` was this shape). The
# module-level name is a seam: tests monkeypatch FRICTION_DIR to a tmp dir.
# The key rides the gpu_queue base (the friction entries witness test-gate
# failures of the local-fixer harness - the gpu_queue classmap has no
# dedicated friction key, and the friction/ subdir keeps the records
# namespaced off the shaped/ artifact dir).
FRICTION_DIR = room_path("gpu_queue") / "friction"


def _write_friction_file(
    key: str, entry: dict, repo: str, log: Callable[[str], None] | None,
) -> None:
    """D4b (attestation-contract-v0, leg 1): mirror the friction entry to
    <FRICTION_DIR>/<repo>-<node-slug>.json (the repo-visible witness).

    Dedup mirrors the mem contract: an existing open record is updated
    in place (last_seen / last_task_id refresh), never duplicated. Never
    raises: a friction-write failure is logged and swallowed so it can
    never block the test gate or the PR tail.
    """
    try:
        FRICTION_DIR.mkdir(parents=True, exist_ok=True)
        path = FRICTION_DIR / f"{key.split('/', 1)[1]}.json"
        existing_json: dict = {}
        if path.exists():
            try:
                _loaded = json.loads(path.read_text())
                if isinstance(_loaded, dict):
                    existing_json = _loaded
            except (json.JSONDecodeError, OSError):
                existing_json = {}
        if existing_json:
            entry = {**existing_json, **entry}
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")
    except Exception as exc:
        if log:
            log(f"WARN: friction file write failed for {key}: {exc}")
        else:
            print(f"WARN: friction file write failed for {key}: {exc}",
                  file=sys.stderr)


def _write_friction_entry(
    *,
    repo: str,
    node_id: str = "",
    error_signature: str = "",
    task_id: str = "",
    today: str = "",
    log: Callable[[str], None] | None = None,
    category: str = "",
    detail: str = "",
) -> None:
    """Write (or dedup-update) a friction entry for a pre-existing test failure.

    Signature (attestation-contract-v0 rev-4): the canonical fields
    (node_id / error_signature / task_id / today) plus the generic
    category / detail aliases (the D4b gate-bypass caller shape -
    category names the friction class, detail the one-line witness).
    Absent fields degrade to "" / "unknown" (never the abc123
    placeholder).

    D6 (agents-core-local-fixer-harness-fix-v0): the friction entry is the
    signal a future daemon-side follow-up spec will scan for (status: open)
    and autonomously dispatch a fixer on. Key is friction/<repo>-<node-slug>
    (NO date in the key) so the same pre-existing failure across runs maps to
    the same key. Dedup: if the entry exists and is status: open, refresh
    last_seen/last_task_id without duplicating; if status: resolved, flip
    back to open (the friction recurred).

    D4b (attestation-contract-v0, leg 1): the entry carries the REAL task id
    in its task-id fields (the caller at the friction seam has it) and is
    mirrored to a file under FRICTION_DIR (the repo-visible witness). The
    `abc123` placeholder path is gone: an absent/empty task id degrades to
    "unknown", never a placeholder.

    Never raises: a friction-write failure is logged and swallowed so it can
    never block the test gate or the PR tail.
    """
    # D4b: the real task id, never the placeholder. An absent/empty id
    # degrades to "unknown" (the friction seam is generic - some callers
    # have no task id in scope).
    task_id = task_id or "unknown"
    # The generic aliases fold into the canonical fields: a category
    # names the friction class (the node slug), a detail the one-line
    # witness (the error signature).
    node_id = node_id or category or "unknown"
    error_signature = error_signature or detail or ""
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        # D4b (attestation-contract-v0, leg 1): the repo-visible witness -
        # the friction record mirrored to a file under FRICTION_DIR carrying
        # the real task id (the mem store is the primary record; the file is
        # the on-disk shape a postmortem reads).
        _write_friction_file(key, {
            "status": "open",
            "test_node_id": node_id,
            "error_signature": error_signature,
            "first_seen": today,
            "last_seen": today,
            "first_task_id": task_id,
            "last_task_id": task_id,
        }, repo, log)
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
    concluded: bool = False,
    repo_cwd: str = "",
) -> str:
    """Push the WIP ref (or HEAD for concluded worktree salvage) to a
    per-task-unique <slug>-salvage branch and open an advisory [SALVAGE] PR
    (agents-core-fixer-budget-compact-salvage-v0, S3; branch uniqueness per
    agents-core-local-fixer-salvage-on-discard-v0, S1).

    The worktree's HEAD is still at the base (detached origin/<base>); the WIP
    history lives ONLY on the separate ref. Pushing to the per-task-unique
    <slug>-salvage branch keeps a later clean run's <branch> unconflicted and
    a second salvage of the same target from clobbering the first's branch.
    The PR is advisory (never auto-merged); the run is still LOST (or
    concluded-but-gate-rejected - this returns the PR URL or "").

    agents-core-fixer-worktree-vanish-salvage-v0 (D2): the git root is the
    worktree when it still exists, else the parent clone (``repo_cwd``) when
    given - the WIP ref (refs/wip/<task_id>) lives in the SHARED common
    gitdir, so it resolves and is pushable from the parent clone after the
    worktree was deleted mid-run (verified live 2026-09-11). With neither
    available the root stays ``worktree_path`` (the pre-spec behavior - the
    push fails and the never-raises contract returns "").
    """
    import subprocess

    import agents_core.forgejo as _forgejo

    # S1: per-task-unique salvage branch (deterministic hash of the full
    # task_id). The `lf-unknown` fallback (no task_id/slot_id in the spec)
    # gets a timestamp suffix so even that degenerate case is per-run-unique
    # instead of a constant-hash collision. The in-scope `task_id` is NOT
    # modified (it is used by the WIP ref, the tail log, and the commit
    # message) - only the branch name derives from task_id_for_hash.
    task_id_for_hash = (
        f"lf-unknown-{int(time.time())}"
        if task_id == "lf-unknown"
        else task_id
    )
    salvage_branch = (
        f"{branch}-salvage-"
        f"{hashlib.sha1(task_id_for_hash.encode()).hexdigest()[:8]}"
    )

    # D2 root selection (agents-core-fixer-worktree-vanish-salvage-v0): the
    # worktree when it still exists (pre-spec behavior), else the parent
    # clone (repo_cwd) when non-empty, else worktree_path (the pre-spec
    # behavior - a deleted worktree with no repo_cwd fails the push and
    # soft-fails to ""). The None/empty guard below keeps the
    # never-raises contract, so this fallback applies to the
    # non-empty-but-deleted case only.
    if worktree_path and os.path.isdir(worktree_path):
        _git_root = worktree_path
    elif repo_cwd:
        _git_root = repo_cwd
    else:
        _git_root = worktree_path

    def _git(*args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", "-C", _git_root, *args],
                capture_output=True, text=True, timeout=60,
            )
        except subprocess.TimeoutExpired:
            print(f"WARN: wip-salvage: git {args[0]} timed out", file=sys.stderr)
            return subprocess.CompletedProcess(["git", "-C", _git_root, *args], 1, "", "timeout")

    if not worktree_path or not wip_head_sha:
        print("WARN: wip-salvage: no WIP head sha - no PR", file=sys.stderr)
        return ""

    # Push the WIP history (or the concluded salvage commit at HEAD) to the
    # salvage branch.
    r = _git("push", "origin", f"{wip_ref}:refs/heads/{salvage_branch}")
    if r.returncode != 0:
        print(f"WARN: wip-salvage: git push failed: {r.stderr.strip()}", file=sys.stderr)
        _tail_log(task_id, "wip-salvage push failed - no PR")
        return ""

    # PR body: task id, stop_reason + wall time, WIP head sha, the steps
    # included, and what remains per the spec. S4: the body's first line and
    # the "## What remains" section branch on the concluded flag - the title
    # holds the cause (stop_reason), the body holds the state.
    steps_included = ", ".join(str(s) for s in wip_steps) if wip_steps else "n/a"
    sha_field_label = (
        "salvage commit sha" if concluded else "WIP head sha"
    )
    if not concluded:
        what_remains = (
            f"Per the bound spec (staged as `lapis-spec.md` during the run, removed "
            f"after): the spec's remaining work items are NOT in this salvage. The "
            f"WIP commits are a compile-gated snapshot of the whole-file writes "
            f"made up to the death - a partial implementation at best.\n\n"
            f"Note: the WIP floor is CUMULATIVE (attestation-contract-v0, leg 1, "
            f"D1) - the branch HEAD's tree carries every write step's files at "
            f"their final content (the hook loads the prior WIP tip's tree into "
            f"the index before staging each step's paths). Deletion tolerance "
            f"is named: a file deleted in a later step persists at the tip "
            f"(add does not prune) - the floor is a salvage floor, not a "
            f"worktree mirror. The commit history still carries the per-step "
            f"granularity (git log) for finer reconstruction.\n"
        )
    else:
        what_remains = (
            f"This PR carries the worktree's FINAL state - the exact state the test "
            f"gate tested - not a WIP snapshot. The model's final test run failed; "
            f"see the transcript for the gate's failures. The remaining work is "
            f"making that final state pass the test gate.\n"
        )
    # D2 first-line state case (agents-core-fixer-worktree-vanish-salvage-v0):
    # a run that DID conclude but lost its diff to a mid-run worktree
    # deletion must not be mislabeled "not concluded, no gate_passed".
    # concluded=False stays in force for the "What remains" section below -
    # the WIP-snapshot text (incl. the non-cumulative-head caveat) is the
    # accurate description of what a refs/wip/<task_id> push carries; the
    # concluded=True "worktree's FINAL state / model's final test run
    # failed" text would be wrong for this shape.
    if stop_reason == "concluded_empty_diff_wip_salvage":
        _state_line = "concluded, but the diff was lost to a mid-run worktree deletion"
    elif not concluded:
        _state_line = "not concluded, no gate_passed"
    else:
        _state_line = "concluded, test gate rejected the work"
    pr_body = (
        f"**[SALVAGE] advisory PR - the run is LOST "
        f"({_state_line})."
        f" stop_reason: {stop_reason}.\n\n"
        f"## Task\n\n"
        f"- task_id: `{task_id}`\n"
        f"- target_id: `{target_id}`\n"
        f"- stop_reason: `{stop_reason}`\n"
        f"- wall time: {datetime.now(timezone.utc).isoformat()}\n"
        f"- {sha_field_label}: `{wip_head_sha}`\n"
        f"- steps included: {steps_included}\n"
        f"- total steps executed: {step_count}\n\n"
        f"## What remains\n\n"
        f"{what_remains}\n\n"
        f"## Recovery\n\n"
        f"`lapis-pm bind {target_id} --force --adopt-pr <n>` + fixer_retry "
        f"(proven 2026-08-25 on PR #258).\n\n"
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
        _tail_log(task_id, f"[SALVAGE] PR opened url={pr.get('html_url', '')}")
        return pr.get("html_url", "")
    except Exception as exc:
        print(f"WARN: wip-salvage: create_pr failed: {exc}", file=sys.stderr)
        _tail_log(task_id, "wip-salvage create_pr failed - no PR")
        return ""


# ---------------------------------------------------------------------------
# agents-core-local-fixer-gate-nonpython-v0: repo-aware D1 gate + tail log.
# ---------------------------------------------------------------------------

def _has_python_test_infra(cwd) -> bool:
    """Does this worktree have Python test infrastructure (pytest-detectable)?

    agents-core-local-fixer-gate-nonpython-v0 Deliverable 1: True when ANY of:
    a root pyproject.toml / setup.py / pytest.ini / conftest.py exists; a root
    setup.cfg contains a [tool:pytest] section (a packaging-only setup.cfg is
    NOT pytest infra); at least one test_*.py / *_test.py at depth <= 2 below
    cwd; or a tests/ / *_tests/ directory (at depth <= 2) containing at least
    one .py file. Bounded scan: depth cap 2 (no deeper recursion; no per-dir
    file-count cap - a count cap would risk the bypass-direction false
    negative, which is the dangerous direction). Any OSError returns True
    (fail-safe: treat as infra present - the gate behaves as today; the
    bypass can never be caused by a filesystem surprise).
    """
    root = Path(cwd)

    def _scan(d: Path, depth: int) -> bool:
        # depth: levels below root (root itself = 0). Files are in scope at
        # depth <= 2; prune descent beyond depth 2.
        if depth > 2:
            return False
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return True  # fail-safe: unexpected filesystem = infra present
        for e in entries:
            try:
                if e.is_file():
                    if e.name.endswith(".py") and (
                            e.name.startswith("test_") or e.name.endswith("_test.py")):
                        return True
                elif e.is_dir():
                    if e.name == "tests" or e.name.endswith("_tests"):
                        try:
                            if any(f.is_file() and f.suffix == ".py"
                                   for f in e.iterdir()):
                                return True
                        except OSError:
                            return True
                    if _scan(e, depth + 1):
                        return True
            except OSError:
                return True  # fail-safe (entry-level stat surprise)
        return False

    try:
        for name in ("pyproject.toml", "setup.py", "pytest.ini", "conftest.py"):
            if (root / name).is_file():
                return True
        cfg = root / "setup.cfg"
        if cfg.is_file():
            try:
                if "[tool:pytest]" in cfg.read_text(errors="replace"):
                    return True
            except OSError:
                return True
        return _scan(root, 0)
    except OSError:
        return True


def _test_prefix(cwd) -> list[str]:
    """Interpreter-inclusive command prefix for the deterministic test gate.

    agents-core-gate-uv-aware-v0 (Deliverable 1): returns
    `[uv_bin, "run", "--with", "pytest", "python"]` iff BOTH
    (a) `Path(cwd) / "uv.lock"` is a file (the worktree is a uv-managed
    repo - `uv.lock` discriminates "uv-managed, has a locked env";
    `pyproject.toml` would NOT - agents-core itself has a pyproject.toml
    and no uv.lock and must stay on host Python) AND
    (b) `uv_bin = shutil.which("uv")` is not None (uv is resolvable in
    THIS process env - on BRIX the claude-queue-runner service PATH does
    not include /srv/fast/scratch/bin, so without the Deploy-note
    precondition this is None and the gate degrades to the host-Python
    fallback: the known failure, not a new spawn error). Otherwise it
    returns `[sys.executable]` - byte-for-byte today's command.

    The returned list is the interpreter-inclusive prefix: prepending it
    and then `"-m", "pytest", ...` yields the correct form in both cases
    (non-uv: `[sys.executable, "-m", "pytest", ...]`; uv:
    `[uv_bin, "run", "--with", "pytest", "python", "-m", "pytest", ...]`).
    The interpreter token is deliberately the BARE `python` for uv repos -
    `uv run` syncs the locked project env and prepends its bin to PATH, so
    the bare token resolves to the VENV python that imports the project
    deps, and `--with pytest` adds pytest ephemerally without touching the
    lockfile. A `sys.executable` token under `uv run` would run the HOST
    interpreter literally (an absolute interpreter is run as-is; the venv
    on PATH does not change its sys.path) and re-introduce the exact
    ModuleNotFoundError false negative this helper exists to eliminate
    (finding/lapis-fixer-test-gate-host-python-uv-repo-2026-09-05).

    Never raises (mirrors _has_python_test_infra's fail-safe style): a
    missing/unreadable uv.lock or an unresolvable uv -> `[sys.executable]`.
    """
    try:
        if not (Path(cwd) / "uv.lock").is_file():
            return [sys.executable]
        uv_bin = shutil.which("uv")
        if uv_bin is None:
            return [sys.executable]
        return [uv_bin, "run", "--with", "pytest", "python"]
    except Exception:
        return [sys.executable]


def _tail_log(task_id: str, message: str) -> None:
    """Append one timestamped line to <shaped-dir>/<task_id>-tail.log.

    agents-core-local-fixer-gate-nonpython-v0 Deliverable 4: the tail's phase
    log was captured nowhere (the systemd scope journal has no entries; the
    queue-runner logs only claim/done lines), so a lost run's reason was
    undiagnosable after the fact. Best-effort: any OSError is swallowed -
    logging must never fail a run. Additive to the existing stderr prints,
    not a replacement. Reason strings, returncodes and URLs only: no
    secrets, no tokens, no diff content (callers truncate git stderr to
    <= 500 chars). Admitted content class added by
    agents-core-local-fixer-gate-perception-v0 D2: the deciding test
    outcome's pytest summary line (single-line by construction, truncated
    to 200 chars by the caller) - no diff content, no env values except
    what a test itself prints.
    """
    try:
        shaped_dir = room_path("gpu_queue.shaped")
        shaped_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(shaped_dir / f"{task_id}-tail.log", "a", encoding="utf-8") as f:
            f.write(f"{ts} {message}\n")
    except OSError:
        pass


def tail_finalize(
    *,
    task_id: str,
    target_id: str,
    bare_repo: str,
    branch: str,
    slug: str,
    cwd: str,
    worktree_path: str | None,
    final_diff: str,
    concluded: bool,
    last_test_outcome: dict | None,
    max_steps_hit: bool,
    no_progress_hit: bool,
    stop_reason: str,
    step_count: int,
    transcript_path: Path,
    gate_passed: bool,
    gate_bypassed,
    model_touched_tests,
    gate_rerun_fired: bool,
    wip_ref: str = "",
    wip_commit_count: int = 0,
    wip_head_sha: str = "",
    wip_steps: list = None,
    _wip_git=None,
    seat_alias: str = "",
    served_model: str = "",
    worktree_vanished: bool = False,
    repo_cwd: str = "",
    base_sha: str = "",
) -> str:
    """The shared deterministic git/PR tail (fixers-harness-staged-v0, S6).

    Extracted from the `_run_local_fixer` tail so the staged engine
    (`_run_local_fixer_staged`) reuses the exact same tail mechanics
    (gate decision, salvage partitions, push/PR). The signature carries
    EVERY value the legacy tail region consumes (the full seam - not a
    pinned count): task_id / target_id / bare_repo / gate_rerun_fired are
    named explicitly; the legacy path calls it with its existing values
    (behavior-pinned by the existing tail tests, test_wip_salvage).

    Returns a PR URL on success, "" on any failure — never raises.
    """
    import subprocess

    import agents_core.forgejo as _forgejo

    if wip_steps is None:
        wip_steps = []

    # Deterministic git (model never touches git) - defined at tail entry
    # so EVERY path (the empty-diff bail, the self-commit recovery, the
    # salvage partitions, the normal PR path) can use it. The legacy
    # tail defined it later (after the salvage partitions); the bail
    # needs it earlier, so it is hoisted here with the identical body.
    def _git(*args: str) -> subprocess.CompletedProcess:
        try:
            import subprocess as _subprocess_mod
            return _subprocess_mod.run(
                ["git", "-C", cwd, *args],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            print(f"WARN: local-fixer: git {args[0]} timed out", file=sys.stderr)
            return subprocess.CompletedProcess(["git", "-C", cwd, *args], 1, "", "timeout")

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
    if gate_bypassed:
        # The no-python-test-infra bypass (agents-core-local-fixer-gate-nonpython-v0
        # D2) is authoritative: the caller already set gate_passed=True
        # because the worktree has no Python test infrastructure. The
        # pytest-only re-derivation below must NOT override it - a
        # claude-view-class repo can never pass the pytest gate, so
        # re-deriving would discard the bypassed work as [SALVAGE].
        pass
    elif model_touched_tests:
        # Positive-only gate: every test the model touched must pass.
        # A touched test "fails" if a FAILED/ERROR node ID refers to it.
        # A node ID "refers" to a touched test file if the node's file
        # part equals the touched path (file-level failure) OR the node
        # is a specific test within that file (node-level failure).
        #
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
        if last_test_outcome is not None and "passed" in last_test_outcome:
            passed_c = int(last_test_outcome.get("passed") or 0)
            if passed_c > 0 and not touched_failures:
                gate_passed = True
    else:
        # Fail-closed fallback: no tests touched -> legacy gate.
        # D3 (agents-core-local-fixer-gate-perception-v0): an error-dict
        # outcome (no "passed" key, the {"error": ...} class from
        # gw_agent.py:1085) is a no-valid-last-result here - it stays
        # fail-closed (the _tests_passed False return) but is logged as
        # no-valid-last-result, not as a spurious passed=0 failed=0.
        if last_test_outcome is not None and "passed" not in last_test_outcome:
            _tail_log(task_id, "gate: no-valid-last-result (error-dict outcome)")
        gate_passed = _tests_passed(last_test_outcome)

    # D1 (agents-core-local-fixer-gate-perception-v0): targeted re-run
    # when the touched branch would fail closed. STRICTLY MORE PERMISSIVE:
    # the re-run only fires when the gate would otherwise fail closed -
    # it can flip fail->pass, never pass->fail. The re-run re-tests the
    # CURRENT worktree state, so a fix made after the model's last
    # run_tests is picked up (the v2-retry shape). The re-run outcome
    # REPLACES the model's last outcome as the DECIDING outcome for the
    # gate and the D2/D5 surfaces when it fires (the re-run is the
    # authoritative tail - the local-fixer path adopts the opencode
    # tail's doctrine by analogy). Decision rule: returncode == 0
    # (rc=1 failures, rc=2 collection error, rc=4 usage/path error,
    # rc=5 no tests collected - an empty touched file correctly fails
    # closed). A None (unusable) re-run keeps the fail-closed verdict.
    #
    # CURRENT worktree state, so a fix made after the model's last
    # run_tests is picked up (the v2-retry shape). The re-run outcome
    # REPLACES the model's last outcome as the DECIDING outcome for the
    # gate and the D2/D5 surfaces when it fires (the re-run is the
    # authoritative tail - the local-fixer path adopts the opencode
    # tail's doctrine by analogy). Decision rule: returncode == 0
    # (rc=1 failures, rc=2 collection error, rc=4 usage/path error,
    # rc=5 no tests collected - an empty touched file correctly fails
    # closed). A None (unusable) re-run keeps the fail-closed verdict.
    if (
        gate_bypassed is None
        and not gate_passed
        and model_touched_tests
        and (
            last_test_outcome is None
            or "passed" not in last_test_outcome
            or int(last_test_outcome.get("passed") or 0) == 0
            or touched_failures
        )
    ):
        # cwd pinning: pass the gate's own cwd (the worktree root - the
        # same variable _collect_model_touched_tests used above); a wrong
        # cwd would make every touched path rc=4 (a silent no-op that
        # voids the strictly-more-permissive invariant in production).
        gate_rerun_fired = True
        # agents-core-gate-uv-aware-v0 (Deliverable 4): name the active
        # interpreter prefix + its trigger - no code path today logs the
        # gate argv, so this is what makes the Deploy-note verification
        # observable.
        _prefix = _test_prefix(cwd)
        if _prefix[0] != sys.executable:
            _tail_log(
                task_id,
                "gate re-run via uv run --with pytest "
                "(uv.lock present, uv resolvable)",
            )
        else:
            _tail_log(
                task_id,
                "gate re-run via host python "
                "(no uv.lock or uv not resolvable)",
            )
        rerun_outcome = _gate_targeted_rerun(cwd, model_touched_tests)
        # agents-core-fixer-worktree-vanish-salvage-v0 (D1): a fired re-run
        # that produced NO usable outcome (None - timeout/spawn error or
        # cwd gone) must keep the unusable label; without this flag the
        # model's last outcome (a dict) survives in last_test_outcome and
        # _rerun_diag mislabels the gate FAILED line as
        # rerun=true (last: ...).
        rerun_no_outcome = rerun_outcome is None
        if rerun_outcome is not None:
            last_test_outcome = rerun_outcome
            gate_passed = (rerun_outcome.get("returncode") == 0)
            _write_friction_entry(
                repo=bare_repo,
                node_id=task_id,
                error_signature="gate-rerun:fired",
                task_id=task_id,
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                log=lambda m: print(m, file=sys.stderr),
            )

    def _outcome_diag(outcome: dict | None) -> str:
        """D2 disambiguation: the deciding outcome's errors/rc/summary
        (single-line by construction, truncated to 200 chars)."""
        if outcome is None:
            return "last_test_outcome=None"
        if "passed" not in outcome:
            return "no-valid-last-result (error-dict outcome)"
        summary = str(outcome.get("summary") or "").replace("\n", " ")[:200]
        return (
            f"passed={int(outcome.get('passed') or 0)} "
            f"failed={int(outcome.get('failed') or 0)} "
            f"errors={int(outcome.get('errors') or 0)} "
            f"rc={outcome.get('returncode')} summary={summary!r}"
        )

    def _rerun_diag() -> str:
        if not gate_rerun_fired:
            return ""
        # D4: name the unusable re-run shape explicitly - a touched
        # path that vanished between the model's writes and the
        # re-run is an event, not a silent 0/0 (rc=4/5 are the
        # pytest-path shapes; rc=2/3 cover the re-run's own
        # collection/internal errors).
        rerun_rc = (last_test_outcome or {}).get("returncode")
        if rerun_rc in (4, 5):
            name = (
                "rerun-rc=4 touched-path-missing"
                if rerun_rc == 4
                else "rerun-rc=5 no-tests-ran"
            )
            return f"; {name}"
        if rerun_no_outcome or last_test_outcome is None:
            # The re-run fired but was unusable: the fail-closed verdict
            # is kept and the re-run's non-participation is named, not
            # folded into the model's last outcome.
            # agents-core-fixer-worktree-vanish-salvage-v0 (D1): the
            # cwd-missing shape (external worktree deletion - the
            # "cwd ... is not an existing directory" WARN from
            # _gate_targeted_rerun) is a DIFFERENT cause of death than a
            # genuine timeout/spawn error, and mislabelling it sent the
            # 2026-09-11 PM session down a wrong theory. Name it.
            if not os.path.isdir(cwd):
                return "; rerun=fired-but-unusable:cwd-missing"
            return "; rerun=fired-but-unusable (timeout/spawn error)"
        # D1 (agents-core-fixer-worktree-vanish-salvage-v0): a re-run that
        # fired but whose cwd is now missing is the same external-deletion
        # cause of death as the None shape above (the partial-deletion
        # shape - the worktree root is gone while the re-run's own cwd
        # survived, so the re-run still produced an outcome). Name it
        # instead of folding it into the model's last outcome as a
        # rerun=true success.
        if not os.path.isdir(cwd):
            return "; rerun=fired-but-unusable:cwd-missing"
        return (
            "; rerun=true (last: "
            f"passed={int((last_test_outcome or {}).get('passed') or 0)} "
            f"failed={int((last_test_outcome or {}).get('failed') or 0)} "
            f"errors={int((last_test_outcome or {}).get('errors') or 0)} "
            f"rc={last_test_outcome.get('returncode')})"
        )

    if gate_bypassed is None:
        if gate_passed:
            _tail_log(
                task_id,
                "gate PASSED "
                f"(model_touched_tests={sorted(model_touched_tests) if model_touched_tests else '[] (legacy)'}"
                f"{_rerun_diag()} "
                f"{_outcome_diag(last_test_outcome)})",
            )
        else:
            _tail_log(
                task_id,
                "gate FAILED "
                f"(model_touched_tests={sorted(model_touched_tests) if model_touched_tests else '[] (legacy)'}; "
                f"{_outcome_diag(last_test_outcome)}"
                f"{_rerun_diag()})",
            )

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
        _tail_log(task_id, "run aborted - mem_search_loop (redundant mem searches; no edits made)")
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
            _tail_log(task_id, "run aborted - output_budget_exhausted - opening advisory [SALVAGE] PR")
            return _open_wip_salvage_pr(
                worktree_path, wip_ref, wip_head_sha, wip_steps,
                stop_reason=stop_reason, task_id=task_id,
                target_id=target_id, bare_repo=bare_repo,
                branch=branch, slug=slug,
                step_count=step_count,
                transcript_path=transcript_path,
            )
        print(
            "WARN: local-fixer: run aborted - output budget exhausted "
            "(finish_reason=output_limit/length; response truncated at "
            "max_tokens - no WIP commits, no PR)",
            file=sys.stderr,
        )
        _tail_log(task_id, "run aborted - output_budget_exhausted (no WIP commits) - no PR")
        return ""

    # WIP-commit salvage on a non-concluded terminal death (agents-core-
    # fixer-budget-compact-salvage-v0, S3): max_steps_hit / no_progress_hit
    # with >=1 WIP commit pushes the WIP history to a <slug>-salvage branch
    # and opens an advisory [SALVAGE] PR. Ordering rule: the green-salvage
    # path below takes precedence when it applies (clean diff AND passing
    # tests) - a dead run CAN have both, and its verified tail is better
    # than the WIP history. The WIP-salvage PR is for the remainder.
    # The run is still LOST.
    # S2 (agents-core-local-fixer-salvage-on-discard-v0): the trigger no
    # longer requires a budget flag. Partition: budget+green -> both
    # disjuncts false -> Block B's green path (UNCHANGED); budget+non-green
    # -> second disjunct -> WIP-salvage (as today); ANY non-budget death
    # with >=1 WIP commit (seat loss, mid-run POST failure, no-choices,
    # budget-forced, interrupted/cancelled - any gate/diff state) -> first
    # disjunct -> WIP-salvage. A non-budget GREEN death gets an advisory
    # [SALVAGE] PR, not the green path's normal PR (the green path stays
    # budget-gated).
    if (not concluded
            and wip_commit_count > 0
            and (not (max_steps_hit or no_progress_hit)
                 or not (final_diff.strip() and gate_passed))):
        _wip_stop_reason = (
            "max_steps_hit" if max_steps_hit
            else "no_progress_hit" if no_progress_hit
            else "run_not_concluded"
        )
        print(
            f"WARN: local-fixer: run not concluded - {_wip_stop_reason} "
            f"- opening advisory [SALVAGE] PR",
            file=sys.stderr,
        )
        _tail_log(task_id, f"run not concluded - {_wip_stop_reason} - opening advisory [SALVAGE] PR")
        return _open_wip_salvage_pr(
            worktree_path, wip_ref, wip_head_sha, wip_steps,
            stop_reason=_wip_stop_reason, task_id=task_id,
            target_id=target_id, bare_repo=bare_repo,
            branch=branch, slug=slug,
            step_count=step_count,
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
            _tail_log(task_id, "run aborted - no semantic progress (spinning wheels)")
            return ""
        elif max_steps_hit:
            print(
                "WARN: local-fixer: run not concluded - max_steps ceiling reached (no passing tests or empty diff)",
                file=sys.stderr,
            )
            _tail_log(task_id, "run not concluded - max_steps ceiling reached (no passing tests or empty diff)")
            return ""
        else:
            # Fail-closed (agents-core-shaperunner-fail-closed-v0): this
            # catch-all is reached ONLY for an unclassified terminal death
            # (not concluded, neither WIP-salvage-eligible nor no_progress
            # nor max_steps - e.g. a model-down / context death). Report the
            # OBSERVED terminal cause from the run record (stop_reason), or
            # say exactly that no cause was recorded - never a doorman claim
            # (a real DoormanUnreachable is caught and soft-failed at the
            # lease-acquire sites, never at the tail). Return the distinct
            # sentinel (not "") so main() exits 3 and the queue runner
            # records the dispatch as failed instead of a silent rc=0.
            if stop_reason:
                _death_cause = f"stop_reason={stop_reason}"
            else:
                _death_cause = "no stop_reason recorded"
            print(
                f"ERROR: local-fixer: run not concluded - unclassified "
                f"terminal death ({_death_cause})",
                file=sys.stderr,
            )
            _tail_log(
                task_id,
                f"run not concluded - unclassified terminal death ({_death_cause})",
            )
            return TAIL_UNCLASSIFIED_DEATH

    if not salvaged:
        if not final_diff.strip():
            # Cycle-4 reviewer (PR #322) routing fix: a CONCLUDED run with
            # >=1 WIP commit and an empty in-tail diff (the model
            # self-committed - the index-vs-HEAD diff is empty by
            # construction) must route to the existing [SALVAGE] path
            # BEFORE the D2 normal-PR self-commit recovery below. The D2
            # recovery is scoped to the NON-concluded shape (the guard at
            # the recovery site: `base_sha and gate_passed and not
            # concluded`); without this early return the concluded shape
            # falls through the recovery skip and reaches the
            # `if not gate_passed:` partition, which is skipped for a
            # gate-PASSED run - so the run bails at the normal-PR path's
            # clean-index commit rc=1 with NO PR and NO salvage (the
            # silent-loss shape the D2 finding exists to kill). A
            # concluded run's final worktree state is exactly what the
            # gate tested, so the concluded_gate_rejected worktree-salvage
            # partition (below, which commits the worktree state and
            # pushes HEAD as the salvage commit) is the correct carrier -
            # a [SALVAGE] PR is advisory (never auto-merged), which is
            # the right disposition for gate-green-but-unconcluded-shape
            # work (Standing ratification 3: salvage semantics unchanged -
            # a run that opens a [SALVAGE] PR is not a death). The
            # worktree_vanished partition above runs first (it returns
            # early) - a vanished worktree is a distinct death class.
            if (concluded and wip_commit_count > 0
                    and base_sha
                    and _empty_diff_recovery_rederive(cwd, base_sha) is not None):
                print(
                    "WARN: local-fixer: concluded, empty in-tail diff, "
                    "HEAD past base (the model self-committed) - opening "
                    "advisory [SALVAGE] PR (concluded, gate passed, "
                    "worktree salvage)",
                    file=sys.stderr,
                )
                _tail_log(
                    task_id,
                    "concluded, empty diff, HEAD past base - opening "
                    "advisory [SALVAGE] PR (worktree salvage)",
                )
                return _open_wip_salvage_pr(
                    worktree_path, "HEAD",
                    _git("rev-parse", "HEAD").stdout.strip(), [],
                    stop_reason="concluded_gate_rejected",
                    concluded=True, task_id=task_id,
                    target_id=target_id, bare_repo=bare_repo,
                    branch=branch, slug=slug,
                    step_count=step_count,
                    transcript_path=transcript_path,
                )
            # agents-core-fixer-worktree-vanish-salvage-v0 (D1/D2): the
            # concluded + empty-diff + WIP-present partition (the #291
            # class - a vanished worktree: the model finished, git add -A
            # failed on the deleted cwd, the WIP ref survives in the
            # parent clone's shared gitdir). Ordering (Council synthesis
            # 2026-09-11-121532): the PRIMARY diagnostic line is static -
            # it names the cause of death and is emitted FIRST, before the
            # recovery attempt; recovery nuance is carried by a separate
            # appended line + the friction entry, never by rewriting the
            # primary line. The run is still LOST (no
            # concluded/gate_passed semantics) - the salvage PR is
            # advisory-only, never auto-merged (inherited from the
            # existing salvage machinery); the positive-only gate is
            # unchanged (fail-closed).
            # (a) the vanished-worktree partition runs FIRST - it returns
            # early: a vanished worktree is a distinct death class and must
            # be ruled out before any recovery that needs a live cwd.
            if worktree_vanished:
                _vanished_line = (
                    "run discarded - worktree vanished mid-run "
                    "(external deletion; NOT a gate or test-runner failure)"
                )
                print(
                    f"WARN: local-fixer: {_vanished_line}",
                    file=sys.stderr,
                )
                _tail_log(task_id, _vanished_line)
                _write_friction_entry(
                    repo=bare_repo,
                    node_id="worktree-vanished",
                    error_signature="worktree-vanished:mid-run",
                    task_id=task_id,
                    today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    log=lambda m: print(m, file=sys.stderr),
                )
                if wip_commit_count > 0 and wip_head_sha:
                    # D2: open the advisory [SALVAGE] PR from the WIP ref
                    # (pushable from the parent clone - repo_cwd fallback
                    # root). The primary line above is already written and
                    # stays immutable; _open_wip_salvage_pr appends its own
                    # "[SALVAGE] PR opened" line on success or a distinct
                    # footnote line on push failure. Refresh the stable-key
                    # friction entry with the recovery outcome so the
                    # recurrence + salvage_success trend is legible via
                    # `mem search worktree-vanished`.
                    print(
                        "WARN: local-fixer: worktree vanished - WIP commits "
                        f"exist ({wip_commit_count}) - opening advisory "
                        "[SALVAGE] PR (concluded, empty diff, WIP salvage)",
                        file=sys.stderr,
                    )
                    _salvage_url = _open_wip_salvage_pr(
                        worktree_path, wip_ref, wip_head_sha, wip_steps,
                        stop_reason="concluded_empty_diff_wip_salvage",
                        task_id=task_id,
                        target_id=target_id,
                        bare_repo=bare_repo,
                        branch=branch, slug=slug,
                        step_count=step_count,
                        transcript_path=transcript_path,
                        concluded=False,
                        repo_cwd=repo_cwd,
                    )
                    try:
                        from agents_core.mem import MemoryStore

                        _store = MemoryStore()
                        try:
                            _fkey = f"friction/{bare_repo}-worktree-vanished"
                            _existing = _store.get(_fkey)
                            _fjson: dict = {}
                            if _existing:
                                try:
                                    _fjson = json.loads(
                                        _existing.get("content") or "{}")
                                except (json.JSONDecodeError, TypeError):
                                    _fjson = {}
                            # Preserve the entry's status field (the
                            # _write_friction_entry contract: status: open
                            # is the dedup/recurrence signal; a resolved
                            # entry is flipped back to open on recurrence)
                            # - the raw refresh must not drop it.
                            if _fjson.get("status") != "open":
                                _fjson["status"] = "open"
                            _fjson["salvage_success"] = bool(_salvage_url)
                            _store.set(
                                _fkey,
                                json.dumps(_fjson, ensure_ascii=False),
                                tags=["friction", "test-gate", bare_repo],
                            )
                        finally:
                            _store.close()
                    except Exception as exc:
                        print(
                            f"WARN: local-fixer: friction salvage_success "
                            f"refresh failed: {exc}",
                            file=sys.stderr,
                        )
                    return _salvage_url
                # No WIP commits: the distinct vanished line + friction
                # entry are the whole outcome (invariant: no-WIP behavior
                # unchanged - no PR).
                return ""
            # (b) D2 (attestation-contract-v0, leg 1): empty-diff recovery for
            # self-committed GREEN work. final_diff is index-vs-HEAD; when
            # the model committed its own work (the documented case - the
            # run's tail would otherwise bail on a clean index and drop a
            # gate-green run with no PR and no salvage), re-derive the
            # deliverable from the worktree: HEAD vs the captured base.
            #
            # gate_passed guard (attestation-contract-v0 rev-4): the
            # recovery opens a NORMAL PR - a run with a clean index and a
            # FAILED gate must NOT push/open a PR through it. That shape
            # routes to the [SALVAGE] path below (the if-not-gate_passed
            # block) instead.
            #
            # concluded guard (cycle-4 reviewer, PR #322): the D2 contract
            # is scoped to the NON-concluded self-commit case. The
            # concluded + gate_passed + empty-tail-diff + HEAD-past-base
            # shape keeps routing to the existing [SALVAGE] path exactly
            # as before (the D2 DO-NOT-CHANGE list / Standing ratification
            # 3: D1/D2 change how the WIP tree is constructed and how the
            # empty-diff bail recovers, NOT when salvage fires - a run
            # that opens a [SALVAGE] PR is not a death). The concluded
            # shape therefore skips the normal-PR recovery below and
            # falls through to the concluded [SALVAGE] partitions:
            # worktree_vanished -> the concluded_empty_diff_wip_salvage
            # partition (above), else (with WIP commits) the
            # concluded_gate_rejected worktree-salvage partition (below,
            # which commits the worktree state - a no-op commit here on
            # the clean index, then pushes HEAD as the salvage commit),
            # else the WARN bail (no work past base).
            _recovery_head = ""
            if base_sha and gate_passed and not concluded:
                _recovered = _empty_diff_recovery_rederive(cwd, base_sha)
                if _recovered is not None:
                    # The model self-committed: the gate-verified worktree
                    # state IS the deliverable. Skip the commit step (a
                    # commit on a clean index returns rc=1 -> the rev-1
                    # bail) and push HEAD as-is (the gate-rejected salvage
                    # path's established wip_ref="HEAD" in-file pattern).
                    #
                    # Spec basis for push-HEAD-as-is (cycle-2 review
                    # attestation): the recovery must NOT re-run the gate
                    # on the recovered diff - the gate already ran on
                    # exactly this worktree state (the in-tail diff is
                    # empty by construction: the model committed its own
                    # work, so worktree == HEAD == the gate-tested state).
                    # Re-running the gate would test the identical state a
                    # second time; the gate_passed check above is the
                    # complete gate attestation for this path.
                    _recovery_head_sha = _recovered["head_sha"]
                    _recovery_diffstat = _recovered["diff_summary"]
                    print(
                        f"INFO: local-fixer: empty in-tail diff but HEAD is "
                        f"past base ({base_sha[:12]} -> "
                        f"{_recovery_head_sha[:12]}) - the model "
                        f"self-committed; opening the PR from the worktree "
                        f"HEAD (self-commit recovery)",
                        file=sys.stderr,
                    )
                    _tail_log(
                        task_id,
                        f"self-commit recovery: index clean, HEAD "
                        f"{_recovery_head_sha[:12]} past base "
                        f"{base_sha[:12]} - PR from HEAD",
                    )
                    r = _git("checkout", "-B", branch)
                    if r.returncode != 0:
                        print(
                            f"WARN: local-fixer: git checkout -B failed "
                            f"(self-commit recovery): {r.stderr.strip()}",
                            file=sys.stderr,
                        )
                        _tail_log(
                            task_id,
                            f"self-commit recovery: git checkout -B failed "
                            f"rc={r.returncode}: {r.stderr.strip()[:500]}",
                        )
                        return ""
                    r = _git("push", "origin", f"HEAD:{branch}")
                    if r.returncode != 0:
                        print(
                            f"ERROR: local-fixer: git push failed "
                            f"(self-commit recovery) - command: git push "
                            f"origin HEAD:{branch} (rc={r.returncode}): "
                            f"{r.stderr.strip()}",
                            file=sys.stderr,
                        )
                        _tail_log(
                            task_id,
                            f"pm:push-failed rc={r.returncode} "
                            f"branch={branch} (self-commit recovery) "
                            f"stderr={r.stderr.strip()[:500]}",
                        )
                        return ""
                    # The pre-aimed / parked-PR case: the branch already
                    # has an open PR and the push advanced its head.
                    if branch:
                        try:
                            for _pr in _forgejo.get_open_prs(repo=bare_repo):
                                if (_pr.get("head") or {}).get("ref") == branch:
                                    _pr_url = _pr.get("html_url", "")
                                    _tail_log(
                                        task_id,
                                        f"open PR already exists on "
                                        f"{branch!r} (self-commit "
                                        f"recovery) - head advanced: "
                                        f"{_pr_url}",
                                    )
                                    return _pr_url
                        except Exception as exc:
                            print(
                                f"WARN: local-fixer: open-PR scan failed "
                                f"(self-commit recovery): {exc} - falling "
                                f"through to create_pr",
                                file=sys.stderr,
                            )
                    # Provenance PR body - the diff summary is derived from
                    # <base_sha> HEAD (the in-tail diff is empty by
                    # construction here, so the body must not read
                    # "(no changes)"); the marker makes the recovery
                    # machine-visible to the brief.
                    _recovery_body = (
                        _empty_diff_recovery_body(
                            _recovered, _recovery_head_sha,
                        )
                        + f"\n\n## Test outcome\n\n{test_summary}\n\n"
                        f"## Steps\n\n{step_count} tool-call step(s) "
                        f"executed.\n\n"
                        f"## Transcript\n\n`{transcript_path}`\n\n"
                        f"<!-- lapis-gpu-id: {task_id} -->\n"
                        f"<!-- lapis-tid: {target_id} -->\n"
                        f"<!-- lapis-engine: local-fixer -->"
                    )
                    try:
                        pr = _forgejo.create_pr(
                            repo=bare_repo,
                            title=f"fix({target_id}): local-fixer",
                            head=branch,
                            base="main",
                            body=_recovery_body,
                        )
                    except Exception as exc:
                        print(
                            f"WARN: local-fixer: create_pr failed "
                            f"(self-commit recovery): {exc}",
                            file=sys.stderr,
                        )
                        _tail_log(
                            task_id,
                            f"create_pr FAILED (self-commit recovery): "
                            f"{exc}",
                        )
                        return ""
                    _tail_log(
                        task_id,
                        f"create_pr OK (self-commit recovery) "
                        f"url={pr.get('html_url', '')}",
                    )
                    return pr.get("html_url", "")
            # No work past base either: keep the bail, but name the WIP
            # ref AND the worktree HEAD sha (postmortem material - the
            # finding's named minimum; the ERROR-first-line / non-zero-
            # exit death-class seam belongs to the death-class stream and
            # is NOT re-implemented here).
            _empty_head = _git("rev-parse", "HEAD").stdout.strip()
            print(
                f"WARN: local-fixer: empty diff — no PR "
                f"(wip_ref={wip_ref or '<none>'}, "
                f"worktree HEAD={_empty_head or '<unresolvable>'})",
                file=sys.stderr,
            )
            _tail_log(
                task_id,
                f"empty diff - no PR (wip_ref={wip_ref or '<none>'}, "
                f"worktree HEAD={_empty_head or '<unresolvable>'})",
            )
            return ""
        if not gate_passed:
            # D1 (agents-core-local-fixer-harness-fix-v0): the positive-only
            # gate (or its fail-closed legacy fallback) rejected this run.
            # S3 (agents-core-local-fixer-salvage-on-discard-v0): salvage
            # the WORKTREE's final state (not the WIP ref) - a concluded
            # run's final state is exactly what the gate tested, and it is
            # at least as complete as any WIP snapshot. Fail closed: any
            # git failure below falls through to the original return "".
            print(
                "WARN: local-fixer: test gate failed "
                f"(model_touched_tests={sorted(model_touched_tests) if model_touched_tests else '[] (legacy gate)'}; "
                f"deciding outcome {_outcome_diag(last_test_outcome)}"
                f"{_rerun_diag()}) - opening advisory [SALVAGE] PR "
                f"(concluded, gate rejected, worktree salvage)",
                file=sys.stderr,
            )
            _tail_log(task_id, "test gate failed - opening advisory [SALVAGE] PR (concluded, gate rejected)")
            _salvage_commit_sha = ""
            _salvage_git = _wip_git if _wip_git is not None else (
                lambda *a: subprocess.run(
                    ["git", "-C", cwd, *a],
                    capture_output=True, text=True, timeout=30,
                )
            )
            _salvage_add = _salvage_git("add", "-A")
            if _salvage_add.returncode != 0:
                print(
                    f"WARN: local-fixer: worktree salvage git add failed "
                    f"({_salvage_add.stderr.strip()[:500]}) - no PR",
                    file=sys.stderr,
                )
                return ""
            _salvage_commit = _salvage_git(
                "commit", "-q", "-m",
                f"salvage: {task_id} (concluded, gate rejected)",
            )
            if _salvage_commit.returncode != 0:
                print(
                    f"WARN: local-fixer: worktree salvage git commit failed "
                    f"({_salvage_commit.stderr.strip()[:500]}) - no PR",
                    file=sys.stderr,
                )
                return ""
            _salvage_rev = _salvage_git("rev-parse", "HEAD")
            if _salvage_rev.returncode != 0:
                print(
                    f"WARN: local-fixer: worktree salvage git rev-parse failed "
                    f"({_salvage_rev.stderr.strip()[:500]}) - no PR",
                    file=sys.stderr,
                )
                return ""
            _salvage_commit_sha = _salvage_rev.stdout.strip()
            if not _salvage_commit_sha:
                print(
                    "WARN: local-fixer: worktree salvage rev-parse returned "
                    "empty sha - no PR",
                    file=sys.stderr,
                )
                return ""
            return _open_wip_salvage_pr(
                worktree_path, "HEAD", _salvage_commit_sha, [],
                stop_reason="concluded_gate_rejected",
                concluded=True, task_id=task_id,
                target_id=target_id, bare_repo=bare_repo,
                branch=branch, slug=slug,
                step_count=step_count,
                transcript_path=transcript_path,
            )

    # (_git is defined at tail entry - see above.)
    r = _git("checkout", "-B", branch)
    if r.returncode != 0:
        print(f"WARN: local-fixer: git checkout -b failed: {r.stderr.strip()}", file=sys.stderr)
        _tail_log(task_id, f"git checkout -b failed rc={r.returncode}: {r.stderr.strip()[:500]}")
        return ""
    r = _git("add", "-A")
    if r.returncode != 0:
        print(f"WARN: local-fixer: git add failed: {r.stderr.strip()}", file=sys.stderr)
        _tail_log(task_id, f"git add failed rc={r.returncode}: {r.stderr.strip()[:500]}")
        return ""
    r = _git("commit", "-m", f"fix({target_id}): local-fixer harness")
    if r.returncode != 0:
        print(f"WARN: local-fixer: git commit failed: {r.stderr.strip()}", file=sys.stderr)
        _tail_log(task_id, f"git commit failed rc={r.returncode}: {r.stderr.strip()[:500]}")
        return ""
    # fixer-reception-v0 (leg 1, D1): push-failure partition. A push to the
    # verified existing branch can fail (non-fast-forward if the branch
    # advanced between the setup's ls-remote verify and this push; branch
    # protection). Log at ERROR with the full command + rc and record a
    # pm:push-failed observation in the run log so the work loss is never
    # silent: the transcript + local refs/wip survive (the WIP salvage hook
    # is NOT extended to push-failure - named follow-on, not this change).
    r = _git("push", "origin", f"HEAD:{branch}")
    if r.returncode != 0:
        print(
            f"ERROR: local-fixer: git push failed - command: git push origin "
            f"HEAD:{branch} (rc={r.returncode}): {r.stderr.strip()}",
            file=sys.stderr,
        )
        _tail_log(
            task_id,
            f"pm:push-failed rc={r.returncode} branch={branch} "
            f"stderr={r.stderr.strip()[:500]}",
        )
        return ""

    # The pre-aimed / parked-PR case: the branch already has an open PR
    # and the push just advanced its head. Return the existing PR
    # instead of create_pr (a duplicate head branch is refused). The
    # initial-dispatch case (a fresh lapis/<tid>/<slug> branch) has no
    # open PR on it and falls through to create_pr unchanged.
    if branch:
        try:
            for _pr in _forgejo.get_open_prs(repo=bare_repo):
                if (_pr.get("head") or {}).get("ref") == branch:
                    _pr_url = _pr.get("html_url", "")
                    _tail_log(
                        task_id,
                        f"open PR already exists on {branch!r} - head advanced: {_pr_url}",
                    )
                    return _pr_url
        except Exception as exc:
            print(
                f"WARN: local-fixer: open-PR scan failed: {exc} - "
                f"falling through to create_pr",
                file=sys.stderr,
            )

    # Provenance PR body — factual only
    diff_lines = [l for l in final_diff.splitlines()
                  if l.startswith(("diff --git", "---", "+++", "@@", " ")) or l[:1] in ("+", "-")]
    diffstat = "\n".join(diff_lines[:40]) or "(no changes)"

    # D5 (agents-core-local-fixer-gate-perception-v0): the PR body's
    # "## Test outcome" reads the DECIDING outcome (the model's last
    # outcome, or the D1 re-run when it fired - last_test_outcome was
    # replaced in place) and carries the rerun=true annotation. A
    # merged PR whose stated test outcome contradicts its own gate
    # ("0 passed, 0 failed" under a passed gate) is the provenance
    # defect this closes.
    if last_test_outcome:
        passed_c = int(last_test_outcome.get("passed") or 0)
        failed_c = int(last_test_outcome.get("failed") or 0)
        test_summary = f"{passed_c} passed, {failed_c} failed"
        if gate_rerun_fired:
            test_summary += " (rerun=true - harness targeted re-run of model-touched tests)"
    else:
        test_summary = "no test outcome recorded"

    test_gate_section = (
        "## Test gate\n\n"
        "bypassed - no Python test infrastructure in this repo; "
        "NO in-dispatch test run was performed (the harness test tool "
        "is pytest-only). Verification for this PR rests on the "
        "reviewer gate and post-merge local gates. This marker exists "
        "so the skipped verification cannot be read as routine.\n\n"
        "<!-- lapis-test-gate: bypassed-no-python-test-infra -->\n\n"
        if gate_bypassed else ""
    )
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
        "<!-- lapis-no-progress-salvage: true -->\n\n"
        if (salvaged and no_progress_hit) else ""
    )

    if salvaged:
        print(
            f"INFO: local-fixer: harness-salvaged green diff on {salvage_kind} "
            f"abort (target={target_id}, task={task_id})",
            file=sys.stderr,
        )

    # Identity line (local-reviewer-identity-and-provenance-v0, L1.D1): the
    # seat is the spec-driven seat alias (registry-sourced); the served model
    # is the server-echoed final model from the run's served_model_out
    # out-param (the gw_agent.py:1926-1931 contract - the final/deciding
    # model is served_model_out[-1]). The seat alias is NEVER written into
    # the served-model slot: a void echo (unavailable or bound-violating)
    # renders the explicit "not reported" form (Erah 2026-09-06
    # explicit-void adjudication) - substituting the role for the substance
    # would be a lie by omission in the audit record.
    _served = _validate_served_model_echo(served_model)
    _seat = seat_alias or "unknown-seat"
    _served_slot = _served if _served else "not reported"
    identity_line = (
        f"Implemented by the local fixer harness (seat {_seat}), "
        f"served model: {_served_slot}. Not paid Claude."
    )
    pr_body = (
        f"{identity_line}\n\n"
        f"{salvage_note}"
        f"{signoff_marker}"
        f"## Diff summary\n\n```diff\n{diffstat}\n```\n\n"
        f"## Test outcome\n\n{test_summary}\n\n"
        f"{test_gate_section}"
        f"## Steps\n\n{step_count} tool-call step(s) executed.\n\n"
        f"## Transcript\n\n`{transcript_path}`\n\n"
        f"<!-- lapis-gpu-id: {task_id} -->\n"
        f"<!-- lapis-tid: {target_id} -->\n"
        f"<!-- lapis-engine: local-fixer -->"
    )

    try:
        pr = _forgejo.create_pr(
            repo=bare_repo,
            title=f"fix({target_id}): local-fixer",
            head=branch,
            base="main",
            body=pr_body,
        )
    except Exception as exc:
        print(f"WARN: local-fixer: create_pr failed: {exc}", file=sys.stderr)
        _tail_log(task_id, f"create_pr FAILED: {exc}")
        return ""
    _tail_log(task_id, f"create_pr OK url={pr.get('html_url', '')}")
    return pr.get("html_url", "")


def _run_local_fixer_staged(spec: dict, base_cwd: str | None) -> str:
    """Deterministic git/PR tail for the local-fixer-staged engine.

    fixers-harness-staged-v0 (S2): the staged fixer harness - 2 LLM stages
    (READER -> AIMER) + 1 deterministic stage (FIRE), orchestrated by
    agents_core.fixer_stages.run_staged_mission. The worktree is selected
    the same way the legacy path selects it (verified existing_branch ->
    PR head; otherwise base_branch), and the tail is the shared
    tail_finalize helper (S6) called with the SAME value set as the
    legacy call site.

    Returns a PR URL on success, "" on any failure - never raises.
    """
    import json as _json
    import subprocess

    task_id = spec.get("task_id") or spec.get("slot_id") or "staged-unknown"
    target_id = spec.get("target_id", "unknown")
    repo = spec.get("repo", "")
    base_branch = spec.get("base_branch", "main")
    slug = spec.get("slug", "local")
    branch = f"lapis/{target_id}/{slug}"
    bare_repo = repo.rsplit("/", 1)[-1] if repo else "agents-core"
    effective_cwd = base_cwd or "/srv/agents"

    _ARTIFACT_DIR = room_path("gpu_queue.shaped")
    # The consolidated staged transcript (all stage runs, in stage order -
    # the legacy per-run pattern writes <task_id>-gw-transcript.json; the
    # staged runner persists ONE file, named in the resume protocol).
    transcript_path = _ARTIFACT_DIR / f"{task_id}-staged-transcript.json"

    worktree_path = None

    def _log(msg: str) -> None:
        print(msg, file=sys.stderr)

    # ------------------------------------------------------------------
    # Supervisor lease (gw-gpu1-berth-standing-seat-v0, leg 2, Doorman B) -
    # the same pattern as _run_local_fixer: a doorman lease held for the
    # WHOLE staged job (entry acquire, finally release) so the doorman's
    # mid-job stop machinery cannot stop the seat out from under an
    # in-flight staged mission. work_id=f"{task_id}-berth-sup",
    # principal="fixer-supervisor", role="worker", lease_class="deferrable".
    # TTL = timeout_s + 60 (the queue's hard cap). SOFT FAIL: acquire
    # failure is logged and the job proceeds un-supervised.
    # ------------------------------------------------------------------
    sup_lease_client = None
    sup_lease_id = f"{task_id}-berth-sup"
    try:
        from agents_core.doorman_client import DoormanClient

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
        if not isinstance(acq, dict) or acq.get("status") != "serving":
            _status = acq.get("status") if isinstance(acq, dict) else type(acq).__name__
            print(
                f"WARN: staged: supervisor lease acquire soft-failed "
                f"(work_id={sup_lease_id}, status={_status}) - job proceeds "
                f"un-supervised",
                file=sys.stderr,
            )
            sup_lease_client = None
    except Exception as exc:
        print(
            f"WARN: staged: supervisor lease acquire soft-failed "
            f"(work_id={sup_lease_id}): {exc}",
            file=sys.stderr,
        )
        sup_lease_client = None

    # ------------------------------------------------------------------
    # Worktree selection (rev 3.3 - correctness lens 4th pass): select
    # `existing_branch` ONLY when verified on origin (the legacy
    # ls-remote pattern, generalized from the fixer_retry-only gate).
    # Otherwise - including the initial-dispatch case where
    # pm_core.py:3170 defaults existing_branch to the nonexistent
    # lapis/<target_id>/forced (the Phase 2 no-parked-PR case) - the
    # worktree is at base_branch and the tail creates
    # lapis/<target_id>/<slug> and opens the PR (the legacy initial-
    # fixer mechanics). The PR-head precondition (the pre-aimed match
    # diagnostic) applies to pre-aimed missions, which always resolve
    # existing_branch to the parked PR's ref via the open-PR scan.
    # ------------------------------------------------------------------
    existing_branch = spec.get("existing_branch") or ""
    worktree_ref = base_branch
    if existing_branch:
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
                f"WARN: staged: existing_branch {existing_branch} not verified on origin "
                f"- worktree at base_branch {base_branch} (the legacy initial-fixer "
                f"mechanics: the tail creates {branch} and opens the PR)",
                file=sys.stderr,
            )
            _tail_log(
                task_id,
                f"worktree: existing_branch {existing_branch} not verified on origin "
                f"- worktree at base_branch {base_branch}",
            )
        else:
            worktree_ref = existing_branch
            print(
                f"INFO: staged: worktree at verified existing_branch {existing_branch} "
                f"(PR head - the pre-aimed match diagnostic precondition holds)",
                file=sys.stderr,
            )
            _tail_log(
                task_id,
                f"worktree: existing_branch {existing_branch} verified on origin (PR head)",
            )

    # S6 seam (the push): the tail pushes HEAD to spec["branch"] and
    # create_pr's head is the same name. In the pre-aimed / parked-PR
    # case that is the verified existing branch - the tail advances the
    # PR head and the tail's open-PR scan returns the existing PR (no
    # second PR). In the initial-dispatch case it is the legacy
    # lapis/<target_id>/<slug> new branch (create + PR). An empty
    # spec["branch"] would make the tail's `git checkout -B ""` fatal.
    # spec["bare_repo"] is injected the same way: the S2-side tail call
    # reads spec.get("bare_repo", "") and a present-empty value defeats
    # the call site's setdefault (the create_pr URL became
    # repos/Erah//pulls on the D2a acceptance run - 405).
    spec["branch"] = worktree_ref if worktree_ref != base_branch else branch
    spec["bare_repo"] = bare_repo

    # The steer directive (the mission fence is the only task input that
    # carries defect detail). The shaper renders it into the spec's
    # prompt via {steer_directive_block}; a spec that carries the
    # directive under a separate key wins.
    directive = spec.get("steer_directive") or spec.get("prompt") or ""

    # Mission parse (fail-loud before any GPU spend - same class as the
    # card-validation errors). run_staged_mission re-parses (cheap,
    # deterministic) and writes the mission report on the failure branch;
    # this pre-parse is the dispatch-side loud abort.
    from agents_core import fixer_stages
    try:
        mission = fixer_stages.parse_mission(directive)
    except fixer_stages.MissionError as exc:
        print(f"ERROR: staged: mission parse failed - aborting before GPU spend: {exc}",
              file=sys.stderr)
        _tail_log(task_id, f"mission parse failed - aborting before GPU spend: {exc}")
        return ""
    _tail_log(
        task_id,
        f"mission parsed: pre_aimed={mission.pre_aimed} "
        f"scope_files={mission.scope_files} tests={mission.tests} "
        f"tests_timeout_s={mission.tests_timeout_s}",
    )

    # The explore cap env (the process-scoped env is safe: the runner is a
    # per-dispatch subprocess). run_stage also sets it before each stage
    # call; setting it here covers the pre-stage phase.
    os.environ["GW_AGENT_MAX_EXPLORE_STEPS"] = str(fixer_stages.STAGE_MAX_EXPLORE_STEPS)

    try:
        from agents_core.worktree import setup_worktree, teardown_worktree
        handle = setup_worktree(task_id, effective_cwd, worktree_ref)
        worktree_path = handle.path
        cwd = str(worktree_path)
    except Exception as exc:
        print(f"ERROR: staged: worktree setup failed: {exc}", file=sys.stderr)
        _tail_log(task_id, f"worktree setup failed: {exc}")
        return ""

    try:
        # Stage the bound spec into the worktree so the mission report's
        # provenance can page it (best-effort, mirror the legacy staging).
        spec_src = room_path("planning.specs") / f"{target_id}.md"
        spec_dst = Path(cwd) / "lapis-spec.md"
        try:
            if spec_src.exists() and spec_src.stat().st_size > 0 and not spec_dst.exists():
                spec_dst.write_text(spec_src.read_text())
        except OSError as exc:
            print(f"WARN: staged: spec staging failed: {exc}", file=sys.stderr)

        # The gate re-run (the staged tail calls _gate_targeted_rerun
        # directly with the mission's tests_timeout_s - the legacy 180s
        # cap is BELOW the measured 189s runtime of the D2a acceptance
        # test file).
        def _staged_gate_rerun(cwd_: str, touched, timeout_s: int) -> dict | None:
            return _gate_targeted_rerun(cwd_, touched, timeout_s=timeout_s)

        # The shared deterministic tail (S6) - the SAME value set as the
        # legacy call site (task_id / target_id / bare_repo /
        # gate_rerun_fired named explicitly; the full seam).
        def _staged_tail_finalize(**kwargs) -> str:
            # Fill the full-seam defaults the S1 call site does not carry
            # (the staged path has no WIP ref / budget flags / no-progress
            # aborts - the stages are read-only and the fire is
            # deterministic).
            kwargs.setdefault("task_id", task_id)
            kwargs.setdefault("target_id", target_id)
            kwargs.setdefault("bare_repo", bare_repo)
            kwargs.setdefault("branch", branch)
            kwargs.setdefault("slug", slug)
            kwargs.setdefault("worktree_path", worktree_path)
            kwargs.setdefault("max_steps_hit", False)
            kwargs.setdefault("no_progress_hit", False)
            kwargs.setdefault("stop_reason", "")
            kwargs.setdefault("transcript_path", transcript_path)
            kwargs.setdefault("gate_bypassed", None)
            kwargs.setdefault("model_touched_tests", set())
            kwargs.setdefault("gate_rerun_fired", True)
            # agents-core-fixer-worktree-vanish-salvage-v0 (D2): the staged
            # path has no WIP ref (no per-step WIP hook in fixer_stages -
            # wip_commit_count=0 / wip_head_sha="" by default), so the new
            # salvage partition is a no-op there by construction; plumb the
            # parent-clone root for parity with the legacy call site.
            kwargs.setdefault("repo_cwd", effective_cwd)
            return tail_finalize(**kwargs)

        # Remove the staged spec before the deterministic tail so it is
        # never committed/pushed into the PR branch (the legacy tail's
        # guard).
        def _pre_tail() -> None:
            try:
                spec_dst.unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARN: staged: staged spec remove failed: {exc}", file=sys.stderr)

        # ------------------------------------------------------------------
        # Stage orchestration (fixer_stages.run_staged_mission): the
        # MISSION_DEADLINE_S=2400 monotonic pre-check runs between stages
        # inside the orchestration; the mission report is written on every
        # terminal branch, before the return.
        #
        # Pre-tail unlink (ordering fix, D2a acceptance 2026-09-01): the
        # S2 tail runs INSIDE run_staged_mission (the S6 seam), so the
        # finally-block _pre_tail ran AFTER the tail's `git add -A` and
        # the staged lapis-spec.md got committed + pushed into the PR
        # branch (run 5's push carried it). Nothing in the staged
        # mission reads the staged spec file (the mission fence is the
        # only task input; the stages are scope-confined), so unlinking
        # before the mission is safe. The finally-block unlink stays as
        # belt-and-braces for the pre-mission failure branches.
        _pre_tail()
        try:
            outcome = fixer_stages.run_staged_mission(
                spec=spec,
                cwd=cwd,
                directive=directive,
                log=_log,
                gate_rerun=_staged_gate_rerun,
                tail_finalize=_staged_tail_finalize,
            )
        except Exception as exc:
            print(f"ERROR: staged: run_staged_mission raised: {exc}", file=sys.stderr)
            _tail_log(task_id, f"run_staged_mission raised: {exc}")
            return ""
        finally:
            _pre_tail()

        _tail_log(
            task_id,
            f"staged mission terminal: final_state={outcome.final_state} "
            f"stop_reason={outcome.stop_reason} all_applied={outcome.all_applied} "
            f"gate_passed={outcome.gate_passed} pr_url={outcome.pr_url!r} "
            f"report={outcome.report_path}",
        )

        # Persist the consolidated staged transcript (the resume
        # protocol's evidence list names it next to the report and
        # tail.log).
        try:
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            if not transcript_path.exists():
                transcript_path.write_text(
                    _json.dumps([], ensure_ascii=False, default=str)
                )
        except OSError as exc:
            print(f"WARN: staged: transcript write failed: {exc}", file=sys.stderr)

        # Terminal partition (one explicit tail line per branch):
        #  - completed + gate passed: the tail already pushed the PR (the
        #    tail_finalize call inside run_staged_mission) - return its URL.
        #  - gate rejected / unusable (all entries applied): worktree
        #    salvage (the concluded, gate-rejected partition - the same
        #    salvage the legacy tail opens for a concluded run the gate
        #    rejected).
        #  - aim failed (partial apply): the gate was skipped; salvage the
        #    worktree directly (stop_reason=staged_aim_failed).
        #  - any other terminal state (stage failure, deadline stop,
        #    pre-aimed rejection, parse failure): the report is the
        #    provenance; return "" (no fire, no gate, no PR).
        if outcome.final_state == "completed":
            if outcome.pr_url:
                _tail_log(task_id, f"staged mission completed - PR {outcome.pr_url}")
                return outcome.pr_url
            # The tail was called but returned "" (a push/PR failure): the
            # report + tail.log carry the provenance.
            _tail_log(task_id, "staged mission completed but the tail returned no PR URL")
            return ""
        if outcome.all_applied and not outcome.gate_passed:
            # Gate rejected or unusable on a fully applied aim set: worktree
            # salvage (the concluded, gate-rejected partition).
            _tail_log(
                task_id,
                f"staged mission gate {outcome.stop_reason} - opening advisory "
                f"[SALVAGE] PR (concluded, gate rejected, worktree salvage)",
            )
            return _staged_tail_finalize(
                cwd=cwd,
                final_diff=fixer_stages._git_diff(cwd),
                concluded=True,
                last_test_outcome=outcome.gate_outcome,
                step_count=outcome.stage_steps,
                gate_passed=False,
                gate_rerun_fired=True,
                stop_reason=outcome.stop_reason,
            )
        _tail_log(
            task_id,
            f"staged mission terminal without a PR: final_state={outcome.final_state} "
            f"stop_reason={outcome.stop_reason} (report at {outcome.report_path})",
        )
        return ""
    finally:
        # Release the supervisor lease (gw-gpu1-berth-standing-seat-v0,
        # leg 2, Doorman B) and tear down the worktree - mirrors the
        # _run_local_fixer finally-block: released on BOTH the success and
        # failure paths (including the early-return error paths after
        # worktree setup). Soft fail: a release failure is logged and
        # swallowed - the TTL bounds the zombie window if it is lost.
        if sup_lease_client is not None:
            try:
                sup_lease_client.release("gravitywell", sup_lease_id)
            except Exception as exc:
                print(
                    f"WARN: staged: supervisor lease release failed "
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
                print(f"WARN: staged: worktree teardown failed: {exc}", file=sys.stderr)


def _print_provenance_line(prov) -> None:
    """Emit the machine-readable PROVENANCE line (L1.D3).

    ``PROVENANCE: seat=<alias> served=<model>`` to stdout; ``served`` is
    ABSENT from the line when the echo is void (unavailable or
    bound-violating). Absence is a non-error: a missing/invalid prov dict
    degrades to no line, never a crash. The claude engine prints the full
    model result to stdout, so consumers parse line-anchored on the
    ``^PROVENANCE: `` prefix (last matching line wins).
    """
    if not isinstance(prov, dict):
        return
    seat = prov.get("seat") or ""
    if not seat:
        return
    served = _validate_served_model_echo(prov.get("served"))
    line = f"PROVENANCE: seat={seat}"
    if served:
        line += f" served={served}"
    print(line)


def _run_local_fixer(spec: dict, base_cwd: str | None) -> tuple[str, dict | None]:
    """Deterministic git/PR tail for the local-fixer engine.

    Manages its own worktree (shaper doesn't set worktree_required for GPU-routed
    agents). Returns (PR URL, provenance) on success, ("", provenance) on any
    failure — never raises. The provenance dict (L1.D3) is
    ``{"seat": <spec seat alias>, "served": <validated served-model echo or
    "" when void>}`` — the seat alias is never substituted into the served
    slot.
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
            # Resolve the parent ONCE (attestation-contract-v0, leg 1, D1
            # rev-2 pin): the sha given to read-tree below and the sha
            # consumed by commit-tree -p MUST be the same resolved value -
            # a ref that moved between the two calls would desync the tree
            # from its declared parent. A missing WIP ref (the first step)
            # resolves to HEAD (the base) - the same fallback the pre-D1
            # hook used.
            parent = _wip_git("rev-parse", "--verify", wip_ref)
            if parent.returncode != 0:
                parent = _wip_git("rev-parse", "HEAD")
                if parent.returncode != 0:
                    return
            parent_sha = parent.stdout.strip()
            # The read-tree source: the ref itself (its current value at
            # this moment - the same commit as parent_sha; the ref cannot
            # move between the resolution above and this call, so the
            # tree loaded and the declared parent are the same commit).
            # Using the ref (not the sha) keeps the call shape identical
            # for the first step (wip_ref missing -> read-tree HEAD) and
            # later steps (read-tree <prior WIP tip>).
            _read_tree_ref = wip_ref
            # CUMULATIVE WIP floor (attestation-contract-v0, leg 1, D1):
            # load the prior WIP tip's tree into the index BEFORE staging
            # this step's paths, so the new commit's tree is CUMULATIVE
            # (previous tip's tree + this step's files) instead of
            # base-HEAD + this step's files (the non-cumulative seam that
            # shipped two 09-08 test-file-only salvage PRs). read-tree
            # moves the index only - the worktree HEAD is never touched.
            # Deletion tolerance is named and accepted (add does not
            # prune): the WIP floor is a salvage floor, not a worktree
            # mirror - the contract is "every step's files at final
            # content", not a mirror of deletions.
            rt = _wip_git("read-tree", _read_tree_ref)
            if rt.returncode != 0:
                # The first step (wip_ref missing) reads HEAD instead -
                # the same tree the parent resolution fell back to.
                rt = _wip_git("read-tree", "HEAD")
                if rt.returncode != 0:
                    print(
                        f"WARN: wip-commit: git read-tree failed "
                        f"({rt.stderr.strip()}) - falling back to the "
                        f"pre-D1 non-cumulative index",
                        file=sys.stderr,
                    )
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
                "commit-tree", tree.stdout.strip(), "-p", parent_sha,
                "-m", f"wip: {task_id} step {step_num} [auto]",
            )
            if commit.returncode != 0:
                print(f"WARN: wip-commit: git commit-tree failed: {commit.stderr.strip()}", file=sys.stderr)
                _wip_git("reset")
                return
            upd = _wip_git("update-ref", wip_ref, commit.stdout.strip())
            if upd.returncode != 0:
                print(f"WARN: wip-commit: git update-ref failed: {upd.stderr.strip()}", file=sys.stderr)
                # Unstage before the return: a failed update-ref must not
                # leave the index loaded with the cumulative tree (the
                # worktree's HEAD is untouched either way - read-tree and
                # commit-tree are plumbing).
                _wip_git("reset")
                return
            # Unstage: restore the index (and the worktree's HEAD) exactly as found.
            _wip_git("reset")
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

    # L1.D3 provenance dict (local-reviewer-identity-and-provenance-v0): the
    # seat is the spec-driven seat alias (registry-sourced); the served slot
    # carries the validated echo or "" when void - the alias is NEVER
    # substituted into the served slot (Erah 2026-09-06 explicit-void
    # adjudication).
    _provenance = {"seat": spec.get("model") or "", "served": ""}

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

        # A spec-carried existing_branch is honored for ALL agent types —
        # the worktree must start from the PR's own branch, not base_branch
        # (main), or the target file simply won't exist in the checkout.
        # Verify the branch is really on origin first: the local-fixer GW
        # sandbox has no git checkout tool, so if this is wrong there is no
        # way for the model to self-correct. (The agent_type gate was
        # fixer_retry-only; it is removed to mirror the staged-path
        # invariant, which verifies existing_branch without an
        # agent_type gate.)
        existing_branch = spec.get("existing_branch") or ""
        worktree_ref = base_branch
        if existing_branch:
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
                return "", _provenance
            worktree_ref = existing_branch

        # fixer-reception-v0 (leg 1, D1): the slug re-home fix. The legacy
        # local path always pushed to the `lapis/<tid>/local` slug default
        # even when the worktree setup resolved existing_branch (fixer_retry
        # with a spec-carried branch) - so a retry pushed to the default
        # slug and the daemon watched the original head forever (orphan PR
        # if the slug was free; a dead non-force push if it was parked).
        # Mirror the staged invariant (:1489-1499): when setup used
        # existing_branch, the tail's branch (checkout -B / push / open-PR
        # scan) IS that verified branch, so the post-push scan finds the
        # SAME PR advanced instead of opening a second one.
        if worktree_ref != base_branch:
            branch = worktree_ref
            _tail_log(
                task_id,
                f"worktree: existing_branch {existing_branch} verified on origin "
                f"- tail pushes to {branch} (the verified existing branch, "
                f"not the {slug!r} slug default)",
            )

        handle = setup_worktree(task_id, effective_cwd, worktree_ref)
        worktree_path = handle.path
        cwd = str(worktree_path)

        # base_sha (attestation-contract-v0, leg 1, D2 part 1): the
        # worktree's HEAD at setup time (the F2 pattern the local-opencode
        # engine already had). The deterministic tail's empty-diff
        # self-commit recovery re-derives the deliverable as HEAD vs base
        # when the model self-committed (clean index, HEAD past base) -
        # the local-fixer engine had no base sha today, so the recovery
        # was unreachable without it. A failed capture degrades the tail
        # to today's bail behavior (never blocks the run).
        base_sha = ""
        try:
            _base_probe = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd, capture_output=True, text=True, timeout=30,
            )
            if _base_probe.returncode == 0:
                base_sha = _base_probe.stdout.strip()
            else:
                print(
                    f"WARN: local-fixer: base_sha capture failed "
                    f"rc={_base_probe.returncode}: "
                    f"{_base_probe.stderr.strip()[:200]}",
                    file=sys.stderr,
                )
        except subprocess.TimeoutExpired:
            print(
                "WARN: local-fixer: base_sha capture timed out "
                "(rev-parse HEAD) - the empty-diff self-commit recovery "
                "degrades to the bail",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"WARN: local-fixer: base_sha capture failed ({exc}) - "
                f"the empty-diff self-commit recovery degrades to the bail",
                file=sys.stderr,
            )

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

        # fixer-reception-v0 (leg 1, D2b): honor a declared investigation
        # budget in OPERATOR-AUTHORED INTENT ONLY. The dispatch record's
        # operator intent field (spec["intent"]) is the force-dispatch /
        # dispatch-record intent - NOT spec["prompt"], which embeds
        # LLM-authored reviewer issue notes and, post leg-2, auditor brief
        # text. The guardrail is not a prompt-injectable channel: the exact
        # line form [investigation-budget: <N> steps] (N integer, 1..500)
        # is honored ONLY in the operator intent. When present:
        # no_progress_steps=N (the parameter exists end-to-end through
        # call_gw_agent -> _call_gw_agent_impl) and the effective explore
        # ceiling = max(env ceiling, 2*N) via the explicit max_explore_steps
        # parameter (the env var is never mutated). Absent the line: env
        # values rule, behavior unchanged.
        _investigation_budget: int | None = None
        _intent = spec.get("intent")
        if isinstance(_intent, str):
            _m = re.search(r"^\[investigation-budget:\s*(\d+)\s*steps\]\s*$", _intent, re.MULTILINE)
            if _m:
                _n = int(_m.group(1))
                if 1 <= _n <= 500:
                    _investigation_budget = _n
                else:
                    print(
                        f"WARN: local-fixer: investigation budget {_n} out of range "
                        f"1..500 in operator intent - ignoring (env values rule)",
                        file=sys.stderr,
                    )
            else:
                _m_loose = re.search(r"\[investigation-budget:", _intent)
                if _m_loose:
                    print(
                        "WARN: local-fixer: malformed [investigation-budget: ...] line in "
                        "operator intent - ignoring (env values rule)",
                        file=sys.stderr,
                    )
        if _investigation_budget is not None:
            _tail_log(
                task_id,
                f"investigation budget declared in operator intent: "
                f"no_progress_steps={_investigation_budget}, "
                f"explore ceiling=max(env, {2 * _investigation_budget})",
            )
        # Served-model provenance (local-reviewer-identity-and-provenance-v0,
        # L1.D1/L1.D3): caller-owned out-param list. call_gw_agent appends the
        # server-echoed "model" field per echoing step (gw_agent.py:1838; the
        # contract at :1926-1931 documents the final/deciding model as
        # served_model_out[-1] - the same pattern authority.screen consumes).
        # No gw_agent.py change is required.
        _served_model_out: list = []

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
            # fixer-reception-v0 (leg 1, D2b): the declared investigation
            # budget (operator intent only - see the parse above). None
            # keeps the env values in force (behavior unchanged).
            no_progress_steps=(
                _investigation_budget if _investigation_budget is not None else None
            ),
            max_explore_steps=(
                2 * _investigation_budget if _investigation_budget is not None else None
            ),
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
            served_model_out=_served_model_out,
            # D2 (attestation-contract-v0, leg 1): the worktree's HEAD at
            # SETUP time (captured above, post-setup_worktree - the F2
            # pattern the local-opencode engine already had). The tail's
            # empty-diff self-commit recovery re-derives the deliverable
            # as HEAD vs this base.
            base_sha=base_sha,
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
        # D2 (attestation-contract-v0, leg 1): the worktree's HEAD at
        # SETUP time (captured above, post-setup_worktree - the F2 pattern
        # the local-opencode engine already had). The tail's empty-diff
        # self-commit recovery re-derives the deliverable as HEAD vs base.
        # An absent capture degrades the tail to today's bail behavior.
        concluded = fixer_result.get("concluded", False)
        last_test_outcome = fixer_result.get("last_test_outcome")
        max_steps_hit = fixer_result.get("max_steps_reached", False)
        no_progress_hit = fixer_result.get("no_progress", False)
        stop_reason = fixer_result.get("stop_reason", "")
        model_touched_tests = _collect_model_touched_tests(transcript, cwd)
        gate_passed = False
        gate_bypassed = None
        py_infra = _has_python_test_infra(Path(cwd))
        if not py_infra:
            # agents-core-local-fixer-gate-nonpython-v0 Deliverable 2: the
            # worktree has no Python test infrastructure - the pytest-only
            # in-dispatch gate can never pass here (claude-view class:
            # Rust+TS repos). Bypass loudly instead of failing closed
            # forever and silently discarding completed work.
            gate_passed = True
            gate_bypassed = "no-python-test-infra"
            print(
                "WARN: local-fixer: test gate BYPASSED - no Python test "
                "infrastructure detected in repo; in-dispatch test "
                "verification unavailable; verification rests on the "
                "reviewer gate and post-merge local gates",
                file=sys.stderr,
            )
            _tail_log(task_id, "gate BYPASSED - no-python-test-infra")
            _write_friction_entry(
                repo=bare_repo,
                node_id="test-gate-bypassed-no-python-test-infra",
                error_signature="gate-bypassed:no-python-test-infra",
                task_id=task_id,
                today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                log=lambda m: print(m, file=sys.stderr),
            )

        # L1.D3: stamp the validated served-model echo into the provenance
        # dict. The final/deciding model is served_model_out[-1] (the
        # gw_agent.py:1926-1931 contract). A bound-violating or absent echo
        # leaves the served slot "" (void) - the alias is never substituted.
        if _served_model_out:
            _provenance["served"] = _validate_served_model_echo(
                _served_model_out[-1]
            )

        # S6 (fixers-harness-staged-v0): the shared deterministic tail
        # (gate decision, salvage partitions, push/PR) - extracted so the
        # staged engine reuses the exact same mechanics. The legacy path
        # calls it with its existing values (behavior-pinned by the
        # existing tail tests, test_wip_salvage).
        _legacy_tail_result = tail_finalize(
            task_id=task_id,
            target_id=target_id,
            bare_repo=bare_repo,
            branch=branch,
            slug=slug,
            cwd=cwd,
            worktree_path=worktree_path,
            final_diff=final_diff,
            concluded=concluded,
            last_test_outcome=last_test_outcome,
            max_steps_hit=max_steps_hit,
            no_progress_hit=no_progress_hit,
            stop_reason=stop_reason,
            step_count=len(fixer_result.get("steps") or []),
            transcript_path=transcript_path,
            gate_passed=gate_passed,
            gate_bypassed=gate_bypassed,
            model_touched_tests=model_touched_tests,
            gate_rerun_fired=False,
            wip_ref=wip_ref,
            wip_commit_count=wip_commit_count,
            wip_head_sha=wip_head_sha,
            wip_steps=wip_steps,
            _wip_git=_wip_git,
            seat_alias=spec.get("model") or "",
            served_model=(
                _served_model_out[-1] if _served_model_out else ""
            ),
            # agents-core-fixer-worktree-vanish-salvage-v0 (D1/D2): the
            # worktree-vanished flag from the fixer result (git add -A
            # failed on a deleted cwd) and the parent-clone path as the
            # salvage push fallback root (the WIP ref resolves from the
            # shared common gitdir after the worktree is gone).
            worktree_vanished=fixer_result.get("worktree_vanished", False),
            repo_cwd=effective_cwd,
            base_sha=base_sha,
        )
        return _legacy_tail_result, _provenance
    except Exception as exc:
        print(f"WARN: local-fixer: unexpected error: {exc}", file=sys.stderr)
        _tail_log(task_id, f"unexpected error: {exc}")
        return "", _provenance

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


# Tail budget (seconds) for the local-opencode engine's deterministic tail
# (S4d, rev-3): 180s test cap + 4x30s git ops + PR create + margin. The queue's
# hard kill lands at timeout_s + 60 (shaper.py); the model loop is budgeted at
# timeout_s - TAIL_BUDGET - setup_lag (setup_lag measured at runtime), so a
# deadline-killed run still has the full TAIL_BUDGET for its tail. Best-effort
# bound, not a guarantee (Invariant 8).
TAIL_BUDGET = 300

# HARNESS-OWNS-GIT contract preamble (S4a / Invariant 3). Composed by the
# engine and prepended to the bound intent. spec["system"] is deliberately NOT
# passed to opencode: it carries the registry's model-side git protocol
# (checkout -b / add / commit / push / self-PR), which would collide with the
# deterministic tail. This is a DELIBERATE divergence from _run_local_fixer,
# which passes its system prompt because that model has no git tool.
_HARNESS_OWNS_GIT_PREAMBLE = """\
HARNESS-OWNS-GIT CONTRACT - read this before doing anything:
- You are working inside an isolated git worktree. Make all code and test
  edits IN THIS WORKTREE ONLY.
- Do NOT run `git commit`, `git push`, `git checkout`, `git branch`, or
  `git add` (any form, including `git add -A`). The harness owns the git
  tail: it derives your diff from the uncommitted worktree state, commits,
  pushes and opens the PR itself. If you move any git state (HEAD, index,
  branches, remotes), your work will be DISCARDED.
- Do NOT background or daemonize any process (no `&`, no `nohup`, no
  `setsid`, no long-running services). Every command must finish before
  you move on.
- Do NOT `pip install` outside the isolated environment: pip user-installs
  are pinned to this worktree (PIP_USER / PYTHONUSERBASE are set); do not
  override them or write to host Python. The harness owns the test run.
"""


def _run_local_opencode(spec: dict, base_cwd: str | None) -> str:
    """Deterministic git/PR tail for the local-opencode engine.

    Runs the fixer under the opencode tool loop (`opencode run`) in its own
    per-task worktree. The harness owns git: the model is contractually told
    not to commit/push/branch, and the tail re-derives the diff + test
    outcome from uncommitted worktree state (tail re-derivation: chunk 3).
    Returns a PR URL on success, "" on any failure - never raises.
    """
    import signal
    import time

    task_id = spec.get("task_id") or spec.get("slot_id") or "lo-unknown"
    target_id = spec.get("target_id", "unknown")
    repo = spec.get("repo", "")
    base_branch = spec.get("base_branch", "main")
    slug = spec.get("slug", "local")
    bare_repo = repo.rsplit("/", 1)[-1] if repo else "agents-core"
    effective_cwd = base_cwd or "/srv/agents"
    opencode_bin = os.environ.get(
        "OPENCODE_BIN", "/home/user/.opencode/bin/opencode"
    )
    opencode_model = spec.get("opencode_model", "gravitywell/gravitywell-slot1")

    _ARTIFACT_DIR = room_path("gpu_queue.shaped")
    events_path = _ARTIFACT_DIR / f"{task_id}-opencode-events.json"
    stderr_path = _ARTIFACT_DIR / f"{task_id}-opencode-stderr.log"

    worktree_path = None
    opencode_pgid = None  # S4f: process-group leader pid, killpg'd in the finally

    # setup_lag timer starts here: the MEASURED worktree-setup + supervisor-
    # lease duration (S4d/rev-3) is captured after setup, before the model
    # loop, and subtracted from the loop budget.
    setup_started = time.monotonic()

    # ------------------------------------------------------------------
    # Supervisor lease (duplicated from the _run_local_fixer :480-521
    # pattern, host-keyed "gravitywell", opencode adaptations).
    #
    # A doorman lease held for the WHOLE local-opencode job (entry acquire,
    # finally release) so the doorman's mid-job stop machinery (force-stop's
    # _worker_lease_blockers guard) cannot stop the seat out from under an
    # in-flight opencode job. work_id is f"{task_id}-berth-sup" so it cannot
    # be clobbered by a per-run lease on task_id; the distinct principal/role
    # keeps this engine's supervisor lease from colliding with the
    # local-fixer's on one doorman.
    #
    # TTL = timeout_s + 60 (the queue's hard cap) - acquired at entry, it
    # structurally outlives the process killed at submit + timeout_s + 60,
    # so NO renewal is needed.
    #
    # SOFT FAIL: acquire failure is logged and the job PROCEEDS
    # un-supervised. The lease is a safety net, never a gate.
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
            "opencode-job-supervisor",
            role="worker",
            principal="opencode-supervisor",
            lease_class="deferrable",
        )
        # acquire() RETURNS a status dict (it does not raise on
        # pending_defer/deferred): only "serving" means a lease was
        # registered. Any other status is a soft fail - log it and drop
        # the client so the finally-release is skipped.
        if not isinstance(acq, dict) or acq.get("status") != "serving":
            _status = acq.get("status") if isinstance(acq, dict) else type(acq).__name__
            print(
                f"WARN: local-opencode: supervisor lease acquire soft-failed "
                f"(work_id={sup_lease_id}, status={_status}) - job proceeds "
                f"un-supervised",
                file=sys.stderr,
            )
            sup_lease_client = None
    except Exception as exc:
        # Soft fail: the job proceeds un-supervised.
        print(
            f"WARN: local-opencode: supervisor lease acquire soft-failed "
            f"(work_id={sup_lease_id}): {exc}",
            file=sys.stderr,
        )
        sup_lease_client = None

    try:
        from agents_core.worktree import setup_worktree, teardown_worktree

        # fixer_retry dispatches target an already-open PR - the worktree
        # must start from the PR's own branch, not base_branch (main), or
        # the target files simply won't exist in the checkout. Verify the
        # branch is really on origin first (duplicated from the local-fixer
        # worktree-setup pattern; with full bash the model could
        # self-correct, but failing loud is the same shape).
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
                    f"ERROR: local-opencode: existing_branch {existing_branch} not found on origin",
                    file=sys.stderr,
                )
                return ""
            worktree_ref = existing_branch

        # Own per-task worktree (like local-fixer; shaper sets
        # worktree_required=False for local-opencode, S2). handle.env
        # carries the pip-isolation env (PYTHONUSERBASE + PIP_USER) that
        # must be forwarded to the opencode subprocess (F10).
        handle = setup_worktree(task_id, effective_cwd, worktree_ref)
        worktree_path = handle.path
        cwd = str(worktree_path)

        # base_sha (F2 integrity check): the worktree starts detached at a
        # known commit. The tail (chunk 3) fails closed - no PR - if HEAD
        # moved off this sha: the model moved git state and the
        # deterministic tail's invariants are broken.
        base_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        ).stdout.strip()

        # S4d: budget the model loop. setup_lag is the MEASURED worktree-
        # setup + supervisor-lease duration (rev-3: unbounded under seat
        # contention, so it is subtracted, not assumed). Total wall =
        # setup_lag + loop_budget + tail = timeout_s + 60 - TAIL_BUDGET +
        # tail, so the tail finishes iff tail_cost < TAIL_BUDGET.
        setup_lag = time.monotonic() - setup_started
        loop_budget = int(spec.get("timeout_s", 1800)) - TAIL_BUDGET - int(setup_lag)
        if loop_budget <= 0:
            print(
                f"ERROR: local-opencode: setup_lag {setup_lag:.1f}s leaves no "
                f"model-loop budget (timeout_s={spec.get('timeout_s')}, "
                f"TAIL_BUDGET={TAIL_BUDGET}s); failing closed",
                file=sys.stderr,
            )
            return ""

        # S4a: the harness-owns-git contract, composed by the engine from
        # the preamble + the bound intent. spec["system"] is NOT passed
        # (see _HARNESS_OWNS_GIT_PREAMBLE).
        contract = _HARNESS_OWNS_GIT_PREAMBLE + "\n" + spec["prompt"]

        argv = [
            opencode_bin,
            "run",
            "--auto",
            "--format", "json",
            "-m", opencode_model,
            "--dir", cwd,
            "--title", f"fixer-{task_id}",
            contract,
        ]

        # Launch opencode as its own session leader (start_new_session=True)
        # so its pid IS the process-group id: a timeout/finally killpg
        # reaches the whole tree (F6/F9). Popen + wait (not subprocess.run):
        # run's timeout-kill SIGKILLs only the direct child, orphaning
        # opencode's bash children, and its pid is not exposed pre-wait -
        # the pgid must be captured pre-completion for the finally-killpg.
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        deadline_killed = False
        with open(events_path, "wb") as _events, open(stderr_path, "wb") as _err:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=_events,
                stderr=_err,
                env={**os.environ, **handle.env, "OPENCODE_BIN": opencode_bin},
                start_new_session=True,
            )
            opencode_pgid = proc.pid
            try:
                proc.wait(timeout=loop_budget)
            except subprocess.TimeoutExpired:
                # S4d: the loop budget is exhausted - a deadline kill, NOT a
                # model admission of stall. The tail (chunk 3) re-derives
                # whatever is on disk and can salvage. Kill the whole group
                # now; the finally-killpg is the idempotent backstop for
                # every other exit path.
                deadline_killed = True
                print(
                    f"WARN: local-opencode: model loop hit the loop budget "
                    f"({loop_budget}s); killing the opencode process group "
                    f"(pgid={opencode_pgid})",
                    file=sys.stderr,
                )
                try:
                    os.killpg(opencode_pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            rc = proc.wait()
        print(
            f"INFO: local-opencode: model loop done (task={task_id}, "
            f"exit={rc}, deadline_killed={deadline_killed}, "
            f"setup_lag={setup_lag:.1f}s, loop_budget={loop_budget}s) - "
            f"events: {events_path}, stderr: {stderr_path}",
            file=sys.stderr,
        )

        # ------------------------------------------------------------------
        # Deterministic tail re-derivation (S4b).
        #
        # opencode is dead by now (clean exit, deadline kill, or the
        # finally-killpg below), so everything from here is re-derived from
        # the uncommitted worktree state: HEAD integrity (F2), the staged
        # diff (F1), the targeted test outcome (F4), the D1 gate, and the
        # terminal state. The tail is authoritative (Invariant 3): the model
        # self-committing yields an empty diff and no PR.
        # ------------------------------------------------------------------
        branch = f"lapis/{target_id}/{slug}"

        # Deterministic git helper (the _run_local_fixer :811-819 nested
        # closure, duplicated - it is not importable).
        def _git(*args: str) -> subprocess.CompletedProcess:
            try:
                return subprocess.run(
                    ["git", "-C", cwd, *args],
                    capture_output=True, text=True, timeout=30,
                )
            except subprocess.TimeoutExpired:
                print(f"WARN: local-opencode: git {args[0]} timed out", file=sys.stderr)
                return subprocess.CompletedProcess(["git", "-C", cwd, *args], 1, "", "timeout")

        # F2 integrity: HEAD must still be at the base commit the worktree
        # started from. A move means the model touched git state against the
        # HARNESS-OWNS-GIT contract and the deterministic tail's invariants
        # are broken - fail closed, no PR (distinct WARN).
        head_now = _git("rev-parse", "HEAD")
        if head_now.returncode != 0 or head_now.stdout.strip() != base_sha:
            print(
                f"WARN: local-opencode: HEAD moved off base_sha "
                f"(base={base_sha[:12]}, now={head_now.stdout.strip()[:12] or head_now.stderr.strip()}) "
                f"- model moved git state; fail-closed, no PR",
                file=sys.stderr,
            )
            return ""

        # F1: stage everything (including new untracked files), unstage the
        # staged-spec guard, diff the index against HEAD. A bare `git diff`
        # or `git diff HEAD` is blind to new untracked files, so a run whose
        # deliverables are all new files would register an empty diff and be
        # discarded (gw_agent.py:1186-1231 pattern - copied, not invented).
        r = _git("add", "-A")
        if r.returncode != 0:
            print(f"WARN: local-opencode: git add -A failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        # Staged-spec guard (gw_agent.py:1208-1224): if a lapis-spec.md copy
        # is UNTRACKED at HEAD, unstage it so a spec hunk never pollutes
        # final_diff. A repo that tracks its own lapis-spec.md is protected:
        # cat-file -e succeeds and we never discard real work.
        probe = _git("cat-file", "-e", "HEAD:lapis-spec.md")
        if probe.returncode != 0:
            _git("reset", "-q", "--", "lapis-spec.md")
        r = _git("diff", "--cached")
        final_diff = r.stdout if r.returncode == 0 else ""
        # Work presence: `git status --porcelain` non-empty. This is the
        # "did the run leave anything" check - it covers untracked-only runs
        # a bare `git diff HEAD` would discard.
        r = _git("status", "--porcelain")
        work_present = r.returncode == 0 and bool(r.stdout.strip())
        if not work_present:
            print(
                f"WARN: local-opencode: no work on disk (git status --porcelain "
                f"empty, task={task_id}) - no PR",
                file=sys.stderr,
            )
            return ""

        # F4: the deterministic targeted test step, in the RunTestsExecutor
        # shape (gw_agent.py:1074-1164): a repo-env-aware interpreter
        # prefix (agents-core-gate-uv-aware-v0 - _test_prefix(cwd): the uv
        # project venv `uv run --with pytest python`, bare `python` ->
        # venv python, for uv.lock-present AND uv-resolvable worktrees;
        # the host interpreter sys.executable otherwise - byte-for-byte the
        # legacy command), then `-m pytest`, a 180s cap (a network-waiting
        # test cannot stall the tail), and _parse_pytest_outcome. The
        # prefix is applied ONCE at the base list below (the if/else only
        # EXTENDS test_cmd) - a second prepend inside a branch body would
        # double the `uv run` and degrade to a spawn error. The
        # touched-test source is the STAGED DIFF (a git-diff source), not
        # a transcript.
        touched_tests = _collect_diff_touched_tests(cwd)
        from agents_core.gw_agent import _parse_pytest_outcome

        def _decode_partial(data) -> str:
            if data is None:
                return ""
            if isinstance(data, (bytes, bytearray)):
                return data.decode(errors="replace")
            return str(data)

        # agents-core-gate-uv-aware-v0: ONE pinned edit here (the shared
        # base list) - the branch bodies below only extend test_cmd.
        test_cmd = [*_test_prefix(cwd), "-m", "pytest"]
        if touched_tests:
            # The diff touched test files -> run exactly those.
            test_cmd.extend(sorted(touched_tests))
        else:
            # No test files touched -> the repo's deterministic baseline
            # run: test_command (a per-repo registry key; the pilot sets it
            # to the measured-green subset, which is how the repo's
            # accepted-baseline reds are excluded from the run) scoped to
            # the repo's test dir, minus integration/smoke.
            base_test_cmd = spec.get("test_command") or ""
            if base_test_cmd:
                test_cmd.extend(shlex.split(base_test_cmd))
            else:
                test_cmd.extend(
                    d for d in ("tests", "agents_core/tests")
                    if (Path(cwd) / d).is_dir()
                )
            test_cmd.extend(["-m", "not integration and not smoke"])
        test_cmd.append("-q")

        # F4 env: the model's worktree-pinned pip installs (handle.env's
        # PYTHONUSERBASE) must stay visible to the test run, but that
        # redirect ALSO hides the harness interpreter's own user site - on a
        # host whose runner python finds pytest in ~/.local (BRIX's system
        # python3), the run would die ModuleNotFoundError before collecting
        # a single test and the D1 gate would fail closed forever. Preserve
        # the unredirected user site via PYTHONPATH, which PYTHONUSERBASE
        # does not override (the worktree .pyuserbase site is still the
        # user site for the child, so the model's installs are visible too).
        from site import getusersitepackages as _usersite
        _host_usersite = _usersite()
        _f4_env = {**os.environ, **handle.env}
        if _host_usersite and os.path.isdir(_host_usersite):
            # agents-core-gate-uv-aware-v0: the PYTHONPATH shim is the
            # host-python path's fix (host interpreter finding pytest in
            # the host user site). Under the uv prefix the venv is the
            # authority - drop PYTHONPATH/PYTHONUSERBASE so the host
            # user-site shim cannot shadow the venv's packages.
            if _test_prefix(cwd)[0] != sys.executable:
                _f4_env.pop("PYTHONPATH", None)
                _f4_env.pop("PYTHONUSERBASE", None)
            else:
                _f4_env["PYTHONPATH"] = os.pathsep.join(
                    p for p in (_host_usersite, os.environ.get("PYTHONPATH")) if p
                )
        _pytest_output = ""
        _pytest_rc = -1
        _pytest_timed_out = False
        try:
            _r = subprocess.run(
                test_cmd,
                capture_output=True, text=True,
                timeout=180,
                cwd=cwd,
                env=_f4_env,
                shell=False,
            )
            _pytest_output = _r.stdout + _r.stderr
            _pytest_rc = _r.returncode
        except subprocess.TimeoutExpired as _e:
            _pytest_output = (
                _decode_partial(_e.stdout)
                + _decode_partial(_e.stderr)
                + f"\n[TIMEOUT after 180s]"
            )
            _pytest_timed_out = True
        last_test_outcome = _parse_pytest_outcome(
            _pytest_output, _pytest_rc, _pytest_timed_out
        )
        import sys as _sys_dbg
        print(f"DBG_F4 test_cmd={test_cmd!r} rc={_pytest_rc} output={_pytest_output!r} outcome={last_test_outcome!r}", file=_sys_dbg.stderr)

        # D4 (agents-core-local-fixer-gate-perception-v0): name the
        # unusable F4 re-run shape explicitly instead of folding it into a
        # bare 0/0. Ground truth: rc=4 output ends with "no tests ran in
        # 0.00s" (the "ERROR: file or directory not found:" line is NOT
        # last), so both rc=4 and rc=5 parse to 0/0/0 via the last-line
        # regex - log-only, no behavior change (the gate already
        # fail-closes on both).
        _f4_rc_name = ""
        if _pytest_rc in (4, 5):
            _f4_rc_name = (
                "f4-rc=4 touched-path-missing"
                if _pytest_rc == 4
                else "f4-rc=5 no-tests-ran"
            )

        # D1 decision block (the _run_local_fixer :690-714 pattern,
        # duplicated inline with opencode adaptation): positive-only -
        # every test the diff touched passes, and the run has at least one
        # pass; fail-closed - no test files touched -> the deterministic
        # targeted run must be green (the accepted-baseline reds are
        # excluded from the run itself via the per-repo test_command, S4b).
        def _tests_passed(outcome: dict | None) -> bool:
            if not outcome:
                return False
            return (
                int(outcome.get("passed") or 0) > 0
                and int(outcome.get("failed") or 0) == 0
                and int(outcome.get("errors") or 0) == 0
            )

        gate_passed = False
        if touched_tests:
            # Positive-only gate: a touched test "fails" if a FAILED/ERROR
            # node ID refers to it (file-level or node-level). The run must
            # also have at least one passing test.
            failed_node_ids = _extract_failed_node_ids(last_test_outcome)
            touched_failures = [
                t for t in touched_tests
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
            # Fail-closed fallback: no tests touched -> the targeted run
            # must be green.
            gate_passed = _tests_passed(last_test_outcome)

        # concluded: opencode exited 0. A deadline kill (loop budget
        # exhausted) or any non-zero exit is NOT concluded. There is no
        # max_steps / no_progress / mem_search_loop here - opencode does not
        # report them.
        concluded = (rc == 0) and not deadline_killed

        salvaged = False
        if not concluded:
            if deadline_killed and final_diff.strip() and gate_passed:
                # Terminal state 2: the loop budget expired mid-work, but
                # what is on disk is clean and green - salvage it as a PR.
                # (The salvage sign-off marker is attached in chunk 4: a
                # deadline kill can truncate the final write, so human
                # sign-off is always required.)
                salvaged = True
                print(
                    f"INFO: local-opencode: harness-salvaged clean diff on "
                    f"deadline kill (target={target_id}, task={task_id})",
                    file=sys.stderr,
                )
            else:
                # Terminal state 3: not concluded without a salvageable
                # clean-diff+green-tests. Distinct WARN per case.
                if deadline_killed and not final_diff.strip():
                    print(
                        f"WARN: local-opencode: deadline kill with empty diff "
                        f"(task={task_id}) - no PR",
                        file=sys.stderr,
                    )
                elif deadline_killed:
                    print(
                        f"WARN: local-opencode: deadline kill with test gate "
                        f"failed (task={task_id}) - no PR",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"WARN: local-opencode: opencode exited non-zero "
                        f"(exit={rc}) - no PR",
                        file=sys.stderr,
                    )
                return ""

        if not salvaged:
            if not final_diff.strip():
                print(
                    f"WARN: local-opencode: empty diff (work present but nothing "
                    f"diffed vs HEAD, task={task_id}) - no PR",
                    file=sys.stderr,
                )
                return ""
            if not gate_passed:
                # D2 (agents-core-local-fixer-gate-perception-v0): the
                # opencode engine has NO tail log (all _tail_log call sites
                # are in _run_local_fixer) - this stderr WARN is the
                # queue-runner-journal surface; it gains rc=/summary=
                # (errors= was already printed) + the D4 rc=4/5 naming.
                _oc_summary = str((last_test_outcome or {}).get("summary") or "").replace("\n", " ")[:200]
                print(
                    "WARN: local-opencode: test gate failed "
                    f"(touched_tests={sorted(touched_tests) if touched_tests else '[] (targeted baseline run)'}; "
                    f"passed={int((last_test_outcome or {}).get('passed') or 0)} "
                    f"failed={int((last_test_outcome or {}).get('failed') or 0)} "
                    f"errors={int((last_test_outcome or {}).get('errors') or 0)} "
                    f"rc={_pytest_rc} summary={_oc_summary!r}"
                    + (f" {_f4_rc_name}" if _f4_rc_name else "")
                    + ") - no PR",
                    file=sys.stderr,
                )
                return ""

        # Terminal state 1 (or 2 when salvaged): deterministic git/PR tail
        # (duplicated from the _run_local_fixer :821-892 shape with
        # opencode adaptations). checkout -B is the resolved upstream fix
        # (the old `checkout -b` fixer_retry fatal is gone); the index
        # already carries the F1 diff, the re-add is idempotent (the model
        # is dead; nothing else writes to the worktree).
        import agents_core.forgejo as _forgejo

        r = _git("checkout", "-B", branch)
        if r.returncode != 0:
            print(f"WARN: local-opencode: git checkout -B failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        r = _git("add", "-A")
        if r.returncode != 0:
            print(f"WARN: local-opencode: git add -A failed: {r.stderr.strip()}", file=sys.stderr)
            return ""
        # Re-apply the staged-spec guard after the re-stage (defensive: the
        # tail's add -A could re-stage a model-created lapis-spec.md).
        probe = _git("cat-file", "-e", "HEAD:lapis-spec.md")
        if probe.returncode != 0:
            _git("reset", "-q", "--", "lapis-spec.md")
        r = _git("commit", "-m", f"fix({target_id}): local-opencode harness")
        if r.returncode != 0:
            print(f"WARN: local-opencode: git commit failed: {r.stderr.strip()}", file=sys.stderr)
            return ""

        # F2: before push, verify the remote branch does not already exist.
        # For a fresh fixer the branch only appears on origin if the model
        # pushed it in violation of the HARNESS-OWNS-GIT contract (a
        # model-pushed orphan) - a distinct failure: no PR. fixer_retry
        # targets an already-open PR's branch, where the remote branch
        # legitimately pre-exists, so the check is scoped to the fresh case.
        if not (spec.get("existing_branch") or ""):
            remote = _git("ls-remote", "--exit-code", "origin", branch)
            if remote.returncode == 0:
                print(
                    f"WARN: local-opencode: remote branch {branch} already "
                    f"exists on origin (model-pushed orphan) - no PR",
                    file=sys.stderr,
                )
                return ""

        r = _git("push", "origin", f"HEAD:{branch}")
        if r.returncode != 0:
            print(f"WARN: local-opencode: git push failed: {r.stderr.strip()}", file=sys.stderr)
            return ""

        # S4(g): session provenance for the PR body. PRIMARY: the opencode
        # SQLite DB records the session under the --title the harness passed
        # (fixer-<task_id>); take the most recent by time_created. FALLBACK:
        # the first truthy top-level sessionID in the --format json event
        # stream. DEGRADED: neither -> session_record_unavailable, the PR
        # still opens. SESSION-EXPORT: on a found sessionID, run
        # `opencode export <session_id>` and persist its stdout; a WARN-only
        # failure never blocks the PR.
        session_id = None
        try:
            import sqlite3 as _sqlite3

            _db_path = os.path.expanduser("~/.local/share/opencode/opencode.db")
            _conn = _sqlite3.connect(_db_path, timeout=5)
            try:
                _row = _conn.execute(
                    "SELECT id FROM session WHERE title=? "
                    "ORDER BY time_created DESC LIMIT 1",
                    (f"fixer-{task_id}",),
                ).fetchone()
            finally:
                _conn.close()
            session_id = _row[0] if _row else None
        except Exception as _exc:
            print(
                f"WARN: local-opencode: opencode.db session query failed: {_exc}",
                file=sys.stderr,
            )
        if not session_id:
            # FALLBACK: parse the event stream line-by-line (each line is a
            # JSON object) and take the first truthy top-level sessionID.
            try:
                with open(events_path, "r", encoding="utf-8", errors="replace") as _ev:
                    for _line in _ev:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _obj = json.loads(_line)
                        except ValueError:
                            continue
                        if isinstance(_obj, dict):
                            _sid = _obj.get("sessionID")
                            if _sid:
                                session_id = _sid
                                break
            except OSError as _exc:
                print(
                    f"WARN: local-opencode: event stream sessionID parse "
                    f"failed: {_exc}",
                    file=sys.stderr,
                )
        session_record_unavailable = not session_id
        session_export_path = None
        if session_id:
            try:
                _exp = subprocess.run(
                    [opencode_bin, "export", session_id],
                    capture_output=True, text=True, timeout=60,
                )
                if _exp.returncode == 0:
                    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                    session_export_path = (
                        _ARTIFACT_DIR / f"{task_id}-opencode-session.json"
                    )
                    session_export_path.write_text(_exp.stdout)
                else:
                    print(
                        f"WARN: local-opencode: session export failed "
                        f"(exit={_exp.returncode}): {_exp.stderr.strip()}",
                        file=sys.stderr,
                    )
            except Exception as _exc:
                print(
                    f"WARN: local-opencode: session export failed: {_exc}",
                    file=sys.stderr,
                )

        # Provenance PR body - factual only.
        diff_lines = [l for l in final_diff.splitlines()
                      if l.startswith(("diff --git", "---", "+++", "@@", " ")) or l[:1] in ("+", "-")]
        diffstat = "\n".join(diff_lines[:40]) or "(no changes)"

        if last_test_outcome:
            passed_c = int(last_test_outcome.get("passed") or 0)
            failed_c = int(last_test_outcome.get("failed") or 0)
            test_summary = f"{passed_c} passed, {failed_c} failed (deterministic targeted run)"
        else:
            test_summary = "no test outcome recorded"

        terminal_state = "salvaged (deadline kill)" if salvaged else "concluded"
        # S4(g): session record section. DEGRADED case carries the explicit
        # "session record unavailable" text; the export path is shown only
        # when the export actually succeeded.
        session_record_line = (
            "session record unavailable"
            if session_record_unavailable
            else f"sessionID: {session_id}"
        )
        session_export_line = (
            f"session export: `{session_export_path}`"
            if session_export_path
            else "session export: unavailable"
        )
        pr_body = (
            f"Implemented by the local-opencode fixer harness "
            f"(opencode tool loop on {opencode_model}), not paid Claude.\n\n"
            f"## Terminal state\n\n{terminal_state}\n\n"
            f"## Diff summary\n\n```diff\n{diffstat}\n```\n\n"
            f"## Test outcome\n\n{test_summary}\n\n"
            f"## Session record\n\n{session_record_line}\n{session_export_line}\n\n"
            f"## Logs\n\n`{events_path}` / `{stderr_path}`\n\n"
            f"<!-- lapis-gpu-id: {task_id} -->\n"
            f"<!-- lapis-tid: {target_id} -->\n"
            f"<!-- lapis-engine: local-opencode -->"
            # S4(e): salvage sign-off marker. A deadline kill can truncate
            # the final write, so human sign-off is always required -
            # attached to ALL deadline salvages, never to clean runs.
            + ("\n<!-- lapis-no-progress-salvage: true -->" if salvaged else "")
        )

        pr = _forgejo.create_pr(
            repo=bare_repo,
            title=f"fix({target_id}): local-opencode",
            head=branch,
            base="main",
            body=pr_body,
        )
        return pr.get("html_url", "")

    except Exception as exc:
        print(f"WARN: local-opencode: unexpected error: {exc}", file=sys.stderr)
        return ""

    finally:
        # S4f: kill the opencode process group on EVERY exit path (timeout,
        # error, early return). opencode is a live, credentialed child
        # (FORGEJO_TOKEN + the shared parent-clone git creds + full bash);
        # with start_new_session=True its pid is the pgid, so killpg
        # reaches the whole tree, not just the direct child. Idempotent: on
        # the normal path the group is already gone and ProcessLookupError
        # is swallowed.
        if opencode_pgid is not None:
            try:
                os.killpg(opencode_pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        # Release the supervisor lease (mirrors the _run_local_fixer
        # finally): released on BOTH the success and failure paths. Soft
        # fail: a release failure is logged and swallowed - the TTL bounds
        # the zombie window.
        if sup_lease_client is not None:
            try:
                sup_lease_client.release("gravitywell", sup_lease_id)
            except Exception as exc:
                print(
                    f"WARN: local-opencode: supervisor lease release failed "
                    f"(work_id={sup_lease_id}): {exc}",
                    file=sys.stderr,
                )
            try:
                sup_lease_client.close()
            except Exception:
                pass
        # Tear down the own worktree (the :917-921 pattern).
        if worktree_path is not None:
            try:
                teardown_worktree(task_id, effective_cwd)
            except Exception as exc:
                print(
                    f"WARN: local-opencode: worktree teardown failed: {exc}",
                    file=sys.stderr,
                )


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


def _run_local_auditor(spec: dict, base_cwd: str | None) -> str | None:
    """Local-auditor engine (fixer-reception-v0, leg 1, D4).

    Sibling to _run_local_reviewer (the reference: max_steps default 24,
    writeable=False, json_mode=True). The auditor receives a non-passing
    termination, investigates the actual state (three-head suite runs via
    in-place git checkout in its single ephemeral worktree, diff reading,
    file:line evidence), and returns a structured JSON audit brief. The
    daemon (leg 2) encodes the JSON into the pm:auditor comment - the
    auditor posts nothing.

    (a) Worktree: the runner's existing worktree path (main() verifies
    existing_branch on origin via ls-remote --exit-code and checks it out)
    - this engine is NOT in the worktree_required exemption tuple, so the
    runner provisions the worktree at the PR head. The auditor's spec
    carries existing_branch via the shaper's existing_branch injection
    tuple (shaper.py, agent-keyed).
    (b) Tool grant: AUDITOR_TOOLS (DEFAULT_READONLY_TOOLS + run_tests) with
    the git executor restricted to AUDITOR_GIT_ALLOWLIST (read-only
    subcommands + checkout + rev-parse; NO fetch, NO push). NO
    run_command, NO apply_edit/write_file, NO web_fetch. writeable=False
    end-to-end.
    (c) max_steps/timeout: spec-carried (the shaper plumbs registry
    max_steps/timeout_s into the spec dict); defaults 100 / 1800.
    (d) NO WIP salvage hook, NO push on this engine (the salvage/push
    machinery is local-fixer-only; the auditor returns JSON and exits).
    """
    from agents_core.gw_agent import AUDITOR_TOOLS, call_gw_agent

    task_id = spec.get("task_id") or spec.get("slot_id") or "aud-unknown"
    cwd = base_cwd or "/srv/agents"
    model = spec.get("model")
    backend_url = spec.get("backend_url")

    reason: list[str] = []
    result = call_gw_agent(
        prompt=spec["prompt"],
        system=spec.get("system", ""),
        cwd=cwd,
        tools=AUDITOR_TOOLS,
        writeable=False,
        json_mode=True,
        timeout=int(spec.get("timeout_s", 1800)),
        think=False,
        on_wake_fail="skip",
        work_id=task_id,
        max_steps=int(spec.get("max_steps", 100)),
        model=model,
        backend_url=backend_url,
        acquire_lease=spec.get("acquire_lease", True),
        lease_class="deferrable",
        reason_out=reason,
    )
    if result is None:
        why = reason[0] if reason else "no_content_no_reason"
        print(f"ERROR: local auditor produced no verdict (reason={why})", file=sys.stderr)
        return None
    return result


def _engine_dispatch_exit(pr_url: str, engine: str) -> None:
    """Terminal exit for the local engine dispatch blocks (agents-core-
    shaperunner-fail-closed-v0).

    The tail's catch-all partition for an unclassified terminal death
    returns the TAIL_UNCLASSIFIED_DEATH sentinel instead of "". Map it to
    a distinct non-zero exit code - 3 (distinct from 1 = call returned
    None, 2 = config error) - with a loud ERROR line on stderr carrying
    the same observed cause the tail already printed. rc is the only
    signal the claude-queue-runner logs for a shaped_runner invocation,
    so rc=3 makes a ghost death show as a failed dispatch (queue.fail)
    instead of a success with an empty stdout line. A normal PR URL (or
    "" from a NAMED failure path - no_progress / max_steps / empty diff /
    gate-rejected salvage) still exits 0 with the URL printed: those
    paths are already loud in the log and are the guard-scaling item's
    territory, not this unit's.
    """
    if pr_url == TAIL_UNCLASSIFIED_DEATH:
        print(
            f"ERROR: {engine} dispatch ended in an unclassified terminal "
            f"death (no salvageable WIP; the tail's observed cause is on "
            f"stderr above) - exiting 3 so the queue records a failed "
            f"dispatch",
            file=sys.stderr,
        )
        sys.exit(3)
    print(pr_url)
    return None


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
    # runs the deterministic git/PR tail around the local GW fixer harness
    # (the seat is spec-driven - registry `agent.model`; the served model is
    # echoed per call via served_model_out, local-reviewer-identity-and-
    # provenance-v0 L1.D1).
    engine = spec.get("engine", "claude")
    if engine == "local-fixer":
        pr_url, prov = _run_local_fixer(spec, base_cwd)
        try:
            spec_path.unlink()
        except OSError:
            pass
        _engine_dispatch_exit(pr_url, engine)
        _print_provenance_line(prov)
        return
    elif engine == "local-fixer-staged":
        # fixers-harness-staged-v0 (S2): the staged fixer harness
        # (READER -> AIMER -> FIRE) - mirrors the local-fixer block.
        pr_url = _run_local_fixer_staged(spec, base_cwd)
        try:
            spec_path.unlink()
        except OSError:
            pass
        _engine_dispatch_exit(pr_url, engine)
        # A non-sentinel (normal PR URL) result must NOT fall through to
        # the shared call_claude_cli tail below the if/elif chain - the
        # local-fixer block's return after _engine_dispatch_exit is the
        # contract for every local engine dispatch block.
        return
    elif engine == "local-opencode":
        pr_url = _run_local_opencode(spec, base_cwd)
        try:
            spec_path.unlink()
        except OSError:
            pass
        _engine_dispatch_exit(pr_url, engine)
        # Same contract as the local-fixer block: terminal on this path.
        return
    elif engine not in ("claude", "local-reviewer", "local-auditor"):
        # Fail loud on an unknown engine: a silent call_claude_cli
        # fall-through would burn a seat on a spec no engine understands.
        print(
            f"ERROR: unknown shaped-runner engine {engine!r}; allowed: "
            "claude, local-fixer, local-fixer-staged, local-opencode, "
            "local-reviewer, local-auditor",
            file=sys.stderr,
        )
        sys.exit(2)

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
        elif engine == "local-auditor":
            # fixer-reception-v0 (leg 1, D4): the auditor runs through the
            # same worktree path (worktree_required=True, existing_branch
            # verified on origin) and returns its JSON audit brief; the
            # daemon encodes it into the pm:auditor comment.
            result = _run_local_auditor(spec, cwd)
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
