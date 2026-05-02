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
