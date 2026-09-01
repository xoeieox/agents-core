"""agents_core.fixer_stages — the staged fixer harness (fixers-harness-staged-v0, S1).

The staged decomposition: 2 LLM stages (READER -> AIMER) + 1 deterministic
stage (FIRE), run by the lapis-pm daemon via shaped_runner's
`_run_local_fixer_staged` engine for `fixer_staged` dispatches.

Design invariants (spec arc rev 4.2, staged spec rev 3.4):

- Both LLM stages run `writeable=True` (so the handler / before_tool /
  after_step hooks and the no-progress guard are active) with the CLOSED
  TOOLSET: `tools={read_file, grep}` schemas AND a restricted
  `tool_executors={ReadFileExecutor, GrepExecutor(line_numbers=True)}` -
  both knobs, so the write executors are absent from the execution map and
  off-schema `apply_edit`/`write_file` calls hit the `unknown tool` error.
  The stage write surface is structurally zero, not prompt-level.
- Persona load: resolve_under_cards_root + validate_deck_card (error
  strings checked loud at dispatch) + PersonaCardEntity.load. The stage
  system prompt is assembled from the four entity attributes (weighted
  primitives / voice_exemplars / domains / kernel_invariants); the
  entity's `system_prompt()` is NEVER called (its hardcoded "You are a
  reviewer persona" framing would leak council context into a fixer stage).
- The mission fence (fenced YAML inside the steer directive) is the only
  task input that carries defect detail. Raw reviewer issues never reach a
  stage.
- Model-generated text is rendered ONLY as data: YAML-dumped into prompt
  blocks and json.dump'd into the report; never a str.format template,
  never a shell argument, never part of a file path.
- The mission report is written on every terminal branch, before the
  return; it round-trips through json.load.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The whole-mission wall clock (monotonic). The deadline is a hard gate
# evaluated BEFORE each stage dispatch: a stage may start at
# elapsed <= MISSION_DEADLINE_S - STAGE_TIMEOUT_S and run its full
# STAGE_TIMEOUT_S (bounded by the loop's own deadline + the ~20s
# forced-conclusion floor overshoot).
MISSION_DEADLINE_S = 2400
STAGE_TIMEOUT_S = 900

# The staged invocation sets this env for the stage subprocess scope (the
# runner is a per-dispatch subprocess, so process-scoped env is safe):
# the loop's internal hard cap on total exploration tool calls
# (gw_agent.py, default 20) would otherwise preempt a progressing 16-step
# Reader at ~step 10 (2 tool calls per response). 64 exceeds the planned
# <=32 calls (16 steps x <=2 calls/step); at >=4 calls/step the cap can
# still fire first - a cap hit is terminal via the stage-failure partition
# (the named bounded residual, not a guarantee).
STAGE_MAX_EXPLORE_STEPS = 64

# Per-stage budgets (explicit - the stages never rely on the loop defaults).
READER_MAX_STEPS = 16
AIMER_MAX_STEPS = 12
STAGE_NO_PROGRESS_STEPS = 6

# Keep-look budget: 2 Aimer rejections; the third is terminal.
KEEP_LOOK_REJECTIONS = 2

# The mission report + tail log + staged transcript live in one named
# provenance directory (room_path("gpu_queue.shaped")).
_REPORT_DIRNAME = "gpu_queue.shaped"

# The mission fence markers (the fenced YAML block inside the steer
# directive).
_FENCE_RE = re.compile(
    r"```mission\s*\n(.*?)```", re.DOTALL
)
# The stage artifact fence (the fenced JSON block in the stage's final
# message): ```map / ```aim.
_MAP_FENCE_RE = re.compile(r"```map\s*\n(.*?)```", re.DOTALL)
_AIM_FENCE_RE = re.compile(r"```aim\s*\n(.*?)```", re.DOTALL)

# The mission's tests_timeout_s range (parse-time fail-loud): the 240 cap
# keeps the worst completed path under the 2760s process kill
# (2420 stage phase + 240 gate + ~60 push = 2720 < 2760, 40s margin); the
# 60 floor is below a gate that tests nothing.
TESTS_TIMEOUT_MIN_S = 60
TESTS_TIMEOUT_MAX_S = 240


class MissionError(Exception):
    """The mission fence is malformed or fails a fail-loud edge.

    Raised at parse time - the dispatch aborts BEFORE any GPU spend (same
    class as the card-validation errors).
    """


# ---------------------------------------------------------------------------
# Path normalization (M1): identical at parse and at gate.
# ---------------------------------------------------------------------------

def _normalize_worktree_path(path: str, cwd: str) -> str:
    """Normalize a model/mission-supplied file path to a worktree-relative
    form. cwd-strip -> os.path.normpath -> exact match against the scope
    set. Absolute paths and `..` escapes are rejected (MissionError).
    """
    if not isinstance(path, str) or not path:
        raise MissionError(f"empty file path: {path!r}")
    if os.path.isabs(path):
        raise MissionError(f"absolute path rejected (worktree-relative required): {path!r}")
    normed = os.path.normpath(path)
    if normed.startswith("..") or "/../" in normed:
        raise MissionError(f"path escapes the worktree: {path!r}")
    return normed


# ---------------------------------------------------------------------------
# parse_mission (S1)
# ---------------------------------------------------------------------------

@dataclass
class Mission:
    """The parsed mission artifact (the only task input that carries
    defect detail). All file paths are worktree-relative (normalized)."""
    pre_aimed: bool
    scope_files: list[str]
    tests: list[str]
    tests_timeout_s: int
    sites: list[dict] = field(default_factory=list)
    recipe: list[dict] = field(default_factory=list)
    raw: str = ""  # the raw fence text (the partial_artifact diagnostic)

    def scope_set(self) -> set[str]:
        return set(self.scope_files)


def _extract_mission_fence(directive: str) -> str:
    """Extract the ```mission fence body from the steer directive text."""
    m = _FENCE_RE.search(directive or "")
    if not m:
        raise MissionError("no ```mission fence found in the directive")
    return m.group(1)


def parse_mission(directive: str) -> Mission:
    """Parse + validate the mission fence.

    Fail-loud edges (each aborts the dispatch before any GPU spend):
      (a) scope_files or tests empty;
      (b) tests_timeout_s absent, non-integer, or outside [60, 240];
      (c) pre_aimed: true with a missing/empty recipe or a recipe entry
          missing file/old/new.
    A YAML syntax error or a truncated fence raises MissionError carrying
    the RAW fence text (a YAML error cannot name the offending field, so
    the raw dump is the diagnostic).
    """
    raw = _extract_mission_fence(directive)
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise MissionError(f"mission fence YAML error: {exc}; raw fence:\n{raw}") from exc
    if not isinstance(data, dict):
        raise MissionError(
            f"mission fence is not a mapping; raw fence:\n{raw}"
        )

    pre_aimed = bool(data.get("pre_aimed", False))

    scope_files = data.get("scope_files") or []
    tests = data.get("tests") or []
    if not isinstance(scope_files, list) or not scope_files:
        raise MissionError("mission scope_files is missing or empty")
    if not isinstance(tests, list) or not tests:
        raise MissionError("mission tests is missing or empty")

    tests_timeout_s = data.get("tests_timeout_s")
    if tests_timeout_s is None or isinstance(tests_timeout_s, bool) \
            or not isinstance(tests_timeout_s, int):
        raise MissionError(
            f"mission tests_timeout_s is absent or non-integer: "
            f"{tests_timeout_s!r}"
        )
    if not (TESTS_TIMEOUT_MIN_S <= tests_timeout_s <= TESTS_TIMEOUT_MAX_S):
        raise MissionError(
            f"mission tests_timeout_s {tests_timeout_s} outside "
            f"[{TESTS_TIMEOUT_MIN_S}, {TESTS_TIMEOUT_MAX_S}]"
        )

    # Normalize the scope + test paths (worktree-relative; absolute and ..
    # rejected at parse time).
    cwd = ""  # parse-time: no cwd to strip against; the gate re-normalizes
    # against the real worktree cwd with the identical rule.
    scope_norm = [_normalize_worktree_path(p, cwd) for p in scope_files]
    tests_norm = [_normalize_worktree_path(p, cwd) for p in tests]

    sites = data.get("sites") or []
    if not isinstance(sites, list):
        raise MissionError("mission sites is not a list")
    recipe = data.get("recipe") or []
    if not isinstance(recipe, list):
        raise MissionError("mission recipe is not a list")

    if pre_aimed:
        if not recipe:
            raise MissionError(
                "pre_aimed: true with a missing/empty recipe"
            )
        for i, entry in enumerate(recipe):
            if not isinstance(entry, dict):
                raise MissionError(f"recipe entry {i} is not a mapping")
            for key in ("file", "old", "new"):
                if key not in entry:
                    raise MissionError(
                        f"recipe entry {i} missing {key!r}"
                    )
            entry["file"] = _normalize_worktree_path(entry["file"], cwd)

    return Mission(
        pre_aimed=pre_aimed,
        scope_files=scope_norm,
        tests=tests_norm,
        tests_timeout_s=tests_timeout_s,
        sites=sites,
        recipe=recipe,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Persona seam (S1) - move-only: data, not code.
# ---------------------------------------------------------------------------

def load_stage_persona(card_relpath: str) -> dict:
    """Load a stage persona card (reader.yaml / aimer.yaml) under the cards
    root.

    resolve_under_cards_root (symlink/absolute-path escape raises) ->
    validate_deck_card (error strings checked loud at dispatch) ->
    PersonaCardEntity.load. Returns the entity's four attributes as a
    plain dict (the harness assembles the system prompt from them
    directly; `system_prompt()` is never called).
    """
    from agents_core.cards import (
        cards_root,
        resolve_under_cards_root,
        validate_deck_card,
    )
    from agents_core.council.persona_card_entity import PersonaCardEntity

    root = cards_root()
    card_path = resolve_under_cards_root(root / card_relpath)
    errors = validate_deck_card(card_path)
    if errors:
        # Fail loud at dispatch - a failing card aborts before any GPU spend.
        raise MissionError(
            f"invalid stage persona card {card_relpath}: {'; '.join(errors)}"
        )
    entity = PersonaCardEntity.load(card_path, llm=None)
    return {
        "slug": entity.slug,
        "primitives": entity.primitives,
        "voice_exemplars": entity.voice_exemplars,
        "domains": entity.domains,
        "kernel_invariants": entity.kernel_invariants,
    }


def build_stage_system_prompt(persona: dict, stage: str) -> str:
    """Assemble the stage system prompt from the four entity attributes.

    (1) the weighted composition, (2) the voice exemplars, (3) the
    domains, (4) the kernel invariants (the canonical 8, verbatim). The
    entity's system_prompt() is never called (its "You are a reviewer
    persona" framing would leak council context into a fixer stage).
    """
    weighted = ", ".join(
        f"{k} ({v:.2f})" for k, v in persona.get("primitives", {}).items()
    )
    exemplars = "\n".join(
        f"- {e}" for e in persona.get("voice_exemplars", [])
    )
    domains = ", ".join(persona.get("domains", []))
    invariants = "\n".join(
        f"- {inv}" for inv in persona.get("kernel_invariants", [])
    )
    return (
        f"You are the {stage.upper()} stage of the staged fixer harness "
        f"(persona: {persona.get('slug', stage)}).\n\n"
        f"Your composition: {weighted}\n\n"
        f"Your voice — speak in this register, calibrated to these exemplars:\n"
        f"{exemplars}\n\n"
        f"Your domains: {domains}\n\n"
        f"Kernel invariants you hold:\n{invariants}\n"
    )


# ---------------------------------------------------------------------------
# The closed stage toolset (the design's security invariant).
# ---------------------------------------------------------------------------

def stage_tools_schema() -> dict:
    """The staged tools SCHEMA shown to the model: read_file + grep only.

    The grep description is the updated one (returns file:line:content
    matches) - the staged harness supplies its own restricted schema; the
    legacy default schema (gw_agent DEFAULT_READONLY_TOOLS) is unchanged.
    """
    from agents_core.gw_agent import DEFAULT_READONLY_TOOLS

    read_file = json.loads(json.dumps(DEFAULT_READONLY_TOOLS["read_file"]))
    grep = json.loads(json.dumps(DEFAULT_READONLY_TOOLS["grep"]))
    grep["function"]["description"] = (
        "Search for a pattern in files using ripgrep. Returns "
        "file:line:content matches (up to 100 matches)."
    )
    return {"read_file": read_file, "grep": grep}


def stage_tool_executors(cwd: str) -> dict:
    """The restricted EXECUTOR map: ReadFileExecutor + GrepExecutor
    (line_numbers=True) ONLY. No apply_edit / write_file / run_tests / git
    / mem / list_open_prs - the stage write surface is structurally zero
    (off-schema calls hit the `unknown tool` error).
    """
    from agents_core.gw_agent import ReadFileExecutor, GrepExecutor

    return {
        "read_file": ReadFileExecutor(cwd),
        "grep": GrepExecutor(cwd, line_numbers=True),
    }


# ---------------------------------------------------------------------------
# Stage task prompt (the no-write-tools override + the leads instruction
# for Reader).
# ---------------------------------------------------------------------------

def build_stage_task_prompt(
    mission: Mission,
    stage: str,
    re_hunt_reasons: list[str] | None = None,
) -> str:
    """The stage's first user turn.

    OPENS with the explicit no-write-tools override (HIGH-2: the
    writeable=True tool block renders "To change code you MUST call
    apply_edit or write_file"; the override neutralizes the contradiction
    in the user turn where the model's attention is).

    The mission block is rendered as a YAML dump; it is NEVER passed
    through str.format (data braces must not be interpreted as format
    placeholders). Rejection reasons (a re-hunt) ride along as a YAML
    block - data-only, never an interpolated template.
    """
    parts: list[str] = []
    parts.append(
        "You have no write tools. Do NOT call apply_edit or write_file - "
        "you will not be given them. Your only job is to output the fenced "
        "artifact."
    )
    parts.append("")
    parts.append("## Mission")
    mission_dump = yaml.safe_dump(
        {
            "pre_aimed": mission.pre_aimed,
            "scope_files": mission.scope_files,
            "tests": mission.tests,
            "tests_timeout_s": mission.tests_timeout_s,
            "sites": mission.sites,
            "recipe": mission.recipe if stage == "aimer" else None,
        },
        default_flow_style=False,
        sort_keys=False,
    )
    parts.append(mission_dump)
    if stage == "reader":
        parts.append(
            "You are the READER. Read the mission's scope files and produce "
            "a candidate file:line map of the defects. Do NOT edit anything. "
            "Your final message MUST end with a fenced ```map JSON block: "
            "an object with 'entries' (a list of {file, line_range, defect, "
            "evidence}) and an OPTIONAL 'leads' array (out-of-scope "
            "observations NOTICED from in-scope reads - {file, note} - "
            "record, do not pursue; the scope gate is unchanged and a lead "
            "is never a tool call)."
        )
    else:
        parts.append(
            "You are the AIMER. Confirm the target: verify each site's "
            "current text, and emit the exact edit artifact. Your final "
            "message MUST end with a fenced ```aim JSON block: an object "
            "with 'verdict' ('confirm' or 'reject_keep_looking'), "
            "'reasons' (a list of strings - on a rejection, name the entry "
            "and the reason), and 'aim' (a list of per-entry "
            "{file, old_string, new_string, evidence})."
        )
    if re_hunt_reasons:
        parts.append("")
        parts.append("## Re-hunt context (the previous Aimer rejection)")
        # Data-only: YAML-dumped, never a str.format template.
        parts.append(yaml.safe_dump(
            {"reasons": list(re_hunt_reasons)},
            default_flow_style=False,
        ))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stage closures (before_tool scope gate / handler_hook redirect /
# after_step ledger).
# ---------------------------------------------------------------------------

@dataclass
class StageLedger:
    """The after_step observation-only ledger for one stage run."""
    entries: list[dict] = field(default_factory=list)
    redirects: list[dict] = field(default_factory=list)
    interventions_used: int = 0
    scope_rejects: int = 0


def _scope_gate(mission: Mission, cwd: str, ledger: "StageLedger"):
    """The deterministic before_tool closure: every read_file call's file
    argument must be in mission.scope_files (normalized per the artifact
    contract). Violation -> {"decision": "reject", "reason": <the scope
    note>}. grep is EXEMPT (its path_glob is a search hint, not a read;
    the executor is cwd-confined to the worktree, so a stage's read
    surface is strictly more constrained than the legacy fixer's).

    Boundedness: a rejected novel path counts as progress in the novelty
    detector (path-based); repeated identical out-of-scope calls are
    killed by the existing repeat-call detector. A scope-ignoring model
    is terminated, not looped.
    """
    scope_set = mission.scope_set()

    def _gate(tool_name: str, tool_args: dict) -> dict:
        if tool_name != "read_file":
            # grep is exempt (the scope-gate grep ruling, rev 3.3).
            return {"decision": "proceed"}
        path_arg = (tool_args or {}).get("path", "")
        try:
            normed = _normalize_worktree_path(path_arg, cwd)
        except MissionError:
            normed = None
        if normed is not None and normed in scope_set:
            return {"decision": "proceed"}
        ledger.scope_rejects += 1
        scope_note = (
            "The mission scope_files (worktree-relative) are: "
            + ", ".join(sorted(scope_set))
            + ". read_file targets outside this scope are rejected. "
            "Out-of-scope observations go in the map's 'leads' field - "
            "record, do not pursue."
        )
        return {"decision": "reject", "reason": scope_note}

    return _gate


def _handler_hook(ledger: StageLedger, mission: Mission, stage: str,
                  max_interventions: int = 2):
    """The deterministic handler_hook closure: fires ONLY when the
    no-progress guard is about to fire (the loop's hook seam does the
    injection; the harness's mission state computes the verdict).

    Verdict contract (gw_agent.py):
      - decision="redirect" + redirect string -> injected as a user
        message, consecutive_no_progress resets, consumes one
        intervention.
      - decision="continue" -> resets the counter, appends nothing,
        consumes one intervention.
      - anything else -> silent fall-through to the static nudge, NOT
        counted against the budget.

    Budget: max 2 redirects per stage run; a third drift falls through to
    the static nudge -> the hardcoded abort -> stage-failure partition
    (terminal, the report carries the redirect ledger). `continue` is
    reserved for the final-verification window and also costs 1 of the 2.
    """
    def _hook(ctx: dict) -> dict:
        if ledger.interventions_used >= max_interventions:
            # Budget exhausted: fall through to the static nudge path
            # (an unrecognized decision - NOT counted against the budget).
            ledger.redirects.append({
                "step": ctx.get("step_num"),
                "decision": "fallthrough",
                "note": "intervention budget exhausted - static nudge path",
            })
            return {"decision": "fallthrough"}
        # The named re-aim: the file:line-range the mission says to look
        # at next, or the next unverified site.
        sites = mission.sites or []
        if sites:
            next_site = sites[0]
            re_aim = (
                f"Re-aim: look at {next_site.get('file', 'the mission scope')} "
                f"around line {next_site.get('line', '?')}. "
                f"Defect: {next_site.get('defect', 'see mission')}. "
                f"Read the window and record it in your artifact."
            )
        else:
            re_aim = (
                "Re-aim: re-read the mission scope files in line windows "
                "and record the map/aim artifact."
            )
        ledger.interventions_used += 1
        ledger.redirects.append({
            "step": ctx.get("step_num"),
            "decision": "redirect",
            "redirect": re_aim,
        })
        return {"decision": "redirect", "redirect": re_aim}

    return _hook


def _after_step(ledger: StageLedger):
    """The observation-only after_step closure: records
    {step_num, tool, args, result-summary} for the mission report. It
    cannot inject or stop (Callable[[dict], None]).
    """
    def _hook(ctx: dict) -> None:
        step_num = ctx.get("step_num")
        for entry in ctx.get("transcript") or []:
            ledger.entries.append({
                "step_num": step_num,
                "tool": entry.get("tool_name"),
                "args": entry.get("arguments"),
                "result_summary": str(entry.get("result"))[:200],
                "error": entry.get("error"),
            })

    return _hook


# ---------------------------------------------------------------------------
# Stage invocation
# ---------------------------------------------------------------------------

def run_stage(
    *,
    stage: str,
    mission: Mission,
    cwd: str,
    persona: dict,
    re_hunt_reasons: list[str] | None = None,
    log=None,
) -> tuple[dict, list[dict], StageLedger]:
    """Run one LLM stage (reader/aimer) with the closed toolset.

    Returns (FixerResult, transcript, ledger). The stage's final message
    is carried on FixerResult["result_text"] (the S0 additive change).
    """
    from agents_core.gw_agent import call_gw_agent

    max_steps = READER_MAX_STEPS if stage == "reader" else AIMER_MAX_STEPS
    ledger = StageLedger()
    system = build_stage_system_prompt(persona, stage)
    prompt = build_stage_task_prompt(mission, stage, re_hunt_reasons)

    # The explore cap env (the process-scoped env is safe: the runner is a
    # per-dispatch subprocess).
    os.environ["GW_AGENT_MAX_EXPLORE_STEPS"] = str(STAGE_MAX_EXPLORE_STEPS)

    try:
        fixer_result, transcript = call_gw_agent(
            prompt=prompt,
            system=system,
            cwd=cwd,
            writeable=True,
            timeout=STAGE_TIMEOUT_S,
            max_steps=max_steps,
            no_progress_steps=STAGE_NO_PROGRESS_STEPS,
            return_transcript=True,
            tools=stage_tools_schema(),
            tool_executors=stage_tool_executors(cwd),
            before_tool=_scope_gate(mission, cwd, ledger),
            handler_hook=_handler_hook(ledger, mission, stage),
            handler_objective=f"{stage} stage: output the fenced artifact",
            handler_max_interventions=2,
            after_step=_after_step(ledger),
            log=log,
        )
    except Exception as exc:  # never-raises: a stage crash is a stage failure
        if log:
            log(f"[fixer_stages] {stage} stage raised: {exc}")
        raise

    return fixer_result, transcript, ledger


# ---------------------------------------------------------------------------
# Stage artifact parsing (the stage-failure partition).
# ---------------------------------------------------------------------------

def parse_stage_artifact(result_text: str, stage: str) -> tuple[dict | None, str]:
    """Parse the fenced artifact (map/aim) out of the stage's final message.

    Returns (artifact_dict_or_None, reasoning). The fenced artifact is
    parsed out; the remainder of the final message (the model's
    reasoning/justification) is the stage's `reasoning` in the mission
    report. A stage that ends without a parseable artifact is handled by
    the stage-failure partition (the caller).

    The partition routes on PARSEABILITY, never on the budget_forced flag:
    a budget-forced stage may still deliver a fully parseable artifact.
    """
    text = result_text or ""
    fence_re = _MAP_FENCE_RE if stage == "reader" else _AIM_FENCE_RE
    m = fence_re.search(text)
    if not m:
        # No fence at all: no artifact; the whole text is the reasoning.
        return None, text
    raw = m.group(1).strip()
    try:
        artifact = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        # Truncated fence / malformed JSON: no artifact; the raw fence
        # text is the partial_artifact diagnostic.
        return None, text
    if not isinstance(artifact, dict):
        return None, text
    # The reasoning is the text OUTSIDE the artifact fence.
    reasoning = (text[:m.start()] + text[m.end():]).strip()
    return artifact, reasoning


# ---------------------------------------------------------------------------
# The pre-aimed match diagnostic (Council Q1).
# ---------------------------------------------------------------------------

def pre_aimed_match_diagnostic(
    mission: Mission,
    cwd: str,
    aim_entries: list[dict] | None = None,
    aimer_stated_reason: str = "",
) -> list[dict]:
    """The deterministic per-site match (old_string presence + count at
    the current worktree state), lifted into the terminal branches.

    Distinguishes the two failure causes the Council named: CODE DRIFT
    (old_string absent or ambiguous - the PR head moved under the recipe)
    from MODEL HALLUCINATION/MISAIM (match clean, the stage still claims
    mismatch). Zero extra LLM cost - the matching is the fire path's own
    check.

    Precondition (stated): the worktree is unmodified from the VERIFIED
    existing_branch checkout (stages are read-only), so the match runs
    against the PR head.
    """
    entries = aim_entries if aim_entries is not None else mission.recipe
    diagnostic = []
    for i, entry in enumerate(entries):
        file = entry.get("file", "")
        old = entry.get("old", "")
        line_hint = entry.get("line")
        found = False
        match_count = 0
        context_check = ""
        try:
            path = Path(cwd) / file
            content = path.read_text()
            match_count = content.count(old)
            found = match_count >= 1
            # The actual current text at the site (N-line window).
            lines = content.splitlines()
            if line_hint is not None and 1 <= int(line_hint) <= len(lines):
                start = max(0, int(line_hint) - 3)
                end = min(len(lines), int(line_hint) + 3)
                context_check = "\n".join(lines[start:end])
        except (OSError, ValueError):
            found = False
            match_count = 0
            context_check = "(unreadable)"
        diagnostic.append({
            "file": file,
            "line_hint": line_hint,
            "old_string_found": found,
            "match_count": match_count,
            "context_check": context_check,
            "aimer_stated_reason": aimer_stated_reason,
        })
    return diagnostic


# ---------------------------------------------------------------------------
# Fire (deterministic).
# ---------------------------------------------------------------------------

class AimError(Exception):
    """A fire-time aim failure: path outside scope, or old_string count
    mismatch (0 or >1) for an entry."""


def _apply_aim_entry(cwd: str, entry: dict, scope_set: set[str]) -> None:
    """Per-edit: (1) path-validate (worktree-relative, normalized, member
    of mission.scope_files - HIGH-1: Aimer-emitted paths are model output
    and get the same treatment as any model output); (2) reuse
    ApplyEditExecutor semantics (file exists, old_string count == 1,
    single apply). A count mismatch = AimError for that entry.
    """
    from agents_core.gw_agent import ApplyEditExecutor

    file = entry.get("file", "")
    try:
        normed = _normalize_worktree_path(file, cwd)
    except MissionError as exc:
        raise AimError(f"entry path invalid: {exc}") from exc
    if normed not in scope_set:
        raise AimError(
            f"entry path {normed!r} outside the mission scope "
            f"{sorted(scope_set)}"
        )
    executor = ApplyEditExecutor(cwd)
    result = executor.execute({
        "path": normed,
        "old_string": entry.get("old", ""),
        "new_string": entry.get("new", ""),
    })
    if isinstance(result, dict) and "error" in result:
        raise AimError(f"apply_edit failed for {normed!r}: {result['error']}")


def fire(
    mission: Mission,
    aim: dict,
    cwd: str,
) -> tuple[bool, list[dict]]:
    """Apply the aim artifact verbatim, per entry, IN ORDER.

    Returns (all_applied, fire_results). Partial apply: any AimError ->
    the gate is skipped by the caller (a gate on a partially applied set
    is meaningless) -> worktree salvage directly.
    """
    scope_set = mission.scope_set()
    entries = aim.get("aim") or []
    fire_results: list[dict] = []
    all_applied = True
    for i, entry in enumerate(entries):
        file = entry.get("file", "")
        try:
            _apply_aim_entry(cwd, entry, scope_set)
            fire_results.append({
                "index": i,
                "file": file,
                "applied": True,
                "error": None,
            })
        except AimError as exc:
            fire_results.append({
                "index": i,
                "file": file,
                "applied": False,
                "error": str(exc),
            })
            all_applied = False
            # Partial apply: stop (the remaining entries are not applied;
            # the gate runs iff ALL entries applied).
            break
    return all_applied, fire_results


# ---------------------------------------------------------------------------
# The mission report writer.
# ---------------------------------------------------------------------------

def _report_dir() -> Path:
    from agents_core.room_paths import room_path

    d = room_path(_REPORT_DIRNAME)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_mission_report(
    task_id: str,
    mission: Mission,
    report: dict,
) -> Path:
    """Write the mission report to
    /srv/lapis/gpu-queue/shaped/<task_id>-staged-report.json.

    Written on every terminal branch, before the return. Model fields
    (reasoning, leads, partial_artifact, aimer_stated_reason) are
    json.dump'd data - the data-only pin. The report round-trips through
    json.load.
    """
    report = dict(report)
    report["mission"] = {
        "pre_aimed": mission.pre_aimed,
        "scope_files": mission.scope_files,
        "tests": mission.tests,
        "tests_timeout_s": mission.tests_timeout_s,
        "sites": mission.sites,
        # The recipe is echoed sanitized (no raw reviewer issues - the
        # mission fence is the only task input that carries defect detail;
        # the recipe itself is PM-proven, not model output).
        "recipe": mission.recipe,
    }
    path = _report_dir() / f"{task_id}-staged-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, default=str, indent=2))
    return path


def write_staged_transcript(
    task_id: str,
    stage_transcripts: list[dict],
) -> Path:
    """Persist ONE consolidated mission transcript to
    /srv/lapis/gpu-queue/shaped/<task_id>-staged-transcript.json (all stage
    runs, in stage order - the legacy per-run pattern writes
    <task_id>-gw-transcript.json regardless of outcome).
    """
    path = _report_dir() / f"{task_id}-staged-transcript.json"
    path.write_text(json.dumps(stage_transcripts, ensure_ascii=False, default=str))
    return path


# ---------------------------------------------------------------------------
# The staged mission orchestration (run by _run_local_fixer_staged).
# ---------------------------------------------------------------------------

@dataclass
class StagedOutcome:
    """The outcome of a staged mission run."""
    all_applied: bool = False
    gate_outcome: dict | None = None
    gate_passed: bool = False
    stop_reason: str = ""
    final_state: str = ""
    aim: dict | None = None
    fire_results: list[dict] = field(default_factory=list)
    pre_aimed_match_diagnostic: list[dict] | None = None
    report_path: Path | None = None
    pr_url: str = ""


def run_staged_mission(
    *,
    spec: dict,
    cwd: str,
    directive: str,
    log=None,
    gate_rerun=None,
    tail_finalize=None,
) -> StagedOutcome:
    """Run the full staged mission: mission parse, stage orchestration
    with MISSION_DEADLINE_S=2400 (monotonic, checked between stages
    pre-dispatch), keep-look (2-rejection budget; pre-aimed terminal
    rejection + the per-site match diagnostic), the stage-failure
    partition, the fire (per-edit path validation + ApplyEditExecutor
    reuse; partial -> skip gate), and the mission report.

    `gate_rerun` is a callable (cwd, touched, timeout_s) -> outcome dict
    | None (the staged tail's _gate_targeted_rerun with the mission's
    tests_timeout_s). `tail_finalize` is the shared tail helper (S6) -
    called with the staged values when all entries applied and the gate
    passes.
    """
    task_id = spec.get("task_id") or spec.get("slot_id") or "staged-unknown"
    target_id = spec.get("target_id", "unknown")

    # Mission parse (fail-loud before any GPU spend).
    try:
        mission = parse_mission(directive)
    except MissionError as exc:
        if log:
            log(f"[fixer_stages] mission parse failed: {exc}")
        if log:
            log(f"[fixer_stages] mission parse failed - aborting before GPU spend")
        report = {
            "final_state": "mission_parse_failed",
            "stop_reason": "mission_parse_failed",
            "error": str(exc),
        }
        path = write_mission_report(task_id, _empty_mission(), report)
        return StagedOutcome(
            stop_reason="mission_parse_failed",
            final_state="mission_parse_failed",
            report_path=path,
        )

    # Persona load (fail-loud before any GPU spend).
    try:
        reader_persona = load_stage_persona("decks/fixer/reader.yaml")
        aimer_persona = load_stage_persona("decks/fixer/aimer.yaml")
    except MissionError as exc:
        if log:
            log(f"[fixer_stages] stage persona card invalid - aborting before GPU spend: {exc}")
        report = {
            "final_state": "persona_card_invalid",
            "stop_reason": "persona_card_invalid",
            "error": str(exc),
        }
        path = write_mission_report(task_id, _empty_mission(), report)
        return StagedOutcome(
            stop_reason="persona_card_invalid",
            final_state="persona_card_invalid",
            report_path=path,
        )
    except Exception as exc:
        # A card escape (resolve_under_cards_root raises on a symlink or
        # absolute-path escape) or a loader crash is the same fail-loud
        # class: abort before any GPU spend.
        if log:
            log(f"[fixer_stages] stage persona load failed - aborting before GPU spend: {exc}")
        report = {
            "final_state": "persona_load_failed",
            "stop_reason": "persona_load_failed",
            "error": str(exc),
        }
        path = write_mission_report(task_id, _empty_mission(), report)
        return StagedOutcome(
            stop_reason="persona_load_failed",
            final_state="persona_load_failed",
            report_path=path,
        )

    mission_start = time.monotonic()
    stage_transcripts: list[dict] = []
    stages_report: list[dict] = []
    map_artifact: dict | None = None
    aim_artifact: dict | None = None
    rejections = 0
    rejection_reasons: list[str] = []
    outcome = StagedOutcome()

    # The deadline check (a hard gate evaluated BEFORE each stage
    # dispatch).
    def _deadline_exceeded() -> bool:
        elapsed = time.monotonic() - mission_start
        return elapsed + STAGE_TIMEOUT_S > MISSION_DEADLINE_S

    # ------------------------------------------------------------------
    # READER (skipped when pre_aimed - the recipe IS the map).
    # ------------------------------------------------------------------
    if not mission.pre_aimed:
        if _deadline_exceeded():
            return _deadline_stop(task_id, mission, stages_report,
                                  stage_transcripts, outcome)
        try:
            fixer, transcript, ledger = run_stage(
                stage="reader", mission=mission, cwd=cwd,
                persona=reader_persona, log=log,
            )
        except Exception as exc:
            return _stage_failure(task_id, mission, "reader", str(exc),
                                  stages_report, stage_transcripts, outcome)
        stage_transcripts.append({"stage": "reader", "transcript": transcript})
        artifact, reasoning = parse_stage_artifact(
            fixer.get("result_text", ""), "reader"
        )
        if artifact is None:
            return _stage_failure(
                task_id, mission, "reader",
                "no parseable map artifact",
                stages_report, stage_transcripts, outcome,
                partial_artifact=fixer.get("result_text", ""),
                ledger=ledger,
            )
        map_artifact = artifact
        stages_report.append(_stage_report_entry(
            "reader", fixer, ledger, artifact, reasoning,
        ))

    # ------------------------------------------------------------------
    # AIMER (confirm | reject_keep_looking; budget: 2 rejections).
    # ------------------------------------------------------------------
    aim_entries: list[dict] = []
    if mission.pre_aimed:
        # The recipe IS the aim entries (the Aimer confirms the PM-proven
        # recipe).
        aim_entries = [
            {
                "file": e.get("file", ""),
                "old_string": e.get("old", ""),
                "new_string": e.get("new", ""),
                "evidence": "pre-aimed recipe entry",
            }
            for e in mission.recipe
        ]
    else:
        aim_entries = (map_artifact or {}).get("entries") or []

    while True:
        re_hunt = rejection_reasons if rejection_reasons else None
        if _deadline_exceeded():
            return _deadline_stop(task_id, mission, stages_report,
                                  stage_transcripts, outcome)
        try:
            fixer, transcript, ledger = run_stage(
                stage="aimer", mission=mission, cwd=cwd,
                persona=aimer_persona, re_hunt_reasons=re_hunt, log=log,
            )
        except Exception as exc:
            return _stage_failure(task_id, mission, "aimer", str(exc),
                                  stages_report, stage_transcripts, outcome)
        stage_transcripts.append({"stage": "aimer", "transcript": transcript})
        artifact, reasoning = parse_stage_artifact(
            fixer.get("result_text", ""), "aimer"
        )
        if artifact is None:
            return _stage_failure(
                task_id, mission, "aimer",
                "no parseable aim artifact",
                stages_report, stage_transcripts, outcome,
                partial_artifact=fixer.get("result_text", ""),
                ledger=ledger,
            )
        aim_artifact = artifact
        stages_report.append(_stage_report_entry(
            "aimer", fixer, ledger, artifact, reasoning,
        ))
        verdict = artifact.get("verdict", "")
        if verdict == "confirm":
            aim_entries = artifact.get("aim") or aim_entries
            break
        # reject_keep_looking.
        rejections += 1
        rejection_reasons = artifact.get("reasons") or []
        if mission.pre_aimed:
            # Pre-aimed: an Aimer rejection is TERMINAL immediately - there
            # is no Reader to re-hunt and the recipe is PM-proven.
            outcome.pre_aimed_match_diagnostic = pre_aimed_match_diagnostic(
                mission, cwd, aim_entries=aim_entries,
                aimer_stated_reason="; ".join(rejection_reasons),
            )
            return _terminal_escalation(
                task_id, mission, stages_report, stage_transcripts,
                outcome, stop_reason="staged_aimer_rejected_pre_aimed",
                final_state="pre_aimed_rejection_terminal",
            )
        if rejections >= KEEP_LOOK_REJECTIONS:
            # The third rejection is terminal.
            outcome.pre_aimed_match_diagnostic = (
                pre_aimed_match_diagnostic(
                    mission, cwd, aim_entries=aim_entries,
                    aimer_stated_reason="; ".join(rejection_reasons),
                )
                if mission.pre_aimed else None
            )
            return _terminal_escalation(
                task_id, mission, stages_report, stage_transcripts,
                outcome, stop_reason="staged_aim_rejected",
                final_state="keep_look_exhausted",
            )
        # Re-hunt: 1 Reader re-hunt (the rejection reasons ride along) +
        # 1 Aimer re-aim (the loop continues).
        if _deadline_exceeded():
            return _deadline_stop(task_id, mission, stages_report,
                                  stage_transcripts, outcome)
        try:
            fixer, transcript, ledger = run_stage(
                stage="reader", mission=mission, cwd=cwd,
                persona=reader_persona, re_hunt_reasons=rejection_reasons,
                log=log,
            )
        except Exception as exc:
            return _stage_failure(task_id, mission, "reader", str(exc),
                                  stages_report, stage_transcripts, outcome)
        stage_transcripts.append({"stage": "reader", "transcript": transcript})
        artifact, reasoning = parse_stage_artifact(
            fixer.get("result_text", ""), "reader"
        )
        if artifact is None:
            return _stage_failure(
                task_id, mission, "reader",
                "no parseable map artifact (re-hunt)",
                stages_report, stage_transcripts, outcome,
                partial_artifact=fixer.get("result_text", ""),
                ledger=ledger,
            )
        map_artifact = artifact
        stages_report.append(_stage_report_entry(
            "reader", fixer, ledger, artifact, reasoning,
        ))
        aim_entries = (map_artifact or {}).get("entries") or []

    # ------------------------------------------------------------------
    # FIRE (deterministic).
    # ------------------------------------------------------------------
    all_applied, fire_results = fire(mission, aim_artifact or {"aim": aim_entries}, cwd)
    outcome.all_applied = all_applied
    outcome.fire_results = fire_results
    outcome.aim = aim_artifact

    if not all_applied:
        # Partial apply: skip the gate -> worktree salvage directly.
        if mission.pre_aimed:
            outcome.pre_aimed_match_diagnostic = pre_aimed_match_diagnostic(
                mission, cwd, aim_entries=aim_entries,
                aimer_stated_reason="fire-time AimError",
            )
        outcome.stop_reason = "staged_aim_failed"
        outcome.final_state = "aim_failed"
        path = _write_report(
            task_id, mission, stages_report, stage_transcripts, outcome,
        )
        outcome.report_path = path
        return outcome

    # ------------------------------------------------------------------
    # Gate (the staged tail calls _gate_targeted_rerun directly with the
    # mission's tests_timeout_s).
    # ------------------------------------------------------------------
    if gate_rerun is not None:
        touched = set(mission.tests)
        gate_outcome = gate_rerun(cwd, touched, mission.tests_timeout_s)
        outcome.gate_outcome = gate_outcome
        outcome.gate_passed = (
            gate_outcome is not None
            and gate_outcome.get("returncode") == 0
        )

    if not outcome.gate_passed:
        # Gate failed or unusable (None): worktree salvage.
        outcome.stop_reason = (
            "staged_gate_unusable" if outcome.gate_outcome is None
            else "concluded_gate_rejected"
        )
        outcome.final_state = "gate_rejected"
        path = _write_report(
            task_id, mission, stages_report, stage_transcripts, outcome,
        )
        outcome.report_path = path
        return outcome

    # ------------------------------------------------------------------
    # Tail (the shared helper - push / PR / salvage).
    #
    # S2 seam: `tail_finalize` is the shaped_runner.tail_finalize full-seam
    # helper (S6) - its signature carries EVERY value the legacy tail
    # region consumes (task_id / target_id / bare_repo / gate_rerun_fired
    # named explicitly). The staged runner (_run_local_fixer_staged)
    # passes the legacy-shaped values; this keyword set is the S1-side
    # contract.
    # ------------------------------------------------------------------
    if tail_finalize is not None:
        outcome.pr_url = tail_finalize(
            task_id=task_id,
            target_id=target_id,
            bare_repo=spec.get("bare_repo", ""),
            branch=spec.get("branch", ""),
            slug=spec.get("slug", "local"),
            cwd=cwd,
            worktree_path=spec.get("worktree_path"),
            final_diff=_git_diff(cwd),
            concluded=True,
            last_test_outcome=outcome.gate_outcome,
            max_steps_hit=False,
            no_progress_hit=False,
            stop_reason="",
            step_count=sum(
                len(s.get("transcript") or [])
                for s in stage_transcripts
            ),
            transcript_path=write_staged_transcript(
                task_id, stage_transcripts
            ),
            gate_passed=True,
            gate_bypassed=None,
            model_touched_tests=set(mission.tests),
            gate_rerun_fired=True,
            _wip_git=None,
        )
    outcome.final_state = "completed"
    path = _write_report(
        task_id, mission, stages_report, stage_transcripts, outcome,
    )
    outcome.report_path = path
    return outcome


def _git_diff(cwd: str) -> str:
    """The staged diff (uncommitted worktree state vs HEAD) for the tail.

    Read-only: NO `git add` - the tail's own `git add -A` stages the
    worktree (the same doctrine as the legacy tail, which derives
    final_diff from the uncommitted state).
    """
    import subprocess

    try:
        r = subprocess.run(
            ["git", "-C", cwd, "diff", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _empty_mission() -> Mission:
    """A degenerate mission for the parse-failure report (the report
    writer needs a Mission; the parse failed so the fields are empty)."""
    return Mission(
        pre_aimed=False, scope_files=[], tests=[], tests_timeout_s=0,
    )


def _stage_report_entry(
    stage: str, fixer: dict, ledger: StageLedger,
    artifact: dict, reasoning: str,
) -> dict:
    """One stage's report entry (steps, budget consumed, stop_reason,
    redirect ledger, reasoning, artifact, leads)."""
    leads = artifact.get("leads") if isinstance(artifact, dict) else None
    return {
        "stage": stage,
        "steps": len(fixer.get("steps") or []),
        "stop_reason": fixer.get("stop_reason", ""),
        "concluded": fixer.get("concluded", False),
        "budget_forced": fixer.get("budget_forced", False),
        "redirects": ledger.redirects,
        "interventions_used": ledger.interventions_used,
        "scope_rejects": ledger.scope_rejects,
        "reasoning": reasoning,
        "artifact": artifact,
        "leads": leads,
    }


def _write_report(
    task_id: str, mission: Mission, stages_report: list[dict],
    stage_transcripts: list[dict], outcome: StagedOutcome,
) -> Path:
    """Write the mission report + the consolidated staged transcript."""
    report = {
        "stages": stages_report,
        "aim": outcome.aim,
        "fire_results": outcome.fire_results,
        "pre_aimed_match_diagnostic": outcome.pre_aimed_match_diagnostic,
        "gate_outcome": outcome.gate_outcome,
        "final_state": outcome.final_state,
        "stop_reason": outcome.stop_reason,
    }
    path = write_mission_report(task_id, mission, report)
    write_staged_transcript(task_id, stage_transcripts)
    return path


def _stage_failure(
    task_id: str, mission: Mission, stage: str, reason: str,
    stages_report: list[dict], stage_transcripts: list[dict],
    outcome: StagedOutcome, partial_artifact: str = "",
    ledger: StageLedger | None = None,
) -> StagedOutcome:
    """The stage-failure partition: a stage ending without a parseable
    artifact is terminal (no re-hunt - the failure mode is the model's
    output, not the search); the report records partial_artifact.
    """
    if ledger is not None:
        stages_report.append({
            "stage": stage,
            "steps": 0,
            "stop_reason": "stage_failure",
            "redirects": ledger.redirects,
            "interventions_used": ledger.interventions_used,
            "reasoning": reason,
            "artifact": None,
            "leads": None,
        })
    else:
        stages_report.append({
            "stage": stage,
            "steps": 0,
            "stop_reason": "stage_failure",
            "reasoning": reason,
            "artifact": None,
            "leads": None,
        })
    outcome.stop_reason = f"staged_{stage}_failed"
    outcome.final_state = "stage_failure"
    # The report records the raw partial_artifact (a YAML error cannot
    # name the offending field, so the raw dump is the diagnostic).
    if partial_artifact:
        stages_report[-1]["partial_artifact"] = partial_artifact
    path = _write_report(task_id, mission, stages_report, stage_transcripts,
                          outcome)
    outcome.report_path = path
    return outcome


def _deadline_stop(
    task_id: str, mission: Mission, stages_report: list[dict],
    stage_transcripts: list[dict], outcome: StagedOutcome,
) -> StagedOutcome:
    """A deadline stop (the keep-look worst case): the stage phase is
    deadline-bounded; the deadline preempts the keep-look logical budget.
    A deadline stop escalates with the report (redirect ledger + partial
    state) regardless of remaining rejections.
    """
    outcome.stop_reason = "staged_deadline"
    outcome.final_state = "deadline_stop"
    path = _write_report(task_id, mission, stages_report, stage_transcripts,
                          outcome)
    outcome.report_path = path
    return outcome


def _terminal_escalation(
    task_id: str, mission: Mission, stages_report: list[dict],
    stage_transcripts: list[dict], outcome: StagedOutcome,
    stop_reason: str, final_state: str,
) -> StagedOutcome:
    """A terminal escalation (no fire, no gate; tail = report write +
    return "").
    """
    outcome.stop_reason = stop_reason
    outcome.final_state = final_state
    path = _write_report(task_id, mission, stages_report, stage_transcripts,
                          outcome)
    outcome.report_path = path
    return outcome
