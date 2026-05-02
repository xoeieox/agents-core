"""Expert dispatch substrate — layered Expert primitive.

Implements the dispatch substrate that turns Experts into persistent
dispatchable entities. Coordinators (Router) call dispatch_expert() to
summon an Expert to weigh in on questions, perform domain work, and
accumulate experience over time via structured post-mortem records.

spec: expert-substrate-v0
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents_core.claude_queue import CLAUDE_QUEUE_DIR, ClaudeQueue
from agents_core.expert_layout import (  # noqa: F401 — re-exported for consumers
    EXPERTS_ROOT,
    dispatches_root,
    expert_root,
    post_mortems_root,
    seeds_root,
)
from agents_core.llm import call_claude_cli

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPERT_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_VALID_LAYERS = frozenset({"persona", "corpus", "memory"})

RECORD_KINDS = frozenset({
    "concept_encountered",
    "pattern_observed",
    "principle_tested",
    "surprise",
    "open_question",
    "decision_made",
})

_VALID_CONFIDENCES = frozenset({"low", "medium", "high"})

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "expert_post_mortem.txt"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertDispatchInput:
    """Input to dispatch_expert(). All fields are validated on dispatch."""
    expert_id: str
    intent: str
    layers: tuple[str, ...]
    model: str = "sonnet"
    requested_by: str = "unknown"
    timeout_s: int = 600


@dataclass(frozen=True)
class ExpertDispatchResult:
    """Result returned by dispatch_expert()."""
    task_id: str
    outcome: str          # completed | abandoned | post_mortem_failed
    intent_path: Path
    notepad_path: Path    # may not exist if dispatch crashed before write
    output_path: Path | None  # None if dispatch crashed before output.md
    post_mortem_path: Path    # .yaml (success) or .yaml.failed (failure)
    error: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_task_id(expert_id: str) -> str:
    """Generate expert_<YYYYMMDD>_<HHMMSS>_<usec>_<expert-id> (UTC)."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    usec = now.strftime("%f")[:4]
    return f"expert_{ts}_{usec}_{expert_id}"


def _compose_system_prompt(
    expert_id: str,
    layers: tuple[str, ...],
    experts_root: Path,
) -> str:
    """Compose the system prompt from persona + optional corpus + optional memory.

    Invariant: if a layer is in ``layers`` but its source dir is empty/absent,
    the section is rendered with ``(none yet)`` — never omitted. This is what
    makes the A/B embodied-baseline experiment structurally clean.
    """
    persona_path = experts_root / expert_id / "persona.md"
    parts = [persona_path.read_text()]

    if "corpus" in layers:
        parts.append("\n## Your accumulated corpus\n")
        seeds_dir = experts_root / expert_id / "seeds"
        if seeds_dir.exists():
            seed_files = sorted(seeds_dir.glob("*.yaml"))
            if seed_files:
                parts.append("\n".join(f.read_text() for f in seed_files))
            else:
                parts.append("(none yet)")
        else:
            parts.append("(none yet)")

    if "memory" in layers:
        parts.append("\n## What you have learned from past dispatches\n")
        pm_dir = experts_root / expert_id / "post-mortems"
        if pm_dir.exists():
            # glob "*.yaml*" to also catch ".yaml.failed"; filter keeps only plain ".yaml"
            pm_files = [
                f for f in sorted(pm_dir.glob("*.yaml*"))
                if not f.name.endswith(".yaml.failed")
            ]
            if pm_files:
                parts.append("\n".join(f.read_text() for f in pm_files))
            else:
                parts.append("(none yet)")
        else:
            parts.append("(none yet)")

    # Field-notebook instruction — always appended last (verbatim from spec)
    parts.append(
        "\nYou have a field notebook at `./notepad.md` in your current working "
        "directory. Write to it freely as you work \u2014 observations, partial "
        "thoughts, dead-ends, surprises, questions. Nothing here is shown to "
        "anyone immediately; the post-mortem step will read it after you finish "
        "to extract structured records. You do not need to be tidy in this file. "
        "When you are done, write your final response to the dispatch intent in "
        "`./output.md`."
    )

    return "\n".join(parts)


def _poll_for_output(
    output_path: Path,
    spec_json_path: Path,
    deadline: float,
    poll_interval: float = 5.0,
) -> bool:
    """Return True if output.md appears with content before deadline.

    Two completion signals:
    - output.md has content → dispatch succeeded (return True immediately)
    - spec JSON deleted (shaped_runner finished) → dispatch ran; check output
    """
    while time.monotonic() < deadline:
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        if not spec_json_path.exists():
            # shaped_runner ran and deleted the spec — give a brief grace period
            # in case output.md is still being flushed to disk
            time.sleep(min(2.0, poll_interval))
            return output_path.exists() and output_path.stat().st_size > 0
        time.sleep(poll_interval)
    # Deadline reached — check one last time
    return output_path.exists() and output_path.stat().st_size > 0


def _run_post_mortem(
    expert_id: str,
    task_id: str,
    intent: str,
    layers_loaded: list[str],
    dispatch_cwd: Path,
    dispatched_at: str,
    dispatched_by: str,
    duration_seconds: int,
    dispatch_outcome: str,
    experts_root: Path,
) -> tuple[str, Path]:
    """Synchronous post-mortem LLM call. Returns (final_outcome, path).

    Reads notepad.md + output.md, calls Sonnet directly (not via ClaudeQueue),
    validates the YAML schema, and writes post-mortems/<task_id>.yaml.

    On any failure: writes .yaml.failed and returns ("post_mortem_failed", path).
    When both notepad and output are absent (crash before any write): writes
    records:[] without an LLM call.
    """
    pm_dir = experts_root / expert_id / "post-mortems"
    pm_dir.mkdir(parents=True, exist_ok=True)

    notepad_path = dispatch_cwd / "notepad.md"
    output_path = dispatch_cwd / "output.md"

    notepad_content = notepad_path.read_text() if notepad_path.exists() else "(no notepad)"
    output_content = output_path.read_text() if output_path.exists() else "(no output)"

    pm_yaml_path = pm_dir / f"{task_id}.yaml"
    pm_failed_path = pm_dir / f"{task_id}.yaml.failed"

    dispatch_info: dict = {
        "task_id": task_id,
        "expert_id": expert_id,
        "intent": intent,
        "dispatched_by": dispatched_by,
        "dispatched_at": dispatched_at,
        "duration_seconds": duration_seconds,
        "outcome": dispatch_outcome,
        "layers_loaded": layers_loaded,
        "notepad_path": f"dispatches/{task_id}/notepad.md",
    }
    # output_path only included when the file actually exists (omitted on abandoned)
    if output_path.exists() and output_path.stat().st_size > 0:
        dispatch_info["output_path"] = f"dispatches/{task_id}/output.md"
    dispatch_block: dict = {"dispatch": dispatch_info}

    # Short-circuit: if nothing was written, skip LLM call
    if notepad_content == "(no notepad)" and output_content == "(no output)":
        full_data = {**dispatch_block, "records": []}
        pm_yaml_path.write_text(
            yaml.dump(full_data, default_flow_style=False, allow_unicode=True)
        )
        return dispatch_outcome, pm_yaml_path

    try:
        template = _TEMPLATE_PATH.read_text()
        prompt = (
            template
            .replace("{{expert_id}}", expert_id)
            .replace("{{intent}}", intent)
            .replace("{{layers_loaded}}", ", ".join(layers_loaded))
            .replace("{{notepad_content}}", notepad_content)
            .replace("{{output_content}}", output_content)
        )

        records_raw = call_claude_cli(
            prompt=prompt,
            model="sonnet",
            timeout=120,
        )
        if records_raw is None:
            raise RuntimeError("post-mortem LLM call returned None (timeout or error)")

        # Strip markdown fences that some LLM outputs wrap around YAML
        stripped = records_raw.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        records_data = yaml.safe_load(stripped)
        if not isinstance(records_data, dict) or "records" not in records_data:
            raise ValueError(
                f"post-mortem output missing 'records' key; raw: {records_raw[:300]!r}"
            )

        records = records_data.get("records") or []
        if not isinstance(records, list):
            raise ValueError(f"'records' field must be a list, got {type(records).__name__}")

        for record in records:
            kind = record.get("kind")
            if kind not in RECORD_KINDS:
                raise ValueError(
                    f"Unknown record kind {kind!r}; valid: {sorted(RECORD_KINDS)}"
                )
            if not record.get("essence", "").strip():
                raise ValueError(f"Record with kind {kind!r} has empty 'essence'")
            if not record.get("context", "").strip():
                raise ValueError(f"Record with kind {kind!r} has empty 'context'")
            confidence = record.get("confidence")
            if confidence not in _VALID_CONFIDENCES:
                raise ValueError(
                    f"Record with kind {kind!r} has invalid confidence {confidence!r};"
                    f" must be one of {sorted(_VALID_CONFIDENCES)}"
                )
            if not isinstance(record.get("domain_tags"), list):
                raise ValueError(
                    f"Record with kind {kind!r} has missing or non-list 'domain_tags'"
                )

        full_data = {**dispatch_block, "records": records}
        pm_yaml_path.write_text(
            yaml.dump(full_data, default_flow_style=False, allow_unicode=True)
        )
        return dispatch_outcome, pm_yaml_path

    except Exception as exc:
        # Post-mortem failed — write .yaml.failed with error info
        failed_block = dict(dispatch_block)
        if dispatch_outcome == "completed":
            failed_block["dispatch"] = dict(dispatch_block["dispatch"])
            failed_block["dispatch"]["outcome"] = "post_mortem_failed"
        failed_block["error"] = str(exc)
        failed_block["error_at"] = datetime.now(timezone.utc).isoformat()
        pm_failed_path.write_text(
            yaml.dump(failed_block, default_flow_style=False, allow_unicode=True)
        )
        return "post_mortem_failed", pm_failed_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dispatch_expert(
    inp: ExpertDispatchInput,
    *,
    _experts_root: Path | None = None,
    _queue_dir: Path | None = None,
    _poll_interval: float = 5.0,
) -> ExpertDispatchResult:
    """Dispatch an Expert and block until dispatch + post-mortem complete.

    Args:
        inp: Validated dispatch input.
        _experts_root: Override /srv/lapis/experts/ root (tests only).
        _queue_dir: Override /srv/lapis/claude-queue/ root (tests only).
        _poll_interval: Polling interval in seconds (tests set this low).

    Returns:
        ExpertDispatchResult with outcome, paths, and optional error.

    Raises:
        ValueError: if expert_id is invalid, 'persona' not in layers, or
            layers contains unknown values, or persona.md is missing.
    """
    # -----------------------------------------------------------------------
    # Step 1: Validate input
    # -----------------------------------------------------------------------
    if not _EXPERT_ID_RE.match(inp.expert_id):
        raise ValueError(
            f"expert_id {inp.expert_id!r} must match ^[a-z][a-z0-9-]*$"
        )
    if "persona" not in inp.layers:
        raise ValueError("'persona' must be in layers — it is always required")
    unknown_layers = set(inp.layers) - _VALID_LAYERS
    if unknown_layers:
        raise ValueError(f"Unknown layer(s): {unknown_layers!r}")

    experts_root = _experts_root or EXPERTS_ROOT
    queue_dir = _queue_dir or CLAUDE_QUEUE_DIR

    persona_path = experts_root / inp.expert_id / "persona.md"
    if not persona_path.exists():
        raise ValueError(
            f"persona.md not found for expert {inp.expert_id!r} at {persona_path}"
        )

    # -----------------------------------------------------------------------
    # Step 2: Generate task_id
    # -----------------------------------------------------------------------
    task_id = _generate_task_id(inp.expert_id)

    # -----------------------------------------------------------------------
    # Step 3: Materialize dispatch CWD + post-mortems dir
    # -----------------------------------------------------------------------
    dispatch_cwd = experts_root / inp.expert_id / "dispatches" / task_id
    dispatch_cwd.mkdir(parents=True, exist_ok=True)
    pm_dir = experts_root / inp.expert_id / "post-mortems"
    pm_dir.mkdir(parents=True, exist_ok=True)

    dispatched_at = datetime.now(timezone.utc).isoformat()
    start_monotonic = time.monotonic()

    # Write intent.yaml (dispatch input + metadata)
    intent_path = dispatch_cwd / "intent.yaml"
    intent_data = {
        "task_id": task_id,
        "expert_id": inp.expert_id,
        "intent": inp.intent,
        "layers": list(inp.layers),
        "model": inp.model,
        "requested_by": inp.requested_by,
        "timeout_s": inp.timeout_s,
        "dispatched_at": dispatched_at,
    }
    intent_path.write_text(
        yaml.dump(intent_data, default_flow_style=False, allow_unicode=True)
    )

    # -----------------------------------------------------------------------
    # Step 4: Compose system prompt
    # -----------------------------------------------------------------------
    system_prompt = _compose_system_prompt(inp.expert_id, inp.layers, experts_root)

    # -----------------------------------------------------------------------
    # Step 5: Write shaped_runner-compatible spec JSON to pending/
    #         and submit YAML task to ClaudeQueue
    # -----------------------------------------------------------------------
    pending_dir = queue_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    spec_json_path = pending_dir / f"{task_id}.json"
    spec = {
        "task_id": task_id,
        "task_type": "subprocess",
        "prompt": inp.intent,
        "system": system_prompt,
        "model": inp.model,
        "timeout_s": inp.timeout_s,
        "json_mode": False,
        "cwd": str(dispatch_cwd),
        "permission_mode": "bypassPermissions",
        "worktree_required": False,
        "capture_meta": False,
    }
    spec_json_path.write_text(json.dumps(spec, ensure_ascii=False))

    q = ClaudeQueue(queue_dir)
    q.submit(
        {
            "task_type": "subprocess",
            "priority": 50,  # Priority.NORMAL
            "timeout_seconds": inp.timeout_s + 60,
            "submitted_by": inp.requested_by,
            "model": inp.model,
            "description": f"expert:{inp.expert_id}",
            "notify": False,
            "payload": {"spec_path": str(spec_json_path)},
        },
        task_id=task_id,
    )

    # -----------------------------------------------------------------------
    # Step 6: Block on completion via polling
    # -----------------------------------------------------------------------
    notepad_path = dispatch_cwd / "notepad.md"
    output_file = dispatch_cwd / "output.md"
    poll_deadline = time.monotonic() + inp.timeout_s + 60

    dispatch_succeeded = _poll_for_output(
        output_file, spec_json_path, poll_deadline, _poll_interval
    )

    duration_seconds = int(time.monotonic() - start_monotonic)

    # -----------------------------------------------------------------------
    # Step 7: Run post-mortem step (synchronous; always runs)
    # -----------------------------------------------------------------------
    dispatch_outcome = "completed" if dispatch_succeeded else "abandoned"
    dispatch_error = (
        None if dispatch_succeeded
        else f"Expert dispatch abandoned: no output.md after {inp.timeout_s + 60}s"
    )

    final_outcome, pm_path = _run_post_mortem(
        expert_id=inp.expert_id,
        task_id=task_id,
        intent=inp.intent,
        layers_loaded=list(inp.layers),
        dispatch_cwd=dispatch_cwd,
        dispatched_at=dispatched_at,
        dispatched_by=inp.requested_by,
        duration_seconds=duration_seconds,
        dispatch_outcome=dispatch_outcome,
        experts_root=experts_root,
    )

    error = dispatch_error
    if final_outcome == "post_mortem_failed" and dispatch_outcome == "completed":
        error = f"Post-mortem failed; see {pm_path}"

    # -----------------------------------------------------------------------
    # Step 8: Return result
    # -----------------------------------------------------------------------
    return ExpertDispatchResult(
        task_id=task_id,
        outcome=final_outcome,
        intent_path=intent_path,
        notepad_path=notepad_path,
        output_path=output_file if output_file.exists() else None,
        post_mortem_path=pm_path,
        error=error,
    )
