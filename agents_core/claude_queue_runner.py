#!/usr/bin/env python3
"""ClaudeQueue runner daemon — bounded-concurrency executor for shaped-agent
`claude -p` subprocesses.

Architecture differs from `gpu_queue_runner.py`:
- asyncio event loop + `asyncio.Semaphore(CLAUDE_QUEUE_WORKERS)` for
  bounded concurrency (Claude subprocesses are I/O-bound on the API).
- The GPU runner uses threading + a single-worker claim loop because Qwen
  inference is GPU-bound and can only run one task at a time.

Config env vars:
- CLAUDE_QUEUE_WORKERS: max concurrent subprocesses (default 2)
- FIXER_MAX_CONCURRENT: max concurrent fixer/fixer_retry tasks (default 5);
  subset cap - restricts fixers below CLAUDE_QUEUE_WORKERS, never past it
- CLAUDE_QUEUE_ENABLED: "0" makes the daemon exit cleanly on next tick

The runner is responsible for capturing `shaped_runner`'s stdout/stderr and
writing the output file under `/srv/lapis/claude-queue/completed/`. `shaped_runner`
prints the result and exits; the file-write lives here, not there, mirroring
`/srv/agents/scripts/gpu_queue_runner.py:execute_subprocess_task`.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import psutil

from agents_core.claude_queue import (
    CLAUDE_QUEUE_DIR,
    ClaudeQueue,
    _acquire_claim_lease as _cq_acquire_claim_lease,
    _release_claim_lease as _cq_release_claim_lease,
    _task_backend_url as _cq_task_backend_url,
)
from agents_core.gpu import PACIFIC, Priority as QueuePriority  # noqa: F401
from agents_core.notify import Priority as PushoverPriority, _capture_event, send_notification
from agents_core.room_paths import room_path
from agents_core.worktree import WORKTREE_ROOT

RUNNER_SCRIPT_MODULE = "agents_core.shaped_runner"
OUTPUT_DIR = CLAUDE_QUEUE_DIR / "completed"

CLONE_ROOTS_GLOB = "/srv/git/*-working"

POLL_INTERVAL_S = 2.0
STARTUP_STALE_GRACE_S = 300

# Distinct exit code so journalctl/systemctl show -p ExecMainStatus can tell
# "claim loop crashed" apart from a generic uncaught startup error.
_CLAIM_LOOP_CRASH_EXIT_CODE = 3

# Council concurrency control — module-level, NOT on Daemon (see docstring).
# asyncio.Semaphore is safe to create at module level in Python 3.10+.
# Constructed below (after `log` and COUNCIL_MAX_CONCURRENT are defined) —
# see _COUNCIL_SEM assignment further down this module.

_COUNCIL_DIR = room_path("council")
_COUNCIL_LOG_DIR = room_path("council.logs")
_COUNCIL_ORPHAN_AGE_SECS = int(os.environ.get("COUNCIL_ORPHAN_AGE_SECS", "3600"))

SILENCED_LOG = room_path("notify_audit.silenced")

# Terminal status sets per mode.
_DELIBERATION_TERMINAL = frozenset({"resolved", "open", "laid-down"})
_SCENE_TERMINAL = frozenset({"closed"})

log = logging.getLogger("claude-queue-runner")


def _parse_concurrency_cap(env_var: str, default: int) -> int:
    """Parse a positive-int concurrency cap from an env var.

    Invalid or <1 values fall back to 1 and log a WARNING (never crash the
    daemon on a bad env var). Missing env var uses `default` silently.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value < 1:
            raise ValueError(f"{env_var}={raw!r} must be >= 1")
        return value
    except ValueError:
        log.warning(
            f"{env_var}={raw!r} is invalid — falling back to 1"
        )
        return 1


COUNCIL_MAX_CONCURRENT = _parse_concurrency_cap("COUNCIL_MAX_CONCURRENT", 2)

_COUNCIL_SEM = asyncio.Semaphore(COUNCIL_MAX_CONCURRENT)
"""Cap: at most COUNCIL_MAX_CONCURRENT council subprocesses at a time
(env COUNCIL_MAX_CONCURRENT, default 2; invalid/<1 falls back to 1 with a
WARNING log — see _parse_concurrency_cap).

NOT placed on Daemon because the existing dispatch pattern is
  Daemon._worker → module-level _run_task(queue, task)
with no Daemon handle threaded through.  Putting it on Daemon would require
reworking every call site.

Acquisition order in Daemon._worker is load-bearing: _COUNCIL_SEM is
acquired BEFORE self.sem.  Reversed order causes the second queued council
task to idle-hold a worker slot while waiting — dropping fixer/reviewer
throughput to zero.  With the outer-first ordering, the second council task
blocks without holding a worker slot.
"""

FIXER_MAX_CONCURRENT = _parse_concurrency_cap("FIXER_MAX_CONCURRENT", 5)
_FIXER_SEM = asyncio.Semaphore(FIXER_MAX_CONCURRENT)
"""Cap: at most FIXER_MAX_CONCURRENT fixer/fixer_retry tasks at a time
(env FIXER_MAX_CONCURRENT, default 5; invalid/<1 falls back to 1 with a
WARNING log - see _parse_concurrency_cap). Subset cap: it can RESTRICT
fixers below CLAUDE_QUEUE_WORKERS but never expand them past the global
worker cap (same semantics as the council sub-cap). Acquisition order in
Daemon._worker is load-bearing: _FIXER_SEM is acquired BEFORE self.sem,
so a claimed fixer task waiting on the sub-cap holds no global slot.
Default 5 matches the live unit's CLAUDE_QUEUE_WORKERS=5 (raised 2026-09-04),
so unset env is behavior-neutral. Fixer-family tasks are identified by the
shaper description prefix (shaper.py:395, f"{agent.name}:{target_id}"):
"fixer:" and "fixer_retry:"."""


def _is_fixer_task(task: dict) -> bool:
    """Fixer admission predicate.

    Shaped tasks are the generic task_type "subprocess"; the agent family
    rides in description (shaper.py:395 writes f"{agent.name}:{target_id}").
    council.run is EXCLUDED at the predicate level: council descriptions are
    free-form operator decision text (council/cli.py:1986-1993) and can legally
    start with "fixer:" (e.g. "fixer: merge or hold?"). The exclusion makes the
    one-directional-interference invariant hold at the predicate level, not only
    by the _worker branch order.
    If the shaper's description format ever changes, THIS PREDICATE MUST
    CHANGE WITH IT (free-form-coupling risk, see Out of scope / known-deferred).
    """
    if task.get("task_type") == "council.run":
        return False
    desc = str(task.get("description") or "")
    return desc.startswith("fixer:") or desc.startswith("fixer_retry:")


# ---------------------------------------------------------------------------
# Ops telemetry — operational primitive extraction (Lapis Ops Layer parity).
#
# Ported from /srv/agents/scripts/gpu_queue_runner.py:64-96. Gated by
# LAPIS_OPS_PRIMITIVES env var (default on); failures are swallowed because
# missing weather data is acceptable at the aggregate. Called on the success
# path only — match GPU's semantics.
# ---------------------------------------------------------------------------

def _extract_ops_primitives(task_id: str, task_type: str,
                            result_text: str | None,
                            output_path: str | None) -> None:
    """Post-completion hook: tag the task output with operational primitives
    and store to mem.db under weather/<date>/<task_id>."""
    if os.environ.get("LAPIS_OPS_PRIMITIVES", "1") == "0":
        return
    try:
        text = result_text or ""
        if (not text or len(text) < 200) and output_path:
            try:
                text = Path(output_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        if not text or len(text.strip()) < 80:
            return  # nothing worth tagging
        import ops_primitives
        prims = ops_primitives.extract_and_store(task_id, task_type, text)
        if prims:
            types = ",".join(p.get("type", "?") for p in prims[:3])
            log.info(f"ops-primitives [{task_type}] {task_id}: {types}")
    except Exception as e:
        log.warning(f"ops-primitives: skip for {task_id}: {e}")


# ---------------------------------------------------------------------------
# Failure classification and silenced-event logging
# ---------------------------------------------------------------------------

# shaped_runner._run_local_reviewer prints every non-verdict as
# "ERROR: local reviewer produced no verdict (reason=<X>)" (shaped_runner.py:558),
# where <X> is drawn from the closed vocabulary documented at gw_agent.py:1318-1332.
# This regex isolates that shape from every other "ERROR:"-prefixed first line so
# extraction and classification stay decoupled — a line that merely starts with
# "ERROR:" must never enter the reason table (agents-core-reviewer-failure-notify-class-v0).
_REVIEWER_NO_VERDICT_RE = re.compile(
    r"^ERROR: local reviewer produced no verdict \(reason=(\w+)\)"
)

# Reason -> class, per agents-core-reviewer-failure-notify-class-v0 C2.
# infra: a host is down or refusing — pages HIGH, a human can act on it.
# execution: model-quality / lifecycle outcomes — no host to fix, silence to audit.
# contention: transient GPU contention / per-step transport failures already
# retried internally (GW_STEP_MAX_RETRIES) — the normal cost of a shared GPU,
# must not page. Recorded distinctly from execution in the audit line (C2, C4).
_REVIEWER_REASON_CLASS = {
    "gw_unreachable": "infra",
    "gw_not_serving": "infra",
    "backend_unreachable": "infra",
    "grounding_failed": "execution",
    "max_steps_exhausted": "execution",
    "budget_exhausted": "execution",
    "no_choices": "execution",
    "interrupted": "execution",
    "gw_defer_timeout": "contention",
    "rate_limited": "contention",
    "request_timeout": "contention",
    "server_error": "contention",
    "request_failed": "contention",
}


def _reviewer_failure_reason(result: str) -> str | None:
    """Extract the reason= token from a reviewer non-verdict first line.

    Returns None when the first line does not match the reviewer non-verdict
    shape at all — that "does not match" case must fall through to today's
    prefix logic in _failure_class untouched (C3), never enter the reason
    table below.
    """
    lines = result.splitlines()
    first = lines[0] if lines else ""
    m = _REVIEWER_NO_VERDICT_RE.match(first)
    return m.group(1) if m else None


def _failure_class(result: str) -> str:
    """Classify a failure result string as 'infra', 'execution', or 'contention'.

    Inspects only the **first line** of the formatted result string
    (e.g. "EXIT 1:\n…", "ERROR: worktree_setup:\n…"), not raw subprocess output.
    This is distinct from _classify_runner_failure which scans all lines.

    Reviewer non-verdicts ("ERROR: local reviewer produced no verdict
    (reason=<X>)") classify by <X> via _REVIEWER_REASON_CLASS rather than by
    the uniform "ERROR:" prefix — see agents-core-reviewer-failure-notify-class-v0.
    An unrecognised reason fails closed to 'infra' (page once, loudly, rather
    than silently swallow an unanticipated outage class).

    Every other first line — fixer EXIT/TIMEOUT/INTERRUPTED, and any other
    "ERROR:"-prefixed string that isn't the reviewer non-verdict shape — keeps
    the original prefix-only contract byte-identical (use startswith, not
    substring 'in', to match prefixes).
    """
    lines = result.splitlines()
    first = lines[0] if lines else ""

    reason = _reviewer_failure_reason(result)
    if reason is not None:
        return _REVIEWER_REASON_CLASS.get(reason, "infra")  # unknown reason → fail closed

    if first.startswith("ERROR:"):
        return "infra"
    if first.startswith(("EXIT ", "TIMEOUT:", "INTERRUPTED ")):
        return "execution"
    return "infra"  # unknown/empty → conservative infra


def _log_silenced(event: str, task: dict, *, failure_class: str | None,
                  demoted_from: str, result: str,
                  reason: str | None = None) -> None:
    """Append one JSON line to SILENCED_LOG for a demoted notification.

    `reason` is the extracted reviewer non-verdict reason (e.g.
    "grounding_failed"), when applicable — so the audit line answers "what
    stopped paging and why" without re-reading the queue (C4).

    Best-effort: wrap the whole body in try/except so a logging failure
    never raises and never causes a push.
    """
    try:
        SILENCED_LOG.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(PACIFIC).isoformat()
        result_head = (result.splitlines()[0] if result else "")[:300]
        entry = {
            "ts": ts,
            "source": "claude_queue_runner",
            "task_id": task.get("id"),
            "description": _fmt_task_label(task),
            "event": event,
            "failure_class": failure_class,
            "demoted_from": demoted_from,
            "result_head": result_head,
            "reason": reason,
        }
        with open(SILENCED_LOG, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        _capture_event(
            source="claude_queue_runner",
            message=result_head,
            title="claude-queue",
            priority=PushoverPriority[demoted_from],
            delivered=None,
            extra={
                "task_id": task.get("id"),
                "description": _fmt_task_label(task),
                "event": event,
                "failure_class": failure_class,
                "demoted_from": demoted_from,
            },
        )
    except Exception as e:
        log.warning(f"failed to log silenced event: {e}")


# ---------------------------------------------------------------------------
# Notification helpers
#
# These are local to the runner. `agents_core.notify` exports
# `send_notification()` + a `Priority` enum — no `notify_completion` /
# `notify_failure` functions exist there. `gpu_queue_runner.py` follows the
# same pattern: thin wrappers that format task context and call
# send_notification().
# ---------------------------------------------------------------------------

def _fmt_task_label(task: dict) -> str:
    desc = task.get("description") or ""
    submitted_by = task.get("submitted_by", "unknown")
    return desc or f"{task.get('task_type','?')} (by {submitted_by})"


def notify_completion(task: dict, output_path: str) -> None:
    if not task.get("notify"):
        _capture_event(
            source="claude_queue_runner",
            message=f"Claude task completed: {_fmt_task_label(task)}\nOutput: {output_path}",
            title="claude-queue",
            priority=PushoverPriority.NORMAL,
            delivered=None,
            extra={"task_id": task.get("id"), "description": _fmt_task_label(task), "notify_flag": False},
        )
        return
    policy = task.get("notify_policy", "always")
    if policy == "infra-only":
        _log_silenced("completion", task, failure_class=None,
                     demoted_from="NORMAL", result="")
        return
    send_notification(
        message=f"Claude task completed: {_fmt_task_label(task)}\nOutput: {output_path}",
        title="claude-queue",
        priority=PushoverPriority.NORMAL,
    )


def notify_failure(task: dict, result: str) -> None:
    if not task.get("notify"):
        summary = result.splitlines()[0][:300] if result else "(no output)"
        _capture_event(
            source="claude_queue_runner",
            message=f"Claude task FAILED: {_fmt_task_label(task)}\n{summary}",
            title="claude-queue",
            priority=PushoverPriority.HIGH,
            delivered=None,
            extra={"task_id": task.get("id"), "description": _fmt_task_label(task), "notify_flag": False},
        )
        return
    policy = task.get("notify_policy", "always")
    if policy == "infra-only":
        cls = _failure_class(result)
        if cls == "infra":
            send_notification(
                message=f"Claude task FAILED (infra): {_fmt_task_label(task)}\n{result.splitlines()[0][:300] if result else '(no output)'}",
                title="claude-queue",
                priority=PushoverPriority.HIGH,
            )
        else:
            # 'execution' and 'contention' both demote to the audit log; the
            # class distinction (and the extracted reviewer reason, if any)
            # is preserved in the audit line rather than collapsed (C2, C4).
            _log_silenced("failure", task, failure_class=cls,
                         demoted_from="HIGH", result=result,
                         reason=_reviewer_failure_reason(result))
        return
    summary = result.splitlines()[0][:300] if result else "(no output)"
    send_notification(
        message=f"Claude task FAILED: {_fmt_task_label(task)}\n{summary}",
        title="claude-queue",
        priority=PushoverPriority.HIGH,
    )


# ---------------------------------------------------------------------------
# Startup sweep
# ---------------------------------------------------------------------------

def startup_sweep(queue: ClaudeQueue, council_dir: Path | None = None) -> None:
    """Crash recovery. Called once before the claim loop begins.

    ``council_dir`` defaults to the module-level ``_COUNCIL_DIR`` constant but
    can be overridden — e.g. to inject a ``tmp_path`` in tests without
    monkeypatching module state. Resolved at call time (not as a default-arg
    value) so tests that monkeypatch ``_COUNCIL_DIR`` still take effect.
    """
    if council_dir is None:
        council_dir = _COUNCIL_DIR
    now = datetime.now(PACIFIC)
    for active_yaml in queue.active_dir.glob("*.yaml"):
        task = queue._read_task(active_yaml)
        if not task:
            continue
        started_at = task.get("started_at")
        timeout = int(task.get("timeout_seconds", 300))
        stale = False
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                age_s = (now - started_dt).total_seconds()
                stale = age_s > (timeout + STARTUP_STALE_GRACE_S)
            except ValueError:
                stale = True
        else:
            stale = True
        if stale:
            log.warning(f"stale active task {task['id']} — moving to failed")
            queue.fail(task["id"], error="runner_crash_recovery")

    # Unconditional in_flight reconciliation: state.json.in_flight may list
    # ids with no backing active/*.yaml file at all (hand-killed + rm'd out
    # of band, before this sweep or fail() ever ran) — the loop above only
    # ever sees files that still exist, so it cannot catch this case. Force
    # a fresh rebuild from the live glob every startup, independent of
    # whether anything above was found stale.
    state = queue._read_state()
    queue._refresh_state(state)
    queue._write_state(state)

    if WORKTREE_ROOT.exists():
        active_ids = {p.stem for p in queue.active_dir.glob("*.yaml")}
        for wt in WORKTREE_ROOT.iterdir():
            if not wt.is_dir() or wt.name in active_ids:
                continue
            log.warning(f"orphaned worktree {wt} — removing")
            for clone in Path("/srv/git").glob("*-working"):
                subprocess.run(
                    ["git", "-C", str(clone), "worktree", "remove",
                     "--force", str(wt)],
                    check=False, capture_output=True, timeout=30)
            shutil.rmtree(wt, ignore_errors=True)

    for clone in Path("/srv/git").glob("*-working"):
        subprocess.run(
            ["git", "-C", str(clone), "worktree", "prune"],
            check=False, capture_output=True, timeout=30)

    # Hoisted out of the council-orphan block below so both it and the
    # narrative-emit orphan-spec pass can rely on queued_ids regardless of
    # whether council_dir exists on this host.
    import yaml as _yaml
    queued_ids: set[str] = set()
    for subdir_name in ("pending", "active", "completed", "failed"):
        subdir = queue.queue_dir / subdir_name
        if subdir.exists():
            for f in subdir.glob("*.yaml"):
                queued_ids.add(f.stem)
    cancelled_dir = queue.queue_dir / "cancelled"
    if cancelled_dir.exists():
        for f in cancelled_dir.glob("*.yaml"):
            queued_ids.add(f.stem)

    # Council orphan recovery: mark deliberating runs that have no
    # corresponding queue task and are older than _COUNCIL_ORPHAN_AGE_SECS.
    if council_dir.exists():
        sweep_now = datetime.now()  # naive — matches council YAML created_at
        for run_yaml in council_dir.glob("*.yaml"):
            try:
                run_data = _yaml.safe_load(run_yaml.read_text())
            except Exception:
                continue
            if not isinstance(run_data, dict):
                continue
            if run_data.get("status") != "deliberating":
                continue
            created_at = run_data.get("created_at")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at)
                    age_s = (sweep_now - created_dt).total_seconds()
                    if age_s <= _COUNCIL_ORPHAN_AGE_SECS:
                        continue
                except ValueError:
                    pass
            run_id = run_data.get("run_id") or run_yaml.stem
            if run_id in queued_ids:
                continue
            log.warning(f"orphan council run {run_id} — marking failed (crash recovery)")
            run_data["status"] = "failed"
            run_data["error"] = "runner_crash_recovery"
            run_data["completed_at"] = sweep_now.isoformat(timespec="seconds")
            try:
                run_yaml.write_text(
                    _yaml.safe_dump(
                        run_data, sort_keys=False, width=100, allow_unicode=True
                    )
                )
            except Exception as exc:
                log.warning(f"council orphan recovery: could not write {run_yaml}: {exc}")

    # narrative-emit orphan-spec recovery: a pending/*.json spec with no
    # matching .yaml anywhere means its submit() (or the atomic rename after
    # it) never completed - dead-letter it rather than leaving it stuck.
    for spec_json in sorted(queue.pending_dir.glob("*.json")):
        name = spec_json.name
        if name.startswith(".") or name.endswith(".tmp"):
            continue
        task_id = spec_json.stem
        if task_id in queued_ids:
            continue
        try:
            dead_letter_dir = queue.queue_dir / "dead-letter"
            dead_letter_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(spec_json), str(dead_letter_dir / spec_json.name))
            log.warning(f"orphaned spec {task_id} has no matching queue entry, moved to dead-letter")
        except Exception as exc:
            log.warning(f"failed to reap orphaned spec {task_id}: {exc}")

    # Layer 2: reap lapis-fixer-*.scope units left behind by a prior crash.
    _reap_orphan_scopes(queue)


# ---------------------------------------------------------------------------
# Runner failure classification
# ---------------------------------------------------------------------------

def _classify_runner_failure(combined: str, rc: int) -> tuple[str, str]:
    """Return (prefix, error_for_queue_fail) for a non-zero _runner.py exit.

    Contract source: ``agents_core.shaped_runner``.  That module emits
    ``ERROR: worktree_setup: <exception>`` as the *first token of a line*
    when the worktree setup itself raises, and
    ``ERROR: local reviewer produced no verdict (reason=<why>)`` as the
    first token of a line when ``_run_local_reviewer`` gets no result from
    ``call_gw_agent``. All other failure modes (call_claude_cli returning
    None, shape failures, unknown return codes) do NOT emit either prefix.

    The local-reviewer line is surfaced verbatim (not collapsed to a bare
    prefix like the worktree case) so the reason string reaches the queue
    record's ``error`` field — a leg that produced nothing must not be
    recorded as ``completed / error: null``, nor as a bare ``EXIT 1`` that
    hides why.

    Any future prefix added to shaped_runner must be reflected here with an
    explicit line-anchored check — do NOT revert to substring ``in combined``
    matching, which was the source of the 2026-04-27 misclassification bug.
    """
    lines = combined.splitlines()
    has_setup_err = any(line.startswith("ERROR: worktree_setup") for line in lines)
    reviewer_err_line = next(
        (line for line in lines if line.startswith("ERROR: local reviewer produced no verdict")),
        None,
    )
    if has_setup_err:
        prefix = "ERROR: worktree_setup"
    elif reviewer_err_line is not None:
        prefix = reviewer_err_line
    else:
        prefix = f"EXIT {rc}"
    return prefix, prefix[:200]


# ---------------------------------------------------------------------------
# Output-write helpers (extracted for testability)
# ---------------------------------------------------------------------------

# Served-model echo boundedness (local-reviewer-identity-and-provenance-v0,
# L1.D3): the PROVENANCE line is machine-parsed on a mixed-stdout channel
# (the claude engine prints the full model result to stdout), so the parse
# is line-anchored on the `^PROVENANCE: ` prefix; the served capture is the
# rest of the line so a whitespace-containing bound-violating token still
# matches and is voided by the bound check (a strict `\S+` capture would
# treat the line as absent instead of void).
_PROVENANCE_LINE_RE = re.compile(r"^PROVENANCE: seat=(\S+)(?: served=(.*))?$")
_SERVED_MODEL_ECHO_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
_SERVED_MODEL_ECHO_MAX_LEN = 200


def _parse_provenance_line(combined: str) -> dict | None:
    """Parse the PROVENANCE line out of a shaped_runner stdout capture.

    Line-anchored on the ``^PROVENANCE: `` prefix (last matching line wins —
    the same line-anchored contract as the failure-prefix classification
    above). Returns ``{"seat": ..., "served": ...}`` where ``served`` is
    None when the echo is void (absent from the line, or a bound-violating
    token — the seat alias is never substituted). Returns None when no
    PROVENANCE line is present (absence is a non-error: no crash, no
    dead-letter).
    """
    seat = None
    served: str | None = None
    for line in (combined or "").splitlines():
        m = _PROVENANCE_LINE_RE.match(line)
        if not m:
            continue
        seat = m.group(1)
        served = m.group(2)
    if seat is None:
        return None
    if served is not None:
        if (
            len(served) > _SERVED_MODEL_ECHO_MAX_LEN
            or not _SERVED_MODEL_ECHO_RE.match(served)
        ):
            served = None  # bound-violating token is VOID, not a value
    return {"seat": seat, "served": served}


def _write_success_output(path: Path, combined: str) -> None:
    """Write the full agent payload to *path*. No truncation on the success path."""
    path.write_text(combined if combined else "(no output)")


# ---------------------------------------------------------------------------
# Per-task execution
# ---------------------------------------------------------------------------

def _claim_lease_ctx(task: dict) -> tuple[str, str, str | None, str]:
    """D5 (attestation-contract-v0, leg 1): resolve the claim-lease context
    for a claimed task.

    Returns (base_url, work_id, backend_url, gw_url) where:
      base_url    - the doorman's own base (DOORMAN_SERVER - the port the
                    doorman serves the /lease/* endpoints on). The lease
                    acquire/release and the serving probe ride this base.
      work_id     - the lease's work_id (the task id - the same key the
                    per-run LLM lease uses, so the claim lease is visible to
                    the park decision for the full claim -> first-call
                    window)
      backend_url - the task's spec backend_url (claude_queue._task_backend_url
                    fail-open contract: None when the task carries no
                    spec_path or no backend_url - non-GW tasks, no lease).
                    This is the GW SEAT endpoint (the doorman's configured
                    gw_url, GW_URL - the same env the doorman probes for its
                    /status "serving" view and agents_core.llm uses for its
                    LLM calls). Consumed by the D5 seams' scope gate INSTEAD
                    of a second spec-JSON read (the claim() call site computes
                    the same value via the same fail-open function). The
                    scope gate compares backend_url against the FIXED GW_URL
                    anchor (os.environ["GW_URL"] - the same env
                    doorman_server.create_app reads for the node's gw_url),
                    NOT against the task's own backend_url (a self-
                    referential comparison is a tautology that scopes
                    nothing - cycle-3 reviewer finding) and NOT against the
                    doorman's own port (cycle-2 reviewer finding: the rev-1
                    gate compared the GW seat URL against DOORMAN_SERVER and
                    failed for every real GW task).

      gw_url      - the FIXED GW_URL env (the doorman's configured gw_url
                    anchor for the scope gate - the same env
                    doorman_server.create_app reads for the node's gw_url).
                    The task's own backend_url is NOT the anchor (cycle-3
                    reviewer finding).

    Never raises: any failure shape degrades to (base, work_id, None, gw_url).
    """
    base_url = os.environ.get("DOORMAN_SERVER", "http://127.0.0.1:8407")
    work_id = task.get("id", "")
    # The D5 scope-gate anchor: the FIXED GW_URL env (the same env
    # doorman_server.create_app reads for the node's gw_url) - NOT the
    # task's own backend_url (a self-referential comparison is a
    # tautology that scopes nothing - cycle-3 reviewer finding).
    gw_url = os.environ.get("GW_URL", "")
    backend_url: str | None = None
    try:
        backend_url = _cq_task_backend_url(task)
    except Exception:
        backend_url = None
    return base_url, work_id, backend_url, gw_url


def _acquire_claim_lease(task: dict) -> bool:
    """D5 claim seam (best-effort): acquire the doorman claim lease after a
    successful GW-backend claim. Never raises; a failure is a WARN line -
    the run proceeds lease-less (the death-class signals cover the
    seat-down case)."""
    base_url, work_id, backend_url, gw_url = _claim_lease_ctx(task)
    if backend_url is None:
        return False
    try:
        return _cq_acquire_claim_lease(
            base_url=base_url,
            work_id=work_id,
            task_id=task.get("id", ""),
            backend_url=backend_url,
            timeout_s=task.get("timeout_seconds"),
            gw_url=gw_url,
        )
    except Exception as e:
        log.warning("claim-lease: acquire failed for %s: %s",
                    task.get("id"), e)
        return False


def _release_claim_lease(task: dict) -> None:
    """D5 release seam: release the claim lease on a run exit path.
    Never raises (a lost release degrades to the TTL-bounded zombie
    window)."""
    base_url, work_id, backend_url, gw_url = _claim_lease_ctx(task)
    if backend_url is None:
        return
    try:
        _cq_release_claim_lease(
            base_url=base_url,
            work_id=work_id,
            task_id=task.get("id", ""),
            backend_url=backend_url,
            gw_url=gw_url,
        )
    except Exception as e:
        log.warning("claim-lease: release failed for %s: %s",
                    task.get("id"), e)


async def _run_shaped_task(queue: ClaudeQueue, task: dict) -> None:
    """Spawn _runner.py, capture output, write output file, mark done.

    Body of the shaped-agent execution path (Step A extraction).
    """
    task_id = task["id"]
    timeout = int(task.get("timeout_seconds", 300))
    spec_path = (task.get("payload") or {}).get("spec_path")
    output_path = str(OUTPUT_DIR / f"{task_id}-output.md")

    if not spec_path:
        msg = "ERROR: task payload missing spec_path"
        Path(output_path).write_text(msg)
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    log.info(f"claim {task_id} model={task.get('model')} timeout={timeout}s")

    # Lane-reality preflight (agents-core-lane-reality-preflight-v0) lives at
    # the CLAIM decision point - ClaudeQueue.claim() - not here: a requeue at
    # this point would head-of-line-block the queue (claim() hands back the
    # single highest-priority row, so a parked row would starve every other
    # pending row behind it). claim() skips parked candidates instead, so a
    # dead lane never burns a cycle and never starves another lane's work.

    # D5 (attestation-contract-v0, leg 1): the claim-time doorman lease.
    # Best-effort: scoped to GW-backend tasks (the spec's backend_url names
    # the doorman's seat), probe-gated (no cold-wake of a down seat),
    # released on every exit path below.
    if _acquire_claim_lease(task):
        log.info(
            "claim-lease: acquired doorman lease for %s "
            "(claim -> first-call window)",
            task_id,
        )

    # Cgroup isolation: wrap in a user-manager scope (fail-closed).
    _orig_argv = [sys.executable, "-m", RUNNER_SCRIPT_MODULE, spec_path]
    global _cage_alert_last_ts, _cage_unavail_consecutive, _cage_unavail_hold_until, _cage_critical_sent
    _cage_ok, _cage_reason = await asyncio.to_thread(_cage_buildable)
    if _cage_ok:
        if _cage_unavail_consecutive:
            log.info("cgroup-isolation: cage restored after %d consecutive failures",
                     _cage_unavail_consecutive)
            _cage_unavail_consecutive = 0
            _cage_unavail_hold_until = 0.0
            _cage_critical_sent = False
        if not await asyncio.to_thread(_check_slice_has_cpu_quota):
            log.warning(
                "cgroup-isolation: %s has no CPUQuota — task %s runs "
                "unbounded-aggregate; install a persistent slice unit to "
                "enforce the aggregate cap",
                _FIXER_SLICE, task_id,
            )
        _launch_argv = _build_scope_argv(task_id, _orig_argv)
    elif os.environ.get("CLAUDE_QUEUE_ALLOW_UNBOUNDED", "0") == "1":
        log.warning(
            "cgroup-isolation: cage unavailable (%s); "
            "CLAUDE_QUEUE_ALLOW_UNBOUNDED=1 — launching %s unbounded",
            _cage_reason, task_id,
        )
        _launch_argv = _orig_argv
    else:
        _now = time.monotonic()
        _cage_unavail_consecutive += 1
        _backoff_s = _CAGE_UNAVAIL_BACKOFF_CAPS_S[
            min(_cage_unavail_consecutive - 1, len(_CAGE_UNAVAIL_BACKOFF_CAPS_S) - 1)
        ]
        _cage_unavail_hold_until = _now + _backoff_s
        _is_critical = _cage_unavail_consecutive >= _CAGE_ESCALATION_THRESHOLD
        _requeue_to_pending(queue, task_id)
        # D5: release on requeue - the requeued task is never lease-less in
        # the park window (it re-acquires on the re-claim).
        _release_claim_lease(task)
        if _is_critical:
            log.critical(
                "cgroup-isolation: user bus PERSISTENTLY unreachable (%s) — "
                "fixer cage down, queue effectively stalled; consecutive=%d; "
                "host/config intervention required",
                _cage_reason, _cage_unavail_consecutive,
            )
        else:
            log.warning(
                "cgroup-isolation: cage unavailable (%s) — task %s requeued "
                "(fail-closed; backoff %ds; consecutive=%d; "
                "set CLAUDE_QUEUE_ALLOW_UNBOUNDED=1 to override)",
                _cage_reason, task_id, _backoff_s, _cage_unavail_consecutive,
            )
        # Notification fires exactly once when first crossing the threshold (not
        # subject to the normal cooldown - a systemic break needs to be loud).
        # log.critical() above fires on every post-threshold claim; only the
        # push notification is gated by _cage_critical_sent.
        if _is_critical and not _cage_critical_sent:
            _cage_critical_sent = True
            send_notification(
                message=(
                    f"CRITICAL: user bus persistently unreachable ({_cage_reason}); "
                    f"fixer cage down — queue stalled after {_cage_unavail_consecutive} "
                    f"consecutive failures. Host/config intervention required."
                ),
                title="claude-queue CRITICAL",
                priority=PushoverPriority.HIGH,
            )
        elif not _is_critical and _now - _cage_alert_last_ts >= _CAGE_ALERT_COOLDOWN_S:
            _cage_alert_last_ts = _now
            send_notification(
                message=(
                    f"claude-queue cage build failed ({_cage_reason}): "
                    f"jobs are held until the user bus is restored or "
                    f"CLAUDE_QUEUE_ALLOW_UNBOUNDED=1 is set."
                ),
                title="claude-queue",
                priority=PushoverPriority.HIGH,
            )
        return

    try:
        proc = await asyncio.create_subprocess_exec(
            *_launch_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        msg = f"ERROR: failed to spawn shaped_runner: {e}"
        Path(output_path).write_text(msg)
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        _release_claim_lease(task)  # D5: release on every run exit path
        return

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        killed_pid = proc.pid
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stdout_b, stderr_b = await proc.communicate()
        combined = (stdout_b + stderr_b).decode(errors="replace")
        result = f"TIMEOUT: exceeded {timeout}s\n{combined[-3000:]}"
        Path(output_path).write_text(result)
        queue.fail(task_id, error=f"timeout after {timeout}s")
        notify_failure(task, result)
        # D4: best-effort fail outstanding gw-admission tickets from the killed subprocess.
        try:
            import socket as _socket
            from agents_core.elevator import ElevatorStore as _ES, DB_DIR as _EDB, DB_PATH as _EDBP
            _estore = _ES(db_path=_EDBP)
            _estore.fail_pending_by_pid(_socket.gethostname(), killed_pid)
            _estore.close()
        except Exception:
            pass
        _release_claim_lease(task)  # D5: release on every run exit path
        return

    combined = (stdout_b + stderr_b).decode(errors="replace").strip()
    rc = proc.returncode or 0

    if rc < 0:
        result = f"INTERRUPTED by signal {-rc}:\n{combined[-3000:]}"
        Path(output_path).write_text(result)
        queue.fail(task_id, error=f"interrupted signal {-rc}")
        notify_failure(task, result)
        _release_claim_lease(task)  # D5: release on every run exit path
        return

    if rc != 0:
        prefix, error_str = _classify_runner_failure(combined, rc)
        result = f"{prefix}:\n{combined[-3000:]}"
        Path(output_path).write_text(result)
        queue.fail(task_id, error=error_str)
        notify_failure(task, result)
        _release_claim_lease(task)  # D5: release on every run exit path
        return

    # Success path: write the full agent response. Consumers (e.g. spec-review
    # JSON-verdict parsers) may need the head of the payload. Failure branches
    # above intentionally tail-slice to bound stderr noise.
    _write_success_output(Path(output_path), combined)
    summary = combined.splitlines()[0][:200] if combined else ""
    # Served-model provenance (local-reviewer-identity-and-provenance-v0,
    # L1.D3): stamp the served_model field into the completed task yaml
    # beside the existing `model:` field (requested alias stays as-is). A
    # void echo (absent PROVENANCE line or bound-violating token) is
    # explicit None - never the seat alias. Absence is a non-error: no
    # crash, no dead-letter.
    prov = _parse_provenance_line(combined)
    queue.complete(
        task_id,
        output_path=output_path,
        result_summary=summary,
        served_model=prov.get("served") if prov else None,
        served_model_provided=prov is not None,
    )
    _extract_ops_primitives(
        task_id,
        task.get("task_type", "subprocess"),
        combined,
        output_path,
    )
    notify_completion(task, output_path)
    _release_claim_lease(task)  # D5: release on every run exit path
    log.info(f"done  {task_id} rc=0")


# ---------------------------------------------------------------------------
# Council task handler
# ---------------------------------------------------------------------------


def _force_council_run_failed(run_yaml_path: Path, task_id: str, worker_error: str) -> None:
    """Best-effort: stamp status:failed + worker_error on a council run YAML.

    Called by the parent watchdog when the child exits non-zero (AC3).
    Never raises — failures are logged and suppressed.
    """
    import tempfile as _tempfile
    import yaml as _yaml
    try:
        data = _yaml.safe_load(run_yaml_path.read_text())
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {"run_id": task_id}
    if data.get("status") in ("resolved", "open", "laid-down", "closed"):
        return
    if data.get("status") == "failed" and data.get("worker_error"):
        return  # self-captured traceback takes priority; parent's generic message is fallback only
    data["status"] = "failed"
    data["worker_error"] = worker_error
    content = _yaml.safe_dump(data, sort_keys=False, width=100, allow_unicode=True)
    try:
        fd, tmp = _tempfile.mkstemp(dir=run_yaml_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(content)
            os.replace(tmp, run_yaml_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        log.error(f"_force_council_run_failed: could not write {run_yaml_path}: {e}")


async def _run_council_task(queue: ClaudeQueue, task: dict) -> None:
    """Spawn `python -m agents_core.council run <run_id>`, read terminal status,
    mark queue complete/failed.

    Output path = run YAML (what dashboard reads).
    Does NOT call _extract_ops_primitives.
    notify_completion / notify_failure gated on task["notify"].
    """
    task_id = task["id"]
    timeout = int(task.get("timeout_seconds", 1200))
    payload = task.get("payload") or {}
    mode = payload.get("mode", "")

    if mode not in ("deliberation", "scene"):
        msg = f"council.run: invalid payload.mode={mode!r}"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    run_yaml_path = _COUNCIL_DIR / f"{task_id}.yaml"
    log_file = _COUNCIL_LOG_DIR / f"{task_id}.log"
    _COUNCIL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"council claim {task_id} mode={mode} timeout={timeout}s")

    try:
        with open(log_file, "ab") as log_fh:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "agents_core.council", "run", task_id,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
            )
    except OSError as e:
        msg = f"ERROR: failed to spawn council subprocess: {e}"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        council_killed_pid = proc.pid
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        msg = f"timeout after {timeout}s"
        _force_council_run_failed(run_yaml_path, task_id, f"council worker {msg}")
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        # D4: best-effort fail outstanding gw-admission tickets from the killed council process.
        try:
            import socket as _socket
            from agents_core.elevator import ElevatorStore as _ES, DB_PATH as _EDBP
            _estore = _ES(db_path=_EDBP)
            _estore.fail_pending_by_pid(_socket.gethostname(), council_killed_pid)
            _estore.close()
        except Exception:
            pass
        return

    rc = proc.returncode

    if rc is not None and rc < 0:
        msg = f"interrupted signal {-rc}"
        _force_council_run_failed(run_yaml_path, task_id, f"council worker {msg}")
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    if rc != 0:
        msg = f"EXIT {rc}: subprocess failed before terminal status"
        _force_council_run_failed(run_yaml_path, task_id, f"council worker exited code={rc}")
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    try:
        import yaml as _yaml
        run_data = _yaml.safe_load(run_yaml_path.read_text())
        status = (run_data or {}).get("status", "")
    except Exception as e:
        msg = f"could not read run YAML after subprocess exit: {e}"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    terminal_set = (
        _DELIBERATION_TERMINAL if mode == "deliberation" else _SCENE_TERMINAL
    )

    if status == "failed":
        error_detail = (run_data or {}).get("error", "run_deliberation raised")
        queue.fail(task_id, error=error_detail)
        notify_failure(task, error_detail)
        return

    if status not in terminal_set:
        msg = "runtime_did_not_set_terminal_status"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    output_path = str(run_yaml_path)
    queue.complete(
        task_id, output_path=output_path,
        result_summary=f"council {mode} {status}",
    )
    notify_completion(task, output_path)
    log.info(f"council done {task_id} status={status}")


# ---------------------------------------------------------------------------
# llm_call handler
# ---------------------------------------------------------------------------

_CLI_MODEL_MAP: dict[str, str] = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
}

_ALLOWED_OPERATOR_CLASSES = frozenset(_CLI_MODEL_MAP)


async def _run_llm_call_task(queue: ClaudeQueue, task: dict) -> None:
    """Handle task_type=llm_call — call claude -p via call_claude_cli.

    payload.operator_class ∈ {"sonnet","opus","haiku"} (required).
    payload.prompt (required, non-empty str).
    payload.system (optional str, default "").
    payload.json_mode (optional bool, default False).

    Rejects qwen (has its own sync path) and any unknown operator_class.
    Does NOT call _extract_ops_primitives — llm_call returns raw model text.
    """
    from agents_core.llm import call_claude_cli

    task_id = task["id"]
    payload = task.get("payload") or {}
    output_path = str(OUTPUT_DIR / f"{task_id}-output.md")

    operator_class = payload.get("operator_class")

    if operator_class == "qwen":
        msg = "operator_class='qwen' is not routable through llm_call — qwen has its own sync path via call_llm()"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    if operator_class == "gravitywell":
        msg = "operator_class='gravitywell' is not routable through llm_call — gravitywell has its own sync path via call_operator()"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    if operator_class not in _ALLOWED_OPERATOR_CLASSES:
        msg = f"llm_call: unknown operator_class={operator_class!r}; must be one of {sorted(_ALLOWED_OPERATOR_CLASSES)}"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    prompt = payload.get("prompt")
    if not prompt:
        msg = "payload.prompt is empty or missing"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    system = payload.get("system", "")
    json_mode = bool(payload.get("json_mode", False))
    cli_model = _CLI_MODEL_MAP[operator_class]
    timeout = int(task.get("timeout_seconds", 300))

    log.info(f"llm_call claim {task_id} operator={operator_class} model={cli_model} timeout={timeout}s")

    text = await asyncio.to_thread(
        call_claude_cli,
        prompt,
        system=system,
        model=cli_model,
        timeout=timeout,
        json_mode=json_mode,
    )

    if text is None:
        msg = "call_claude_cli returned None (subprocess failure or empty response)"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    Path(output_path).write_text(text)
    summary = text.splitlines()[0][:200] if text else ""
    queue.complete(task_id, output_path=output_path, result_summary=summary)
    notify_completion(task, output_path)
    log.info(f"llm_call done {task_id} chars={len(text)}")


# ---------------------------------------------------------------------------
# Task dispatch (Step B)
# ---------------------------------------------------------------------------

async def _run_task(queue: ClaudeQueue, task: dict) -> None:
    """Dispatch to the correct handler based on task_type.

    Dispatch happens BEFORE any field validation so council tasks never
    reach the spec_path check in _run_shaped_task.
    """
    tt = task.get("task_type", "subprocess")
    if tt == "council.run":
        return await _run_council_task(queue, task)
    if tt == "llm_call":
        return await _run_llm_call_task(queue, task)
    return await _run_shaped_task(queue, task)


# ---------------------------------------------------------------------------
# Freeze-guard: pure-psutil RAM/swap-headroom claim gate
# ---------------------------------------------------------------------------
# guaardvark@51d9829c131d — plugins/swarm/service/orchestrator.py
# Cheap pure-psutil RAM/swap floor; re-checked before bringing on each new
# subprocess so parallel claude -p workers can't drive the box to a swap freeze.
# NOTE: thresholds re-calibrated for BRIX (28GB RAM / 8GB swap, ~3.7GB swap at
# idle) — guaardvark's 60GB-box values (6.0 / 1.0) do NOT transfer; see spec.

SPAWN_MIN_RAM_AVAIL_GB = float(os.environ.get("CLAUDE_QUEUE_MIN_RAM_GB", "4.0"))
SPAWN_MIN_SWAP_FREE_GB = float(os.environ.get("CLAUDE_QUEUE_MIN_SWAP_FREE_GB", "1.5"))
_GB = 1024 ** 3

_guard_blocking: bool = False


def _spawn_freeze_guard_block_reason() -> str | None:
    """Reason to withhold claiming a new subprocess, or None. Pure psutil so it
    never blocks on a missing probe; on read failure returns None (the guard fails
    open, never the thing that blocks all work)."""
    try:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        ram_avail_gb = vm.available / _GB
        swap_free_gb = (sw.total - sw.used) / _GB
    except Exception:
        return None
    if ram_avail_gb < SPAWN_MIN_RAM_AVAIL_GB:
        return f"RAM available {ram_avail_gb:.1f} GB < {SPAWN_MIN_RAM_AVAIL_GB:.1f} GB floor"
    if swap_free_gb < SPAWN_MIN_SWAP_FREE_GB:
        return f"swap free {swap_free_gb:.1f} GB < {SPAWN_MIN_SWAP_FREE_GB:.1f} GB headroom"
    return None


# ---------------------------------------------------------------------------
# Cgroup isolation: user-manager scope + slice (Layer 1)
# ---------------------------------------------------------------------------

_FIXER_SLICE = "lapis-fixer.slice"
_CAGE_ALERT_COOLDOWN_S = 300.0
_cage_alert_last_ts: float = 0.0

# Anti-spin backoff for persistent cage-unavailable (Facets trickster requirement).
# Escalating seconds: 5 → 30 → 120, capped at 120s.
_CAGE_UNAVAIL_BACKOFF_CAPS_S: tuple = (5, 30, 120)
# Consecutive cage-unavailable count before escalating to CRITICAL severity.
_CAGE_ESCALATION_THRESHOLD: int = 5
_cage_unavail_consecutive: int = 0
_cage_unavail_hold_until: float = 0.0

# Set once at startup if the runner self-heals XDG_RUNTIME_DIR (queryable
# without re-logging; True means the systemd unit/drop-in has a config gap).
_LAPIS_RUNNER_SELF_HEALED: bool = False

# Tracks whether the CRITICAL escalation notification has been sent for the
# current cage-unavailable episode; resets when the cage restores.
_cage_critical_sent: bool = False


def _cage_buildable() -> tuple[bool, str]:
    """Check whether a systemd-run user scope cage can be built.

    Returns (ok, reason). Runs a real connectivity probe via `systemctl --user
    show` (≤5s timeout) to verify the user bus is actually reachable - not
    just that the socket file exists. The probe runs in the same env as the
    eventual launch (after the XDG_RUNTIME_DIR self-heal fires at startup),
    so True ⇒ a systemd-run --user call will actually connect.
    """
    if not shutil.which("systemd-run"):
        return False, "systemd-run not in PATH"
    uid = os.getuid()
    bus = f"/run/user/{uid}/bus"
    if not os.path.exists(bus):
        return False, f"user bus socket not found: {bus}"
    # Connectivity probe: the socket file alone is not sufficient (system
    # services can have the socket present via linger but lack XDG_RUNTIME_DIR
    # in their env, making systemd-run --user fail with "No medium found").
    try:
        r = subprocess.run(
            ["systemctl", "--user", "--no-pager", "show", "-p", "Version"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if r.returncode != 0:
            return False, f"user bus not reachable: systemctl --user rc={r.returncode}"
    except subprocess.TimeoutExpired:
        return False, "user bus not reachable: probe timed out"
    except Exception as e:
        return False, f"user bus not reachable: {e}"
    return True, ""


def _check_slice_has_cpu_quota() -> bool:
    """Return True if lapis-fixer.slice has a finite CPUQuota (persistent
    slice unit installed). False means transient / uncapped aggregate.

    systemd exposes the quota as CPUQuotaPerSecUSec= (e.g. "10s" for 1000%),
    never as "CPUQuota=". Treat LoadState=not-found, CPUQuotaPerSecUSec=infinity,
    or the property absent as uncapped (return False). Fail open (return True)
    only on subprocess error so warnings are never spurious on transient errors.
    """
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", _FIXER_SLICE,
             "--property=CPUQuotaPerSecUSec,LoadState", "--no-pager"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        out = r.stdout
        if "LoadState=not-found" in out:
            return False
        for line in out.splitlines():
            if line.startswith("CPUQuotaPerSecUSec="):
                val = line.split("=", 1)[1].strip()
                return val not in ("infinity", "", "0")
        return True  # property absent — fail open
    except Exception:
        return True


def _build_scope_argv(task_id: str, orig_argv: list) -> list:
    """Return the systemd-run-wrapped argv for task_id.

    Per-job quota read from env at launch time so tuning needs no redeploy.
    """
    cpu_quota = os.environ.get("CLAUDE_QUEUE_JOB_CPUQUOTA", "300%")
    mem_max = os.environ.get("CLAUDE_QUEUE_JOB_MEMMAX", "6G")
    return [
        "systemd-run", "--user", "--scope", "--collect",
        f"--unit=lapis-fixer-{task_id}.scope",
        f"--slice={_FIXER_SLICE}",
        "-p", f"CPUQuota={cpu_quota}",
        "-p", f"MemoryMax={mem_max}",
        "--",
        *orig_argv,
    ]


def _requeue_to_pending(queue: ClaudeQueue, task_id: str) -> None:
    """Move an already-claimed task back to pending (cage unavailable, hold).

    Atomically renames the active YAML to pending so the next poll picks it
    up without data loss. status/started_at are stale but harmless — claim()
    overwrites them on the next acquisition.
    """
    active_path = queue.active_dir / f"{task_id}.yaml"
    pending_path = queue.pending_dir / f"{task_id}.yaml"
    try:
        active_path.rename(pending_path)
        log.info("scope-cage: task %s requeued to pending", task_id)
    except OSError as e:
        log.warning("scope-cage: requeue %s failed: %s", task_id, e)


# ---------------------------------------------------------------------------
# Scope reaper (Layer 2) — called from startup_sweep
# ---------------------------------------------------------------------------

def _reap_orphan_scopes(queue: ClaudeQueue) -> None:
    """Stop lapis-fixer-*.scope units with no matching active queue task.

    Runs once at startup to reap scopes left by a prior crash or hard restart.
    Best-effort: any error is logged and swallowed; startup is never aborted.
    """
    if not shutil.which("systemctl"):
        return
    try:
        r = subprocess.run(
            ["systemctl", "--user", "list-units", "--no-pager", "--no-legend",
             "--plain", "lapis-fixer-*.scope"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception as e:
        log.warning("scope-reaper: list-units error: %s", e)
        return

    if r.returncode != 0:
        log.warning("scope-reaper: systemctl list-units rc=%d — skipping", r.returncode)
        return

    active_ids = {p.stem for p in queue.active_dir.glob("*.yaml")}

    for line in r.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        unit_name = parts[0]
        if not (unit_name.startswith("lapis-fixer-") and unit_name.endswith(".scope")):
            continue
        task_id = unit_name[len("lapis-fixer-"):-len(".scope")]
        if task_id in active_ids:
            continue
        log.warning("scope-reaper: orphan scope %s (task %s not active) — stopping",
                    unit_name, task_id)
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", unit_name],
                capture_output=True, timeout=15, check=False,
            )
        except Exception as e:
            log.warning("scope-reaper: stop %s failed: %s", unit_name, e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

class Daemon:
    def __init__(self, workers: int, council_dir: Path | None = None):
        self.queue = ClaudeQueue()
        self.sem = asyncio.Semaphore(workers)
        self.stop_claiming = asyncio.Event()
        self.in_flight: set[asyncio.Task] = set()
        # Overridable so tests can inject a tmp_path instead of monkeypatching
        # the module-level _COUNCIL_DIR constant — threaded into startup_sweep().
        # Resolved at call time (not as a default-arg value) so tests that
        # monkeypatch _COUNCIL_DIR still take effect.
        self.council_dir = council_dir if council_dir is not None else _COUNCIL_DIR

    async def _worker(self, task: dict):
        # Branch order is load-bearing: council.run FIRST (council descriptions
        # are free-form decision text and can start with "fixer:"). The
        # family sub-cap is acquired BEFORE the global worker semaphore and
        # is load-bearing too (a claimed task waiting on the sub-cap must
        # not hold a global slot; council precedent).
        # D5 (attestation-contract-v0, leg 1): the claim lease is released
        # on EVERY exit path - including the raise path (the handler's
        # return paths release themselves; this finally covers the
        # exception escape so a run that raises is never lease-pinned).
        if task.get("task_type") == "council.run":
            async with _COUNCIL_SEM:
                async with self.sem:
                    await self._run_guarded(task)
        elif _is_fixer_task(task):
            async with _FIXER_SEM:
                async with self.sem:
                    await self._run_guarded(task)
        else:
            async with self.sem:
                await self._run_guarded(task)

    async def _run_guarded(self, task: dict):
        try:
            await _run_task(self.queue, task)
        except Exception as e:
            log.exception(
                f"unhandled error in task {task.get('id')}: {e}"
            )
        finally:
            _release_claim_lease(task)

    async def run(self):
        startup_sweep(self.queue, council_dir=self.council_dir)
        log.info("startup sweep complete, entering claim loop")
        while not self.stop_claiming.is_set():
            try:
                if os.environ.get("CLAUDE_QUEUE_ENABLED", "1") == "0":
                    log.info("CLAUDE_QUEUE_ENABLED=0 — exiting")
                    break

                if self.sem.locked():
                    await asyncio.sleep(POLL_INTERVAL_S)
                    continue

                global _guard_blocking
                if os.environ.get("CLAUDE_QUEUE_FREEZE_GUARD", "1") != "0":
                    block_reason = _spawn_freeze_guard_block_reason()
                    if block_reason is not None:
                        if not _guard_blocking:
                            _guard_blocking = True
                            log.warning("claude-queue freeze-guard ENGAGED: withholding claims (%s)", block_reason)
                        else:
                            log.debug("claude-queue freeze-guard still engaged (%s)", block_reason)
                        await asyncio.sleep(POLL_INTERVAL_S)
                        continue
                    elif _guard_blocking:
                        _guard_blocking = False
                        log.warning("claude-queue freeze-guard CLEARED: resuming claims")

                # Anti-spin: suppress claims during a cage-unavailable backoff window.
                if time.monotonic() < _cage_unavail_hold_until:
                    await asyncio.sleep(POLL_INTERVAL_S)
                    continue

                task = self.queue.claim()
                if task is None:
                    await asyncio.sleep(POLL_INTERVAL_S)
                    continue

                t = asyncio.create_task(self._worker(task))
                self.in_flight.add(t)
                t.add_done_callback(self.in_flight.discard)
            except (asyncio.CancelledError, GeneratorExit):
                # Legitimate shutdown-cancellation, not a crash - never route
                # this through the crash-exit path below.
                raise
            except Exception:
                pid = os.getpid()
                run_id = str(uuid.uuid4())[:8]
                log.critical(
                    "claim loop crashed (pid=%d run_id=%s); exiting for systemd restart",
                    pid, run_id, exc_info=True,
                )
                send_notification(
                    message=(
                        f"claude-queue-runner claim loop crashed (pid={pid}, "
                        f"run_id={run_id}); exiting for systemd restart."
                    ),
                    title="claude-queue CRITICAL",
                    priority=PushoverPriority.HIGH,
                )
                sys.exit(_CLAIM_LOOP_CRASH_EXIT_CODE)

        if self.in_flight:
            longest = max((int(self._task_timeout(t)) for t in self.in_flight),
                          default=300)
            log.info(f"draining {len(self.in_flight)} in-flight tasks "
                     f"(grace {longest+60}s)")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.in_flight, return_exceptions=True),
                    timeout=longest + 60,
                )
            except asyncio.TimeoutError:
                log.warning("grace deadline exceeded; some tasks may be stranded")

    def _task_timeout(self, _task: asyncio.Task) -> int:
        return 600

    def request_stop(self):
        log.info("SIGTERM received; stopping claim loop")
        self.stop_claiming.set()


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


async def _amain():
    workers = int(os.environ.get("CLAUDE_QUEUE_WORKERS", "2"))
    daemon = Daemon(workers=workers)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, daemon.request_stop)

    log.info(f"claude-queue-runner starting workers={workers} "
             f"queue_dir={daemon.queue.queue_dir}")
    await daemon.run()
    log.info("claude-queue-runner exited")


def _self_heal_user_bus_env() -> None:
    """Set XDG_RUNTIME_DIR at runner startup if the unit/drop-in failed to set it.

    Live drop-in: /etc/systemd/system/claude-queue-runner.service.d/10-user-bus-env.conf
    sets Environment=XDG_RUNTIME_DIR=/run/user/1000. This self-heal makes the runner
    correct even without that drop-in - surviving host rebuilds or unit reinstalls.

    Emits WARNING exactly once at startup (friction-as-signal: keeps the config gap
    visible without spamming per-launch). Sets _LAPIS_RUNNER_SELF_HEALED so the
    healed state is queryable without re-logging.
    """
    global _LAPIS_RUNNER_SELF_HEALED
    if os.environ.get("XDG_RUNTIME_DIR"):
        return
    uid = os.getuid()
    cand = f"/run/user/{uid}"
    if os.path.isdir(cand):
        os.environ["XDG_RUNTIME_DIR"] = cand
        _LAPIS_RUNNER_SELF_HEALED = True
        log.warning(
            "self-heal: XDG_RUNTIME_DIR was unset; set to %s — "
            "the systemd unit/drop-in SHOULD set this (config gap)", cand
        )
    else:
        log.warning(
            "self-heal: /run/user/%d absent; user-scope cage will be unavailable", uid
        )


def main():
    _setup_logging()
    _self_heal_user_bus_env()
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
