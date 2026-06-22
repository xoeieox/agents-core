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
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

from agents_core.claude_queue import CLAUDE_QUEUE_DIR, ClaudeQueue
from agents_core.gpu import PACIFIC, Priority as QueuePriority  # noqa: F401
from agents_core.notify import Priority as PushoverPriority, send_notification

from agents_core.worktree import WORKTREE_ROOT

RUNNER_SCRIPT_MODULE = "agents_core.shaped_runner"
OUTPUT_DIR = CLAUDE_QUEUE_DIR / "completed"

CLONE_ROOTS_GLOB = "/srv/git/*-working"

POLL_INTERVAL_S = 2.0
STARTUP_STALE_GRACE_S = 300

# Council concurrency control — module-level, NOT on Daemon (see docstring).
# asyncio.Semaphore is safe to create at module level in Python 3.10+.
_COUNCIL_SEM = asyncio.Semaphore(1)
"""Hard cap: at most one council subprocess at a time.

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

_COUNCIL_DIR = Path("/srv/lapis/council")
_COUNCIL_LOG_DIR = Path("/srv/lapis/council/logs")
_COUNCIL_ORPHAN_AGE_SECS = int(os.environ.get("COUNCIL_ORPHAN_AGE_SECS", "3600"))

SILENCED_LOG = Path("/srv/lapis/notify-audit/silenced.jsonl")

# Terminal status sets per mode.
_DELIBERATION_TERMINAL = frozenset({"resolved", "open", "laid-down"})
_SCENE_TERMINAL = frozenset({"closed"})

log = logging.getLogger("claude-queue-runner")


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

def _failure_class(result: str) -> str:
    """Classify a failure result string as 'infra' or 'execution'.

    Inspects only the **first line** of the formatted result string
    (e.g. "EXIT 1:\n…", "ERROR: worktree_setup:\n…"), not raw subprocess output.
    This is distinct from _classify_runner_failure which scans all lines.

    Contract: use startswith (not substring 'in') to match prefixes.
    """
    lines = result.splitlines()
    first = lines[0] if lines else ""

    if first.startswith("ERROR:"):
        return "infra"
    if first.startswith(("EXIT ", "TIMEOUT:", "INTERRUPTED ")):
        return "execution"
    return "infra"  # unknown/empty → conservative infra


def _log_silenced(event: str, task: dict, *, failure_class: str | None,
                  demoted_from: str, result: str) -> None:
    """Append one JSON line to SILENCED_LOG for a demoted notification.

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
        }
        with open(SILENCED_LOG, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
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
            _log_silenced("failure", task, failure_class=cls,
                         demoted_from="HIGH", result=result)
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

def startup_sweep(queue: ClaudeQueue) -> None:
    """Crash recovery. Called once before the claim loop begins."""
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

    # Council orphan recovery: mark deliberating runs that have no
    # corresponding queue task and are older than _COUNCIL_ORPHAN_AGE_SECS.
    import yaml as _yaml
    if _COUNCIL_DIR.exists():
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

        sweep_now = datetime.now()  # naive — matches council YAML created_at
        for run_yaml in _COUNCIL_DIR.glob("*.yaml"):
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

    # Layer 2: reap lapis-fixer-*.scope units left behind by a prior crash.
    _reap_orphan_scopes(queue)


# ---------------------------------------------------------------------------
# Runner failure classification
# ---------------------------------------------------------------------------

def _classify_runner_failure(combined: str, rc: int) -> tuple[str, str]:
    """Return (prefix, error_for_queue_fail) for a non-zero _runner.py exit.

    Contract source: ``agents_core.shaped_runner``.  That module emits
    ``ERROR: worktree_setup: <exception>`` as the *first token of a line*
    when the worktree setup itself raises.  All other failure modes
    (call_claude_cli returning None, shape failures, unknown return codes)
    do NOT emit that prefix.

    Any future prefix added to shaped_runner must be reflected here with an
    explicit line-anchored check — do NOT revert to substring ``in combined``
    matching, which was the source of the 2026-04-27 misclassification bug.
    """
    has_setup_err = any(
        line.startswith("ERROR: worktree_setup")
        for line in combined.splitlines()
    )
    prefix = "ERROR: worktree_setup" if has_setup_err else f"EXIT {rc}"
    return prefix, prefix[:200]


# ---------------------------------------------------------------------------
# Output-write helpers (extracted for testability)
# ---------------------------------------------------------------------------

def _write_success_output(path: Path, combined: str) -> None:
    """Write the full agent payload to *path*. No truncation on the success path."""
    path.write_text(combined if combined else "(no output)")


# ---------------------------------------------------------------------------
# Per-task execution
# ---------------------------------------------------------------------------

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

    # Cgroup isolation: wrap in a user-manager scope (fail-closed).
    _orig_argv = [sys.executable, "-m", RUNNER_SCRIPT_MODULE, spec_path]
    _cage_ok, _cage_reason = _cage_buildable()
    if _cage_ok:
        if not _check_slice_has_cpu_quota():
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
        global _cage_alert_last_ts
        _now = time.monotonic()
        _requeue_to_pending(queue, task_id)
        if _now - _cage_alert_last_ts >= _CAGE_ALERT_COOLDOWN_S:
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
        log.error(
            "cgroup-isolation: cage unavailable (%s) — task %s requeued "
            "(fail-closed; set CLAUDE_QUEUE_ALLOW_UNBOUNDED=1 to override)",
            _cage_reason, task_id,
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
        return

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
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
        return

    combined = (stdout_b + stderr_b).decode(errors="replace").strip()
    rc = proc.returncode or 0

    if rc < 0:
        result = f"INTERRUPTED by signal {-rc}:\n{combined[-3000:]}"
        Path(output_path).write_text(result)
        queue.fail(task_id, error=f"interrupted signal {-rc}")
        notify_failure(task, result)
        return

    if rc != 0:
        prefix, error_str = _classify_runner_failure(combined, rc)
        result = f"{prefix}:\n{combined[-3000:]}"
        Path(output_path).write_text(result)
        queue.fail(task_id, error=error_str)
        notify_failure(task, result)
        return

    # Success path: write the full agent response. Consumers (e.g. spec-review
    # JSON-verdict parsers) may need the head of the payload. Failure branches
    # above intentionally tail-slice to bound stderr noise.
    _write_success_output(Path(output_path), combined)
    summary = combined.splitlines()[0][:200] if combined else ""
    queue.complete(task_id, output_path=output_path, result_summary=summary)
    _extract_ops_primitives(
        task_id,
        task.get("task_type", "subprocess"),
        combined,
        output_path,
    )
    notify_completion(task, output_path)
    log.info(f"done  {task_id} rc=0")


# ---------------------------------------------------------------------------
# Council task handler
# ---------------------------------------------------------------------------

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
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        msg = f"timeout after {timeout}s"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    rc = proc.returncode

    if rc is not None and rc < 0:
        msg = f"interrupted signal {-rc}"
        queue.fail(task_id, error=msg)
        notify_failure(task, msg)
        return

    if rc != 0:
        msg = f"EXIT {rc}: subprocess failed before terminal status"
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


def _cage_buildable() -> tuple[bool, str]:
    """Check whether a systemd-run user scope cage can be built.

    Returns (ok, reason). Synchronous; uses no subprocess — binary existence
    and socket path checks only (negligible latency).
    """
    if not shutil.which("systemd-run"):
        return False, "systemd-run not in PATH"
    uid = os.getuid()
    bus = f"/run/user/{uid}/bus"
    if not os.path.exists(bus):
        return False, f"user bus socket not found: {bus}"
    return True, ""


def _check_slice_has_cpu_quota() -> bool:
    """Return True if lapis-fixer.slice has a finite CPUQuota (persistent
    slice unit installed). False means transient / uncapped aggregate.

    Fails open (returns True) on any error so warnings are never spurious.
    """
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", _FIXER_SLICE,
             "--property=CPUQuota,LoadState", "--no-pager"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        out = r.stdout
        if "LoadState=not-found" in out:
            return False
        for line in out.splitlines():
            if line.startswith("CPUQuota="):
                val = line.split("=", 1)[1].strip()
                return "infinity" not in val and val not in ("", "0")
        return True  # key absent — fail open
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
    def __init__(self, workers: int):
        self.queue = ClaudeQueue()
        self.sem = asyncio.Semaphore(workers)
        self.stop_claiming = asyncio.Event()
        self.in_flight: set[asyncio.Task] = set()

    async def _worker(self, task: dict):
        if task.get("task_type") == "council.run":
            # _COUNCIL_SEM acquired BEFORE self.sem — order is load-bearing.
            async with _COUNCIL_SEM:
                async with self.sem:
                    try:
                        await _run_task(self.queue, task)
                    except Exception as e:
                        log.exception(
                            f"unhandled error in task {task.get('id')}: {e}"
                        )
        else:
            async with self.sem:
                try:
                    await _run_task(self.queue, task)
                except Exception as e:
                    log.exception(
                        f"unhandled error in task {task.get('id')}: {e}"
                    )

    async def run(self):
        startup_sweep(self.queue)
        log.info("startup sweep complete, entering claim loop")
        while not self.stop_claiming.is_set():
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

            task = self.queue.claim()
            if task is None:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue

            t = asyncio.create_task(self._worker(task))
            self.in_flight.add(t)
            t.add_done_callback(self.in_flight.discard)

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


def main():
    _setup_logging()
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
