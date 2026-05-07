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
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from agents_core.claude_queue import CLAUDE_QUEUE_DIR, ClaudeQueue
from agents_core.gpu import PACIFIC, Priority as QueuePriority  # noqa: F401
from agents_core.notify import Priority as PushoverPriority, send_notification

from agents_core.worktree import WORKTREE_ROOT

RUNNER_SCRIPT_MODULE = "agents_core.shaped_runner"
OUTPUT_DIR = CLAUDE_QUEUE_DIR / "completed"

CLONE_ROOTS_GLOB = "/srv/git/*-working"

POLL_INTERVAL_S = 2.0
STARTUP_STALE_GRACE_S = 300

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
    send_notification(
        message=f"Claude task completed: {_fmt_task_label(task)}\nOutput: {output_path}",
        title="claude-queue",
        priority=PushoverPriority.NORMAL,
    )


def notify_failure(task: dict, result: str) -> None:
    if not task.get("notify"):
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
# Per-task execution
# Step A: verbatim body extracted into _run_shaped_task; _run_task delegates.
# Step B (next commit): _run_task gains task_type dispatch + council handler.
# ---------------------------------------------------------------------------

async def _run_shaped_task(queue: ClaudeQueue, task: dict) -> None:
    """Spawn _runner.py, capture output, write output file, mark done.

    Body of the shaped-agent execution path.  Errors are caught at the top
    level; any unhandled exception drops the task in active/ for the next
    startup sweep to clean up.
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

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", RUNNER_SCRIPT_MODULE, spec_path,
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

    Path(output_path).write_text(combined[-3000:] if combined else "(no output)")
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


async def _run_task(queue: ClaudeQueue, task: dict) -> None:
    """Route to shaped handler (Step A: single route; Step B adds dispatch)."""
    return await _run_shaped_task(queue, task)


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
        async with self.sem:
            try:
                await _run_task(self.queue, task)
            except Exception as e:
                log.exception(f"unhandled error in task {task.get('id')}: {e}")

    async def run(self):
        startup_sweep(self.queue)
        log.info("startup sweep complete, entering claim loop")
        while not self.stop_claiming.is_set():
            if os.environ.get("CLAUDE_QUEUE_ENABLED", "1") == "0":
                log.info("CLAUDE_QUEUE_ENABLED=0 — exiting")
                break

            # Only claim when a worker slot is free. This avoids pulling
            # tasks into active/ while all workers are busy (the task would
            # sit under our id for the full duration of someone else's run).
            if self.sem.locked():
                await asyncio.sleep(POLL_INTERVAL_S)
                continue

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
