"""Synchronous ClaudeQueue wrapper — submit a task and block until it completes.

Public surface
--------------
submit_and_wait(task_dict, *, timeout_s, poll_interval_s, on_state_change)

The wrapper is the synchronous bridge between call_operator() callers and the
ClaudeQueue daemon.  It polls the queue's directory layout directly (no
q.poll() API — directory presence is the authoritative signal).

State-change callback (opt-in)
-------------------------------
Pass on_state_change=<callable> to receive a StateUpdate on every detected
transition: pending → active → completed/failed.  The callback is invoked
with a single StateUpdate positional argument.  Exceptions from the callback
are caught and logged at WARNING level — they never abort the wrapper.

Default on_state_change=None preserves byte-for-byte behavior parity with the
pre-v0 harness-engineer wrapper so import-only migrators (Leg 2) need no
signature changes at their call sites.
"""

import logging
import time
from typing import Callable, NamedTuple, Optional

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ClaudeQueue import — at module top so tests can patch agents_core.claude_queue_sync.ClaudeQueue
# ---------------------------------------------------------------------------

try:
    from agents_core.claude_queue import ClaudeQueue
except Exception:  # pragma: no cover
    ClaudeQueue = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class StateUpdate(NamedTuple):
    """State transition signal emitted from submit_and_wait's polling loop."""
    state: str        # "pending" | "active" | "completed" | "failed"
    task_id: str
    elapsed_s: float  # wall-clock seconds since submit_and_wait was called


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_emit(cb: Optional[Callable[[StateUpdate], None]], update: StateUpdate) -> None:
    """Call cb(update) swallowing any exception. No-op when cb is None."""
    if cb is None:
        return
    try:
        cb(update)
    except Exception as exc:
        log.warning("on_state_change callback raised (ignored): %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def submit_and_wait(
    task_dict: dict,
    *,
    timeout_s: float = 300,
    poll_interval_s: float = 2.0,
    on_state_change: Optional[Callable[[StateUpdate], None]] = None,
) -> str:
    """Submit a task to ClaudeQueue and block until it completes.

    Returns the text content of the completed task's output_path file.
    Raises RuntimeError if the task fails (queue.fail()'d) or output_path is missing.
    Raises TimeoutError if timeout_s elapses without completion or failure.

    If on_state_change is provided, the callable is invoked on every detected
    state transition: "pending" immediately after submit, then "active" when
    the daemon claims the task (YAML appears in q.active_dir), then terminal
    "completed" or "failed" when the YAML lands in the corresponding dir.
    Callback exceptions are caught and logged at WARNING level — they never
    abort the wrapper or change its return value. This is the legibility
    surface the Mirror Council asked for: callers watching long-running calls
    can render real-time state without polling the queue separately.

    Race condition note: a fast task may move pending→active→completed between
    two poll intervals, in which case the "active" transition is never observed
    and the callback receives only "pending" then "completed". This is
    acceptable — the wrapper is a best-effort observer of the daemon's state,
    not an authoritative state machine.
    """
    if ClaudeQueue is None:
        raise RuntimeError("agents_core.claude_queue not available")

    q = ClaudeQueue()

    start = time.monotonic()
    task_id = q.submit(task_dict)
    prev_state = "pending"
    _safe_emit(on_state_change, StateUpdate("pending", task_id, 0.0))

    completed_yaml = q.completed_dir / f"{task_id}.yaml"
    failed_yaml = q.failed_dir / f"{task_id}.yaml"
    active_yaml = q.active_dir / f"{task_id}.yaml"

    deadline = start + timeout_s
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - start

        if failed_yaml.exists():
            _safe_emit(on_state_change, StateUpdate("failed", task_id, elapsed))
            try:
                raw = yaml.safe_load(failed_yaml.read_text())
                error = (raw or {}).get("error") or "task failed (no error field)"
            except Exception as exc:
                error = f"task failed; could not read failed YAML: {exc}"
            raise RuntimeError(f"ClaudeQueue task {task_id!r} failed: {error}")

        if completed_yaml.exists():
            _safe_emit(on_state_change, StateUpdate("completed", task_id, elapsed))
            try:
                raw = yaml.safe_load(completed_yaml.read_text())
            except Exception as exc:
                raise RuntimeError(
                    f"ClaudeQueue task {task_id!r} completed but YAML is unreadable: {exc}"
                ) from exc
            output_path = (raw or {}).get("output_path")
            if not output_path:
                raise RuntimeError(
                    f"ClaudeQueue task {task_id!r} completed but output_path field is missing"
                )
            try:
                from pathlib import Path
                return Path(output_path).read_text()
            except OSError as exc:
                raise RuntimeError(
                    f"ClaudeQueue task {task_id!r} completed but output file is unreadable: {exc}"
                ) from exc

        if prev_state == "pending" and active_yaml.exists():
            _safe_emit(on_state_change, StateUpdate("active", task_id, elapsed))
            prev_state = "active"

        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"ClaudeQueue task {task_id!r} did not complete within {timeout_s}s"
    )
