#!/usr/bin/env python3
"""GPU Task Queue — centralized GPU scheduler for StarHouse.

Directory-based priority queue for all GPU work. Any process can submit
tasks; the gpu_queue_runner daemon executes them sequentially by priority.

Queue directory layout:
    /srv/lapis/gpu-queue/
      pending/          YAML task files waiting to run
      active/           At most one YAML — currently executing
      completed/        Done tasks (pruned after 48h)
      failed/           Failed tasks (kept for review)
      history.jsonl     Append-only event log
      state.json        Snapshot: mode, current task, queue depth, model

Usage as library:
    from gpu_queue import GPUQueue, Priority
    q = GPUQueue()
    task_id = q.submit({
        "task_type": "pytest",
        "priority": Priority.NORMAL,
        "payload": {"target": "tests/test_attention.py"},
    })

Usage as CLI (status/management only — use gpu-submit for task submission):
    python3 gpu_queue.py status
    python3 gpu_queue.py pending
    python3 gpu_queue.py cancel <task_id> [reason]
    python3 gpu_queue.py cleanup [--max-age 48]
    python3 gpu_queue.py history [--limit 20]
"""

import fcntl
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

PACIFIC = ZoneInfo("America/Los_Angeles")

QUEUE_DIR = Path("/srv/lapis/gpu-queue")
PENDING_DIR = QUEUE_DIR / "pending"
ACTIVE_DIR = QUEUE_DIR / "active"
COMPLETED_DIR = QUEUE_DIR / "completed"
FAILED_DIR = QUEUE_DIR / "failed"
HISTORY_PATH = QUEUE_DIR / "history.jsonl"
STATE_PATH = QUEUE_DIR / "state.json"

MAX_COMPLETED = 100  # Keep last N completed tasks before pruning


# ---------------------------------------------------------------------------
# Task coordinator hook (Lapis Ops Layer).
#
# agents_core.gpu is a primitive — it knows nothing about the ops layer or
# intention-negotiation policy. Instead it exposes `register_coordinator()`
# as an extension point. A coordinator that is registered receives lifecycle
# events (submit/complete/fail) and decides how to coordinate intentions.
#
# When no coordinator is registered, lifecycle hooks are silent no-ops and
# the queue behaves like a plain priority queue.
#
# Coordinator contract — the object passed to register_coordinator() must
# expose the following callables:
#
#   project_from_task(task, *, projected_by, proposed_change, target_heading,
#                     task_id) -> ProjectionResult | None
#       Called on every non-opt-out submit. ProjectionResult must have:
#           .decision       str — "projected" | "match_reinforce" |
#                                 "match_manifested" | "sibling"
#           .shared_task_id str | None — non-None when decision is a match,
#                                        points at the prior task's id
#           .intention      object with `.intention_id` and `.reinforces`
#
#   manifest(intention_id, *, linked_task_id) -> None
#       Called when a task completes. Moves the intention from in-flight to
#       manifested. Must reinforce-match the consensus-negotiation rule (the
#       original reinforcers cascade too — this is the coordinator's concern,
#       not agents_core's).
#
#   compost(intention_id, *, reason) -> None
#       Called when a task fails. Moves the intention to composted.
#
# Registration patterns:
#   - ops-layer's intention_registry.py auto-registers itself at module load
#     (import side-effect), so `import intention_registry` is sufficient.
#   - `ops_layer_init.register()` is an explicit helper for callers that
#     don't want the side-effect import.
#   - Any future coordinator plugs in the same way; this module does not
#     know about intention_registry by name.
# ---------------------------------------------------------------------------

import logging as _logging

_coord_log = _logging.getLogger("gpu.coordinator")

_coordinator = None


def register_coordinator(coordinator) -> None:
    """Bind a task coordinator. See module docstring for the required surface.

    Safe to call multiple times — the last registration wins. Pass None to
    unregister (tests can restore no-coordinator mode this way).
    """
    global _coordinator
    _coordinator = coordinator


def get_coordinator():
    """Return the currently registered coordinator, or None."""
    return _coordinator


def _project_intention_for_task(task: dict, task_id: str):
    """Run intention projection via the registered coordinator.

    task_id is passed through so the coordinator can write its
    linked_task_id atomically with the intention, closing the attach-race
    window. Returns the coordinator's ProjectionResult (or None if no
    coordinator is registered or the call raises)."""
    reg = _coordinator
    if reg is None:
        return None
    try:
        return reg.project_from_task(
            task,
            projected_by=task.get("submitted_by") or "unknown",
            proposed_change=task.get("description") or task.get("task_type"),
            target_heading=(task.get("payload") or {}).get("target_heading"),
            task_id=task_id,
        )
    except Exception as e:
        _coord_log.warning(
            f"coordinator.project_from_task failed for {task.get('id')}: {e}")
        return None


def _manifest_intention_for_task(task: dict) -> None:
    """Notify the coordinator a task completed."""
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _coordinator
    if reg is None:
        return
    try:
        reg.manifest(intention_id, linked_task_id=task.get("id"))
    except Exception as e:
        _coord_log.warning(
            f"coordinator.manifest failed for {intention_id}: {e}")


def _compost_intention_for_task(task: dict, reason: str) -> None:
    """Notify the coordinator a task failed."""
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _coordinator
    if reg is None:
        return
    try:
        reg.compost(intention_id, reason=reason)
    except Exception as e:
        _coord_log.warning(
            f"coordinator.compost failed for {intention_id}: {e}")


class Priority:
    """Priority levels — lower number = higher priority."""
    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 80
    IDLE = 99


# Required fields in a task dict (everything else is optional)
REQUIRED_FIELDS = {"task_type"}

# Default values for optional fields
TASK_DEFAULTS = {
    "priority": Priority.NORMAL,
    "submitted_by": "unknown",
    "timeout_seconds": 300,
    "preemptible": False,
    "notify": False,
    "model": None,
    "stop_llm_server": False,
    "payload": {},
    "callback_path": None,
    "status": "pending",
    "started_at": None,
    "completed_at": None,
    "duration_seconds": None,
    "error": None,
}


def _now_iso() -> str:
    return datetime.now(PACIFIC).isoformat(timespec="seconds")


def _now_pacific() -> datetime:
    return datetime.now(PACIFIC)


class GPUQueue:
    """File-based GPU task queue with priority scheduling."""

    def __init__(self, queue_dir: str | Path = QUEUE_DIR):
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.active_dir = self.queue_dir / "active"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        self.history_path = self.queue_dir / "history.jsonl"
        self.state_path = self.queue_dir / "state.json"
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.pending_dir, self.active_dir,
                  self.completed_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, task_type: str) -> str:
        ts = _now_pacific().strftime("%Y%m%d_%H%M%S")
        slug = task_type.replace(" ", "").replace("-", "")[:16]
        # Add microseconds for uniqueness when multiple tasks submit per second
        usec = _now_pacific().strftime("%f")[:4]
        return f"gpu_{ts}_{usec}_{slug}"

    def _append_event(self, event: dict):
        """Append event to JSONL history with file locking."""
        with open(self.history_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _read_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "mode": "idle",
            "pid": None,
            "current_task": None,
            "current_model": None,
            "queue_depth": 0,
            "tasks_completed_today": 0,
            "last_activity": None,
        }

    def _write_state(self, state: dict):
        state["updated"] = _now_iso()
        fd, tmp = tempfile.mkstemp(
            dir=self.queue_dir, suffix=".tmp", prefix="state-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
                f.write("\n")
            os.replace(tmp, self.state_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_task(self, directory: Path, task: dict):
        """Atomically write a task YAML file."""
        path = directory / f"{task['id']}.yaml"
        fd, tmp = tempfile.mkstemp(
            dir=directory, suffix=".tmp", prefix="task-")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(task, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def _read_task(self, path: Path) -> dict | None:
        try:
            return yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError):
            return None

    def _update_queue_depth(self, state: dict):
        """Count pending tasks and update state."""
        state["queue_depth"] = len(list(self.pending_dir.glob("*.yaml")))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, task_dict: dict) -> str:
        """Submit a task to the queue. Always returns a task_id (str).

        Intention-registry behavior (opt-out via
        payload['_ignore_intention_registry']=True):

        - Novel signature → a new GPU task is queued; returns the new id.
        - Matches an in-flight intention → no new GPU task is queued; this
          caller joins the in-flight task by receiving the existing (shared)
          task_id. Callers that poll `completed_dir/<task_id>.yaml` will see
          the same result as the original submitter when the task finishes.
        - Matches a recently-manifested intention → no new GPU task; returns
          the prior task_id. `find_similar` gates this to cases where the
          prior task file still exists in completed/failed, so callers never
          receive an id whose yaml has been pruned.

        The task_id is generated before intention projection and passed into
        the registry so the intention's linked_task_id is written atomically,
        preventing races where a concurrent submit reads the intention before
        the task_id has been attached.
        """
        missing = REQUIRED_FIELDS - set(task_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        task = dict(TASK_DEFAULTS)
        task.update(task_dict)
        task["id"] = task.get("id") or self._generate_id(task["task_type"])
        task["submitted_at"] = _now_iso()
        task["status"] = "pending"

        payload = task.get("payload") or {}
        ignore_registry = bool(payload.get("_ignore_intention_registry"))

        projection = None
        if not ignore_registry:
            # Pass task["id"] so the registry can write the intention with
            # linked_task_id already set, eliminating the attach-race window.
            projection = _project_intention_for_task(task, task["id"])

        if projection is not None and projection.decision in (
            "match_reinforce", "match_manifested"
        ):
            shared = projection.shared_task_id
            self._append_event({
                "event": f"intention_{projection.decision}",
                "id": task["id"],
                "timestamp": task["submitted_at"],
                "task_type": task["task_type"],
                "submitted_by": task.get("submitted_by", "unknown"),
                "intention_id": projection.intention.intention_id,
                "reinforces": projection.intention.reinforces,
                "shared_task_id": shared,
            })
            # Both match_reinforce and match_manifested: return the shared
            # task_id (not None) so callers can poll the same completed-dir
            # entry. find_similar guarantees the prior task file still exists.
            if shared:
                return shared
            # Defensive fallback: shouldn't happen given find_similar gating,
            # but if we somehow get a match with no task_id, fall through and
            # queue a fresh task rather than returning None to callers.
            _ir_log.warning(
                f"intention match for {task['id']} had no shared_task_id; "
                f"falling back to queueing a fresh task"
            )

        # projected / sibling / registry disabled → queue the task normally.
        if projection is not None:
            task["intention_id"] = projection.intention.intention_id

        self._write_task(self.pending_dir, task)

        self._append_event({
            "event": "submitted",
            "id": task["id"],
            "timestamp": task["submitted_at"],
            "task_type": task["task_type"],
            "priority": task["priority"],
            "model": task.get("model"),
            "submitted_by": task.get("submitted_by", "unknown"),
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        self._update_queue_depth(state)
        state["last_activity"] = task["submitted_at"]
        self._write_state(state)

        return task["id"]

    def claim(self, current_model: str | None = None) -> dict | None:
        """Claim the highest-priority pending task. Returns task dict or None.

        Model-affinity tiebreak: when multiple tasks share the same priority,
        prefer the one matching current_model to avoid model swaps.
        """
        pending_files = sorted(self.pending_dir.glob("*.yaml"))
        if not pending_files:
            return None

        # Load all pending tasks
        tasks = []
        for p in pending_files:
            t = self._read_task(p)
            if t:
                t["_path"] = str(p)
                tasks.append(t)

        if not tasks:
            return None

        # Sort: priority ASC, then model-affinity (matching model first)
        def sort_key(t):
            pri = t.get("priority", Priority.NORMAL)
            # Same-model tasks sort before different-model tasks at same priority
            model_match = 0 if (current_model and t.get("model") == current_model) else 1
            submitted = t.get("submitted_at", "")
            return (pri, model_match, submitted)

        tasks.sort(key=sort_key)
        chosen = tasks[0]
        src_path = Path(chosen.pop("_path"))

        # Atomically move to active
        chosen["status"] = "running"
        chosen["started_at"] = _now_iso()
        self._write_task(self.active_dir, chosen)
        src_path.unlink(missing_ok=True)

        self._append_event({
            "event": "claimed",
            "id": chosen["id"],
            "timestamp": chosen["started_at"],
            "task_type": chosen["task_type"],
            "priority": chosen["priority"],
            "model": chosen.get("model"),
        })

        state = self._read_state()
        state["mode"] = "processing"
        state["current_task"] = {
            "id": chosen["id"],
            "task_type": chosen["task_type"],
            "priority": chosen["priority"],
            "started_at": chosen["started_at"],
            "model": chosen.get("model"),
        }
        self._update_queue_depth(state)
        state["last_activity"] = chosen["started_at"]
        self._write_state(state)

        return chosen

    def complete(self, task_id: str, output_path: str | None = None,
                 result_summary: str | None = None):
        """Mark a task as completed and move to completed dir."""
        active_path = self.active_dir / f"{task_id}.yaml"
        task = self._read_task(active_path)
        if not task:
            return

        now = _now_iso()
        task["status"] = "completed"
        task["completed_at"] = now
        if output_path:
            task["output_path"] = output_path
        if result_summary:
            task["result_summary"] = result_summary

        # Calculate duration
        if task.get("started_at"):
            try:
                dt_start = datetime.fromisoformat(task["started_at"])
                dt_end = datetime.fromisoformat(now)
                task["duration_seconds"] = int(
                    (dt_end - dt_start).total_seconds())
            except (ValueError, TypeError):
                pass

        self._write_task(self.completed_dir, task)
        active_path.unlink(missing_ok=True)

        _manifest_intention_for_task(task)

        self._append_event({
            "event": "completed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "output_path": output_path,
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        state["mode"] = "idle"
        state["current_task"] = None
        self._update_queue_depth(state)
        state["last_activity"] = now
        # Reset daily counter at day boundary
        today = _now_pacific().strftime("%Y-%m-%d")
        if state.get("_counter_date") != today:
            state["tasks_completed_today"] = 0
            state["_counter_date"] = today
        state["tasks_completed_today"] = state.get(
            "tasks_completed_today", 0) + 1
        self._write_state(state)

    def fail(self, task_id: str, error: str = "Unknown error"):
        """Mark a task as failed and move to failed dir."""
        active_path = self.active_dir / f"{task_id}.yaml"
        task = self._read_task(active_path)
        if not task:
            return

        now = _now_iso()
        task["status"] = "failed"
        task["completed_at"] = now
        task["error"] = error

        if task.get("started_at"):
            try:
                dt_start = datetime.fromisoformat(task["started_at"])
                dt_end = datetime.fromisoformat(now)
                task["duration_seconds"] = int(
                    (dt_end - dt_start).total_seconds())
            except (ValueError, TypeError):
                pass

        self._write_task(self.failed_dir, task)
        active_path.unlink(missing_ok=True)

        _compost_intention_for_task(task, reason=f"task_failed: {error[:120]}")

        self._append_event({
            "event": "failed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "error": error[:500],
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        state["mode"] = "idle"
        state["current_task"] = None
        self._update_queue_depth(state)
        state["last_activity"] = now
        self._write_state(state)

    def preempt(self, task_id: str):
        """Re-queue a preemptible task back to pending."""
        active_path = self.active_dir / f"{task_id}.yaml"
        task = self._read_task(active_path)
        if not task:
            return

        now = _now_iso()
        task["status"] = "pending"
        task["started_at"] = None

        self._write_task(self.pending_dir, task)
        active_path.unlink(missing_ok=True)

        self._append_event({
            "event": "preempted",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
        })

        state = self._read_state()
        state["mode"] = "idle"
        state["current_task"] = None
        self._update_queue_depth(state)
        state["last_activity"] = now
        self._write_state(state)

    def cancel(self, task_id: str, reason: str = ""):
        """Cancel a pending task (removes from queue)."""
        pending_path = self.pending_dir / f"{task_id}.yaml"
        if not pending_path.exists():
            return False

        task = self._read_task(pending_path)
        pending_path.unlink(missing_ok=True)

        # Compost the intention so it doesn't linger in-flight forever.
        # fail() already does this; cancel() was missing the cascade and
        # left intentions orphaned (2026-04-23: 3 psych-rerun-batch zombies).
        if task:
            _compost_intention_for_task(
                task, reason=f"task_cancelled: {reason or 'no reason given'}"
            )

        self._append_event({
            "event": "cancelled",
            "id": task_id,
            "timestamp": _now_iso(),
            "task_type": task.get("task_type", "unknown") if task else "unknown",
            "reason": reason,
        })

        state = self._read_state()
        self._update_queue_depth(state)
        self._write_state(state)
        return True

    def should_preempt(self, current_task: dict) -> bool:
        """Check if a higher-priority task is waiting."""
        if not current_task.get("preemptible", False):
            return False

        current_pri = current_task.get("priority", Priority.NORMAL)

        for p in self.pending_dir.glob("*.yaml"):
            t = self._read_task(p)
            if t and t.get("priority", Priority.NORMAL) < current_pri:
                return True
        return False

    def get_state(self) -> dict:
        """Return current queue state snapshot."""
        state = self._read_state()
        self._update_queue_depth(state)
        return state

    def get_pending(self) -> list[dict]:
        """Return all pending tasks, sorted by priority."""
        tasks = []
        for p in sorted(self.pending_dir.glob("*.yaml")):
            t = self._read_task(p)
            if t:
                tasks.append(t)
        tasks.sort(key=lambda t: (
            t.get("priority", Priority.NORMAL),
            t.get("submitted_at", ""),
        ))
        return tasks

    def get_active(self) -> dict | None:
        """Return the currently active task, if any."""
        active_files = list(self.active_dir.glob("*.yaml"))
        if active_files:
            return self._read_task(active_files[0])
        return None

    def get_recent_completed(self, limit: int = 10) -> list[dict]:
        """Return recently completed tasks."""
        files = sorted(
            self.completed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_recent_failed(self, limit: int = 10) -> list[dict]:
        """Return recently failed tasks."""
        files = sorted(
            self.failed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_history(self, limit: int = 50) -> list[dict]:
        """Return recent events from JSONL history."""
        if not self.history_path.exists():
            return []
        events = []
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events[-limit:]

    def cleanup(self, max_age_hours: float = 48):
        """Remove completed tasks older than max_age_hours."""
        cutoff = _now_pacific() - timedelta(hours=max_age_hours)
        removed = 0
        for p in self.completed_dir.glob("*.yaml"):
            try:
                mtime = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=PACIFIC)
                if mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
        return removed

    def update_runner_state(self, **kwargs):
        """Update runner-specific fields in state.json (mode, pid, model)."""
        state = self._read_state()
        state.update(kwargs)
        self._write_state(state)

    def pause(self):
        """Pause the queue — runner finishes current task then stops claiming."""
        state = self._read_state()
        state["paused"] = True
        self._write_state(state)
        self._append_event({"event": "paused", "timestamp": _now_iso()})

    def resume(self):
        """Resume a paused queue."""
        state = self._read_state()
        state["paused"] = False
        self._write_state(state)
        self._append_event({"event": "resumed", "timestamp": _now_iso()})

    def is_paused(self) -> bool:
        return self._read_state().get("paused", False)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: gpu_queue.py <command> [args...]", file=sys.stderr)
        print("Commands: submit, status, pending, cancel, cleanup, history",
              file=sys.stderr)
        sys.exit(1)

    q = GPUQueue()
    cmd = sys.argv[1]

    if cmd == "status":
        state = q.get_state()
        active = q.get_active()
        pending = q.get_pending()
        print(json.dumps({
            "mode": state.get("mode", "unknown"),
            "current_task": active,
            "queue_depth": len(pending),
            "pending": pending,
            "current_model": state.get("current_model"),
            "tasks_completed_today": state.get("tasks_completed_today", 0),
        }, indent=2, default=str))

    elif cmd == "pending":
        for t in q.get_pending():
            pri = t.get("priority", "?")
            tt = t.get("task_type", "?")
            tid = t.get("id", "?")
            model = t.get("model") or "-"
            sub = t.get("submitted_by", "?")
            print(f"  [{pri:>3}] {tid}  {tt:20s}  model={model}  by={sub}")

    elif cmd == "cancel":
        if len(sys.argv) < 3:
            print("Usage: gpu_queue.py cancel <task_id> [reason]",
                  file=sys.stderr)
            sys.exit(1)
        reason = sys.argv[3] if len(sys.argv) > 3 else ""
        if q.cancel(sys.argv[2], reason):
            print(f"Cancelled: {sys.argv[2]}")
        else:
            print(f"Not found in pending: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "cleanup":
        max_age = 48
        if len(sys.argv) >= 4 and sys.argv[2] == "--max-age":
            max_age = float(sys.argv[3])
        removed = q.cleanup(max_age_hours=max_age)
        print(f"Removed {removed} old completed task(s)")

    elif cmd == "history":
        limit = 20
        if len(sys.argv) >= 4 and sys.argv[2] == "--limit":
            limit = int(sys.argv[3])
        for event in q.get_history(limit):
            print(json.dumps(event))

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Coordinator auto-discovery (entry points).
#
# Plugins declare themselves via:
#
#     [project.entry-points."agents_core.gpu_coordinators"]
#     <name> = "<module_that_calls_register_coordinator_on_import>"
#
# At module load we iterate matching entry points and import them. Each
# plugin is expected to call register_coordinator() as a side effect of its
# own import (that is the plugin's contract, not ours — we only load).
#
# Failures are swallowed: a broken plugin must never prevent agents_core.gpu
# from loading. The queue degrades to no-coordinator mode.
#
# Callers can still register_coordinator() manually at any time; entry-point
# discovery is a convenience, not the only supported path.
# ---------------------------------------------------------------------------

def _autoload_coordinators() -> None:
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover — 3.10+ always has this
        return
    try:
        eps = entry_points(group="agents_core.gpu_coordinators")
    except Exception as e:
        _coord_log.debug(f"entry_points lookup failed: {e}")
        return
    for ep in eps:
        try:
            ep.load()  # side-effect: plugin registers itself
        except Exception as e:
            _coord_log.warning(
                f"coordinator plugin {ep.name!r} failed to load: {e}")


_autoload_coordinators()


if __name__ == "__main__":
    main()
