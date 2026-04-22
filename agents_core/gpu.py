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
# Intention registry bridge (Lapis Ops Layer, 2026-04-22).
# Soft-coupled: the intention_registry module lives under /srv/agents/scripts
# for now. If the import fails (registry disabled, fresh install), the queue
# falls back to pre-registry behavior.
# ---------------------------------------------------------------------------

def _load_intention_registry():
    """Lazy-load the intention_registry module, returning None on failure."""
    try:
        import importlib
        import sys as _sys
        scripts_dir = "/srv/agents/scripts"
        if scripts_dir not in _sys.path:
            _sys.path.append(scripts_dir)
        return importlib.import_module("intention_registry")
    except Exception:
        return None


def _project_intention_for_task(task: dict):
    """Run intention projection for a pending task. Returns ProjectionResult or None."""
    reg = _load_intention_registry()
    if reg is None:
        return None
    try:
        return reg.project_from_task(
            task,
            projected_by=task.get("submitted_by") or "unknown",
            proposed_change=task.get("description") or task.get("task_type"),
            target_heading=(task.get("payload") or {}).get("target_heading"),
        )
    except Exception:
        return None


def _attach_task_id_to_intention(intention_id: str, task_id: str) -> None:
    """Record the GPU task id on a freshly-projected intention."""
    reg = _load_intention_registry()
    if reg is None:
        return
    try:
        reg.attach_task_id(intention_id, task_id)
    except Exception:
        pass


def _manifest_intention_for_task(task: dict) -> None:
    """Move the task's intention from in-flight to manifested."""
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _load_intention_registry()
    if reg is None:
        return
    try:
        reg.manifest(intention_id, linked_task_id=task.get("id"))
    except Exception:
        pass


def _compost_intention_for_task(task: dict, reason: str) -> None:
    """Move the task's intention to composted when a task fails."""
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _load_intention_registry()
    if reg is None:
        return
    try:
        reg.compost(intention_id, reason=reason)
    except Exception:
        pass


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

    def submit(self, task_dict: dict) -> str | None:
        """Submit a task to the queue. Returns task ID, or None when the
        intention registry says the work is already done (match_manifested
        within lookback). When a second agent reinforces an in-flight
        intention, returns the shared task_id without queuing new GPU work.

        Opt out of the registry by setting payload['_ignore_intention_registry']
        to True (e.g. for stochastic-variance reruns).
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
            projection = _project_intention_for_task(task)

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
            if projection.decision == "match_reinforce":
                # Second agent joins work in flight; return the shared task_id
                # so the caller can monitor/poll the same work.
                return shared
            # match_manifested — prior work already done within lookback.
            # Return None; caller can read the prior output if it wants.
            return None

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

        if projection is not None:
            _attach_task_id_to_intention(projection.intention.intention_id, task["id"])

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


if __name__ == "__main__":
    main()
