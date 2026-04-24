#!/usr/bin/env python3
"""ClaudeQueue — file-based queue for `claude -p` subprocess tasks.

Separate from agents_core.gpu.GPUQueue because Qwen-shaped semantics
(preempt / stop_llm_server / model-affinity tiebreak / intention-registry
coordinator) do not apply to API-backed shaped-agent subprocesses. See
/srv/lapis/planning/specs/agents-core-claude-queue.md for the design rationale.

Storage:
    /srv/lapis/claude-queue/
      pending/          YAML task files waiting to run
      active/           Up to CLAUDE_QUEUE_WORKERS YAMLs — in-flight
      completed/        Done tasks
      failed/           Failed tasks
      history.jsonl     Append-only event log
      state.json        Snapshot: in-flight list, queue depth

Usage:
    from agents_core.claude_queue import ClaudeQueue
    q = ClaudeQueue()
    task_id = q._generate_id(slug="fixer-target_x")
    q.submit({
        "task_type": "subprocess",
        "priority": 10,
        "timeout_seconds": 600,
        "submitted_by": "lapis-pm",
        "model": "sonnet",
        "payload": {"command": "...", "spec_path": "..."},
    }, task_id=task_id)
"""

import fcntl
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from agents_core.gpu import PACIFIC, Priority, _now_iso, _now_pacific  # noqa: F401

CLAUDE_QUEUE_DIR = Path("/srv/lapis/claude-queue")

REQUIRED_FIELDS = {"task_type"}

TASK_DEFAULTS = {
    "priority": Priority.NORMAL,
    "submitted_by": "unknown",
    "timeout_seconds": 300,
    "notify": False,
    "model": None,
    "payload": {},
    "description": None,
    "base_branch": "main",
    "worktree_required": False,
    "status": "pending",
    "started_at": None,
    "completed_at": None,
    "duration_seconds": None,
    "error": None,
}


class ClaudeQueue:
    """File-based queue for `claude -p` subprocess tasks."""

    def __init__(self, queue_dir: str | Path = CLAUDE_QUEUE_DIR):
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

    def _generate_id(self, slug: str = "") -> str:
        # Single _now_pacific() call so the seconds and microseconds come
        # from the same instant. GPUQueue's _generate_id calls it twice,
        # which can straddle a second boundary (see spec §Interface).
        now = _now_pacific()
        ts = now.strftime("%Y%m%d_%H%M%S")
        usec = now.strftime("%f")[:4]
        clean = (slug or "task").replace(" ", "").replace("-", "").replace("/", "")[:24]
        return f"claude_{ts}_{usec}_{clean}"

    def _append_event(self, event: dict):
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
            "in_flight": [],
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

    def _refresh_state(self, state: dict):
        state["queue_depth"] = len(list(self.pending_dir.glob("*.yaml")))
        state["in_flight"] = sorted(
            p.stem for p in self.active_dir.glob("*.yaml")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, task_dict: dict, task_id: str | None = None) -> str:
        """Submit a task. Returns its task_id.

        `task_id` override eliminates a write-back race: the shaper generates
        the id via `_generate_id()`, writes it into the spec JSON file along
        with `worktree_required`/`base_branch`, then submits with that id.
        Without this, the shaper would have to re-write the spec after
        `submit()` returns, and the runner could claim the task in between —
        seeing a spec missing `task_id` and `worktree_required`. See spec
        §Shaper routing.
        """
        missing = REQUIRED_FIELDS - set(task_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        task = dict(TASK_DEFAULTS)
        task.update(task_dict)
        task["id"] = task_id or task.get("id") or self._generate_id(
            slug=task.get("description") or task["task_type"]
        )
        task["submitted_at"] = _now_iso()
        task["status"] = "pending"

        self._write_task(self.pending_dir, task)

        self._append_event({
            "event": "submitted",
            "id": task["id"],
            "timestamp": task["submitted_at"],
            "task_type": task["task_type"],
            "priority": task["priority"],
            "model": task.get("model"),
            "submitted_by": task.get("submitted_by", "unknown"),
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = task["submitted_at"]
        self._write_state(state)

        return task["id"]

    def claim(self) -> dict | None:
        """Claim the highest-priority pending task. Returns task dict or None.

        Atomic move from pending/ to active/ via `os.replace` on the source
        YAML. If multiple runner-daemon workers race to claim, only one
        succeeds per file because `_write_task` uses atomic replace; the
        losers find a missing source file and try the next task.

        No model-affinity tiebreak (Qwen-specific), no preempt. Priority ASC,
        then submitted_at ASC.
        """
        pending_files = sorted(self.pending_dir.glob("*.yaml"))
        if not pending_files:
            return None

        tasks = []
        for p in pending_files:
            t = self._read_task(p)
            if t:
                t["_path"] = str(p)
                tasks.append(t)

        if not tasks:
            return None

        def sort_key(t):
            return (
                t.get("priority", Priority.NORMAL),
                t.get("submitted_at", ""),
            )

        tasks.sort(key=sort_key)

        for chosen in tasks:
            src_path = Path(chosen.pop("_path"))
            chosen["status"] = "running"
            chosen["started_at"] = _now_iso()
            self._write_task(self.active_dir, chosen)
            try:
                src_path.unlink()
            except FileNotFoundError:
                # Another worker claimed it first — revert our active write.
                (self.active_dir / f"{chosen['id']}.yaml").unlink(missing_ok=True)
                continue

            self._append_event({
                "event": "claimed",
                "id": chosen["id"],
                "timestamp": chosen["started_at"],
                "task_type": chosen["task_type"],
                "priority": chosen["priority"],
                "model": chosen.get("model"),
            })

            state = self._read_state()
            self._refresh_state(state)
            state["last_activity"] = chosen["started_at"]
            self._write_state(state)

            return chosen

        return None

    def complete(self, task_id: str, output_path: str | None = None,
                 result_summary: str | None = None):
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

        self._append_event({
            "event": "completed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "output_path": output_path,
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = now
        today = _now_pacific().strftime("%Y-%m-%d")
        if state.get("_counter_date") != today:
            state["tasks_completed_today"] = 0
            state["_counter_date"] = today
        state["tasks_completed_today"] = state.get(
            "tasks_completed_today", 0) + 1
        self._write_state(state)

    def fail(self, task_id: str, error: str = "Unknown error"):
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

        self._append_event({
            "event": "failed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "error": error[:500],
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = now
        self._write_state(state)

    def cancel(self, task_id: str, reason: str = "") -> bool:
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
        self._refresh_state(state)
        self._write_state(state)
        return True

    def status(self) -> dict:
        state = self._read_state()
        self._refresh_state(state)
        return {
            "depth": state["queue_depth"],
            "in_flight": state["in_flight"],
        }

    def get_pending(self) -> list[dict]:
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

    def get_active(self) -> list[dict]:
        return [t for p in self.active_dir.glob("*.yaml")
                if (t := self._read_task(p))]

    def get_recent_completed(self, limit: int = 10) -> list[dict]:
        files = sorted(
            self.completed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_recent_failed(self, limit: int = 10) -> list[dict]:
        files = sorted(
            self.failed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_history(self, limit: int = 50) -> list[dict]:
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

    def cleanup(self, max_age_hours: float = 48) -> int:
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


def main():
    if len(sys.argv) < 2:
        print("Usage: claude_queue.py <status|pending|cancel|cleanup|history>",
              file=sys.stderr)
        sys.exit(1)

    q = ClaudeQueue()
    cmd = sys.argv[1]

    if cmd == "status":
        print(json.dumps({
            **q.status(),
            "active": q.get_active(),
            "pending": q.get_pending(),
        }, indent=2, default=str))

    elif cmd == "pending":
        for t in q.get_pending():
            pri = t.get("priority", "?")
            tt = t.get("task_type", "?")
            tid = t.get("id", "?")
            model = t.get("model") or "-"
            sub = t.get("submitted_by", "?")
            print(f"  [{pri:>3}] {tid}  {tt:12s}  model={model}  by={sub}")

    elif cmd == "cancel":
        if len(sys.argv) < 3:
            print("Usage: claude_queue.py cancel <task_id> [reason]",
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
