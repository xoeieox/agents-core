#!/usr/bin/env python3
"""Target Store — persistent work target management for the Conductor agent.

Manages YAML-based targets in /srv/lapis/targets/. Each target tracks a work thread
with status, urgency, work mode, stages, and decay detection.

Usage as library:
    from target_store import TargetStore, SCHEDULE
    store = TargetStore()
    for t in store.active_targets():
        print(f"{t.id}: {t.title} (idle {t.days_idle}d)")

Usage as CLI:
    python3 target_store.py summary          # all targets
    python3 target_store.py decay            # tick decay + show alerts
    python3 target_store.py schedule         # today's schedule-filtered targets
    python3 target_store.py schedule Wednesday  # specific day
"""

import re
import sys
import yaml
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
TARGETS_DIR = Path("/srv/lapis/targets")

URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Target categories
CATEGORY_ACTIVE_WORK = "active-work"  # Erah's personal work — drives nudges
CATEGORY_RESEARCH = "research"        # Local agent orchestration — no nudges

SCHEDULE = {
    "Monday":    {"modes": ["anywhere"], "hours": None, "label": "Family day"},
    "Tuesday":   {"modes": ["library", "anywhere"], "hours": "10 AM-4 PM", "label": "Remote work"},
    "Wednesday": {"modes": ["workshop", "library", "anywhere"], "hours": "9 AM-4:30 PM", "label": "Home"},
    "Thursday":  {"modes": ["library", "anywhere"], "hours": "10 AM-4 PM", "label": "Remote work"},
    "Friday":    {"modes": ["workshop", "library", "anywhere"], "hours": "9 AM-4:30 PM", "label": "Home"},
    "Saturday":  {"modes": ["workshop", "library", "anywhere"], "hours": "9 AM-2 PM", "label": "Home"},
    "Sunday":    {"modes": ["anywhere"], "hours": None, "label": "Family day"},
}

# EOD windows: 30 minutes before end of work hours
EOD_WINDOWS = {
    "Monday":    None,
    "Tuesday":   (15, 30, 16, 0),   # 3:30-4:00 PM
    "Wednesday": (16, 0, 16, 30),    # 4:00-4:30 PM
    "Thursday":  (15, 30, 16, 0),    # 3:30-4:00 PM
    "Friday":    (16, 0, 16, 30),    # 4:00-4:30 PM
    "Saturday":  (13, 30, 14, 0),    # 1:30-2:00 PM
    "Sunday":    None,
}


class Target:
    """Single work target loaded from a YAML file."""

    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = path

    @property
    def id(self) -> str:
        return self.data.get("id", self.path.stem)

    @property
    def title(self) -> str:
        return self.data.get("title", self.id)

    @property
    def status(self) -> str:
        return self.data.get("status", "active")

    @property
    def urgency(self) -> str:
        return self.data.get("urgency", "medium")

    @property
    def category(self) -> str:
        return self.data.get("category", CATEGORY_RESEARCH)

    @property
    def work_mode(self) -> str:
        return self.data.get("work_mode", "anywhere")

    @property
    def days_idle(self) -> int:
        return self.data.get("decay_days", 0)

    @property
    def decay_threshold(self) -> int:
        return self.data.get("decay_threshold", 7)

    @property
    def is_decaying(self) -> bool:
        return self.days_idle >= self.decay_threshold

    @property
    def touched_date(self) -> date | None:
        val = self.data.get("touched")
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            try:
                return date.fromisoformat(val)
            except ValueError:
                return None
        return None

    @property
    def current_stage(self) -> str | None:
        """Return the name of the first active stage, or None."""
        for stage in self.data.get("stages", []):
            if stage.get("status") == "active":
                return stage["name"]
        return None

    @property
    def created_date(self) -> date | None:
        val = self.data.get("created")
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            try:
                return date.fromisoformat(val)
            except ValueError:
                return None
        return None

    @property
    def product(self) -> str:
        return self.data.get("product") or "Research"

    @property
    def stages_progress(self) -> str | None:
        stages = self.data.get("stages", [])
        if not stages:
            return None
        done = sum(1 for s in stages if s.get("status") == "completed")
        return f"{done}/{len(stages)}"

    @property
    def stages_list(self) -> list[dict]:
        out = []
        for stage in self.data.get("stages", []):
            out.append({
                "name": stage.get("name", "—"),
                "status": stage.get("status", "pending"),
                "note": stage.get("note", ""),
            })
        return out

    @property
    def arc_doc_path(self) -> str:
        return f"/srv/lapis/lapis-state/{self.id}.md"

    @property
    def arc_doc_exists(self) -> bool:
        return Path(self.arc_doc_path).is_file()

    def to_dashboard_dict(self) -> dict:
        stages = self.data.get("stages", [])
        total = len(stages)
        done = sum(1 for s in stages if s.get("status") == "completed")
        current = None
        for s in stages:
            if s.get("status") == "active" and current is None:
                current = s.get("name", "—")
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "category": self.category,
            "product": self.product,
            "urgency": self.urgency,
            "work_mode": self.work_mode,
            "current_stage": current,
            "stages_done": done,
            "stages_total": total,
            "decay_days": self.days_idle,
            "decay_threshold": self.decay_threshold,
            "is_decaying": self.is_decaying,
            "touched": str(self.data.get("touched", "")),
            "created": str(self.data.get("created", "")),
            "tags": self.data.get("tags", []),
            "description": self.data.get("description", ""),
            "stages": self.stages_list,
            "pm_bound": self.pm_bound,
            "pm_repo": self.pm_repo,
            "pm_authority": self.pm_authority,
            "paused": self.paused,
            "paused_reason": self.paused_reason,
        }

    # --- Lapis PM fields (set by `lapis-pm bind`) ---
    @property
    def pm_bound(self) -> bool:
        return bool(self.data.get("pm_bound", False))

    @property
    def pm_repo(self) -> str | None:
        return self.data.get("pm_repo")

    @property
    def pm_authority(self) -> str:
        return self.data.get("pm_authority", "advisory")

    @property
    def pr_count(self) -> int:
        """Number of PRs that must merge before auto-land fires. Default 1."""
        return int(self.data.get("pr_count", 1))

    @property
    def paused(self) -> bool:
        return bool(self.data.get("paused", False))

    @property
    def paused_reason(self) -> str | None:
        return self.data.get("paused_reason")

    def set_paused(self, paused: bool, reason: str | None = None):
        """Toggle pause state. Save() must be called by caller."""
        if paused:
            self.data["paused"] = True
            if reason:
                self.data["paused_reason"] = reason
            self.data["paused_at"] = datetime.now(PACIFIC).isoformat(timespec="seconds")
        else:
            self.data["paused"] = False
            self.data.pop("paused_reason", None)
            self.data.pop("paused_at", None)

    def bind_pm(self, repo: str, authority: str = "advisory"):
        """Mark target as PM-managed. Save() must be called by caller."""
        if authority not in ("advisory", "auto"):
            raise ValueError("authority must be 'advisory' or 'auto'")
        self.data["pm_bound"] = True
        self.data["pm_repo"] = repo
        self.data["pm_authority"] = authority

    def unbind_pm(self):
        """Remove PM binding. Save() must be called by caller."""
        self.data["pm_bound"] = False
        self.data.pop("pm_repo", None)
        self.data.pop("pm_authority", None)

    def touch(self, when: date | None = None):
        """Mark target as touched. Resets decay_days to 0."""
        if when is None:
            when = datetime.now(PACIFIC).date()
        self.data["touched"] = when.isoformat()
        self.data["decay_days"] = 0
        self.data["updated"] = when.isoformat()

    def save(self):
        """Write target back to its YAML file."""
        # Convert date objects to strings for clean YAML output
        write_data = {}
        for key, val in self.data.items():
            if isinstance(val, date):
                write_data[key] = val.isoformat()
            else:
                write_data[key] = val
        self.path.write_text(yaml.dump(write_data, default_flow_style=False,
                                       sort_keys=False, allow_unicode=True))


class TargetStore:
    """Manages all targets in a directory."""

    def __init__(self, targets_dir: Path = TARGETS_DIR):
        self.targets_dir = targets_dir
        self.targets_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        target_id: str,
        title: str,
        urgency: str = "medium",
        work_mode: str = "anywhere",
        description: str = "",
        stages: list[dict] | None = None,
        decay_threshold: int = 7,
        category: str = CATEGORY_RESEARCH,
    ) -> Target:
        """Create a new target YAML file. Returns the created Target."""
        path = self.targets_dir / f"{target_id}.yaml"
        if path.exists():
            raise FileExistsError(f"Target {target_id} already exists")

        today = datetime.now(PACIFIC).date().isoformat()
        data = {
            "id": target_id,
            "title": title,
            "status": "active",
            "category": category,
            "urgency": urgency,
            "work_mode": work_mode,
            "created": today,
            "touched": today,
            "updated": today,
            "decay_days": 0,
            "decay_threshold": decay_threshold,
        }
        if description:
            data["description"] = description
        if stages:
            data["stages"] = stages

        target = Target(data, path)
        target.save()
        return target

    def archive(self, target_id: str) -> bool:
        """Set a target's status to 'archived'. Returns True if successful."""
        target = self.get(target_id)
        if target is None:
            return False
        target.data["status"] = "archived"
        target.data["updated"] = datetime.now(PACIFIC).date().isoformat()
        target.save()
        return True

    def load_all(self) -> list[Target]:
        """Load all .yaml files from the targets directory."""
        targets = []
        for path in sorted(self.targets_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text())
                if data and isinstance(data, dict):
                    targets.append(Target(data, path))
            except Exception as e:
                print(f"Warning: failed to load {path.name}: {e}", file=sys.stderr)
        return targets

    def get(self, target_id: str) -> Target | None:
        """Load a single target by ID."""
        path = self.targets_dir / f"{target_id}.yaml"
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text())
            if data and isinstance(data, dict):
                return Target(data, path)
        except Exception:
            pass
        return None

    def active_targets(self, category: str | None = None) -> list[Target]:
        """Return targets with status='active', sorted by urgency (high first).

        If category is given, filter to that category only.
        """
        active = [t for t in self.load_all() if t.status == "active"]
        if category:
            active = [t for t in active if t.category == category]
        active.sort(key=lambda t: URGENCY_ORDER.get(t.urgency, 99))
        return active

    def active_work_targets(self) -> list[Target]:
        """Return active targets in the active-work category (drives nudges)."""
        return self.active_targets(category=CATEGORY_ACTIVE_WORK)

    def research_targets(self) -> list[Target]:
        """Return active targets in the research category (local agent)."""
        return self.active_targets(category=CATEGORY_RESEARCH)

    def filter_by_schedule(self, day: str, category: str | None = None) -> list[Target]:
        """Filter active targets by work_mode appropriate for the given day.

        If category is given, filter to that category only.
        """
        sched = SCHEDULE.get(day)
        if not sched:
            return []
        available_modes = sched["modes"]
        return [t for t in self.active_targets(category=category)
                if t.work_mode in available_modes]

    def tick_daily_decay(self, today: date | None = None) -> list[Target]:
        """Increment decay_days for untouched targets, reset for touched ones.

        Idempotent within a day: checks the touched date before incrementing.
        Returns list of targets that are at or past their decay threshold.
        """
        if today is None:
            today = datetime.now(PACIFIC).date()

        alerts = []
        for target in self.load_all():
            if target.status != "active":
                continue

            touched = target.touched_date
            if touched and touched >= today:
                # Touched today — ensure decay is 0
                if target.days_idle != 0:
                    target.data["decay_days"] = 0
                    target.data["updated"] = today.isoformat()
                    target.save()
            else:
                # Not touched today — compute days since last touch
                if touched:
                    delta = (today - touched).days
                    target.data["decay_days"] = delta
                else:
                    # No touch date recorded; increment by 1
                    target.data["decay_days"] = target.days_idle + 1
                target.data["updated"] = today.isoformat()
                target.save()

            if target.is_decaying:
                alerts.append(target)

        return alerts

    def decay_alerts(self) -> list[Target]:
        """Return active targets where decay_days >= decay_threshold."""
        return [t for t in self.active_targets() if t.is_decaying]

    def status_summary(self) -> str:
        """Return a markdown summary of all active targets, grouped by category."""
        targets = self.active_targets()
        if not targets:
            return "(no active targets)"

        active_work = [t for t in targets if t.category == CATEGORY_ACTIVE_WORK]
        research = [t for t in targets if t.category == CATEGORY_RESEARCH]

        lines = []
        if active_work:
            lines.append("### Active Work")
            for t in active_work:
                stage = t.current_stage or "—"
                decay_flag = " **DECAYING**" if t.is_decaying else ""
                lines.append(
                    f"- [{t.urgency.upper()}] **{t.title}** ({t.work_mode}) "
                    f"— {stage} | idle {t.days_idle}d{decay_flag}"
                )
        if research:
            lines.append("### Research Threads")
            for t in research:
                stage = t.current_stage or "—"
                decay_flag = " **DECAYING**" if t.is_decaying else ""
                lines.append(
                    f"- [{t.urgency.upper()}] **{t.title}** ({t.work_mode}) "
                    f"— {stage} | idle {t.days_idle}d{decay_flag}"
                )
        return "\n".join(lines)

    def schedule_summary(self, day: str) -> str:
        """Return markdown summary filtered by today's schedule."""
        sched = SCHEDULE.get(day)
        if not sched:
            return f"(no schedule data for {day})"

        filtered = self.filter_by_schedule(day)
        header = f"**{day}** | {sched['label']} | {sched['hours'] or 'No scheduled hours'}"
        header += f"\n**Available modes:** {', '.join(sched['modes'])}\n"

        if not filtered:
            return header + "\n(no targets match today's schedule)"

        lines = [header]
        for t in filtered:
            stage = t.current_stage or "—"
            lines.append(f"- [{t.urgency.upper()}] **{t.title}** — {stage}")
        return "\n".join(lines)


    def complete_stage(self, target_id: str, stage_name: str) -> bool:
        """Mark a stage as completed and activate the next pending stage.

        Touches the target (resets decay). If all stages are now completed,
        sets target status to 'completed'. Returns True if the stage was found
        and marked.
        """
        target = self.get(target_id)
        if target is None:
            return False

        stages = target.data.get("stages", [])
        found = False
        for stage in stages:
            if stage.get("name") == stage_name and stage.get("status") != "completed":
                stage["status"] = "completed"
                found = True
                break

        if not found:
            return False

        # Activate the next pending stage
        for stage in stages:
            if stage.get("status") in (None, "pending"):
                stage["status"] = "active"
                break

        # If all stages completed, mark the target done
        if all(s.get("status") == "completed" for s in stages):
            target.data["status"] = "completed"

        target.touch()
        target.save()
        return True

    def advance_stage(self, target_id: str, note: str | None = None) -> str | None:
        """Complete the current active stage and activate the next one.

        Convenience wrapper: finds whichever stage is active, completes it,
        and advances. Returns the name of the completed stage, or None if
        no active stage was found. Optionally appends a note to the stage.
        """
        target = self.get(target_id)
        if target is None:
            return None

        stages = target.data.get("stages", [])
        completed_name = None
        for stage in stages:
            if stage.get("status") == "active":
                completed_name = stage["name"]
                stage["status"] = "completed"
                if note:
                    stage["note"] = note
                break

        if completed_name is None:
            return None

        # Activate next pending stage
        for stage in stages:
            if stage.get("status") in (None, "pending"):
                stage["status"] = "active"
                break

        # If all stages completed, mark the target done
        if all(s.get("status") == "completed" for s in stages):
            target.data["status"] = "completed"

        target.touch()
        target.save()
        return completed_name

    def stale_but_relevant(self, findings_text: str) -> list[dict]:
        """Find targets that are idle but relevant to tonight's findings.

        Cross-references tonight's output text against target titles,
        descriptions, and stage names. Returns targets sorted by relevance.
        """
        findings_lower = findings_text.lower()
        results = []

        for target in self.active_targets():
            if target.days_idle < 1:
                continue  # recently touched, not stale

            # Build a keyword set from the target
            keywords = set()
            title_words = re.findall(r'[a-z]{4,}', target.title.lower())
            keywords.update(title_words)

            # Add stage names
            for stage in target.data.get("stages", []):
                stage_words = re.findall(
                    r'[a-z]{4,}', stage.get("name", "").lower()
                )
                keywords.update(stage_words)

            # Add description words if present
            desc = target.data.get("description", "")
            if desc:
                desc_words = re.findall(r'[a-z]{4,}', desc.lower())
                keywords.update(desc_words)

            # Filter common words
            stops = {
                "this", "that", "with", "from", "into", "have", "been",
                "more", "some", "implementation", "system", "update",
                "create", "design", "build", "phase", "stage", "active",
            }
            keywords -= stops

            if not keywords:
                continue

            # Count how many target keywords appear in findings
            hits = sum(1 for kw in keywords if kw in findings_lower)
            if hits == 0:
                continue

            relevance = hits / len(keywords)
            results.append({
                "target_id": target.id,
                "title": target.title,
                "days_idle": target.days_idle,
                "decay_threshold": target.decay_threshold,
                "current_stage": target.current_stage,
                "relevance": round(relevance, 3),
                "keyword_hits": hits,
                "total_keywords": len(keywords),
            })

        results.sort(key=lambda r: (-r["relevance"], -r["days_idle"]))
        return results

    def staleness_report_markdown(self, findings_text: str) -> str:
        """Format staleness report for injection into LLM prompts."""
        matches = self.stale_but_relevant(findings_text)
        if not matches:
            return "**Targets:** no idle targets relevant to tonight's findings"

        lines = [f"**Targets:** {len(matches)} idle target(s) relevant to tonight's work"]
        for m in matches[:5]:
            decay_warn = " **DECAYING**" if m["days_idle"] >= m["decay_threshold"] else ""
            lines.append(
                f"- **{m['title']}** — idle {m['days_idle']}d, "
                f"stage: {m['current_stage'] or '—'}, "
                f"relevance: {m['relevance']:.0%}{decay_warn}"
            )
        return "\n".join(lines)


def main():
    """CLI interface for target_store."""
    if len(sys.argv) < 2:
        print("Usage: target_store.py <summary|decay|schedule|create|archive> [args]")
        sys.exit(1)

    command = sys.argv[1]
    store = TargetStore()

    if command == "summary":
        print(store.status_summary())

    elif command == "decay":
        alerts = store.tick_daily_decay()
        if alerts:
            print(f"Decay alerts ({len(alerts)}):")
            for t in alerts:
                print(f"  {t.id}: {t.days_idle}d idle (threshold: {t.decay_threshold})")
        else:
            print("No decay alerts.")

    elif command == "schedule":
        day = sys.argv[2] if len(sys.argv) > 2 else datetime.now(PACIFIC).strftime("%A")
        print(store.schedule_summary(day))

    elif command == "create":
        # target_store.py create <id> <title> [urgency] [work_mode] [decay_threshold]
        if len(sys.argv) < 4:
            print("Usage: target_store.py create <id> <title> [urgency] [work_mode] [decay_threshold]",
                  file=sys.stderr)
            sys.exit(1)
        tid = sys.argv[2]
        title = sys.argv[3]
        urgency = sys.argv[4] if len(sys.argv) > 4 else "medium"
        work_mode = sys.argv[5] if len(sys.argv) > 5 else "anywhere"
        decay_threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 7
        try:
            t = store.create(tid, title, urgency=urgency, work_mode=work_mode,
                             decay_threshold=decay_threshold)
            print(f"Created target: {t.id} — {t.title}")
        except FileExistsError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif command == "archive":
        if len(sys.argv) < 3:
            print("Usage: target_store.py archive <id>", file=sys.stderr)
            sys.exit(1)
        tid = sys.argv[2]
        if store.archive(tid):
            print(f"Archived target: {tid}")
        else:
            print(f"Target not found: {tid}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
