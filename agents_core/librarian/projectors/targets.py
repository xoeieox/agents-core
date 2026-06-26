"""current-targets-state projector.

Reads /srv/lapis/targets/*.yaml (filtered to pm_bound: true) and joins with
mem keys (pm/cursor/<tid>, pm/dispatched/<tid>, pm/outstanding-brief/<tid>)
to produce the exact field shape that claude-view's
extract_targets_from_answer + direct_read_targets_fallback already produce.

The contract is *consumer-frozen*: every key in the per-target dict below is
load-bearing on the Rust side. Do not rename, do not remove, do not change
types in this slice. New fields are additive; consumers ignore unknown keys.

mem.get() returns a dict with schema:
  {"key": str, "content": str, "tags": str, "source": str,
   "created_at": str, "updated_at": str}
The value field is "content". The defensive isinstance branches cover both
the dict-row return and any future bare-string return; do not remove them.

Read-only discipline: projectors must never call mem.set() or write any
file. This is enforced by convention and exercised in tests.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents_core.room_paths import room_path

TARGETS_DIR = room_path("targets")


def current_targets_state(scope: dict, freshness: int, policy: str) -> dict:
    """Return {"targets": [TargetSummary, ...], "generated_at": "<iso>"}.

    Per-target shape (ALL keys required, in this exact form):
      target_id, title, pm_repo, pm_authority, paused, cursor,
      dispatched_total, dispatched_pending, outstanding_brief_id,
      tags, urgency, category
    """
    targets: list[dict[str, Any]] = []
    for yaml_path in sorted(TARGETS_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not data.get("pm_bound"):
            continue
        tid = data.get("id") or yaml_path.stem
        dispatched = _get_dispatched(tid)
        targets.append({
            "target_id": tid,
            "title": data.get("title", tid),
            "pm_repo": data.get("pm_repo"),
            "pm_authority": data.get("pm_authority"),
            "paused": bool(data.get("paused", False)),
            "cursor": _get_cursor(tid),
            "dispatched_total": len(dispatched),
            "dispatched_pending": sum(1 for d in dispatched if d.get("status") == "pending"),
            "outstanding_brief_id": _get_outstanding_brief(tid),
            "tags": data.get("tags", []),
            "urgency": data.get("urgency"),
            "category": data.get("category"),
        })
    return {
        "targets": targets,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Mem helpers — each wraps _mem_get so tests can monkeypatch _mem_get once
# ---------------------------------------------------------------------------

def _get_dispatched(tid: str) -> list[dict]:
    try:
        raw = _mem_get(f"pm/dispatched/{tid}")
    except Exception:
        return []
    if raw is None:
        return []
    try:
        v = json.loads(raw["content"]) if isinstance(raw, dict) else json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def _get_cursor(tid: str) -> str | None:
    try:
        raw = _mem_get(f"pm/cursor/{tid}")
    except Exception:
        return None
    if raw is None:
        return None
    return raw["content"] if isinstance(raw, dict) else str(raw)


def _get_outstanding_brief(tid: str) -> str | None:
    try:
        raw = _mem_get(f"pm/outstanding-brief/{tid}")
    except Exception:
        return None
    if raw is None:
        return None
    return raw["content"] if isinstance(raw, dict) else str(raw)


def _mem_get(key: str) -> dict | None:
    """Thin wrapper around MemoryStore.get().

    Lazily instantiates a shared MemoryStore. Tests monkeypatch this
    function directly to avoid touching mem.db.
    """
    return _get_store().get(key)


_store = None


def _get_store():
    global _store
    if _store is None:
        from agents_core.mem import MemoryStore
        _store = MemoryStore()
    return _store
