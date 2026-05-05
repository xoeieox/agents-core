"""Agent observation substrate — append-only per-agent-per-date JSONL.

Storage: /srv/lapis/agent-observations/<agent_id>/<YYYY-MM-DD>.jsonl
Override root via AGENT_OBSERVATIONS_ROOT env var (primarily for tests).

Entry schema (each JSONL line):
    {
        agent_id, session_id, timestamp (ISO8601 UTC), observation_type,
        context, content, target_id, tags, intervention_shape, extra
    }

observation_type: friction | decision | lesson | anomaly | intervention
intervention_shape: question | pointer | counter-example | frame-shift | constraint | why-trace
    (required iff observation_type == "intervention")
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/srv/lapis/agent-observations")

VALID_OBSERVATION_TYPES = frozenset({
    "friction", "decision", "lesson", "anomaly", "intervention"
})
VALID_INTERVENTION_SHAPES = frozenset({
    "question", "pointer", "counter-example", "frame-shift", "constraint", "why-trace"
})


def root() -> Path:
    """Return the resolved observations root, honoring AGENT_OBSERVATIONS_ROOT env override.
    Default: /srv/lapis/agent-observations/"""
    override = os.environ.get("AGENT_OBSERVATIONS_ROOT")
    return Path(override) if override else DEFAULT_ROOT


def record(
    agent_id: str,
    observation_type: str,
    context: str,
    content: str,
    *,
    session_id: str | None = None,
    target_id: str | None = None,
    tags: list[str] | None = None,
    intervention_shape: str | None = None,
    extra: dict | None = None,
    now: datetime | None = None,
) -> Path:
    """Append one observation entry to the per-agent-per-date JSONL file.
    Returns the path written to."""
    if observation_type not in VALID_OBSERVATION_TYPES:
        raise ValueError(
            f"observation_type must be one of {sorted(VALID_OBSERVATION_TYPES)}, "
            f"got {observation_type!r}"
        )
    if observation_type == "intervention":
        if intervention_shape is None:
            raise ValueError(
                "intervention_shape is required when observation_type == 'intervention'"
            )
        if intervention_shape not in VALID_INTERVENTION_SHAPES:
            raise ValueError(
                f"intervention_shape must be one of {sorted(VALID_INTERVENTION_SHAPES)}, "
                f"got {intervention_shape!r}"
            )
    else:
        if intervention_shape is not None:
            raise ValueError(
                f"intervention_shape is only valid when observation_type == 'intervention', "
                f"got observation_type={observation_type!r}"
            )

    ts = now if now is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    ts_utc = ts.astimezone(timezone.utc)
    ts_iso = ts_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    date_str = ts_utc.strftime("%Y-%m-%d")

    entry: dict = {
        "agent_id": agent_id,
        "session_id": session_id,
        "timestamp": ts_iso,
        "observation_type": observation_type,
        "context": context,
        "content": content,
        "target_id": target_id,
        "tags": list(tags) if tags else [],
        "intervention_shape": intervention_shape,
        "extra": extra,
    }

    obs_root = root()
    agent_dir = obs_root / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / f"{date_str}.jsonl"

    # O_APPEND on POSIX guarantees atomic appends for writes under the pipe-buffer
    # threshold (~4 KiB on Linux). Callers accept best-effort interleave-safety
    # for larger lines; each line is independently parseable.
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)

    return path


def search(
    *,
    agent_id: str | None = None,
    observation_type: str | None = None,
    target_id: str | None = None,
    tags_any: list[str] | None = None,
    tags_all: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    substring: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read across observation files matching filters. Streams JSONL files;
    does not load everything into memory. Returns entries sorted by timestamp ascending."""
    obs_root = root()
    if not obs_root.exists():
        return []

    substring_lower = substring.lower() if substring else None

    # Normalise since/until to UTC-aware
    since_aware: datetime | None = None
    if since is not None:
        since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)

    until_aware: datetime | None = None
    if until is not None:
        until_aware = until if until.tzinfo else until.replace(tzinfo=timezone.utc)

    # Determine which agent directories to scan
    if agent_id is not None:
        agent_dirs = [obs_root / agent_id]
    else:
        try:
            agent_dirs = sorted(p for p in obs_root.iterdir() if p.is_dir())
        except OSError:
            return []

    results: list[dict] = []

    for agent_dir in agent_dirs:
        if not agent_dir.exists():
            continue
        try:
            jsonl_files = sorted(agent_dir.glob("*.jsonl"))
        except OSError:
            continue

        for jsonl_path in jsonl_files:
            try:
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                "skipping malformed JSONL line in %s", jsonl_path
                            )
                            continue
                        if not isinstance(entry, dict):
                            continue

                        # Filter: observation_type
                        if (observation_type is not None
                                and entry.get("observation_type") != observation_type):
                            continue

                        # Filter: target_id
                        if target_id is not None and entry.get("target_id") != target_id:
                            continue

                        # Filter: tags_any (OR — at least one tag matches)
                        if tags_any is not None:
                            entry_tags = set(entry.get("tags") or [])
                            if not any(t in entry_tags for t in tags_any):
                                continue

                        # Filter: tags_all (AND — all tags must be present)
                        if tags_all is not None:
                            entry_tags = set(entry.get("tags") or [])
                            if not all(t in entry_tags for t in tags_all):
                                continue

                        # Filter: since
                        if since_aware is not None:
                            ts_str = entry.get("timestamp", "")
                            try:
                                entry_ts = datetime.fromisoformat(ts_str)
                                if entry_ts.tzinfo is None:
                                    entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                                if entry_ts < since_aware:
                                    continue
                            except (ValueError, TypeError):
                                continue

                        # Filter: until
                        if until_aware is not None:
                            ts_str = entry.get("timestamp", "")
                            try:
                                entry_ts = datetime.fromisoformat(ts_str)
                                if entry_ts.tzinfo is None:
                                    entry_ts = entry_ts.replace(tzinfo=timezone.utc)
                                if entry_ts > until_aware:
                                    continue
                            except (ValueError, TypeError):
                                continue

                        # Filter: substring (case-insensitive on content + context)
                        if substring_lower is not None:
                            content_lower = (entry.get("content") or "").lower()
                            context_lower = (entry.get("context") or "").lower()
                            if (substring_lower not in content_lower
                                    and substring_lower not in context_lower):
                                continue

                        results.append(entry)
            except OSError:
                continue

    results.sort(key=lambda e: e.get("timestamp", ""))

    if limit is not None:
        results = results[:limit]

    return results


# ---------------------------------------------------------------------------
# CLI entry-point  (python -m agents_core.observations [record|search] ...)
# ---------------------------------------------------------------------------

def _cli_record(args: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="observations record")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--type", dest="observation_type", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--content", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--target-id")
    parser.add_argument("--tag", dest="tags", action="append", default=[])
    parser.add_argument("--intervention-shape")
    parser.add_argument("--extra-json")
    ns = parser.parse_args(args)

    extra: dict | None = None
    if ns.extra_json:
        extra = json.loads(ns.extra_json)

    path = record(
        agent_id=ns.agent_id,
        observation_type=ns.observation_type,
        context=ns.context,
        content=ns.content,
        session_id=ns.session_id,
        target_id=ns.target_id,
        tags=ns.tags or None,
        intervention_shape=ns.intervention_shape,
        extra=extra,
    )
    print(str(path))


def _cli_search(args: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="observations search")
    parser.add_argument("--agent-id")
    parser.add_argument("--type", dest="observation_type")
    parser.add_argument("--target-id")
    parser.add_argument("--tags-any")
    parser.add_argument("--tags-all")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--substring")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format", dest="fmt", choices=["json", "text"], default="text")
    ns = parser.parse_args(args)

    tags_any = [t.strip() for t in ns.tags_any.split(",")] if ns.tags_any else None
    tags_all = [t.strip() for t in ns.tags_all.split(",")] if ns.tags_all else None
    since = datetime.fromisoformat(ns.since) if ns.since else None
    until = datetime.fromisoformat(ns.until) if ns.until else None

    entries = search(
        agent_id=ns.agent_id,
        observation_type=ns.observation_type,
        target_id=ns.target_id,
        tags_any=tags_any,
        tags_all=tags_all,
        since=since,
        until=until,
        substring=ns.substring,
        limit=ns.limit,
    )

    if ns.fmt == "json":
        for entry in entries:
            print(json.dumps(entry, ensure_ascii=False))
    else:
        for entry in entries:
            ts = entry.get("timestamp", "?")
            aid = entry.get("agent_id", "?")
            otype = entry.get("observation_type", "?")
            content = entry.get("content", "")
            context = entry.get("context", "")
            print(f"{ts} [{aid}:{otype}] {content}")
            if context:
                print(f"  context: {context}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "search"):
        print("usage: python -m agents_core.observations [record|search] ...",
              file=sys.stderr)
        sys.exit(1)

    subcommand = sys.argv[1]
    rest = sys.argv[2:]

    if subcommand == "record":
        _cli_record(rest)
    else:
        _cli_search(rest)
