"""Agent observation substrate — append-only per-agent-per-date JSONL.

Storage: /srv/lapis/agent-observations/<agent_id>/<YYYY-MM-DD>.jsonl
Override root via AGENT_OBSERVATIONS_ROOT env var (primarily for tests).

Entry schema (each JSONL line):
    {
        obs_id (v0.1, stable hash of agent_id+timestamp+type+content),
        agent_id, session_id, timestamp (ISO8601 UTC), observation_type,
        context, content, target_id, tags, intervention_shape,
        informed_by (v0.1, list[str] of cited obs_ids, default []),
        signal_strength (v0.1, "high" | "normal", default "normal"),
        extra
    }

observation_type: friction | decision | lesson | anomaly | intervention
intervention_shape: question | pointer | counter-example | frame-shift | constraint | why-trace
    (required iff observation_type == "intervention")
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
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

VALID_SIGNAL_STRENGTHS = frozenset({"high", "normal"})
DEFAULT_SIGNAL_STRENGTH = "normal"


def root() -> Path:
    """Return the resolved observations root, honoring AGENT_OBSERVATIONS_ROOT env override.
    Default: /srv/lapis/agent-observations/"""
    override = os.environ.get("AGENT_OBSERVATIONS_ROOT")
    return Path(override) if override else DEFAULT_ROOT


def compute_obs_id(agent_id: str, timestamp_iso: str,
                   observation_type: str, content: str) -> str:
    """Stable, content-derived identifier. 16 hex chars from SHA-256.
    Same inputs always produce the same obs_id, across invocations
    and across processes."""
    import hashlib
    key = f"{agent_id}|{timestamp_iso}|{observation_type}|{content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


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
    informed_by: list[str] | None = None,
    signal_strength: str = DEFAULT_SIGNAL_STRENGTH,
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

    if signal_strength not in VALID_SIGNAL_STRENGTHS:
        raise ValueError(
            f"signal_strength must be one of {sorted(VALID_SIGNAL_STRENGTHS)}, "
            f"got {signal_strength!r}"
        )

    if informed_by is not None:
        if not isinstance(informed_by, list) or not all(
            isinstance(x, str) for x in informed_by
        ):
            raise TypeError("informed_by must be a list of strings")

    ts = now if now is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    ts_utc = ts.astimezone(timezone.utc)
    ts_iso = ts_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    date_str = ts_utc.strftime("%Y-%m-%d")

    obs_id = compute_obs_id(agent_id, ts_iso, observation_type, content)

    entry: dict = {
        "obs_id": obs_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "timestamp": ts_iso,
        "observation_type": observation_type,
        "context": context,
        "content": content,
        "target_id": target_id,
        "tags": list(tags) if tags else [],
        "intervention_shape": intervention_shape,
        "informed_by": list(informed_by) if informed_by else [],
        "signal_strength": signal_strength,
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


def _backfill_v0_defaults(entry: dict) -> dict:
    """Apply v0.1 read-time defaults to v0 entries that lack new fields.
    Modifies the dict in place and returns it."""
    if "obs_id" not in entry:
        entry["obs_id"] = compute_obs_id(
            entry.get("agent_id", ""),
            entry.get("timestamp", ""),
            entry.get("observation_type", ""),
            entry.get("content", ""),
        )
    if "informed_by" not in entry:
        entry["informed_by"] = []
    if "signal_strength" not in entry:
        entry["signal_strength"] = DEFAULT_SIGNAL_STRENGTH
    return entry


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
    min_signal_strength: str | None = None,
    informed_by: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read across observation files matching filters. Iterates JSONL files line by line
    (files are not slurped whole), but accumulates all matching entries in memory before
    returning. Returns entries sorted by timestamp ascending."""
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

                        # Backfill v0 entries with v0.1 defaults at read time
                        _backfill_v0_defaults(entry)

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

                        # Filter: min_signal_strength
                        if min_signal_strength == "high":
                            if entry.get("signal_strength") != "high":
                                continue
                        # "normal" is a no-op (everything passes); None disables filter

                        # Filter: informed_by (entry must cite the given obs_id)
                        if informed_by is not None:
                            if informed_by not in (entry.get("informed_by") or []):
                                continue

                        results.append(entry)
            except OSError:
                continue

    results.sort(key=lambda e: e.get("timestamp", ""))

    if limit is not None:
        results = results[:limit]

    return results


def lineage(
    obs_id: str,
    *,
    direction: str = "both",
    max_depth: int = 10,
) -> dict:
    """Return the citation graph rooted at obs_id.

    Returns: {
      "root": <entry or None if obs_id not found>,
      "forward": [<entry>, ...],   # entries that (transitively) cite obs_id
      "backward": [<entry>, ...],  # entries (transitively) in obs_id's informed_by chain
    }

    Each entry in forward/backward has _lineage_depth added (int, distance from root).
    Direction filter empties the unwanted list. Cycles broken by visited-set on obs_id.
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError(
            f"direction must be 'forward', 'backward', or 'both', got {direction!r}"
        )
    all_entries = search()

    # Build index: obs_id -> entry (last-seen wins per Invariant 4)
    by_id: dict[str, dict] = {}
    for e in all_entries:
        eid = e.get("obs_id")
        if eid:
            by_id[eid] = e

    # Build reverse-citation index: obs_id -> list of obs_ids that cite it
    cited_by: dict[str, list[str]] = {}
    for e in all_entries:
        eid = e.get("obs_id")
        if not eid:
            continue
        for parent_id in (e.get("informed_by") or []):
            cited_by.setdefault(parent_id, []).append(eid)

    root_entry = by_id.get(obs_id)

    forward: list[dict] = []
    backward: list[dict] = []

    if direction in ("forward", "both"):
        visited: set[str] = {obs_id}
        queue: deque[tuple[str, int]] = deque([(obs_id, 0)])
        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for child_id in cited_by.get(current_id, []):
                if child_id in visited:
                    continue
                visited.add(child_id)
                child_entry = by_id.get(child_id)
                if child_entry is not None:
                    entry_copy = dict(child_entry)
                    entry_copy["_lineage_depth"] = depth + 1
                    forward.append(entry_copy)
                    queue.append((child_id, depth + 1))

    if direction in ("backward", "both"):
        visited_b: set[str] = {obs_id}
        queue_b: deque[tuple[str, int]] = deque([(obs_id, 0)])
        while queue_b:
            current_id, depth = queue_b.popleft()
            if depth >= max_depth:
                continue
            current_entry = by_id.get(current_id)
            if current_entry is None:
                continue
            for parent_id in (current_entry.get("informed_by") or []):
                if parent_id in visited_b:
                    continue
                visited_b.add(parent_id)
                parent_entry = by_id.get(parent_id)
                if parent_entry is not None:
                    entry_copy = dict(parent_entry)
                    entry_copy["_lineage_depth"] = depth + 1
                    backward.append(entry_copy)
                    queue_b.append((parent_id, depth + 1))

    root_copy = dict(root_entry) if root_entry is not None else None
    return {"root": root_copy, "forward": forward, "backward": backward}


def cite(
    informed_by: list[str],
    *,
    agent_id: str,
    observation_type: str,
    context: str,
    content: str,
    signal_strength: str = DEFAULT_SIGNAL_STRENGTH,
    **record_kwargs,
) -> Path:
    """Record an observation that cites prior observations.

    Equivalent to record(..., informed_by=informed_by, signal_strength=...) — exists
    purely for readability at the call site: cite([obs1, obs2], agent_id=..., ...).
    """
    return record(
        agent_id=agent_id,
        observation_type=observation_type,
        context=context,
        content=content,
        informed_by=informed_by,
        signal_strength=signal_strength,
        **record_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry-point  (python -m agents_core.observations [record|search|lineage] ...)
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
    parser.add_argument("--informed-by", dest="informed_by", action="append", default=[])
    parser.add_argument(
        "--signal-strength", dest="signal_strength",
        choices=["high", "normal"], default="normal",
    )
    ns = parser.parse_args(args)

    extra: dict | None = None
    if ns.extra_json:
        extra = json.loads(ns.extra_json)

    # Generate now upfront so we can compute obs_id without a post-write search scan.
    now = datetime.now(timezone.utc)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    obs_id_out = compute_obs_id(ns.agent_id, ts_iso, ns.observation_type, ns.content)

    path = record(
        agent_id=ns.agent_id,
        observation_type=ns.observation_type,
        context=ns.context,
        content=ns.content,
        session_id=ns.session_id,
        target_id=ns.target_id,
        tags=ns.tags or None,
        intervention_shape=ns.intervention_shape,
        informed_by=ns.informed_by if ns.informed_by else None,
        signal_strength=ns.signal_strength,
        extra=extra,
        now=now,
    )

    print(f"{path}\t{obs_id_out}")


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
    parser.add_argument(
        "--min-signal-strength", dest="min_signal_strength",
        choices=["high", "normal"], default=None,
    )
    parser.add_argument("--informed-by", dest="informed_by", default=None)
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
        min_signal_strength=ns.min_signal_strength,
        informed_by=ns.informed_by,
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
            obs_id = entry.get("obs_id", "????????????????")
            print(f"{obs_id} {ts} [{aid}:{otype}] {content}")
            if context:
                print(f"  context: {context}")


def _cli_lineage(args: list[str]) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="observations lineage")
    parser.add_argument("obs_id")
    parser.add_argument(
        "--direction", choices=["both", "forward", "backward"], default="both"
    )
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--format", dest="fmt", choices=["json", "text"], default="text")
    ns = parser.parse_args(args)

    result = lineage(ns.obs_id, direction=ns.direction, max_depth=ns.max_depth)

    if ns.fmt == "json":
        print(json.dumps(result, ensure_ascii=False, default=str))
    else:
        root_entry = result["root"]
        if root_entry is None:
            print(f"<missing: {ns.obs_id}>")
        else:
            obs_id = root_entry.get("obs_id", "?")
            ts = root_entry.get("timestamp", "?")
            otype = root_entry.get("observation_type", "?")
            content = root_entry.get("content", "")
            print(f"root: {obs_id} {ts} [{otype}] {content}")

        if result["forward"]:
            print("forward citations (entries that cite root):")
            for e in result["forward"]:
                depth = e.get("_lineage_depth", "?")
                obs_id = e.get("obs_id", "?")
                ts = e.get("timestamp", "?")
                otype = e.get("observation_type", "?")
                content = e.get("content", "")
                indent = "  " + "↳ " * depth
                print(f"{indent}{obs_id} {ts} [{otype}] {content}")

        if result["backward"]:
            print("backward citations (entries root was informed by):")
            for e in result["backward"]:
                depth = e.get("_lineage_depth", "?")
                obs_id = e.get("obs_id", "?")
                ts = e.get("timestamp", "?")
                otype = e.get("observation_type", "?")
                content = e.get("content", "")
                indent = "  " + "↰ " * depth
                print(f"{indent}{obs_id} {ts} [{otype}] {content}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("record", "search", "lineage"):
        print(
            "usage: python -m agents_core.observations [record|search|lineage] ...",
            file=sys.stderr,
        )
        sys.exit(1)

    subcommand = sys.argv[1]
    rest = sys.argv[2:]

    if subcommand == "record":
        _cli_record(rest)
    elif subcommand == "search":
        _cli_search(rest)
    else:
        _cli_lineage(rest)
