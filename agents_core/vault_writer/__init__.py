"""agents_core.vault_writer — canonical write path for vault corpus files.

Every non-legacy write to the vault (inertia-vault-working, /room) must go
through this module. It provides:

  - Atomic writes (temp file + os.rename, never a partial write visible to readers)
  - Per-file exclusive lock (fcntl.LOCK_EX) so concurrent writers serialize
  - Audit log append (vault_audit) with content-hash chain
  - Write-event emission: in-process asyncio pub/sub AND on-disk JSONL tail
  - Attribution frontmatter stamping (idempotent, YAML-safe)

Usage::

    from agents_core.vault_writer import write, stamp_attribution, subscribe
    record = write(
        "/srv/git/inertia-vault-working/Lapis/foo.md",
        new_content,
        agent_id="lapis-pm",
        intent="update target summary",
    )

Event bus::

    # In an asyncio context:
    async for event in subscribe("Lapis/"):
        print(event.record.path, event.record.content_hash)

Hash convention:
    content_hash values always carry the ``sha256:`` prefix (e.g.
    ``"sha256:abcdef..."``).  On-disk paths strip the prefix (no colon in
    directory names).

Write-event JSONL tail:
    Every write appends one JSON line to ``/data/vault-events.jsonl``
    (override via ``VAULT_EVENTS_JSONL`` env var) so out-of-process
    subscribers can ``tail -f`` the file.

Visibility note (PR 2+):
    This module does NOT enforce citable-field rules; that is the librarian's
    responsibility.  vault_writer writes whatever it's given.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from agents_core import vault_audit

# ---------------------------------------------------------------------------
# Public types (from spec Interface Contracts)
# ---------------------------------------------------------------------------


class Citation(TypedDict):
    path: str           # corpus-relative path of cited source
    content_hash: str   # "sha256:<hex>" form
    quoted_snippet: str  # substring of cited chunk text


@dataclass
class WriteRecord:
    path: str
    content_hash: str       # "sha256:<hex>" form
    prev_hash: str | None   # previous content_hash at this path, or None on first write
    ts: datetime
    agent_id: str
    intent: str


@dataclass
class WriteEvent:
    record: WriteRecord
    topic: str  # path-prefix-shaped topic for subscription filtering


PolicyName = Literal["auto-update", "on-trigger", "never-update"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EVENTS_JSONL_DEFAULT = Path("/data/vault-events.jsonl")


def _events_jsonl_path() -> Path:
    return Path(os.environ.get("VAULT_EVENTS_JSONL", str(_EVENTS_JSONL_DEFAULT)))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_hash(data: bytes) -> str:
    return f"sha256:{_sha256_hex(data)}"


def _read_prev_hash(path: Path) -> str | None:
    """Return content_hash of current on-disk file, or None if absent."""
    if not path.exists():
        return None
    return _content_hash(path.read_bytes())


def _atomic_write(path: Path, data: bytes) -> None:
    """Write data to path atomically via a temp file + os.rename in same dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp.")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        os.rename(tmp_name, str(path))
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# In-process pub/sub event bus
# ---------------------------------------------------------------------------

@dataclass
class _Subscriber:
    topic: str  # prefix filter; "*" matches all
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    loop: asyncio.AbstractEventLoop | None = None  # owning loop captured at register


_subscribers: list[_Subscriber] = []
_subscribers_lock = asyncio.Lock()


async def _register_subscriber(topic: str) -> _Subscriber:
    sub = _Subscriber(topic=topic, loop=asyncio.get_running_loop())
    async with _subscribers_lock:
        _subscribers.append(sub)
    return sub


async def _unregister_subscriber(sub: _Subscriber) -> None:
    async with _subscribers_lock:
        try:
            _subscribers.remove(sub)
        except ValueError:
            pass


def _publish_sync(event: WriteEvent) -> None:
    """Deliver event to subscribers via each subscriber's owning loop.

    Works from inside any event loop, a different loop, or a plain
    sync/threaded caller.  Each subscriber's asyncio.Queue is only
    mutated from its own loop's thread via ``call_soon_threadsafe``,
    so cross-loop dispatch is safe.
    """
    # GIL-protected snapshot — we don't take the async lock from sync code.
    targets = list(_subscribers)
    for sub in targets:
        if not (sub.topic == "*" or event.record.path.startswith(sub.topic)):
            continue
        target_loop = sub.loop
        if target_loop is None or target_loop.is_closed():
            continue
        try:
            target_loop.call_soon_threadsafe(sub.queue.put_nowait, event)
        except RuntimeError:
            # Loop closed mid-call; drop event (best-effort).
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write(
    path: str | Path,
    content: str | bytes,
    *,
    agent_id: str,
    intent: str,
    citations: list[Citation] | None = None,
    policy: PolicyName = "auto-update",
    stamp_frontmatter: bool = True,
) -> WriteRecord:
    """Write *content* to *path* through the vault-writer pipeline.

    Acquires an exclusive file lock, writes atomically (temp + rename),
    appends to the audit log, and emits a write event.

    Parameters
    ----------
    path:
        Absolute (or relative) filesystem path to write.
    content:
        String (UTF-8) or bytes to write.
    agent_id:
        Identifier of the writing agent, e.g. ``"lapis-pm"``.
    intent:
        Human-readable reason for the write, e.g. ``"update target summary"``.
    citations:
        Optional list of Citation dicts — sources that informed the write.
    policy:
        Update-policy hint stored in the audit row; default ``"auto-update"``.
    stamp_frontmatter:
        If True (default), add/update an ``attribution:`` block in YAML
        frontmatter before writing.  Set False for binary or non-markdown files.

    Returns
    -------
    WriteRecord with hashes and timestamp of the completed write.
    """
    path = Path(path)
    data: bytes = content.encode("utf-8") if isinstance(content, str) else content

    # Attribution stamp (markdown/text only)
    if stamp_frontmatter and isinstance(content, str):
        stamped = stamp_attribution(content, agent_id=agent_id, intent=intent, citations=citations or [])
        data = stamped.encode("utf-8")

    # Per-file exclusive lock — acquire before reading prev_hash so the
    # hash chain is accurate even under concurrent writers.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    lock_fh = open(lock_path, "w")  # noqa: WPS515 — intentional resource held
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)

        prev_hash = _read_prev_hash(path)
        _atomic_write(path, data)
        now = datetime.now(tz=timezone.utc)
        new_hash = _content_hash(data)

        # Audit log
        vault_audit.append(
            ts=now,
            path=str(path),
            content_hash=new_hash,
            prev_hash=prev_hash,
            agent_id=agent_id,
            intent=intent,
            citations=list(citations) if citations else [],
        )

        record = WriteRecord(
            path=str(path),
            content_hash=new_hash,
            prev_hash=prev_hash,
            ts=now,
            agent_id=agent_id,
            intent=intent,
        )

    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()

    # Build topic from path (use str form, publish after lock released)
    topic = str(path)
    event = WriteEvent(record=record, topic=topic)

    # Persist to JSONL tail (best-effort; failures are non-fatal)
    _append_jsonl(event)

    # In-process pub/sub
    _publish_sync(event)

    return record


async def subscribe(topic: str = "*") -> AsyncIterator[WriteEvent]:
    """Async generator that yields WriteEvents matching *topic* prefix.

    Parameters
    ----------
    topic:
        Path prefix to filter on.  ``"*"`` (default) matches all events.
        E.g. ``"Lapis/"`` matches writes whose path starts with ``"Lapis/"``.

    Usage::

        async for event in subscribe("Lapis/"):
            print(event.record.path)
    """
    sub = await _register_subscriber(topic)
    try:
        while True:
            event = await sub.queue.get()
            yield event
    finally:
        await _unregister_subscriber(sub)


def _append_jsonl(event: WriteEvent) -> None:
    """Append a JSON line for *event* to the JSONL tail file."""
    jsonl_path = _events_jsonl_path()
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = event.record
        line = json.dumps(
            {
                "topic": event.topic,
                "path": record.path,
                "content_hash": record.content_hash,
                "prev_hash": record.prev_hash,
                "ts": record.ts.isoformat(),
                "agent_id": record.agent_id,
                "intent": record.intent,
            },
            ensure_ascii=False,
        )
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
    except OSError:
        pass  # non-fatal; JSONL tail is best-effort


# ---------------------------------------------------------------------------
# Frontmatter helper
# ---------------------------------------------------------------------------

# Matches a YAML frontmatter block at the very start of a markdown file.
_FM_RE = re.compile(r"^---\r?\n(.*?\r?\n)---\r?\n", re.DOTALL)


def stamp_attribution(
    content: str,
    agent_id: str,
    intent: str,
    citations: list[Citation] | None = None,
) -> str:
    """Add or update an ``attribution:`` block in YAML frontmatter.

    Idempotent: stamping the same content twice produces the same result.

    If the file has no frontmatter, a minimal ``---`` block is prepended.
    The ``attribution:`` key is added or replaced without disturbing other
    frontmatter fields.

    Parameters
    ----------
    content:
        The markdown file content (string).
    agent_id:
        Identifier of the writing agent.
    intent:
        Human-readable description of the write.
    citations:
        Optional list of Citation dicts to record in the attribution block.

    Returns
    -------
    Modified file content with an up-to-date ``attribution:`` block.
    """
    citations = citations or []

    # Build the attribution YAML value (as a mapping, inline-style for compactness)
    cit_list = [
        {"path": c["path"], "content_hash": c["content_hash"]}
        for c in citations
    ]
    attr_value: dict = {"agent_id": agent_id, "intent": intent}
    if cit_list:
        attr_value["citations"] = cit_list

    # Serialize citation list into an inline YAML fragment
    attr_yaml = _attribution_to_yaml_fragment(attr_value)

    match = _FM_RE.match(content)
    if match:
        fm_body = match.group(1)
        rest = content[match.end():]

        # Remove existing attribution: block (single-key or multi-line)
        fm_body = _remove_attribution_key(fm_body)

        # Append the new attribution block
        new_fm = f"---\n{fm_body}attribution: {attr_yaml}\n---\n"
        return new_fm + rest
    else:
        # No frontmatter — prepend a new block
        return f"---\nattribution: {attr_yaml}\n---\n{content}"


def _attribution_to_yaml_fragment(attr: dict) -> str:
    """Produce a single-line YAML representation of the attribution dict."""
    # We keep it simple: build a JSON-compatible inline YAML (flow mapping)
    # which is valid YAML and readable.
    parts = [f'agent_id: "{attr["agent_id"]}", intent: "{attr["intent"]}"']
    citations = attr.get("citations", [])
    if citations:
        cit_parts = []
        for c in citations:
            cit_parts.append(
                f'{{path: "{c["path"]}", content_hash: "{c["content_hash"]}"}}'
            )
        parts.append(f"citations: [{', '.join(cit_parts)}]")
    return "{" + ", ".join(parts) + "}"


def _remove_attribution_key(fm_body: str) -> str:
    """Remove an existing ``attribution:`` key (and its continuation lines) from frontmatter body."""
    lines = fm_body.split("\n")
    result = []
    skip = False
    for line in lines:
        if line.startswith("attribution:"):
            skip = True
            continue
        if skip and line and (line[0] == " " or line[0] == "\t"):
            # Continuation of the previous multi-line value
            continue
        skip = False
        result.append(line)
    return "\n".join(result)
