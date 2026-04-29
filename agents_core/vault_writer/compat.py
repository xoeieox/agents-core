"""agents_core.vault_writer.compat — explicit-call migration helper.

Provides write_compat() so existing callers can be migrated from direct
Path.write_text() calls to vault-writer mediation with a one-liner grep-and-
replace, without requiring them to supply full agent metadata.

Migration pattern::

    # Before:
    path.write_text(content)

    # After:
    from agents_core.vault_writer.compat import write_compat
    write_compat(path, content, agent_id="my-agent", intent="specific reason")

Or for unmigrated callers that are just being swept in bulk::

    write_compat(path, content)  # defaults: agent_id="legacy", intent="unmigrated"

Design decisions:
  - NO monkey-patching of Path.write_text.  Monkey-patching was rejected because
    subtle write paths (.open('w'), os.fsync, third-party libs) silently bypass
    it, giving false confidence that all writes are tracked.  Explicit call
    conversion is honest about untracked legacy origin.
  - Attribution frontmatter is stamped with agent_id="legacy" / intent="unmigrated"
    for untracked conversions, so the audit log is honest about the origin.
  - write_compat() produces the same on-disk bytes as Path.write_text(content)
    EXCEPT for the attribution frontmatter block prepended/updated on markdown
    files (stamp_frontmatter=True by default, same as write()).
  - If you need byte-for-byte identical output to the original write_text call
    (e.g. for binary blobs or files that must not have a frontmatter block),
    pass stamp_frontmatter=False.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from agents_core.vault_writer import Citation, WriteRecord, write

PolicyName = Literal["auto-update", "on-trigger", "never-update"]


def write_compat(
    path: str | Path,
    content: str | bytes,
    agent_id: str = "legacy",
    intent: str = "unmigrated",
    *,
    citations: list[Citation] | None = None,
    policy: PolicyName = "auto-update",
    stamp_frontmatter: bool = True,
) -> WriteRecord:
    """Drop-in replacement for Path.write_text() that goes through vault-writer.

    Parameters
    ----------
    path:
        Destination path (str or Path).
    content:
        File contents — str (UTF-8) or bytes, matching write_text() behaviour.
    agent_id:
        Identifies the writing agent.  Defaults to ``"legacy"`` for untracked
        callers; supply a real id once the caller is properly migrated.
    intent:
        Human-readable write reason.  Defaults to ``"unmigrated"`` for legacy
        callers; supply a specific intent once properly migrated.
    citations:
        Optional source citations (same as vault_writer.write()).
    policy:
        Update-policy hint; default ``"auto-update"``.
    stamp_frontmatter:
        Whether to add/update the ``attribution:`` block.  Same default (True)
        as vault_writer.write().  Set False for binary or non-markdown files.

    Returns
    -------
    WriteRecord — same as vault_writer.write().
    """
    return write(
        path,
        content,
        agent_id=agent_id,
        intent=intent,
        citations=citations,
        policy=policy,
        stamp_frontmatter=stamp_frontmatter,
    )
