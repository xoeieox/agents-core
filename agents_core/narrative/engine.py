"""Narrative template engine — emit_draft().

Compose canonical sources × audience frame × ask, dispatch one Opus call
via ClaudeQueue, and return the draft + provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Lazy-import chub_loader from /srv/agents/scripts (runtime dep only)
# ---------------------------------------------------------------------------
_CHUB_SCRIPTS = "/srv/agents/scripts"
if _CHUB_SCRIPTS not in sys.path:
    sys.path.insert(0, _CHUB_SCRIPTS)

from agents_core.claude_queue import CLAUDE_QUEUE_DIR, ClaudeQueue
from agents_core.narrative.audiences import AUDIENCE_REGISTRY, AudienceProfile

_VAULT_ROOT = Path("/srv/git/inertia-vault-working")

# Canonical sources (load order is the invariant for prompt determinism)
_CANONICAL_SOURCES: list[str] = [
    "Lapis/Constitution-Kernel.md",
    "<chub:conductor/lapis-ecosystem>",
    "Lapis/Lapis-Philosophy.md",
    "Lapis/Vision-Bidirectional-Mirror.md",
]

_LENGTH_TARGETS = {"short": "~250 words", "medium": "~600 words", "long": "~1500 words"}

_DISPATCH_BASE = Path("/tmp/narrative-emit")


@dataclass
class SourceRef:
    path: str
    sha256: str


@dataclass
class EmitResult:
    draft: str
    audience: str
    ask: str
    length_target: str
    sources: list[SourceRef]
    prompt_hash: str
    model: str
    dispatched_at: str
    returned_at: str
    # Internal: assembled prompt (populated only on dry_run)
    _prompt: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_vault(rel_path: str) -> str:
    """Read a vault file strictly (UTF-8, binary read). Raises on any failure."""
    full = _VAULT_ROOT / rel_path
    raw = full.read_bytes()  # raises FileNotFoundError / PermissionError
    return raw.decode("utf-8", errors="strict")  # raises UnicodeDecodeError on bad bytes


def _load_chub() -> str:
    """Load the lapis-ecosystem chub bundle. Raises RuntimeError on empty return."""
    from chub_loader import load_chub_bundle  # noqa: PLC0415 (lazy import per CLAUDE.md)
    content = load_chub_bundle("conductor/lapis-ecosystem")
    if not content:
        raise RuntimeError(
            "chub bundle 'conductor/lapis-ecosystem' returned empty content; "
            "verify chub CLI installed and bundle present"
        )
    return content


def _resolve_sources(profile: AudienceProfile) -> tuple[list[SourceRef], dict[str, str]]:
    """Load all sources in fixed canonical order. Returns (refs, bodies_by_path)."""
    refs: list[SourceRef] = []
    bodies: dict[str, str] = {}

    # a. Constitution Kernel
    body = _read_vault("Lapis/Constitution-Kernel.md")
    refs.append(SourceRef(path="Lapis/Constitution-Kernel.md", sha256=_sha256(body)))
    bodies["Lapis/Constitution-Kernel.md"] = body

    # b. Chub bundle — check for empty at call site (loud failure invariant)
    chub_body = _load_chub()
    if not chub_body:
        raise RuntimeError(
            "chub bundle 'conductor/lapis-ecosystem' returned empty content; "
            "verify chub CLI installed and bundle present"
        )
    chub_key = "<chub:conductor/lapis-ecosystem>"
    refs.append(SourceRef(path=chub_key, sha256=_sha256(chub_body)))
    bodies[chub_key] = chub_body

    # c. Philosophy
    body = _read_vault("Lapis/Lapis-Philosophy.md")
    refs.append(SourceRef(path="Lapis/Lapis-Philosophy.md", sha256=_sha256(body)))
    bodies["Lapis/Lapis-Philosophy.md"] = body

    # d. Vision
    body = _read_vault("Lapis/Vision-Bidirectional-Mirror.md")
    refs.append(SourceRef(path="Lapis/Vision-Bidirectional-Mirror.md", sha256=_sha256(body)))
    bodies["Lapis/Vision-Bidirectional-Mirror.md"] = body

    # e. Audience extra sources
    for rel in profile.extra_sources:
        body = _read_vault(rel)
        refs.append(SourceRef(path=rel, sha256=_sha256(body)))
        bodies[rel] = body

    return refs, bodies


def _compose_prompt(
    profile: AudienceProfile,
    source_refs: list[SourceRef],
    bodies: dict[str, str],
    ask: str,
    length_target: str,
) -> tuple[str, str]:
    """Return (system_block, user_block). No wall-clock values embedded."""
    fence = "=" * 72

    # System block: audience frame
    frame_parts = [
        "## Audience Frame",
        f"**Audience:** {profile.title}",
        "",
        "**Frame:**",
        profile.frame,
        "",
        f"**Register:** {profile.register}",
        "",
        "**Emphasize:**",
        *[f"- {e}" for e in profile.emphasize],
        "",
        "**De-emphasize:**",
        *[f"- {d}" for d in profile.deemphasize],
        "",
        "**Characteristic phrasings:**",
        *[f"- {c}" for c in profile.example_callouts],
        "",
        f"**Length target:** {_LENGTH_TARGETS[length_target]}",
    ]
    frame_block = "\n".join(frame_parts)

    # Sources block
    source_parts = ["## Canonical Sources"]
    for ref in source_refs:
        body = bodies[ref.path]
        source_parts.append(f"\n### {ref.path}\n\n{fence}\n{body}\n{fence}")
    sources_block = "\n".join(source_parts)

    system_block = f"{frame_block}\n\n{sources_block}"

    # User block: the ask
    user_block = (
        f"Using the canonical sources and audience frame above, draft the following:\n\n{ask}"
    )

    return system_block, user_block


def _generate_task_id(audience_slug: str) -> str:
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    usec = now.strftime("%f")[:4]
    return f"narrative_{ts}_{usec}_{audience_slug}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit_draft(
    audience_slug: str,
    ask: str,
    *,
    length_target: Literal["short", "medium", "long"] = "medium",
    dry_run: bool = False,
    _queue_dir: Path | None = None,
    _poll_interval: float = 2.0,
) -> EmitResult:
    """Compose canonical sources × audience frame × ask → EmitResult.

    Raises FileNotFoundError if any pinned source path is missing.
    Raises KeyError if audience_slug is not registered.
    Raises RuntimeError if chub bundle returns empty content.
    """
    # 1. Resolve audience
    profile = AUDIENCE_REGISTRY[audience_slug]  # KeyError if unknown

    # 2. Resolve sources (fails loud on any missing path or empty chub)
    source_refs, bodies = _resolve_sources(profile)

    # 3. Compose prompt deterministically (no wall-clock values)
    system_block, user_block = _compose_prompt(profile, source_refs, bodies, ask, length_target)
    full_prompt = f"{system_block}\n\n---\n\n{user_block}"
    prompt_hash = _sha256(full_prompt)

    # 4. Dry-run: return without calling the queue
    if dry_run:
        result = EmitResult(
            draft="",
            audience=audience_slug,
            ask=ask,
            length_target=length_target,
            sources=source_refs,
            prompt_hash=prompt_hash,
            model="",
            dispatched_at="",
            returned_at="",
            _prompt=full_prompt,
        )
        return result

    # 5. Dispatch via submit-and-poll (mirrors expert.py:395-440)
    queue_dir = _queue_dir or CLAUDE_QUEUE_DIR
    task_id = _generate_task_id(audience_slug)
    dispatch_cwd = _DISPATCH_BASE / task_id
    dispatch_cwd.mkdir(parents=True, exist_ok=True)

    pending_dir = queue_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "task_id": task_id,
        "task_type": "subprocess",
        "prompt": user_block,
        "system": system_block,
        "model": "opus",
        "timeout_s": 300,
        "json_mode": False,
        "cwd": str(dispatch_cwd),
        "permission_mode": "bypassPermissions",
        "worktree_required": False,
        "capture_meta": False,
    }
    spec_path = pending_dir / f"{task_id}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False))

    dispatched_at = datetime.now(timezone.utc).isoformat()

    q = ClaudeQueue(queue_dir)
    q.submit(
        {
            "task_type": "subprocess",
            "priority": 50,
            "timeout_seconds": 360,
            "submitted_by": "narrative-emit",
            "model": "opus",
            "description": f"narrative:{audience_slug}",
            "notify": False,
            "payload": {"spec_path": str(spec_path)},
        },
        task_id=task_id,
    )

    # 6. Poll for output.md (≤360s hard deadline, 2s interval)
    output_file = dispatch_cwd / "output.md"
    deadline = time.monotonic() + 360

    while time.monotonic() < deadline:
        if output_file.exists() and output_file.stat().st_size > 0:
            break
        time.sleep(_poll_interval)

    if not (output_file.exists() and output_file.stat().st_size > 0):
        raise TimeoutError(
            f"narrative-emit dispatch {task_id} did not write output.md within 360s; "
            "check /srv/lapis/claude-queue/failed/"
        )

    returned_at = datetime.now(timezone.utc).isoformat()
    draft = output_file.read_text(encoding="utf-8")

    return EmitResult(
        draft=draft,
        audience=audience_slug,
        ask=ask,
        length_target=length_target,
        sources=source_refs,
        prompt_hash=prompt_hash,
        model="claude-opus-4-7",
        dispatched_at=dispatched_at,
        returned_at=returned_at,
    )
