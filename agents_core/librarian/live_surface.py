"""agents_core.librarian.live_surface — weekly Live-Surface digest renderer.

Implements ``render_live_surface(out_path) → WriteRecord``.

Idempotency
-----------
The ``generated`` timestamp in the frontmatter is stable when the underlying
corpus is unchanged.  A mapping of ``<render_key> → <generated_ts>`` is
stored in a small JSON file at
``/data/synthesis-cache/live-surface-ts.json`` (overridable via
``LIVE_SURFACE_TS_STORE`` env var).  If the same corpus_snapshots arrive on
a re-render the stored timestamp is reused, producing byte-identical output.

The render key is ``sha256(thread_corpus_snapshot | follow_corpus_snapshot)``
— both "unavailable" strings when the librarian is degraded, which is still
stable for the duration of the outage.

Frontmatter contract (required by the spec)::

    type: live-surface
    generated: <iso-8601 timestamp>
    freshness: weekly
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents_core.librarian import SynthesisArtifact
    from agents_core.vault_writer import WriteRecord

_TS_STORE_DEFAULT = "/data/synthesis-cache/live-surface-ts.json"
_TS_STORE_MAX_ENTRIES = 100  # ~2 years of weekly renders; prune oldest on overflow


def _ts_store_path() -> Path:
    return Path(os.environ.get("LIVE_SURFACE_TS_STORE", _TS_STORE_DEFAULT))


def _load_ts_store() -> dict:
    p = _ts_store_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_ts_store(store: dict) -> None:
    p = _ts_store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Cap to most recent entries — values are ISO timestamps, lexicographic sort is correct.
    if len(store) > _TS_STORE_MAX_ENTRIES:
        store = dict(sorted(store.items(), key=lambda kv: kv[1])[-_TS_STORE_MAX_ENTRIES:])
    data = json.dumps(store, ensure_ascii=False, sort_keys=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.rename(p)


def _render_key(
    thread_artifact: "SynthesisArtifact | None",
    follow_artifact: "SynthesisArtifact | None",
) -> str:
    """Compute a stable render key from both artifacts' corpus_snapshots."""
    parts = [
        thread_artifact.corpus_snapshot if thread_artifact is not None else "unavailable",
        follow_artifact.corpus_snapshot if follow_artifact is not None else "unavailable",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def render_live_surface(out_path: "str | Path") -> "WriteRecord":
    """Render the weekly Live-Surface digest to *out_path*.

    Fuses Thread Weaver state + Follow-ups + recently-landed into a markdown
    digest with the required frontmatter::

        type: live-surface
        generated: <iso-timestamp>
        freshness: weekly

    Writes via ``vault_writer`` with ``agent_id="librarian"`` and
    ``stamp_frontmatter=False`` (frontmatter is already embedded in the
    content string).

    Idempotent: re-rendering with the same corpus snapshot produces
    byte-identical output — the ``generated`` timestamp is stable as long as
    the corpus snapshots of both synthesised artifacts are unchanged.

    Parameters
    ----------
    out_path:
        Destination path for the digest markdown file, e.g.
        ``/srv/git/inertia-vault-working/Lapis/Live-Surface.md``.

    Returns
    -------
    ``WriteRecord`` from ``vault_writer.write()``.
    """
    # Lazy imports to avoid circular dependency at module load time.
    import agents_core.librarian as _lib
    from agents_core import vault_writer

    out_path = Path(out_path)

    # -- Synthesise the two pane contents (freshness=0 → always re-run) -----
    thread_result = _lib.corroborate(
        "thread-weaver-state",
        _lib.Scope(corpus=["vault-rag", "mem"]),
        freshness=0,
        policy="auto-update",
    )
    follow_result = _lib.corroborate(
        "recent-follow-ups",
        _lib.Scope(corpus=["mem", "vault-rag"]),
        freshness=0,
        policy="auto-update",
    )

    thread_artifact = thread_result if isinstance(thread_result, _lib.SynthesisArtifact) else None
    follow_artifact = follow_result if isinstance(follow_result, _lib.SynthesisArtifact) else None

    # -- Idempotency: reuse stored timestamp when corpus is unchanged --------
    render_key = _render_key(thread_artifact, follow_artifact)
    ts_store = _load_ts_store()
    if render_key in ts_store:
        generated_ts = ts_store[render_key]
    else:
        generated_ts = datetime.now(tz=timezone.utc).isoformat()
        ts_store[render_key] = generated_ts
        _save_ts_store(ts_store)

    # -- Render text content from artifacts ----------------------------------
    thread_text = (
        thread_artifact.answer.get("text", str(thread_artifact.answer))
        if thread_artifact is not None
        else "(unavailable — librarian degraded)"
    )
    follow_text = (
        follow_artifact.answer.get("text", str(follow_artifact.answer))
        if follow_artifact is not None
        else "(unavailable — librarian degraded)"
    )

    content = (
        "---\n"
        "type: live-surface\n"
        f"generated: {generated_ts}\n"
        "freshness: weekly\n"
        "---\n"
        "\n"
        "# Live Surface\n"
        "\n"
        f"*Generated {generated_ts}*\n"
        "\n"
        "## Threads in Motion\n"
        "\n"
        f"{thread_text}\n"
        "\n"
        "## Follow-ups\n"
        "\n"
        f"{follow_text}\n"
    )

    return vault_writer.write(
        out_path,
        content,
        agent_id="librarian",
        intent="weekly live-surface digest",
        stamp_frontmatter=False,  # frontmatter already embedded above
    )
