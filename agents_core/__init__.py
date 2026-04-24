"""agents-core — shared infrastructure primitives for StarHouse Claude agents.

Modules:
    llm       — llama-server (call_llm) + Claude CLI (call_claude_cli)
    forgejo   — Forgejo REST helper (create_pr, merge_pr, get_pr_diff, ...)
    notify    — Pushover wrapper (send_notification, Priority)
    targets   — /srv/lapis/targets YAML store (Target, TargetStore)
    comments  — per-target comment log (Comment, CommentStore)
    gpu       — GPU task queue (GPUQueue, Priority)
    mem       — cross-instance memory store library (MemoryStore)
    retrieval — Synapse retrieval engine (Hit, retrieve)
"""

from agents_core.retrieval import Hit, retrieve

__version__ = "0.3.0"

__all__ = ["Hit", "retrieve"]
