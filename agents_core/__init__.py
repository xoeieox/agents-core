"""agents-core — shared infrastructure primitives for StarHouse Claude agents.

Modules:
    llm       — llama-server (call_llm) + Claude CLI (call_claude_cli)
    forgejo   — Forgejo REST helper (create_pr, merge_pr, get_pr_diff, ...)
    notify    — Pushover wrapper (send_notification, Priority)
    targets   — /srv/lapis/targets YAML store (Target, TargetStore)
    comments  — per-target comment log (Comment, CommentStore)
    gpu       — GPU task queue (GPUQueue, Priority)
    mem       — cross-instance memory store library (MemoryStore)
"""

__version__ = "0.2.0"
