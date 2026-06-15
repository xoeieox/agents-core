"""agents-core — shared infrastructure primitives for StarHouse Claude agents.

Modules:
    llm       — llama-server (call_llm) + Claude CLI (call_claude_cli)
    gw_agent  — GravityWell review-agent harness (call_gw_agent)
    forgejo   — Forgejo REST helper (create_pr, merge_pr, get_pr_diff, ...)
    notify    — Pushover wrapper (send_notification, Priority)
    targets   — /srv/lapis/targets YAML store (Target, TargetStore)
    comments  — per-target comment log (Comment, CommentStore)
    gpu       — GPU task queue (GPUQueue, Priority)
    mem       — cross-instance memory store library (MemoryStore)
    retrieval — Synapse retrieval engine (Hit, retrieve)
    shaper    — shaped-agent registry + dispatch (Shaper, ShapedAgent, DispatchResult)
    worktree  — per-task git worktree lifecycle (WORKTREE_ROOT, setup_worktree, ...)
"""

from agents_core.gw_agent import call_gw_agent
from agents_core.retrieval import Hit, retrieve
from agents_core.shaper import DispatchResult, ShapedAgent, Shaper

__version__ = "0.3.0"

__all__ = ["call_gw_agent", "Hit", "retrieve", "Shaper", "ShapedAgent", "DispatchResult"]
