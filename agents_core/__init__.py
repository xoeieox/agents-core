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

__version__ = "0.3.0"

__all__ = ["call_gw_agent", "Hit", "retrieve", "Shaper", "ShapedAgent", "DispatchResult"]

_lazy_map: dict[str, tuple[str, str]] = {
    "call_gw_agent": ("agents_core.gw_agent", "call_gw_agent"),
    "Hit": ("agents_core.retrieval", "Hit"),
    "retrieve": ("agents_core.retrieval", "retrieve"),
    "Shaper": ("agents_core.shaper", "Shaper"),
    "ShapedAgent": ("agents_core.shaper", "ShapedAgent"),
    "DispatchResult": ("agents_core.shaper", "DispatchResult"),
}


def __getattr__(name: str):
    if name in _lazy_map:
        import importlib
        module_name, attr = _lazy_map[name]
        mod = importlib.import_module(module_name)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'agents_core' has no attribute {name!r}")
