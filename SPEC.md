# SPEC — agents-core

## Identity

**agents-core** is the shared-primitives package for all Claude-driven agents
running on StarHouse. It provides the minimum contract surface an agent needs
to participate in the StarHouse ecosystem: talk to the LLM, talk to Forgejo,
notify humans, track work threads and comments, dispatch GPU tasks, and
read/write the cross-instance memory store.

## What agents-core owns

- `agents_core.llm` — `call_llm()` (llama-server OpenAI-compatible client) and
  `call_claude_cli()` (subprocess wrapper around `claude -p`, Max subscription).
- `agents_core.forgejo` — Forgejo REST client for the single-owner workflow
  (OWNER hardcoded to `Erah`). PR creation, merge, diff, comments, issues,
  branch protection.
- `agents_core.notify` — Pushover notification wrapper with `Priority` enum.
- `agents_core.targets` — `/srv/lapis/targets/<id>.yaml` work-thread store. Owns
  the `Target` dataclass schema, including PM-extension fields (`pm_bound`,
  `authority`, etc.) until a second agent wants its own fields.
- `agents_core.comments` — per-target comment log (`/srv/lapis/targets/<id>/comments.jsonl`).
- `agents_core.gpu` — priority queue at `/srv/lapis/gpu-queue/`. `GPUQueue.submit()`,
  `Priority` enum (CRITICAL=0, HIGH=10, NORMAL=50, LOW=80, IDLE=99).
- `agents_core.mem` — `MemoryStore` class over `/data/memory/mem.db` (SQLite + FTS5).
  Library only. The `mem` CLI stays in `/srv/agents/scripts/mem.py`.

## What agents-core does NOT own

- **Any particular agent's perceive → decide → act logic.** That lives in
  `lapis-pm/`, and future agents (code-review-agent, session-weaver, etc.)
  will live in their own repos.
- **LLM routing policy.** Callers choose `call_llm` (local, free) vs.
  `call_claude_cli` (Max, better) per-task.
- **Notification rules or throttling.** Callers decide when to notify.
- **GPU task semantics.** `GPUQueue` is a queue primitive; the task runner
  (`gpu-queue-runner.service` at `/srv/agents/scripts/gpu_queue_runner.py`)
  executes submitted tasks and stays outside this package.
- **Chub content or bundles.** `agents_core.llm.call_llm` imports
  `chub_broker` lazily when `bundle_ids=` is passed, so `/srv/agents/scripts/`
  must be on PYTHONPATH if you use that feature. `chub_broker` itself is
  a candidate for future inclusion in agents-core once a second consumer asks.
- **Orchestration schedules, research-domain logic, anthro/TTRPG/psych miners,
  convergence analysis, dashboards.** All stay in `/srv/agents/` as consumers.

## Deferred candidates

These modules in `/srv/agents/scripts/` have multiple callers and may join
agents-core when a second consumer materializes:

- `gpu_coordinator` (18 callers) — strong candidate, intimately paired with `gpu`
- `taskqueue` (9 callers)
- `research_schedule` (7 callers)
- `embed_utils` (8 callers)
- `rag_cache` (3 callers)
- `chub_loader` / `chub_broker`

## Shim migration convention

When a module is extracted into this package, the old path at
`/srv/agents/scripts/<name>.py` (or `/srv/agents/dashboard/<name>.py` for
`comment_store`) is converted to a re-export shim:

```python
# /srv/agents/scripts/target_store.py (shim)
"""Compatibility shim. Canonical source: agents_core.targets.
This file will be deleted once all callers use the namespaced import.
"""
from agents_core.targets import *  # noqa: F401, F403
from agents_core.targets import (  # names a wildcard might miss
    Target, TargetStore, URGENCY_ORDER, SCHEDULE, ...
)
```

Consumers then get flipped in batches to `from agents_core.<module> import ...`,
and once all consumers are flipped the shim file is deleted. This pattern keeps
timer-fired services stable across the cutover.

## Governance

- PRs required to merge into `main`. Same Forgejo branch protection as the rest
  of the StarHouse repos.
- Schema changes to `Target` / `Comment` / `MemoryStore` rows are explicit
  breaking changes — call them out in PR titles with `schema:` prefix so
  consumers (dashboard, night orch, lapis-pm) can be updated in lockstep.
- No new abstractions (`AgentRunner` base classes, unified `LLMCall` types,
  etc.) without a second consumer asking for them. Move-only is the default.

## Install model

Editable install: `pip install -e /srv/git/agents-core-working` into
`/home/user/.local/lib/python3.12/site-packages`. Systemd units already
have that path on their PYTHONPATH (see `/srv/agents/systemd/*.service`).

No PyPI. The git SHA is the version.
