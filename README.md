# agents-core

Shared infrastructure primitives for StarHouse Claude agents. Editable-install
locally; every Python process on the box gets `import agents_core.*` without
`sys.path` tricks.

## Modules

| Import | What it does |
|--------|--------------|
| `agents_core.llm` | `call_llm` (llama-server), `call_claude_cli` (Claude CLI, Max subscription) |
| `agents_core.forgejo` | Forgejo REST helper: `create_pr`, `merge_pr`, `get_pr_diff`, `add_comment`, ... |
| `agents_core.notify` | `send_notification`, `Priority` — Pushover wrapper |
| `agents_core.targets` | `Target`, `TargetStore` — YAML targets at `/srv/lapis/targets/` |
| `agents_core.comments` | `Comment`, `CommentStore` — per-target comment log |
| `agents_core.gpu` | `GPUQueue`, `Priority` — GPU task queue at `/srv/lapis/gpu-queue/` |
| `agents_core.mem` | `MemoryStore` — SQLite + FTS5 cross-instance memory library |

The `mem` CLI stays at `/srv/agents/scripts/mem.py` (imports `MemoryStore` from
this package) and is invoked via `/usr/local/bin/mem`.

## Install

```bash
pip install -e /srv/git/agents-core-working
```

Install is already done on StarHouse. After editing any file in this repo, the
change is live in every Python process that re-imports the module.

## Tests

```bash
cd /srv/git/agents-core-working && python3 -m pytest
```

## Scope

See [SPEC.md](SPEC.md) for what this package owns vs. doesn't, and the shim
migration convention used to extract these modules without breaking consumers.
