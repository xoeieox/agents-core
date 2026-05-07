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
| `agents_core.observations` | `record`, `search`, `root` — append-only per-agent observation substrate |

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

## Agent Observation Substrate

Cross-cutting primitive for substrate-mediated learning across all Lapis agents. Every
agent (Lapis PM, Tech-Kami, Code Reviewer, Harness Engineer, future Experts) can record
observations — friction, decisions, lessons, anomalies, interventions — to a shared
append-only JSONL store. Sessions can then query prior observations to avoid re-discovering
the same lessons.

### Where files land

```
/srv/lapis/agent-observations/<agent_id>/<YYYY-MM-DD>.jsonl
```

Override the root for tests via `AGENT_OBSERVATIONS_ROOT` env var.

### Log an observation from Python

```python
from agents_core.observations import record

record(
    agent_id="lapis-pm",
    observation_type="friction",       # friction | decision | lesson | anomaly | intervention
    context="running the test suite",
    content="pytest discovery was slow due to missing __init__.py",
    session_id="sess-abc",             # optional
    target_id="t-42",                  # optional, lapis-pm target_id
    tags=["agents-core", "tests"],     # optional free-form
)

# intervention requires intervention_shape:
record(
    agent_id="tech-kami",
    observation_type="intervention",
    context="reviewing PR #17",
    content="The proposed abstraction leaks implementation details upward.",
    intervention_shape="counter-example",  # see vocab below
)
```

### Log an observation from CLI

```bash
python -m agents_core.observations record \
  --agent-id lapis-pm \
  --type friction \
  --context "running tests" \
  --content "pytest was slow" \
  --tag agents-core --tag tests

python -m agents_core.observations record \
  --agent-id tech-kami \
  --type intervention \
  --context "PR review" \
  --content "Abstraction leaks details." \
  --intervention-shape counter-example
```

### Search observations

```python
from agents_core.observations import search
from datetime import datetime, timezone

entries = search(
    agent_id="lapis-pm",
    observation_type="friction",
    since=datetime(2026, 5, 1, tzinfo=timezone.utc),
    substring="slow",
    limit=20,
)
```

```bash
python -m agents_core.observations search \
  --agent-id lapis-pm --type friction \
  --since 2026-05-01T00:00:00+00:00 \
  --substring slow --limit 20 --format text

# JSONL output (one entry per line):
python -m agents_core.observations search --format json
```

### Enum vocabulary

**`observation_type`** (required):

| Value | Meaning |
|-------|---------|
| `friction` | Something that slowed the agent down or caused repeated effort |
| `decision` | A choice made and why |
| `lesson` | A non-obvious thing learned that generalises across sessions |
| `anomaly` | Unexpected state, behaviour, or output observed |
| `intervention` | The agent issued a corrective signal to another agent or human |

**`intervention_shape`** (required iff `observation_type == "intervention"`):

| Value | Meaning |
|-------|---------|
| `question` | Asked a clarifying question |
| `pointer` | Pointed to an existing authority (spec, doc, prior decision) |
| `counter-example` | Offered a concrete counter-example to challenge a claim |
| `frame-shift` | Reframed the problem at a higher or different level |
| `constraint` | Stated a hard constraint that rules out an approach |
| `why-trace` | Traced the causal chain explaining why something is the way it is |

### Architectural rationale

See `architecture/agent-observation-substrate-v0` in mem.db for full design
context and the harness program framing (`decision/harness-program-phased-ordering-2026-05-05`).

## Agent Bundle Loader

`agents_core.bundle` is the runtime loader for agent bundle directories. A bundle is a directory containing a `manifest.yaml` plus component files (`system_prompt.md.j2`, `context_injection.yaml`, `tool_allowlist.yaml`, and optionally `retry_strategy.yaml`, `eval_criteria.yaml`).

### Invoke a bundle from Python

```python
from agents_core.bundle import load, invoke

# Inspect the parsed bundle (cheap — no I/O beyond file reads)
bundle = load("/path/to/my-agent/")
print(bundle.agent_id, bundle.operator_class)

# Full end-to-end invocation
result = invoke(
    "/path/to/my-agent/",
    task_intent="Review PR #42 in repo agents-core",
    target_id="pr-42",           # optional — stored in the observation
    record_observation=True,     # default: writes a 'decision' observation entry
)
print(result.response)
print(result.rendered_prompt)    # the system prompt that was sent
print(result.context_blocks)     # block_id -> resolved text (post-truncation)
```

For sandbox / test use, inject fake backends:

```python
result = invoke(
    bundle_path,
    task_intent="...",
    backends={
        "mem": fake_mem_store,          # must expose .list_by_prefix(prefix, limit=...)
        "observations": fake_obs_mod,   # must expose .search(...) and .record(...)
    },
)
```

### Supported operator classes

| Class | Backend | Notes |
|-------|---------|-------|
| `qwen` | Local llama-server (synchronous) | Production path |
| `sonnet` | ClaudeQueue | v0 gap — raises `NotImplementedError`; see `agents-core-claude-queue-sync-surface-v0` |
| `opus` | ClaudeQueue | Same gap |
| `haiku` | ClaudeQueue | Same gap |

### Supported context-injection sources

| Source name | What it calls | Required params |
|-------------|--------------|-----------------|
| `mem.search` | `MemoryStore.list_by_prefix(key_prefix, limit=50)` | `key_prefix` |
| `agent_observations` | `observations.search(agent_id=..., tags_all=..., limit=...)` | `agent_id` |

Forward pointer: additional sources (e.g. `forgejo.get_file`, `vault.search`) are deferred to future binds — add a new `elif source == "..."` branch in `bundle.py:_resolve_context_blocks`.

### v0 invariants

- **No tool execution.** `tool_allowlist.yaml` is rendered as descriptive context for the operator prompt; the loader does not invoke tools or interpret tool-use protocol responses.
- **No retry enforcement.** `retry_strategy.yaml` is parsed and exposed via `Bundle.retry_strategy` but the loader does not loop or back off. Callers wanting retry semantics implement them around `invoke()`.
- **No eval_criteria enforcement.** Same: parsed and exposed, not consumed.
- **No caching.** `cache_scope` is parsed but ignored.
- **Strict undefined.** Jinja2 templates referencing unset variables raise `jinja2.UndefinedError` at render time — silent empty-string fallbacks hide bugs.
- **No `harness_id` shim.** The loader reads `agent_id` only. Bundles with `harness_id:` (the three code-reviewer bundles) are migrated in `code-reviewer-bundle-wire-v0`.

### First real consumer

`code-reviewer-bundle-wire-v0` wires `bundle.invoke()` into `code_reviewer/review.py`, replacing the hardcoded Python-format system prompt, and renames `harness_id` → `agent_id` in the three existing bundle manifests.

## Scope

See [SPEC.md](SPEC.md) for what this package owns vs. doesn't, and the shim
migration convention used to extract these modules without breaking consumers.
