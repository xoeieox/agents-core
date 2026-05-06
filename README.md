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

## Scope

See [SPEC.md](SPEC.md) for what this package owns vs. doesn't, and the shim
migration convention used to extract these modules without breaking consumers.
