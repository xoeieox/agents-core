# AGENTS.md - working notes for AI agents in this repo

agents-core is the **shared infrastructure primitives** package for the
StarHouse agent fleet: the library surface (`agents_core.*`) that agent
processes import without `sys.path` tricks - LLM plumbing, Forgejo client,
notifications, targets/comments, GPU queue, memory (SQLite + FTS5), shaped
agent execution, doorman leases, slots - plus thin FastAPI servers wrapping
the `mem` and `gpu` modules for cross-host access. See `README.md` for the
module table and `SPEC.md` for the contract surface.

## Hard rules

- Library first: the importable module is the contract. The servers
  (`mem_server`, `gpu_server`, `slot_server`, `doorman_server`,
  `dowser_server`) are thin wrappers; do not add logic to a server layer
  that is not available through the library module.
- No credentials in-repo, ever. Tokens and base URLs come from the
  environment (see `docs/deploying-*.md`).
- Shared state (`/srv/lapis/targets/`, `/srv/lapis/gpu-queue/`,
  `/srv/lapis/comments/`) is on-disk state consumed by every agent process.
  A change to a store's on-disk shape is a breaking change for all
  consumers - brief the owner first.
- `council/` and some server modules import sibling packages
  (`archetypes_core`, `lapis_engine`). Do not inline sibling code here;
  depend on them instead.
- Conventional commits (`feat(scope): ...`, `fix(scope): ...`).

## Dev loop

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
pytest
```

The deploy guides in `docs/` cover running `mem-server` and the GPU queue
server as systemd services.
