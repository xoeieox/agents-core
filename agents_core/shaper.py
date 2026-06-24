"""Shaped-agent registry + dispatch.

Instantiate with a per-consumer registry.yaml:

    from agents_core.shaper import Shaper
    _SHAPER = Shaper(Path(__file__).parent / "registry.yaml")

Then dispatch via:

    result = _SHAPER.dispatch(agent_type, target_id, user_prompt, vars_=...)

The module-level singleton must be constructed at import time (not lazily)
to preserve fail-fast semantics: a malformed registry.yaml crashes the daemon
at import, not at first dispatch.

The runner is invoked as `python3 -m agents_core.shaped_runner <spec.json>`
(module-resolved, no hardcoded paths).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agents_core.claude_queue import ClaudeQueue
from agents_core.gpu import GPUQueue, Priority

# chub_broker still lives in /srv/agents/scripts/ (deferred from the agents-core
# day-one scope). Keep this last sys.path shim until chub_broker moves into
# agents-core, at which point this block + the import can be deleted.
if "/srv/agents/scripts" not in sys.path:
    sys.path.insert(0, "/srv/agents/scripts")

try:
    from chub_broker import select_bundles_by_ids, compose_with_system  # noqa: E402
except Exception:
    select_bundles_by_ids = None  # type: ignore
    compose_with_system = None  # type: ignore


SPEC_DIR = Path("/srv/lapis/gpu-queue/shaped")
RUNNER_MODULE = "agents_core.shaped_runner"

# Per-repo working-clone convention. A repo name like "lapis-engine" maps to
# /srv/git/lapis-engine-working/. Shaped agents dispatch with this as cwd so
# chub-inject.py (SessionStart hook) finds the repo CLAUDE.md and injects its
# @chub: bundles + per-project auto-memory into the subprocess context.
_REPO_CWD_TEMPLATE = "/srv/git/{repo}-working"
# Fallback when the working clone doesn't exist (test runs, unknown repo). The
# subprocess still works — just without repo-specific hook injection.
_DEFAULT_CWD = "/srv/agents"


def _slugify(text: str, max_len: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "task")[:max_len]


def _record_dispatch_slot(
    *,
    slot_id: str,
    project_id: str,
    agent_type: str,
    task_id: str,
    vars_: dict | None,
    user_prompt: str,
) -> None:
    """Open a project-slot on the BRIX blackboard for a shaped dispatch.

    Best-effort: never raises. Slot-store unavailability (off-master, server down,
    not-yet-deployed) must not block a dispatch — the dispatch is the real work; the
    slot is coordination metadata. The project is the target; the slot is this unit of
    work; the contributor-of-record is the dispatched agent (keyed by task_id).
    """
    try:
        from agents_core.slots import SlotStore

        horizon = {
            "project_summary": (vars_ or {}).get("spec_summary") or "",
            "immediate_goal": (user_prompt or "").strip()[:280],
            "adjacent_slots": [],
        }
        SlotStore().create_slot(
            project_id=project_id,
            contributor={"type": agent_type, "id": task_id},
            horizon=horizon,
            slot_id=slot_id,
        )
    except Exception as exc:  # never block dispatch
        print(f"[slots:dispatch-open-failed] {slot_id}: {exc}", file=sys.stderr)


@dataclass
class ShapedAgent:
    name: str
    chub_bundles: list[str]
    system_template: str
    model: str
    timeout_s: int
    capture_meta: bool = False
    notify: bool = False
    notify_policy: str = "always"
    engine: str = "claude"


@dataclass
class DispatchResult:
    task_id: str
    agent_type: str
    spec_path: str
    spec_id: str        # short uuid embedded in spec filename (used to find meta sidecar)
    output_path: str    # where the queue runner will write the result


class Shaper:
    """Shaped-agent registry and dispatcher.

    One instance per consumer registry.yaml. Constructed at module scope
    (fail-fast on malformed registry). Each consumer owns its own registry.yaml;
    agents-core ships none.
    """

    def __init__(self, registry_path: Path | str):
        self.registry_path = Path(registry_path)
        if not self.registry_path.exists():
            raise RuntimeError(f"Shaper registry missing: {self.registry_path}")
        self._shared_preamble: str = ""
        self._registry: dict[str, ShapedAgent] = {}
        self._force_gpu_warned: bool = False
        self._load_registry()

    def _load_registry(self) -> None:
        """Parse registry.yaml into self._shared_preamble + self._registry.

        Missing `agents:` key → empty registry (no crash). Missing/malformed
        YAML raises RuntimeError (fail-fast at import time).
        """
        try:
            raw = yaml.safe_load(self.registry_path.read_text()) or {}
        except Exception as e:
            raise RuntimeError(f"Shaper registry malformed: {self.registry_path}: {e}") from e
        self._shared_preamble = raw.get("shared_preamble") or ""
        registry: dict[str, ShapedAgent] = {}
        for name, body in (raw.get("agents") or {}).items():
            registry[name] = ShapedAgent(
                name=name,
                chub_bundles=list(body.get("chub_bundles") or []),
                system_template=body.get("system_template", ""),
                model=body.get("model", "haiku"),
                timeout_s=int(body.get("timeout_s", 300)),
                capture_meta=bool(body.get("capture_meta", False)),
                notify=bool(body.get("notify", False)),
                notify_policy=str(body.get("notify_policy", "always")),
                engine=str(body.get("engine", "claude")),
            )
        self._registry = registry

    def reload_registry(self) -> None:
        """Re-read registry.yaml. Useful after an in-place registry edit."""
        self._load_registry()

    def list_agents(self) -> list[str]:
        return sorted(self._registry.keys())

    def get_agent(self, name: str) -> ShapedAgent:
        if name not in self._registry:
            raise KeyError(f"Unknown shaped agent: {name}. Known: {self.list_agents()}")
        return self._registry[name]

    @staticmethod
    def resolve_repo_cwd(repo: str) -> str:
        """Map a repo name to its working-clone path, falling back to _DEFAULT_CWD
        when the clone doesn't exist. Returned path is what the shaped-agent
        subprocess uses as cwd — determining which CLAUDE.md + hooks fire."""
        if not repo:
            return _DEFAULT_CWD
        # Accept bare name ("lapis-engine") or owner-prefixed ("Erah/lapis-engine").
        bare = repo.rsplit("/", 1)[-1]
        candidate = _REPO_CWD_TEMPLATE.format(repo=bare)
        return candidate if Path(candidate).is_dir() else _DEFAULT_CWD

    def _compose_system(self, agent: ShapedAgent, vars_: dict) -> str:
        # Preamble is prepended to every agent's composed prompt. Its format
        # vars (repo, repo_cwd, target_id) are supplied by dispatch(). Missing
        # vars in the preamble are a programmer error — fail loud at format time
        # rather than silently producing a half-rendered preamble.
        preamble = self._shared_preamble.format(**vars_) if self._shared_preamble else ""
        base = agent.system_template.format(**vars_)
        composed = f"{preamble}\n\n---\n\n{base}" if preamble else base
        if agent.chub_bundles and select_bundles_by_ids and compose_with_system:
            try:
                sel = select_bundles_by_ids(agent.chub_bundles)
                return compose_with_system(sel, composed)
            except Exception:
                return composed
        return composed

    def dispatch(
        self,
        agent_type: str,
        target_id: str,
        user_prompt: str,
        vars_: dict | None = None,
        priority: int = Priority.HIGH,
        submitted_by: str = "agents-core",
    ) -> DispatchResult:
        """Submit a shaped-agent invocation to the appropriate queue.

        vars_ are interpolated into the agent's system_template via str.format.
        user_prompt is passed straight through to Claude as the user message.
        """
        agent = self.get_agent(agent_type)
        vars_ = dict(vars_ or {})

        # Derive the subprocess cwd from the repo. The shaped agent's
        # SessionStart hooks (chub-inject.py, per-project auto-memory) key off
        # this — see _REPO_CWD_TEMPLATE docstring. Always populate repo_cwd in
        # vars so the shared_preamble can reference it.
        repo_cwd = self.resolve_repo_cwd(vars_.get("repo") or "")
        vars_.setdefault("repo_cwd", repo_cwd)

        SPEC_DIR.mkdir(parents=True, exist_ok=True)

        system = self._compose_system(agent, vars_)

        spec = {
            "agent_type": agent.name,
            "target_id": target_id,
            "repo": vars_.get("repo", ""),
            "engine": agent.engine,
            "model": agent.model,
            "timeout_s": agent.timeout_s,
            "system": system,
            "prompt": user_prompt,
            # cwd that shaped_runner.py passes to call_claude_cli. Hooks fire
            # against the CLAUDE.md at this path — that's what pulls chubs +
            # auto-memory into the shaped agent's context.
            "cwd": repo_cwd,
            # Shaped agents run headless with no human to answer Claude Code's
            # workspace-trust prompt. `claude -p` skips that prompt's dialog and
            # returns "please allow writes to <path>" when the workspace isn't
            # cached-trusted. bypassPermissions sidesteps that — safe because
            # the authority gate upstream (spec + --authority flag) already
            # bounds what the shaped agent is allowed to do.
            "permission_mode": "bypassPermissions",
            # Data-driven per agent. Default false (safer than name-based inference).
            "capture_meta": agent.capture_meta,
        }

        spec_id = uuid.uuid4().hex[:12]
        spec_path = SPEC_DIR / f"{target_id}-{agent.name}-{spec_id}.json"
        # The slot_id IS the spec_id: stable, unique per dispatch, already carried on
        # DispatchResult and recorded by lapis-pm, so the contributor-of-record that
        # completes the slot later keys off the same id. Hand it to the runner too so
        # mid-run checkpoints (Reality Snap) can target this slot.
        # slot_id is set here so the runner spec is self-contained regardless of
        # whether _record_dispatch_slot (best-effort, below) succeeds. The slot
        # may not exist in the blackboard if the store is unavailable, but the
        # runner can still carry the id for mid-run checkpoints.
        spec["slot_id"] = spec_id

        cmd = f"python3 -m {RUNNER_MODULE} {shlex.quote(str(spec_path))}"

        # Route by agent.model. Anthropic-API models (sonnet/haiku/opus) →
        # ClaudeQueue (API-backed, parallel, per-task worktree). Qwen → GPUQueue
        # (GPU-serialized, no worktree, TOU-paused 4–9 PM because it uses real
        # local GPU hardware). Opus was previously routed to GPUQueue as a cutover
        # oversight; it shares the Anthropic API path with sonnet/haiku and has no
        # local-GPU resource to gate on.
        #
        # AGENTS_CORE_FORCE_GPU_QUEUE=1 is the emergency rollback knob.
        # LAPIS_PM_FORCE_GPU_QUEUE=1 is the deprecated alias (one merge cycle).
        force_gpu = os.getenv("AGENTS_CORE_FORCE_GPU_QUEUE") == "1"
        if not force_gpu and os.getenv("LAPIS_PM_FORCE_GPU_QUEUE") == "1":
            force_gpu = True
            if not self._force_gpu_warned:
                print(
                    "DeprecationWarning: LAPIS_PM_FORCE_GPU_QUEUE is deprecated; "
                    "use AGENTS_CORE_FORCE_GPU_QUEUE",
                    file=sys.stderr,
                )
                self._force_gpu_warned = True

        # local-fixer does remote HTTP work via GravityWell — it has no local-GPU
        # dependency and no GPU-queue consumer on BRIX. Route it to ClaudeQueue so
        # the live claude-queue-runner executes it and lapis-pm can reconcile the
        # dispatch record via ClaudeQueue.get_recent_failed/completed.
        # force_gpu does not apply to local-fixer (there is no GPU path to fall back to).
        route_to_claude = (
            agent.engine == "local-fixer"
            or (agent.model in {"sonnet", "haiku", "opus"} and not force_gpu)
        )

        if route_to_claude:
            # Generate task_id BEFORE the single spec write so the runner cannot
            # claim a spec that's missing task_id or worktree_required. See spec
            # §Shaper routing ("Why generate before write, not after submit").
            queue = ClaudeQueue()
            task_id = queue._generate_id(slug=f"{agent.name}-{target_id}")
            spec["task_id"] = task_id
            spec["base_branch"] = "main"
            # local-fixer manages its own worktree inside _run_local_fixer;
            # setting worktree_required=True would cause the runner to set up a
            # competing worktree before the engine even starts.
            spec["worktree_required"] = agent.engine != "local-fixer"
            spec_path.write_text(json.dumps(spec, ensure_ascii=False))
            queue.submit({
                "task_type": "subprocess",
                "priority": priority,
                "timeout_seconds": agent.timeout_s + 60,
                "submitted_by": submitted_by,
                "model": agent.model,
                "description": f"{agent.name}:{target_id}",
                "notify": agent.notify,
                "notify_policy": agent.notify_policy,
                "payload": {"command": cmd, "spec_path": str(spec_path)},
            }, task_id=task_id)
            output_path = f"/srv/lapis/claude-queue/completed/{task_id}-output.md"
        else:
            spec_path.write_text(json.dumps(spec, ensure_ascii=False))
            queue = GPUQueue()
            task_id = queue.submit({
                "task_type": "subprocess",
                "priority": priority,
                "timeout_seconds": agent.timeout_s + 60,
                "submitted_by": submitted_by,
                "model": agent.model,
                "payload": {"command": cmd},
            })
            output_path = f"/srv/lapis/gpu-queue/completed/{task_id}-output.md"

        # Open a project-slot on the blackboard for this dispatch. Best-effort:
        # slot-store trouble must never block a real dispatch (mirrors
        # router_portfolio's emit discipline). slot_id == spec_id; contributor-of-
        # record == task_id, which is what lapis-pm matches on at completion.
        _record_dispatch_slot(
            slot_id=spec_id,
            project_id=target_id,
            agent_type=agent.name,
            task_id=task_id,
            vars_=vars_,
            user_prompt=user_prompt,
        )

        return DispatchResult(
            task_id=task_id,
            agent_type=agent.name,
            spec_path=str(spec_path),
            spec_id=spec_id,
            output_path=output_path,
        )
