"""Agent bundle loader - load, resolve, render, and invoke agent bundle directories.

Public surface:
    Bundle            - parsed bundle (cheap; holds raw component data)
    InvocationResult  - result of a full invoke() round-trip
    load(path)        - read manifest + component files from disk
    invoke(path, ...) - end-to-end: load → resolve context → render → call_operator → record

Consumers:
    from agents_core.bundle import load, invoke

v0 invariants:
    - No tool execution (tool_allowlist is descriptive context only)
    - No retry_strategy enforcement (parsed, exposed, not consumed)
    - No eval_criteria enforcement (parsed, exposed, not consumed)
    - No caching (cache_scope parsed but ignored)
    - Strict undefined in Jinja2 - missing variables raise at render, not silently empty
    - No backward-compat shim for harness_id - loader reads agent_id only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agents_core.llm import call_operator  # module-level import enables patch("agents_core.bundle.call_operator")

logger = logging.getLogger(__name__)

_VALID_OPERATOR_CLASSES = frozenset({"qwen", "sonnet", "opus", "haiku", "gravitywell"})
_RESERVED_OPERATOR_KWARGS = frozenset({"operator_class", "prompt", "system", "model", "bundle_ids"})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bundle:
    """Parsed bundle. Cheap to construct; holds raw component data, no resolved context."""
    path: Path
    agent_id: str
    operator_class: str            # one of: "qwen", "sonnet", "opus", "haiku"
    task: str                      # e.g. "code-reviewer"
    manifest: dict                 # full manifest.yaml as dict (for forward fields)
    system_prompt_template: str    # raw .j2 source
    context_injection: dict        # parsed context_injection.yaml
    tool_allowlist: dict           # parsed tool_allowlist.yaml
    retry_strategy: dict | None    # parsed retry_strategy.yaml (or None if absent)
    eval_criteria: dict | None     # parsed eval_criteria.yaml (or None if absent)


@dataclass(frozen=True)
class InvocationResult:
    response: str | None           # operator response text, or None when operator returned empty content.
                                   # May propagate agents_core.llm.OperatorUnreachableError if the
                                   # operator backend is unreachable after retries. response=None means
                                   # the operator returned empty content (semantic empty response); use
                                   # the exception to distinguish backend outages.
    rendered_prompt: str           # the fully-rendered system prompt that was sent
    agent_id: str
    operator_class: str
    context_blocks: dict[str, str] # block_id -> resolved content (post-truncation, post-budget)
    observation_path: Path | None  # where the observation was written, or None if not recorded


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def load(bundle_path: str | Path) -> Bundle:
    """Read manifest.yaml and component files from disk.

    Raises ValueError with a file-pointing message on any malformed component.
    Does NOT resolve context blocks or render the template.
    """
    bundle_path = Path(bundle_path)
    manifest_file = bundle_path / "manifest.yaml"

    if not manifest_file.exists():
        raise ValueError(f"Bundle manifest not found: {manifest_file}")

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse {manifest_file}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Expected YAML mapping in {manifest_file}, got {type(manifest).__name__}")

    # Validate schema_version
    schema_version = manifest.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported schema_version {schema_version!r} in {manifest_file}; expected 1"
        )

    # Validate agent_id
    agent_id = manifest.get("agent_id")
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError(f"Missing or empty 'agent_id' in {manifest_file}")

    # Validate operator_class
    operator_class = manifest.get("operator_class")
    if operator_class not in _VALID_OPERATOR_CLASSES:
        raise ValueError(
            f"Invalid operator_class {operator_class!r} in {manifest_file}; "
            f"must be one of {sorted(_VALID_OPERATOR_CLASSES)}"
        )

    # Validate task
    task = manifest.get("task")
    if not task or not isinstance(task, str):
        raise ValueError(f"Missing or empty 'task' in {manifest_file}")

    # Validate components structure
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"Missing or malformed 'components' mapping in {manifest_file}")

    required_components = ("system_prompt", "context_injection", "tool_allowlist")
    for key in required_components:
        if key not in components:
            raise ValueError(
                f"Missing required component key '{key}' in {manifest_file}; "
                f"required: {list(required_components)}"
            )

    # Load required component files
    system_prompt_template = _load_component_file(bundle_path, components, "system_prompt", manifest_file)
    context_injection_raw = _load_component_file(bundle_path, components, "context_injection", manifest_file)
    tool_allowlist_raw = _load_component_file(bundle_path, components, "tool_allowlist", manifest_file)

    # Parse YAML component files
    context_injection = _parse_yaml_component(context_injection_raw, bundle_path, components["context_injection"])
    tool_allowlist = _parse_yaml_component(tool_allowlist_raw, bundle_path, components["tool_allowlist"])

    # Optional components
    retry_strategy: dict | None = None
    if "retry_strategy" in components:
        raw = _load_component_file(bundle_path, components, "retry_strategy", manifest_file)
        retry_strategy = _parse_yaml_component(raw, bundle_path, components["retry_strategy"])

    eval_criteria: dict | None = None
    if "eval_criteria" in components:
        raw = _load_component_file(bundle_path, components, "eval_criteria", manifest_file)
        eval_criteria = _parse_yaml_component(raw, bundle_path, components["eval_criteria"])

    return Bundle(
        path=bundle_path,
        agent_id=agent_id,
        operator_class=operator_class,
        task=task,
        manifest=manifest,
        system_prompt_template=system_prompt_template,
        context_injection=context_injection,
        tool_allowlist=tool_allowlist,
        retry_strategy=retry_strategy,
        eval_criteria=eval_criteria,
    )


def _load_component_file(bundle_path: Path, components: dict, key: str, manifest_file: Path) -> str:
    """Load the content of a component file, raising ValueError if missing."""
    filename = components[key]
    file_path = bundle_path / filename
    if not file_path.exists():
        raise ValueError(
            f"Component file not found for '{key}': {file_path} "
            f"(declared in {manifest_file})"
        )
    return file_path.read_text(encoding="utf-8")


def _parse_yaml_component(raw: str, bundle_path: Path, filename: str) -> dict:
    """Parse a YAML component file, returning an empty dict if the content is None/empty."""
    file_path = bundle_path / filename
    try:
        result = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML component {file_path}: {exc}") from exc
    # Empty YAML files are valid (e.g. empty tool list) - return empty dict
    if result is None:
        return {}
    if not isinstance(result, dict):
        raise ValueError(
            f"Expected YAML mapping in {file_path}, got {type(result).__name__} - "
            f"component files must be top-level mappings, not lists or scalars"
        )
    return result


# ---------------------------------------------------------------------------
# invoke()
# ---------------------------------------------------------------------------

def invoke(
    bundle_path: str | Path,
    *,
    task_intent: str,
    retry_state: str | None = None,
    target_id: str | None = None,
    backends: dict | None = None,
    record_observation: bool = True,
    **operator_kwargs,
) -> InvocationResult:
    """End-to-end: load → resolve context → render template → call_operator → record observation.

    backends: optional dict for sandbox use. Keys "mem" and "observations" override the
        default agents_core.mem and agents_core.observations modules.
    operator_kwargs: forwarded verbatim to call_operator() (timeout, json_mode, temperature, etc.).

    May propagate agents_core.llm.OperatorUnreachableError if the operator backend is
    unreachable after retries. InvocationResult.response=None means the operator returned
    empty content (semantic empty response); use the exception to distinguish backend outages.
    """
    # Resolve backends at the top - pass them down, never re-import inside helpers
    if backends is None:
        backends = {}

    mem_backend = backends.get("mem")   # None = lazy-init MemoryStore on first need
    obs_backend = backends.get("observations")  # None = use agents_core.observations

    if obs_backend is None:
        from agents_core import observations as obs_module
        obs_backend = obs_module

    # Pre-validate reserved kwargs before touching anything else
    for reserved in _RESERVED_OPERATOR_KWARGS:
        if reserved in operator_kwargs:
            raise ValueError(
                f"operator_kwargs contains reserved key {reserved!r}; "
                f"the loader supplies this argument - remove it from operator_kwargs"
            )

    bundle = load(bundle_path)

    # Resolve context blocks
    mem_backend_resolved, context_blocks = _resolve_context_blocks(
        bundle, mem_backend=mem_backend, obs_backend=obs_backend,
    )

    # Render tool allowlist
    tools_str = _render_tool_allowlist(bundle.tool_allowlist)

    # Render system prompt template
    rendered_prompt = _render_template(
        bundle.system_prompt_template,
        task_intent=task_intent,
        retry_state=retry_state,
        context_blocks=context_blocks,
        tools_str=tools_str,
    )

    # Call operator
    response = call_operator(
        operator_class=bundle.operator_class,
        prompt=task_intent,
        system=rendered_prompt,
        **operator_kwargs,
    )

    # Record observation
    observation_path: Path | None = None
    if record_observation:
        try:
            observation_path = obs_backend.record(
                agent_id=bundle.agent_id,
                observation_type="decision",
                context=f"bundle invocation: {task_intent[:120]}",
                content=response or "(operator returned None)",
                target_id=target_id,
                tags=[bundle.agent_id, f"operator:{bundle.operator_class}", f"task:{bundle.task}"],
                extra={
                    "rendered_prompt_chars": len(rendered_prompt),
                    "context_block_ids": list(context_blocks.keys()),
                },
            )
        except Exception as exc:
            logger.warning("Observation write failed (invocation result unaffected): %s", exc)
            observation_path = None

    return InvocationResult(
        response=response,
        rendered_prompt=rendered_prompt,
        agent_id=bundle.agent_id,
        operator_class=bundle.operator_class,
        context_blocks=context_blocks,
        observation_path=observation_path,
    )


# ---------------------------------------------------------------------------
# Context block resolution
# ---------------------------------------------------------------------------

def _resolve_context_blocks(
    bundle: Bundle,
    *,
    mem_backend: Any,
    obs_backend: Any,
) -> tuple[Any, dict[str, str]]:
    """Resolve all context_injection blocks. Returns (mem_backend_used, blocks_dict)."""
    blocks_cfg = bundle.context_injection.get("blocks") or []
    total_budget = bundle.context_injection.get("total_budget_chars") or 0

    resolved: dict[str, str] = {}    # block_id -> text
    block_order: list[str] = []      # declaration order (for budget trimming)

    for block in blocks_cfg:
        block_id = block["id"]
        source = block["source"]
        params = block.get("params") or {}
        max_chars = block.get("max_chars") or 0
        truncation = block.get("truncation", "tail")
        on_empty = block.get("on_empty", "omit")

        if source == "mem.search":
            mem_backend, text = _resolve_mem_search(block_id, params, mem_backend)
        elif source == "agent_observations":
            text = _resolve_agent_observations(block_id, params, obs_backend)
        else:
            raise ValueError(
                f"Unsupported context source {source!r} in block {block_id!r}; "
                f"v0 supports: 'mem.search', 'agent_observations'"
            )

        if not text:
            text = _apply_on_empty(block_id, on_empty)

        if text and max_chars and len(text) > max_chars:
            text = _truncate(text, max_chars, truncation)

        resolved[block_id] = text
        block_order.append(block_id)

    # Total budget enforcement
    if total_budget and total_budget > 0:
        total = sum(len(v) for v in resolved.values())
        if total > total_budget:
            per_block_sizes = {k: len(v) for k, v in resolved.items()}
            overage = total - total_budget
            logger.warning(
                "Context block budget exceeded: total=%d, budget=%d, overage=%d, per_block=%s",
                total, total_budget, overage, per_block_sizes,
            )
            resolved = _enforce_budget(resolved, block_order, total_budget, bundle.context_injection.get("blocks") or [])

    return mem_backend, resolved


def _resolve_mem_search(block_id: str, params: dict, mem_backend: Any) -> tuple[Any, str]:
    """Resolve a mem.search block. Returns (mem_backend, text)."""
    # Validate params: only key_prefix allowed in v0
    unsupported = set(params.keys()) - {"key_prefix"}
    if unsupported:
        raise ValueError(
            f"mem.search block {block_id!r}: unsupported params {sorted(unsupported)}; "
            f"v0 only supports 'key_prefix'"
        )

    key_prefix = params.get("key_prefix")
    if not key_prefix:
        raise ValueError(
            f"mem.search block {block_id!r}: 'key_prefix' param is required"
        )

    if mem_backend is None:
        from agents_core.mem import MemoryStore
        mem_backend = MemoryStore()

    entries = mem_backend.list_by_prefix(prefix=key_prefix, limit=50)
    if not entries:
        return mem_backend, ""

    lines = []
    for entry in entries:
        key = entry.get("key", "")
        content = (entry.get("content") or "").replace("\n", " ")
        lines.append(f"{key}: {content}")
    return mem_backend, "\n".join(lines)


def _resolve_agent_observations(block_id: str, params: dict, obs_backend: Any) -> str:
    """Resolve an agent_observations block."""
    # Validate params: only agent_id, tags, limit allowed in v0
    unsupported = set(params.keys()) - {"agent_id", "tags", "limit"}
    if unsupported:
        raise ValueError(
            f"agent_observations block {block_id!r}: unsupported params {sorted(unsupported)}; "
            f"v0 supports: 'agent_id', 'tags', 'limit'"
        )

    agent_id = params.get("agent_id")
    if not agent_id:
        raise ValueError(
            f"agent_observations block {block_id!r}: 'agent_id' param is required"
        )

    tags = params.get("tags") or None
    limit = params.get("limit", 10)

    entries = obs_backend.search(agent_id=agent_id, tags_all=tags, limit=limit)
    if not entries:
        return ""

    lines = []
    for entry in entries:
        ts = entry.get("timestamp", "")
        obs_type = entry.get("observation_type", "")
        content = (entry.get("content") or "").replace("\n", " ")
        lines.append(f"[{ts} {obs_type}] {content}")
    return "\n".join(lines)


def _apply_on_empty(block_id: str, on_empty: str) -> str:
    """Apply on_empty policy. Returns the replacement text (empty string for 'omit')."""
    if on_empty == "omit":
        return ""
    if on_empty.startswith("placeholder:"):
        return on_empty[len("placeholder:"):]
    # Default to omit for unrecognized policies
    return ""


def _truncate(text: str, max_chars: int, truncation: str) -> str:
    """Truncate text to max_chars using head or tail strategy, appending marker."""
    marker = f"\n[truncated to {max_chars} chars]"
    if truncation == "head":
        return text[:max_chars] + marker
    else:  # tail
        return text[-max_chars:] + marker


def _enforce_budget(
    resolved: dict[str, str],
    block_order: list[str],
    total_budget: int,
    blocks_cfg: list[dict],
) -> dict[str, str]:
    """Truncate blocks in reverse declaration order until total fits budget.

    Uses plain head/tail slicing (no marker appended) so the resulting total
    is strictly within total_budget. The per-block max_chars marker is separate.
    """
    truncation_rules: dict[str, str] = {
        block["id"]: block.get("truncation", "tail") for block in blocks_cfg
    }

    result = dict(resolved)

    for block_id in reversed(block_order):
        current_total = sum(len(v) for v in result.values())
        if current_total <= total_budget:
            break
        remaining_budget = total_budget - sum(
            len(v) for k, v in result.items() if k != block_id
        )
        text = result[block_id]
        if len(text) <= max(remaining_budget, 0):
            continue
        if remaining_budget <= 0:
            result[block_id] = ""
        else:
            trunc_rule = truncation_rules.get(block_id, "tail")
            if trunc_rule == "head":
                result[block_id] = text[:remaining_budget]
            else:
                result[block_id] = text[-remaining_budget:]

    return result


# ---------------------------------------------------------------------------
# Tool allowlist rendering
# ---------------------------------------------------------------------------

def _render_tool_allowlist(tool_allowlist: dict) -> str:
    """Render tool_allowlist.yaml tools as a markdown bullet list."""
    tools = tool_allowlist.get("tools") or []
    if not tools:
        return ""
    lines = []
    for tool in tools:
        name = tool.get("name", "")
        kind = tool.get("kind", "")
        purpose = tool.get("purpose", "")
        lines.append(f"- {name} ({kind}): {purpose}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

class _ContextNamespace:
    """Simple namespace object so templates can write {{ context.block_id }}."""
    def __init__(self, blocks: dict[str, str]):
        for k, v in blocks.items():
            setattr(self, k, v)


def _render_template(
    template_src: str,
    *,
    task_intent: str,
    retry_state: str | None,
    context_blocks: dict[str, str],
    tools_str: str,
) -> str:
    """Render the Jinja2 system prompt template with all required variables."""
    import jinja2

    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    template = env.from_string(template_src)

    # Normalize retry_state: None → "" so {% if retry_state %} evaluates falsy
    # and {{ retry_state }} renders as empty string (StrictUndefined would not
    # stringify None correctly in all contexts)
    retry_state_value = retry_state if retry_state is not None else ""

    context_ns = _ContextNamespace(context_blocks)

    return template.render(
        task_intent=task_intent,
        retry_state=retry_state_value,
        context=context_ns,
        tools=tools_str,
    )
