"""Tests for agents_core.bundle — loader, context resolution, template rendering, invoke().

All 22 test cases from the spec:
  1.  load() happy path
  2.  load() rejects unknown schema_version
  3.  load() rejects unknown operator_class
  4.  load() rejects missing required component key
  5.  load() rejects missing component file on disk
  6.  load() preserves forward-compat fields
  7.  Context resolution: mem.search — two records, placeholder on empty
  8.  Context resolution: agent_observations — two entries, omit on empty
  9.  Context resolution: max_chars truncation (tail)
  10. Context resolution: total_budget_chars enforcement
  11. Context resolution: unknown source raises ValueError
  12. Template rendering: variables resolve correctly
  13. Template rendering: missing variable raises UndefinedError
  14. Template rendering: retry_state None evaluates falsy; renders empty
  15. Tool allowlist rendering — three-tool list
  16. invoke() qwen path end-to-end (mocked call_operator)
  17. invoke() Anthropic path raises NotImplementedError
  18. invoke() backend injection — real modules NOT imported
  19. invoke() record_observation=False
  20. invoke() observation write failure → WARN logged, result still returned
  21. invoke() rejects reserved operator_kwargs
  22. MemoryStore.list_by_prefix correctness
"""
from __future__ import annotations

import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures path
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "bundles"


# ---------------------------------------------------------------------------
# Helper: build a temporary bundle directory
# ---------------------------------------------------------------------------

def _make_bundle(
    tmp_path: Path,
    manifest: dict,
    system_prompt: str = "Hello {{ task_intent }}\n",
    context_injection: dict | None = None,
    tool_allowlist: dict | None = None,
    retry_strategy: dict | None = None,
    eval_criteria: dict | None = None,
) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    ci = context_injection if context_injection is not None else {"blocks": []}
    ta = tool_allowlist if tool_allowlist is not None else {"tools": []}

    components: dict[str, Any] = {
        "system_prompt": "system_prompt.md.j2",
        "context_injection": "context_injection.yaml",
        "tool_allowlist": "tool_allowlist.yaml",
    }

    (bundle_dir / "system_prompt.md.j2").write_text(system_prompt, encoding="utf-8")
    (bundle_dir / "context_injection.yaml").write_text(yaml.dump(ci), encoding="utf-8")
    (bundle_dir / "tool_allowlist.yaml").write_text(yaml.dump(ta), encoding="utf-8")

    if retry_strategy is not None:
        components["retry_strategy"] = "retry_strategy.yaml"
        (bundle_dir / "retry_strategy.yaml").write_text(yaml.dump(retry_strategy), encoding="utf-8")

    if eval_criteria is not None:
        components["eval_criteria"] = "eval_criteria.yaml"
        (bundle_dir / "eval_criteria.yaml").write_text(yaml.dump(eval_criteria), encoding="utf-8")

    manifest = dict(manifest, components=components)
    (bundle_dir / "manifest.yaml").write_text(yaml.dump(manifest), encoding="utf-8")

    return bundle_dir


# ---------------------------------------------------------------------------
# 1. load() happy path
# ---------------------------------------------------------------------------

def test_load_happy_path():
    """minimal-qwen fixture loads with all Bundle fields populated correctly."""
    from agents_core.bundle import load, Bundle

    bundle = load(FIXTURES / "minimal-qwen")

    assert isinstance(bundle, Bundle)
    assert bundle.agent_id == "test-minimal-qwen"
    assert bundle.operator_class == "qwen"
    assert bundle.task == "test"
    assert bundle.path == FIXTURES / "minimal-qwen"
    assert "{{ task_intent }}" in bundle.system_prompt_template
    assert isinstance(bundle.context_injection, dict)
    assert isinstance(bundle.tool_allowlist, dict)
    assert bundle.retry_strategy is None
    assert bundle.eval_criteria is None
    assert bundle.manifest["schema_version"] == 1


# ---------------------------------------------------------------------------
# 2. load() rejects unknown schema_version
# ---------------------------------------------------------------------------

def test_load_rejects_unknown_schema_version(tmp_path):
    """schema_version: 2 raises ValueError mentioning the file."""
    from agents_core.bundle import load

    bundle_dir = _make_bundle(tmp_path, {
        "schema_version": 2,
        "agent_id": "test",
        "operator_class": "qwen",
        "task": "test",
    })

    with pytest.raises(ValueError, match="schema_version"):
        load(bundle_dir)


# ---------------------------------------------------------------------------
# 3. load() rejects unknown operator_class
# ---------------------------------------------------------------------------

def test_load_rejects_unknown_operator_class():
    """bad-operator-class fixture raises ValueError listing valid classes."""
    from agents_core.bundle import load

    with pytest.raises(ValueError, match="operator_class") as exc_info:
        load(FIXTURES / "bad-operator-class")

    msg = str(exc_info.value)
    # Must mention valid classes
    assert "qwen" in msg
    assert "sonnet" in msg


# ---------------------------------------------------------------------------
# 4. load() rejects missing required component key
# ---------------------------------------------------------------------------

def test_load_rejects_missing_required_component(tmp_path):
    """manifest without system_prompt key in components raises ValueError."""
    from agents_core.bundle import load

    # Build a manifest that omits system_prompt
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    manifest = {
        "schema_version": 1,
        "agent_id": "test",
        "operator_class": "qwen",
        "task": "test",
        "components": {
            # system_prompt intentionally omitted
            "context_injection": "context_injection.yaml",
            "tool_allowlist": "tool_allowlist.yaml",
        },
    }
    (bundle_dir / "manifest.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (bundle_dir / "context_injection.yaml").write_text(yaml.dump({"blocks": []}), encoding="utf-8")
    (bundle_dir / "tool_allowlist.yaml").write_text(yaml.dump({"tools": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="system_prompt"):
        load(bundle_dir)


# ---------------------------------------------------------------------------
# 5. load() rejects missing component file on disk
# ---------------------------------------------------------------------------

def test_load_rejects_missing_component_file(tmp_path):
    """components.system_prompt pointing to a non-existent file raises ValueError with path."""
    from agents_core.bundle import load

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    manifest = {
        "schema_version": 1,
        "agent_id": "test",
        "operator_class": "qwen",
        "task": "test",
        "components": {
            "system_prompt": "missing.j2",  # does not exist
            "context_injection": "context_injection.yaml",
            "tool_allowlist": "tool_allowlist.yaml",
        },
    }
    (bundle_dir / "manifest.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (bundle_dir / "context_injection.yaml").write_text(yaml.dump({"blocks": []}), encoding="utf-8")
    (bundle_dir / "tool_allowlist.yaml").write_text(yaml.dump({"tools": []}), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load(bundle_dir)

    # Error should mention the resolved path
    assert "missing.j2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. load() preserves forward-compat fields
# ---------------------------------------------------------------------------

def test_load_preserves_forward_compat_fields(tmp_path):
    """Extra top-level keys (pass, shippable, provenance) survive in Bundle.manifest."""
    from agents_core.bundle import load

    bundle_dir = _make_bundle(tmp_path, {
        "schema_version": 1,
        "agent_id": "test",
        "operator_class": "qwen",
        "task": "test",
        "pass": 3,
        "shippable": True,
        "provenance": {"authored_by": "qwen"},
    })

    bundle = load(bundle_dir)

    assert bundle.manifest.get("pass") == 3
    assert bundle.manifest.get("shippable") is True
    assert bundle.manifest.get("provenance") == {"authored_by": "qwen"}


# ---------------------------------------------------------------------------
# 7. Context resolution: mem.search
# ---------------------------------------------------------------------------

def _make_mem_entry(key: str, content: str) -> dict:
    return {
        "key": key,
        "content": content,
        "tags": "",
        "source": "test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def test_context_mem_search_two_records(tmp_path):
    """fake mem backend returning two records produces a two-line block."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.conventions }}",
        context_injection={
            "blocks": [{
                "id": "conventions",
                "source": "mem.search",
                "params": {"key_prefix": "review/convention/"},
                "max_chars": 2000,
                "truncation": "tail",
                "on_empty": "placeholder:NONE",
            }]
        },
    )

    fake_mem = MagicMock()
    fake_mem.list_by_prefix.return_value = [
        _make_mem_entry("review/convention/a", "Convention A"),
        _make_mem_entry("review/convention/b", "Convention B"),
    ]
    fake_obs = MagicMock()
    fake_obs.search.return_value = []

    bundle = load(bundle_dir)
    _, blocks = bundle_mod._resolve_context_blocks(
        bundle, mem_backend=fake_mem, obs_backend=fake_obs
    )

    assert blocks["conventions"] == "review/convention/a: Convention A\nreview/convention/b: Convention B"
    fake_mem.list_by_prefix.assert_called_once_with(prefix="review/convention/", limit=50)


def test_context_mem_search_empty_placeholder(tmp_path):
    """Empty mem result with on_empty: placeholder:NONE produces 'NONE'."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.conventions }}",
        context_injection={
            "blocks": [{
                "id": "conventions",
                "source": "mem.search",
                "params": {"key_prefix": "review/convention/"},
                "max_chars": 2000,
                "truncation": "tail",
                "on_empty": "placeholder:NONE",
            }]
        },
    )

    fake_mem = MagicMock()
    fake_mem.list_by_prefix.return_value = []
    fake_obs = MagicMock()

    bundle = load(bundle_dir)
    _, blocks = bundle_mod._resolve_context_blocks(
        bundle, mem_backend=fake_mem, obs_backend=fake_obs
    )

    assert blocks["conventions"] == "NONE"


# ---------------------------------------------------------------------------
# 8. Context resolution: agent_observations
# ---------------------------------------------------------------------------

def test_context_agent_observations_two_entries(tmp_path):
    """fake observations backend returning two entries produces a two-line block."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.obs }}",
        context_injection={
            "blocks": [{
                "id": "obs",
                "source": "agent_observations",
                "params": {"agent_id": "test-agent", "tags": ["test"], "limit": 10},
                "max_chars": 1500,
                "truncation": "tail",
                "on_empty": "omit",
            }]
        },
    )

    fake_obs = MagicMock()
    fake_obs.search.return_value = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "observation_type": "decision", "content": "Did X"},
        {"timestamp": "2026-01-01T01:00:00+00:00", "observation_type": "lesson", "content": "Learned Y"},
    ]
    fake_mem = MagicMock()

    bundle = load(bundle_dir)
    _, blocks = bundle_mod._resolve_context_blocks(
        bundle, mem_backend=fake_mem, obs_backend=fake_obs
    )

    lines = blocks["obs"].split("\n")
    assert len(lines) == 2
    assert "Did X" in lines[0]
    assert "Learned Y" in lines[1]

    fake_obs.search.assert_called_once_with(agent_id="test-agent", tags_all=["test"], limit=10)


def test_context_agent_observations_empty_omit(tmp_path):
    """Empty observations result with on_empty: omit produces empty string."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.obs }}",
        context_injection={
            "blocks": [{
                "id": "obs",
                "source": "agent_observations",
                "params": {"agent_id": "test-agent"},
                "max_chars": 1500,
                "truncation": "tail",
                "on_empty": "omit",
            }]
        },
    )

    fake_obs = MagicMock()
    fake_obs.search.return_value = []
    fake_mem = MagicMock()

    bundle = load(bundle_dir)
    _, blocks = bundle_mod._resolve_context_blocks(
        bundle, mem_backend=fake_mem, obs_backend=fake_obs
    )

    assert blocks["obs"] == ""


# ---------------------------------------------------------------------------
# 9. Context resolution: max_chars truncation (tail)
# ---------------------------------------------------------------------------

def test_context_max_chars_truncation_tail(tmp_path):
    """Block exceeding max_chars: 50 with truncation: tail keeps last 50 chars + marker."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    long_content = "A" * 100  # 100 chars, well above max_chars=50

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.bigblock }}",
        context_injection={
            "blocks": [{
                "id": "bigblock",
                "source": "mem.search",
                "params": {"key_prefix": "trunc/"},
                "max_chars": 50,
                "truncation": "tail",
                "on_empty": "placeholder:empty",
            }]
        },
    )

    fake_mem = MagicMock()
    fake_mem.list_by_prefix.return_value = [
        _make_mem_entry("trunc/key1", long_content)
    ]
    fake_obs = MagicMock()

    bundle = load(bundle_dir)
    _, blocks = bundle_mod._resolve_context_blocks(
        bundle, mem_backend=fake_mem, obs_backend=fake_obs
    )

    text = blocks["bigblock"]
    assert "[truncated to 50 chars]" in text
    # The 50 chars kept should be the tail (last 50 chars of the rendered line)
    rendered_line = f"trunc/key1: {long_content}"
    assert text.startswith(rendered_line[-50:])


# ---------------------------------------------------------------------------
# 10. Context resolution: total_budget_chars enforcement
# ---------------------------------------------------------------------------

def test_context_total_budget_chars(tmp_path, caplog):
    """Three blocks summing to >600 chars with total_budget_chars:600 truncates from last to first."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    # Each block will have ~300 chars, total ~900 > budget 600
    content = "X" * 280

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.a }}{{ context.b }}{{ context.c }}",
        context_injection={
            "total_budget_chars": 600,
            "blocks": [
                {
                    "id": "a",
                    "source": "mem.search",
                    "params": {"key_prefix": "budget/a/"},
                    "max_chars": 5000,
                    "truncation": "tail",
                    "on_empty": "placeholder:empty",
                },
                {
                    "id": "b",
                    "source": "mem.search",
                    "params": {"key_prefix": "budget/b/"},
                    "max_chars": 5000,
                    "truncation": "tail",
                    "on_empty": "placeholder:empty",
                },
                {
                    "id": "c",
                    "source": "mem.search",
                    "params": {"key_prefix": "budget/c/"},
                    "max_chars": 5000,
                    "truncation": "tail",
                    "on_empty": "placeholder:empty",
                },
            ],
        },
    )

    fake_mem = MagicMock()
    fake_mem.list_by_prefix.side_effect = lambda prefix, limit: [
        _make_mem_entry(f"{prefix}key", content)
    ]
    fake_obs = MagicMock()

    bundle = load(bundle_dir)
    with caplog.at_level(logging.WARNING, logger="agents_core.bundle"):
        _, blocks = bundle_mod._resolve_context_blocks(
            bundle, mem_backend=fake_mem, obs_backend=fake_obs
        )

    # Total must now be <= 600
    total = sum(len(v) for v in blocks.values())
    assert total <= 600

    # Warning must have been logged
    assert any("budget" in r.message.lower() or "overage" in r.message.lower()
               for r in caplog.records)

    # Block a (first) should be least truncated; block c (last) most truncated
    assert len(blocks["a"]) >= len(blocks["c"])


# ---------------------------------------------------------------------------
# 11. Context resolution: unknown source raises ValueError
# ---------------------------------------------------------------------------

def test_context_unknown_source_raises(tmp_path):
    """source: redis.get raises ValueError naming the unsupported source."""
    from agents_core.bundle import load
    from agents_core import bundle as bundle_mod

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.foo }}",
        context_injection={
            "blocks": [{
                "id": "foo",
                "source": "redis.get",
                "params": {},
                "on_empty": "omit",
            }]
        },
    )

    bundle = load(bundle_dir)
    fake_obs = MagicMock()
    with pytest.raises(ValueError, match="redis.get"):
        bundle_mod._resolve_context_blocks(
            bundle, mem_backend=None, obs_backend=fake_obs
        )


# ---------------------------------------------------------------------------
# 12. Template rendering: variables resolve correctly
# ---------------------------------------------------------------------------

def test_template_rendering_variables_resolve(tmp_path):
    """{{ task_intent }} and {{ context.foo }} render correctly."""
    from agents_core.bundle import _render_template

    result = _render_template(
        "Intent: {{ task_intent }} | Foo: {{ context.foo }}",
        task_intent="do something",
        retry_state=None,
        context_blocks={"foo": "bar-content"},
        tools_str="",
    )

    assert "do something" in result
    assert "bar-content" in result


# ---------------------------------------------------------------------------
# 13. Template rendering: missing variable raises UndefinedError
# ---------------------------------------------------------------------------

def test_template_rendering_missing_variable_raises():
    """{{ unknown_var }} raises jinja2.UndefinedError mentioning 'unknown_var'."""
    import jinja2
    from agents_core.bundle import _render_template

    with pytest.raises(jinja2.UndefinedError, match="unknown_var"):
        _render_template(
            "{{ unknown_var }}",
            task_intent="x",
            retry_state=None,
            context_blocks={},
            tools_str="",
        )


# ---------------------------------------------------------------------------
# 14. Template rendering: retry_state None evaluates falsy; renders empty
# ---------------------------------------------------------------------------

def test_template_rendering_retry_state_none():
    """retry_state=None: {% if retry_state %} is false; {{ retry_state }} renders as ''."""
    from agents_core.bundle import _render_template

    template_src = (
        "{% if retry_state %}RETRY: {{ retry_state }}{% endif %}"
        "DONE:{{ retry_state }}END"
    )
    result = _render_template(
        template_src,
        task_intent="x",
        retry_state=None,
        context_blocks={},
        tools_str="",
    )

    assert "RETRY" not in result
    assert "DONE:END" in result


# ---------------------------------------------------------------------------
# 15. Tool allowlist rendering
# ---------------------------------------------------------------------------

def test_tool_allowlist_rendering():
    """Three tools render as a three-line markdown bullet list."""
    from agents_core.bundle import _render_tool_allowlist

    tool_allowlist = {
        "tools": [
            {"name": "tool.alpha", "kind": "read", "purpose": "Does alpha"},
            {"name": "tool.beta", "kind": "write", "purpose": "Does beta"},
            {"name": "tool.gamma", "kind": "network", "purpose": "Does gamma"},
        ]
    }

    result = _render_tool_allowlist(tool_allowlist)
    lines = result.split("\n")

    assert len(lines) == 3
    assert lines[0] == "- tool.alpha (read): Does alpha"
    assert lines[1] == "- tool.beta (write): Does beta"
    assert lines[2] == "- tool.gamma (network): Does gamma"


# ---------------------------------------------------------------------------
# 16. invoke() qwen path end-to-end
# ---------------------------------------------------------------------------

def test_invoke_qwen_end_to_end(tmp_path):
    """invoke() with mocked call_operator returns correct InvocationResult; observation written."""
    from agents_core.bundle import invoke, InvocationResult

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test-agent", "operator_class": "qwen", "task": "test"},
        system_prompt="Hello {{ task_intent }}",
    )

    fake_obs = MagicMock()
    fake_obs.record.return_value = Path("/tmp/fake-obs-path.jsonl")

    with patch("agents_core.bundle.call_operator", return_value="operator response") as mock_op:
        result = invoke(
            bundle_dir,
            task_intent="do the thing",
            backends={"observations": fake_obs},
        )

    assert isinstance(result, InvocationResult)
    assert result.response == "operator response"
    assert result.agent_id == "test-agent"
    assert result.operator_class == "qwen"
    assert "do the thing" in result.rendered_prompt
    assert result.observation_path == Path("/tmp/fake-obs-path.jsonl")

    mock_op.assert_called_once()
    call_kwargs = mock_op.call_args
    assert call_kwargs.kwargs.get("operator_class") == "qwen" or call_kwargs.args[0] == "qwen"

    fake_obs.record.assert_called_once()
    record_kwargs = fake_obs.record.call_args.kwargs
    assert record_kwargs["agent_id"] == "test-agent"
    assert record_kwargs["observation_type"] == "decision"


# ---------------------------------------------------------------------------
# 17. invoke() Anthropic path raises NotImplementedError
# ---------------------------------------------------------------------------

def test_invoke_anthropic_raises_not_implemented(tmp_path):
    """bundle with operator_class: sonnet propagates NotImplementedError with gap name."""
    from agents_core.bundle import invoke

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "sonnet", "task": "test"},
    )

    with pytest.raises(NotImplementedError) as exc_info:
        invoke(bundle_dir, task_intent="hello")

    assert "agents-core-claude-queue-sync-surface-v0" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 18. invoke() backend injection — real modules NOT imported
# ---------------------------------------------------------------------------

def test_invoke_backend_injection_no_real_import(tmp_path):
    """Fake mem and observations backends are used; real agents_core.mem/.observations not imported."""
    from agents_core.bundle import invoke

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
        system_prompt="{{ context.block_a }}",
        context_injection={
            "blocks": [{
                "id": "block_a",
                "source": "mem.search",
                "params": {"key_prefix": "test/"},
                "max_chars": 1000,
                "truncation": "tail",
                "on_empty": "placeholder:INJECTED",
            }]
        },
    )

    fake_mem = MagicMock()
    fake_mem.list_by_prefix.return_value = [
        _make_mem_entry("test/key", "injected-content")
    ]

    fake_obs = MagicMock()
    fake_obs.record.return_value = Path("/tmp/test-obs.jsonl")

    # Patch call_operator; also ensure agents_core.mem.MemoryStore is NOT instantiated
    with patch("agents_core.bundle.call_operator", return_value="ok") as mock_op, \
         patch("agents_core.mem.MemoryStore") as mock_store_class:

        result = invoke(
            bundle_dir,
            task_intent="test",
            backends={"mem": fake_mem, "observations": fake_obs},
        )

    # Real MemoryStore must NOT have been instantiated
    mock_store_class.assert_not_called()
    fake_mem.list_by_prefix.assert_called_once()
    assert "injected-content" in result.rendered_prompt


# ---------------------------------------------------------------------------
# 19. invoke() record_observation=False
# ---------------------------------------------------------------------------

def test_invoke_no_record_observation(tmp_path):
    """record_observation=False: no observation written; observation_path is None."""
    from agents_core.bundle import invoke

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
    )

    fake_obs = MagicMock()

    with patch("agents_core.bundle.call_operator", return_value="response"):
        result = invoke(
            bundle_dir,
            task_intent="hello",
            record_observation=False,
            backends={"observations": fake_obs},
        )

    fake_obs.record.assert_not_called()
    assert result.observation_path is None
    assert result.response == "response"


# ---------------------------------------------------------------------------
# 20. invoke() observation write failure — WARN logged, result still returned
# ---------------------------------------------------------------------------

def test_invoke_observation_write_failure_logged(tmp_path, caplog):
    """observation backend's record() raises OSError; invoke returns successfully; WARN logged."""
    from agents_core.bundle import invoke

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
    )

    fake_obs = MagicMock()
    fake_obs.record.side_effect = OSError("disk full")

    with patch("agents_core.bundle.call_operator", return_value="response"), \
         caplog.at_level(logging.WARNING, logger="agents_core.bundle"):
        result = invoke(
            bundle_dir,
            task_intent="hello",
            backends={"observations": fake_obs},
        )

    assert result.response == "response"
    assert result.observation_path is None
    assert any("observation" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# 21. invoke() rejects reserved operator_kwargs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved_key", ["system", "bundle_ids", "operator_class", "prompt", "model"])
def test_invoke_rejects_reserved_operator_kwargs(tmp_path, reserved_key):
    """Passing reserved keys in operator_kwargs raises ValueError naming the offending key."""
    from agents_core.bundle import invoke

    bundle_dir = _make_bundle(
        tmp_path,
        {"schema_version": 1, "agent_id": "test", "operator_class": "qwen", "task": "test"},
    )

    with pytest.raises(ValueError, match=reserved_key):
        invoke(bundle_dir, task_intent="hello", **{reserved_key: "bad-value"})


# ---------------------------------------------------------------------------
# 22. MemoryStore.list_by_prefix
# ---------------------------------------------------------------------------

def test_list_by_prefix_basic_correctness(tmp_path):
    """Insert keys with two prefixes; list_by_prefix returns only the matching ones, key-ordered."""
    from agents_core.mem import MemoryStore

    db_path = tmp_path / "test.db"
    store = MemoryStore(db_path=db_path)

    # Insert convention keys
    store.set("review/convention/alpha", "Convention alpha")
    store.set("review/convention/beta", "Convention beta")
    store.set("review/convention/gamma", "Convention gamma")
    # Insert pattern keys (should not appear)
    store.set("review/pattern/x", "Pattern x")
    store.set("review/pattern/y", "Pattern y")

    results = store.list_by_prefix("review/convention/")

    assert len(results) == 3
    keys = [r["key"] for r in results]
    assert keys == sorted(keys)  # ordered by key ascending
    assert all(k.startswith("review/convention/") for k in keys)


def test_list_by_prefix_empty_prefix_raises(tmp_path):
    """Empty prefix raises ValueError."""
    from agents_core.mem import MemoryStore

    db_path = tmp_path / "test.db"
    store = MemoryStore(db_path=db_path)

    with pytest.raises(ValueError, match="non-empty prefix"):
        store.list_by_prefix("")


def test_list_by_prefix_sql_wildcard_escape(tmp_path):
    """SQL-LIKE wildcards in prefix are escaped so they match literally."""
    from agents_core.mem import MemoryStore

    db_path = tmp_path / "test.db"
    store = MemoryStore(db_path=db_path)

    # Insert a key with literal underscore in prefix
    store.set("review/x_y/foo", "Underscore key")
    # Insert a key that would match if _ were treated as wildcard
    store.set("review/xay/foo", "Wildcard victim")

    results = store.list_by_prefix("review/x_y/")
    keys = [r["key"] for r in results]

    assert "review/x_y/foo" in keys
    assert "review/xay/foo" not in keys
