"""Integration tests for v0.next run YAML shape (position cast end-to-end)."""
from __future__ import annotations

from datetime import datetime
import pytest
import yaml


def _make_run_yaml(tmp_path, mode="deliberation", turns_cap=8):
    run_id = "2026-05-07-int-test"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": mode,
        "decision": "Integration test decision for v0.next contract.",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "entity-a", "role": "first_voice"},
            {"id": "entity-b", "role": "second_voice"},
        ],
        "selection_reasoning": "integration test",
        "voicing": "sonnet",
        "turns_cap": turns_cap,
        "turns": [],
    }
    run_yaml = tmp_path / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    return run_id, run_yaml


def test_full_deliberation_run_yaml_shape_all_agree(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

    run_id, run_yaml = _make_run_yaml(tmp_path)
    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())

    syn = result.get("synthesis", {})
    assert result["status"] == "resolved"
    assert syn.get("confidence") == "converged"
    assert syn.get("output_class") == "cohesion-finding"
    assert isinstance(syn.get("positions"), list)
    assert len(syn["positions"]) == 2
    assert all(p["position"] == "agree" for p in syn["positions"])
    assert isinstance(syn.get("stood_aside"), list)
    assert isinstance(syn.get("blocks"), list)
    assert isinstance(syn.get("invariants_implicated"), list)
    assert isinstance(syn.get("actionable"), bool)
    assert syn["actionable"] is True

    # Cache entry should have been written
    cache_dir = tmp_path / "cache"
    assert cache_dir.exists()
    cache_files = [f for f in cache_dir.glob("*.yaml") if not f.name.startswith(".tmp-")]
    assert len(cache_files) == 1


def test_full_deliberation_run_yaml_shape_with_block(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,block")

    run_id, run_yaml = _make_run_yaml(tmp_path, turns_cap=1)
    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())

    assert result["status"] == "laid-down"
    syn = result["synthesis"]
    assert syn["confidence"] == "laid-down"
    assert len(syn["blocks"]) > 0
    assert syn["actionable"] is False
    assert syn["output_class"] == "none"
    # No cache write for laid-down (Invariant 6)
    if cache_dir.exists():
        cache_files = [f for f in cache_dir.glob("*.yaml") if not f.name.startswith(".tmp-")]
        assert len(cache_files) == 0


def test_position_cast_tail_runs_in_stub_branch(tmp_path, monkeypatch):
    """Guard against H2 from v3->v4: stub branch MUST run the position-cast tail."""
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

    run_id, run_yaml = _make_run_yaml(tmp_path)
    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())

    syn = result.get("synthesis", {})
    # All v0.next keys must be present — missing any means tail didn't run in stub
    for key in ("positions", "output_class", "actionable", "stood_aside", "blocks"):
        assert key in syn, f"stub branch did not run position-cast tail: missing {key!r}"


def test_scene_run_yaml_unchanged_from_v0(tmp_path, monkeypatch):
    """Scene mode must produce no synthesis key, no positions key, status=closed."""
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    run_id = "2026-05-07-scene-v0next"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": "scene",
        "decision": "A kitchen at midnight.",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "entity-a", "role": "scene_slot_0"},
            {"id": "entity-b", "role": "scene_slot_1"},
        ],
        "selection_reasoning": "test",
        "voicing": "sonnet",
        "turns_cap": 2,
        "turns": [],
    }
    run_yaml = tmp_path / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))

    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())

    assert result["status"] == "closed"
    assert "synthesis" not in result
    assert "positions" not in result
