"""Tests for agents_core.council.cache module (v0.next)."""
from __future__ import annotations

import os
from pathlib import Path
import pytest
import yaml


def _make_run(output_class="cohesion-finding", confidence="converged",
              decision="Should we adopt daily cohesion sweeps?",
              run_id="2026-05-07-cache-test-aabbcc",
              invariants=None):
    positions = [
        {"voice": "entity-a", "position": "agree", "reason": "", "basis": None},
        {"voice": "entity-b", "position": "agree", "reason": "", "basis": None},
    ]
    return {
        "run_id": run_id,
        "decision": decision,
        "synthesis": {
            "landing": "We agree on the approach. It is sound and reversible.",
            "confidence": confidence,
            "positions": positions,
            "invariants_implicated": invariants if invariants is not None else ["4"],
            "stood_aside": [],
            "blocks": [],
            "actionable": output_class == "cohesion-finding",
            "output_class": output_class,
        },
    }


def test_write_finding_writes_yaml_at_hash_path(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")
    run = _make_run()
    result = cache.write_finding(run, "0.2.0")
    assert result is not None
    assert result.exists()
    data = yaml.safe_load(result.read_text())
    assert "decision_text_hash" in data
    assert data["run_id"] == run["run_id"]
    assert data["kernel_version"] == "0.2.0"
    assert data["output_class"] == "cohesion-finding"


def test_write_finding_skipped_for_none_output_class(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    cache_dir = tmp_path / "cohesion"
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    run = _make_run(output_class="none")
    result = cache.write_finding(run, "0.2.0")
    assert result is None
    # No yaml files should have been written
    if cache_dir.exists():
        assert len([f for f in cache_dir.glob("*.yaml") if not f.name.startswith(".tmp-")]) == 0


def test_write_finding_uses_same_fs_tmp(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    cache_dir = tmp_path / "cohesion"
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    renamed_from = []
    original_rename = os.rename

    def _capture_rename(src, dst):
        renamed_from.append(str(src))
        return original_rename(src, dst)

    monkeypatch.setattr(os, "rename", _capture_rename)
    run = _make_run()
    cache.write_finding(run, "0.2.0")
    assert len(renamed_from) == 1
    assert ".tmp-" in renamed_from[0]
    assert str(cache_dir) in renamed_from[0]


def test_write_finding_atomic_on_disk_full(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")

    def _always_raise(src, dst):
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "rename", _always_raise)
    run = _make_run()
    # Should not raise; returns None on write failure
    result = cache.write_finding(run, "0.2.0")
    assert result is None


def test_find_related_by_decision_text_hash_exact_match(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")
    run = _make_run()
    cache.write_finding(run, "0.2.0")
    results = cache.find_related(decision_text=run["decision"], current_kernel_version="0.2.0")
    assert len(results) >= 1
    assert results[0]["run_id"] == run["run_id"]


def test_find_related_by_invariant_overlap(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")

    run1 = _make_run(
        decision="Decision about invariant 4 and 8",
        run_id="run-1",
        invariants=["4", "8"],
    )
    cache.write_finding(run1, "0.2.0")

    run2 = _make_run(
        decision="A completely different topic",
        run_id="run-2",
        invariants=["2"],
    )
    cache.write_finding(run2, "0.2.0")

    results = cache.find_related(invariants=["4"], current_kernel_version="0.2.0")
    assert results[0]["run_id"] == "run-1"


def test_find_related_marks_stale_when_kernel_version_differs(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")
    run = _make_run()
    cache.write_finding(run, "0.2.0")
    results = cache.find_related(decision_text=run["decision"], current_kernel_version="0.3.0")
    assert len(results) >= 1
    assert results[0]["stale"] is True


def test_find_related_orders_stale_last(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")

    run_fresh = _make_run(decision="Fresh decision text here", run_id="run-fresh")
    cache.write_finding(run_fresh, "0.3.0")

    run_stale = _make_run(decision="Stale decision text here", run_id="run-stale")
    cache.write_finding(run_stale, "0.2.0")

    results = cache.find_related(current_kernel_version="0.3.0")
    assert results[0]["run_id"] == "run-fresh"


def test_find_related_returns_empty_when_cache_dir_missing(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    missing = tmp_path / "nonexistent" / "cache"
    monkeypatch.setattr(cache, "CACHE_DIR", missing)
    results = cache.find_related()
    assert results == []


def test_read_kernel_version_parses_frontmatter(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    kernel_file = tmp_path / "Constitution-Kernel.md"
    kernel_file.write_text("---\nversion: 0.2.0\ntitle: Lapis Constitution Kernel\n---\n")
    monkeypatch.setattr(cache, "KERNEL_FILE", kernel_file)
    assert cache.read_kernel_version() == "0.2.0"


def test_read_kernel_version_returns_unknown_when_file_missing(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "KERNEL_FILE", tmp_path / "no-such-file.md")
    assert cache.read_kernel_version() == "unknown"


def test_read_kernel_version_returns_unknown_when_no_version_line(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    kernel_file = tmp_path / "Kernel.md"
    kernel_file.write_text("---\ntitle: Lapis\n---\nNo version here at all.")
    monkeypatch.setattr(cache, "KERNEL_FILE", kernel_file)
    assert cache.read_kernel_version() == "unknown"


def test_find_related_treats_unknown_kernel_version_as_stale(tmp_path, monkeypatch):
    import agents_core.council.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cohesion")
    run = _make_run()
    cache.write_finding(run, "unknown")
    results = cache.find_related(decision_text=run["decision"], current_kernel_version="0.2.0")
    assert len(results) >= 1
    assert results[0]["stale"] is True


def test_selector_prompt_includes_cohesion_findings_section(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    captured = {}

    def mock_call(*args, **kwargs):
        captured["prompt"] = kwargs.get("prompt", args[0] if args else "")
        return '{"selected": ["entity-a", "entity-b"], "reasoning": "test"}'

    monkeypatch.setattr(council_cli, "call_claude_cli", mock_call)
    monkeypatch.setattr(council_cli, "find_card_path", lambda x: Path("/fake/path.yaml"))

    context = {
        "terms": ["test"],
        "hits": [],
        "cohesion_findings": [
            {
                "run_id": "run-abc",
                "kernel_version": "0.2.0",
                "stale": False,
                "synthesis": {"landing": "We found an issue with Invariant 4 coupling."},
            }
        ],
    }

    council_cli.select_entities(
        decision="Should we adopt daily sweeps?",
        roster=[
            {"character_id": "entity-a", "character_name": "Entity A",
             "pool": "test", "cultural_context": "Test entity context."},
            {"character_id": "entity-b", "character_name": "Entity B",
             "pool": "test", "cultural_context": "Another test context."},
        ],
        context=context,
        n=2,
    )
    assert "RELATED PRIOR COHESION FINDINGS:" in captured["prompt"]
    assert "--- END PRIOR FINDINGS ---" in captured["prompt"]


def test_selector_prompt_truncates_long_findings_to_300_chars(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    captured = {}

    def mock_call(*args, **kwargs):
        captured["prompt"] = kwargs.get("prompt", args[0] if args else "")
        return '{"selected": ["entity-a", "entity-b"], "reasoning": "test"}'

    monkeypatch.setattr(council_cli, "call_claude_cli", mock_call)
    monkeypatch.setattr(council_cli, "find_card_path", lambda x: Path("/fake/path.yaml"))

    long_landing = "A" * 500
    context = {
        "terms": [],
        "hits": [],
        "cohesion_findings": [
            {
                "run_id": "run-xyz",
                "kernel_version": "0.2.0",
                "stale": False,
                "synthesis": {"landing": long_landing},
            }
        ],
    }

    council_cli.select_entities(
        decision="test decision",
        roster=[
            {"character_id": "entity-a", "character_name": "A",
             "pool": "test", "cultural_context": "x"},
            {"character_id": "entity-b", "character_name": "B",
             "pool": "test", "cultural_context": "y"},
        ],
        context=context,
        n=2,
    )
    # Should have ellipsis suffix and not contain the full 500-char string
    assert "\u2026" in captured["prompt"]
    assert "A" * 301 not in captured["prompt"]


def test_selector_prompt_renders_none_when_no_findings(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    captured = {}

    def mock_call(*args, **kwargs):
        captured["prompt"] = kwargs.get("prompt", args[0] if args else "")
        return '{"selected": ["entity-a", "entity-b"], "reasoning": "test"}'

    monkeypatch.setattr(council_cli, "call_claude_cli", mock_call)
    monkeypatch.setattr(council_cli, "find_card_path", lambda x: Path("/fake/path.yaml"))

    context = {"terms": [], "hits": [], "cohesion_findings": []}

    council_cli.select_entities(
        decision="test decision",
        roster=[
            {"character_id": "entity-a", "character_name": "A",
             "pool": "test", "cultural_context": "x"},
            {"character_id": "entity-b", "character_name": "B",
             "pool": "test", "cultural_context": "y"},
        ],
        context=context,
        n=2,
    )
    assert "RELATED PRIOR COHESION FINDINGS:" in captured["prompt"]
    assert "(none)" in captured["prompt"]


def test_render_transcript_returns_no_turns_recorded_when_empty():
    from agents_core.council.cli import _render_transcript
    assert _render_transcript([]) == "(no turns recorded)"


def test_render_transcript_filters_to_deliberation_turn_only():
    from agents_core.council.cli import _render_transcript
    turns = [
        {"type": "deliberation_turn", "speaker": "a", "content": "hello deliberation"},
        {"type": "synthesis", "speaker": "synthesis", "content": "synthesis text"},
        {"type": "unknown", "speaker": "x", "content": "noise"},
    ]
    result = _render_transcript(turns)
    assert "hello deliberation" in result
    assert "synthesis text" not in result
    assert "noise" not in result
