"""Tests for v0.next synthesis schema extensions (position cast + aggregator)."""
from __future__ import annotations

import pytest
from datetime import datetime
from pathlib import Path
import yaml


def _make_run(tmp_path, mode="deliberation", turns_cap=8):
    run_id = "2026-05-07-syn-test"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": mode,
        "decision": "Test: should we adopt daily cohesion sweeps?",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "entity-a", "role": "first_voice"},
            {"id": "entity-b", "role": "second_voice"},
        ],
        "selection_reasoning": "test",
        "voicing": "sonnet",
        "turns_cap": turns_cap,
        "turns": [],
    }
    run_yaml = tmp_path / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    return run_id, run_yaml


def _run_stub(tmp_path, monkeypatch, positions="agree,agree", turns_cap=8):
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", positions)
    run_id, run_yaml = _make_run(tmp_path, turns_cap=turns_cap)
    council_cli.run_deliberation(run_id)
    return yaml.safe_load(run_yaml.read_text())


def test_synthesis_carries_positions_after_run_deliberation(tmp_path, monkeypatch):
    result = _run_stub(tmp_path, monkeypatch)
    syn = result["synthesis"]
    for key in ("positions", "stood_aside", "blocks", "invariants_implicated",
                "evidence", "actionable", "output_class"):
        assert key in syn, f"synthesis missing key: {key!r}"


def test_actionable_true_when_converged(tmp_path, monkeypatch):
    result = _run_stub(tmp_path, monkeypatch, positions="agree,agree")
    assert result["synthesis"]["actionable"] is True
    assert result["synthesis"]["output_class"] == "cohesion-finding"


def test_actionable_true_when_converged_with_reservation(tmp_path, monkeypatch):
    result = _run_stub(tmp_path, monkeypatch, positions="agree,stand-aside")
    assert result["synthesis"]["confidence"] == "converged-with-reservation"
    assert result["synthesis"]["actionable"] is True
    assert result["synthesis"]["output_class"] == "cohesion-finding"


def test_actionable_false_when_open_or_laid_down(tmp_path, monkeypatch):
    # block + turns_cap=1 → laid-down
    result = _run_stub(tmp_path, monkeypatch, positions="agree,block", turns_cap=1)
    assert result["synthesis"]["actionable"] is False
    assert result["synthesis"]["output_class"] == "none"


def test_invariants_implicated_extracted_kernel_dot_form():
    from agents_core.council.cli import _extract_invariants_implicated
    result = _extract_invariants_implicated(
        "This relates to kernel.invariant.4 in our system.", []
    )
    assert "4" in result


def test_invariants_implicated_extracted_prose_form():
    from agents_core.council.cli import _extract_invariants_implicated
    result = _extract_invariants_implicated(
        "This violates Invariant 8 as described.", []
    )
    assert "8" in result


def test_invariants_implicated_extracted_paren_form():
    from agents_core.council.cli import _extract_invariants_implicated
    result = _extract_invariants_implicated("See invariant (7) for details.", [])
    assert "7" in result


def test_invariants_implicated_extracted_from_position_basis():
    from agents_core.council.cli import _extract_invariants_implicated
    positions = [
        {"voice": "a", "position": "block", "basis": "kernel.invariant.5", "reason": "bad"}
    ]
    result = _extract_invariants_implicated("No explicit cite here.", positions)
    assert "5" in result


def test_invariants_implicated_empty_when_no_citation():
    from agents_core.council.cli import _extract_invariants_implicated
    result = _extract_invariants_implicated("Nothing cited here at all.", [])
    assert result == []


def test_invariants_implicated_dedup_and_sort():
    from agents_core.council.cli import _extract_invariants_implicated
    text = "Invariant 4 violated. Also kernel.invariant.1 applies. And Invariant 4 again."
    result = _extract_invariants_implicated(text, [])
    assert result == ["1", "4"]


def test_status_from_synthesis_handles_converged_with_reservation():
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({"confidence": "converged-with-reservation"}) == "resolved"


def test_status_from_synthesis_handles_laid_down():
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({"confidence": "laid-down"}) == "laid-down"


def test_status_from_synthesis_legacy_diverged_maps_to_open():
    """backward-compat for pre-v0.next run YAMLs — Invariant 10."""
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({"confidence": "diverged"}) == "open"


def test_status_from_synthesis_partial_maps_to_open():
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({"confidence": "partial"}) == "open"


def test_status_from_synthesis_empty_dict_maps_to_open():
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({}) == "open"


def test_aggregator_overrides_partial_confidence_when_all_agree(tmp_path, monkeypatch):
    """Aggregator overrides parser's partial fallback when positions all agree."""
    result = _run_stub(tmp_path, monkeypatch, positions="agree,agree")
    # Stub synthesis has CONFIDENCE: converged; aggregator confirms converged
    assert result["synthesis"]["confidence"] == "converged"
    assert result["status"] == "resolved"


def test_aggregator_overrides_llm_diverged_when_all_agree(tmp_path, monkeypatch):
    """Invariant 11: aggregator output overrides parsed confidence.
    Even if _parse_synthesis returns 'diverged', positions=agree,agree → converged.
    """
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

    run_id, run_yaml = _make_run(tmp_path)

    # Patch _parse_synthesis to return diverged, simulating "LLM said diverged"
    original_parse = council_cli._parse_synthesis

    def _patched_parse(text):
        result = original_parse(text)
        result["confidence"] = "diverged"
        return result

    monkeypatch.setattr(council_cli, "_parse_synthesis", _patched_parse)

    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())
    # Aggregator override: positions are agree,agree → confidence must be converged
    assert result["synthesis"]["confidence"] == "converged"
    assert result["status"] == "resolved"
