"""Tests for agents_core.council module — pure runtime, no LLM, no subprocess."""
from __future__ import annotations
import inspect
import os
from datetime import datetime, timedelta
from pathlib import Path
import pytest
import yaml


def test_run_deliberation_signature_matches_legacy():
    from agents_core.council.cli import run_deliberation
    sig = inspect.signature(run_deliberation)
    params = list(sig.parameters.keys())
    assert params == ["run_id"], f"Unexpected parameters: {params}"
    ret = sig.return_annotation
    assert ret is None or ret == "None" or ret is inspect.Parameter.empty


def test_run_deliberation_writes_turns_and_synthesis(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    council_dir = tmp_path
    run_id = "2026-05-06-000000-aabbcc"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "Should we unify the orchestrators?",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "ada-lovelace-canonical", "role": "first_voice"},
            {"id": "benjamin-franklin-canonical", "role": "second_voice"},
        ],
        "selection_reasoning": "test",
        "voicing": "sonnet",
        "turns_cap": 8,
        "turns": [],
    }
    run_yaml = council_dir / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", council_dir)
    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())
    assert len(result["turns"]) >= 1
    assert result["status"] in {"resolved", "open", "diverged"}
    assert "synthesis" in result
    assert result["synthesis"]["confidence"] == "converged"


def test_run_deliberation_scene_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    council_dir = tmp_path
    run_id = "2026-05-06-000001-ccddee"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": "scene",
        "decision": "A kitchen at 2am.",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "char-a", "role": "scene_slot_0"},
            {"id": "char-b", "role": "scene_slot_1"},
        ],
        "selection_reasoning": "test",
        "voicing": "sonnet",
        "turns_cap": 6,
        "turns": [],
    }
    run_yaml = council_dir / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", council_dir)
    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())
    assert result["status"] == "closed"
    assert len(result["turns"]) >= 1
    assert "synthesis" not in result


def test_run_deliberation_does_not_call_send_notification(tmp_path, monkeypatch):
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    council_dir = tmp_path
    run_id = "2026-05-06-000002-ffaabb"
    run = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "deliberating",
        "mode": "deliberation",
        "decision": "Test decision",
        "context_gathered": {"terms": [], "hits": []},
        "selected_entities": [
            {"id": "entity-a", "role": "first_voice"},
            {"id": "entity-b", "role": "second_voice"},
        ],
        "selection_reasoning": "",
        "voicing": "sonnet",
        "turns_cap": 4,
        "turns": [],
    }
    run_yaml = council_dir / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))
    from agents_core.council import cli as council_cli
    from agents_core.council import cache
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", council_dir)
    # L2 (v4->v4.1): pin cache dir to tmp_path so position-cast tail doesn't
    # leak fixture files to /srv/lapis/council/cache/cohesion/.
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    call_log: list = []
    import agents_core.notify as notify_mod
    monkeypatch.setattr(notify_mod, "send_notification", lambda *a, **kw: call_log.append((a, kw)))
    council_cli.run_deliberation(run_id)
    assert call_log == [], "send_notification was called from run_deliberation"


def test_parse_synthesis_landing_questions_confidence():
    from agents_core.council.cli import _parse_synthesis
    text = """LANDING: We agreed the costs outweigh the gains in this context.
OPEN QUESTIONS:
- How does this interact with the Kami budget cap?
- What happens at the 06:30 soft cutoff?
CONFIDENCE: converged"""
    out = _parse_synthesis(text)
    assert "costs outweigh" in out["landing"]
    assert out["open_questions"] == [
        "How does this interact with the Kami budget cap?",
        "What happens at the 06:30 soft cutoff?",
    ]
    assert out["confidence"] == "converged"


def test_parse_synthesis_empty_questions():
    from agents_core.council.cli import _parse_synthesis
    text = "LANDING: Clear path forward.\nOPEN QUESTIONS: none\nCONFIDENCE: converged"
    out = _parse_synthesis(text)
    assert out["open_questions"] == []
    assert out["confidence"] == "converged"


def test_parse_synthesis_diverged():
    from agents_core.council.cli import _parse_synthesis
    text = """LANDING: We could not agree on the primary lever.
OPEN QUESTIONS:
- Whether urgency should override reversibility
CONFIDENCE: diverged"""
    out = _parse_synthesis(text)
    assert out["confidence"] == "diverged"
    assert len(out["open_questions"]) == 1


def test_status_mapping():
    from agents_core.council.cli import _status_from_synthesis
    assert _status_from_synthesis({"confidence": "converged"}) == "resolved"
    assert _status_from_synthesis({"confidence": "partial"}) == "open"
    assert _status_from_synthesis({"confidence": "diverged"}) == "open"
    assert _status_from_synthesis({}) == "open"


def test_extract_search_terms_drops_stopwords():
    from agents_core.council.cli import _extract_search_terms
    terms = _extract_search_terms("Should we unify the night and daytime orchestrators?")
    assert "unify" in terms and "night" in terms and "daytime" in terms
    assert "should" not in terms


def test_extract_search_terms_dedupes_preserving_order():
    from agents_core.council.cli import _extract_search_terms
    terms = _extract_search_terms("Kami budget Kami cache Kami")
    assert terms == ["kami", "budget", "cache"]


def test_parse_mem_keys_extracts_hits():
    from agents_core.council.cli import _parse_mem_keys
    raw = """Found 3 result(s) for: kami
  infra/kami-budget [budget,kami,ops]  (2026-04-16)
    Some description...
  architecture/kami-dual-mode [kami,architecture]  (2026-04-10)
    Other description...
"""
    keys = _parse_mem_keys(raw)
    assert keys == ["infra/kami-budget", "architecture/kami-dual-mode"]


def test_extract_json_strips_code_fences():
    from agents_core.council.cli import _extract_json
    text = '```json\n{"selected": ["a", "b"], "reasoning": "x"}\n```'
    assert _extract_json(text) == {"selected": ["a", "b"], "reasoning": "x"}


def test_extract_json_finds_embedded_object():
    from agents_core.council.cli import _extract_json
    text = 'Here\'s:\n{"selected": ["a", "b"], "reasoning": "x"}\nDone.'
    assert _extract_json(text)["selected"] == ["a", "b"]


def test_validate_mode_n_rejects_unknown_mode():
    from agents_core.council.cli import _validate_mode_n
    with pytest.raises(ValueError, match="Unknown mode"):
        _validate_mode_n("interview", 2)


def test_validate_mode_n_deliberation_requires_n2():
    from agents_core.council.cli import _validate_mode_n
    _validate_mode_n("deliberation", 2)
    with pytest.raises(ValueError, match="deliberation mode requires n=2"):
        _validate_mode_n("deliberation", 3)


def test_validate_mode_n_scene_allows_2_or_3():
    from agents_core.council.cli import _validate_mode_n
    _validate_mode_n("scene", 2)
    _validate_mode_n("scene", 3)
    with pytest.raises(ValueError, match="scene mode requires n in"):
        _validate_mode_n("scene", 4)


def test_validate_narrator_requires_scene_mode():
    from agents_core.council.cli import _validate_mode_n
    with pytest.raises(ValueError, match="--narrator requires --mode=scene"):
        _validate_mode_n("deliberation", 2, narrator=True)


def test_validate_narrator_requires_n3():
    from agents_core.council.cli import _validate_mode_n
    with pytest.raises(ValueError, match="--narrator requires n=3"):
        _validate_mode_n("scene", 2, narrator=True)
    _validate_mode_n("scene", 3, narrator=True)


def test_validate_narrator_cannot_be_pinned_with():
    from agents_core.council.cli import _validate_mode_n
    with pytest.raises(ValueError, match="cannot pin the narrator"):
        _validate_mode_n("scene", 3, narrator=True, with_entity="narrator")


def test_role_assignments_deliberation():
    from agents_core.council.cli import _role_assignments
    roles = _role_assignments(["a", "b"], "deliberation")
    assert roles == [{"id": "a", "role": "first_voice"}, {"id": "b", "role": "second_voice"}]


def test_role_assignments_scene_two():
    from agents_core.council.cli import _role_assignments
    roles = _role_assignments(["a", "b"], "scene")
    assert roles == [{"id": "a", "role": "scene_slot_0"}, {"id": "b", "role": "scene_slot_1"}]


def test_role_assignments_scene_three():
    from agents_core.council.cli import _role_assignments
    roles = _role_assignments(["a", "b", "c"], "scene")
    assert [r["role"] for r in roles] == ["scene_slot_0", "scene_slot_1", "scene_slot_2"]


def test_role_assignments_scene_with_narrator_default_voice():
    from agents_core.council.cli import _role_assignments
    roles = _role_assignments(["ada-lovelace-canonical", "benjamin-franklin-canonical"], "scene", narrator=True)
    assert len(roles) == 3
    assert roles[2] == {"id": "narrator", "role": "narrator"}
    assert "voice" not in roles[2]


def test_role_assignments_scene_with_narrator_voice_override():
    from agents_core.council.cli import _role_assignments
    roles = _role_assignments(["a", "b"], "scene", narrator=True, narrator_voice="terse and cinematic")
    assert roles[2] == {"id": "narrator", "role": "narrator", "voice": "terse and cinematic"}


def test_role_assignments_narrator_requires_two_characters():
    from agents_core.council.cli import _role_assignments
    with pytest.raises(ValueError, match="narrator scene requires 2 character ids"):
        _role_assignments(["a"], "scene", narrator=True)


class _StubLLM:
    def __init__(self, response: str = "ok"):
        self.response = response
        self.last_system = None
        self.last_messages = None
    def chat(self, system, messages):
        self.last_system = system
        self.last_messages = messages
        return self.response


def test_narrator_entity_system_prompt_default():
    pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity, DEFAULT_NARRATOR_SYSTEM, NARRATOR_ID
    n = NarratorEntity(llm=_StubLLM())
    assert n.id == NARRATOR_ID
    assert n.system_prompt(ctx=None) == DEFAULT_NARRATOR_SYSTEM


def test_narrator_entity_system_prompt_with_voice():
    pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity, DEFAULT_NARRATOR_SYSTEM
    n = NarratorEntity(llm=_StubLLM(), voice="terse and cinematic")
    prompt = n.system_prompt(ctx=None)
    assert DEFAULT_NARRATOR_SYSTEM in prompt
    assert "Your narrative voice: terse and cinematic" in prompt


def test_narrator_entity_act_passes_through_llm():
    lapis = pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity
    stub = _StubLLM(response="the kitchen light hums")
    n = NarratorEntity(llm=stub, voice="quiet")
    out = n.act("what happens next?", ctx=None)
    assert out == "the kitchen light hums"
    assert stub.last_messages == [lapis.Message(role="user", content="what happens next?")]


def test_narrator_entity_satisfies_entity_protocol():
    lapis = pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity
    assert isinstance(NarratorEntity(llm=_StubLLM()), lapis.Entity)


def test_build_entity_narrator_slot_returns_narrator_entity():
    pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity
    from agents_core.council.cli import _build_entity
    class _FakeCharacterEntity:
        @classmethod
        def load(cls, path, adapter): raise AssertionError("should not call load for narrator")
    e = _build_entity({"id": "narrator", "role": "narrator", "voice": "quiet"}, _StubLLM(), _FakeCharacterEntity, NarratorEntity)
    assert isinstance(e, NarratorEntity) and e.voice == "quiet"


def test_build_entity_unknown_character_raises():
    pytest.importorskip("lapis_engine")
    from agents_core.council.narrator_entity import NarratorEntity
    from agents_core.council.cli import _build_entity
    class _FakeCE:
        @classmethod
        def load(cls, path, adapter): raise AssertionError("should not reach load")
    with pytest.raises(RuntimeError, match="No card for entity id"):
        _build_entity({"id": "obviously-not-a-real-character", "role": "scene_slot_0"}, _StubLLM(), _FakeCE, NarratorEntity)


def test_new_run_id_is_unique_and_formatted():
    import re
    from agents_core.council.cli import new_run_id
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert re.match(r"^\d{4}-\d{2}-\d{2}-\d{6}-[0-9a-f]{6}$", a)


def test_build_roster_finds_historical_and_fiction():
    from agents_core.council.cli import build_roster, CARDS_ROOT
    if not CARDS_ROOT.exists():
        pytest.skip("archetypal-intelligence cards dir not present")
    roster = build_roster(["personal", "historical", "fiction"])
    assert len(roster) > 50


def test_find_card_path_resolves_known_id():
    from agents_core.council.cli import find_card_path, CARDS_ROOT
    if not CARDS_ROOT.exists():
        pytest.skip("archetypal-intelligence cards dir not present")
    path = find_card_path("Erah-canonical")
    assert path is not None and path.exists()


def test_find_card_path_unknown_returns_none():
    from agents_core.council.cli import find_card_path, CARDS_ROOT
    if not CARDS_ROOT.exists():
        pytest.skip("archetypal-intelligence cards dir not present")
    assert find_card_path("obviously-not-a-real-character") is None
