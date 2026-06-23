"""Tests for agents_core.council CLI — cmd_submit paths, queue vs no-queue,
write ordering, notify flag, description truncation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_submit_args(**kwargs):
    defaults = {
        "decision": "Should we unify the orchestrators?",
        "mode": "deliberation",
        "n": None,
        "turns": 8,
        "voicing": "sonnet",
        "with_entity": None,
        "narrator": False,
        "narrator_voice": None,
        "no_queue": False,
        "notify": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# cmd_submit: run YAML is written before queue/fork
# ---------------------------------------------------------------------------

def test_cmd_submit_queue_path_writes_run_yaml(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["ada-lovelace-canonical", "benjamin-franklin-canonical"], "reasoning": "tension"}

    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append((task, task_id))
            return task_id or "fake-id"

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    # Patch ClaudeQueue inside cmd_submit's local import
    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    args = _make_submit_args()
    rc = council_cli.cmd_submit(args)

    assert rc == 0
    yamls = list(tmp_path.glob("*.yaml"))
    assert len(yamls) == 1
    run_data = yaml.safe_load(yamls[0].read_text())
    assert run_data["status"] == "deliberating"
    assert run_data["mode"] == "deliberation"
    assert len(run_data["selected_entities"]) == 2


def test_cmd_submit_no_queue_path_forks_not_queues(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["ada-lovelace-canonical", "benjamin-franklin-canonical"], "reasoning": "tension"}
    forked = []

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)
    monkeypatch.setattr(council_cli, "_fork_runtime", lambda run_id, log_file: forked.append(run_id) or object())
    monkeypatch.setattr(council_cli, "_watch_startup", lambda proc, run_id, log_file, **kw: None)

    args = _make_submit_args(no_queue=True)
    rc = council_cli.cmd_submit(args)

    assert rc == 0
    assert len(forked) == 1
    yamls = list(tmp_path.glob("*.yaml"))
    assert len(yamls) == 1


def test_cmd_submit_run_yaml_written_before_queue_submit(tmp_path, monkeypatch):
    """The run YAML must exist on disk before queue.submit() is called."""
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    write_order = []

    original_save_run = council_cli.save_run
    def _tracking_save(run):
        write_order.append(("save_run", run["run_id"]))
        original_save_run(run)
    monkeypatch.setattr(council_cli, "save_run", _tracking_save)

    class _FakeQueue:
        def submit(self, task, task_id=None):
            write_order.append(("queue_submit", task_id))
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args())

    # save_run must appear before queue_submit in write_order
    save_idx = next(i for i, e in enumerate(write_order) if e[0] == "save_run")
    queue_idx = next(i for i, e in enumerate(write_order) if e[0] == "queue_submit")
    assert save_idx < queue_idx, f"save_run must precede queue_submit; got order {write_order}"


def test_cmd_submit_distinct_run_ids(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    submitted_ids = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_ids.append(task_id)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args())
    council_cli.cmd_submit(_make_submit_args())

    assert len(submitted_ids) == 2
    assert submitted_ids[0] != submitted_ids[1], "Each submit must produce a unique run_id"


def test_cmd_submit_notify_flag_set_in_task(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append(task)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    # With notify=True
    council_cli.cmd_submit(_make_submit_args(notify=True))
    assert submitted_tasks[-1]["notify"] is True

    # With notify=False (default)
    council_cli.cmd_submit(_make_submit_args(notify=False))
    assert submitted_tasks[-1]["notify"] is False


def test_cmd_submit_description_truncated_at_80(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    long_decision = "x" * 120
    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append(task)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args(decision=long_decision))
    desc = submitted_tasks[-1]["description"]
    assert len(desc) <= 81, f"description too long: {len(desc)}"
    assert desc.endswith("\u2026"), "truncated description should end with ellipsis"


def test_cmd_submit_scene_mode_sets_longer_timeout(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append(task)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args(mode="scene", n=2))
    assert submitted_tasks[-1]["timeout_seconds"] == 2400

    submitted_tasks.clear()
    council_cli.cmd_submit(_make_submit_args(mode="deliberation"))
    assert submitted_tasks[-1]["timeout_seconds"] == 1200


def test_cmd_submit_queue_task_type_is_council_run(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["a", "b"], "reasoning": "x"}
    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append(task)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args())
    assert submitted_tasks[-1]["task_type"] == "council.run"
    assert submitted_tasks[-1]["payload"]["mode"] == "deliberation"


def test_build_parser_has_no_queue_and_notify_flags():
    from agents_core.council.cli import build_parser
    p = build_parser()
    args = p.parse_args(["submit", "some decision", "--no-queue", "--notify"])
    assert args.no_queue is True
    assert args.notify is True


def test_build_parser_defaults_no_queue_false_notify_false():
    from agents_core.council.cli import build_parser
    p = build_parser()
    args = p.parse_args(["submit", "some decision"])
    assert args.no_queue is False
    assert args.notify is False


def test_cmd_list_empty(tmp_path, monkeypatch, capsys):
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path / "nonexistent")
    rc = council_cli.cmd_list(argparse.Namespace())
    assert rc == 0
    out = capsys.readouterr().out
    assert "no runs yet" in out


def test_cmd_show_missing(tmp_path, monkeypatch, capsys):
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    rc = council_cli.cmd_show(argparse.Namespace(run_id="2026-05-06-000000-zzzzzz"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "no such run" in out


def test_cmd_show_prints_yaml(tmp_path, monkeypatch, capsys):
    from agents_core.council import cli as council_cli
    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    run_id = "2026-05-06-000000-aabbcc"
    run_data = {"run_id": run_id, "status": "resolved", "decision": "test", "turns": []}
    (tmp_path / f"{run_id}.yaml").write_text(yaml.safe_dump(run_data))
    rc = council_cli.cmd_show(argparse.Namespace(run_id=run_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved" in out


# ---------------------------------------------------------------------------
# Three-seat deliberation tests
# ---------------------------------------------------------------------------


def test_cmd_submit_deliberation_default_n_is_3(tmp_path, monkeypatch):
    """When mode=deliberation and args.n is None, default n should be 3."""
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {
        "selected": ["ada-lovelace-canonical", "benjamin-franklin-canonical", "alan-turing-canonical"],
        "reasoning": "tension + far-reach",
    }
    submitted_tasks = []

    class _FakeQueue:
        def submit(self, task, task_id=None):
            submitted_tasks.append(task)
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)
    monkeypatch.setattr(council_cli, "recent_pair_entities", lambda: [])

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    args = _make_submit_args(n=None)  # No explicit n
    rc = council_cli.cmd_submit(args)

    assert rc == 0
    yamls = list(tmp_path.glob("*.yaml"))
    assert len(yamls) == 1
    run_data = yaml.safe_load(yamls[0].read_text())
    assert len(run_data["selected_entities"]) == 3
    roles = [e["role"] for e in run_data["selected_entities"]]
    assert roles == ["first_voice", "second_voice", "third_voice"]


def test_validate_mode_n_allows_n3_deliberation():
    from agents_core.council.cli import _validate_mode_n

    # Should not raise
    _validate_mode_n("deliberation", 3)


def test_validate_mode_n_rejects_n4_deliberation():
    from agents_core.council.cli import _validate_mode_n

    with pytest.raises(ValueError, match="deliberation mode requires n in"):
        _validate_mode_n("deliberation", 4)


def test_role_assignments_three_seat():
    from agents_core.council.cli import _role_assignments

    selected = ["alice", "bob", "charlie"]
    result = _role_assignments(selected, "deliberation")
    assert len(result) == 3
    assert result[0] == {"id": "alice", "role": "first_voice"}
    assert result[1] == {"id": "bob", "role": "second_voice"}
    assert result[2] == {"id": "charlie", "role": "third_voice"}


def test_recent_pair_entities_excludes_third_voice(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)

    # Write two deliberation runs with 3 entities each
    run1 = {
        "run_id": "2026-05-01-000000-aaaaaa",
        "mode": "deliberation",
        "selected_entities": [
            {"id": "alice", "role": "first_voice"},
            {"id": "bob", "role": "second_voice"},
            {"id": "charlie", "role": "third_voice"},
        ],
    }
    run2 = {
        "run_id": "2026-05-02-000000-bbbbbb",
        "mode": "deliberation",
        "selected_entities": [
            {"id": "diana", "role": "first_voice"},
            {"id": "eve", "role": "second_voice"},
            {"id": "frank", "role": "third_voice"},
        ],
    }

    (tmp_path / "2026-05-01-000000-aaaaaa.yaml").write_text(yaml.safe_dump(run1))
    (tmp_path / "2026-05-02-000000-bbbbbb.yaml").write_text(yaml.safe_dump(run2))

    result = council_cli.recent_pair_entities()

    # Should return [diana, eve, alice, bob] (newest first, excluding third_voice roles)
    assert len(result) == 4
    assert "charlie" not in result
    assert "frank" not in result
    assert result[0] in ("diana", "eve")  # Most recent
    assert result[1] in ("diana", "eve")


def test_recent_pair_entities_empty_dir(tmp_path, monkeypatch):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path / "nonexistent")
    result = council_cli.recent_pair_entities()
    assert result == []


def test_cmd_submit_passes_recent_pair_ids_to_select_entities(tmp_path, monkeypatch, capsys):
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {"selected": ["x", "y", "z"], "reasoning": "..."}
    captured_kwargs = {}

    def _mock_select(**kwargs):
        captured_kwargs.update(kwargs)
        return mock_selection

    class _FakeQueue:
        def submit(self, task, task_id=None):
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", _mock_select)
    monkeypatch.setattr(council_cli, "recent_pair_entities", lambda: ["ada-lovelace-canonical"])

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args())

    # Verify recent_pair_ids was passed to select_entities
    assert "recent_pair_ids" in captured_kwargs
    assert captured_kwargs["recent_pair_ids"] == ["ada-lovelace-canonical"]

    # Verify the recency penalty message was printed
    out = capsys.readouterr().out
    assert "recency penalty" in out
    assert "ada-lovelace-canonical" in out


# ---------------------------------------------------------------------------
# Provenance and paid_spend tests
# ---------------------------------------------------------------------------


def test_select_entities_stub_path_returns_provenance_fields(tmp_path, monkeypatch):
    """Verify stub mode returns selection_operator and selection_degraded, and does not call call_operator."""
    from agents_core.council import cli as council_cli

    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")

    # Mock call_operator to ensure it's never called in stub mode
    mock_call_operator = MagicMock()
    monkeypatch.setattr(council_cli, "call_operator", mock_call_operator)

    # Build minimal roster
    roster = [
        {"character_id": "a", "character_name": "A", "pool": "test", "cultural_context": ""},
        {"character_id": "b", "character_name": "B", "pool": "test", "cultural_context": ""},
    ]

    result = council_cli.select_entities(
        decision="test",
        roster=roster,
        context={"terms": [], "hits": []},
        n=2,
    )

    assert "selected" in result
    assert "reasoning" in result
    assert result["selection_operator"] == "stub"
    assert result["selection_degraded"] is False
    # Hermeticity check: call_operator must not be called in stub mode
    mock_call_operator.assert_not_called()


def test_select_entities_real_path_with_mocked_call_operator(tmp_path, monkeypatch):
    """Verify real-path select_entities with mocked call_operator derives selection_operator and selection_degraded."""
    from agents_core.council import cli as council_cli

    # Ensure COUNCIL_ENGINE_STUB is NOT set (real path)
    monkeypatch.delenv("COUNCIL_ENGINE_STUB", raising=False)

    # Build minimal roster with resolvable character_ids
    roster = [
        {"character_id": "a", "character_name": "A", "pool": "test", "cultural_context": ""},
        {"character_id": "b", "character_name": "B", "pool": "test", "cultural_context": ""},
    ]

    # Mock call_operator to return valid JSON selection
    mock_response = '{"selected": ["a", "b"], "reasoning": "tension"}'

    def mock_call_operator(operator, prompt, **kwargs):
        # Populate _provenance_out if provided
        prov_out = kwargs.get("_provenance_out")
        if prov_out is not None:
            prov_out.append(("success", operator))
        return mock_response

    monkeypatch.setattr(council_cli, "call_operator", mock_call_operator)

    # Mock find_card_path to make all ids resolvable
    monkeypatch.setattr(council_cli, "find_card_path", lambda cid: Path(f"/fake/{cid}.yaml"))

    result = council_cli.select_entities(
        decision="test",
        roster=roster,
        context={"terms": [], "hits": []},
        n=2,
    )

    # Verify provenance fields are set correctly
    assert result["selection_operator"] == "gravitywell"
    assert result["selection_degraded"] is False
    assert result["selected"] == ["a", "b"]
    assert "reasoning" in result


def test_cmd_submit_run_dict_includes_selection_provenance(tmp_path, monkeypatch):
    """Verify run dict includes selection_voicing and selection_degraded."""
    from agents_core.council import cli as council_cli

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(council_cli, "LOG_DIR", tmp_path / "logs")

    mock_selection = {
        "selected": ["a", "b"],
        "reasoning": "test",
        "selection_operator": "gravitywell",
        "selection_degraded": False,
    }

    class _FakeQueue:
        def submit(self, task, task_id=None):
            return task_id

    monkeypatch.setattr(council_cli, "gather_mem_context", lambda d: {"terms": [], "hits": []})
    monkeypatch.setattr(council_cli, "build_roster", lambda: [])
    monkeypatch.setattr(council_cli, "select_entities", lambda **kw: mock_selection)

    import agents_core.claude_queue as cq_mod
    monkeypatch.setattr(cq_mod, "ClaudeQueue", lambda: _FakeQueue())

    council_cli.cmd_submit(_make_submit_args())

    yamls = list(tmp_path.glob("*.yaml"))
    assert len(yamls) == 1
    run_data = yaml.safe_load(yamls[0].read_text())
    assert run_data["selection_voicing"] == "gravitywell"
    assert run_data["selection_degraded"] is False


def test_stub_deliberation_includes_paid_spend(tmp_path, monkeypatch):
    """Verify stub-mode run sets paid_spend correctly."""
    from agents_core.council import cli as council_cli
    from agents_core.council import cache

    monkeypatch.setattr(council_cli, "COUNCIL_DIR", tmp_path)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("COUNCIL_ENGINE_STUB", "1")
    monkeypatch.setenv("COUNCIL_STUB_POSITIONS", "agree,agree")

    run_id = "2026-06-17-test-paid-spend"
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
        "selection_reasoning": "stub test",
        "selection_voicing": "stub",
        "selection_degraded": False,
        "voicing": "gravitywell",
        "turns_cap": 8,
        "turns": [],
    }
    run_yaml = tmp_path / f"{run_id}.yaml"
    run_yaml.write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True))

    council_cli.run_deliberation(run_id)
    result = yaml.safe_load(run_yaml.read_text())

    assert "paid_spend" in result
    # With selection_degraded=False and voicing="gravitywell", paid_spend should be False
    assert result["paid_spend"] is False


def test_paid_spend_flag_set_when_selection_degraded(tmp_path, monkeypatch):
    """Verify paid_spend is True when selection_degraded is True."""
    from agents_core.council import cli as council_cli

    run = {
        "selection_degraded": True,
        "selection_voicing": "sonnet",
        "voicing_degraded": False,
        "voicing": "gravitywell",
    }
    assert council_cli._calculate_paid_spend(run) is True


def test_paid_spend_flag_set_when_voicing_degraded(tmp_path, monkeypatch):
    """Verify paid_spend is True when voicing_degraded is True."""
    from agents_core.council import cli as council_cli

    run = {
        "selection_degraded": False,
        "selection_voicing": "gravitywell",
        "voicing_degraded": True,
        "voicing": "sonnet",
    }
    assert council_cli._calculate_paid_spend(run) is True


def test_paid_spend_flag_set_when_explicit_paid_voicing(tmp_path, monkeypatch):
    """Verify paid_spend is True when voicing is explicitly paid."""
    from agents_core.council import cli as council_cli

    run = {
        "selection_degraded": False,
        "selection_voicing": "gravitywell",
        "voicing_degraded": False,
        "voicing": "sonnet",
    }
    assert council_cli._calculate_paid_spend(run) is True


def test_paid_spend_flag_false_for_free_voicing(tmp_path, monkeypatch):
    """Verify paid_spend is False for free voicing and selection."""
    from agents_core.council import cli as council_cli

    run = {
        "selection_degraded": False,
        "selection_voicing": "gravitywell",
        "voicing_degraded": False,
        "voicing": "gravitywell",
    }
    assert council_cli._calculate_paid_spend(run) is False
