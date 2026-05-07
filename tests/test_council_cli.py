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
    monkeypatch.setattr(council_cli, "_fork_runtime", lambda run_id, log_file: forked.append(run_id))

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
