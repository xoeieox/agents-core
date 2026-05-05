"""Tests for agents_core.observations — all 10 spec test cases."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from agents_core.observations import record, search, root, VALID_OBSERVATION_TYPES, VALID_INTERVENTION_SHAPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_seconds: int = 0) -> datetime:
    """Return a UTC-aware datetime offset by seconds from a fixed base."""
    base = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


# ---------------------------------------------------------------------------
# 1. Round-trip basic
# ---------------------------------------------------------------------------

def test_roundtrip_basic(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    path = record(
        "lapis-pm", "friction", "running tests", "pytest was slow",
        session_id="sess-1",
        target_id="t-42",
        tags=["agents-core", "tests"],
        now=_ts(),
    )
    assert path.exists()
    entries = search()
    assert len(entries) == 1
    e = entries[0]
    assert e["agent_id"] == "lapis-pm"
    assert e["observation_type"] == "friction"
    assert e["context"] == "running tests"
    assert e["content"] == "pytest was slow"
    assert e["session_id"] == "sess-1"
    assert e["target_id"] == "t-42"
    assert e["tags"] == ["agents-core", "tests"]
    assert e["timestamp"] == "2026-05-05T12:00:00+00:00"
    assert e["intervention_shape"] is None
    assert e["extra"] is None


# ---------------------------------------------------------------------------
# 2. Per-agent-per-date file layout
# ---------------------------------------------------------------------------

def test_per_agent_per_date_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    day1 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 5, 2, 10, 0, 0, tzinfo=timezone.utc)

    record("lapis-pm", "decision", "ctx", "content-A", now=day1)
    record("lapis-pm", "lesson", "ctx", "content-B", now=day2)
    record("tech-kami", "anomaly", "ctx", "content-C", now=day1)
    record("tech-kami", "friction", "ctx", "content-D", now=day2)

    assert (tmp_path / "lapis-pm" / "2026-05-01.jsonl").exists()
    assert (tmp_path / "lapis-pm" / "2026-05-02.jsonl").exists()
    assert (tmp_path / "tech-kami" / "2026-05-01.jsonl").exists()
    assert (tmp_path / "tech-kami" / "2026-05-02.jsonl").exists()
    # No cross-agent bleed
    assert not (tmp_path / "lapis-pm" / "2026-05-03.jsonl").exists()


# ---------------------------------------------------------------------------
# 3. Append semantics
# ---------------------------------------------------------------------------

def test_append_semantics(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    ts = datetime(2026, 5, 5, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        record("agent-x", "friction", "ctx", f"entry-{i}", now=ts)

    jsonl_path = tmp_path / "agent-x" / "2026-05-05.jsonl"
    lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 5
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["content"] == f"entry-{i}"


# ---------------------------------------------------------------------------
# 4. Enum validation at write
# ---------------------------------------------------------------------------

def test_enum_validation_observation_type(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="observation_type"):
        record("agent", "bogus", "ctx", "content")


def test_enum_validation_intervention_shape_bogus(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="intervention_shape"):
        record("agent", "intervention", "ctx", "content", intervention_shape="bogus")


def test_enum_validation_intervention_shape_wrong_type(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="intervention_shape"):
        record("agent", "friction", "ctx", "content", intervention_shape="question")


def test_enum_validation_intervention_missing_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="intervention_shape"):
        record("agent", "intervention", "ctx", "content")


def test_intervention_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    path = record(
        "agent", "intervention", "ctx", "content",
        intervention_shape="question", now=_ts(),
    )
    entries = search()
    assert entries[0]["intervention_shape"] == "question"


# ---------------------------------------------------------------------------
# 5. Search filters
# ---------------------------------------------------------------------------

@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """Populate a fixture corpus for filter tests."""
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    base = datetime(2026, 5, 5, 8, 0, 0, tzinfo=timezone.utc)

    record("lapis-pm", "friction", "ctx-A", "content friction", target_id="t-1",
           tags=["repo-A", "slow"], now=base)
    record("lapis-pm", "decision", "ctx-B", "content decision", target_id="t-2",
           tags=["repo-A", "arch"], now=base + timedelta(minutes=1))
    record("tech-kami", "lesson", "ctx-C", "content lesson MATCH",
           tags=["repo-B"], now=base + timedelta(minutes=2))
    record("tech-kami", "anomaly", "ctx-D", "content anomaly", target_id="t-1",
           tags=["repo-B", "arch"], now=base + timedelta(minutes=3))
    record("lapis-pm", "intervention", "ctx-E", "content intervention",
           intervention_shape="pointer", tags=["repo-A"],
           now=base + timedelta(minutes=4))
    return tmp_path


def test_filter_agent_id(corpus):
    res = search(agent_id="tech-kami")
    assert all(e["agent_id"] == "tech-kami" for e in res)
    assert len(res) == 2


def test_filter_observation_type(corpus):
    res = search(observation_type="friction")
    assert all(e["observation_type"] == "friction" for e in res)
    assert len(res) == 1


def test_filter_target_id(corpus):
    res = search(target_id="t-1")
    assert all(e["target_id"] == "t-1" for e in res)
    assert len(res) == 2


def test_filter_tags_any(corpus):
    res = search(tags_any=["slow", "repo-B"])
    # "slow" matches 1 lapis-pm entry; "repo-B" matches 2 tech-kami entries
    assert len(res) == 3


def test_filter_tags_all(corpus):
    res = search(tags_all=["repo-B", "arch"])
    assert len(res) == 1
    assert res[0]["observation_type"] == "anomaly"


def test_filter_since(corpus):
    cutoff = datetime(2026, 5, 5, 8, 2, 0, tzinfo=timezone.utc)
    res = search(since=cutoff)
    assert all(e["timestamp"] >= "2026-05-05T08:02:00" for e in res)
    assert len(res) == 3


def test_filter_until(corpus):
    cutoff = datetime(2026, 5, 5, 8, 2, 0, tzinfo=timezone.utc)
    res = search(until=cutoff)
    assert len(res) == 3


def test_filter_substring(corpus):
    res = search(substring="MATCH")
    assert len(res) == 1
    assert "MATCH" in res[0]["content"]


def test_filter_substring_context(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent", "friction", "unique-ctx-xyz", "no match here", now=_ts())
    res = search(substring="unique-ctx-xyz")
    assert len(res) == 1


def test_filter_limit(corpus):
    res = search(limit=2)
    assert len(res) == 2
    # Should be the two earliest
    assert res[0]["timestamp"] <= res[1]["timestamp"]


def test_filter_combined(corpus):
    res = search(agent_id="lapis-pm", tags_any=["repo-A"], observation_type="decision")
    assert len(res) == 1
    assert res[0]["observation_type"] == "decision"


# ---------------------------------------------------------------------------
# 6. Forward-compat tolerance
# ---------------------------------------------------------------------------

def test_forward_compat_tolerance(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    agent_dir = tmp_path / "future-agent"
    agent_dir.mkdir()
    future_entry = {
        "agent_id": "future-agent",
        "session_id": None,
        "timestamp": "2026-05-05T12:00:00+00:00",
        "observation_type": "lesson",
        "context": "ctx",
        "content": "from the future",
        "target_id": None,
        "tags": [],
        "intervention_shape": None,
        "extra": None,
        "new_field_v2": "some_value",         # unknown future field
        "another_future_field": {"nested": 1},
    }
    (agent_dir / "2026-05-05.jsonl").write_text(
        json.dumps(future_entry) + "\n", encoding="utf-8"
    )
    entries = search()
    assert len(entries) == 1
    assert entries[0]["new_field_v2"] == "some_value"
    assert entries[0]["another_future_field"] == {"nested": 1}


# ---------------------------------------------------------------------------
# 7. Concurrent append safety
# ---------------------------------------------------------------------------

def test_concurrent_append_safety(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    n = 100
    ts = datetime(2026, 5, 5, 6, 0, 0, tzinfo=timezone.utc)
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(n):
                record(
                    "concurrent-agent", "friction",
                    f"thread-{thread_id}", f"entry-{thread_id}-{i}",
                    now=ts,
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Writer threads raised: {errors}"

    jsonl_path = tmp_path / "concurrent-agent" / "2026-05-05.jsonl"
    lines = [l for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2 * n, f"Expected {2 * n} lines, got {len(lines)}"
    for line in lines:
        obj = json.loads(line)  # must not raise
        assert obj["agent_id"] == "concurrent-agent"


# ---------------------------------------------------------------------------
# 8. CLI parity
# ---------------------------------------------------------------------------

def test_cli_record_parity(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))

    env = {"AGENT_OBSERVATIONS_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "record",
            "--agent-id", "cli-agent",
            "--type", "lesson",
            "--context", "cli context",
            "--content", "cli content",
            "--tag", "tag-a",
            "--tag", "tag-b",
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0, result.stderr

    entries = search()
    assert len(entries) == 1
    e = entries[0]
    assert e["agent_id"] == "cli-agent"
    assert e["observation_type"] == "lesson"
    assert e["context"] == "cli context"
    assert e["content"] == "cli content"
    assert "tag-a" in e["tags"]
    assert "tag-b" in e["tags"]


def test_cli_search_json_format(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("search-agent", "anomaly", "ctx", "needle content", now=_ts())

    env = {"AGENT_OBSERVATIONS_ROOT": str(tmp_path)}
    result = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "search",
            "--agent-id", "search-agent",
            "--format", "json",
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0, result.stderr
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["agent_id"] == "search-agent"
    assert obj["content"] == "needle content"


# ---------------------------------------------------------------------------
# 9. Env override
# ---------------------------------------------------------------------------

def test_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-root"
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(override))

    assert root() == override

    path = record("env-agent", "friction", "ctx", "content", now=_ts())
    assert str(override) in str(path)
    assert path.exists()

    entries = search()
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# 10. Empty / missing files
# ---------------------------------------------------------------------------

def test_search_empty_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path / "nonexistent"))
    assert search() == []


def test_search_missing_agent_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    result = search(agent_id="ghost-agent")
    assert result == []


def test_search_malformed_lines_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    agent_dir = tmp_path / "bad-agent"
    agent_dir.mkdir()
    good_entry = {
        "agent_id": "bad-agent", "session_id": None,
        "timestamp": "2026-05-05T10:00:00+00:00",
        "observation_type": "friction", "context": "c", "content": "good",
        "target_id": None, "tags": [], "intervention_shape": None, "extra": None,
    }
    (agent_dir / "2026-05-05.jsonl").write_text(
        '{"bad json": \n'           # malformed line
        + json.dumps(good_entry) + "\n",
        encoding="utf-8",
    )
    entries = search()
    # Malformed line is skipped; good entry is returned
    assert len(entries) == 1
    assert entries[0]["content"] == "good"
