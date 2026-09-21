"""Tests for agents_core.observations — all 10 spec test cases."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from agents_core.observations import (
    record, search, root, lineage, cite, compute_obs_id,
    VALID_OBSERVATION_TYPES, VALID_INTERVENTION_SHAPES,
    VALID_SIGNAL_STRENGTHS, DEFAULT_SIGNAL_STRENGTH,
)


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


def test_filter_substring_context(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent", "friction", "unique-ctx-xyz", "no match here", now=_ts())
    res = search(substring="unique-ctx-xyz")
    assert len(res) == 1


def test_filter_limit(corpus):
    res = search(limit=2)
    assert len(res) == 2
    # Should be the two earliest
    assert res[0]["timestamp"] <= res[1]["timestamp"]


def test_limit_equivalence(corpus):
    """Bounded-limit accumulation must equal the unbounded path's output exactly."""
    total = len(search(limit=None))
    assert total > 0
    for k in (0, 1, 2, total // 2, total, total + 3):
        assert search(limit=k) == (search(limit=None) or [])[:k], f"mismatch at k={k}"


def test_limit_tie_break_scan_order(tmp_path, monkeypatch):
    """Two entries sharing an identical timestamp: limit=1 returns the one
    encountered first in scan order (dir/file/line order)."""
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    ts = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    # agent-a sorts before agent-b; within a file, line order is scan order
    record("agent-a", "friction", "ctx", "first-scanned", now=ts)
    record("agent-b", "friction", "ctx", "second-scanned", now=ts)
    res = search(limit=1)
    assert len(res) == 1
    assert res[0]["content"] == "first-scanned"


def test_limit_oldest_selection(tmp_path, monkeypatch):
    """With >limit matching entries scanned out of timestamp order, limit=1
    returns the EARLIEST timestamp (not the first-scanned entry)."""
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    base = datetime(2026, 5, 5, 12, 0, 0, tzinfo=timezone.utc)
    # Scanned in this order, but earliest timestamp is "earliest-ts"
    record("agent-a", "friction", "ctx", "scanned-first", now=base + timedelta(minutes=2))
    record("agent-a", "friction", "ctx", "earliest-ts", now=base)
    record("agent-a", "friction", "ctx", "scanned-last", now=base + timedelta(minutes=5))
    res = search(limit=1)
    assert len(res) == 1
    assert res[0]["content"] == "earliest-ts"
    assert res[0]["timestamp"] == base.isoformat()


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


# ---------------------------------------------------------------------------
# v0.1 — obs_id
# ---------------------------------------------------------------------------

def test_obs_id_stable_across_invocations(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    ts = _ts()
    record("agent-a", "lesson", "ctx", "same content", now=ts)
    record("agent-a", "lesson", "ctx", "same content", now=ts)
    entries = search()
    assert len(entries) == 2
    assert entries[0]["obs_id"] == entries[1]["obs_id"]


def test_obs_id_unique_per_distinct_content(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    ts = _ts()
    record("agent-a", "lesson", "ctx", "content-X", now=ts)
    record("agent-a", "lesson", "ctx", "content-Y", now=ts)
    entries = search()
    assert entries[0]["obs_id"] != entries[1]["obs_id"]


def test_v0_entries_get_obs_id_on_read(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    agent_dir = tmp_path / "v0-agent"
    agent_dir.mkdir()
    v0_entry = {
        "agent_id": "v0-agent",
        "session_id": None,
        "timestamp": "2026-05-05T12:00:00+00:00",
        "observation_type": "lesson",
        "context": "ctx",
        "content": "v0 content",
        "target_id": None,
        "tags": [],
        "intervention_shape": None,
        "extra": None,
    }
    (agent_dir / "2026-05-05.jsonl").write_text(
        json.dumps(v0_entry) + "\n", encoding="utf-8"
    )
    # Disk has no obs_id
    raw = json.loads((agent_dir / "2026-05-05.jsonl").read_text())
    assert "obs_id" not in raw

    entries = search()
    assert len(entries) == 1
    assert "obs_id" in entries[0]
    assert len(entries[0]["obs_id"]) == 16

    # Disk still has no obs_id (not rewritten)
    raw2 = json.loads((agent_dir / "2026-05-05.jsonl").read_text())
    assert "obs_id" not in raw2


# ---------------------------------------------------------------------------
# v0.1 — informed_by / cite
# ---------------------------------------------------------------------------

def test_informed_by_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    path_a = record("agent-a", "lesson", "ctx", "observation A", now=_ts(0))
    entries_a = search(agent_id="agent-a", substring="observation A")
    obs_id_a = entries_a[0]["obs_id"]

    cite([obs_id_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="based on A", now=_ts(1))

    entries_b = search(agent_id="agent-a", observation_type="decision")
    assert len(entries_b) == 1
    assert entries_b[0]["informed_by"] == [obs_id_a]


# ---------------------------------------------------------------------------
# v0.1 — search new filters
# ---------------------------------------------------------------------------

def test_search_min_signal_strength(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent-a", "lesson", "ctx", "high entry", signal_strength="high", now=_ts(0))
    record("agent-a", "lesson", "ctx", "normal entry", signal_strength="normal", now=_ts(1))

    highs = search(min_signal_strength="high")
    assert len(highs) == 1
    assert highs[0]["content"] == "high entry"

    all_entries = search(min_signal_strength="normal")
    assert len(all_entries) == 2

    no_filter = search()
    assert len(no_filter) == 2


def test_search_informed_by(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent-a", "lesson", "ctx", "entry A", now=_ts(0))
    obs_id_a = search(agent_id="agent-a", substring="entry A")[0]["obs_id"]

    cite([obs_id_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="B cites A", now=_ts(1))
    cite([obs_id_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="C cites A", now=_ts(2))
    record("agent-a", "lesson", "ctx", "entry D unrelated", now=_ts(3))

    results = search(informed_by=obs_id_a)
    assert len(results) == 2
    contents = {e["content"] for e in results}
    assert contents == {"B cites A", "C cites A"}


# ---------------------------------------------------------------------------
# v0.1 — lineage
# ---------------------------------------------------------------------------

def test_lineage_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent-a", "lesson", "ctx", "A", now=_ts(0))
    obs_a = search(agent_id="agent-a", substring="A")[0]["obs_id"]

    cite([obs_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="B", now=_ts(1))
    obs_b = search(agent_id="agent-a", substring="B",
                   observation_type="decision")[0]["obs_id"]

    cite([obs_b], agent_id="agent-a", observation_type="decision",
         context="ctx", content="C", now=_ts(2))

    result = lineage(obs_a, direction="forward")
    assert result["root"]["obs_id"] == obs_a
    forward_contents = {e["content"] for e in result["forward"]}
    assert "B" in forward_contents
    assert "C" in forward_contents
    assert result["backward"] == []

    depths = {e["content"]: e["_lineage_depth"] for e in result["forward"]}
    assert depths["B"] == 1
    assert depths["C"] == 2


def test_lineage_backward(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent-a", "lesson", "ctx", "A", now=_ts(0))
    obs_a = search(agent_id="agent-a", observation_type="lesson")[0]["obs_id"]

    cite([obs_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="B", now=_ts(1))

    cite_entries = search(agent_id="agent-a", observation_type="decision")
    obs_b = cite_entries[0]["obs_id"]

    cite([obs_b], agent_id="agent-a", observation_type="decision",
         context="ctx", content="C", now=_ts(2))

    all_decisions = search(agent_id="agent-a", observation_type="decision")
    obs_c = all_decisions[1]["obs_id"]  # sorted by timestamp; C is second

    result = lineage(obs_c, direction="backward")
    assert result["root"]["obs_id"] == obs_c
    backward_contents = {e["content"] for e in result["backward"]}
    assert "B" in backward_contents
    assert "A" in backward_contents
    assert result["forward"] == []

    depths = {e["content"]: e["_lineage_depth"] for e in result["backward"]}
    assert depths["B"] == 1
    assert depths["A"] == 2


def test_lineage_both(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    record("agent-a", "lesson", "ctx", "A", now=_ts(0))
    obs_a = search(agent_id="agent-a", substring="A")[0]["obs_id"]

    cite([obs_a], agent_id="agent-a", observation_type="decision",
         context="ctx", content="B", now=_ts(1))
    obs_b = search(agent_id="agent-a", observation_type="decision",
                   substring="B")[0]["obs_id"]

    cite([obs_b], agent_id="agent-a", observation_type="decision",
         context="ctx", content="C", now=_ts(2))

    result = lineage(obs_b, direction="both")
    forward_contents = {e["content"] for e in result["forward"]}
    backward_contents = {e["content"] for e in result["backward"]}
    assert "C" in forward_contents
    assert "A" in backward_contents


def test_lineage_cycle_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    # Write entries first to get obs_ids, then write cycle entries manually
    agent_dir = tmp_path / "cycle-agent"
    agent_dir.mkdir()

    # Compute obs_ids for the cycle entries
    ts_a = "2026-05-05T12:00:00+00:00"
    ts_b = "2026-05-05T12:00:01+00:00"
    obs_id_a = compute_obs_id("cycle-agent", ts_a, "lesson", "A cycle")
    obs_id_b = compute_obs_id("cycle-agent", ts_b, "lesson", "B cycle")

    entry_a = {
        "obs_id": obs_id_a,
        "agent_id": "cycle-agent",
        "session_id": None,
        "timestamp": ts_a,
        "observation_type": "lesson",
        "context": "ctx",
        "content": "A cycle",
        "target_id": None,
        "tags": [],
        "intervention_shape": None,
        "informed_by": [obs_id_b],  # A cites B
        "signal_strength": "normal",
        "extra": None,
    }
    entry_b = {
        "obs_id": obs_id_b,
        "agent_id": "cycle-agent",
        "session_id": None,
        "timestamp": ts_b,
        "observation_type": "lesson",
        "context": "ctx",
        "content": "B cycle",
        "target_id": None,
        "tags": [],
        "intervention_shape": None,
        "informed_by": [obs_id_a],  # B cites A
        "signal_strength": "normal",
        "extra": None,
    }
    (agent_dir / "2026-05-05.jsonl").write_text(
        json.dumps(entry_a) + "\n" + json.dumps(entry_b) + "\n",
        encoding="utf-8",
    )

    # Must not raise or loop infinitely
    result = lineage(obs_id_a)
    assert result["root"] is not None


def test_lineage_max_depth(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    # Build a 5-deep chain
    prev_id = None
    ids = []
    for i in range(5):
        ib = [prev_id] if prev_id else []
        record("agent-a", "lesson", "ctx", f"depth-{i}",
               informed_by=ib or None, now=_ts(i))
        e = search(agent_id="agent-a", substring=f"depth-{i}")[0]
        prev_id = e["obs_id"]
        ids.append(prev_id)

    root_id = ids[0]
    result = lineage(root_id, direction="forward", max_depth=2)
    # Only entries at depth 1 and 2 should appear
    depths = [e["_lineage_depth"] for e in result["forward"]]
    assert all(d <= 2 for d in depths)
    assert max(depths) <= 2


def test_lineage_missing_obs_id(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    result = lineage("nonexistent0000")
    assert result == {"root": None, "forward": [], "backward": []}


# ---------------------------------------------------------------------------
# v0.1 — signal_strength default on v0 entries
# ---------------------------------------------------------------------------

def test_signal_strength_default_normal_on_v0_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    agent_dir = tmp_path / "v0-signal-agent"
    agent_dir.mkdir()
    v0_entry = {
        "agent_id": "v0-signal-agent",
        "session_id": None,
        "timestamp": "2026-05-05T12:00:00+00:00",
        "observation_type": "lesson",
        "context": "ctx",
        "content": "v0 no signal",
        "target_id": None,
        "tags": [],
        "intervention_shape": None,
        "extra": None,
    }
    (agent_dir / "2026-05-05.jsonl").write_text(
        json.dumps(v0_entry) + "\n", encoding="utf-8"
    )
    entries = search()
    assert len(entries) == 1
    assert entries[0]["signal_strength"] == "normal"
    assert entries[0]["informed_by"] == []


# ---------------------------------------------------------------------------
# v0.1 — CLI parity for new flags
# ---------------------------------------------------------------------------

def test_cli_record_with_informed_by_and_signal_strength(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    env = {"AGENT_OBSERVATIONS_ROOT": str(tmp_path)}

    # First record to get an obs_id
    r1 = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "record",
            "--agent-id", "cli-v01",
            "--type", "lesson",
            "--context", "cli ctx",
            "--content", "first observation",
            "--signal-strength", "high",
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert r1.returncode == 0, r1.stderr
    # Output is path<TAB>obs_id
    parts = r1.stdout.strip().split("\t")
    assert len(parts) == 2
    obs_id_first = parts[1]
    assert len(obs_id_first) == 16

    # Second record citing the first
    r2 = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "record",
            "--agent-id", "cli-v01",
            "--type", "decision",
            "--context", "cli ctx",
            "--content", "second cites first",
            "--informed-by", obs_id_first,
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert r2.returncode == 0, r2.stderr
    parts2 = r2.stdout.strip().split("\t")
    assert len(parts2) == 2

    entries = search(agent_id="cli-v01")
    first = next(e for e in entries if e["observation_type"] == "lesson")
    second = next(e for e in entries if e["observation_type"] == "decision")

    assert first["signal_strength"] == "high"
    assert second["informed_by"] == [obs_id_first]


def test_cli_lineage_subcommand(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    env = {"AGENT_OBSERVATIONS_ROOT": str(tmp_path)}

    # Build A <- B chain via CLI
    r1 = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "record",
            "--agent-id", "lineage-cli",
            "--type", "lesson",
            "--context", "ctx",
            "--content", "root entry",
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert r1.returncode == 0, r1.stderr
    obs_id_root = r1.stdout.strip().split("\t")[1]

    r2 = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "record",
            "--agent-id", "lineage-cli",
            "--type", "decision",
            "--context", "ctx",
            "--content", "child cites root",
            "--informed-by", obs_id_root,
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert r2.returncode == 0, r2.stderr

    # Run lineage subcommand in JSON mode
    r3 = subprocess.run(
        [
            sys.executable, "-m", "agents_core.observations", "lineage",
            obs_id_root,
            "--direction", "forward",
            "--format", "json",
        ],
        capture_output=True, text=True, env={**__import__("os").environ, **env},
    )
    assert r3.returncode == 0, r3.stderr
    data = json.loads(r3.stdout)
    assert data["root"]["obs_id"] == obs_id_root
    assert len(data["forward"]) == 1
    assert data["forward"][0]["content"] == "child cites root"
    assert data["forward"][0]["_lineage_depth"] == 1


# ---------------------------------------------------------------------------
# Bounded-accumulation guarantee (reviewer debt 6157790b94): search() with a
# small limit must NOT materialise the entire result set in memory — it
# streams line-by-line and caps retention at `limit` entries (bisect-insert),
# so memory is O(limit), not O(total matching entries). test_limit_equivalence
# above pins the output behaviour; the test below pins the MEMORY BOUND
# directly with tracemalloc: a bounded scan's peak allocation must stay
# small (O(limit)) while the unbounded scan of the same store holds the
# full corpus. A regression to full-scan-then-slice would pass the
# behaviour tests but fail this one.
# ---------------------------------------------------------------------------

def test_search_limit_does_not_materialise_full_store(tmp_path, monkeypatch):
    """Memory bound: with limit=1 over a store with many matching entries,
    search() never holds more than O(limit) entries at once. Pinned by
    asserting (a) the result is the single earliest entry even though the
    store is scanned in an order where the earliest is NOT first-scanned,
    and (b) the bounded scan's peak traced allocation is a small fraction
    of the unbounded scan's peak over the same store."""
    import tracemalloc

    monkeypatch.setenv("AGENT_OBSERVATIONS_ROOT", str(tmp_path))
    base = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 500 entries, each ~1 KiB of content, out of timestamp order:
    # earliest ts is written last. Unbounded scan holds ~500 KiB.
    filler = "x" * 1000
    for i in range(500):
        record("agent-x", "friction", "ctx", f"entry-{i}-{filler}",
               now=base + timedelta(minutes=i))

    # Warm up (module imports, json codecs, dir listing) outside tracing.
    search(limit=1)

    tracemalloc.start()
    res = search(limit=1)
    _, bounded_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    res_unbounded = search(limit=None)
    _, unbounded_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Oldest is entry-0 (base + 0 min).
    assert len(res) == 1
    assert res[0]["content"].startswith("entry-0-")
    assert len(res_unbounded) == 500

    # The unbounded scan must actually hold the corpus (sanity: the
    # comparison is meaningful, not both tiny).
    assert unbounded_peak > 100_000

    # The bounded scan must hold far less than the full corpus —
    # O(limit) retention, not O(total matching entries).
    assert bounded_peak < unbounded_peak // 4, (
        f"bounded scan peak {bounded_peak} B is not much smaller than "
        f"unbounded peak {unbounded_peak} B — search(limit=1) appears to "
        "materialise the full result set"
    )
