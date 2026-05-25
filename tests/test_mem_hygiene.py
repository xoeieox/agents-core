"""Tests for agents_core.mem_hygiene — RecencyGuard + HygieneVerdict."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agents_core.mem import MemoryStore
from agents_core.mem_hygiene import HygieneVerdict, LatchState, RecencyGuard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "test.db")


def _insert_artifact(store: MemoryStore, key: str, age_days: float) -> None:
    """Insert an artifact with a specific age by back-dating created_at."""
    store.set(key, f"content for {key}")
    # Back-date by patching the created_at directly in SQLite.
    created_at = (
        datetime.now(timezone.utc) - timedelta(days=age_days)
    ).isoformat()
    store._conn.execute(
        "UPDATE memories SET created_at = ? WHERE key = ?",
        (created_at, key),
    )
    store._conn.commit()


# ---------------------------------------------------------------------------
# Downgrade: deprecate and delete on fresh artifacts
# ---------------------------------------------------------------------------

def test_fresh_deprecate_downgrades_to_keep_both(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh-key", age_days=10)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("candidate/fresh-key", "deprecate")

    assert verdict.final_action == "keep_both"
    assert verdict.fired is False
    assert verdict.staying_reason == "recency_guard"
    assert verdict.proposed_action == "deprecate"
    assert verdict.latch_state == LatchState.PROBE


def test_fresh_delete_downgrades_to_keep_both(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh-key", age_days=10)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("candidate/fresh-key", "delete")

    assert verdict.final_action == "keep_both"
    assert verdict.fired is False
    assert verdict.staying_reason == "recency_guard"


# ---------------------------------------------------------------------------
# Pass-through: old artifacts
# ---------------------------------------------------------------------------

def test_old_artifact_deprecate_fires(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "decision/old-key", age_days=120)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("decision/old-key", "deprecate")

    assert verdict.final_action == "deprecate"
    assert verdict.fired is True
    assert verdict.staying_reason is None


# ---------------------------------------------------------------------------
# Boundary: exactly 30 days uses strict < (passes through)
# ---------------------------------------------------------------------------

def test_boundary_exactly_30_days_fires(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "decision/boundary-key", age_days=30)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("decision/boundary-key", "deprecate")

    # age >= threshold → fires through (strict <)
    assert verdict.fired is True
    assert verdict.final_action == "deprecate"


# ---------------------------------------------------------------------------
# merge and keep_both always fire regardless of age
# ---------------------------------------------------------------------------

def test_fresh_merge_always_fires(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh-merge", age_days=5)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("candidate/fresh-merge", "merge")

    assert verdict.fired is True
    assert verdict.final_action == "merge"


def test_fresh_keep_both_always_fires(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh-keep", age_days=5)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("candidate/fresh-keep", "keep_both")

    assert verdict.fired is True
    assert verdict.final_action == "keep_both"


# ---------------------------------------------------------------------------
# Artifact not found: fires through with artifact_age_days=None
# ---------------------------------------------------------------------------

def test_artifact_not_found_fires_through(tmp_path):
    store = _store(tmp_path)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("nonexistent/key", "deprecate")

    assert verdict.artifact_age_days is None
    assert verdict.fired is True
    assert verdict.final_action == "deprecate"


# ---------------------------------------------------------------------------
# Latch states
# ---------------------------------------------------------------------------

def test_latch_probe_absent_key(tmp_path):
    """No latch key → PROBE, guard downgrades normally."""
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh", age_days=5)
    guard = RecencyGuard(store)

    assert guard.latch_state == LatchState.PROBE
    verdict = guard.evaluate("candidate/fresh", "deprecate")
    assert verdict.fired is False
    assert verdict.latch_state == LatchState.PROBE


def test_latch_absorbed_stops_downgrade(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh", age_days=5)
    store.set("decision/decrystallization-full-ratified", "absorbed")
    guard = RecencyGuard(store)

    assert guard.latch_state == LatchState.ABSORBED
    verdict = guard.evaluate("candidate/fresh", "deprecate")
    assert verdict.fired is True
    assert verdict.latch_state == LatchState.ABSORBED


def test_latch_replaced_stops_downgrade(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh", age_days=5)
    store.set("decision/decrystallization-full-ratified", "replaced")
    guard = RecencyGuard(store)

    assert guard.latch_state == LatchState.REPLACED
    verdict = guard.evaluate("candidate/fresh", "deprecate")
    assert verdict.fired is True
    assert verdict.latch_state == LatchState.REPLACED


def test_latch_forgotten_unrecognized_content(tmp_path):
    """Key exists but content is unrecognized → FORGOTTEN, guard stops downgrading."""
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh", age_days=5)
    store.set("decision/decrystallization-full-ratified", "some-unrecognized-value")
    guard = RecencyGuard(store)

    assert guard.latch_state == LatchState.FORGOTTEN
    verdict = guard.evaluate("candidate/fresh", "deprecate")
    assert verdict.fired is True
    assert verdict.latch_state == LatchState.FORGOTTEN


# ---------------------------------------------------------------------------
# Configurable threshold
# ---------------------------------------------------------------------------

def test_configurable_threshold_15_days(tmp_path):
    """With threshold=10, artifact at 15 days passes through."""
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/medium", age_days=15)
    guard = RecencyGuard(store, recency_threshold_days=10)

    verdict = guard.evaluate("candidate/medium", "deprecate")

    assert verdict.fired is True
    assert verdict.final_action == "deprecate"


def test_configurable_threshold_keeps_artifact_below(tmp_path):
    """With threshold=60, artifact at 15 days is stayed."""
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/medium", age_days=15)
    guard = RecencyGuard(store, recency_threshold_days=60)

    verdict = guard.evaluate("candidate/medium", "deprecate")

    assert verdict.fired is False
    assert verdict.final_action == "keep_both"


# ---------------------------------------------------------------------------
# JSON-serializable
# ---------------------------------------------------------------------------

def test_verdict_json_serializable(tmp_path):
    store = _store(tmp_path)
    _insert_artifact(store, "candidate/fresh", age_days=5)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("candidate/fresh", "deprecate", sweep_id="sweep-001")

    # LatchState inherits str so asdict produces plain strings
    as_dict = dataclasses.asdict(verdict)
    serialized = json.dumps(as_dict)
    parsed = json.loads(serialized)
    assert parsed["final_action"] == "keep_both"
    assert parsed["latch_state"] == "probe"


# ---------------------------------------------------------------------------
# Latch is per-instance cached
# ---------------------------------------------------------------------------

def test_latch_cached_per_instance(tmp_path):
    """Latch is read once and cached; updating the store does not change result."""
    store = _store(tmp_path)
    guard = RecencyGuard(store)

    # First access — no latch key, so PROBE
    assert guard.latch_state == LatchState.PROBE

    # Now write the latch key — but the cache should still return PROBE
    store.set("decision/decrystallization-full-ratified", "absorbed")
    assert guard.latch_state == LatchState.PROBE  # cached


# ---------------------------------------------------------------------------
# sweep_id propagated through
# ---------------------------------------------------------------------------

def test_sweep_id_propagated(tmp_path):
    store = _store(tmp_path)
    guard = RecencyGuard(store)

    verdict = guard.evaluate("nonexistent/key", "keep_both", sweep_id="s-42")

    assert verdict.sweep_id == "s-42"
