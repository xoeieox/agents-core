"""Tests for agents_core.librarian.projectors.targets (current-targets-state)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_target(directory: Path, name: str, data: dict) -> Path:
    p = directory / f"{name}.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_registry_exposes_current_targets_state():
    """lookup('current-targets-state') returns a callable."""
    from agents_core.librarian.projectors import lookup
    fn = lookup("current-targets-state")
    assert callable(fn)


def test_registry_unknown_claim_returns_none():
    """lookup for an unregistered claim returns None."""
    from agents_core.librarian.projectors import lookup
    assert lookup("nonexistent") is None
    assert lookup("") is None
    assert lookup("CURRENT-TARGETS-STATE") is None  # case-sensitive


# ---------------------------------------------------------------------------
# Projector filter tests
# ---------------------------------------------------------------------------

def test_projector_filters_pm_bound_only(tmp_path):
    """Only targets with pm_bound: true appear in the answer."""
    _write_target(tmp_path, "bound", {"id": "t-bound", "pm_bound": True, "title": "Bound"})
    _write_target(tmp_path, "unbound", {"id": "t-unbound", "pm_bound": False, "title": "Unbound"})
    _write_target(tmp_path, "no_field", {"id": "t-nofield", "title": "No pm_bound"})

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", return_value=None):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    assert len(result["targets"]) == 1
    assert result["targets"][0]["target_id"] == "t-bound"


# ---------------------------------------------------------------------------
# Mem key join tests
# ---------------------------------------------------------------------------

def test_projector_joins_mem_keys(tmp_path):
    """cursor, dispatched_total, dispatched_pending, outstanding_brief_id populated from mem."""
    _write_target(tmp_path, "t1", {"id": "t1", "pm_bound": True})

    dispatched = [
        {"status": "pending"},
        {"status": "done"},
        {"status": "pending"},
    ]

    def fake_mem_get(key: str):
        if key == "pm/cursor/t1":
            return {"content": "main", "key": key, "tags": "", "source": "x", "created_at": "", "updated_at": ""}
        if key == "pm/dispatched/t1":
            return {"content": json.dumps(dispatched), "key": key, "tags": "", "source": "x", "created_at": "", "updated_at": ""}
        if key == "pm/outstanding-brief/t1":
            return {"content": "brief-abc", "key": key, "tags": "", "source": "x", "created_at": "", "updated_at": ""}
        return None

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", side_effect=fake_mem_get):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    t = result["targets"][0]
    assert t["cursor"] == "main"
    assert t["dispatched_total"] == 3
    assert t["dispatched_pending"] == 2
    assert t["outstanding_brief_id"] == "brief-abc"


# ---------------------------------------------------------------------------
# Frozen field-set test
# ---------------------------------------------------------------------------

def test_projector_frozen_field_set(tmp_path):
    """Per-target dict has exactly the 12 frozen keys — guards against accidental drop."""
    REQUIRED_KEYS = {
        "target_id", "title", "pm_repo", "pm_authority", "paused", "cursor",
        "dispatched_total", "dispatched_pending", "outstanding_brief_id",
        "tags", "urgency", "category",
    }
    _write_target(tmp_path, "t1", {"id": "t1", "pm_bound": True})

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", return_value=None):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    assert len(result["targets"]) == 1
    assert set(result["targets"][0].keys()) == REQUIRED_KEYS


# ---------------------------------------------------------------------------
# Default values for missing optional fields
# ---------------------------------------------------------------------------

def test_projector_missing_optional_fields_default(tmp_path):
    """A YAML with only id + pm_bound:true → safe defaults for all optional fields."""
    _write_target(tmp_path, "minimal", {"id": "min1", "pm_bound": True})

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", return_value=None):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    t = result["targets"][0]
    assert t["target_id"] == "min1"
    assert t["title"] == "min1"         # defaults to id
    assert t["pm_repo"] is None
    assert t["pm_authority"] is None
    assert t["paused"] is False
    assert t["cursor"] is None
    assert t["dispatched_total"] == 0
    assert t["dispatched_pending"] == 0
    assert t["outstanding_brief_id"] is None
    assert t["tags"] == []
    assert t["urgency"] is None
    assert t["category"] is None


# ---------------------------------------------------------------------------
# Mem unavailability is non-fatal
# ---------------------------------------------------------------------------

def test_projector_mem_unavailable_nonfatal(tmp_path):
    """If _mem_get raises, the projector still returns the target with safe defaults."""
    _write_target(tmp_path, "t1", {"id": "t1", "pm_bound": True})

    def raising_mem_get(key: str):
        raise RuntimeError("mem unavailable")

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", side_effect=raising_mem_get):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    assert len(result["targets"]) == 1
    t = result["targets"][0]
    assert t["cursor"] is None
    assert t["dispatched_total"] == 0
    assert t["dispatched_pending"] == 0
    assert t["outstanding_brief_id"] is None


# ---------------------------------------------------------------------------
# generated_at is ISO-8601 UTC
# ---------------------------------------------------------------------------

def test_projector_generated_at_iso8601_utc(tmp_path):
    """generated_at is parseable as ISO-8601 with timezone info."""
    _write_target(tmp_path, "t1", {"id": "t1", "pm_bound": True})

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", return_value=None):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    dt = datetime.fromisoformat(result["generated_at"])
    assert dt.tzinfo is not None  # UTC-aware


# ---------------------------------------------------------------------------
# id falls back to yaml stem when id key absent
# ---------------------------------------------------------------------------

def test_projector_id_fallback_to_stem(tmp_path):
    """When YAML has no 'id' key, target_id defaults to the filename stem."""
    # Write a YAML with no id field (but pm_bound: true)
    p = tmp_path / "my-target-slug.yaml"
    p.write_text(yaml.dump({"pm_bound": True, "title": "No id field"}), encoding="utf-8")

    with patch("agents_core.librarian.projectors.targets.TARGETS_DIR", tmp_path), \
         patch("agents_core.librarian.projectors.targets._mem_get", return_value=None):
        from agents_core.librarian.projectors.targets import current_targets_state
        result = current_targets_state({}, 60, "auto-update")

    assert result["targets"][0]["target_id"] == "my-target-slug"
