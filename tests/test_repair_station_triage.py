"""Tests for agents_core.repair_station.triage — Leg 3 one-shot dry-run triage.

Covers:
  1. Provenance classification (_is_test_pollution): run-id / commit-sha provenance,
     not keyword matching.
  2. Grouping + disposition precedence (_group_incidents): collapsed-signature
     grouping across per-run-unique fields, and the dismiss/close/keep-open rules.
  3. main(): writes exactly the markdown report + JSON sidecar, and — the DoD 4
     invariant — causes ZERO writes to the incidents DB.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Provenance classification
# ---------------------------------------------------------------------------

def test_is_test_pollution_flags_sentinel_run_id():
    from agents_core.repair_station.triage import _is_test_pollution

    row = {
        "error_signal": {"run_id": "2026-01-01-000000-stale01", "reason": "x"},
        "created_at": "2026-08-17T12:00:00+00:00",
    }
    assert _is_test_pollution(row) is True


def test_is_test_pollution_flags_literal_test_run_id():
    from agents_core.repair_station.triage import _is_test_pollution

    row = {"error_signal": {"run_id": "test-run", "reason": "x"}, "created_at": "2026-08-17T12:00:00+00:00"}
    assert _is_test_pollution(row) is True


def test_is_test_pollution_accepts_well_formed_run_id_near_created_at():
    from agents_core.repair_station.triage import _is_test_pollution

    row = {
        "error_signal": {"run_id": "2026-08-17-115900-6eadc2", "reason": "x"},
        "created_at": "2026-08-17T12:00:05+00:00",
    }
    assert _is_test_pollution(row) is False


def test_is_test_pollution_flags_well_formed_run_id_with_skewed_date():
    """A run_id that matches the production shape but whose embedded timestamp is
    wildly inconsistent with created_at did not come from a real fire."""
    from agents_core.repair_station.triage import _is_test_pollution

    row = {
        "error_signal": {"run_id": "2020-01-01-000000-abcdef", "reason": "x"},
        "created_at": "2026-08-17T12:00:00+00:00",
    }
    assert _is_test_pollution(row) is True


def test_is_test_pollution_flags_short_resolved_sha():
    from agents_core.repair_station.triage import _is_test_pollution

    row = {"error_signal": {"resolved_sha": "abc", "case": "denied"}, "created_at": "2026-08-17T12:00:00+00:00"}
    assert _is_test_pollution(row) is True


def test_is_test_pollution_accepts_real_resolved_sha():
    from agents_core.repair_station.triage import _is_test_pollution

    row = {
        "error_signal": {"resolved_sha": "018a6dcd5ce89d53019b9b4affd9ab66cea91985", "case": "denied"},
        "created_at": "2026-08-17T12:00:00+00:00",
    }
    assert _is_test_pollution(row) is False


def test_is_test_pollution_no_signal_defaults_to_not_pollution():
    """Absence of a positive signal is not evidence — rows with neither run_id nor
    resolved_sha are not asserted as pollution."""
    from agents_core.repair_station.triage import _is_test_pollution

    row = {
        "error_signal": {"case": "absent", "skip_reason": "no_repo_in_context"},
        "created_at": "2026-08-17T12:00:00+00:00",
    }
    assert _is_test_pollution(row) is False


# ---------------------------------------------------------------------------
# 2. Grouping + disposition
# ---------------------------------------------------------------------------

def test_group_incidents_collapses_across_excluded_fields():
    from agents_core.repair_station.triage import _group_incidents

    rows = [
        {
            "incident_id": "inc-1", "station_id": "s", "status": "open",
            "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-01-000000-aaaaaa", "last_heartbeat": None},
        },
        {
            "incident_id": "inc-2", "station_id": "s", "status": "open",
            "created_at": "2026-08-02T00:00:00+00:00",
            "error_signal": {
                "reason": "boom", "run_id": "2026-08-02-000000-bbbbbb",
                "last_heartbeat": "2026-08-02T00:00:01",
            },
        },
    ]
    groups = _group_incidents(rows)

    assert len(groups) == 1
    assert groups[0]["member_count"] == 2


def test_group_incidents_all_pollution_proposes_dismiss():
    from agents_core.repair_station.triage import _group_incidents

    rows = [
        {
            "incident_id": "inc-1", "station_id": "s", "status": "open",
            "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "test-run"},
        },
        {
            "incident_id": "inc-2", "station_id": "s", "status": "open",
            "created_at": "2026-08-02T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-01-01-000000-stale01"},
        },
    ]
    groups = _group_incidents(rows)

    assert len(groups) == 1
    g = groups[0]
    assert g["proposed_disposition"] == "dismiss:test-pollution"
    assert g["provenance_counts"]["test-pollution"] == 2
    assert g["human_verify"] == "required"


def test_group_incidents_duplicates_propose_close_keeping_earliest_genuine():
    from agents_core.repair_station.triage import _group_incidents

    rows = [
        {
            "incident_id": "inc-1", "station_id": "s", "status": "open",
            "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-01-000000-aaaaaa"},
        },
        {
            "incident_id": "inc-2", "station_id": "s", "status": "open",
            "created_at": "2026-08-02T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-02-000000-bbbbbb"},
        },
    ]
    groups = _group_incidents(rows)

    assert len(groups) == 1
    g = groups[0]
    assert g["proposed_disposition"] == "close:duplicate"
    assert g["provenance_counts"]["genuine"] == 1
    assert g["provenance_counts"]["duplicate-of-open-group"] == 1
    assert g["sample_incident_id"] == "inc-1"  # earliest row is the kept "genuine" one


def test_group_incidents_singleton_proposes_keep_open():
    from agents_core.repair_station.triage import _group_incidents

    rows = [
        {
            "incident_id": "inc-1", "station_id": "s", "status": "open",
            "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-01-000000-aaaaaa"},
        },
    ]
    groups = _group_incidents(rows)

    assert len(groups) == 1
    assert groups[0]["proposed_disposition"] == "keep-open"
    assert groups[0]["provenance_counts"]["genuine"] == 1


def test_group_incidents_separates_by_station_id():
    from agents_core.repair_station.triage import _group_incidents

    rows = [
        {
            "incident_id": "inc-1", "station_id": "council/worker-fast-fail", "status": "open",
            "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-01-000000-aaaaaa"},
        },
        {
            "incident_id": "inc-2", "station_id": "shared-deliberation/facets-grounding-denied",
            "status": "open", "created_at": "2026-08-01T00:00:00+00:00",
            "error_signal": {"reason": "boom", "run_id": "2026-08-01-000000-aaaaaa"},
        },
    ]
    groups = _group_incidents(rows)
    assert len(groups) == 2


# ---------------------------------------------------------------------------
# 3. main(): report files + zero DB writes
# ---------------------------------------------------------------------------

def test_main_writes_only_report_files_zero_db_writes(tmp_path):
    from agents_core.repair_station import escalate, Tier, first
    from agents_core.repair_station._db import get_db
    from agents_core.repair_station.triage import main

    db_path = tmp_path / "rs.db"
    # A well-formed, NOT test-pollution run_id: embedded timestamp near "now" (the
    # fixture's created_at), so this row classifies as "genuine" (see
    # _is_test_pollution) rather than "test-pollution".
    now_run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S") + "-aaaaaa"
    escalate(
        station_id="test/triage-station",
        stable_pointer="agents_core/test_module.py",
        error_signal={"reason": "boom", "run_id": now_run_id},
        author_intent="triage fixture",
        escalation_policy=first(),
        tier=Tier.NORMAL,
        db_path=db_path,
    )

    # Checkpoint the WAL so the on-disk main file reflects the fired incident before
    # hashing, and again after main() runs, so the comparison isn't polluted by
    # normal WAL flush timing.
    get_db(db_path).execute("PRAGMA wal_checkpoint(FULL)")
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()

    report_dir = tmp_path / "reports"
    result = main(db_path=db_path, report_dir=report_dir)

    get_db(db_path).execute("PRAGMA wal_checkpoint(FULL)")
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()

    assert before == after, "triage main() must write nothing to the incidents DB"

    written = sorted(p.name for p in report_dir.iterdir())
    expected = sorted([Path(result["md_path"]).name, Path(result["json_path"]).name])
    assert written == expected

    sidecar = json.loads(Path(result["json_path"]).read_text())
    assert sidecar["totals"]["incidents_total"] == 1
    assert sidecar["groups"][0]["human_verify"] == "required"
    assert sidecar["groups"][0]["proposed_disposition"] == "keep-open"

    md_text = Path(result["md_path"]).read_text()
    assert "DRY RUN" in md_text
    assert "human-verify: required" in md_text


def test_main_default_report_dir_is_db_parent(tmp_path):
    from agents_core.repair_station.triage import main

    db_path = tmp_path / "sub" / "rs.db"
    result = main(db_path=db_path)

    assert Path(result["md_path"]).parent == db_path.parent
    assert Path(result["json_path"]).parent == db_path.parent
    assert result["totals"]["incidents_total"] == 0
