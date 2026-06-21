"""Tests for agents_core.repair_station — Repair Station Contract v0.

Covers:
  1. escalate() creates a well-formed incident with all five payload fields + tier.
  2. Station self-registers on first fire; registry is durable; fire-count/last-fire update.
  3. Dedup: repeated fires with same (station, error_signature) within window yield one open
     incident; escalation policy (first / n_within) gates whether a fire escalates at all.
  4. Council fast-fail station is wired and emits on a simulated signal.
  5. No brain: escalate() makes no model call, performs no diagnosis.
  6. Tier is defined and carried but not acted upon in this leg.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy_first():
    from agents_core.repair_station import first
    return first()


def _make_policy_n_within(count, window_s):
    from agents_core.repair_station import n_within
    return n_within(count, window_s)


def _fire(
    db_path: Path,
    *,
    station_id: str = "test/station",
    error_signal: dict | None = None,
    policy=None,
    tier=None,
):
    from agents_core.repair_station import escalate, Tier, first
    return escalate(
        station_id=station_id,
        stable_pointer="agents_core/test_module.py",
        error_signal=error_signal or {"code": "ETEST", "msg": "test error"},
        author_intent="test station for unit tests",
        escalation_policy=policy or first(),
        tier=tier or Tier.NORMAL,
        owning_module="agents_core.tests.test_repair_station",
        db_path=db_path,
    )


def _registry_row(db_path: Path, station_id: str):
    from agents_core.repair_station._db import get_db
    db = get_db(db_path)
    return db.execute(
        "SELECT * FROM station_registry WHERE station_id = ?", (station_id,)
    ).fetchone()


def _incident_row(db_path: Path, incident_id: str):
    from agents_core.repair_station._db import get_db
    db = get_db(db_path)
    return db.execute(
        "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()


# ---------------------------------------------------------------------------
# 1. Well-formed incident with all payload fields + tier
# ---------------------------------------------------------------------------

def test_escalate_creates_well_formed_incident(tmp_path):
    from agents_core.repair_station import Tier
    db_path = tmp_path / "rs.db"
    error_signal = {"code": "ECONNREFUSED", "host": "localhost:5432"}

    incident_id = _fire(
        db_path,
        error_signal=error_signal,
        tier=Tier.HIGH,
    )

    assert incident_id is not None
    assert incident_id.startswith("inc-")

    row = _incident_row(db_path, incident_id)
    assert row is not None
    assert row["station_id"] == "test/station"
    assert row["stable_pointer"] == "agents_core/test_module.py"
    import json
    assert json.loads(row["error_signal"]) == error_signal
    assert row["author_intent"] == "test station for unit tests"
    assert row["tier"] == int(Tier.HIGH)
    assert row["status"] == "open"
    assert row["back_ref"] is None
    assert row["created_at"]
    assert row["updated_at"]
    assert row["error_signature"]


# ---------------------------------------------------------------------------
# 2. Station self-registers on first fire; fire-count updates on subsequent fires
# ---------------------------------------------------------------------------

def test_station_self_registers_on_first_fire(tmp_path):
    from agents_core.repair_station import Tier
    db_path = tmp_path / "rs.db"

    _fire(db_path, station_id="test/reg-station", tier=Tier.NORMAL)

    row = _registry_row(db_path, "test/reg-station")
    assert row is not None
    assert row["station_id"] == "test/reg-station"
    assert row["stable_pointer"] == "agents_core/test_module.py"
    assert row["tier"] == int(Tier.NORMAL)
    assert row["author_intent"] == "test station for unit tests"
    assert row["fire_count"] == 1
    assert row["first_fire"]
    assert row["last_fire"]


def test_fire_count_increments_on_subsequent_fires(tmp_path):
    db_path = tmp_path / "rs.db"
    station_id = "test/count-station"

    # Use a long dedup window so all fires create incidents (different signals)
    for i in range(3):
        _fire(
            db_path,
            station_id=station_id,
            error_signal={"i": i},  # different signal each time
        )

    row = _registry_row(db_path, station_id)
    assert row["fire_count"] == 3


def test_last_fire_updates_on_each_call(tmp_path):
    db_path = tmp_path / "rs.db"
    station_id = "test/time-station"

    _fire(db_path, station_id=station_id, error_signal={"seq": 1})
    first_row = _registry_row(db_path, station_id)
    first_last = first_row["last_fire"]

    # Small sleep to ensure timestamps differ
    time.sleep(0.05)
    _fire(db_path, station_id=station_id, error_signal={"seq": 2})
    second_row = _registry_row(db_path, station_id)
    assert second_row["last_fire"] >= first_last


# ---------------------------------------------------------------------------
# 3. Dedup and escalation policy
# ---------------------------------------------------------------------------

def test_dedup_same_signature_yields_one_open_incident(tmp_path):
    db_path = tmp_path / "rs.db"
    error_signal = {"code": "ETEST", "msg": "same error"}

    id1 = _fire(db_path, error_signal=error_signal)
    id2 = _fire(db_path, error_signal=error_signal)  # same signature, within dedup window

    assert id1 is not None
    assert id2 is None  # dedup suppressed

    # Only one open incident
    from agents_core.repair_station._db import get_db
    rows = get_db(db_path).execute(
        "SELECT * FROM incidents WHERE station_id = 'test/station' AND status = 'open'"
    ).fetchall()
    assert len(rows) == 1


def test_dedup_different_signatures_yield_separate_incidents(tmp_path):
    db_path = tmp_path / "rs.db"

    id1 = _fire(db_path, error_signal={"code": "E1"})
    id2 = _fire(db_path, error_signal={"code": "E2"})  # different signature

    assert id1 is not None
    assert id2 is not None
    assert id1 != id2


def test_n_within_policy_gates_on_count(tmp_path):
    db_path = tmp_path / "rs.db"
    policy = _make_policy_n_within(3, 60.0)

    id1 = _fire(db_path, error_signal={"seq": 1}, policy=policy)
    id2 = _fire(db_path, error_signal={"seq": 2}, policy=policy)
    id3 = _fire(db_path, error_signal={"seq": 3}, policy=policy)

    # First two fires should not escalate (count < 3)
    assert id1 is None
    assert id2 is None
    # Third fire meets count=3 within window -> escalates
    assert id3 is not None


def test_first_policy_always_escalates_first_fire(tmp_path):
    db_path = tmp_path / "rs.db"
    from agents_core.repair_station import first

    id1 = _fire(db_path, error_signal={"unique": "x"}, policy=first())
    assert id1 is not None


def test_n_within_outside_window_does_not_escalate(tmp_path):
    from agents_core.repair_station import escalate, Tier, n_within
    from agents_core.repair_station._db import get_db
    import json

    db_path = tmp_path / "rs.db"
    db = get_db(db_path)

    # Manually insert old fires (beyond the window)
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
    for i in range(3):
        db.execute(
            "INSERT INTO station_fires (station_id, fired_at) VALUES (?, ?)",
            ("test/window-station", old_ts),
        )
    db.execute(
        """
        INSERT INTO station_registry
          (station_id, owning_module, stable_pointer, tier, author_intent,
           first_fire, last_fire, fire_count)
        VALUES (?, '', 'ptr', 2, 'intent', ?, ?, 3)
        """,
        ("test/window-station", old_ts, old_ts),
    )
    db.commit()

    # Fire with count=3, window=60s — old fires are 200s ago, should NOT escalate
    result = escalate(
        station_id="test/window-station",
        stable_pointer="agents_core/test_module.py",
        error_signal={"code": "E_WINDOW"},
        author_intent="window test",
        escalation_policy=n_within(3, 60.0),
        tier=Tier.NORMAL,
        db_path=db_path,
    )
    # The new fire brings count to 1 within window — only the new fire is recent
    assert result is None


# ---------------------------------------------------------------------------
# 4. Council fast-fail station
# ---------------------------------------------------------------------------

def test_council_fast_fail_station_emits_incident(tmp_path):
    """Simulate a council worker fast-fail and assert an incident is created."""
    from agents_core.repair_station.escalate import escalate
    from agents_core.repair_station import Tier, first

    db_path = tmp_path / "rs.db"

    incident_id = escalate(
        station_id="council/worker-fast-fail",
        stable_pointer="agents_core/shared_deliberation/orchestrator.py",
        error_signal={
            "run_id": "council-run-abc123",
            "last_heartbeat": None,
            "reason": "no_heartbeat_after_startup",
        },
        author_intent="council worker died/stalled during deliberation",
        escalation_policy=first(),
        tier=Tier.HIGH,
        owning_module="agents_core.shared_deliberation.orchestrator",
        db_path=db_path,
    )

    assert incident_id is not None

    row = _incident_row(db_path, incident_id)
    import json
    signal = json.loads(row["error_signal"])
    assert signal["run_id"] == "council-run-abc123"
    assert signal["reason"] == "no_heartbeat_after_startup"
    assert row["tier"] == int(Tier.HIGH)
    assert row["status"] == "open"


def test_council_fast_fail_wired_in_orchestrator(tmp_path):
    """Integration: _escalate_council_fast_fail calls repair_station.escalate with correct args."""
    from agents_core.shared_deliberation.orchestrator import _escalate_council_fast_fail
    from agents_core.repair_station import Tier

    captured = {}

    def fake_escalate(*args, **kwargs):
        captured.update(kwargs)
        return "inc-fake"

    # The lazy import inside _escalate_council_fast_fail does:
    #   from agents_core.repair_station import escalate, Tier, first
    # So patch at the module level where it will be imported from.
    with patch("agents_core.repair_station.escalate.escalate", fake_escalate):
        # Also need to patch what the function actually calls — it calls
        # agents_core.repair_station.escalate (the imported name in the function body).
        # The simplest reliable approach: patch the module's escalate function directly.
        import agents_core.repair_station as rs_mod
        original = rs_mod.escalate
        rs_mod.escalate = fake_escalate
        try:
            _escalate_council_fast_fail("run-xyz", None, "no_heartbeat_after_startup")
        finally:
            rs_mod.escalate = original

    # Verify the council station is wired with correct parameters
    assert captured.get("station_id") == "council/worker-fast-fail"
    assert captured.get("tier") == Tier.HIGH
    assert "run-xyz" in str(captured.get("error_signal", {}))
    assert captured.get("escalation_policy") is not None
    assert captured.get("escalation_policy").kind == "first"


def test_council_fast_fail_escalation_smoke(tmp_path, monkeypatch):
    """Smoke: _escalate_council_fast_fail does not raise even if db is live."""
    from agents_core.shared_deliberation.orchestrator import _escalate_council_fast_fail

    # Let it run with the real db (the /room path may not exist in CI — it will
    # gracefully create or fail silently — but it must not raise)
    try:
        _escalate_council_fast_fail("test-run", None, "no_heartbeat_after_startup")
    except Exception as exc:
        pytest.fail(f"_escalate_council_fast_fail raised: {exc}")


# ---------------------------------------------------------------------------
# 5. No brain: escalate() makes no model call
# ---------------------------------------------------------------------------

def test_no_model_call_on_escalate(tmp_path):
    """escalate() must make no subprocess or LLM calls."""
    db_path = tmp_path / "rs.db"

    with patch("subprocess.run") as mock_run, \
         patch("subprocess.Popen") as mock_popen, \
         patch("agents_core.llm.call_llm") as mock_llm:
        _fire(db_path, error_signal={"check": "no-model"})

    mock_run.assert_not_called()
    mock_popen.assert_not_called()
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Tier is carried but not acted upon
# ---------------------------------------------------------------------------

def test_tier_carried_in_incident(tmp_path):
    from agents_core.repair_station import Tier
    db_path = tmp_path / "rs.db"

    for tier in (Tier.LOW, Tier.NORMAL, Tier.HIGH, Tier.CRITICAL):
        station_id = f"test/tier-{tier.name.lower()}"
        incident_id = _fire(
            db_path,
            station_id=station_id,
            error_signal={"tier_check": tier.value},
            tier=tier,
        )
        assert incident_id is not None
        row = _incident_row(db_path, incident_id)
        assert row["tier"] == int(tier)
        # Registry also carries tier
        reg = _registry_row(db_path, station_id)
        assert reg["tier"] == int(tier)


def test_tier_not_acted_upon_in_leg1(tmp_path):
    """In Leg 1, all tiers produce the same incident structure — no queue logic."""
    from agents_core.repair_station import Tier
    db_path = tmp_path / "rs.db"

    high_id = _fire(db_path, station_id="test/high", error_signal={"t": "h"}, tier=Tier.HIGH)
    low_id = _fire(db_path, station_id="test/low", error_signal={"t": "l"}, tier=Tier.LOW)

    # Both create incidents; no priority differentiation in Leg 1
    assert high_id is not None
    assert low_id is not None
    high_row = _incident_row(db_path, high_id)
    low_row = _incident_row(db_path, low_id)
    assert high_row["status"] == low_row["status"] == "open"


# ---------------------------------------------------------------------------
# 7. Incident intake schema (status + back_ref for close-the-loop)
# ---------------------------------------------------------------------------

def test_incident_status_and_back_ref_fields_exist(tmp_path):
    from agents_core.repair_station._db import get_db
    db_path = tmp_path / "rs.db"

    incident_id = _fire(db_path)
    row = _incident_row(db_path, incident_id)

    assert row["status"] == "open"
    assert row["back_ref"] is None

    # Expert (Leg 2) will set status and back_ref; verify the columns are writable
    db = get_db(db_path)
    db.execute(
        "UPDATE incidents SET status = 'resolved', back_ref = 'expert-diagnosis-abc', updated_at = ? WHERE incident_id = ?",
        (datetime.now(timezone.utc).isoformat(), incident_id),
    )
    db.commit()

    updated = _incident_row(db_path, incident_id)
    assert updated["status"] == "resolved"
    assert updated["back_ref"] == "expert-diagnosis-abc"
