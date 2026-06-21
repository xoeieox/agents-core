"""escalate() — thin escalation client for repair stations.

Call this from any module to report a structured failure. On a qualifying fire
(passes escalation policy + dedup gate), writes an incident to the intake and
self-registers the station. Never calls a model. Fast enough to call synchronously
from hot paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .types import EscalationPolicy, Incident, Tier
from ._db import get_db

log = logging.getLogger("repair-station")

# Default dedup window: suppress duplicate incidents for the same
# (station_id, error_signature) if one is already open and was created
# within this many seconds.
DEFAULT_DEDUP_WINDOW_S = 3600.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_signature(error_signal: dict) -> str:
    data = json.dumps(error_signal, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _upsert_registry(
    db,
    station_id: str,
    owning_module: str,
    stable_pointer: str,
    tier: Tier,
    author_intent: str,
    now_iso: str,
) -> int:
    """Insert or update station registry row. Returns updated fire_count."""
    row = db.execute(
        "SELECT fire_count FROM station_registry WHERE station_id = ?",
        (station_id,),
    ).fetchone()

    if row is None:
        db.execute(
            """
            INSERT INTO station_registry
              (station_id, owning_module, stable_pointer, tier, author_intent,
               first_fire, last_fire, fire_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (station_id, owning_module, stable_pointer, int(tier), author_intent,
             now_iso, now_iso),
        )
        db.commit()
        return 1
    else:
        new_count = row["fire_count"] + 1
        db.execute(
            """
            UPDATE station_registry
               SET last_fire = ?, fire_count = ?,
                   stable_pointer = ?, tier = ?, author_intent = ?,
                   owning_module = ?
             WHERE station_id = ?
            """,
            (now_iso, new_count, stable_pointer, int(tier), author_intent,
             owning_module, station_id),
        )
        db.commit()
        return new_count


def _record_fire(db, station_id: str, now_iso: str) -> None:
    db.execute(
        "INSERT INTO station_fires (station_id, fired_at) VALUES (?, ?)",
        (station_id, now_iso),
    )
    db.commit()


def _policy_gates(
    db,
    policy: EscalationPolicy,
    station_id: str,
    fire_count: int,
    now_iso: str,
) -> bool:
    """Return True if the escalation policy allows creating an incident."""
    if policy.kind == "first":
        return True  # Always escalate (dedup handles idempotency)

    if policy.kind == "n_within":
        # Count fires within the window
        cutoff = _iso_minus_seconds(now_iso, policy.window_s)
        row = db.execute(
            """
            SELECT COUNT(*) AS cnt FROM station_fires
             WHERE station_id = ? AND fired_at >= ?
            """,
            (station_id, cutoff),
        ).fetchone()
        return row["cnt"] >= policy.count

    log.warning("Unknown escalation policy kind %r — defaulting to escalate", policy.kind)
    return True


def _has_open_incident(
    db,
    station_id: str,
    error_signature: str,
    now_iso: str,
    dedup_window_s: float,
) -> bool:
    """Return True if there is already an open incident for this (station, signature)
    created within dedup_window_s seconds."""
    cutoff = _iso_minus_seconds(now_iso, dedup_window_s)
    row = db.execute(
        """
        SELECT incident_id FROM incidents
         WHERE station_id = ?
           AND error_signature = ?
           AND status = 'open'
           AND created_at >= ?
         LIMIT 1
        """,
        (station_id, error_signature, cutoff),
    ).fetchone()
    return row is not None


def _create_incident(
    db,
    station_id: str,
    stable_pointer: str,
    error_signal: dict,
    author_intent: str,
    tier: Tier,
    error_signature: str,
    now_iso: str,
) -> str:
    incident_id = f"inc-{uuid.uuid4().hex[:12]}"
    db.execute(
        """
        INSERT INTO incidents
          (incident_id, station_id, stable_pointer, error_signal, author_intent,
           tier, error_signature, status, back_ref, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?)
        """,
        (
            incident_id,
            station_id,
            stable_pointer,
            json.dumps(error_signal, default=str),
            author_intent,
            int(tier),
            error_signature,
            now_iso,
            now_iso,
        ),
    )
    db.commit()
    return incident_id


def _iso_minus_seconds(iso: str, seconds: float) -> str:
    from datetime import timedelta
    dt = datetime.fromisoformat(iso)
    return (dt - timedelta(seconds=seconds)).isoformat()


def escalate(
    station_id: str,
    stable_pointer: str,
    error_signal: dict,
    author_intent: str,
    escalation_policy: EscalationPolicy,
    tier: Tier,
    *,
    owning_module: str = "",
    dedup_window_s: float = DEFAULT_DEDUP_WINDOW_S,
    db_path: Path | None = None,
) -> str | None:
    """Fire a repair station.

    Applies the escalation policy and dedup gate. On a qualifying fire, writes
    an incident to the intake and self-registers (or updates) the station.

    Args:
        station_id:          Stable dot-or-slash namespaced ID, e.g. "council/worker-fast-fail".
        stable_pointer:      Where the subject + docs live (module path, mem-key, RAG scope).
                             The Expert reads context fresh from here; nothing rich is embedded.
        error_signal:        Structured failure payload (typed opaque dict).
        author_intent:       One line — why this station exists. The only embedded prose.
        escalation_policy:   When to escalate (first() | n_within(count, window_s)).
        tier:                Priority tier (carried to Leg 2; not acted on here).
        owning_module:       Optional: module path that owns this station (for registry).
        dedup_window_s:      Suppress duplicate incidents within this many seconds (default 1h).
        db_path:             Override DB path (used in tests).

    Returns:
        incident_id (str) if an incident was created, else None.
    """
    now_iso = _now_iso()
    error_signature = _compute_signature(error_signal)
    db = get_db(db_path)

    try:
        fire_count = _upsert_registry(
            db, station_id, owning_module, stable_pointer, tier, author_intent, now_iso
        )
        _record_fire(db, station_id, now_iso)

        if not _policy_gates(db, escalation_policy, station_id, fire_count, now_iso):
            log.debug(
                "repair-station %s: policy gate blocked (count=%d, policy=%s)",
                station_id, fire_count, escalation_policy,
            )
            return None

        if _has_open_incident(db, station_id, error_signature, now_iso, dedup_window_s):
            log.debug(
                "repair-station %s: dedup suppressed (signature=%s)",
                station_id, error_signature,
            )
            return None

        incident_id = _create_incident(
            db, station_id, stable_pointer, error_signal,
            author_intent, tier, error_signature, now_iso,
        )
        log.info(
            "repair-station %s: incident created %s (tier=%s)",
            station_id, incident_id, tier.name,
        )
        return incident_id

    except Exception:
        log.exception("repair-station escalate() failed for station %r — suppressed", station_id)
        return None


def get_incident(incident_id: str, *, db_path: Path | None = None) -> Incident | None:
    """Fetch a single incident by ID. Returns None if not found."""
    db = get_db(db_path)
    row = db.execute(
        "SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_incident(row)


def list_open_incidents(
    station_id: str | None = None, *, db_path: Path | None = None
) -> list[Incident]:
    """List open incidents, optionally filtered by station_id."""
    db = get_db(db_path)
    if station_id:
        rows = db.execute(
            "SELECT * FROM incidents WHERE status = 'open' AND station_id = ? ORDER BY created_at DESC",
            (station_id,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM incidents WHERE status = 'open' ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_incident(r) for r in rows]


def _row_to_incident(row) -> Incident:
    return Incident(
        incident_id=row["incident_id"],
        station_id=row["station_id"],
        stable_pointer=row["stable_pointer"],
        error_signal=json.loads(row["error_signal"]),
        author_intent=row["author_intent"],
        tier=Tier(row["tier"]),
        error_signature=row["error_signature"],
        status=row["status"],
        back_ref=row["back_ref"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
