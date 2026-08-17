"""One-shot dry-run triage for the repair-station incident backlog (Leg 3 of
repair-station-close-dedup-triage-v0).

Reads every incident (all statuses) from the repair-station DB, recomputes a
COLLAPSED signature per row (error_signal minus the per-run-unique keys named
by types.PER_RUN_UNIQUE_KEYS — the same exclusion vocabulary Leg 2's
`signature_fields` station amendments in orchestrator.py use), groups by
(station_id, collapsed_signature), classifies each row's provenance, and
writes a markdown report + a JSON sidecar of PROPOSED dispositions.

Dry-run is the ONLY mode in this unit: this module never calls close_incident()
or dismiss_incident(), never UPDATEs or DELETEs a row. It reads, and it writes
two report files. Bulk close/dismiss of the backlog waits on Erah's
ratification of the report — that apply pass is a separate follow-up unit (see
Non-goals in the repair-station-close-dedup-triage-v0 spec).

Usage:
    python3 -m agents_core.repair_station.triage
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from agents_core.room_paths import room_path

from ._db import get_db
from .escalate import _compute_signature
from .types import PER_RUN_UNIQUE_KEYS

# ---------------------------------------------------------------------------
# Read (no writes)
# ---------------------------------------------------------------------------


def _read_all_incidents(db_path: Path | None = None) -> list[dict]:
    """Read every incident row, all statuses. Read-only."""
    db = get_db(db_path)
    rows = db.execute("SELECT * FROM incidents ORDER BY created_at ASC").fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["error_signal"] = json.loads(d["error_signal"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Collapsed signature — same exclusion vocabulary as Leg 2's signature_fields
# ---------------------------------------------------------------------------


def _collapsed_signature(error_signal: dict) -> tuple[dict, str]:
    subset = {k: v for k, v in error_signal.items() if k not in PER_RUN_UNIQUE_KEYS}
    return subset, _compute_signature(subset)


# ---------------------------------------------------------------------------
# Provenance classification — run-id / commit-sha provenance, not keyword match
# ---------------------------------------------------------------------------

# Production run_ids are minted by agents_core/council/cli.py::new_run_id() as
# "YYYY-MM-DD-HHMMSS-<6-hex>" at fire time (verified this session, cli.py:194-197).
_RUN_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{6})-([0-9a-f]{6})$")
# A real git commit hash, full or abbreviated.
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

_RUN_ID_MAX_SKEW_S = 86400.0  # a day of slack between run_id's embedded time and created_at


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_test_pollution(row: dict) -> bool:
    """Provenance check, not keyword matching: ask whether the identifying field in
    the payload could plausibly have been produced by the real generating process,
    not whether it contains the substring 'test'
    (correction/repair-station-reverified-92-open-and-test-pollution-widened-2026-08-03's
    method).

    council/worker-fast-fail: a production run_id matches
    "YYYY-MM-DD-HHMMSS-<6-hex>" and its embedded timestamp lands close to the
    incident's created_at. A run_id that doesn't match that shape at all (e.g. the
    literal "test-run"/"run-xyz" fixtures in tests/test_repair_station.py), or one
    that matches but whose embedded time is off by more than a day (e.g. the
    "2026-01-01-000000-*" sentinel in tests/test_council_resilience.py, live-verified
    this session against real dates in the 2026-06 through 2026-08 range), did not
    come from a real fire.

    shared-deliberation/facets-grounding-denied: resolved_sha is expected to be a
    real git commit hash (>=7 hex chars) when grounding actually ran; a shorter or
    non-hex value (e.g. the "abc" fixture value observed live in the backlog) could
    not have come from git.

    Rows carrying neither field are NOT asserted as pollution — absence of a
    positive signal is not evidence.
    """
    signal = row["error_signal"]

    run_id = signal.get("run_id")
    if isinstance(run_id, str):
        m = _RUN_ID_RE.match(run_id)
        if not m:
            return True
        try:
            embedded = datetime.strptime(m.group(1), "%Y-%m-%d-%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return True
        created = _parse_iso(row.get("created_at"))
        if created is not None and abs((created - embedded).total_seconds()) > _RUN_ID_MAX_SKEW_S:
            return True
        return False

    resolved_sha = signal.get("resolved_sha")
    if isinstance(resolved_sha, str) and resolved_sha:
        return not bool(_SHA_RE.match(resolved_sha))

    return False


# ---------------------------------------------------------------------------
# Grouping + disposition
# ---------------------------------------------------------------------------


def _group_incidents(rows: list[dict]) -> list[dict]:
    """Group by (station_id, collapsed_signature); classify each row's provenance;
    compute per-group aggregates and a proposed disposition. Pure function — no I/O,
    no DB writes. Mutates each row dict in place with _collapsed_signature,
    _collapsed_subset, _provenance. Returns groups sorted by (station_id, first_seen).
    """
    for row in rows:
        subset, sig = _collapsed_signature(row["error_signal"])
        row["_collapsed_subset"] = subset
        row["_collapsed_signature"] = sig
        row["_is_pollution"] = _is_test_pollution(row)

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["station_id"], row["_collapsed_signature"])].append(row)

    groups = []
    for (station_id, sig), members in by_key.items():
        members = sorted(members, key=lambda r: r["created_at"])
        non_pollution = [r for r in members if not r["_is_pollution"]]
        genuine_id = non_pollution[0]["incident_id"] if non_pollution else None

        for r in members:
            if r["_is_pollution"]:
                r["_provenance"] = "test-pollution"
            elif r["incident_id"] == genuine_id:
                r["_provenance"] = "genuine"
            else:
                r["_provenance"] = "duplicate-of-open-group"

        provenance_counts = Counter(r["_provenance"] for r in members)
        status_counts = Counter(r["status"] for r in members)

        # Disposition precedence, most conservative first: only propose dismissing a
        # group outright when EVERY member is test-pollution provenance. A group with
        # any duplicate rows proposes closing the redundant ones (the earliest/genuine
        # row is not touched). A clean singleton with no pollution and no duplicates
        # proposes nothing.
        if provenance_counts["test-pollution"] == len(members):
            disposition = "dismiss:test-pollution"
        elif provenance_counts["duplicate-of-open-group"] > 0:
            disposition = "close:duplicate"
        else:
            disposition = "keep-open"

        groups.append(
            {
                "station_id": station_id,
                "collapsed_signature": sig,
                "collapsed_subset": members[0]["_collapsed_subset"],
                "member_count": len(members),
                "status_counts": dict(status_counts),
                "provenance_counts": {
                    "test-pollution": provenance_counts.get("test-pollution", 0),
                    "duplicate-of-open-group": provenance_counts.get(
                        "duplicate-of-open-group", 0
                    ),
                    "genuine": provenance_counts.get("genuine", 0),
                },
                "first_seen": members[0]["created_at"],
                "last_seen": members[-1]["created_at"],
                "sample_incident_id": members[0]["incident_id"],
                "sample_error_signal": members[0]["error_signal"],
                "member_incident_ids": [r["incident_id"] for r in members],
                "proposed_disposition": disposition,
                "human_verify": "required",
            }
        )

    groups.sort(key=lambda g: (g["station_id"], g["first_seen"]))
    return groups


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_markdown(
    groups: list[dict], totals: dict, report_id: str, generated_at: str, db_path: Path, out_path: Path
) -> None:
    lines = [
        f"# Repair Station Triage Report — {report_id}",
        "",
        f"Generated: {generated_at} · DB: `{db_path}`",
        "",
        "**DRY RUN.** This is a proposal for Erah to ratify, not an executed action —",
        "no incident was closed, dismissed, or otherwise modified by this report. Every",
        "disposition below carries `human-verify: required`; applying dispositions is a",
        "separate follow-up unit gated on explicit per-group human ratification (see",
        "Non-goals in the repair-station-close-dedup-triage-v0 spec).",
        "",
        "## Summary",
        "",
        f"- Total incidents (all statuses): {totals['incidents_total']}",
        f"- Status: {_fmt_counts(totals['status_counts'])}",
        f"- Provenance: {_fmt_counts(totals['provenance_counts'])}",
        f"- Groups: {totals['group_count']}",
        f"- Proposed dispositions: {_fmt_counts(totals['disposition_counts'])}",
        "",
        "## Groups",
        "",
    ]

    for g in groups:
        lines.append(
            f"### `{g['station_id']}` — {g['collapsed_signature']}  "
            f"(`{g['proposed_disposition']}`)"
        )
        lines.append("")
        lines.append(f"- Members: {g['member_count']} — status: {_fmt_counts(g['status_counts'])}")
        lines.append(f"- Provenance: {_fmt_counts(g['provenance_counts'])}")
        lines.append(f"- First seen: {g['first_seen']}  ·  Last seen: {g['last_seen']}")
        lines.append(f"- Collapsed fields: `{json.dumps(g['collapsed_subset'], sort_keys=True, default=str)}`")
        lines.append(f"- Sample incident: `{g['sample_incident_id']}`")
        lines.append(
            f"- Sample payload: `{json.dumps(g['sample_error_signal'], sort_keys=True, default=str)}`"
        )
        lines.append(f"- **Proposed disposition: `{g['proposed_disposition']}`** — human-verify: required")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def _fmt_counts(counts: dict) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _write_json(groups: list[dict], totals: dict, report_id: str, generated_at: str, db_path: Path, out_path: Path) -> None:
    sidecar = {
        "report_id": report_id,
        "generated_at": generated_at,
        "db_path": str(db_path),
        "totals": totals,
        "groups": groups,
    }
    out_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(db_path: Path | None = None, report_dir: Path | None = None) -> dict:
    """Run the dry-run triage pass. Writes NOTHING to the incidents DB — only the
    markdown report and JSON sidecar. Returns the summary dict."""
    resolved_db_path = db_path or room_path("repair_station")
    resolved_report_dir = report_dir or resolved_db_path.parent
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_all_incidents(db_path)
    groups = _group_incidents(rows)

    generated_at = datetime.now(timezone.utc).isoformat()
    date_str = generated_at[:10]
    report_id = f"triage-report-{date_str}"

    totals = {
        "incidents_total": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "provenance_counts": dict(Counter(r["_provenance"] for r in rows)),
        "group_count": len(groups),
        "disposition_counts": dict(Counter(g["proposed_disposition"] for g in groups)),
    }

    md_path = resolved_report_dir / f"{report_id}.md"
    json_path = resolved_report_dir / f"{report_id}.json"

    _write_markdown(groups, totals, report_id, generated_at, resolved_db_path, md_path)
    _write_json(groups, totals, report_id, generated_at, resolved_db_path, json_path)

    return {
        "report_id": report_id,
        "md_path": str(md_path),
        "json_path": str(json_path),
        "totals": totals,
    }


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, sort_keys=True))
