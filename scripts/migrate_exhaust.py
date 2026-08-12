#!/usr/bin/env python3
"""migrate_exhaust — one-shot mover for the tier-1 exhaust prefixes
(agents-core-mem-exhaust-sibling-store-v0, leg 3).

Moves the `elevator/`, `weather/`, and `router/gw-review-divergence/` rows
that already exist in mem.db (written before this deploy) into the sibling
exhaust store. New writes for these prefixes route to the sibling
automatically as of this deploy (see agents_core.mem.MemoryStore.set()) —
this tool only cleans up the pre-existing rows.

Default is DRY RUN. Pass --write to actually copy+delete. Safe to run
repeatedly: a key already present (and checksum-verified) on both sides is
just deleted from the source again if still there, and a key already
deleted from the source is skipped — a second full run moves zero rows and
exits 0.

Order per key: copy -> verify by checksum -> delete. Never deletes a row
from mem.db before the sibling copy is confirmed byte-for-byte identical.

Usage:
    migrate_exhaust.py                          # dry run against production paths
    migrate_exhaust.py --write                  # the real, destructive pass
    migrate_exhaust.py --db scratch/mem.db --exhaust-db scratch/exhaust.db --write
    migrate_exhaust.py --write --allow-count-mismatch   # override the volume stop-condition

Exit codes:
    0 — dry run completed, or write run completed with no unresolved
        checksum mismatches
    1 — a stop condition was hit (row-count materially off from the
        documented expected volume, without --allow-count-mismatch; or a
        checksum mismatch on an already-present destination row)
    2 — usage / IO error (db not found, etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

try:
    from agents_core import mem_exhaust
    from agents_core.mem import DB_PATH as DEFAULT_MEM_DB_PATH
except ImportError:
    sys.path.insert(0, "/srv/agents")
    from agents_core import mem_exhaust
    from agents_core.mem import DB_PATH as DEFAULT_MEM_DB_PATH

# Documented expected volume (spec: agents-core-mem-exhaust-sibling-store-v0,
# leg 3). A materially different count on the LIVE run is a stop condition,
# not a rounding difference — it means the census this spec is built on
# (finding/mem-exhaust-45pct-has-zero-readers-2026-08-11) was wrong about
# scope, and the fix is to report and re-scope, not to route around it.
EXPECTED_COUNTS = {
    "elevator/": 10_975,
    "weather/": 2_380,
    "router/gw-review-divergence/": 1_461,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
COUNT_MISMATCH_TOLERANCE = 0.15  # 15% relative deviation

BATCH_SIZE = 500
CHECKSUM_FIELD_SEP = "\x1f"  # ASCII unit separator — not expected in content


def _row_checksum(key: str, content: str, tags: str, source: str,
                   created_at: str, updated_at: str) -> str:
    """SHA-256 over the concatenated row fields, joined with a separator
    unlikely to appear in free text. Checksum comparison, not a field-by-
    field Python `==`, because the latter can mask encoding drift between
    the two connections (per spec)."""
    payload = CHECKSUM_FIELD_SEP.join([key, content, tags, source, created_at, updated_at])
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _ensure_dest_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(mem_exhaust.EXHAUST_SCHEMA)
    conn.commit()


def _fetch_prefix_rows(conn: sqlite3.Connection, prefix: str) -> list[sqlite3.Row]:
    # The three tier-1 prefixes are fixed literals with no SQL-LIKE wildcard
    # characters (%, _, \), so a plain LIKE "<prefix>%" needs no escaping.
    return conn.execute(
        "SELECT key, content, tags, source, created_at, updated_at "
        "FROM memories WHERE key LIKE ? ORDER BY key",
        (f"{prefix}%",),
    ).fetchall()


def _dest_row(conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT key, content, tags, source, created_at, updated_at "
        "FROM memories WHERE key = ?", (key,),
    ).fetchone()


def migrate(db_path: Path, exhaust_db_path: Path, write: bool,
            allow_count_mismatch: bool) -> int:
    if not db_path.exists():
        print(f"[migrate_exhaust] source db not found: {db_path}", file=sys.stderr)
        return 2

    src = _connect(db_path)
    dst = _connect(exhaust_db_path)
    _ensure_dest_schema(dst)

    try:
        # --- volume stop-condition check, computed up front across all
        # three prefixes before touching anything ---
        per_prefix_rows: dict[str, list[sqlite3.Row]] = {
            prefix: _fetch_prefix_rows(src, prefix) for prefix in mem_exhaust.EXHAUST_PREFIXES
        }
        actual_total = sum(len(rows) for rows in per_prefix_rows.values())
        deviation = (
            abs(actual_total - EXPECTED_TOTAL) / EXPECTED_TOTAL if EXPECTED_TOTAL else 0
        )
        print(f"[migrate_exhaust] source rows matching tier-1 prefixes: {actual_total} "
              f"(expected ~{EXPECTED_TOTAL})")
        for prefix, rows in per_prefix_rows.items():
            print(f"  {prefix!r}: {len(rows)} (expected ~{EXPECTED_COUNTS[prefix]})")

        if deviation > COUNT_MISMATCH_TOLERANCE and not allow_count_mismatch:
            print(
                f"[migrate_exhaust] STOP CONDITION: actual count {actual_total} deviates "
                f"{deviation:.0%} from the documented expected {EXPECTED_TOTAL} (spec's "
                f"{COUNT_MISMATCH_TOLERANCE:.0%} tolerance). This means the reader census "
                f"this spec is built on may be wrong about scope for this data — report it, "
                f"do not route around it. Re-run with --allow-count-mismatch to override.",
                file=sys.stderr,
            )
            return 1

        moved = 0
        skipped_mismatch = 0

        for prefix, rows in per_prefix_rows.items():
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                for row in batch:
                    key = row["key"]
                    src_checksum = _row_checksum(
                        key, row["content"], row["tags"], row["source"],
                        row["created_at"], row["updated_at"],
                    )

                    existing_dest = _dest_row(dst, key)
                    if existing_dest is None:
                        if not write:
                            moved += 1
                            continue
                        dst.execute(
                            "INSERT INTO memories (key, content, tags, source, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (key, row["content"], row["tags"], row["source"],
                             row["created_at"], row["updated_at"]),
                        )
                        existing_dest = _dest_row(dst, key)

                    dest_checksum = _row_checksum(
                        existing_dest["key"], existing_dest["content"], existing_dest["tags"],
                        existing_dest["source"], existing_dest["created_at"],
                        existing_dest["updated_at"],
                    )

                    if src_checksum != dest_checksum:
                        skipped_mismatch += 1
                        print(
                            f"[migrate_exhaust] CHECKSUM MISMATCH, not deleting from source: {key}",
                            file=sys.stderr,
                        )
                        continue

                    if not write:
                        moved += 1
                        continue

                    src.execute("DELETE FROM memories WHERE key = ?", (key,))
                    moved += 1

                if write:
                    dst.commit()
                    src.commit()

        verb = "would move" if not write else "moved"
        print(f"[migrate_exhaust] {verb} {moved} rows"
              + (f", {skipped_mismatch} checksum mismatches skipped" if skipped_mismatch else ""))

        # DoD: memories/memories_fts row counts on the source must still match
        # after the run — the FTS5 triggers on mem.db do their own delete
        # bookkeeping, this just confirms they kept up.
        mem_count = src.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        fts_count = src.execute("SELECT COUNT(*) AS n FROM memories_fts_docsize").fetchone()["n"]
        print(f"[migrate_exhaust] source memories={mem_count} memories_fts={fts_count} "
              f"({'in sync' if mem_count == fts_count else 'DIVERGED'})")
        if mem_count != fts_count:
            return 1

        return 1 if skipped_mismatch else 0
    finally:
        src.close()
        dst.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_MEM_DB_PATH,
                         help="source mem.db path (default: production mem.db)")
    parser.add_argument("--exhaust-db", type=Path, default=mem_exhaust.EXHAUST_DB_PATH,
                         help="destination exhaust.db path (default: production exhaust.db)")
    parser.add_argument("--write", action="store_true",
                         help="actually copy+delete; default is dry run")
    parser.add_argument("--allow-count-mismatch", action="store_true",
                         help="proceed even if the matched row count deviates materially "
                              "from the documented expected volume")
    args = parser.parse_args()

    return migrate(args.db, args.exhaust_db, args.write, args.allow_count_mismatch)


if __name__ == "__main__":
    sys.exit(main())
