"""agents_core.synthesis_cache — dedicated writer for the synthesis cache.

Separate from vault_writer; the synthesis cache is NOT vault corpus.

Content-addressed: entries are immutable once written at a given artifact_id.
Writes are atomic (temp + rename); no flock needed because content-addressed
paths can never collide on a single host.

On-disk layout::

    /data/synthesis-cache/              (override: SYNTHESIS_CACHE_ROOT)
        by-hash/<artifact_id_hex>/
            entry.json      # canonical sorted-key UTF-8 JSON, no trailing newline
            entry.sig       # hex Ed25519 signature over entry.json bytes
        index/
            by-claim.sqlite          # cache hit lookup (eager-maintained)
            by-source.sqlite         # source-path → artifacts citing it (eager-maintained)
            trigger-requests.sqlite  # consumer refresh-pressure signal

Artifact-id canonicalization
-----------------------------
The artifact_id for a synthesis entry is::

    artifact_id = "sha256:" + sha256_hex(
        canonical_json({
            "claim":              claim.strip(),         # case preserved
            "corpus":             sorted(scope_corpus),  # sorted for stability
            "policy":             policy,
            "format_schema_hash": "sha256:" + sha256_hex(canonical_json(format or {})),
        })
    )

Excluded from identity: ``freshness`` (read-time policy), the literal ``format``
value (only its schema hash counts).  ``format=None`` and ``format={}`` collapse
to the same hash; callers wanting structured output must pass a non-empty schema.

On-disk paths strip the ``sha256:`` prefix (colon is filesystem-valid but annoying).
Values inside ``entry.json`` and ``WriteRecord`` carry the prefix.

Index maintenance
-----------------
``put()`` eagerly updates all three indexes in the same logical operation.
``update_source_tracking()`` is called by the librarian's vault-event subscriber
to update ``latest_content_hash`` as the corpus evolves.
``rebuild_indexes()`` is a recovery tool that reconstructs by-claim and by-source
from on-disk entries (trigger-requests are not stored in entries and cannot be
rebuilt).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_CACHE_ROOT = Path("/data/synthesis-cache")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _cache_root() -> Path:
    return Path(os.environ.get("SYNTHESIS_CACHE_ROOT", str(_DEFAULT_CACHE_ROOT)))


def strip_hash_prefix(artifact_id: str) -> str:
    """Strip ``sha256:`` prefix for use in filesystem paths."""
    return artifact_id.removeprefix("sha256:")


def add_hash_prefix(hex_id: str) -> str:
    """Add ``sha256:`` prefix to a bare hex id."""
    if hex_id.startswith("sha256:"):
        return hex_id
    return f"sha256:{hex_id}"


def _entry_dir(artifact_id: str) -> Path:
    return _cache_root() / "by-hash" / strip_hash_prefix(artifact_id)


def _index_dir() -> Path:
    return _cache_root() / "index"


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """Stable sorted-key JSON with no trailing newline."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_artifact_id(
    claim: str,
    corpus: list[str],
    policy: str,
    format: dict | None = None,
) -> str:
    """Compute the deterministic artifact_id for a (claim, scope, policy, format) tuple.

    This is the cache-lookup key: same query params always produce the same
    artifact_id regardless of freshness or the literal format value.

    Note: ``format=None`` and ``format={}`` produce the same hash.  Callers
    wanting structured output must pass a non-empty schema.
    """
    canonical = {
        "claim": claim.strip(),
        "corpus": sorted(corpus),
        "policy": policy,
        "format_schema_hash": f"sha256:{sha256_hex(canonical_json(format or {}))}",
    }
    return f"sha256:{sha256_hex(canonical_json(canonical))}"


def compute_format_schema_hash(format: dict | None = None) -> str:
    return f"sha256:{sha256_hex(canonical_json(format or {}))}"


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp.")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        os.rename(tmp_name, str(path))
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# SQLite connection singletons (resettable for tests)
# ---------------------------------------------------------------------------

_claim_conn: sqlite3.Connection | None = None
_source_conn: sqlite3.Connection | None = None
_trigger_conn: sqlite3.Connection | None = None


def reset_connections() -> None:
    """Close and clear all cached connections. Used in tests to pick up env changes."""
    global _claim_conn, _source_conn, _trigger_conn
    for conn in (_claim_conn, _source_conn, _trigger_conn):
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    _claim_conn = _source_conn = _trigger_conn = None


def _connect_claim() -> sqlite3.Connection:
    global _claim_conn
    if _claim_conn is None:
        _claim_conn = _open_claim_db()
    return _claim_conn


def _connect_source() -> sqlite3.Connection:
    global _source_conn
    if _source_conn is None:
        _source_conn = _open_source_db()
    return _source_conn


def _connect_trigger() -> sqlite3.Connection:
    global _trigger_conn
    if _trigger_conn is None:
        _trigger_conn = _open_trigger_db()
    return _trigger_conn


def _open_claim_db() -> sqlite3.Connection:
    path = _index_dir() / "by-claim.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS by_claim (
            artifact_id         TEXT PRIMARY KEY,
            claim_normalized    TEXT NOT NULL,
            scope_corpus_json   TEXT NOT NULL,
            scope_policy        TEXT NOT NULL,
            format_schema_hash  TEXT NOT NULL,
            synthesized_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_claim_lookup
        ON by_claim(claim_normalized, scope_corpus_json, scope_policy, format_schema_hash)
    """)
    conn.commit()
    return conn


def _open_source_db() -> sqlite3.Connection:
    path = _index_dir() / "by-source.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS by_source (
            source_path             TEXT NOT NULL,
            source_content_hash     TEXT NOT NULL,
            artifact_id             TEXT NOT NULL,
            latest_content_hash     TEXT NOT NULL,
            latest_observed_at      TEXT NOT NULL,
            PRIMARY KEY (source_path, artifact_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source_path ON by_source(source_path)")
    conn.commit()
    return conn


def _open_trigger_db() -> sqlite3.Connection:
    path = _index_dir() / "trigger-requests.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trigger_requests (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ts           TEXT NOT NULL,
            artifact_id  TEXT NOT NULL,
            requester_id TEXT NOT NULL,
            reason       TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trigger_artifact ON trigger_requests(artifact_id, ts)")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def put(entry: dict, sig_hex: str) -> None:
    """Write a synthesis entry to disk and eagerly update all three indexes.

    Parameters
    ----------
    entry:
        Full entry dict matching the spec JSON schema.  ``entry["artifact_id"]``
        must carry the ``sha256:`` prefix.
    sig_hex:
        Hex-encoded Ed25519 signature over the entry.json canonical bytes
        (sorted keys, UTF-8, no trailing newline).
    """
    artifact_id: str = entry["artifact_id"]
    entry_dir = _entry_dir(artifact_id)
    entry_dir.mkdir(parents=True, exist_ok=True)

    # entry.json: sorted-key UTF-8, no trailing newline
    entry_bytes = canonical_json(entry).encode("utf-8")
    _atomic_write(entry_dir / "entry.json", entry_bytes)
    _atomic_write(entry_dir / "entry.sig", sig_hex.encode("utf-8"))

    _index_put(entry)


def _index_put(entry: dict) -> None:
    """Eagerly insert/replace rows in all three indexes for a newly written entry."""
    artifact_id = entry["artifact_id"]
    claim_normalized = entry["claim"].strip()
    scope = entry["scope_identity"]
    scope_corpus_json = canonical_json(scope["corpus"])
    scope_policy = scope["policy"]
    format_schema_hash = scope["format_schema_hash"]
    synthesized_at = entry["synthesized_at"]

    # by-claim.sqlite: one row per artifact (INSERT OR REPLACE since artifact_id is PK)
    cc = _connect_claim()
    cc.execute("""
        INSERT OR REPLACE INTO by_claim
            (artifact_id, claim_normalized, scope_corpus_json, scope_policy, format_schema_hash, synthesized_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (artifact_id, claim_normalized, scope_corpus_json, scope_policy, format_schema_hash, synthesized_at))
    cc.commit()

    # by-source.sqlite: one row per (source_path, artifact_id)
    sc = _connect_source()
    now_ts = datetime.now(tz=timezone.utc).isoformat()
    for citation in entry.get("citations", []):
        source_path = citation["path"]
        source_content_hash = citation["content_hash"]
        sc.execute("""
            INSERT OR REPLACE INTO by_source
                (source_path, source_content_hash, artifact_id, latest_content_hash, latest_observed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (source_path, source_content_hash, artifact_id, source_content_hash, now_ts))
    sc.commit()


def get(artifact_id: str) -> dict | None:
    """Load a cache entry by artifact_id. Returns None if not found."""
    entry_path = _entry_dir(artifact_id) / "entry.json"
    if not entry_path.exists():
        return None
    return json.loads(entry_path.read_text(encoding="utf-8"))


def get_sig(artifact_id: str) -> str | None:
    """Load the hex signature for an artifact. Returns None if not found."""
    sig_path = _entry_dir(artifact_id) / "entry.sig"
    if not sig_path.exists():
        return None
    return sig_path.read_text(encoding="utf-8").strip()


def lookup_by_claim(
    claim: str,
    corpus: list[str],
    policy: str,
    format_schema_hash: str,
) -> str | None:
    """Return the most recent artifact_id matching these query params, or None.

    Parameters correspond to the cache-lookup key fields in the spec.
    """
    claim_normalized = claim.strip()
    scope_corpus_json = canonical_json(sorted(corpus))
    conn = _connect_claim()
    cur = conn.execute("""
        SELECT artifact_id FROM by_claim
        WHERE claim_normalized = ?
          AND scope_corpus_json = ?
          AND scope_policy = ?
          AND format_schema_hash = ?
        ORDER BY synthesized_at DESC
        LIMIT 1
    """, (claim_normalized, scope_corpus_json, policy, format_schema_hash))
    row = cur.fetchone()
    return row[0] if row else None


def update_source_tracking(source_path: str, content_hash: str, observed_at: str) -> None:
    """Update latest_content_hash for all artifacts citing source_path.

    Called by the librarian's vault-event subscriber on every relevant vault write.
    """
    conn = _connect_source()
    conn.execute("""
        UPDATE by_source
        SET latest_content_hash = ?, latest_observed_at = ?
        WHERE source_path = ?
    """, (content_hash, observed_at, source_path))
    conn.commit()


def get_source_rows(artifact_id: str) -> list[dict]:
    """Return all by-source rows for an artifact (for degree-of-shift computation)."""
    conn = _connect_source()
    cur = conn.execute("""
        SELECT source_path, source_content_hash, latest_content_hash, latest_observed_at
        FROM by_source
        WHERE artifact_id = ?
    """, (artifact_id,))
    return [
        {
            "source_path": r[0],
            "source_content_hash": r[1],
            "latest_content_hash": r[2],
            "latest_observed_at": r[3],
        }
        for r in cur.fetchall()
    ]


def get_max_observed_at() -> str | None:
    """Return the most recent latest_observed_at across all source rows, or None."""
    conn = _connect_source()
    cur = conn.execute("SELECT MAX(latest_observed_at) FROM by_source")
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def add_trigger_request(
    artifact_id: str,
    requester_id: str,
    reason: str,
    ts: str | None = None,
) -> None:
    """Record a consumer refresh-pressure request."""
    conn = _connect_trigger()
    if ts is None:
        ts = datetime.now(tz=timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO trigger_requests (ts, artifact_id, requester_id, reason)
        VALUES (?, ?, ?, ?)
    """, (ts, artifact_id, requester_id, reason))
    conn.commit()


def get_trigger_requests(*, since: datetime | None = None) -> list[dict]:
    """Return trigger-request rows, optionally filtered to ts >= since."""
    conn = _connect_trigger()
    if since is not None:
        cur = conn.execute("""
            SELECT id, ts, artifact_id, requester_id, reason
            FROM trigger_requests
            WHERE ts >= ?
            ORDER BY ts
        """, (since.isoformat(),))
    else:
        cur = conn.execute(
            "SELECT id, ts, artifact_id, requester_id, reason FROM trigger_requests ORDER BY ts"
        )
    return [
        {"id": r[0], "ts": r[1], "artifact_id": r[2], "requester_id": r[3], "reason": r[4]}
        for r in cur.fetchall()
    ]


def rebuild_indexes() -> None:
    """Reconstruct by-claim and by-source indexes from authoritative by-hash entries.

    Recovery tool; not part of normal operation.  Trigger-request history cannot
    be reconstructed (it is not stored in entries) and is cleared.

    Usage::

        synthesis_cache.rebuild_indexes()
    """
    # Drop existing databases
    for name in ("by-claim.sqlite", "by-source.sqlite", "trigger-requests.sqlite"):
        db_path = _index_dir() / name
        if db_path.exists():
            db_path.unlink()

    reset_connections()

    # Re-initialise (creates fresh empty tables)
    _connect_claim()
    _connect_source()
    _connect_trigger()

    # Rebuild by-claim + by-source from every by-hash entry
    by_hash_dir = _cache_root() / "by-hash"
    if not by_hash_dir.exists():
        return

    for entry_path in sorted(by_hash_dir.glob("*/entry.json")):
        try:
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            _index_put(entry)
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # skip malformed or unreadable entries
