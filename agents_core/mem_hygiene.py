"""mem_hygiene — Recency guard + bounded quarantine hygiene for mem.db.

Two independent mechanisms live here:

1. RecencyGuard (probe-with-latch) + HygieneVerdict — the PR #46 chassis for
   hygiene passes that *propose* pruning mem.db artifacts. Kept unchanged.

2. The dead-stream quarantine runner (mem-hygiene-automation-v0, Erah
   2026-09-14 inform-don't-ask ruling) — moves machine-state rows from a
   dead producer's prefix into ``memories_quarantine`` (quarantine, not
   delete), in one transaction per run, bounded by a per-run batch cap on
   the scheduled path.

USAGE CONVENTION (RecencyGuard):

Any caller that runs a hygiene pass and calls RecencyGuard.evaluate() MUST
pass the resulting HygieneVerdict to weaver.hygiene_audit.AuditLog.record().
The guard is only as useful as the chassis that records its decisions.
A guard that stays a prune and has no log entry was never there.

The guard does not import from weaver. The convention is enforced by the
caller, not at runtime here. The in-repo provenance writer for the
quarantine runner below is the local ``QuarantineVerdict`` dataclass + the
``decision/`` batch line written via ``mem set`` by the conductor night
node (the caller passes the verdict to weaver's AuditLog AFTER the run —
weaver is a separate package and agents-core must not import it).

LATCH LIFECYCLE (RecencyGuard):

The guard is in PROBE state until a future Facets-ratified, comprehensive
de-crystallization session writes latch_key to mem.db with one of:
  - "absorbed"  — Q2-full subsumed the guard into a broader policy
  - "replaced"  — Q2-full superseded with a different mechanism
  - "forgotten" — Q2-full landed without addressing the guard (process gap)

When latch_state != PROBE the guard stops downgrading. The caller must record
a final latch-state entry to the chassis.

QUARANTINE RUNNER DESIGN (mem-hygiene-automation-v0):

D1 — quarantine, not delete. Candidates move to ``memories_quarantine``
(same columns as ``memories`` + quarantined_at, quarantine_reason, run_id)
via a normal DELETE from ``memories`` so the FTS triggers
(memories_ai/ad/au) keep ``memories_fts`` honest. One transaction per run:
a crash rolls back cleanly, which is what makes the idempotency invariant
true by construction. The quarantine PK is (key, run_id) with
INSERT OR IGNORE, so re-running a crashed run is a clean no-op. Purge-from-
quarantine only after the rollback window (default 14 days).

D2 — dead-stream predicate. A prefix is quarantine-eligible when (a) it is
on the machine-state allowlist (explicit config, NOT auto-inferred), (b)
MAX(updated_at) is older than N days (default 30), measured across BOTH
stores (mem.db + the exhaust twin for exhaust-routed prefixes — the
classifier uses direct SQL on the ``memories`` table, never the
fall-through store API which merges stores and mis-reports last-write for
routed prefixes), and (c) the prefix's ``source`` values map to the
registered-dead-producer set. (c) FAILS CLOSED (ineligible) when a
prefix's sources are unmapped (e.g. hostname sources — exactly
``router/``'s shape today), so a live stream can never be quarantined
without explicit manual registration.

D3 — bounded + logged. The per-run batch cap (default 5,000) governs the
scheduled weekly runs; an above-cap run (the first pass) is legal with
``allow_over_cap=True`` and the mandatory db-file backup is the caller's
job (the conductor node runs it). Dry-run is first-class: the candidate
list is returned and written as a named artifact before any mutation.

D6 — what automation must NOT touch. Any key under an atom-class prefix
(decision/, finding/, correction/, reference/, pattern/, intent/,
project/, candidate/, state/, scar/, ...) is NEVER quarantined — the
allowlist is machine-state-only and the atom-class prefix guard is
enforced in code, not just config. The live router/ stream (incl.
router/lapis-pm/) is protected by the (c) fail-closed registry, not by
this list.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from agents_core import mem_exhaust
from agents_core.mem import MemoryStore

# ---------------------------------------------------------------------------
# RecencyGuard chassis (PR #46) — unchanged
# ---------------------------------------------------------------------------


class LatchState(str, Enum):
    PROBE     = "probe"      # guard is active; no Q2-full ratification seen
    ABSORBED  = "absorbed"   # Q2-full subsumed the guard into a broader policy
    REPLACED  = "replaced"   # Q2-full superseded with a different mechanism
    FORGOTTEN = "forgotten"  # Q2-full landed without addressing the guard (process gap)


@dataclass
class HygieneVerdict:
    artifact_key:      str
    artifact_age_days: float | None    # None if artifact not found in store
    proposed_action:   str             # 'deprecate' | 'merge' | 'keep_both' | 'delete'
    final_action:      str             # same enum; equals proposed_action if fired
    fired:             bool
    staying_reason:    str | None      # populated when fired=False
    latch_state:       LatchState
    sweep_id:          str | None = None


class RecencyGuard:
    """Probe-with-latch recency guard for mem.db hygiene passes.

    Reads artifact age from the store and downgrades destructive proposed
    actions (deprecate, delete) to keep_both when the artifact is younger
    than recency_threshold_days. Safe actions (merge, keep_both) always fire.

    The latch is per-instance cached — read once on first access and stored
    for the lifetime of the instance. One RecencyGuard per sweep pass.
    """

    def __init__(
        self,
        store: MemoryStore,
        recency_threshold_days: float = 30,
        latch_key: str = "decision/decrystallization-full-ratified",
    ):
        self._store = store
        self._recency_threshold_days = recency_threshold_days
        self._latch_key = latch_key
        self._cached_latch_state: LatchState | None = None

    def evaluate(
        self,
        artifact_key: str,
        proposed_action: Literal["deprecate", "merge", "keep_both", "delete"],
        sweep_id: str | None = None,
    ) -> HygieneVerdict:
        """Snap to actual artifact age and decide whether to fire or stay.

        If proposed_action would delete a newer record, downgrades to
        keep_both. Returns HygieneVerdict for the chassis to record.

        Does NOT execute the action — action execution is the caller's
        responsibility.
        """
        record = self._store.get(artifact_key)

        if record is None:
            # Cannot measure age — fire through with artifact_age_days=None.
            return HygieneVerdict(
                artifact_key=artifact_key,
                artifact_age_days=None,
                proposed_action=proposed_action,
                final_action=proposed_action,
                fired=True,
                staying_reason=None,
                latch_state=self.latch_state,
                sweep_id=sweep_id,
            )

        created_at_str = record["created_at"]
        created_at = datetime.fromisoformat(created_at_str)
        # Ensure tz-aware for comparison with now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        artifact_age_days = (
            datetime.now(timezone.utc) - created_at
        ).total_seconds() / 86400

        current_latch = self.latch_state
        destructive = proposed_action in ("deprecate", "delete")

        if (
            current_latch == LatchState.PROBE
            and destructive
            and artifact_age_days < self._recency_threshold_days
        ):
            return HygieneVerdict(
                artifact_key=artifact_key,
                artifact_age_days=artifact_age_days,
                proposed_action=proposed_action,
                final_action="keep_both",
                fired=False,
                staying_reason="recency_guard",
                latch_state=current_latch,
                sweep_id=sweep_id,
            )

        return HygieneVerdict(
            artifact_key=artifact_key,
            artifact_age_days=artifact_age_days,
            proposed_action=proposed_action,
            final_action=proposed_action,
            fired=True,
            staying_reason=None,
            latch_state=current_latch,
            sweep_id=sweep_id,
        )

    @property
    def latch_state(self) -> LatchState:
        """Check mem.db for latch key. Per-instance cached after first read."""
        if self._cached_latch_state is not None:
            return self._cached_latch_state

        record = self._store.get(self._latch_key)
        if record is None:
            state = LatchState.PROBE
        else:
            content = (record.get("content") or "").strip().lower()
            try:
                state = LatchState(content)
            except ValueError:
                # Key exists but content is unrecognized — FORGOTTEN (process gap).
                state = LatchState.FORGOTTEN

        self._cached_latch_state = state
        return state


# ---------------------------------------------------------------------------
# Dead-stream quarantine runner (mem-hygiene-automation-v0)
# ---------------------------------------------------------------------------

# D6 — atom-class prefixes that hygiene automation must NEVER touch, no
# matter what the allowlist config says. The allowlist is machine-state-only
# (elevator/, weather/, test/, arc-scratch-* dated snapshots, ...); this
# prefix set is the load-bearing backstop that keeps an operator mistake in
# the config file from reaching an atom.
#
# NOTE: ``router/`` is NOT a member. It is a LIVE machine-state stream, not
# an atom class, so it does not belong on this list — it is protected by
# the D2(c) fail-closed registry instead (its sources are hostnames —
# brix/starhouse — which can never satisfy the registered-dead test
# without explicit manual registration). D6's must-not-touch list is the
# ATOM classes (decision/, finding/, correction/, ...).
ATOM_CLASS_PREFIXES: tuple[str, ...] = (
    "decision/",
    "finding/",
    "correction/",
    "reference/",
    "pattern/",
    "intent/",
    "project/",
    "candidate/",
    "state/",
    "scar/",
)

# Rollback window for purge-from-quarantine (D1). The 2026-09-14 gate
# recorded the Council divergence (7-day proof-by-silence vs 0-day
# backup-gated purge); the spec text carries 14 days as the v0 value and
# the gate folded that as the standing number until Erah's plate lands.
DEFAULT_ROLLBACK_WINDOW_DAYS = 14

# Per-run batch cap for the SCHEDULED (weekly) path (D3). An above-cap run
# (the first pass) is legal with allow_over_cap=True + the mandatory
# db-file backup; the cap is NOT a hard ceiling on the store, it is the
# bound on the unattended scheduled path.
DEFAULT_BATCH_CAP = 5000

# D2(b) default age threshold (days) for the dead-stream predicate.
DEFAULT_DEAD_STREAM_AGE_DAYS = 30

# D2(b) dual-store predicate — the NAMED pair the spec's dual-store note
# names: elevator/ and weather/. The same producer (ops-primitives) kept
# writing weather/ into the exhaust twin until 2026-08-22, 9 days past the
# mem.db-only "last write" date, so single-store measurement makes the
# predicate non-deterministic across implementation layers.
#
# Deliberate divergence from mem_exhaust.EXHAUST_PREFIXES (reviewer high,
# PR #329 cycle 1): EXHAUST_PREFIXES also carries
# router/gw-review-divergence/ — a LIVE router/ sub-stream whose last-write
# MUST be measured across both stores (the routing table in mem_exhaust.py
# is the write-routing truth), but which is NOT on the dead-producer
# registry (fail-closed, router/ stays live until its producer dies).
# The DUAL_STORE set here is the CLASSIFIER's measurement set (dead-stream
# allowlist members only); the two sets coincide today and must be kept in
# sync when either changes.
DUAL_STORE_PREFIXES: tuple[str, ...] = ("elevator/", "weather/")

# Named default for the dry-run candidate artifact (D3). The conductor
# node overrides this per run with <date>-<runid>.
DEFAULT_CANDIDATE_ARTIFACT_DIR = Path("/data/slots")

QUARANTINE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories_quarantine (
    key               TEXT NOT NULL,
    content           TEXT NOT NULL,
    tags              TEXT DEFAULT '',
    source            TEXT DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    quarantined_at    TEXT NOT NULL,
    quarantine_reason TEXT DEFAULT '',
    run_id            TEXT NOT NULL,
    PRIMARY KEY (key, run_id)
);
"""

def is_atom_class_key(key: str) -> bool:
    """True if `key` sits under an atom-class prefix (D6 must-not-touch)."""
    return key.startswith(ATOM_CLASS_PREFIXES)


def _like_escape(prefix: str) -> str:
    """Escape SQL-LIKE wildcards so prefix queries are exact-prefix."""
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass
class DeadProducerRegistry:
    """D2(c) — prefix -> exact source-value-set mapping.

    Materializes as a versioned config file (agents_core/config/
    dead_producers.json), NOT auto-inferred and NOT in code. A prefix whose
    live sources are hostnames or otherwise unmapped FAILS CLOSED
    (ineligible) — which is exactly ``router/``'s shape today (sources =
    brix/starhouse hostnames), so router/ can never be quarantined without
    explicit manual registration.

    Interaction with the D4 write guard (explicit, no gap): the guard
    rejects writes whose EXPLICIT source matches the test-provenance
    pattern (``_TEST_SOURCE_PATTERN`` in agents_core.mem); the registry
    here maps exact production source values to dead producers. The two
    operate on the same ``source`` column but at different layers — the
    guard at write time (rejecting test harnesses), the registry at
    classify time (proving a producer is dead) — and a source can be
    registered-dead while also matching the test pattern (e.g. a
    ``test/``-family source); nothing depends on the two being disjoint.
    """

    version: int
    # prefix (with trailing slash) -> set of exact source values that prove
    # the producer behind it is dead. A prefix whose observed sources are a
    # SUPERSET of its registered set is NOT dead (new producer appeared).
    dead_sources: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "DeadProducerRegistry":
        raw = json.loads(Path(path).read_text())
        return cls(
            version=int(raw["version"]),
            dead_sources={k: set(v) for k, v in raw["dead_sources"].items()},
        )

    def registered_sources(self, prefix: str) -> set[str]:
        return self.dead_sources.get(prefix, set())

    def is_dead(self, prefix: str, observed_sources: set[str]) -> bool:
        """(c) — True only when observed sources are a non-empty SUBSET of
        the registered dead set. Unmapped (empty registration) or any
        unregistered source observed -> False (fail closed)."""
        registered = self.registered_sources(prefix)
        if not registered:
            return False
        return bool(observed_sources) and observed_sources <= registered


@dataclass
class HygieneConfig:
    """The machine-state allowlist + thresholds (D2(a), D3).

    Loaded from an explicit named config file via the MEM_HYGIENE_CONFIG
    env var (NOT auto-inferred). ``allowlist`` is the machine-state-only
    prefix set; atom-class prefixes are excluded by is_atom_class_key()
    regardless of what the file says (D6 backstop).
    """

    registry: DeadProducerRegistry
    allowlist: tuple[str, ...]
    dead_stream_age_days: int = DEFAULT_DEAD_STREAM_AGE_DAYS
    batch_cap: int = DEFAULT_BATCH_CAP
    rollback_window_days: int = DEFAULT_ROLLBACK_WINDOW_DAYS

    @classmethod
    def load(cls, path: Path | str) -> "HygieneConfig":
        raw = json.loads(Path(path).read_text())
        registry_path = raw["registry"]
        # A relative registry path resolves next to the config file (the
        # repo ships both files in agents_core/config/); an absolute path
        # is used as-is (the BRIX deploy copies the pair to /srv/agents/
        # agents_core/config/).
        registry = DeadProducerRegistry.load(
            Path(path).parent / registry_path
            if not Path(registry_path).is_absolute()
            else Path(registry_path)
        )
        return cls(
            registry=registry,
            allowlist=tuple(raw["allowlist"]),
            dead_stream_age_days=int(raw.get("dead_stream_age_days", DEFAULT_DEAD_STREAM_AGE_DAYS)),
            batch_cap=int(raw.get("batch_cap", DEFAULT_BATCH_CAP)),
            rollback_window_days=int(raw.get("rollback_window_days", DEFAULT_ROLLBACK_WINDOW_DAYS)),
        )

    @classmethod
    def from_env(cls) -> "HygieneConfig":
        """Load from MEM_HYGIENE_CONFIG (named config file path). Raises
        RuntimeError when unset — the allowlist is explicit config, not
        auto-inference."""
        path = os.environ.get("MEM_HYGIENE_CONFIG", "").strip()
        if not path:
            raise RuntimeError(
                "MEM_HYGIENE_CONFIG is not set — the machine-state "
                "allowlist is explicit config, never auto-inferred"
            )
        return cls.load(path)


@dataclass
class CandidatePrefix:
    """One prefix's classification result (D2)."""

    prefix: str
    eligible: bool
    reason: str
    row_count: int = 0
    last_write: str | None = None
    observed_sources: set[str] = field(default_factory=set)


@dataclass
class QuarantineVerdict:
    """Per-run provenance record (the in-repo writer; the conductor node
    passes this to weaver's AuditLog after the run)."""

    run_id: str
    ts_utc: str
    mode: str                      # 'dry-run' | 'run'
    eligible_prefixes: list[CandidatePrefix] = field(default_factory=list)
    ineligible_prefixes: list[CandidatePrefix] = field(default_factory=list)
    candidate_count: int = 0
    candidate_artifact: str | None = None
    quarantined: int = 0
    already_quarantined: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    fts_integrity_ok: bool | None = None
    batch_cap: int = DEFAULT_BATCH_CAP
    allow_over_cap: bool = False
    db_row_count_after: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ts_utc": self.ts_utc,
            "mode": self.mode,
            "eligible_prefixes": [
                {
                    "prefix": p.prefix,
                    "reason": p.reason,
                    "row_count": p.row_count,
                    "last_write": p.last_write,
                    "observed_sources": sorted(p.observed_sources),
                }
                for p in self.eligible_prefixes
            ],
            "ineligible_prefixes": [
                {
                    "prefix": p.prefix,
                    "reason": p.reason,
                    "row_count": p.row_count,
                    "last_write": p.last_write,
                    "observed_sources": sorted(p.observed_sources),
                }
                for p in self.ineligible_prefixes
            ],
            "candidate_count": self.candidate_count,
            "candidate_artifact": self.candidate_artifact,
            "quarantined": self.quarantined,
            "already_quarantined": self.already_quarantined,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "fts_integrity_ok": self.fts_integrity_ok,
            "batch_cap": self.batch_cap,
            "allow_over_cap": self.allow_over_cap,
            "db_row_count_after": self.db_row_count_after,
        }


class HygieneAborted(RuntimeError):
    """A run aborted before or after mutation (count mismatch, cap
    breach, FTS integrity failure). The transaction is rolled back."""


class MemHygieneRunner:
    """Bounded, transactional dead-stream quarantine runner.

    All ``memories`` mutations go through trigger-covered DML (the
    quarantine DELETE and the restore INSERT..SELECT both fire
    memories_ai/ad/au); no raw FTS writes. One transaction per run.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: HygieneConfig,
        run_id: str | None = None,
        artifact_dir: Path = DEFAULT_CANDIDATE_ARTIFACT_DIR,
    ):
        self._store = store
        self._config = config
        self._artifact_dir = artifact_dir
        now = datetime.now(timezone.utc)
        self.run_id = run_id or f"hyg-{now.strftime('%Y%m%dT%H%M%SZ')}-{now.microsecond:06d}"
        self._schema_ready = False

    def ensure_schema(self) -> None:
        """Bootstrap the quarantine table ONCE per runner instance.

        Reviewer medium (PR #329 cycle 1): the schema bootstrap used to run
        in __init__ and thus re-executed the (idempotent) CREATE TABLE IF
        NOT EXISTS + commit on every construction — the mem-server builds a
        runner per request. It is now lazy: called at the top of each
        run/restore/ageout mutation, executed at most once per instance,
        and kept OUTSIDE the run transaction (the spec's one-transaction-
        per-run invariant covers the run, not the bootstrap).
        """
        if self._schema_ready:
            return
        with self._store._lock:
            self._store._conn.executescript(QUARANTINE_SCHEMA)
            self._store._conn.commit()
        self._schema_ready = True

    # ------------------------------------------------------------------
    # D2 — dead-stream classifier (direct SQL, both stores)
    # ------------------------------------------------------------------

    def _max_updated(self, conn: sqlite3.Connection, prefix: str) -> str | None:
        row = conn.execute(
            "SELECT MAX(updated_at) FROM memories WHERE key LIKE ? ESCAPE '\\'",
            (_like_escape(prefix) + "%",),
        ).fetchone()
        return row[0] if row and row[0] else None

    def _exhaust_max_updated(self, prefix: str) -> str | None:
        """(b) dual-store: the exhaust twin's MAX(updated_at) for `prefix`.

        Read through the store's OWN ExhaustStore instance (its connection
        AND its lock), never a fresh sqlite3.connect() on the exhaust path:
        a raw second connection bypasses the sibling store's lock and, under
        WAL mode with a live writer, could observe a torn WAL state (reviewer
        high, PR #329 cycle 1). The sibling store is still direct SQL on the
        exhaust file's `memories` table — never the fall-through store API.

        Lock ordering (reviewer low, PR #331 cycle 1): this acquires
        es._lock while classify_prefixes already holds self._store._lock
        (store lock -> exhaust lock). The ordering is documented here so
        any future code path that acquires them in reverse (exhaust lock
        -> store lock) is a known deadlock hazard; no such reverse path
        exists in the codebase today. Both locks are RLocks, so the
        hazard is cross-path, not same-thread-reentrant.
        """
        es = self._store._exhaust_store()
        with es._lock:
            return self._max_updated(es._conn, prefix)

    def _prefix_sources(self, conn: sqlite3.Connection, prefix: str) -> set[str]:
        rows = conn.execute(
            "SELECT DISTINCT source FROM memories WHERE key LIKE ? ESCAPE '\\'",
            (_like_escape(prefix) + "%",),
        ).fetchall()
        return {r[0] for r in rows if r[0]}

    def _prefix_count(self, conn: sqlite3.Connection, prefix: str) -> int:
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE key LIKE ? ESCAPE '\\'",
            (_like_escape(prefix) + "%",),
        ).fetchone()
        return int(row[0]) if row else 0

    def classify_prefixes(self, now: datetime | None = None) -> list[CandidatePrefix]:
        """Classify every allowlisted prefix against the D2 predicate.

        (b) measures MAX(updated_at) across BOTH stores for exhaust-routed
        prefixes (the exhaust twin is a separate file with its own
        ``memories`` table — direct SQL, never the fall-through API).
        """
        now = now or datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=self._config.dead_stream_age_days)).isoformat()
        results: list[CandidatePrefix] = []

        with self._store._lock:
            conn = self._store._conn
            for prefix in self._config.allowlist:
                if is_atom_class_key(prefix):
                    # D6 backstop: an atom-class prefix is NEVER eligible,
                    # even if a config file names it.
                    results.append(CandidatePrefix(
                        prefix=prefix, eligible=False,
                        reason="atom-class prefix (D6 must-not-touch)",
                    ))
                    continue

                count = self._prefix_count(conn, prefix)
                last_write = self._max_updated(conn, prefix)
                # (b) dual-store: for exhaust-routed prefixes the age math
                # measures MAX(updated_at) across BOTH stores — the
                # exhaust twin is a separate file with its own memories
                # table (direct SQL, never the fall-through API).
                if prefix in DUAL_STORE_PREFIXES:
                    # (b) dual-store: measure MAX(updated_at) across BOTH
                    # stores. The exhaust twin is a separate file with its
                    # own memories table — read it through the store's own
                    # ExhaustStore instance + lock (no raw second
                    # connection; see _exhaust_max_updated).
                    e_last = self._exhaust_max_updated(prefix)
                    if e_last and (last_write is None or e_last > last_write):
                        last_write = e_last

                if count == 0:
                    results.append(CandidatePrefix(
                        prefix=prefix, eligible=False,
                        reason="no rows in mem.db",
                    ))
                    continue

                sources = self._prefix_sources(conn, prefix)
                if not self._config.registry.is_dead(prefix, sources):
                    results.append(CandidatePrefix(
                        prefix=prefix, eligible=False,
                        reason=f"sources {sorted(sources)} not registered-dead "
                               f"(fail-closed)",
                        row_count=count, last_write=last_write,
                        observed_sources=sources,
                    ))
                    continue

                if last_write is None or last_write >= cutoff:
                    results.append(CandidatePrefix(
                        prefix=prefix, eligible=False,
                        reason=f"last write {last_write} within "
                               f"{self._config.dead_stream_age_days}d window",
                        row_count=count, last_write=last_write,
                        observed_sources=sources,
                    ))
                    continue

                results.append(CandidatePrefix(
                    prefix=prefix, eligible=True,
                    reason=f"dead stream: sources {sorted(sources)} registered, "
                           f"last write {last_write} older than "
                           f"{self._config.dead_stream_age_days}d",
                    row_count=count, last_write=last_write,
                    observed_sources=sources,
                ))
        return results

    # ------------------------------------------------------------------
    # D3 — dry-run candidate artifact
    # ------------------------------------------------------------------

    def write_candidate_artifact(
        self,
        candidates: list[dict],
        eligible: list[CandidatePrefix],
        mode: str = "dry-run",
    ) -> str:
        """Write the candidate list as a named artifact (atomic).
        Returns the path (recorded in the decision/ line).

        The artifact is the DRY-RUN snapshot of the candidate set even
        when it is written ahead of a real run (the night node writes it
        BEFORE any mutation and then quarantines the same enumeration) —
        but the ``mode`` field records the CALLER's run mode so a real
        run's artifact does not claim it was a dry-run (reviewer medium,
        PR #331 cycle 1: the field said 'dry-run' on real runs)."""
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._artifact_dir / f"mem-hygiene-candidates-{date}-{self.run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "eligible_prefixes": [p.prefix for p in eligible],
            "candidate_count": len(candidates),
            "candidates": candidates,
        }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".mem-hyg-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.rename(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return str(path)

    # ------------------------------------------------------------------
    # D1 — quarantine / restore / ageout (one transaction per run)
    # ------------------------------------------------------------------

    def _fts_integrity_ok(self, conn: sqlite3.Connection) -> bool:
        """FTS integrity probe (the halt condition the Council agreed on).

        A corrupted external-content FTS index raises
        ``sqlite3.DatabaseError`` (``database disk image is malformed``),
        not ``OperationalError`` — both are caught so the halt condition
        fires on either failure mode.

        The integrity command returns ZERO rows on a healthy index (it
        only raises on corruption), so the probe is the exception path,
        not the row count: no exception = OK. (A row-count check would
        make this probe always-False and every real run would abort —
        the bug this wording exists to keep from regressing.)
        """
        try:
            conn.execute(
                "SELECT 'ok' FROM memories_fts WHERE memories_fts='integrity'"
            ).fetchall()
            return True
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return False

    def _eligible_keys(self, conn: sqlite3.Connection, eligible: list[CandidatePrefix]) -> list[str]:
        """Enumerate the candidate keys across all eligible prefixes.

        DE-DUPLICATED (reviewer high, PR #329 cycle 1): the shipped
        allowlist entries are disjoint prefixes, but a key can match two
        allowlist prefixes in general (e.g. `elevator/` and a broader
        `elevator` entry). A duplicate key would be enumerated twice but
        DELETEd only once, tripping the count-mismatch guard with a false
        abort. The key is the PRIMARY KEY of `memories`, so a key is a
        single row no matter how many prefixes match it — dedup is exact.
        """
        seen: set[str] = set()
        keys: list[str] = []
        for p in eligible:
            rows = conn.execute(
                "SELECT key FROM memories WHERE key LIKE ? ESCAPE '\\' ORDER BY key",
                (_like_escape(p.prefix) + "%",),
            ).fetchall()
            for r in rows:
                if r[0] not in seen:
                    seen.add(r[0])
                    keys.append(r[0])
        return keys

    def list_candidates(self, now: datetime | None = None) -> tuple[list[dict], list[CandidatePrefix]]:
        """Dry-run enumeration: (candidate rows, prefix classifications).
        Read-only — no mutation, no transaction."""
        prefixes = self.classify_prefixes(now=now)
        eligible = [p for p in prefixes if p.eligible]
        with self._store._lock:
            conn = self._store._conn
            keys = self._eligible_keys(conn, eligible)
            rows: list[dict] = []
            if keys:
                # Chunk the IN-list (sqlite bound-parameter cap is 999).
                for i in range(0, len(keys), 500):
                    chunk = keys[i:i + 500]
                    qmarks = ",".join("?" for _ in chunk)
                    for r in conn.execute(
                        f"SELECT key, content, tags, source, created_at, updated_at "
                        f"FROM memories WHERE key IN ({qmarks}) ORDER BY key",
                        chunk,
                    ).fetchall():
                        rows.append(dict(r))
        return rows, prefixes

    def run_quarantine(
        self,
        *,
        dry_run: bool = False,
        allow_over_cap: bool = False,
        candidates: list[dict] | None = None,
        prefixes: list[CandidatePrefix] | None = None,
    ) -> QuarantineVerdict:
        """One hygiene run (D1 + D3). dry_run=True writes the candidate
        artifact and returns without mutating. Otherwise: one transaction
        — INSERT OR IGNORE into memories_quarantine + DELETE from
        memories — with the count-mismatch abort and the FTS integrity
        halt. Raises HygieneAborted on any abort condition.

        ``candidates``/``prefixes`` (reviewer medium, PR #329 cycle 1): a
        caller that already ran the dry-run enumeration (the night node
        writes the candidate artifact BEFORE any mutation) may pass its
        result through so the mutation operates on the SAME enumeration —
        no double-classification, and the store cannot drift between the
        artifact and the quarantine. When omitted, this method enumerates
        itself (the server endpoints keep that shape)."""
        now = datetime.now(timezone.utc)
        verdict = QuarantineVerdict(
            run_id=self.run_id,
            ts_utc=now.isoformat(),
            mode="dry-run" if dry_run else "run",
            batch_cap=self._config.batch_cap,
            allow_over_cap=allow_over_cap,
        )
        if candidates is None or prefixes is None:
            candidates, prefixes = self.list_candidates(now=now)
        eligible = [p for p in prefixes if p.eligible]
        verdict.eligible_prefixes = eligible
        verdict.ineligible_prefixes = [p for p in prefixes if not p.eligible]
        verdict.candidate_count = len(candidates)

        if dry_run:
            verdict.candidate_artifact = self.write_candidate_artifact(candidates, eligible)
            with self._store._lock:
                verdict.db_row_count_after = int(
                    self._store._conn.execute(
                        "SELECT COUNT(*) FROM memories"
                    ).fetchone()[0]
                )
            return verdict

        # Cap check (D3): the scheduled path is bounded; above-cap runs are
        # legal only with allow_over_cap (the first pass, backup-mandatory).
        if len(candidates) > self._config.batch_cap and not allow_over_cap:
            verdict.aborted = True
            verdict.abort_reason = (
                f"candidate count {len(candidates)} exceeds batch cap "
                f"{self._config.batch_cap} and allow_over_cap is not set"
            )
            raise HygieneAborted(verdict.abort_reason)

        if not candidates:
            # Nothing eligible — still a clean, logged run.
            with self._store._lock:
                verdict.db_row_count_after = int(
                    self._store._conn.execute(
                        "SELECT COUNT(*) FROM memories"
                    ).fetchone()[0]
                )
            return verdict

        reason = "dead-stream quarantine (mem-hygiene-automation-v0)"
        self.ensure_schema()
        with self._store._lock:
            conn = self._store._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                keys = [c["key"] for c in candidates]
                # 1. Stage into the quarantine table (PK (key, run_id),
                #    INSERT OR IGNORE — a crashed re-run is a clean no-op).
                #    Chunked IN-lists: sqlite's bound-parameter cap is 999
                #    (the first pass is ~14,043 keys); same 500-key chunk
                #    size list_candidates uses. One transaction spans all
                #    chunks — a crash mid-batch still rolls back cleanly.
                for i in range(0, len(keys), 500):
                    chunk = keys[i:i + 500]
                    qmarks = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"""
                        INSERT OR IGNORE INTO memories_quarantine (
                            key, content, tags, source, created_at, updated_at,
                            quarantined_at, quarantine_reason, run_id
                        )
                        SELECT key, content, tags, source, created_at, updated_at,
                               ?, ?, ?
                        FROM memories
                        WHERE key IN ({qmarks})
                        """,
                        [now.isoformat(), reason, self.run_id] + chunk,
                    )
                # 2. Delete from memories (trigger-covered: memories_ad
                #    keeps memories_fts honest). Chunked for the same
                #    bound-parameter reason; the DELETE rowcount is the
                #    load-bearing count-mismatch guard (step 3) — a
                #    derived "already = candidates - quarantined" would
                #    be a tautology that never fires.
                deleted = 0
                for i in range(0, len(keys), 500):
                    chunk = keys[i:i + 500]
                    qmarks = ",".join("?" for _ in chunk)
                    cur = conn.execute(
                        f"DELETE FROM memories WHERE key IN ({qmarks})",
                        chunk,
                    )
                    deleted += cur.rowcount
                # 3. Count-mismatch abort (D3): the DELETE rowcount must
                #    equal the candidate count — the candidates were
                #    enumerated from `memories` moments ago and no other
                #    writer holds this connection (single-connection
                #    store, BEGIN IMMEDIATE), so a mismatch means the
                #    store drifted mid-run and the batch must roll back.
                if deleted != len(candidates):
                    raise HygieneAborted(
                        f"count mismatch: deleted {deleted} != "
                        f"{len(candidates)} candidates"
                    )
                # 4. FTS integrity halt (the Council's agreed halt
                #    condition): a failure rolls back the whole batch.
                if not self._fts_integrity_ok(conn):
                    raise HygieneAborted("FTS integrity check failed post-mutation")
                conn.execute("COMMIT")
            except HygieneAborted as exc:
                conn.execute("ROLLBACK")
                verdict.aborted = True
                verdict.abort_reason = str(exc)
                raise
            except BaseException as exc:
                conn.execute("ROLLBACK")
                verdict.aborted = True
                verdict.abort_reason = f"transaction rolled back: {exc}"
                raise

        # 5. Post-commit provenance counts: quarantined = rows this run_id
        #    staged (== deleted, the mismatch guard already enforced
        #    equality with the candidate count); already_quarantined =
        #    rows for the SAME keys from an EARLIER run_id still sitting
        #    in memories_quarantine (the INSERT OR IGNORE path — a
        #    crashed re-run with the same run_id is a clean no-op by
        #    construction).
        #
        # Reviewer medium (PR #331 cycle 1): restore_prefix DRAINS the
        # quarantine table for the restored prefix, so after a
        # restore-then-run cycle the earlier run's rows are gone and this
        # count is 0 for that path. The count is informational only (the
        # verdict field, not a guard); it is NOT a restore-then-run
        # detector.
        with self._store._lock:
            conn = self._store._conn
            quarantined = int(conn.execute(
                "SELECT COUNT(*) FROM memories_quarantine WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()[0])
            already = 0
            if keys:
                for i in range(0, len(keys), 500):
                    chunk = keys[i:i + 500]
                    qmarks = ",".join("?" for _ in chunk)
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM memories_quarantine "
                        f"WHERE key IN ({qmarks}) AND run_id != ?",
                        chunk + [self.run_id],
                    ).fetchone()
                    already += int(row[0])
        verdict.quarantined = quarantined
        verdict.already_quarantined = already
        with self._store._lock:
            verdict.fts_integrity_ok = self._fts_integrity_ok(self._store._conn)
            verdict.db_row_count_after = int(
                self._store._conn.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()[0]
            )
        return verdict

    def restore_prefix(self, prefix: str) -> int:
        """Restore every quarantined row under `prefix` back into
        ``memories`` (D-2: INSERT..SELECT + DELETE pair — a literal
        cross-table UPDATE is not executable SQL). The memories_ai
        trigger re-indexes FTS. One transaction. Returns the restored
        row count.

        Named exception (panel F6): this is a direct store write from the
        maintenance path; it bypasses the D4 set-guard by design and is
        logged by the caller in the run's provenance line.
        """
        if is_atom_class_key(prefix):
            raise ValueError(f"refusing to restore atom-class prefix {prefix!r}")
        pattern = _like_escape(prefix) + "%"
        self.ensure_schema()
        with self._store._lock:
            conn = self._store._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memories (
                        key, content, tags, source, created_at, updated_at
                    )
                    SELECT key, content, tags, source, created_at, updated_at
                    FROM memories_quarantine
                    WHERE key LIKE ? ESCAPE '\\'
                    """,
                    (pattern,),
                )
                cur = conn.execute(
                    "DELETE FROM memories_quarantine WHERE key LIKE ? ESCAPE '\\'",
                    (pattern,),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return cur.rowcount

    def ageout(self, window_days: int | None = None) -> int:
        """Purge quarantined rows older than the rollback window (D1:
        purge-from-quarantine only after the window). Returns the purged
        row count. One transaction."""
        window = window_days if window_days is not None else self._config.rollback_window_days
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=window)
        ).isoformat()
        self.ensure_schema()
        with self._store._lock:
            conn = self._store._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "DELETE FROM memories_quarantine WHERE quarantined_at < ?",
                    (cutoff,),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return cur.rowcount

    def quarantine_stats(self) -> dict[str, Any]:
        """``mem hygiene list`` surface: quarantine table census."""
        self.ensure_schema()
        with self._store._lock:
            conn = self._store._conn
            by_run = [
                dict(r) for r in conn.execute(
                    """
                    SELECT run_id, MIN(quarantined_at) AS quarantined_at,
                           COUNT(*) AS rows
                    FROM memories_quarantine
                    GROUP BY run_id
                    ORDER BY MIN(quarantined_at) DESC
                    """
                ).fetchall()
            ]
            by_prefix = [
                dict(r) for r in conn.execute(
                    """
                    SELECT substr(key, 1, instr(key, '/') - 1) || '/' AS prefix,
                           COUNT(*) AS rows
                    FROM memories_quarantine
                    WHERE instr(key, '/') > 0
                    GROUP BY prefix
                    ORDER BY rows DESC
                    """
                ).fetchall()
            ]
            total = int(conn.execute(
                "SELECT COUNT(*) FROM memories_quarantine"
            ).fetchone()[0])
        return {"total": total, "by_run": by_run, "by_prefix": by_prefix}
