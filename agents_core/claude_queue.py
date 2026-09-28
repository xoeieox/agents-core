#!/usr/bin/env python3
"""ClaudeQueue — file-based queue for `claude -p` subprocess tasks.

Separate from agents_core.gpu.GPUQueue because Qwen-shaped semantics
(preempt / stop_llm_server / model-affinity tiebreak) do not apply to
API-backed shaped-agent subprocesses. See
/srv/lapis/planning/specs/agents-core-claude-queue.md for the design rationale.

Intention-registry parity with GPUQueue (2026-04-24 ops-layer integration,
see /srv/lapis/planning/specs/agents-core-claude-queue-ops-layer-integration.md):
ClaudeQueue carries its own coordinator slot, separate from agents_core.gpu.
The coordinator plumbing is duplicated rather than shared; factor out to
agents_core/_coordinator.py only when a third queue arrives.

Storage:
    /srv/lapis/claude-queue/
      pending/          YAML task files waiting to run
      active/           Up to CLAUDE_QUEUE_WORKERS YAMLs — in-flight
      completed/        Done tasks
      failed/           Failed tasks
      history.jsonl     Append-only event log
      state.json        Snapshot: in-flight list, queue depth

Seat serialization: claim() defers a pending candidate when its seat is
already occupied by an active task. Seats are the explicit set from env
CLAUDE_QUEUE_SERIAL_SEATS (comma-separated; a bare `model` matches any
backend_url, a `backend_url|model` entry matches the exact seat); unset or
empty falls back to the safe default {gravitywell-122b, ninfer-27b}. A
deferred serialized-seat candidate may be overtaken by a lower-priority
other-seat candidate claimed in the same call (candidate-skip semantics).

Usage:
    from agents_core.claude_queue import ClaudeQueue
    q = ClaudeQueue()
    task_id = q._generate_id(slug="fixer-target_x")
    q.submit({
        "task_type": "subprocess",
        "priority": 10,
        "timeout_seconds": 600,
        "submitted_by": "lapis-pm",
        "model": "sonnet",
        "payload": {"command": "...", "spec_path": "..."},
    }, task_id=task_id)
"""

import fcntl
import json
import logging as _logging
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from agents_core.gpu import PACIFIC, Priority, _now_iso, _now_pacific  # noqa: F401
from agents_core.room_paths import room_path

CLAUDE_QUEUE_DIR = room_path("claude_queue")

# ---------------------------------------------------------------------------
# Task coordinator hook (Lapis Ops Layer parity).
#
# Mirrors agents_core.gpu's coordinator extension point but maintains its own
# registration slot. The coordinator contract (project_from_task / manifest /
# compost) is identical — intention_registry satisfies both GPU and Claude
# because the interface is queue-agnostic. See gpu.py's module docstring for
# the full contract.
#
# Registration patterns:
#   - _autoload_coordinators() below attempts to import intention_registry at
#     module load and bind it, so any process importing agents_core.claude_queue
#     gets coordination wired automatically.
#   - Callers may override via claude_queue.register_coordinator(obj); pass
#     None to unregister (tests use this for isolation).
#
# TODO (factor-out): when a third queue arrives, move register_coordinator /
# get_coordinator / _project_* / _manifest_* / _compost_* into
# agents_core/_coordinator.py. Deduplication is deferred per the 2026-04-24
# ops-layer integration spec (§Architecture item 1(a)).
# ---------------------------------------------------------------------------

_coord_log = _logging.getLogger("claude_queue.coordinator")

_coordinator = None


def register_coordinator(coordinator) -> None:
    """Bind a task coordinator. See agents_core.gpu for the required surface.

    Safe to call multiple times — the last registration wins. Pass None to
    unregister (tests can restore no-coordinator mode this way).
    """
    global _coordinator
    _coordinator = coordinator


def get_coordinator():
    """Return the currently registered coordinator, or None."""
    return _coordinator


def _project_intention_for_task(task: dict, task_id: str):
    reg = _coordinator
    if reg is None:
        return None
    try:
        return reg.project_from_task(
            task,
            projected_by=task.get("submitted_by") or "unknown",
            proposed_change=task.get("description") or task.get("task_type"),
            target_heading=(task.get("payload") or {}).get("target_heading"),
            task_id=task_id,
        )
    except Exception as e:
        _coord_log.warning(
            f"coordinator.project_from_task failed for {task.get('id')}: {e}")
        return None


def _manifest_intention_for_task(task: dict) -> None:
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _coordinator
    if reg is None:
        return
    try:
        reg.manifest(intention_id, linked_task_id=task.get("id"))
    except Exception as e:
        _coord_log.warning(
            f"coordinator.manifest failed for {intention_id}: {e}")


def _compost_intention_for_task(task: dict, reason: str) -> None:
    intention_id = task.get("intention_id")
    if not intention_id:
        return
    reg = _coordinator
    if reg is None:
        return
    try:
        reg.compost(intention_id, reason=reason)
    except Exception as e:
        _coord_log.warning(
            f"coordinator.compost failed for {intention_id}: {e}")


def _unlink_orphaned_spec(spec_path: str | None, shared_task_id: str | None) -> None:
    """On match_reinforce / match_manifested, the second submitter's pre-written
    spec JSON is orphaned — its referenced task_id diverges from the shared
    one. Unlink it and log a warning identifying what was dropped.

    Resolves §1 of the ops-layer integration spec. Safe to call with None.
    """
    if not spec_path:
        return
    try:
        p = Path(spec_path)
        if p.exists():
            p.unlink()
            _coord_log.warning(
                f"match fired (shared_task_id={shared_task_id}); "
                f"unlinked orphan spec {spec_path}")
    except OSError as e:
        _coord_log.warning(
            f"failed to unlink orphan spec {spec_path}: {e}")

REQUIRED_FIELDS = {"task_type"}

TASK_DEFAULTS = {
    "priority": Priority.NORMAL,
    "submitted_by": "unknown",
    "timeout_seconds": 300,
    "notify": False,
    "notify_policy": "always",
    "model": None,
    "payload": {},
    "description": None,
    "base_branch": "main",
    "worktree_required": False,
    "status": "pending",
    "started_at": None,
    "completed_at": None,
    "duration_seconds": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Seat serialization guard (per-seat admission cap, claim-time deferral).
#
# Generalizes the former hard-coded gravitywell-122b "Defect 3" guard to an
# explicit set of serialized seats keyed on (backend_url, model), read from
# env CLAUDE_QUEUE_SERIAL_SEATS on every claim() call (never cached).
# See the module docstring's "Seat serialization" note.
# ---------------------------------------------------------------------------

# Grace beyond a job's timeout_seconds before its active YAML is treated as a
# stale orphan (runner restarted mid-job) and stops pinning its seat. Mirrors
# startup_sweep's staleness arithmetic in claude_queue_runner.py online: the
# runner kills a job at its wall-clock timeout, so a genuinely live job can
# never outlive timeout + this grace.
SEAT_STALE_GRACE_S = 120

_DEFAULT_SERIAL_SEATS = ("gravitywell-122b", "ninfer-27b")


def _serial_seat_entries() -> list[tuple[str, str | None, str]]:
    """Parse CLAUDE_QUEUE_SERIAL_SEATS fresh on each call.

    Returns a list of (kind, url, model) tuples: ("exact", url, model) for a
    `backend_url|model` entry (the LAST pipe is the separator — the URL side
    may contain pipes, the model side may not) or ("model", None, model) for
    a bare-model entry. Whitespace stripped; duplicate entries collapse.
    Unset OR empty value -> the default bare-model set (a stray env edit
    cannot silently disable serialization).
    """
    raw = os.environ.get("CLAUDE_QUEUE_SERIAL_SEATS")
    if raw is None or raw.strip() == "":
        return [("model", None, m) for m in _DEFAULT_SERIAL_SEATS]
    seen: set[tuple[str, str | None, str]] = set()
    entries: list[tuple[str, str | None, str]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "|" in tok:
            url, model = tok.rsplit("|", 1)
            url, model = url.strip(), model.strip()
            if not url or not model:
                continue
            e = ("exact", url, model)
        else:
            if not tok:
                continue
            e = ("model", None, tok)
        if e not in seen:
            seen.add(e)
            entries.append(e)
    return entries


def _task_spec_dict(task: dict) -> dict:
    """Read the task's spec JSON as a dict (lane-reality-preflight-v0).
    Same fail-open contract as ``_task_backend_url``: any failure shape
    (missing spec_path, unreadable file, JSON error, non-dict top level)
    returns ``{}``, which the preflight reads as "nothing to gate" and the
    run's own error paths report. Never raises.
    """
    try:
        payload = task.get("payload") or {}
        spec_path = payload.get("spec_path")
        if not spec_path:
            return {}
        data = json.loads(Path(spec_path).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _task_spec_write(task: dict, spec: dict) -> bool:
    """Atomically rewrite the task's spec JSON (lane-reality-preflight-v0,
    Deliverable 3 at the claim point).

    Used ONLY to re-pin a reviewer-family row onto the registry's ACTIVE lane
    (backend_url + served model). Best-effort: any failure returns False and
    the run proceeds against its original spec (the pre-existing behavior) -
    a failed re-pin must never lose the queued work.
    """
    try:
        payload = task.get("payload") or {}
        spec_path = payload.get("spec_path")
        if not spec_path:
            return False
        path = Path(spec_path)
        if not path.parent.is_dir():
            return False
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp",
                                   prefix="lane-spec-")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(spec, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        return True
    except Exception:
        return False


def _task_backend_url(task: dict) -> str | None:
    """Read backend_url from the task's spec JSON (payload.spec_path).
    Fail-open contract: ANY failure shape (missing field, unreadable file,
    JSON parse error, non-dict top level, missing/non-string backend_url)
    returns None. The catch is deliberately broad (`except Exception`) so no
    malformed spec can ever escape claim() into the runner's claim-loop
    catch-all (which pages and exits the whole daemon).
    """
    try:
        payload = task.get("payload") or {}
        spec_path = payload.get("spec_path")
        if not spec_path:
            raise ValueError("no payload.spec_path")
        data = json.loads(Path(spec_path).read_text())
        if not isinstance(data, dict):
            raise ValueError("spec top level is not a dict")
        url = data["backend_url"]
        if not isinstance(url, str):
            raise ValueError("backend_url is not a string")
        return url
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Claim-time doorman lease (attestation-contract-v0, leg 1, D5).
#
# The 2026-09-08 cohort: three claimed GW-backend tasks died
# `gw_not_serving` - the seat parked under claimed work because the run's
# lease is acquired at its FIRST LLM call (ensure_serving), not at claim.
# The idle-park decision runs only while the lease set is empty, so a
# claimed-but-not-yet-calling GW task reads as idle and the seat can park
# under it. D5: on a successful GW-backend claim, best-effort acquire a
# doorman lease for the run; release it on every run exit path.
#
# The wake trap: the lease-acquire path cold-wakes a down seat
# (ensure_serving), so the acquire is preceded by a cheap SERVING PROBE
# that reads the doorman's own view (liveness + seat state), NOT the GW
# seat directly. Doorman unreachable or seat not serving -> SKIP the lease
# (the first LLM call's existing ensure_serving wake path handles that
# case; the death-class signals cover its failure).
#
# Scope: GW-backend tasks ONLY, matched by URL identity against the
# doorman's configured gw_url - the GW SEAT base URL (GW_URL, the same
# env the doorman probes for its own /status "serving" view and the same
# env agents_core.llm uses for its LLM calls - doorman_server.py
# create_app reads GW_URL for the node's gw_url; llm.py GW_URL). The
# task's backend_url is the GW seat LLM endpoint (e.g.
# http://127.0.0.1:8081), NOT the doorman's own port (DOORMAN_SERVER,
# e.g. http://127.0.0.1:8407) - comparing against the doorman's own
# port would fail for every real GW task and the lease would never
# acquire (cycle-2 reviewer finding). The anchor is the FIXED GW_URL env
# (the same env doorman_server.create_app reads for the node's gw_url),
# NOT the task's own backend_url: a self-referential
# backend_url != backend_url gate is a tautology that scopes nothing
# (cycle-3 reviewer finding: the rev-2 call sites passed
# gw_url=backend_url, so a non-GW-backend task still probed + acquired a
# doorman lease). No backend_url / other seat -> no lease.
#
# Best-effort contract: an acquire failure NEVER blocks the claim (the run
# proceeds lease-less; the death-class signals cover the seat-down case).
# ---------------------------------------------------------------------------

# The claim lease covers the claim -> first-LLM-call window (worktree
# setup, ~minutes) plus margin. The queue's hard cap is submit +
# timeout_s + 60 (shaper), so the lease TTL is the task's own timeout +
# 60 - the same arithmetic as the local-fixer supervisor lease, which
# structurally outlives a hard-killed run and needs no renewal.
_CLAIM_LEASE_TTL_MARGIN_S = 60

# The doorman's own liveness endpoint (doorman_server.py /healthz route -
# the doorman process's health, NOT the GW seat's /health).
_DOORMAN_HEALTHZ_PATH = "/healthz"

# The doorman's /status snapshot (doorman_server.py status_snapshot()).
# The shape is FLAT at the top level: {"serving": bool|None,
# "serving_mode": str, "service_stopped": bool, "lease_count": int,
# ...} - the doorman's own cached view of whether its GW seat is serving
# (_is_serving probes {gw_url}/health on the refresh tick and caches the
# result in _cached_serving, which status_snapshot surfaces as
# "serving"). The three-state seat probe vocabulary (down /
# up_registered / up_unverified) belongs to the doorman's separate
# flash-next (:30000) seat axis (the "flashnext" block of the snapshot) -
# NOT the GW seat the claim lease protects.
_DOORMAN_STATUS_PATH = "/status"


def _doorman_probe_serving(base_url: str, timeout: float = 2.0) -> bool | None:
    """The D5 serving probe (the wake trap).

    Reads the doorman's view of the GW seat (liveness via the doorman's
    own /healthz endpoint + the seat state the doorman itself computes -
    the flat top-level "serving" field of its /status snapshot, cached
    from the doorman's own {gw_url}/health probe), NOT the GW seat
    directly - so the claim path never cold-wakes a down seat. Returns:

      True  - doorman up AND its own view says the seat is serving
              (acquire the lease)
      False - doorman up but its own view says the seat is not serving
              (SKIP - no wake)
      None  - doorman unreachable / unreadable (SKIP - the first LLM
              call's existing ensure_serving wake path handles it)

    Never raises: any failure shape degrades to None (fail-open to
    "skip the lease", never to "wake the seat").

    Transport: the EXISTING DoormanClient (doorman_client.py, httpx -
    the same client the LLM path and the lease seams use, incl. its
    bearer-token header) - no second HTTP library in the claim path
    (cycle-2 reviewer finding: the rev-1 requests import was redundant
    with the httpx client the module already routes through; requests
    IS a declared dependency, but the probe rides the shared client).
    """
    try:
        from agents_core.doorman_client import DoormanClient

        client = DoormanClient(base_url=base_url, timeout=timeout)
        try:
            # Liveness first (the doorman's own /healthz endpoint): an
            # unreachable doorman is None (skip) - never a wake.
            health = client.healthz()
            if not isinstance(health, dict):
                return None
            # The seat state: the doorman's own view of its GW seat -
            # the flat top-level "serving" field of the /status snapshot
            # (status_snapshot(): serving = self._cached_serving,
            # refreshed from the doorman's own {gw_url}/health probe on
            # the refresh tick). None (pre-first-refresh) -> not serving
            # (skip, no wake).
            data = client.status()
        finally:
            client.close()
        if not isinstance(data, dict):
            return None
        return data.get("serving") is True
    except Exception:
        return None


def _doorman_lease_acquire(
    base_url: str, work_id: str, ttl_sec: int = 1860,
    reason: str = "claim-lease", timeout: float = 10.0,
) -> bool:
    """Best-effort doorman lease acquire for the claim -> first-call
    window.

    Reuses the EXISTING DoormanClient.acquire() seam (doorman_client.py -
    the same client the LLM path uses, incl. its bearer-token header and
    the /lease/acquire contract) with the existing lease-class semantics:
    role="worker" + the default deferrable class (the LLM path's
    _LEASE_CLASS_DEFAULT; fixers are deferrable by the class-aware
    gate's own docstring). No new endpoint, no hand-built body.

    Returns True when a lease was registered (status "serving"); False on
    any refusal/unreachable/exception (the claim is never blocked). Never
    raises.
    """
    try:
        from agents_core.doorman_client import DoormanClient

        client = DoormanClient(base_url=base_url, timeout=timeout)
        try:
            res = client.acquire(
                "gravitywell", work_id, ttl_sec=ttl_sec, reason=reason,
                role="worker",
            )
        finally:
            client.close()
        return (res or {}).get("status") == "serving"
    except Exception:
        return False


def _doorman_lease_release(base_url: str, work_id: str, timeout: float = 10.0) -> bool:
    """Best-effort doorman lease release (idempotent - unknown work_id is
    a no-op server-side) via the EXISTING DoormanClient.release() seam
    (doorman_client.py). Returns True on success; False on any
    unreachable/exception. Never raises: a lost release degrades to the
    TTL-bounded zombie window (the existing lease contract)."""
    try:
        from agents_core.doorman_client import DoormanClient

        client = DoormanClient(base_url=base_url, timeout=timeout)
        try:
            client.release("gravitywell", work_id)
        finally:
            client.close()
        return True
    except Exception:
        return False


def _acquire_claim_lease(
    *,
    base_url: str,
    work_id: str,
    task_id: str,
    spec_dir=None,
    timeout_s: int | None = None,
    backend_url: str | None = None,
    gw_url: str | None = None,
) -> bool:
    """The D5 claim seam: probe -> acquire for a GW-backend claim.

    Called from the runner's claim loop after a successful GW-backend
    claim (the spec's backend_url must name the doorman-managed GW seat -
    URL identity against the doorman's configured gw_url, not a string
    guess). Returns True when a lease is live for the run.

    Scope gate: the task's backend_url is taken from the
    ``backend_url`` parameter - the _task_backend_url(chosen) result the
    claim() call site already computed (the spec JSON is NOT re-read
    here). The spec_dir/task_id parameters are retained for the
    historical seam shape (the runner's ctx resolution) and are unused
    in the scope gate.

    The ``gw_url`` parameter is the doorman's configured GW SEAT base -
    the FIXED GW_URL env (the same env doorman_server.create_app reads
    for the node's gw_url, and the same env agents_core.llm uses for its
    LLM calls). The task's backend_url is that GW seat endpoint - NOT
    the doorman's own port (DOORMAN_SERVER): the lease itself is acquired
    against the doorman's base (``base_url``), and the SCOPE gate
    compares the task's backend_url against this fixed anchor (cycle-2
    reviewer finding: comparing against the doorman's own port fails for
    every real GW task and the lease never acquired; cycle-3 reviewer
    finding: the anchor must be the fixed GW_URL env, NOT the task's own
    backend_url - a self-referential comparison is a tautology that
    scopes nothing).

    Contract:
      - non-GW-backend task (backend_url absent or != gw_url) -> False,
        no probe, no acquire (scoped, per the spec)
      - probe False (seat down) -> False and the acquire path is NOT
        entered (no wake triggered from the claim path)
      - probe None (doorman unreachable) -> False (skip; the first LLM
        call's ensure_serving wake path handles that case)
      - acquire failure -> False (best-effort: the claim is not blocked)
    """
    # Scope gate: GW-backend tasks only, by URL identity against the
    # FIXED GW_URL anchor (the doorman's configured gw_url - the same
    # env doorman_server.create_app reads for the node's gw_url). The
    # anchor is NOT the task's own backend_url: a self-referential
    # comparison is a tautology that scopes nothing (cycle-3 reviewer
    # finding).
    task_backend = backend_url
    if not isinstance(task_backend, str) or not task_backend:
        return False
    if task_backend.rstrip("/") != (gw_url or "").rstrip("/"):
        return False

    # Serving probe BEFORE acquire (the wake trap): the probe reads the
    # doorman's view (liveness + seat state), NOT the GW seat directly.
    serving = _doorman_probe_serving(base_url)
    if serving is not True:
        # Seat down (False) or doorman unreachable (None): skip the lease.
        # The acquire path is deliberately NOT entered - a blind acquire
        # would cold-wake a down seat (the wake trap).
        return False

    ttl = (
        int(timeout_s) + _CLAIM_LEASE_TTL_MARGIN_S
        if isinstance(timeout_s, (int, float)) and timeout_s > 0
        else 1860
    )
    return _doorman_lease_acquire(
        base_url, work_id, ttl_sec=ttl, reason="claim-lease",
    )


def _release_claim_lease(
    *,
    base_url: str,
    work_id: str,
    task_id: str,
    spec_dir=None,
    backend_url: str | None = None,
    gw_url: str | None = None,
) -> bool:
    """The D5 release seam: release the claim lease on a run exit path.

    Called on EVERY enumerated exit (success, failure, timeout-kill,
    requeue) so the seat is never lease-pinned by a dead run beyond its
    TTL. Requeue releases here and re-acquires on the re-claim (via
    _acquire_claim_lease), so a requeued task is never lease-less in the
    park window. Returns True on a 200; False otherwise (never raises).

    The scope gate mirrors _acquire_claim_lease: the task's backend_url
    is the ``backend_url`` parameter (the _task_backend_url result the
    claim() call site computed) and the identity anchor is the FIXED
    GW_URL env (``gw_url`` - the doorman's configured gw_url, the same
    env doorman_server.create_app reads for the node's gw_url; NOT the
    task's own backend_url - a self-referential comparison is a
    tautology that scopes nothing - and NOT the doorman's own port) -
    the spec JSON is NOT re-read here.
    """
    task_backend = backend_url
    if not isinstance(task_backend, str) or not task_backend:
        return False
    if task_backend.rstrip("/") != (gw_url or "").rstrip("/"):
        return False
    return _doorman_lease_release(base_url, work_id)


def _active_occupies(active_task: dict, now_dt: datetime) -> bool:
    """Does an active task still occupy its seat?

    A task occupies only while `now - started_at < timeout_seconds +
    SEAT_STALE_GRACE_S`. If started_at is absent or unparseable the task is
    treated as FRESH (occupying) — the conservative default for the guard,
    deliberately the OPPOSITE of startup_sweep's eviction direction (which
    treats missing as stale). Normal claims always stamp started_at before
    the atomic active write, so this pin covers hand-placed or partial-state
    YAMLs only.
    """
    started_raw = active_task.get("started_at")
    if not started_raw:
        return True
    try:
        started_dt = datetime.fromisoformat(started_raw)
    except (TypeError, ValueError):
        return True
    timeout = active_task.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        # No usable timeout bound: treat as fresh (conservative).
        return True
    return (now_dt - started_dt).total_seconds() < timeout + SEAT_STALE_GRACE_S


class ClaudeQueue:
    """File-based queue for `claude -p` subprocess tasks."""

    def __init__(self, queue_dir: str | Path = CLAUDE_QUEUE_DIR):
        self.queue_dir = Path(queue_dir)
        self.pending_dir = self.queue_dir / "pending"
        self.active_dir = self.queue_dir / "active"
        self.completed_dir = self.queue_dir / "completed"
        self.failed_dir = self.queue_dir / "failed"
        self.history_path = self.queue_dir / "history.jsonl"
        self.state_path = self.queue_dir / "state.json"
        # In-memory log-dedup sets for the seat-serialization guard: ids of
        # tasks already logged at INFO (deferral) or WARN (unreadable spec).
        # Pruned to pending stems at the top of each claim() scan and dropped
        # on successful claim, so they stay bounded in a long-lived daemon.
        self._logged_deferrals: set[str] = set()
        self._warned_spec_ids: set[str] = set()
        # Lane-reality preflight log/ledger dedup (lane-reality-preflight-v0):
        # (task_id, tag) pairs already surfaced / already written to the
        # lane-status ledger, so a parked row does not re-log or re-append on
        # every 2 s poll. Pruned to pending stems, like the sets above. The
        # park line keeps its OWN set so a row that was earlier deferred by
        # the serialized-seat guard still logs its lane_down park (and vice
        # versa) instead of one guard silencing the other.
        self._logged_lane_tags: set[tuple] = set()
        self._logged_lane_parks: set[str] = set()
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in (self.pending_dir, self.active_dir,
                  self.completed_dir, self.failed_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, slug: str = "") -> str:
        # Single _now_pacific() call so the seconds and microseconds come
        # from the same instant. GPUQueue's _generate_id calls it twice,
        # which can straddle a second boundary (see spec §Interface).
        now = _now_pacific()
        ts = now.strftime("%Y%m%d_%H%M%S")
        usec = now.strftime("%f")[:4]
        clean = (slug or "task").replace(" ", "").replace("-", "").replace("/", "")[:24]
        return f"claude_{ts}_{usec}_{clean}"

    def _append_event(self, event: dict):
        with open(self.history_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

    def _read_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "in_flight": [],
            "queue_depth": 0,
            "tasks_completed_today": 0,
            "last_activity": None,
        }

    def _write_state(self, state: dict):
        state["updated"] = _now_iso()
        fd, tmp = tempfile.mkstemp(
            dir=self.queue_dir, suffix=".tmp", prefix="state-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
                f.write("\n")
            os.replace(tmp, self.state_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_task(self, directory: Path, task: dict):
        path = directory / f"{task['id']}.yaml"
        fd, tmp = tempfile.mkstemp(
            dir=directory, suffix=".tmp", prefix="task-")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(task, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def _read_task(self, path: Path) -> dict | None:
        try:
            return yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError):
            return None

    def _refresh_state(self, state: dict):
        state["queue_depth"] = len(list(self.pending_dir.glob("*.yaml")))
        state["in_flight"] = sorted(
            p.stem for p in self.active_dir.glob("*.yaml")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, task_dict: dict, task_id: str | None = None) -> str:
        """Submit a task. Returns its task_id.

        `task_id` override eliminates a write-back race: the shaper generates
        the id via `_generate_id()`, writes it into the spec JSON file along
        with `worktree_required`/`base_branch`, then submits with that id.
        Without this, the shaper would have to re-write the spec after
        `submit()` returns, and the runner could claim the task in between —
        seeing a spec missing `task_id` and `worktree_required`. See spec
        §Shaper routing.

        Intention-registry behavior (opt-out via
        payload['_ignore_intention_registry']=True), parity with GPUQueue:

        - Novel signature → new task queued; returns the new id.
        - Matches an in-flight intention → no new task queued; caller joins
          the in-flight task by receiving the shared id. The caller's
          pre-written spec (payload['spec_path']) is unlinked because the
          shared task_id points at the first submitter's spec.
        - Matches a recently-manifested intention → no new task; returns the
          prior id. Caller's orphan spec is unlinked.
        """
        missing = REQUIRED_FIELDS - set(task_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        task = dict(TASK_DEFAULTS)
        task.update(task_dict)
        task["id"] = task_id or task.get("id") or self._generate_id(
            slug=task.get("description") or task["task_type"]
        )
        task["submitted_at"] = _now_iso()
        task["status"] = "pending"

        payload = task.get("payload") or {}
        ignore_registry = bool(payload.get("_ignore_intention_registry"))

        projection = None
        if not ignore_registry:
            projection = _project_intention_for_task(task, task["id"])

        if projection is not None and projection.decision in (
            "match_reinforce", "match_manifested"
        ):
            shared = projection.shared_task_id
            self._append_event({
                "event": f"intention_{projection.decision}",
                "id": task["id"],
                "timestamp": task["submitted_at"],
                "task_type": task["task_type"],
                "submitted_by": task.get("submitted_by", "unknown"),
                "intention_id": projection.intention.intention_id,
                "reinforces": projection.intention.reinforces,
                "shared_task_id": shared,
            })
            if shared:
                # §1 resolution (b): unlink the pre-written orphan spec.
                _unlink_orphaned_spec(payload.get("spec_path"), shared)
                return shared
            _coord_log.warning(
                f"intention match for {task['id']} had no shared_task_id; "
                f"falling back to queueing a fresh task"
            )

        # projected / sibling / registry disabled → queue the task normally.
        if projection is not None:
            task["intention_id"] = projection.intention.intention_id

        self._write_task(self.pending_dir, task)

        self._append_event({
            "event": "submitted",
            "id": task["id"],
            "timestamp": task["submitted_at"],
            "task_type": task["task_type"],
            "priority": task["priority"],
            "model": task.get("model"),
            "submitted_by": task.get("submitted_by", "unknown"),
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = task["submitted_at"]
        self._write_state(state)

        return task["id"]

    def _seat_key(self, task: dict, need_url: bool = False) -> tuple[str | None, object]:
        """Return (backend_url, model) for a task.

        backend_url is resolved LAZILY: it stays None unless an exact
        url|model serial entry forces resolution (need_url=True), in which
        case the spec JSON is read via _task_backend_url's fail-open
        contract. The default bare-model set therefore performs zero added
        spec-file I/O. Any failure degrades to (None, model) and logs one
        WARN per task id (never file content or parsed values — the spec
        JSON carries system/prompt instruction text).
        """
        model = task.get("model")
        if not need_url:
            return (None, model)
        tid = task.get("id", "<unknown>")
        url = _task_backend_url(task)
        if url is None and tid not in self._warned_spec_ids:
            self._warned_spec_ids.add(tid)
            payload = task.get("payload") or {}
            spec_path = payload.get("spec_path")
            try:
                # Classify the failure for the WARN without touching content.
                data = json.loads(Path(spec_path).read_text())
                if not isinstance(data, dict):
                    cls = "non-dict-spec"
                elif not isinstance(data.get("backend_url"), str):
                    cls = "bad-backend-url"
                else:
                    cls = "unexpected"
            except Exception as e:  # noqa: BLE001 - classification only
                cls = type(e).__name__
            _coord_log.warning(
                "claude-queue: unreadable spec for %s (spec_path=%s, "
                "failure=%s); seat key degrades to (None, %s)",
                tid, spec_path, cls, model)
        return (url, model)

    def _entry_matches(self, entry: tuple, task: dict) -> bool:
        """Does a serial-seat entry match this task?

        Bare ("model", None, m): matches on model alone. Exact
        ("exact", url, m): matches on model AND the task's spec backend_url
        (fail-open: an unreadable spec has url None and matches no exact
        entry, but still matches its bare entries).
        """
        kind, entry_url, entry_model = entry
        if task.get("model") != entry_model:
            return False
        if kind == "model":
            return True
        url = _task_backend_url(task)
        if url is None:
            tid = task.get("id", "<unknown>")
            if tid not in self._warned_spec_ids:
                self._warned_spec_ids.add(tid)
                _coord_log.warning(
                    "claude-queue: unreadable spec for %s; exact seat "
                    "entry %s|%s cannot match (degraded to model-only)",
                    tid, entry_url[:120], entry_model[:120])
            return False
        return url == entry_url

    def claim(self) -> dict | None:
        """Claim the highest-priority pending task. Returns task dict or None.

        Atomic move from pending/ to active/ via `os.replace` on the source
        YAML. If multiple runner-daemon workers race to claim, only one
        succeeds per file because `_write_task` uses atomic replace; the
        losers find a missing source file and try the next task.

        No model-affinity tiebreak (Qwen-specific), no preempt. Priority ASC,
        then submitted_at ASC — with one serialized-seat exception: a
        deferred high-priority candidate on a busy serialized seat may be
        overtaken by a lower-priority other-seat candidate claimed in the
        same call (candidate-skip semantics; see the module docstring's
        "Seat serialization" note).
        """
        pending_files = sorted(self.pending_dir.glob("*.yaml"))
        if not pending_files:
            return None

        # Prune the log-dedup sets to ids whose YAML is still in pending/ so
        # they neither re-log INFO every poll nor grow unbounded.
        pending_stems = {p.stem for p in pending_files}
        self._logged_deferrals &= pending_stems
        self._warned_spec_ids &= pending_stems
        self._logged_lane_tags = {
            k for k in self._logged_lane_tags
            if isinstance(k, tuple) and k[0] in pending_stems
        }
        self._logged_lane_parks &= pending_stems

        tasks = []
        for p in pending_files:
            t = self._read_task(p)
            if t:
                t["_path"] = str(p)
                tasks.append(t)

        if not tasks:
            return None

        def sort_key(t):
            p = t.get("priority", Priority.NORMAL)
            if not isinstance(p, int):
                p = Priority.NORMAL
            return (p, t.get("submitted_at", ""))

        tasks.sort(key=sort_key)

        serial_entries = _serial_seat_entries()
        now_dt = datetime.fromisoformat(_now_iso())
        active_tasks: list[dict] = []
        for ap in self.active_dir.glob("*.yaml"):
            at = self._read_task(ap)
            if at:
                active_tasks.append(at)

        for chosen in tasks:
            # Seat-serialization guard (generalized from the former
            # hard-coded gravitywell-122b "Defect 3" block): a candidate is
            # deferred iff any serial seat it matches has an active task that
            # also matches the same entry and still occupies its seat.
            # Deferral is a candidate skip — other seats claim normally, so
            # a busy serialized seat never starves them.
            needs_url = any(
                e[0] == "exact" and chosen.get("model") == e[2]
                for e in serial_entries
            )
            # Lazy seat-key resolution: the spec JSON is read only when an
            # exact url|model entry forces it (default bare set -> zero I/O).
            self._seat_key(chosen, need_url=needs_url)
            deferred_entry = None
            deferred_active = None
            for entry in serial_entries:
                if not self._entry_matches(entry, chosen):
                    continue
                for at in active_tasks:
                    if not self._entry_matches(entry, at):
                        continue
                    if not _active_occupies(at, now_dt):
                        continue  # stale orphan YAML: does not pin the seat
                    deferred_entry = entry
                    deferred_active = at
                    break
                if deferred_entry is not None:
                    break
            if deferred_entry is not None:
                _, d_url, d_model = deferred_entry
                tid = chosen.get("id", "<unknown>")
                active_id = deferred_active.get("id", "<unknown>")
                log_line = (
                    "claude-queue: deferring %s - serialized seat %s|%s "
                    "busy (active: %s)"
                    % (tid, (d_url or "-")[:120], d_model[:120], active_id)
                )
                if tid in self._logged_deferrals:
                    _coord_log.debug(log_line)
                else:
                    self._logged_deferrals.add(tid)
                    _coord_log.info(log_line)
                continue

            # -----------------------------------------------------------------
            # Lane-reality preflight at the CLAIM decision point
            # (agents-core-lane-reality-preflight-v0, Deliverable 1/2).
            #
            # Covers rows that were already queued when the lane died - the
            # 2026-09-27 shape (reviewer rows pinned :8081 while the registry
            # reported 8081:down, and the plain fixer that died "local seat
            # returned no text"). A parked candidate is SKIPPED, exactly like
            # the serialized-seat deferral above: it stays pending, other
            # candidates claim normally in this same call (no head-of-line
            # block), and NOTHING is written to completed/ or failed/ - so the
            # PM's fixer/reviewer attempt counters (which count queue records)
            # cannot increment and the gw_seat_occupied ceiling keys stop
            # accumulating. Registry-blind fails open: one probe per target
            # per 10-min tick claims, the rest of the window is skipped.
            # Fail-open on any internal fault - this gate must never become a
            # single point of failure for the whole queue.
            # -----------------------------------------------------------------
            try:
                from agents_core import lane_preflight as _lane_preflight

                _lane_spec = _task_spec_dict(chosen)
                _lane_decision = _lane_preflight.preflight(
                    _lane_spec,
                    probe_key=(
                        str(_lane_spec.get("target_id")
                            or chosen.get("description") or chosen.get("id", "")),
                        chosen.get("model"),
                    ),
                )
                # Record EVERY decision (including a clean fire), deduped per
                # (task, tag): read_lane_status reports the last line per
                # target, so a fire line is what clears a stale lane_down once
                # the lane comes back.
                _lane_key = (chosen.get("id"), _lane_decision.tag)
                if _lane_key not in self._logged_lane_tags:
                    self._logged_lane_tags.add(_lane_key)
                    _lane_preflight.record_lane_tag(
                        _lane_decision,
                        target_id=str(_lane_spec.get("target_id") or ""),
                        task_id=chosen.get("id", ""),
                        agent_type=str(_lane_spec.get("agent_type") or ""),
                        engine=str(_lane_spec.get("engine") or ""),
                    )
                if not _lane_decision.fire:
                    if chosen.get("id") not in self._logged_lane_parks:
                        self._logged_lane_parks.add(chosen.get("id", ""))
                        _coord_log.info(
                            "claude-queue: parking %s - %s (seat=%s port=%s "
                            "reason=%s); stays pending, no failure record",
                            chosen.get("id", "<unknown>"), _lane_decision.tag,
                            _lane_decision.seat, _lane_decision.port,
                            _lane_decision.reason,
                        )
                    continue
                if _lane_decision.reroute_base_url:
                    # Deliverable 3 at the claim point: a reviewer-family row
                    # queued BEFORE the lane died follows the registry's
                    # ACTIVE lane. The spec is rewritten (endpoint + served id
                    # from the registry row) so the run dials the live seat
                    # instead of the dead pin.
                    _task_spec_write(chosen, {
                        **_lane_spec,
                        "backend_url": _lane_decision.reroute_base_url,
                        **({"model": _lane_decision.reroute_model}
                           if _lane_decision.reroute_model else {}),
                    })
                    # Keep the queue row's model in step with the seat it now
                    # dials: the serialized-seat guard keys on task["model"],
                    # so a re-routed row must pin the ACTIVE seat's admission
                    # cap (the flash-next ops cap), not the dead seat's.
                    if _lane_decision.reroute_model:
                        chosen["model"] = _lane_decision.reroute_model
                if _lane_decision.tag:
                    _coord_log.warning(
                        "claude-queue: firing %s on %s (seat=%s port=%s "
                        "reason=%s probe=%s)",
                        chosen.get("id", "<unknown>"), _lane_decision.tag,
                        _lane_decision.seat, _lane_decision.port,
                        _lane_decision.reason, _lane_decision.probe,
                    )
            except Exception as _lane_exc:
                _coord_log.warning(
                    "claude-queue: lane-preflight failed for %s (%s) - firing "
                    "fail-open", chosen.get("id", "<unknown>"),
                    type(_lane_exc).__name__,
                )

            src_path = Path(chosen.pop("_path"))
            chosen["status"] = "running"
            chosen["started_at"] = _now_iso()
            self._write_task(self.active_dir, chosen)
            try:
                src_path.unlink()
            except FileNotFoundError:
                # Another worker claimed it first — revert our active write.
                (self.active_dir / f"{chosen['id']}.yaml").unlink(missing_ok=True)
                continue

            self._append_event({
                "event": "claimed",
                "id": chosen["id"],
                "timestamp": chosen["started_at"],
                "task_type": chosen["task_type"],
                "priority": chosen["priority"],
                "model": chosen.get("model"),
            })
            # The claimed task leaves pending/: drop its dedup entries.
            self._logged_deferrals.discard(chosen["id"])
            self._warned_spec_ids.discard(chosen["id"])
            self._logged_lane_parks.discard(chosen["id"])
            self._logged_lane_tags = {
                k for k in self._logged_lane_tags if k[0] != chosen["id"]
            }

            # D5 (attestation-contract-v0, leg 1): on a successful
            # GW-backend claim, best-effort acquire the doorman claim lease
            # (the claim -> first-LLM-call window). The probe reads the
            # doorman's view (liveness + seat state), NOT the GW seat
            # directly, so a down seat is never cold-woken from the claim
            # path; the run's exit paths release the lease
            # (_release_claim_lease). The scope gate consumes the
            # _task_backend_url(chosen) result computed here (the spec
            # JSON is read exactly once per claim, by _task_backend_url's
            # fail-open contract). Best-effort: any failure shape is
            # swallowed (a lease failure never blocks the claim - the
            # first LLM call's existing ensure_serving wake path handles
            # the seat-down case, and the death-class signals cover its
            # failure).
            try:
                # The doorman's own base (the port the doorman serves
                # the /lease/* endpoints on) - the same env
                # doorman_client.DoormanClient defaults to.
                _claim_doorman_base = os.environ.get(
                    "DOORMAN_SERVER", "http://127.0.0.1:8407",
                )
                _claim_backend_url = _task_backend_url(chosen)
                if _claim_backend_url:
                    # The lease is acquired against the doorman's base
                    # (DOORMAN_SERVER - the doorman's own port), but the
                    # scope gate compares the task's backend_url against
                    # the FIXED GW_URL anchor (the same env
                    # doorman_server.create_app reads for the node's
                    # gw_url - the GW seat base the doorman probes for
                    # its /status "serving" view and agents_core.llm
                    # uses for its LLM calls). The anchor is NOT the
                    # task's own backend_url (a self-referential
                    # comparison is a tautology that scopes nothing -
                    # cycle-3 reviewer finding). The two URLs are
                    # DIFFERENT: the task backend_url is the GW seat
                    # endpoint (e.g. http://127.0.0.1:8081), the
                    # doorman base is its own port (e.g.
                    # http://127.0.0.1:8407).
                    _acquire_claim_lease(
                        base_url=_claim_doorman_base,
                        work_id=chosen["id"],
                        task_id=chosen["id"],
                        backend_url=_claim_backend_url,
                        gw_url=os.environ.get("GW_URL", ""),
                    )
            except Exception:
                pass  # best-effort: a lease failure never blocks the claim

            state = self._read_state()
            self._refresh_state(state)
            state["last_activity"] = chosen["started_at"]
            self._write_state(state)

            return chosen

        return None

    def complete(self, task_id: str, output_path: str | None = None,
                 result_summary: str | None = None,
                 served_model: str | None = None,
                 served_model_provided: bool = False):
        active_path = self.active_dir / f"{task_id}.yaml"
        task = self._read_task(active_path)
        if not task:
            return

        now = _now_iso()
        task["status"] = "completed"
        task["completed_at"] = now
        if output_path:
            task["output_path"] = output_path
        if result_summary:
            task["result_summary"] = result_summary
        # Served-model provenance (local-reviewer-identity-and-provenance-v0,
        # L1.D3): the ACTUAL served model echoed by the run, stamped beside
        # the existing `model:` field (the requested seat alias, kept as-is).
        # A void echo is explicit None (never the seat alias - Erah 2026-09-06
        # explicit-void adjudication). The field is set when the runner
        # observed a PROVENANCE line (served_model_provided) - explicit null
        # on a void echo; it stays absent when no PROVENANCE line was emitted
        # (pre-Leg-1 / non-local engines) - that absence is a non-error.
        if served_model_provided:
            task["served_model"] = served_model

        if task.get("started_at"):
            try:
                dt_start = datetime.fromisoformat(task["started_at"])
                dt_end = datetime.fromisoformat(now)
                task["duration_seconds"] = int(
                    (dt_end - dt_start).total_seconds())
            except (ValueError, TypeError):
                pass

        self._write_task(self.completed_dir, task)
        active_path.unlink(missing_ok=True)

        _manifest_intention_for_task(task)

        self._append_event({
            "event": "completed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "output_path": output_path,
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = now
        today = _now_pacific().strftime("%Y-%m-%d")
        if state.get("_counter_date") != today:
            state["tasks_completed_today"] = 0
            state["_counter_date"] = today
        state["tasks_completed_today"] = state.get(
            "tasks_completed_today", 0) + 1
        self._write_state(state)

    def fail(self, task_id: str, error: str = "Unknown error"):
        active_path = self.active_dir / f"{task_id}.yaml"
        task = self._read_task(active_path)
        if not task:
            # Active file already gone (hand-killed + rm'd, or never existed) —
            # state.json's in_flight may still list this id from before it was
            # removed. Reconcile against the live glob so the ghost doesn't wedge
            # forever (gotcha/claude-queue-orphan-task-clean-recovery-2026-06-22's
            # "second kill not reaped" case). Nothing else to do: no task data to
            # write to failed/, no history event, no intention to compost.
            state = self._read_state()
            self._refresh_state(state)
            state["last_activity"] = _now_iso()
            self._write_state(state)
            return

        now = _now_iso()
        task["status"] = "failed"
        task["completed_at"] = now
        task["error"] = error

        if task.get("started_at"):
            try:
                dt_start = datetime.fromisoformat(task["started_at"])
                dt_end = datetime.fromisoformat(now)
                task["duration_seconds"] = int(
                    (dt_end - dt_start).total_seconds())
            except (ValueError, TypeError):
                pass

        self._write_task(self.failed_dir, task)
        active_path.unlink(missing_ok=True)

        _compost_intention_for_task(task, reason=f"task_failed: {error[:120]}")

        self._append_event({
            "event": "failed",
            "id": task_id,
            "timestamp": now,
            "task_type": task.get("task_type", "unknown"),
            "duration_seconds": task.get("duration_seconds"),
            "error": error[:500],
            "intention_id": task.get("intention_id"),
        })

        state = self._read_state()
        self._refresh_state(state)
        state["last_activity"] = now
        self._write_state(state)

    def cancel(self, task_id: str, reason: str = "") -> bool:
        pending_path = self.pending_dir / f"{task_id}.yaml"
        if not pending_path.exists():
            return False

        task = self._read_task(pending_path)
        pending_path.unlink(missing_ok=True)

        # Compost the intention so it doesn't linger in-flight forever
        # (mirrors gpu.py's 2026-04-23 cancel-compost cascade fix).
        if task:
            _compost_intention_for_task(
                task, reason=f"task_cancelled: {reason or 'no reason given'}"
            )

        self._append_event({
            "event": "cancelled",
            "id": task_id,
            "timestamp": _now_iso(),
            "task_type": task.get("task_type", "unknown") if task else "unknown",
            "reason": reason,
        })

        state = self._read_state()
        self._refresh_state(state)
        self._write_state(state)
        return True

    def lane_status(self, target_id: str | None = None) -> dict:
        """Latest lane-reality tag per target (lane-reality-preflight-v0,
        Deliverable 4).

        ``{target_id: {tag, action, reason, model, port, seat, ts,
        agent_type}}`` read from the lane-status ledger, so a target parked by
        the preflight is visible as ``noop:lane_down:model=...`` in the
        operator-facing queue status rather than only in JSONL. Best-effort:
        an unreadable ledger reads as {} (never raises).
        """
        from agents_core.lane_preflight import read_lane_status

        return read_lane_status(target_id)

    def status(self) -> dict:
        state = self._read_state()
        self._refresh_state(state)
        out = {
            "depth": state["queue_depth"],
            "in_flight": state["in_flight"],
        }
        # Lane-reality surfacing (lane-reality-preflight-v0, Deliverable 4):
        # per-target lane tags ride the queue status the PM already reads, so
        # a parked target's WHY is operator-visible. Only targets whose LATEST
        # decision was a park or a tagged fire are listed - a clean fire
        # (recorded so a recovered lane clears its stale tag) is not
        # interesting to a PM asking "why is this target sitting?", and with
        # nothing to report the dict keeps its pre-preflight shape exactly.
        lane = {
            tid: entry for tid, entry in self.lane_status().items()
            if entry.get("action") != "fire" or entry.get("tag")
        }
        if lane:
            out["lane_status"] = lane
        return out

    def get_pending(self) -> list[dict]:
        tasks = []
        for p in sorted(self.pending_dir.glob("*.yaml")):
            t = self._read_task(p)
            if t:
                tasks.append(t)
        tasks.sort(key=lambda t: (
            t.get("priority", Priority.NORMAL),
            t.get("submitted_at", ""),
        ))
        return tasks

    def get_active(self) -> list[dict]:
        return [t for p in self.active_dir.glob("*.yaml")
                if (t := self._read_task(p))]

    def get_recent_completed(self, limit: int = 10) -> list[dict]:
        files = sorted(
            self.completed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_recent_failed(self, limit: int = 10) -> list[dict]:
        files = sorted(
            self.failed_dir.glob("*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        return [t for p in files if (t := self._read_task(p))]

    def get_history(self, limit: int = 50) -> list[dict]:
        if not self.history_path.exists():
            return []
        events = []
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events[-limit:]

    def cleanup(self, max_age_hours: float = 48) -> int:
        cutoff = _now_pacific() - timedelta(hours=max_age_hours)
        removed = 0
        for p in self.completed_dir.glob("*.yaml"):
            try:
                mtime = datetime.fromtimestamp(
                    p.stat().st_mtime, tz=PACIFIC)
                if mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
        return removed


# ---------------------------------------------------------------------------
# Coordinator autoload (ops-layer v0 bridge).
#
# Unlike agents_core.gpu which uses entry-point discovery via the
# `agents_core.gpu_coordinators` group, ClaudeQueue hard-codes an
# intention_registry probe. Rationale: ops_layer_init (the existing
# entry-point plugin) only registers with GPUQueue's slot, and editing
# ops-layer's pyproject is out of scope for the 2026-04-24 integration.
# When ops-layer (or another coordinator) publishes a ClaudeQueue-aware
# entry point, migrate to entry-point discovery mirroring gpu.py.
# ---------------------------------------------------------------------------

def _autoload_coordinators() -> None:
    try:
        import intention_registry  # noqa: F401
    except ImportError:
        return
    try:
        register_coordinator(intention_registry)
    except Exception as e:  # pragma: no cover — defensive
        _coord_log.warning(f"failed to register intention_registry: {e}")


_autoload_coordinators()


def main():
    if len(sys.argv) < 2:
        print("Usage: claude_queue.py <status|pending|cancel|cleanup|history>",
              file=sys.stderr)
        sys.exit(1)

    q = ClaudeQueue()
    cmd = sys.argv[1]

    if cmd == "status":
        print(json.dumps({
            **q.status(),
            "active": q.get_active(),
            "pending": q.get_pending(),
        }, indent=2, default=str))

    elif cmd == "pending":
        for t in q.get_pending():
            pri = t.get("priority", "?")
            tt = t.get("task_type", "?")
            tid = t.get("id", "?")
            model = t.get("model") or "-"
            sub = t.get("submitted_by", "?")
            print(f"  [{pri:>3}] {tid}  {tt:12s}  model={model}  by={sub}")

    elif cmd == "cancel":
        if len(sys.argv) < 3:
            print("Usage: claude_queue.py cancel <task_id> [reason]",
                  file=sys.stderr)
            sys.exit(1)
        reason = sys.argv[3] if len(sys.argv) > 3 else ""
        if q.cancel(sys.argv[2], reason):
            print(f"Cancelled: {sys.argv[2]}")
        else:
            print(f"Not found in pending: {sys.argv[2]}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "cleanup":
        max_age = 48
        if len(sys.argv) >= 4 and sys.argv[2] == "--max-age":
            max_age = float(sys.argv[3])
        removed = q.cleanup(max_age_hours=max_age)
        print(f"Removed {removed} old completed task(s)")

    elif cmd == "history":
        limit = 20
        if len(sys.argv) >= 4 and sys.argv[2] == "--limit":
            limit = int(sys.argv[3])
        for event in q.get_history(limit):
            print(json.dumps(event))

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
