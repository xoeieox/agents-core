"""Core orchestration logic for shared deliberation service.

Wraps Facets (subprocess) + Mirror Council, runs them concurrently,
unifies results into a DeliberationEnvelope.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from agents_core.shared_deliberation.envelope import DeliberationEnvelope, DeliberationRequest
from agents_core.room_paths import room_path

log = logging.getLogger("shared-deliberation")

_COUNCIL_STATION_ID = "council/worker-fast-fail"
_GROUNDING_STATION_ID = "shared-deliberation/facets-grounding-denied"

# Root under which every repo's persistent local clone lives, and the suffix on the
# clone dir name. Single constant so a future multi-host port has one place to change --
# this module already hardcodes /srv/git/facets-working elsewhere (see facets_repo below),
# so this does not make the module portable on its own.
_GROUNDING_CLONE_ROOT = "/srv/git"
_GROUNDING_CLONE_SUFFIX = "-working"


class GroundingHandoffError(Exception):
    """Raised when grounding_result_file is set but the file is absent or empty.

    A producer that sets this field has promised a pre-computed grounding result.
    Silently falling back to auto-grounding would mask the contract violation and
    make any swarm:<id> provenance label a lie. Hard-fail instead.
    """


def _escalate_council_fast_fail(run_id: str, last_heartbeat, reason: str) -> None:
    """Fire the council fast-fail repair station (Leg 1 of repair-expert-v0).

    Imported lazily so the repair_station module is not loaded on every
    orchestrator import — only on an actual fast-fail event.
    """
    try:
        from agents_core.repair_station import escalate, Tier, first
        escalate(
            station_id=_COUNCIL_STATION_ID,
            stable_pointer="agents_core/shared_deliberation/orchestrator.py",
            error_signal={
                "run_id": run_id,
                "last_heartbeat": last_heartbeat,
                "reason": reason,
            },
            author_intent="council worker died/stalled during deliberation",
            escalation_policy=first(),
            tier=Tier.HIGH,
            owning_module="agents_core.shared_deliberation.orchestrator",
        )
    except Exception:
        log.exception("repair-station escalation failed for council fast-fail — suppressed")


def _resolve_grounding_target(context: dict) -> tuple[Optional[str], str, dict]:
    """Resolve a trustworthy, read-only grounding target for Facets codebase verification.

    Grounds against a detached worktree of the EXISTING LOCAL CLONE's `origin/main` ref
    (Erah ruling, 2026-08-01) -- purely local, no `git clone`, no `git fetch`, no
    hardcoded remote URL. `origin/main` in a local clone is only as current as that
    clone's last fetch; the resolved sha is captured in provenance so a reader can tell
    what was actually verified against, rather than assuming it is current.

    Returns (path, skip_reason, provenance). Never raises -- any failure yields
    (None, <reason>, {}) so a missing/stale grounding target never fails the
    deliberation itself.
    """
    raw_repo = (context or {}).get("repo")
    if not raw_repo or not isinstance(raw_repo, str) or not raw_repo.strip():
        return (None, "no_repo_in_context", {})

    repo = raw_repo.strip().rsplit("/", 1)[-1]
    if not repo or any(ch.isspace() for ch in repo) or ".." in repo or "/" in repo or "\\" in repo:
        return (None, "no_repo_in_context", {})

    clone_dir = Path(_GROUNDING_CLONE_ROOT) / f"{repo}{_GROUNDING_CLONE_SUFFIX}"
    if not clone_dir.is_dir():
        return (None, "grounding_target_unavailable", {})

    tmpdir = tempfile.mkdtemp(prefix=f"grounding-{repo}-")
    try:
        subprocess.run(
            ["git", "-C", str(clone_dir), "worktree", "add", "--detach", tmpdir, "origin/main"],
            check=True, capture_output=True, text=True, timeout=60,
        )
        sha_result = subprocess.run(
            ["git", "-C", str(clone_dir), "rev-parse", "origin/main"],
            check=True, capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:
        log.warning(
            "[shared-deliberation:grounding] worktree resolution failed for repo=%s: %s",
            repo, exc,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
        return (None, "grounding_target_unavailable", {})

    provenance = {
        "source_repo": repo,
        "resolved_sha": sha_result.stdout.strip(),
        "clone_dir": str(clone_dir),
        "worktree_path": tmpdir,
    }
    return (tmpdir, "", provenance)


def _cleanup_grounding_worktree(provenance: dict) -> None:
    """Remove a worktree created by `_resolve_grounding_target`. Never raises."""
    clone_dir = (provenance or {}).get("clone_dir")
    worktree_path = (provenance or {}).get("worktree_path")
    if not clone_dir or not worktree_path:
        return
    try:
        subprocess.run(
            ["git", "-C", clone_dir, "worktree", "remove", "--force", worktree_path],
            check=False, capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:
        log.warning(
            "[shared-deliberation:grounding] worktree cleanup failed for %s: %s",
            worktree_path, exc,
        )
    finally:
        shutil.rmtree(worktree_path, ignore_errors=True)


def _extract_denied_codebase_surfaces(facets_dict: Optional[dict]) -> tuple[list, list, list]:
    """Scan every non-final round's sim_failures for codebase-surface entries.

    Pinned to round index (round_num != max round_num among the rounds present), not to
    shape: the final round's sim_failures is always {} by adapter.py invariant 2
    (unfulfilled final-round sim_requests are recorded, never treated as failures), but
    this exemption is enforced explicitly here rather than relied upon implicitly.
    """
    rounds = (facets_dict or {}).get("rounds") or []
    if not rounds:
        return ([], [], [])

    max_round_num = max((r.get("round_num", 0) for r in rounds), default=0)
    surfaces: list = []
    reasons: list = []
    rounds_affected: list = []
    for rnd in rounds:
        if rnd.get("round_num") == max_round_num:
            continue
        sim_failures = rnd.get("sim_failures") or {}
        hit = False
        for key, reason in sim_failures.items():
            if key.split(":", 1)[0] == "codebase":
                surfaces.append(key)
                reasons.append(reason)
                hit = True
        if hit:
            rounds_affected.append(rnd.get("round_num"))
    return (surfaces, reasons, rounds_affected)


def _maybe_escalate_grounding_denial(
    *,
    context: dict,
    grounding_result_file: Optional[str],
    skip_reason: str,
    provenance: dict,
    facets_dict: Optional[dict],
) -> None:
    """Fire the facets-grounding-denied repair station, at most once per deliberation.

    Mirrors `_escalate_council_fast_fail` for structure: module-level station-ID
    constant, lazy import of repair_station (not loaded on every orchestrator import),
    bare except-suppress. That except is load-bearing, not cosmetic -- `escalate()`
    calls `get_db()` outside its own try block, so an unwritable dir or corrupt DB
    raises straight out of the call.
    """
    try:
        # Suppressions that apply regardless of case -- none of these leave anything
        # real to report: the elevator path already grounded fully, the leg never ran
        # (stub/disabled/missing-repo), or the leg died with no rounds to inspect.
        if grounding_result_file:
            return
        if os.getenv("SHARED_DELIBERATION_FACETS_STUB") == "1":
            return
        if os.getenv("FACETS_DISPATCH_DISABLED") == "1":
            return
        if not Path("/srv/git/facets-working").exists():
            return
        if facets_dict is None:
            return

        denied_surfaces, reasons, rounds_affected = _extract_denied_codebase_surfaces(facets_dict)

        if skip_reason:
            case = "absent"
        elif denied_surfaces:
            case = "denied"
        else:
            return

        from agents_core.repair_station import escalate, Tier, first

        tier = Tier.HIGH if case == "denied" else Tier.LOW

        error_signal = {
            "case": case,
            "repo": (context or {}).get("repo"),
            "skip_reason": skip_reason,
            "denied_surfaces": sorted(set(denied_surfaces)),
            "reasons": sorted(set(reasons)),
            "rounds_affected": sorted(set(rounds_affected)),
            "source_repo": provenance.get("source_repo") if provenance else None,
            "resolved_sha": provenance.get("resolved_sha") if provenance else None,
        }

        escalate(
            station_id=_GROUNDING_STATION_ID,
            stable_pointer="agents_core/shared_deliberation/orchestrator.py",
            error_signal=error_signal,
            author_intent="Facets codebase grounding was denied or structurally unavailable",
            escalation_policy=first(),
            tier=tier,
            owning_module="agents_core.shared_deliberation.orchestrator",
        )
    except Exception:
        log.exception("repair-station escalation failed for grounding denial — suppressed")


# Bounded concurrency for Facets subprocesses (gate against GW lane stampede)
_facets_semaphore: Optional[asyncio.Semaphore] = None

# Reserved seam modes for jagged-seam tap (v0: empty; H3 will register modes here)
_registered_seam_modes: list = []


def init_facets_semaphore(max_concurrent: int = 2) -> None:
    """Initialize the Facets concurrency gate. Called once at service startup."""
    global _facets_semaphore
    _facets_semaphore = asyncio.Semaphore(max_concurrent)


def register_seam_mode(mode) -> None:
    """Register a seam mode for jagged-seam deliberation (H3 tap).

    The mode must implement: async def run(seam_config, text, context) -> dict
    with keys: name, ok, result, errors.
    """
    global _registered_seam_modes
    _registered_seam_modes.append(mode)


async def _facets_subprocess(
    text: str,
    context: dict,
    operator: str = "gravitywell",
    grounding_result_file: Optional[str] = None,
    gw_principal: Optional[str] = None,
    target_repo: Optional[str] = None,
) -> tuple[bool, Optional[dict], Optional[str], Optional[str]]:
    """Invoke Facets via subprocess. Returns (ok, deliberation_dict, deliberation_id, errors).

    The subprocess is executed in a thread to avoid blocking the event loop.
    """
    if _facets_semaphore is None:
        raise RuntimeError("Facets semaphore not initialized; call init_facets_semaphore() at startup")

    if os.getenv("SHARED_DELIBERATION_FACETS_STUB") == "1":
        # Stub mode for testing: skip subprocess, return fixture
        return (True, {"stub": True, "methodology": {"synthesis_operator": operator}}, "stub-id", None)

    if os.getenv("FACETS_DISPATCH_DISABLED") == "1":
        return (False, None, None, "Facets dispatch disabled via FACETS_DISPATCH_DISABLED")

    facets_repo = Path("/srv/git/facets-working")
    if not facets_repo.exists():
        return (False, None, None, f"Facets repo not found at {facets_repo}")

    async with _facets_semaphore:
        return await asyncio.to_thread(
            _run_facets_subprocess, text, context, operator, facets_repo,
            grounding_result_file, gw_principal, target_repo,
        )


def _run_facets_subprocess(
    text: str,
    context: dict,
    operator: str,
    facets_repo: Path,
    grounding_result_file: Optional[str] = None,
    gw_principal: Optional[str] = None,
    target_repo: Optional[str] = None,
) -> tuple[bool, Optional[dict], Optional[str], Optional[str]]:
    """Synchronous subprocess invocation (runs in thread)."""
    # Grounding handoff guard: validate before building argv.
    if grounding_result_file is not None:
        grf_path = Path(grounding_result_file)
        if not grf_path.exists() or grf_path.stat().st_size == 0:
            log.error(
                "[shared-deliberation:grounding] GroundingHandoffError: "
                "producer promised grounding_result_file=%s but it is absent "
                "-- refusing to silently auto-ground",
                grounding_result_file,
            )
            raise GroundingHandoffError(
                f"grounding_result_file={grounding_result_file!r} is absent or empty; "
                "refusing to silently auto-ground (producer contract violation)"
            )

    try:
        # Create temp context file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"text": text, "spec_text": text, **context}, f)
            context_file = f.name

        try:
            facets_env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    p for p in (str(facets_repo), os.environ.get("PYTHONPATH", "")) if p
                ),
            }
            if gw_principal:
                facets_env["GW_GATE_PRINCIPAL"] = gw_principal
            argv = [
                "python3", "-m", "facets.adapter", "deliberate",
                text,
                "--context-file", context_file,
                "--format", "json",
            ]
            if operator and operator != "haiku":
                argv += ["--persona-operator", operator, "--synthesis-operator", operator]

            if target_repo:
                argv += ["--target-repo", target_repo]

            if grounding_result_file is not None:
                argv += ["--grounding-result-file", grounding_result_file, "--no-auto-ground"]
                log.info(
                    "[shared-deliberation:grounding] injected pre-computed grounding from %s",
                    grounding_result_file,
                )

            idle_kill_secs = float(os.environ.get("FACETS_ORCH_IDLE_KILL_SECS", "600"))
            hard_ceiling_secs = float(os.environ.get("FACETS_ORCH_HARD_CEILING_SECS", "1800"))

            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=facets_env,
            )

            # Track liveness via stderr reader thread
            last_stderr_at = [time.monotonic()]
            stderr_lines = []

            def _read_stderr():
                for line in proc.stderr:
                    last_stderr_at[0] = time.monotonic()
                    stderr_lines.append(line)

            stdout_chunks = []

            def _read_stdout():
                for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            t_stderr = threading.Thread(target=_read_stderr, daemon=True)
            t_stdout = threading.Thread(target=_read_stdout, daemon=True)
            t_stderr.start()
            t_stdout.start()

            start_time = time.monotonic()
            kill_reason = None

            while True:
                # Check if process has finished
                retcode = proc.poll()
                if retcode is not None:
                    break

                now = time.monotonic()
                elapsed = now - start_time
                idle = now - last_stderr_at[0]

                if elapsed >= hard_ceiling_secs:
                    kill_reason = "hard ceiling"
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                        proc.wait()
                    break

                if idle >= idle_kill_secs:
                    kill_reason = "silence"
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                        proc.wait()
                    break

                time.sleep(0.5)

            t_stderr.join(timeout=5.0)
            t_stdout.join(timeout=5.0)

            if kill_reason == "silence":
                error_msg = "Facets timeout (silence)"
                log.error(error_msg)
                return (False, None, None, error_msg)

            if kill_reason == "hard ceiling":
                error_msg = "Facets timeout (hard ceiling)"
                log.error(error_msg)
                return (False, None, None, error_msg)

            retcode = proc.poll()
            if retcode != 0:
                stderr_text = "".join(stderr_lines)
                error_msg = f"Facets subprocess exited {retcode}: {stderr_text}"
                log.error(error_msg)
                return (False, None, None, error_msg)

            stdout_text = "".join(stdout_chunks)
            deliberation_json = json.loads(stdout_text)
            deliberation_id = deliberation_json.get("deliberation_id")
            if not deliberation_id:
                error_msg = "No deliberation_id in Facets output"
                log.error(error_msg)
                return (False, None, None, error_msg)

            log.info(f"Facets complete: deliberation_id={deliberation_id}")
            return (True, deliberation_json, deliberation_id, None)

        finally:
            try:
                os.unlink(context_file)
            except OSError:
                pass

    except Exception as e:
        error_msg = f"Facets error: {e}"
        log.error(error_msg)
        return (False, None, None, error_msg)


async def _council_subprocess(
    text: str,
    voicing: str = "gravitywell",
    gw_principal: Optional[str] = None,
) -> tuple[bool, Optional[str], Optional[dict], Optional[str]]:
    """Submit council deliberation and poll until terminal.

    Returns (ok, run_id, council_data, errors).
    council_data includes status, landing, confidence, open_questions, positions, etc.
    """
    if os.getenv("SHARED_DELIBERATION_COUNCIL_STUB") == "1":
        # Stub mode for testing
        return (True, "stub-council-id", {"status": "resolved", "positions": []}, None)

    # Read timeout from env; default 1800s (30 min)
    timeout_s = int(os.environ.get("SHARED_DELIBERATION_COUNCIL_TIMEOUT_S", "1800"))
    # Bounded retry (defense-in-depth): a liveness died/stalled fast-fail can be a
    # false positive from serial-queue contention on the single-model GW endpoint.
    # Re-submit once with a fresh run_id before giving up. Does NOT fire on a
    # terminal Council status (error is None) or on a genuine timeout_s backstop
    # (error has no "died/stalled" marker) — only on the structured liveness signal.
    max_retries = int(os.environ.get("COUNCIL_MAX_RETRIES", "1"))

    attempt = 0
    while True:
        run_id = await asyncio.to_thread(_submit_council, text, voicing, gw_principal)
        if not run_id:
            return (False, None, None, "Failed to submit council")

        council_data, error = await asyncio.to_thread(_poll_council, run_id, timeout_s)
        if not error:
            return (True, run_id, council_data, None)

        if "died/stalled" in error and attempt < max_retries:
            attempt += 1
            log.warning(
                f"Council liveness fast-fail on run_id={run_id}; "
                f"retrying ({attempt}/{max_retries}): {error}"
            )
            continue

        return (False, run_id, None, error)


def _submit_council(text: str, voicing: str, gw_principal: Optional[str] = None) -> Optional[str]:
    """Submit council deliberation via agents_core.council.cli.cmd_submit."""
    try:
        from agents_core.council.cli import cmd_submit, DEFAULT_TURNS
        import argparse
        import io
        import contextlib
        import re

        args = argparse.Namespace(
            decision=text,
            voicing=voicing,
            mode="deliberation",
            n=None,
            turns=DEFAULT_TURNS,
            with_entity=None,
            narrator=False,
            narrator_voice=None,
            no_queue=False,
            notify=False,
            gw_principal=gw_principal,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_submit(args)
        output = buf.getvalue()
        m = re.search(r"task_id=(\S+)", output)
        if not m:
            log.error(f"Could not parse run_id from council output: {output!r}")
            return None
        return m.group(1)
    except Exception as e:
        log.error(f"Council submit error: {e}")
        return None


def _queue_task_alive(run_id: str) -> Optional[bool]:
    """Consult ClaudeQueue for run_id's presence among pending/active tasks.

    Returns True if the task is still queued or running — the worker is not dead,
    just serialized behind another task on the single-model GW endpoint (queued !=
    dead). Returns False if the queue is reachable but run_id is absent from both
    lists (never enqueued, or already reaped). Returns None if the queue itself
    cannot be consulted (import failure, lookup error) — this must never
    propagate as a poller failure; callers treat None the same as False (degrade
    to the tolerant startup-grace fallback, never to a false positive).
    """
    try:
        from agents_core.claude_queue import ClaudeQueue
        queue = ClaudeQueue()
        for task in queue.get_pending():
            if task.get("id") == run_id:
                return True
        for task in queue.get_active():
            if task.get("id") == run_id:
                return True
        return False
    except Exception as e:
        log.warning(f"Council queue liveness check failed for run_id={run_id}: {e}")
        return None


def _poll_council(run_id: str, timeout_s: int = 1800) -> tuple[Optional[dict], Optional[str]]:
    """Poll council run YAML until terminal status or timeout.

    Returns (council_data, error_message).
    council_data extracts status, landing, confidence, open_questions, positions.

    Liveness ladder (queued != dead), most-live rung first:
      1. heartbeat_at present -> tight COUNCIL_STALL_S (env, default 180s)
         inter-heartbeat clock, unchanged from before.
      2. else started_at present (worker running, pre-first-heartbeat) ->
         COUNCIL_STARTUP_GRACE_S (env, default 400s) clock from started_at.
         Covers the doorman-acquire window (~210s) plus first-call/GC headroom.
      3. else still queued (neither field set) -> ask ClaudeQueue. Pending/running
         there means keep polling, bounded only by the timeout_s backstop below —
         serial-queue wait must never read as worker death. Absent from the queue,
         or the queue lookup itself failing, degrades to the COUNCIL_STARTUP_GRACE_S
         clock from created_at (never the tight stall clock — a missing/lying queue
         must degrade to tolerant, not to a false positive).

    The timeout_s backstop is the ultimate ceiling on every rung, including the
    queued rung: a queue that never stops reporting "pending" can at worst delay
    the verdict to timeout_s, never hang the gate.
    """
    import yaml

    council_dir = room_path("council")
    start_time = time.time()
    poll_interval = 5
    stall_s = int(os.environ.get("COUNCIL_STALL_S", "180"))
    startup_grace_s = int(os.environ.get("COUNCIL_STARTUP_GRACE_S", "400"))

    while time.time() - start_time < timeout_s:
        run_path = council_dir / f"{run_id}.yaml"
        try:
            if not run_path.exists():
                time.sleep(poll_interval)
                continue

            run = yaml.safe_load(run_path.read_text())
            if not isinstance(run, dict):
                error = f"Council YAML is not a dict: {run_path}"
                log.error(error)
                return (None, error)

            status = run.get("status")
            if status in ("resolved", "open", "laid-down", "failed", "closed"):
                # Terminal status reached
                synthesis = run.get("synthesis", {})
                return (
                    {
                        "status": status,
                        "landing": synthesis.get("landing"),
                        "confidence": synthesis.get("confidence"),
                        "open_questions": synthesis.get("open_questions", []),
                        "positions": synthesis.get("positions", []),
                        "voicing_effective": run.get("voicing"),
                        "voicing_degraded": run.get("selection_degraded", False),
                        "voicing_degraded_reason": None,  # TODO: extract from run if available
                    },
                    None,
                )

            # Liveness ladder (queued != dead) — see docstring. Most-live rung first.
            last_heartbeat = run.get("heartbeat_at")
            started_at = run.get("started_at")
            if last_heartbeat:
                ref_str = last_heartbeat
                clock_s = stall_s
            elif started_at:
                ref_str = started_at
                clock_s = startup_grace_s
            else:
                queue_alive = _queue_task_alive(run_id)
                if queue_alive is True:
                    # Still queued behind another serialized task — alive, keep
                    # polling. Bounded only by the timeout_s backstop below.
                    time.sleep(poll_interval)
                    continue
                # Absent from the queue, or the queue lookup failed/unavailable —
                # tolerant fallback, never the tight stall clock.
                ref_str = run.get("created_at")
                clock_s = startup_grace_s

            if ref_str:
                try:
                    ref_ts = datetime.fromisoformat(ref_str).timestamp()
                except (ValueError, TypeError):
                    ref_ts = None
                if ref_ts is not None and time.time() - ref_ts > clock_s:
                    reason = "heartbeat_stale" if last_heartbeat else "no_heartbeat_after_startup"
                    error = (
                        f"council worker died/stalled "
                        f"(run_id={run_id}, last_heartbeat={last_heartbeat!r}, reason={reason})"
                    )
                    log.error(error)
                    _escalate_council_fast_fail(run_id, last_heartbeat, reason)
                    return (None, error)

            time.sleep(poll_interval)
        except Exception as e:
            error = f"Council poll error: {e}"
            log.error(error)
            return (None, error)

    # Timeout (backstop — liveness fast-fail is the primary path for dead workers)
    error = f"Council poll timeout after {timeout_s}s"
    log.error(error)
    return (None, error)


async def run_deliberation(request: DeliberationRequest) -> DeliberationEnvelope:
    """Main orchestrator: run Facets and Council concurrently, unify results.

    Partial-failure semantics:
    - If Facets fails but Council succeeds, return HTTP 200 with facets_ok=False
    - If Council fails but Facets succeeds, return HTTP 200 with council_ok=False
    - If both fail, return HTTP 200 with both _ok=False (caller must check flags)
    """
    import uuid

    request_id = f"deliberation-{uuid.uuid4().hex[:12]}"
    triage = request.triage or "full"
    triage_escalated = False
    triage_reason = "caller-requested"

    # Deliberation-spanning GW keepawake hold (shared-deliberation-gate-spanning-keepawake-v0).
    # Placed BEFORE Facets dispatch so GW stays warm across the full Facets + Council window.
    # This is the primary guarantee; the council leg's own hold (council/cli.py) remains as
    # defense-in-depth but is independent of this one; a dead council worker drops its own
    # lease but does NOT affect this orchestrator hold.
    # TTL covers the max deliberation duration; refreshed periodically via background thread.
    # Auto-expires if the orchestrator process dies (no unconditional pinning).
    _gw_in_play = (
        request.council_voicing == "gravitywell"
        or request.facets_operator == "gravitywell"
    )
    _span_work_id = f"shared-delib-{request_id}"
    _span_doorman = None
    _span_stop = threading.Event()
    _span_refresh_thread = None

    if _gw_in_play:
        try:
            from agents_core.doorman_client import DoormanClient, _gw_acquire_timeout
            _span_doorman = DoormanClient()
            # Max deliberation window: used as absolute deadline for the refresh loop.
            _span_ttl = (
                int(os.environ.get("SHARED_DELIBERATION_COUNCIL_TIMEOUT_S", "1800")) + 600
            )
            # Refresh interval and short per-lease TTL (Fix 2: hold self-expires if refresh stops).
            # _refresh_ttl < _span_ttl: a leaked hold expires within 2x refresh intervals
            # without depending on the finally-release firing.
            _refresh_interval = int(os.environ.get("SHARED_DELIB_SPAN_REFRESH_S", "300"))
            _refresh_ttl = _refresh_interval * 2
            _span_deadline_abs = time.time() + _span_ttl  # absolute deadline for refresh loop

            _span_principal = request.gw_principal or _span_work_id
            _hold_res = await asyncio.to_thread(
                _span_doorman.acquire,
                "gravitywell", _span_work_id, _refresh_ttl,
                "shared-deliberation-span-hold",
                timeout=_gw_acquire_timeout(),
                principal=_span_principal,
                lease_kind="coordination",
                lease_class="protected",
            )
            _hold_status = _hold_res.get("status")
            if _hold_status == "serving":
                log.info(
                    "[shared-deliberation] span hold placed request_id=%s status=%s",
                    request_id, _hold_status,
                )
            else:
                log.warning(
                    "[shared-deliberation] span hold NOT placed request_id=%s status=%s"
                    " - GW may idle-suspend during deliberation",
                    request_id, _hold_status,
                )
            if _hold_status == "serving":
                def _span_refresh_loop(
                    _dc=_span_doorman,
                    _wid=_span_work_id,
                    _principal=_span_principal,
                    _refresh_ttl=_refresh_ttl,
                    _stop=_span_stop,
                    _iv=_refresh_interval,
                    _deadline=_span_deadline_abs,
                ):
                    # Short-interval poll: responsive to stop event and deadline check.
                    _POLL_S = 15
                    _next_refresh = time.time() + _iv
                    while not _stop.wait(_POLL_S):
                        now = time.time()
                        if now >= _deadline:
                            # Past absolute deliberation deadline — stop refreshing.
                            # Lease expires within _refresh_ttl seconds on its own.
                            break
                        if now >= _next_refresh:
                            try:
                                _dc.acquire(
                                    "gravitywell", _wid, _refresh_ttl,
                                    "shared-deliberation-span-refresh",
                                    timeout=10.0,
                                    principal=_principal,
                                    lease_kind="coordination",
                                    lease_class="protected",
                                )
                            except Exception as _ref_err:
                                log.warning(
                                    "[shared-deliberation] span hold refresh failed: %s",
                                    _ref_err,
                                )
                            _next_refresh = now + _iv

                _span_refresh_thread = threading.Thread(
                    target=_span_refresh_loop,
                    daemon=True,
                    name=f"span-hold-refresh-{request_id}",
                )
                _span_refresh_thread.start()
        except Exception as _hold_err:
            log.warning(
                "[shared-deliberation] span hold acquire failed (non-fatal): %s",
                _hold_err,
            )

    _grounding_path, _grounding_skip_reason, _grounding_provenance = await asyncio.to_thread(
        _resolve_grounding_target, request.context
    )
    if _grounding_skip_reason:
        log.info(
            "[shared-deliberation:grounding] no target resolved request_id=%s reason=%s",
            request_id, _grounding_skip_reason,
        )

    try:
        # Run both legs concurrently via asyncio.gather
        async def _facets_leg():
            return await _facets_subprocess(
                request.text,
                request.context,
                request.facets_operator,
                request.grounding_result_file,
                request.gw_principal,
                _grounding_path,
            )

        async def _council_leg():
            # Council only runs if triage == "full"; otherwise return (False, None, None, None)
            if triage != "full":
                return (False, None, None, None)
            return await _council_subprocess(
                request.text,
                request.council_voicing,
                request.gw_principal,
            )

        facets_result, council_result = await asyncio.gather(
            _facets_leg(),
            _council_leg(),
        )

        facets_ok, facets_dict, facets_id, facets_error = facets_result
        council_ok, council_run_id, council_data, council_error = council_result

        _maybe_escalate_grounding_denial(
            context=request.context,
            grounding_result_file=request.grounding_result_file,
            skip_reason=_grounding_skip_reason,
            provenance=_grounding_provenance,
            facets_dict=facets_dict,
        )

        # Extract operator info from Facets
        operator_requested = None
        operator_effective = None
        if facets_dict:
            methodology = facets_dict.get("methodology", {})
            operator_requested = methodology.get("operator_requested", request.facets_operator)
            operator_effective = methodology.get("synthesis_operator", request.facets_operator)

        # Extract council info
        council_status = None
        council_landing = None
        council_confidence = None
        council_open_questions = []
        council_positions = []
        council_voicing_effective = None
        council_voicing_degraded = False
        council_voicing_degraded_reason = None

        if council_data and council_ok:
            council_status = council_data.get("status")
            council_landing = council_data.get("landing")
            council_confidence = council_data.get("confidence")
            council_open_questions = council_data.get("open_questions", [])
            council_positions = council_data.get("positions", [])
            council_voicing_effective = council_data.get("voicing_effective")
            council_voicing_degraded = council_data.get("voicing_degraded", False)
            council_voicing_degraded_reason = council_data.get("voicing_degraded_reason")

        # Collect errors
        errors = {}
        if facets_error:
            errors["facets"] = facets_error
        if council_error:
            errors["council"] = council_error

        envelope = DeliberationEnvelope(
            deliberation_request_id=request_id,
            triage=triage,
            triage_reason=triage_reason,
            triage_escalated=triage_escalated,
            # Facets
            facets_ok=facets_ok,
            facets=facets_dict,
            facets_deliberation_id=facets_id,
            operator_requested=operator_requested,
            operator_effective=operator_effective,
            # Council
            council_ok=council_ok,
            council_run_id=council_run_id,
            council_status=council_status,
            council_landing=council_landing,
            council_confidence=council_confidence,
            council_open_questions=council_open_questions,
            council_positions=council_positions,
            council_voicing_requested=request.council_voicing,
            council_voicing_effective=council_voicing_effective,
            council_voicing_degraded=council_voicing_degraded,
            council_voicing_degraded_reason=council_voicing_degraded_reason,
            # Extensions and errors
            extra_modes=[],  # Reserved for jagged-seam tap (v0: empty)
            errors=errors,
        )

        return envelope
    finally:
        # Remove the grounding worktree (if one was created) on any exit path.
        if _grounding_provenance:
            await asyncio.to_thread(_cleanup_grounding_worktree, _grounding_provenance)
        # Stop the refresh thread and release the span hold on any exit path.
        _span_stop.set()
        if _span_refresh_thread is not None:
            _span_refresh_thread.join(timeout=15.0)
        if _span_doorman is not None:
            try:
                await asyncio.to_thread(
                    _span_doorman.release, "gravitywell", _span_work_id
                )
            except Exception as _rel_err:
                log.warning(
                    "[shared-deliberation] span hold release failed: %s", _rel_err
                )
            try:
                _span_doorman.close()
            except Exception:
                pass
