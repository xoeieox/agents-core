"""MCP stdio server exposing `lapis-pm` (status/list/brief/bind/tick) as typed
tools for opencode, via subprocess + stdout parsing.

`lapis-pm` is a pure argparse CLI with no importable library API (see spec
opencode-mcp-mem-lapispm-v0). This wrapper shells the exact same subcommands
a human/bash-shelling model would run and parses stdout into structured
fields, per a strict parsing contract:

- every response carries `raw_output`, `exit_code`, `parse_status`
- a non-zero exit short-circuits parsing entirely (`parse_status: cli_error`)
- on exit 0, a field is only included if its pattern matches exactly once;
  ambiguous/absent fields are omitted rather than guessed
- `parse_status` is "ok" only if every expected field for that call parsed
  cleanly; otherwise "partial" — callers should fall back to `raw_output`
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lapis-pm")

# `status` in particular makes a live Forgejo network call under the hood; bound
# so a slow/unreachable Forgejo can't block this stdio server indefinitely.
_SUBPROCESS_TIMEOUT_S = 30.0


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["lapis-pm", *argv],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\n[mcp_lapispm] timed out after {_SUBPROCESS_TIMEOUT_S}s",
        )


def _cli_error(proc: subprocess.CompletedProcess) -> dict:
    return {
        "raw_output": proc.stdout,
        "exit_code": proc.returncode,
        "stderr": proc.stderr,
        "parse_status": "cli_error",
    }


def _coerce_bool(value: str) -> bool | str:
    if value == "True":
        return True
    if value == "False":
        return False
    return value


_STATUS_FIELD_PATTERNS = {
    "title": r"^\s*title:\s+(.*)$",
    "pm_bound": r"^\s*pm_bound:\s+(\S+)$",
    "pm_repo": r"^\s*pm_repo:\s+(.*)$",
    "pm_authority": r"^\s*pm_authority:\s+(.*)$",
    "pm_verification": r"^\s*pm_verification:\s+(.*)$",
    "paused": r"^\s*paused:\s+(\S+)",
    "cursor": r"^\s*cursor:\s+(.*)$",
    "dispatched": r"^\s*dispatched:\s+(.*)$",
    "outstanding": r"^\s*outstanding:\s+(.*)$",
}


def _parse_status(stdout: str, target_id: str) -> tuple[dict, str]:
    blocks = re.findall(r"^=== (?!review-gate)(\S+) ===\n((?:(?!^===).)*)", stdout, re.M | re.S)
    if target_id:
        bodies = [body for tid, body in blocks if tid == target_id]
    else:
        bodies = [body for _, body in blocks]

    if len(bodies) != 1:
        return {}, "partial"

    body = bodies[0]
    parsed: dict = {}
    for field, pattern in _STATUS_FIELD_PATTERNS.items():
        matches = re.findall(pattern, body, re.M)
        if len(matches) == 1:
            value = matches[0].strip()
            parsed[field] = _coerce_bool(value) if field in ("pm_bound", "paused") else value

    parse_status = "ok" if len(parsed) == len(_STATUS_FIELD_PATTERNS) else "partial"
    return parsed, parse_status


@mcp.tool()
def lapispm_status(target_id: str = "", explain: bool = False) -> dict:
    """Show PM state for a bound target (or all bound targets if target_id is empty)."""
    argv = ["status"]
    if target_id:
        argv.append(target_id)
    if explain:
        argv.append("--explain")
    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)
    parsed, parse_status = _parse_status(proc.stdout, target_id)
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


@mcp.tool()
def lapispm_list() -> dict:
    """List all pm-bound targets."""
    proc = _run(["list", "--json"])
    if proc.returncode != 0:
        return _cli_error(proc)
    try:
        targets = json.loads(proc.stdout)
        return {
            "raw_output": proc.stdout,
            "exit_code": 0,
            "parse_status": "ok",
            "targets": targets,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "raw_output": proc.stdout,
            "exit_code": 0,
            "parse_status": "partial",
        }


_BRIEF_FIELD_PATTERNS = {
    "brief_path": r"^Brief written:\s+(.+)$",
    "symlink": r"^Symlink updated:\s+(.+)$",
}


@mcp.tool()
def lapispm_brief() -> dict:
    """Generate a live state-of-work brief."""
    proc = _run(["brief", "--period", "live"])
    if proc.returncode != 0:
        return _cli_error(proc)
    parsed: dict = {}
    for field, pattern in _BRIEF_FIELD_PATTERNS.items():
        matches = re.findall(pattern, proc.stdout, re.M)
        if len(matches) == 1:
            parsed[field] = matches[0].strip()
    parse_status = "ok" if len(parsed) == len(_BRIEF_FIELD_PATTERNS) else "partial"
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


@mcp.tool()
def lapispm_bind(
    target_id: str,
    spec_from: str,
    repo: str,
    authority: str = "advisory",
    verification: str = "",
    force: bool = False,
    create: bool = False,
    title: str = "",
    description: str = "",
    destination_slug: str = "",
    destination_name: str = "",
    destination_when: str = "",
) -> dict:
    """Bind a target to the PM with a spec. Mirrors `lapis-pm bind`'s common flags."""
    argv = [
        "bind", target_id,
        "--spec-from", spec_from,
        "--repo", repo,
        "--authority", authority,
    ]
    if verification:
        argv += ["--verification", verification]
    if force:
        argv.append("--force")
    if create:
        argv.append("--create")
    if title:
        argv += ["--title", title]
    if description:
        argv += ["--description", description]
    if destination_slug:
        argv += ["--destination-slug", destination_slug]
    if destination_name:
        argv += ["--destination-name", destination_name]
    if destination_when:
        argv += ["--destination-when", destination_when]

    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)

    parsed: dict = {}
    bound_match = re.findall(
        r"^Bound (\S+) .*repo=([^,]+), authority=([^,]+), verification=(\S+)$",
        proc.stdout, re.M,
    )
    if len(bound_match) == 1:
        tid, bound_repo, bound_authority, bound_verification = bound_match[0]
        parsed["bound_target_id"] = tid
        parsed["bound_repo"] = bound_repo
        parsed["bound_authority"] = bound_authority
        parsed["bound_verification"] = bound_verification
    spec_match = re.findall(r"^Spec:\s+(\d+) chars$", proc.stdout, re.M)
    if len(spec_match) == 1:
        parsed["spec_chars"] = int(spec_match[0])

    parse_status = "ok" if len(bound_match) == 1 and len(spec_match) == 1 else "partial"
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


_TICK_LINE_RE = re.compile(
    r"^\[(?P<target_id>\S+)\] skipped=(?P<skipped>\S+) reason=(?P<reason>.*?)"
    r"(?: reconciled=(?P<reconciled>\S+))? encoded=(?P<encoded>\S+) decision=(?P<decision>\S+)$"
)
_FORCE_DISPATCH_RE = re.compile(r"^Dispatched:\s+(\S+)\s+task_id=(\S+)$")


@mcp.tool()
def lapispm_tick(target_id: str = "", force_dispatch: str = "") -> dict:
    """Run one PM tick (single target, all bound targets, or a forced smoke dispatch)."""
    argv = ["tick"]
    if target_id:
        argv += ["--target", target_id]
    if force_dispatch:
        argv += ["--force-dispatch", force_dispatch]

    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)

    if force_dispatch:
        match = _FORCE_DISPATCH_RE.search(proc.stdout)
        if match:
            return {
                "raw_output": proc.stdout,
                "exit_code": 0,
                "parse_status": "ok",
                "agent_type": match.group(1),
                "task_id": match.group(2),
            }
        return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "partial"}

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    ticks = []
    unmatched = 0
    for line in lines:
        match = _TICK_LINE_RE.match(line)
        if match:
            ticks.append({k: v for k, v in match.groupdict().items() if v is not None})
        else:
            unmatched += 1

    if unmatched > 0:
        return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "partial", "ticks": ticks}
    return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "ok", "ticks": ticks}


# ---------------------------------------------------------------------------
# spec-review: background-launch + poll pair
#
# `spec-review` blocks for 5-15 minutes (Facets + Mirror Council legs), so it
# cannot fit the `_run` blocking-subprocess-with-timeout shape above. These
# two tools launch it detached and let the caller poll on its own cadence.
# See spec opencode-mcp-lapispm-spec-review-v0.
# ---------------------------------------------------------------------------

# In-process Popen handles, keyed by run_id. Populated by `start` within this
# server's lifetime; used by `poll` to get a real exit code via .poll() when
# available. A `poll` call against a run this process didn't launch (a
# different server instance, or a restart) won't find an entry here — it
# falls back to PID-liveness + best-effort exit-code inference from the
# parsed brief (see `_inferred_exit_code`).
_POPEN_REGISTRY: dict[str, subprocess.Popen] = {}


def _spec_review_runtime_dir() -> Path:
    return Path(f"/run/user/{os.getuid()}")


def _spec_review_lock_path() -> Path:
    """Canonical CLI lock path — must match lapis_pm.spec_review._spec_review_lock_path()."""
    return _spec_review_runtime_dir() / "lapis-pm-spec-review.lock"


def _spec_review_runs_dir() -> Path:
    return _spec_review_runtime_dir() / "lapis-pm-spec-review-runs"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just owned by someone else
    except OSError:
        return False
    return True


def _read_lock_info(lock_path: Path) -> dict | None:
    """Read-only: return the lock file's JSON payload, or None if absent/empty/unparsable."""
    try:
        data = lock_path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _try_nonblocking_flock(lock_path: Path) -> tuple[bool, str]:
    """Read-only observer: test whether the lock is actually free without ever
    creating, deleting, or rewriting the lock file (kernel owns lock state —
    lapis_pm.spec_review's fail-closed invariant, inherited unchanged here).

    Returns (acquired, flock_error). flock_error is the errno/strerror from the
    failed attempt, or "" if acquired (or the file doesn't exist, i.e. no lock).
    """
    try:
        fd = os.open(str(lock_path), os.O_RDONLY)
    except OSError:
        return True, ""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        return False, f"[Errno {e.errno}] {e.strerror}"
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True, ""
    finally:
        os.close(fd)


def _iter_registry_entries(runs_dir: Path) -> list[dict]:
    entries = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            entries.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return entries


def _read_registry_entry(runs_dir: Path, run_id: str) -> dict | None:
    try:
        return json.loads((runs_dir / f"{run_id}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_live_registry_run(runs_dir: Path, spec_path: str) -> dict | None:
    for entry in _iter_registry_entries(runs_dir):
        pid = entry.get("pid")
        if entry.get("spec_path") == spec_path and isinstance(pid, int) and _pid_alive(pid):
            return entry
    return None


def _most_recent_registry_run(runs_dir: Path, spec_path: str) -> dict | None:
    matches = [e for e in _iter_registry_entries(runs_dir) if e.get("spec_path") == spec_path]
    if not matches:
        return None
    matches.sort(key=lambda e: e.get("started_at", ""))
    return matches[-1]


def _new_run_id(spec_path: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", Path(spec_path).stem.lower()).strip("-") or "spec"
    return f"{int(time.time())}-{slug}-{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _elapsed_seconds(started_at: str) -> float:
    try:
        started = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0
    return (datetime.now(timezone.utc) - started).total_seconds()


def _read_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _spec_review_argv(
    spec_path: str,
    council_voicing: str,
    facets_operator: str,
    no_sonnet_reviewer: bool,
    no_facets: bool,
    authority: str,
    timeout_s: int,
) -> list[str]:
    """Build the `lapis-pm spec-review` argv — flags map 1:1 to the CLI's own parser."""
    argv = [
        "lapis-pm", "spec-review", spec_path,
        "--council-voicing", council_voicing,
        "--facets-operator", facets_operator,
        "--timeout", str(timeout_s),
    ]
    if no_sonnet_reviewer:
        argv.append("--no-sonnet-reviewer")
    if no_facets:
        argv.append("--no-facets")
    if authority:
        argv += ["--authority", authority]
    return argv


# format_brief() (lapis_pm/spec_review.py) renders these fields uniquely at the
# top of every brief regardless of which legs ran; used to reconstruct a
# structured "done" response from the log's raw markdown.
_BRIEF_TOP_FIELD_PATTERNS = {
    "target_id": r"^# Spec Review:\s+(.+)$",
    "spec_path": r"^\*\*Spec:\*\*\s+(.+)$",
    "repo": r"^\*\*Repo:\*\*\s+(.+)$",
    "elapsed_s": r"^\*\*Elapsed:\*\*\s+([\d.]+)s$",
    "recommendation": r"^\*\*Recommendation:\*\*\s+(.+)$",
}

# "Run ID:" also appears under the Facets and Sonnet sections when those legs
# ran, so council fields are extracted from the isolated Mirror Council block
# (same block-extraction convention _parse_status uses above) rather than
# matched against the whole document.
_COUNCIL_SECTION_RE = re.compile(r"^## Mirror Council deliberation\n((?:(?!^##).)*)", re.M | re.S)
_COUNCIL_FIELD_PATTERNS = {
    "council_status": r"^-\s+\*\*Status:\*\*\s+(\S+?),",
    "council_confidence": r"confidence\s+(.+)$",
    "council_run_id": r"^-\s+\*\*Run ID:\*\*\s+(.+)$",
    "council_landing": r"^-\s+\*\*Landing:\*\*\s+(.+)$",
}


def _parse_spec_review_brief(raw_output: str) -> tuple[dict, str]:
    parsed: dict = {}
    ok = True

    for field, pattern in _BRIEF_TOP_FIELD_PATTERNS.items():
        matches = re.findall(pattern, raw_output, re.M)
        if len(matches) == 1:
            parsed[field] = matches[0].strip()
        else:
            ok = False

    council_blocks = _COUNCIL_SECTION_RE.findall(raw_output)
    if len(council_blocks) == 1:
        section = council_blocks[0]
        for field, pattern in _COUNCIL_FIELD_PATTERNS.items():
            matches = re.findall(pattern, section, re.M)
            if len(matches) == 1:
                parsed[field] = matches[0].strip()
            else:
                ok = False
    else:
        ok = False

    parse_status = "ok" if ok and parsed else "partial"
    return parsed, parse_status


def _inferred_exit_code(recommendation: str) -> int:
    """Recompute cmd_spec_review's own exit-code mapping (lapis_pm/cli.py) from the
    parsed recommendation — a deterministic re-derivation, not a guess, for the case
    where the launching process (and its real Popen handle) is no longer available.
    """
    return 1 if recommendation in {"parse_failed", "incomplete"} else 0


@mcp.tool()
def lapispm_spec_review_start(
    spec_path: str,
    council_voicing: str = "gravitywell",
    facets_operator: str = "gravitywell",
    no_sonnet_reviewer: bool = True,
    no_facets: bool = False,
    authority: str = "",
    timeout_s: int = 1800,
) -> dict:
    """Launch `lapis-pm spec-review` as a detached background process; returns
    immediately with a run_id to poll via lapispm_spec_review_poll. Refuses to
    double-launch if a review is already active (this server's own registry, or
    the CLI's own cross-session lock file — see spec opencode-mcp-lapispm-spec-review-v0).
    """
    runs_dir = _spec_review_runs_dir()
    lock_path = _spec_review_lock_path()

    existing = _find_live_registry_run(runs_dir, spec_path)
    if existing is not None:
        return {
            "status": "already_running",
            "held_by_pid": existing.get("pid"),
            "spec_path": existing.get("spec_path"),
            "started_at": existing.get("started_at"),
            "run_id": existing.get("run_id"),
        }

    lock_info = _read_lock_info(lock_path)
    if lock_info is not None:
        held_pid = lock_info.get("pid")
        if isinstance(held_pid, int) and _pid_alive(held_pid):
            return {
                "status": "already_running",
                "held_by_pid": held_pid,
                "spec_path": lock_info.get("spec_path"),
                "started_at": lock_info.get("started_at"),
                "run_id": None,
            }
        acquired, flock_error = _try_nonblocking_flock(lock_path)
        if not acquired:
            return {
                "status": "stale_lock",
                "held_by_pid": held_pid,
                "spec_path": lock_info.get("spec_path"),
                "flock_error": flock_error,
            }

    argv = _spec_review_argv(
        spec_path, council_voicing, facets_operator,
        no_sonnet_reviewer, no_facets, authority, timeout_s,
    )
    run_id = _new_run_id(spec_path)
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / f"{run_id}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    log_fh = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
    finally:
        log_fh.close()

    started_at = _now_iso()
    entry = {
        "run_id": run_id,
        "pid": proc.pid,
        "spec_path": spec_path,
        "log_path": str(log_path),
        "started_at": started_at,
        "cli_argv": argv,
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(entry), encoding="utf-8")
    _POPEN_REGISTRY[run_id] = proc

    return {
        "run_id": run_id,
        "pid": proc.pid,
        "log_path": str(log_path),
        "status": "started",
    }


@mcp.tool()
def lapispm_spec_review_poll(run_id: str = "", spec_path: str = "") -> dict:
    """Non-blocking check on a spec-review run started via lapispm_spec_review_start.
    Looks up by run_id, or by spec_path (most recent run for that spec) if run_id is
    omitted. Never waits for the process to finish — poll repeatedly on your own cadence.
    """
    runs_dir = _spec_review_runs_dir()

    entry: dict | None = None
    if run_id:
        entry = _read_registry_entry(runs_dir, run_id)
    elif spec_path:
        entry = _most_recent_registry_run(runs_dir, spec_path)
        if entry is None:
            lock_path = _spec_review_lock_path()
            lock_info = _read_lock_info(lock_path)
            if lock_info is not None:
                held_pid = lock_info.get("pid")
                if not (isinstance(held_pid, int) and _pid_alive(held_pid)):
                    acquired, flock_error = _try_nonblocking_flock(lock_path)
                    if not acquired:
                        return {
                            "status": "stale_lock",
                            "held_by_pid": held_pid,
                            "spec_path": lock_info.get("spec_path"),
                            "flock_error": flock_error,
                        }

    if entry is None:
        return {"status": "not_found"}

    resolved_run_id = entry.get("run_id", run_id)
    pid = entry.get("pid")
    log_path = Path(entry["log_path"])
    started_at = entry.get("started_at", "")

    popen_obj = _POPEN_REGISTRY.get(resolved_run_id)
    if popen_obj is not None:
        rc = popen_obj.poll()
        running = rc is None
        exit_code = rc
    else:
        running = isinstance(pid, int) and _pid_alive(pid)
        exit_code = None

    log_text = _read_log(log_path)

    if running:
        return {
            "status": "running",
            "elapsed_s": _elapsed_seconds(started_at),
            "tail": log_text[-2000:],
        }

    parsed, parse_status = _parse_spec_review_brief(log_text)
    if exit_code is None:
        recommendation = parsed.get("recommendation")
        exit_code = _inferred_exit_code(recommendation) if recommendation is not None else None

    if exit_code is None and parse_status != "ok":
        return {"status": "unknown", "raw_output": log_text}

    return {
        "status": "done",
        "exit_code": exit_code,
        "raw_output": log_text,
        "parse_status": parse_status,
        **parsed,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
