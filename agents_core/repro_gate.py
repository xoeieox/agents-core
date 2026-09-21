"""P1 reproduce-before-retry: fresh-clone gate reproduction harness.

lapis-pm-test-gate-hermeticity-v0 (rev 3, D1/D3/D4/D5/D6).

Before the salvage path labels a test-gate failure as
``concluded_gate_rejected`` and opens the PR, the failed suite is re-run in
a FRESH THROWAWAY CLONE at the gate-run's provenance head SHA. The verdict
partition is 5 classes (RED / GREEN / ERROR / INCONCLUSIVE / SHA-MOVED):

- RED: the failed node-ids (or the full gate suite in the fallback) fail at
  the head in the repro -> the concluded_gate_rejected label is EARNED.
- GREEN: the FULL gate suite is green at the head in the repro (GREEN is
  issuable ONLY on the full suite, never on the targeted subset alone) ->
  NO concluded_gate_rejected label (the clean-push / salvage-green routes).
- ERROR: the repro itself could not run (missing deps, OOM, ENOSPC,
  env-dirty) -> infra-noise classification: NO salvage PR with a defect
  label, NO review cycle consumed.
- INCONCLUSIVE: the repro started and at least one re-run node-id is
  untested at verdict time (the mid-suite timeout/OOM shape) -> may NEVER
  be GREEN; consumes one of the 2/tick budget; defers to the next tick.
- SHA-MOVED: the repro SHA != the current head at repro time -> defer to
  the next tick and re-probe the new head; NEVER classify the old
  gate-red from a moved-SHA repro.

Hermeticity pins (the proven failure class is rootdir/conftest-import, NOT
PYTHONPATH - finding/tmp-agents-core-shadowing-pytest-2026-09-19):
``--confcutdir=<clone-root>`` AND ``--rootdir=<clone-root>``, explicit
PYTHONPATH reset to repo paths only, ``-p no:cacheprovider``, and the
repro's OWN sys.path shadow check (stray package roots under cwd's
ancestors) + the child-side fail-closed runtime env assertion (env-dirty
subcode).

Two-phase env: the CLONE step runs with the credentialed env (it fetches a
private org's ref); the REPRO pytest subprocess runs with an EXPLICIT
allow-list (PYTHONPATH + the interpreter path and nothing else) - it must
NOT carry FORGEJO_TOKEN, the parent-clone git creds, DOORMAN_BEARER_TOKEN,
PHALA_API_KEY, or GPU_QUEUE_BEARER_TOKEN. The repro OUTPUT is serialized
into the mem ledger + the PR body (a persistence surface), so a secret in
the child env or a test that prints os.environ would persist into the
authoritative ledger.

Location + retention: /srv/fast/lapis-repro/<target>-<head>-<epoch> (off
/tmp - the /tmp shadow vector this spec kills, and the BRIX storage rule:
bulk data off the OS root). Delete-on-verdict: the repro dir is removed
once the verdict + artifact are written. df preflight before clone;
ENOSPC is a NAMED ERROR subcode.

Budget: 2 repro runs per target per tick, defer to the next tick.
Chain-depth guard: the PRIMARY count is a persistent per-target counter in
mem (mirroring the daemon's _fixer_retry_count pattern - bumped at
salvage-open, update-in-place, no network on the hot path); the Forgejo
marker count is a RECONCILIATION-ONLY fallback (a periodic consistency
check, never the hot-path read).

Activation: LAPIS_PM_REPRO_ENFORCE=shadow (the DEFAULT post-merge) runs
every gate-red and records the repro verdict + the would-be disposition as
an observation, but the salvage path behaves EXACTLY AS IT DOES TODAY.
``on`` = full enforcement; ``off`` = today's behavior (the kill switch).
The flip is a MANUAL PM ACT.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Module-level path constants (overridable in tests by env - the repo's
# established pattern; the env reads are per-call so tests can monkeypatch
# the env without re-importing).
# ---------------------------------------------------------------------------

def _repro_root() -> Path:
    """The repro clone surface: /srv/fast/lapis-repro (off /tmp - the /tmp
    shadow vector this spec kills; bulk data off the OS root)."""
    return Path(os.environ.get("LAPIS_REPRO_ROOT", "/srv/fast/lapis-repro"))


def _quarantine_root() -> Path:
    """The quarantine surface for shadowed stale paths:
    /srv/fast/lapis-quarantine (off /tmp; "recoverable beats gone")."""
    return Path(os.environ.get("LAPIS_QUARANTINE_ROOT", "/srv/fast/lapis-quarantine"))


# The repro subprocess timeout budget: >= the gate's own 180s cap
# (the gate's [TIMEOUT after 180s] marker).
REPRO_TIMEOUT_S = int(os.environ.get("LAPIS_REPRO_TIMEOUT_S", "300"))

# Per-scope memory ceiling for the repro subprocess (finding/fixer-oom-
# 12gb-bundle-wave-2026-09-19: a 12GB ballooned process oom-killed the
# whole 30GB box; one oom-kill takes the fixer legs, the queue, and
# everything down). 4G covers a targeted pytest suite (live-check: confirm
# per-suite peak on first live run).
REPRO_MEMORY_MAX_KB = int(os.environ.get("LAPIS_REPRO_MEMORY_MAX_KB", str(4 * 1024 * 1024)))

# The 2-repro-per-target-per-tick budget (the ratified lean promoted to
# requirement text). At the live 60s tick cadence the cap sits far above
# measured demand (~38 gate-reds/day); the operative bound is the
# gate-red conversion rate.
REPRO_BUDGET_PER_TARGET_PER_TICK = int(
    os.environ.get("LAPIS_REPRO_BUDGET_PER_TICK", "2")
)

# Chain-depth guard threshold (rev 2 MEDIUM / rev 3 Facets refinement):
# the mem counter read at salvage-open time; > 3 => stop + page (Matrix
# only, deduped per target per chain - never Pushover).
CHAIN_DEPTH_STOP = int(os.environ.get("LAPIS_SALVAGE_CHAIN_DEPTH_STOP", "3"))

# Quarantine retention: age-then-purge after 7 days.
QUARANTINE_PURGE_DAYS = 7

# Preexisting-failure ledger label expiry (rev 2 M4): a label older than
# 7d is NOT re-verified per-use against base (the uncached, ownerless,
# unbounded recurring cost) - P1's repro-at-head is the verification.
PREEXISTING_LABEL_EXPIRY_DAYS = 7

# The env keys the repro subprocess must NEVER carry (two-phase env,
# rev 2 H5 + rev 3 fail-closed runtime assertion). The clone step is
# credentialed; the repro pytest subprocess runs on an explicit allow-list
# WITHOUT these.
REPRO_DISALLOWED_ENV_KEYS: tuple[str, ...] = (
    "FORGEJO_TOKEN",
    "DOORMAN_BEARER_TOKEN",
    "PHALA_API_KEY",
    "GPU_QUEUE_BEARER_TOKEN",
)

# The 5-class partition verdicts.
VERDICT_RED = "RED"
VERDICT_GREEN = "GREEN"
VERDICT_ERROR = "ERROR"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_SHA_MOVED = "SHA-MOVED"

# ERROR subcodes (the named subcode the provenance block carries).
SUBCODE_ENOSPC = "enospc"
SUBCODE_OOM = "oom"
SUBCODE_MISSING_DEPS = "missing-deps"
SUBCODE_ENV_DIRTY = "env-dirty"
SUBCODE_CLONE_FAILED = "clone-failed"
SUBCODE_TIMEOUT = "timeout"
SUBCODE_UNKNOWN = "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()



def _open_mem(mem_db_path: Path | None = None) -> "MemoryStore | None":
    """Open a MemoryStore (the mem-on-brix canon: MEM_DB_PATH). Returns
    None when the import fails (the caller degrades to the no-mem
    shape)."""
    try:
        from agents_core.mem import MemoryStore
    except Exception:
        return None
    if mem_db_path:
        return MemoryStore(db_path=mem_db_path)
    return MemoryStore()

def _log(log: Callable[[str], None] | None, message: str) -> None:
    if log:
        log(message)
    else:
        print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# The activation switch (per-tick env read - same-call-time discipline).
# shadow (the DEFAULT post-merge) = observe-only (today's behavior
# preserved, the would-be disposition is recorded as an observation);
# off = today's behavior byte-identically (the kill switch); on = full
# enforcement. The flip is a MANUAL PM ACT.
# ---------------------------------------------------------------------------

ENFORCE_SHADOW = "shadow"
ENFORCE_ON = "on"
ENFORCE_OFF = "off"


def read_enforce_mode(env: dict | None = None) -> str:
    """Read LAPIS_PM_REPRO_ENFORCE at call time.

    ``shadow`` is the DEFAULT post-merge (absent/empty/unknown value ->
    shadow - the refusal is measured, not enforced). ``on`` = full
    enforcement; ``off`` = today's behavior. Never raises.
    """
    src = os.environ if env is None else env
    val = (src.get("LAPIS_PM_REPRO_ENFORCE") or "").strip().lower()
    if val in (ENFORCE_ON, ENFORCE_OFF):
        return val
    return ENFORCE_SHADOW


# ---------------------------------------------------------------------------
# Full short-test-summary-section parse (rev 2 H2).
# ---------------------------------------------------------------------------

_FAILED_LINE_RE = re.compile(r"^(FAILED|ERROR)\s+(\S+)")
_ERROR_AT_RE = re.compile(r"^ERROR at (setup|teardown) of (\S+)")


def parse_failed_node_ids_full(output: str) -> list[str]:
    """Parse the FULL pytest output (NOT the 20-line output_tail the
    legacy extractor reads - it silently drops failure lists longer than
    20 lines) for short-test-summary failure lines.

    Returns the node-ids in first-seen order (deduped). Returns [] when
    no failure lines are visible (the collection-error / rc=4 touched-
    path-missing / rc=5 no-tests-ran / 180s-timeout shapes - the caller
    falls back to the full gate suite, never a vacuous GREEN).
    """
    node_ids: list[str] = []
    for line in (output or "").splitlines():
        line = line.strip()
        m = _ERROR_AT_RE.match(line)
        if m:
            node = m.group(2)
            if node and node not in node_ids:
                node_ids.append(node)
            continue
        m = _FAILED_LINE_RE.match(line)
        if m:
            node = m.group(2)
            if node and node not in node_ids:
                node_ids.append(node)
    return node_ids


def gate_is_red(
    last_test_outcome: dict | None,
    *,
    gate_bypassed=None,
    timeout_marker: str = "[TIMEOUT after",
) -> bool:
    """The red-outcome predicate (rev 2 P3 - replaces the presence check).

    A gate outcome is RED ONLY when the parsed summary shows failed>0 OR
    errors>0 OR a named fail-closed marker (rc in {2,4,5} per the existing
    f4-rc naming, or the [TIMEOUT after 180s] marker). A presence check
    ("a pytest summary block exists") is NOT sufficient - the 09-18
    instances had complete "N passed" summary blocks and still got the
    wrong label.
    """
    if gate_bypassed:
        return False
    if not last_test_outcome:
        # No outcome at all: the gate could not conclude green - the
        # predicate is fail-closed (red) so the repro decides.
        return True
    if int(last_test_outcome.get("failed") or 0) > 0:
        return True
    if int(last_test_outcome.get("errors") or 0) > 0:
        return True
    rc = last_test_outcome.get("returncode")
    if rc in (2, 4, 5):
        return True
    summary = str(last_test_outcome.get("summary") or "")
    output_tail = str(last_test_outcome.get("output_tail") or "")
    if timeout_marker in summary or timeout_marker in output_tail:
        return True
    return False


# ---------------------------------------------------------------------------
# D3: gate provenance block + run_not_concluded refusal.
# ---------------------------------------------------------------------------

def _sys_path_shadow_check(cwd: str) -> list[str]:
    """The repro's OWN sys.path shadow check (independent of the gate's
    start audit): stray package roots under cwd's ancestors.

    A parent directory that contains a directory with __init__.py (or a
    pyproject.toml naming the package) would be importable via pytest's
    prepend import mode - the proven failure class. Returns the list of
    offending ancestor paths (empty = clean).
    """
    import agents_core as _ac
    pkg_name = _ac.__name__.split(".")[0]
    offending: list[str] = []
    cur = Path(cwd).resolve()
    while True:
        cand = cur / pkg_name
        try:
            if cand.is_dir() and (cand / "__init__.py").is_file():
                offending.append(str(cur))
            elif (cur / "pyproject.toml").is_file():
                try:
                    text = (cur / "pyproject.toml").read_text(errors="replace")
                except OSError:
                    text = ""
                if re.search(rf'^name\s*=\s*["\']{re.escape(pkg_name)}["\']',
                             text, re.MULTILINE):
                    offending.append(str(cur))
        except OSError:
            pass
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return offending


def build_provenance_block(
    *,
    repo: str,
    head_sha: str,
    cwd: str,
    resolved_package_path: str = "",
    shadow_check: list[str] | None = None,
    confcutdir: str = "",
    quarantine_state: str = "",
) -> dict:
    """D3: the gate provenance block.

    An unaudited gate run (a missing field) yields ``run_not_concluded``,
    never ``concluded_*`` - kills both false-label classes at the source.
    The resolved-package-path field records where THIS process resolves
    the package to (the D3 live-check residual: confirm the gate runner's
    process resolves agents_core to the /srv/agents editable install).
    """
    import agents_core as _ac
    if not resolved_package_path:
        try:
            resolved_package_path = str(Path(_ac.__file__).resolve().parent)
        except Exception:
            resolved_package_path = ""
    if shadow_check is None:
        try:
            shadow_check = _sys_path_shadow_check(cwd)
        except Exception:
            shadow_check = []
    return {
        "repo": repo,
        "head_sha": head_sha,
        "cwd": cwd,
        "resolved_package_path": resolved_package_path,
        "sys_path_shadow_check": shadow_check or [],
        "confcutdir": confcutdir,
        "quarantine_state": quarantine_state,
        "ts": _now_iso(),
    }


def provenance_complete(block: dict | None) -> bool:
    """True when EVERY required provenance field is present + non-empty.

    A missing field means the gate run is unaudited: the caller yields
    ``run_not_concluded``, never ``concluded_*``.
    """
    if not isinstance(block, dict):
        return False
    for field in ("repo", "head_sha", "cwd", "resolved_package_path"):
        if not block.get(field):
            return False
    if block.get("sys_path_shadow_check"):
        # A NON-EMPTY shadow list is a refusal, not a completion.
        return False
    return True


# ---------------------------------------------------------------------------
# D3: quarantine, not delete (rev 2 M9 discipline).
# ---------------------------------------------------------------------------

def quarantine_shadow_path(
    shadow_path: str,
    quarantine_root: Path | None = None,
) -> dict:
    """Move a stale shadow path into the quarantine tree.

    On shadow detection, move the stale path to /srv/fast/lapis-quarantine/
    (off /tmp; the "recoverable beats gone" red line is satisfied MORE by
    the move, not violated). Dir 0700; resolve realpath before the move
    and REFUSE (-> run_not_concluded + log, do NOT move) any shadow whose
    realpath is outside the quarantine tree or is a symlink (a stale tree
    may carry stale secrets; a default-umask world-readable quarantine
    re-exposes them); move into a unique mktemp subdir (never a
    predictable destination filename); atomic rename within the same
    mount; age-then-purge after 7 days.

    Returns {"status": "moved"|"refused"|"absent", "dest"/"reason"}.
    Never raises.
    """
    root = quarantine_root or _quarantine_root()
    result: dict = {"status": "absent", "reason": ""}
    src = Path(shadow_path)
    if not src.exists() and not src.is_symlink():
        return result
    if src.is_symlink():
        # A symlink shadow: REFUSE (the target may be live state or carry
        # stale secrets; moving it re-exposes them).
        return {"status": "refused", "reason": "symlink"}
    try:
        real = src.resolve(strict=True)
    except OSError as exc:
        return {"status": "refused", "reason": f"unresolvable: {exc}"}
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    root_real = root.resolve(strict=True)
    try:
        real.relative_to(root_real)
        return {"status": "moved", "dest": str(real),
                "reason": "already-quarantined"}
    except ValueError:
        pass
    # Unique mktemp subdir (never a predictable destination filename).
    dest_dir = Path(tempfile.mkdtemp(prefix="q-", dir=str(root)))
    try:
        os.chmod(dest_dir, 0o700)
    except OSError:
        pass
    dest = dest_dir / src.name
    try:
        # Atomic rename within the same mount (same filesystem as root -
        # the mkdtemp above guarantees it).
        os.rename(str(src), str(dest))
    except OSError as exc:
        shutil.rmtree(str(dest_dir), ignore_errors=True)
        return {"status": "refused", "reason": f"rename-failed: {exc}"}
    return {"status": "moved", "dest": str(dest)}


def purge_quarantine(
    quarantine_root: Path | None = None,
    max_age_days: int = QUARANTINE_PURGE_DAYS,
    now: float | None = None,
) -> list[str]:
    """Age-then-purge: remove quarantine subdirs older than
    ``max_age_days``. Returns the purged paths. Never raises."""
    root = quarantine_root or _quarantine_root()
    now = now if now is not None else time.time()
    cutoff = now - max_age_days * 86400
    purged: list[str] = []
    if not root.is_dir():
        return purged
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(str(entry), ignore_errors=True)
                purged.append(str(entry))
        except OSError:
            continue
    return purged


# ---------------------------------------------------------------------------
# Two-phase env (rev 2 H5 + rev 3 fail-closed runtime assertion).
# ---------------------------------------------------------------------------

def build_repro_env(
    clone_root: str,
    *,
    host_env: dict | None = None,
    pythonpath: str | None = None,
) -> dict:
    """The REPRO pytest subprocess env: an EXPLICIT allow-list.

    (a) The CLONE step runs with the credentialed env (it fetches a
    private org's ref) - the caller's ambient env. (b) This allow-list is
    the REPRO step: PYTHONPATH (repo paths only) + the venv/interpreter
    path and NOTHING else. It must NOT carry FORGEJO_TOKEN, the
    parent-clone git creds, DOORMAN_BEARER_TOKEN, PHALA_API_KEY, or
    GPU_QUEUE_BEARER_TOKEN - justified: the clone is already on disk; the
    repro run needs no network or git auth. The repro OUTPUT is serialized
    into the mem ledger + the PR body (a persistence surface).
    """
    src = os.environ if host_env is None else host_env
    if pythonpath is None:
        pythonpath = clone_root
    env = {
        "PYTHONPATH": pythonpath,
        "PATH": src.get("PATH") or "/usr/local/bin:/usr/bin:/bin",
        "HOME": src.get("HOME") or "/root",
        "USER": src.get("USER") or "",
        "LANG": src.get("LANG") or "C.UTF-8",
        "LC_ALL": src.get("LC_ALL") or "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Belt-and-braces: the allow-list is constructed WITHOUT the
    # disallowed keys; this assertion catches a future allow-list drift.
    for key in REPRO_DISALLOWED_ENV_KEYS:
        if key in env:
            raise ValueError(f"repro env allow-list carries {key!r}")
    return env


def _child_env_guard_py() -> str:
    """The child-side fail-closed runtime env assertion (rev 3, Facets
    consensus refinement): the repro subprocess entrypoint re-checks its
    OWN os.environ at start and ABORTS with the named ERROR subcode
    (env-dirty) if any disallowed key is present in the child's
    environment - the label's provenance is verified at the point of
    execution, not at the point of construction.

    The parent-side construction test is necessary but NOT sufficient:
    this guard runs INSIDE the child, before any test runs.
    """
    keys = ",".join(REPRO_DISALLOWED_ENV_KEYS)
    return (
        "import os, sys\n"
        f"for _k in {keys!r}.split(','):\n"
        "    if _k in os.environ:\n"
        "        sys.stderr.write('repro-env-dirty: ' + _k + '\\n')\n"
        "        sys.stderr.write('lapis-repro-subcode: env-dirty\\n')\n"
        "        sys.exit(3)\n"
    )


def run_repro_pytest(
    clone_root: str,
    argv_paths: list[str],
    *,
    host_env: dict | None = None,
    timeout_s: int = REPRO_TIMEOUT_S,
    memory_max_kb: int = REPRO_MEMORY_MAX_KB,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run the repro pytest subprocess in the fresh clone.

    Hermeticity pins: ``--confcutdir=<clone-root>`` AND
    ``--rootdir=<clone-root>`` (conftest collection is cut at the clone
    boundary - the proven failure class is rootdir/conftest-import, NOT
    PYTHONPATH), ``-p no:cacheprovider``, the two-phase allow-listed env
    (build_repro_env), the child-side env-dirty guard (the fail-closed
    runtime assertion), and the per-scope MemoryMax (the OOM-lane guard -
    best-effort: an unsupported cgroup write degrades to the unbounded
    shape with a WARN, never a crash).

    Returns {"returncode", "output", "timed_out", "subcode"}.
    Never raises.
    """
    env = build_repro_env(clone_root, host_env=host_env)
    # The child entrypoint: the env-dirty guard (the fail-closed runtime
    # assertion) runs FIRST, then pytest.main with the real argv. (The
    # runpy.run_module route is NOT used: with ``-c`` the argv[0]
    # rewriting + module-name resolution drops the positional path args
    # - pytest.main with the raw argv is the reliable shape.)
    cmd = [
        sys.executable,
        "-c",
        _child_env_guard_py() + "import sys\n"
        "import pytest\n"
        "sys.exit(pytest.main(sys.argv[1:]))\n",
        "--confcutdir=" + clone_root,
        "--rootdir=" + clone_root,
        "-p", "no:cacheprovider",
        "-q",
        *argv_paths,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=clone_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
    except Exception as exc:
        _log(log, f"WARN: repro: spawn failed: {exc}")
        return {
            "returncode": -1,
            "output": f"spawn failed: {exc}",
            "timed_out": False,
            "subcode": SUBCODE_UNKNOWN,
        }
    # Per-scope MemoryMax (best-effort cgroup v2): a 12GB ballooned
    # process oom-killed the whole box on 2026-09-19; one oom-kill takes
    # the fixer legs, the queue, and everything down. An unsupported
    # cgroup write degrades to the unbounded shape (WARN, not crash).
    cgroup_path: str | None = None
    try:
        cg = Path(f"/sys/fs/cgroup/lapis-repro-{os.getpid()}-{proc.pid}")
        cg.mkdir(parents=True, exist_ok=True)
        (cg / "memory.max").write_text(f"{memory_max_kb}K")
        (cg / "cgroup.procs").write_text(str(proc.pid))
        cgroup_path = str(cg)
    except Exception as exc:
        _log(log, f"WARN: repro: MemoryMax cgroup unavailable ({exc}) - "
                  "unbounded shape")
    subcode = SUBCODE_UNKNOWN
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        subcode = SUBCODE_TIMEOUT
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        returncode = -9
    finally:
        if cgroup_path:
            try:
                shutil.rmtree(cgroup_path, ignore_errors=True)
            except Exception:
                pass
    output = (out or "") + (err or "")
    if "repro-env-dirty" in output:
        subcode = SUBCODE_ENV_DIRTY
    elif "No module named" in output and "ModuleNotFoundError" in output:
        subcode = SUBCODE_MISSING_DEPS
    elif "MemoryError" in output:
        subcode = SUBCODE_OOM
    return {
        "returncode": returncode,
        "output": output,
        "timed_out": timed_out,
        "subcode": subcode,
    }


# ---------------------------------------------------------------------------
# D4: preexisting-failure ledger filter (failure-at-head re-entry).
# ---------------------------------------------------------------------------

def _parse_ledger_date(key: str) -> str:
    """The <date> suffix of a finding/preexisting-failure/.../<date> key
    (YYYY-MM-DD; "" when absent/malformed)."""
    tail = key.rsplit("/", 1)[-1]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})$", tail)
    return m.group(1) if m else ""


def preexisting_ledger_filter(
    node_ids: list[str],
    *,
    mem_db_path: Path | None = None,
    repo: str = "",
    today: str = "",
    expiry_days: int = PREEXISTING_LABEL_EXPIRY_DAYS,
) -> dict:
    """D4: filter failures against the preexisting-failure ledger.

    Reads ``finding/preexisting-failure/<repo>/<test>/<date>`` keys (newest
    record per test wins). Re-entry rule (rev 2 M3): a filtered node-id
    that FAILS in the P1 repro re-enters the retry trigger -
    failure-at-head is the trigger; NEWness is not required (proving NEW
    would require a base run, which this spec does not perform). The
    benign direction is stated: a filtered node-id that is GREEN at head
    is a no-op - the label stands until re-verify.

    7d expiry (rev 2 M4): a label older than ``expiry_days`` is NOT
    re-verified per-use against base (the uncached, ownerless, unbounded
    recurring cost, largely redundant with P1 which already re-runs the
    node-ids at head) - P1's repro-at-head is the verification, and the
    filter decision is recorded ``stale-deferred-to-repro`` in the
    provenance block.

    Returns {"filtered": [...], "stale": [...], "unlabeled": [...]}:
    filtered = labeled + fresh (excluded from attribution); stale =
    labeled + expired (the decision is stale-deferred-to-repro);
    unlabeled = no ledger record (attributed to the PR).
    """
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store = _open_mem(mem_db_path)
    if store is None:
        return {"filtered": [], "stale": [], "unlabeled": list(node_ids)}
    filtered: list[str] = []
    stale: list[str] = []
    unlabeled: list[str] = []
    try:
        prefix = f"finding/preexisting-failure/{repo}/" if repo else (
            "finding/preexisting-failure/"
        )
        rows = store.list_by_prefix(prefix, limit=500)
        # Newest record per test wins: index by the test key
        # (finding/preexisting-failure/<repo>/<test>) -> date.
        latest: dict[str, str] = {}
        for row in rows:
            key = row.get("key") or ""
            if not key.startswith(prefix):
                continue
            test_key = key[: key.rfind("/")]
            date = _parse_ledger_date(key)
            if not date:
                continue
            if date >= latest.get(test_key, ""):
                latest[test_key] = date
        for node in node_ids:
            node_file = node.split("::")[0]
            test_key = f"{prefix}{node_file}"
            date = latest.get(test_key)
            if not date:
                unlabeled.append(node)
                continue
            try:
                label_date = datetime.strptime(date, "%Y-%m-%d")
                today_date = datetime.strptime(today, "%Y-%m-%d")
                age_days = (today_date - label_date).days
            except ValueError:
                age_days = 0
            if age_days > expiry_days:
                # stale-deferred-to-repro: P1's repro-at-head is the
                # verification; the stale label does NOT exclude.
                stale.append(node)
            else:
                filtered.append(node)
    except Exception:
        # A ledger read failure degrades to "no filter" (the attribution
        # stands - the fail-safe direction is the conservative one for
        # the label).
        return {"filtered": [], "stale": [], "unlabeled": list(node_ids)}
    finally:
        try:
            store.close()
        except Exception:
            pass
    return {"filtered": filtered, "stale": stale, "unlabeled": unlabeled}


# ---------------------------------------------------------------------------
# The repro orchestrator (D1 entrypoint).
# ---------------------------------------------------------------------------

def _df_preflight_ok(path: Path, min_free_bytes: int = 1 << 30) -> bool:
    """df preflight before clone: refuse to start when the scratch fs is
    below the floor (ENOSPC mid-clone would silently reclassifify every
    repro as infra-noise while the disk fills)."""
    try:
        usage = os.statvfs(str(path))
        free = usage.f_bavail * usage.f_frsize
        return free >= min_free_bytes
    except OSError:
        return False


def _clone_fresh(
    *,
    target: str,
    head_sha: str,
    worktree: str,
    wip_ref: str,
    clone_dest: Path,
    log: Callable[[str], None] | None,
) -> dict:
    """Phase (a): the CLONE step - runs with the CREDENTIALED env (it
    fetches a private org's ref).

    Ref source (rev 2 H4): the head SHA is typically UNPUSHED at the
    enforcement point (the gate runs before any push), so the fresh clone
    seeds from the worktree / the WIP ref (refs/wip/<task_id> - the ref
    the salvage path already resolves; it lives in the shared parent
    clone's gitdir) when the SHA is not on origin.

    Returns {"ok": bool, "head_sha": <the checked-out sha>|"",
    "subcode": ...}. Never raises.
    """
    clone_dest.mkdir(parents=True, exist_ok=True)
    if not _df_preflight_ok(clone_dest):
        _log(log, f"WARN: repro: df preflight failed at {clone_dest}")
        return {"ok": False, "head_sha": "", "subcode": SUBCODE_ENOSPC}
    git = ["git"]
    # The WIP ref lives in the shared parent clone's gitdir - seed from
    # the worktree (its gitdir) so the unpushed head is reachable.
    try:
        r = subprocess.run(
            git + ["clone", "--no-checkout", "--local", str(worktree),
                   str(clone_dest)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            shutil.rmtree(str(clone_dest), ignore_errors=True)
            _log(log, f"WARN: repro: clone failed: {r.stderr[-500:]}")
            return {"ok": False, "head_sha": "",
                    "subcode": SUBCODE_CLONE_FAILED}
        # Fetch the WIP ref (if named) from the parent gitdir.
        if wip_ref:
            subprocess.run(
                git + ["-C", str(clone_dest), "fetch", str(worktree),
                       wip_ref, "--"],
                capture_output=True, text=True, timeout=300,
            )
        # Resolve the head SHA: try the exact SHA first, then the WIP ref.
        resolved = ""
        for ref in (head_sha, wip_ref):
            if not ref:
                continue
            rr = subprocess.run(
                git + ["-C", str(clone_dest), "rev-parse", f"{ref}^{{commit}}"],
                capture_output=True, text=True, timeout=60,
            )
            if rr.returncode == 0:
                resolved = rr.stdout.strip()
                break
        if not resolved:
            shutil.rmtree(str(clone_dest), ignore_errors=True)
            _log(log, "WARN: repro: head SHA not resolvable in fresh clone")
            return {"ok": False, "head_sha": "",
                    "subcode": SUBCODE_CLONE_FAILED}
        rc = subprocess.run(
            git + ["-C", str(clone_dest), "checkout", "-q", resolved],
            capture_output=True, text=True, timeout=300,
        )
        if rc.returncode != 0:
            shutil.rmtree(str(clone_dest), ignore_errors=True)
            return {"ok": False, "head_sha": "",
                    "subcode": SUBCODE_CLONE_FAILED}
        return {"ok": True, "head_sha": resolved, "subcode": ""}
    except subprocess.TimeoutExpired:
        shutil.rmtree(str(clone_dest), ignore_errors=True)
        return {"ok": False, "head_sha": "", "subcode": SUBCODE_TIMEOUT}
    except Exception as exc:
        shutil.rmtree(str(clone_dest), ignore_errors=True)
        _log(log, f"WARN: repro: clone exception: {exc}")
        return {"ok": False, "head_sha": "", "subcode": SUBCODE_UNKNOWN}


def _sha_moved(
    worktree: str,
    wip_ref: str,
    head_sha: str,
    log: Callable[[str], None] | None,
) -> bool:
    """SHA identity (rev 2 H4): at repro time, re-fetch and verify the
    repro SHA == the current head (or the salvage branch head if
    pre-adopt). SHA-MOVED is its own class: defer to the next tick and
    re-probe the new head; NEVER classify the old gate-red from a
    moved-SHA repro (the misclassification direction is the
    merge-direction one)."""
    if not worktree:
        return False
    candidates: list[str] = []
    if wip_ref:
        candidates.append(wip_ref)
    try:
        for ref in candidates:
            r = subprocess.run(
                ["git", "-C", str(worktree), "rev-parse",
                 f"{ref}^{{commit}}"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                current = r.stdout.strip()
                if current and current != head_sha:
                    _log(log, f"WARN: repro: SHA-MOVED {head_sha[:12]} "
                              f"-> {current[:12]}")
                    return True
                return False
    except Exception:
        pass
    return False


def run_repro(
    *,
    target: str,
    head_sha: str,
    worktree: str,
    wip_ref: str = "",
    failed_node_ids: list[str] | None = None,
    full_suite_paths: list[str] | None = None,
    gate_output: str = "",
    current_head_sha: str = "",
    mem_db_path: Path | None = None,
    repro_root: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """D1 entrypoint: run the fresh-clone reproduction at the head SHA.

    Returns the repro verdict dict:
      {"verdict": RED|GREEN|ERROR|INCONCLUSIVE|SHA-MOVED,
       "subcode": <ERROR subcode or "">,
       "full_suite": bool,
       "node_ids": [...],
       "artifact": <path|"">,
       "provenance": <the D3 block>,
       "ts": <iso>}

    The 5-class partition:
    - SHA-MOVED: the head moved between gate and repro -> defer.
    - ERROR: the repro itself could not run (clone failed, ENOSPC,
      OOM, missing-deps, env-dirty).
    - INCONCLUSIVE: the repro started but timed out mid-suite (at least
      one re-run node-id untested) -> may NEVER be GREEN.
    - RED: the failed node-ids (or the full gate suite in the fallback)
      fail at the head.
    - GREEN: the FULL gate suite is green at the head (GREEN is
      issuable ONLY on the full suite - the targeted subset alone is a
      vacuous-GREEN hole).
    """
    root = repro_root or _repro_root()
    epoch = int(time.time())
    clone_dest = root / f"{target}-{head_sha[:12]}-{epoch}"
    # SHA identity check FIRST (a moved head poisons every downstream
    # classification).
    if _sha_moved(worktree, wip_ref, head_sha, log):
        return {
            "verdict": VERDICT_SHA_MOVED, "subcode": "", "full_suite": False,
            "node_ids": list(failed_node_ids or []), "artifact": "",
            "provenance": {}, "ts": _now_iso(),
        }
    # Full-summary-section parse (rev 2 H2): the FULL gate output, NOT
    # the 20-line tail. Empty/short input -> the full gate suite in the
    # fresh clone (never a vacuous GREEN).
    parsed = parse_failed_node_ids_full(gate_output)
    node_ids = list(failed_node_ids or [])
    full_suite = False
    if not parsed or (node_ids and len(parsed) < len(node_ids)):
        # Empty or lossy parse: fall back to the full gate suite.
        full_suite = True
        run_paths = list(full_suite_paths or [])
        if not run_paths:
            # No suite paths at all: the repro cannot prove anything -
            # ERROR, not a vacuous GREEN.
            return {
                "verdict": VERDICT_ERROR, "subcode": SUBCODE_UNKNOWN,
                "full_suite": False, "node_ids": node_ids, "artifact": "",
                "provenance": {}, "ts": _now_iso(),
            }
    else:
        run_paths = parsed
    clone = _clone_fresh(
        target=target, head_sha=head_sha, worktree=worktree,
        wip_ref=wip_ref, clone_dest=clone_dest, log=log,
    )
    if not clone["ok"]:
        return {
            "verdict": VERDICT_ERROR, "subcode": clone["subcode"],
            "full_suite": full_suite, "node_ids": node_ids, "artifact": "",
            "provenance": {}, "ts": _now_iso(),
        }
    provenance = build_provenance_block(
        repo=target, head_sha=clone["head_sha"],
        cwd=str(clone_dest), confcutdir=str(clone_dest),
    )
    result = run_repro_pytest(
        str(clone_dest), run_paths, log=log,
    )
    output = result["output"]
    rc = result["returncode"]
    artifact = ""
    # Write the artifact BEFORE the delete-on-verdict (keep the verdict
    # artifact; the clone dir itself is removed).
    try:
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / f"{target}-{head_sha[:12]}-{epoch}.txt"
        artifact.write_text(
            f"verdict-pending\nhead: {clone['head_sha']}\n"
            f"full_suite: {full_suite}\nrun_paths: {run_paths}\n"
            f"returncode: {rc}\n--- output ---\n{output}\n",
            encoding="utf-8",
        )
    except OSError:
        artifact = ""
    # Delete-on-verdict retention: the clone dir is removed once the
    # verdict + artifact are written.
    shutil.rmtree(str(clone_dest), ignore_errors=True)
    # Classify.
    if result["subcode"] in (SUBCODE_ENV_DIRTY, SUBCODE_OOM,
                             SUBCODE_MISSING_DEPS, SUBCODE_ENOSPC):
        verdict = VERDICT_ERROR
        subcode = result["subcode"]
    elif result["timed_out"] or rc == -9:
        # The mid-suite timeout/OOM shape: at least one re-run node-id is
        # untested at verdict time -> INCONCLUSIVE (may NEVER be GREEN).
        verdict = VERDICT_INCONCLUSIVE
        subcode = result["subcode"] or SUBCODE_TIMEOUT
    elif rc == 0:
        if full_suite:
            verdict = VERDICT_GREEN
        else:
            # A green run of ONLY the targeted subset is not a GREEN
            # verdict (the vacuous-GREEN hole) - re-run the full suite.
            if not full_suite_paths:
                verdict = VERDICT_ERROR
                subcode = SUBCODE_UNKNOWN
            else:
                clone2 = _clone_fresh(
                    target=target, head_sha=head_sha, worktree=worktree,
                    wip_ref=wip_ref,
                    clone_dest=root / f"{target}-{head_sha[:12]}-{epoch}-full",
                    log=log,
                )
                if not clone2["ok"]:
                    verdict = VERDICT_ERROR
                    subcode = clone2["subcode"]
                else:
                    r2 = run_repro_pytest(
                        str(root / f"{target}-{head_sha[:12]}-{epoch}-full"),
                        list(full_suite_paths), log=log,
                    )
                    shutil.rmtree(
                        str(root / f"{target}-{head_sha[:12]}-{epoch}-full"),
                        ignore_errors=True,
                    )
                    if r2["timed_out"] or r2["returncode"] == -9:
                        verdict = VERDICT_INCONCLUSIVE
                        subcode = r2["subcode"] or SUBCODE_TIMEOUT
                    elif r2["returncode"] == 0:
                        verdict = VERDICT_GREEN
                    else:
                        verdict = VERDICT_RED
                        subcode = ""
                    if r2["output"]:
                        try:
                            if artifact:
                                Path(artifact).write_text(
                                    Path(artifact).read_text()
                                    + f"--- full-suite output ---\n"
                                    + r2["output"] + "\n",
                                    encoding="utf-8",
                                )
                        except OSError:
                            pass
    else:
        # rc != 0 and not timed out: the re-run failed at the head.
        # (A subset failure is RED; the full-suite fallback failure is
        # RED.)
        verdict = VERDICT_RED
        subcode = ""
    return {
        "verdict": verdict, "subcode": subcode,
        "full_suite": full_suite, "node_ids": node_ids or run_paths,
        "artifact": str(artifact) if artifact else "",
        "provenance": provenance, "ts": _now_iso(),
    }


# ---------------------------------------------------------------------------
# D5: salvage title/label truth + chain-depth guard + gate-noise counter.
# ---------------------------------------------------------------------------

def repro_verdict_title_tag(verdict: str) -> str:
    """D5: the repro verdict in the salvage title.

    [REPRO-RED] / [REPRO-GREEN] / [REPRO-ERROR]. INCONCLUSIVE and
    SHA-MOVED open no PR - they write an observation and defer.
    """
    if verdict == VERDICT_RED:
        return "[REPRO-RED]"
    if verdict == VERDICT_GREEN:
        return "[REPRO-GREEN]"
    if verdict == VERDICT_ERROR:
        return "[REPRO-ERROR]"
    return ""


def repro_body_block(verdict: dict) -> str:
    """The repro verdict block that rides in the PR body (the fixer sees
    the actual failing invocation). The daemon-leg consumption of this
    block by _act_dispatch_fixer_retry's payload is a FOLLOW-UP lapis-pm
    target (the payload is built from the reviewer's issues list - a
    cross-repo seam, out of scope for this agents-core unit)."""
    v = (verdict or {}).get("verdict") or ""
    lines = [
        "<!-- lapis-repro: start -->",
        f"verdict: {v}",
        f"subcode: {(verdict or {}).get('subcode') or ''}",
        f"full_suite: {(verdict or {}).get('full_suite')}",
        f"node_ids: {(verdict or {}).get('node_ids')}",
        f"artifact: {(verdict or {}).get('artifact') or ''}",
        f"ts: {(verdict or {}).get('ts') or ''}",
        "<!-- lapis-repro: end -->",
    ]
    return "\n".join(lines)


# The per-target repro budget: a module-level per-tick counter (the
# 2-repro-per-target-per-tick budget with defer-to-next-tick). The
# caller resets it each tick via reset_repro_budget().
_REPRO_BUDGET: dict[str, int] = {}
_REPRO_TICK: str = ""


def _tick_id() -> str:
    """A per-tick identity: the minute-granularity wall clock (the live
    60s tick cadence)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def reset_repro_budget() -> None:
    """Reset the per-target repro budget at the start of each tick."""
    global _REPRO_TICK
    _REPRO_TICK = _tick_id()
    _REPRO_BUDGET.clear()


def repro_budget_available(target: str) -> bool:
    """True when the target has repro budget left this tick (2/tick)."""
    tick = _tick_id()
    if tick != _REPRO_TICK:
        # A new tick: the budget resets.
        reset_repro_budget()
    used = _REPRO_BUDGET.get(target, 0)
    return used < REPRO_BUDGET_PER_TARGET_PER_TICK


def consume_repro_budget(target: str) -> None:
    """Consume one unit of the target's repro budget this tick."""
    global _REPRO_TICK
    tick = _tick_id()
    if tick != _REPRO_TICK:
        _REPRO_TICK = tick
        _REPRO_BUDGET.clear()
    _REPRO_BUDGET[target] = _REPRO_BUDGET.get(target, 0) + 1


def salvage_chain_depth(
    target: str,
    *,
    mem_db_path: Path | None = None,
) -> int:
    """D5: the chain-depth guard - the PRIMARY count is a persistent
    per-target counter in mem (mirroring the daemon's _fixer_retry_count
    pattern - bumped at salvage-open, update-in-place, no network on the
    hot path). The Forgejo API count of open + closed PRs carrying the
    ``<!-- lapis-salvage: true -->`` body marker is a RECONCILIATION-ONLY
    fallback (a periodic consistency check - if the API count exceeds the
    mem counter, the mem counter is corrected UP and a divergence row is
    written). The guard reads the mem counter at salvage-open time -
    EXPLICITLY not title parsing (the API path would introduce a network
    failure mode on the hot salvage path)."""
    key = f"pm/salvage-chain/{target}"
    store = _open_mem(mem_db_path)
    if store is None:
        return 0
    try:
        rows = store.list_by_prefix(key, limit=5)
        for row in rows:
            if row.get("key") == key:
                try:
                    val = json.loads(row.get("content") or "{}")
                    return int(val.get("count") or 0)
                except (ValueError, TypeError):
                    return 0
        return 0
    except Exception:
        return 0
    finally:
        try:
            store.close()
        except Exception:
            pass


def bump_salvage_chain(
    target: str,
    *,
    mem_db_path: Path | None = None,
) -> int:
    """Bump the persistent per-target salvage-chain counter at
    salvage-open (update-in-place - bounded rows, no append-only growth).
    Returns the new count."""
    key = f"pm/salvage-chain/{target}"
    store = _open_mem(mem_db_path)
    if store is None:
        return 0
    try:
        current = salvage_chain_depth(target, mem_db_path=mem_db_path)
        new_count = current + 1
        store.set(
            key,
            json.dumps({"count": new_count, "ts": _now_iso()}),
        )
        return new_count
    except Exception:
        return 0
    finally:
        try:
            store.close()
        except Exception:
            pass


def reconcile_salvage_chain(
    target: str,
    forgejo_pr_count: int,
    *,
    mem_db_path: Path | None = None,
) -> bool:
    """The RECONCILIATION-ONLY fallback (a periodic consistency check):
    if the Forgejo marker count exceeds the mem counter, the mem counter
    is corrected UP and a divergence row is written. Returns True when a
    divergence was found + corrected."""
    key = f"pm/salvage-chain/{target}"
    mem_count = salvage_chain_depth(target, mem_db_path=mem_db_path)
    if forgejo_pr_count <= mem_count:
        return False
    store = _open_mem(mem_db_path)
    if store is None:
        return False
    try:
        store.set(
            key,
            json.dumps({"count": forgejo_pr_count, "ts": _now_iso(),
                        "reconciled": True}),
        )
        store.set(
            f"pm/salvage-chain-divergence/{target}",
            json.dumps({"mem": mem_count, "forgejo": forgejo_pr_count,
                        "ts": _now_iso()}),
        )
        return True
    except Exception:
        return False
    finally:
        try:
            store.close()
        except Exception:
            pass


def chain_depth_exceeded(
    target: str,
    *,
    depth: int | None = None,
    stop: int = CHAIN_DEPTH_STOP,
    mem_db_path: Path | None = None,
) -> bool:
    """True when the chain depth exceeds the stop threshold (> 3 => stop
    + brief + one Matrix page per target per chain - never Pushover).
    Chains already at depth >= 4 page ONCE on enablement day - one-time,
    expected, not a storm."""
    if depth is None:
        depth = salvage_chain_depth(target, mem_db_path=mem_db_path)
    return depth > stop


def gate_noise_count(
    repo: str,
    *,
    delta: int = 1,
    mem_db_path: Path | None = None,
) -> int:
    """The gate-noise counter key ``pm/gate-noise/<repo>`` (update-in-
    place, bounded rows - mem.db is the authoritative ledger on a
    15G-free root; no append-only low-value growth). The ERROR /
    INCONCLUSIVE / SHA-MOVED observations count toward it with the named
    subcode. Returns the new count."""
    key = f"pm/gate-noise/{repo}"
    store = _open_mem(mem_db_path)
    if store is None:
        return 0
    try:
        rows = store.list_by_prefix(key, limit=5)
        current = 0
        for row in rows:
            if row.get("key") == key:
                try:
                    current = int(json.loads(row.get("content") or "{}")
                                   .get("count") or 0)
                except (ValueError, TypeError):
                    current = 0
                break
        new_count = current + delta
        store.set(key, json.dumps({"count": new_count, "ts": _now_iso()}))
        return new_count
    except Exception:
        return 0
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shadow-mode observation writer (rev 3: the would-be disposition is
# recorded as an observation - the `pm/repro-shadow/<tid>/<head>` row +
# the gate-noise counter).
# ---------------------------------------------------------------------------

def write_shadow_observation(
    *,
    target: str,
    head_sha: str,
    verdict: dict,
    would_be_disposition: str,
    mem_db_path: Path | None = None,
) -> str:
    """Shadow mode (LAPIS_PM_REPRO_ENFORCE=shadow, the DEFAULT post-
    merge): P1 runs every gate-red and records the repro verdict + the
    would-be disposition (label / clean-push / salvage-green / ERROR /
    INCONCLUSIVE) as an observation, but the salvage path behaves EXACTLY
    AS IT DOES TODAY. Nothing is paused, nothing is dropped, nothing is
    blocked - the refusal is measured, not enforced.

    Returns the observation key ("" on write failure).
    """
    key = f"pm/repro-shadow/{target}/{head_sha[:12]}"
    store = _open_mem(mem_db_path)
    if store is None:
        return ""
    try:
        row = {
            "verdict": (verdict or {}).get("verdict") or "",
            "subcode": (verdict or {}).get("subcode") or "",
            "would_be_disposition": would_be_disposition,
            "ts": _now_iso(),
        }
        store.set(key, json.dumps(row))
        return key
    except Exception:
        return ""
    finally:
        try:
            store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# D6: the day-surface census line (the repro-green rate + the per-repo
# gate-noise count + the current LAPIS_PM_REPRO_ENFORCE value - a missing
# or drifted value is visible, not silent).
# ---------------------------------------------------------------------------

def day_surface_line(
    *,
    enforce_mode: str | None = None,
    repro_green_rate: float | None = None,
    gate_noise: dict[str, int] | None = None,
    repro_verdict_counts: dict[str, int] | None = None,
    env_dirty_aborts: int = 0,
) -> str:
    """D6: the day-surface line (the Nzinga loud-trace pattern).

    Carries: the current LAPIS_PM_REPRO_ENFORCE value (a missing or
    drifted value is visible, not silent - the named field the
    activation switch promises), the repro-green rate, the per-repo
    gate-noise count (a counter nobody renders violates the
    agent-as-destination flame-shape invariant), the repro-verdict
    distribution, and the env-dirty abort count (must be zero - the
    fail-closed runtime assertion never fires on a clean box).
    """
    if enforce_mode is None:
        enforce_mode = read_enforce_mode()
    parts = [
        f"lapis-repro: enforce={enforce_mode}",
    ]
    if repro_green_rate is not None:
        parts.append(f"repro-green-rate={repro_green_rate:.2f}")
    if repro_verdict_counts:
        dist = ",".join(
            f"{k}={v}" for k, v in sorted(repro_verdict_counts.items())
        )
        parts.append(f"verdicts[{dist}]")
    if gate_noise:
        noise = ",".join(f"{k}:{v}" for k, v in sorted(gate_noise.items()))
        parts.append(f"gate-noise[{noise}]")
    parts.append(f"env-dirty-aborts={env_dirty_aborts}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# D2: the salvage-labeling enforcement (the orchestration entrypoint).
# ---------------------------------------------------------------------------

def salvage_label_decision(
    *,
    target: str,
    head_sha: str,
    gate_passed: bool,
    last_test_outcome: dict | None,
    gate_output: str = "",
    failed_node_ids: list[str] | None = None,
    full_suite_paths: list[str] | None = None,
    wip_commit_count: int = 0,
    empty_diff: bool = False,
    head_past_base: bool = False,
    worktree: str = "",
    wip_ref: str = "",
    current_head_sha: str = "",
    mem_db_path: Path | None = None,
    log: Callable[[str], None] | None = None,
    run_repro_fn: Callable[..., dict] | None = None,
) -> dict:
    """D2: the enforcement at the SALVAGE-LABELING moment.

    Returns a decision dict:
      {"stop_reason": <the stop_reason the salvage path emits>,
       "title_tag": <the [REPRO-*] title tag or "">,
       "body_block": <the <!-- lapis-repro: ... --> block or "">,
       "disposition": <clean-push | salvage-red | salvage-green |
                       salvage-error | defer-inconclusive |
                       defer-sha-moved | run-not-concluded |
                       concluded-gate-rejected-legacy>,
       "repro": <the repro verdict dict or {}>,
       "shadow_observation": <the observation key or "">,
       "enforce_mode": <shadow|on|off>}

    The gate_passed branch (class 4a fix): gate-PASSED + concluded +
    WIP + empty-diff routes to the CLEAN-PUSH disposition, NEVER
    concluded_gate_rejected.

    In shadow mode (the DEFAULT post-merge) the salvage path behaves
    EXACTLY AS IT DOES TODAY (the concluded_gate_rejected label emits as
    before, including the class-4a mislabel - which the shadow
    observation quantifies live); the repro verdict + the would-be
    disposition are recorded as observations only.

    In off mode (the kill switch) the salvage path is byte-identical to
    today's behavior (no repro run at all).

    In on mode the full P1/P2/P3 enforcement applies: no
    concluded_gate_rejected label without [REPRO-RED]; the red-outcome
    predicate gates the label; the 2-per-target-per-tick budget defers
    to the next tick; the chain-depth guard stops at > 3.
    """
    mode = read_enforce_mode()
    decision: dict = {
        "stop_reason": "concluded_gate_rejected",
        "title_tag": "",
        "body_block": "",
        "disposition": "concluded-gate-rejected-legacy",
        "repro": {},
        "shadow_observation": "",
        "enforce_mode": mode,
    }

    # The class-4a shape: gate-PASSED + concluded + WIP + empty-diff.
    # Today's behavior: the concluded_gate_rejected label emits
    # UNCONDITIONALLY (the mislabel - the 09-18 finding's 3rd instance).
    # The would-be disposition: the CLEAN-PUSH disposition.
    class4a = (
        gate_passed
        and wip_commit_count > 0
        and empty_diff
        and head_past_base
    )

    if mode == ENFORCE_OFF:
        # The kill switch: today's behavior byte-identically (no repro
        # run, no observation - the salvage path is untouched).
        return decision

    # shadow + on: run the repro (budget permitting).
    if not repro_budget_available(target):
        # Budget exhaustion: defer to the next tick (the observation is
        # written; no PR, no label decision this tick).
        decision["disposition"] = "defer-budget"
        decision["stop_reason"] = "defer-budget"
        _log(log, f"WARN: repro: budget exhausted for {target} - "
                  "defer to next tick")
        return decision

    consume_repro_budget(target)
    repro_fn = run_repro_fn or run_repro
    try:
        verdict = repro_fn(
            target=target,
            head_sha=head_sha,
            worktree=worktree,
            wip_ref=wip_ref,
            failed_node_ids=failed_node_ids,
            full_suite_paths=full_suite_paths,
            gate_output=gate_output,
            current_head_sha=current_head_sha,
            mem_db_path=mem_db_path,
            log=log,
        )
    except Exception as exc:
        # A repro harness crash is infra-noise: no defect label, no
        # cycle consumed.
        verdict = {
            "verdict": VERDICT_ERROR, "subcode": SUBCODE_UNKNOWN,
            "full_suite": False, "node_ids": [], "artifact": "",
            "provenance": {}, "ts": _now_iso(),
        }
        _log(log, f"WARN: repro: harness exception: {exc}")
    decision["repro"] = verdict
    v = verdict.get("verdict") or ""
    subcode = verdict.get("subcode") or ""

    # The would-be disposition (what enforcement WOULD do).
    if v == VERDICT_SHA_MOVED:
        would_be = "defer-sha-moved"
    elif v == VERDICT_INCONCLUSIVE:
        would_be = "defer-inconclusive"
    elif v == VERDICT_ERROR:
        would_be = "salvage-error"
    elif class4a:
        would_be = "clean-push"
    elif v == VERDICT_RED:
        would_be = "salvage-red"
    elif v == VERDICT_GREEN:
        would_be = "salvage-green"
    else:
        would_be = "concluded-gate-rejected-legacy"

    # The gate-noise counter (ERROR / INCONCLUSIVE / SHA-MOVED count
    # toward it with the named subcode).
    if v in (VERDICT_ERROR, VERDICT_INCONCLUSIVE, VERDICT_SHA_MOVED):
        gate_noise_count(target, mem_db_path=mem_db_path)

    if mode == ENFORCE_SHADOW:
        # Observe-only: the salvage path behaves EXACTLY AS IT DOES
        # TODAY (the concluded_gate_rejected label emits as before,
        # including the class-4a mislabel); the repro verdict + the
        # would-be disposition are recorded as observations.
        decision["shadow_observation"] = write_shadow_observation(
            target=target, head_sha=head_sha, verdict=verdict,
            would_be_disposition=would_be, mem_db_path=mem_db_path,
        )
        # The today-behavior label stands (the mislabel quantified live).
        return decision

    # mode == ENFORCE_ON: full enforcement.
    if v == VERDICT_SHA_MOVED:
        decision["disposition"] = "defer-sha-moved"
        decision["stop_reason"] = "defer-sha-moved"
        decision["title_tag"] = ""
        return decision
    if v == VERDICT_INCONCLUSIVE:
        # INCONCLUSIVE may NEVER be GREEN; consumes the budget (already
        # consumed); defers to the next tick.
        decision["disposition"] = "defer-inconclusive"
        decision["stop_reason"] = "defer-inconclusive"
        decision["title_tag"] = ""
        return decision
    if v == VERDICT_ERROR:
        # Infra-noise classification: NO salvage PR with a defect
        # label, NO review cycle consumed. The subcode is in the
        # provenance.
        decision["disposition"] = "salvage-error"
        decision["stop_reason"] = f"repro-error-{subcode or 'unknown'}"
        decision["title_tag"] = repro_verdict_title_tag(v)
        decision["body_block"] = repro_body_block(verdict)
        return decision
    if class4a:
        # The class-4a shape: the CLEAN-PUSH disposition, NEVER
        # concluded_gate_rejected.
        decision["disposition"] = "clean-push"
        decision["stop_reason"] = "clean-push"
        decision["title_tag"] = ""
        return decision
    if v == VERDICT_RED:
        # The label is EARNED: the repro proved the red at the head.
        decision["disposition"] = "salvage-red"
        decision["stop_reason"] = "concluded_gate_rejected"
        decision["title_tag"] = repro_verdict_title_tag(v)
        decision["body_block"] = repro_body_block(verdict)
        return decision
    if v == VERDICT_GREEN:
        # The gate-RED-but-repro-GREEN shape: the SALVAGE-GREEN path -
        # open the PR as [SALVAGE] with [REPRO-GREEN] in the title
        # (the advisory PR is NEVER auto-merged - the merge is a PM
        # act, and the PM merge gate re-runs the suite at head).
        decision["disposition"] = "salvage-green"
        decision["stop_reason"] = "salvage-green"
        decision["title_tag"] = repro_verdict_title_tag(v)
        decision["body_block"] = repro_body_block(verdict)
        return decision
    # Fallback: a verdict we did not recognize - the conservative
    # direction is the today-behavior label (never a silent drop).
    decision["disposition"] = "concluded-gate-rejected-legacy"
    return decision
