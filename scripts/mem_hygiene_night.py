#!/usr/bin/env python3
"""Mem-Hygiene Night Node — bounded dead-stream quarantine for mem.db.

Deterministic, no-LLM, BRIX-local. Runs as a shell-lane command node in the
night DAG (see lapis-spec.md, target mem-hygiene-automation-v0; the
clone-currency-sync family: --dry-run, atomic artifact, exit-code mapping).

Pipeline (one scheduled weekly run):
  1. Load the machine-state allowlist + dead-producer registry from the
     MEM_HYGIENE_CONFIG-named config (explicit config, never
     auto-inferred). The default is the deployed repo copy
     (/srv/agents/agents_core/config/mem_hygiene.json).
  2. Dry-run FIRST: classify every allowlisted prefix against the D2
     dead-stream predicate (allowlisted + MAX(updated_at) older than N
     days across BOTH stores + sources registered-dead, fail-closed) and
     write the candidate artifact
     /data/slots/mem-hygiene-candidates-<date>-<runid>.json (atomic).
  3. If candidates exist and the batch cap is not exceeded: quarantine in
     one transaction (INSERT OR IGNORE into memories_quarantine + DELETE
     from memories — trigger-covered FTS), with the count-mismatch abort
     and the FTS integrity halt.
  4. Age out quarantined rows past the rollback window (default 14 days).
  5. Deposit the provenance line:
       key:   decision/mem-hygiene-run-<date>-<runid>   (per-RUN key —
              mem set upserts; a per-date key would be silently
              overwritten on multi-run days)
       tags:  lapis-pm,mem-hygiene  (the 08:00 PT Lapis PM morning brief
              ingests mem.list_all(tag="lapis-pm") filtered to decision/,
              truncated to 120 chars — the one-liner budget is <=120)
     via the `mem` CLI (HTTP when MEM_SERVER is set, direct store
     otherwise).

Exit codes: 0 = clean (including a no-op run with zero candidates);
1 = any step failed or the run aborted.

--dry-run: steps 2 only — candidate artifact written, no mutation, no
deposit. Same exit-code mapping.

Invariants: one transaction per run (crash = clean rollback); nothing
atom-class is ever auto-mutated (D6 backstop in the library); all
memories mutations are trigger-covered DML; the scheduled path is bounded
by the batch cap (above-cap runs are the first pass's job, run manually
with the db-file backup).

Usage:
    python3 mem_hygiene_night.py              # real run
    python3 mem_hygiene_night.py --dry-run    # report only
    python3 mem_hygiene_night.py --config FILE --slots-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CONFIG = "/srv/agents/agents_core/config/mem_hygiene.json"
DEFAULT_SLOTS_DIR = Path("/data/slots")
DEFAULT_MEM_CLI = "/usr/local/bin/mem"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _log(msg: str) -> None:
    print(f"[mem-hygiene-night] {msg}", flush=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic write: tmp file in the same directory, then os.rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _mem_set(key: str, content: str, tags: str, mem_cli: str) -> None:
    """Deposit a provenance line via the `mem` CLI.

    The CLI routes through mem-server (MEM_SERVER) or the direct store;
    either way the row lands in the store the morning brief reads.
    """
    cmd = [mem_cli, "set", key, content, "--tags", tags]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`mem set {key}` failed (rc={proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def _run_pipeline(config: str, slots_dir: Path, dry_run: bool,
                   mem_cli: str) -> tuple[dict, int]:
    """Execute the hygiene pipeline. Returns (report, exit_code)."""
    os.environ.setdefault("MEM_HYGIENE_CONFIG", config)

    from agents_core.mem import MemoryStore  # noqa: PLC0415
    from agents_core.mem_hygiene import (  # noqa: PLC0415
        HygieneConfig,
        HygieneAborted,
        MemHygieneRunner,
    )

    cfg = HygieneConfig.load(config)
    run_id = "hyg-night-" + _utcnow().strftime("%Y%m%dT%H%M%SZ")
    date = _utcnow().strftime("%Y-%m-%d")
    store = MemoryStore()
    report: dict = {
        "run_id": run_id,
        "ts_utc": _utcnow().isoformat(),
        "mode": "dry-run" if dry_run else "run",
        "config": config,
        "steps": {},
    }
    exit_code = 0
    try:
        runner = MemHygieneRunner(store, cfg, run_id=run_id,
                                  artifact_dir=slots_dir)

        # Step 1+2: dry-run classification + candidate artifact.
        candidates, prefixes = runner.list_candidates()
        eligible = [p for p in prefixes if p.eligible]
        report["steps"]["classify"] = {
            "eligible": [p.prefix for p in eligible],
            "ineligible": [
                {"prefix": p.prefix, "reason": p.reason}
                for p in prefixes if not p.eligible
            ],
            "candidate_count": len(candidates),
        }
        # The artifact is the dry-run snapshot of the candidate set, but
        # its mode field records THIS run's mode — on a real run the
        # artifact is written ahead of the mutation and the mutation
        # quarantines this exact enumeration, so the artifact is the
        # pre-mutation snapshot of a run, not a dry-run (reviewer medium,
        # PR #331 cycle 1).
        artifact = runner.write_candidate_artifact(
            candidates, eligible,
            mode="dry-run" if dry_run else "run",
        )
        report["steps"]["candidate_artifact"] = artifact
        _log(f"candidate artifact: {artifact} ({len(candidates)} rows)")

        if dry_run:
            report["steps"]["mutation"] = "skipped (dry-run)"
            return report, exit_code

        # Step 3: quarantine (one transaction; cap-bounded scheduled path).
        # Pass the Step 1+2 enumeration THROUGH (reviewer medium, PR #329
        # cycle 1): the mutation operates on the SAME candidate set the
        # artifact was written from — no double-classification, so the
        # store cannot drift between the artifact and the quarantine.
        # (The count-mismatch guard still fires on mid-run drift.)
        try:
            verdict = runner.run_quarantine(
                dry_run=False, allow_over_cap=False,
                candidates=candidates, prefixes=prefixes,
            )
            report["steps"]["quarantine"] = {
                "quarantined": verdict.quarantined,
                "already_quarantined": verdict.already_quarantined,
                "fts_integrity_ok": verdict.fts_integrity_ok,
                "db_row_count_after": verdict.db_row_count_after,
            }
            _log(f"quarantine: {verdict.quarantined} quarantined, "
                 f"{verdict.already_quarantined} already quarantined")
        except HygieneAborted as exc:
            report["steps"]["quarantine"] = {"aborted": str(exc)}
            _log(f"quarantine ABORTED: {exc}")
            return report, 1

        # Step 4: age out past the rollback window.
        purged = runner.ageout()
        report["steps"]["ageout"] = {"purged": purged,
                                     "window_days": cfg.rollback_window_days}
        _log(f"ageout: {purged} purged (>{cfg.rollback_window_days}d)")

        # Step 5: provenance deposit (per-RUN key, <=120 char one-liner).
        stats = runner.quarantine_stats()
        one_liner = (
            f"hygiene: {report['steps']['quarantine']['quarantined']} quarantined, "
            f"{report['steps']['quarantine']['already_quarantined']} already aged out - "
            f"deleted; db at {report['steps']['quarantine']['db_row_count_after']} rows"
        )
        if len(one_liner) > 120:
            one_liner = one_liner[:117] + "..."
        decision_key = f"decision/mem-hygiene-run-{date}-{run_id}"
        ineligible_str = "; ".join(
            f"{i['prefix']} ({i['reason']})"
            for i in report["steps"]["classify"]["ineligible"]
        ) or "none"
        content = (
            f"{one_liner}\n\n"
            f"run_id: {run_id}\n"
            f"eligible: {', '.join(report['steps']['classify']['eligible']) or 'none'}\n"
            f"ineligible: {ineligible_str}\n"
            f"candidate_artifact: {artifact}\n"
            f"quarantine_total: {stats['total']}\n"
            f"fts_integrity_ok: {report['steps']['quarantine']['fts_integrity_ok']}"
        )
        _mem_set(decision_key, content, "lapis-pm,mem-hygiene", mem_cli)
        report["steps"]["deposit"] = decision_key
        _log(f"provenance line deposited: {decision_key}")
    except Exception as exc:  # noqa: BLE001 - the node maps any failure to rc=1
        report["steps"]["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED: {exc}")
        exit_code = 1
    finally:
        store.close()
    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded dead-stream quarantine run for mem.db "
                    "(mem-hygiene-automation-v0 night node)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="classify + write the candidate artifact only; no mutation, "
             "no provenance deposit",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="named hygiene config file (default: the deployed repo copy "
             f"{DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--slots-dir",
        default=str(DEFAULT_SLOTS_DIR),
        help="directory for the candidate artifact (default: /data/slots)",
    )
    parser.add_argument(
        "--mem-cli",
        default=DEFAULT_MEM_CLI,
        help="path to the mem CLI for the provenance deposit "
             "(default: /usr/local/bin/mem)",
    )
    args = parser.parse_args(argv)

    report, exit_code = _run_pipeline(
        config=args.config,
        slots_dir=Path(args.slots_dir),
        dry_run=args.dry_run,
        mem_cli=args.mem_cli,
    )
    if args.dry_run:
        print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
