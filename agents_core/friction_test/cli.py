"""agents_core.friction_test.cli — CLI entry point (callable from shim).

Invoked via: python3 /srv/agents/scripts/friction_test.py run --target ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import run


def main(argv: list[str] | None = None) -> int:
    """Parse args, call orchestrator.run(), print summary, return exit code."""
    parser = argparse.ArgumentParser(
        prog="friction-test",
        description="Friction Tester v0 — Dissonance Engine for agent-operable flows.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run friction test against a target")
    run_p.add_argument(
        "--target",
        required=True,
        choices=["radio-op", "cockpit"],
        help="Target to test",
    )
    run_p.add_argument(
        "--scenario-set",
        default="smoke",
        choices=["smoke"],
        help="Scenario set to run (v0: smoke only)",
    )
    run_p.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        metavar="N",
        help="Cap on scenarios per family",
    )
    run_p.add_argument(
        "--invariant-mode",
        default="declared",
        choices=["declared", "inferred", "both"],
        help="Which invariants to run",
    )
    run_p.add_argument(
        "--target-base-url",
        default=None,
        help="Override driver's default base URL",
    )
    run_p.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Override vault output directory",
    )
    run_p.add_argument(
        "--qwen-endpoint",
        default="http://203.0.113.12:8081/v1/chat/completions",
        help="Qwen API endpoint",
    )
    run_p.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit 1 on harness errors (target unreachable, vault_writer failure)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        report = run(
            target=args.target,
            scenario_set=args.scenario_set,
            n_max=args.n_scenarios,
            invariant_mode=args.invariant_mode,
            target_base_url=args.target_base_url,
            out_dir=args.out_dir,
            qwen_endpoint=args.qwen_endpoint,
            strict=args.strict,
        )
    except Exception as exc:
        if args.strict:
            print(f"ERROR: harness error — {exc}", file=sys.stderr)
            return 1
        print(f"WARNING: harness error — {exc}", file=sys.stderr)
        return 0

    # Check for harness errors in report
    harness_errors = sum(1 for o in report.observations if o.harness_error)

    # Summary
    nd = report.n_dissonances
    print(
        f"friction-test {report.target} {report.scenario_set} | "
        f"scenarios={report.n_scenarios} "
        f"invariants={report.n_invariants_declared} "
        f"inferred={report.n_invariants_inferred} | "
        f"dissonances: system_likely={nd.get('system_likely',0)} "
        f"model_likely={nd.get('model_likely',0)} "
        f"total={nd.get('total',0)} | "
        f"report={report.md_path}"
    )

    if report.harness_warnings:
        for w in report.harness_warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    if args.strict and harness_errors:
        return 1

    return 0
