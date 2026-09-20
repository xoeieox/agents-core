"""mem CLI — the `promote` subcommand + `--store` flag + `MEM_PRINCIPAL`
env default (openclaw-memdb-influx-reader-v0, D2 / Files-changed / panel F7).

This module is the AGENTS-CORE half of the spec's Files-changed line. The
LIVE CLI surface is `conductor/scripts/mem.py` (a separate repo); the
conductor change is a thin dispatch to this module (one `import` + one
`add_parser`), which is the conductor PR's job. This module ships the
argparse wiring + the HTTP dispatch so the curation surface
(`mem promote`) is exercisable from agents-core alone, and so the
`MEM_PRINCIPAL` env default is named + tested here (the BRIX-side
principal that covers the mem CLI + conductor node scripts + PM machinery,
D1).

The `promote` verb (D2):

    mem promote <key> [--content <body>] --from <agent-store-ref>
                      [--by <curator-principal>] [--tags tag1,tag2]
                      [--rationale one-line-why] [--store atoms|machinery]

    (argparse: all options come AFTER the positional <key>; the optional
    body is the --content option, not a positional — see build_parser().)

Routes through MemClient when MEM_SERVER is set (the HTTP path — the
curation surface runs BRIX-side against the BRIX master). When MEM_SERVER
is unset, `promote` is REFUSED (loud exit 2): promotion is an explicit,
server-side verb with a provenance shape the server owns (the server
builds the exact header + the batch decision key); a local
MemoryStore-only promote would bypass the server's --from shape check +
batch key, which is exactly the side-effect the spec forbids ("promotion
is an explicit verb with a provenance shape, never a side effect").

`--store` (D2 / Files-changed): RESCOPED (rev-2) — the machinery store is
the EXISTING exhaust store, so the flag is MOOT at the HTTP layer (the
server routes machine-state keys transparently). It is validated (loud
ValueError on a typo) and accepted so the spec's Files-changed line is
honored; it does NOT change routing.

`MEM_PRINCIPAL` env default (D1 / panel F7): the BRIX-side principal that
covers the mem CLI. When MEM_PRINCIPAL is unset and the command is a WRITE
(promote), the CLI defaults to `brix-pm` (the BRIX-resident curator
principal) so the primary local path (the mem CLI over the tailscale IP)
lands as the registered writer — NOT as a reader (fail-closed). This is
the "one env default + one header line" the panel named; loopback must NOT
default to brix-pm (the panel call), so the default is keyed on the WRITE
verb, not on the transport.

Usage as CLI (this module):
    mem promote <key> [--content <body>] --from <agent>/<store>
                      [--by <curator-principal>] [--tags tag1,tag2]
                      [--rationale one-line-why] [--store atoms|machinery]

WIRING (reviewer PR #338 cycle 1 [med] — the dead-in-repo module is no
longer dead): this module IS wired into a live entry point in agents-core:
the `mem-cli` console_scripts entry in pyproject.toml
(`agents_core.mem_cli:main`) exposes the promote subcommand + --store flag
+ MEM_PRINCIPAL default as a standalone binary. The spec's Files-changed
line names conductor/scripts/mem.py (a separate repo) as the LIVE curation
surface — the conductor PR adds a thin dispatch to this module (one
`import` + one `add_parser`) so the conductor `mem` binary and this
`mem-cli` binary share the same argparse wiring + HTTP dispatch. Until the
conductor PR lands, `mem-cli` (this entry point) is the live surface
shipped in THIS repo; the module is exercised by tests/test_mem_cli.py AND
by the console-script entry point.
"""

from __future__ import annotations

import argparse
import os
import sys

from agents_core.mem_client import (
    INVALID_REF_CHARS_RE,
    MemClient,
    MemHTTPError,
    STORE_ATOMS,
    STORE_MACHINERY,
    validate_store,
)

# The BRIX-side default principal for the mem CLI (D1 / panel F7). When
# MEM_PRINCIPAL is unset and the command is a write (promote), the CLI
# defaults to this so the primary local path lands as the registered
# writer, not a reader. Loopback must NOT default to brix-pm (panel call) —
# the default is keyed on the WRITE verb, not the transport.
DEFAULT_CLI_PRINCIPAL = "brix-pm"


def _resolve_principal(args: argparse.Namespace) -> str:
    """The principal for the command: explicit --by (promote) or the
    MEM_PRINCIPAL env default. For a write verb with no explicit principal,
    default to DEFAULT_CLI_PRINCIPAL (brix-pm) so the primary local path
    lands as the registered writer (D1 / panel F7)."""
    # Explicit --by (promote) wins.
    by = getattr(args, "by", None)
    if by:
        return by
    # MEM_PRINCIPAL env (the caller's principal, D1).
    env = os.environ.get("MEM_PRINCIPAL", "").strip()
    if env:
        return env
    # Write verb with no principal: default to the BRIX-side curator.
    if getattr(args, "command", None) == "promote":
        return DEFAULT_CLI_PRINCIPAL
    return ""


def _cmd_promote(args: argparse.Namespace) -> int:
    """Dispatch the promote verb through MemClient (HTTP).

    REFUSES (loud exit 2) when MEM_SERVER is unset: promotion is an
    explicit, server-side verb with a provenance shape the server owns; a
    local MemoryStore-only promote would bypass the server's --from shape
    check + batch key (the side-effect the spec forbids)."""
    if not os.environ.get("MEM_SERVER"):
        print(
            "[mem] promote requires MEM_SERVER (the HTTP path): promotion is "
            "an explicit, server-side verb with a provenance shape the server "
            "owns; a local-only promote is refused (D2: never a side effect)",
            file=sys.stderr,
        )
        return 2

    principal = _resolve_principal(args)
    if not principal:
        print(
            "[mem] promote requires a curator principal (--by or MEM_PRINCIPAL)",
            file=sys.stderr,
        )
        return 2

    # Validate --store (loud ValueError on a typo) — the rescoped flag is
    # MOOT at the HTTP layer but must be a known value.
    try:
        validate_store(args.store)
    except ValueError as e:
        print(f"[mem] {e}", file=sys.stderr)
        return 2

    # Client-side one-line rationale check (the server repeats it with a
    # loud 400 bad_rationale): the batch decision key is a line-based
    # listing, so a multi-line/control-char rationale is rejected before
    # any request is sent.
    if INVALID_REF_CHARS_RE.search(args.rationale):
        print(
            "[mem] --rationale must be a single line (no newlines or "
            "control chars)",
            file=sys.stderr,
        )
        return 2

    try:
        client = MemClient(principal=principal)
        row = client.promote(
            args.key,
            args.from_ref,
            principal,
            args.content,
            tags=args.tags,
            rationale=args.rationale,
        )
        print(f"Promoted: {args.key} (batch: {row.get('batch_key', '?')})")
        return 0
    except MemHTTPError as e:
        print(f"[mem] MEM_SERVER error: {e.status_code} {e.body}", file=sys.stderr)
        return 2
    except ValueError as e:
        # Client-side --from shape validation (newline/control-char ref).
        print(f"[mem] {e}", file=sys.stderr)
        return 2
    except Exception:
        # BRIX offline / server unreachable / timeout.
        print("[mem] BRIX offline: promote unavailable", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser for the promote subcommand + --store flag.

    The conductor CLI (scripts/mem.py) adds this subparser to its own
    `sub` group; this module exposes it standalone so it is testable from
    agents-core alone (the spec's Files-changed line names the conductor
    CLI as the live surface, but the wiring + the MEM_PRINCIPAL default
    live here)."""
    parser = argparse.ArgumentParser(
        prog="mem",
        description="Cross-instance memory store (promote subcommand)",
    )
    sub = parser.add_subparsers(dest="command")

    # promote (D2)
    p_promote = sub.add_parser(
        "promote",
        help="Promote one row from an agent store into mem.db (D2)",
    )
    # NOTE: argparse ordering constraint. The promoted body is the --content
    # OPTION (not a positional) so that the documented canonical invocation
    #
    #   mem promote <key> --from <agent>/<store> --by <curator> [--content ...]
    #
    # parses: argparse stops consuming positionals at the first unrecognized
    # token, so with a positional [content] the optional options (--from,
    # --by, ...) could only come AFTER the positionals — the documented form
    # (`mem promote <key> --from ...`) then failed to parse (reviewer PR #333
    # cycle 1 [low]). With a single positional (key) and all options, any
    # option order after <key> works.
    p_promote.add_argument("key", help="The mem.db key to write")
    p_promote.add_argument(
        "--from",
        dest="from_ref",
        required=True,
        help="The --from agent-store ref, '<agent>/<store>' (path-like, one "
             "slash).",
    )
    p_promote.add_argument(
        "--content",
        default="",
        help="The promoted body (the provenance header line is prepended "
             "server-side); omit to promote an empty body.",
    )
    p_promote.add_argument(
        "--by",
        default="",
        help="The curator principal (defaults to MEM_PRINCIPAL, then brix-pm).",
    )
    p_promote.add_argument(
        "--tags",
        default="",
        help="Comma-separated extra tags (the 'promoted' tag is always added).",
    )
    p_promote.add_argument(
        "--rationale",
        default="",
        help="One-line rationale (the D2 named decision artifact).",
    )
    p_promote.add_argument(
        "--store",
        default=STORE_ATOMS,
        choices=sorted((STORE_ATOMS, STORE_MACHINERY, "exhaust")),
        help="The target store (RESCOPED: MOOT at the HTTP layer — the server "
             "routes machine-state keys transparently; validated for a loud typo).",
    )

    return parser


def cli(argv: list[str] | None = None) -> int:
    """The CLI entry point. Returns the exit code (0 = success, 2 = error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "promote":
        return _cmd_promote(args)

    parser.print_help()
    return 1


def main() -> None:
    """The console-script entry point (pyproject.toml: mem-cli).

    Exits with the CLI's exit code (0 = success, 1 = no/unknown command,
    2 = error). The spec's Files-changed line names conductor/scripts/mem.py
    (a separate repo) as the LIVE curation surface; the conductor PR
    dispatches to this module. Until that lands, `mem-cli` is the live
    surface shipped in THIS repo (reviewer PR #338 cycle 1 [med])."""
    sys.exit(cli() or 0)


if __name__ == "__main__":
    main()
