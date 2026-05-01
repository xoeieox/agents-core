"""CLI entry for agents_core.librarian.

Usage::

    python3 -m agents_core.librarian render-live-surface --out <path>

Subcommands
-----------
render-live-surface --out PATH
    Render the weekly Live-Surface digest to PATH via vault_writer.
    Idempotent: re-running with unchanged corpus produces byte-identical output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_render_live_surface(args: argparse.Namespace) -> int:
    from agents_core.librarian.live_surface import render_live_surface

    out_path = Path(args.out)
    record = render_live_surface(out_path)
    print(f"written: {record.path}")
    print(f"  hash:  {record.content_hash}")
    print(f"  agent: {record.agent_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m agents_core.librarian",
        description="agents_core.librarian CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    rls = sub.add_parser(
        "render-live-surface",
        help="Render the weekly Live-Surface digest markdown file",
    )
    rls.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Destination path for the digest (e.g. /srv/git/inertia-vault-working/Lapis/Live-Surface.md)",
    )

    args = parser.parse_args(argv)

    if args.command == "render-live-surface":
        return _cmd_render_live_surface(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
