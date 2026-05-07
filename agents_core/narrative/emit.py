"""CLI entry point: narrative-emit.

Usage:
    narrative-emit --audience SLUG --ask TEXT [--length-target {short|medium|long}]
                   [--out PATH] [--dry-run]
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import yaml

from agents_core.narrative.audiences import VALID_SLUGS
from agents_core.narrative.engine import EmitResult, emit_draft


def _slug_from_ask(ask: str, max_chars: int = 40) -> str:
    """Convert ask text to a kebab-case slug, truncated to max_chars."""
    slug = ask.lower()
    slug = re.sub(r"[^a-z0-9 ]+", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    return slug[:max_chars].rstrip("-")


def _default_out(audience: str, ask: str) -> Path:
    today = date.today().isoformat()
    slug = _slug_from_ask(ask)
    return Path("/srv/lapis/narratives") / audience / f"{today}-{slug}.md"


def _build_front_matter(result: EmitResult) -> str:
    """Render YAML front-matter block from EmitResult."""
    sources_list = [
        {"path": ref.path, "sha256": ref.sha256} for ref in result.sources
    ]
    data = {
        "schema_version": "narrative-emit-v0",
        "caller": "narrative-emit",
        "audience": result.audience,
        "ask": result.ask,
        "length_target": result.length_target,
        "model": result.model,
        "dispatched_at": result.dispatched_at,
        "returned_at": result.returned_at,
        "prompt_hash": result.prompt_hash,
        "sources": sources_list,
    }
    return "---\n" + yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False) + "---\n"


def _write_output(result: EmitResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    front_matter = _build_front_matter(result)
    out_path.write_text(front_matter + "\n" + result.draft, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="narrative-emit",
        description="Compose canonical sources × audience frame × ask → draft.md",
    )
    parser.add_argument(
        "--audience",
        required=True,
        choices=VALID_SLUGS,
        help="Target audience slug",
    )
    parser.add_argument(
        "--ask",
        required=True,
        help="One-sentence description of what the draft needs to do",
    )
    parser.add_argument(
        "--length-target",
        default="medium",
        choices=["short", "medium", "long"],
        dest="length_target",
        help="Approximate draft length (default: medium ~600 words)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: /srv/lapis/narratives/<audience>/<date>-<slug>.md)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print assembled prompt to stdout and exit without calling the model",
    )

    args = parser.parse_args(argv)

    out_path = Path(args.out) if args.out else _default_out(args.audience, args.ask)

    try:
        result = emit_draft(
            audience_slug=args.audience,
            ask=args.ask,
            length_target=args.length_target,
            dry_run=args.dry_run,
        )
    except KeyError as e:
        print(
            f"narrative-emit: unknown audience {e}. Valid: {', '.join(VALID_SLUGS)}",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, RuntimeError) as e:
        print(f"narrative-emit: source error: {e}", file=sys.stderr)
        return 1
    except TimeoutError as e:
        print(f"narrative-emit: dispatch timeout: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(result._prompt)
        return 0

    _write_output(result, out_path)
    print(f"narrative-emit: draft written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
