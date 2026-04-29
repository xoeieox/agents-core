#!/usr/bin/env python3
"""agents_core/vault_visibility_audit.py — Vault citable-field auditor.

Walks the vault, identifies markdown docs missing the ``citable:`` frontmatter
field, suggests defaults from per-directory allow-list rules, and writes them
back in batch (--apply) or per-doc-interactive (--review).

Default rules:
  - Path contains ``/Personal/``               ⇒  citable: false
  - Path contains ``/Daily-Notes/private/``    ⇒  citable: false
  - Frontmatter contains ``private: true``     ⇒  citable: false
  - All other docs                             ⇒  citable: true

Usage::

    # Report unaudited docs (exits non-zero if any found):
    python3 -m agents_core.vault_visibility_audit --check /srv/git/inertia-vault-working

    # Batch-write defaults (non-interactive):
    python3 -m agents_core.vault_visibility_audit --apply /srv/git/inertia-vault-working

    # Per-doc interactive review:
    python3 -m agents_core.vault_visibility_audit --review /srv/git/inertia-vault-working

The script is re-runnable: docs that already have a ``citable:`` field are
skipped.  Only docs without the field are touched.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Directory-based exclusion patterns (case-sensitive path contains checks)
# ---------------------------------------------------------------------------

_EXCLUDE_PATTERNS: list[str] = [
    "/Personal/",
    "/Daily-Notes/private/",
]

# Regex for a YAML frontmatter block at the start of a file.
_FM_RE = re.compile(r"^---\r?\n(.*?\r?\n)---\r?\n", re.DOTALL)
_PRIVATE_RE = re.compile(r"^\s*private\s*:\s*true\s*$", re.MULTILINE | re.IGNORECASE)
_CITABLE_RE = re.compile(r"^\s*citable\s*:", re.MULTILINE)


def _suggested_citable(path: Path, fm_body: str) -> bool:
    """Return the suggested citable value for a doc based on path and frontmatter."""
    path_str = str(path)
    for pattern in _EXCLUDE_PATTERNS:
        if pattern in path_str:
            return False
    if _PRIVATE_RE.search(fm_body):
        return False
    return True


def _has_citable_field(content: str) -> bool:
    """Return True if the file already has a ``citable:`` frontmatter field."""
    match = _FM_RE.match(content)
    if not match:
        return False
    return bool(_CITABLE_RE.search(match.group(1)))


def _inject_citable(content: str, path: Path, citable: bool) -> str:
    """Insert ``citable: true/false`` into the frontmatter, or create a block.

    Inserts *after* the last existing frontmatter line, before the closing
    ``---``.  If there is no frontmatter, a minimal block is prepended.
    """
    value = "true" if citable else "false"
    match = _FM_RE.match(content)
    if match:
        fm_body = match.group(1)
        rest = content[match.end():]
        # Ensure fm_body ends with a newline before appending
        if not fm_body.endswith("\n"):
            fm_body += "\n"
        return f"---\n{fm_body}citable: {value}\n---\n{rest}"
    else:
        return f"---\ncitable: {value}\n---\n{content}"


def _collect_unaudited(vault_root: Path) -> list[tuple[Path, bool]]:
    """Walk vault_root, return list of (path, suggested_citable) for unaudited docs."""
    unaudited: list[tuple[Path, bool]] = []
    for md_path in sorted(vault_root.rglob("*.md")):
        # Skip hidden files / .git internals
        if any(part.startswith(".") for part in md_path.parts):
            continue
        try:
            content = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _has_citable_field(content):
            continue
        # Get frontmatter body for private detection
        fm_match = _FM_RE.match(content)
        fm_body = fm_match.group(1) if fm_match else ""
        suggested = _suggested_citable(md_path, fm_body)
        unaudited.append((md_path, suggested))
    return unaudited


def _write_citable(path: Path, citable: bool) -> None:
    """Inject citable field into *path*'s frontmatter and save via Path.write_text.

    This module intentionally uses Path.write_text directly (not vault_writer)
    because it is a maintenance script, not a vault-content writer.  The
    vault_visibility_audit itself is not a corpus document.
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    new_content = _inject_citable(content, path, citable)
    path.write_text(new_content, encoding="utf-8")


def mode_check(vault_root: Path) -> int:
    """Report unaudited docs; exit 1 if any found, 0 if clean."""
    unaudited = _collect_unaudited(vault_root)
    if not unaudited:
        print("vault-visibility-audit: OK — all docs have a citable: field.")
        return 0
    print(f"vault-visibility-audit: {len(unaudited)} doc(s) missing citable: field:")
    for path, suggested in unaudited:
        flag = "true" if suggested else "false"
        print(f"  [{flag}]  {path}")
    print("\nRun with --apply to write defaults, or --review for per-doc review.")
    return 1


def mode_apply(vault_root: Path) -> int:
    """Batch-write suggested citable defaults for all unaudited docs."""
    unaudited = _collect_unaudited(vault_root)
    if not unaudited:
        print("vault-visibility-audit: nothing to do — all docs already audited.")
        return 0
    for path, suggested in unaudited:
        _write_citable(path, suggested)
        flag = "true" if suggested else "false"
        print(f"  wrote citable: {flag}  →  {path}")
    print(f"\nvault-visibility-audit: stamped {len(unaudited)} doc(s).")
    return 0


def mode_review(vault_root: Path) -> int:
    """Per-doc interactive review: confirm or override suggested citable value."""
    unaudited = _collect_unaudited(vault_root)
    if not unaudited:
        print("vault-visibility-audit: nothing to do — all docs already audited.")
        return 0

    applied = 0
    skipped = 0
    for path, suggested in unaudited:
        flag = "true" if suggested else "false"
        rel = path.relative_to(vault_root) if path.is_relative_to(vault_root) else path
        print(f"\n  {rel}")
        print(f"  Suggested citable: {flag}")
        while True:
            resp = input("  Accept [y], flip [f], skip [s], quit [q]? ").strip().lower()
            if resp in ("y", ""):
                _write_citable(path, suggested)
                print(f"  → wrote citable: {flag}")
                applied += 1
                break
            elif resp == "f":
                flipped = not suggested
                _write_citable(path, flipped)
                print(f"  → wrote citable: {'true' if flipped else 'false'} (flipped)")
                applied += 1
                break
            elif resp == "s":
                print("  → skipped")
                skipped += 1
                break
            elif resp == "q":
                print(f"\nvault-visibility-audit: applied {applied}, skipped {skipped + (len(unaudited) - applied - skipped - 1)}, quit early.")
                return 0
            else:
                print("  Please enter y, f, s, or q.")

    print(f"\nvault-visibility-audit: applied {applied}, skipped {skipped}.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit vault docs for missing citable: frontmatter field.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "vault_root",
        nargs="?",
        default="/srv/git/inertia-vault-working",
        help="Path to the vault root (default: /srv/git/inertia-vault-working)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check",
        action="store_true",
        help="Report-only mode; exits non-zero if unaudited docs exist.",
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="Batch-write suggested citable defaults for all unaudited docs.",
    )
    group.add_argument(
        "--review",
        action="store_true",
        help="Per-doc interactive review; confirm or override each suggestion.",
    )
    args = parser.parse_args(argv)
    vault_root = Path(args.vault_root)
    if not vault_root.exists():
        print(f"vault-visibility-audit: vault root not found: {vault_root}", file=sys.stderr)
        return 2

    if args.check:
        return mode_check(vault_root)
    elif args.apply:
        return mode_apply(vault_root)
    elif args.review:
        return mode_review(vault_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
