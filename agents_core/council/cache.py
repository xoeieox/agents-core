"""agents_core.council.cache — cohesion-finding cache for the Mirror Council.

Cache lives at /srv/lapis/council/cache/cohesion/<decision_hash>.yaml.
One file per landed cohesion-finding (output_class == "cohesion-finding").
none-class and laid-down outputs are NOT cached (Invariant 6).

Write is atomic (tmp-file on same fs + os.rename per M8) and non-blocking
(failure logs and returns None, does not raise, per Invariant 7).

find_related returns [] when cache dir does not exist (first-deploy safety
per M5 in v3→v4), so the first deliberation under v0.next never raises.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from agents_core.room_paths import room_path

CACHE_DIR = room_path("council.cache_cohesion")
KERNEL_FILE = Path("/srv/git/inertia-vault-working/Lapis/Constitution-Kernel.md")

log = logging.getLogger("council.cache")


def _normalize_decision_text(text: str) -> str:
    """Lowercase and collapse whitespace — used for hash input."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _parse_ts(ts: str) -> float:
    """Parse ISO timestamp to float for sorting; 0.0 on error."""
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def cache_path(decision_text_hash: str) -> Path:
    """Return the canonical cache file path for a given hash (first 16 hex chars)."""
    return CACHE_DIR / f"{decision_text_hash[:16]}.yaml"


def read_kernel_version() -> str:
    """Parse version: line from Constitution-Kernel.md frontmatter.

    Returns 'unknown' on missing/unparseable file (per Invariant 12).
    Cache entries tagged 'unknown' are treated as stale on read.
    """
    try:
        text = KERNEL_FILE.read_text()
        for line in text.splitlines():
            m = re.match(r"^version:\s*(.+)$", line.strip())
            if m:
                return m.group(1).strip()
        return "unknown"
    except Exception:
        return "unknown"


def write_finding(run: dict, kernel_version: str) -> Path | None:
    """Write a cohesion-finding cache entry for the given run.

    Only writes when run["synthesis"]["output_class"] == "cohesion-finding".
    Returns the written Path on success, None on skip or error.

    Write is atomic: tmp file under CACHE_DIR (same fs) + os.rename.
    Failure (disk full, permissions) logs and returns None without raising
    (cache is derived index; run YAML is the source of truth).
    """
    synthesis = run.get("synthesis", {})
    if synthesis.get("output_class") != "cohesion-finding":
        return None

    decision_text = run.get("decision", "")
    normalized = _normalize_decision_text(decision_text)
    text_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("cache mkdir failed (non-blocking): %s", e)
        return None

    positions = synthesis.get("positions", [])
    divergence_voices = [
        p["voice"] for p in positions if p.get("position") != "agree"
    ]
    stood_aside_basis = [
        {"voice": p["voice"], "reason": p.get("reason", "")}
        for p in positions
        if p.get("position") == "stand-aside"
    ]

    entry = {
        "decision_text_hash": text_hash,
        "decision_text_normalized": normalized,
        "run_id": run["run_id"],
        "landed_at": datetime.now(tz=timezone.utc).isoformat(),
        "kernel_version": kernel_version,
        "output_class": "cohesion-finding",
        "synthesis": {
            "landing": synthesis.get("landing", ""),
            "confidence": synthesis.get("confidence", ""),
            "positions": positions,
            "invariants_implicated": synthesis.get("invariants_implicated", []),
            "stood_aside": synthesis.get("stood_aside", []),
            "blocks": synthesis.get("blocks", []),
        },
        "divergence_voices": divergence_voices,
        "stood_aside_basis": stood_aside_basis,
    }

    dest = cache_path(text_hash)
    tmp = CACHE_DIR / f".tmp-{text_hash}"

    try:
        tmp.write_text(
            yaml.safe_dump(entry, sort_keys=False, width=100, allow_unicode=True)
        )
        os.rename(str(tmp), str(dest))
        return dest
    except OSError as e:
        log.warning("cache write failed (non-blocking): %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def find_related(
    decision_text: str = None,
    invariants: list[str] = None,
    voices: list[str] = None,
    limit: int = 3,
    current_kernel_version: str = None,
) -> list[dict]:
    """Return matching cache entries sorted by score desc, stale last.

    Missing cache dir returns [] without raising (M5 first-deploy safety).
    Stale entries (kernel_version mismatch or 'unknown') are returned but
    sorted last and flagged with stale: True.
    """
    try:
        files = list(CACHE_DIR.glob("*.yaml"))
    except (FileNotFoundError, OSError):
        return []

    if not files:
        return []

    if current_kernel_version is None:
        current_kernel_version = read_kernel_version()

    query_hash: str | None = None
    if decision_text:
        query_hash = hashlib.sha256(
            _normalize_decision_text(decision_text).encode()
        ).hexdigest()[:16]

    results = []
    for fpath in files:
        if fpath.name.startswith(".tmp-"):
            continue
        try:
            entry = yaml.safe_load(fpath.read_text())
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue

        score = 0
        if query_hash and entry.get("decision_text_hash") == query_hash:
            score += 10

        entry_invariants = entry.get("synthesis", {}).get("invariants_implicated", [])
        if invariants:
            for inv in invariants:
                if inv in entry_invariants:
                    score += 3

        entry_divergence = entry.get("divergence_voices", [])
        if voices:
            for v in voices:
                if v in entry_divergence:
                    score += 1

        entry_kv = entry.get("kernel_version", "unknown")
        stale = (entry_kv != current_kernel_version) or (entry_kv == "unknown")
        if stale:
            score = score // 2

        result = dict(entry)
        result["stale"] = stale
        result["_score"] = score
        result["_ts"] = _parse_ts(entry.get("landed_at", ""))
        results.append(result)

    # Sort: non-stale first, then by score desc, then by landed_at desc (newest first)
    results.sort(key=lambda r: (r["stale"], -r["_score"], -r["_ts"]))

    for r in results:
        r.pop("_score", None)
        r.pop("_ts", None)

    return results[:limit]
