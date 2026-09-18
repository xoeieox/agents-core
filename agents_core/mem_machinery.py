"""Machine-state faucet allowlist — the shared artifact both the mem-server
prefix guard (openclaw-memdb-influx-reader-v0, D3) and mem-hygiene-automation
read.

ONE file both specs read: ``/srv/agents/config/mem-machine-state-prefixes.json``
(panel 2026-09-14 F6). One entry per prefix:

    {
      "prefix": "elevator/",          # LITERAL — the codebase's startswith-only
                                      # convention (mem_exhaust.py:64-71). No globs.
      "producer_principal": "brix-pm",# the registered bot principal allowed to
                                      # write this prefix (faucet reads this)
      "store": "machinery",           # where the producer's writes land
      "state": "live",                # "live" | "dead" (hygiene reads this)
      "dead_since": null              # ISO date or null (hygiene reads this)
    }

Faucet reads ``prefix`` / ``producer_principal`` / ``store``; hygiene reads
``state`` / ``dead_since``.

Alignment with ``EXHAUST_PREFIXES`` (mem_exhaust.py:72-76): on a prefix present
in BOTH, the ROUTING table (EXHAUST_PREFIXES) wins for where new rows land and
this file wins for quarantine/death state. The safe desync direction is the
hygiene list as SUPERSET of the faucet list (quarantine is recoverable inside
its 14-day window; silent accumulation in the machinery store is the worse
failure).

Startup-validation guard (gate technical-integrity + transmuter, hard
requirement): ``load_allowlist()`` REFUSES on a missing/malformed/unreadable
file by raising ``AllowlistError`` — a fail-open bypass where a missing
allowlist silently disables the prefix-reject is the silent fail-open the gate
names. ``mem_server.create_app()`` calls this at boot and the server refuses
to start on failure (no fail-open).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Default path of the shared allowlist artifact. Env-overridable (mirrors
# MEM_DB_PATH's override convention) so tests point at a tmp file.
ALLOWLIST_PATH = Path(
    os.environ.get(
        "MEM_MACHINE_STATE_PREFIXES_PATH",
        "/srv/agents/config/mem-machine-state-prefixes.json",
    )
)

# The repo-shipped copy of the artifact (config/mem-machine-state-prefixes.json).
# The live copy at ALLOWLIST_PATH is the deployment target; when it is not yet
# present (e.g. a fresh clone, a test env, or pre-deploy), the guard falls back
# to this repo artifact so the server can still boot. This is NOT a fail-open
# bypass: the fallback is a VALID, concrete allowlist shipped in the PR, and a
# missing/malformed file at EITHER path still refuses to start.
_REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ALLOWLIST_PATH = _REPO_ROOT / "config" / "mem-machine-state-prefixes.json"


def default_allowlist_path() -> Path:
    """The effective default allowlist path: the live path if it exists, else
    the repo-shipped copy. Used by create_app() when no explicit path is given."""
    if ALLOWLIST_PATH.exists():
        return ALLOWLIST_PATH
    return REPO_ALLOWLIST_PATH

# The store a machine-state write lands in. RESCOPED (rev-2, 2026-09-14
# gate proceed-to-bind): the machinery store is the EXISTING exhaust store —
# the ``route_to_exhaust`` mechanism — extended with a reconciled narrow
# prefix list. No fork, no second sqlite (no mem_machinery.db). ``store``
# values in the allowlist are therefore all "machinery" == "exhaust".
STORE_MACHINERY = "machinery"


class AllowlistError(Exception):
    """Raised when the shared allowlist artifact is missing, malformed, or
    unreadable. The server MUST refuse to start on this (fail-closed)."""


@dataclass(frozen=True)
class MachineStateEntry:
    prefix: str
    producer_principal: str
    store: str
    state: str
    dead_since: str | None = None


@dataclass
class MachineStateAllowlist:
    """The parsed, validated allowlist.

    ``entries`` preserves file order. ``prefixes`` is the tuple used for
    ``str.startswith`` matching (the startswith-only convention).
    """

    entries: list[MachineStateEntry] = field(default_factory=list)

    @property
    def prefixes(self) -> tuple[str, ...]:
        return tuple(e.prefix for e in self.entries)

    def producer_for(self, key: str) -> str | None:
        """The registered producer principal for a key under a machine-state
        prefix, or None if the key is not machine-state."""
        for e in self.entries:
            if key.startswith(e.prefix):
                return e.producer_principal
        return None

    def is_machine_state(self, key: str) -> bool:
        return any(key.startswith(e.prefix) for e in self.entries)


def _validate_entry(raw: Any, idx: int) -> MachineStateEntry:
    if not isinstance(raw, dict):
        raise AllowlistError(f"allowlist entry #{idx} is not an object: {raw!r}")
    prefix = raw.get("prefix")
    if not isinstance(prefix, str) or not prefix:
        raise AllowlistError(f"allowlist entry #{idx} has a missing/empty 'prefix'")
    if not prefix.endswith("/"):
        # The startswith-only convention is a trailing-slash literal prefix.
        raise AllowlistError(
            f"allowlist entry #{idx} prefix {prefix!r} must be a literal "
            f"trailing-slash prefix (startswith-only convention)"
        )
    producer = raw.get("producer_principal")
    if not isinstance(producer, str) or not producer:
        raise AllowlistError(
            f"allowlist entry #{idx} ({prefix!r}) has a missing/empty "
            f"'producer_principal'"
        )
    store = raw.get("store")
    if not isinstance(store, str) or not store:
        raise AllowlistError(
            f"allowlist entry #{idx} ({prefix!r}) has a missing/empty 'store'"
        )
    state = raw.get("state")
    if state not in ("live", "dead"):
        raise AllowlistError(
            f"allowlist entry #{idx} ({prefix!r}) has invalid 'state' {state!r} "
            f"(must be 'live' or 'dead')"
        )
    dead_since = raw.get("dead_since", None)
    if dead_since is not None and not isinstance(dead_since, str):
        raise AllowlistError(
            f"allowlist entry #{idx} ({prefix!r}) has a non-string 'dead_since'"
        )
    return MachineStateEntry(
        prefix=prefix,
        producer_principal=producer,
        store=store,
        state=state,
        dead_since=dead_since,
    )


def load_allowlist(path: Path | str = ALLOWLIST_PATH) -> MachineStateAllowlist:
    """Load and validate the shared allowlist artifact.

    Raises ``AllowlistError`` (fail-closed) when the file is missing,
    unreadable, not valid JSON, not a JSON object with a top-level ``"prefixes"``
    list, or contains a malformed entry. A missing/malformed allowlist must
    never silently disable the prefix-reject — that is the fail-open bypass
    the gate names.
    """
    p = Path(path)
    try:
        raw_text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise AllowlistError(f"allowlist file is missing: {p}") from None
    except OSError as e:
        raise AllowlistError(f"allowlist file is unreadable: {p} ({e})") from None

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise AllowlistError(f"allowlist file is malformed JSON: {p} ({e})") from None

    if not isinstance(data, dict):
        raise AllowlistError(
            f"allowlist file must be a JSON object: {p}"
        )
    prefixes = data.get("prefixes")
    if not isinstance(prefixes, list):
        raise AllowlistError(
            f"allowlist file must have a top-level 'prefixes' list: {p}"
        )

    entries = [_validate_entry(raw, i) for i, raw in enumerate(prefixes)]

    # Duplicate prefixes are a config error (ambiguous producer).
    seen: set[str] = set()
    for e in entries:
        if e.prefix in seen:
            raise AllowlistError(
                f"allowlist file has duplicate prefix {e.prefix!r}: {p}"
            )
        seen.add(e.prefix)

    return MachineStateAllowlist(entries=entries)


def machine_state_prefixes(path: Path | str = ALLOWLIST_PATH) -> tuple[str, ...]:
    """Convenience: just the prefix tuple (raises AllowlistError as above)."""
    return load_allowlist(path).prefixes
