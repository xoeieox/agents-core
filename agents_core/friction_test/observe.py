"""agents_core.friction_test.observe — Observation dataclass.

Pure data; no judgment. Captures everything a driver collected during
one scenario execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Observation:
    """All raw data captured during one scenario execution.

    Fields are intentionally wide — each driver populates the fields
    relevant to its target and leaves others empty.
    """

    scenario_id: str
    started_at: str  # ISO8601
    finished_at: str  # ISO8601

    # HTTP calls: list of {method, url, status, body, latency_ms, ts}
    http_calls: list[dict[str, Any]] = field(default_factory=list)

    # SSE events received on stream endpoint
    sse_events: list[dict[str, Any]] = field(default_factory=list)

    # mem.db writes observed (before/after diff)
    mem_writes: list[dict[str, Any]] = field(default_factory=list)

    # vault writes observed (before/after diff on audit chain)
    vault_writes: list[dict[str, Any]] = field(default_factory=list)

    # log appends: new lines appended to watched JSONL files
    log_appends: list[dict[str, Any]] = field(default_factory=list)

    # errors / harness problems; does NOT crash the run
    errors: list[dict[str, Any]] = field(default_factory=list)

    # harness_error flag: True when driver couldn't run the scenario at all
    harness_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "http_calls": self.http_calls,
            "sse_events": self.sse_events,
            "mem_writes": self.mem_writes,
            "vault_writes": self.vault_writes,
            "log_appends": self.log_appends,
            "errors": self.errors,
            "harness_error": self.harness_error,
        }
