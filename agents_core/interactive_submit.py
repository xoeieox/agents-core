"""Interactive submit interface — the H4 contract for enqueuing and polling turns.

submit(turn, context, destination, *, operator=None, principal)
    destination ∈ {"queue-gw-interactive", "route-to-fast"}

For "queue-gw-interactive": enqueues an interactive baton and returns a handle.
For "route-to-fast": routes directly to an operator (default: qwen/StarHouse).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents_core.elevator import ElevatorStore
from agents_core.llm import call_operator


def submit(
    turn: str,
    context: dict | None = None,
    destination: str = "queue-gw-interactive",
    *,
    operator: str | None = None,
    principal: str = "default",
) -> dict[str, Any]:
    """Submit a turn to either the queue or direct routing.

    Args:
        turn: the user's prompt/message text
        context: optional system context (will be JSON-serialized as system=)
        destination: "queue-gw-interactive" (async queue) or "route-to-fast" (direct)
        operator: operator to use for "route-to-fast" (default: "qwen")
        principal: identity of the submitter (default: "default")

    Returns for "queue-gw-interactive":
        {
            "item_id": "...",
            "poll": "/v0/elevator/item/{item_id}",
            "node_state": {"queue": {...}, "gw": {...}, "as_of": "..."}
        }

    Returns for "route-to-fast":
        {
            "result": "...",
            "provenance": [...],
            "served_by": "operator_name"
        }
    """
    if destination == "queue-gw-interactive":
        # Read db_path from environment (for test isolation).
        db_path_env = os.environ.get("ELEVATOR_DB_PATH")
        db_path = Path(db_path_env) if db_path_env else None
        elevator = ElevatorStore(db_path) if db_path else ElevatorStore()
        try:
            payload = {
                "prompt": turn,
                "context": context or {},
            }
            item_id = elevator.enqueue(
                lane="interactive",
                kind="session-turn",
                payload=payload,
                principal=principal,
                latency_class="interactive",
            )
            node_state = elevator.state()
            return {
                "item_id": item_id,
                "poll": f"/v0/elevator/item/{item_id}",
                "node_state": node_state,
            }
        finally:
            elevator.close()

    elif destination == "route-to-fast":
        operator_class = operator or "qwen"
        system = json.dumps(context) if context else None
        provenance: list[tuple[str, str]] = []
        result = call_operator(
            operator_class,
            prompt=turn,
            system=system,
            _provenance_out=provenance,
            lease_class="deferrable",
        )
        return {
            "result": result,
            "provenance": provenance,
            "served_by": operator_class,
        }

    else:
        raise ValueError(
            f"Unknown destination {destination!r}; "
            "must be 'queue-gw-interactive' or 'route-to-fast'"
        )
