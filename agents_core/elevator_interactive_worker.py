"""Interactive serving worker — claims interactive batons and serves them on GravityWell.

This is a separate managed companion (role='worker') to slot_server. It polls for
interactive batons, serves them via GravityWell, and handles deferred/wake-failed
requeuing without paying fallback operators.

Entry point: elevator-interactive-server (console_script in pyproject.toml).
Systemd unit: systemd/elevator-interactive-server.service (held, not auto-enabled).

Environment:
  ELEVATOR_DB_PATH — path to queue.db (default /data/elevator/queue.db)
  GW_SERVE_TIMEOUT_SEC — GW serve timeout (default 300, must match doorman lease TTL)
  MAX_WAKE_FAIL_RETRIES — max retries before marking as failed (default 5)
  INTERACTIVE_CLAIM_TTL_SEC — claim TTL (default 360 = GW_SERVE_TIMEOUT_SEC + 60)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from agents_core.doorman_client import DoormanClient, DoormanUnreachable
from agents_core.elevator import ElevatorStore
from agents_core.llm import call_operator

# Configuration
DB_PATH = Path(os.environ.get("ELEVATOR_DB_PATH", "/data/elevator/queue.db"))
GW_SERVE_TIMEOUT_SEC = int(os.environ.get("GW_SERVE_TIMEOUT_SEC", "300"))
MAX_WAKE_FAIL_RETRIES = int(os.environ.get("MAX_WAKE_FAIL_RETRIES", "5"))
INTERACTIVE_CLAIM_TTL_SEC = int(os.environ.get(
    "INTERACTIVE_CLAIM_TTL_SEC",
    str(GW_SERVE_TIMEOUT_SEC + 60)  # 360 = 300 + 60
))

logger = logging.getLogger(__name__)


def _reconstruct_call_args(payload: dict) -> tuple[str, str | None]:
    """Reconstruct prompt and system args from the enqueued payload.

    Returns (prompt, system) where system is JSON-stringified context or None.
    """
    prompt = payload.get("prompt", "")
    context = payload.get("context")
    system = json.dumps(context) if context else None
    return prompt, system


def _classify_deferred_reason(provenance: list[tuple[str, str]]) -> str | None:
    """Classify deferred vs wake_failed vs doorman_unreachable from provenance tuple.

    If doorman_unreachable is in the tuple, return "doorman_unreachable".
    Otherwise, do a cached doorman /status read to determine deferred vs wake_failed.

    Returns "deferred", "wake_failed", "doorman_unreachable", or None if unknown.
    """
    # Check if doorman was unreachable during the serve attempt.
    for reason, _ in provenance:
        if reason == "doorman_unreachable":
            return "doorman_unreachable"

    # For gw_not_serving, do a cached /status read to differentiate.
    try:
        client = DoormanClient()
        status_resp = client.status()
        client.close()
        gw_state = status_resp.get("nodes", {}).get("gravitywell", {})
        # If serving=false, the GW is in swarm (deferred);
        # if the key is missing, it's a system issue (treat as wake_failed).
        if gw_state.get("serving") is False:
            return "deferred"
    except (DoormanUnreachable, Exception):
        # Doorman unreachable — treat as wake_failed (will retry with backoff).
        pass

    return "wake_failed"


def serve_interactive_baton(item: dict) -> bool:
    """Serve one interactive baton via GravityWell.

    Returns True if successfully served, False if should be requeued/failed.
    """
    # Read configuration from environment (for test isolation).
    db_path_env = os.environ.get("ELEVATOR_DB_PATH")
    db_path = Path(db_path_env) if db_path_env else DB_PATH
    max_wake_fail_retries = int(os.environ.get("MAX_WAKE_FAIL_RETRIES", str(MAX_WAKE_FAIL_RETRIES)))

    elevator = ElevatorStore(db_path)
    item_id = item["item_id"]
    payload = item["payload"]
    attempts = item.get("attempts", 0)

    try:
        prompt, system = _reconstruct_call_args(payload)
        provenance: list[tuple[str, str]] = []

        # Call GravityWell with on_wake_fail="skip" (no paid fallback).
        result = call_operator(
            "gravitywell",
            prompt=prompt,
            system=system,
            on_wake_fail="skip",
            _provenance_out=provenance,
        )

        if result is not None:
            # Serve succeeded — write result + provenance + ack.
            provenance_dict = {"serves": [{"reason": p[0], "operator": p[1]} for p in provenance]}
            elevator.ack(
                item_id,
                result=result,
                provenance=provenance_dict,
            )
            logger.info(f"Served {item_id} on GravityWell")
            return True

        # result is None — GW was deferred, wake_failed, or doorman unreachable.
        reason = _classify_deferred_reason(provenance)

        if reason == "deferred":
            # GW in swarm — immediate requeue (will end when swarm flips).
            elevator.requeue(item_id)
            logger.info(f"Requeued {item_id} (GW deferred)")
            return False

        # wake_failed or doorman_unreachable — bounded backoff.
        if attempts >= max_wake_fail_retries:
            elevator.fail(item_id)
            logger.warning(
                f"Marked {item_id} as failed (wake_failed after {MAX_WAKE_FAIL_RETRIES} retries)"
            )
            return False

        # Requeue with bounded backoff (exponential: 1s, 2s, 4s, 8s, 16s).
        elevator.requeue(item_id)
        backoff = 2 ** min(attempts, 4)  # Cap at 16s.
        logger.warning(
            f"Requeued {item_id} (wake_failed, attempt {attempts + 1}/{max_wake_fail_retries}, "
            f"backoff {backoff}s)"
        )
        return False

    except Exception as e:
        logger.exception(f"Error serving {item_id}: {e}")
        # On unexpected error, requeue or fail based on attempt count.
        attempts = item.get("attempts", 0)
        if attempts >= MAX_WAKE_FAIL_RETRIES:
            elevator.fail(item_id)
        else:
            elevator.requeue(item_id)
        return False
    finally:
        elevator.close()


def worker_loop():
    """Main worker loop: claim interactive batons and serve them."""
    elevator = ElevatorStore(DB_PATH)
    try:
        logger.info(
            f"Interactive worker started (claim_ttl={INTERACTIVE_CLAIM_TTL_SEC}s, "
            f"max_retries={MAX_WAKE_FAIL_RETRIES})"
        )
        while True:
            item = elevator.claim(
                lanes=["interactive"],
                owner="elevator-interactive-worker",
                claim_ttl_sec=INTERACTIVE_CLAIM_TTL_SEC,
            )
            if item is None:
                # No pending interactive items — sleep and retry.
                time.sleep(5)
                continue

            serve_interactive_baton(item)
    except KeyboardInterrupt:
        logger.info("Interactive worker stopped")
    except Exception as e:
        logger.exception(f"Worker loop error: {e}")
    finally:
        elevator.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    worker_loop()


if __name__ == "__main__":
    main()
