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
  MODE_PEEK_FRESHNESS_SEC — how long last_known_serve_big is considered fresh (default 60)
  MODE_PEEK_BACKOFF_CAP_SEC — max exponential backoff on a cold peek failure (default 16)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

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
MODE_PEEK_FRESHNESS_SEC = int(os.environ.get("MODE_PEEK_FRESHNESS_SEC", "60"))
MODE_PEEK_BACKOFF_CAP_SEC = int(os.environ.get("MODE_PEEK_BACKOFF_CAP_SEC", "16"))

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
    """Classify deferred vs wake_failed vs doorman_unreachable.

    Reads the L1a provenance list FIRST — the exact reason is already present.
    Falls back to a legacy /status re-probe only when the provenance carries none
    of the expected L1a reasons (defensive: older call_operator builds).

    Returns "deferred", "wake_failed", "doorman_unreachable", or None if unknown.
    """
    for reason, _ in provenance:
        if reason == "gw_deferred_swarm":
            return "deferred"
        if reason == "doorman_unreachable":
            return "doorman_unreachable"
        if reason == "gw_not_serving":
            return "wake_failed"

    # Legacy fallback: provenance carries none of the three L1a reasons — fall back
    # to a /status re-probe (pre-L1a call_operator or unknown provenance).
    try:
        client = DoormanClient()
        status_resp = client.status()
        client.close()
        gw_state = status_resp.get("nodes", {}).get("gravitywell", {})
        if gw_state.get("serving") is False:
            return "deferred"
    except (DoormanUnreachable, Exception):
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
        # _admission_bypass=True: the interactive worker's claimed baton IS its admission;
        # re-entering the elevator would self-deadlock.
        result = call_operator(
            "gravitywell",
            prompt=prompt,
            system=system,
            on_wake_fail="skip",
            _provenance_out=provenance,
            _admission_bypass=True,
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


class _ModePeekState:
    """Carries last-known serve-big state + exponential-backoff counter for peek failures."""

    def __init__(self, freshness_sec: int, backoff_cap_sec: int):
        self.freshness_sec = freshness_sec
        self.backoff_cap_sec = backoff_cap_sec
        # last known serve-big result + the monotonic timestamp it was recorded
        self._last_value: Optional[bool] = None
        self._last_ts: float = 0.0
        # consecutive-failure counter for exponential backoff
        self._fail_count: int = 0

    def record_success(self, serve_big: bool) -> None:
        self._last_value = serve_big
        self._last_ts = time.monotonic()
        self._fail_count = 0

    def is_fresh(self) -> bool:
        return (
            self._last_value is not None
            and (time.monotonic() - self._last_ts) < self.freshness_sec
        )

    def last_known_serve_big(self) -> Optional[bool]:
        return self._last_value

    def next_backoff_sec(self) -> int:
        """Exponential backoff for successive peek failures, capped."""
        return min(2 ** self._fail_count, self.backoff_cap_sec)

    def record_failure(self) -> None:
        self._fail_count += 1

    def past_cap(self) -> bool:
        """True once we have been failing long enough to have hit the cap."""
        return self._fail_count > 0 and 2 ** (self._fail_count - 1) >= self.backoff_cap_sec


def _peek_serve_big(client: DoormanClient) -> bool:
    """Return True iff the cached doorman /status says GW is in big mode."""
    status = client.status()
    gw = status.get("nodes", {}).get("gravitywell", {})
    return bool(gw.get("serving") is True and gw.get("serving_mode") == "big")


def worker_loop(
    _peek_state: Optional[_ModePeekState] = None,
    _doorman_client: Optional[DoormanClient] = None,
):
    """Main worker loop: peek GW mode, claim interactive batons, and serve them.

    The pre-claim mode-peek (AC1/AC2/AC3) reads the cached doorman /status before
    claiming. If the mode is not big (swarm/stopped), the baton is left PENDING and
    the loop sleeps the idle interval — removing claim-then-defer churn.

    On a peek failure the worker falls back to last_known_serve_big + bounded
    exponential backoff rather than blind-claiming (AC3 hardened fail).
    """
    elevator = ElevatorStore(DB_PATH)
    peek_state = _peek_state or _ModePeekState(
        freshness_sec=MODE_PEEK_FRESHNESS_SEC,
        backoff_cap_sec=MODE_PEEK_BACKOFF_CAP_SEC,
    )
    # Allow injection of a DoormanClient for tests; else create a shared one.
    _own_client = _doorman_client is None
    client = _doorman_client or DoormanClient()

    IDLE_SLEEP_SEC = 5

    try:
        logger.info(
            f"Interactive worker started (claim_ttl={INTERACTIVE_CLAIM_TTL_SEC}s, "
            f"max_retries={MAX_WAKE_FAIL_RETRIES})"
        )
        while True:
            # --- Pre-claim mode-peek ---
            try:
                serve_big = _peek_serve_big(client)
                peek_state.record_success(serve_big)
            except (DoormanUnreachable, Exception) as exc:
                if peek_state.is_fresh():
                    # Use the last-known state to decide. Do NOT record a failure
                    # here — _fail_count drives the stale-case backoff ramp and must
                    # start from 0 when freshness expires; incrementing during the
                    # fresh window would bypass the 1→2→4→...→cap ramp at transition.
                    serve_big = peek_state.last_known_serve_big()
                    logger.debug(
                        f"Peek failed ({exc!r}); using last-known serve_big={serve_big}"
                    )
                else:
                    # No fresh last-known state — skip with bounded exponential backoff.
                    # Compute backoff BEFORE recording the failure so that the first
                    # skip sleeps 1s (2^0), the second 2s (2^1), etc.
                    backoff = peek_state.next_backoff_sec()
                    peek_state.record_failure()
                    logger.warning(
                        f"Peek failed and no fresh last-known state; "
                        f"skipping claim, backoff {backoff}s ({exc!r})"
                    )
                    time.sleep(backoff)
                    # After the cap has been reached, allow one claim attempt so a
                    # sustained doorman outage cannot permanently starve interactive.
                    if peek_state.past_cap():
                        logger.warning(
                            "Peek backoff cap reached; attempting one claim despite unknown mode"
                        )
                        serve_big = True  # let the claim proceed
                    else:
                        continue

            if not serve_big:
                # GW is not in big mode — leave batons PENDING, sleep idle interval.
                logger.debug("GW not in big mode (peek); skipping claim this cycle")
                time.sleep(IDLE_SLEEP_SEC)
                continue

            # --- Claim + serve ---
            # AC3b: deterministic deliberation-lane stale-claim sweep on each worker tick.
            # Runs independent of GW admission traffic; this loop already ticks on a timer
            # and holds no GW resources — safe host for the sweep.
            elevator.reclaim_stale("deliberation")
            item = elevator.claim(
                lanes=["interactive"],
                owner="elevator-interactive-worker",
                claim_ttl_sec=INTERACTIVE_CLAIM_TTL_SEC,
            )
            if item is None:
                # No pending interactive items — sleep and retry.
                time.sleep(IDLE_SLEEP_SEC)
                continue

            serve_interactive_baton(item)
    except KeyboardInterrupt:
        logger.info("Interactive worker stopped")
    except Exception as e:
        logger.exception(f"Worker loop error: {e}")
    finally:
        elevator.close()
        if _own_client:
            client.close()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    worker_loop()


if __name__ == "__main__":
    main()
