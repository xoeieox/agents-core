#!/usr/bin/env python3
"""Pushover notification module — single shared implementation for all Conductor scripts.

Usage:
    from notify import send_notification, Priority
    send_notification("Something happened", title="Alert", priority=Priority.HIGH)

Credentials are read from env vars PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN,
which are set via systemd EnvironmentFile (conductor.env).
"""

import json
import logging
import os
from datetime import datetime
from enum import IntEnum

import requests

from agents_core.room_paths import room_path

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

# Pushover limits
MAX_MESSAGE_LENGTH = 1024
MAX_TITLE_LENGTH = 250

CAPTURE_LOG = room_path("notify_audit.captured")

log = logging.getLogger(__name__)


class Priority(IntEnum):
    LOWEST = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1


def _capture_event(
    *,
    source: str,
    message: str,
    title: str,
    priority: Priority,
    delivered: bool | None,
    extra: dict | None = None,
) -> None:
    """Append one JSON line to CAPTURE_LOG for a notification event.

    Best-effort: wrap the whole body in try/except so a logging failure
    never raises and never blocks or duplicates a push.
    """
    try:
        CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now().astimezone().isoformat(),
            "source": source,
            "title": title,
            "message_head": message[:300],
            "priority": Priority(priority).name,
            "delivered": delivered,
            "extra": extra or {},
        }
        with open(CAPTURE_LOG, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except Exception as e:
        log.warning(f"failed to log captured event: {e}")


def send_notification(
    message: str,
    title: str = "Conductor",
    priority: Priority = Priority.NORMAL,
    url: str = "",
    url_title: str = "",
    source: str = "unknown",
) -> bool:
    """Send a Pushover notification. Returns True on success."""
    # Truncate to Pushover limits
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[: MAX_MESSAGE_LENGTH - 3] + "..."
    if len(title) > MAX_TITLE_LENGTH:
        title = title[: MAX_TITLE_LENGTH - 3] + "..."

    user_key = os.environ.get("PUSHOVER_USER_KEY", "")
    app_token = os.environ.get("PUSHOVER_APP_TOKEN", "")
    if not user_key or not app_token:
        delivered = False
    else:
        payload = {
            "token": app_token,
            "user": user_key,
            "message": message,
            "title": title,
            "priority": int(priority),
        }
        if url:
            payload["url"] = url
        if url_title:
            payload["url_title"] = url_title

        try:
            resp = requests.post(PUSHOVER_API_URL, data=payload, timeout=15)
            delivered = resp.status_code == 200
        except Exception:
            delivered = False

    _capture_event(source=source, message=message, title=title, priority=priority, delivered=delivered)
    return delivered
