#!/usr/bin/env python3
"""Pushover notification module — single shared implementation for all Conductor scripts.

Usage:
    from notify import send_notification, Priority
    send_notification("Something happened", title="Alert", priority=Priority.HIGH)

Credentials are read from env vars PUSHOVER_USER_KEY and PUSHOVER_APP_TOKEN,
which are set via systemd EnvironmentFile (conductor.env).
"""

import os
from enum import IntEnum

import requests

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

# Pushover limits
MAX_MESSAGE_LENGTH = 1024
MAX_TITLE_LENGTH = 250


class Priority(IntEnum):
    LOWEST = -2
    LOW = -1
    NORMAL = 0
    HIGH = 1


def send_notification(
    message: str,
    title: str = "Conductor",
    priority: Priority = Priority.NORMAL,
    url: str = "",
    url_title: str = "",
) -> bool:
    """Send a Pushover notification. Returns True on success."""
    user_key = os.environ.get("PUSHOVER_USER_KEY", "")
    app_token = os.environ.get("PUSHOVER_APP_TOKEN", "")

    if not user_key or not app_token:
        return False

    # Truncate to Pushover limits
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[: MAX_MESSAGE_LENGTH - 3] + "..."
    if len(title) > MAX_TITLE_LENGTH:
        title = title[: MAX_TITLE_LENGTH - 3] + "..."

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
        return resp.status_code == 200
    except Exception:
        return False
