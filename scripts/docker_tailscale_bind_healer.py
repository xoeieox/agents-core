#!/usr/bin/env python3
"""Docker Tailscale Bind Healer - see docker_health.py / docker-tailscale-bind-healer-v0
spec for the full incident writeup. Runs once at boot: waits (bounded) for BRIX's
Tailscale IP to actually be present on tailscale0, then force-recreates any
container left broken by the boot-ordering race against tailscaled.

SAFE BY DEFAULT: with no flags, this only LOGS what it would do (--dry-run is the
implicit default). Pass --apply to actually run recreate commands. The systemd unit
ships invoking this with NO flags (dry-run) until Erah's manual verification cycle
(see spec Deploy note) confirms zero false positives on this host, at which point
the unit's ExecStart is edited to add --apply.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, "/srv/agents/scripts")
from agents_core.notify import send_notification, Priority
from docker_health import is_container_broken, has_unexpected_bind_ip, compose_recreate_command

PACIFIC = ZoneInfo("America/Los_Angeles")
LOG_FILE = Path("/srv/agents/logs/docker-tailscale-bind-healer.log")
TAILSCALE_IP = "203.0.113.10"
LAN_IP = "192.168.50.69"
KNOWN_GOOD_IPS = {TAILSCALE_IP, LAN_IP, "127.0.0.1"}
WAIT_TIMEOUT_SECONDS = 60
WAIT_POLL_SECONDS = 2


def log(msg: str):
    ts = datetime.now(PACIFIC).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def wait_for_tailscale_ip() -> bool:
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["ip", "-4", "-o", "addr", "show", "tailscale0"],
                capture_output=True, text=True, timeout=5,
            )
            if TAILSCALE_IP in out.stdout:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(WAIT_POLL_SECONDS)
    return False


def list_containers() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout.split()


def inspect(name: str) -> dict | None:
    try:
        out = subprocess.run(
            ["docker", "inspect", name],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return json.loads(out.stdout)[0]
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            json.JSONDecodeError, IndexError) as exc:
        log(f"SKIP {name}: could not inspect ({exc})")
        return None


def remediate(name: str, inspect_data: dict, apply: bool) -> str:
    """Returns one of: 'healed', 'failed', 'skipped-no-compose-labels',
    'skipped-unexpected-bind-ip', 'dry-run'."""
    if has_unexpected_bind_ip(inspect_data, KNOWN_GOOD_IPS):
        log(f"SKIP {name}: configured bind IP not in {KNOWN_GOOD_IPS} - this may be "
            f"genuine config drift (e.g. Tailscale re-key), not the boot race this "
            f"tool targets. Needs human judgment, not auto-recreate.")
        return "skipped-unexpected-bind-ip"

    cmd = compose_recreate_command(inspect_data)
    if cmd is None:
        log(f"SKIP {name}: broken but not compose-managed (no project labels), "
            f"cannot safely recreate")
        return "skipped-no-compose-labels"

    if not apply:
        log(f"DRY-RUN would recreate {name}: {' '.join(cmd)}")
        return "dry-run"

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        log(f"Recreated {name}: {' '.join(cmd)}")
        return "healed"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        log(f"Recreate FAILED for {name}: {exc}")
        return "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                         help="Actually run recreate commands. Without this, only logs "
                              "what would happen (default: dry-run).")
    args = parser.parse_args()

    ip_ready = wait_for_tailscale_ip()
    log(f"tailscale0 has {TAILSCALE_IP}: {ip_ready}" if ip_ready else
        f"WARNING: {TAILSCALE_IP} not seen on tailscale0 after "
        f"{WAIT_TIMEOUT_SECONDS}s, proceeding anyway")
    log(f"mode: {'APPLY (will recreate)' if args.apply else 'DRY-RUN (log only)'}")

    try:
        names = list_containers()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        log(f"FATAL: could not list containers: {exc}")
        return

    results = {}
    for name in names:
        data = inspect(name)
        if data is None:
            continue
        if is_container_broken(data):
            log(f"BROKEN: {name}")
            results[name] = remediate(name, data, args.apply)
        else:
            log(f"ok: {name}")

    if results:
        lines = [f"Post-boot healer ran ({'APPLY' if args.apply else 'DRY-RUN'} mode)."]
        for outcome in ("healed", "dry-run", "failed",
                        "skipped-no-compose-labels", "skipped-unexpected-bind-ip"):
            names_for = [n for n, o in results.items() if o == outcome]
            if names_for:
                lines.append(f"{outcome}: {', '.join(names_for)}")
        send_notification(
            "\n".join(lines),
            title="Docker Tailscale Bind Healer",
            priority=Priority.HIGH,  # confirmed valid: agents_core/notify.py's Priority enum
                                     # is {LOWEST=-2, LOW=-1, NORMAL=0, HIGH=1}
        )
    log(f"Done. {results or 'nothing broken'}")


if __name__ == "__main__":
    main()
