"""Detect Docker containers left in a broken port-binding state by a boot-ordering
race against tailscaled (or any other reason a configured publish never went live).
See docker-tailscale-bind-healer-v0 spec / infra/vault-rag-oom-crashloop-host-oom-2026-07-16.

Deliberately NOT part of the agents_core package - single-consumer, BRIX-host-specific
logic stays in scripts/ per this repo's own SPEC.md ("Deferred candidates" section).
"""
from __future__ import annotations

from typing import Any


def is_container_broken(inspect_data: dict[str, Any]) -> bool:
    """True if this container is configured to publish ports, is supposed to be
    running (RestartPolicy in {'unless-stopped', 'always'} - both mean "Docker
    should keep this running"; only 'no'/'on-failure' are treated as intentionally
    stoppable), and its actual live state doesn't match: either it isn't running at
    all, or at least one of its configured published ports is missing/null in
    NetworkSettings.Ports.

    Containers with any other RestartPolicy (e.g. on-demand/proxy-managed services
    running restart: "no") are never considered broken by this check - their
    stopped state is by design, not a failure.

    Containers with no configured PortBindings at all (nothing published at the
    Docker level - e.g. a container only reachable via another container's Docker
    network, fronted by a proxy) are never considered broken - there's nothing to
    verify.
    """
    host_config = inspect_data.get("HostConfig") or {}
    if (host_config.get("RestartPolicy") or {}).get("Name") not in ("unless-stopped", "always"):
        return False

    configured = host_config.get("PortBindings") or {}
    if not configured:
        return False

    if not (inspect_data.get("State") or {}).get("Running"):
        return True

    live = (inspect_data.get("NetworkSettings") or {}).get("Ports") or {}
    return any(not live.get(port_key) for port_key in configured)


def configured_bind_ips(inspect_data: dict[str, Any]) -> set[str]:
    """All distinct HostIp values across this container's configured PortBindings."""
    configured = (inspect_data.get("HostConfig") or {}).get("PortBindings") or {}
    ips = set()
    for entries in configured.values():
        for entry in entries or []:
            host_ip = entry.get("HostIp")
            if host_ip:
                ips.add(host_ip)
    return ips


def has_unexpected_bind_ip(inspect_data: dict[str, Any], known_good_ips: set[str]) -> bool:
    """True if any configured bind IP is NOT one of the host's current known-good
    IPs. Guards against the healer "fixing" a container that's actually failing for
    a DIFFERENT reason than the boot race this tool targets - e.g. genuine Tailscale
    re-keying/subnet migration, or hand-edited config pointing at a retired IP.
    Force-recreating in that case wouldn't even help (same broken config, same
    failure) and would mask a real config-drift problem behind a "healed" log line.
    """
    configured = configured_bind_ips(inspect_data)
    return bool(configured - known_good_ips)


def compose_recreate_command(inspect_data: dict[str, Any]) -> list[str] | None:
    """Build the `docker compose ... up -d --force-recreate <service>` argv needed
    to fix a broken container, using its own compose-project labels (works
    regardless of which repo/compose file owns it - no hardcoded per-service
    mapping). Returns None if the container isn't compose-managed (no project
    labels), meaning it can't be safely reconstructed by this script.
    """
    labels = (inspect_data.get("Config") or {}).get("Labels") or {}
    config_files = labels.get("com.docker.compose.project.config_files")
    working_dir = labels.get("com.docker.compose.project.working_dir")
    service = labels.get("com.docker.compose.service")
    if not (config_files and working_dir and service):
        return None
    return [
        "docker", "compose",
        "--project-directory", working_dir,
        "-f", config_files,
        "up", "-d", "--force-recreate", service,
    ]
