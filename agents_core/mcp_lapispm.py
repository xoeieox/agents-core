"""MCP stdio server exposing `lapis-pm` (status/list/brief/bind/tick) as typed
tools for opencode, via subprocess + stdout parsing.

`lapis-pm` is a pure argparse CLI with no importable library API (see spec
opencode-mcp-mem-lapispm-v0). This wrapper shells the exact same subcommands
a human/bash-shelling model would run and parses stdout into structured
fields, per a strict parsing contract:

- every response carries `raw_output`, `exit_code`, `parse_status`
- a non-zero exit short-circuits parsing entirely (`parse_status: cli_error`)
- on exit 0, a field is only included if its pattern matches exactly once;
  ambiguous/absent fields are omitted rather than guessed
- `parse_status` is "ok" only if every expected field for that call parsed
  cleanly; otherwise "partial" — callers should fall back to `raw_output`
"""

from __future__ import annotations

import json
import re
import subprocess

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lapis-pm")


def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["lapis-pm", *argv], capture_output=True, text=True)


def _cli_error(proc: subprocess.CompletedProcess) -> dict:
    return {
        "raw_output": proc.stdout,
        "exit_code": proc.returncode,
        "stderr": proc.stderr,
        "parse_status": "cli_error",
    }


def _coerce_bool(value: str) -> bool | str:
    if value == "True":
        return True
    if value == "False":
        return False
    return value


_STATUS_FIELD_PATTERNS = {
    "title": r"^\s*title:\s+(.*)$",
    "pm_bound": r"^\s*pm_bound:\s+(\S+)$",
    "pm_repo": r"^\s*pm_repo:\s+(.*)$",
    "pm_authority": r"^\s*pm_authority:\s+(.*)$",
    "pm_verification": r"^\s*pm_verification:\s+(.*)$",
    "paused": r"^\s*paused:\s+(\S+)",
    "cursor": r"^\s*cursor:\s+(.*)$",
    "dispatched": r"^\s*dispatched:\s+(.*)$",
    "outstanding": r"^\s*outstanding:\s+(.*)$",
}


def _parse_status(stdout: str, target_id: str) -> tuple[dict, str]:
    blocks = re.findall(r"^=== (?!review-gate)(\S+) ===\n((?:(?!^===).)*)", stdout, re.M | re.S)
    if target_id:
        bodies = [body for tid, body in blocks if tid == target_id]
    else:
        bodies = [body for _, body in blocks]

    if len(bodies) != 1:
        return {}, "partial"

    body = bodies[0]
    parsed: dict = {}
    for field, pattern in _STATUS_FIELD_PATTERNS.items():
        matches = re.findall(pattern, body, re.M)
        if len(matches) == 1:
            value = matches[0].strip()
            parsed[field] = _coerce_bool(value) if field in ("pm_bound", "paused") else value

    parse_status = "ok" if len(parsed) == len(_STATUS_FIELD_PATTERNS) else "partial"
    return parsed, parse_status


@mcp.tool()
def lapispm_status(target_id: str = "", explain: bool = False) -> dict:
    """Show PM state for a bound target (or all bound targets if target_id is empty)."""
    argv = ["status"]
    if target_id:
        argv.append(target_id)
    if explain:
        argv.append("--explain")
    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)
    parsed, parse_status = _parse_status(proc.stdout, target_id)
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


@mcp.tool()
def lapispm_list() -> dict:
    """List all pm-bound targets."""
    proc = _run(["list", "--json"])
    if proc.returncode != 0:
        return _cli_error(proc)
    try:
        targets = json.loads(proc.stdout)
        return {
            "raw_output": proc.stdout,
            "exit_code": 0,
            "parse_status": "ok",
            "targets": targets,
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "raw_output": proc.stdout,
            "exit_code": 0,
            "parse_status": "partial",
        }


_BRIEF_FIELD_PATTERNS = {
    "brief_path": r"^Brief written:\s+(.+)$",
    "symlink": r"^Symlink updated:\s+(.+)$",
}


@mcp.tool()
def lapispm_brief() -> dict:
    """Generate a live state-of-work brief."""
    proc = _run(["brief", "--period", "live"])
    if proc.returncode != 0:
        return _cli_error(proc)
    parsed: dict = {}
    for field, pattern in _BRIEF_FIELD_PATTERNS.items():
        matches = re.findall(pattern, proc.stdout, re.M)
        if len(matches) == 1:
            parsed[field] = matches[0].strip()
    parse_status = "ok" if len(parsed) == len(_BRIEF_FIELD_PATTERNS) else "partial"
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


@mcp.tool()
def lapispm_bind(
    target_id: str,
    spec_from: str,
    repo: str,
    authority: str = "advisory",
    verification: str = "",
    force: bool = False,
    create: bool = False,
    title: str = "",
    description: str = "",
    destination_slug: str = "",
    destination_name: str = "",
    destination_when: str = "",
) -> dict:
    """Bind a target to the PM with a spec. Mirrors `lapis-pm bind`'s common flags."""
    argv = [
        "bind", target_id,
        "--spec-from", spec_from,
        "--repo", repo,
        "--authority", authority,
    ]
    if verification:
        argv += ["--verification", verification]
    if force:
        argv.append("--force")
    if create:
        argv.append("--create")
    if title:
        argv += ["--title", title]
    if description:
        argv += ["--description", description]
    if destination_slug:
        argv += ["--destination-slug", destination_slug]
    if destination_name:
        argv += ["--destination-name", destination_name]
    if destination_when:
        argv += ["--destination-when", destination_when]

    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)

    parsed: dict = {}
    bound_match = re.findall(
        r"^Bound (\S+) .*repo=([^,]+), authority=([^,]+), verification=(\S+)$",
        proc.stdout, re.M,
    )
    if len(bound_match) == 1:
        tid, bound_repo, bound_authority, bound_verification = bound_match[0]
        parsed["bound_target_id"] = tid
        parsed["bound_repo"] = bound_repo
        parsed["bound_authority"] = bound_authority
        parsed["bound_verification"] = bound_verification
    spec_match = re.findall(r"^Spec:\s+(\d+) chars$", proc.stdout, re.M)
    if len(spec_match) == 1:
        parsed["spec_chars"] = int(spec_match[0])

    parse_status = "ok" if len(bound_match) == 1 and len(spec_match) == 1 else "partial"
    return {
        "raw_output": proc.stdout,
        "exit_code": 0,
        "parse_status": parse_status,
        **parsed,
    }


_TICK_LINE_RE = re.compile(
    r"^\[(?P<target_id>\S+)\] skipped=(?P<skipped>\S+) reason=(?P<reason>\S+)"
    r"(?: reconciled=(?P<reconciled>\S+))? encoded=(?P<encoded>\S+) decision=(?P<decision>\S+)$"
)
_FORCE_DISPATCH_RE = re.compile(r"^Dispatched:\s+(\S+)\s+task_id=(\S+)$")


@mcp.tool()
def lapispm_tick(target_id: str = "", force_dispatch: str = "") -> dict:
    """Run one PM tick (single target, all bound targets, or a forced smoke dispatch)."""
    argv = ["tick"]
    if target_id:
        argv += ["--target", target_id]
    if force_dispatch:
        argv += ["--force-dispatch", force_dispatch]

    proc = _run(argv)
    if proc.returncode != 0:
        return _cli_error(proc)

    if force_dispatch:
        match = _FORCE_DISPATCH_RE.search(proc.stdout)
        if match:
            return {
                "raw_output": proc.stdout,
                "exit_code": 0,
                "parse_status": "ok",
                "agent_type": match.group(1),
                "task_id": match.group(2),
            }
        return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "partial"}

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    ticks = []
    unmatched = 0
    for line in lines:
        match = _TICK_LINE_RE.match(line)
        if match:
            ticks.append({k: v for k, v in match.groupdict().items() if v is not None})
        else:
            unmatched += 1

    if unmatched > 0:
        return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "partial", "ticks": ticks}
    return {"raw_output": proc.stdout, "exit_code": 0, "parse_status": "ok", "ticks": ticks}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
