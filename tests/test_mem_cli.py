"""Tests for the mem CLI promote subcommand + --store flag + MEM_PRINCIPAL
env default (openclaw-memdb-influx-reader-v0, D2 / Files-changed / panel F7).

The spec's Files-changed line names the conductor CLI (scripts/mem.py, a
separate repo) as the live surface; this module (agents_core/mem_cli.py)
ships the argparse wiring + the HTTP dispatch so the curation surface
(`mem promote`) is exercisable from agents-core alone, and so the
MEM_PRINCIPAL env default is named + tested here.

Covers:
  - promote routes through MemClient (HTTP) with the right principal
  - MEM_PRINCIPAL env default (the BRIX-side principal for the mem CLI)
  - the brix-pm default when MEM_PRINCIPAL is unset (D1 / panel F7)
  - --store flag validation (loud ValueError on a typo; rescoped MOOT)
  - promote REFUSES (loud exit 2) when MEM_SERVER is unset (D2: never a
    side effect — the server owns the provenance shape + batch key)
  - --from newline-injection rejected client-side (loud exit 2)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import respx

from agents_core.mem_cli import (
    DEFAULT_CLI_PRINCIPAL,
    build_parser,
    cli,
)
from agents_core.mem_client import (
    MemClient,
    STORE_ATOMS,
    STORE_MACHINERY,
    validate_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = "http://test-mem-server:8403"


# ---------------------------------------------------------------------------
# --store flag validation (rescoped MOOT, but loud on a typo)
# ---------------------------------------------------------------------------

def test_validate_store_known_values():
    assert validate_store(STORE_ATOMS) == STORE_ATOMS
    assert validate_store(STORE_MACHINERY) == STORE_MACHINERY
    # 'exhaust' is an alias for 'machinery' (the rescoped store name).
    assert validate_store("exhaust") == STORE_MACHINERY


def test_validate_store_rejects_typo():
    with pytest.raises(ValueError, match="unknown --store"):
        validate_store("machinery-typo")
    with pytest.raises(ValueError, match="unknown --store"):
        validate_store("")


# ---------------------------------------------------------------------------
# MEM_PRINCIPAL env default (D1 / panel F7)
# ---------------------------------------------------------------------------

def test_default_cli_principal_is_brix_pm():
    """The BRIX-side default principal for the mem CLI is brix-pm (D1 /
    panel F7): the primary local path (the mem CLI over the tailscale IP)
    lands as the registered writer, not a reader (fail-closed)."""
    assert DEFAULT_CLI_PRINCIPAL == "brix-pm"


@respx.mock
def test_cli_promote_uses_mem_principal_env(monkeypatch):
    """MEM_PRINCIPAL env is the caller's principal (D1): when set, the
    promote dispatch uses it (not the brix-pm default)."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.setenv("MEM_PRINCIPAL", "zephyr-deposit")
    route = respx.post(f"{BASE}/v0/promote").mock(
        return_value=httpx.Response(200, json={
            "key": "finding/x", "content": "h", "tags": "promoted",
            "source": "promoted:openclaw/gw", "created_at": "x",
            "updated_at": "y", "batch_key": "decision/memdb-promotion-20260914-zephyr-deposit",
        })
    )
    rc = cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding"])
    assert rc == 0
    sent = route.calls[0].request
    # The MEM_PRINCIPAL env principal is sent as the X-Mem-Principal header.
    assert sent.headers["X-Mem-Principal"] == "zephyr-deposit"


@respx.mock
def test_cli_promote_defaults_to_brix_pm_when_no_env(monkeypatch):
    """When MEM_PRINCIPAL is unset and no --by is given, the promote
    dispatch defaults to brix-pm (the BRIX-side curator principal, D1 /
    panel F7) so the primary local path lands as the registered writer."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    route = respx.post(f"{BASE}/v0/promote").mock(
        return_value=httpx.Response(200, json={
            "key": "finding/x", "content": "h", "tags": "promoted",
            "source": "promoted:openclaw/gw", "created_at": "x",
            "updated_at": "y", "batch_key": "decision/memdb-promotion-20260914-brix-pm",
        })
    )
    rc = cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding"])
    assert rc == 0
    sent = route.calls[0].request
    assert sent.headers["X-Mem-Principal"] == "brix-pm"


@respx.mock
def test_cli_promote_by_flag_wins_over_env(monkeypatch):
    """An explicit --by wins over the MEM_PRINCIPAL env default."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.setenv("MEM_PRINCIPAL", "zephyr-deposit")
    route = respx.post(f"{BASE}/v0/promote").mock(
        return_value=httpx.Response(200, json={
            "key": "finding/x", "content": "h", "tags": "promoted",
            "source": "promoted:openclaw/gw", "created_at": "x",
            "updated_at": "y", "batch_key": "decision/memdb-promotion-20260914-brix-pm",
        })
    )
    rc = cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding",
              "--by", "brix-pm"])
    assert rc == 0
    sent = route.calls[0].request
    assert sent.headers["X-Mem-Principal"] == "brix-pm"


# ---------------------------------------------------------------------------
# --store flag (rescoped MOOT, but accepted + validated)
# ---------------------------------------------------------------------------

@respx.mock
def test_cli_promote_accepts_store_machinery(monkeypatch):
    """The --store machinery flag is accepted (rescoped MOOT at the HTTP
    layer — the server routes machine-state keys transparently)."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    route = respx.post(f"{BASE}/v0/promote").mock(
        return_value=httpx.Response(200, json={
            "key": "finding/x", "content": "h", "tags": "promoted",
            "source": "promoted:openclaw/gw", "created_at": "x",
            "updated_at": "y", "batch_key": "decision/memdb-promotion-20260914-brix-pm",
        })
    )
    rc = cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding",
              "--store", "machinery"])
    assert rc == 0


@respx.mock
def test_cli_promote_rejects_unknown_store(monkeypatch, capsys):
    """An unknown --store value is rejected by argparse (loud, exit 2)."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    with pytest.raises(SystemExit) as exc:
        cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding",
             "--store", "machinery-typo"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# promote REFUSES when MEM_SERVER is unset (D2: never a side effect)
# ---------------------------------------------------------------------------

def test_cli_promote_refused_without_mem_server(monkeypatch, capsys):
    """Promote is REFUSED (loud exit 2) when MEM_SERVER is unset: promotion
    is an explicit, server-side verb with a provenance shape the server
    owns; a local MemoryStore-only promote would bypass the server's
    --from shape check + batch key (the side-effect the spec forbids)."""
    monkeypatch.delenv("MEM_SERVER", raising=False)
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    rc = cli(["promote", "--from", "openclaw/gw", "finding/x", "the finding"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "MEM_SERVER" in err


# ---------------------------------------------------------------------------
# --from newline-injection rejected client-side (loud exit 2)
# ---------------------------------------------------------------------------

def test_cli_promote_rejects_newline_ref(monkeypatch, capsys):
    """A --from ref with a newline is rejected client-side (loud exit 2):
    the client validates before any request is sent (panel security F6)."""
    monkeypatch.setenv("MEM_SERVER", BASE)
    monkeypatch.delenv("MEM_PRINCIPAL", raising=False)
    rc = cli(["promote", "--from", "openclaw/gw\n[evil]", "finding/x",
              "the finding"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "path-like" in err or "control" in err or "newline" in err.lower()


# ---------------------------------------------------------------------------
# The parser wiring (the conductor CLI adds this subparser)
# ---------------------------------------------------------------------------

def test_parser_promote_subcommand_exists():
    """The promote subcommand is wired into the parser (the conductor CLI
    adds this subparser to its own `sub` group). The canonical invocation
    puts --from BEFORE the positionals (argparse cannot handle a required
    option sandwiched between positionals)."""
    parser = build_parser()
    args = parser.parse_args(
        ["promote", "--from", "openclaw/gw", "finding/x", "the finding"]
    )
    assert args.command == "promote"
    assert args.key == "finding/x"
    assert args.from_ref == "openclaw/gw"
    assert args.content == "the finding"
    assert args.by == ""
    assert args.tags == ""
    assert args.rationale == ""
    assert args.store == STORE_ATOMS


def test_parser_promote_optional_flags():
    """The optional options (--by, --tags, --rationale, --store) must come
    AFTER the positionals (key, content) on the command line (argparse
    ordering constraint)."""
    parser = build_parser()
    args = parser.parse_args(
        ["promote", "--from", "openclaw/gw", "finding/x", "the finding",
         "--by", "brix-pm", "--tags", "openclaw,friction",
         "--rationale", "friction-cluster insight",
         "--store", "machinery"]
    )
    assert args.by == "brix-pm"
    assert args.tags == "openclaw,friction"
    assert args.rationale == "friction-cluster insight"
    assert args.store == STORE_MACHINERY
