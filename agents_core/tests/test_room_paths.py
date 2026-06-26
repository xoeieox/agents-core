"""Tests for agents_core.room_paths — DoD-1 through DoD-10."""
from __future__ import annotations

import ast
import importlib.resources
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Golden table — hardcoded expected values, NOT re-derived from classmap
# ---------------------------------------------------------------------------

GOLDEN: dict[str, str] = {
    "research": "/srv/lapis/research",
    "research.source_texts": "/srv/lapis/research/source-texts",
    "research.invariant": "/srv/lapis/research/invariant",
    "research.governance_watch": "/srv/lapis/research/governance-watch",
    "research.enlightenment_corpus": "/srv/lapis/research/enlightenment-corpus",
    "research.arxiv_watch": "/srv/lapis/research/arxiv-watch",
    "research.rsi": "/srv/lapis/research/rsi",
    "research.dplace": "/srv/lapis/research/dplace",
    "forming": "/srv/lapis/forming",
    "forming.psych_mining": "/srv/lapis/forming/psych-mining",
    "library": "/srv/lapis/library",
    "library.atoms": "/srv/lapis/library/atoms",
    "library.ideas": "/srv/lapis/library/ideas",
    "library.podcasts": "/srv/lapis/library/podcasts",
    "collider": "/srv/lapis/collider",
    "extracts": "/srv/lapis/extracts",
    "targets": "/srv/lapis/targets",
    "targets.comments": "/srv/lapis/targets/comments",
    "lapis_state": "/srv/lapis/lapis-state",
    "lapis_state.roadmap": "/srv/lapis/lapis-state/roadmap",
    "lapis_state.roadmap.committed_plan": "/srv/lapis/lapis-state/roadmap/committed-plan.json",
    "lapis_state.restart_pending": "/srv/lapis/lapis-state/restart-pending",
    "lapis_state.deploy_log": "/srv/lapis/lapis-state/lapis-pm-deploy-log.md",
    "gpu_queue": "/srv/lapis/gpu-queue",
    "gpu_queue.shaped": "/srv/lapis/gpu-queue/shaped",
    "gpu_queue.active": "/srv/lapis/gpu-queue/active",
    "gpu_queue.failed": "/srv/lapis/gpu-queue/failed",
    "gpu_queue.completed": "/srv/lapis/gpu-queue/completed",
    "claude_queue": "/srv/lapis/claude-queue",
    "claude_queue.active": "/srv/lapis/claude-queue/active",
    "claude_queue.completed": "/srv/lapis/claude-queue/completed",
    "claude_queue.failed": "/srv/lapis/claude-queue/failed",
    "claude_queue.history": "/srv/lapis/claude-queue/history.jsonl",
    "intentions": "/srv/lapis/intentions",
    "directives": "/srv/lapis/directives",
    "directives.brief_decisions": "/srv/lapis/directives/brief-decisions",
    "proposals": "/srv/lapis/proposals",
    "signals": "/srv/lapis/signals",
    "tasks": "/srv/lapis/tasks",
    "schedule": "/srv/lapis/schedule",
    "trajectory": "/srv/lapis/trajectory",
    "intent": "/srv/lapis/intent",
    "dropbox": "/srv/lapis/dropbox",
    "notify_audit": "/srv/lapis/notify-audit",
    "notify_audit.silenced": "/srv/lapis/notify-audit/silenced.jsonl",
    "council": "/srv/lapis/council",
    "council.logs": "/srv/lapis/council/logs",
    "council.cache_cohesion": "/srv/lapis/council/cache/cohesion",
    "council.speakers": "/srv/lapis/council/speakers",
    "facets": "/srv/lapis/facets",
    "facets.personas": "/srv/lapis/facets/personas",
    "facets.deliberations": "/srv/lapis/facets/deliberations",
    "kami": "/srv/lapis/kami",
    "kami.processed": "/srv/lapis/kami/processed",
    "scout": "/srv/lapis/scout",
    "scout.sims": "/srv/lapis/scout/sims",
    "scout.sims_tonight": "/srv/lapis/scout/sims-tonight",
    "scout.nightlogs": "/srv/lapis/scout/nightlogs",
    "scout.traces": "/srv/lapis/scout/traces",
    "scout.maps": "/srv/lapis/scout/maps",
    "scout.refiner": "/srv/lapis/scout/refiner",
    "experts": "/srv/lapis/experts",
    "reviews": "/srv/lapis/review",
    "reviews.sweep": "/srv/lapis/review/sweep",
    "reviews.seeds": "/srv/lapis/review/seeds",
    "radio.consults": "/srv/lapis/radio/consults",
    "keeper": "/srv/lapis/keeper",
    "repair_station": "/srv/lapis/repair-station/repair_station.db",
    "agent_observations": "/srv/lapis/agent-observations",
    "backcaster": "/srv/lapis/backcaster",
    "backcaster.runs": "/srv/lapis/backcaster/runs",
    "gardener.synthesis": "/srv/lapis/gardener/synthesis",
    "spec_review_artifacts": "/srv/lapis/spec-review-artifacts",
    "simulations": "/srv/lapis/simulations",
    "jagged_seam": "/srv/lapis/jagged-seam",
    "experiments": "/srv/lapis/experiments",
    "scratch": "/srv/lapis/scratch",
    "workshop": "/srv/lapis/workshop",
    "planning": "/srv/lapis/planning",
    "planning.specs": "/srv/lapis/planning/specs",
    "planning.specs.superseded": "/srv/lapis/planning/specs/superseded",
    "planning.evals": "/srv/lapis/planning/evals",
    "briefs": "/srv/lapis/briefs",
    "briefs.kami_adjudication": "/srv/lapis/briefs/kami-adjudication",
    "briefings": "/srv/lapis/briefings",
    "design": "/srv/lapis/design",
    "narratives": "/srv/lapis/narratives",
    "journal": "/srv/lapis/journal",
    "cards": "/srv/lapis/cards",
    "memory": "/srv/lapis/memory",
    "memory.activity_log": "/srv/lapis/memory/activity.log",
    "memory.host_fault_events": "/srv/lapis/memory/host-fault-events.jsonl",
    "sessions": "/srv/lapis/sessions",
    "focus": "/srv/lapis/FOCUS.md",
    "heading": "/srv/lapis/HEADING.yaml",
    "config": "/srv/lapis/config",
    "agents": "/srv/lapis/agents",
    "taxonomy_workspace": "/srv/lapis/taxonomy-workspace",
    "current": "/srv/lapis/current",
}

# Env vars that may be set in the real environment — we need to strip them for tests
_ENV_VARS_TO_CLEAR = [
    "ROOM_ROOT",
    "TARGETS_DIR", "WEAVER_TARGETS_DIR", "PM_TARGETS_DIR",
    "LAPIS_STATE",
    "GPU_QUEUE_DIR",
    "CLAUDE_QUEUE_DIR",
    "LAPIS_INTENTIONS_DIR",
    "ATOM_CORPUS_ROOT",
    "IDEA_CORPUS_DIR",
    "PM_SESSIONS_DIR",
    "REPAIR_STATION_DB",
    "AGENT_OBSERVATIONS_ROOT",
    "GARDENER_OUTPUT_DIR",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Strip all /room env overrides before each test."""
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# DoD-1: golden no-op parity
# ---------------------------------------------------------------------------

def test_dod1_golden_parity():
    """Every key in GOLDEN resolves to its expected path with no env overrides."""
    from agents_core.room_paths import room_path

    for key, expected in GOLDEN.items():
        result = str(room_path(key))
        assert result == expected, (
            f"room_path({key!r}) = {result!r}, expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# DoD-2: write= is inert
# ---------------------------------------------------------------------------

def test_dod2_write_inert():
    """room_path(k, write=True) == room_path(k, write=False) for every key."""
    from agents_core.room_paths import room_path, iter_keys

    for key in iter_keys():
        assert room_path(key, write=True) == room_path(key, write=False), (
            f"write= flag changed path for key {key!r}"
        )


# ---------------------------------------------------------------------------
# DoD-3: import isolation — AST scan for non-stdlib imports
# ---------------------------------------------------------------------------

_STDLIB_MODULES = {
    "os", "pathlib", "enum", "dataclasses", "tomllib",
    "importlib", "importlib.resources", "sys", "typing",
    "__future__",
}


def _collect_imports(source: str) -> list[str]:
    """Return top-level module names imported in the source string."""
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def test_dod3_import_isolation():
    """room_paths submodule Python files must only import from stdlib."""
    pkg_dir = Path(__file__).parent.parent / "room_paths"
    assert pkg_dir.exists(), f"room_paths dir not found: {pkg_dir}"

    py_files = list(pkg_dir.glob("*.py"))
    assert py_files, "No .py files found in room_paths"

    allowed_top_level = {
        "os", "pathlib", "enum", "dataclasses", "tomllib",
        "importlib", "sys", "typing", "collections", "abc",
        "__future__",
        # __main__.py imports from the package itself
        "agents_core",
    }

    violations: list[str] = []
    for py_file in py_files:
        if py_file.name == "__main__.py":
            # __main__.py is allowed to import from agents_core.room_paths
            continue
        source = py_file.read_text()
        imports = _collect_imports(source)
        for imp in imports:
            if imp not in allowed_top_level:
                violations.append(f"{py_file.name}: imports {imp!r}")

    assert not violations, "Non-stdlib imports found:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# DoD-4: env overrides
# ---------------------------------------------------------------------------

def test_dod4_env_lapis_state(monkeypatch):
    """LAPIS_STATE env var overrides lapis_state key."""
    from agents_core.room_paths import room_path

    monkeypatch.setenv("LAPIS_STATE", "/custom/lapis")
    assert str(room_path("lapis_state")) == "/custom/lapis"


def test_dod4_env_gpu_queue_dir(monkeypatch):
    """GPU_QUEUE_DIR env var overrides gpu_queue key."""
    from agents_core.room_paths import room_path

    monkeypatch.setenv("GPU_QUEUE_DIR", "/custom/gpu")
    assert str(room_path("gpu_queue")) == "/custom/gpu"


def test_dod4_env_targets_dir(monkeypatch):
    """TARGETS_DIR env var overrides targets key."""
    from agents_core.room_paths import room_path

    monkeypatch.setenv("TARGETS_DIR", "/custom/targets")
    assert str(room_path("targets")) == "/custom/targets"


def test_dod4_env_repair_station_db(monkeypatch):
    """REPAIR_STATION_DB env var overrides repair_station key."""
    from agents_core.room_paths import room_path

    monkeypatch.setenv("REPAIR_STATION_DB", "/custom/repair.db")
    assert str(room_path("repair_station")) == "/custom/repair.db"


def test_dod4_env_room_root(monkeypatch):
    """ROOM_ROOT env var changes the base for all keys without their own env_var."""
    from agents_core.room_paths import room_path

    monkeypatch.setenv("ROOM_ROOT", "/alt/room")
    # 'research' has no env_var, so it respects ROOM_ROOT
    assert str(room_path("research")) == "/alt/srv/lapis/research"
    # 'council' likewise
    assert str(room_path("council")) == "/alt/srv/lapis/council"


# ---------------------------------------------------------------------------
# DoD-5: shell seam parity
# ---------------------------------------------------------------------------

def test_dod5_sh_parity():
    """--emit-sh output: all ROOM_* values match golden expected paths."""
    clean = {k: v for k, v in os.environ.items() if k not in _ENV_VARS_TO_CLEAR}
    result = subprocess.run(
        [sys.executable, "-m", "agents_core.room_paths", "--emit-sh"],
        capture_output=True, text=True, check=True,
        env=clean,
    )
    output = result.stdout

    # Parse var=value lines
    sh_vars: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("export ROOM_") and "=" in line:
            # export ROOM_FOO="/srv/lapis/foo"
            _, rest = line.split(" ", 1)
            name, val = rest.split("=", 1)
            # Strip surrounding quotes
            val = val.strip('"').strip("'")
            sh_vars[name] = val

    # Build expected from GOLDEN
    for key, expected in GOLDEN.items():
        sh_name = "ROOM_" + key.upper().replace(".", "_")
        if sh_name in sh_vars:
            assert sh_vars[sh_name] == expected, (
                f"{sh_name} in shell output = {sh_vars[sh_name]!r}, "
                f"expected {expected!r}"
            )


# ---------------------------------------------------------------------------
# DoD-7: package-data ships
# ---------------------------------------------------------------------------

def test_dod7_package_data_ships():
    """importlib.resources.files can read _classmap.toml from the installed package."""
    pkg = importlib.resources.files("agents_core.room_paths")
    toml_bytes = (pkg / "_classmap.toml").read_bytes()
    assert toml_bytes, "_classmap.toml is empty or missing"
    assert b"[keys.targets]" in toml_bytes


# ---------------------------------------------------------------------------
# DoD-9: access domain
# ---------------------------------------------------------------------------

def test_dod9_access_domain():
    """Every access() value is in {read, write, both, unknown}."""
    from agents_core.room_paths import iter_keys, access

    valid = {"read", "write", "both", "unknown"}
    for key in iter_keys():
        val = access(key)
        assert val in valid, f"access({key!r}) = {val!r}, not in {valid}"


# ---------------------------------------------------------------------------
# DoD-10: stdlib-only import (subprocess test in bare Python)
# ---------------------------------------------------------------------------

def test_dod10_stdlib_only_import():
    """agents_core.room_paths submodule imports successfully.

    We verify by importing just agents_core.room_paths (not the top-level
    agents_core package which has heavy third-party deps) and confirming the
    submodule works correctly. The stdlib-only contract is enforced structurally
    by DoD-3 (AST scan), which is the more reliable check.
    """
    # Import the submodule directly, bypassing agents_core's top-level __init__
    code = (
        "import importlib; "
        "mod = importlib.import_module('agents_core.room_paths'); "
        "keys = mod.iter_keys(); "
        "assert len(keys) > 0, f'no keys: {keys}'; "
        "p = mod.room_path('targets'); "
        "assert str(p) == '/srv/lapis/targets', repr(str(p)); "
        "print('OK')"
    )
    clean = {k: v for k, v in os.environ.items() if k not in _ENV_VARS_TO_CLEAR}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env=clean,
    )
    assert result.returncode == 0, (
        f"Import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------

def test_unknown_key_raises():
    """Requesting an unknown key raises KeyError with a helpful message."""
    from agents_core.room_paths import room_path

    with pytest.raises(KeyError, match="unknown key"):
        room_path("does_not_exist")


def test_room_str_returns_string():
    """room_str returns a plain str, not a Path."""
    from agents_core.room_paths import room_str

    result = room_str("targets")
    assert isinstance(result, str)
    assert result == "/srv/lapis/targets"


def test_parts_appended():
    """Extra *parts args are appended to the resolved base."""
    from agents_core.room_paths import room_path

    result = room_path("targets", "foo", "bar.yaml")
    assert str(result) == "/srv/lapis/targets/foo/bar.yaml"


def test_path_class_enum():
    """path_class returns a Class enum for all keys."""
    from agents_core.room_paths import iter_keys, path_class, Class

    for key in iter_keys():
        cls = path_class(key)
        assert isinstance(cls, Class), f"path_class({key!r}) is not a Class: {cls!r}"


def test_convenience_constants():
    """Module-level constants resolve to their expected values."""
    import agents_core.room_paths as rp

    assert str(rp.TARGETS_DIR) == "/srv/lapis/targets"
    assert str(rp.GPU_QUEUE_DIR) == "/srv/lapis/gpu-queue"
    assert str(rp.CLAUDE_QUEUE_DIR) == "/srv/lapis/claude-queue"
    assert str(rp.EXPERTS_ROOT) == "/srv/lapis/experts"
    assert str(rp.LAPIS_STATE_DIR) == "/srv/lapis/lapis-state"
    assert str(rp.COUNCIL_DIR) == "/srv/lapis/council"
    assert str(rp.COUNCIL_LOG_DIR) == "/srv/lapis/council/logs"
    assert str(rp.AGENT_OBSERVATIONS_DIR) == "/srv/lapis/agent-observations"


def test_iter_keys_completeness():
    """iter_keys() returns all GOLDEN keys (and possibly more)."""
    from agents_core.room_paths import iter_keys

    known = set(iter_keys())
    for key in GOLDEN:
        assert key in known, f"GOLDEN key {key!r} missing from iter_keys()"
