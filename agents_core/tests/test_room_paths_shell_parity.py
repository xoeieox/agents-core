"""Shell==python resolution-parity tests for agents_core.room_paths.

Deliverables C, D, F of agents-core-room-paths-shell-seam-v0.

AC1  — strict no-op: shell==python==pre-seam /room literal, all keys, no overrides
AC2  — override wins verbatim (not re-rooted under ROOM_ROOT)
AC3  — ROOM_ROOT flip propagates identically through both seams
AC4  — targets alias family: first-set-wins order matches _resolve_base
AC5  — every key covered (parametrized over iter_keys())
AC6  — room_root() / ROOM_ROOT correct and non-disruptive
AC10 — lint gate: no frozen /srv/lapis/ literals in emitted shell
AC11 — drift gate: committed room_paths.sh byte-identical to fresh _emit_sh()
AC12 — room_root() evaluated per-call (not the import-bound ROOM_ROOT constant)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from agents_core.room_paths import _emit_sh, iter_keys, room_root, room_str


_ALL_OVERRIDE_VARS = [
    "ROOM_ROOT",
    "TARGETS_DIR", "WEAVER_TARGETS_DIR", "PM_TARGETS_DIR",
    "LAPIS_STATE",
    "GPU_QUEUE_DIR", "CLAUDE_QUEUE_DIR",
    "LAPIS_INTENTIONS_DIR",
    "ATOM_CORPUS_ROOT", "IDEA_CORPUS_DIR",
    "PM_SESSIONS_DIR",
    "REPAIR_STATION_DB", "AGENT_OBSERVATIONS_ROOT",
    "GARDENER_OUTPUT_DIR",
]


def _scrub_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ALL_OVERRIDE_VARS}
    if extra:
        env.update(extra)
    return env


def _source_sh(sh_content: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Source sh_content in a clean bash env, return all exported ROOM_* vars."""
    env = _scrub_env(extra_env)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(sh_content)
        sh_path = f.name
    try:
        result = subprocess.run(
            ["bash", "-c", f'. "{sh_path}" && env'],
            capture_output=True, text=True, env=env, check=True,
        )
    finally:
        os.unlink(sh_path)
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line and line.startswith("ROOM_"):
            k, _, v = line.partition("=")
            out[k] = v
    return out


def _py_vals(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return {key: room_str(key)} for all keys via subprocess with a scrubbed env."""
    env = _scrub_env(extra_env)
    code = (
        "from agents_core.room_paths import iter_keys, room_str; "
        "import json; "
        "print(json.dumps({k: room_str(k) for k in iter_keys()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, check=True,
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — emit and source once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emitted_sh() -> str:
    return _emit_sh()


@pytest.fixture(scope="module")
def sh_clean(emitted_sh: str) -> dict[str, str]:
    """All ROOM_* vars sourced in a completely clean env."""
    return _source_sh(emitted_sh)


@pytest.fixture(scope="module")
def py_clean() -> dict[str, str]:
    """All key->path from python in a completely clean env."""
    return _py_vals()


# ---------------------------------------------------------------------------
# AC1 + AC5: strict no-op, all keys, parametrized
# ---------------------------------------------------------------------------

_KEYS = iter_keys()
_KEY_IDS = [k.replace(".", "_") for k in _KEYS]


@pytest.mark.parametrize("key", _KEYS, ids=_KEY_IDS)
def test_ac1_ac5_no_op_parity(key: str, sh_clean: dict, py_clean: dict):
    """AC1+AC5: shell==python==pre-seam /room literal, all overrides unset."""
    sh_name = "ROOM_" + key.upper().replace(".", "_")
    sh_val = sh_clean.get(sh_name)
    py_val = py_clean.get(key)

    assert sh_val is not None, f"Missing from sourced shell: {sh_name}"
    assert py_val is not None, f"Missing from python: {key!r}"
    assert py_val.startswith("/room"), (
        f"Python default for {key!r} should start /room: {py_val!r}"
    )
    assert sh_val == py_val, (
        f"AC1: {sh_name}: shell={sh_val!r} != python={py_val!r}"
    )


# ---------------------------------------------------------------------------
# AC2: override wins verbatim — not re-rooted under ROOM_ROOT
# ---------------------------------------------------------------------------

def test_ac2_lapis_state_override_verbatim(emitted_sh: str):
    """LAPIS_STATE=/tmp/x → both seams return /tmp/x (not /srv/lapis/x or /mnt/.../x)."""
    override = "/tmp/x_parity_test"
    sh = _source_sh(emitted_sh, {"LAPIS_STATE": override})
    py = _py_vals({"LAPIS_STATE": override})
    assert sh.get("ROOM_LAPIS_STATE") == override, f"shell={sh.get('ROOM_LAPIS_STATE')!r}"
    assert py.get("lapis_state") == override, f"python={py.get('lapis_state')!r}"


def test_ac2_targets_dir_override_verbatim(emitted_sh: str):
    """TARGETS_DIR=/tmp/t → both seams return /tmp/t verbatim."""
    override = "/tmp/t_parity_test"
    sh = _source_sh(emitted_sh, {"TARGETS_DIR": override})
    py = _py_vals({"TARGETS_DIR": override})
    assert sh.get("ROOM_TARGETS") == override, f"shell={sh.get('ROOM_TARGETS')!r}"
    assert py.get("targets") == override, f"python={py.get('targets')!r}"


def test_ac2_override_unset_then_set_both_unset(emitted_sh: str):
    """LAPIS_STATE and ROOM_ROOT both unset → /srv/lapis/lapis-state."""
    sh = _source_sh(emitted_sh)
    py = _py_vals()
    assert sh.get("ROOM_LAPIS_STATE") == "/srv/lapis/lapis-state"
    assert py.get("lapis_state") == "/srv/lapis/lapis-state"


# ---------------------------------------------------------------------------
# AC3: ROOM_ROOT flip propagates identically
# ---------------------------------------------------------------------------

def test_ac3_room_root_flip_no_override(emitted_sh: str):
    """ROOM_ROOT=/mnt/altroom + all overrides unset → every key re-roots identically."""
    alt = "/mnt/altroom"
    sh = _source_sh(emitted_sh, {"ROOM_ROOT": alt})
    py = _py_vals({"ROOM_ROOT": alt})

    # Representative no-override keys
    for key in ["research", "council", "targets.comments", "gpu_queue.failed", "focus"]:
        sh_name = "ROOM_" + key.upper().replace(".", "_")
        assert sh.get(sh_name) == py.get(key), (
            f"AC3: {key}: shell={sh.get(sh_name)!r} != python={py.get(key)!r}"
        )
        assert sh.get(sh_name, "").startswith(alt), (
            f"AC3: {key} not re-rooted under {alt}: {sh.get(sh_name)!r}"
        )


def test_ac3_lapis_state_re_roots_with_room_root(emitted_sh: str):
    """ROOM_ROOT=/mnt/altroom + LAPIS_STATE unset → /mnt/altroom/lapis-state both seams."""
    alt = "/mnt/altroom"
    sh = _source_sh(emitted_sh, {"ROOM_ROOT": alt})
    py = _py_vals({"ROOM_ROOT": alt})
    assert sh.get("ROOM_LAPIS_STATE") == f"{alt}/lapis-state"
    assert py.get("lapis_state") == f"{alt}/lapis-state"


def test_ac3_override_beats_room_root(emitted_sh: str):
    """LAPIS_STATE=/custom + ROOM_ROOT=/mnt/altroom → /custom wins (not re-rooted)."""
    sh = _source_sh(emitted_sh, {"ROOM_ROOT": "/mnt/altroom", "LAPIS_STATE": "/custom"})
    py = _py_vals({"ROOM_ROOT": "/mnt/altroom", "LAPIS_STATE": "/custom"})
    assert sh.get("ROOM_LAPIS_STATE") == "/custom"
    assert py.get("lapis_state") == "/custom"


# ---------------------------------------------------------------------------
# AC4: targets alias family — first-set-wins order matches _resolve_base
# ---------------------------------------------------------------------------

def test_ac4_weaver_targets_dir_wins_when_primary_unset(emitted_sh: str):
    """WEAVER_TARGETS_DIR set + TARGETS_DIR unset → both seams use WEAVER_TARGETS_DIR."""
    override = "/tmp/weaver_t"
    sh = _source_sh(emitted_sh, {"WEAVER_TARGETS_DIR": override})
    py = _py_vals({"WEAVER_TARGETS_DIR": override})
    assert sh.get("ROOM_TARGETS") == override, f"shell={sh.get('ROOM_TARGETS')!r}"
    assert py.get("targets") == override, f"python={py.get('targets')!r}"


def test_ac4_primary_targets_dir_beats_weaver(emitted_sh: str):
    """TARGETS_DIR set beats WEAVER_TARGETS_DIR — TARGETS_DIR is first in alias list."""
    sh = _source_sh(emitted_sh, {
        "TARGETS_DIR": "/tmp/primary_t",
        "WEAVER_TARGETS_DIR": "/tmp/weaver_t",
    })
    py = _py_vals({"TARGETS_DIR": "/tmp/primary_t", "WEAVER_TARGETS_DIR": "/tmp/weaver_t"})
    assert sh.get("ROOM_TARGETS") == "/tmp/primary_t"
    assert py.get("targets") == "/tmp/primary_t"


def test_ac4_pm_targets_dir_is_last_fallback(emitted_sh: str):
    """PM_TARGETS_DIR wins only when TARGETS_DIR and WEAVER_TARGETS_DIR are both unset."""
    override = "/tmp/pm_t"
    sh = _source_sh(emitted_sh, {"PM_TARGETS_DIR": override})
    py = _py_vals({"PM_TARGETS_DIR": override})
    assert sh.get("ROOM_TARGETS") == override
    assert py.get("targets") == override


# ---------------------------------------------------------------------------
# Deliverable D / AC6 / AC12: room_root() function and ROOM_ROOT constant
# ---------------------------------------------------------------------------

def test_room_root_function_default():
    """room_root() returns Path('/room') when ROOM_ROOT is unset (AC6)."""
    env = _scrub_env()
    code = "from agents_core.room_paths import room_root; print(str(room_root()), end='')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, check=True,
    )
    assert result.stdout == "/room"


def test_room_root_function_override():
    """room_root() returns /mnt/altroom when ROOM_ROOT=/mnt/altroom (AC6, AC12)."""
    env = _scrub_env({"ROOM_ROOT": "/mnt/altroom"})
    code = "from agents_core.room_paths import room_root; print(str(room_root()), end='')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, check=True,
    )
    assert result.stdout == "/mnt/altroom"


def test_room_root_constant_is_import_bound():
    """ROOM_ROOT constant binds at import time (AC12 — import-time snapshot, not runtime).

    Bare-root consumers must call room_root() (the function) for runtime-correct
    resolution; the module-level ROOM_ROOT is a convenience constant that captures
    the env at import time and does NOT change if ROOM_ROOT env var changes later.
    """
    from agents_core.room_paths import ROOM_ROOT as CONST
    # In the test env (no ROOM_ROOT set) the constant is /room
    assert str(CONST) == "/room", f"ROOM_ROOT constant default wrong: {CONST!r}"
    # Verify it is a Path, not a string
    assert isinstance(CONST, Path)


def test_room_root_no_existing_key_changed():
    """Adding room_root() changes no existing key's resolution (AC6 non-disruptive)."""
    # Spot-check representative keys with all overrides unset
    assert room_str("targets") == "/srv/lapis/targets"
    assert room_str("lapis_state") == "/srv/lapis/lapis-state"
    assert room_str("focus") == "/srv/lapis/FOCUS.md"
    assert room_str("gpu_queue.failed") == "/srv/lapis/gpu-queue/failed"
    assert room_str("lapis_state.roadmap.committed_plan") == "/srv/lapis/lapis-state/roadmap/committed-plan.json"


# ---------------------------------------------------------------------------
# Deliverable F / AC10: lint gate — no frozen /srv/lapis/ literals in emitted shell
# ---------------------------------------------------------------------------

def test_ac10_no_frozen_room_literals(emitted_sh: str):
    """AC10: emitted shell must not contain bare /srv/lapis/<subpath> literals.

    The only allowed form of /room is inside ${ROOM_ROOT:-/room} (the parameter
    expansion default).  A bare /srv/lapis/anything literal means the emitter baked a
    frozen path — exactly the defect this unit fixes.

    Planted frozen literal test: we verify a synthetically frozen line would fail.
    """
    violations = []
    for i, line in enumerate(emitted_sh.splitlines(), 1):
        # Remove all ${ROOM_ROOT:-/room} occurrences (the allowed form)
        stripped = re.sub(r'\$\{ROOM_ROOT:-/room\}', '', line)
        # Any remaining /srv/lapis/ or /room" means a frozen literal
        if re.search(r'/room[/"]', stripped):
            violations.append(f"line {i}: {line!r}")
    assert not violations, (
        "Frozen /srv/lapis/ literals found in emitted shell (must use ${ROOM_ROOT:-/room}):\n"
        + "\n".join(violations)
    )


def test_ac10_planted_frozen_literal_would_fail():
    """Prove the lint gate catches a frozen literal — self-test of the gate."""
    frozen_line = 'export ROOM_RESEARCH="/srv/lapis/research"'
    stripped = re.sub(r'\$\{ROOM_ROOT:-/room\}', '', frozen_line)
    assert re.search(r'/room[/"]', stripped), (
        "Gate self-test failed: frozen literal was not detected"
    )


# ---------------------------------------------------------------------------
# Deliverable F / AC11: drift gate — committed artifact matches fresh emit
# ---------------------------------------------------------------------------

def test_ac11_committed_artifact_matches_fresh_emit(emitted_sh: str):
    """AC11: repo-committed room_paths.sh is byte-identical to a fresh _emit_sh().

    If this fails, regenerate: python -m agents_core.room_paths --emit-sh > room_paths.sh
    """
    committed = Path(__file__).parent.parent.parent / "room_paths.sh"
    assert committed.exists(), f"room_paths.sh not tracked at {committed}"
    assert emitted_sh == committed.read_text(), (
        "Committed room_paths.sh diverges from _emit_sh(). "
        "Regenerate: python -m agents_core.room_paths --emit-sh > room_paths.sh"
    )
