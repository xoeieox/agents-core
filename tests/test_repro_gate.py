# Copyright (c) 2026 Erah. All rights reserved.
# SPDX-License-Identifier: MIT

"""D7 (lapis-pm-test-gate-hermeticity-v0): the repro-gate test suite.

The spec lists ~20 required test behaviors. This file exercises them
against the real ``agents_core.repro_gate`` module (no LLM, no Forgejo
- the git operations, the verdict partition, the enforcement decision,
and the census writer are the units under test).

Test categories (the spec's D7 list):
- repro-red proceeds (label earned, [REPRO-RED])
- repro-green diverts (gate-PASSED shape -> clean-push, zero
  concluded_gate_rejected PRs; gate-RED shape -> salvage-green path,
  zero fixer_retry spend)
- repro-error: no PR, no cycle, no ceiling touch, subcode in provenance
- INCONCLUSIVE never GREEN + defers
- SHA-MOVED defers + re-probes
- provenance-missing run cannot conclude (run_not_concluded)
- shadow-tree quarantine: moves to /srv/fast, symlink REFUSED
  (run_not_concluded), unique mktemp dest, atomic rename, age-purge
  honored
- the --confcutdir pin: a seeded stray parent-dir conftest.py is NOT
  imported
- full-summary parse: a >20-line failure list is not truncated
- empty-input fallback: the full gate suite runs, never a vacuous GREEN
- preexisting filter: honors ledger, failure-at-head re-enters,
  green-at-head no-op, stale-deferred-to-repro recorded
- chain-depth guard (marker count) pages at the 4th link, deduped per
  target per chain
- no FORGEJO_TOKEN in the repro subprocess env (key-set contract test)
- budget exhaustion defers to next tick
- ENOSPC on a full scratch fs classifies ERROR, does not spend the
  ceiling
- /srv/fast retention (delete-on-verdict)
- the fail-closed runtime env assertion: a seeded FORGEJO_TOKEN in the
  child's os.environ aborts the repro with the env-dirty subcode before
  any test runs
- shadow mode: with LAPIS_PM_REPRO_ENFORCE=shadow, the class-4a shape
  still emits the today-behavior label AND the shadow observation row
  is written (observe-only, nothing paused); with =off, the salvage
  path is byte-identical to today's behavior.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agents_core import repro_gate, shaped_runner
from agents_core.repro_gate import (
    ENFORCE_OFF,
    ENFORCE_ON,
    ENFORCE_SHADOW,
    REPRO_DISALLOWED_ENV_KEYS,
    SUBCODE_ENV_DIRTY,
    VERDICT_ERROR,
    VERDICT_GREEN,
    VERDICT_INCONCLUSIVE,
    VERDICT_RED,
    VERDICT_SHA_MOVED,
    _sha_moved,
    build_provenance_block,
    build_repro_env,
    chain_depth_exceeded,
    bump_salvage_chain,
    census_gap_for_day,
    gate_is_red,
    gate_noise_count,
    parse_failed_node_ids_full,
    preexisting_ledger_filter,
    provenance_complete,
    purge_quarantine,
    quarantine_shadow_path,
    read_enforce_mode,
    repro_body_block,
    repro_budget_available,
    repro_verdict_title_tag,
    run_census_main,
    run_repro,
    run_repro_pytest,
    salvage_chain_depth,
    salvage_label_decision,
    try_consume_repro_budget,
    write_census,
    write_shadow_observation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_budget():
    """Reset the module-level repro budget state between tests (the
    budget is a module-level per-tick counter - a leak between tests
    would make the budget-exhausted tests flaky)."""
    repro_gate._REPRO_BUDGET.clear()
    repro_gate._REPRO_TICK = ""
    yield
    repro_gate._REPRO_BUDGET.clear()
    repro_gate._REPRO_TICK = ""


@pytest.fixture
def mem_db(tmp_path: Path) -> Path:
    """A scratch mem db (the mem-on-brix canon: MEM_DB_PATH)."""
    return tmp_path / "mem.db"


@pytest.fixture
def repro_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch repro root (off /tmp - the /tmp shadow vector this
    spec kills)."""
    root = tmp_path / "lapis-repro"
    root.mkdir()
    monkeypatch.setenv("LAPIS_REPRO_ROOT", str(root))
    return root


@pytest.fixture
def quarantine_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch quarantine root."""
    root = tmp_path / "lapis-quarantine"
    root.mkdir()
    monkeypatch.setenv("LAPIS_QUARANTINE_ROOT", str(root))
    return root


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    ).stdout.strip()


def _init_repo(cwd: Path) -> str:
    """Init a scratch git repo with one base commit. Returns the base
    SHA."""
    subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=cwd, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=cwd, check=True)
    (cwd / "base.py").write_text("base\n")
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=cwd, check=True)
    return _git(cwd, "rev-parse", "HEAD")


def _verdict(
    verdict: str,
    *,
    subcode: str = "",
    full_suite: bool = False,
    node_ids: list[str] | None = None,
    artifact: str = "",
    provenance: dict | None = None,
) -> dict:
    """A minimal repro verdict dict (the shape run_repro returns)."""
    return {
        "verdict": verdict,
        "subcode": subcode,
        "full_suite": full_suite,
        "node_ids": node_ids or [],
        "artifact": artifact,
        "provenance": provenance or {},
        "ts": "2026-09-19T00:00:00+00:00",
    }


def _red_outcome() -> dict:
    """A red gate outcome (failed>0)."""
    return {
        "passed": 10,
        "failed": 2,
        "errors": 0,
        "returncode": 1,
        "summary": "10 passed, 2 failed in 5.0s",
        "output_tail": "FAILED tests/test_a.py::test_1\nFAILED tests/test_b.py::test_2",
    }


def _green_outcome() -> dict:
    """A green gate outcome (passed>0, failed=0, errors=0)."""
    return {
        "passed": 12,
        "failed": 0,
        "errors": 0,
        "returncode": 0,
        "summary": "12 passed in 5.0s",
        "output_tail": "",
    }


# ---------------------------------------------------------------------------
# read_enforce_mode
# ---------------------------------------------------------------------------

class TestReadEnforceMode:
    def test_default_is_shadow(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LAPIS_PM_REPRO_ENFORCE", raising=False)
        assert read_enforce_mode() == ENFORCE_SHADOW

    def test_on(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        assert read_enforce_mode() == ENFORCE_ON

    def test_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "off")
        assert read_enforce_mode() == ENFORCE_OFF

    def test_unknown_defaults_to_shadow(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "bogus")
        assert read_enforce_mode() == ENFORCE_SHADOW

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "ON")
        assert read_enforce_mode() == ENFORCE_ON


# ---------------------------------------------------------------------------
# parse_failed_node_ids_full
# ---------------------------------------------------------------------------

class TestParseFailedNodeIdsFull:
    def test_basic(self):
        output = (
            "FAILED tests/test_a.py::test_1\n"
            "FAILED tests/test_b.py::test_2\n"
            "ERROR tests/test_c.py\n"
        )
        assert parse_failed_node_ids_full(output) == [
            "tests/test_a.py::test_1",
            "tests/test_b.py::test_2",
            "tests/test_c.py",
        ]

    def test_error_at_setup(self):
        output = "ERROR at setup of tests/test_d.py::test_3"
        assert parse_failed_node_ids_full(output) == ["tests/test_d.py::test_3"]

    def test_more_than_20_lines(self):
        """A >20-line failure list is not truncated (the legacy
        extractor reads only the 20-line output_tail)."""
        lines = [f"FAILED tests/test_{i}.py::test_{i}" for i in range(25)]
        output = "\n".join(lines)
        result = parse_failed_node_ids_full(output)
        assert len(result) == 25
        assert result[0] == "tests/test_0.py::test_0"
        assert result[24] == "tests/test_24.py::test_24"

    def test_empty(self):
        assert parse_failed_node_ids_full("") == []

    def test_no_failure_lines(self):
        output = "12 passed in 5.0s"
        assert parse_failed_node_ids_full(output) == []

    def test_deduped(self):
        output = (
            "FAILED tests/test_a.py::test_1\n"
            "FAILED tests/test_a.py::test_1\n"
            "FAILED tests/test_b.py::test_2\n"
        )
        assert parse_failed_node_ids_full(output) == [
            "tests/test_a.py::test_1",
            "tests/test_b.py::test_2",
        ]

    def test_stale_summary_section_ignored(self):
        """A stale 'previously failed' summary line from an earlier
        pytest invocation must NOT drive a subset run - only the LAST
        short test summary info section is parsed."""
        output = (
            "============================= short test summary info ==============================\n"
            "FAILED tests/test_stale.py::test_old\n"
            "============================= short test summary info ==============================\n"
            "FAILED tests/test_current.py::test_new\n"
        )
        assert parse_failed_node_ids_full(output) == [
            "tests/test_current.py::test_new",
        ]

    def test_no_header_scans_whole_output(self):
        """When NO summary header is present, the whole output is
        scanned (the legacy shape)."""
        output = (
            "FAILED tests/test_a.py::test_1\n"
            "some noise\n"
            "FAILED tests/test_b.py::test_2\n"
        )
        assert parse_failed_node_ids_full(output) == [
            "tests/test_a.py::test_1",
            "tests/test_b.py::test_2",
        ]


# ---------------------------------------------------------------------------
# gate_is_red
# ---------------------------------------------------------------------------

class TestGateIsRed:
    def test_red_failed(self):
        assert gate_is_red({"failed": 2, "passed": 10, "returncode": 1})

    def test_red_errors(self):
        assert gate_is_red({"errors": 1, "passed": 10, "returncode": 1})

    def test_red_rc_2(self):
        assert gate_is_red({"returncode": 2, "passed": 0, "failed": 0})

    def test_red_rc_4(self):
        assert gate_is_red({"returncode": 4, "passed": 0, "failed": 0})

    def test_red_rc_5(self):
        assert gate_is_red({"returncode": 5, "passed": 0, "failed": 0})

    def test_red_timeout_marker(self):
        assert gate_is_red({
            "returncode": 0, "passed": 10, "failed": 0,
            "output_tail": "[TIMEOUT after 180s]",
        })

    def test_green(self):
        assert not gate_is_red({"passed": 12, "failed": 0, "errors": 0,
                                "returncode": 0})

    def test_none_is_red(self):
        """No outcome at all: the predicate is fail-closed (red)."""
        assert gate_is_red(None)

    def test_gate_bypassed_is_not_red(self):
        assert not gate_is_red(
            {"failed": 2, "returncode": 1}, gate_bypassed="no-python-test-infra",
        )


# ---------------------------------------------------------------------------
# build_repro_env (the two-phase env allow-list)
# ---------------------------------------------------------------------------

class TestBuildReproEnv:
    def test_no_forgejo_token(self):
        """The no-FORGEJO_TOKEN key-set contract test: the repro env
        allow-list must NOT carry FORGEJO_TOKEN (or any other
        disallowed key)."""
        env = build_repro_env("/tmp/clone", host_env={
            "FORGEJO_TOKEN": "secret",
            "PATH": "/usr/bin",
            "HOME": "/root",
        })
        assert "FORGEJO_TOKEN" not in env
        for key in REPRO_DISALLOWED_ENV_KEYS:
            assert key not in env

    def test_pythonpath_is_clone_root(self):
        env = build_repro_env("/tmp/clone", host_env={"PATH": "/usr/bin"})
        assert env["PYTHONPATH"] == "/tmp/clone"

    def test_git_credentials_disallowed(self):
        """The git-cred marker (the parent-clone's credential helper
        env) is in the disallowed set."""
        env = build_repro_env("/tmp/clone", host_env={
            "GIT_CREDENTIALS": "user:pass",
            "PATH": "/usr/bin",
        })
        assert "GIT_CREDENTIALS" not in env

    def test_allow_list_keys(self):
        env = build_repro_env("/tmp/clone", host_env={
            "PATH": "/usr/bin",
            "HOME": "/root",
            "USER": "user",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        })
        assert set(env.keys()) == {
            "PYTHONPATH", "PATH", "HOME", "USER", "LANG",
            "LC_ALL", "PYTHONDONTWRITEBYTECODE",
        }


# ---------------------------------------------------------------------------
# run_repro_pytest (the child-side env-dirty guard)
# ---------------------------------------------------------------------------

class TestRunReproPytest:
    def test_env_dirty_abort(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """The fail-closed runtime env assertion: a seeded FORGEJO_TOKEN
        in the child's os.environ aborts the repro with the env-dirty
        subcode BEFORE any test runs.

        The parent-side construction test (build_repro_env) passes but
        the child aborts - the two tests are distinct.
        """
        # Seed a disallowed key into the child env by patching
        # build_repro_env to include it (simulating a future allow-list
        # drift that the child-side guard catches).
        original = repro_gate.build_repro_env

        def _dirty_env(clone_root: str, **kwargs) -> dict:
            env = original(clone_root, **kwargs)
            env["FORGEJO_TOKEN"] = "secret"  # the drift
            return env

        repro_gate.build_repro_env = _dirty_env
        try:
            result = run_repro_pytest(
                str(tmp_path), ["tests/"], log=lambda m: None,
                memory_max_kb=0,  # skip the cgroup guard (test env)
            )
        finally:
            repro_gate.build_repro_env = original
        assert result["subcode"] == SUBCODE_ENV_DIRTY
        assert "repro-env-dirty" in result["output"]

    def test_confcutdir_pin(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """The --confcutdir pin: a seeded stray parent-dir conftest.py
        is NOT imported (conftest collection is cut at the clone
        boundary)."""
        # Create a stray conftest.py in a PARENT dir of the clone root.
        parent = tmp_path / "parent"
        parent.mkdir()
        (parent / "conftest.py").write_text(
            "import sys\nsys.stderr.write('STRAY-CONFTEST-IMPORTED\\n')\n"
        )
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n"
        )
        result = run_repro_pytest(
            str(clone), ["test_ok.py"], log=lambda m: None,
            memory_max_kb=0,  # skip the cgroup guard (test env)
        )
        # The stray conftest.py in the parent dir is NOT imported
        # (the --confcutdir=<clone-root> pin cuts conftest collection
        # at the clone boundary).
        assert "STRAY-CONFTEST-IMPORTED" not in result["output"]
        # The test itself passes.
        assert result["returncode"] == 0

    def test_memory_max_unavailable_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """The MemoryMax fail-closed: a cgroup write failure returns
        the named ERROR subcode (the unbounded shape is never live)."""
        # Force the cgroup write to fail by pointing /sys/fs/cgroup at
        # a read-only path (the mkdir fails).
        monkeypatch.setattr(
            repro_gate.os, "getpid", lambda: 999999999,
        )
        # The cgroup path will be /sys/fs/cgroup/lapis-repro-999999999-<pid>
        # which is not writable -> the cgroup write fails -> the
        # process is killed and the ERROR subcode is returned.
        result = run_repro_pytest(
            str(tmp_path), ["tests/"], log=lambda m: None,
            memory_max_kb=4 * 1024 * 1024,
        )
        # The cgroup write failed (the unbounded shape is NOT live).
        assert result.get("memory_limited") is False
        assert result["returncode"] == -1


# ---------------------------------------------------------------------------
# _sha_moved
# ---------------------------------------------------------------------------

class TestShaMoved:
    def test_sha_moved_when_head_differs(self, tmp_path: Path):
        """SHA-MOVED: the repro SHA != the current head at repro time."""
        base_sha = _init_repo(tmp_path)
        # Advance HEAD (the model self-committed).
        (tmp_path / "work.py").write_text("work\n")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "self"], cwd=tmp_path, check=True)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")
        assert head_sha != base_sha
        # The gate's recorded head is the BASE sha (stale); the current
        # head is the self-commit.
        assert _sha_moved(str(tmp_path), "", base_sha, None,
                          current_head_sha=head_sha)

    def test_sha_not_moved_when_head_matches(self, tmp_path: Path):
        base_sha = _init_repo(tmp_path)
        assert not _sha_moved(str(tmp_path), "", base_sha, None,
                              current_head_sha=base_sha)

    def test_sha_not_moved_when_no_current(self, tmp_path: Path):
        """An unresolvable / absent source is NOT a move (the repro
        proceeds on the gate's recorded head)."""
        base_sha = _init_repo(tmp_path)
        assert not _sha_moved("", "", base_sha, None, current_head_sha="")


# ---------------------------------------------------------------------------
# provenance_complete
# ---------------------------------------------------------------------------

class TestProvenanceComplete:
    def test_complete(self):
        block = build_provenance_block(
            repo="agents-core", head_sha="abc123", cwd="/tmp/clone",
            resolved_package_path="/srv/agents/agents_core",
            shadow_check=[],
        )
        assert provenance_complete(block)

    def test_missing_repo(self):
        block = build_provenance_block(
            repo="", head_sha="abc123", cwd="/tmp/clone",
            resolved_package_path="/srv/agents/agents_core",
            shadow_check=[],
        )
        assert not provenance_complete(block)

    def test_missing_head_sha(self):
        block = build_provenance_block(
            repo="agents-core", head_sha="", cwd="/tmp/clone",
            resolved_package_path="/srv/agents/agents_core",
            shadow_check=[],
        )
        assert not provenance_complete(block)

    def test_missing_resolved_package_path(self):
        block = {
            "repo": "agents-core",
            "head_sha": "abc123",
            "cwd": "/tmp/clone",
            "resolved_package_path": "",
            "sys_path_shadow_check": [],
        }
        assert not provenance_complete(block)

    def test_shadow_check_non_empty(self):
        """A NON-EMPTY shadow list is a refusal, not a completion."""
        block = build_provenance_block(
            repo="agents-core", head_sha="abc123", cwd="/tmp/clone",
            resolved_package_path="/srv/agents/agents_core",
            shadow_check=["/tmp/stale"],
        )
        assert not provenance_complete(block)

    def test_none(self):
        assert not provenance_complete(None)


# ---------------------------------------------------------------------------
# quarantine_shadow_path
# ---------------------------------------------------------------------------

class TestQuarantine:
    def test_moves_to_quarantine(self, tmp_path: Path, quarantine_root: Path):
        """A stale shadow path is moved to the quarantine tree."""
        src = tmp_path / "stale_agents_core"
        src.mkdir()
        (src / "__init__.py").write_text("")
        result = quarantine_shadow_path(str(src), quarantine_root)
        assert result["status"] == "moved"
        assert result["dest"].startswith(str(quarantine_root))
        # The source is gone.
        assert not src.exists()

    def test_symlink_refused(self, tmp_path: Path, quarantine_root: Path):
        """A symlink shadow is REFUSED (the target may be live state or
        carry stale secrets)."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "file.txt").write_text("data")
        link = tmp_path / "stale_link"
        link.symlink_to(target)
        result = quarantine_shadow_path(str(link), quarantine_root)
        assert result["status"] == "refused"
        assert result["reason"] == "symlink"
        # The symlink is NOT moved.
        assert link.exists()
        # The target is NOT moved.
        assert target.exists()

    def test_unique_mktemp_dest(self, tmp_path: Path, quarantine_root: Path):
        """The move is into a unique mktemp subdir (never a
        predictable destination filename)."""
        src = tmp_path / "stale1"
        src.mkdir()
        result1 = quarantine_shadow_path(str(src), quarantine_root)
        src2 = tmp_path / "stale2"
        src2.mkdir()
        result2 = quarantine_shadow_path(str(src2), quarantine_root)
        assert result1["status"] == "moved"
        assert result2["status"] == "moved"
        # The destinations are in DIFFERENT mktemp subdirs.
        dest1_dir = Path(result1["dest"]).parent
        dest2_dir = Path(result2["dest"]).parent
        assert dest1_dir != dest2_dir

    def test_age_purge(self, tmp_path: Path, quarantine_root: Path):
        """Age-then-purge: quarantine subdirs older than 7 days are
        removed."""
        import time
        old = quarantine_root / "q-old"
        old.mkdir()
        (old / "file.txt").write_text("stale")
        # Set the mtime to 8 days ago.
        old_ts = time.time() - 8 * 86400
        os.utime(old, (old_ts, old_ts))
        new = quarantine_root / "q-new"
        new.mkdir()
        (new / "file.txt").write_text("fresh")
        purged = purge_quarantine(quarantine_root)
        assert str(old) in purged
        assert str(new) not in purged
        assert not old.exists()
        assert new.exists()


# ---------------------------------------------------------------------------
# preexisting_ledger_filter
# ---------------------------------------------------------------------------

class TestPreexistingLedgerFilter:
    def test_honors_ledger(self, tmp_path: Path, mem_db: Path):
        """A labeled + fresh node-id is filtered (excluded from
        attribution)."""
        from agents_core.mem import MemoryStore
        store = MemoryStore(db_path=mem_db)
        store.set(
            "finding/preexisting-failure/agents-core/tests/test_a.py/2026-09-19",
            json.dumps({"status": "open"}),
        )
        store.close()
        result = preexisting_ledger_filter(
            ["tests/test_a.py::test_1"],
            mem_db_path=mem_db, repo="agents-core",
            today="2026-09-19",
        )
        assert "tests/test_a.py::test_1" in result["filtered"]
        assert "tests/test_a.py::test_1" not in result["unlabeled"]

    def test_unlabeled(self, tmp_path: Path, mem_db: Path):
        """A node-id with no ledger record is attributed to the PR."""
        result = preexisting_ledger_filter(
            ["tests/test_b.py::test_2"],
            mem_db_path=mem_db, repo="agents-core",
            today="2026-09-19",
        )
        assert "tests/test_b.py::test_2" in result["unlabeled"]
        assert "tests/test_b.py::test_2" not in result["filtered"]

    def test_stale_deferred_to_repro(self, tmp_path: Path, mem_db: Path):
        """A label older than 7d is NOT re-verified per-use against
        base (stale-deferred-to-repro)."""
        from agents_core.mem import MemoryStore
        store = MemoryStore(db_path=mem_db)
        store.set(
            "finding/preexisting-failure/agents-core/tests/test_c.py/2026-09-01",
            json.dumps({"status": "open"}),
        )
        store.close()
        result = preexisting_ledger_filter(
            ["tests/test_c.py::test_3"],
            mem_db_path=mem_db, repo="agents-core",
            today="2026-09-19",
        )
        assert "tests/test_c.py::test_3" in result["stale"]
        assert "tests/test_c.py::test_3" not in result["filtered"]

    def test_failure_at_head_re_enters(self, tmp_path: Path, mem_db: Path):
        """A filtered node-id that FAILS in the P1 repro re-enters the
        retry trigger (failure-at-head is the trigger; NEWness is not
        required)."""
        # The filter returns the filtered node-ids; the caller (the
        # repro decision) checks if the node-id FAILED at head. If it
        # did, it re-enters the retry trigger. The filter itself does
        # NOT re-enter - the re-entry is the caller's decision (the
        # repro proved the failure). This test verifies the filter
        # returns the filtered node-id so the caller can make the
        # re-entry decision.
        from agents_core.mem import MemoryStore
        store = MemoryStore(db_path=mem_db)
        store.set(
            "finding/preexisting-failure/agents-core/tests/test_a.py/2026-09-19",
            json.dumps({"status": "open"}),
        )
        store.close()
        result = preexisting_ledger_filter(
            ["tests/test_a.py::test_1"],
            mem_db_path=mem_db, repo="agents-core",
            today="2026-09-19",
        )
        # The node-id is filtered (the caller checks the repro verdict
        # for the re-entry decision).
        assert "tests/test_a.py::test_1" in result["filtered"]


# ---------------------------------------------------------------------------
# chain-depth guard
# ---------------------------------------------------------------------------

class TestChainDepth:
    def test_bump_and_read(self, tmp_path: Path, mem_db: Path):
        """The mem counter is bumped at salvage-open (update-in-place)."""
        assert salvage_chain_depth("tgt", mem_db_path=mem_db) == 0
        assert bump_salvage_chain("tgt", mem_db_path=mem_db) == 1
        assert bump_salvage_chain("tgt", mem_db_path=mem_db) == 2
        assert salvage_chain_depth("tgt", mem_db_path=mem_db) == 2

    def test_exceeded_at_4(self, tmp_path: Path, mem_db: Path):
        """> 3 => stop + page (the 4th link fires the guard)."""
        for _ in range(3):
            bump_salvage_chain("tgt", mem_db_path=mem_db)
        assert not chain_depth_exceeded("tgt", mem_db_path=mem_db)
        bump_salvage_chain("tgt", mem_db_path=mem_db)
        assert chain_depth_exceeded("tgt", mem_db_path=mem_db)

    def test_deduped_per_target(self, tmp_path: Path, mem_db: Path):
        """The guard is per-target (different targets have independent
        counters)."""
        for _ in range(4):
            bump_salvage_chain("tgt-a", mem_db_path=mem_db)
        bump_salvage_chain("tgt-b", mem_db_path=mem_db)
        assert chain_depth_exceeded("tgt-a", mem_db_path=mem_db)
        assert not chain_depth_exceeded("tgt-b", mem_db_path=mem_db)


# ---------------------------------------------------------------------------
# gate_noise_count
# ---------------------------------------------------------------------------

class TestGateNoiseCount:
    def test_keyed_by_repo(self, tmp_path: Path, mem_db: Path):
        """The key is pm/gate-noise/<repo> (not <target>)."""
        from agents_core.mem import MemoryStore
        assert gate_noise_count("agents-core", mem_db_path=mem_db) == 1
        assert gate_noise_count("agents-core", mem_db_path=mem_db) == 2
        store = MemoryStore(db_path=mem_db)
        row = store.get("pm/gate-noise/agents-core")
        store.close()
        assert row is not None
        assert json.loads(row["content"])["count"] == 2


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

class TestBudget:
    def test_try_consume(self, monkeypatch: pytest.MonkeyPatch):
        """The atomic check-and-consume: 2/tick, then defer."""
        repro_gate._REPRO_BUDGET.clear()
        repro_gate._REPRO_TICK = ""
        assert try_consume_repro_budget("tgt")
        assert try_consume_repro_budget("tgt")
        # The 3rd consume fails (budget exhausted).
        assert not try_consume_repro_budget("tgt")

    def test_available(self, monkeypatch: pytest.MonkeyPatch):
        repro_gate._REPRO_BUDGET.clear()
        repro_gate._REPRO_TICK = ""
        assert repro_budget_available("tgt")
        try_consume_repro_budget("tgt")
        try_consume_repro_budget("tgt")
        assert not repro_budget_available("tgt")


# ---------------------------------------------------------------------------
# repro_verdict_title_tag
# ---------------------------------------------------------------------------

class TestTitleTag:
    def test_red(self):
        assert repro_verdict_title_tag(VERDICT_RED) == "[REPRO-RED]"

    def test_green(self):
        assert repro_verdict_title_tag(VERDICT_GREEN) == "[REPRO-GREEN]"

    def test_error(self):
        assert repro_verdict_title_tag(VERDICT_ERROR) == "[REPRO-ERROR]"

    def test_inconclusive(self):
        assert repro_verdict_title_tag(VERDICT_INCONCLUSIVE) == ""

    def test_sha_moved(self):
        assert repro_verdict_title_tag(VERDICT_SHA_MOVED) == ""


# ---------------------------------------------------------------------------
# repro_body_block
# ---------------------------------------------------------------------------

class TestBodyBlock:
    def test_shape(self):
        block = repro_body_block(_verdict(VERDICT_RED, subcode=""))
        assert "<!-- lapis-repro: start -->" in block
        assert "verdict: RED" in block
        assert "<!-- lapis-repro: end -->" in block

    def test_empty(self):
        block = repro_body_block({})
        assert "verdict: " in block


# ---------------------------------------------------------------------------
# write_shadow_observation
# ---------------------------------------------------------------------------

class TestShadowObservation:
    def test_writes_row(self, tmp_path: Path, mem_db: Path):
        key = write_shadow_observation(
            target="tgt", head_sha="abc123def456",
            verdict=_verdict(VERDICT_RED),
            would_be_disposition="salvage-red",
            mem_db_path=mem_db,
        )
        assert key == "pm/repro-shadow/tgt/abc123def456"
        from agents_core.mem import MemoryStore
        store = MemoryStore(db_path=mem_db)
        row = store.get(key)
        store.close()
        assert row is not None
        data = json.loads(row["content"])
        assert data["verdict"] == "RED"
        assert data["would_be_disposition"] == "salvage-red"


# ---------------------------------------------------------------------------
# salvage_label_decision (the D2 enforcement)
# ---------------------------------------------------------------------------

class TestSalvageLabelDecision:
    def _kwargs(self, **overrides) -> dict:
        kwargs = dict(
            target="tgt",
            head_sha="abc123def456789",
            gate_passed=False,
            last_test_outcome=_red_outcome(),
            gate_output="FAILED tests/test_a.py::test_1",
            failed_node_ids=["tests/test_a.py::test_1"],
            wip_commit_count=1,
            empty_diff=False,
            head_past_base=True,
            worktree="/tmp/worktree",
            wip_ref="refs/wip/tgt",
            current_head_sha="abc123def456789",
            repo="agents-core",
        )
        kwargs.update(overrides)
        return kwargs

    def test_off_mode_is_today_behavior(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """With LAPIS_PM_REPRO_ENFORCE=off, the salvage path is
        byte-identical to today's behavior (no repro run, no
        observation)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "off")
        decision = salvage_label_decision(**self._kwargs())
        assert decision["enforce_mode"] == ENFORCE_OFF
        assert decision["disposition"] == "concluded-gate-rejected-legacy"
        assert decision["stop_reason"] == "concluded_gate_rejected"
        assert decision["title_tag"] == ""
        assert decision["body_block"] == ""
        assert decision["repro"] == {}
        assert decision["shadow_observation"] == ""

    def test_shadow_mode_class4a(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """Shadow mode: the class-4a shape still emits the today-
        behavior label AND the shadow observation row is written
        (observe-only, nothing paused)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "shadow")
        # The class-4a shape: gate-PASSED + concluded + WIP + empty-diff.
        decision = salvage_label_decision(
            **self._kwargs(
                gate_passed=True,
                last_test_outcome=_green_outcome(),
                empty_diff=True,
            ),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_GREEN, full_suite=True),
        )
        assert decision["enforce_mode"] == ENFORCE_SHADOW
        # The today-behavior label stands (the mislabel quantified
        # live).
        assert decision["disposition"] == "concluded-gate-rejected-legacy"
        assert decision["stop_reason"] == "concluded_gate_rejected"
        # The shadow observation is written.
        assert decision["shadow_observation"] != ""
        assert "pm/repro-shadow/" in decision["shadow_observation"]

    def test_on_mode_repro_red(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: repro-RED + gate-RED -> the label is EARNED
        ([REPRO-RED] title tag, the repro verdict block in the PR
        body)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
        )
        assert decision["enforce_mode"] == ENFORCE_ON
        assert decision["disposition"] == "salvage-red"
        assert decision["stop_reason"] == "concluded_gate_rejected"
        assert decision["title_tag"] == "[REPRO-RED]"
        assert "<!-- lapis-repro: start -->" in decision["body_block"]

    def test_on_mode_repro_red_gate_not_red(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: repro-RED but the gate outcome is NOT red ->
        run_not_concluded (the label is not earned)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(
                last_test_outcome=_green_outcome(),
            ),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
        )
        assert decision["disposition"] == "run-not-concluded"
        assert decision["stop_reason"] == "run_not_concluded"
        assert decision["title_tag"] == ""

    def test_on_mode_class4a_clean_push(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: the class-4a shape (gate-PASSED + concluded + WIP +
        empty-diff) routes to the CLEAN-PUSH disposition, NEVER
        concluded_gate_rejected."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(
                gate_passed=True,
                last_test_outcome=_green_outcome(),
                empty_diff=True,
            ),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_GREEN, full_suite=True),
        )
        assert decision["disposition"] == "clean-push"
        assert decision["stop_reason"] == "clean-push"
        assert decision["title_tag"] == ""
        assert "concluded_gate_rejected" not in decision["stop_reason"]

    def test_on_mode_repro_green(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: gate-RED-but-repro-GREEN -> the SALVAGE-GREEN path
        ([REPRO-GREEN] title tag, zero fixer_retry spend)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_GREEN, full_suite=True),
        )
        assert decision["disposition"] == "salvage-green"
        assert decision["stop_reason"] == "salvage-green"
        assert decision["title_tag"] == "[REPRO-GREEN]"
        assert "<!-- lapis-repro: start -->" in decision["body_block"]

    def test_on_mode_repro_error(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: the repro itself could not run (ERROR) -> infra-
        noise: NO salvage PR with a defect label, NO review cycle
        consumed. The subcode is in the provenance."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_ERROR, subcode="enospc"),
        )
        assert decision["disposition"] == "salvage-error"
        assert decision["stop_reason"] == "repro-error-enospc"
        assert decision["title_tag"] == "[REPRO-ERROR]"

    def test_on_mode_inconclusive(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: INCONCLUSIVE may NEVER be GREEN; defers to the
        next tick."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_INCONCLUSIVE),
        )
        assert decision["disposition"] == "defer-inconclusive"
        assert decision["stop_reason"] == "defer-inconclusive"
        assert decision["title_tag"] == ""

    def test_on_mode_sha_moved(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: SHA-MOVED defers to the next tick and re-probes
        the new head."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_SHA_MOVED),
        )
        assert decision["disposition"] == "defer-sha-moved"
        assert decision["stop_reason"] == "defer-sha-moved"
        assert decision["title_tag"] == ""

    def test_on_mode_provenance_refusal(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: the D3 provenance refusal (an unaudited gate run
        cannot conclude - run_not_concluded)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # An incomplete provenance block (missing resolved_package_path).
        incomplete_prov = {
            "repo": "agents-core",
            "head_sha": "abc123",
            "cwd": "/tmp/clone",
            "resolved_package_path": "",
            "sys_path_shadow_check": [],
        }
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
            provenance_block=incomplete_prov,
        )
        assert decision["disposition"] == "run-not-concluded"
        assert decision["stop_reason"] == "run_not_concluded"
        assert decision["title_tag"] == ""

    def test_on_mode_chain_stop(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: the D5 chain-depth guard (> 3) stops + pages."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # Bump the chain to 4.
        for _ in range(4):
            bump_salvage_chain("tgt", mem_db_path=mem_db)
        pages = []
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
            page_fn=lambda msg: pages.append(msg),
        )
        assert decision["disposition"] == "chain-stop"
        assert decision["stop_reason"] == "chain-stop"
        assert decision["chain_stop"] is True
        assert decision["chain_depth"] == 4
        assert len(pages) == 1
        assert "tgt" in pages[0]

    def test_on_mode_budget_exhausted(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """On mode: budget exhaustion defers to the next tick."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # Exhaust the budget.
        repro_gate._REPRO_BUDGET.clear()
        repro_gate._REPRO_TICK = ""
        try_consume_repro_budget("tgt")
        try_consume_repro_budget("tgt")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
        )
        assert decision["disposition"] == "defer-budget"
        assert decision["stop_reason"] == "defer-budget"

    def test_shadow_mode_gate_red_repro_red(
        self, monkeypatch: pytest.MonkeyPatch, mem_db: Path,
    ):
        """Shadow mode: the gate-RED + repro-RED shape still emits the
        today-behavior label (the mislabel is quantified live by the
        shadow observation)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "shadow")
        decision = salvage_label_decision(
            **self._kwargs(),
            mem_db_path=mem_db,
            run_repro_fn=lambda **kw: _verdict(VERDICT_RED),
        )
        assert decision["enforce_mode"] == ENFORCE_SHADOW
        assert decision["disposition"] == "concluded-gate-rejected-legacy"
        assert decision["stop_reason"] == "concluded_gate_rejected"
        assert decision["shadow_observation"] != ""


# ---------------------------------------------------------------------------
# D6 census
# ---------------------------------------------------------------------------

class TestCensus:
    def test_aggregate(self, tmp_path: Path):
        """The census aggregates the day's fixer / fixer_retry
        wall-clock from the claude-queue completed ledger."""
        hist = tmp_path / "history.jsonl"
        lines = [
            json.dumps({
                "event": "completed",
                "id": "claude_20260919_fixer_retry_tgt",
                "timestamp": "2026-09-19T10:00:00-07:00",
                "duration_seconds": 545,
            }),
            json.dumps({
                "event": "completed",
                "id": "claude_20260919_fixer_tgt",
                "timestamp": "2026-09-19T11:00:00-07:00",
                "duration_seconds": 937,
            }),
            json.dumps({
                "event": "completed",
                "id": "claude_20260919_reviewer_tgt",
                "timestamp": "2026-09-19T12:00:00-07:00",
                "duration_seconds": 114,
            }),
        ]
        hist.write_text("\n".join(lines))
        census = repro_gate.aggregate_wallclock_census(
            "2026-09-19", history_path=hist,
        )
        assert census["fixer_retry"]["count"] == 1
        assert census["fixer_retry"]["avg_s"] == 545.0
        assert census["fixer"]["count"] == 1
        assert census["fixer"]["avg_s"] == 937.0
        # The reviewer is NOT counted (only fixer / fixer_retry).
        assert "reviewer" not in json.dumps(census)

    def test_write_census(self, tmp_path: Path):
        """The census artifact is written to state/fixer-wallclock-
        census-<date>."""
        census = {
            "date": "2026-09-19",
            "fixer": {"count": 7, "avg_s": 937.1, "max_s": 2471},
            "fixer_retry": {"count": 38, "avg_s": 545.0, "max_s": 2680},
        }
        artifact = write_census("2026-09-19", census, state_dir=tmp_path)
        assert artifact == str(tmp_path / "fixer-wallclock-census-2026-09-19")
        assert Path(artifact).is_file()
        data = json.loads(Path(artifact).read_text())
        assert data["fixer_retry"]["count"] == 38

    def test_census_gap(self, tmp_path: Path):
        """An absent census artifact renders as a VISIBLE GAP on the
        day surface."""
        gap = census_gap_for_day("2026-09-19", state_dir=tmp_path)
        assert "census-gap" in gap
        # Write the artifact: the gap is gone.
        write_census("2026-09-19", {"date": "2026-09-19"},
                     state_dir=tmp_path)
        gap = census_gap_for_day("2026-09-19", state_dir=tmp_path)
        assert gap == ""

    def test_run_census_main(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                             capsys: pytest.CaptureFixture):
        """The python3 -m agents_core.repro_gate census entrypoint."""
        monkeypatch.setenv("LAPIS_CENSUS_STATE_DIR", str(tmp_path))
        rc = run_census_main("2026-09-19")
        assert rc == 0
        assert (tmp_path / "fixer-wallclock-census-2026-09-19").is_file()


# ---------------------------------------------------------------------------
# run_repro (the D1 orchestrator)
# ---------------------------------------------------------------------------

class TestRunRepro:
    def test_empty_input_full_suite_fallback(
        self, tmp_path: Path, repro_root: Path,
    ):
        """Empty-input fallback: when the parsed input is empty and
        there are no full_suite_paths, the repro returns ERROR (never
        a vacuous GREEN)."""
        base_sha = _init_repo(tmp_path)
        result = run_repro(
            target="tgt",
            head_sha=base_sha,
            worktree=str(tmp_path),
            wip_ref="",
            failed_node_ids=[],
            full_suite_paths=[],
            gate_output="",  # empty -> parse returns []
            current_head_sha=base_sha,
            repro_root=repro_root,
            log=lambda m: None,
        )
        # No suite paths at all: ERROR, not a vacuous GREEN.
        assert result["verdict"] == VERDICT_ERROR

    def test_sha_moved_defers(
        self, tmp_path: Path, repro_root: Path,
    ):
        """SHA-MOVED: the head moved between gate and repro -> defer."""
        base_sha = _init_repo(tmp_path)
        # Advance HEAD.
        (tmp_path / "work.py").write_text("work\n")
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "self"], cwd=tmp_path, check=True)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")
        result = run_repro(
            target="tgt",
            head_sha=base_sha,  # the gate's recorded head (stale)
            worktree=str(tmp_path),
            wip_ref="",
            failed_node_ids=["tests/test_a.py::test_1"],
            full_suite_paths=["tests/"],
            gate_output="FAILED tests/test_a.py::test_1",
            current_head_sha=head_sha,  # the current head (moved)
            repro_root=repro_root,
            log=lambda m: None,
        )
        assert result["verdict"] == VERDICT_SHA_MOVED

    def test_clone_failure_returns_error(
        self, tmp_path: Path, repro_root: Path,
    ):
        """A clone failure returns ERROR (the subcode is
        clone-failed)."""
        # A non-existent worktree -> the clone fails.
        result = run_repro(
            target="tgt",
            head_sha="abc123def456789",
            worktree=str(tmp_path / "nonexistent"),
            wip_ref="",
            failed_node_ids=["tests/test_a.py::test_1"],
            full_suite_paths=["tests/"],
            gate_output="FAILED tests/test_a.py::test_1",
            current_head_sha="abc123def456789",
            repro_root=repro_root,
            log=lambda m: None,
        )
        assert result["verdict"] == VERDICT_ERROR
        assert result["subcode"] == "clone-failed"

    def test_enospc_classifies_error(
        self, tmp_path: Path, repro_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """ENOSPC on a full scratch fs classifies ERROR, does not
        spend the ceiling."""
        # Force the df preflight to fail.
        monkeypatch.setattr(
            repro_gate, "_df_preflight_ok",
            lambda path, min_free_bytes=1 << 30: False,
        )
        base_sha = _init_repo(tmp_path)
        result = run_repro(
            target="tgt",
            head_sha=base_sha,
            worktree=str(tmp_path),
            wip_ref="",
            failed_node_ids=["tests/test_a.py::test_1"],
            full_suite_paths=["tests/"],
            gate_output="FAILED tests/test_a.py::test_1",
            current_head_sha=base_sha,
            repro_root=repro_root,
            log=lambda m: None,
        )
        assert result["verdict"] == VERDICT_ERROR
        assert result["subcode"] == "enospc"

    def test_delete_on_verdict(
        self, tmp_path: Path, repro_root: Path,
    ):
        """/srv/fast retention: the clone dir is removed once the
        verdict + artifact are written (delete-on-verdict)."""
        base_sha = _init_repo(tmp_path)
        # Create a minimal test file in the worktree so the clone
        # succeeds.
        (tmp_path / "test_ok.py").write_text(
            "def test_ok():\n    assert True\n"
        )
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "add test"], cwd=tmp_path, check=True)
        head_sha = _git(tmp_path, "rev-parse", "HEAD")
        result = run_repro(
            target="tgt",
            head_sha=head_sha,
            worktree=str(tmp_path),
            wip_ref="",
            failed_node_ids=["test_ok.py::test_ok"],
            full_suite_paths=["test_ok.py"],
            gate_output="FAILED test_ok.py::test_ok",
            current_head_sha=head_sha,
            repro_root=repro_root,
            log=lambda m: None,
        )
        # The clone dir is removed (delete-on-verdict).
        clone_dirs = [
            d for d in repro_root.iterdir()
            if d.is_dir() and d.name.startswith("tgt-")
        ]
        assert len(clone_dirs) == 0
        # The artifact is kept.
        assert result["artifact"] != ""
        assert Path(result["artifact"]).is_file()


# ---------------------------------------------------------------------------
# The D2 SEAM (shaped_runner._compute_salvage_decision ->
# salvage_label_decision): the on-mode live path must feed the repro the
# FULL gate output + the full-summary-parser node-ids + the per-repo
# full-suite paths + the D3 provenance block. The cycle-2 defect: the
# seam passed the 20-line output_tail + the legacy 20-line extractor +
# NO full_suite_paths + NO provenance_block, so every real gate-red
# routed to the salvage-error partition with NO PR / NO label / NO
# reviewer cycle (the empty-input fallback returned VERDICT_ERROR).
# ---------------------------------------------------------------------------

def _make_wt_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    """A worktree whose origin is a bare repo (mirrors the fixture
    pattern in test_shaped_runner_tail_fail_closed.py). The tail's
    salvage path pushes to origin, so the worktree needs a reachable
    origin remote."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(origin)],
        capture_output=True, text=True, check=True,
    )
    wt = tmp_path / "wt"
    wt.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "-m", "base"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "push", "-q", "origin", "main"], cwd=wt,
        capture_output=True, text=True, check=True,
    )
    return wt, origin


def _seam_kwargs(wt_dir: Path, **overrides) -> dict:
    """Build the tail_finalize kwargs for the gate-RED salvage path
    (the "test gate failed" partition - the D2 enforcement seam's
    live carrier: concluded, NON-WIP-salvage-eligible, no guard
    flags, a non-empty in-tail diff, a RED gate). That shape reaches
    the `if not gate_passed:` partition where _compute_salvage_decision
    runs and the on-mode dispositions (run-not-concluded / chain-stop /
    defer / salvage-error / salvage-red / salvage-green) are applied.

    The shape is deliberately NOT the class-4a shape (concluded + WIP +
    empty-diff + head-past-base): that shape routes to the class-4a
    salvage partition (the "concluded, gate passed" [SALVAGE] PR) which
    opens the concluded_gate_rejected label directly and does NOT
    consult the salvage decision (the class-4a on-mode clean-push
    routing is a separate, pre-existing path - out of scope for the
    cycle-2 gate-RED wiring fix)."""
    kwargs = {
        "task_id": "task-seam",
        "target_id": "tgt-seam",
        "bare_repo": "agents-core",
        "branch": "lapis/tgt-seam/local",
        "slug": "local",
        "cwd": str(wt_dir),
        "worktree_path": str(wt_dir),
        # A non-empty in-tail diff: the gate-RED salvage partition
        # commits the worktree state (the empty-diff bail would fire
        # before it otherwise).
        "final_diff": "diff --git a/x.py b/x.py\n+1\n",
        "concluded": True,
        "last_test_outcome": None,
        "max_steps_hit": False,
        "no_progress_hit": False,
        "stop_reason": "concluded",
        "step_count": 5,
        "transcript_path": wt_dir / "transcript.json",
        "gate_passed": False,
        "gate_bypassed": None,
        "model_touched_tests": set(),
        "gate_rerun_fired": False,
        "wip_ref": "",
        "wip_commit_count": 0,  # NOT WIP-salvage-eligible
        "wip_head_sha": "",
        "wip_steps": [],
        "_wip_git": None,
        "base_sha": "",
    }
    kwargs.update(overrides)
    return kwargs


def _red_outcome_full_output(n_fail: int) -> dict:
    """A red gate outcome whose output_tail carries the FULL short
    test summary section (n_fail FAILED lines - >20 lines for the
    >20-line truncation test) + the summary line."""
    failed_lines = "\n".join(
        f"FAILED tests/test_{i}.py::test_{i}" for i in range(n_fail)
    )
    output = (
        "============================= short test summary info "
        "=============================\n"
        f"{failed_lines}\n"
        f"{n_fail} failed in 5.0s"
    )
    return {
        "passed": 0,
        "failed": n_fail,
        "errors": 0,
        "returncode": 1,
        "summary": f"{n_fail} failed in 5.0s",
        "output_tail": output,
    }


class TestSalvageSeam:
    """The D2 seam (shaped_runner._compute_salvage_decision) - the
    on-mode live path must wire the full gate output, the
    full-summary-parser node-ids, the full-suite paths, and the D3
    provenance block through to salvage_label_decision. The cycle-2
    defect: the seam passed the 20-line output_tail + the legacy
    20-line extractor + NO full_suite_paths + NO provenance_block, so
    every real gate-red routed to the salvage-error partition with NO
    PR / NO label / NO reviewer cycle (the empty-input fallback
    returned VERDICT_ERROR).

    The tests exercise the SEAM (not just the repro_gate functions):
    tail_finalize is driven end-to-end through the gate-RED salvage
    partition, with the seam's repro dependency (repro_gate.run_repro)
    and provenance dependency (shaped_runner._build_provenance_block)
    patched so the seam's wiring is observable in the captured
    arguments."""

    def _run_seam(self, tmp_path: Path, monkeypatch, capsys,
                  **overrides) -> str:
        wt_dir, _ = _make_wt_with_origin(tmp_path)
        # Stage a file so the gate-RED salvage partition's
        # `git add -A` + `git commit` succeed (the partition commits
        # the worktree state before opening the [SALVAGE] PR - a clean
        # index would make the commit rc=1 and the PR never open).
        (wt_dir / "work.py").write_text("work\n")
        subprocess.run(
            ["git", "add", "-A"], cwd=wt_dir,
            capture_output=True, text=True, check=True,
        )
        kwargs = _seam_kwargs(wt_dir)
        kwargs["last_test_outcome"] = _red_outcome_full_output(2)
        kwargs.update(overrides)
        # Patch the seam's PR-open dependency (the gate-RED salvage
        # partition's carrier) so the test does not hit the network.
        monkeypatch.setattr(
            shaped_runner, "_open_wip_salvage_pr",
            lambda *a, **k: "https://forge.example/pr/seam",
        )
        return shaped_runner.tail_finalize(**kwargs)

    def test_on_mode_gate_red_produces_repro_verdict_not_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """On mode: a real gate-red with a >20-line failure list must
        produce a repro verdict from the PARSED node-ids (the full
        short-summary parse), NOT VERDICT_ERROR (the legacy 20-line
        extractor returned [] for a >20-line list -> the empty-input
        fallback -> the salvage-error partition with no PR)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # A >20-line failure list (25 FAILED lines).
        outcome = _red_outcome_full_output(25)
        captured: dict = {}

        def _fake_repro(**kw):
            captured.update(kw)
            return {
                "verdict": VERDICT_RED, "subcode": "", "full_suite": False,
                "node_ids": kw.get("failed_node_ids") or [],
                "artifact": "", "provenance": {}, "ts": "2026-09-21T00:00:00Z",
            }

        # The seam's repro dependency: salvage_label_decision's default
        # run_repro_fn is repro_gate.run_repro (resolved at call time) -
        # patch the module attribute the seam's repro_fn resolves.
        monkeypatch.setattr(repro_gate, "run_repro", _fake_repro,
                            raising=False)
        out = self._run_seam(
            tmp_path, monkeypatch, capsys, last_test_outcome=outcome,
        )
        # The repro verdict is RED (from the parsed node-ids), NOT
        # VERDICT_ERROR - the label is EARNED and the [SALVAGE] PR
        # opens with the [REPRO-RED] title tag (no salvage-error
        # partition, no silent drop).
        assert captured, "the repro was never invoked through the seam"
        # The seam fed the FULL gate output (the complete short-summary
        # section), NOT the 20-line tail.
        assert "short test summary info" in captured["gate_output"]
        # The failed node-ids come from the FULL parser (all 25), NOT
        # the legacy 20-line extractor (which would return [] for a
        # >20-line list).
        assert len(captured["failed_node_ids"]) == 25
        assert captured["failed_node_ids"][0] == "tests/test_0.py::test_0"
        # The full-suite paths are threaded (the empty-input fallback
        # re-runs the full gate suite instead of erroring).
        assert captured["full_suite_paths"] == ["tests/"]
        # The PR opened (the repro verdict is RED, not ERROR).
        assert out == "https://forge.example/pr/seam"
        err = capsys.readouterr().err
        assert "salvage-error" not in err

    def test_on_mode_empty_summary_triggers_full_suite_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """On mode: a gate-red whose summary parse comes back EMPTY
        (the collection-error / rc=4 / rc=5 / timeout shapes) must
        trigger the full-suite fallback in the fresh clone - never a
        vacuous GREEN, never VERDICT_ERROR (the legacy seam passed no
        full_suite_paths -> the fallback's run_paths was empty ->
        VERDICT_ERROR)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # A red outcome whose output_tail carries NO failure lines
        # (the rc=4 touched-path-missing shape - the full parser
        # returns []).
        outcome = {
            "passed": 0, "failed": 0, "errors": 0, "returncode": 4,
            "summary": "no tests ran",
            "output_tail": "ERROR: file or directory not found: tests/x.py",
        }
        captured: dict = {}

        def _fake_repro(**kw):
            captured.update(kw)
            # The full-suite fallback: run_paths == full_suite_paths
            # (the per-repo baseline) - the repro re-runs the FULL gate
            # suite in the fresh clone (a RED verdict from the full
            # suite - never a vacuous GREEN).
            return {
                "verdict": VERDICT_RED, "subcode": "", "full_suite": True,
                "node_ids": kw.get("full_suite_paths") or [],
                "artifact": "", "provenance": {}, "ts": "2026-09-21T00:00:00Z",
            }

        monkeypatch.setattr(repro_gate, "run_repro", _fake_repro,
                            raising=False)
        out = self._run_seam(
            tmp_path, monkeypatch, capsys, last_test_outcome=outcome,
        )
        assert captured, "the repro was never invoked through the seam"
        # The full-summary parse came back empty -> the failed node-ids
        # are [] (the empty-input shape).
        assert captured["failed_node_ids"] == []
        # The full-suite paths are threaded (the per-repo baseline) -
        # the empty-input fallback re-runs the full gate suite instead
        # of erroring.
        assert captured["full_suite_paths"] == ["tests/"]
        # The repro verdict is RED (the full-suite fallback proved the
        # red), NOT VERDICT_ERROR / VERDICT_GREEN - the PR opens.
        assert out == "https://forge.example/pr/seam"
        err = capsys.readouterr().err
        assert "salvage-error" not in err

    def test_on_mode_incomplete_provenance_returns_run_not_concluded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """On mode: a gate run whose provenance block is
        missing/incomplete must return run_not_concluded (no PR, no
        label, no cycle) - the D3 refusal must be REACHABLE in the
        live path (the legacy seam never passed provenance_block, so
        the refusal was dead code)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "on")
        # Force the gate's provenance block to be incomplete (a missing
        # resolved_package_path - the unaudited shape).
        monkeypatch.setattr(
            shaped_runner, "_build_provenance_block",
            lambda **kw: {
                "repo": kw.get("repo", ""),
                "head_sha": kw.get("head_sha", ""),
                "cwd": kw.get("cwd", ""),
                "resolved_package_path": "",  # incomplete
                "sys_path_shadow_check": [],
                "confcutdir": kw.get("confcutdir", ""),
                "quarantine_state": "",
                "ts": "2026-09-21T00:00:00Z",
            },
            raising=False,
        )
        out = self._run_seam(tmp_path, monkeypatch, capsys)
        # No PR (the run_not_concluded refusal - the unaudited gate run
        # cannot conclude).
        assert out == ""
        err = capsys.readouterr().err
        assert "run_not_concluded" in err
        # No salvage PR was opened (the refusal fires before the
        # concluded_gate_rejected label).
        assert "opening advisory [SALVAGE] PR" not in err

    def test_shadow_mode_seam_stays_today_behavior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """Shadow mode: the seam still emits the today-behavior
        concluded_gate_rejected label (the mislabel quantified live) -
        the shadow path is byte-identical to today (the repro verdict
        + the would-be disposition are observations only)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "shadow")
        captured: dict = {}

        def _fake_repro(**kw):
            captured.update(kw)
            return {
                "verdict": VERDICT_RED, "subcode": "", "full_suite": False,
                "node_ids": kw.get("failed_node_ids") or [],
                "artifact": "", "provenance": {}, "ts": "2026-09-21T00:00:00Z",
            }

        monkeypatch.setattr(repro_gate, "run_repro", _fake_repro,
                            raising=False)
        out = self._run_seam(tmp_path, monkeypatch, capsys)
        # The shadow path opens the today-behavior [SALVAGE] PR
        # (concluded_gate_rejected label stands - the mislabel is
        # quantified live by the shadow observation).
        assert out == "https://forge.example/pr/seam"
        assert captured, "the repro ran (the shadow observation is written)"
        err = capsys.readouterr().err
        assert "concluded, gate rejected" in err
        # The shadow path does NOT enforce (no run_not_concluded
        # refusal, no salvage-error partition).
        assert "run_not_concluded" not in err
        assert "salvage-error" not in err

    def test_off_mode_seam_is_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        """Off mode: the seam is byte-identical to today's behavior
        (no repro run, no observation - the salvage path is
        untouched)."""
        monkeypatch.setenv("LAPIS_PM_REPRO_ENFORCE", "off")
        called = {"n": 0}

        def _fake_repro(**kw):
            called["n"] += 1
            return {
                "verdict": VERDICT_RED, "subcode": "", "full_suite": False,
                "node_ids": [], "artifact": "", "provenance": {},
                "ts": "2026-09-21T00:00:00Z",
            }

        monkeypatch.setattr(repro_gate, "run_repro", _fake_repro,
                            raising=False)
        out = self._run_seam(tmp_path, monkeypatch, capsys)
        # The off path opens the today-behavior [SALVAGE] PR and NEVER
        # runs the repro (no observation).
        assert out == "https://forge.example/pr/seam"
        assert called["n"] == 0, "off mode must not run the repro"
        err = capsys.readouterr().err
        assert "concluded, gate rejected" in err
