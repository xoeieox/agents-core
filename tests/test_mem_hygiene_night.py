"""Hermetic tests for scripts/mem_hygiene_night.py — D-3 of
mem-hygiene-automation-v0 (the conductor night node in the
clone-currency-sync family).

The pipeline runs against a tmp_path fixture db (MEM_DB_PATH) with the
config pair pointed at a tmp_path file; the provenance deposit goes
through a fake `mem` CLI (a stub python script that appends its argv to a
JSONL file). No live mem.db, no live /data/slots, no network.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import mem_hygiene_night as mhn  # noqa: E402

NOW = datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    """A full hermetic environment: fixture db, config pair, fake mem CLI."""
    db = tmp_path / "mem.db"
    monkeypatch.setenv("MEM_DB_PATH", str(db))
    monkeypatch.delenv("MEM_HYGIENE_CONFIG", raising=False)
    monkeypatch.delenv("MEM_SERVER", raising=False)
    monkeypatch.delenv("MEM_ALLOW_TEST_WRITE", raising=False)

    (tmp_path / "dead_producers.json").write_text(json.dumps({
        "version": 1,
        "dead_sources": {"elevator/": ["elevator_scheduler"]},
    }))
    cfg = tmp_path / "mem_hygiene.json"
    cfg.write_text(json.dumps({
        "registry": "dead_producers.json",
        "allowlist": ["elevator/"],
        "dead_stream_age_days": 30,
        "batch_cap": 5000,
        "rollback_window_days": 14,
    }))

    # Fake mem CLI: appends its argv to a JSONL file and exits 0.
    deposits = tmp_path / "deposits.jsonl"
    cli = tmp_path / "fake_mem.py"
    cli.write_text(
        "import json, sys\n"
        f"with open({str(deposits)!r}, 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0)\n"
    )
    return {
        "db": db,
        "config": cfg,
        "slots": tmp_path / "slots",
        "cli": f"python3 {cli}",
        "deposits": deposits,
    }


def _seed_dead_stream(db: Path, n: int = 4, age_days: float = 45) -> None:
    from agents_core.mem import MemoryStore
    store = MemoryStore(db_path=db)
    for i in range(n):
        store.set(f"elevator/row-{i}", f"machine state {i}",
                  tags=["machine"], source="elevator_scheduler")
    ts = (NOW - timedelta(days=age_days)).isoformat()
    store._conn.execute(
        "UPDATE memories SET created_at=?, updated_at=? WHERE key LIKE 'elevator/%'",
        (ts, ts),
    )
    store._conn.commit()
    store.close()


def _run(env: dict, *extra: str) -> tuple[int, str]:
    # PYTHONPATH -> the staged tree so the night script imports the
    # DEPLOYED agents_core (this worktree), not whatever else is importable
    # (e.g. a stale /srv/agents checkout on sys.path).
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(mhn.__file__).parent.parent)
    proc = subprocess.run(
        ["python3", str(Path(mhn.__file__)),
         "--config", str(env["config"]),
         "--slots-dir", str(env["slots"]),
         "--mem-cli", env["cli"], *extra],
        capture_output=True, text=True, timeout=120, env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

def test_dry_run_writes_artifact_no_mutation_no_deposit(env):
    _seed_dead_stream(env["db"], n=4)
    rc, out = _run(env, "--dry-run")
    assert rc == 0, out

    # The report is printed as the last JSON document on stdout (after the
    # [mem-hygiene-night] log lines). Start at the last newline-brace
    # boundary — a bare rindex("{") can land inside a log line's JSON
    # fragment and json.loads() blows up.
    idx = out.rfind("\n{")
    report_text = out[idx + 1:] if idx != -1 else out[out.rindex("{"):]
    report = json.loads(report_text)
    assert report["mode"] == "dry-run"
    assert report["steps"]["classify"]["candidate_count"] == 4
    assert report["steps"]["mutation"] == "skipped (dry-run)"

    artifact = Path(report["steps"]["candidate_artifact"])
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["candidate_count"] == 4
    # The artifact's mode field records the caller's run mode (reviewer
    # medium, PR #331 cycle 1): a dry-run artifact says 'dry-run'.
    assert payload["mode"] == "dry-run"

    # No mutation.
    from agents_core.mem import MemoryStore
    store = MemoryStore(db_path=env["db"])
    assert len(store.list_by_prefix("elevator/")) == 4
    store.close()

    # No provenance deposit.
    assert not env["deposits"].exists()


def test_dry_run_no_candidates_is_clean(env):
    rc, out = _run(env, "--dry-run")
    assert rc == 0, out


# ---------------------------------------------------------------------------
# real run
# ---------------------------------------------------------------------------

def test_real_run_quarantines_and_deposits(env):
    _seed_dead_stream(env["db"], n=4)
    rc, out = _run(env)
    assert rc == 0, out

    # Quarantine happened.
    from agents_core.mem import MemoryStore
    store = MemoryStore(db_path=env["db"])
    assert store.list_by_prefix("elevator/") == []
    store.close()

    # The candidate artifact exists.
    artifacts = list(env["slots"].glob("mem-hygiene-candidates-*.json"))
    assert len(artifacts) == 1
    # The artifact is the pre-mutation snapshot of THIS run, so its mode
    # field records the run mode, not 'dry-run' (reviewer medium,
    # PR #331 cycle 1: it used to say 'dry-run' on a real run).
    payload = json.loads(artifacts[0].read_text())
    assert payload["mode"] == "run"
    assert payload["candidate_count"] == 4

    # The provenance line was deposited via the mem CLI with the right
    # key shape, the per-RUN key, and the lapis-pm,mem-hygiene tags.
    lines = [json.loads(l) for l in env["deposits"].read_text().splitlines()]
    assert len(lines) == 1
    argv = lines[0]
    assert argv[:2] == ["set", "decision/mem-hygiene-run-"]
    assert argv[3:5] == ["--tags", "lapis-pm,mem-hygiene"]
    content = argv[2]
    # The one-liner budget: <=120 chars on the first line.
    first_line = content.splitlines()[0]
    assert len(first_line) <= 120
    assert "quarantined" in first_line
    assert "candidate_artifact:" in content


def test_real_run_no_candidates_is_clean_no_deposit(env):
    rc, out = _run(env)
    assert rc == 0, out
    assert not env["deposits"].exists()


def test_real_run_cap_breach_fails(env):
    _seed_dead_stream(env["db"], n=6)
    cfg = json.loads(env["config"].read_text())
    cfg["batch_cap"] = 3
    env["config"].write_text(json.dumps(cfg))
    rc, out = _run(env)
    assert rc == 1, out
    assert "ABORTED" in out
    # Nothing mutated.
    from agents_core.mem import MemoryStore
    store = MemoryStore(db_path=env["db"])
    assert len(store.list_by_prefix("elevator/")) == 6
    store.close()
    assert not env["deposits"].exists()


def test_real_run_config_missing_fails(env):
    rc, out = _run(env)
    # Point at a nonexistent config via a fresh invocation.
    proc = subprocess.run(
        ["python3", str(Path(mhn.__file__)),
         "--config", str(env["slots"] / "nope.json"),
         "--slots-dir", str(env["slots"]),
         "--mem-cli", env["cli"]],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert "FAILED" in proc.stdout + proc.stderr
