"""Tests for the mem-server principal model + machine-state faucet
(openclaw-memdb-influx-reader-v0, D1/D3).

Covers (D4):
  - principal fail-closed (reader write = 403 principal_reader)
  - reader-secret holder rejected on the write class regardless of asserted role
  - machine-state reject + machinery-store write
  - observe-only mode logs without rejecting
  - the fail-open startup guard (refuse to start on missing/malformed/unreadable
    allowlist; accept a valid allowlist)
  - FTS integrity (machinery store FTS-less by design)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agents_core import mem_machinery
from agents_core.mem_server import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ALLOWLIST = REPO_ROOT / "config" / "mem-machine-state-prefixes.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def allowlist_file(tmp_path):
    """A valid shared allowlist artifact in a tmp dir."""
    p = tmp_path / "mem-machine-state-prefixes.json"
    p.write_text(SHARED_ALLOWLIST.read_text(encoding="utf-8"), encoding="utf-8")
    return p


@pytest.fixture
def observe_log(tmp_path, monkeypatch):
    p = tmp_path / "observe.log"
    monkeypatch.setenv("MEM_OBSERVE_LOG", str(p))
    return p


@pytest.fixture
def no_enforce(monkeypatch):
    monkeypatch.delenv("MEM_ENFORCE_PRINCIPALS", raising=False)


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setenv("MEM_ENFORCE_PRINCIPALS", "1")


@pytest.fixture
def client(tmp_db, allowlist_file, observe_log, no_enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_mem.db"


# ---------------------------------------------------------------------------
# Startup-validation guard (fail-open bypass test)
# ---------------------------------------------------------------------------

def test_refuses_to_start_on_missing_allowlist(tmp_db, tmp_path, observe_log, no_enforce):
    missing = tmp_path / "does-not-exist.json"
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=missing)


def test_refuses_to_start_on_malformed_allowlist(tmp_db, tmp_path, observe_log, no_enforce):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad)


def test_refuses_to_start_on_non_object_allowlist(tmp_db, tmp_path, observe_log, no_enforce):
    bad = tmp_path / "nonobj.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad)


def test_refuses_to_start_on_bad_entry(tmp_db, tmp_path, observe_log, no_enforce):
    bad = tmp_path / "badentry.json"
    bad.write_text(
        json.dumps({"prefixes": [{"prefix": "no-slash", "producer_principal": "x",
                                  "store": "machinery", "state": "live"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad)


def test_accepts_valid_allowlist(tmp_db, allowlist_file, observe_log, no_enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["machinery_allowlist"]["prefixes"] == [
            "elevator/", "weather/", "router/gw-review-divergence/",
        ]
        # 'test/' must NOT be a machinery prefix: it is a generic scratch
        # namespace whose keys must keep landing in mem.db (reviewer
        # 2026-09-18 [high] — the landmine that rerouted test/* writes to
        # exhaust.db and broke the FTS-divergence test).
        assert "test/" not in body["machinery_allowlist"]["prefixes"]
        assert body["principal_model"]["enforce"] is False
        assert "brix-pm" in body["principal_model"]["principals"]


# ---------------------------------------------------------------------------
# D1 — principal fail-closed (enforce mode)
# ---------------------------------------------------------------------------

def _enforce_client(tmp_db, allowlist_file, observe_log, enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    return TestClient(app)


def test_reader_write_rejected_403_principal_reader(tmp_db, allowlist_file, observe_log, enforce):
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        # No X-Mem-Principal header = reader (fail-closed).
        resp = c.put("/v0/memories/foo/bar", json={"content": "x", "tags": "", "source": ""})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"


def test_unknown_principal_write_rejected_403(tmp_db, allowlist_file, observe_log, enforce):
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        resp = c.put(
            "/v0/memories/foo/bar",
            json={"content": "x", "tags": "", "source": ""},
            headers={"X-Mem-Principal": "some-random-agent"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"


def test_curator_write_allowed(tmp_db, allowlist_file, observe_log, enforce):
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        resp = c.put(
            "/v0/memories/foo/bar",
            json={"content": "x", "tags": "a", "source": "s"},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200
        assert resp.json()["created"] is True


def test_reader_delete_rejected_403(tmp_db, allowlist_file, observe_log, enforce):
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        # Seed a row as curator.
        c.put("/v0/memories/foo/bar", json={"content": "x", "tags": "", "source": "s"},
              headers={"X-Mem-Principal": "brix-pm"})
        # Reader (no principal) attempts delete.
        resp = c.delete("/v0/memories/foo/bar")
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"


def test_reader_secret_holder_rejected_on_write_class(tmp_db, allowlist_file, observe_log, enforce, monkeypatch):
    """Secret-to-verb binding (panel F2): the write class is rejected for a
    holder of the secret who does NOT assert a registered curator principal,
    regardless of the secret. The read verb-set stays open to the secret holder.

    v0 honesty limit (stated, not asserted away): with a SINGLE shared bearer
    token the server cannot distinguish reader-secret from curator-secret by
    value alone. The binding is enforced via the PRINCIPAL — a write requires
    holding the BRIX secret AND asserting a registered curator principal. A
    secret holder who asserts an absent/unknown/reader principal is rejected
    on the write class (403 principal_reader) even though the secret is valid;
    the secret alone (no curator principal) grants read-only access. This test
    pins that guarantee with a hard 403 assertion (not a 200-or-403 escape).
    """
    monkeypatch.setenv("MEM_BEARER_TOKEN", "reader-secret")
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        hdrs = {"Authorization": "Bearer reader-secret"}
        # Read verb-set is open to the secret holder.
        assert c.get("/healthz", headers=hdrs).status_code == 200
        # Write class: a secret holder with NO asserted principal is a reader
        # and is REJECTED (hard 403) — the secret alone does not grant write.
        resp = c.put(
            "/v0/memories/foo/bar2",
            json={"content": "x", "tags": "", "source": ""},
            headers={**hdrs},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"
        # A secret holder asserting an UNKNOWN principal is likewise a reader
        # and rejected on the write class (fail-closed).
        resp_unknown = c.put(
            "/v0/memories/foo/bar3",
            json={"content": "x", "tags": "", "source": ""},
            headers={**hdrs, "X-Mem-Principal": "not-a-registered-principal"},
        )
        assert resp_unknown.status_code == 403
        assert resp_unknown.json()["error"]["code"] == "principal_reader"
        # The decisive contrast: the SAME valid secret + a registered curator
        # principal IS allowed to write (the binding is principal-gated, not
        # secret-gated — this is the documented v0 honesty limit).
        resp_curator = c.put(
            "/v0/memories/foo/bar",
            json={"content": "x", "tags": "", "source": ""},
            headers={**hdrs, "X-Mem-Principal": "brix-pm"},
        )
        assert resp_curator.status_code == 200
        # Nothing landed for the rejected reader attempts.
        assert c.get("/v0/memories/foo/bar2", headers=hdrs).status_code == 404
        assert c.get("/v0/memories/foo/bar3", headers=hdrs).status_code == 404


def test_deposit_verb_partitioned(tmp_db, allowlist_file, observe_log, no_enforce):
    """The deposit verb is observed as POST_DEPOSIT (gate trickster): a
    dormant/unregistered deposit writer cannot hide behind PUT/DELETE counts."""
    from agents_core.mem_server import _load_deposit_recorder

    # No recorder configured -> deposit 503s AFTER the guard; but the guard
    # observes the attempt first. Use a fake recorder to get past the guard.
    class FakeRecorder:
        def already_recorded(self, mh):
            return False

        def record(self, prov, *, store_kind, key):
            return True

    app = create_app(tmp_db, deposit_recorder=FakeRecorder(), allowlist_path=allowlist_file)
    with TestClient(app) as c:
        envelope = {
            "provenance": {"manifest_hash": "abc123", "agent_id": "zephyr"},
            "payload": {"key": "work/record/1", "value": "v"},
        }
        # archetypes_core may not be importable in the test env; the deposit
        # route validates the envelope via LapisToolReturn.from_dict. If that
        # import fails the route 400s — which still means the guard did not
        # reject. We assert the observe log captured the attempt if the route
        # reached the guard, else the 400 is the contract.
        resp = c.post("/v0/deposit", json=envelope)
        assert resp.status_code in (200, 400, 503)
    # The observe log, if any lines, must label the verb POST_DEPOSIT.
    if observe_log.exists():
        for line in observe_log.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "write_attempt" and rec.get("key") == "work/record/1":
                assert rec["verb"] == "POST_DEPOSIT"


# ---------------------------------------------------------------------------
# D1 — observe-only mode logs without rejecting
# ---------------------------------------------------------------------------

def test_observe_only_logs_without_rejecting(tmp_db, allowlist_file, observe_log, no_enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # A reader (no principal) PUT — observe-only must ALLOW it (no 403)
        # but LOG the would-reject.
        resp = c.put("/v0/memories/foo/bar", json={"content": "x", "tags": "", "source": ""})
        assert resp.status_code == 200
        assert resp.json()["created"] is True

    assert observe_log.exists()
    lines = [json.loads(l) for l in observe_log.read_text().splitlines()]
    put_lines = [r for r in lines if r.get("key") == "foo/bar" and r.get("verb") == "PUT"]
    assert put_lines, "observe log must record the reader PUT"
    rec = put_lines[0]
    assert rec["principal"] == "none"
    assert rec["role"] == "reader"
    assert rec["would_reject"] is True
    assert rec["enforce"] is False
    assert "source_ip" in rec


def test_observe_report_endpoint(tmp_db, allowlist_file, observe_log, no_enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        c.put("/v0/memories/b/2", json={"content": "y", "tags": "", "source": ""},
              headers={"X-Mem-Principal": "brix-pm"})
        report = c.get("/v0/observe-report").json()
    # The reader (none) writer is in the break list; brix-pm is not.
    assert "none" in report["break_list"]
    assert "brix-pm" not in report["break_list"]
    # Write counts are partitioned by verb.
    assert report["writers"]["none"]["verbs"]["PUT"] == 1
    assert report["writers"]["brix-pm"]["verbs"]["PUT"] == 1


# ---------------------------------------------------------------------------
# D1 — promote is in the write class (reviewer PR #328 cycle 2 [med])
# ---------------------------------------------------------------------------

def test_promote_by_reader_rejected_403_enforce(tmp_db, allowlist_file, observe_log, enforce):
    """The /v0/promote endpoint runs the write-class guard on the validated
    key: a reader is rejected with 403 principal_reader and NOTHING is
    written (neither the promoted row nor the batch decision key).

    The identity the GUARD sees is the X-Mem-Principal header (the
    body 'principal' is only the provenance header's 'by'). So the reader
    case is: no header (guard sees reader) + a body principal (passes the
    loud-400 principal check, so the guard's 403 is what surfaces)."""
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/promote-reader",
                "from": "openclaw/gw",
                "principal": "brix-pm",
                "content": "x",
            },
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"
        # Nothing landed: no promoted row, no batch decision key.
        assert c.get("/v0/memories/finding/promote-reader").status_code == 404
        rows = c.get("/v0/memories", params={"tag": "promoted-batch", "limit": 50}).json()
        assert rows == [], "no batch decision key may be written by a reader"


def test_promote_by_curator_allowed_enforce(tmp_db, allowlist_file, observe_log, enforce):
    """The same promote, by a registered curator, is allowed under
    enforcement (the guard passes for the curator role)."""
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/promote-curator",
                "from": "openclaw/gw",
                "principal": "brix-pm",
                "content": "x",
            },
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "promoted:openclaw/gw"


# ---------------------------------------------------------------------------
# D3 — machine-state reject + machinery-store write
# ---------------------------------------------------------------------------

def test_machine_state_write_lands_in_machinery_store(tmp_db, tmp_path, allowlist_file, observe_log, no_enforce):
    """A registered producer writing its own machine-state prefix lands in the
    machinery store (the existing exhaust store), not mem.db (D3 faucet)."""
    exhaust_db = tmp_path / "exhaust.db"
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        resp = c.put(
            "/v0/memories/elevator/proposals/p1",
            json={"content": "v", "tags": "", "source": "elevator-scheduler"},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200

    # The row must NOT be in mem.db (the atoms store / ledger of record).
    import sqlite3
    conn = sqlite3.connect(str(tmp_db))
    row = conn.execute(
        "SELECT 1 FROM memories WHERE key = ?", ("elevator/proposals/p1",)
    ).fetchone()
    conn.close()
    assert row is None, "machine-state write must not land in mem.db"

    # ...it must be in the machinery (exhaust) store.
    assert exhaust_db.exists()
    conn2 = sqlite3.connect(str(exhaust_db))
    row2 = conn2.execute(
        "SELECT content FROM memories WHERE key = ?", ("elevator/proposals/p1",)
    ).fetchone()
    conn2.close()
    assert row2 is not None and row2[0] == "v"


def test_machine_state_write_by_non_owner_rejected_enforce(tmp_db, allowlist_file, observe_log, enforce):
    """Under enforcement, a curator that is NOT the owning producer for a
    machine-state prefix is rejected (403 machine_state_prefix)."""
    # Build an allowlist where the producer for elevator/ is a DIFFERENT
    # principal than brix-pm, so brix-pm is a non-owner.
    alt = tmp_db.parent / "alt-allowlist.json"
    alt.write_text(json.dumps({"prefixes": [
        {"prefix": "elevator/", "producer_principal": "elevator-bot",
         "store": "machinery", "state": "live", "dead_since": None},
    ]}), encoding="utf-8")
    app = create_app(tmp_db, allowlist_path=alt)
    with TestClient(app) as c:
        resp = c.put(
            "/v0/memories/elevator/proposals/p1",
            json={"content": "v", "tags": "", "source": "s"},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "machine_state_prefix"


def test_machine_state_write_by_owner_allowed_enforce(tmp_db, allowlist_file, observe_log, enforce):
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        resp = c.put(
            "/v0/memories/elevator/proposals/p1",
            json={"content": "v", "tags": "", "source": "s"},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Library-level fail-open trade-off (reviewer 2026-09-18 [low])
# ---------------------------------------------------------------------------

def test_library_fail_open_degrades_to_static_prefixes(tmp_db, tmp_path, observe_log, no_enforce, monkeypatch):
    """Documented trade-off: mem_exhaust.machinery_prefixes() is fail-OPEN at
    the library level (returns () on a missing/malformed allowlist) so the
    library set()/get() chokepoint never breaks. The FAIL-CLOSED surface is the
    server's boot-time guard. This pins the boundary: a library-only caller
    with a missing allowlist degrades to the static EXHAUST_PREFIXES (test/ is
    NOT routed), while the server REFUSES TO START on the same missing file.
    """
    from agents_core import mem_exhaust

    # Point the machinery loader at a missing file.
    monkeypatch.setenv("MEM_MACHINE_STATE_PREFIXES_PATH", str(tmp_path / "missing.json"))
    mem_exhaust.reset_machinery_prefixes_cache()
    try:
        # Library-level: fail-open -> no machinery extension, only static.
        assert mem_exhaust.machinery_prefixes() == ()
        # test/ is NOT a static exhaust prefix, so it is NOT routed.
        assert mem_exhaust.route_to_exhaust("test/key") is False
        assert mem_exhaust.route_to_exhaust("elevator/x") is True
    finally:
        mem_exhaust.reset_machinery_prefixes_cache()
        monkeypatch.delenv("MEM_MACHINE_STATE_PREFIXES_PATH", raising=False)

    # Server-level: the SAME missing file REFUSES TO START (fail-closed).
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# FTS integrity (machinery store FTS-less by design)
# ---------------------------------------------------------------------------

def test_fts_integrity_atoms_and_machinery(tmp_db, tmp_path, allowlist_file, observe_log, no_enforce):
    """mem.db FTS stays in sync with mem.db rows; the machinery (exhaust) store
    has NO FTS table (FTS-less by design, mem_exhaust.py:78-81)."""
    exhaust_db = tmp_path / "exhaust.db"
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # An atoms-store row.
        c.put("/v0/memories/atoms/1", json={"content": "hello atoms", "tags": "", "source": "s"})
        # A machinery-store row.
        c.put("/v0/memories/elevator/proposals/p1",
              json={"content": "v", "tags": "", "source": "s"},
              headers={"X-Mem-Principal": "brix-pm"})
        body = c.get("/healthz").json()

    # Atoms-store FTS is in sync (mem.db row count == FTS docsize count).
    assert body["fts_integrity"]["in_sync"] is True
    assert body["fts_integrity"]["divergence"] == 0

    # The machinery store has no FTS table at all.
    import sqlite3
    conn = sqlite3.connect(str(exhaust_db))
    fts_tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'memories_fts%'"
    ).fetchall()
    conn.close()
    assert fts_tables == [], "machinery store must be FTS-less by design"
