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
# Excluded-prefix parsing (reviewer 2026-09-18, PR #330 cycle 1 [low])
# ---------------------------------------------------------------------------

def test_excluded_prefix_is_machine_readable(tmp_path):
    """The 'excluded' section of the allowlist is parsed and validated so
    the exclusion rationale is machine-readable (not just human-readable in
    the top-level 'note' string). A parser reading the JSON alone can see
    WHY a prefix is absent from the faucet list."""
    from agents_core.mem_machinery import load_allowlist

    allowlist = load_allowlist(SHARED_ALLOWLIST)
    # The shared allowlist has 'test/' in its excluded section.
    assert "test/" in allowlist.excluded_prefixes
    # The excluded prefix is NOT in the machine-state entries.
    assert "test/" not in allowlist.prefixes
    # The rationale is machine-readable (not just a human note).
    test_entry = next(e for e in allowlist.excluded if e.prefix == "test/")
    assert test_entry.state == "excluded"
    assert test_entry.reason  # non-empty
    # The excluded prefix is never treated as machine-state.
    assert not allowlist.is_machine_state("test/key")
    assert allowlist.is_excluded("test/key")


def test_excluded_prefix_malformed_refuses(tmp_db, tmp_path, observe_log, no_enforce):
    """A malformed 'excluded' entry (missing reason, wrong state, or a
    prefix that is also in 'prefixes') REFUSES TO START (fail-closed)."""
    # Missing reason.
    bad = tmp_path / "bad-excluded.json"
    bad.write_text(json.dumps({"prefixes": [], "excluded": [
        {"prefix": "test/", "state": "excluded"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad)

    # Wrong state.
    bad2 = tmp_path / "bad-state.json"
    bad2.write_text(json.dumps({"prefixes": [], "excluded": [
        {"prefix": "test/", "state": "live", "reason": "x"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad2)

    # Contradictory: a prefix in both 'prefixes' and 'excluded'.
    bad3 = tmp_path / "contradictory.json"
    bad3.write_text(json.dumps({
        "prefixes": [{"prefix": "test/", "producer_principal": "x",
                      "store": "machinery", "state": "live"}],
        "excluded": [{"prefix": "test/", "state": "excluded", "reason": "x"}],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="REFUSES TO START"):
        create_app(tmp_db, allowlist_path=bad3)


def test_excluded_in_healthz(tmp_db, allowlist_file, observe_log, no_enforce):
    """The healthz endpoint surfaces the excluded prefixes (machine-readable
    rationale) so an operator can see WHY a prefix is absent from the
    faucet list."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        body = c.get("/healthz").json()
    excluded = body["machinery_allowlist"]["excluded"]
    # The shared allowlist has 'test/' excluded.
    assert any(e["prefix"] == "test/" for e in excluded)
    test_entry = next(e for e in excluded if e["prefix"] == "test/")
    assert test_entry["reason"]  # non-empty


# ---------------------------------------------------------------------------
# Uniform rejection envelope (spec D-1 named rejection contract)
# ---------------------------------------------------------------------------

def test_rejection_envelope_is_flat_error(tmp_db, allowlist_file, observe_log, enforce):
    """The spec's named rejection contract is the FLAT envelope
    {"error": {"code", "message"}} — the 403 principal_reader path is the
    reference shape, and EVERY other mem-server error path must match it.

    Reviewer PR #342 cycle 1 [med]: the 400 promote/land paths previously
    raised HTTPException(detail=_error(...)), which FastAPI serializes as
    {"detail":{"error":{...}}} — a DIFFERENT envelope than the spec's named
    contract. This test pins the uniform envelope across the error classes:
    404 not_found, 400 bad_request (search), 400 bad_request (promote
    requires a principal), and 400 bad_disposition (observe-report/land).
    The server-side chokepoint is _http_error() in mem_server.py — a new
    HTTPException(detail=...) would break this test.
    """
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # 404 not_found (GET a missing key).
        resp = c.get("/v0/memories/does/not/exist")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "not_found"
        assert "message" in body["error"]
        assert "detail" not in body, "must be the flat {error:{...}} envelope"

        # 400 bad_request (search with empty q).
        resp = c.get("/v0/search", params={"q": ""})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "bad_request"
        assert "detail" not in body

        # 400 bad_request (promote without a curator principal).
        resp = c.post(
            "/v0/promote",
            json={"key": "finding/x", "from": "openclaw/gw", "content": "x"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "bad_request"
        assert "detail" not in body

        # 400 bad_disposition (observe-report/land with a bad disposition).
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        resp = c.post(
            "/v0/observe-report/land",
            json={"disposition": {"none": "not-a-real-disposition"}},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "bad_disposition"
        assert "detail" not in body


# ---------------------------------------------------------------------------
# D1 — principal fail-closed (enforce mode)
# ---------------------------------------------------------------------------

def _server_client_observe(tmp_db, allowlist_file, observe_log, no_enforce):
    """An observe-only (non-enforce) TestClient for the promote tests."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    return TestClient(app)


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
    dormant/unregistered deposit writer cannot hide behind PUT/DELETE counts.

    Deterministic (reviewer PR #334 cycle 1 [med]): the envelope is built via
    archetypes_core.provenance.to_lapis_return (the same helper
    tests/test_mem_deposit.py uses), so LapisToolReturn.from_dict in the
    route is guaranteed to parse it and the guard RUNS — the assertion below
    is on a guaranteed log line, not a vacuous 'if the route reached the
    guard' conditional. archetypes_core is a hard dep of this test (it
    already is for test_mem_deposit.py; the mem_server.py route imports it
    lazily, so the package stays importable without it). It is NOT declared
    in pyproject.toml [project.optional-dependencies] test (reviewer PR
    #339 cycle 1 [med]) — the gate environment has it installed (the
    existing test_mem_deposit.py suite depends on it). Reviewer PR #342
    cycle 1 [low] fix: the import is now GUARDED — a clean environment
    without archetypes_core skips this test (with a named reason) instead of
    failing at import; the deposit path is still covered by
    test_mem_deposit.py in the gate env (which has the dep)."""
    try:
        from archetypes_core.provenance import to_lapis_return
    except ImportError:
        pytest.skip(
            "archetypes_core not installed (undeclared hard test dependency, "
            "named in the docstring; the gate env has it — test_mem_deposit.py "
            "covers the deposit path there)"
        )

    class FakeRecorder:
        def already_recorded(self, mh):
            return False

        def record(self, prov, *, store_kind, key):
            return True

    app = create_app(tmp_db, deposit_recorder=FakeRecorder(), allowlist_path=allowlist_file)
    with TestClient(app) as c:
        envelope = to_lapis_return(
            {"key": "work/record/1", "value": "v"},
            agent_id="zephyr",
            tool="pytest",
            summary="deposit verb-partition test",
        )
        resp = c.post("/v0/deposit", json=json.loads(envelope.to_json()))
        # A valid envelope + a configured recorder must be ACCEPTED — if this
        # 400s, the guard never ran and the verb-partition assertion below
        # would be vacuous; fail loudly instead.
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "accepted"
    # The observe log MUST contain the attempt, labeled POST_DEPOSIT.
    assert observe_log.exists(), "observe log must exist after a write attempt"
    lines = [json.loads(l) for l in observe_log.read_text().splitlines()]
    deposit_lines = [
        r for r in lines
        if r.get("event") == "write_attempt" and r.get("key") == "work/record/1"
    ]
    assert deposit_lines, "observe log must record the deposit attempt"
    assert all(r["verb"] == "POST_DEPOSIT" for r in deposit_lines)


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
    # The break-list entry carries a disposition (the escalation-path
    # signal) — default "none" (undispositioned = a silent deferral).
    assert report["break_list"]["none"]["disposition"] == "none"
    # The escalation verdict: a non-empty break list with an
    # undispositioned writer escalates (the flip gate escalates to Erah).
    assert report["escalate"] is True


def test_observe_report_land_named_key(tmp_db, allowlist_file, observe_log, no_enforce):
    """The observe-week report LANDS at the named mem key
    state/memdb-influx-observe-week-<YYYYMMDD> (report body) + the raw
    log (D-1 / D-4 named deliverable; reviewer PR #338 cycle 1 [high]).

    The /v0/observe-report endpoint returning the in-process aggregation
    is NOT the landing artifact — landing writes the report body through
    store.upsert_line() into the named key. The landing is a WRITE
    (POST /v0/observe-report/land, observed as POST_LAND) and is
    curator-gated under enforcement."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # A reader write (break-list entry) + a curator write.
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        c.put("/v0/memories/b/2", json={"content": "y", "tags": "", "source": ""},
              headers={"X-Mem-Principal": "brix-pm"})

        # Land the report (curator principal — the landing is a write).
        resp = c.post(
            "/v0/observe-report/land",
            json={},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

    # The named key shape: state/memdb-influx-observe-week-<YYYYMMDD>.
    from datetime import datetime, timezone
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert body["key"] == f"state/memdb-influx-observe-week-{day}"
    # The raw log path is the named landing artifact's raw log.
    assert body["raw_log"] == str(observe_log)

    # The report body landed in the named mem key (read it back).
    app2 = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app2) as c2:
        row = c2.get(f"/v0/memories/{body['key']}").json()
    assert row["key"] == body["key"]
    # The content is the header line + a JSON report line (upsert_line
    # shape: header + line).
    lines = row["content"].splitlines()
    assert lines[0].startswith(f"Observe-week report {day}")
    # The report line is valid JSON carrying the per-writer table +
    # break list + escalation verdict.
    report_line = json.loads(lines[1])
    assert report_line["writers"]["none"]["verbs"]["PUT"] == 1
    assert "none" in report_line["break_list"]
    assert report_line["break_list"]["none"]["disposition"] == "none"
    assert report_line["escalate"] is True


def test_observe_report_land_reader_rejected_enforce(tmp_db, allowlist_file, observe_log, enforce):
    """The landing is a WRITE (write class): a reader cannot land the
    report under enforcement (403 principal_reader) — the landing key is
    machine-state-free, so the guard's role check is the whole gate."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # A reader write (break-list entry).
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        # Reader attempts to land the report — rejected.
        resp = c.post("/v0/observe-report/land", json={})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "principal_reader"
        # Nothing landed: the named key does not exist.
        from datetime import datetime, timezone
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert c.get(f"/v0/memories/state/memdb-influx-observe-week-{day}").status_code == 404


def test_observe_report_land_observed_as_post_land(tmp_db, allowlist_file, observe_log, no_enforce):
    """The landing verb is observed as POST_LAND — its OWN verb in the
    per-writer table (gate trickster verb partitioning): a curator
    landing the report must not be mislabeled as a PUT."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        resp = c.post(
            "/v0/observe-report/land",
            json={},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200
    assert observe_log.exists()
    for line in observe_log.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("verb") == "POST_LAND":
            assert rec["principal"] == "brix-pm"
            break
    else:
        raise AssertionError("observe log must record the landing as POST_LAND")


def test_observe_report_disposition_migrations_line(tmp_db, allowlist_file, observe_log, no_enforce):
    """The escalation path (spec D-1 hard requirement): a 'migration
    line' is a dispositioned OBLIGATION, not a silent deferral. The
    landing accepts a disposition mapping; a dispositioned writer does
    NOT escalate, but an undispositioned one does."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # Two reader writes (two break-list entries: 'none').
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        c.put("/v0/memories/b/2", json={"content": "y", "tags": "", "source": ""})

        # Disposition the break-list writer 'none' as a named migration
        # line (the obligation text).
        resp = c.post(
            "/v0/observe-report/land",
            json={"disposition": {"none": "migration-line:migrate to registered bot by 2026-09-21"}},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200, resp.text
        report = resp.json()["report"]

    # The disposition is recorded on the break-list entry.
    assert report["break_list"]["none"]["disposition"] == "migration-line:migrate to registered bot by 2026-09-21"
    # A dispositioned writer does NOT escalate (the migration line is the
    # obligation; the gate does not escalate on a dispositioned writer).
    assert report["escalate"] is False


def test_observe_report_disposition_bare_migration_line_rejected(tmp_db, allowlist_file, observe_log, no_enforce):
    """A bare 'migration-line' (no line text) is a silent deferral
    disguised as a migration line — REJECTED with a loud 400 (the gate
    must not accept a silent deferral as a dispositioned obligation)."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        resp = c.post(
            "/v0/observe-report/land",
            json={"disposition": {"none": "migration-line"}},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "bad_disposition"


def test_observe_report_disposition_unknown_rejected(tmp_db, allowlist_file, observe_log, no_enforce):
    """An unknown disposition value is rejected with a loud 400 (fail-
    closed: a typo in the disposition must not silently land)."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        c.put("/v0/memories/a/1", json={"content": "x", "tags": "", "source": ""})
        resp = c.post(
            "/v0/observe-report/land",
            json={"disposition": {"none": "not-a-real-disposition"}},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "bad_disposition"


def test_observe_report_disposition_unobserved_writer_rejected(tmp_db, allowlist_file, observe_log, no_enforce):
    """A disposition for a writer that was never observed is a
    configuration error (nothing to disposition) — loud 400, not silent."""
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        # No writes at all — 'ghost' is not in the break list.
        resp = c.post(
            "/v0/observe-report/land",
            json={"disposition": {"ghost": "brix-pm"}},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "bad_disposition"


# ---------------------------------------------------------------------------
# D1 — promote is in the write class (reviewer PR #328 cycle 2 [med])
# ---------------------------------------------------------------------------

def test_promote_by_reader_rejected_403_enforce(tmp_db, allowlist_file, observe_log, enforce):
    """The /v0/promote endpoint runs the write-class guard on the validated
    key: a reader is rejected with 403 principal_reader and NOTHING is
    written (neither the promoted row nor the batch decision key).

    The identity the GUARD sees is the X-Mem-Principal header, and the
    audit artifact (provenance 'by' + batch key) must carry that SAME
    verified identity — a body 'principal' that differs from the header
    is rejected first (400 principal_mismatch, reviewer PR #333 cycle 1
    [med]), and a body principal can never substitute for the header
    (the guard keys off the header). So the reader case is: NO header
    (guard sees reader) + NO body principal — the guard's 403 is what
    surfaces, not a forged artifact."""
    with _enforce_client(tmp_db, allowlist_file, observe_log, enforce) as c:
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/promote-reader",
                "from": "openclaw/gw",
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


def test_promote_body_principal_mismatch_rejected_400(tmp_db, allowlist_file, observe_log, no_enforce):
    """The body 'principal' is a client-supplied string: when it differs
    from the VERIFIED X-Mem-Principal header, the promote is rejected with
    a loud 400 principal_mismatch and NOTHING is written (reviewer PR #333
    cycle 1 [med]). The audit artifact (provenance header 'by' + batch
    decision key) must carry the verified principal — a caller must not be
    able to assert a reader header (the guard sees reader) while forging
    the artifact to any name."""
    with _server_client_observe(tmp_db, allowlist_file, observe_log, no_enforce) as c:
        # Reader header + a body principal that is NOT the header.
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/forge-attempt",
                "from": "openclaw/gw",
                "principal": "brix-pm",
                "content": "x",
            },
            headers={"X-Mem-Principal": "some-reader"},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "principal_mismatch"
        # Nothing landed: no promoted row, no batch decision key.
        assert c.get("/v0/memories/finding/forge-attempt").status_code == 404
        rows = c.get("/v0/memories", params={"tag": "promoted-batch", "limit": 50}).json()
        assert rows == [], "no batch decision key may carry a forged principal"


def test_promote_body_principal_absent_uses_header(tmp_db, allowlist_file, observe_log, no_enforce):
    """When the body 'principal' is absent, the VERIFIED header principal
    is used for the provenance artifact (spec D2: --by is the curator
    principal; the guard is the principal check)."""
    with _server_client_observe(tmp_db, allowlist_file, observe_log, no_enforce) as c:
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/header-only",
                "from": "openclaw/gw",
                "content": "x",
            },
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["batch_key"].endswith("-brix-pm")
        from agents_core.mem_client import parse_promoted_header
        parsed = parse_promoted_header(body["content"])
        assert parsed["principal"] == "brix-pm"


def test_promote_verb_partitioned_post_promote(tmp_db, allowlist_file, observe_log, no_enforce):
    """The /v0/promote verb is observed as POST_PROMOTE — its OWN verb in
    the per-writer table, NOT a PUT (reviewer PR #333 cycle 1 [med]; gate
    trickster verb partitioning): a curator promote must not be
    mislabeled as a PUT in the observe log or the report."""
    with _server_client_observe(tmp_db, allowlist_file, observe_log, no_enforce) as c:
        resp = c.post(
            "/v0/promote",
            json={
                "key": "finding/verb-partition",
                "from": "openclaw/gw",
                "principal": "brix-pm",
                "content": "x",
            },
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200

        # The observe log labels the verb POST_PROMOTE, not PUT.
        assert observe_log.exists()
        for line in observe_log.read_text().splitlines():
            import json as _json
            rec = _json.loads(line)
            if rec.get("key") == "finding/verb-partition":
                assert rec["verb"] == "POST_PROMOTE"

        # The report partitions the count under POST_PROMOTE; the PUT count
        # for the same writer is 0 (no conflation with the PUT write class).
        report = c.get("/v0/observe-report").json()
        w = report["writers"]["brix-pm"]
        assert w["verbs"]["POST_PROMOTE"] == 1
        assert w["verbs"]["PUT"] == 0
        assert w["total"] == 1


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


def test_machinery_row_covered_by_checkpoint_wal(tmp_db, tmp_path, allowlist_file, observe_log, no_enforce):
    """The checkpoint mechanism covers the machinery (exhaust) store
    (spec D-3: the mem.py:371-373 checkpoint pattern + mem-checkpoint.service).

    RESCOPED (rev-2): the machinery store IS the existing exhaust store, so
    the D-3 'extend the checkpoint mechanism' requirement is satisfied by
    MemoryStore.checkpoint_wal() checkpointing the exhaust sibling
    (mem.py:420-424). This test PINS that coverage (reviewer PR #342 cycle
    1 [med]: the rescoping was documented in comments only, not pinned by a
    test): a machine-state (elevator/) row written through the server is
    covered by store.checkpoint_wal() — the WAL is truncated (checkpoint
    returns (0, 0, 0): busy=0, log=0, remaining=0) and the row survives the
    checkpoint in the machinery store.
    """
    from agents_core.mem import MemoryStore

    exhaust_db = tmp_path / "exhaust.db"
    app = create_app(tmp_db, allowlist_path=allowlist_file)
    with TestClient(app) as c:
        resp = c.put(
            "/v0/memories/elevator/checkpoint/p1",
            json={"content": "v", "tags": "", "source": "elevator-scheduler"},
            headers={"X-Mem-Principal": "brix-pm"},
        )
        assert resp.status_code == 200

    # A fresh store instance on the same db (the mem-checkpoint.service
    # pattern: a fresh MemoryStore() every firing) checkpoints the WAL.
    store = MemoryStore(tmp_db)
    try:
        # The machinery row must be in the exhaust sibling (not mem.db).
        assert exhaust_db.exists()
        import sqlite3
        conn = sqlite3.connect(str(exhaust_db))
        before = conn.execute(
            "SELECT content FROM memories WHERE key = ?",
            ("elevator/checkpoint/p1",),
        ).fetchone()
        conn.close()
        assert before is not None and before[0] == "v"

        # checkpoint_wal() covers the machinery (exhaust) sibling: the WAL
        # is truncated (busy=0, log=0, remaining=0) and the row survives.
        result = store.checkpoint_wal()
        assert result == (0, 0, 0)

        conn = sqlite3.connect(str(exhaust_db))
        after = conn.execute(
            "SELECT content FROM memories WHERE key = ?",
            ("elevator/checkpoint/p1",),
        ).fetchone()
        conn.close()
        assert after is not None and after[0] == "v"
    finally:
        store.close()


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


def test_machinery_prefixes_cache_stale_without_reset(tmp_db, tmp_path, observe_log, no_enforce, monkeypatch):
    """Pins the documented latent footgun (reviewer PR #337 cycle 1
    [low]): the machinery prefix cache is NOT invalidated by a
    mid-process allowlist change. A caller that rotates the allowlist at
    runtime (here: pointing MEM_MACHINE_STATE_PREFIXES_PATH at a new
    file) silently keeps the STALE prefix set until it calls
    reset_machinery_prefixes_cache(). This test proves both halves:
    (1) a rotation WITHOUT the reset keeps the old prefixes, and
    (2) the reset makes the new prefixes take effect."""
    from agents_core import mem_exhaust

    # NOTE: both allowlists use a NON-STATIC prefix (pm/) — elevator/ and
    # weather/ are static EXHAUST_PREFIXES, so routing them would not
    # distinguish the machinery extension from the static list.
    # First allowlist: pm/ is a machinery prefix.
    allowlist_a = tmp_path / "allowlist-a.json"
    allowlist_a.write_text(json.dumps({"prefixes": [
        {"prefix": "pm/", "producer_principal": "brix-pm",
         "store": "machinery", "state": "live", "dead_since": None},
    ]}), encoding="utf-8")
    monkeypatch.setenv("MEM_MACHINE_STATE_PREFIXES_PATH", str(allowlist_a))
    mem_exhaust.reset_machinery_prefixes_cache()
    try:
        assert mem_exhaust.machinery_prefixes() == ("pm/",)

        # Rotate the allowlist at runtime WITHOUT resetting the cache.
        allowlist_b = tmp_path / "allowlist-b.json"
        allowlist_b.write_text(json.dumps({"prefixes": [
            {"prefix": "router/gw-review-divergence/", "producer_principal": "brix-pm",
             "store": "machinery", "state": "live", "dead_since": None},
        ]}), encoding="utf-8")
        # (router/gw-review-divergence/ is static too — the point is that the
        # NEW prefix is different from the cached one; the stale assertion
        # below uses pm/, which is in NEITHER the static list nor the new
        # allowlist.)
        monkeypatch.setenv("MEM_MACHINE_STATE_PREFIXES_PATH", str(allowlist_b))

        # STALE: the cache still serves the old prefix set (the footgun).
        assert mem_exhaust.machinery_prefixes() == ("pm/",)
        assert mem_exhaust.route_to_exhaust("pm/x") is True

        # The reset makes the new allowlist take effect.
        mem_exhaust.reset_machinery_prefixes_cache()
        assert mem_exhaust.machinery_prefixes() == ("router/gw-review-divergence/",)
        assert mem_exhaust.route_to_exhaust("pm/x") is False
    finally:
        mem_exhaust.reset_machinery_prefixes_cache()
        monkeypatch.delenv("MEM_MACHINE_STATE_PREFIXES_PATH", raising=False)


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
