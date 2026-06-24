"""Tests for agents_core.shared_deliberation service.

Fixtures stub Facets and Council via env vars, avoiding live GW/Council/Facets calls.
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from agents_core.shared_deliberation.envelope import DeliberationRequest, DeliberationEnvelope
from agents_core.shared_deliberation.orchestrator import (
    run_deliberation,
    init_facets_semaphore,
    GroundingHandoffError,
    _run_facets_subprocess,
)
from agents_core.shared_deliberation.client import SharedDeliberationClient
from agents_core.shared_deliberation.compat import derive_spec_review_brief_compat


@pytest.fixture
def stub_facets(monkeypatch):
    """Enable Facets stub via env var."""
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")


@pytest.fixture
def stub_council(monkeypatch):
    """Enable Council stub via env var."""
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")


@pytest.fixture
def init_semaphore():
    """Initialize Facets semaphore for tests."""
    init_facets_semaphore(2)
    yield


# Test 1: Facets leg via the service
@pytest.mark.asyncio
async def test_facets_leg_stub(stub_facets, stub_council, init_semaphore):
    """Submit a deliberation request, assert structured envelope returns with Facets result."""
    request = DeliberationRequest(
        text="Review this spec for scope and fit.",
        context={"source": "test", "repo": "test-repo"},
    )
    envelope = await run_deliberation(request)

    assert envelope.deliberation_request_id
    assert envelope.facets_ok is True
    assert envelope.facets is not None
    assert envelope.facets.get("stub") is True
    assert envelope.triage == "full"


# Test 2: operator_requested surfaced
@pytest.mark.asyncio
async def test_operator_requested_surfaced(stub_facets, stub_council, init_semaphore):
    """With GW unavailable (stub), assert operator_requested is surfaced."""
    request = DeliberationRequest(
        text="Deliberate on this.",
        context={},
        facets_operator="gravitywell",
    )
    envelope = await run_deliberation(request)

    assert envelope.facets_ok is True
    # In stub mode, operator_requested should be set from the request
    assert envelope.operator_requested is not None


# Test 3: Triage branch
@pytest.mark.asyncio
async def test_triage_lightweight_skips_council(stub_facets, stub_council, init_semaphore):
    """With triage=lightweight, Council is NOT submitted."""
    request = DeliberationRequest(
        text="Quick check",
        context={},
        triage="lightweight",
    )
    envelope = await run_deliberation(request)

    assert envelope.triage == "lightweight"
    assert envelope.council_run_id is None
    assert envelope.council_ok is False  # Not attempted


@pytest.mark.asyncio
async def test_triage_full_runs_both(stub_facets, stub_council, init_semaphore):
    """With triage=full, both Facets and Council run."""
    request = DeliberationRequest(
        text="Full review",
        context={},
        triage="full",
    )
    envelope = await run_deliberation(request)

    assert envelope.triage == "full"
    assert envelope.facets_ok is True
    assert envelope.council_ok is True
    assert envelope.council_run_id is not None


# Test 4: Typed seam present
@pytest.mark.asyncio
async def test_seam_field_present(stub_facets, stub_council, init_semaphore):
    """Request/envelope carry seam field; orchestrator iterates extra_modes without error."""
    request = DeliberationRequest(
        text="With seam",
        context={},
        seam={"mode": "test-mode"},
    )
    envelope = await run_deliberation(request)

    assert envelope.extra_modes == []  # v0: empty
    assert envelope.deliberation_request_id


# Test 5: Three callers (spec-review-shaped, Gardener stub, Sessions stub)
@pytest.mark.asyncio
async def test_spec_review_shaped_caller(stub_facets, stub_council, init_semaphore):
    """spec-review-shaped payload hits the same interface, derives SpecReviewBrief."""
    request = DeliberationRequest(
        text="Review this spec for ecosystem fit.",
        context={
            "source": "pre-bind-wire",
            "spec_path": "/srv/lapis/planning/specs/test-target.md",
            "repo": "test-repo",
            "authority": "advisory",
        },
        caller="spec-review",
        triage="full",
    )
    envelope = await run_deliberation(request)

    # Verify envelope structure
    assert envelope.facets_ok is True
    assert envelope.council_ok is True
    assert envelope.council_positions is not None

    # Derive SpecReviewBrief-equivalent
    brief = derive_spec_review_brief_compat(
        envelope,
        spec_path=Path("/srv/lapis/planning/specs/test-target.md"),
        target_id="test-target",
        repo="test-repo",
        elapsed_s=0.5,
    )
    assert brief.council_status in ("resolved", "open", "laid-down", "failed", "timeout", "error", "closed")
    assert brief.council_positions == []  # Stub returns empty


@pytest.mark.asyncio
async def test_gardener_shaped_caller(stub_facets, stub_council, init_semaphore):
    """Gardener-shaped payload (recovered item + weaver-digest) hits the same interface."""
    request = DeliberationRequest(
        text="Recovered compost item for routing.",
        context={
            "source": "weaver-digest",
            "recovery_reason": "conflict-signal",
        },
        caller="gardener",
        triage="full",
    )
    envelope = await run_deliberation(request)

    assert envelope.deliberation_request_id
    assert envelope.facets_ok is True
    assert envelope.council_ok is True


@pytest.mark.asyncio
async def test_sessions_shaped_caller(stub_facets, stub_council, init_semaphore):
    """Sessions-shaped payload (selected text + session context) hits the same interface."""
    request = DeliberationRequest(
        text="Selected text from the session.",
        context={
            "source": "session-selection",
            "session_id": "session-123",
            "user_id": "user-456",
        },
        caller="sessions",
        triage="lightweight",  # Sessions may use lightweight
    )
    envelope = await run_deliberation(request)

    assert envelope.deliberation_request_id
    assert envelope.facets_ok is True
    assert envelope.council_ok is False  # Not attempted due to lightweight


# Test 6: Partial-failure semantics
@pytest.mark.asyncio
async def test_facets_failure_council_success(monkeypatch, stub_council, init_semaphore):
    """If Facets fails but Council succeeds, HTTP 200 with partial envelope."""
    monkeypatch.setenv("FACETS_DISPATCH_DISABLED", "1")  # Disable Facets

    request = DeliberationRequest(
        text="Test",
        context={},
        triage="full",
    )
    envelope = await run_deliberation(request)

    # Partial failure: return both legs' state
    assert envelope.facets_ok is False
    assert envelope.council_ok is True  # Stub succeeds
    assert "facets" in envelope.errors or envelope.facets is None
    # No 5xx error; caller must check flags


@pytest.mark.asyncio
async def test_both_legs_fail(monkeypatch, init_semaphore):
    """If both legs fail, HTTP service returns 500."""
    from fastapi.testclient import TestClient
    from agents_core.shared_deliberation.service import create_app
    from agents_core.shared_deliberation import orchestrator

    monkeypatch.setenv("FACETS_DISPATCH_DISABLED", "1")
    # Monkeypatch _submit_council to return None (council leg fails)
    monkeypatch.setattr(orchestrator, "_submit_council", lambda text, voicing: None)

    app = create_app()
    client = TestClient(app)

    request_data = {
        "text": "Test",
        "context": {},
        "triage": "full",
    }
    response = client.post("/v0/deliberate", json=request_data)

    # Both failed -> HTTP 500
    assert response.status_code == 500


# Test 7: Client integration
@pytest.mark.asyncio
async def test_client_ship_and_return(stub_facets, stub_council, init_semaphore):
    """SharedDeliberationClient sends request, receives envelope."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agents_core.shared_deliberation.service import create_app

    app = create_app()
    client = TestClient(app)

    request_data = {
        "text": "Test deliberation",
        "context": {"test": True},
        "triage": "full",
        "caller": "test",
    }
    response = client.post("/v0/deliberate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["deliberation_request_id"]
    assert data["facets_ok"] is True
    assert data["triage"] == "full"


@pytest.mark.asyncio
async def test_client_health_check(stub_facets, init_semaphore):
    """Health check endpoint works."""
    from fastapi.testclient import TestClient
    from agents_core.shared_deliberation.service import create_app

    app = create_app()
    client = TestClient(app)

    response = client.get("/v0/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- H5b grounding passthrough tests (AC1-AC4) ---

# AC1: default parity — grounding_result_file unset, argv unchanged
def test_ac1_default_parity_no_grounding_field(tmp_path):
    """AC1: grounding_result_file=None produces identical argv (no --grounding-result-file)."""
    from pathlib import Path
    from unittest.mock import patch, MagicMock

    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    context_file_holder = []

    def fake_run(argv, **kwargs):
        context_file_holder.append(argv)
        m = MagicMock()
        m.returncode = 0
        import json
        m.stdout = json.dumps({"deliberation_id": "test-id"})
        m.stderr = ""
        return m

    with patch("subprocess.run", side_effect=fake_run):
        _run_facets_subprocess("text", {}, "gravitywell", facets_repo, grounding_result_file=None)

    argv = context_file_holder[0]
    assert "--grounding-result-file" not in argv
    assert "--no-auto-ground" not in argv


# AC2: injection — set + exists -> argv gains --grounding-result-file + --no-auto-ground
def test_ac2_injection_set_and_exists(tmp_path):
    """AC2: grounding_result_file pointing to an existing file injects argv flags."""
    from unittest.mock import patch, MagicMock

    grounding_file = tmp_path / "grounding.json"
    grounding_file.write_text('{"result": "ok"}')
    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    captured = []

    def fake_run(argv, **kwargs):
        captured.append(argv)
        m = MagicMock()
        m.returncode = 0
        import json
        m.stdout = json.dumps({"deliberation_id": "test-id"})
        m.stderr = ""
        return m

    with patch("subprocess.run", side_effect=fake_run):
        _run_facets_subprocess(
            "text", {}, "gravitywell", facets_repo,
            grounding_result_file=str(grounding_file),
        )

    argv = captured[0]
    assert "--grounding-result-file" in argv
    assert str(grounding_file) in argv
    assert "--no-auto-ground" in argv
    # flags must be adjacent
    idx = argv.index("--grounding-result-file")
    assert argv[idx + 1] == str(grounding_file)


# AC3: set-but-missing -> GroundingHandoffError, no subprocess spawned
def test_ac3_set_but_missing_raises_handoff_error(tmp_path):
    """AC3: grounding_result_file set to non-existent path raises GroundingHandoffError."""
    from unittest.mock import patch

    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    missing_path = str(tmp_path / "does_not_exist.json")

    with patch("subprocess.run") as mock_run:
        with pytest.raises(GroundingHandoffError):
            _run_facets_subprocess(
                "text", {}, "gravitywell", facets_repo,
                grounding_result_file=missing_path,
            )
        mock_run.assert_not_called()


def test_ac3_set_but_empty_raises_handoff_error(tmp_path):
    """AC3: grounding_result_file set to an empty file raises GroundingHandoffError."""
    from unittest.mock import patch

    facets_repo = tmp_path / "facets"
    facets_repo.mkdir()
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("")

    with patch("subprocess.run") as mock_run:
        with pytest.raises(GroundingHandoffError):
            _run_facets_subprocess(
                "text", {}, "gravitywell", facets_repo,
                grounding_result_file=str(empty_file),
            )
        mock_run.assert_not_called()


# AC4: serialization — grounding_result_file round-trips via to_dict()
def test_ac4_serialization_round_trip():
    """AC4: grounding_result_file is present in to_dict() output."""
    req = DeliberationRequest(
        text="test",
        context={},
        grounding_result_file="/tmp/grounding.json",
    )
    d = req.to_dict()
    assert "grounding_result_file" in d
    assert d["grounding_result_file"] == "/tmp/grounding.json"


def test_ac4_serialization_none_default():
    """AC4: grounding_result_file defaults to None and round-trips as None."""
    req = DeliberationRequest(text="test", context={})
    d = req.to_dict()
    assert "grounding_result_file" in d
    assert d["grounding_result_file"] is None


# --- GW span hold tests (shared-deliberation-gate-spanning-keepawake-v0) ---

def _make_mock_doorman():
    """Return a mock DoormanClient with acquire returning serving."""
    m = Mock()
    m.acquire = Mock(return_value={"status": "serving"})
    m.release = Mock()
    m.close = Mock()
    return m


@pytest.mark.asyncio
async def test_span_hold_acquire_before_facets_and_release_after(
    monkeypatch, init_semaphore
):
    """GW span hold: acquire fires before Facets dispatch; work_id spans the call; released in finally."""
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")

    import agents_core.shared_deliberation.orchestrator as orch

    call_order = []

    orig_facets = orch._facets_subprocess
    async def tracked_facets(*a, **kw):
        call_order.append("facets")
        return await orig_facets(*a, **kw)
    monkeypatch.setattr(orch, "_facets_subprocess", tracked_facets)

    mock_doorman = _make_mock_doorman()
    def tracked_acquire(*a, **kw):
        call_order.append("acquire")
        return {"status": "serving"}
    mock_doorman.acquire = tracked_acquire

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_doorman), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={}, council_voicing="gravitywell"
        )
        await run_deliberation(request)

    assert "acquire" in call_order
    assert "facets" in call_order
    assert call_order.index("acquire") < call_order.index("facets"), (
        f"acquire must precede facets dispatch; got order: {call_order}"
    )
    mock_doorman.release.assert_called_once()
    release_args = mock_doorman.release.call_args
    work_id_used = release_args.args[1] if release_args.args else release_args.kwargs.get("work_id", "")
    assert work_id_used.startswith("shared-delib-"), (
        f"span hold work_id must start with 'shared-delib-', got: {work_id_used!r}"
    )


@pytest.mark.asyncio
async def test_span_hold_released_on_facets_error(monkeypatch, init_semaphore):
    """Span hold is released in finally even when the Facets leg errors."""
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")

    import agents_core.shared_deliberation.orchestrator as orch

    async def failing_facets(*a, **kw):
        raise RuntimeError("facets exploded")
    monkeypatch.setattr(orch, "_facets_subprocess", failing_facets)

    mock_doorman = _make_mock_doorman()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_doorman), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={}, council_voicing="gravitywell"
        )
        with pytest.raises(RuntimeError, match="facets exploded"):
            await run_deliberation(request)

    mock_doorman.release.assert_called_once()


@pytest.mark.asyncio
async def test_span_hold_released_on_council_error(monkeypatch, init_semaphore):
    """Span hold is released in finally even when the Council leg errors."""
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")

    import agents_core.shared_deliberation.orchestrator as orch

    async def failing_council(*a, **kw):
        raise RuntimeError("council exploded")
    monkeypatch.setattr(orch, "_council_subprocess", failing_council)

    mock_doorman = _make_mock_doorman()
    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_doorman), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={}, council_voicing="gravitywell"
        )
        with pytest.raises(RuntimeError, match="council exploded"):
            await run_deliberation(request)

    mock_doorman.release.assert_called_once()


@pytest.mark.asyncio
async def test_council_worker_death_no_span_hold_drop(monkeypatch, init_semaphore):
    """Simulated council-worker death (heartbeat stale) does NOT release the orchestrator span hold.

    The span hold is the orchestrator's own lease; it stays active until run_deliberation
    returns, independent of the council worker's own hold. Specifically: when council_subprocess
    returns early with a heartbeat_stale error, the span hold is not released mid-flight.
    """
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")

    import agents_core.shared_deliberation.orchestrator as orch

    release_count = [0]
    span_released_during_council = [False]

    mock_doorman = _make_mock_doorman()

    def tracked_release(*a, **kw):
        release_count[0] += 1

    mock_doorman.release = tracked_release

    # Simulate worker death: council checks that release has NOT been called yet,
    # then returns an error (like heartbeat_stale)
    async def dead_worker_council(*a, **kw):
        span_released_during_council[0] = release_count[0] > 0
        return (False, "council-run-dead", None, "council worker died/stalled (heartbeat_stale)")
    monkeypatch.setattr(orch, "_council_subprocess", dead_worker_council)

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_doorman), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={}, council_voicing="gravitywell"
        )
        envelope = await run_deliberation(request)

    # council_ok=False (worker died) but span hold release happens exactly once in finally
    assert envelope.council_ok is False
    assert envelope.council_run_id == "council-run-dead"
    assert release_count[0] == 1, (
        f"Expected exactly one release call (in finally), got {release_count[0]}"
    )
    assert not span_released_during_council[0], (
        "Span hold must not be released while council_subprocess is still executing"
    )


@pytest.mark.asyncio
async def test_non_gw_voicing_no_span_hold(monkeypatch, init_semaphore):
    """Non-GW voicing (local/sonnet) does NOT acquire a GW span hold."""
    monkeypatch.setenv("SHARED_DELIBERATION_FACETS_STUB", "1")
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")

    mock_doorman = _make_mock_doorman()
    mock_class = Mock(return_value=mock_doorman)

    with patch("agents_core.doorman_client.DoormanClient", mock_class), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={},
            council_voicing="sonnet",
            facets_operator="haiku",
        )
        await run_deliberation(request)

    mock_class.assert_not_called()
    mock_doorman.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_span_hold_refresh_fires(monkeypatch, init_semaphore):
    """Span hold refresh thread fires at least once during a long deliberation."""
    monkeypatch.setenv("SHARED_DELIBERATION_COUNCIL_STUB", "1")
    monkeypatch.setenv("SHARED_DELIB_SPAN_REFRESH_S", "1")

    import agents_core.shared_deliberation.orchestrator as orch

    async def slow_facets(*a, **kw):
        await asyncio.sleep(1.5)
        return (True, {"stub": True, "methodology": {"synthesis_operator": "gravitywell"}}, "stub-id", None)
    monkeypatch.setattr(orch, "_facets_subprocess", slow_facets)

    acquire_count = [0]
    mock_doorman = Mock()
    def count_acquire(*a, **kw):
        acquire_count[0] += 1
        return {"status": "serving"}
    mock_doorman.acquire = count_acquire
    mock_doorman.release = Mock()
    mock_doorman.close = Mock()

    with patch("agents_core.doorman_client.DoormanClient", return_value=mock_doorman), \
         patch("agents_core.doorman_client._gw_acquire_timeout", return_value=10.0):
        request = DeliberationRequest(
            text="test", context={}, council_voicing="gravitywell"
        )
        await run_deliberation(request)

    assert acquire_count[0] >= 2, (
        f"Expected initial acquire + at least one refresh, got {acquire_count[0]} acquire calls"
    )
