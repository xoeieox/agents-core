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
async def test_facets_leg_stub(stub_facets, init_semaphore):
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
async def test_operator_requested_surfaced(stub_facets, init_semaphore):
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
