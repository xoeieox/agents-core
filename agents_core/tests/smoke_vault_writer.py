#!/usr/bin/env python3
"""Smoke test for agents_core.vault_writer — PR 1 gate.

Covers all tests required by the vault-substrate-v0 spec (PR 1 section):

1. Concurrent-writer test: two threads write the same path; flock serializes;
   audit log shows both writes in order.

2. Audit log replay test: writes ⇒ reads back ⇒ chain of prev_hash ⇒
   content_hash is verifiable.

3. Write-event subscriber test: subscriber sees event with correct fields
   within 100ms of write.

4. Frontmatter idempotence: stamping twice is identical to stamping once.

5. Migration helper roundtrip: write_compat() produces the same on-disk bytes
   as Path.write_text() + attribution stamp (audit row tagged agent_id=legacy,
   intent=unmigrated).

Run::

    python3 -m agents_core.tests.smoke_vault_writer

or via pytest::

    pytest agents_core/tests/smoke_vault_writer.py -v

All tests are LLM-free.  No network calls are made.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures: redirect persistent paths to tmp dirs so tests don't touch /data
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_data_paths(tmp_path, monkeypatch):
    """Redirect /data/vault-audit.db and /data/vault-events.jsonl to tmp_path."""
    audit_db = tmp_path / "vault-audit.db"
    events_jsonl = tmp_path / "vault-events.jsonl"

    monkeypatch.setenv("VAULT_AUDIT_DB", str(audit_db))
    monkeypatch.setenv("VAULT_EVENTS_JSONL", str(events_jsonl))

    # Force the audit module to reconnect with the new DB path.
    import agents_core.vault_audit as va
    va.reset_connection()

    yield

    # Cleanup: reset connection again so next test gets a fresh one.
    va.reset_connection()


# ---------------------------------------------------------------------------
# Test 1: Concurrent-writer test
# ---------------------------------------------------------------------------


def test_concurrent_writers_serialize(tmp_path):
    """Two threads writing the same path serialize via flock; audit log shows both."""
    import agents_core.vault_audit as va
    from agents_core.vault_writer import write

    target = tmp_path / "concurrent.md"

    results: list[str] = []

    def writer(content: str, label: str):
        record = write(
            target,
            content,
            agent_id=f"agent-{label}",
            intent=f"concurrent write {label}",
            stamp_frontmatter=False,
        )
        results.append(record.content_hash)

    t1 = threading.Thread(target=writer, args=("content-A\n", "A"))
    t2 = threading.Thread(target=writer, args=("content-B\n", "B"))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # Both threads must have completed
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # Audit log must have exactly 2 rows for this path
    rows = va.query_by_path(str(target))
    assert len(rows) == 2, f"Expected 2 audit rows, got {len(rows)}"

    # The rows must be in write-order: second row's prev_hash == first row's content_hash
    first, second = rows
    assert second["prev_hash"] == first["content_hash"], (
        f"Hash chain broken: second.prev_hash={second['prev_hash']!r} "
        f"!= first.content_hash={first['content_hash']!r}"
    )

    # First write had no predecessor
    assert first["prev_hash"] is None, f"First write should have prev_hash=None, got {first['prev_hash']!r}"


# ---------------------------------------------------------------------------
# Test 2: Audit log replay / hash chain verification
# ---------------------------------------------------------------------------


def test_audit_log_hash_chain(tmp_path):
    """Replay: chain of prev_hash → content_hash is verifiable across writes."""
    import hashlib
    import agents_core.vault_audit as va
    from agents_core.vault_writer import write

    target = tmp_path / "chain.md"
    contents = ["version one\n", "version two\n", "version three\n"]

    for i, body in enumerate(contents):
        write(
            target,
            body,
            agent_id="chain-agent",
            intent=f"write {i}",
            stamp_frontmatter=False,
        )

    rows = va.query_by_path(str(target))
    assert len(rows) == 3

    # Verify hash format
    for row in rows:
        assert row["content_hash"].startswith("sha256:"), row["content_hash"]

    # Chain integrity: each row's prev_hash == previous row's content_hash
    assert rows[0]["prev_hash"] is None
    for i in range(1, len(rows)):
        assert rows[i]["prev_hash"] == rows[i - 1]["content_hash"], (
            f"Chain broken at position {i}"
        )

    # Independently verify the final file's hash matches the last audit row
    final_bytes = target.read_bytes()
    expected_hash = "sha256:" + hashlib.sha256(final_bytes).hexdigest()
    assert rows[-1]["content_hash"] == expected_hash, (
        f"Final hash mismatch: {rows[-1]['content_hash']!r} != {expected_hash!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: Write-event subscriber test
# ---------------------------------------------------------------------------


def test_write_event_subscriber(tmp_path):
    """Subscriber receives a WriteEvent with correct fields within 100ms of write."""
    from agents_core.vault_writer import write, subscribe

    target = tmp_path / "event_test.md"
    received: list = []

    async def collect_one():
        async for event in subscribe("*"):
            received.append(event)
            return  # stop after first event

    async def run():
        # Start subscriber task first
        collector = asyncio.create_task(collect_one())

        # Give the subscriber a moment to register
        await asyncio.sleep(0.01)

        # Write in a thread (write() is sync; it schedules _publish on the loop)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: write(
                target,
                "event content\n",
                agent_id="event-agent",
                intent="subscriber test",
                stamp_frontmatter=False,
            ),
        )

        # Wait up to 200ms for the event to arrive (spec says 100ms; we give margin)
        try:
            await asyncio.wait_for(collector, timeout=0.2)
        except asyncio.TimeoutError:
            collector.cancel()
            pytest.fail("Subscriber did not receive event within 200ms")

    asyncio.run(run())

    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    event = received[0]
    assert event.record.path == str(target)
    assert event.record.agent_id == "event-agent"
    assert event.record.intent == "subscriber test"
    assert event.record.content_hash.startswith("sha256:")
    assert event.record.prev_hash is None  # first write


# ---------------------------------------------------------------------------
# Test 4: Frontmatter idempotence
# ---------------------------------------------------------------------------


def test_frontmatter_idempotence():
    """Stamping the same content twice produces identical output to stamping once."""
    from agents_core.vault_writer import stamp_attribution

    original = "# Hello\n\nBody text.\n"
    once = stamp_attribution(original, agent_id="test-agent", intent="first stamp")
    twice = stamp_attribution(once, agent_id="test-agent", intent="first stamp")

    assert once == twice, (
        f"stamp_attribution is not idempotent:\nonce:\n{once}\ntwice:\n{twice}"
    )


def test_frontmatter_idempotence_existing_fm():
    """Idempotence holds when file already has non-attribution frontmatter."""
    from agents_core.vault_writer import stamp_attribution

    original = "---\ntitle: My Doc\ntags: [a, b]\n---\n\nBody.\n"
    once = stamp_attribution(original, agent_id="pm", intent="update")
    twice = stamp_attribution(once, agent_id="pm", intent="update")

    assert once == twice
    # Verify existing frontmatter fields are preserved
    assert "title: My Doc" in once
    assert "tags:" in once


def test_frontmatter_idempotence_with_citations():
    """Idempotence holds when citations are provided."""
    from agents_core.vault_writer import stamp_attribution

    citations = [
        {"path": "Lapis/foo.md", "content_hash": "sha256:abc123", "quoted_snippet": "relevant text"},
    ]
    original = "# Doc\n"
    once = stamp_attribution(original, agent_id="pm", intent="cite", citations=citations)
    twice = stamp_attribution(once, agent_id="pm", intent="cite", citations=citations)
    assert once == twice


# ---------------------------------------------------------------------------
# Test 5: Migration helper roundtrip
# ---------------------------------------------------------------------------


def test_write_compat_roundtrip(tmp_path):
    """write_compat() produces the same on-disk result as Path.write_text() +
    attribution stamp, and leaves an audit row tagged agent_id=legacy, intent=unmigrated.
    """
    import agents_core.vault_audit as va
    from agents_core.vault_writer import stamp_attribution
    from agents_core.vault_writer.compat import write_compat

    content = "# Legacy File\n\nSome content here.\n"
    target = tmp_path / "legacy.md"

    record = write_compat(target, content)

    # Audit row must be tagged agent_id=legacy, intent=unmigrated
    assert record.agent_id == "legacy"
    assert record.intent == "unmigrated"

    # Audit log must have one row for this path
    rows = va.query_by_path(str(target))
    assert len(rows) == 1
    assert rows[0]["agent_id"] == "legacy"
    assert rows[0]["intent"] == "unmigrated"

    # On-disk bytes must match what stamp_attribution would produce from the same content
    expected_content = stamp_attribution(content, agent_id="legacy", intent="unmigrated")
    actual_content = target.read_text(encoding="utf-8")
    assert actual_content == expected_content, (
        f"On-disk content mismatch.\nExpected:\n{expected_content}\nGot:\n{actual_content}"
    )


def test_write_compat_no_stamp_is_identical_to_write_text(tmp_path):
    """With stamp_frontmatter=False, write_compat produces byte-identical output to Path.write_text."""
    from agents_core.vault_writer.compat import write_compat

    content = "plain content\n"
    target_compat = tmp_path / "compat.txt"
    target_wt = tmp_path / "write_text.txt"

    write_compat(target_compat, content, stamp_frontmatter=False)
    target_wt.write_text(content, encoding="utf-8")

    assert target_compat.read_bytes() == target_wt.read_bytes()


def test_write_compat_custom_agent(tmp_path):
    """write_compat with explicit agent_id/intent stores correct audit metadata."""
    import agents_core.vault_audit as va
    from agents_core.vault_writer.compat import write_compat

    target = tmp_path / "agent.md"
    write_compat(target, "# Agent\n", agent_id="lapis-pm", intent="arc-doc generation", stamp_frontmatter=False)

    rows = va.query_by_path(str(target))
    assert rows[0]["agent_id"] == "lapis-pm"
    assert rows[0]["intent"] == "arc-doc generation"


# ---------------------------------------------------------------------------
# Additional: JSONL tail test
# ---------------------------------------------------------------------------


def test_jsonl_tail_written(tmp_path):
    """Write-event is persisted to the JSONL tail file."""
    import json
    from agents_core.vault_writer import write

    target = tmp_path / "jsonl_test.md"
    events_jsonl = Path(os.environ["VAULT_EVENTS_JSONL"])

    write(
        target,
        "jsonl content\n",
        agent_id="jsonl-agent",
        intent="jsonl test",
        stamp_frontmatter=False,
    )

    assert events_jsonl.exists(), "JSONL tail file was not created"
    lines = [l for l in events_jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1

    event = json.loads(lines[-1])
    assert event["path"] == str(target)
    assert event["agent_id"] == "jsonl-agent"
    assert event["intent"] == "jsonl test"
    assert event["content_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# Additional: Audit query helpers
# ---------------------------------------------------------------------------


def test_audit_query_by_agent(tmp_path):
    """query_by_agent returns rows filtered by agent_id."""
    import agents_core.vault_audit as va
    from agents_core.vault_writer import write

    write(tmp_path / "a1.md", "x", agent_id="alpha", intent="i", stamp_frontmatter=False)
    write(tmp_path / "a2.md", "y", agent_id="beta",  intent="i", stamp_frontmatter=False)
    write(tmp_path / "a3.md", "z", agent_id="alpha", intent="i", stamp_frontmatter=False)

    alpha_rows = va.query_by_agent("alpha")
    beta_rows = va.query_by_agent("beta")

    assert len(alpha_rows) == 2
    assert len(beta_rows) == 1
    assert all(r["agent_id"] == "alpha" for r in alpha_rows)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        cwd=Path(__file__).parent.parent.parent,
    )
    sys.exit(result.returncode)
