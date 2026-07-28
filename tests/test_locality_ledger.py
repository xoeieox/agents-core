"""Tests for the locality ledger (agents-core-locality-ledger-v0, leg 1).

Covers: agents_core.locality (record/summarize/is_ledger_healthy/rotation) and the
three write-point chokepoints in llm.py (_ret in call_claude_cli, call_operator) and
gw_agent.py (call_gw_agent).

conftest.py's autouse _locality_ledger_isolated fixture already points
LOCALITY_LEDGER_ROOT at a per-test tmp_path, so tests here that just want an
isolated root can rely on it; tests that need to *read* what was written still
resolve the root explicitly via agents_core.locality.root().
"""
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest

from agents_core import locality


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_all_entries(root_path):
    entries = []
    for path in sorted(root_path.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _make_gw_sse_resp_with_model(content_text, model_name, captured_dict=None):
    """Same shape as test_llm_call_operator.py's _make_gw_sse_resp, but the
    streamed chunk also echoes a "model" field (the response-echo path that
    feeds _served_model_out)."""
    def fake_post(url, json=None, timeout=None, stream=None):
        if captured_dict is not None:
            captured_dict["payload"] = json

        def lines():
            chunk = {
                "model": model_name,
                "choices": [{"delta": {"content": content_text}, "finish_reason": None}],
            }
            yield f"data: {__import__('json').dumps(chunk)}"
            yield "data: [DONE]"

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=lines())
        resp.close = MagicMock()
        return resp

    return fake_post


def _gw_mock_client(status="serving"):
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    resp = {"status": status, "node": "gravitywell", "work_id": "w1"}
    if status == "serving":
        resp["drain_cleared"] = True
    mock.acquire.return_value = resp
    return mock


def _gw_dc(status="serving"):
    instance = _gw_mock_client(status)
    dc = MagicMock(return_value=instance)
    dc.is_deferred = lambda resp: resp.get("status") == "deferred"
    dc.is_contended = lambda resp: bool(resp.get("contended"))
    return dc, instance


# ---------------------------------------------------------------------------
# locality.record() / summarize() / is_ledger_healthy() — unit level
# ---------------------------------------------------------------------------

def test_record_writes_one_jsonl_line_with_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))

    locality.record(
        requested_operator="qwen",
        served_model="qwen3.6-35b-a3b",
        host="http://203.0.113.12:8081",
        cost_class="local-sh",
        seam="call_operator",
        duration_ms=12.5,
    )

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["seam"] == "call_operator"
    assert entry["requested_operator"] == "qwen"
    assert entry["served_model"] == "qwen3.6-35b-a3b"
    assert entry["cost_class"] == "local-sh"
    assert entry["fallback_fired"] is False
    assert entry["fallback_reason"] is None
    assert entry["ok"] is True
    assert "ts" in entry


def test_record_never_raises_and_does_not_alter_caller_result(tmp_path, monkeypatch):
    """DoD 6: an unwritable root degrades to a WARNING, never an exception, and
    never changes the caller's return value. A regular FILE at the ledger root
    path makes every mkdir()/open() underneath it fail with a portable, non-root
    OSError regardless of filesystem permissions."""
    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("i am a file, not a directory")
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(bad_root))

    # record() itself must swallow the failure silently.
    locality.record(
        requested_operator="qwen",
        served_model="qwen3.6-35b-a3b",
        host="http://x",
        cost_class="local-sh",
        seam="call_operator",
    )  # must not raise

    # And a real call_operator() call routed through the same broken root must
    # still return its normal result.
    from agents_core.llm import call_operator

    with patch("agents_core.llm._call_qwen_backend", return_value="ok"):
        result = call_operator("qwen", prompt="hi")
    assert result == "ok"


def test_cost_class_outside_enum_is_normalized_to_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    locality.record(
        requested_operator="mystery",
        served_model=None,
        host=None,
        cost_class="totally-not-a-real-class",
        seam="call_operator",
    )
    entries = _read_all_entries(tmp_path)
    assert entries[0]["cost_class"] == "unknown"


def test_summarize_seeded_fixture_pct_local_and_fallback_breakdown(tmp_path, monkeypatch):
    """DoD 5: summarize() over a seeded fixture returns a correct pct_local and a
    correct by_fallback_reason breakdown."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))

    # 3 local, 1 paid-anthropic straight, 1 paid-anthropic via GW fallback.
    locality.record(requested_operator="qwen", served_model="qwen3.6-35b-a3b",
                     host="h", cost_class="local-sh", seam="call_operator")
    locality.record(requested_operator="gravitywell", served_model="gravitywell-122b",
                     host="h", cost_class="local-gw", seam="call_operator")
    locality.record(requested_operator="gravitywell", served_model="gravitywell-122b",
                     host="h", cost_class="local-gw", seam="call_gw_agent")
    locality.record(requested_operator="haiku", served_model="claude-haiku-4-5-20251001",
                     host="claude-cli", cost_class="paid-anthropic", seam="call_claude_cli",
                     cost_usd=0.001)
    locality.record(requested_operator="gravitywell", served_model="gravitywell-122b",
                     host="h", cost_class="local-gw", seam="call_operator",
                     fallback_fired=True, fallback_reason="gw_not_serving")
    locality.record(requested_operator="sonnet", served_model="claude-sonnet-4-6",
                     host="claude-cli", cost_class="paid-anthropic", seam="call_operator",
                     cost_usd=0.02)

    result = locality.summarize()

    assert result["total"] == 6
    # 4 of 6 entries are local-{gw,sh}.
    assert result["pct_local"] == pytest.approx(4 / 6 * 100.0)
    assert result["by_cost_class"]["local-gw"] == 3
    assert result["by_cost_class"]["local-sh"] == 1
    assert result["by_cost_class"]["paid-anthropic"] == 2
    assert result["fallback_count"] == 1
    assert result["by_fallback_reason"] == {"gw_not_serving": 1}
    assert result["paid_cost_usd"] == pytest.approx(0.021)


def test_summarize_since_until_window_filters_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    locality.record(requested_operator="qwen", served_model="qwen3.6-35b-a3b",
                     host="h", cost_class="local-sh", seam="call_operator")

    now = datetime.now(timezone.utc)
    future_result = locality.summarize(since=now + timedelta(hours=1))
    assert future_result["total"] == 0

    past_result = locality.summarize(since=now - timedelta(hours=1))
    assert past_result["total"] == 1


def test_is_ledger_healthy_false_when_no_records(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    healthy, reason = locality.is_ledger_healthy()
    assert healthy is False
    assert reason


def test_is_ledger_healthy_true_immediately_after_a_record(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    locality.record(requested_operator="qwen", served_model="qwen3.6-35b-a3b",
                     host="h", cost_class="local-sh", seam="call_operator")
    healthy, reason = locality.is_ledger_healthy(max_silence_hours=24.0)
    assert healthy is True


def test_is_ledger_healthy_false_when_stale(tmp_path, monkeypatch):
    """An empty/stale ledger must never read as a good (all-local) week — silence
    must present as friction, per the gate-required health check."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    )[:-3] + "+00:00"
    stale_entry = {
        "ts": stale_ts, "seam": "call_operator", "requested_operator": "qwen",
        "served_model": "qwen3.6-35b-a3b", "host": "h", "cost_class": "local-sh",
        "fallback_fired": False, "fallback_reason": None, "cost_usd": None,
        "duration_ms": None, "ok": True,
    }
    day_file = tmp_path / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    day_file.parent.mkdir(parents=True, exist_ok=True)
    day_file.write_text(json.dumps(stale_entry) + "\n", encoding="utf-8")

    healthy, reason = locality.is_ledger_healthy(max_silence_hours=24.0)
    assert healthy is False
    assert "24" in reason or "silence" in reason.lower() or "no ledger record" in reason


def test_rotation_creates_numbered_archive_and_keeps_appending(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCALITY_LEDGER_MAX_BYTES", "100")

    for i in range(10):
        locality.record(requested_operator="qwen", served_model="qwen3.6-35b-a3b",
                         host="h", cost_class="local-sh", seam="call_operator",
                         extra={"i": i})

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    active = tmp_path / f"{day}.jsonl"
    assert active.exists()
    rotated = list(tmp_path.glob(f"{day}.jsonl.*"))
    assert len(rotated) > 0, "expected at least one rotated archive file"


def test_room_paths_locality_key_registered(monkeypatch):
    """Registered under its own env override, independent of the
    LOCALITY_LEDGER_ROOT the autouse test-isolation fixture also sets."""
    monkeypatch.delenv("LOCALITY_LEDGER_ROOT", raising=False)
    from agents_core.room_paths import room_path

    p = room_path("locality")
    assert p.name == "locality"


def test_provenance_comment_block_documents_stream_culled_and_fallback():
    """DoD 9: the llm.py:101-124 vocabulary comment block must list the two
    reasons that were already appended in code but missing from the docs."""
    import agents_core.llm as llm_mod
    import inspect

    source = inspect.getsource(llm_mod)
    comment_start = source.index("GW admission provenance vocabulary")
    comment_end = source.index("GW_PROVENANCE_PRECEDENCE orders")
    comment_block = source[comment_start:comment_end]
    assert "stream_culled" in comment_block
    assert "fallback " in comment_block or "fallback\n" in comment_block or "fallback  " in comment_block


# ---------------------------------------------------------------------------
# chokepoint B — call_operator()
# ---------------------------------------------------------------------------

def test_call_operator_qwen_writes_one_local_sh_record(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_operator

    with patch("agents_core.llm._call_qwen_backend", return_value="ok"):
        result = call_operator("qwen", prompt="hi")
    assert result == "ok"

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["seam"] == "call_operator"
    assert entries[0]["requested_operator"] == "qwen"
    assert entries[0]["cost_class"] == "local-sh"
    assert entries[0]["served_model"] == "qwen3.6-35b-a3b"
    assert entries[0]["ok"] is True


def test_call_operator_anthropic_path_writes_paid_anthropic_record(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_operator

    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="claude says hi"):
        result = call_operator("haiku", prompt="hi")
    assert result == "claude says hi"

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["seam"] == "call_operator"
    assert entries[0]["requested_operator"] == "haiku"
    assert entries[0]["cost_class"] == "paid-anthropic"


def test_call_operator_gravitywell_ledger_uses_observed_served_model_not_static_default(
    tmp_path, monkeypatch
):
    """DoD 3: served_model in the ledger must come from the response echo
    (_served_model_out plumbed through the gw_kwargs allowlist), not a static
    OPERATOR_DEFAULTS guess. Proven by resolving to a non-default model
    (GW_MODEL=gravitywell-27b, never OPERATOR_DEFAULTS["gravitywell"]) and
    asserting the ledger reflects exactly that, not "gravitywell-122b"."""
    from agents_core import llm as llm_mod
    from agents_core.llm import call_operator, OPERATOR_DEFAULTS

    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    monkeypatch.setenv("GW_MODEL", "gravitywell-27b")
    assert "gravitywell-27b" != OPERATOR_DEFAULTS["gravitywell"]

    with llm_mod._gw_handshake_lock:
        llm_mod._gw_handshake_cache[(llm_mod.GW_URL, "gravitywell-27b")] = True

    dc, _mock_client = _gw_dc(status="serving")
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("requests.post", side_effect=_make_gw_sse_resp_with_model("hi", "gravitywell-27b")):
        result = call_operator("gravitywell", prompt="test")

    assert result == "hi"
    entries = _read_all_entries(tmp_path)
    gw_entries = [e for e in entries if e["requested_operator"] == "gravitywell"]
    assert len(gw_entries) == 1
    assert gw_entries[0]["served_model"] == "gravitywell-27b"
    assert gw_entries[0]["cost_class"] == "local-gw"


def test_call_operator_gw_wake_fail_fallback_records_fallback_fired_and_reason(
    tmp_path, monkeypatch
):
    """DoD 4: a GW->paid fallback via _apply_wake_fail records fallback_fired=True
    with a fallback_reason drawn from the llm.py provenance vocabulary. Two
    records are expected — the failed gravitywell attempt and the paid haiku
    fallback that answered it — both seam="call_operator", per the module's
    documented double-counting-is-attributable design."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_operator

    dc, _mock_client = _gw_dc(status="deferred")  # not-serving -> triggers wake_fail
    with patch("agents_core.doorman_client.DoormanClient", dc), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="haiku says hi"):
        with pytest.warns(RuntimeWarning):
            result = call_operator("gravitywell", prompt="test", on_wake_fail="haiku")

    assert result == "haiku says hi"

    entries = _read_all_entries(tmp_path)
    gw_entry = next(e for e in entries if e["requested_operator"] == "gravitywell")
    haiku_entry = next(e for e in entries if e["requested_operator"] == "haiku")

    assert gw_entry["fallback_fired"] is True
    assert gw_entry["fallback_reason"] == "gw_deferred_swarm"
    assert haiku_entry["cost_class"] == "paid-anthropic"
    assert haiku_entry["fallback_fired"] is False


# ---------------------------------------------------------------------------
# chokepoint A — call_claude_cli() / _ret()
# ---------------------------------------------------------------------------

def _fake_completed_process(stdout_obj, returncode=0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = json.dumps(stdout_obj)
    proc.stderr = ""
    return proc


def test_call_claude_cli_records_cost_usd_when_envelope_carries_it(tmp_path, monkeypatch):
    """DoD 7: cost_usd populated when the envelope carries total_cost_usd."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_claude_cli

    envelope = {"result": "hello", "total_cost_usd": 0.0034, "duration_ms": 812}
    with patch("subprocess.run", return_value=_fake_completed_process(envelope)):
        result = call_claude_cli("hi", model="haiku")
    assert result == "hello"

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["seam"] == "call_claude_cli"
    assert entries[0]["cost_class"] == "paid-anthropic"
    assert entries[0]["cost_usd"] == pytest.approx(0.0034)
    assert entries[0]["duration_ms"] == pytest.approx(812)


def test_call_claude_cli_cost_usd_none_when_envelope_lacks_it(tmp_path, monkeypatch):
    """DoD 7 (converse): no branch depends on cost_usd's presence — its absence
    just yields None, not an error or a skipped record."""
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_claude_cli

    envelope = {"result": "hello"}
    with patch("subprocess.run", return_value=_fake_completed_process(envelope)):
        result = call_claude_cli("hi", model="sonnet")
    assert result == "hello"

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["cost_usd"] is None
    assert entries[0]["ok"] is True


def test_call_claude_cli_failure_still_records_ok_false(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core.llm import call_claude_cli

    with patch("subprocess.run", return_value=_fake_completed_process({}, returncode=1)):
        result = call_claude_cli("hi", model="haiku")
    assert result is None

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["ok"] is False
    assert entries[0]["cost_class"] == "paid-anthropic"


# ---------------------------------------------------------------------------
# chokepoint C — call_gw_agent()
# ---------------------------------------------------------------------------

def test_call_gw_agent_writes_one_record_tagged_by_seam(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core import gw_agent as gw_agent_mod

    with patch.object(gw_agent_mod, "_call_gw_agent_impl", return_value="answer") as mock_impl:
        result = gw_agent_mod.call_gw_agent(prompt="hi")

    assert result == "answer"
    mock_impl.assert_called_once()

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["seam"] == "call_gw_agent"
    assert entries[0]["requested_operator"] == "gravitywell"
    assert entries[0]["cost_class"] == "local-gw"
    assert entries[0]["ok"] is True


def test_call_gw_agent_exception_still_writes_record_then_reraises(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core import gw_agent as gw_agent_mod

    with patch.object(gw_agent_mod, "_call_gw_agent_impl", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            gw_agent_mod.call_gw_agent(prompt="hi")

    entries = _read_all_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["ok"] is False


def test_call_gw_agent_served_model_out_prefers_last_echoed_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALITY_LEDGER_ROOT", str(tmp_path))
    from agents_core import gw_agent as gw_agent_mod

    def fake_impl(*args, **kwargs):
        served_out = kwargs.get("served_model_out")
        if served_out is not None:
            served_out.append("gravitywell-122b")
            served_out.append("gravitywell-27b")
        return "answer"

    with patch.object(gw_agent_mod, "_call_gw_agent_impl", side_effect=fake_impl):
        result = gw_agent_mod.call_gw_agent(prompt="hi")

    assert result == "answer"
    entries = _read_all_entries(tmp_path)
    assert entries[0]["served_model"] == "gravitywell-27b"
