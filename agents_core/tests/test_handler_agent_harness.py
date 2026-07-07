"""Tests for agents_core.calibration.handler_agent_harness.

All tests are offline: httpx calls are mocked with respx, no live
GravityWell/AgentWorld/StarHouse dependency (see
agents-core-handler-agent-harness-promotion-v0 §4 — a live run is out of
scope for this unit).
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from agents_core.calibration.handler_agent_harness import (
    _build_parser,
    _extract_json,
    ask_json,
    run_smoke,
)

GW_URL = "http://gw.example/v1"
SH_URL = "http://sh.example/v1"


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_clean():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_markdown():
    raw = 'Sure, here you go:\n```json\n{"a": 1}\n```\n'
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_think_stripped():
    raw = '<think>let me reason about this...</think>\n{"a": 1}'
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_unparseable_returns_none():
    assert _extract_json("sorry I cannot answer") is None


def test_ask_json_retries_then_raises(respx_mock):
    respx_mock.post(f"{SH_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]},
        )
    )
    with pytest.raises(RuntimeError, match="LABEL: could not parse JSON after 2 attempts"):
        ask_json(SH_URL, "handler-agent", "system", "user", temp=0.2, label="LABEL")


# ---------------------------------------------------------------------------
# argparse defaults / env var precedence (§1f)
# ---------------------------------------------------------------------------

def test_gw_agentworld_url_default_when_unset(monkeypatch):
    monkeypatch.delenv("GW_AGENTWORLD_URL", raising=False)
    args = _build_parser().parse_args([])
    assert args.gw_agentworld_url == "http://203.0.113.11:8090/v1"


def test_gw_agentworld_url_from_env(monkeypatch):
    monkeypatch.setenv("GW_AGENTWORLD_URL", "http://env-gw.example/v1")
    args = _build_parser().parse_args([])
    assert args.gw_agentworld_url == "http://env-gw.example/v1"


def test_gw_agentworld_url_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("GW_AGENTWORLD_URL", "http://env-gw.example/v1")
    args = _build_parser().parse_args(["--gw-agentworld-url", "http://flag-gw.example/v1"])
    assert args.gw_agentworld_url == "http://flag-gw.example/v1"


def test_sh_pair_url_default_when_unset(monkeypatch):
    monkeypatch.delenv("SH_PAIR_URL", raising=False)
    args = _build_parser().parse_args([])
    assert args.sh_pair_url == "http://203.0.113.12:8082/v1"


def test_sh_pair_url_from_env(monkeypatch):
    monkeypatch.setenv("SH_PAIR_URL", "http://env-sh.example/v1")
    args = _build_parser().parse_args([])
    assert args.sh_pair_url == "http://env-sh.example/v1"


def test_sh_pair_url_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("SH_PAIR_URL", "http://env-sh.example/v1")
    args = _build_parser().parse_args(["--sh-pair-url", "http://flag-sh.example/v1"])
    assert args.sh_pair_url == "http://flag-sh.example/v1"


def test_cli_only_flags_have_no_env_override():
    args = _build_parser().parse_args([])
    assert args.agentworld_model == "agentworld"
    assert args.pair_model == "handler-agent"
    assert args.max_turns == 8
    assert str(args.out).endswith("calibration/handler-agent-harness/report.json")


def test_cli_only_flags_set_explicitly():
    args = _build_parser().parse_args([
        "--agentworld-model", "aw-model",
        "--pair-model", "pair-model",
        "--max-turns", "3",
        "--out", "/tmp/somewhere/report.json",
    ])
    assert args.agentworld_model == "aw-model"
    assert args.pair_model == "pair-model"
    assert args.max_turns == 3
    assert str(args.out) == "/tmp/somewhere/report.json"


# ---------------------------------------------------------------------------
# preflight-failure report-write path (§1e)
# ---------------------------------------------------------------------------

def test_preflight_failure_writes_report_and_exits_nonzero(tmp_path, respx_mock):
    out = tmp_path / "nested" / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(side_effect=httpx.ConnectError("connection refused"))
    respx_mock.get(f"{SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-agent"}]})
    )

    with pytest.raises(SystemExit) as exc_info:
        run_smoke(
            gw_agentworld_url=GW_URL, sh_pair_url=SH_URL,
            agentworld_model="agentworld", pair_model="handler-agent",
            max_turns=8, out=out,
        )

    assert exc_info.value.code != 0
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["turns"] == 0
    assert report["goal_met"] is False
    assert report["final_counts"] is None
    assert report["transcript"] == []
    assert "GW/AgentWorld" in report["preflight_error"]
    assert "connection refused" in report["preflight_error"]


# ---------------------------------------------------------------------------
# mid-run turn failure (§1e)
# ---------------------------------------------------------------------------

def _agent_response(event, job_id):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(
            {"event": event, "id": job_id, "task_type": "test", "model": "m1", "reason": "why"}
        )}}]},
    )


def _state_response(counts):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps({
            "counts": counts,
            "in_flight": [],
            "workers": {"capacity": 2, "utilization": 0.0},
            "stasis_duration": 0.0,
            "stasis_velocity": 0.0,
        })}}]},
    )


def _handler_response(decision="continue"):
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(
            {"on_track": True, "note": "ok", "anomaly": None, "decision": decision, "redirect": None}
        )}}]},
    )


def test_turn_failure_writes_partial_report_and_exits_nonzero(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    respx_mock.get(f"{SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-agent"}]})
    )

    # Turn 1 succeeds fully (agent -> env -> handler). Turn 2's agent call
    # returns unparseable JSON on both attempts, so ask_json raises and the
    # turn-level try/except in run_smoke must catch it.
    sh_route = respx_mock.post(f"{SH_URL}/chat/completions")
    sh_route.side_effect = [
        _agent_response("claimed", "j1"),
        _handler_response("continue"),
        httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": "still not json"}}]}),
    ]
    respx_mock.post(f"{GW_URL}/chat/completions").mock(
        return_value=_state_response({"pending": 1, "active": 1, "completed": 0, "failed": 0})
    )

    with pytest.raises(SystemExit) as exc_info:
        run_smoke(
            gw_agentworld_url=GW_URL, sh_pair_url=SH_URL,
            agentworld_model="agentworld", pair_model="handler-agent",
            max_turns=8, out=out,
        )

    assert exc_info.value.code != 0
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["goal_met"] is False
    transcript = report["transcript"]
    assert len(transcript) == 2
    assert transcript[0]["turn"] == 1
    assert "action" in transcript[0]
    assert transcript[1] == {"turn": 2, "error": transcript[1]["error"]}
    assert "AGENT: could not parse JSON" in transcript[1]["error"]
    assert report["final_counts"] == {"pending": 1, "active": 1, "completed": 0, "failed": 0}


def test_goal_met_writes_report_and_exits_zero(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    respx_mock.get(f"{SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-agent"}]})
    )
    respx_mock.post(f"{SH_URL}/chat/completions").mock(
        side_effect=[_agent_response("completed", "j2"), _handler_response("done")]
    )
    respx_mock.post(f"{GW_URL}/chat/completions").mock(
        return_value=_state_response({"pending": 0, "active": 0, "completed": 2, "failed": 0})
    )

    with pytest.raises(SystemExit) as exc_info:
        run_smoke(
            gw_agentworld_url=GW_URL, sh_pair_url=SH_URL,
            agentworld_model="agentworld", pair_model="handler-agent",
            max_turns=8, out=out,
        )

    assert exc_info.value.code == 0
    report = json.loads(out.read_text())
    assert report["goal_met"] is True
    assert report["turns"] == 1
    assert "preflight_error" not in report


@pytest.fixture
def respx_mock():
    with respx.mock(assert_all_called=False) as mock:
        yield mock
