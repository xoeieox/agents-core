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
    _resolve_pair_endpoints,
    ask_json,
    preflight,
    run_smoke,
)

GW_URL = "http://gw.example/v1"
SH_URL = "http://sh.example/v1"
AGENT_SH_URL = "http://agent-sh.example/v1"
HANDLER_SH_URL = "http://handler-sh.example/v1"


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
            gw_agentworld_url=GW_URL, agentworld_model="agentworld",
            agent_url=SH_URL, agent_model="handler-agent",
            handler_url=SH_URL, handler_model="handler-agent",
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
            gw_agentworld_url=GW_URL, agentworld_model="agentworld",
            agent_url=SH_URL, agent_model="handler-agent",
            handler_url=SH_URL, handler_model="handler-agent",
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
            gw_agentworld_url=GW_URL, agentworld_model="agentworld",
            agent_url=SH_URL, agent_model="handler-agent",
            handler_url=SH_URL, handler_model="handler-agent",
            max_turns=8, out=out,
        )

    assert exc_info.value.code == 0
    report = json.loads(out.read_text())
    assert report["goal_met"] is True
    assert report["turns"] == 1
    assert "preflight_error" not in report


# ---------------------------------------------------------------------------
# _resolve_pair_endpoints (agents-core-handler-agent-endpoint-split-v0)
# ---------------------------------------------------------------------------

def _clear_role_env(monkeypatch):
    monkeypatch.delenv("AGENT_URL", raising=False)
    monkeypatch.delenv("HANDLER_URL", raising=False)
    monkeypatch.delenv("SH_PAIR_URL", raising=False)


def test_resolve_pair_endpoints_backward_compat_defaults(monkeypatch):
    _clear_role_env(monkeypatch)
    args = _build_parser().parse_args([])
    agent_url, agent_model, handler_url, handler_model = _resolve_pair_endpoints(args)
    assert agent_url == handler_url == "http://203.0.113.12:8082/v1"
    assert agent_model == handler_model == "handler-agent"


def test_resolve_pair_endpoints_pair_default_fans_out(monkeypatch):
    _clear_role_env(monkeypatch)
    args = _build_parser().parse_args(["--sh-pair-url", "http://x/v1", "--pair-model", "mX"])
    agent_url, agent_model, handler_url, handler_model = _resolve_pair_endpoints(args)
    assert agent_url == handler_url == "http://x/v1"
    assert agent_model == handler_model == "mX"


def test_resolve_pair_endpoints_independent_override(monkeypatch):
    _clear_role_env(monkeypatch)
    args = _build_parser().parse_args([
        "--agent-url", "http://a/v1", "--agent-model", "mA",
        "--handler-url", "http://h/v1", "--handler-model", "mH",
    ])
    agent_url, agent_model, handler_url, handler_model = _resolve_pair_endpoints(args)
    assert agent_url == "http://a/v1"
    assert agent_model == "mA"
    assert handler_url == "http://h/v1"
    assert handler_model == "mH"


def test_resolve_pair_endpoints_partial_override(monkeypatch):
    _clear_role_env(monkeypatch)
    args = _build_parser().parse_args([
        "--sh-pair-url", "http://p/v1", "--pair-model", "mP", "--agent-model", "mA",
    ])
    agent_url, agent_model, handler_url, handler_model = _resolve_pair_endpoints(args)
    assert agent_url == "http://p/v1"
    assert agent_model == "mA"
    assert handler_url == "http://p/v1"
    assert handler_model == "mP"


def test_resolve_pair_endpoints_env_precedence(monkeypatch):
    _clear_role_env(monkeypatch)
    monkeypatch.setenv("AGENT_URL", "http://env-agent.example/v1")
    args = _build_parser().parse_args([])
    agent_url, _, _, _ = _resolve_pair_endpoints(args)
    assert agent_url == "http://env-agent.example/v1"

    args = _build_parser().parse_args(["--agent-url", "http://flag-agent.example/v1"])
    agent_url, _, _, _ = _resolve_pair_endpoints(args)
    assert agent_url == "http://flag-agent.example/v1"


def test_resolve_pair_endpoints_empty_env_coerced_to_unset(monkeypatch):
    _clear_role_env(monkeypatch)
    monkeypatch.setenv("AGENT_URL", "")
    args = _build_parser().parse_args([])
    agent_url, _, _, _ = _resolve_pair_endpoints(args)
    assert agent_url == "http://203.0.113.12:8082/v1"


def test_resolve_pair_endpoints_whitespace_env_coerced_to_unset(monkeypatch):
    _clear_role_env(monkeypatch)
    monkeypatch.setenv("AGENT_URL", "   ")
    args = _build_parser().parse_args([])
    agent_url, _, _, _ = _resolve_pair_endpoints(args)
    assert agent_url == "http://203.0.113.12:8082/v1"


# ---------------------------------------------------------------------------
# preflight: distinct endpoints + dedup (agents-core-handler-agent-endpoint-split-v0)
# ---------------------------------------------------------------------------

def test_preflight_probes_distinct_endpoints(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    gw_route = respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    agent_route = respx_mock.get(f"{AGENT_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agent-model"}]})
    )
    handler_route = respx_mock.get(f"{HANDLER_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-model"}]})
    )

    preflight(GW_URL, AGENT_SH_URL, "agent-model", HANDLER_SH_URL, "handler-model", out)

    assert gw_route.call_count == 1
    assert agent_route.call_count == 1
    assert handler_route.call_count == 1
    assert not out.exists()


def test_preflight_dedup_probes_shared_endpoint_once(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    sh_route = respx_mock.get(f"{SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-agent"}]})
    )

    preflight(GW_URL, SH_URL, "handler-agent", SH_URL, "handler-agent", out)

    assert sh_route.call_count == 1
    assert not out.exists()


def test_preflight_distinct_endpoint_failure_identifies_role(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    respx_mock.get(f"{AGENT_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agent-model"}]})
    )
    respx_mock.get(f"{HANDLER_SH_URL}/models").mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(SystemExit) as exc_info:
        preflight(GW_URL, AGENT_SH_URL, "agent-model", HANDLER_SH_URL, "handler-model", out)

    assert exc_info.value.code != 0
    assert out.exists()
    report = json.loads(out.read_text())
    assert "handler" in report["preflight_error"]
    assert "connection refused" in report["preflight_error"]


# ---------------------------------------------------------------------------
# report fields + per-role call routing (agents-core-handler-agent-endpoint-split-v0)
# ---------------------------------------------------------------------------

def test_report_has_agent_and_handler_fields_not_pair(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    respx_mock.get(f"{AGENT_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agent-model"}]})
    )
    respx_mock.get(f"{HANDLER_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-model"}]})
    )
    respx_mock.post(f"{AGENT_SH_URL}/chat/completions").mock(
        return_value=_agent_response("completed", "j2")
    )
    respx_mock.post(f"{HANDLER_SH_URL}/chat/completions").mock(
        return_value=_handler_response("done")
    )
    respx_mock.post(f"{GW_URL}/chat/completions").mock(
        return_value=_state_response({"pending": 0, "active": 0, "completed": 2, "failed": 0})
    )

    with pytest.raises(SystemExit) as exc_info:
        run_smoke(
            gw_agentworld_url=GW_URL, agentworld_model="agentworld",
            agent_url=AGENT_SH_URL, agent_model="agent-model",
            handler_url=HANDLER_SH_URL, handler_model="handler-model",
            max_turns=8, out=out,
        )

    assert exc_info.value.code == 0
    report = json.loads(out.read_text())
    assert "pair" not in report
    assert report["agent"] == f"{AGENT_SH_URL} (agent-model)"
    assert report["handler"] == f"{HANDLER_SH_URL} (handler-model)"
    assert report["environment"] == GW_URL


def test_run_smoke_routes_agent_and_handler_calls_to_distinct_endpoints(tmp_path, respx_mock):
    out = tmp_path / "report.json"
    respx_mock.get(f"{GW_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agentworld"}]})
    )
    respx_mock.get(f"{AGENT_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "agent-model"}]})
    )
    respx_mock.get(f"{HANDLER_SH_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "handler-model"}]})
    )
    agent_route = respx_mock.post(f"{AGENT_SH_URL}/chat/completions").mock(
        return_value=_agent_response("completed", "j2")
    )
    handler_route = respx_mock.post(f"{HANDLER_SH_URL}/chat/completions").mock(
        return_value=_handler_response("done")
    )
    respx_mock.post(f"{GW_URL}/chat/completions").mock(
        return_value=_state_response({"pending": 0, "active": 0, "completed": 2, "failed": 0})
    )

    with pytest.raises(SystemExit) as exc_info:
        run_smoke(
            gw_agentworld_url=GW_URL, agentworld_model="agentworld",
            agent_url=AGENT_SH_URL, agent_model="agent-model",
            handler_url=HANDLER_SH_URL, handler_model="handler-model",
            max_turns=8, out=out,
        )

    assert exc_info.value.code == 0
    assert agent_route.call_count == 1
    assert handler_route.call_count == 1


@pytest.fixture
def respx_mock():
    with respx.mock(assert_all_called=False) as mock:
        yield mock
