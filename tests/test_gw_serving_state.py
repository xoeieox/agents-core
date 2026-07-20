"""Tests for the canonical GW serving-state + model-name resolver
(gw-serving-state-resolver-v0).

Mock-only — no live network call (AC7). Covers:
  - registry alias round-trips + unknown_model + GwRegistryError on malformed (AC1, AC5)
  - resolver in each state: single-122B, :8408 down, endpoint down, multi-model,
    reachable/unreachable/equal slot-2 (AC2, AC3, AC4)
  - source_freshness granularity
  - swarm_serving / swarm_model external-behavior pin (AC6)
"""

import textwrap
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agents_core.llm import (
    AuthorityGapError,
    GwRegistryError,
    GwServingState,
    ModelEntry,
    _gw_registry_lookup,
    _load_gw_model_registry,
    gw_serving_state,
    gw_slot2_url,
    require_authoritative_mode,
    swarm_model,
    swarm_serving,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_alias_roundtrip_display_label_and_canonical_id():
    """qwen3.5-122b-a10b <-> gravitywell-122b reconciliation (AC5)."""
    by_canonical = _gw_registry_lookup("gravitywell-122b")
    by_display = _gw_registry_lookup("qwen3.5-122b-a10b")
    assert by_canonical is not None
    assert by_canonical == by_display
    assert by_canonical.canonical_id == "gravitywell-122b"
    assert by_canonical.display_label == "qwen3.5-122b-a10b"
    assert by_canonical.mode_alias == "big"


def test_registry_lookup_by_operator_and_mode_alias():
    entry = _gw_registry_lookup("gravitywell-122b")
    assert _gw_registry_lookup(entry.operator_alias) == entry
    assert _gw_registry_lookup(entry.mode_alias) == entry


def test_registry_unknown_alias_returns_none():
    assert _gw_registry_lookup("some-other-model-id") is None
    assert _gw_registry_lookup(None) is None


def test_registry_malformed_missing_models_key_raises(tmp_path):
    bad = tmp_path / "gw_models.yaml"
    bad.write_text(yaml.safe_dump({"not_models": []}))
    with pytest.raises(GwRegistryError):
        _load_gw_model_registry(bad)


def test_registry_malformed_missing_field_raises(tmp_path):
    bad = tmp_path / "gw_models.yaml"
    bad.write_text(textwrap.dedent("""\
        models:
          - canonical_id: foo
            mode_alias: big
        """))
    with pytest.raises(GwRegistryError):
        _load_gw_model_registry(bad)


def test_registry_absent_file_raises(tmp_path):
    with pytest.raises(GwRegistryError):
        _load_gw_model_registry(tmp_path / "does-not-exist.yaml")


def test_registry_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "gw_models.yaml"
    bad.write_text("models: [this is not: valid: yaml:::")
    with pytest.raises(GwRegistryError):
        _load_gw_model_registry(bad)


def test_registry_valid_file_loads_model_entries(tmp_path):
    good = tmp_path / "gw_models.yaml"
    good.write_text(textwrap.dedent("""\
        models:
          - canonical_id: foo-1
            mode_alias: big
            operator_alias: foo
            display_label: Foo-One
            weights_hint: "Foo-1"
        """))
    entries = _load_gw_model_registry(good)
    assert entries == [ModelEntry("foo-1", "big", "foo", "Foo-One", "Foo-1")]


# ---------------------------------------------------------------------------
# gw_slot2_url
# ---------------------------------------------------------------------------

def test_gw_slot2_url_default_port(monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    monkeypatch.delenv("GW_SLOT2_PORT", raising=False)
    assert gw_slot2_url("http://1.2.3.4:8081") == "http://1.2.3.4:8082"


def test_gw_slot2_url_env_override_wins(monkeypatch):
    monkeypatch.setenv("GW_SLOT2_URL", "http://override:9999")
    assert gw_slot2_url("http://1.2.3.4:8081") == "http://override:9999"


def test_gw_slot2_url_custom_port(monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    monkeypatch.setenv("GW_SLOT2_PORT", "9090")
    assert gw_slot2_url("http://1.2.3.4:8081") == "http://1.2.3.4:9090"


# ---------------------------------------------------------------------------
# gw_serving_state — mocked HTTP helper
# ---------------------------------------------------------------------------

def _resp(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    return resp


def _mock_get(url_map, default=None):
    """Return a side_effect fn: url_map is a dict of substring -> response-or-exception."""
    def _get(url, timeout=None):
        for substr, outcome in url_map.items():
            if substr in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        if default is not None:
            if isinstance(default, Exception):
                raise default
            return default
        raise ConnectionErrorStub(url)
    return _get


class ConnectionErrorStub(Exception):
    pass


FLIP_STATUS_BIG = {
    "node": "gravitywell",
    "mode": "big",
    "model_label": "qwen3.5-122b-a10b",
    "healthy": True,
    "units": {"llama-server": "active"},
    "in_flight_flip": False,
}


@patch("agents_core.llm.requests.get")
def test_single_122b_serving(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "8082/v1/models": ConnectionErrorStub("slot2 down"),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.serving is True
    assert state.reachable is True
    assert state.served_id == "gravitywell-122b"
    assert state.served_ids == ["gravitywell-122b"]
    assert state.canonical.display_label == "qwen3.5-122b-a10b"
    assert state.unknown_model is False
    assert state.mode == "big"
    assert state.authority_gap is False
    assert state.mode_inferred == "big"
    assert state.distinct_second_model is False
    assert state.source_freshness["flip_controller"]["status"] == "answered"
    assert state.source_freshness["models_endpoint"]["status"] == "answered"
    assert state.source_freshness["health"]["status"] == "answered"
    assert state.source_freshness["slot2"]["status"] == "unreachable"


@patch("agents_core.llm.requests.get")
def test_flip_controller_unreachable_sets_authority_gap(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": ConnectionErrorStub("flip-controller down"),
        "8082/v1/models": ConnectionErrorStub("slot2 down"),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.mode is None
    assert state.authority_gap is True
    assert state.mode_inferred == "big"
    assert state.serving is True
    assert state.served_id == "gravitywell-122b"
    assert state.source_freshness["flip_controller"]["status"] == "unreachable"

    with pytest.raises(AuthorityGapError):
        require_authoritative_mode(state)


@patch("agents_core.llm.requests.get")
def test_endpoint_down_mode_still_resolves(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": ConnectionErrorStub("endpoint down"),
        "/v1/models": ConnectionErrorStub("endpoint down"),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.serving is False
    assert state.reachable is False
    assert state.served_id is None
    assert state.canonical is None
    assert state.unknown_model is False
    assert state.mode == "big"
    assert state.authority_gap is False
    # No live model to infer from — mode_inferred stays None (not a substitute mode).
    assert state.mode_inferred is None


@patch("agents_core.llm.requests.get")
def test_unknown_served_model_is_loud(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "8082/v1/models": ConnectionErrorStub("slot2 down"),
        "/v1/models": _resp(200, {"data": [{"id": "some-brand-new-model"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.served_id == "some-brand-new-model"
    assert state.canonical is None
    assert state.unknown_model is True


@patch("agents_core.llm.requests.get")
def test_multi_model_endpoint_distinct_second(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "/v1/models": _resp(200, {"data": [
            {"id": "gravitywell-122b"},
            {"id": "some-other-resident-model"},
        ]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.served_ids == ["gravitywell-122b", "some-other-resident-model"]
    assert state.distinct_second_model is True


@patch("agents_core.llm.requests.get")
def test_slot2_reachable_distinct_model(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "8082/v1/models": _resp(200, {"data": [{"id": "second-resident-model"}]}),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.distinct_second_model is True
    assert state.source_freshness["slot2"]["status"] == "answered"


@patch("agents_core.llm.requests.get")
def test_slot2_unreachable_is_false_with_freshness_note(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "8082/v1/models": ConnectionErrorStub("slot2 down"),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.distinct_second_model is False
    assert state.source_freshness["slot2"]["status"] == "unreachable"
    assert state.source_freshness["slot2"]["checked_at"] is None


@patch("agents_core.llm.requests.get")
def test_slot2_equal_primary_is_false(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": _resp(200, FLIP_STATUS_BIG),
        "8082/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    state = gw_serving_state(endpoint="http://gw:8081")

    assert state.distinct_second_model is False


# ---------------------------------------------------------------------------
# swarm_serving / swarm_model — external-behavior pin (AC6)
# ---------------------------------------------------------------------------

@patch("agents_core.llm.requests.get")
def test_swarm_serving_true_when_models_and_health_ok(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": ConnectionErrorStub("flip-controller irrelevant to swarm_serving"),
        "8082/v1/models": ConnectionErrorStub("slot2 irrelevant"),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    assert swarm_serving(swarm_url="http://swarm:8081") is True


@patch("agents_core.llm.requests.get")
def test_swarm_serving_false_when_models_empty(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": ConnectionErrorStub("n/a"),
        "8082/v1/models": ConnectionErrorStub("n/a"),
        "/v1/models": _resp(200, {"data": []}),
    })

    assert swarm_serving(swarm_url="http://swarm:8081") is False


@patch("agents_core.llm.requests.get")
def test_swarm_serving_false_when_health_down(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": ConnectionErrorStub("health down"),
        "/v0/status": ConnectionErrorStub("n/a"),
        "8082/v1/models": ConnectionErrorStub("n/a"),
        "/v1/models": _resp(200, {"data": [{"id": "gravitywell-122b"}]}),
    })

    assert swarm_serving(swarm_url="http://swarm:8081") is False


@patch("agents_core.llm.requests.get")
def test_swarm_model_returns_served_id(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": ConnectionErrorStub("n/a"),
        "8082/v1/models": ConnectionErrorStub("n/a"),
        "/v1/models": _resp(200, {"data": [{"id": "Qwen2.5-3B"}]}),
    })

    assert swarm_model(swarm_url="http://swarm:8081") == "Qwen2.5-3B"


@patch("agents_core.llm.requests.get")
def test_swarm_model_none_when_not_serving(mock_get, monkeypatch):
    monkeypatch.delenv("GW_SLOT2_URL", raising=False)
    mock_get.side_effect = _mock_get({
        "/health": _resp(200),
        "/v0/status": ConnectionErrorStub("n/a"),
        "8082/v1/models": ConnectionErrorStub("n/a"),
        "/v1/models": ConnectionErrorStub("models down"),
    })

    assert swarm_model(swarm_url="http://swarm:8081") is None
