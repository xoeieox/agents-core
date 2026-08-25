"""Step-payload thinking control for the swarm (NInfer berth) path
(agents-core-fixer-thinking-off-knob-v0).

Pins the 4-case matrix from the spec: swarm+think=False sends top-level
enable_thinking:False; swarm+think=True sends nothing (server default);
non-swarm keeps chat_template_kwargs untouched (both think values); plus the
_force_conclusion parity pin (seam 2).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agents_core.gw_agent import _force_conclusion, build_step_payload

_MSGS = [{"role": "user", "content": "hi"}]
_TOOLS = {"read_file": {"name": "read_file", "description": "d", "parameters": {}}}


def _payload(**kw):
    return build_step_payload(model="m", messages=_MSGS, tools=_TOOLS, **kw)


def test_swarm_think_false_sends_top_level_enable_thinking_false():
    p = _payload(is_swarm=True, think=False)
    assert p.get("enable_thinking") is False
    assert "chat_template_kwargs" not in p


def test_swarm_think_true_sends_no_thinking_field():
    p = _payload(is_swarm=True, think=True)
    assert "enable_thinking" not in p
    assert "chat_template_kwargs" not in p


def test_nonswarm_think_false_uses_chat_template_kwargs():
    p = _payload(is_swarm=False, think=False)
    assert p.get("chat_template_kwargs") == {"enable_thinking": False}
    assert "enable_thinking" not in p


def test_nonswarm_think_true_uses_chat_template_kwargs():
    p = _payload(is_swarm=False, think=True)
    assert p.get("chat_template_kwargs") == {"enable_thinking": True}
    assert "enable_thinking" not in p


def test_force_conclusion_swarm_sends_top_level_enable_thinking_false():
    captured = {}
    fake_resp = MagicMock()
    fake_resp.raise_for_status = lambda: None
    fake_resp.json = lambda: {
        "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
        "model": "m",
    }

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return fake_resp

    with patch("agents_core.gw_agent.requests.post", side_effect=_fake_post):
        _force_conclusion(
            messages=[{"role": "user", "content": "x"}],
            backend_url="http://berth/v1",
            timeout=30,
            json_mode=False,
            log=None,
            is_swarm=True,
        )
    assert captured["json"].get("enable_thinking") is False
    assert "chat_template_kwargs" not in captured["json"]


def test_force_conclusion_nonswarm_unchanged():
    captured = {}
    fake_resp = MagicMock()
    fake_resp.raise_for_status = lambda: None
    fake_resp.json = lambda: {
        "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
        "model": "m",
    }

    def _fake_post(url, json=None, timeout=None):
        captured["json"] = json
        return fake_resp

    with patch("agents_core.gw_agent.requests.post", side_effect=_fake_post):
        _force_conclusion(
            messages=[{"role": "user", "content": "x"}],
            backend_url="http://gw/v1",
            timeout=30,
            json_mode=False,
            log=None,
            is_swarm=False,
        )
    assert captured["json"].get("chat_template_kwargs") == {"enable_thinking": False}
    assert "enable_thinking" not in captured["json"]