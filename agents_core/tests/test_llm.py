"""Unit tests for agents_core.llm multi-operator routing (v0).

Covers:
  - call_operator() dispatches to the right backend per operator class
  - Unknown operator class raises ValueError
  - call_llm() legacy behavior is byte-identical to pre-bind

No real network calls — all backends are mocked.
"""

import json

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llama_response(text: str):
    """Minimal requests.Response mock for a successful llama-server reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text, "reasoning_content": None}}]
    }
    return resp


# ---------------------------------------------------------------------------
# test_call_operator_qwen_dispatches_to_existing_endpoint
# ---------------------------------------------------------------------------

def test_call_operator_qwen_dispatches_to_existing_endpoint():
    """qwen operator_class posts to the local llama-server URL."""
    from agents_core import llm as llm_mod

    fake_resp = _make_llama_response("qwen answer")

    with patch("agents_core.llm.requests.post", return_value=fake_resp) as mock_post, \
         patch("agents_core.llm._call_qwen_backend",
               wraps=llm_mod._call_qwen_backend) as spy:

        result = llm_mod.call_operator(
            "qwen",
            prompt="hello",
            system="sys",
            bundle_ids=None,
        )

    assert result == "qwen answer"
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert "8081" in called_url
    assert "/v1/chat/completions" in called_url


def test_call_operator_qwen_uses_default_model_constant():
    """The qwen default model string is the expected production value."""
    from agents_core.llm import OPERATOR_DEFAULTS
    assert OPERATOR_DEFAULTS["qwen"] == "qwen3.6-35b-a3b"


# ---------------------------------------------------------------------------
# test_call_operator_sonnet_dispatches_to_claude_queue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("operator_class,expected_model", [
    ("sonnet", "claude-sonnet-4-6"),
    ("opus",   "claude-opus-4-7"),
    ("haiku",  "claude-haiku-4-5-20251001"),
])
def test_call_operator_anthropic_dispatches_to_claude_queue(
        operator_class, expected_model):
    """Anthropic-family classes route via submit_and_wait and return the response string."""
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="y") as mock_saw:
        from agents_core import llm as llm_mod
        result = llm_mod.call_operator(operator_class, prompt="x")

    assert result == "y"
    mock_saw.assert_called_once()
    task_dict = mock_saw.call_args[0][0]
    assert task_dict["task_type"] == "llm_call"
    assert task_dict["model"] == expected_model
    assert task_dict["payload"]["operator_class"] == operator_class
    assert task_dict["payload"]["prompt"] == "x"
    assert task_dict["payload"]["system"] == ""
    assert task_dict["payload"]["json_mode"] is False
    assert task_dict["payload"]["_ignore_intention_registry"] is True


def test_call_operator_sonnet_dispatches_to_claude_queue():
    """Alias for the parametrized test — spec-named test function."""
    test_call_operator_anthropic_dispatches_to_claude_queue(
        "sonnet", "claude-sonnet-4-6"
    )


# ---------------------------------------------------------------------------
# test_call_operator_unknown_class_raises
# ---------------------------------------------------------------------------

def test_call_operator_unknown_class_raises():
    """Unknown operator_class raises ValueError with the class name."""
    from agents_core.llm import call_operator
    with pytest.raises(ValueError, match="gpt4"):
        call_operator("gpt4", prompt="irrelevant")


def test_call_operator_unknown_class_names_valid_options():
    """ValueError message lists the valid operator classes."""
    from agents_core.llm import call_operator, OPERATOR_DEFAULTS
    with pytest.raises(ValueError) as exc_info:
        call_operator("unknown-op", prompt="x")
    err = str(exc_info.value)
    for valid in OPERATOR_DEFAULTS:
        assert valid in err


# ---------------------------------------------------------------------------
# test_call_llm_legacy_behavior_byte_identical
# ---------------------------------------------------------------------------

def test_call_llm_legacy_behavior_byte_identical():
    """call_llm() returns the same result as the pre-bind qwen path.

    Regression guard: same inputs → same output shape and same HTTP endpoint.
    """
    fake_resp = _make_llama_response("legacy response text")

    with patch("agents_core.llm.requests.post", return_value=fake_resp) as mock_post:
        from agents_core.llm import call_llm
        result = call_llm(prompt="legacy prompt", system="sys override")

    assert result == "legacy response text"
    mock_post.assert_called_once()
    called_url = mock_post.call_args[0][0]
    assert "/v1/chat/completions" in called_url
    payload = mock_post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "sys override"}
    assert payload["messages"][1] == {"role": "user", "content": "legacy prompt"}


def test_call_llm_raises_on_unreachable_after_retries():
    """call_llm() raises OperatorUnreachableError after exhausting retries on ConnectionError."""
    import requests as req
    from agents_core.llm import call_llm, OperatorUnreachableError, LLAMACPP_URL

    conn_err = req.exceptions.ConnectionError("down")

    with patch("agents_core.llm.requests.post", side_effect=conn_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            call_llm(prompt="will fail", system="s")

    exc = exc_info.value
    assert exc.url == LLAMACPP_URL
    assert exc.last_error is conn_err


def test_call_llm_raises_on_http_error_after_retries():
    """call_llm() raises OperatorUnreachableError after exhausting retries on HTTPError."""
    import requests as req
    from agents_core.llm import call_llm, OperatorUnreachableError, LLAMACPP_URL

    http_err = req.exceptions.HTTPError("503")

    with patch("agents_core.llm.requests.post", side_effect=http_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            call_llm(prompt="will fail", system="s")

    exc = exc_info.value
    assert exc.url == LLAMACPP_URL
    assert exc.last_error is http_err


def test_call_llm_returns_none_on_empty_content():
    """call_llm() returns None (not exception) when operator returns empty content."""
    from agents_core.llm import call_llm

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": None}}]
    }

    with patch("agents_core.llm.requests.post", return_value=resp):
        result = call_llm(prompt="empty reply", system="s")

    assert result is None


def test_call_llm_retries_then_succeeds():
    """call_llm() retries on ConnectionError and returns content on eventual success."""
    import requests as req
    from agents_core.llm import call_llm

    success_resp = _make_llama_response("recovered")
    conn_err = req.exceptions.ConnectionError("transient")

    with patch("agents_core.llm.requests.post",
               side_effect=[conn_err, conn_err, success_resp]), \
         patch("agents_core.llm.time.sleep"):
        result = call_llm(prompt="retry me", system="s")

    assert result == "recovered"


def test_call_llm_returns_none_on_generic_exception():
    """call_llm() returns None (not exception) for non-Connection/HTTP exceptions."""
    from agents_core.llm import call_llm

    with patch("agents_core.llm.requests.post", side_effect=ValueError("unexpected")):
        result = call_llm(prompt="generic error", system="s")

    assert result is None


def test_operator_unreachable_error_carries_diagnostic():
    """OperatorUnreachableError exposes .url and .last_error; str() includes both."""
    from agents_core.llm import OperatorUnreachableError

    cause = ValueError("boom")
    exc = OperatorUnreachableError("http://x:1234", cause)

    assert exc.url == "http://x:1234"
    assert exc.last_error is cause
    msg = str(exc)
    assert "http://x:1234" in msg
    assert "boom" in msg


def test_call_llm_delegates_to_call_operator():
    """call_llm() is a thin wrapper — it calls call_operator('qwen', ...)."""
    from agents_core import llm as llm_mod

    with patch.object(llm_mod, "call_operator", return_value="delegated") as mock_op:
        result = llm_mod.call_llm(
            prompt="p", system="s", timeout=99, json_mode=True,
            temperature=0.1, log=None, bundle_ids=["x"]
        )

    assert result == "delegated"
    mock_op.assert_called_once_with(
        operator_class="qwen",
        prompt="p",
        system="s",
        timeout=99,
        json_mode=True,
        temperature=0.1,
        log=None,
        bundle_ids=["x"],
    )


# ---------------------------------------------------------------------------
# GW repoint (agents-core-local-llm-gw-repoint-v0): LLAMACPP_URL env-override,
# enable_thinking suppression, reasoning-key fallback, no stale model field.
# No network calls — all backends mocked, per AC1/AC2/AC3/AC4.
# ---------------------------------------------------------------------------

def test_llamacpp_url_defaults_to_gw_url(monkeypatch):
    """AC1: with LOCAL_LLM_URL unset, the local-LLM endpoint resolves to GW_URL."""
    from agents_core import llm as llm_mod

    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    assert llm_mod._llamacpp_url() == llm_mod.GW_URL


def test_llamacpp_url_honors_local_llm_url_override(monkeypatch):
    """AC1: LOCAL_LLM_URL, when set, wins over GW_URL — read at call time."""
    from agents_core import llm as llm_mod

    monkeypatch.setenv("LOCAL_LLM_URL", "http://example-override:9999")
    assert llm_mod._llamacpp_url() == "http://example-override:9999"

    # Call-time, not import-time: unsetting again reverts on the next call.
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    assert llm_mod._llamacpp_url() == llm_mod.GW_URL


def test_call_qwen_backend_payload_defaults_enable_thinking_false(monkeypatch):
    """AC2: default payload carries chat_template_kwargs.enable_thinking = false."""
    from agents_core.llm import call_llm

    monkeypatch.delenv("LOCAL_LLM_THINK", raising=False)
    resp = _make_llama_response("answer")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post:
        call_llm(prompt="p", system="s")

    payload = mock_post.call_args[1]["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_call_qwen_backend_payload_opts_into_thinking(monkeypatch):
    """AC2: LOCAL_LLM_THINK=1 flips enable_thinking to true."""
    from agents_core.llm import call_llm

    monkeypatch.setenv("LOCAL_LLM_THINK", "1")
    resp = _make_llama_response("answer")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post:
        call_llm(prompt="p", system="s")

    payload = mock_post.call_args[1]["json"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("message,expected", [
    ({"content": None, "reasoning": "trace", "reasoning_content": None}, "trace"),
    ({"content": None, "reasoning": None, "reasoning_content": "legacy trace"}, "legacy trace"),
    ({"content": "plain content", "reasoning": None, "reasoning_content": None}, "plain content"),
])
def test_call_qwen_backend_reasoning_key_fallback(message, expected):
    """AC3: parser falls back content -> reasoning (vLLM) -> reasoning_content (llama.cpp)."""
    from agents_core.llm import call_llm

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": message}]}

    with patch("agents_core.llm.requests.post", return_value=resp):
        result = call_llm(prompt="p", system="s")

    assert result == expected


def test_call_qwen_backend_never_sends_model_field():
    """AC4: the qwen-operator payload omits `model` entirely (GW serves one resident
    model and 404s on a stale name — OPERATOR_DEFAULTS['qwen'] is not served)."""
    from agents_core.llm import call_llm

    resp = _make_llama_response("answer")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post:
        call_llm(prompt="p", system="s")

    payload = mock_post.call_args[1]["json"]
    assert "model" not in payload


def test_call_operator_qwen_never_forwards_operator_defaults_model():
    """AC4 (dispatcher level): OPERATOR_DEFAULTS['qwen'] never reaches the wire even
    when a caller explicitly passes it as model=."""
    from agents_core.llm import call_operator, OPERATOR_DEFAULTS

    resp = _make_llama_response("answer")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post:
        call_operator("qwen", prompt="p", model=OPERATOR_DEFAULTS["qwen"])

    payload = mock_post.call_args[1]["json"]
    assert "model" not in payload


def test_call_llm_backoff_bounded_by_remaining_timeout_budget(monkeypatch):
    """Item 7: backoff never sleeps past the caller's remaining timeout budget —
    control flow (retry/raise/swallow) is unchanged, only the sleep duration shrinks."""
    import requests as req
    from agents_core.llm import call_llm, OperatorUnreachableError

    conn_err = req.exceptions.ConnectionError("down")
    sleeps = []

    def fake_sleep(secs):
        sleeps.append(secs)

    # timeout=5: even the first nominal 10s backoff must be clamped down to
    # whatever budget remains (<=5s), never the full 10s/20s ladder.
    with patch("agents_core.llm.requests.post", side_effect=conn_err), \
         patch("agents_core.llm.time.sleep", side_effect=fake_sleep):
        with pytest.raises(OperatorUnreachableError):
            call_llm(prompt="p", system="s", timeout=5)

    assert all(s <= 5 for s in sleeps)


def test_locality_host_by_operator_qwen_reflects_gw(monkeypatch):
    """Item 6: qwen's locality-ledger host attribution is the real (GW) host, not
    the retired StarHouse literal."""
    from agents_core import llm as llm_mod

    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    assert llm_mod._LOCALITY_HOST_BY_OPERATOR["qwen"] != "http://203.0.113.12:8081"


# ---------------------------------------------------------------------------
# Voicing Provenance Tests (GravityWell effective-operator tracking)
# ---------------------------------------------------------------------------

def test_gravitywell_successful_call_records_provenance():
    """call_operator('gravitywell') on success appends ('success', 'gravitywell') to provenance."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "gw answer"
    assert ("success", "gravitywell") in provenance


# ---------------------------------------------------------------------------
# lease_class threading (doorman-lease-class-consumers-v0)
# ---------------------------------------------------------------------------

def test_call_operator_gravitywell_default_lease_class_is_deferrable():
    """Omitting lease_class sends class=deferrable to the doorman (the safe default)."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}) as mock_acquire, \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        result = llm_mod.call_operator("gravitywell", prompt="test")

    assert result == "gw answer"
    assert mock_acquire.call_args.kwargs["lease_class"] == "deferrable"


def test_call_operator_gravitywell_explicit_protected_lease_class():
    """A measured-gate caller passing lease_class='protected' reaches the doorman."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}) as mock_acquire, \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        llm_mod.call_operator("gravitywell", prompt="test", lease_class="protected")

    assert mock_acquire.call_args.kwargs["lease_class"] == "protected"


def test_call_operator_gravitywell_explicit_deferrable_lease_class():
    """A background caller passing lease_class='deferrable' explicitly reaches the doorman
    (deliverable #3: explicit, not relying on the default)."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}) as mock_acquire, \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        llm_mod.call_operator("gravitywell", prompt="test", lease_class="deferrable")

    assert mock_acquire.call_args.kwargs["lease_class"] == "deferrable"


def test_call_operator_new_caller_forgetting_lease_class_still_lands_deferrable():
    """Adversarial case (trickster): a new caller that forgets lease_class must land
    deferrable, not silently escalate to protected - this is what proves the default
    is an enforcement barrier and not merely documentation."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}) as mock_acquire, \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"):

        # Simulates a brand-new call site nobody updated for lease_class yet.
        llm_mod.call_operator("gravitywell", prompt="test", on_wake_fail="skip", timeout=42)

    assert mock_acquire.call_args.kwargs["lease_class"] == "deferrable"


def test_call_operator_lease_class_omission_warns_once_per_module(caplog):
    """Omitting lease_class emits one WARN naming the calling module - once per
    process, not once per call (a busy night DAG must not drown the log)."""
    import logging
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    llm_mod._lease_class_default_warned.clear()

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"), \
         caplog.at_level(logging.WARNING, logger="agents_core.llm"):

        llm_mod.call_operator("gravitywell", prompt="test one")
        llm_mod.call_operator("gravitywell", prompt="test two")

    warn_records = [r for r in caplog.records if "lease-class" in r.getMessage()]
    assert len(warn_records) == 1


def test_call_operator_explicit_lease_class_never_warns():
    """A caller that passes lease_class explicitly (either value) never trips the
    silence warning - only omission does."""
    import logging
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    llm_mod._lease_class_default_warned.clear()

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"), \
         patch.object(llm_mod, "_warn_lease_class_defaulted") as mock_warn:

        llm_mod.call_operator("gravitywell", prompt="test", lease_class="deferrable")
        llm_mod.call_operator("gravitywell", prompt="test", lease_class="protected")

    mock_warn.assert_not_called()


def test_call_operator_acquire_lease_false_bypass_lease_class_ignored():
    """The acquire_lease=False bypass takes no lease at all, so lease_class is moot -
    no acquire() call is made and no silence-warning fires (C2b: out of scope for
    classification, since there is no lease to classify)."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    llm_mod._lease_class_default_warned.clear()

    with patch.object(DoormanClient, "acquire") as mock_acquire, \
         patch("agents_core.llm._call_gravitywell_backend", return_value="gw answer"), \
         patch.object(llm_mod, "_warn_lease_class_defaulted") as mock_warn:

        result = llm_mod.call_operator("gravitywell", prompt="test", acquire_lease=False)

    assert result == "gw answer"
    mock_acquire.assert_not_called()
    mock_warn.assert_not_called()


def test_gravitywell_fallback_on_doorman_unreachable():
    """call_operator('gravitywell') appends doorman_unreachable reason when doorman is down."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    provenance = []
    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("doorman down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("doorman_unreachable", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_gravitywell_fallback_on_gw_not_serving():
    """call_operator('gravitywell') appends gw_not_serving reason when doorman says not serving."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "offline"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("gw_not_serving", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_gravitywell_fallback_on_http_error():
    """call_operator('gravitywell') appends serving_http_error when GW returns HTTP error."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient
    from agents_core.llm import OperatorUnreachableError

    provenance = []
    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend",
               side_effect=OperatorUnreachableError("http://gw:8081", Exception("500"))), \
         patch("agents_core.claude_queue_sync.submit_and_wait", return_value="fallback answer"):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == "fallback answer"
    assert ("serving_http_error", "gravitywell") in provenance
    assert ("fallback", "sonnet") in provenance


def test_non_gravitywell_operator_records_success_provenance():
    """call_operator('sonnet') appends ('success', 'sonnet') to provenance."""
    from agents_core import llm as llm_mod

    provenance = []
    with patch("agents_core.claude_queue_sync.submit_and_wait", return_value="sonnet answer"):
        result = llm_mod.call_operator(
            "sonnet", prompt="test",
            _provenance_out=provenance,
        )

    assert result == "sonnet answer"
    assert ("success", "sonnet") in provenance


# ---------------------------------------------------------------------------
# on_wake_fail="park" tests (agents-core-council-park-not-degrade-v0)
# ---------------------------------------------------------------------------

def test_apply_wake_fail_park_raises_and_makes_no_fallback_call():
    """_apply_wake_fail(policy='park') raises GWParkedError and never calls call_operator
    (i.e. no paid fallback is dispatched — this is the one behavior a naive test could
    pass without actually proving no spend occurred)."""
    from agents_core import llm as llm_mod

    with patch.object(llm_mod, "call_operator") as mock_call_operator:
        with pytest.raises(llm_mod.GWParkedError) as exc_info:
            llm_mod._apply_wake_fail("park", "gravitywell", "test prompt")

    mock_call_operator.assert_not_called()
    assert "gravitywell" in str(exc_info.value)
    assert "no paid fallback" in str(exc_info.value).lower()


def test_gwparked_error_is_operator_unreachable_subclass():
    """GWParkedError is a distinguishable OperatorUnreachableError subclass (callers that
    catch OperatorUnreachableError generically still catch it; callers that need the
    parked-vs-generic-fault distinction can catch GWParkedError specifically)."""
    from agents_core.llm import GWParkedError, OperatorUnreachableError

    assert issubclass(GWParkedError, OperatorUnreachableError)


def test_gravitywell_park_raises_gwparked_error_makes_no_paid_call():
    """call_operator('gravitywell', on_wake_fail='park') raises GWParkedError when GW is
    unreachable and never invokes the paid-operator (Claude CLI / submit_and_wait) path."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient, DoormanUnreachable

    provenance = []
    with patch.object(DoormanClient, "acquire", side_effect=DoormanUnreachable("doorman down")), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.claude_queue_sync.submit_and_wait") as mock_submit:

        with pytest.raises(llm_mod.GWParkedError):
            llm_mod.call_operator(
                "gravitywell", prompt="test",
                on_wake_fail="park",
                _provenance_out=provenance,
            )

    mock_submit.assert_not_called()
    assert ("doorman_unreachable", "gravitywell") in provenance
    assert not any(reason == "fallback" for reason, _ in provenance)


# ---------------------------------------------------------------------------
# GravityWell Retry Hardening Tests (gw-backend-retry-hardening-v0)
# ---------------------------------------------------------------------------

def _make_gw_response(text: str):
    """Minimal requests.Response mock for a successful GravityWell reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text, "reasoning_content": None}}]
    }
    return resp


def test_gravitywell_retries_on_timeout():
    """_call_gravitywell_backend retries on ReadTimeout and returns completion on success."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    timeout_err = req.exceptions.Timeout("read timeout")
    success_resp = _make_gw_response("recovered after timeout")

    with patch("agents_core.llm.requests.post",
               side_effect=[timeout_err, timeout_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry me")

    assert result == "recovered after timeout"
    # Verify backoff was called: attempt 0 (2s) and attempt 1 (4s)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(2)
    mock_sleep.assert_any_call(4)


def test_gravitywell_raises_on_persistent_timeout():
    """_call_gravitywell_backend raises OperatorUnreachableError after timeout exhaustion."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend, OperatorUnreachableError, GW_URL

    timeout_err = req.exceptions.Timeout("persistent timeout")

    with patch("agents_core.llm.requests.post", side_effect=timeout_err), \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError) as exc_info:
            _call_gravitywell_backend(prompt="will fail")

    exc = exc_info.value
    assert exc.url == GW_URL
    assert isinstance(exc.last_error, req.exceptions.Timeout)


def test_gravitywell_retries_on_chunked_encoding_error():
    """_call_gravitywell_backend retries on ChunkedEncodingError (stateless calls)."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    chunked_err = req.exceptions.ChunkedEncodingError("connection broken")
    success_resp = _make_gw_response("recovered from chunked error")

    with patch("agents_core.llm.requests.post",
               side_effect=[chunked_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry chunked")

    assert result == "recovered from chunked error"
    # Only one backoff (after attempt 0)
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(2)


def test_gravitywell_empty_content_returns_none():
    """_call_gravitywell_backend returns None for successful 200 with empty content (unchanged behavior)."""
    from agents_core.llm import _call_gravitywell_backend

    empty_resp = MagicMock()
    empty_resp.raise_for_status = MagicMock()
    empty_resp.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning_content": None}}]
    }

    with patch("agents_core.llm.requests.post", return_value=empty_resp):
        result = _call_gravitywell_backend(prompt="empty reply")

    assert result is None


def test_gravitywell_malformed_body_returns_none_via_generic_handler():
    """_call_gravitywell_backend returns None for malformed response (KeyError/IndexError)."""
    from agents_core.llm import _call_gravitywell_backend

    # Missing "choices" key
    malformed_resp = MagicMock()
    malformed_resp.raise_for_status = MagicMock()
    malformed_resp.json.return_value = {"error": "unexpected structure"}

    with patch("agents_core.llm.requests.post", return_value=malformed_resp):
        result = _call_gravitywell_backend(prompt="malformed response")

    assert result is None


def test_gravitywell_empty_choices_list_returns_none():
    """_call_gravitywell_backend returns None for empty choices list (IndexError)."""
    from agents_core.llm import _call_gravitywell_backend

    empty_choices_resp = MagicMock()
    empty_choices_resp.raise_for_status = MagicMock()
    empty_choices_resp.json.return_value = {"choices": []}

    with patch("agents_core.llm.requests.post", return_value=empty_choices_resp):
        result = _call_gravitywell_backend(prompt="empty choices")

    assert result is None


def test_gravitywell_backoff_timing_cumulative():
    """_call_gravitywell_backend total backoff (6s) does not exceed timeout_per_persona budget."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend, OperatorUnreachableError

    timeout_err = req.exceptions.Timeout("timeout")

    with patch("agents_core.llm.requests.post", side_effect=timeout_err), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        with pytest.raises(OperatorUnreachableError):
            _call_gravitywell_backend(prompt="backoff test")

    # Two backoffs: 2s (after attempt 0) and 4s (after attempt 1)
    assert mock_sleep.call_count == 2
    total_sleep = sum(call[0][0] for call in mock_sleep.call_args_list)
    assert total_sleep == 6
    assert total_sleep <= 6  # Should not exceed the 6s cap


def test_gravitywell_http_error_is_retryable():
    """_call_gravitywell_backend retries on HTTPError (e.g., 503)."""
    import requests as req
    from agents_core.llm import _call_gravitywell_backend

    http_err = req.exceptions.HTTPError("503 Service Unavailable")
    success_resp = _make_gw_response("recovered from 503")

    with patch("agents_core.llm.requests.post",
               side_effect=[http_err, success_resp]), \
         patch("agents_core.llm.time.sleep") as mock_sleep:
        result = _call_gravitywell_backend(prompt="retry http")

    assert result == "recovered from 503"
    mock_sleep.assert_called_once_with(2)


# ---------------------------------------------------------------------------
# GW Generation Guards (spec-review-gw-generation-guards-v0)
# ---------------------------------------------------------------------------

def _make_gw_sse_response(content: str, model: str = "gravitywell-122b"):
    """Fake streaming requests.Response: one content delta then a clean [DONE]."""
    import json as _json
    lines = [
        _json.dumps(
            {"model": model, "choices": [{"delta": {"content": content}}]}
        ).encode(),
        b"[DONE]",
    ]
    lines = [b"data: " + lines[0], b"data: " + lines[1]]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines = MagicMock(return_value=iter(lines))
    resp.close = MagicMock()
    return resp


def test_gravitywell_streaming_payload_includes_max_tokens_default(monkeypatch):
    """AC1: streaming payload carries max_tokens, defaulting to GW_MAX_TOKENS_DEFAULT."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.delenv("GW_MAX_TOKENS", raising=False)
    resp = _make_gw_sse_response("hello")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        result = _call_gravitywell_backend(prompt="hi")

    assert result == "hello"
    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 4096


def test_gravitywell_streaming_payload_max_tokens_env_override(monkeypatch):
    """AC1: GW_MAX_TOKENS env var overrides the default."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("GW_MAX_TOKENS", "777")
    resp = _make_gw_sse_response("hello")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="hi")

    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 777


def test_gravitywell_streaming_payload_max_tokens_caller_param_wins(monkeypatch):
    """AC1: an explicit max_tokens= caller param wins over the env default."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("GW_MAX_TOKENS", "777")
    resp = _make_gw_sse_response("hello")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="hi", max_tokens=42)

    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 42


def test_post_chat_completion_includes_max_tokens_default(monkeypatch):
    """AC1: the non-streaming sibling (_post_chat_completion) also carries max_tokens."""
    from agents_core.llm import _post_chat_completion

    monkeypatch.delenv("GW_MAX_TOKENS", raising=False)
    resp = _make_llama_response("hi there")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post:
        _post_chat_completion(
            base_url="http://x:1", model="m",
            messages=[{"role": "user", "content": "p"}],
        )

    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 4096


def test_gravitywell_llamacpp_backend_includes_repeat_penalty(monkeypatch):
    """AC2: the llama.cpp payload branch carries a repeat_penalty (env-overridable)."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.delenv("GW_REPEAT_PENALTY", raising=False)
    resp = _make_gw_sse_response("hello")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="hi")

    payload = mock_post.call_args[1]["json"]
    assert payload["repeat_penalty"] == 1.1


def test_gravitywell_repeat_penalty_env_override(monkeypatch):
    """AC2: GW_REPEAT_PENALTY env var overrides the default on the llamacpp branch."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("GW_BACKEND", "llamacpp")
    monkeypatch.setenv("GW_REPEAT_PENALTY", "1.3")
    resp = _make_gw_sse_response("hello")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="hi")

    payload = mock_post.call_args[1]["json"]
    assert payload["repeat_penalty"] == 1.3


def test_gravitywell_vllm_backend_omits_repeat_penalty(monkeypatch):
    """AC2: the vLLM branch never gets a repeat_penalty (llamacpp-only guard)."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("GW_BACKEND", "vllm")
    resp = _make_gw_sse_response("hello", model="gravitywell-27b")

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="hi")

    payload = mock_post.call_args[1]["json"]
    assert "repeat_penalty" not in payload
    assert "cache_prompt" not in payload


def test_gravitywell_hard_ceiling_cull_salvages_partial(monkeypatch, tmp_path):
    """AC3: on hard_ceiling_exceeded with buffered content, the accumulated partial is
    returned (with a degraded marker) instead of None."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    cull = ("hard_ceiling_exceeded", 300.2, 0.0)

    with patch("agents_core.llm._gw_stream_attempt",
               return_value=("partial voice content", cull, "gravitywell-122b")), \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        result = _call_gravitywell_backend(prompt="runaway turn")

    assert result is not None
    assert result.startswith("partial voice content")
    assert "gw-degraded" in result
    assert "hard_ceiling_exceeded" in result


def test_gravitywell_hard_ceiling_cull_persists_partial_artifact(monkeypatch, tmp_path):
    """AC4: a culled stream's partial is written to a run artifact path."""
    from agents_core.llm import _call_gravitywell_backend

    monkeypatch.setenv("ROOM_ROOT", str(tmp_path))
    cull = ("hard_ceiling_exceeded", 300.2, 0.0)

    with patch("agents_core.llm._gw_stream_attempt",
               return_value=("the runaway partial", cull, "gravitywell-122b")), \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        _call_gravitywell_backend(prompt="runaway turn")

    cull_dir = tmp_path / "council" / "gw-cull"
    files = list(cull_dir.glob("*.txt"))
    assert len(files) == 1
    assert "the runaway partial" in files[0].read_text()


def test_gravitywell_no_token_cull_still_returns_none(monkeypatch):
    """AC3 (negative): a cull with no accumulated text (nothing ever arrived) stays None."""
    from agents_core.llm import _call_gravitywell_backend

    cull = ("first_token_grace_exceeded", 600.0, 600.0)

    with patch("agents_core.llm._gw_stream_attempt", return_value=(None, cull, None)), \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        result = _call_gravitywell_backend(prompt="dead stream")

    assert result is None


def test_gravitywell_healthy_stream_unaffected_by_guards(monkeypatch):
    """AC7 (regression): a healthy (uncalled) turn returns full text, no degraded marker."""
    from agents_core.llm import _call_gravitywell_backend

    with patch("agents_core.llm._gw_stream_attempt",
               return_value=("clean full response", None, "gravitywell-122b")), \
         patch("agents_core.llm._gw_probe_served_model", return_value=None):
        result = _call_gravitywell_backend(prompt="normal turn")

    assert result == "clean full response"
    assert "gw-degraded" not in result


def test_call_operator_gravitywell_records_stream_culled_provenance():
    """AC6: call_operator('gravitywell') records 'stream_culled' (not 'success') provenance
    when the backend returns a degraded/salvaged partial."""
    from agents_core import llm as llm_mod
    from agents_core.doorman_client import DoormanClient

    degraded = "partial content" + llm_mod._gw_degraded_marker("hard_ceiling_exceeded", 300.2, 0.0)
    provenance = []

    with patch.object(DoormanClient, "acquire", return_value={"status": "serving"}), \
         patch.object(DoormanClient, "release"), \
         patch("agents_core.llm._call_gravitywell_backend", return_value=degraded):

        result = llm_mod.call_operator(
            "gravitywell", prompt="test",
            on_wake_fail="sonnet",
            _provenance_out=provenance,
        )

    assert result == degraded
    assert ("stream_culled", "gravitywell") in provenance
    assert ("success", "gravitywell") not in provenance


# ---------------------------------------------------------------------------
# Swarm Tests: call_swarm, swarm_serving, swarm_model
# ---------------------------------------------------------------------------

def _make_swarm_response(text: str):
    """Minimal requests.Response mock for a successful swarm reply."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}]
    }
    return resp


def _make_models_response(models: list[str]):
    """Mock /v1/models response with a list of model IDs."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": [{"id": model} for model in models]
    }
    return resp


def test_swarm_serving_returns_true_when_both_healthy():
    """swarm_serving() returns True when /v1/models (200+non-empty) and /health (200)."""
    from agents_core.llm import swarm_serving

    models_resp = _make_models_response(["Qwen2.5-3B"])
    health_resp = MagicMock()
    health_resp.raise_for_status = MagicMock()

    with patch("agents_core.llm.requests.get") as mock_get:
        mock_get.side_effect = [models_resp, health_resp]
        result = swarm_serving("http://localhost:8081")

    assert result is True
    assert mock_get.call_count == 2


def test_swarm_serving_false_when_models_empty():
    """swarm_serving() returns False when /v1/models is empty."""
    from agents_core.llm import swarm_serving

    empty_models = MagicMock()
    empty_models.raise_for_status = MagicMock()
    empty_models.json.return_value = {"data": []}

    with patch("agents_core.llm.requests.get", return_value=empty_models):
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_serving_false_when_health_down():
    """swarm_serving() returns False when /health fails."""
    from agents_core.llm import swarm_serving

    models_resp = _make_models_response(["Qwen2.5-3B"])
    health_err = MagicMock()
    health_err.raise_for_status = MagicMock(side_effect=Exception("health check failed"))

    with patch("agents_core.llm.requests.get") as mock_get:
        mock_get.side_effect = [models_resp, health_err]
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_serving_false_when_unreachable():
    """swarm_serving() returns False when endpoint is unreachable."""
    from agents_core.llm import swarm_serving
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = swarm_serving("http://localhost:8081")

    assert result is False


def test_swarm_model_returns_model_id():
    """swarm_model() returns the first model id from /v1/models."""
    from agents_core.llm import swarm_model

    models_resp = _make_models_response(["Qwen2.5-3B", "other-model"])

    with patch("agents_core.llm.requests.get", return_value=models_resp):
        result = swarm_model("http://localhost:8081")

    assert result == "Qwen2.5-3B"


def test_swarm_model_returns_none_when_unreachable():
    """swarm_model() returns None when endpoint is unreachable."""
    from agents_core.llm import swarm_model
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = swarm_model("http://localhost:8081")

    assert result is None


def test_call_swarm_single_prompt_success():
    """call_swarm() with one prompt returns [result]."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("swarm answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp):
        result = call_swarm(["hello"])

    assert result == ["swarm answer"]


def test_call_swarm_multiple_prompts_preserves_order():
    """call_swarm() with multiple prompts returns results in input order."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])

    def mock_post_func(url, json=None, timeout=None):
        """Return response keyed on prompt content."""
        if json and "messages" in json:
            messages = json["messages"]
            user_msg = next((m for m in messages if m.get("role") == "user"), {})
            prompt_content = user_msg.get("content", "")
            if prompt_content == "prompt1":
                return _make_swarm_response("answer1")
            elif prompt_content == "prompt2":
                return _make_swarm_response("answer2")
        return _make_swarm_response("default")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", side_effect=mock_post_func):
        result = call_swarm(["prompt1", "prompt2"])

    assert len(result) == 2
    assert result[0] == "answer1"
    assert result[1] == "answer2"


def test_call_swarm_returns_all_none_when_not_serving():
    """call_swarm() returns [None, None, ...] when swarm not serving."""
    from agents_core.llm import call_swarm
    import requests as req

    with patch("agents_core.llm.requests.get",
               side_effect=req.exceptions.ConnectionError("down")):
        result = call_swarm(["p1", "p2", "p3"])

    assert result == [None, None, None]


def test_call_swarm_isolates_per_prompt_errors():
    """call_swarm() maps per-prompt errors to None, preserves successes."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    success_resp = _make_swarm_response("success")
    error_resp = MagicMock()
    error_resp.raise_for_status = MagicMock(side_effect=Exception("error"))

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post") as mock_post:
        mock_post.side_effect = [success_resp, error_resp, success_resp]
        result = call_swarm(["p1", "p2", "p3"])

    assert len(result) == 3
    assert result[0] == "success"
    assert result[1] is None
    assert result[2] == "success"


def test_call_swarm_uses_provided_model():
    """call_swarm() uses the model param when provided (no /v1/models call)."""
    from agents_core.llm import call_swarm

    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get") as mock_get, \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        result = call_swarm(["hello"], model="custom-model")

    # Should NOT call /v1/models when model is provided
    mock_get.assert_not_called()
    assert result == ["answer"]

    # POST payload should include the custom model
    posted_data = mock_post.call_args[1]["json"]
    assert posted_data["model"] == "custom-model"


def test_call_swarm_does_not_import_doorman_client():
    """call_swarm module scope does NOT import DoormanClient (static guarantee)."""
    import agents_core.llm as llm_module

    # Verify that DoormanClient is not in the module's top-level namespace
    # (it may be imported inside call_operator, but not at module scope)
    assert "DoormanClient" not in dir(llm_module)
    assert not hasattr(llm_module, "DoormanClient")


def test_call_swarm_acquire_never_called():
    """Behavioral test: call_swarm never calls DoormanClient.acquire (even if imported)."""
    from agents_core.llm import call_swarm
    from agents_core.doorman_client import DoormanClient

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch.object(DoormanClient, "acquire") as mock_acquire, \
         patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp):
        result = call_swarm(["hello"])

    mock_acquire.assert_not_called()
    assert result == ["answer"]


def test_call_swarm_empty_prompts_list_returns_empty():
    """call_swarm([]) returns []."""
    from agents_core.llm import call_swarm

    result = call_swarm([])

    assert result == []


def test_call_swarm_respects_max_concurrent():
    """call_swarm() fans out bounded by max_concurrent (default SWARM_MAX_CONCURRENT)."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    # Patch ThreadPoolExecutor at the agents_core.llm module level where it's used
    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp), \
         patch("agents_core.llm.ThreadPoolExecutor") as mock_executor_class:

        mock_executor = MagicMock()
        mock_executor_class.return_value = mock_executor
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=None)
        mock_executor.submit = MagicMock()
        mock_executor.submit.return_value.result = MagicMock(return_value=(0, "answer", "success"))

        # Patch as_completed to return futures in order
        with patch("agents_core.llm.as_completed") as mock_as_completed:
            mock_as_completed.return_value = []
            call_swarm(["p1", "p2"], max_concurrent=2)

        mock_executor_class.assert_called_once_with(max_workers=2)


def test_call_swarm_post_includes_system_prompt():
    """call_swarm() includes system prompt in message payload."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        call_swarm(["hello"], system="You are helpful.")

    posted_data = mock_post.call_args[1]["json"]
    messages = posted_data["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# debt/call-swarm-cannot-disable-thinking-2026-06-22 regression guards
# ---------------------------------------------------------------------------

def test_call_swarm_default_sends_enable_thinking_false():
    """call_swarm() default (think=False) explicitly sends enable_thinking:false."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        call_swarm(["hello"])

    posted_data = mock_post.call_args[1]["json"]
    assert posted_data["chat_template_kwargs"] == {"enable_thinking": False}


def test_call_swarm_think_true_sends_enable_thinking_true():
    """call_swarm(think=True) sends enable_thinking:true."""
    from agents_core.llm import call_swarm

    models_resp = _make_models_response(["Qwen2.5-3B"])
    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.get", return_value=models_resp), \
         patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        call_swarm(["hello"], think=True)

    posted_data = mock_post.call_args[1]["json"]
    assert posted_data["chat_template_kwargs"] == {"enable_thinking": True}


def test_post_chat_completion_think_false_sends_enable_thinking_false():
    """_post_chat_completion(think=False) sends enable_thinking:false (not omitted)."""
    from agents_core.llm import _post_chat_completion

    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        _post_chat_completion(
            base_url="http://fake:8080",
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
            think=False,
        )

    posted_data = mock_post.call_args[1]["json"]
    assert posted_data["chat_template_kwargs"] == {"enable_thinking": False}


def test_post_chat_completion_no_thinking_omits_chat_template_kwargs():
    """_post_chat_completion(_no_thinking=True) structurally omits chat_template_kwargs."""
    from agents_core.llm import _post_chat_completion

    completion_resp = _make_swarm_response("answer")

    with patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        _post_chat_completion(
            base_url="http://fake:8080",
            model="fake-model",
            messages=[{"role": "user", "content": "hello"}],
            think=False,
            _no_thinking=True,
        )

    posted_data = mock_post.call_args[1]["json"]
    assert "chat_template_kwargs" not in posted_data


def test_call_operator_quest_omits_chat_template_kwargs():
    """call_operator('quest') pins _no_thinking=True — payload stays byte-identical
    to pre-fix behavior (no chat_template_kwargs), a deterministic guard against
    silently re-drifting QUEST into thinking-off without a decision."""
    from agents_core import llm as llm_mod

    completion_resp = _make_swarm_response("quest answer")

    with patch("agents_core.llm.requests.post", return_value=completion_resp) as mock_post:
        result = llm_mod.call_operator(
            "quest",
            prompt="hello",
        )

    assert result == "quest answer"
    posted_data = mock_post.call_args[1]["json"]
    assert "chat_template_kwargs" not in posted_data


# ---------------------------------------------------------------------------
# _post_chat_completion json_object degrade (agents-core-post-chat-json-object-degrade-v0)
# ---------------------------------------------------------------------------

def _make_json_response(body: dict):
    """A 200 requests.Response mock whose message content is the JSON-encoded body."""
    import json as _json
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [{"message": {"content": _json.dumps(body), "reasoning_content": None}}]
    }
    return resp


def _make_400_error():
    """An HTTPError as raised by Response.raise_for_status() for a 400, with
    .response.status_code populated (mirrors requests' real behavior)."""
    import requests as req
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    err = req.exceptions.HTTPError("400 Client Error: Bad Request")
    err.response = fake_resp
    return err


def test_post_chat_completion_json_mode_happy_path_byte_identical():
    """DoD-1: json_mode=True, 200 on first try — exactly one POST, response_format
    present, every other payload field unchanged, degrade path never entered."""
    from agents_core.llm import _post_chat_completion

    resp = _make_json_response({"answer": "x", "citations": []})

    with patch("agents_core.llm.requests.post", return_value=resp) as mock_post, \
         patch("agents_core.llm._log") as mock_log:
        result = _post_chat_completion(
            base_url="http://fake:1234", model="m",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True, cache_prompt=True,
        )

    assert mock_post.call_count == 1
    payload = mock_post.call_args[1]["json"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["model"] == "m"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["temperature"] == 0.7
    assert payload["cache_prompt"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert result is not None
    mock_log.warning.assert_not_called()


def test_post_chat_completion_degrades_on_400_then_succeeds():
    """DoD-2/DoD-3: first POST 400s on json_object, second POST (no response_format)
    succeeds with a parseable JSON body — the function returns that text."""
    from agents_core.llm import _post_chat_completion

    body = {"answer": "x", "citations": []}
    success_resp = _make_json_response(body)

    with patch("agents_core.llm.requests.post",
               side_effect=[_make_400_error(), success_resp]) as mock_post, \
         patch("agents_core.llm.time.sleep"):
        result = _post_chat_completion(
            base_url="http://fake:1234", model="m",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

    assert mock_post.call_count == 2
    first_payload = mock_post.call_args_list[0][1]["json"]
    second_payload = mock_post.call_args_list[1][1]["json"]
    assert first_payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in second_payload
    for key in ("model", "messages", "temperature", "max_tokens", "chat_template_kwargs"):
        assert first_payload[key] == second_payload[key]

    assert isinstance(result, str)
    assert json.loads(result) == body


def test_post_chat_completion_json_mode_false_400_unchanged():
    """DoD-4: json_mode=False — a 400 is just a normal transient error; existing
    retry ladder applies, no response_format ever appears, no extra request."""
    from agents_core.llm import _post_chat_completion, OperatorUnreachableError

    with patch("agents_core.llm.requests.post",
               side_effect=_make_400_error()) as mock_post, \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError):
            _post_chat_completion(
                base_url="http://fake:1234", model="m",
                messages=[{"role": "user", "content": "hi"}],
                json_mode=False,
            )

    assert mock_post.call_count == 3
    for c in mock_post.call_args_list:
        assert "response_format" not in c[1]["json"]


def test_post_chat_completion_degrade_bounded_then_terminal():
    """DoD-5: both the json_object POST and the degraded POST (and its own
    retries) keep 400ing — degrade fires exactly once, then the function
    follows the existing terminal path with a bounded number of requests."""
    import itertools
    from agents_core.llm import _post_chat_completion, OperatorUnreachableError

    with patch("agents_core.llm.requests.post",
               side_effect=itertools.repeat(_make_400_error())) as mock_post, \
         patch("agents_core.llm.time.sleep"):
        with pytest.raises(OperatorUnreachableError):
            _post_chat_completion(
                base_url="http://fake:1234", model="m",
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True, max_retries=3,
            )

    # Bounded: 1 json_object attempt + max_retries degraded attempts, then terminal.
    assert mock_post.call_count == 4
    payloads = [c[1]["json"] for c in mock_post.call_args_list]
    assert "response_format" in payloads[0]
    assert all("response_format" not in p for p in payloads[1:])


def test_post_chat_completion_degrade_fires_on_final_transient_attempt():
    """DoD-6: the qualifying 400 lands on what would have been the last transient
    attempt under the old (non-degraded) ladder — the degraded POST must still fire."""
    from agents_core.llm import _post_chat_completion
    import requests as req

    body = {"answer": "recovered"}
    with patch("agents_core.llm.requests.post",
               side_effect=[
                   req.exceptions.Timeout("t1"),
                   req.exceptions.Timeout("t2"),
                   _make_400_error(),
                   _make_json_response(body),
               ]) as mock_post, \
         patch("agents_core.llm.time.sleep"):
        result = _post_chat_completion(
            base_url="http://fake:1234", model="m",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True, max_retries=3,
        )

    assert mock_post.call_count == 4
    assert json.loads(result) == body
    last_payload = mock_post.call_args_list[-1][1]["json"]
    assert "response_format" not in last_payload


def test_post_chat_completion_transient_errors_preserved():
    """DoD-7: ConnectionError, Timeout, and non-400 HTTPError (503) still retry with
    the existing backoff and raise OperatorUnreachableError after max_retries —
    the degrade never engages for non-400 failures."""
    import itertools
    from agents_core.llm import _post_chat_completion, OperatorUnreachableError
    import requests as req

    def make_503():
        fake_resp = MagicMock()
        fake_resp.status_code = 503
        err = req.exceptions.HTTPError("503 Service Unavailable")
        err.response = fake_resp
        return err

    for err_factory in (
        lambda: req.exceptions.ConnectionError("conn"),
        lambda: req.exceptions.Timeout("timeout"),
        make_503,
    ):
        with patch("agents_core.llm.requests.post",
                   side_effect=itertools.repeat(err_factory())) as mock_post, \
             patch("agents_core.llm.time.sleep") as mock_sleep:
            with pytest.raises(OperatorUnreachableError):
                _post_chat_completion(
                    base_url="http://fake:1234", model="m",
                    messages=[{"role": "user", "content": "hi"}],
                    json_mode=True, max_retries=3,
                )

        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2)
        mock_sleep.assert_any_call(4)


def test_post_chat_completion_signature_unchanged():
    """DoD-8: the public signature carries no new required parameter."""
    import inspect
    from agents_core.llm import _post_chat_completion

    sig = inspect.signature(_post_chat_completion)
    required = [
        name for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
    ]
    assert required == ["base_url", "model", "messages"]


def test_post_chat_completion_degrade_emits_warning_log(caplog):
    """DoD-9: the degrade emits a WARNING-level log naming the response_format
    pivot; no in-core JSON validation happens (there is none by design)."""
    import logging
    from agents_core.llm import _post_chat_completion

    success_resp = _make_json_response({"answer": "x"})

    with patch("agents_core.llm.requests.post",
               side_effect=[_make_400_error(), success_resp]), \
         patch("agents_core.llm.time.sleep"), \
         caplog.at_level(logging.WARNING, logger="agents_core.llm"):
        _post_chat_completion(
            base_url="http://fake:1234", model="m",
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "response_format" in r.getMessage() and "400" in r.getMessage()
        for r in warnings
    )


# ---------------------------------------------------------------------------
# gravitywell-27b registry row (agents-core-gw-models-register-27b-seat-v0)
# ---------------------------------------------------------------------------
#
# R1: register the live 27B solo-posture served id in gw_models.yaml.
# DoD 1: round-trip every alias of the new row, and pin that appending at
# end-of-file left every PRE-EXISTING alias lookup byte-identical.
# DoD 3: gw_serving_state() against a stubbed models endpoint serving
# gravitywell-27b resolves unknown_model=False.

def test_gw27b_registry_lookup_round_trips_all_aliases():
    """DoD 1: lookup by each alias of the new row (canonical_id, mode_alias,
    operator_alias, display_label) returns the same entry, and lookup("solo")
    resolves to the 27B row specifically (mode_alias "solo" is the only
    live-lookup key unique to this row — operator_alias/canonical_id/
    display_label are shared with or reused across other rows)."""
    from agents_core.llm import _gw_registry_lookup

    by_canonical_id = _gw_registry_lookup("gravitywell-27b")
    assert by_canonical_id is not None
    assert by_canonical_id.canonical_id == "gravitywell-27b"
    assert by_canonical_id.mode_alias == "solo"
    assert by_canonical_id.operator_alias == "gravitywell"
    assert by_canonical_id.display_label == "qwen3.8-27b-uncensored-nvfp4"
    assert by_canonical_id.weights_hint == "Qwen3.8-27B-Uncensored-NVFP4"

    by_display_label = _gw_registry_lookup("qwen3.8-27b-uncensored-nvfp4")
    assert by_display_label == by_canonical_id

    by_mode_alias = _gw_registry_lookup("solo")
    assert by_mode_alias == by_canonical_id


def test_gw27b_registry_append_preserves_existing_alias_ordering():
    """DoD 1: alias-resolution ordering is pinned — lookup("big"),
    lookup("dual"), lookup("gravitywell") resolve to the SAME rows as before
    this change. _gw_registry_lookup returns the first file-order match, so
    appending gravitywell-27b at end-of-file must not shift any of these
    (byte-identical rows alone would not catch an ordering regression)."""
    from agents_core.llm import _gw_registry_lookup

    big = _gw_registry_lookup("big")
    assert big is not None
    assert big.canonical_id == "gravitywell-122b"

    dual = _gw_registry_lookup("dual")
    assert dual is not None
    assert dual.canonical_id == "gravitywell-a3b-nvfp4"

    operator = _gw_registry_lookup("gravitywell")
    assert operator is not None
    assert operator.canonical_id == "gravitywell-122b"


def _fake_gw_requests_get(url, timeout=None, **kwargs):
    """Hermetic stand-in for requests.get across all three gw_serving_state
    sources (endpoint /health, endpoint+slot2 /v1/models, flip-controller
    /v0/status) — dispatches on URL suffix, never a live call."""
    resp = MagicMock()
    if url.endswith("/health"):
        resp.status_code = 200
        resp.json.return_value = {}
    elif url.endswith("/v1/models"):
        resp.status_code = 200
        resp.json.return_value = {"data": [{"id": "gravitywell-27b"}]}
    elif url.endswith("/v0/status"):
        resp.status_code = 200
        resp.json.return_value = {"mode": "dual", "units": {}, "in_flight_flip": False}
    else:
        resp.status_code = 404
    return resp


def test_gw_serving_state_27b_resolves_unknown_model_false():
    """DoD 3: gw_serving_state() against a stubbed /v1/models serving
    gravitywell-27b returns unknown_model=False and a canonical entry —
    before R1 this served id had no registry row and unknown_model was True
    (agents-core-gw-models-register-27b-seat-v0 problem statement)."""
    from agents_core.llm import gw_serving_state

    with patch("agents_core.llm.requests.get", side_effect=_fake_gw_requests_get):
        state = gw_serving_state(endpoint="http://fake-gw:8081")

    assert state.served_id == "gravitywell-27b"
    assert state.unknown_model is False
    assert state.canonical is not None
    assert state.canonical.canonical_id == "gravitywell-27b"
    assert state.canonical.mode_alias == "solo"
