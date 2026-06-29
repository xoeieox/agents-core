"""AgentWorld predictor for queue-runner calibration (agents-core-agentworld-scorer-v0).

Encodes (state_before, event) into an in-context AgentWorld prompt, calls the
configured endpoint, and returns a predicted state_after.

AgentWorld transport is injectable: any callable matching AgentWorldClient can be
swapped in, so tests run fully offline with a deterministic fake.

Parse failures are NOT silently excluded.  They are recorded as parse_ok=False
and scored as fidelity=0 in the aggregate (a distinct zero-valued category).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

log = logging.getLogger("calibration.predictor")

# ---------------------------------------------------------------------------
# Transport protocol (injectable)
# ---------------------------------------------------------------------------

class AgentWorldClient(Protocol):
    """Interface the predictor uses to call AgentWorld.

    Implementations: _HttpAgentWorldClient (real), FakeAgentWorldClient (tests).
    """

    def predict(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        chat_template_kwargs: dict,
    ) -> tuple[str, str]:
        """Return (content, finish_reason).  finish_reason is 'stop' or 'length'."""
        ...


# ---------------------------------------------------------------------------
# Real HTTP client (wraps the vLLM OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class _HttpAgentWorldClient:
    """Thin wrapper over the vLLM OpenAI-compatible /v1/chat/completions endpoint.

    Uses only stdlib + httpx (already a project dep).
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8090/v1") -> None:
        self.endpoint = endpoint.rstrip("/")

    def predict(
        self,
        messages: list[dict],
        model: str = "agentworld",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
        chat_template_kwargs: dict | None = None,
    ) -> tuple[str, str]:
        import httpx

        if chat_template_kwargs is None:
            chat_template_kwargs = {"enable_thinking": False}

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
        }
        resp = httpx.post(
            f"{self.endpoint}/chat/completions",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # vLLM: content in message.content (NOT reasoning / reasoning_content)
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "stop") or "stop"
        return content, finish_reason


# ---------------------------------------------------------------------------
# System prompts for the queue-runner environment
# ---------------------------------------------------------------------------

# with-rules: v0 verbatim prompt (instruction-following benchmark, NOT fidelity)
_SYSTEM_PROMPT_WITH_RULES = """\
You are AgentWorld — a learned world model of the Lapis StarHouse claude-queue runner.

## Queue-runner state machine

The queue-runner has four lifecycle events:

- submitted  → pending count +1; no change to active/completed/failed
- claimed    → pending count -1; active count +1; job enters in_flight
- completed  → active count -1; completed count +1; job leaves in_flight
- failed     → active count -1; failed count +1; job leaves in_flight

in_flight is the list of currently active jobs.  Each job has: id, task_type,
model (may be null), claimed_at (ISO timestamp), stasis_duration (seconds since
claimed_at at snapshot time).

workers.capacity is static (CLAUDE_QUEUE_WORKERS env, typically 2).
workers.utilization = len(in_flight) / capacity.

stasis_duration (queue-level): seconds since the previous lifecycle event.
  After an event fires, state_after always has stasis_duration = 0.
stasis_velocity: stasis_duration_this - stasis_duration_prev.
  In state_after it is always 0 (reset on event).

## Your task

Given state_before and the incoming event, predict the exact state_after JSON
object.  Output ONLY valid JSON matching the State schema below, with no
explanation, markdown fences, or surrounding text.

## State schema

{
  "counts": {"pending": <int>, "active": <int>, "completed": <int>, "failed": <int>},
  "in_flight": [
    {"id": "<str>", "task_type": "<str>", "model": "<str|null>",
     "claimed_at": "<ISO8601>", "stasis_duration": <float>}
  ],
  "workers": {"capacity": <int>, "utilization": <float>},
  "stasis_duration": <float>,
  "stasis_velocity": <float>
}
"""

# no-rules: role + schema + task only — the state-machine rule list is OMITTED so that
# fidelity measures AgentWorld's learned prior rather than its ability to follow stated rules.
_SYSTEM_PROMPT_NO_RULES = """\
You are AgentWorld — a learned world model of the Lapis StarHouse claude-queue runner.

in_flight is the list of currently active jobs.  Each job has: id, task_type,
model (may be null), claimed_at (ISO timestamp), stasis_duration (seconds since
claimed_at at snapshot time).

workers.capacity is static (CLAUDE_QUEUE_WORKERS env, typically 2).
workers.utilization = len(in_flight) / capacity.

stasis_duration (queue-level): seconds since the previous lifecycle event.
  After an event fires, state_after always has stasis_duration = 0.
stasis_velocity: stasis_duration_this - stasis_duration_prev.
  In state_after it is always 0 (reset on event).

## Your task

Given state_before and the incoming event, predict the exact state_after JSON
object.  Output ONLY valid JSON matching the State schema below, with no
explanation, markdown fences, or surrounding text.

## State schema

{
  "counts": {"pending": <int>, "active": <int>, "completed": <int>, "failed": <int>},
  "in_flight": [
    {"id": "<str>", "task_type": "<str>", "model": "<str|null>",
     "claimed_at": "<ISO8601>", "stasis_duration": <float>}
  ],
  "workers": {"capacity": <int>, "utilization": <float>},
  "stasis_duration": <float>,
  "stasis_velocity": <float>
}
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    state_before: dict,
    event: dict,
    few_shot_examples: list[dict] | None = None,
    prompt_mode: str = "no-rules",
) -> list[dict]:
    """Build the messages list for an AgentWorld chat completion.

    few_shot_examples: list of transition dicts (state_before/event/state_after).
    They are inserted as (user, assistant) turns before the actual query.
    prompt_mode: 'no-rules' (default) omits the state-machine rule list to test
    learned prior; 'with-rules' uses the v0 verbatim prompt (instruction-following).
    """
    system_prompt = (
        _SYSTEM_PROMPT_WITH_RULES if prompt_mode == "with-rules" else _SYSTEM_PROMPT_NO_RULES
    )
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    for ex in (few_shot_examples or []):
        user_content = _make_user_content(ex["state_before"], ex["event"])
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["state_after"], separators=(",", ":")),
        })

    messages.append({
        "role": "user",
        "content": _make_user_content(state_before, event),
    })
    return messages


def _make_user_content(state_before: dict, event: dict) -> str:
    return (
        f"state_before:\n{json.dumps(state_before, indent=2)}\n\n"
        f"event:\n{json.dumps(event, indent=2)}\n\n"
        "Predict state_after:"
    )


# ---------------------------------------------------------------------------
# Resilient JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, tolerating code fences and prose."""
    text = text.strip()
    # Strip code fences: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Find the first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Calls AgentWorld to predict state_after for a (state_before, event) pair."""

    def __init__(
        self,
        client: AgentWorldClient | None = None,
        endpoint: str = "http://127.0.0.1:8090/v1",
        model: str = "agentworld",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
        prompt_mode: str = "no-rules",
    ) -> None:
        self.client: AgentWorldClient = client or _HttpAgentWorldClient(endpoint)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.prompt_mode = prompt_mode

    def predict(
        self,
        state_before: dict,
        event: dict,
        few_shot_examples: list[dict] | None = None,
    ) -> dict:
        """Return a prediction result dict.

        Always returns a dict with keys:
          - predicted_state: dict | None
          - parse_ok: bool
          - raw_response: str
          - parse_failure_type: str | None  ('truncated', 'malformed', 'error', or None)
        """
        messages = build_prompt(state_before, event, few_shot_examples, prompt_mode=self.prompt_mode)
        try:
            raw, finish_reason = self.client.predict(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception as exc:
            log.warning("AgentWorld call failed: %s", exc)
            return {
                "predicted_state": None,
                "parse_ok": False,
                "raw_response": str(exc),
                "parse_failure_type": "error",
            }

        is_truncated = finish_reason == "length"
        parsed = _extract_json(raw)
        if parsed is None:
            failure_type = "truncated" if is_truncated else "malformed"
            log.warning("failed to parse AgentWorld response (type=%s): %r", failure_type, raw[:200])
            return {
                "predicted_state": None,
                "parse_ok": False,
                "raw_response": raw,
                "parse_failure_type": failure_type,
            }

        return {
            "predicted_state": parsed,
            "parse_ok": True,
            "raw_response": raw,
            "parse_failure_type": None,
        }


# ---------------------------------------------------------------------------
# Fake client for offline tests
# ---------------------------------------------------------------------------

class FakeAgentWorldClient:
    """Deterministic test double.  predict() applies exact queue-runner rules."""

    def predict(
        self,
        messages: list[dict],
        model: str = "agentworld",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
        chat_template_kwargs: dict | None = None,
    ) -> tuple[str, str]:
        # Extract state_before and event from the last user message
        user_msg = next(
            m["content"] for m in reversed(messages) if m["role"] == "user"
        )
        state_before = json.loads(
            re.search(r"state_before:\n(\{.*?\})\n\nevent:", user_msg, re.DOTALL).group(1)
        )
        event = json.loads(
            re.search(r"event:\n(\{.*?})\n\nPredict", user_msg, re.DOTALL).group(1)
        )
        predicted = _apply_rules(state_before, event)
        return json.dumps(predicted), "stop"


def _apply_rules(state_before: dict, event: dict) -> dict:
    """Apply exact queue-runner state-machine rules (used by FakeAgentWorldClient)."""
    import copy
    s = copy.deepcopy(state_before)
    ev = event.get("event", "")
    eid = event.get("id", "")

    counts = s["counts"]
    in_flight = s.get("in_flight", [])

    if ev == "submitted":
        counts["pending"] = max(0, counts["pending"] + 1)
    elif ev == "claimed":
        counts["pending"] = max(0, counts["pending"] - 1)
        counts["active"] = counts["active"] + 1
        in_flight.append({
            "id": eid,
            "task_type": event.get("task_type", "unknown"),
            "model": event.get("model"),
            "claimed_at": event.get("timestamp", ""),
            "stasis_duration": 0.0,
        })
    elif ev == "completed":
        counts["active"] = max(0, counts["active"] - 1)
        counts["completed"] = counts["completed"] + 1
        in_flight = [j for j in in_flight if j["id"] != eid]
    elif ev == "failed":
        counts["active"] = max(0, counts["active"] - 1)
        counts["failed"] = counts["failed"] + 1
        in_flight = [j for j in in_flight if j["id"] != eid]

    cap = s["workers"]["capacity"]
    utilization = len(in_flight) / cap if cap > 0 else 0.0
    return {
        "counts": counts,
        "in_flight": in_flight,
        "workers": {"capacity": cap, "utilization": round(utilization, 4)},
        "stasis_duration": 0.0,
        "stasis_velocity": 0.0,
    }
