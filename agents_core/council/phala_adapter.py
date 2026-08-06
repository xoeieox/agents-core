"""Phala adapter for Mirror Council voicing.

Routes council voicing through the local phala-test-key.service (a sealed,
non-Anthropic TEE inference seat) via call_operator("phala", ...). Implements
the LanguageModel protocol (duck-typed; does not import from lapis_engine),
mirroring GravityWellAdapter — not LlamaAdapter — because this is a Council
voicing surface, not a generic lapis-engine client.

Fail-closed by design (agents-core-phala-gate-voicing-v0): Phala has no wake
concept — it is a single sealed HTTP endpoint on loopback, not a box that can
be woken — so call_operator("phala", ...) raises PhalaOperatorUnavailable on
any unreachable shape and never silently escalates to a paid Anthropic
fallback. This adapter does not catch that exception; it propagates to the
caller exactly as GWParkedError does for GravityWellAdapter.

EXPLICIT BANNER (ratified by Erah 2026-08-04): a Phala-voiced verdict must
never read as a 122B one. `voicing_events` is populated on every successful
call with the model id actually requested (Phala's `model=` catalog swap
means the served model can differ run to run — see MODEL_CATALOG's
`served_model_unreported` note in phala_tee_proxy.py), so
`council/cli.py`'s existing requested-vs-effective voicing machinery can
surface "phala:<model-id>" by name rather than folding it into an anonymous
"voicing ok" line.

CONFIDENTIALITY IS NOT CONTENT-TRUST — see agents_core/phala_tee.py's module
docstring. A sealed channel says nothing about the judgment quality of the
model behind it.
"""

from __future__ import annotations
from dataclasses import dataclass

from agents_core.llm import call_operator, OPERATOR_DEFAULTS


def _flatten_messages(messages) -> str:
    """Flatten a message list to a single user prompt.

    Same shape as gravitywell_adapter._flatten_messages: single-message fast
    path, role-tagged join for multi-turn. Duck-typed (reads m.role/m.content),
    does not import Message from lapis_engine.
    """
    if len(messages) == 1:
        return messages[0].content

    parts = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", str(m))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@dataclass
class PhalaAdapter:
    """LanguageModel adapter routing council voicing to Phala's sealed TEE
    seat via call_operator("phala", ...).

    No doorman, no lease, no wake-fail policy to configure — unreachable
    always raises PhalaOperatorUnavailable (llm.py), never a silent degrade
    and never a paid fallback.

    model: catalog override forwarded to call_operator (defaults to
    OPERATOR_DEFAULTS["phala"], "deepseek/deepseek-v4-flash-0731").

    Tracks the model actually requested per turn in voicing_events, same
    shape as GravityWellAdapter, for council/cli.py's requested-vs-effective
    voicing provenance and DEGRADED-banner machinery.
    """
    model: str | None = None
    temperature: float = 0.8
    timeout: int = 300
    voicing_events: list = None

    def __post_init__(self):
        if self.voicing_events is None:
            self.voicing_events = []

    def chat(self, system: str, messages) -> str:
        """Invoke Phala as a Mirror Council voice.

        Args:
            system: Entity persona / character card (passed verbatim).
            messages: List of Message objects (duck-typed; role, content attrs).

        Returns:
            str: The voiced response (empty string if Phala returns None).
            Raises PhalaOperatorUnavailable if the seat is unreachable — no
            paid fallback is attempted, propagates to the caller unchanged.

        `content: null` responses (openai/gpt-oss-120b, qwen/qwen3.5-122b-a10b
        verified live 2026-08-04) put the answer in `reasoning_content`
        instead — `_post_chat_completion` (llm.py) already falls back to
        `reasoning_content` when `content` is empty/null, so this adapter
        gets real text either way, never an empty string masking a live
        answer. `agents_core.phala_tee.E2eeChannel.open_response` decrypts
        `reasoning_content` under the same seal as `content` before this
        adapter ever sees it, and `ReasoningContentDecryptionError` refuses
        to substitute placeholder text on a failed decrypt — this adapter's
        job is only to let that exception propagate, never swallow it.

        Side effect: appends to voicing_events list with per-call provenance.
        """
        prompt = _flatten_messages(messages)
        resolved_model = self.model or OPERATOR_DEFAULTS["phala"]
        result = call_operator(
            "phala", prompt,
            model=resolved_model,
            system=system,
            temperature=self.temperature,
            timeout=self.timeout,
        )
        self.voicing_events.append({
            "effective_operator": f"phala:{resolved_model}",
            "reason": "success",
        })
        return result or ""
