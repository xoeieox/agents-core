"""GravityWell adapter for Mirror Council voicing.

Routes council voicing through the GravityWell 122B model via call_operator()
under the doorman. Implements the LanguageModel protocol (duck-typed; does not
import from lapis_engine).

Per-call doorman lease is safe: the doorman's 600s idle-grace keeps GW warm
across a deliberation's voices. on_wake_fail='sonnet' means council voicing
falls back to paid Sonnet on any GW unreachable shape (wake-fail, doorman-down,
or serving-then-HTTP-exhausted), returning str always, never raising.
"""

from __future__ import annotations
from dataclasses import dataclass

from agents_core.llm import call_operator


def _flatten_messages(messages) -> str:
    """Flatten a message list to a single user prompt.

    Single user-message fast path → messages[0].content.
    Multi-message → role-tagged join (defensive for multi-turn / NarratorEntity).

    Reads m.role / m.content by duck-typing; does NOT import Message from
    lapis_engine. Council voicing passes a single user Message always, so the
    fast path covers the live case.
    """
    if len(messages) == 1:
        return messages[0].content

    # Multi-message join: defensive path for non-council callers.
    parts = []
    for m in messages:
        role = getattr(m, "role", "user")
        content = getattr(m, "content", str(m))
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


@dataclass
class GravityWellAdapter:
    """LanguageModel adapter routing council voicing to the GravityWell 122B
    via call_operator() under the doorman.

    Quality surface → on_wake_fail='sonnet' (paid fallback, logged loudly).
    Per-call doorman lease keeps GW warm across a deliberation's voices.
    """
    temperature: float = 0.8
    timeout: int = 300
    on_wake_fail: str = "sonnet"

    def chat(self, system: str, messages) -> str:
        """Invoke GravityWell as a Mirror Council voice.

        Args:
            system: Entity persona / character card (passed verbatim to GW).
            messages: List of Message objects (duck-typed; role, content attrs).

        Returns:
            str: The voiced response (empty string if GW returns None).
            Never raises OperatorUnreachableError (on_wake_fail handles all
            GW-unreachable shapes and returns str or falls back to Sonnet).
        """
        prompt = _flatten_messages(messages)
        result = call_operator(
            "gravitywell", prompt,
            system=system,
            temperature=self.temperature,
            timeout=self.timeout,
            on_wake_fail=self.on_wake_fail,
        )
        return result or ""
