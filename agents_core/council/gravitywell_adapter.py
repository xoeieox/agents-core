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

    Tracks effective operator per turn in voicing_events for observability.
    """
    temperature: float = 0.8
    timeout: int = 300
    on_wake_fail: str = "sonnet"
    voicing_events: list = None

    def __post_init__(self):
        if self.voicing_events is None:
            self.voicing_events = []

    def chat(self, system: str, messages) -> str:
        """Invoke GravityWell as a Mirror Council voice.

        Args:
            system: Entity persona / character card (passed verbatim to GW).
            messages: List of Message objects (duck-typed; role, content attrs).

        Returns:
            str: The voiced response (empty string if GW returns None).
            Never raises OperatorUnreachableError (on_wake_fail handles all
            GW-unreachable shapes and returns str or falls back to Sonnet).

        Side effect: appends to voicing_events list with per-call provenance.
        """
        prompt = _flatten_messages(messages)
        provenance: list = []
        result = call_operator(
            "gravitywell", prompt,
            system=system,
            temperature=self.temperature,
            timeout=self.timeout,
            on_wake_fail=self.on_wake_fail,
            _provenance_out=provenance,
        )
        # provenance is a list of (reason, operator) tuples.
        # Find the effective operator and the actual failure reason (if any).
        # The effective operator is from the last "success" entry.
        # The failure reason is from the first non-"success", non-"fallback" entry.
        if provenance:
            effective_operator = None
            failure_reason = None

            for reason, op in provenance:
                if reason == "success":
                    effective_operator = op
                elif reason != "fallback" and failure_reason is None:
                    failure_reason = reason

            # Determine final reason: use failure reason if present, else success
            final_reason = failure_reason or "success"

            # Fallback: use last entry's operator if we didn't find a success
            if effective_operator is None and provenance:
                _, effective_operator = provenance[-1]

            self.voicing_events.append({
                "effective_operator": effective_operator or "unknown",
                "reason": final_reason,
            })
        return result or ""
