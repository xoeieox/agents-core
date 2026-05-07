"""NarratorEntity — scene narrator that satisfies the lapis_engine Entity protocol.

The narrator has no character identity or opinion. It advances scene context
between character turns with brief environmental and contextual details.

NARRATOR_ID is the canonical entity id for the narrator slot in the run YAML.
DEFAULT_NARRATOR_SYSTEM is the base system prompt; voice overrides append a
single line at the end.
"""
from __future__ import annotations

from lapis_engine import Entity, Message
from lapis_engine.types import Event, RunContext

NARRATOR_ID = "narrator"

DEFAULT_NARRATOR_SYSTEM = (
    "You are the Narrator of a council scene. "
    "You have no character identity and no opinion on the subject being discussed. "
    "Your role is to advance the scene between character turns with brief "
    "environmental and contextual details that give the scene physical texture:\n"
    "- A physical detail (the fire burns lower, a door swings open)\n"
    "- A shift in atmosphere (the silence stretches, the mist thickens)\n"
    "- A change in the space between speakers (one steps forward, one looks away)\n"
    "- A new sensory element (a sound from outside, the smell of something burning)\n"
    "- An object that becomes relevant (a glass set down hard, a letter on the table)\n"
    "Keep every response to 1-2 sentences. Never more. "
    "Write in present tense, third person. "
    "Do not name what characters are feeling — only what can be seen or heard. "
    "Do not summarize what was just said. Do not take sides. Do not explain."
)


class NarratorEntity:
    """Scene narrator implementing the lapis_engine Entity protocol.

    Parameters
    ----------
    llm:
        Any object satisfying ``lapis_engine.LanguageModel`` (has a
        ``chat(system, messages) -> str`` method).
    voice:
        Optional prose description of the narrator's voice style.
        When provided, appended to DEFAULT_NARRATOR_SYSTEM as
        ``"Your narrative voice: <voice>"``.
    """

    id: str = NARRATOR_ID

    def __init__(self, llm, voice: str | None = None) -> None:
        self.llm = llm
        self.voice = voice

    def system_prompt(self, ctx: RunContext | None) -> str:  # noqa: ARG002
        base = DEFAULT_NARRATOR_SYSTEM
        if self.voice:
            return base + f"\nYour narrative voice: {self.voice}"
        return base

    def act(self, prompt: str, ctx: RunContext | None) -> str:  # noqa: ARG002
        system = self.system_prompt(ctx)
        return self.llm.chat(system, [Message(role="user", content=prompt)])

    def observe(self, event: Event) -> None:  # noqa: ARG002
        """Narrator does not maintain internal state from observations."""
        pass
