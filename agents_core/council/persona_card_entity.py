"""PersonaCardEntity — voices a reviewer/persona deck card built from the
Composition schema (slug, composition.primitives, voice_exemplars, domains,
kernel_invariants; see agents_core.cards.validate_deck_card).

Distinct from CharacterEntity (archetypes.engine.character_entity), which
loads the older, richer CharacterComposition schema (character_id,
behavioral_synthesis, cultural_flavor, shadow_dynamics, ...). Same
lapis_engine Entity protocol duck-type as CharacterEntity/NarratorEntity:
id, act(prompt, ctx), observe(event). Voicing still flows through the same
GravityWellAdapter — no new voicing path, only a new card schema.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from lapis_engine import Message
from lapis_engine.types import Event, RunContext


class PersonaCardEntity:
    """Entity that voices a reviewer persona defined by a deck Composition card."""

    def __init__(
        self,
        slug: str,
        primitives: dict,
        voice_exemplars: list,
        domains: list,
        kernel_invariants: list,
        llm,
    ) -> None:
        self.slug = slug
        self.primitives = primitives
        self.voice_exemplars = voice_exemplars
        self.domains = domains
        self.kernel_invariants = kernel_invariants
        self.llm = llm
        self._observed: list[Event] = []

    @property
    def id(self) -> str:
        return self.slug

    @classmethod
    def load(cls, card_path: str | Path, llm) -> "PersonaCardEntity":
        path = Path(card_path)
        if not path.exists():
            raise FileNotFoundError(f"No persona card at: {path}")
        data = yaml.safe_load(path.read_text())
        composition = data.get("composition") or {}
        return cls(
            slug=data["slug"],
            primitives=composition.get("primitives") or {},
            voice_exemplars=data.get("voice_exemplars") or [],
            domains=data.get("domains") or [],
            kernel_invariants=data.get("kernel_invariants") or [],
            llm=llm,
        )

    def system_prompt(self, ctx: RunContext | None) -> str:  # noqa: ARG002
        weighted = ", ".join(f"{k} ({v:.2f})" for k, v in self.primitives.items())
        exemplars = "\n".join(f"- {e}" for e in self.voice_exemplars)
        domains = ", ".join(self.domains)
        invariants = ", ".join(self.kernel_invariants)
        return (
            f"You are a reviewer persona: {self.slug}.\n\n"
            f"Your composition: {weighted}\n\n"
            f"Your voice — speak in this register, calibrated to these exemplars:\n"
            f"{exemplars}\n\n"
            f"Your domains of scrutiny: {domains}\n\n"
            f"Kernel invariants you hold: {invariants}\n\n"
            f"Speak as yourself, in first person, from inside your reviewing stance. "
            f"Do not analyze yourself from outside. Do not prefix with 'As a reviewer' "
            f"or 'I am'. Be specific and concrete; vividness matters more than length."
        )

    def act(self, prompt: str, ctx: RunContext | None) -> str:
        system = self.system_prompt(ctx)
        return self.llm.chat(system, [Message(role="user", content=prompt)])

    def observe(self, event: Event) -> None:
        self._observed.append(event)
