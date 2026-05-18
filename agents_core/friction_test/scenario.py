"""agents_core.friction_test.scenario — Scenario dataclass and smoke generator.

Scenarios are seed-driven at v0 (no Qwen generation). Output is deterministic
and reviewable. Qwen variant generation lands at v1.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scenario:
    """One concrete probe to run against a target.

    ``scenario_id`` is content-derived so the same seed always produces the
    same ID and IDs are insensitive to key reordering in the seed file.

    ``expected_class`` is a human-readable label used by the report renderer
    only; it is NOT used for programmatic comparison.
    """

    scenario_id: str
    target: str
    family_id: str
    kind: str  # "happy" | "obvious_break"
    inputs: dict[str, Any]
    expected_class: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "target": self.target,
            "family_id": self.family_id,
            "kind": self.kind,
            "inputs": self.inputs,
            "expected_class": self.expected_class,
            "notes": self.notes,
        }


def _make_scenario_id(target: str, family_id: str, inputs: dict[str, Any]) -> str:
    """Stable content-derived ID; insensitive to key ordering."""
    payload = json.dumps(
        {"target": target, "family_id": family_id, "inputs": inputs},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _expand_family(target: str, family: dict[str, Any], n_max: int | None) -> list[Scenario]:
    """Expand one seed-file family into 1-N concrete Scenario objects."""
    family_id = family["id"]
    kind = family.get("kind", "happy")
    expected_class = family.get("expected_class", "")
    notes = family.get("notes", "")
    base_inputs = family.get("inputs", {})
    n_scenarios = family.get("n_scenarios", 1)

    if n_max is not None:
        n_scenarios = min(n_scenarios, n_max)

    scenarios: list[Scenario] = []
    for i in range(n_scenarios):
        # For families that request multiple scenarios, differentiate inputs
        # by adding a copy_index so IDs don't collide.
        if n_scenarios > 1:
            inputs = copy.deepcopy(base_inputs)
            inputs["_copy_index"] = i
        else:
            inputs = copy.deepcopy(base_inputs)

        sid = _make_scenario_id(target, family_id, inputs)
        scenarios.append(
            Scenario(
                scenario_id=sid,
                target=target,
                family_id=family_id,
                kind=kind,
                inputs=inputs,
                expected_class=expected_class,
                notes=notes,
            )
        )
    return scenarios


def _seeds_path(target: str) -> Path:
    here = Path(__file__).parent
    return here / "scenario_seeds" / f"{target}.yaml"


def generate_smoke_scenarios(target: str, n_max: int | None = None) -> list[Scenario]:
    """Load seed file for ``target`` and expand into concrete Scenario list.

    Args:
        target: target name, e.g. ``"radio-op"`` or ``"cockpit"``.
        n_max: optional cap on scenarios per family (useful in tests).

    Returns:
        Flat list of Scenario objects.
    """
    seed_path = _seeds_path(target)
    if not seed_path.exists():
        raise FileNotFoundError(f"No scenario seed file for target '{target}': {seed_path}")

    with seed_path.open() as fh:
        data = yaml.safe_load(fh)

    families = data.get("families", [])
    result: list[Scenario] = []
    for family in families:
        result.extend(_expand_family(target, family, n_max))
    return result
