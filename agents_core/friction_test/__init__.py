"""agents_core.friction_test — Dissonance Engine for agent-operable flows.

Public API re-exports. Concrete implementations live in submodules.
"""
from .critique import Invariant, InvariantResult
from .driver import Driver
from .observe import Observation
from .orchestrator import run
from .report import FrictionReport
from .scenario import Scenario

__all__ = [
    "Driver",
    "Scenario",
    "Observation",
    "Invariant",
    "InvariantResult",
    "FrictionReport",
    "run",
]
