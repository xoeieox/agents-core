"""agents_core.friction_test.orchestrator — run() entry point.

Loads driver, generates scenarios, runs each, collects observations,
runs critique, assembles and writes FrictionReport.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .critique import (
    QWEN_ENDPOINT,
    critique,
    infer_invariants,
    load_declared_invariants,
)
from .driver import CockpitDriver, RadioOpDriver
from .observe import Observation
from .report import FrictionReport, write as write_report
from .scenario import Scenario, generate_smoke_scenarios

logger = logging.getLogger(__name__)

_DRIVER_REGISTRY = {
    "radio-op": RadioOpDriver,
    "cockpit": CockpitDriver,
}


def run(
    target: str,
    scenario_set: str = "smoke",
    n_max: int | None = None,
    invariant_mode: Literal["declared", "inferred", "both"] = "declared",
    target_base_url: str | None = None,
    out_dir: Path | None = None,
    qwen_endpoint: str = QWEN_ENDPOINT,
    strict: bool = False,
) -> FrictionReport:
    """Run friction-test for ``target`` and return a FrictionReport.

    Args:
        target: target name, e.g. "radio-op" or "cockpit".
        scenario_set: "smoke" at v0.
        n_max: cap on scenarios per family.
        invariant_mode: which invariants to run.
        target_base_url: override driver's default base URL.
        out_dir: override vault output dir (mainly for tests).
        qwen_endpoint: Qwen API endpoint.
        strict: if True, re-raise harness errors (exit 1 behavior).

    Returns:
        FrictionReport (also written to vault / out_dir).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    harness_warnings: list[str] = []

    # 1. Load driver
    if target not in _DRIVER_REGISTRY:
        raise ValueError(f"Unknown target: {target!r}. Known: {list(_DRIVER_REGISTRY)}")

    driver_cls = _DRIVER_REGISTRY[target]
    if target_base_url:
        driver = driver_cls(base_url=target_base_url)
    else:
        driver = driver_cls()

    # 2. Driver setup
    try:
        driver.setup()
    except Exception as exc:
        if strict:
            raise
        harness_warnings.append(f"Driver setup failed: {exc}")
        logger.warning("Driver setup failed for %s: %s", target, exc)
        # Return a minimal report
        scenarios = _safe_generate(target, n_max)
        obs_list = [
            Observation(
                scenario_id=s.scenario_id,
                started_at=started_at,
                finished_at=started_at,
                errors=[{"stage": "driver_setup", "error": str(exc)}],
                harness_error=True,
            )
            for s in scenarios
        ]
        return _assemble_report(
            target=target,
            scenario_set=scenario_set,
            started_at=started_at,
            scenarios=scenarios,
            obs_list=obs_list,
            invariant_results=[],
            inferred_invs=[],
            harness_warnings=harness_warnings,
            n_invariants_declared=0,
            n_invariants_inferred=0,
            out_dir=out_dir,
        )

    # 3. Generate scenarios
    scenarios = _safe_generate(target, n_max)

    # 4. Run each scenario
    obs_list: list[Observation] = []
    for s in scenarios:
        try:
            obs = driver.run_scenario(s)
        except Exception as exc:
            obs = Observation(
                scenario_id=s.scenario_id,
                started_at=datetime.now(timezone.utc).isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                errors=[{"stage": "run_scenario", "error": str(exc)}],
                harness_error=True,
            )
            harness_warnings.append(f"Scenario {s.scenario_id} raised: {exc}")
        obs_list.append(obs)

    # 5. Teardown
    try:
        driver.teardown()
    except Exception as exc:
        harness_warnings.append(f"Driver teardown error: {exc}")

    # 6. Load declared invariants
    declared_invs = []
    if invariant_mode in ("declared", "both"):
        try:
            declared_invs = load_declared_invariants(target)
        except Exception as exc:
            harness_warnings.append(f"Failed to load declared invariants: {exc}")
            logger.warning("Could not load invariants for %s: %s", target, exc)

    # 7. Infer invariants via Qwen
    inferred_invs = []
    if invariant_mode in ("inferred", "both"):
        inferred_invs = infer_invariants(obs_list, qwen_endpoint=qwen_endpoint)
        if not inferred_invs and invariant_mode != "inferred":
            harness_warnings.append("Qwen unreachable; skipped inferred-invariants pass")

    all_invs = declared_invs + inferred_invs

    # 8. Critique
    invariant_results = []
    if all_invs:
        invariant_results = critique(scenarios, obs_list, all_invs, qwen_endpoint=qwen_endpoint)

    return _assemble_report(
        target=target,
        scenario_set=scenario_set,
        started_at=started_at,
        scenarios=scenarios,
        obs_list=obs_list,
        invariant_results=invariant_results,
        inferred_invs=inferred_invs,
        harness_warnings=harness_warnings,
        n_invariants_declared=len(declared_invs),
        n_invariants_inferred=len(inferred_invs),
        out_dir=out_dir,
    )


def _safe_generate(target: str, n_max: int | None) -> list[Scenario]:
    try:
        return generate_smoke_scenarios(target, n_max)
    except Exception:
        return []


def _assemble_report(
    *,
    target: str,
    scenario_set: str,
    started_at: str,
    scenarios: list[Scenario],
    obs_list: list[Observation],
    invariant_results: list,
    inferred_invs: list,
    harness_warnings: list[str],
    n_invariants_declared: int,
    n_invariants_inferred: int,
    out_dir: Path | None,
) -> FrictionReport:
    finished_at = datetime.now(timezone.utc).isoformat()

    n_system = sum(1 for r in invariant_results if r.classification == "dissonant_system_likely")
    n_model = sum(1 for r in invariant_results if r.classification == "dissonant_model_likely")

    report = FrictionReport(
        target=target,
        scenario_set=scenario_set,
        started_at=started_at,
        finished_at=finished_at,
        n_scenarios=len(scenarios),
        n_invariants_declared=n_invariants_declared,
        n_invariants_inferred=n_invariants_inferred,
        n_dissonances={
            "system_likely": n_system,
            "model_likely": n_model,
            "total": n_system + n_model,
        },
        scenarios=scenarios,
        observations=obs_list,
        invariant_results=invariant_results,
        inferred_invariants=inferred_invs,
        harness_warnings=harness_warnings,
    )

    write_report(report, out_dir=out_dir)
    return report
