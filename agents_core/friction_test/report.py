"""agents_core.friction_test.report — FrictionReport dataclass + write().

write() renders Markdown + JSON sidecar and calls vault_writer.write() twice.
Vault paths: Lapis/Friction-Reports/<target>-<scenario-set>-<ts>.md + .json
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents_core.vault_writer import write as vault_write

from .critique import Invariant, InvariantResult
from .observe import Observation
from .scenario import Scenario

VAULT_ROOT = Path("/srv/git/inertia-vault-working")


@dataclass
class FrictionReport:
    target: str
    scenario_set: str
    started_at: str
    finished_at: str
    n_scenarios: int
    n_invariants_declared: int
    n_invariants_inferred: int
    n_dissonances: dict[str, int]  # {system_likely, model_likely, total}
    scenarios: list[Scenario]
    observations: list[Observation]
    invariant_results: list[InvariantResult]
    inferred_invariants: list[Invariant]
    harness_warnings: list[str] = field(default_factory=list)
    md_path: str = ""
    json_path: str = ""


def write(
    report: FrictionReport,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Render report to Markdown + JSON sidecar, write both through vault_writer.

    Returns (md_path, json_path).
    """
    from .report_render import render_markdown, render_json

    # Compute vault-relative path
    ts_slug = report.started_at.replace(":", "-").replace("+", "p").replace(" ", "T")[:19]
    rel = f"Lapis/Friction-Reports/{report.target}-{report.scenario_set}-{ts_slug}"

    if out_dir is not None:
        md_path = out_dir / f"{report.target}-{report.scenario_set}-{ts_slug}.md"
        json_path = out_dir / f"{report.target}-{report.scenario_set}-{ts_slug}.json"
    else:
        md_path = VAULT_ROOT / f"{rel}.md"
        json_path = VAULT_ROOT / f"{rel}.json"

    report.md_path = str(md_path)
    report.json_path = str(json_path)

    md_content = render_markdown(report)
    json_content = render_json(report)

    vault_write(
        md_path,
        md_content,
        agent_id="friction-tester-v0",
        intent=f"friction-report:{report.target}:{report.scenario_set}",
        citations=None,
        policy="auto-update",
        stamp_frontmatter=True,
    )
    vault_write(
        json_path,
        json_content,
        agent_id="friction-tester-v0",
        intent=f"friction-report:{report.target}:{report.scenario_set}:json",
        citations=None,
        policy="auto-update",
        stamp_frontmatter=False,
    )

    return md_path, json_path
