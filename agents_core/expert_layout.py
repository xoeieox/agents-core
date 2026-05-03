"""Path constants and helpers for the Expert mini-vault layout.

Single source of truth for /srv/lapis/experts/<expert-id>/ directory structure.
Consumed by agents_core.expert and any future Expert-aware code.

Layout:
    /srv/lapis/experts/<expert-id>/
    ├── persona.md               ← layer 1 (authored per concrete-Expert spec)
    ├── seeds/                   ← layer 2 (written by Kami filter)
    │   └── <seed-id>.yaml
    ├── post-mortems/            ← layer 3 (written by dispatch_expert)
    │   └── <task_id>.yaml
    └── dispatches/              ← layer 4 (written during dispatch)
        └── <task_id>/
            ├── intent.yaml
            ├── notepad.md
            └── output.md
"""
from pathlib import Path

EXPERTS_ROOT = Path("/srv/lapis/experts")


def expert_root(expert_id: str) -> Path:
    """Return /srv/lapis/experts/<expert-id>/."""
    return EXPERTS_ROOT / expert_id


def seeds_root(expert_id: str) -> Path:
    """Return /srv/lapis/experts/<expert-id>/seeds/."""
    return expert_root(expert_id) / "seeds"


def post_mortems_root(expert_id: str) -> Path:
    """Return /srv/lapis/experts/<expert-id>/post-mortems/."""
    return expert_root(expert_id) / "post-mortems"


def dispatches_root(expert_id: str, task_id: str) -> Path:
    """Return /srv/lapis/experts/<expert-id>/dispatches/<task_id>/."""
    return expert_root(expert_id) / "dispatches" / task_id
