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
    ├── dispatches/              ← layer 4 (written during dispatch)
    │   └── <task_id>/
    │       ├── intent.yaml
    │       ├── notepad.md
    │       └── output.md
    └── corpora/<mode>/          ← layer 5 (assembled by corpus builders)
        └── <source-kind>/
            └── <stable-id>.md
"""
from pathlib import Path

from agents_core.room_paths import room_path

EXPERTS_ROOT = room_path("experts")

_VALID_CORPUS_MODES = frozenset({"build", "adversary"})


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


def corpus_root(expert_id: str, mode: str = "build") -> Path:
    """Return /srv/lapis/experts/<expert-id>/corpora/<mode>/.

    Raises ValueError for unknown mode. Valid modes: 'build', 'adversary'.
    """
    if mode not in _VALID_CORPUS_MODES:
        raise ValueError(
            f"corpus_root: unknown mode {mode!r}; valid modes are {sorted(_VALID_CORPUS_MODES)}"
        )
    return expert_root(expert_id) / "corpora" / mode
