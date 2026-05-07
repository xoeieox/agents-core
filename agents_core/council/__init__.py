"""agents_core.council — Mirror Council runtime package.

Relocated from /srv/agents/scripts/council.py.

The AI_ROOT sys.path injection below is a packaging shim: archetypes.engine is
not pip-installed; it lives at /srv/git/archetypal-intelligence-working.
Proper fix (pip-install editable or merge) is v0.next.

archetypes.engine.character_entity is imported function-locally inside
run_deliberation; this injection ensures it is resolvable at that point.
"""
import sys
from pathlib import Path

# Archetypes-engine packaging shim — load-bearing for every council run.
AI_ROOT = Path("/srv/git/archetypal-intelligence-working")
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from agents_core.council.cli import (  # noqa: E402, F401
    COUNCIL_DIR,
    LOG_DIR,
    CARDS_ROOT,
    DEFAULT_POOLS,
    DEFAULT_TURNS,
    DEFAULT_VOICING,
    DEFAULT_MODE,
    VALID_MODES,
    SCENE_N_RANGE,
    ROLE_NARRATOR,
    DASHBOARD_BASE,
    run_path,
    load_run,
    save_run,
    new_run_id,
    build_roster,
    find_card_path,
    gather_mem_context,
    select_entities,
    run_deliberation,
    _build_entity,
    _build_director,
    _build_adapter,
    _parse_synthesis,
    _status_from_synthesis,
    _validate_mode_n,
    _role_assignments,
    _extract_search_terms,
    _parse_mem_keys,
    _extract_json,
    _fork_runtime,
)
