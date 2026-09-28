"""Root conftest: skip suites that require fleet-host state.

This repo ships the full fleet test suite. A few families exercise state
that only exists on a fleet host and will hang or error elsewhere:

- the council suite imports ``agents_core.cards``, which requires the
  Archetypal Intelligence cards library (set ``ARCHETYPAL_CARDS_PATH`` to
  a cards checkout to run them);
- the doorman serving-admission family drives ``doorman_server``'s live
  serving loop, which is enabled on fleet hosts (set
  ``DOORMAN_SERVING_ADMISSION=1`` to run them).

Off-host, these files are excluded from collection so a clean
``pip install -e '.[test]' && pytest`` completes. The ``integration`` and
``smoke`` markers cover the remaining live-backend suites (see pyproject).
"""

import os
from pathlib import Path

collect_ignore_glob = []

_cards = os.environ.get(
    "ARCHETYPAL_CARDS_PATH", "/srv/git/archetypal-intelligence-working/cards"
)
if not Path(_cards).is_dir():
    collect_ignore_glob += [
        "tests/test_council_*.py",
        "agents_core/tests/test_council_*.py",
    ]

if os.environ.get("DOORMAN_SERVING_ADMISSION", "0") != "1":
    collect_ignore_glob += [
        "tests/test_doorman_atomic_acquire.py",
        "tests/test_gw_admission_pending_orphan_reclaim.py",
        "agents_core/tests/test_doorman_serving_admission_v0.py",
    ]
