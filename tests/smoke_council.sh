#!/usr/bin/env bash
# smoke_council.sh — smoke test for agents_core.council (no LLM, no subprocess fork).
#
# Uses COUNCIL_ENGINE_STUB=1 to skip LLM calls entirely.
# Tests deliberation routing, scene routing, and startup_sweep orphan recovery.
#
# Exit codes:
#   0 — all checks passed
#   1 — at least one check failed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export COUNCIL_ENGINE_STUB=1

COUNCIL_DIR="$(mktemp -d)"
LOG_DIR="${COUNCIL_DIR}/logs"
mkdir -p "${LOG_DIR}"

cleanup() { rm -rf "${COUNCIL_DIR}"; }
trap cleanup EXIT

PASS=0
FAIL=0

check() {
    local label="$1"
    local result="$2"
    if [ "${result}" = "ok" ]; then
        echo "  PASS: ${label}"
        PASS=$((PASS+1))
    else
        echo "  FAIL: ${label} — ${result}"
        FAIL=$((FAIL+1))
    fi
}

echo "[smoke_council] Starting smoke tests..."

# ---------------------------------------------------------------------------
# Test 1: deliberation mode stub run
# ---------------------------------------------------------------------------
echo ""
echo "[smoke_council] Test 1: deliberation stub run"

RUN_ID="smoke-deliberation-$(date +%Y%m%d%H%M%S)"
python3 - <<PYEOF
import sys, yaml
from pathlib import Path
from datetime import datetime

council_dir = Path("${COUNCIL_DIR}")

run = {
    "run_id": "${RUN_ID}",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "status": "deliberating",
    "mode": "deliberation",
    "decision": "Should the smoke test pass?",
    "context_gathered": {"terms": [], "hits": []},
    "selected_entities": [
        {"id": "entity-a", "role": "first_voice"},
        {"id": "entity-b", "role": "second_voice"},
    ],
    "selection_reasoning": "smoke",
    "voicing": "sonnet",
    "turns_cap": 4,
    "turns": [],
}
(council_dir / f"${RUN_ID}.yaml").write_text(
    yaml.safe_dump(run, sort_keys=False, allow_unicode=True)
)

# Patch COUNCIL_DIR and run
from agents_core.council import cli as council_cli
council_cli.COUNCIL_DIR = council_dir
council_cli.run_deliberation("${RUN_ID}")

result = yaml.safe_load((council_dir / f"${RUN_ID}.yaml").read_text())

errors = []
if result["status"] not in ("resolved", "open", "diverged"):
    errors.append(f"bad status: {result['status']}")
if "synthesis" not in result:
    errors.append("missing synthesis")
if not result.get("turns"):
    errors.append("no turns")

if errors:
    print("ERRORS: " + "; ".join(errors))
    sys.exit(1)
print("ok")
PYEOF
check "deliberation stub run produces resolved/open/diverged status" "ok"

# ---------------------------------------------------------------------------
# Test 2: scene mode stub run
# ---------------------------------------------------------------------------
echo ""
echo "[smoke_council] Test 2: scene stub run"

RUN_ID_SCENE="smoke-scene-$(date +%Y%m%d%H%M%S)-x"
python3 - <<PYEOF
import sys, yaml
from pathlib import Path
from datetime import datetime

council_dir = Path("${COUNCIL_DIR}")

run = {
    "run_id": "${RUN_ID_SCENE}",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "status": "deliberating",
    "mode": "scene",
    "decision": "A kitchen at 2am, after an argument.",
    "context_gathered": {"terms": [], "hits": []},
    "selected_entities": [
        {"id": "char-a", "role": "scene_slot_0"},
        {"id": "char-b", "role": "scene_slot_1"},
    ],
    "selection_reasoning": "smoke",
    "voicing": "sonnet",
    "turns_cap": 4,
    "turns": [],
}
(council_dir / f"${RUN_ID_SCENE}.yaml").write_text(
    yaml.safe_dump(run, sort_keys=False, allow_unicode=True)
)

from agents_core.council import cli as council_cli
council_cli.COUNCIL_DIR = council_dir
council_cli.run_deliberation("${RUN_ID_SCENE}")

result = yaml.safe_load((council_dir / f"${RUN_ID_SCENE}.yaml").read_text())

errors = []
if result["status"] != "closed":
    errors.append(f"scene status should be closed, got: {result['status']}")
if "synthesis" in result:
    errors.append("scene run must NOT have synthesis key")
if not result.get("turns"):
    errors.append("no turns in scene run")

if errors:
    print("ERRORS: " + "; ".join(errors))
    sys.exit(1)
print("ok")
PYEOF
check "scene stub run produces closed status without synthesis" "ok"

# ---------------------------------------------------------------------------
# Test 3: startup_sweep orphan recovery
# ---------------------------------------------------------------------------
echo ""
echo "[smoke_council] Test 3: orphan recovery via startup_sweep"

RUN_ID_ORPHAN="smoke-orphan-$(date +%Y%m%d%H%M%S)-y"
python3 - <<PYEOF
import sys, yaml, asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

council_dir = Path("${COUNCIL_DIR}")

# Create a deliberating run that is 2 hours old (past the 1h threshold)
old_created_at = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
run_data = {
    "run_id": "${RUN_ID_ORPHAN}",
    "status": "deliberating",
    "created_at": old_created_at,
    "mode": "deliberation",
    "decision": "orphan smoke test",
    "turns": [],
}
(council_dir / f"${RUN_ID_ORPHAN}.yaml").write_text(
    yaml.safe_dump(run_data, sort_keys=False, allow_unicode=True)
)

import agents_core.claude_queue_runner as runner_mod
runner_mod._COUNCIL_DIR = council_dir
runner_mod._COUNCIL_ORPHAN_AGE_SECS = 3600

# Build minimal fake queue with empty dirs
import tempfile
queue_tmp = Path("${COUNCIL_DIR}") / "queue"
for sub in ("pending", "active", "completed", "failed"):
    (queue_tmp / sub).mkdir(parents=True, exist_ok=True)

fake_queue = MagicMock()
fake_queue.active_dir = queue_tmp / "active"
fake_queue.queue_dir = queue_tmp

with patch("subprocess.run"):
    runner_mod.startup_sweep(fake_queue)

result = yaml.safe_load((council_dir / f"${RUN_ID_ORPHAN}.yaml").read_text())

if result["status"] != "failed":
    print(f"ERRORS: orphan run should be failed, got {result['status']}")
    sys.exit(1)
if result.get("error") != "runner_crash_recovery":
    print(f"ERRORS: orphan error should be runner_crash_recovery, got {result.get('error')}")
    sys.exit(1)
print("ok")
PYEOF
check "startup_sweep marks old deliberating orphan run as failed" "ok"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[smoke_council] Results: ${PASS} passed, ${FAIL} failed"
if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
exit 0
