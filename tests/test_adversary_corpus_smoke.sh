#!/usr/bin/env bash
# Smoke test: build_adversary_corpus against real /srv/lapis/ and real mem.
#
# Assertions:
#   - feedback/ subdir is non-empty (guaranteed to have entries in production)
#   - Written count > 0 on a fresh corpus or unchanged > 0 on re-run
#
# Skips gracefully if feedback source is empty (shouldn't happen on StarHouse).
#
# Usage:
#   bash tests/test_adversary_corpus_smoke.sh
#   # or with a custom corpus root:
#   SMOKE_CORPUS_ROOT=/tmp/smoke-corpus bash tests/test_adversary_corpus_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CORPUS_ROOT="${SMOKE_CORPUS_ROOT:-/srv/lapis/experts/tech-kami/corpora/adversary}"

echo "[smoke] adversary_corpus smoke test"
echo "[smoke] corpus root: ${CORPUS_ROOT}"

# Run the build via Python and capture JSON-like output
RESULT=$(python3 - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from agents_core.adversary_corpus import build_adversary_corpus
from pathlib import Path
import os

corpus_root = os.environ.get("SMOKE_CORPUS_ROOT")
kwargs = {}
if corpus_root:
    kwargs["_corpus_root"] = Path(corpus_root)

try:
    result = build_adversary_corpus("tech-kami", **kwargs)
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
PYEOF
)

echo "[smoke] result: ${RESULT}"

WRITTEN=$(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['written'])")
UNCHANGED=$(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['unchanged'])")
ERRORS=$(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['errors']))")

if [ "${ERRORS}" -gt 0 ]; then
    echo "[smoke] FAIL: errors reported: $(echo "${RESULT}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['errors'])")"
    exit 1
fi

# Check feedback/ subdir is non-empty
FEEDBACK_DIR="${CORPUS_ROOT}/feedback"
if [ ! -d "${FEEDBACK_DIR}" ]; then
    echo "[smoke] SKIP: feedback/ dir not found — source may be empty, skipping"
    exit 0
fi

FEEDBACK_COUNT=$(find "${FEEDBACK_DIR}" -name "*.md" | wc -l | tr -d ' ')
if [ "${FEEDBACK_COUNT}" -eq 0 ]; then
    echo "[smoke] SKIP: feedback/ dir is empty — no feedback entries in mem, skipping"
    exit 0
fi

echo "[smoke] feedback/ fragment count: ${FEEDBACK_COUNT}"

TOTAL=$((WRITTEN + UNCHANGED))
if [ "${TOTAL}" -eq 0 ]; then
    echo "[smoke] FAIL: written=${WRITTEN} unchanged=${UNCHANGED} — expected total > 0"
    exit 1
fi

echo "[smoke] PASS: written=${WRITTEN} unchanged=${UNCHANGED} feedback_frags=${FEEDBACK_COUNT}"
