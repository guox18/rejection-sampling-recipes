#!/bin/bash
#
# CPU task demo runner
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
RECIPE_NAME="cpu_task_demo"

export LOG_DIR="${LOG_DIR:-/tmp/logs/${RECIPE_NAME}}"
mkdir -p "${LOG_DIR}"

cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python}"

"$PY_BIN" -m recipes.cpu_task_demo.entrypoint.run \
  --input "$PROJECT_ROOT/tests/mock/cpu_task.jsonl" \
  --config "$SCRIPT_DIR/../config.yaml"
