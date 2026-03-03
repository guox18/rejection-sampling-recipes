#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-python}"

"$PY_BIN" -m recipes.cpu_task_demo.entrypoint.benchmark \
  --ray-address auto \
  --target-concurrency 80 \
  --batch-size 1 \
  --prime-limit 180000 \
  --rounds 4 \
  --input tests/mock/cpu_task_heavy_3000.jsonl \
  --generate-mock-if-missing \
  --mock-num-items 3000
