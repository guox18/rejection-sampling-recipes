#!/bin/bash
#
# Simple text-only recipe runner
#

set -e

export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.75

ray stop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
RECIPE_NAME="text_sft_simple"

export LOG_DIR="${LOG_DIR:-/tmp/logs/${RECIPE_NAME}}"
mkdir -p "${LOG_DIR}"

# Disable Ray Data progress bars if needed:
#   export RAY_DATA_DISABLE_PROGRESS_BARS=1

cd "$PROJECT_ROOT"

python -m recipes.text_sft_simple.entrypoint.run \
  --input "$PROJECT_ROOT/tests/mock/text.jsonl" \
  --config "$SCRIPT_DIR/../config.yaml" \
  --no-resume
