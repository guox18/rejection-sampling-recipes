#!/bin/bash
#
# IFBench recipe runner
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
RECIPE_NAME="ifbench"

export PYTHONPATH="$PROJECT_ROOT"

ray stop 2>/dev/null || true

export RAY_DEDUP_LOGS=1
export LOG_DIR="${LOG_DIR:-/tmp/logs/${RECIPE_NAME}}"
mkdir -p "${LOG_DIR}"

cd "$PROJECT_ROOT"

# 数据路径
INPUT_FILE="${1:-/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/tests/mock/instruction_following.jsonl}"
OUTPUT_DIR="${2:-/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/outputs/ifbench}"

if [[ "$OUTPUT_DIR" == *.jsonl ]]; then
  echo "Output path looks like a file; using directory ${OUTPUT_DIR%.jsonl}"
  OUTPUT_DIR="${OUTPUT_DIR%.jsonl}"
fi

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "IFBench Recipe Test"
echo "=============================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_DIR"
echo "Config: $SCRIPT_DIR/../config.yaml"
echo "Logs:   $LOG_DIR/pipeline.log"
echo "=============================================="

# Add --no-resume to disable auto-resume
uv run python3 -m recipes.ifbench.entrypoint.run \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --config "$SCRIPT_DIR/../config.yaml"
