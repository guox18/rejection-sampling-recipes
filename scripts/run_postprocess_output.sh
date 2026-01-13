#!/bin/bash
# Post-process pipeline output data.
#
# Usage:
#   ./run_postprocess_output.sh <input_file> [output_dir]
#
# Operations:
#   1. Remove framework fields (_resume_id, _failed, _error, _traceback)
#   2. Remove rollouts field
#   3. Split: items with metadata.used_ground_truth=False -> xxx_update.jsonl

set -e

SCRIPT_DIR=$(dirname $(realpath $0))
PROJECT_DIR=$(dirname ${SCRIPT_DIR})

if [ -z "$1" ]; then
    echo "Usage: $0 <input_file> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 ./output.jsonl"
    echo "  $0 ./output.jsonl ./processed/"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="${2:-}"

if [ -n "$OUTPUT_DIR" ]; then
    uv run python ${SCRIPT_DIR}/postprocess_output.py \
        -i "${INPUT_PATH}" \
        -o "${OUTPUT_DIR}"
else
    uv run python ${SCRIPT_DIR}/postprocess_output.py \
        -i "${INPUT_PATH}"
fi

