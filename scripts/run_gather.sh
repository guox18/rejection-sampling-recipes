#!/bin/bash
set -euo pipefail

# Example: run data processing script

# Get absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "$ROOT_DIR/scripts"

# Path inference script
# Input: list of 235b file paths
# Output: auto-derive 30b path, original path, and train path, then call python script

# 235b file path list
INPUT_235B_LIST=(
    ""
)

# Iterate file paths
for INPUT_235B in "${INPUT_235B_LIST[@]}"; do
    # Extract filename
    FILENAME=$(basename "$INPUT_235B")
    
    # Strip between _abs and .jsonl to get original filename
    ORIGINAL_FILENAME=$(echo "$FILENAME" | sed 's/_abs.*\.jsonl/.jsonl/')
    
    # Get 235b file directory
    DIR_235B=$(dirname "$INPUT_235B")
    
    # Go up two levels to get jsonl directory
    JSONL_DIR=$(dirname $(dirname "$DIR_235B"))
    
    # Original file path
    ORIGINAL_PATH="${JSONL_DIR}/${ORIGINAL_FILENAME}"
    
    # Train path (replace .jsonl with _train.jsonl)
    TRAIN_PATH="${JSONL_DIR}/$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_train.jsonl/')"
    
    # Find latest date subdir under sft-30b
    SFT_30B_DIR="${JSONL_DIR}/sft-30b"
    LATEST_30B_DATE=$(ls -1 "$SFT_30B_DIR" | sort -r | head -n 1)
    
    # 30b file path (replace .jsonl with _abs_sft-30b.jsonl)
    FILENAME_30B=$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_abs_sft-30b.jsonl/')
    PATH_30B="${SFT_30B_DIR}/${LATEST_30B_DATE}/${FILENAME_30B}"
    
    # Print inferred paths
    echo "train path: $TRAIN_PATH"
    
    # Check files exist (excluding train)
    if [ ! -f "$INPUT_235B" ]; then
        echo "Error: 235b file not found: $INPUT_235B"
        exit 1
    fi
    
    if [ ! -f "$PATH_30B" ]; then
        echo "Error: 30b file not found: $PATH_30B"
        exit 1
    fi
    
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "Error: original file not found: $ORIGINAL_PATH"
        exit 1
    fi
    
    # Call gather.py
    python gather.py \
        --input_path "$ORIGINAL_PATH" \
        --qwen3vl30ba3bthinking_path "$PATH_30B" \
        --qwen3vl235ba22bthinking_path "$INPUT_235B" \
        --output_path "$TRAIN_PATH"
done

echo "All files processed!"
