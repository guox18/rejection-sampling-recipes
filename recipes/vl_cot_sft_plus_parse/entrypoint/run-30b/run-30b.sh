#!/bin/bash
#
# SFT recipe runner
#
#   - Supports multiple input files (reuses the same Ray session)
#   - Auto-generates output paths (preserves input filenames)
#
# Usage:
#   bash run.sh
#


set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RECIPE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RECIPE_NAME="$(basename "$RECIPE_DIR")"

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

export LOG_DIR="${LOG_DIR:-/tmp/logs/${RECIPE_NAME}}"
mkdir -p "${LOG_DIR}"

# ============================================================
# Config - only pass run.py CLI args; edit YAML for model/concurrency
# ============================================================
# Config file path (can point to a custom config)
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

# Input files (supports multiple files, space-separated)
INPUT_FILES=(
    # test
    "$PROJECT_ROOT/tests/mock/text-pic.jsonl"
)

# Output directory (auto-creates sft/YYYYMMDD_HHMMSS under input dir, or set manually)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${TIMESTAMP}"

LATEST=""                  # Set to "--latest" to resume from latest timestamp dir
# Output filename suffix
OUTPUT_SUFFIX="_sft"
SFT_SUBDIR="sft"              # SFT output subdir name (default: "sft")

# Other options (run.py args only; do not change YAML)
NO_RESUME=""               # Set to "--no-resume" to disable resume
NO_PRESERVE_ORDER=""       # Set to "--no-preserve-order" to disable order preservation

# Error detection options
ERROR_DETECTION="--error-detection"  # Set to enable error detection (empty to disable)
ERROR_THRESHOLD=""         # InternalServerError rate threshold (default 0.5)

# =========================== End of config ============================

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"

# Silence Ray warnings/logs
export RAY_RUNTIME_ENV_HOOK_ENABLED=0
export RAY_DEDUP_LOGS=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_LOG_TO_STDERR=0
export PYTHONWARNINGS=ignore
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_worker_register_timeout_seconds=30


# Config
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}

# Get head node address
get_master_address() {
    local ip_address=$(getent hosts ${MASTER_ADDR} | awk '{print $1}')
    echo "${ip_address}:${MASTER_PORT}"
}

# Start Ray head node
start_ray_head() {
    echo "[INFO] Stopping existing Ray processes..."
    ray stop --force 2>/dev/null || true
    
    echo "[INFO] Starting Ray head node..."
    ray start --head \
        --port ${MASTER_PORT} \
        --system-config='{"enable_metrics_collection":false,"metrics_report_interval_ms":0}' \
        --disable-usage-stats
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to start Ray head node"
        exit 1
    fi
}

# Connect to Ray cluster
connect_to_ray() {
    local master_address=$1
    
    echo "[INFO] Stopping existing Ray processes..."
    ray stop --force 2>/dev/null || true
    
    echo "[INFO] Connecting to Ray cluster: ${master_address}..."
    ray start --address ${master_address} \
        --disable-usage-stats
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to connect to Ray cluster"
        exit 1
    fi
}

# Run pipeline
run_pipeline() {
    echo "[INFO] Running SFT pipeline..."
    cd "$PROJECT_ROOT"
    
    # Build CLI args
    local args=(--ray-address auto)
    
    # Add input files (supports multiple)
    if [ ${#INPUT_FILES[@]} -gt 0 ]; then
        args+=(--input "${INPUT_FILES[@]}")
    fi
    

    # Add output directory
    [ -n "$OUTPUT_DIR" ] && args+=(--output-dir "$OUTPUT_DIR")
    [ -n "$OUTPUT_SUFFIX" ] && args+=(--output-suffix "$OUTPUT_SUFFIX")
    [ -n "$SFT_SUBDIR" ] && args+=(--sft-subdir "$SFT_SUBDIR")
    
    # Add config file
    [ -n "$CONFIG_FILE" ] && args+=(--config "$CONFIG_FILE")
    
    # Add other options
    [ -n "$LATEST" ] && args+=("$LATEST")
    [ -n "$NO_RESUME" ] && args+=("$NO_RESUME")
    [ -n "$NO_PRESERVE_ORDER" ] && args+=("$NO_PRESERVE_ORDER")
    
    # Add error detection options
    [ -n "$ERROR_DETECTION" ] && args+=("$ERROR_DETECTION")
    [ -n "$ERROR_THRESHOLD" ] && args+=(--error-threshold "$ERROR_THRESHOLD")
    
    python -m recipes.vl_cot_sft_plus_parse.entrypoint.run "${args[@]}"
}

# Main
main() {
    echo "============================================================"
    echo "SFT Recipe"
    echo "============================================================"
    echo "  NODE_RANK:   ${NODE_RANK}"
    echo "  MASTER_ADDR: ${MASTER_ADDR:-localhost}"
    echo "  MASTER_PORT: ${MASTER_PORT}"
    echo "  PYTHON:      $(which python)"
    echo "============================================================"
    
    if [ ${NODE_RANK} -eq 0 ]; then
        # Head node: start Ray and run pipeline
        start_ray_head
        
        if [ -n "${MASTER_ADDR}" ]; then
            local master_address=$(get_master_address)
            echo "[INFO] Worker join command: ray start --address ${master_address}"
        fi

        echo "[INFO] Input files: ${#INPUT_FILES[@]}"
        echo "[INFO] Output dir: ${OUTPUT_DIR}"

        # Run pipeline (output paths auto-generated)
        run_pipeline
    else
        # Worker node: connect to Ray and wait
        local master_address=$(get_master_address)
        connect_to_ray ${master_address}
        
        echo "[INFO] Worker connected, waiting..."
        while true; do
            sleep 60
        done
    fi
}

main "$@"
