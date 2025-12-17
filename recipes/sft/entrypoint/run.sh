#!/bin/bash
#
# SFT Recipe 执行脚本
#
# 功能:
#   - 支持处理多个输入文件（复用同一个 Ray session）
#   - 自动生成输出路径（保留输入文件名）
#
# 使用方式:
#   bash run.sh
#


set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ============================================================
# 配置项 - 在此处修改参数
# ============================================================
# 配置文件路径
CONFIG_FILE="${PROJECT_ROOT}/recipes/sft/config.yaml"

# 输入文件路径（支持多个文件，用空格分隔）
INPUT_FILES=(
    "${PROJECT_ROOT}/data/Nemotron-Post-Training-Dataset-v2/datasets/train_30.jsonl"
    "${PROJECT_ROOT}/data/Nemotron-Post-Training-Dataset-v2/datasets/train_5.jsonl"
)

# 输出目录（自动创建 sft/YYYYMMDD_HHMMSS 格式的目录，用户也可以手动指定）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# TIMESTAMP="20251217_120458"
OUTPUT_DIR="${PROJECT_ROOT}/data/Nemotron-Post-Training-Dataset-v2/datasets/more_sft/${TIMESTAMP}"

# 输出文件后缀
OUTPUT_SUFFIX="_sft"

# Pipeline 配置
BATCH_SIZE=""              # 留空使用配置文件默认值
CONCURRENCY=""             # 留空使用配置文件默认值
SAMPLER_CONCURRENCY=""     # 留空使用配置文件默认值
VERIFIER_CONCURRENCY=""    # 留空使用配置文件默认值

# 其他选项
NO_RESUME=""               # 设置为 "--no-resume" 禁用断点续传
NO_PRESERVE_ORDER=""       # 设置为 "--no-preserve-order" 禁用顺序保持

# =========================== 以上为配置项 ============================

# 激活虚拟环境(确保 Ray worker 使用正确的 Python)
source "$PROJECT_ROOT/.venv/bin/activate"

# 禁用 Ray 的各种警告和日志
export RAY_RUNTIME_ENV_HOOK_ENABLED=0
export RAY_DEDUP_LOGS=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_LOG_TO_STDERR=0
export PYTHONWARNINGS=ignore
export RAY_IGNORE_UNHANDLED_ERRORS=1
export RAY_worker_register_timeout_seconds=30


# 配置
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-6379}

# 获取 head 节点地址
get_master_address() {
    local ip_address=$(getent hosts ${MASTER_ADDR} | awk '{print $1}')
    echo "${ip_address}:${MASTER_PORT}"
}

# 启动 Ray head 节点
start_ray_head() {
    echo "[INFO] 停止已有的 Ray 进程..."
    ray stop --force 2>/dev/null || true
    
    echo "[INFO] 启动 Ray head 节点..."
    ray start --head \
        --port ${MASTER_PORT} \
        --system-config='{"enable_metrics_collection":false,"metrics_report_interval_ms":0}' \
        --disable-usage-stats
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] 启动 Ray head 节点失败"
        exit 1
    fi
}

# 连接到 Ray 集群
connect_to_ray() {
    local master_address=$1
    
    echo "[INFO] 停止已有的 Ray 进程..."
    ray stop --force 2>/dev/null || true
    
    echo "[INFO] 连接到 Ray 集群: ${master_address}..."
    ray start --address ${master_address} \
        --disable-usage-stats
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] 连接 Ray 集群失败"
        exit 1
    fi
}

# 执行 Pipeline
run_pipeline() {
    echo "[INFO] 执行 SFT Pipeline..."
    cd "$PROJECT_ROOT"
    
    # 构建命令行参数
    local args=(--ray-address auto)
    
    # 添加输入文件（支持多个）
    if [ ${#INPUT_FILES[@]} -gt 0 ]; then
        args+=(--input "${INPUT_FILES[@]}")
    fi
    
    # 添加输出目录
    [ -n "$OUTPUT_DIR" ] && args+=(--output-dir "$OUTPUT_DIR")
    [ -n "$OUTPUT_SUFFIX" ] && args+=(--output-suffix "$OUTPUT_SUFFIX")
    
    # 添加配置文件
    [ -n "$CONFIG_FILE" ] && args+=(--config "$CONFIG_FILE")
    
    # 添加 Pipeline 配置
    [ -n "$BATCH_SIZE" ] && args+=(--batch-size "$BATCH_SIZE")
    [ -n "$CONCURRENCY" ] && args+=(--concurrency "$CONCURRENCY")
    [ -n "$SAMPLER_CONCURRENCY" ] && args+=(--sampler-concurrency "$SAMPLER_CONCURRENCY")
    [ -n "$VERIFIER_CONCURRENCY" ] && args+=(--verifier-concurrency "$VERIFIER_CONCURRENCY")
    
    # 添加其他选项
    [ -n "$NO_RESUME" ] && args+=("$NO_RESUME")
    [ -n "$NO_PRESERVE_ORDER" ] && args+=("$NO_PRESERVE_ORDER")
    
    python "$SCRIPT_DIR/run.py" "${args[@]}"
}

# 主函数
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
        # Head 节点：启动 Ray 并执行 Pipeline
        start_ray_head
        
        if [ -n "${MASTER_ADDR}" ]; then
            local master_address=$(get_master_address)
            echo "[INFO] Worker 加入命令: ray start --address ${master_address}"
        fi

        echo "[INFO] 输入文件数量: ${#INPUT_FILES[@]}"
        echo "[INFO] 输出目录: ${OUTPUT_DIR}"

        # 运行 Pipeline（输出路径会自动生成）
        run_pipeline
    else
        # Worker 节点：连接到 Ray 集群并等待
        local master_address=$(get_master_address)
        connect_to_ray ${master_address}
        
        echo "[INFO] Worker 节点已连接, 进入等待状态..."
        while true; do
            sleep 60
        done
    fi
}

main "$@"
