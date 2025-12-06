#!/bin/bash
# ============================================================
# Nemotron-Post-Training-Dataset-v2 数据处理流水线
# ============================================================
#
# 用法:
#   bash run_pipeline.sh [HF_TOKEN]
#
# 流程:
#   1. 下载数据 (stem.jsonl)
#   2. 解析 boxed 答案，转换为框架格式 (stem_processed.jsonl)
#   3. 分割为 train/val/test
#
# ============================================================

set -e  # 遇到错误时停止

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 数据输出目录
DATA_DIR="$PROJECT_ROOT/data/Nemotron-Post-Training-Dataset-v2"
WORK_DIR="$DATA_DIR"
OUTPUT_DIR="$DATA_DIR/datasets"

echo "============================================================"
echo "Nemotron-Post-Training-Dataset-v2 数据处理流水线"
echo "============================================================"
echo "脚本目录: $SCRIPT_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo "数据目录: $DATA_DIR"
echo ""

# 创建数据目录
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# HuggingFace Token (可选)
HF_TOKEN="${1:-$HF_TOKEN}"

# ============================================================
# Step 1: 下载数据
# ============================================================
echo "[Step 1/3] 下载数据..."

if [ -f "$WORK_DIR/stem.jsonl" ]; then
    echo "  stem.jsonl 已存在，跳过下载"
else
    echo "  开始下载 nvidia/Nemotron-Post-Training-Dataset-v2 的 stem 数据..."

    if [ -n "$HF_TOKEN" ]; then
        python "$SCRIPT_DIR/download.py" \
            --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
            --pattern "data/stem-*.parquet" \
            --output "$WORK_DIR/stem.jsonl" \
            --token "$HF_TOKEN"
    else
        python "$SCRIPT_DIR/download.py" \
            --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \
            --pattern "data/stem-*.parquet" \
            --output "$WORK_DIR/stem.jsonl"
    fi
fi

echo ""

# ============================================================
# Step 2: 解析 boxed 答案，转换格式
# ============================================================
echo "[Step 2/3] 解析 boxed 答案，转换为框架格式..."

if [ -f "$WORK_DIR/stem_processed.jsonl" ]; then
    echo "  stem_processed.jsonl 已存在，跳过处理"
else
    python "$SCRIPT_DIR/process.py" \
        --input "$WORK_DIR/stem.jsonl" \
        --output "$WORK_DIR/stem_processed.jsonl"
fi

echo ""

# ============================================================
# Step 3: 分割数据
# ============================================================
echo "[Step 3/3] 分割数据为 train/val/test..."

if [ -f "$OUTPUT_DIR/train.jsonl" ] && [ -f "$OUTPUT_DIR/val.jsonl" ] && [ -f "$OUTPUT_DIR/test.jsonl" ]; then
    echo "  train.jsonl, val.jsonl, test.jsonl 已存在，跳过分割"
else
    python "$SCRIPT_DIR/split.py" \
        --input "$WORK_DIR/stem_processed.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --split_ratio "0.95,0.003,0.047" \
        --shuffle \
        --seed 42
fi

echo ""

# ============================================================
# 完成
# ============================================================
echo "============================================================"
echo "✓ 数据处理完成！"
echo ""
echo "生成的文件 (在 $DATA_DIR):"
echo "  - stem.jsonl           (原始下载数据)"
echo "  - stem_processed.jsonl (处理后的数据)"
echo ""
echo "最终数据集 (在 $OUTPUT_DIR):"
echo "  - train.jsonl          (训练集)"
echo "  - val.jsonl            (验证集)"
echo "  - test.jsonl           (测试集)"
echo ""
echo "数据格式 (符合拒绝采样框架要求):"
echo '  {"id": "...", "messages": [{"role": "user", "content": "..."}], "metadata": {"answer": "A"}}'
echo "============================================================"
