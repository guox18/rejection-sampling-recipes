#!/bin/bash
set -euo pipefail

# 数据处理脚本调用示例

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

cd "$ROOT_DIR/scripts"

# 路径推导脚本
# 输入: 235b 文件路径列表
# 输出: 自动推导出 30b 路径、原文件路径和 train 路径，并调用 python 脚本

# 235b 文件路径列表
INPUT_235B_LIST=(
    ""
)

# 遍历文件路径列表
for INPUT_235B in "${INPUT_235B_LIST[@]}"; do
    # 提取文件名
    FILENAME=$(basename "$INPUT_235B")
    
    # 去除 _abs 和 .jsonl 之间的部分，得到原始文件名
    ORIGINAL_FILENAME=$(echo "$FILENAME" | sed 's/_abs.*\.jsonl/.jsonl/')
    
    # 获取 235b 文件的目录
    DIR_235B=$(dirname "$INPUT_235B")
    
    # 倒退两级目录，得到 jsonl 目录
    JSONL_DIR=$(dirname $(dirname "$DIR_235B"))
    
    # 原文件路径
    ORIGINAL_PATH="${JSONL_DIR}/${ORIGINAL_FILENAME}"
    
    # train 路径（原文件名去掉 .jsonl 后缀，加上 _train.jsonl）
    TRAIN_PATH="${JSONL_DIR}/$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_train.jsonl/')"
    
    # 找到 sft-30b 目录下最新的日期子文件夹
    SFT_30B_DIR="${JSONL_DIR}/sft-30b"
    LATEST_30B_DATE=$(ls -1 "$SFT_30B_DIR" | sort -r | head -n 1)
    
    # 30b 文件路径（原始文件名去掉 .jsonl，加上 _abs_sft-30b.jsonl）
    FILENAME_30B=$(echo "$ORIGINAL_FILENAME" | sed 's/\.jsonl$/_abs_sft-30b.jsonl/')
    PATH_30B="${SFT_30B_DIR}/${LATEST_30B_DATE}/${FILENAME_30B}"
    
    # 打印推导结果
    echo "train 路径: $TRAIN_PATH"
    
    # 检查文件是否存在（除了 train）
    if [ ! -f "$INPUT_235B" ]; then
        echo "错误: 235b 文件不存在: $INPUT_235B"
        exit 1
    fi
    
    if [ ! -f "$PATH_30B" ]; then
        echo "错误: 30b 文件不存在: $PATH_30B"
        exit 1
    fi
    
    if [ ! -f "$ORIGINAL_PATH" ]; then
        echo "错误: 原文件不存在: $ORIGINAL_PATH"
        exit 1
    fi
    
    # 调用 gather.py
    python gather.py \
        --input_path "$ORIGINAL_PATH" \
        --qwen3vl30ba3bthinking_path "$PATH_30B" \
        --qwen3vl235ba22bthinking_path "$INPUT_235B" \
        --output_path "$TRAIN_PATH"
done

echo "所有文件处理完成！"
