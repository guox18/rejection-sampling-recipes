#!/usr/bin/env python3
"""
收集脚本：从 pipeline 输出中分类数据

用法:
    python collect.py <输入文件> <输出目录>

生成两份文件:
1. <输出目录>/feasible_and_pass.jsonl - SFT数据: 通过验证且采样模型有有效输出
2. <输出目录>/feasible_but_too_hard.jsonl - RL数据: 通过验证但采样模型无有效输出

分类规则:
1. metadata.used_ground_truth == false → SFT数据 (数据通过验证,采样模型产生有效输出)
2. _error == "No response passed and no gold answer" → RL数据 (数据通过验证,但采样模型无有效输出)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def load_jsonl(filepath: str) -> list[dict]:
    """加载 JSONL 文件"""
    items = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: line {line_num} JSON parse failed: {e}")
                items.append(None)
    return items


def is_sft_data(item: dict) -> bool:
    """
    判断是否为 SFT 数据
    条件: metadata.used_ground_truth == false
    说明: 数据通过了全部验证, 且采样模型产生了有效输出
    """
    if item is None:
        return False
    metadata = item.get("metadata", {})
    # 必须明确为 False (不是 None)
    return metadata.get("used_ground_truth") is False


def is_rl_data(item: dict) -> bool:
    """
    判断是否为 RL 数据
    条件: _error == "No response passed and no gold answer"
    说明: 数据通过了全部验证, 但采样模型没有产生有效输出 (太难了)
    """
    if item is None:
        return False
    return item.get("_error") == "No response passed and no gold answer"


def generate_id_ddm(item: dict) -> str:
    """生成唯一的 id_ddm (20位 hex hash)"""
    # 使用 item 的关键内容生成稳定的 hash
    content = json.dumps(item, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha1(content.encode("utf-8"))
    return hash_obj.hexdigest()[:20]


def format_sft_item(item: dict) -> dict:
    """
    格式化 SFT 数据输出

    输出格式:
    {
        "dialogs": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<think>...</think>\n\n..."}
        ],
        "id_ddm": "12598dc19f214789b283"
    }
    """
    messages = item.get("messages", [])
    dialogs = []
    for msg in messages:
        dialogs.append({"role": msg.get("role"), "content": msg.get("content")})

    return {"dialogs": dialogs, "id_ddm": generate_id_ddm(item)}


def format_rl_item(item: dict) -> dict:
    """
    格式化 RL 数据输出

    输出格式:
    {
        "key": "0",
        "prompt": "",
        "instruction_id_list": ["count:word_count_range", "format:list"],
        "kwargs": [{"min_words": 50, "max_words": 60}, {"sep": ","}]
    }
    """
    return {
        "key": str(item.get("key", "")),
        "prompt": item.get("prompt", ""),
        "instruction_id_list": item.get("instruction_id_list", []),
        "kwargs": item.get("kwargs", []),
    }


def save_jsonl(items: list[dict], filepath: Path):
    """保存 JSONL 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="收集脚本：从 pipeline 输出中分类数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input_file", "-i", required=True, help="Pipeline 输出文件 (JSONL)")
    parser.add_argument("--output_dir", "-o", required=True, help="目标输出目录")
    parser.add_argument("--force", "-f", action="store_true", help="强制覆盖已存在的输出文件")

    args = parser.parse_args()

    # 检查文件存在
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input_file}")
        sys.exit(1)

    # 输出文件路径
    output_dir = Path(args.output_dir)
    sft_data_path = output_dir / "feasible_and_pass.jsonl"
    rl_data_path = output_dir / "feasible_but_too_hard.jsonl"

    # 检查输出文件是否已存在
    if not args.force:
        existing = []
        if sft_data_path.exists():
            existing.append(str(sft_data_path))
        if rl_data_path.exists():
            existing.append(str(rl_data_path))
        if existing:
            print(f"Error: output file(s) already exist: {', '.join(existing)}")
            print("Use --force to overwrite")
            sys.exit(1)

    # 加载数据
    print(f"\nLoading data from: {input_path}")
    items = load_jsonl(args.input_file)
    total_count = len(items)
    print(f"Total items: {total_count}")

    # 分类数据
    print("Classifying data...")
    sft_items = []
    rl_items = []
    skipped_count = 0
    other_count = 0

    for item in items:
        if item is None:
            skipped_count += 1
            continue

        if is_sft_data(item):
            sft_items.append(format_sft_item(item))
        elif is_rl_data(item):
            rl_items.append(format_rl_item(item))
        else:
            other_count += 1

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入文件
    print(f"Writing: {sft_data_path}")
    save_jsonl(sft_items, sft_data_path)

    print(f"Writing: {rl_data_path}")
    save_jsonl(rl_items, rl_data_path)

    # 统计信息
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"  Total items:      {total_count}")
    print(f"  SFT data:         {len(sft_items)} (passed validation, model succeeded)")
    print(f"  RL data:          {len(rl_items)} (passed validation, model failed)")
    print(f"  Other:            {other_count} (infeasible or other errors)")
    print(f"  Skipped:          {skipped_count} (parse errors)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"   - feasible_and_pass.jsonl     ({len(sft_items)} items) - SFT data")
    print(f"   - feasible_but_too_hard.jsonl ({len(rl_items)} items) - RL data")
    print("=" * 60)


if __name__ == "__main__":
    main()
