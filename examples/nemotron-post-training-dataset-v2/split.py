"""
数据分割脚本：将数据集分割为训练集、验证集和测试集

用法：
    python split.py --input stem_processed.jsonl \\
                    --output_dir ./datasets \\
                    --split_ratio "0.95,0.003,0.047" \\
                    --shuffle \\
                    --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path


def read_jsonl(datapath: str) -> list:
    """读取 JSONL 格式文件"""
    res = []
    print(f"读取文件: {datapath}")
    with open(datapath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                res.append(json.loads(line))
    return res


def write_jsonl(data: list, datapath: str) -> None:
    """保存数据为 JSONL 格式"""
    os.makedirs(os.path.dirname(datapath) if os.path.dirname(datapath) else ".", exist_ok=True)
    print(f"保存文件至: {datapath}")
    with open(datapath, "w", encoding="utf-8") as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")


def split_data(
    data: list, split_ratio: list[float], shuffle: bool = True, seed: int = 42
) -> tuple[list, list, list]:
    """
    将数据按比例分割为训练集、验证集和测试集

    Args:
        data: 输入数据列表
        split_ratio: 分割比例 [train_ratio, val_ratio, test_ratio]
        shuffle: 是否随机打乱数据
        seed: 随机种子

    Returns:
        train_data, val_data, test_data
    """
    # 验证分割比例
    if len(split_ratio) != 3:
        raise ValueError(f"split_ratio 必须包含3个元素 [train, val, test]，当前为: {split_ratio}")

    if not abs(sum(split_ratio) - 1.0) < 1e-6:
        raise ValueError(f"split_ratio 总和必须为 1.0，当前为: {sum(split_ratio)}")

    if any(r < 0 or r > 1 for r in split_ratio):
        raise ValueError("split_ratio 中的每个值必须在 [0, 1] 范围内")

    # 复制数据
    data_copy = data.copy()

    # 随机打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(data_copy)
        print(f"已使用随机种子 {seed} 打乱数据")

    # 计算分割点
    total = len(data_copy)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    # 分割数据
    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]

    print("\n数据分割完成:")
    print(f"  训练集: {len(train_data)} 条 ({len(train_data) / total * 100:.2f}%)")
    print(f"  验证集: {len(val_data)} 条 ({len(val_data) / total * 100:.2f}%)")
    print(f"  测试集: {len(test_data)} 条 ({len(test_data) / total * 100:.2f}%)")

    return train_data, val_data, test_data


def parse_split_ratio(ratio_str: str) -> list[float]:
    """解析分割比例字符串"""
    try:
        ratios = [float(x.strip()) for x in ratio_str.split(",")]
        if len(ratios) != 3:
            raise ValueError
        return ratios
    except Exception as e:
        raise ValueError(f"无效的分割比例格式: {ratio_str}，应该形如 '0.8,0.1,0.1'") from e


def main():
    parser = argparse.ArgumentParser(
        description="将数据集分割为训练集、验证集和测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录",
    )

    parser.add_argument(
        "--split_ratio",
        type=str,
        default="0.95,0.003,0.047",
        help='分割比例 (格式: "train,val,test"，默认: "0.95,0.003,0.047")',
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="是否随机打乱数据 (默认: 是)",
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="不打乱数据",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )

    args = parser.parse_args()

    # 确定输入输出路径
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # 解析分割比例
    split_ratio = parse_split_ratio(args.split_ratio)
    shuffle = not args.no_shuffle

    print("=" * 60)
    print("数据分割脚本")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"分割比例: 训练集={split_ratio[0]}, 验证集={split_ratio[1]}, 测试集={split_ratio[2]}")
    print(f"随机打乱: {'是' if shuffle else '否'}")
    if shuffle:
        print(f"随机种子: {args.seed}")
    print("=" * 60)

    # 读取数据
    data = read_jsonl(str(input_path))
    print(f"读取到 {len(data)} 条数据")

    if len(data) == 0:
        raise ValueError("输入文件为空，无数据可分割")

    # 分割数据
    train_data, val_data, test_data = split_data(data, split_ratio, shuffle=shuffle, seed=args.seed)

    # 保存数据
    write_jsonl(train_data, str(output_dir / "train.jsonl"))
    write_jsonl(val_data, str(output_dir / "val.jsonl"))
    write_jsonl(test_data, str(output_dir / "test.jsonl"))

    print("\n" + "=" * 60)
    print("✓ 处理完成！")
    print(f"  训练集: {output_dir / 'train.jsonl'}")
    print(f"  验证集: {output_dir / 'val.jsonl'}")
    print(f"  测试集: {output_dir / 'test.jsonl'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
