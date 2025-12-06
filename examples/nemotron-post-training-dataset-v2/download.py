"""
Hugging Face 数据集下载工具

用于从 Hugging Face Hub 下载数据集并保存为 JSONL 格式。
支持从 parquet 文件直接加载，适用于大型数据集。

    python download.py --dataset nvidia/Nemotron-Post-Training-Dataset-v2 \\
                      --pattern "data/stem-*.parquet" \\
                      --output stem.jsonl \\
                      --token YOUR_HF_TOKEN
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


def setup_hf_environment(
    token: str | None = None,
    cache_dir: Path | None = None,
    endpoint: str = "https://hf-mirror.com",
    timeout: int = 60,
) -> Path:
    """
    设置 Hugging Face 环境变量

    Args:
        token: Hugging Face API token (可选)
        cache_dir: 缓存目录路径 (默认: ~/.cache/huggingface)
        endpoint: HF endpoint URL (默认: https://hf-mirror.com)
        timeout: 下载超时时间(秒) (默认: 60)

    Returns:
        Path: 缓存目录路径
    """
    if token:
        os.environ["HF_TOKEN"] = token

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface"

    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)

    return cache_dir


def download_parquet_dataset(
    dataset_name: str, file_pattern: str, cache_dir: Path, split_name: str = "train"
) -> Any:
    """
    从 Hugging Face 下载 parquet 格式的数据集

    Args:
        dataset_name: 数据集名称 (例如: nvidia/Nemotron-Post-Training-Dataset-v2)
        file_pattern: parquet 文件匹配模式 (例如: data/stem-*.parquet)
        cache_dir: 缓存目录
        split_name: 数据集分割名称 (默认: train)

    Returns:
        下载的数据集对象
    """
    hf_url = f"hf://datasets/{dataset_name}/{file_pattern}"

    print(f"开始下载数据集: {dataset_name}")
    print(f"文件模式: {file_pattern}")

    dataset = load_dataset(
        "parquet",
        data_files={split_name: hf_url},
        cache_dir=str(cache_dir),
    )

    return dataset[split_name]


def save_to_jsonl(dataset: Any, output_path: Path, verbose: bool = True) -> None:
    """
    将数据集保存为 JSONL 格式

    Args:
        dataset: 要保存的数据集
        output_path: 输出文件路径
        verbose: 是否显示详细信息
    """
    if verbose:
        print(f"开始保存到 {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    if verbose:
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ 成功保存到 {output_path}")
        print(f"  记录数: {len(dataset)}")
        print(f"  文件大小: {file_size_mb:.2f} MB")


def download_and_save(
    dataset_name: str,
    file_pattern: str,
    output_path: Path,
    token: str | None = None,
    cache_dir: Path | None = None,
    endpoint: str = "https://hf-mirror.com",
    split_name: str = "train",
) -> None:
    """
    下载数据集并保存为 JSONL 格式 (主函数)

    Args:
        dataset_name: 数据集名称
        file_pattern: parquet 文件匹配模式
        output_path: 输出文件路径
        token: Hugging Face API token (可选)
        cache_dir: 缓存目录 (可选)
        endpoint: HF endpoint URL
        split_name: 数据集分割名称
    """
    try:
        # 设置环境
        cache_dir = setup_hf_environment(token=token, cache_dir=cache_dir, endpoint=endpoint)

        # 下载数据集
        dataset = download_parquet_dataset(
            dataset_name=dataset_name,
            file_pattern=file_pattern,
            cache_dir=cache_dir,
            split_name=split_name,
        )

        print(f"✓ 成功加载数据集，共 {len(dataset)} 条记录")

        # 保存为 JSONL
        save_to_jsonl(dataset, output_path)

    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从 Hugging Face Hub 下载数据集并保存为 JSONL 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="nvidia/Nemotron-Post-Training-Dataset-v2",
        help="数据集名称 (默认: nvidia/Nemotron-Post-Training-Dataset-v2)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="data/stem-*.parquet",
        help="parquet 文件匹配模式 (默认: data/stem-*.parquet)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: 当前目录下的 <pattern_name>.jsonl)",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (可选，也可通过环境变量 HF_TOKEN 设置)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="缓存目录路径 (默认: ~/.cache/huggingface)",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint URL (默认: https://hf-mirror.com)",
    )

    parser.add_argument("--split", type=str, default="train", help="数据集分割名称 (默认: train)")

    return parser.parse_args()


def main():
    """主入口函数"""
    args = parse_args()

    # 如果没有指定 token，尝试从环境变量获取
    token = args.token or os.environ.get("HF_TOKEN")

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        # 从 pattern 中提取名称
        pattern_name = (
            args.pattern.split("/")[-1].replace("-*.parquet", "").replace("*.parquet", "data")
        )
        output_path = Path(f"{pattern_name}.jsonl")

    # 解析缓存目录
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # 执行下载和保存
    download_and_save(
        dataset_name=args.dataset,
        file_pattern=args.pattern,
        output_path=output_path,
        token=token,
        cache_dir=cache_dir,
        endpoint=args.endpoint,
        split_name=args.split,
    )


if __name__ == "__main__":
    main()
