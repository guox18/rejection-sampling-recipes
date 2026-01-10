#!/usr/bin/env python3
"""Simple text-only recipe entrypoint.

Input JSONL (minimal):
  - messages: OpenAI-style list
  - gold answer: last assistant message OR metadata.short_answer/metadata.answer

Usage:
    python -m recipes.text_sft_simple.entrypoint.run \
        --input tests/mock/text.jsonl
"""

import os

# Avoid Ray runtime_env overhead in shared storage environments.
os.environ.setdefault("RAY_RUNTIME_ENV_HOOK_ENABLED", "0")
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import signal
import sys
from datetime import datetime

import ray

from recipes.text_sft_simple.config import TextSFTConfig
from recipes.text_sft_simple.recipe import TextSFTRecipe
from src.pipeline import Pipeline

current_processing_file = None


def signal_handler(signum, frame):
    """Handle Ctrl+C and print progress."""
    print("\n\n" + "=" * 60)
    print("⚠️  收到中断信号 (Ctrl+C)")
    print("=" * 60)
    if current_processing_file:
        print("📄 当前正在处理的文件:")
        print(f"  输入文件: {current_processing_file['input']}")
        print(f"  输出文件: {current_processing_file['output']}")
        print(
            f"  进度: {current_processing_file['index']}/{current_processing_file['total']}"
        )
    else:
        print("  当前没有正在处理的文件")
    print("=" * 60)
    print("🛑 程序已中止")
    sys.exit(130)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Simple text-only recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="输入数据路径 (JSONL 格式)，支持多个文件",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="输出目录(可选)。不指定则在输入目录下创建输出子目录",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_text",
        help="输出文件名后缀，例如 train.jsonl -> train_text.jsonl",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="text",
        help="输出子目录名称",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="配置文件路径",
    )

    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray 集群地址。不指定则启动本地模式；'auto' 自动检测",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Ray 本地模式使用的 CPU 数量(仅本地模式生效)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Ray 本地模式使用的 GPU 数量(仅本地模式生效)",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传, 重新处理所有数据",
    )
    parser.add_argument(
        "--no-preserve-order",
        action="store_true",
        help="禁用顺序保持, 可提高性能但输出顺序可能不一致",
    )

    return parser.parse_args()


def init_ray(args: argparse.Namespace) -> None:
    """Initialize Ray once."""
    if ray.is_initialized():
        print("✅ Ray 已初始化, 使用现有连接")
        return

    init_kwargs = {
        "runtime_env": {},
        "logging_level": "ERROR",
        "log_to_driver": True,
    }

    if args.ray_address:
        print(f"🔗 连接到 Ray 集群: {args.ray_address}")
        init_kwargs["address"] = args.ray_address
    else:
        if args.num_cpus is not None:
            init_kwargs["num_cpus"] = args.num_cpus
        if args.num_gpus is not None:
            init_kwargs["num_gpus"] = args.num_gpus
        print("🚀 启动 Ray 本地模式")

    ray.init(**init_kwargs)


def generate_output_path(input_path: str, output_dir: str, suffix: str) -> str:
    """Generate output path from input path and suffix."""
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    output_filename = f"{name_without_ext}{suffix}.jsonl"
    return os.path.join(output_dir, output_filename)


def main() -> None:
    """Run the text-only recipe."""
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()
    input_files = args.input if isinstance(args.input, list) else [args.input]

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            sys.exit(1)

    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_files = []
    for input_file in input_files:
        if args.output_dir:
            base_dir = os.path.join(args.output_dir, args.output_subdir, timestamp)
        else:
            input_dir = os.path.dirname(os.path.abspath(input_file))
            base_dir = os.path.join(input_dir, args.output_subdir, timestamp)
        output_files.append(generate_output_path(input_file, base_dir, args.output_suffix))

    for output_file in output_files:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    config = TextSFTConfig.from_yaml(args.config)
    config.sampler_concurrency = config.sampler_concurrency or config.concurrency
    config.verifier_concurrency = config.verifier_concurrency or config.concurrency

    print("=" * 60)
    print("Simple Text SFT Recipe")
    print("=" * 60)
    print(f"  Config:      {args.config}")
    print(f"  Model:       {config.model}")
    print(f"  Base URL:    {config.base_url}")
    print(f"  N Samples:   {config.n_samples}")
    print(f"  Batch Size:  {config.batch_size}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Sampler:     {config.sampler_concurrency}")
    print(f"  Verifier:    {config.verifier_concurrency}")
    print(f"  Resume:      {'Disabled' if args.no_resume else 'Enabled'}")
    print(f"  Files:       {len(input_files)}")
    print("=" * 60)

    if output_files:
        output_dir_display = os.path.dirname(output_files[0])
        file_names = [os.path.basename(f) for f in output_files[:3]]
        print("\n📁 Output Directory:")
        print(f"  {output_dir_display}")
        print(f"\n  Files: {len(output_files)} → {', '.join(file_names)}")
        if len(output_files) > 3:
            print(f"         ... and {len(output_files) - 3} more")
        print()

    init_ray(args)

    recipe = TextSFTRecipe(config)
    pipeline = Pipeline(
        recipe=recipe,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
        stage_concurrency={
            "SamplerStage": config.sampler_concurrency,
            "VerifierStage": config.verifier_concurrency,
            "FormatterStage": 1,
        },
        preserve_order=not args.no_preserve_order,
        resume=not args.no_resume,
    )

    print("🚀 开始执行 Pipeline...\n")

    total_success = 0
    total_failed = 0

    for i, (input_file, output_file) in enumerate(zip(input_files, output_files), 1):
        global current_processing_file
        current_processing_file = {
            "input": input_file,
            "output": output_file,
            "index": i,
            "total": len(input_files),
        }

        print("=" * 60)
        print(f"Processing file {i}/{len(input_files)}: {os.path.basename(input_file)}")
        print("=" * 60)

        try:
            pipeline.run(input_file, output_file)
            total_success += 1
        except Exception as exc:
            print(f"❌ Failed to process {input_file}: {exc}")
            total_failed += 1

        print()

    current_processing_file = None

    print("=" * 60)
    print("📊 Final Summary")
    print("=" * 60)
    print(f"  Total files:     {len(input_files)}")
    print(f"  Success:         {total_success}")
    print(f"  Failed:          {total_failed}")
    print("=" * 60)
    print("\n✅ All files processed!")


if __name__ == "__main__":
    main()
