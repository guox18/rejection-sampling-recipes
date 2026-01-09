#!/usr/bin/env python3
"""
SFT Recipe 执行入口.

Usage:
    # 处理单个文件（默认在输入文件的父目录下创建 sft/timestamp/ 子目录）
    python run.py --input data/train.jsonl
    # 输出: data/sft/YYYYMMDD_HHMMSS/sft_train.jsonl

    # 处理多个文件（复用同一个 Ray session）
    python run.py --input data/train.jsonl data/test.jsonl data/val.jsonl
    # 输出: data/sft/YYYYMMDD_HHMMSS/sft_train.jsonl
    #       data/sft/YYYYMMDD_HHMMSS/sft_test.jsonl
    #       data/sft/YYYYMMDD_HHMMSS/sft_val.jsonl

    # 自定义输出目录
    python run.py --input data/*.jsonl --output-dir results/exp001
    # 输出: results/exp001/sft_YYYYMMDD_HHMMSS/sft_*.jsonl

    # 自定义 SFT 子目录名称
    python run.py --input data/train.jsonl --sft-subdir custom_output
    # 输出: data/custom_output/YYYYMMDD_HHMMSS/train_sft.jsonl

    # 连接到 Ray 集群
    python run.py --input data/*.jsonl --ray-address auto
"""

import os

# 禁用 Ray runtime_env 自动检测(共享项目路径, 不需要传输代码)
os.environ.setdefault("RAY_RUNTIME_ENV_HOOK_ENABLED", "0")

# 屏蔽 Ray 的各种警告信息
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # 屏蔽 Python 警告

import argparse
import json
import signal
import sys

# 使用相对引用，python -m 启动时工作目录已在项目根目录
PROJECT_ROOT = os.getcwd()

# 动态导入当前 recipe 的模块（方便复制目录）
from importlib import import_module
_recipe_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFTConfig = import_module(f"recipes.{_recipe_name}.config").SFTConfig
SFTRecipe = import_module(f"recipes.{_recipe_name}.recipe").SFTRecipe

import ray
from src.pipeline import Pipeline

# 全局变量：跟踪当前处理的文件路径
current_processing_file = None


def signal_handler(signum, frame):
    """处理 Ctrl+C 信号."""
    print("\n\n" + "=" * 60)
    print("⚠️  收到中断信号 (Ctrl+C)")
    print("=" * 60)
    if current_processing_file:
        print(f"📄 当前正在处理的文件:")
        print(f"  输入文件: {current_processing_file['input']}")
        print(f"  输出文件: {current_processing_file['output']}")
        print(f"  进度: {current_processing_file['index']}/{current_processing_file['total']}")
    else:
        print("  当前没有正在处理的文件")
    print("=" * 60)
    print("🛑 程序已中止")
    sys.exit(130)  # 130 = 128 + SIGINT(2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="SFT Recipe: 采样 → 验证 → 格式化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="采样模型名",
    )
    
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="judge 模型名",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="采样模型 API base URL",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="judge 模型 API base URL",
    )

    parser.add_argument(
        "--semaphore-per-sampler",
        type=int,
        default=None,
        help="SamplerStage 并发度",
    )


    
    # 数据路径
    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",  # 支持多个输入文件
        required=True,
        help="输入数据路径 (JSONL 格式)，支持多个文件",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录。如果指定，会在此目录下创建 sft_YYYYMMDD_HHMMSS/ 子目录；如果不指定，会在输入文件的父目录下创建 sft/YYYYMMDD_HHMMSS/ 子目录",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_sft",
        help="输出文件名后缀，例如 train.jsonl -> train_sft.jsonl",
    )
    parser.add_argument(
        "--sft-subdir",
        type=str,
        default="sft",
        help="SFT 输出子目录名称，默认为 'sft'",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="从输出目录下的最新时间戳目录续传。如果找不到已有的 sft_YYYYMMDD_HHMMSS 目录，则创建新的",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="配置文件路径",
    )
    # Pipeline 配置(可覆盖配置文件中的默认值)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每个 batch 的数据量(覆盖配置文件)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="默认并发度, 仅对没有配置并发度的 Stages 起作用(Stage 的 actor 数量, 覆盖配置文件)",
    )
    parser.add_argument(
        "--sampler-concurrency",
        type=int,
        default=None,
        help="SamplerStage 并发度(覆盖配置文件)",
    )
    parser.add_argument(
        "--verifier-concurrency",
        type=int,
        default=None,
        help="VerifierStage 并发度(覆盖配置文件)",
    )
    
    # Ray 配置
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray 集群地址. 不指定则启动本地模式；'auto' 表示自动检测；或指定 'ray://IP:10001'",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Ray 本地模式使用的 CPU 数量(仅在 --ray-address 未指定时生效)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Ray 本地模式使用的 GPU 数量(仅在 --ray-address 未指定时生效)",
    )
    
    # 其他选项
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
    
    # 错误检测选项
    parser.add_argument(
        "--error-detection",
        action="store_true",
        help="启用错误检测机制，当 InternalServerError 错误率过高时自动停止",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.5,
        help="InternalServerError 错误率阈值 (0.0-1.0)，默认 0.5 (50%%)",
    )
    
    return parser.parse_args()


def init_ray(args: argparse.Namespace) -> None:
    """初始化 Ray."""
    if ray.is_initialized():
        print(f"✅ Ray 已初始化, 使用现有连接")
        return
    
    # 共享存储模式：禁用 runtime_env, 不需要 Ray 传输代码, 所有节点直接访问共享存储上的项目
    init_kwargs = {
        "runtime_env": {},
        "logging_level": "ERROR",  # 只显示错误信息, 屏蔽警告
        "log_to_driver": False,  # 禁用驱动程序日志输出
    }
    
    if args.ray_address:
        # 连接到已有集群
        print(f"🔗 连接到 Ray 集群: {args.ray_address}")
        init_kwargs["address"] = args.ray_address
    else:
        # 本地模式
        if args.num_cpus is not None:
            init_kwargs["num_cpus"] = args.num_cpus
        if args.num_gpus is not None:
            init_kwargs["num_gpus"] = args.num_gpus
        print(f"🚀 启动 Ray 本地模式")
    
    ray.init(**init_kwargs)
    
    # 打印集群信息
    resources = ray.cluster_resources()
    print(f"   CPU: {resources.get('CPU', 0):.0f}")
    print(f"   GPU: {resources.get('GPU', 0):.0f}")
    print(f"   Memory: {resources.get('memory', 0) / 1e9:.1f} GB")


def find_latest_timestamp_dir(output_dir: str) -> str | None:
    """
    在输出目录下查找最新的时间戳目录 (YYYYMMDD_HHMMSS 格式).
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        最新时间戳目录的完整路径，如果找不到返回 None
    """
    if not os.path.exists(output_dir):
        return None
    
    # 找所有时间戳格式的子目录 (YYYYMMDD_HHMMSS: 8位日期_6位时间)
    timestamp_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and len(item) == 15 and item[8] == '_':
            # 简单检查：长度15，第9个字符是下划线
            timestamp_dirs.append(item)
    
    if not timestamp_dirs:
        return None
    
    # 按字符串排序，时间戳格式天然可排序，最新的在最后
    latest = sorted(timestamp_dirs)[-1]
    latest_path = os.path.join(output_dir, latest)
    return latest_path


def generate_output_path(input_path: str, output_dir: str, suffix: str) -> str:
    """
    根据输入文件路径生成输出文件路径.
    
    规则:
        - 保留输入文件名，添加后缀
        - 输出文件放在 output_dir 目录下
    
    示例:
        输入: a/b/c/train.jsonl    → output_dir/train_sft.jsonl
    """
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    output_filename = f"{name_without_ext}{suffix}.jsonl"
    
    return os.path.join(output_dir, output_filename)


def check_internal_server_errors(output_file: str, error_threshold: float) -> tuple[bool, int, int, int]:
    """
    检查输出文件中的 InternalServerError 错误率.
    
    遍历输出文件，统计包含 InternalServerError 的错误数量。
    如果错误率超过阈值，返回 True 表示应该停止处理。
    
    Args:
        output_file: 输出文件路径
        error_threshold: InternalServerError 错误率阈值 (0.0-1.0)
    
    Returns:
        (should_stop, total_items, failed_items, internal_server_errors)
        - should_stop: 是否应该停止处理
        - total_items: 总数据条数
        - failed_items: 失败的数据条数（所有错误）
        - internal_server_errors: InternalServerError 错误数量
    """
    if not os.path.exists(output_file):
        return False, 0, 0, 0
    
    total_items = 0
    non_skipped_items = 0
    failed_items = 0
    internal_server_errors = 0
    
    with open(output_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    total_items += 1

                    if item.get("metadata", {}).get("skipped") is not True:
                        non_skipped_items += 1
                    
                    # 检查是否失败
                    if item.get('_failed') is True:
                        failed_items += 1
                        
                        # 检查 traceback 中是否包含 InternalServerError
                        traceback_str = (item.get('_traceback') or '')
                        if 'InternalServerError' in traceback_str:
                            internal_server_errors += 1
                            
                except json.JSONDecodeError:
                    continue
    
    if total_items == 0:
        return False, 0, 0, 0
    
    # 计算 InternalServerError 错误率
    error_rate = internal_server_errors / non_skipped_items
    should_stop = error_rate > error_threshold
    
    return should_stop, non_skipped_items, failed_items, internal_server_errors


def main():
    """运行 SFT Recipe."""
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_args()
    
    # 处理输入文件列表
    input_files = args.input if isinstance(args.input, list) else [args.input]
    
    # 检查输入文件
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            sys.exit(1)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 自动生成输出路径
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 决定输出路径
    if args.output_dir:
        # 用户指定了输出目录
        if args.latest:
            # 尝试从最新时间戳目录续传
            latest_dir = find_latest_timestamp_dir(args.output_dir)
            if latest_dir:
                output_base_dir = latest_dir
            else:
                output_base_dir = os.path.join(args.output_dir, f"{args.sft_subdir}_{timestamp}")
        else:
            # 直接创建新的时间戳目录
            output_base_dir = os.path.join(args.output_dir, f"{args.sft_subdir}_{timestamp}")
        
        output_files = [
            generate_output_path(input_file, output_base_dir, args.output_suffix)
            for input_file in input_files
        ]
    else:
        # 用户没有指定输出目录：每个输入文件在自己的父目录下创建 sft/YYYYMMDD_HHMMSS/
        output_files = []
        for input_file in input_files:
            input_dir = os.path.dirname(os.path.abspath(input_file))
            sft_base_dir = os.path.join(input_dir, args.sft_subdir)
            
            if args.latest:
                # 尝试从最新时间戳目录续传
                latest_dir = find_latest_timestamp_dir(sft_base_dir)
                if latest_dir:
                    output_base_dir = latest_dir
                else:
                    output_base_dir = os.path.join(sft_base_dir, timestamp)
            else:
                output_base_dir = os.path.join(sft_base_dir, timestamp)
            
            output_file = generate_output_path(input_file, output_base_dir, args.output_suffix)
            output_files.append(output_file)
    
    # 创建输出目录
    for output_file in output_files:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 加载配置
    config = SFTConfig.from_yaml(args.config)
    
    # 命令行参数覆盖配置文件(自动检测所有匹配的参数)
    for key in dir(config):
        if not key.startswith('_') and hasattr(args, key):
            arg_value = getattr(args, key, None)
            if arg_value is not None:
                setattr(config, key, arg_value)
    
    # 处理默认值：sampler/verifier_concurrency 默认使用 concurrency
    config.sampler_concurrency = config.sampler_concurrency or config.concurrency
    config.verifier_concurrency = config.verifier_concurrency or config.concurrency
    
    print("=" * 60)
    print("SFT Recipe")
    print("=" * 60)
    print(f"  Config:      {args.config}")
    print(f"  Model:       {config.model}")
    print(f"  Base URL:    {config.base_url}")
    print(f"  N Samples:   {config.n_samples}")
    print(f"  Batch Size:  {config.batch_size}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Sampler:     {config.sampler_concurrency}")
    print(f"  Verifier:    {config.verifier_concurrency}")
    print(f"  Resume:      {'Latest' if args.latest else 'Disabled' if args.no_resume else 'Enabled'}")
    print(f"  Files:       {len(input_files)}")
    print("=" * 60)
    
    # 显示输入输出映射（简化路径显示）
    print("\n📁 Output Directory:")
    # 取第一个输出文件的目录作为代表
    if output_files:
        output_dir_display = os.path.dirname(output_files[0])
        print(f"  {output_dir_display}")
        print(f"\n  Files: {len(output_files)} → {', '.join([os.path.basename(f) for f in output_files[:3]])}")
        if len(output_files) > 3:
            print(f"         ... and {len(output_files) - 3} more")
    print()
    
    # 初始化 Ray（只初始化一次，所有文件复用）
    init_ray(args)
    
    # 创建 Recipe（只创建一次）
    recipe = SFTRecipe(config)
    
    # 创建 Pipeline（只创建一次）
    pipeline = Pipeline(
        recipe=recipe,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
        stage_concurrency={
            "DataConverterStage": 1,
            "SamplerStage": config.sampler_concurrency,
            "VerifierStage": config.verifier_concurrency,
            "FormatterStage": 1,
        },
        preserve_order=not args.no_preserve_order,
        resume=not args.no_resume,
    )
    
    # 循环处理所有文件（复用同一个 Ray session 和 Pipeline）
    print("🚀 开始执行 Pipeline...\n")
    
    total_success = 0
    total_failed = 0
    
    for i, (input_file, output_file) in enumerate(zip(input_files, output_files), 1):
        # 更新全局变量：当前处理的文件信息
        global current_processing_file
        current_processing_file = {
            'input': input_file,
            'output': output_file,
            'index': i,
            'total': len(input_files)
        }
        
        print("=" * 60)
        print(f"Processing file {i}/{len(input_files)}: {os.path.basename(input_file)}")
        print("=" * 60)
        
        try:
            pipeline.run(input_file, output_file)
            total_success += 1
            
            # 错误检测：检查 InternalServerError 错误率
            if args.error_detection:
                should_stop, non_skipped_item, failed, internal_errors = check_internal_server_errors(
                    output_file, args.error_threshold
                )
                
                if non_skipped_item > 0:
                    error_rate = internal_errors / non_skipped_item
                    print(f"\n🔍 Error Detection:")
                    print(f"  Non_skipped_item items:           {non_skipped_item}")
                    print(f"  Failed items:          {failed}")
                    print(f"  InternalServerError:   {internal_errors}")
                    print(f"  Error rate:            {error_rate:.2%}")
                    print(f"  Threshold:             {args.error_threshold:.2%}")
                    
                    if should_stop:
                        print(f"\n⚠️  WARNING: InternalServerError 错误率过高！")
                        print(f"  错误率 {error_rate:.2%} 超过阈值 {args.error_threshold:.2%}")
                        print(f"  这可能表示远程服务已关闭或出现严重问题")
                        print(f"  自动停止处理，避免产生大量失败数据")
                        print(f"  已处理文件数: {i}/{len(input_files)}")
                        print(f"  输出目录: {os.path.dirname(output_file)}")
                        sys.exit(1)
                
        except Exception as e:
            print(f"❌ Failed to process {input_file}: {e}")
            total_failed += 1
        
        print()
    
    # 清空全局变量
    current_processing_file = None
    
    # 总结
    print("=" * 60)
    print("📊 Final Summary")
    print("=" * 60)
    print(f"  Total files:     {len(input_files)}")
    print(f"  Success:         {total_success}")
    print(f"  Failed:          {total_failed}")
    print("=" * 60)
    print()
    print("✅ All files processed!")


if __name__ == "__main__":
    main()
