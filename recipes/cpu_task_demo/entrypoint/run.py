#!/usr/bin/env python3
"""CPU task demo recipe entrypoint."""

import os

# Avoid Ray runtime_env overhead in shared storage environments.
os.environ.setdefault("RAY_RUNTIME_ENV_HOOK_ENABLED", "0")
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import argparse
import sys
from datetime import datetime
from pathlib import Path

import ray

from recipes.cpu_task_demo.config import CpuTaskDemoConfig
from recipes.cpu_task_demo.recipe import CpuTaskDemoRecipe
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="CPU task demo recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL path")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSONL path (auto-generated when omitted)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="Config file path",
    )

    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address. Empty starts local mode; 'auto' to detect",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="CPUs for local Ray mode (only when --ray-address is not set)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPUs for local Ray mode (only when --ray-address is not set)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume and reprocess all data",
    )
    parser.add_argument(
        "--no-preserve-order",
        action="store_true",
        help="Disable order preservation (faster but output order may differ)",
    )
    return parser.parse_args()


def init_ray(args: argparse.Namespace) -> None:
    """Initialize Ray once."""
    if ray.is_initialized():
        print("[INFO] Ray already initialized; reusing existing connection")
        return

    init_kwargs = {
        "runtime_env": {},
        "logging_level": "ERROR",
        "log_to_driver": True,
    }

    if args.ray_address:
        print(f"[INFO] Connecting to Ray cluster: {args.ray_address}")
        init_kwargs["address"] = args.ray_address
    else:
        if args.num_cpus is not None:
            init_kwargs["num_cpus"] = args.num_cpus
        if args.num_gpus is not None:
            init_kwargs["num_gpus"] = args.num_gpus
        print("[INFO] Starting Ray in local mode")

    ray.init(**init_kwargs)


def default_output_path(input_path: Path) -> Path:
    """Generate default output path for demo."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "cpu_task_demo" / timestamp
    return out_dir / f"{input_path.stem}_cpu.jsonl"


def main() -> None:
    """Run CPU task demo recipe."""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = CpuTaskDemoConfig.from_yaml(str(config_path))
    config.cpu_stage_concurrency = config.cpu_stage_concurrency or config.concurrency

    print("=" * 60)
    print("CPU Task Demo Recipe")
    print("=" * 60)
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Config:      {config_path}")
    print(f"  prime_limit: {config.prime_limit}")
    print(f"  rounds:      {config.rounds}")
    print(f"  batch_size:  {config.batch_size}")
    print(f"  concurrency: {config.concurrency}")
    print("=" * 60)

    init_ray(args)

    recipe = CpuTaskDemoRecipe(config)
    pipeline = Pipeline(
        recipe=recipe,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
        stage_concurrency={"CpuIntensiveStage": config.cpu_stage_concurrency},
        preserve_order=not args.no_preserve_order,
        resume=not args.no_resume,
        flush_interval=config.flush_interval,
    )

    print("[INFO] Running CPU-intensive pipeline...\n")
    pipeline.run(str(input_path), str(output_path))
    print(f"\n[DONE] Output written to: {output_path}")


if __name__ == "__main__":
    main()
