#!/usr/bin/env python3
"""Benchmark runner for cpu_task_demo across Ray cluster sizes."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import ray

from recipes.cpu_task_demo.config import CpuTaskDemoConfig
from recipes.cpu_task_demo.entrypoint.generate_heavy_mock import generate_heavy_mock
from recipes.cpu_task_demo.recipe import CpuTaskDemoRecipe
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cpu_task_demo on current Ray cluster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="tests/mock/cpu_task_heavy.jsonl",
        help="Benchmark input JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (auto-generated if omitted)",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help="Ray cluster address",
    )
    parser.add_argument(
        "--target-concurrency",
        type=int,
        default=80,
        help="Upper bound of stage concurrency",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Pipeline batch size (1 recommended for CPU saturation)",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=100,
        help="Output flush interval",
    )
    parser.add_argument(
        "--prime-limit",
        type=int,
        default=180000,
        help="Fallback prime limit when input item has no prime_limit",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Fallback rounds when input item has no rounds",
    )
    parser.add_argument(
        "--generate-mock-if-missing",
        action="store_true",
        help="Generate heavy mock file if --input does not exist",
    )
    parser.add_argument(
        "--mock-num-items",
        type=int,
        default=240,
        help="Used when --generate-mock-if-missing is enabled",
    )
    return parser.parse_args()


def maybe_generate_mock(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if input_path.exists():
        return
    if not args.generate_mock_if_missing:
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Use --generate-mock-if-missing to create one."
        )

    generate_heavy_mock(
        output=args.input,
        num_items=args.mock_num_items,
        base_prime_limit=args.prime_limit,
        prime_step=3000,
        rounds=args.rounds,
    )


def choose_concurrency(total_cpu: float, target_concurrency: int) -> int:
    """Pick a safe concurrency from current cluster CPU resources."""
    usable = max(1, int(math.floor(total_cpu)) - 2)
    return max(1, min(target_concurrency, usable))


def summarize_output(output_path: Path) -> tuple[int, Counter, float]:
    """Return item count, host distribution, and average elapsed_ms."""
    count = 0
    host_counter: Counter[str] = Counter()
    elapsed_sum = 0.0
    elapsed_count = 0

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            count += 1

            metadata = item.get("metadata") or {}
            host = metadata.get("worker_host", "unknown")
            host_counter[host] += 1

            elapsed = metadata.get("elapsed_ms")
            if isinstance(elapsed, (int, float)):
                elapsed_sum += float(elapsed)
                elapsed_count += 1

    avg_elapsed = elapsed_sum / elapsed_count if elapsed_count > 0 else 0.0
    return count, host_counter, avg_elapsed


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    args.input = str(input_path)

    maybe_generate_mock(args)

    ray.init(address=args.ray_address)
    cluster = ray.cluster_resources()
    total_cpu = float(cluster.get("CPU", 0.0))
    concurrency = choose_concurrency(total_cpu, args.target_concurrency)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path("outputs") / "cpu_task_demo_benchmark" / stamp / "benchmark_output.jsonl"
        ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = CpuTaskDemoConfig(
        prime_limit=args.prime_limit,
        rounds=args.rounds,
        batch_size=args.batch_size,
        concurrency=concurrency,
        cpu_stage_concurrency=concurrency,
        flush_interval=args.flush_interval,
    )

    recipe = CpuTaskDemoRecipe(config)
    pipeline = Pipeline(
        recipe=recipe,
        batch_size=config.batch_size,
        concurrency=config.concurrency,
        stage_concurrency={"CpuIntensiveStage": config.cpu_stage_concurrency},
        preserve_order=False,
        resume=False,
        flush_interval=config.flush_interval,
    )

    print("=" * 70)
    print("CPU Task Benchmark")
    print("=" * 70)
    print(f"Ray address:         {args.ray_address}")
    print(f"Cluster total CPU:   {total_cpu}")
    print(f"Chosen concurrency:  {concurrency}")
    print(f"Input:               {input_path}")
    print(f"Output:              {output_path}")
    print(f"Batch size:          {config.batch_size}")
    print(f"Fallback prime:      {config.prime_limit}")
    print(f"Fallback rounds:     {config.rounds}")
    print("=" * 70)

    start = time.perf_counter()
    pipeline.run(str(input_path), str(output_path))
    wall_seconds = time.perf_counter() - start

    total_items, host_counter, avg_elapsed_ms = summarize_output(output_path)

    print("\n" + "=" * 70)
    print("Benchmark Result")
    print("=" * 70)
    print(f"Wall time (s):       {wall_seconds:.2f}")
    print(f"Processed items:     {total_items}")
    print(f"Avg item elapsed ms: {avg_elapsed_ms:.2f}")
    print("Worker distribution:")
    for host, num in sorted(host_counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {host}: {num}")
    print("=" * 70)


if __name__ == "__main__":
    main()
