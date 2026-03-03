#!/usr/bin/env python3
"""Generate a heavier mock input for CPU task demo benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_heavy_mock(
    output: str,
    num_items: int,
    base_prime_limit: int,
    prime_step: int,
    rounds: int,
) -> None:
    """Generate heavy JSONL mock data for CPU benchmark."""
    if num_items < 1:
        raise ValueError("num_items must be >= 1")
    if base_prime_limit < 2:
        raise ValueError("base_prime_limit must be >= 2")
    if prime_step < 1:
        raise ValueError("prime_step must be >= 1")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(num_items):
            prime_limit = base_prime_limit + (idx % 7) * prime_step
            item = {
                "id": f"cpu-heavy-{idx + 1:04d}",
                "prime_limit": prime_limit,
                "rounds": rounds,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heavy JSONL input for cpu_task_demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/mock/cpu_task_heavy.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument("--num-items", type=int, default=240, help="Number of items")
    parser.add_argument(
        "--base-prime-limit",
        type=int,
        default=180000,
        help="Base upper bound for prime counting",
    )
    parser.add_argument(
        "--prime-step",
        type=int,
        default=3000,
        help="Delta used to vary prime_limit across items",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="How many repeated rounds per item",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_heavy_mock(
        output=args.output,
        num_items=args.num_items,
        base_prime_limit=args.base_prime_limit,
        prime_step=args.prime_step,
        rounds=args.rounds,
    )

    output_path = Path(args.output)

    print("[DONE] Heavy mock generated")
    print(f"  Path:       {output_path}")
    print(f"  Num items:  {args.num_items}")
    print(f"  Prime base: {args.base_prime_limit}")
    print(f"  Prime step: {args.prime_step}")
    print(f"  Rounds:     {args.rounds}")


if __name__ == "__main__":
    main()
