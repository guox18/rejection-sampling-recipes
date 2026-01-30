#!/usr/bin/env python3
"""
SFT recipe entrypoint.

Notes:
    - If images are included, run scripts/preprocess_images.py to add absolute
      image path fields (defaults to abs_image_path_field in config.yaml).
    - If metadata.short_answer is empty, it will be parsed from the assistant
      answer and filled in automatically.

Usage:
    # Process a single file (defaults to sft/timestamp/ under the input parent)
    python run.py --input data/train.jsonl
    # Output: data/sft/YYYYMMDD_HHMMSS/sft_train.jsonl

    # Process multiple files (reuses the same Ray session)
    python run.py --input data/train.jsonl data/test.jsonl data/val.jsonl
    # Output: data/sft/YYYYMMDD_HHMMSS/sft_train.jsonl
    #         data/sft/YYYYMMDD_HHMMSS/sft_test.jsonl
    #         data/sft/YYYYMMDD_HHMMSS/sft_val.jsonl

    # Custom output directory
    python run.py --input data/*.jsonl --output-dir results/exp001
    # Output: results/exp001/sft_YYYYMMDD_HHMMSS/sft_*.jsonl

    # Custom SFT subdirectory name
    python run.py --input data/train.jsonl --sft-subdir custom_output
    # Output: data/custom_output/YYYYMMDD_HHMMSS/train_sft.jsonl

    # Connect to a Ray cluster
    python run.py --input data/*.jsonl --ray-address auto
"""

import os

# Disable Ray runtime_env auto-detection (shared storage, no code shipping needed).
os.environ.setdefault("RAY_RUNTIME_ENV_HOOK_ENABLED", "0")

# Silence Ray warnings.
os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
os.environ.setdefault("RAY_DISABLE_MEMORY_MONITOR", "1")
os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # Silence Python warnings.

import argparse
import json
import signal
import sys

# When launched via python -m, the working directory is repo root.
PROJECT_ROOT = os.getcwd()

# Dynamically import the current recipe module (easy to copy directories).
from importlib import import_module  # noqa: E402

_recipe_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFTConfig = import_module(f"recipes.{_recipe_name}.config").SFTConfig
SFTRecipe = import_module(f"recipes.{_recipe_name}.recipe").SFTRecipe

import ray  # noqa: E402

from src.pipeline import Pipeline  # noqa: E402

# Global: track current file being processed.
current_processing_file = None


def signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    print("\n\n" + "=" * 60)
    print("⚠️  Received interrupt signal (Ctrl+C)")
    print("=" * 60)
    if current_processing_file:
        print("📄 Currently processing:")
        print(f"  Input:  {current_processing_file['input']}")
        print(f"  Output: {current_processing_file['output']}")
        print(f"  Progress: {current_processing_file['index']}/{current_processing_file['total']}")
    else:
        print("  No file is being processed")
    print("=" * 60)
    print("🛑 Aborted")
    sys.exit(130)  # 130 = 128 + SIGINT(2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Recipe: sample → verify → format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",  # Supports multiple input files.
        required=True,
        help="Input data path(s) in JSONL format (supports multiple files)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help=(
            "Output directory. If set, creates sft_YYYYMMDD_HHMMSS/ under it; "
            "otherwise creates sft/YYYYMMDD_HHMMSS/ under each input's parent."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_sft",
        help="Output filename suffix, e.g. train.jsonl -> train_sft.jsonl",
    )
    parser.add_argument(
        "--sft-subdir",
        type=str,
        default="sft",
        help="SFT output subdirectory name (default: 'sft')",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Resume from the latest timestamped output directory. If none exists, create a new one."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        help="Config file path",
    )

    # Ray configuration
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help=(
            "Ray cluster address. Empty starts local mode; 'auto' to detect; or 'ray://IP:10001'."
        ),
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

    # Other options
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

    # Error detection options
    parser.add_argument(
        "--error-detection",
        action="store_true",
        help="Enable error detection; stop when InternalServerError rate is too high",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.5,
        help="InternalServerError rate threshold (0.0-1.0), default 0.5 (50%%)",
    )

    return parser.parse_args()


def init_ray(args: argparse.Namespace) -> None:
    """Initialize Ray."""
    if ray.is_initialized():
        print("✅ Ray already initialized; reusing existing connection")
        return

    # Shared storage mode: disable runtime_env; all nodes access the shared repo directly.
    init_kwargs = {
        "runtime_env": {},
        "logging_level": "ERROR",  # Errors only.
        "log_to_driver": False,  # Disable driver logs.
    }

    if args.ray_address:
        # Connect to an existing cluster.
        print(f"🔗 Connecting to Ray cluster: {args.ray_address}")
        init_kwargs["address"] = args.ray_address
    else:
        # Local mode.
        if args.num_cpus is not None:
            init_kwargs["num_cpus"] = args.num_cpus
        if args.num_gpus is not None:
            init_kwargs["num_gpus"] = args.num_gpus
        print("🚀 Starting Ray in local mode")

    ray.init(**init_kwargs)

    # Print cluster info.
    resources = ray.cluster_resources()
    print(f"   CPU: {resources.get('CPU', 0):.0f}")
    print(f"   GPU: {resources.get('GPU', 0):.0f}")
    print(f"   Memory: {resources.get('memory', 0) / 1e9:.1f} GB")


def find_latest_timestamp_dir(output_dir: str) -> str | None:
    """
    Find the latest timestamp directory (YYYYMMDD_HHMMSS) under an output dir.

    Args:
        output_dir: output directory path

    Returns:
        full path of the latest timestamp directory, or None if not found
    """
    if not os.path.exists(output_dir):
        return None

    # Find subdirs matching YYYYMMDD_HHMMSS (8-digit date + 6-digit time).
    timestamp_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and len(item) == 15 and item[8] == "_":
            # Simple check: length 15, underscore at position 9.
            timestamp_dirs.append(item)

    if not timestamp_dirs:
        return None

    # Sort as strings; timestamp format is naturally sortable.
    latest = sorted(timestamp_dirs)[-1]
    latest_path = os.path.join(output_dir, latest)
    return latest_path


def generate_output_path(input_path: str, output_dir: str, suffix: str) -> str:
    """
    Generate output path from input path.

    Rules:
        - keep input filename, add suffix
        - place output under output_dir

    Example:
        input: a/b/c/train.jsonl    → output_dir/train_sft.jsonl
    """
    input_filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    output_filename = f"{name_without_ext}{suffix}.jsonl"

    return os.path.join(output_dir, output_filename)


def check_internal_server_errors(
    output_file: str, error_threshold: float
) -> tuple[bool, int, int, int]:
    """
    Check InternalServerError rate in an output file.

    Scan the output file and count InternalServerError occurrences.
    Returns True if the error rate exceeds the threshold.

    Args:
        output_file: output file path
        error_threshold: InternalServerError rate threshold (0.0-1.0)

    Returns:
        (should_stop, total_items, failed_items, internal_server_errors)
        - should_stop: whether to stop processing
        - total_items: total items
        - failed_items: failed items (all errors)
        - internal_server_errors: InternalServerError count
    """
    if not os.path.exists(output_file):
        return False, 0, 0, 0

    total_items = 0
    non_skipped_items = 0
    failed_items = 0
    internal_server_errors = 0

    with open(output_file) as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    total_items += 1

                    if item.get("metadata", {}).get("skipped") is not True:
                        non_skipped_items += 1

                    # Check failures.
                    if item.get("_failed") is True:
                        failed_items += 1

                        # Check traceback for InternalServerError.
                        traceback_str = item.get("_traceback") or ""
                        if "InternalServerError" in traceback_str:
                            internal_server_errors += 1

                except json.JSONDecodeError:
                    continue

    if total_items == 0:
        return False, 0, 0, 0

    # Compute InternalServerError rate.
    error_rate = internal_server_errors / non_skipped_items
    should_stop = error_rate > error_threshold

    return should_stop, non_skipped_items, failed_items, internal_server_errors


def main():
    """Run the SFT recipe."""
    # Register signal handler.
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    # Normalize input files.
    input_files = args.input if isinstance(args.input, list) else [args.input]

    # Validate input files.
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            sys.exit(1)

    # Validate config file.
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Auto-generate output paths.
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Decide output paths.
    if args.output_dir:
        # User specified output dir.
        if args.latest:
            # Try resuming from latest timestamp dir.
            latest_dir = find_latest_timestamp_dir(args.output_dir)
            if latest_dir:
                output_base_dir = latest_dir
            else:
                output_base_dir = os.path.join(args.output_dir, f"{args.sft_subdir}_{timestamp}")
        else:
            # Create a new timestamp dir.
            output_base_dir = os.path.join(args.output_dir, f"{args.sft_subdir}_{timestamp}")

        output_files = [
            generate_output_path(input_file, output_base_dir, args.output_suffix)
            for input_file in input_files
        ]
    else:
        # No output dir: create sft/YYYYMMDD_HHMMSS/ under each input parent.
        output_files = []
        for input_file in input_files:
            input_dir = os.path.dirname(os.path.abspath(input_file))
            sft_base_dir = os.path.join(input_dir, args.sft_subdir)

            if args.latest:
                # Try resuming from latest timestamp dir.
                latest_dir = find_latest_timestamp_dir(sft_base_dir)
                if latest_dir:
                    output_base_dir = latest_dir
                else:
                    output_base_dir = os.path.join(sft_base_dir, timestamp)
            else:
                output_base_dir = os.path.join(sft_base_dir, timestamp)

            output_file = generate_output_path(input_file, output_base_dir, args.output_suffix)
            output_files.append(output_file)

    # Create output directories.
    for output_file in output_files:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load config.
    config = SFTConfig.from_yaml(args.config)

    # Defaults: sampler/verifier_concurrency fall back to concurrency.
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
    print(
        f"  Resume:      {'Latest' if args.latest else 'Disabled' if args.no_resume else 'Enabled'}"
    )
    print(f"  Files:       {len(input_files)}")
    print("=" * 60)

    # Show input/output mapping (simplified paths).
    print("\n📁 Output Directory:")
    # Use the first output dir as representative.
    if output_files:
        output_dir_display = os.path.dirname(output_files[0])
        print(f"  {output_dir_display}")
        print(
            f"\n  Files: {len(output_files)} → {', '.join([os.path.basename(f) for f in output_files[:3]])}"
        )
        if len(output_files) > 3:
            print(f"         ... and {len(output_files) - 3} more")
    print()

    # Initialize Ray once (reused across files).
    init_ray(args)

    # Create recipe once.
    recipe = SFTRecipe(config)

    # Create pipeline once.
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

    # Process all files (reuse the same Ray session and pipeline).
    print("🚀 Running pipeline...\n")

    total_success = 0
    total_failed = 0

    for i, (input_file, output_file) in enumerate(zip(input_files, output_files, strict=True), 1):
        # Update global current file info.
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

            # Error detection: check InternalServerError rate.
            if args.error_detection:
                should_stop, non_skipped_item, failed, internal_errors = (
                    check_internal_server_errors(output_file, args.error_threshold)
                )

                if non_skipped_item > 0:
                    error_rate = internal_errors / non_skipped_item
                    print("\n🔍 Error Detection:")
                    print(f"  Non_skipped_item items:           {non_skipped_item}")
                    print(f"  Failed items:          {failed}")
                    print(f"  InternalServerError:   {internal_errors}")
                    print(f"  Error rate:            {error_rate:.2%}")
                    print(f"  Threshold:             {args.error_threshold:.2%}")

                    if should_stop:
                        print("\n⚠️  WARNING: InternalServerError rate too high!")
                        print(
                            f"  Rate {error_rate:.2%} exceeds threshold {args.error_threshold:.2%}"
                        )
                        print("  This may indicate the remote service is down or unstable.")
                        print("  Stopping to avoid producing large amounts of failed data.")
                        print(f"  Files processed: {i}/{len(input_files)}")
                        print(f"  Output dir: {os.path.dirname(output_file)}")
                        sys.exit(1)

        except Exception as e:
            print(f"❌ Failed to process {input_file}: {e}")
            total_failed += 1

        print()

    # Clear global state.
    current_processing_file = None

    # Summary
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
