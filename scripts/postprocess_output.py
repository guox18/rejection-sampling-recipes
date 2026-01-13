#!/usr/bin/env python3
"""
Post-process pipeline output data.

This script performs the following operations:
1. Remove framework internal fields (_resume_id, _failed, _error, _traceback)
2. Remove rollouts field
3. Split data based on metadata.used_ground_truth:
   - If used_ground_truth is False: output to xxx_update.jsonl (successfully generated new content)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Set, Tuple

# Framework internal fields (same as src/utils/framework.py)
FRAMEWORK_FIELDS: Set[str] = {"_resume_id", "_failed", "_error", "_traceback"}

# Additional fields to remove
FIELDS_TO_REMOVE: Set[str] = FRAMEWORK_FIELDS | {"rollouts"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def clean_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove framework fields and rollouts from a single item.

    Args:
        item: The dictionary item to clean

    Returns:
        A new dictionary with specified fields removed
    """
    return {k: v for k, v in item.items() if k not in FIELDS_TO_REMOVE}


def is_updated_content(item: Dict[str, Any]) -> bool:
    """
    Check if the item has successfully generated new assistant content.

    An item is considered "updated" if metadata.used_ground_truth is False,
    meaning the model generated new content instead of using the original.

    Args:
        item: The dictionary item to check

    Returns:
        True if used_ground_truth is False, False otherwise
    """
    metadata = item.get("metadata") or {}
    # Be explicit: only return True when used_ground_truth is exactly False
    return metadata.get("used_ground_truth") is False


def process_file(
    input_path: str, output_dir: str = None, verbose: bool = True
) -> Tuple[int, int, int]:
    """
    Process a JSONL file: clean fields and split based on used_ground_truth.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory for output files. If None, uses input file's directory
        verbose: Whether to print detailed information

    Returns:
        Tuple of (total_lines, updated_lines, error_lines)

    Raises:
        FileNotFoundError: If input file does not exist
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output paths
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = input_file.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem  # filename without extension
    update_output_path = out_dir / f"{stem}_update.jsonl"

    total_lines = 0
    updated_lines = 0
    error_lines = 0

    with (
        input_file.open("r", encoding="utf-8") as fin,
        update_output_path.open("w", encoding="utf-8") as f_update,
    ):
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            try:
                item = json.loads(line)

                # Clean the item (remove framework fields and rollouts)
                cleaned_item = clean_item(item)

                # Check if this is an updated item (used_ground_truth is False)
                if is_updated_content(item):
                    updated_lines += 1
                    f_update.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                error_lines += 1
                if verbose:
                    logger.warning("Line %s: JSON decode error - %s", line_num, e)
                continue

    if verbose:
        logger.info("=" * 60)
        logger.info("Processing complete")
        logger.info("=" * 60)
        logger.info("Input file: %s", input_path)
        logger.info("Total lines processed: %s", total_lines)
        logger.info("Updated content (used_ground_truth=False): %s", updated_lines)
        logger.info("Parse errors: %s", error_lines)
        logger.info("-" * 60)
        logger.info("Output file (updated): %s", update_output_path)
        if updated_lines > 0:
            update_rate = updated_lines / total_lines * 100
            logger.info("Update rate: %.2f%%", update_rate)

    return total_lines, updated_lines, error_lines


def main():
    parser = argparse.ArgumentParser(
        description="Post-process pipeline output data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python postprocess_output.py -i output.jsonl

  # Process with custom output directory
  python postprocess_output.py -i output.jsonl -o ./processed/

Operations performed:
  1. Remove framework fields: """
        + ", ".join(sorted(FRAMEWORK_FIELDS))
        + """
  2. Remove rollouts field
  3. Split data: items with metadata.used_ground_truth=False -> xxx_update.jsonl
""",
    )

    parser.add_argument("-i", "--input", required=True, help="Input JSONL file path")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory (default: same as input file directory)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    if not args.quiet:
        print(f"[Info] Input: {args.input}")
        print(f"[Info] Output directory: {args.output_dir or 'same as input'}")
        print(f"[Info] Removing fields: {', '.join(sorted(FIELDS_TO_REMOVE))}")
        print()

    process_file(args.input, args.output_dir, verbose=not args.quiet)

    if not args.quiet:
        print("\n[Done]")


if __name__ == "__main__":
    main()

