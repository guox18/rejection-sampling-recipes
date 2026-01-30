#!/usr/bin/env python3
"""
Image path preprocessing script.

Functions:
1. Read JSONL files
2. Compute absolute image paths for items with images
3. Store absolute paths in meta_info.abs_image_path (or user-specified field)
4. Check whether image files exist and warn if missing
5. Write a new JSONL file (xxx_abs.jsonl)

Usage:
    # Option 1: provide full image base path directly (no doc_loc needed)
    python scripts/preprocess_images.py \
        --input data/train.jsonl data/test.jsonl \
        --image-base-path /mnt/.../internxx/P~xxx~1.0.0~0.0/multimodal_elements \
        [--abs-image-path-field abs_path]

    # Option 2: infer from doc_loc using a base dir (doc_loc required)
    python scripts/preprocess_images.py \
        --input data/train.jsonl data/test.jsonl \
        --image-base-dir /mnt/.../internxx \
        [--abs-image-path-field abs_path]
"""

import argparse
import json
import os
import re
from pathlib import Path


def infer_image_base_path_from_doc_loc(doc_loc: str, image_base_dir: str) -> str | None:
    """
    Infer image base path from doc_loc.

    Args:
        doc_loc: S3 path, e.g. "s3://.../P~xxx~1.0.0~0.0_suffix/jsonl/part-001.jsonl"
        image_base_dir: base directory that contains image files

    Returns:
        full image path like "{image_base_dir}/P~xxx~1.0.0~0.0/multimodal_elements",
        or None if it cannot be inferred

    Example:
        input:  "s3://.../P~Document_QA~unknown~xxx~1.0.0~0.0_Bo1f7/jsonl/part-001.jsonl"
        config: image_base_dir = "/mnt/.../internxx"
        output: "/mnt/.../internxx/P~Document_QA~unknown~xxx~1.0.0~0.0/multimodal_elements"

    Note:
        Dataset names in doc_loc may include random suffixes (e.g. _Bo1f7-xxx).
        The actual directory name ends at the version (~1.0.0~0.0), so the suffix
        should be stripped.
    """
    if not image_base_dir:
        return None

    # Regex: extract dataset name (P~... up to version).
    # Format: P~xxx~xxx~xxx~x.x.x~x.x
    pattern = r"(P~[^/]+?~\d+\.\d+\.\d+~\d+\.\d+)(?:_[^/]+)?/jsonl/"
    match = re.search(pattern, doc_loc)
    if not match:
        return None

    dataset_name = match.group(1)
    image_path = os.path.join(image_base_dir, dataset_name, "multimodal_elements")

    return image_path


def has_image_content(item: dict) -> bool:
    """
    Check whether an item contains image content.

    Args:
        item: data item

    Returns:
        True if images exist, False if text-only
    """
    for msg in item.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])

        # content may be a string (text-only) or a list (structured).
        if isinstance(content, str):
            continue  # String content has no images.

        # List content: check for image_url elements.
        for content_item in content:
            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                return True

    return False


def extract_image_relative_paths(item: dict) -> list[str]:
    """
    Extract all image relative paths from an item.

    Args:
        item: data item

    Returns:
        list of relative image paths
    """
    image_paths = []

    for msg in item.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])

        # content must be a list to contain images.
        if not isinstance(content, list):
            continue

        for content_item in content:
            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                image_url_data = content_item.get("image_url", {})
                relative_path = image_url_data.get("url", "")
                if relative_path:
                    image_paths.append(relative_path)

    return image_paths


def set_nested_field(item: dict, field_path: str, value) -> None:
    """
    Set a nested field value.

    Args:
        item: data dict
        field_path: dotted path, e.g. "meta_info.abs_image_path" or "abs_path"
        value: value to set
    """
    parts = field_path.split(".")
    current = item

    # Walk to the second-to-last level, creating dicts as needed.
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # Set the final field.
    current[parts[-1]] = value


def preprocess_file(
    input_path: str,
    output_path: str,
    image_base_dir: str,
    abs_image_path_field: str,
    image_base_path: str = None,
) -> dict:
    """
    Preprocess a single JSONL file.

    Args:
        input_path: input JSONL file path
        output_path: output JSONL file path
        image_base_dir: base directory for images (used to infer from doc_loc)
        abs_image_path_field: field name to store absolute path (supports nesting)
        image_base_path: full image base path (optional; if set, skip doc_loc inference)

    Returns:
        stats dict
    """
    stats = {
        "total_items": 0,
        "items_with_images": 0,
        "items_without_images": 0,
        "total_image_files": 0,
        "image_files_exist": 0,
        "image_files_missing": 0,
        "missing_image_details": [],  # Store details of missing images.
    }

    with (
        open(input_path, encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                stats["total_items"] += 1

                # Check for images.
                if not has_image_content(item):
                    stats["items_without_images"] += 1
                    # Text-only: write as-is.
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                stats["items_with_images"] += 1

                # Get image base path.
                if image_base_path:
                    # User provided full path.
                    item_image_base_path = image_base_path
                else:
                    # Infer from doc_loc.
                    doc_loc = item.get("doc_loc", "")
                    item_image_base_path = infer_image_base_path_from_doc_loc(
                        doc_loc, image_base_dir
                    )

                    if not item_image_base_path:
                        print(f"⚠️  Line {line_num}: cannot infer image path, skipping")
                        print(f"   doc_loc: {doc_loc}")
                        print(
                            "   Tip: if your data has no doc_loc, pass --image-base-path "
                            "to specify the full path directly"
                        )
                        continue

                # Extract all relative image paths.
                relative_paths = extract_image_relative_paths(item)

                if not relative_paths:
                    stats["items_without_images"] += 1
                    stats["items_with_images"] -= 1
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                # Check whether image files exist.
                for relative_path in relative_paths:
                    full_path = os.path.join(item_image_base_path, relative_path)
                    stats["total_image_files"] += 1

                    if os.path.exists(full_path):
                        stats["image_files_exist"] += 1
                    else:
                        stats["image_files_missing"] += 1
                        stats["missing_image_details"].append(
                            {
                                "line": line_num,
                                "item_id": item.get("id", "unknown"),
                                "relative_path": relative_path,
                                "full_path": full_path,
                            }
                        )
                        print(f"❌ Line {line_num}: image file not found")
                        print(f"   ID: {item.get('id', 'unknown')}")
                        print(f"   Relative path: {relative_path}")
                        print(f"   Full path: {full_path}")

                # Store absolute path in target field.
                set_nested_field(item, abs_image_path_field, item_image_base_path)

                # Write output.
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"⚠️  Line {line_num}: JSON decode error - {e}")
            except Exception as e:
                print(f"⚠️  Line {line_num}: processing error - {e}")

    return stats


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Image path preprocessing: add absolute image paths to JSONL items",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="Input JSONL file paths (supports multiple files)",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default=None,
        help="Base directory for images (infer from doc_loc), e.g. /mnt/.../internxx",
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        default=None,
        help=(
            "Full image base path (if set, skip doc_loc inference), "
            "e.g. /mnt/.../internxx/P~xxx~1.0.0~0.0/multimodal_elements"
        ),
    )
    parser.add_argument(
        "--abs-image-path-field",
        type=str,
        default="abs_path",
        help="Field name to store absolute paths (supports nesting, e.g. meta_info.abs_image_path)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_abs",
        help="Output filename suffix, e.g. train.jsonl -> train_abs.jsonl",
    )

    args = parser.parse_args()

    # Validate input files.
    input_files = args.input
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return

    # Validate image path config.
    if not args.image_base_dir and not args.image_base_path:
        print("❌ You must specify --image-base-dir or --image-base-path")
        return

    if args.image_base_dir and not os.path.exists(args.image_base_dir):
        print(f"❌ Image base dir not found: {args.image_base_dir}")
        return

    if args.image_base_path and not os.path.exists(args.image_base_path):
        print(f"❌ Image base path not found: {args.image_base_path}")
        return

    print("=" * 80)
    print("Image Path Preprocessing")
    print("=" * 80)
    print(f"Input files: {len(input_files)}")
    if args.image_base_path:
        print(f"Image base path: {args.image_base_path} (explicit)")
    else:
        print(f"Image base dir:  {args.image_base_dir} (inferred from doc_loc)")
    print(f"Absolute path field: {args.abs_image_path_field}")
    print(f"Output suffix:        {args.output_suffix}")
    print("=" * 80)
    print()

    # Process each file.
    total_stats = {
        "total_items": 0,
        "items_with_images": 0,
        "items_without_images": 0,
        "total_image_files": 0,
        "image_files_exist": 0,
        "image_files_missing": 0,
        "missing_image_details": [],
    }

    for input_file in input_files:
        print(f"\n{'=' * 80}")
        print(f"Processing file: {input_file}")
        print(f"{'=' * 80}")

        # Build output file path.
        input_path = Path(input_file)
        output_filename = f"{input_path.stem}{args.output_suffix}{input_path.suffix}"
        output_path = input_path.parent / output_filename

        print(f"Output file: {output_path}")

        # Preprocess file.
        stats = preprocess_file(
            input_path=str(input_path),
            output_path=str(output_path),
            image_base_dir=args.image_base_dir,
            abs_image_path_field=args.abs_image_path_field,
            image_base_path=args.image_base_path,
        )

        # Print stats.
        print("\nFile stats:")
        print(f"  Total items:     {stats['total_items']}")
        print(f"  With images:     {stats['items_with_images']}")
        print(f"  Without images:  {stats['items_without_images']}")
        print(f"  Total images:    {stats['total_image_files']}")
        print(f"  Images present:  {stats['image_files_exist']}")
        print(f"  Images missing:  {stats['image_files_missing']}")

        if stats["image_files_missing"] > 0:
            print(f"\n⚠️  Warning: {stats['image_files_missing']} missing image files")

        # Accumulate totals.
        for key in total_stats:
            if key == "missing_image_details":
                total_stats[key].extend(stats[key])
            else:
                total_stats[key] += stats[key]

    # Print overall stats.
    print(f"\n\n{'=' * 80}")
    print("Overall Stats")
    print(f"{'=' * 80}")
    print(f"Total items:     {total_stats['total_items']}")
    print(f"With images:     {total_stats['items_with_images']}")
    print(f"Without images:  {total_stats['items_without_images']}")
    print(f"Total images:    {total_stats['total_image_files']}")
    print(f"Images present:  {total_stats['image_files_exist']}")
    print(f"Images missing:  {total_stats['image_files_missing']}")

    if total_stats["total_image_files"] > 0:
        exist_rate = 100 * total_stats["image_files_exist"] / total_stats["total_image_files"]
        print(f"Image existence rate:   {exist_rate:.2f}%")

    # Print missing image details.
    if total_stats["missing_image_details"]:
        print(f"\n\n{'=' * 80}")
        print("Missing image details (first 3)")
        print(f"{'=' * 80}")
        for detail in total_stats["missing_image_details"][:3]:
            print(f"  Line {detail['line']}, ID {detail['item_id']}")
            print(f"    Relative path: {detail['relative_path']}")
            print(f"    Full path:     {detail['full_path']}")
            print()

        if len(total_stats["missing_image_details"]) > 3:
            print(f"  ... and {len(total_stats['missing_image_details']) - 3} more missing records")

    print(f"\n{'=' * 80}")
    print("✅ Preprocessing complete")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
