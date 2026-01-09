"""
Framework fields and utility functions for data processing.

This module provides constants and functions for managing framework internal fields
that are used by the pipeline system.
"""

import json
from pathlib import Path
from typing import Any, Dict, Set, Tuple

# 框架内部字段（由 pipeline 自动保留）
FRAMEWORK_FIELDS: Set[str] = {"_resume_id", "_failed", "_error", "_traceback"}

def remove_framework_fields(item: Dict[str, Any], fields: Set[str] = None) -> Dict[str, Any]:
    """
    Remove framework internal fields from a single item.

    Args:
        item: The dictionary item to clean
        fields: Set of field names to remove. If None, uses FRAMEWORK_FIELDS

    Returns:
        A new dictionary with framework fields removed

    Example:
        >>> item = {"id": 1, "text": "hello", "_resume_id": "abc123"}
        >>> remove_framework_fields(item)
        {"id": 1, "text": "hello"}
    """
    if fields is None:
        fields = FRAMEWORK_FIELDS

    return {k: v for k, v in item.items() if k not in fields}


def clean_framework_fields_from_file(
    input_path: str, output_path: str, fields: Set[str] = None, verbose: bool = True
) -> Tuple[int, int, int]:
    """
    Clean framework internal fields from a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        fields: Set of field names to remove. If None, uses FRAMEWORK_FIELDS
        verbose: Whether to print detailed information

    Returns:
        Tuple of (total_lines, cleaned_lines, error_lines)

    Raises:
        FileNotFoundError: If input file does not exist

    Example:
        >>> total, cleaned, errors = clean_framework_fields_from_file(
        ...     "input.jsonl", "output.jsonl"
        ... )
        >>> print(f"Cleaned {cleaned} lines out of {total}")
    """
    if fields is None:
        fields = FRAMEWORK_FIELDS

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    cleaned_lines = 0
    error_lines = 0

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            try:
                item = json.loads(line)

                # Track if any fields were removed
                had_fields = any(field in item for field in fields)
                if had_fields:
                    cleaned_lines += 1

                # Remove framework fields
                cleaned_item = remove_framework_fields(item, fields)

                # Write cleaned data
                fout.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                error_lines += 1
                if verbose:
                    print(f"  [Warning] Line {line_num}: JSON decode error - {e}")
                continue

    if verbose:
        print(f"✓ Processed {total_lines} lines")
        print(f"  - Cleaned: {cleaned_lines}")
        print(f"  - Errors: {error_lines}")

    return total_lines, cleaned_lines, error_lines


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="清理 JSONL 文件中的框架内部字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 清理所有框架字段 (默认)
  python -m src.utils.framework -i input.jsonl -o output.jsonl

  # 清理特定字段
  python -m src.utils.framework -i input.jsonl -o output.jsonl -f _resume_id _error

默认清理的字段: """ + ", ".join(sorted(FRAMEWORK_FIELDS)),
    )

    parser.add_argument("-i", "--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        help=f"要清理的字段列表（默认: {' '.join(sorted(FRAMEWORK_FIELDS))}）",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    # 确定要清理的字段
    fields_to_clean = set(args.fields) if args.fields else FRAMEWORK_FIELDS

    if not args.quiet:
        print(f"[Info] Input: {args.input}")
        print(f"[Info] Output: {args.output}")
        print(f"[Info] Removing fields: {', '.join(sorted(fields_to_clean))}")
        print()

    # 清理文件
    clean_framework_fields_from_file(
        args.input, args.output, fields=fields_to_clean, verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\n[Done] 已清理: {args.output}")