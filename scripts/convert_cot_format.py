#!/usr/bin/env python3
"""
CoT Format Converter - Convert between R1 and OSS CoT formats.

Formats:
- R1 format: <think>thinking content</think>final answer
- OSS format: <|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>answer

Usage:
    # Convert R1 to OSS format
    python scripts/convert_cot_format.py input.jsonl output.jsonl --to oss

    # Convert OSS to R1 format
    python scripts/convert_cot_format.py input.jsonl output.jsonl --to r1

    # Specify custom field path for non-standard JSONL
    python scripts/convert_cot_format.py input.jsonl output.jsonl --to r1 --field response

    # Read from stdin, write to stdout
    cat input.jsonl | python scripts/convert_cot_format.py - - --to oss
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

# ============================================================================
# Format detection and conversion
# ============================================================================


def detect_format(text: str) -> str | None:
    """
    Detect the CoT format of a text.

    Returns:
        "r1" if R1 format (<think>...</think>)
        "oss" if OSS format (<|channel|>analysis<|message|>...)
        None if no recognized format
    """
    if "<think>" in text or "</think>" in text:
        return "r1"
    if "<|channel|>analysis<|message|>" in text:
        return "oss"
    return None


def parse_r1_format(text: str) -> tuple[str, str]:
    """
    Parse R1 format into (thinking, answer) parts.

    Returns:
        Tuple of (thinking_content, final_answer)
    """
    thinking = ""
    answer = text.strip()

    if "<think>" in text:
        if "</think>" in text:
            # Complete format
            parts = text.split("</think>", 1)
            thinking_part = parts[0]
            if "<think>" in thinking_part:
                thinking = thinking_part.split("<think>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            answer = parts[-1].strip()
        else:
            # Incomplete - only thinking, no answer
            thinking = text.replace("<think>", "").strip()
            answer = ""
    elif "</think>" in text:
        # Edge case: </think> without <think>
        parts = text.split("</think>", 1)
        thinking = parts[0].strip()
        answer = parts[-1].strip()

    return thinking, answer


def parse_oss_format(text: str) -> tuple[str, str]:
    """
    Parse OSS format into (thinking, answer) parts.

    Returns:
        Tuple of (thinking_content, final_answer)
    """
    thinking = ""
    answer = text.strip()

    separator = "<|end|><|start|>assistant<|channel|>final<|message|>"

    if "<|channel|>analysis<|message|>" in text:
        if separator in text:
            # Complete format
            parts = text.split(separator, 1)
            thinking_part = parts[0]
            if "<|channel|>analysis<|message|>" in thinking_part:
                thinking = thinking_part.split("<|channel|>analysis<|message|>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            answer = parts[-1].strip()
        else:
            # Incomplete - only analysis, no final
            thinking = text.replace("<|channel|>analysis<|message|>", "").strip()
            answer = ""

    return thinking, answer


def to_r1_format(thinking: str, answer: str) -> str:
    """
    Convert thinking and answer to R1 format.
    """
    if thinking:
        return f"<think>{thinking}</think>{answer}"
    return answer


def to_oss_format(thinking: str, answer: str) -> str:
    """
    Convert thinking and answer to OSS format.
    """
    if thinking:
        return f"<|channel|>analysis<|message|>{thinking}<|end|><|start|>assistant<|channel|>final<|message|>{answer}"
    return answer


def convert_text(text: str, target_format: str) -> str:
    """
    Convert text from one format to another.

    Args:
        text: The text containing CoT
        target_format: "r1" or "oss"

    Returns:
        Converted text
    """
    source_format = detect_format(text)

    if source_format is None:
        # No CoT detected, return as-is
        return text

    if source_format == target_format:
        # Already in target format
        return text

    # Parse based on source format
    if source_format == "r1":
        thinking, answer = parse_r1_format(text)
    else:  # oss
        thinking, answer = parse_oss_format(text)

    # Convert to target format
    if target_format == "r1":
        return to_r1_format(thinking, answer)
    else:  # oss
        return to_oss_format(thinking, answer)


# ============================================================================
# JSONL processing
# ============================================================================


def get_nested_value(obj: dict, field_path: str):
    """Get value from nested dict using dot notation."""
    keys = field_path.split(".")
    value = obj
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def set_nested_value(obj: dict, field_path: str, value) -> dict:
    """Set value in nested dict using dot notation."""
    keys = field_path.split(".")
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return obj


def convert_messages_format(record: dict, target_format: str) -> dict:
    """
    Convert CoT format in standard messages format.

    Looks for assistant messages and converts their content.
    """
    if "messages" not in record:
        return record

    record = record.copy()
    messages = record["messages"].copy()

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "content" in msg:
            messages[i] = msg.copy()
            messages[i]["content"] = convert_text(msg["content"], target_format)

    record["messages"] = messages
    return record


def convert_record(record: dict, target_format: str, field: str | None = None) -> dict:
    """
    Convert a single record.

    Args:
        record: JSON record to convert
        target_format: "r1" or "oss"
        field: Optional specific field to convert. If None, uses standard "messages" format.

    Returns:
        Converted record
    """
    if field:
        # Custom field specified
        value = get_nested_value(record, field)
        if value and isinstance(value, str):
            converted = convert_text(value, target_format)
            record = record.copy()
            set_nested_value(record, field, converted)
        return record
    else:
        # Standard messages format
        return convert_messages_format(record, target_format)


def process_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    target_format: str,
    field: str | None = None,
    verbose: bool = False,
):
    """
    Process JSONL file and convert formats.

    Args:
        input_stream: Input file stream
        output_stream: Output file stream
        target_format: "r1" or "oss"
        field: Optional specific field to convert
        verbose: Print progress info
    """
    total = 0
    converted = 0
    errors = 0

    for line_num, line in enumerate(input_stream, 1):
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            converted_record = convert_record(record, target_format, field)
            output_stream.write(json.dumps(converted_record, ensure_ascii=False) + "\n")
            total += 1

            # Check if conversion happened
            if record != converted_record:
                converted += 1

        except json.JSONDecodeError as e:
            errors += 1
            if verbose:
                print(f"Warning: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
            # Write original line as-is
            output_stream.write(line + "\n")

    if verbose:
        print(f"\nProcessed: {total} records", file=sys.stderr)
        print(f"Converted: {converted} records", file=sys.stderr)
        if errors:
            print(f"Errors: {errors} lines", file=sys.stderr)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoT format between R1 (<think>) and OSS (<|channel|>) formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert R1 to OSS format
  %(prog)s input.jsonl output.jsonl --to oss

  # Convert OSS to R1 format
  %(prog)s input.jsonl output.jsonl --to r1

  # Use stdin/stdout (use - for stdin/stdout)
  cat input.jsonl | %(prog)s - - --to oss > output.jsonl

  # Convert a specific field instead of messages
  %(prog)s input.jsonl output.jsonl --to r1 --field response

  # Verbose output
  %(prog)s input.jsonl output.jsonl --to oss -v

Formats:
  R1:  <think>thinking content</think>final answer
  OSS: <|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>answer
        """,
    )

    parser.add_argument(
        "input",
        help="Input JSONL file (use - for stdin)",
    )
    parser.add_argument(
        "output",
        help="Output JSONL file (use - for stdout)",
    )
    parser.add_argument(
        "--to",
        "-t",
        required=True,
        choices=["r1", "oss"],
        dest="target_format",
        help="Target format: r1 (<think>) or oss (<|channel|>)",
    )
    parser.add_argument(
        "--field",
        "-f",
        default=None,
        help="Specific field to convert (dot notation for nested, e.g., 'response' or 'data.content'). "
        "If not specified, converts assistant messages in standard 'messages' format.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    # Handle input stream
    if args.input == "-":
        input_stream = sys.stdin
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        input_stream = open(input_path, encoding="utf-8")

    # Handle output stream
    if args.output == "-":
        output_stream = sys.stdout
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_stream = open(output_path, "w", encoding="utf-8")

    try:
        if args.verbose:
            print(f"Converting to {args.target_format.upper()} format...", file=sys.stderr)

        process_jsonl(
            input_stream,
            output_stream,
            args.target_format,
            args.field,
            args.verbose,
        )

        if args.verbose:
            print("Done!", file=sys.stderr)

    finally:
        if args.input != "-":
            input_stream.close()
        if args.output != "-":
            output_stream.close()


if __name__ == "__main__":
    main()
