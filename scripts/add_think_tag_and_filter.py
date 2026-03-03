#!/usr/bin/env python3
"""
Add <think> tag to the beginning of assistant output and filter out records
where the output doesn't contain </think> (truncated outputs).

Usage:
    python add_think_tag_and_filter.py input.jsonl output.jsonl
"""

import argparse
import json


def process_file(input_file: str, output_file: str) -> dict:
    """
    Process JSONL file:
    1. Add <think> at the beginning of the last assistant message content
    2. Filter out records where content doesn't contain </think>

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file

    Returns:
        Statistics dict with total, kept, and filtered_out counts
    """
    total = 0
    kept = 0
    filtered_out = 0

    with open(input_file, "r", encoding="utf-8") as fin:
        with open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                total += 1
                data = json.loads(line.strip())

                # Find the last assistant message
                last_assistant_idx = None
                for i in range(len(data["messages"]) - 1, -1, -1):
                    if data["messages"][i]["role"] == "assistant":
                        last_assistant_idx = i
                        break

                if last_assistant_idx is None:
                    filtered_out += 1
                    continue

                content = data["messages"][last_assistant_idx]["content"]

                # Filter: skip if no </think> in content (truncated output)
                if "</think>" not in content:
                    filtered_out += 1
                    continue

                # Add <think> at the beginning
                data["messages"][last_assistant_idx]["content"] = "<think>" + content

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1

    return {"total": total, "kept": kept, "filtered_out": filtered_out}


def main():
    parser = argparse.ArgumentParser(
        description="Add <think> tag and filter records without </think>"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Processing: {args.input_file}")
    stats = process_file(args.input_file, args.output_file)

    print(f"\n=== Results ===")
    print(f"Total records: {stats['total']}")
    print(f"Kept (with </think>): {stats['kept']}")
    print(f"Filtered out (no </think>): {stats['filtered_out']}")
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()
