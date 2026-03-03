#!/usr/bin/env python3
"""
Filter out records where the assistant output token count exceeds a threshold.

Usage:
    python filter_by_token_count.py input.jsonl output.jsonl \
        --tokenizer /path/to/tokenizer \
        --max-tokens 8192
"""

import argparse
import json

from transformers import AutoTokenizer


def process_file(
    input_file: str,
    output_file: str,
    tokenizer_path: str,
    max_tokens: int,
) -> dict:
    """
    Filter JSONL file by token count of the last assistant message.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        tokenizer_path: Path to HuggingFace tokenizer
        max_tokens: Maximum allowed token count

    Returns:
        Statistics dict with total, kept, and filtered_out counts
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("Tokenizer loaded.")

    total = 0
    kept = 0
    filtered_out = 0

    with open(input_file, encoding="utf-8") as fin:
        with open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                total += 1
                data = json.loads(line.strip())

                # Get the last assistant message content
                content = data["messages"][-1]["content"]

                # Count tokens
                tokens = tokenizer.encode(content, add_special_tokens=False)
                token_count = len(tokens)

                # Filter: skip if token count > max_tokens
                if token_count > max_tokens:
                    filtered_out += 1
                    continue

                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1

                if total % 2000 == 0:
                    print(f"Processed {total} records...")

    return {"total": total, "kept": kept, "filtered_out": filtered_out}


def main():
    parser = argparse.ArgumentParser(
        description="Filter records by token count of assistant output"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSONL file path")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to HuggingFace tokenizer",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum allowed token count (default: 8192)",
    )
    args = parser.parse_args()

    print(f"Processing: {args.input_file}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max tokens: {args.max_tokens}")

    stats = process_file(
        args.input_file,
        args.output_file,
        args.tokenizer,
        args.max_tokens,
    )

    print("\n=== Results ===")
    print(f"Total records: {stats['total']}")
    print(f"Kept (tokens <= {args.max_tokens}): {stats['kept']}")
    print(f"Filtered out (tokens > {args.max_tokens}): {stats['filtered_out']}")
    print(f"Output file: {args.output_file}")


if __name__ == "__main__":
    main()
