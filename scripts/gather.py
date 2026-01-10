import os
import pandas as pd
import json
import argparse
from tqdm import tqdm
from typing import Dict


def write_jsonl(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    # print(f'saving file at {datapath}')
    with open(datapath, "w", encoding="utf-8") as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")


def write_json(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    # print(f'saving file at {datapath}')
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(datapath, "w", encoding="utf-8") as json_file:
        json_file.write(json_str)


def read_jsonl(datapath):
    res = []
    # print(f'reading file at {datapath}')
    with open(datapath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def read_json(datapath):
    # print(f'reading file at {datapath}')
    with open(datapath, "r", encoding="utf-8") as f:
        res = json.load(f)
    return res


def write_parquet(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    # print(f'saving parquet file at {datapath}')
    df = pd.DataFrame(data)
    df.to_parquet(datapath, index=False)


### Here! Modify the processing function
def process_fn(example: Dict, qwen30ba3bfile_dict: Dict, qwen235ba22bfile_dict: Dict) -> Dict:
    """
    Process a single sample.

    Args:
        example: original sample
        qwen30ba3bfile: 30b file
        qwen235ba22bfile: 235b file

    Returns:
        processed sample
    """
    assert example["id"] == qwen30ba3bfile_dict["id"]
    assert example["id"] == qwen235ba22bfile_dict["id"]

    def get_assistant_content(example):
        for msg in example.get("messages", []):
            if msg.get("role") == "assistant":
                return msg.get("content")
        return None

    def set_assistant_content(example, content):
        for msg in example.get("messages", []):
            if msg.get("role") == "assistant":
                msg["content"] = content
                break
        return example

    # Initialize workload field.
    workload = {
        "original_assistant_content": get_assistant_content(example),
        "source": "origin",  # Default to original data.
    }

    qwen235b_metadata = qwen235ba22bfile_dict.get("metadata") or {}
    if qwen235b_metadata.get("used_ground_truth") is False:
        # Be careful with defaults: branches include True, False, and None/missing.
        # Prefer `is` checks; `is not` can be ambiguous here.
        workload["source"] = "qwen3vl_235b_a22b_thinking"
        workload["n_passed"] = qwen235ba22bfile_dict["metadata"]["n_passed"]
        workload["n_total"] = qwen235ba22bfile_dict["metadata"]["n_total"]
        set_assistant_content(example, get_assistant_content(qwen235ba22bfile_dict))

    qwen30b_metadata = qwen30ba3bfile_dict.get("metadata") or {}
    if qwen30b_metadata.get("used_ground_truth") is False:
        workload["source"] = "qwen3vl_30b_a3b_thinking"
        workload["n_passed"] = qwen30ba3bfile_dict["metadata"]["n_passed"]
        workload["n_total"] = qwen30ba3bfile_dict["metadata"]["n_total"]
        set_assistant_content(example, get_assistant_content(qwen30ba3bfile_dict))

    # If neither model generated new data, keep original.
    example["workload"] = workload

    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data processing script")
    # Here! Fill in input/output.
    parser.add_argument("--input_path", type=str, default="", help="Input file path")
    parser.add_argument(
        "--qwen3vl30ba3bthinking_path", type=str, default="", help="Input file path"
    )
    parser.add_argument(
        "--qwen3vl235ba22bthinking_path", type=str, default="", help="Input file path"
    )
    parser.add_argument(
        "--output_path", type=str, default="", help="Output file path"
    )  # If empty, defaults to _train.jsonl

    args = parser.parse_args()

    # Read data.
    data = read_jsonl(args.input_path)
    qwen30ba3bfile = read_jsonl(args.qwen3vl30ba3bthinking_path)
    qwen235ba22bfile = read_jsonl(args.qwen3vl235ba22bthinking_path)
    assert len(data) == len(qwen30ba3bfile)
    assert len(data) == len(qwen235ba22bfile)
    # print(f'Read {len(data)} items')

    # Process data.
    data = [
        process_fn(item, qwen30ba3bfile[idx], qwen235ba22bfile[idx])
        for idx, item in tqdm(enumerate(data))
    ]

    # Count how many are not origin.
    n_not_origin = sum(1 for item in data if item["workload"]["source"] != "origin")
    n_total = len(data)
    print(f"Replacement rate: {n_not_origin / n_total * 100}%")
    # Save data.
    write_jsonl(data, args.output_path)
    # print(f'Completed processing {len(data)} items')
