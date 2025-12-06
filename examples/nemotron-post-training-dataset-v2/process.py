"""
数据处理脚本：解析 boxed 答案并转换为拒绝采样框架格式

功能：
1. 从原始数据中提取 \\boxed{} 中的答案
2. 只保留单选题（答案为 A-Z 的题目）
3. 转换为框架要求的格式：
   {
       "id": str,
       "messages": [{"role": "user", "content": str}],
       "metadata": {"answer": str, ...}
   }

用法：
    python process.py --input stem.jsonl --output stem_processed.jsonl
"""

import argparse
import json
import os
import re
from pathlib import Path

# ============== 答案提取工具函数 ==============


def remove_boxed(s: str) -> str | None:
    """从 \\boxed{...} 格式中提取内容"""
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string: str) -> str | None:
    """找到字符串中最后一个 \\boxed{} 或 \\fbox{} 并返回完整的 boxed 字符串"""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def extract_boxed_answer(solution: str) -> str | None:
    """从 LaTeX \\boxed{} 中提取答案"""
    boxed_str = last_boxed_only_string(solution)
    if boxed_str is None:
        return None
    return remove_boxed(boxed_str)


def extract_answer(passage: str) -> str | None:
    """
    从文本中提取答案，支持多种格式：
    1. <answer>...</answer> 标签
    2. \\boxed{...} LaTeX 格式
    """
    if "<answer>" in passage and "</answer>" in passage:
        return passage.split("<answer>")[1].split("</answer>")[0]
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def normalize_answer(answer: str) -> str:
    """
    标准化答案，去除常见的包装格式
    例如: \\text{A} -> A, \\textbf{B} -> B
    """
    if answer is None:
        return None

    answer = answer.strip()

    # 处理 \text{X}, \textbf{X}, \mathrm{X} 等格式
    patterns = [
        r"^\\text\{(.+?)\}$",
        r"^\\textbf\{(.+?)\}$",
        r"^\\mathrm\{(.+?)\}$",
        r"^\\mathbf\{(.+?)\}$",
    ]

    for pattern in patterns:
        m = re.match(pattern, answer)
        if m:
            answer = m.group(1).strip()
            break

    return answer


def is_valid_mcq_answer(answer: str) -> bool:
    """检查答案是否为有效的单选题答案 (A-Z)"""
    if answer is None:
        return False
    normalized = normalize_answer(answer)
    return (
        normalized is not None
        and len(normalized) == 1
        and normalized.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )


# ============== 文件 I/O ==============


def read_jsonl(datapath: str) -> list:
    """读取 JSONL 格式文件"""
    res = []
    print(f"读取文件: {datapath}")
    with open(datapath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                res.append(json.loads(line))
    return res


def write_jsonl(data: list, datapath: str) -> None:
    """保存数据为 JSONL 格式"""
    os.makedirs(os.path.dirname(datapath) if os.path.dirname(datapath) else ".", exist_ok=True)
    print(f"保存文件至: {datapath}")
    with open(datapath, "w", encoding="utf-8") as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")


# ============== 数据处理 ==============


class ProcessStats:
    """处理统计信息"""

    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed_no_answer = 0
        self.failed_not_mcq = 0
        self.failed_logs = []

    def log_failure(self, idx: int, item: dict, reason: str, extracted: str | None = None):
        self.failed_logs.append(
            {
                "idx": idx,
                "uuid": item.get("uuid", "unknown"),
                "reason": reason,
                "extracted_answer": extracted,
            }
        )

    def print_summary(self):
        print("\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        print(f"总数据量: {self.total}")
        print(f"成功处理: {self.success} ({self.success / self.total * 100:.2f}%)")
        print(f"失败 - 无法提取答案: {self.failed_no_answer}")
        print(f"失败 - 非单选题: {self.failed_not_mcq}")
        print("=" * 60)


def process_item(item: dict, idx: int, stats: ProcessStats) -> dict | None:
    """
    处理单个数据样本

    Args:
        item: 原始数据样本（包含 messages 字段）
        idx: 样本索引
        stats: 统计信息对象

    Returns:
        转换后的数据样本，如果无效则返回 None
    """
    stats.total += 1

    # 提取 assistant 的回复内容
    assistant_content = None
    user_content = None

    if "messages" not in item:
        stats.failed_no_answer += 1
        stats.log_failure(idx, item, "no_messages_field")
        return None

    for message in item["messages"]:
        if message["role"] == "assistant":
            assistant_content = message.get("content", "")
        elif message["role"] == "user":
            user_content = message.get("content", "")

    if assistant_content is None:
        stats.failed_no_answer += 1
        stats.log_failure(idx, item, "no_assistant_message")
        return None

    if user_content is None:
        stats.failed_no_answer += 1
        stats.log_failure(idx, item, "no_user_message")
        return None

    # 提取答案
    extracted_answer = extract_answer(assistant_content)

    if extracted_answer is None:
        stats.failed_no_answer += 1
        stats.log_failure(idx, item, "cannot_extract_answer")
        return None

    # 标准化答案
    normalized_answer = normalize_answer(extracted_answer)

    # 检查是否为单选题答案
    if not is_valid_mcq_answer(normalized_answer):
        stats.failed_not_mcq += 1
        stats.log_failure(idx, item, "not_mcq_answer", normalized_answer)
        return None

    # 转换为框架格式
    stats.success += 1

    return {
        "id": item.get("uuid", f"item_{idx}"),
        "messages": [{"role": "user", "content": user_content}],
        "metadata": {
            "answer": normalized_answer.upper(),
            "original_uuid": item.get("uuid"),
            "source": item.get("source"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="解析 boxed 答案并转换为拒绝采样框架格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: <input>_processed.jsonl)",
    )

    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="失败日志文件路径 (默认: <output>_failed.jsonl)",
    )

    args = parser.parse_args()

    # 确定输入输出路径
    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_processed.jsonl"

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = output_path.parent / f"{output_path.stem}_failed.jsonl"

    # 读取数据
    data = read_jsonl(str(input_path))
    print(f"读取到 {len(data)} 条数据")

    # 处理数据
    stats = ProcessStats()
    processed_data = []

    for idx, item in enumerate(data):
        result = process_item(item, idx, stats)
        if result is not None:
            processed_data.append(result)

    # 保存处理后的数据
    write_jsonl(processed_data, str(output_path))

    # 打印统计信息
    stats.print_summary()

    # 保存失败日志
    if stats.failed_logs:
        write_jsonl(stats.failed_logs, str(log_path))
        print(f"失败日志已保存到: {log_path}")


if __name__ == "__main__":
    main()
