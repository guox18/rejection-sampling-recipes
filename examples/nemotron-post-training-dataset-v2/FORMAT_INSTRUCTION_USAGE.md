# 格式指令选项使用说明

## 概述

现在数据处理默认**不会**在问题末尾添加格式指令后缀。这是一个可选功能，可以通过命令行选项开启。

## 默认行为（无格式指令）

默认情况下，处理后的数据只包含原始问题，不添加任何后缀：

```bash
# 使用 run_pipeline.sh
bash examples/nemotron-post-training-dataset-v2/run_pipeline.sh

# 直接使用 process.py
python examples/nemotron-post-training-dataset-v2/process.py \
    --input stem.jsonl \
    --output stem_processed.jsonl
```

输出示例：
```json
{
  "id": "uuid-123",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid"
    }
  ],
  "metadata": {
    "answer": "B"
  }
}
```

## 启用格式指令后缀

如果需要在问题末尾添加格式指令（引导模型使用 `\boxed{}` 格式），可以使用 `--add-format-instruction` 选项：

```bash
# 使用 run_pipeline.sh
bash examples/nemotron-post-training-dataset-v2/run_pipeline.sh --add-format-instruction

# 直接使用 process.py
python examples/nemotron-post-training-dataset-v2/process.py \
    --input stem.jsonl \
    --output stem_processed.jsonl \
    --add-format-instruction
```

输出示例：
```json
{
  "id": "uuid-123",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    }
  ],
  "metadata": {
    "answer": "B"
  }
}
```

## 自定义格式指令

你也可以自定义格式指令内容：

```bash
python examples/nemotron-post-training-dataset-v2/process.py \
    --input stem.jsonl \
    --output stem_processed.jsonl \
    --add-format-instruction \
    --format-instruction "\n\n请逐步推理，并将最终答案放在 \\boxed{} 中。"
```

## 默认格式指令内容

当使用 `--add-format-instruction` 但不指定 `--format-instruction` 时，会使用以下默认指令：

```
Please reason step by step, and put your final answer within \boxed{}.
```

## 适用场景

- **不添加格式指令（默认）**：适用于模型已经训练过 MCQ 格式，或者你想保持原始问题不变
- **添加格式指令**：适用于引导模型输出特定格式的答案，特别是当你使用的 verifier 需要 `\boxed{}` 格式时
