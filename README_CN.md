# Rejection Sampling Recipes

用于 LLM/VLM 数据合成的拒绝采样可复现 recipes。

## 为什么用这个仓库

- **好用易上手**：选一个 recipe 直接跑，不需要额外脚手架。
- **开箱即用**：内置文本/多模态流程，包含答案解析、成熟 judge prompt、图片 resize 兜底等实用能力。
- **可规模化**：基于 Ray Data 的流水式处理，支持流式处理、批处理、并发和断点续传。

## 安装

```bash
# 方式 A：uv（推荐；示例脚本默认使用 .venv）
uv sync

# 方式 B：conda + pip
conda create -n rsr python=3.12
conda activate rsr
pip install -r requirements.txt
```

## 核心概念

- **Stage**：单个处理步骤（如采样、验证、格式化）。
- **Recipe**：由多个 Stage 组成的完整数据处理流程。
- **Pipeline**：执行引擎，负责批处理、错误处理与断点续传。

## 项目结构

```
├── src/                          # 核心框架
│   ├── base.py                  # Stage 与 BaseRecipe 基类
│   ├── pipeline.py              # Pipeline 执行引擎
│   └── utils/                   # 数据读写与工具
├── recipes/                     # Recipe 实现
│   ├── text_sft_simple/         # 纯文本 recipe
│   ├── vl_cot_sft_plus_parse/   # 文本+图像 recipe（含答案解析）
│   └── cpu_task_demo/           # CPU 密集型 demo recipe
├── scripts/                     # 工具脚本
└── tests/                       # 测试与 mock 数据
```

## Recipes（快速上手）

所有 recipe 都读取 JSONL，每条数据需包含 OpenAI 格式的 `messages`。

### 最小 JSONL 示例

纯文本（答案可放在 assistant）：
```json
{"id": 1, "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]}
```

多模态（含图片）：
```json
{
  "id": 1,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "images/foo.jpg", "image_wh": [640, 480]}},
        {"type": "text", "text": "What is shown?"}
      ]
    },
    {"role": "assistant", "content": "A short answer."}
  ],
  "abs_path": "/abs/path/to/image/base"
}
```

### 1) `text_sft_simple`（纯文本）

```bash
bash recipes/text_sft_simple/entrypoint/run.sh
```

### 2) `vl_cot_sft_plus_parse`（文本 + 图像 + 答案解析）

```bash
# 1) 写入图片绝对路径（abs_path）
python scripts/preprocess_images.py \
  --input tests/mock/text-pic.jsonl \
  --image-base-path /abs/path/to/images \
  --abs-image-path-field abs_path

# 2) 运行 recipe
bash recipes/vl_cot_sft_plus_parse/entrypoint/run-30b/run-30b.sh
```

### 3) `cpu_task_demo`（CPU 密集型 Demo）

用于验证 CPU 密集型任务在 Ray 集群中的分布式执行（示例为质数统计）。

```bash
bash recipes/cpu_task_demo/entrypoint/run.sh
```

如果你要做多节点 vs 单机的 CPU 扩展性对比，建议直接跑重负载 benchmark：

```bash
# 若不存在，会自动生成 tests/mock/cpu_task_heavy_3000.jsonl。
# 脚本会自动识别集群 CPU，并将并发上限设为 80。
bash recipes/cpu_task_demo/entrypoint/benchmark.sh
```

对比方式：先在“有额外 worker”状态跑一次，再停止 worker 后跑一次，比较 `Wall time (s)`。

## Launch Serve

模型服务的启动方式请看 `scripts/launch_serve/README.md`。
注意：默认脚本里包含 `ray stop`。如需连续运行多个脚本，建议使用不同机器，或移除 `ray stop`。

## 日志

- 默认日志目录为 `logs/`：`pipeline.log`（driver）和 `pipeline_worker_<pid>.log`（worker）。
  如果 `logs/` 不可写，会自动回退到 `/tmp/rejection-sampling-recipes-logs/`。
- 环境变量覆盖：`LOG_DIR`（目录）、`LOG_MAX_BYTES`（默认 10MB）、`LOG_BACKUP_COUNT`（默认 5）、
  `LOG_FILE_LEVEL`（默认 DEBUG）、`LOG_CONSOLE_LEVEL`（默认 INFO）。
- VL recipe 复用全局日志机制；可用 `LOG_DIR` 将 driver/worker 日志分组到指定目录（如
  `/tmp/logs/vl_cot_sft_plus_parse`）。

## License

MIT
