# Rejection Sampling Recipes

这个项目是希望复现一些 rejection sampling 的工作，提供可复现的数据合成基线。

## 背景

市面上已经有很多推理/训练(LLaMA-Factory、veRL)/评测(lm-eval-harness、OpenCompass)框架，但缺少合成数据的规范框架。虽然合成数据的门槛较低，不涉及复杂的代码逻辑，但新手可能会犯一些常见错误：
- 输出被截断（max_tokens 设置不当 / cot 解析逻辑错误）
- 采样参数不合理（temperature 设置）
- 评估有漏洞（答案提取逻辑错误）
- 忘记保存通过率，想要筛选难度时还需要重复推理
- 推理效率低（大量 rollout 预算浪费在简单题，难题 rollout 次数不足）

目前缺少一些可复现的数据合成基线（如 RLVR, rubrics/reward model）。

### 😱 踩坑案例

> **案例1**：蒸馏 DeepSeek-R1，rollout 一切正常，速度还挺快。蒸馏和训练跑了 3 天。等到评估时发现分数不对。回头检查才发现，`max_tokens` 只设了 2048，R1 的长思维链全部被截断，数据全废了。
>
> **案例2**：Rollout 完成后想筛选简单题和难题分开训练，才发现没保存每道题的通过率，只能重新跑一遍。
>
> **案例3**：用 自己写的 json 解析工具处理模型的长输出，pass rate 异常偏低。排查发现不小心把 thinking 过程中出现的 `{"answer": "B"}` 解析出来了，而不是模型最终输出的答案。


## 项目贡献

1. **提供端到端 Recipe**：数据准备 → 合成 → 训练脚本 → 评测脚本
2. **基础功能完善**：断点续推、智能早停、质量分析
3. **可复现基线**：提供完整的配置、日志、结果，方便用户参照和修改

## Scope 定义

### 聚焦方法
- **Rejection Sampling**：对同一 prompt 采样多次，选择通过验证的 response
- **Best-of-N**：对同一 prompt 采样 N 次，选择得分最高的 response

### 支持任务

| 任务类型 | 验证方式 |
|---------|---------|
| 数学推理 | Rule-based（答案提取+比对） |
| 学科选择题 | Rule-based（选项匹配） |
| 通用对话 | LLM-as-Judge / Reward Model |

### 支持的推理后端 (Sampler)

| 类型 | 说明 |
|-----|------|
| `openai-compatible-api` | 支持 OpenAI、DeepSeek、vLLM serve 等，asyncio 并发 |
| `vllm-offline` | 本地离线推理，支持数据并行 |

**截断处理**：默认丢弃被截断的 response（`drop_truncated: true`）

| 后端 | 检测方式 |
|-----|---------|
| `openai-compatible-api` | `finish_reason == "length"` |
| `vllm-offline` | 末尾无 `eos_token`（从 tokenizer_config.json 读取） |

截断的 response 直接丢弃，不保存、不计入有效 rollout。通过增大 `max_steps` 来补偿截断带来的损失。

### 支持的验证器 (Verifier)

| 类型 | 适用场景 |
|-----|---------|
| `math-rlvr` | 数学推理（答案提取 + 数值比较） |
| `mcq-rlvr` | 选择题（规则提取选项） |
| `mcq-llm-as-judge` | 选择题（非 R1 模型，选项不在 `\boxed{}` 中，需 LLM 提取） |

### 支持的数据格式化器 (Formatter)

支持同时运行多个 formatter，一次 rollout 可同时生成 SFT 和 DPO 数据。

| 类型 | 说明 | 早停条件 |
|-----|------|---------|
| `sft` | 取得分最高的 response | 有 1 个 pass（score >= pass_threshold） |
| `dpo` | 取最高分 + 最低分的 response | 有 1 个 pass + 1 个 fail（score <= fail_threshold） |

---

## 工作路径设计

采用**时间戳路径**组织实验，便于追踪、复现和 resume。

```
output/20251206_143052/
├── config.yaml                   # 实验配置（自动保存）
├── state.json                    # 运行状态（进度、断点）
├── data/
│   └── input.jsonl               # 预处理后的数据
├── rollout/                      # 推理+评测结果（分 shard 存储）
│   ├── shard_0000.jsonl
│   └── ...
├── train/                        # 训练数据
│   ├── sft.jsonl
│   └── dpo.jsonl
└── summary/                      # 分析结果
    └── stats.json
```

### 数据预处理

**流程**：
```
原始数据 → DataPreprocessor → 格式检查 → data/input.jsonl
                ↓
          transform (可选)
```

**逻辑**：
1. 检查 `work_dir/data/input.jsonl` 是否存在
2. 如果存在 → 跳过预处理（resume 场景）
3. 如果不存在 → 读取原始数据 → transform（可选）→ 格式检查 → 写入

**格式要求**：
```python
{
    "id": str,                           # 必须：唯一标识
    "messages": [                        # 必须：OpenAI messages 格式
        {"role": "user", "content": str}
    ],
    "metadata": {                        # 必须：元数据
        "answer": str,                   # 可选：标准答案（无则打印警告）
        ...
    }
}
```

**Transform 函数接口**：
```python
# transforms/gsm8k.py
def transform(item: dict) -> dict | None:
    """Transform raw item to required format. Return None to skip."""
    return {
        "id": item["id"],
        "messages": [{"role": "user", "content": item["question"]}],
        "metadata": {"answer": item["answer"]}
    }
```

**使用示例**：
```bash
# 数据已符合格式，直接复制
python run.py data.input_path=/path/to/formatted.jsonl

# 需要转换
python run.py data.input_path=/path/to/raw.jsonl \
  data.preprocess.transform=transforms/gsm8k.py:transform

# resume，已有 data/input.jsonl，跳过预处理
python run.py work_dir=output/20251206_143052/
```

### 分 Shard 存储

Rollout 结果按 shard 分片存储（默认每 10000 条一个 shard），好处：
- 支持大规模数据（10w+）而不爆内存
- 断点续推时只需重跑未完成的 shard
- 便于并行处理

---

## 配置管理

使用 **Hydra** 进行配置管理，支持 YAML 配置 + 命令行覆盖。

### 配置示例

```yaml
# Rejection Sampling Recipes Configuration

data:
  input_path: ???                # Required: path to input jsonl file
  preprocess:
    transform: null              # null = direct copy (data already formatted)
                                 # or specify: transforms/gsm8k.py:transform

work_dir: null                   # null = auto generate timestamp path (output/YYYYMMDD_HHMMSS/)

sampling:
  max_rollouts: 16               # Target: collect this many valid rollouts
  step_size: 4                   # Rollouts per step
  max_steps: 8                   # Max steps (set higher to handle truncation/timeout error)
  early_stop: true               # Enable smart early stopping based on formatter needs

sampler:
  type: openai-compatible-api    # Options: openai-compatible-api, vllm-offline
  model: DeepSeek-R1
  base_url: null                 # Only used for openai-compatible-api
  model_path: null               # Only used for vllm-offline
  temperature: 0.7
  max_tokens: 2048
  top_p: 1.0
  concurrent_requests: 50
  timeout: 300
  drop_truncated: true           # Drop truncated responses

verifier:
  type: math-rlvr                # Options: math-rlvr, mcq-rlvr, mcq-llm-as-judge
  score_type: float

formatter:
  - type: sft                    # Options: sft, dpo
    pass_threshold: 1.0          # score >= pass_threshold is considered as passed
    fail_threshold: 0.0          # score <= fail_threshold is considered as failed

shard:
  size: 10000                    # Samples per shard file
```

---

## 用户接口

```bash
# 启动新实验
python run.py data.input_path=/path/to/data.jsonl

# 覆盖配置
python run.py data.input_path=/path/to/data.jsonl \
  sampler.model=deepseek-chat \
  sampling.max_rollouts=32

# Resume
python run.py work_dir=output/20251206_143052/
```

---

## 数据格式

采用 **Messages 格式**（OpenAI 标准）。

### 输入格式

```jsonl
{"id": "001", "messages": [{"role": "user", "content": "问题..."}], "metadata": {"answer": "42"}}
```

### Rollout 输出格式

```jsonl
{
  "id": "001",
  "messages": [{"role": "user", "content": "问题..."}],
  "metadata": {"answer": "42"},
  "rollouts": [
    {"response": "...", "score": 1.0},
    {"response": "...", "score": 0.0}
  ]
}
```

### 训练数据格式

**SFT：**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO：**
```jsonl
{"prompt": [{"role": "user", "content": "..."}], "chosen": [...], "rejected": [...]}
```

---

## 核心功能

### 1. 采样流程

```
目标: 收集 max_rollouts 条有效 rollout

step 1: roll step_size 条 → 丢弃截断 → 保留有效 → 检查早停
step 2: roll step_size 条 → 丢弃截断 → 保留有效 → 检查早停
...
停止条件：有效 rollout >= max_rollouts 或 step >= max_steps 或早停满足
```

**配置示例**：
- `max_rollouts=16, step_size=4, max_steps=4`：无截断时刚好 4 轮
- `max_rollouts=16, step_size=4, max_steps=8`：允许 2 倍轮数，应对截断

### 2. 智能早停

根据 formatter 需求提前停止采样：

- SFT 早停条件：有 1 个 pass
- DPO 早停条件：有 1 个 pass + 1 个 fail
- 多 formatter：满足所有 formatter 才停止

### 2. 断点续推

- `state.json` 记录已完成的 shard 列表
- 重启时自动跳过已完成的 shard

### 3. 质量分析

统计通过率、token 分布、平均采样次数等，保存到 `summary/stats.json`。

---

## 项目结构

```
rejection-sampling-recipes/
├── configs/                     # Hydra 配置
├── src/
│   ├── sampler/                 # 采样器
│   ├── verifier/                # 验证器
│   ├── formatter/               # 格式化器
│   ├── pipeline.py              # 主流程
│   └── analysis.py              # 质量分析
├── run.py                       # 入口
├── recipes/                     # 示例 Recipe
├── pyproject.toml               # uv
└── requirements.txt             # pip
```

---

## 环境管理

支持两种方式：

**uv（推荐）：**
```bash
uv sync
uv run python run.py ...
```

**conda + pip：**
```bash
conda create -n rsr python=3.12 -y
conda activate rsr
pip install -r requirements.txt
python run.py ...
```

---

## 开发规范

### 分支策略

- `main`：稳定分支，初始开发直接 push，后续只接受 PR
- `feat/*`：功能分支，完成后 PR 到 main
- `fix/*`：修复分支

### 代码规范

- **语言**：代码注释、docstring、commit message 全部使用英文
- **Linter**：使用 ruff（lint + format）
- **类型提示**：推荐使用 type hints

### CI 配置

GitHub Actions 自动运行：
- ruff check（lint）
- ruff format --check（format）
- pytest（单元测试）

### 项目文件清单

```
rejection-sampling-recipes/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI 配置
├── .gitignore
├── .pre-commit-config.yaml      # pre-commit hooks
├── LICENSE                      # MIT
├── README.md                    # 英文，面向开源社区
├── docs/
│   └── design.md                # 中文设计文档
├── pyproject.toml               # 项目配置 + ruff 配置
├── requirements.txt
├── configs/
├── src/
├── tests/                       # 单元测试
└── run.py
```

---

## 开发流程

### 模块交互关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Pipeline                                    │
│                                                                          │
│  ┌────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │   Data     │───▶│ Sampler  │───▶│ Verifier │───▶│Formatter │        │
│  │Preprocessor│    │          │    │ (Judge)  │    │          │        │
│  └────────────┘    └──────────┘    └──────────┘    └──────────┘        │
│        │                │               │               │               │
│        ▼                ▼               ▼               ▼               │
│  data/input.jsonl  responses[]      scores[]      train/*.jsonl        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │  State Manager   │
                         │  (checkpoint)    │
                         └──────────────────┘
```

### 模块职责

| 模块 | 输入 | 输出 | 职责 |
|------|------|------|------|
| **DataPreprocessor** | raw jsonl | `data/input.jsonl` | 转换格式 + 校验 |
| **Sampler** | messages | `List[str]` | 调用 LLM 生成 response |
| **Verifier** | response + metadata | `float` | 评估 response，返回分数 |
| **Formatter** | item + rollouts | `List[dict]` | 筛选并格式化为训练数据 |
| **StateManager** | - | - | 管理断点续推状态 |

### 设计决策

**Sampler**：简单工厂函数（只有两种类型）
```python
def get_sampler(cfg):
    if cfg.type == "openai-compatible-api":
        return OpenAISampler(cfg)
    elif cfg.type == "vllm-offline":
        return VLLMSampler(cfg)
```

**Verifier**：注册器模式（类型多，用户可能扩展）
```python
@register_verifier("math-rlvr")
class MathRLVRVerifier(BaseVerifier): ...

@register_verifier("mcq-rlvr")
class MCQRLVRVerifier(BaseVerifier): ...

# 使用
verifier = get_verifier(cfg.verifier.type)
```

**Formatter**：注册器模式（用户可能扩展）
```python
@register_formatter("sft")
class SFTFormatter(BaseFormatter): ...      # 取最高分

@register_formatter("dpo")
class DPOFormatter(BaseFormatter): ...      # 取最高 + 最低

@register_formatter("top_k")
class TopKFormatter(BaseFormatter): ...     # 取前 k 个高于阈值的

# 使用
formatter = get_formatter(cfg.type)

### 开发阶段

#### Phase 1: Sampler（推理模块）

**目标**：实现稳定的推理能力

**任务**：
- [ ] 实现 `OpenAISampler`（asyncio 并发）
- [ ] 实现重试、超时、错误处理
- [ ] 支持 batch 采样（利用 `n` 参数）

**测试**：
- 基本功能：能否正常调用 API 并返回结果
- 并发：高并发下是否稳定
- 错误处理：超时、限流是否能正确重试

**产出**：
- `src/sampler/openai_sampler.py`
- `tests/test_sampler.py`
- 一批真实的推理结果（用于后续测试 Verifier）

---

#### Phase 2: Verifier（评估模块）

**目标**：实现准确的评估能力

**任务**：
- [ ] 实现 `MCQVerifier`（选项提取 + 匹配）
- [ ] 处理不同模型的输出格式差异：
  - 有/无推理过程
  - `\boxed{}`、`【答案】`、直接输出等格式
  - special tokens 差异

**测试**：
- 用 Phase 1 的真实推理结果构造测试用例
- 覆盖各种边界情况：
  - 正常格式
  - 格式变体（中英文、全角半角）
  - 无法提取答案的情况
  - 数值精度问题（0.3333 vs 1/3）

**产出**：
- `src/verifier/math_verifier.py`
- `src/verifier/mcq_verifier.py`
- `tests/test_verifier.py`（大量测试用例）
- `tests/fixtures/` 真实推理结果 fixtures

---

#### Phase 3: Formatter（格式化模块）

**目标**：实现灵活的数据筛选和格式化

**任务**：
- [ ] 实现 `SFTFormatter`（取最高分）
- [ ] 实现 `DPOFormatter`（取最高 + 最低）
- [ ] 实现早停条件检查 `is_satisfied()`

**测试**：
- 筛选逻辑是否正确
- 边界情况：全 pass、全 fail、只有一个

**产出**：
- `src/formatter/sft_formatter.py`
- `src/formatter/dpo_formatter.py`
- `tests/test_formatter.py`

---

#### Phase 4: Pipeline（整体流程）

**目标**：串联所有模块，实现完整流程

**任务**：
- [ ] 实现 `Pipeline` 主流程
- [ ] 实现 `StateManager`（断点续推）
- [ ] 实现 shard 分片存储
- [ ] 实现智能早停逻辑
- [ ] 集成 Hydra 配置

**测试**：
- 端到端测试：输入 → 输出
- 断点续推：中断后能否正确恢复
- shard 存储：大数据量是否正常
- 早停：是否按预期减少采样次数

**产出**：
- `src/pipeline.py`
- `src/state.py`
- `tests/test_pipeline.py`
- `run.py`

---

#### Phase 5: 质量分析 + 文档

**任务**：
- [ ] 实现 `Analysis` 统计模块
- [ ] 完善 README 和使用文档
- [ ] 提供示例 Recipe

### 测试策略

```
tests/
├── fixtures/                    # 测试数据
│   ├── sample_inputs.jsonl      # 输入样例
│   └── sample_outputs/          # Phase 1 产出的真实推理结果
│       ├── math_responses.jsonl
│       └── mcq_responses.jsonl
├── test_sampler.py              # Phase 1
├── test_verifier.py             # Phase 2（核心，用例最多）
├── test_formatter.py            # Phase 3
├── test_pipeline.py             # Phase 4（集成测试）
└── conftest.py                  # pytest fixtures
```

### 开发顺序建议

```
Week 1: Phase 1 (Sampler)
        ├── 实现 OpenAISampler
        └── 收集真实推理结果作为测试数据

Week 2: Phase 2 (Verifier) ← 核心，花时间最多
        ├── 实现 MathVerifier
        ├── 实现 MCQVerifier
        └── 大量测试用例

Week 3: Phase 3 + 4 (Formatter + Pipeline)
        ├── 实现 Formatter
        ├── 实现 Pipeline
        └── 断点续推测试

Week 4: Phase 5 + 收尾
        ├── 质量分析
        ├── 文档完善
        └── 示例 Recipe
```
