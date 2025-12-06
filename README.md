<div align="center">

# 🧪 Rejection Sampling Recipes

**Reproducible recipes for rejection sampling in synthetic data generation**

[![CI](https://github.com/guox18/rejection-sampling-recipes/actions/workflows/ci.yml/badge.svg)](https://github.com/guox18/rejection-sampling-recipes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/guox18/rejection-sampling-recipes/branch/main/graph/badge.svg)](https://codecov.io/gh/guox18/rejection-sampling-recipes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Why This Project?](#-why-this-project) • [Features](#-whats-included) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

</div>

---

## 🤔 Why This Project?

The community already has great tools for inference ([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)), training ([LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [veRL](https://github.com/volcengine/verl)), and evaluation ([lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness), [OpenCompass](https://github.com/open-compass/opencompass)). But when it comes to **synthetic data generation**, most of us end up writing one-off scripts—and learning the same lessons the hard way. What's missing are **reproducible recipes** that provide end-to-end solutions with concrete configs, parameters, and workflows you can follow.

This project started after a few embarrassing moments:

> 😱 Distilled long-reasoning traces, trained for days, everything looked fine. Only when evaluation scores came back terrible did we dig into the data—`max_tokens=2048` had silently truncated all the chain-of-thought.

> 😱 Wanted to split easy vs hard problems for curriculum learning. Realized we never saved pass rates. Had to re-run everything.

> 😱 Pass rate looked suspiciously low. Turned out our answer extractor was grabbing `{"answer": "B"}` from the model's *thinking process*, not its final answer. Thousands of correct responses marked wrong.

**RSR is our attempt to avoid repeating these mistakes.** It handles the trivial-but-important details: truncation detection, checkpoint/resume, pass rate tracking, and answer extraction that actually works across different model output formats.

---

## ✨ What's Included

- **End-to-end workflow** — From raw data to training-ready JSONL, with configs you can actually reproduce
- **Smart early stopping** — Stop sampling once you have what you need (1 pass for SFT, 1 pass + 1 fail for DPO)
- **Checkpoint & resume** — Shard-based storage that handles 100k+ samples; resume from any interruption
- **Quality stats** — Pass rates, token distributions, and sampling efficiency, saved automatically
- **Truncation handling** — Detects and discards truncated responses so you don't train on garbage

## 📋 Supported Tasks

| Task | Verifier | Status |
|------|----------|--------|
| Multiple Choice (Single) | Rule-based | ✅ |
| Multiple Choice (Multi) | LLM-as-Judge | ✅ |
| Math Reasoning | Rule-based | ✅ |
| General Chat | Reward Model | 🚧 TODO |

## 📦 Installation

**Using uv (recommended):**

```bash
git clone https://github.com/guox18/rejection-sampling-recipes.git
cd rejection-sampling-recipes
uv sync
```

**Using pip:**

```bash
git clone https://github.com/guox18/rejection-sampling-recipes.git
cd rejection-sampling-recipes
pip install -r requirements.txt
```

## 🚀 Quick Start

**Using vLLM offline inference:**

```bash
uv run python run.py \
  data.input_path=data/your_data.jsonl \
  data.preprocess.transform=transforms/your_transform.py:transform \
  sampler.type=vllm-offline \
  sampler.model_path=/path/to/model \
  sampler.tensor_parallel_size=2 \
  sampler.data_parallel_size=2 \
  sampler.max_tokens=4096 \
  sampling.max_rollouts=4 \
  sampling.step_size=2 \
  verifier.type=mcq-rlvr
```

**Using OpenAI-compatible API:**

```bash
uv run python run.py \
  data.input_path=data/your_data.jsonl \
  data.preprocess.transform=transforms/your_transform.py:transform \
  sampler.type=openai-compatible-api \
  sampler.base_url=http://localhost:8000/v1 \
  sampler.model=your-model \
  sampler.max_tokens=4096 \
  sampler.concurrent_requests=128 \
  sampling.max_rollouts=4 \
  sampling.step_size=2 \
  verifier.type=mcq-rlvr
```

**Resume from checkpoint:**

```bash
# Auto-loads config from work_dir/config.yaml
uv run python run.py work_dir=output/20251206_143052/

# Resume with parameter override
uv run python run.py work_dir=output/20251206_143052/ sampler.concurrent_requests=256
```

## 📖 Documentation

- [Design Document](docs/design.md) | [中文版](docs/design_cn.md)

### Output Structure

```
output/20251206_143052/
├── config.yaml          # Experiment config (auto-saved)
├── state.json           # Progress & checkpoint state
├── data/
│   └── input.jsonl      # Preprocessed input data
├── rollout/             # Rollout results (sharded)
│   ├── shard_0000.jsonl
│   └── ...
├── train/               # Training-ready data
│   ├── sft.jsonl
│   └── dpo.jsonl
└── summary/
    └── stats.json       # Quality statistics
```

### Configuration Example

```yaml
data:
  input_path: /path/to/data.jsonl
  preprocess:
    transform: null

work_dir: null                   # null = auto generate: output/YYYYMMDD_HHMMSS/
verbose: false                   # Enable verbose logging

sampling:
  max_rollouts: 16               # Target rollouts per sample
  step_size: 4                   # Rollouts per step
  max_steps: 8                   # Max steps (handles truncation retries)
  early_stop: true               # Stop when formatter requirements met

sampler:
  type: openai-compatible-api    # Options: openai-compatible-api, vllm-offline
  model: DeepSeek-R1
  base_url: null                 # API base URL (defaults to OpenAI)
  api_key: null                  # Defaults to OPENAI_API_KEY env var
  model_path: null               # For vllm-offline only
  tensor_parallel_size: 1        # vllm-offline: GPUs per worker
  data_parallel_size: null       # vllm-offline: number of workers (auto-calculated if unset : total_gpus / tensor_parallel_size)
  temperature: 0.7
  max_tokens: 2048               # Set high enough for long CoT!
  top_p: 1.0
  concurrent_requests: 128        # Concurrent batch size (API mode)
  timeout: 300                   # Request timeout in seconds
  drop_truncated: true           # Auto-discard truncated responses
  extra_params: {}               # Extra params (e.g., reasoning_effort: high)

verifier:
  type: mcq-rlvr                 # Options: mcq-rlvr, mcq-llm-as-judge, math-rlvr
  # For mcq-llm-as-judge:
  model: null                    # Judge model (e.g., gpt-4o-mini)
  base_url: null                 # Judge API base URL
  api_key: null                  # Judge API key

formatter:
  - type: sft                    # Options: sft, dpo, multi_sft
    pass_threshold: 1.0          # score >= threshold = passed
    fail_threshold: 0.0          # score <= threshold = failed
    # For multi_sft: num_responses: 32

shard:
  size: 10000                    # Samples per shard file
```

### Data Format

<details>
<summary>Click to expand</summary>

**Input (after preprocessing):**
```jsonl
{
  "id": "5ed129f9-8548-4cbd-abd4-7ff362f7facc",
  "messages": [
    {"role": "user", "content": "Which of the following best explains...?\n\nA: Option A\nB: Option B\nC: Option C"}
  ],
  "metadata": {"answer": "B", "category": "stem"}
}
```

**Output (SFT):**
```jsonl
{
  "messages": [
    {"role": "user", "content": "Which of the following best explains...?\n\nA: Option A\nB: Option B\nC: Option C"},
    {"role": "assistant", "content": "**B – Option B**\n\nThis is the correct answer because..."}
  ]
}
```

**Rollout (intermediate):**
```jsonl
{
  "id": "5ed129f9-8548-4cbd-abd4-7ff362f7facc",
  "messages": [...],
  "metadata": {"answer": "B"},
  "rollouts": [
    {"response": "**B – Option B**\n\n...", "score": 1.0},
    {"response": "**A – Option A**\n\n...", "score": 0.0}
  ]
}
```

</details>

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --all-extras

# Setup pre-commit
uv run pre-commit install

# Run linter
uv run ruff check .

# Run tests
uv run pytest
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training recipes
- [Hydra](https://github.com/facebookresearch/hydra) for configuration management

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐!**

</div>
