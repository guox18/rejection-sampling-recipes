<div align="center">

# 🧪 Rejection Sampling Recipes

**Reproducible recipes for rejection sampling in LLM data synthesis**

[![CI](https://github.com/yourname/rejection-sampling-recipes/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/rejection-sampling-recipes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourname/rejection-sampling-recipes/branch/main/graph/badge.svg)](https://codecov.io/gh/yourname/rejection-sampling-recipes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

<!-- [![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-RSR--Examples-orange)](https://huggingface.co/datasets/yourname/rsr-examples)
[![📊 WandB](https://img.shields.io/badge/📊%20WandB-Experiments-blue)](https://wandb.ai/yourname/rejection-sampling-recipes) -->

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## ✨ Features

- 🚀 **End-to-end Recipes** — Data preparation → Synthesis → Training scripts → Evaluation
- ⚡ **Smart Early Stopping** — Batch sampling with formatter-aware early stop
- 💾 **Checkpoint & Resume** — Shard-based storage for large-scale data (100k+)
- 📊 **Quality Analysis** — Pass rate, token distribution, sampling efficiency stats
- 🔧 **Flexible Config** — Hydra-based configuration with CLI overrides

## 📋 Supported Tasks

| Task | Verifier | Status |
|------|----------|--------|
| Math Reasoning | Rule-based | ✅ |
| Multiple Choice | Rule-based / LLM-as-Judge | ✅ |
| General Chat | Reward Model | 🚧 |

## 📦 Installation

**Using uv (recommended):**

```bash
git clone https://github.com/yourname/rejection-sampling-recipes.git
cd rejection-sampling-recipes
uv sync
```

**Using pip:**

```bash
git clone https://github.com/yourname/rejection-sampling-recipes.git
cd rejection-sampling-recipes
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# Start a new experiment
python run.py data.input_path=/path/to/data.jsonl

# Override config
python run.py data.input_path=/path/to/data.jsonl \
  sampler.model=deepseek-chat \
  sampling.max_rollouts=32

# Resume from checkpoint
python run.py work_dir=output/20251206_143052/
```

## 📖 Documentation

- [Design Document](docs/design.md) (中文)

### Configuration Example

```yaml
data:
  input_path: /path/to/data.jsonl

sampling:
  max_rollouts: 16
  step_size: 4
  early_stop: true

sampler:
  type: openai-compatible-api    # or vllm-offline
  model: DeepSeek-R1
  temperature: 0.7

verifier:
  type: math-rlvr                # math-rlvr, mcq-rlvr, mcq-llm-as-judge

formatter:
  - type: sft
    pass_threshold: 1.0
```

### Data Format

<details>
<summary>Click to expand</summary>

**Input:**
```jsonl
{"id": "001", "messages": [{"role": "user", "content": "..."}], "metadata": {"answer": "42"}}
```

**Output (SFT):**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Output (DPO):**
```jsonl
{"prompt": [...], "chosen": [...], "rejected": [...]}
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

1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

## 👥 Contributors

<a href="https://github.com/yourname/rejection-sampling-recipes/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yourname/rejection-sampling-recipes" />
</a>

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
