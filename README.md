# Rejection Sampling Recipes

[![Ruff](https://github.com/guox18/rejection-sampling-recipes/actions/workflows/ruff.yml/badge.svg)](https://github.com/guox18/rejection-sampling-recipes/actions/workflows/ruff.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Reproducible recipes for rejection sampling in LLM/VLM data synthesis.

## Why This Repo

- **Easy to run**: Pick a recipe and run it; no extra scaffolding.
- **Ready-to-use recipes**: Text + multimodal flows with answer parsing, a solid judge
  prompt, and safe image-resize fallbacks.
- **Scales when data grows**: Ray Data based pipeline, which gives streaming-style processing,
  batching, concurrency, and checkpoint/resume out of the box.

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Stage** | A single processing step (e.g., sampling, verification, formatting) |
| **Recipe** | A sequence of stages that defines a complete data processing workflow |
| **Pipeline** | The execution engine that runs recipes with batching, error handling, and checkpoint/resume |

## Project Structure

```
rejection-sampling-recipes/
├── src/                          # Core framework
│   ├── base.py                   # Stage and BaseRecipe base classes
│   ├── pipeline.py               # Pipeline execution engine
│   └── utils/                    # Data I/O utilities
├── recipes/                      # Recipe implementations
│   ├── text_sft_simple/          # Text-only recipe
│   ├── vl_cot_sft_plus_parse/    # Text + image recipe (with answer parsing)
│   └── cpu_task_demo/            # CPU-intensive demo recipe
├── scripts/                      # Utility scripts
└── tests/                        # Test files and mock data
```

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) [uv](https://github.com/astral-sh/uv) for faster dependency management

### Option A: Using uv (Recommended)

```bash
uv sync
```

### Option B: Using pip

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

All recipes read JSONL. Each item should have `messages` in OpenAI format.

### Input Format Examples

**Text-only** (answer inside assistant response):

```json
{"id": 1, "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]}
```

**Multimodal** (with images):

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

### Available Recipes

#### 1. `text_sft_simple` (Text-only)

A simple text-only recipe for basic SFT data processing.

```bash
bash recipes/text_sft_simple/entrypoint/run.sh
```

#### 2. `vl_cot_sft_plus_parse` (Text + Image + Answer Parsing)

A multimodal recipe with Chain-of-Thought and answer parsing.

```bash
# Step 1: Add absolute image paths ("abs_path")
python scripts/preprocess_images.py \
  --input tests/mock/text-pic.jsonl \
  --image-base-path /abs/path/to/images \
  --abs-image-path-field abs_path

# Step 2: Run the recipe
bash recipes/vl_cot_sft_plus_parse/entrypoint/run-30b/run-30b.sh
```

#### 3. `cpu_task_demo` (CPU-intensive Demo)

A minimal recipe for CPU-heavy processing (prime counting) to verify distributed CPU execution.

```bash
bash recipes/cpu_task_demo/entrypoint/run.sh
```

For multi-node CPU scaling tests, use the heavy benchmark script:

```bash
# Auto-generates tests/mock/cpu_task_heavy_3000.jsonl when missing.
# The script auto-detects cluster CPU and chooses concurrency up to 80.
bash recipes/cpu_task_demo/entrypoint/benchmark.sh
```

To compare distributed vs single-node performance, run it once with extra Ray workers,
then stop extra workers and run it again. Compare `Wall time (s)`.

## Model Service Setup

See [`scripts/launch_serve/README.md`](scripts/launch_serve/README.md) for model service setup and launch steps.

> **Note**: The default launch scripts include `ray stop`. If you need to run multiple
> scripts, use separate machines or remove `ray stop`.

## Logging

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LOG_DIR` | `logs/` | Log directory (falls back to `/tmp/rejection-sampling-recipes-logs/`) |
| `LOG_MAX_BYTES` | `10MB` | Maximum log file size |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files |
| `LOG_FILE_LEVEL` | `DEBUG` | File logging level |
| `LOG_CONSOLE_LEVEL` | `INFO` | Console logging level |

Log files:
- `pipeline.log` - Driver logs
- `pipeline_worker_<pid>.log` - Ray worker logs

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

```bash
# Install dev dependencies
uv sync --extra dev

# Run linting
uvx ruff check .

# Run formatting
uvx ruff format .

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please consider citing:

```bibtex
@software{rejection_sampling_recipes,
  title = {Rejection Sampling Recipes},
  author = {guox18},
  year = {2024},
  url = {https://github.com/guox18/rejection-sampling-recipes}
}
```
