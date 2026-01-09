# Rejection Sampling Recipes

Reproducible recipes for rejection sampling in LLM data synthesis.

## Install

Choose one:

```bash
# uv (recommended)
uv sync

# pip
pip install -r requirements.txt
```

Note: run commands from the repo root so `src/` and `recipes/` are importable.

## Core Concepts

- **Stage**: A single processing step (e.g., sampling, verification, formatting). Implement `process_item(item: dict) -> dict` for automatic batching, or override `process(batch: list[dict]) -> list[dict]` for custom batch processing. Stages may filter/expand/reorder items; the pipeline restores framework fields based on `_resume_id`.
- **Recipe**: A sequence of stages that defines a complete data processing workflow.
- **Pipeline**: The execution engine that runs recipes with batching, error handling, and checkpoint/resume.

## Project Structure

```
├── src/                          # Core framework
│   ├── base.py                  # Stage and BaseRecipe base classes
│   ├── pipeline.py              # Pipeline execution engine
│   └── utils/                   # Data I/O utilities
├── recipes/                     # Recipe implementations
│   └── vl_cot_sft_plus_parse/   # Current SFT recipe
├── scripts/                     # Utility scripts
└── tests/                       # Test files and mock data
```

## Example Usage

### CLI (recommended)

```bash
# If using OpenAI-compatible APIs
export OPENAI_API_KEY=your_key

# 1) Optional: add absolute image paths for multimodal data
python scripts/preprocess_images.py \
  --input tests/mock/text-pic.jsonl \
  --image-base-path /abs/path/to/images \
  --abs-image-path-field abs_path

# 2) Run recipe
python recipes/vl_cot_sft_plus_parse/entrypoint/run.py \
  --input tests/mock/text-pic.jsonl \
  --config recipes/vl_cot_sft_plus_parse/config.yaml
```

### Python API

```python
from recipes.vl_cot_sft_plus_parse import SFTRecipe, SFTConfig
from src.pipeline import Pipeline

config = SFTConfig()
recipe = SFTRecipe(config)
pipeline = Pipeline(recipe, batch_size=4, concurrency=4)
pipeline.run("data/train.jsonl", "output/train_sft.jsonl")
```

## License

MIT
