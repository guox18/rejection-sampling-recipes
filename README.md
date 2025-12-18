# Rejection Sampling Recipes

Reproducible recipes for rejection sampling in LLM data synthesis.

## Quick Start

```bash
uv sync
```

## Core Concepts

- **Stage**: A single processing step (e.g., sampling, verification, formatting). Implement `process_item(item: dict) -> dict` for automatic batching, or override `process(batch: list[dict]) -> list[dict]` for custom batch processing.
- **Recipe**: A sequence of stages that defines a complete data processing workflow.
- **Pipeline**: The execution engine that runs recipes with batching, error handling, and checkpoint/resume.

## Project Structure

```
├── src/                    # Core framework
│   ├── base.py            # Stage and BaseRecipe base classes
│   ├── pipeline.py        # Pipeline execution engine
│   └── utils/             # Data I/O utilities
├── recipes/               # Recipe implementations
│   ├── sft/              # Standard SFT recipe
├── scripts/              # Utility scripts
└── tests/                # Test files and mock data
```

## Example Usage

```python
from recipes.sft import SFTRecipe, SFTConfig
from src.pipeline import Pipeline

# Configure recipe
config = SFTConfig(
    input_path="data/train.jsonl",
    output_dir="output/",
    # ... other config options
)

# Run pipeline
recipe = SFTRecipe(config)
pipeline = Pipeline(recipe, config)
pipeline.run()
```

## License

MIT

