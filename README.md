# Rejection Sampling Recipes

Reproducible recipes for rejection sampling in LLM/VLM data synthesis.

## Why this repo

- **Easy to run**: pick a recipe and run it; no extra scaffolding.
- **Ready-to-use recipes**: text + multimodal flows with answer parsing, a solid judge
  prompt, and safe image-resize fallbacks.
- **Scales when data grows**: Ray Data based pipeline, which gives streaming-style processing,
  batching, concurrency, and checkpoint/resume out of the box.

## Install

```bash
# Option A: uv (recommended; the multimodal recipe uses this env)
uv sync

# Option B: conda + pip
conda create -n rsr python=3.12
conda activate rsr
pip install -r requirements.txt
```
## Core Concepts

- **Stage**: A single processing step (e.g., sampling, verification, formatting).
- **Recipe**: A sequence of stages that defines a complete data processing workflow.
- **Pipeline**: The execution engine that runs recipes with batching, error handling,
  and checkpoint/resume.

## Project Structure

```
├── src/                          # Core framework
│   ├── base.py                  # Stage and BaseRecipe base classes
│   ├── pipeline.py              # Pipeline execution engine
│   └── utils/                   # Data I/O utilities
├── recipes/                     # Recipe implementations
│   ├── text_sft_simple/         # Text-only recipe
│   └── vl_cot_sft_plus_parse/   # Text + image recipe (with answer parsing)
├── scripts/                     # Utility scripts
└── tests/                       # Test files and mock data
```

## Recipes (Quick Start)

All recipes read JSONL. Each item should have `messages` in OpenAI format.

### Minimal JSONL examples

Text-only (You can put answer inside assistant response):
```json
{"id": 1, "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]}
```

Multimodal (with images):
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

### 1) `text_sft_simple` (text-only, simple)

```bash
bash recipes/text_sft_simple/entrypoint/run.sh
```

### 2) `vl_cot_sft_plus_parse` (text + image + answer parsing)


For example:
```bash
# 1) Add absolute image paths ("abs_path")
python scripts/preprocess_images.py \
  --input tests/mock/text-pic.jsonl \
  --image-base-path /abs/path/to/images \
  --abs-image-path-field abs_path

# 2) Run the recipe
bash recipes/vl_cot_sft_plus_parse/entrypoint/run-30b/run-30b.sh
```

## Launch Serve

See `scripts/launch_serve/README.md` for model service setup and launch steps.

## Logging
- Default log files live in `logs/`: `pipeline.log` for the driver and
  `pipeline_worker_<pid>.log` for Ray workers. If `logs/` is not writable, they
  fall back to `/tmp/rejection-sampling-recipes-logs/`.
- Environment overrides: `LOG_DIR` (log directory), `LOG_MAX_BYTES` (default 10MB),
  `LOG_BACKUP_COUNT` (default 5), `LOG_FILE_LEVEL` (default DEBUG),
  `LOG_CONSOLE_LEVEL` (default INFO).
- The VL recipe reuses the global logging pipeline. Set `LOG_DIR` to group
  driver/worker logs for this recipe under a specific directory (e.g.,
  `/tmp/logs/vl_cot_sft_plus_parse`).

## License

MIT
