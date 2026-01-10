# Rejection Sampling Recipes

Reproducible recipes for rejection sampling in LLM data synthesis.

## Install

Python >= 3.10. Choose one (keep it simple):

```bash
# Option A: uv (recommended; the multimodal recipe uses this env)
uv sync

# run scripts inside the uv env
uv run python -m recipes.text_sft_simple.entrypoint.run --help

# Option B: conda + pip (traditional)
conda create -n rsr python=3.12
conda activate rsr
pip install -r requirements.txt
```

Note: run commands from the repo root so `src/` and `recipes/` are importable.

## Core Concepts

- **Stage**: A single processing step (e.g., sampling, verification, formatting).
  Implement `process_item(item: dict) -> dict` for automatic batching, or override
  `process(batch: list[dict]) -> list[dict]` for custom batch processing. Stages may
  filter/expand/reorder items; the pipeline restores framework fields based on
  `_resume_id`.
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
If you installed with uv, just prefix commands with `uv run`.

### 1) `text_sft_simple` (text-only, simple)

Input requirements (minimal):
- `messages` with at least one user message
- Gold answer is taken from the last assistant message if present, otherwise from
  `metadata.short_answer` / `metadata.answer`

```bash
bash recipes/text_sft_simple/entrypoint/run.sh
```

### 2) `vl_cot_sft_plus_parse` (text + image + answer parsing)

Input requirements (minimal):
- `messages` can include `image_url` parts (relative path in `image_url.url`)
- Provide an absolute image base path via `abs_path` (or run the preprocessor)
- If `metadata.short_answer` is missing, the recipe parses `metadata.answer`
  (extracted from the assistant message) to create it for judging

For example:
```bash
# 1) Add absolute image paths (required if you have images)
python scripts/preprocess_images.py \
  --input tests/mock/text-pic.jsonl \
  --image-base-path /abs/path/to/images \
  --abs-image-path-field abs_path

# 2) Run the recipe
python recipes/vl_cot_sft_plus_parse/entrypoint/run.py \
  --input tests/mock/text-pic.jsonl \
  --config recipes/vl_cot_sft_plus_parse/config.yaml
```

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
