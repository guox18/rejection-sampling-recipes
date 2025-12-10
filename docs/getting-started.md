# Getting Started

Get RSR running quickly.

## Installation

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

## Prepare Your Data

RSR expects input in JSONL format with `messages` and `metadata`:

```jsonl
{"id": "q001", "messages": [{"role": "user", "content": "What is 2+2?\n\nA: 3\nB: 4\nC: 5"}], "metadata": {"answer": "B"}}
{"id": "q002", "messages": [{"role": "user", "content": "Capital of France?\n\nA: London\nB: Berlin\nC: Paris"}], "metadata": {"answer": "C"}}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier for each prompt |
| `messages` | Yes | OpenAI-style message list |
| `metadata` | Optional | Contains ground truth (e.g., `answer`) |

## Run Your First Job

### Option 1: OpenAI-Compatible API

```bash
uv run python run.py \
  data.input_path=data/your_data.jsonl \
  sampler.type=openai-compatible-api \
  sampler.base_url=http://localhost:8000/v1 \
  sampler.model=your-model \
  sampler.max_tokens=4096 \
  sampling.max_rollouts=4 \
  sampling.step_size=2 \
  verifier.type=mcq-rlvr
```

### Option 2: vLLM Offline

```bash
uv run python run.py \
  data.input_path=data/your_data.jsonl \
  sampler.type=vllm-offline \
  sampler.model_path=/path/to/model \
  sampler.tensor_parallel_size=8 \
  sampler.max_tokens=16384 \
  sampling.max_rollouts=4 \
  sampling.step_size=1 \
  sampling.max_steps=6 \
  verifier.type=mcq-rlvr
```

## Check Results

After the run completes:

```bash
# View generated training data
head output/*/train/sft.jsonl

# Check statistics
cat output/*/summary/stats.json
```

## Resume from Checkpoint

If a job is interrupted, resume it:

```bash
# Auto-loads config from work_dir/config.yaml
uv run python run.py work_dir=output/20251206_143052/

# Resume with parameter override
uv run python run.py work_dir=output/20251206_143052/ sampler.concurrent_requests=256
```

## Next Steps

- [Configuration](configuration.md) — Understand all parameters
- [Recipes](recipes.md) — SFT, DPO, Multi-SFT workflows
- [Early Stop](early-stop.md) — Optimize sampling efficiency
