# Configuration Reference

RSR uses [Hydra](https://hydra.cc/) for configuration. All parameters can be set via YAML files or command-line overrides.

## Quick Reference

```bash
# Use a config file
uv run python run.py --config-name=sft data.input_path=data.jsonl sampler.model=gpt-oss

# Override any parameter
uv run python run.py sampler.temperature=0.8 sampling.max_rollouts=8
```

## Full Configuration

### Data

```yaml
data:
  input_path: ???                 # Path to input JSONL file (required)
  preprocess:
    transform: null               # Transform script path (null = no transform)
```

### Working Directory

```yaml
work_dir: null                    # Output directory
                                  # null = auto-generate: output/YYYYMMDD_HHMMSS/
                                  # Specify existing path to resume from checkpoint
verbose: false                    # Enable verbose logging
```

### Sampling

```yaml
sampling:
  max_rollouts: 16                # Target valid rollouts per prompt
  step_size: 4                    # Responses sampled per step
  max_steps: 5                    # Maximum sampling steps
  early_stop: true                # Stop when formatter is satisfied
```

**Key relationships:**

- `max_steps × step_size` = total sampling budget
- Should be ≥ `max_rollouts` to handle truncation

**Example configurations:**

| Scenario | max_rollouts | step_size | max_steps | Budget |
|----------|--------------|-----------|-----------|--------|
| SFT (find 1 correct) | 16 | 4 | 4 | 16 |
| DPO (need pass+fail) | 32 | 8 | 4 | 32 |
| Multi-SFT (32 unique) | 64 | 8 | 8 | 64 |

### Sampler

```yaml
sampler:
  # Common parameters
  type: openai-compatible-api     # openai-compatible-api | vllm-offline
  temperature: 0.7                # Sampling temperature
  max_tokens: 2048                # Max tokens per response
  top_p: 1.0                      # Nucleus sampling threshold
  drop_truncated: true            # Discard truncated responses
  extra_params: {}                # Extra API parameters

  # API mode parameters
  model: ???                      # Model name (required for API)
  base_url: null                  # API endpoint (null = OpenAI default)
  api_key: null                   # API key (null = env OPENAI_API_KEY)
  concurrent_requests: 50         # Concurrent API requests
  timeout: 300                    # Request timeout in seconds

  # vLLM offline parameters
  model_path: null                # Local model path
  tensor_parallel_size: 1         # GPUs per worker
  data_parallel_size: null        # Number of workers (auto if null)
  gpu_memory_utilization: 0.9     # GPU memory ratio
```

### Verifier

```yaml
verifier:
  type: mcq-rlvr                  # mcq-rlvr | math-rlvr | mcq-llm-as-judge

  # For mcq-llm-as-judge only:
  model: null                     # Judge model (e.g., gpt-4o-mini)
  base_url: null                  # Judge API endpoint
  api_key: null                   # Judge API key
  temperature: 0.0                # Judge temperature
  max_tokens: 10                  # Judge max tokens
```

**Verifier types:**

| Type | Use Case | Speed |
|------|----------|-------|
| `mcq-rlvr` | MCQ with `\boxed{}` format | Very Fast |
| `mcq-llm-as-judge` | MCQ without structured format | Fast |

### Formatter

```yaml
formatter:
  - type: sft                     # sft | dpo | multi_sft
    pass_threshold: 1.0           # Score >= this is passed
    fail_threshold: 0.0           # Score <= this is failed
    # num_responses: 32           # For multi_sft only
```

Multiple formatters can run in parallel:

```yaml
formatter:
  - type: sft
    pass_threshold: 1.0
  - type: dpo
    pass_threshold: 1.0
    fail_threshold: 0.0
```

> **Note:** When using multiple formatters with `early_stop: true`, sampling stops only when **all formatters are satisfied**. This means all formatters will output the same set of prompts (though in different formats). For example, with SFT + DPO, only prompts that have both passed and failed responses will be included in both outputs.

### Shard

```yaml
shard:
  size: 10000                     # Items per shard file
```

Sharding enables:
- Memory-efficient processing of large datasets
- Checkpoint/resume at shard boundaries

## Pre-built Configs

RSR provides ready-to-use configs in `configs/`:

| Config | Description |
|--------|-------------|
| `config.yaml` | Default with all parameters |
| `sft.yaml` | SFT recipe (1 best correct) |
| `dpo.yaml` | DPO recipe (chosen + rejected) |
| `multi_sft.yaml` | Multi-SFT (N unique correct) |

Use with:

```bash
uv run python run.py --config-name=sft data.input_path=data.jsonl sampler.model=xxx
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Default API key for OpenAI-compatible endpoints |
| `CUDA_VISIBLE_DEVICES` | GPU selection for vLLM offline |
