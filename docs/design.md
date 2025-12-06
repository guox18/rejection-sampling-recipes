# Rejection Sampling Recipes

This project aims to reproduce rejection sampling workflows and provide reproducible baselines for data synthesis.

## Background

There are many existing frameworks for inference/training (LLaMA-Factory, veRL) and evaluation (lm-eval-harness, OpenCompass), but there's a lack of standardized frameworks for synthetic data generation. Although synthetic data has a lower barrier to entry without complex code logic, beginners may make common mistakes:
- Output truncation (improper `max_tokens` setting / incorrect CoT parsing logic)
- Unreasonable sampling parameters (temperature settings)
- Evaluation loopholes (incorrect answer extraction logic)
- Forgetting to save pass rates, requiring re-inference when filtering by difficulty
- Low inference efficiency (lots of rollout budget wasted on easy problems, insufficient rollouts for hard problems)

Currently, there's a lack of reproducible data synthesis baselines (like RLVR, rubrics/reward model).

### 😱 Real-World Pitfalls

> **Case 1**: Distilling DeepSeek-R1, rollout seemed normal and was reasonably fast. Distillation and training ran for 3 days. When evaluating, the scores were off. Checking revealed `max_tokens` was only set to 2048, truncating all of R1's long chain-of-thought outputs. All data was wasted.
>
> **Case 2**: After completing rollouts, wanted to separate easy and hard problems for different training. Realized pass rates weren't saved, had to rerun everything.
>
> **Case 3**: Used a custom JSON parsing tool for long model outputs, pass rate was abnormally low. Investigation found it was extracting `{"answer": "B"}` from the thinking process instead of the model's final answer.


## Project Contributions

1. **End-to-End Recipes**: Data preparation → Synthesis → Training scripts → Evaluation scripts
2. **Complete Basic Features**: Checkpoint resume, smart early stopping, quality analysis
3. **Reproducible Baselines**: Complete configurations, logs, and results for reference and modification

## Scope Definition

### Focus Methods
- **Rejection Sampling**: Sample multiple times for the same prompt, select responses that pass verification
- **Best-of-N**: Sample N times for the same prompt, select the highest-scoring response

### Supported Tasks

| Task Type | Verification Method |
|-----------|-------------------|
| Math Reasoning | Rule-based (answer extraction + comparison) |
| Subject MCQ | Rule-based (option matching) |
| General Chat | LLM-as-Judge / Reward Model |

### Supported Inference Backends (Sampler)

| Type | Description |
|------|-------------|
| `openai-compatible-api` | Supports OpenAI, DeepSeek, vLLM serve, SGLang, etc., with asyncio concurrency |
| `vllm-offline` | Local offline inference, supports Ray data parallelism |

**Parameter Extension**: Different models/services may have special parameters (e.g., `reasoning_effort` for OSS models), passed through `extra_params`:

```yaml
sampler:
  type: openai-compatible-api
  model: qwen
  base_url: http://localhost:30120/v1
  extra_params:
    reasoning_effort: high        # OSS model specific parameter
```

**Truncation Handling**: Truncated responses are dropped by default (`drop_truncated: true`)

| Backend | Detection Method |
|---------|-----------------|
| `openai-compatible-api` | `finish_reason == "length"` |
| `vllm-offline` | No `eos_token` at the end (read from tokenizer_config.json) |

Truncated responses are directly discarded, not saved, and don't count toward valid rollouts. Increase `max_steps` to compensate for truncation losses.

### Supported Verifiers

| Type | Use Case |
|------|----------|
| `math-rlvr` | Math reasoning (answer extraction + numerical comparison) |
| `mcq-rlvr` | Multiple choice (rule-based option extraction) |
| `mcq-llm-as-judge` | Multiple choice (non-R1 models where options aren't in `\boxed{}`, requires LLM extraction) |

### Supported Formatters

Supports running multiple formatters simultaneously, generating both SFT and DPO data from one rollout.

| Type | Description | Early Stop Condition |
|------|-------------|---------------------|
| `sft` | Select highest-scoring response | Has 1 pass (score >= pass_threshold) |
| `dpo` | Select highest + lowest scoring responses | Has 1 pass + 1 fail (score <= fail_threshold) |

---

## Work Directory Design

Uses **timestamp-based paths** to organize experiments for easy tracking, reproduction, and resume.

```
output/20251206_143052/
├── config.yaml                   # Experiment config (auto-saved)
├── state.json                    # Run state (progress, checkpoint)
├── data/
│   └── input.jsonl               # Preprocessed data
├── rollout/                      # Inference + evaluation results (sharded storage)
│   ├── shard_0000.jsonl
│   └── ...
├── train/                        # Training data
│   ├── sft.jsonl
│   └── dpo.jsonl
└── summary/                      # Analysis results
    └── stats.json
```

### Data Preprocessing

**Flow**:
```
Raw data → DataPreprocessor → Format check → data/input.jsonl
                ↓
          transform (optional)
```

**Logic**:
1. Check if `work_dir/data/input.jsonl` exists
2. If exists → Skip preprocessing (resume scenario)
3. If not exists → Read raw data → transform (optional) → Format check → Write

**Format Requirements**:
```python
{
    "id": str,                           # Required: unique identifier
    "messages": [                        # Required: OpenAI messages format
        {"role": "user", "content": str}
    ],
    "metadata": {                        # Required: metadata
        "answer": str,                   # Optional: ground truth answer (warning printed if missing)
        ...
    }
}
```

**Transform Function Interface**:
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

**Usage Examples**:
```bash
# Data already formatted, direct copy
python run.py data.input_path=/path/to/formatted.jsonl

# Needs transformation
python run.py data.input_path=/path/to/raw.jsonl \
  data.preprocess.transform=transforms/gsm8k.py:transform

# Resume, already has data/input.jsonl, skip preprocessing
python run.py work_dir=output/20251206_143052/
```

### Sharded Storage

Rollout results are stored in shards (default 10000 items per shard), benefits:
- Supports large-scale data (100k+) without memory explosion
- Only need to rerun incomplete shards on resume
- Facilitates parallel processing

---

## Configuration Management

Uses **Hydra** for configuration management, supporting YAML config + command line overrides.

### Configuration Example

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
  num_gpus: null                 # Only used for vllm-offline, null = auto detect
  temperature: 0.7
  max_tokens: 2048
  top_p: 1.0
  concurrent_requests: 50        # Only used for openai-compatible-api
  timeout: 300
  drop_truncated: true           # Drop truncated responses
  extra_params: {}               # Extra params for specific models (e.g., reasoning_effort for OSS models)

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

## User Interface

```bash
# Start a new experiment
python run.py data.input_path=/path/to/data.jsonl

# Override config
python run.py data.input_path=/path/to/data.jsonl \
  sampler.model=deepseek-chat \
  sampling.max_rollouts=32

# Resume
python run.py work_dir=output/20251206_143052/
```

---

## Data Formats

Uses **Messages format** (OpenAI standard).

### Input Format

```jsonl
{"id": "001", "messages": [{"role": "user", "content": "Question..."}], "metadata": {"answer": "42"}}
```

### Rollout Output Format

```jsonl
{
  "id": "001",
  "messages": [{"role": "user", "content": "Question..."}],
  "metadata": {"answer": "42"},
  "rollouts": [
    {"response": "...", "score": 1.0},
    {"response": "...", "score": 0.0}
  ]
}
```

### Training Data Formats

**SFT:**
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**DPO:**
```jsonl
{"prompt": [{"role": "user", "content": "..."}], "chosen": [...], "rejected": [...]}
```

---

## Core Features

### 1. Sampling Flow

```
Goal: Collect max_rollouts valid rollouts

step 1: roll step_size items → drop truncated → keep valid → check early stop
step 2: roll step_size items → drop truncated → keep valid → check early stop
...
Stop conditions: valid rollouts >= max_rollouts OR step >= max_steps OR early stop satisfied
```

**Configuration Examples**:
- `max_rollouts=16, step_size=4, max_steps=4`: Exactly 4 rounds with no truncation
- `max_rollouts=16, step_size=4, max_steps=8`: Allow 2x rounds to handle truncation

### 2. Smart Early Stopping

Stop sampling early based on formatter needs:

- SFT early stop condition: Has 1 pass
- DPO early stop condition: Has 1 pass + 1 fail
- Multiple formatters: Stop only when all formatters are satisfied

### 3. Checkpoint Resume

- `state.json` records list of completed shards
- Automatically skip completed shards on restart

### 4. Quality Analysis

Statistics on pass rate, token distribution, average sampling count, etc., saved to `summary/stats.json`.

---

## Project Structure

```
rejection-sampling-recipes/
├── configs/                     # Hydra configs
├── src/
│   ├── sampler/                 # Samplers
│   ├── verifier/                # Verifiers
│   ├── formatter/               # Formatters
│   ├── pipeline.py              # Main pipeline
│   └── analysis.py              # Quality analysis
├── run.py                       # Entry point
├── recipes/                     # Example recipes
├── pyproject.toml               # uv
└── requirements.txt             # pip
```

---

## Environment Management

Two options supported:

**uv (recommended):**
```bash
uv sync
uv run python run.py ...
```

**conda + pip:**
```bash
conda create -n rsr python=3.12 -y
conda activate rsr
pip install -r requirements.txt
python run.py ...
```

---

## Development Standards

### Branch Strategy

- `main`: Stable branch, direct push during initial development, only accepts PRs later
- `feat/*`: Feature branches, PR to main when complete
- `fix/*`: Fix branches

### Code Standards

- **Language**: Code comments, docstrings, commit messages all in English
- **Linter**: Use ruff (lint + format)
- **Type Hints**: Recommended

### CI Configuration

GitHub Actions automatically runs:
- ruff check (lint)
- ruff format --check (format)
- pytest (unit tests)

### Project File Checklist

```
rejection-sampling-recipes/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI config
├── .gitignore
├── .pre-commit-config.yaml      # pre-commit hooks
├── LICENSE                      # MIT
├── README.md                    # English, for open source community
├── docs/
│   └── design.md                # Design document
│   └── design_cn.md             # Chinese design document
├── pyproject.toml               # Project config + ruff config
├── requirements.txt
├── configs/
├── src/
├── tests/                       # Unit tests
└── run.py
```

---

## Development Flow

### Module Interaction

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

### Module Responsibilities

| Module | Input | Output | Responsibility |
|--------|-------|--------|----------------|
| **DataPreprocessor** | raw jsonl | `data/input.jsonl` | Format conversion + validation |
| **Sampler** | messages | `List[str]` | Call LLM to generate responses |
| **Verifier** | response + metadata | `float` | Evaluate response, return score |
| **Formatter** | item + rollouts | `List[dict]` | Filter and format into training data |
| **StateManager** | - | - | Manage checkpoint resume state |

### Design Decisions

**Sampler**: Simple factory function (only two types)
```python
def get_sampler(cfg):
    if cfg.type == "openai-compatible-api":
        return OpenAISampler(cfg)
    elif cfg.type == "vllm-offline":
        return VLLMSampler(cfg)
```

**Verifier**: Registry pattern (many types, users may extend)
```python
@register_verifier("math-rlvr")
class MathRLVRVerifier(BaseVerifier): ...

@register_verifier("mcq-rlvr")
class MCQRLVRVerifier(BaseVerifier): ...

# Usage
verifier = get_verifier(cfg.verifier.type)
```

**Formatter**: Registry pattern (users may extend)
```python
@register_formatter("sft")
class SFTFormatter(BaseFormatter): ...      # Select highest score

@register_formatter("dpo")
class DPOFormatter(BaseFormatter): ...      # Select highest + lowest

@register_formatter("top_k")
class TopKFormatter(BaseFormatter): ...     # Select top k above threshold

# Usage
formatter = get_formatter(cfg.type)
```

### Development Phases

#### Phase 1: Sampler (Inference Module)

**Goal**: Implement stable inference capability

**Tasks**:
- [ ] Implement `OpenAISampler` (asyncio concurrent)
- [ ] Implement retry, timeout, error handling
- [ ] Support batch sampling (using `n` parameter)

**Testing**:
- Basic functionality: Can it call API and return results
- Concurrency: Is it stable under high concurrency
- Error handling: Can it correctly retry on timeout/rate limiting

**Deliverables**:
- `src/sampler/openai_sampler.py`
- `tests/test_sampler.py`
- A batch of real inference results (for testing Verifier later)

---

#### Phase 2: Verifier (Evaluation Module)

**Goal**: Implement accurate evaluation capability

**Tasks**:
- [ ] Implement `MCQVerifier` (option extraction + matching)
- [ ] Handle different model output format differences:
  - With/without reasoning process
  - `\boxed{}`, `【Answer】`, direct output formats
  - Special tokens differences

**Testing**:
- Use real inference results from Phase 1 to construct test cases
- Cover various edge cases:
  - Normal format
  - Format variations (Chinese/English, full-width/half-width)
  - Cases where answer cannot be extracted
  - Numerical precision issues (0.3333 vs 1/3)

**Deliverables**:
- `src/verifier/math_verifier.py`
- `src/verifier/mcq_verifier.py`
- `tests/test_verifier.py` (extensive test cases)
- `tests/fixtures/` real inference result fixtures

---

#### Phase 3: Formatter (Formatting Module)

**Goal**: Implement flexible data filtering and formatting

**Tasks**:
- [ ] Implement `SFTFormatter` (select highest score)
- [ ] Implement `DPOFormatter` (select highest + lowest)
- [ ] Implement early stop condition check `is_satisfied()`

**Testing**:
- Is filtering logic correct
- Edge cases: all pass, all fail, only one

**Deliverables**:
- `src/formatter/sft_formatter.py`
- `src/formatter/dpo_formatter.py`
- `tests/test_formatter.py`

---

#### Phase 4: Pipeline (Overall Flow)

**Goal**: Connect all modules, implement complete flow

**Tasks**:
- [ ] Implement `Pipeline` main flow
- [ ] Implement `StateManager` (checkpoint resume)
- [ ] Implement shard storage
- [ ] Implement smart early stop logic
- [ ] Integrate Hydra configuration

**Testing**:
- End-to-end test: input → output
- Checkpoint resume: Can it correctly recover after interruption
- Shard storage: Does it work with large data volumes
- Early stop: Does it reduce sampling as expected

**Deliverables**:
- `src/pipeline.py`
- `src/state.py`
- `tests/test_pipeline.py`
- `run.py`

---

#### Phase 5: Quality Analysis + Documentation

**Tasks**:
- [ ] Implement `Analysis` statistics module
- [ ] Complete README and usage documentation
- [ ] Provide example recipes

### Testing Strategy

```
tests/
├── fixtures/                    # Test data
│   ├── sample_inputs.jsonl      # Input samples
│   └── sample_outputs/          # Real inference results from Phase 1
│       ├── math_responses.jsonl
│       └── mcq_responses.jsonl
├── test_sampler.py              # Phase 1
├── test_verifier.py             # Phase 2 (core, most test cases)
├── test_formatter.py            # Phase 3
├── test_pipeline.py             # Phase 4 (integration tests)
└── conftest.py                  # pytest fixtures
```

### Suggested Development Timeline

```
Week 1: Phase 1 (Sampler)
        ├── Implement OpenAISampler
        └── Collect real inference results as test data

Week 2: Phase 2 (Verifier) ← Core, takes most time
        ├── Implement MathVerifier
        ├── Implement MCQVerifier
        └── Extensive test cases

Week 3: Phase 3 + 4 (Formatter + Pipeline)
        ├── Implement Formatter
        ├── Implement Pipeline
        └── Checkpoint resume testing

Week 4: Phase 5 + Wrap-up
        ├── Quality analysis
        ├── Documentation completion
        └── Example recipes
```
