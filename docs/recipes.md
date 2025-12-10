# Recipes

RSR supports three training data formats: **SFT**, **DPO**, and **Multi-SFT**.

## SFT (Supervised Fine-Tuning)

Collect the **best correct response** for each prompt.

### When to Use

- Knowledge distillation from stronger models
- Curating high-quality training data
- You only need one good answer per question

### Configuration

```yaml
# configs/sft.yaml
sampling:
  max_rollouts: 16
  step_size: 4
  max_steps: 4
  early_stop: true        # Stop once we have 1 correct

formatter:
  - type: sft
    pass_threshold: 1.0   # score >= 1.0 is correct
```

### Early Stop Condition

Satisfied when: **≥1 passed response**

### Output Format

Compatible with [LLaMA-Factory OpenAI format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md):

```json
{
  "messages": [
    {"role": "user", "content": "Which is correct?\n\nA: X\nB: Y"},
    {"role": "assistant", "content": "The answer is B because..."}
  ]
}
```

---

## DPO (Direct Preference Optimization)

Collect **(chosen, rejected) pairs** for preference learning.

### When to Use

- Training with preference optimization (DPO, ORPO, etc.)
- You need both correct and incorrect examples
- Building contrast pairs for alignment

### Configuration

```yaml
# configs/dpo.yaml
sampling:
  max_rollouts: 32
  step_size: 8
  max_steps: 4
  early_stop: true        # Stop once we have both pass and fail

formatter:
  - type: dpo
    pass_threshold: 1.0   # score >= 1.0 → chosen
    fail_threshold: 0.0   # score <= 0.0 → rejected
```

### Early Stop Condition

Satisfied when: **≥1 passed AND ≥1 failed response**

### Output Format

Compatible with [LLaMA-Factory ShareGPT preference format](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md):

```json
{
  "conversations": [
    {"from": "human", "value": "Which is correct?\n\nA: X\nB: Y"}
  ],
  "chosen": {"from": "gpt", "value": "The answer is B..."},
  "rejected": {"from": "gpt", "value": "The answer is A..."}
}
```

---

## Multi-SFT (Diversity Sampling)

Collect **N unique correct responses** for each prompt.

### When to Use

- Diversity-focused distillation
- Building response pools for reranking

### Configuration

```yaml
# configs/multi_sft.yaml
sampling:
  max_rollouts: 64
  step_size: 8
  max_steps: 8
  early_stop: true

sampler:
  temperature: 0.8        # Higher for diversity

formatter:
  - type: multi_sft
    num_responses: 16     # Collect 16 unique correct responses
    pass_threshold: 1.0
```

### Early Stop Condition

Satisfied when: **≥num_responses unique passed responses**

### Output Format

```json
{
  "messages": [
    {"role": "user", "content": "Which is correct?\n\nA: X\nB: Y"}
  ],
  "responses": [
    "The answer is B because reason 1...",
    "B is correct. Here's why...",
    "Looking at this problem, B...",
    // ... up to num_responses
  ]
}
```

---

## Combining Formatters

You can run multiple formatters simultaneously:

```yaml
formatter:
  - type: sft
    pass_threshold: 1.0
  - type: dpo
    pass_threshold: 1.0
    fail_threshold: 0.0
```

This generates both `sft.jsonl` and `dpo.jsonl` from the same sampling run.

**Note:** When combining formatters, early stopping triggers when **all** formatters are satisfied.

---

## Recipe Comparison

| Recipe | Goal | Early Stop | Typical Budget |
|--------|------|------------|----------------|
| SFT | 1 best correct | ≥1 pass | 16 samples |
| DPO | 1 pass + 1 fail | ≥1 pass AND ≥1 fail | 32 samples |
| Multi-SFT | N unique correct | ≥N unique pass | 64 samples |
