# Early Stop

Early stopping is a key efficiency feature that terminates sampling once enough data is collected.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Sampling Loop                            │
│                                                             │
│   for step in 1..max_steps:                                 │
│       responses = sample(step_size)                         │
│       scores = verify(responses)                            │
│                                                             │
│       if early_stop AND all_formatters_satisfied():         │
│           break  ◄── Stop early, save compute               │
│                                                             │
│   # Continue if not satisfied or early_stop=false           │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

```yaml
sampling:
  early_stop: true    # Enable/disable early stopping
```

## When to Enable (default)

**Enable `early_stop: true` when:**

- ✅ Goal is data collection (SFT distillation, DPO pairs)
- ✅ You want to minimize compute cost
- ✅ Easy questions should finish fast

**Example:** For SFT, once you find 1 correct response, there's no need to sample more.

## When to Disable

**Set `early_stop: false` when:**

- ❌ You need accurate pass rate / difficulty estimation
- ❌ You want consistent sample counts per prompt
- ❌ Downstream tasks need pass@k statistics

## Satisfaction Conditions

Each formatter defines when it's "satisfied":

| Formatter | Satisfied When |
|-----------|---------------|
| SFT | ≥1 passed response |
| DPO | ≥1 passed AND ≥1 failed |
| Multi-SFT | ≥num_responses unique passed |

When multiple formatters are configured, early stop triggers when **all** are satisfied.

> **Note:** This means all formatters will output the **same set of prompts** (though in different formats). For example, with SFT + DPO, a prompt is only included if it has both passed and failed responses — prompts with only passed responses are excluded from both outputs.

## Efficiency Impact

Real-world example (MCQ dataset, 10k questions):

| Config | Easy (80% pass) | Hard (20% pass) | Total Samples |
|--------|-----------------|-----------------|---------------|
| `early_stop: true` | ~1.2 avg | ~8.5 avg | ~45k |
| `early_stop: false` | 16 each | 16 each | 160k |

Early stop reduces compute by ~70% while collecting the same training data.

## Interaction with Sampling Parameters

```yaml
sampling:
  max_rollouts: 16      # Soft target
  step_size: 4          # Batch size per step
  max_steps: 4          # Hard limit (4 × 4 = 16 max samples)
  early_stop: true
```

- **With early_stop:** May stop after 1-4 steps for easy questions
- **Without early_stop:** Always runs all 4 steps (16 samples)

## Best Practices

1. **For distillation:** Use `early_stop: true` (default)
2. **For difficulty analysis:** Use `early_stop: false`
3. **For DPO:** Keep `early_stop: true` — need both pass and fail quickly
4. **For Multi-SFT:** Depends on diversity needs; `early_stop: true` is usually fine
