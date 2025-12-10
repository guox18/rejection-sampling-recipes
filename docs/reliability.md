# Reliability Features

RSR includes features for robust large-scale sampling: truncation detection and checkpoint/resume.

---

## Truncation Detection

### The Problem

When `max_tokens` is too low, LLM responses get silently truncated:

```
User: Explain quantum computing in detail.

Model: Quantum computing uses qubits which can exist in superposition...
       [TRUNCATED - response cut off mid-sentence]
```

Training on truncated responses leads to:
- Degraded reasoning quality (especially for CoT)
- Wasted compute on useless samples

### The Solution

RSR automatically detects and drops truncated responses:

```yaml
sampler:
  max_tokens: 16384         # Set high enough for your task!
  drop_truncated: true      # Auto-discard truncated (default)
```

### How It Works

Detection is based on the `finish_reason` from the API:

| finish_reason | Meaning | Action |
|---------------|---------|--------|
| `stop` | Natural completion | ✅ Keep |
| `length` | Hit max_tokens limit | ❌ Drop |
| `content_filter` | Filtered by safety | ❌ Drop |

### Impact of High Truncation

When truncation rate is high:

1. **More sampling steps needed** — Each step yields fewer valid responses, so early_stop takes longer to satisfy
2. **Lower efficiency** — You're paying for tokens that get discarded
3. **May hit max_steps limit** — If `max_steps × step_size` isn't enough buffer, some prompts won't collect enough rollouts

**Recommendation:** If you notice sampling is slow or `avg_rollouts_per_item` is much lower than `max_rollouts`, try increasing `max_tokens`.

---

## Checkpoint & Resume

### The Problem

Large-scale sampling jobs can take hours or days. Failures (OOM, network issues, preemption) shouldn't mean starting over.

### The Solution

RSR saves progress at shard boundaries:

```
output/20251206_143052/
├── state.json              # Progress tracking
├── rollout/
│   ├── shard_0000.jsonl   # ✅ Completed
│   ├── shard_0001.jsonl   # ✅ Completed
│   └── shard_0002.jsonl   # 🔄 In progress (will resume here)
```

### How It Works

1. **Sharding:** Data is split into chunks (default 10k items each)
2. **State tracking:** `state.json` records completed shards
3. **Resume:** On restart, completed shards are skipped

```yaml
shard:
  size: 10000    # Items per shard file
```

### Resuming a Job

Simply point to the existing work directory:

```bash
# Resume from where it stopped
uv run python run.py work_dir=output/20251206_143052/

# Resume with modified parameters
uv run python run.py work_dir=output/20251206_143052/ sampler.concurrent_requests=256
```

RSR will:
1. Load `config.yaml` from the work directory
2. Check `state.json` for completed shards
3. Skip completed shards
4. Continue from the first incomplete shard

### State File

`state.json` tracks progress:

```json
{
  "status": "running",
  "total_items": 50000,
  "processed_items": 20000,
  "completed_shards": [0, 1],
  "current_shard": 2,
  "started_at": "2024-12-06T14:30:52",
  "error": null
}
```

### Best Practices

1. **Shard size:** Balance between checkpoint frequency and overhead
   - Small shards (1k): More checkpoints, more files
   - Large shards (50k): Fewer checkpoints, lose more on failure
   - Default (10k): Good balance for most cases

2. **Work directory:** Use explicit `work_dir` for important jobs:
   ```bash
   uv run python run.py work_dir=output/my_experiment/ ...
   ```


### Recovery Scenarios

| Scenario | What Happens |
|----------|--------------|
| Clean exit | All shards completed, state = "completed" |
| Ctrl+C | Current shard may be partial, resume re-processes it |
| OOM / Crash | Same as Ctrl+C |
| API timeout | Failed requests logged, continue with next batch |

---
