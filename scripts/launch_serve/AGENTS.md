# vLLM Smart Sharding (Agent Runbook)

## Goal

Use `scripts/launch_serve/start_vllm_service.sh` to launch vLLM on a single 8-GPU node
with automatic sharding based on `TP/DP/PP`, and auto-register instances to the SGLang
Router.

## Hard Rules (Must Follow)

- If `TP * DP * PP <= 8`, the script starts multiple vLLM instances on one 8-GPU node.
- `8` must be divisible by `TP * DP * PP`, or the script exits with guidance.
- Some MoE models do not support `DP > 1`; recommand using `dp: 1`.

## Configuration and Launch

### Scenario A: Large model (multi-GPU parallelism, e.g. TP=8)

Config example (same format as `scripts/launch_serve/model_config_example.yaml`):
```yaml
qwen3_vl_235b_a22b_thinking:
  tp: 8
  dp: 1
  pp: 1
```

Launch (submit 8 rjob tasks):
```bash
# scripts/launch_serve/submit_and_test_until.sh
NUM_INSTANCES=8

# Result: 8 tasks x 1 instance = 8 vLLM instances
```

### Scenario B: Small model (single GPU, e.g. TP=1)

Config example:
```yaml
qwen3_vl_30b_a3b_thinking:
  tp: 1
  dp: 1
  pp: 1
```

Launch (submit 1 rjob task):
```bash
# scripts/launch_serve/submit_and_test_until.sh
NUM_INSTANCES=1

# Result: 1 task auto-starts 8 instances (ports 8000-8007)
```

## Valid Sharding Combinations (Examples)

- `TP=1, DP=1, PP=1` -> 8 instances
- `TP=2, DP=1, PP=1` -> 4 instances
- `TP=4, DP=1, PP=1` -> 2 instances
- `TP=8, DP=1, PP=1` -> 1 instance

## Main Flow (Recommended)

```bash
# 1) Set model and task count
# scripts/launch_serve/submit_and_test_until.sh
MODEL_NAME="qwen3_vl_30b_a3b_thinking"
NUM_INSTANCES=1

# 2) Launch
bash scripts/launch_serve/submit_and_test_until.sh
```

The script automatically:
- Submits the rjob task (8 GPUs)
- Starts multiple vLLM instances per the sharding rules
- Waits for instances to register with the Router
- Verifies service availability

## Ports and GPU Mapping

- Base port is set via `--local-port` (default 8000).
- Instance ports: 8000, 8001, ..., 8007.
- `CUDA_VISIBLE_DEVICES` is auto-assigned (instance0 -> GPU0, instance1 -> GPU1, ...).

## Logs and Process Info

- vLLM logs: `/tmp/vllm_<port>.log`
- Registration logs: `/tmp/vllm_register_<port>.log`
- PID files: `/tmp/vllm_<port>.pid`

The main process tails the first instance log; `Ctrl+C` stops tailing only, not the service.

## Troubleshooting (In Order)

1) Check configuration:
```bash
cat scripts/launch_serve/model_config_example.yaml
```

2) Inspect instance logs:
```bash
tail -f /tmp/vllm_800*.log
```

3) Check port usage:
```bash
lsof -i :8000-8007
```

4) Verify Router registration:
```bash
curl http://<router-ip>:<router-port>/list_workers
```
