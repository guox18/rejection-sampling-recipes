Quick Start
1) Deploy and start the SGLang Router
```bash
pip install sglang-router
python -m sglang_router.launch_router --host 0.0.0.0 --port 21001 --prometheus-port 29001
```
2) Set `MODEL_NAME` and `NUM_INSTANCES` in `scripts/launch_serve/submit_and_test_until.sh`
3) Launch and wait for registration to complete:
```bash
bash scripts/launch_serve/submit_and_test_until.sh
```
4) After services are ready, run your downstream script (example):
```bash
bash xxx.sh
```

Notes:
- `MODEL_NAME` must match the model name in the config file
- `NUM_INSTANCES` depends on the model parallelism setup (see `scripts/launch_serve/AGENTS.md`)

## Add CPU Workers to Existing Ray Cluster

When you already have a Ray head node on the current machine, you can let newly created
machines join it as CPU workers.

1) On the Ray head machine, save head IP to shared storage:
```bash
hostname -i | awk '{print $1}' > scripts/launch_serve/.ray_head_ip
```

2) In your rjob task (running on the new machine), execute:
```bash
bash scripts/launch_serve/join_ray_cpu_worker.sh
```

The script auto-detects local CPU/memory and joins `<head-ip>:6379`. It keeps running to keep
the worker alive.
