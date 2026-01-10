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
