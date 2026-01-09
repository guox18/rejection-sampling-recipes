Quick Start
1) 部署并启动 SGLang Router
2) 在 `scripts/launch_serve/submit_and_test_until.sh` 设置 `MODEL_NAME` 和 `NUM_INSTANCES`
3) 启动并等待注册完成：
```bash
bash scripts/launch_serve/submit_and_test_until.sh
```
4) 服务就绪后执行你的下游脚本（示例）：
```bash
bash xxx.sh
```

补充说明：
- `MODEL_NAME` 需与配置文件里的模型名一致
- `NUM_INSTANCES` 与模型并行方式相关（具体规则见 `scripts/launch_serve/AGENTS.md`）
