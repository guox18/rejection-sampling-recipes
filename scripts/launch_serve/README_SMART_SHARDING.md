# vLLM 智能分片启动说明

## 新功能

`start_vllm_service.sh` 现在支持**智能GPU分片**：
- 当 `TP * DP * PP < 8` 时，自动在单个8卡节点上启动多个 vLLM 实例
- 每个实例使用独立的 GPU 和端口，自动注册到 SGLang Router

## 使用方法

### 场景1：大模型（需要多卡并行，如 TP=8）

**配置示例：**
```yaml
qwen3_vl_235b_a22b_thinking:
  tp: 8      # 需要8卡并行
  dp: 1
  pp: 1
```

**启动方式：**
```bash
# submit_and_test_until.sh 中设置
NUM_INSTANCES=8    # 提交8个rjob任务

# 结果：8个任务 × 1个实例 = 8个 vLLM 实例
```

### 场景2：小模型（单卡即可，如 TP=1）⭐ 推荐用法

**配置示例：**
```yaml
qwen3_vl_30b_a3b_thinking:
  tp: 1      # 单卡即可
  dp: 1      # ⚠️ MoE模型暂不支持 DP>1
  pp: 1
```

**启动方式：**
```bash
# submit_and_test_until.sh 中设置
NUM_INSTANCES=1    # 只提交1个rjob任务

# 结果：1个任务会自动启动8个实例，端口 8000-8007
```

## 配置验证

脚本会自动检查配置合理性：
- ✅ 8 能被 (TP×DP×PP) 整除
- ❌ 8 不能被 (TP×DP×PP) 整除 → 报错并给出建议

**合理的配置示例：**
- `TP=1, DP=1, PP=1` → 启动 8 个实例
- `TP=2, DP=1, PP=1` → 启动 4 个实例
- `TP=4, DP=1, PP=1` → 启动 2 个实例
- `TP=8, DP=1, PP=1` → 启动 1 个实例

## 日志文件

每个实例有独立的日志：
- vLLM 日志：`/tmp/vllm_<端口>.log`
- 注册日志：`/tmp/vllm_register_<端口>.log`
- 进程 PID：`/tmp/vllm_<端口>.pid`

## 完整示例

```bash
# 1. 修改 submit_and_test_until.sh
MODEL_NAME="qwen3_vl_30b_a3b_thinking"
NUM_INSTANCES=1  # ⚠️ 对于 TP=1 的模型，改为 1

# 2. 执行启动
bash scripts/launch_serve/submit_and_test_until.sh

# 3. 脚本会自动：
#    - 提交 1 个 rjob 任务（8卡）
#    - 该任务会启动 8 个 vLLM 实例（端口 8000-8007）
#    - 等待所有实例注册到 Router
#    - 验证服务可用性
```

## 注意事项

1. **MoE 模型暂不支持 DP > 1**
   - 使用 `dp: 1` 配置
   - 可以使用 `tp` 进行张量并行

2. **端口自动分配**
   - 基础端口（默认 8000）由 `--local-port` 指定
   - 实例端口为：8000, 8001, 8002, ..., 8007

3. **GPU 分配**
   - 使用 `CUDA_VISIBLE_DEVICES` 自动分配
   - 例如：实例0用GPU 0，实例1用GPU 1，...

4. **查看日志**
   - 主进程会 `tail -f` 第一个实例的日志
   - 按 Ctrl+C 退出日志查看（不影响服务运行）

## 问题排查

如果服务启动失败，检查：

1. **配置是否合理**
   ```bash
   # 检查配置文件
   cat scripts/launch_serve/model_config_example.yaml
   ```

2. **查看实例日志**
   ```bash
   # 查看所有实例日志
   tail -f /tmp/vllm_800*.log
   
   # 查看特定实例
   tail -f /tmp/vllm_8000.log
   ```

3. **检查端口占用**
   ```bash
   lsof -i :8000-8007
   ```

4. **验证 Router 注册**
   ```bash
   curl http://<router-ip>:<router-port>/list_workers
   ```

