# 模型服务启动脚本

自动启动模型服务并注册到 SGLang Router 的便捷脚本。

## 功能特性

- ✅ 读取 YAML 配置文件，支持灵活配置
- ✅ 自动健康检查，等待服务完全启动
- ✅ 自动注册到指定的 SGLang Router
- ✅ 支持自定义端口和模型名称
- ✅ 完整的错误处理和日志记录

## 快速开始

### 1. 准备配置文件

创建一个 YAML 配置文件，包含所有可用模型的配置（参考 `model_config_example.yaml`）:

```yaml
# 配置文件包含多个模型
qwen3_vl_235b_a22b_instruct:
  tp: 8
  dp: 1
  pp: 1
  worker_count: 1
  memory_per_task: 1200
  cpus_per_task: 32
  model_path: "/path/to/qwen3-vl-235b-instruct"

qwen3_vl_235b_a22b_thinking:
  tp: 8
  dp: 1
  pp: 1
  worker_count: 1
  memory_per_task: 1200
  cpus_per_task: 32
  model_path: "/path/to/qwen3-vl-235b-thinking"

qwen2_72b_instruct:
  tp: 4
  dp: 1
  pp: 1
  worker_count: 1
  memory_per_task: 600
  cpus_per_task: 16
  model_path: "/path/to/qwen2-72b-instruct"
```

### 2. 申请机器并启动服务

**方式一: 申请机器后手动启动**

```bash
# 申请机器
job_name=rollout-$(date +%m%d-%H%M%S) && \
REPLICAS=1 && \
namespace=ailab-puyullmgpunew && \
charged_group=puyullmgpunew_gpu && \
rjob submit \
  -e DISTRIBUTED_JOB=true \
  -e NCCL_DEBUG_SUBSYS=ALL \
  --image=registry.h.pjlab.org.cn/ailab-puyullmgpu-puyullm_gpu/lvhaijun:qwen3-vl-thz-251009-v006-official \
  --namespace $namespace \
  --host-network=true \
  --name $job_name \
  -P $REPLICAS \
  --gpu 8 \
  --cpu 80 \
  --memory 800000 \
  --charged-group $charged_group \
  --private-machine='group' \
  --gang-start=true \
  --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
  --mount=gpfs://gpfs1/ailab-hx:/mnt/shared-storage-user/ailab-hx \
  --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
  --custom-resources rdma/mlnx_shared=8 \
  --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-user/intern-pretrain-shared02 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  --enable-sshd \
  -- bash -c 'sleep inf'

# SSH 登录到机器后，启动服务
cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/launch_serve
bash start_vllm_service.sh \
  --config model_config.yaml \
  --model qwen3_vl_235b_a22b_instruct \
  --router-ip 100.102.249.23 \
  --router-port 21001
```

**方式二: 申请机器时直接启动服务**

```bash
job_name=rollout-$(date +%m%d-%H%M%S) && \
REPLICAS=1 && \
namespace=ailab-puyullmgpunew && \
charged_group=puyullmgpunew_gpu && \
rjob submit \
  -e DISTRIBUTED_JOB=true \
  -e NCCL_DEBUG_SUBSYS=ALL \
  --image=registry.h.pjlab.org.cn/ailab-puyullmgpu-puyullm_gpu/lvhaijun:qwen3-vl-thz-251009-v006-official \
  --namespace $namespace \
  --host-network=true \
  --name $job_name \
  -P $REPLICAS \
  --gpu 8 \
  --cpu 80 \
  --memory 800000 \
  --charged-group $charged_group \
  --private-machine='group' \
  --gang-start=true \
  --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
  --mount=gpfs://gpfs1/ailab-hx:/mnt/shared-storage-user/ailab-hx \
  --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
  --custom-resources rdma/mlnx_shared=8 \
  --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-user/intern-pretrain-shared02 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  --enable-sshd \
  -- bash -c 'cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/launch_serve && bash start_vllm_service.sh --config model_config.yaml --model qwen3_vl_235b_a22b_instruct --router-ip 100.102.249.23 --router-port 21001'
```

### 3. 使用 tmux 在后台运行

如果想在后台持续运行（推荐）:

```bash
# SSH 登录后
cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/launch_serve

# 使用 tmux 启动
tmux new-session -d -s model-service 'bash start_vllm_service.sh --config model_config.yaml --model qwen3_vl_235b_a22b_instruct --router-ip 100.102.249.23 --router-port 21001'

# 查看日志
tmux attach -t model-service

# 分离 tmux 会话: 按 Ctrl+B 然后按 D
```

## 命令行参数

### 必需参数

- `--config FILE`: 模型配置文件路径 (YAML 格式，包含多个模型)
- `--model NAME`: 要启动的模型名称 (必须在配置文件中定义)
- `--router-ip IP`: SGLang Router 的 IP 地址
- `--router-port PORT`: SGLang Router 的端口

### 可选参数

- `--local-port PORT`: 本地服务端口，默认 8000
- `--model-name NAME`: 覆盖使用的模型名称 (用于注册到 Router)
- `--help`: 显示帮助信息

## 配置文件说明

配置文件为 YAML 格式，包含以下字段：

```yaml
模型名称:
  tp: 张量并行度 (Tensor Parallel)
  dp: 数据并行度 (Data Parallel)
  pp: 流水线并行度 (Pipeline Parallel)
  worker_count: Worker 数量
  memory_per_task: 每个任务的内存 (MB)
  cpus_per_task: 每个任务的 CPU 核心数
  model_path: 模型文件路径
```

## 日志文件

- 注册日志: `/tmp/sglang_register_<端口>.log`
- 服务日志: 直接输出到终端

查看注册日志：

```bash
tail -f /tmp/sglang_register_8000.log
```

## 常见问题

### 1. 端口被占用

脚本会自动尝试释放端口，如果失败，可以手动杀掉进程：

```bash
lsof -t -i :8000 | xargs kill -9
```

### 2. 注册失败

检查 Router 是否正常运行：

```bash
curl http://100.102.249.23:21001/health
```

查看注册日志：

```bash
cat /tmp/sglang_register_8000.log
```

### 3. 模型路径不存在

确保配置文件中的 `model_path` 正确，且在当前机器上可访问。

### 4. 手动注册到 Router

如果自动注册失败，可以手动注册：

```bash
WORKER_URL="http://$(hostname -I | awk '{print $1}'):8000"
ROUTER_URL="http://100.102.249.23:21001"
curl -X POST "${ROUTER_URL}/add_worker?url=${WORKER_URL}"
```

## 使用示例

### 示例 1: 启动 Qwen3 VL Instruct 模型

```bash
bash start_vllm_service.sh \
  --config model_config.yaml \
  --model qwen3_vl_235b_a22b_instruct \
  --router-ip 100.102.249.23 \
  --router-port 21001
```

### 示例 2: 启动 Qwen3 VL Thinking 模型，使用自定义端口

```bash
bash start_vllm_service.sh \
  --config model_config.yaml \
  --model qwen3_vl_235b_a22b_thinking \
  --router-ip 100.102.249.23 \
  --router-port 21001 \
  --local-port 8001
```

### 示例 3: 启动 Qwen2 72B 模型，覆盖模型名称

```bash
bash start_vllm_service.sh \
  --config model_config.yaml \
  --model qwen2_72b_instruct \
  --router-ip 100.102.249.23 \
  --router-port 21001 \
  --model-name my-custom-model-name
```

### 示例 4: 在同一台机器上启动多个模型 (使用不同端口)

```bash
# 启动第一个模型在端口 8000
tmux new-session -d -s model-8000 'bash start_vllm_service.sh --config model_config.yaml --model qwen3_vl_235b_a22b_instruct --router-ip 100.102.249.23 --router-port 21001 --local-port 8000'

# 启动第二个模型在端口 8001
tmux new-session -d -s model-8001 'bash start_vllm_service.sh --config model_config.yaml --model qwen2_72b_instruct --router-ip 100.102.249.23 --router-port 21001 --local-port 8001'
```

## 脚本工作流程

1. 解析命令行参数和配置文件
2. 激活 conda 环境
3. 检查并释放端口占用
4. 启动后台注册任务
   - 轮询健康检查 (最多 40 次 × 30 秒 = 20 分钟)
   - 健康检查通过后注册到 Router
5. 启动模型服务 (前台运行)

## 维护建议

1. **定期检查服务状态**: 确保服务正常运行且已注册到 Router
2. **监控日志**: 定期查看注册日志和服务日志
3. **资源监控**: 使用 `nvidia-smi` 监控 GPU 使用情况
4. **备份配置**: 保存好配置文件以便快速恢复

## 联系方式

如有问题，请联系脚本维护者或查看项目文档。


