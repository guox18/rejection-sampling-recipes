#!/bin/bash
set -euo pipefail

# ============================================================================
# 完整工作流示例：通过 rjob 启动服务 -> 运行任务
# ============================================================================

echo "=========================================="
echo "完整工作流示例（rjob 模式）"
echo "=========================================="
echo ""

# ------------- 配置参数 -------------
ROUTER_IP="100.102.249.23"
# ROUTER_PORT="21001"
# MODEL_NAME="qwen3_vl_235b_a22b_thinking"

# ROUTER_PORT="21002"
# MODEL_NAME="qwen3_vl_30b_a3b_thinking"

ROUTER_PORT="21003"
MODEL_NAME="qwen25_32b_instruct"

# ⭐ 重要: NUM_INSTANCES 的含义已改变！
# - 对于 TP=8 的大模型（如 235B）: NUM_INSTANCES=8 (8个rjob任务，每个任务1个vllm实例)
# - 对于 TP=1 的小模型（如 30B）:  NUM_INSTANCES=1 (1个rjob任务，自动启动8个vllm实例)
NUM_INSTANCES=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Step 1: 通过 rjob 提交 ${NUM_INSTANCES} 个模型服务任务..."
echo ""

# 提交 rjob 任务（脚本会阻塞直到所有服务就绪）
bash "${SCRIPT_DIR}/submit_rjob_instances.sh" \
  -n ${NUM_INSTANCES} \
  --config model_config_example.yaml \
  --model ${MODEL_NAME} \
  --router-ip ${ROUTER_IP} \
  --router-port ${ROUTER_PORT}

# 如果上面的命令成功返回，说明所有服务都已启动并注册成功
echo ""
echo "=========================================="
echo "Step 2: 运行使用服务的任务..."
echo "=========================================="
echo ""

# 现在可以安全地运行依赖这些服务的任务
echo "服务已就绪，开始运行任务..."
echo "Router 地址: http://${ROUTER_IP}:${ROUTER_PORT}"
echo ""

# 示例：测试服务是否可用
echo "测试服务连接..."
if curl -sf "http://${ROUTER_IP}:${ROUTER_PORT}/health" >/dev/null; then
  echo "✓ Router 连接正常"
  echo ""
  echo "健康状态:"
  curl -s "http://${ROUTER_IP}:${ROUTER_PORT}/health"
  echo ""
  echo ""
  echo "可用模型:"
  curl -s "http://${ROUTER_IP}:${ROUTER_PORT}/v1/models" | python3 -m json.tool || true
else
  echo "✗ Router 连接失败"
  exit 1
fi

echo "=========================================="
echo "✓ 启动完成！"
echo "=========================================="
echo ""
echo "提示: 任务完成后，记得清理 rjob 任务"
echo "  查看任务: cat /tmp/vllm_rjobs_${ROUTER_PORT}.txt"
echo "  停止任务: 参考上面文件中的任务名称，使用 rjob stop <job-name>"

