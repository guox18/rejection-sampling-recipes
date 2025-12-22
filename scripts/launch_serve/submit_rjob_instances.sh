#!/bin/bash
set -euo pipefail

# ============================================================================
# 通过 rjob 提交多个模型服务实例
# 每个实例在独立的 GPU 节点上运行
# ============================================================================

# ------------- 使用说明 -------------
usage() {
  cat <<EOF
用法: $0 -n <实例数量> --config <配置文件> --model <模型名称> --router-ip <IP> --router-port <端口> [选项]

必需参数:
  -n NUM                实例数量（提交 N 个 rjob 任务）
  --config FILE         模型配置文件（YAML 格式）
  --model NAME          要启动的模型名称
  --router-ip IP        SGLang Router 的 IP 地址
  --router-port PORT    SGLang Router 的端口

可选参数:
  --start-port PORT     起始端口号（默认: 8000）
  --namespace NS        rjob namespace（默认: ailab-puyullmgpunew）
  --charged-group GRP   计费组（默认: puyullmgpunew_gpu）
  --image IMG           Docker 镜像
  --help                显示此帮助信息

示例:
  $0 -n 4 --config model_config_example.yaml --model qwen3_vl_235b_a22b_thinking --router-ip 100.102.249.23 --router-port 21001

EOF
  exit 1
}

# ------------- 参数解析 -------------
NUM_INSTANCES=""
CONFIG_FILE=""
MODEL_NAME=""
ROUTER_IP=""
ROUTER_PORT=""
START_PORT=8000
NAMESPACE="ailab-puyullmgpunew"
CHARGED_GROUP="puyullmgpunew_gpu"
IMAGE="registry.h.pjlab.org.cn/ailab-puyullmgpu/vllm-openai:v0.11.0"

while [[ $# -gt 0 ]]; do
  case $1 in
    -n)
      NUM_INSTANCES="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --router-ip)
      ROUTER_IP="$2"
      shift 2
      ;;
    --router-port)
      ROUTER_PORT="$2"
      shift 2
      ;;
    --start-port)
      START_PORT="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --charged-group)
      CHARGED_GROUP="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "错误: 未知参数 $1"
      usage
      ;;
  esac
done

# ------------- 检查必需参数 -------------
if [ -z "$NUM_INSTANCES" ] || [ -z "$CONFIG_FILE" ] || [ -z "$MODEL_NAME" ] || [ -z "$ROUTER_IP" ] || [ -z "$ROUTER_PORT" ]; then
  echo "错误: 缺少必需参数"
  usage
fi

if ! [[ "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [ "$NUM_INSTANCES" -lt 1 ]; then
  echo "错误: 实例数量必须为正整数"
  exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查配置文件
if [ ! -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]; then
  echo "错误: 配置文件不存在: ${CONFIG_FILE}"
  exit 1
fi

# 检查 rjob 命令
if ! command -v rjob &> /dev/null; then
  echo "错误: rjob 命令不可用，请确保在正确的环境中运行"
  exit 1
fi

# ------------- 读取模型配置计算实际实例数 -------------
echo "正在读取模型配置..."
read_model_config() {
  python3 <<EOF
import yaml
import sys

try:
    with open("${SCRIPT_DIR}/${CONFIG_FILE}", "r") as f:
        config = yaml.safe_load(f)
    
    if "${MODEL_NAME}" not in config:
        print("错误: 模型不存在于配置文件中", file=sys.stderr)
        sys.exit(1)
    
    model_config = config["${MODEL_NAME}"]
    tp = model_config.get('tp', 1)
    dp = model_config.get('dp', 1)
    pp = model_config.get('pp', 1)
    
    gpus_per_instance = tp * dp * pp
    total_gpus = 8
    instances_per_job = total_gpus // gpus_per_instance
    
    print(f"TP={tp}")
    print(f"DP={dp}")
    print(f"PP={pp}")
    print(f"GPUS_PER_INSTANCE={gpus_per_instance}")
    print(f"INSTANCES_PER_JOB={instances_per_job}")
    
except Exception as e:
    print(f"错误: 解析配置文件失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

eval $(read_model_config)

# 计算期望的总实例数（用于健康检查）
EXPECTED_TOTAL_INSTANCES=$((NUM_INSTANCES * INSTANCES_PER_JOB))

# ------------- 显示配置信息 -------------
echo ""
echo "=========================================="
echo "提交 rjob 任务"
echo "=========================================="
echo "模型:         ${MODEL_NAME}"
echo "模型配置:     TP=${TP}, DP=${DP}, PP=${PP}"
echo "每个任务GPU:  ${GPUS_PER_INSTANCE}"
echo "rjob任务数:   ${NUM_INSTANCES}"
echo "每任务实例:   ${INSTANCES_PER_JOB}"
echo "总实例数:     ${EXPECTED_TOTAL_INSTANCES}"
echo "起始端口:     ${START_PORT}"
echo "Router:       ${ROUTER_IP}:${ROUTER_PORT}"
echo "Namespace:    ${NAMESPACE}"
echo "计费组:       ${CHARGED_GROUP}"
echo "=========================================="
echo ""

# ------------- 提交 rjob 任务 -------------
SUBMITTED_JOBS=()
SUBMITTED_PORTS=()

for ((i=0; i<NUM_INSTANCES; i++)); do
  # PORT=$((START_PORT + i))
  PORT=$START_PORT
  TIMESTAMP=$(date +%m%d-%H%M%S)
  JOB_NAME="vllm-${MODEL_NAME}-${PORT}-${TIMESTAMP}"
  
  echo "[$((i+1))/${NUM_INSTANCES}] 提交任务: ${JOB_NAME} (端口 ${PORT})..."
  
  # 构建启动命令
 STARTUP_CMD="cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/launch_serve && \
bash start_vllm_service.sh \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --router-ip ${ROUTER_IP} \
  --router-port ${ROUTER_PORT} \
  --local-port ${PORT}"

  echo $STARTUP_CMD
  
  # 提交 rjob 任务
  SUBMIT_OUTPUT=$(rjob submit \
    -e DISTRIBUTED_JOB=true \
    -e NCCL_DEBUG_SUBSYS=ALL \
    --image="${IMAGE}" \
    --namespace "${NAMESPACE}" \
    --host-network=true \
    --name "${JOB_NAME}" \
    -P 1 \
    --gpu 8 \
    --cpu 80 \
    --memory 800000 \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/ailab-hx:/mnt/shared-storage-user/ailab-hx \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources rdma/mlnx_shared=8 \
    --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-user/intern-pretrain-shared02 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --enable-sshd \
    -- bash -c "${STARTUP_CMD}" 2>&1)
  
  if [ $? -eq 0 ]; then
    echo "  ✓ 任务已提交: ${JOB_NAME}"
    SUBMITTED_JOBS+=("${JOB_NAME}")
    SUBMITTED_PORTS+=("${PORT}")
  else
    echo "  ✗ 任务提交失败"
    echo "${SUBMIT_OUTPUT}"
  fi
  
  # 稍微等待，避免任务名冲突
  sleep 2
done

# ------------- 显示提交结果 -------------
echo ""
echo "=========================================="
if [ ${#SUBMITTED_JOBS[@]} -eq 0 ]; then
  echo "✗ 所有任务提交失败！"
  echo "=========================================="
  exit 1
else
  echo "✓ 已提交 ${#SUBMITTED_JOBS[@]}/${NUM_INSTANCES} 个任务"
fi
echo "=========================================="
echo ""

echo "已提交的任务:"
for ((i=0; i<${#SUBMITTED_JOBS[@]}; i++)); do
  echo "  - ${SUBMITTED_JOBS[$i]} (端口 ${SUBMITTED_PORTS[$i]})"
done
echo ""

# ------------- 等待任务启动 -------------
# 注意: rjob list 检测不太准确，跳过此阶段，直接监听 router 端口
echo "=========================================="
echo "跳过任务状态检查（假设任务已启动）..."
echo "=========================================="
echo ""
echo "✓ 跳过 rjob list 检查，直接进入服务健康检查阶段"
echo ""

# ------------- 等待服务健康检查 -------------
echo ""
echo "=========================================="
echo "等待服务健康检查和注册..."
echo "=========================================="
echo ""

echo "提示: 这可能需要 5-20 分钟（模型加载时间）"
echo "将持续检查 SGLang Router 直到所有服务注册成功..."
echo "Router 地址: http://${ROUTER_IP}:${ROUTER_PORT}/list_workers"
echo "超时时间: 30 分钟"
echo ""

HEALTH_TIMEOUT=1800  # 30分钟
START_TIME=$(date +%s)

# 注册验证函数 - 统计所有注册的 worker 数量
check_registration() {
  local workers=$(curl -sf --connect-timeout 3 "http://${ROUTER_IP}:${ROUTER_PORT}/list_workers" 2>/dev/null || echo "")
  if [ -z "$workers" ]; then
    echo 0
    return 0
  fi
  
  # 计算注册的实例数量（统计包含 http 的行数）
  local count=$(echo "$workers" | grep -o "http://" | wc -l)
  
  echo $count
  return 0
}

ALL_REGISTERED=false
CHECK_ITERATION=0
while true; do
  CHECK_ITERATION=$((CHECK_ITERATION + 1))
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  ELAPSED_MIN=$((ELAPSED / 60))
  ELAPSED_SEC=$((ELAPSED % 60))
  
  if [ $ELAPSED -ge $HEALTH_TIMEOUT ]; then
    echo ""
    echo "⚠ 健康检查超时（${HEALTH_TIMEOUT}秒）"
    echo "部分服务可能仍在启动中"
    break
  fi
  
  # 检查注册状态
  REGISTERED_COUNT=$(check_registration)
  
  if [ "$REGISTERED_COUNT" -ge "$EXPECTED_TOTAL_INSTANCES" ]; then
    ALL_REGISTERED=true
    echo ""
    echo "✓ 所有服务已成功注册到 Router！"
    echo "  注册实例数: ${REGISTERED_COUNT}"
    echo "  总共耗时: ${ELAPSED_MIN}分${ELAPSED_SEC}秒"
    break
  fi
  
  # 显示详细的检查信息
  printf "[检查 #%d] 等待服务注册到 SGLang Router... (%d/%d 已注册, 已等待 %d分%d秒)\r" \
    "$CHECK_ITERATION" "$REGISTERED_COUNT" "$EXPECTED_TOTAL_INSTANCES" "$ELAPSED_MIN" "$ELAPSED_SEC"
  
  sleep 10
done

# ------------- 最终结果 -------------
echo ""
echo "=========================================="
if [ "$ALL_REGISTERED" = true ]; then
  echo "✓✓✓ 所有服务已准备就绪！ ✓✓✓"
  echo "已注册实例数: ${REGISTERED_COUNT}"
else
  echo "⚠ 部分服务可能未就绪"
  echo "已注册: ${REGISTERED_COUNT}/${EXPECTED_TOTAL_INSTANCES}"
fi
echo "=========================================="
echo ""

echo "已提交的任务:"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "  - ${JOB}"
done
echo ""

echo "Router 地址: http://${ROUTER_IP}:${ROUTER_PORT}"
echo ""

echo "管理命令:"
echo "  查看任务状态: rjob list | grep vllm"
echo "  查看任务日志: rjob logs <job-name>"
echo "  停止所有任务:"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "    rjob stop ${JOB}"
done
echo ""

# 保存任务信息到文件
JOBS_FILE="/tmp/vllm_rjobs_${ROUTER_PORT}.txt"
echo "# vLLM rjob 任务列表" > "${JOBS_FILE}"
echo "# 创建时间: $(date)" >> "${JOBS_FILE}"
echo "# Router: ${ROUTER_IP}:${ROUTER_PORT}" >> "${JOBS_FILE}"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "${JOB}" >> "${JOBS_FILE}"
done
echo ""
echo "任务信息已保存到: ${JOBS_FILE}"

echo "=========================================="

# 返回状态
if [ "$ALL_REGISTERED" = true ]; then
  exit 0
else
  exit 1
fi

