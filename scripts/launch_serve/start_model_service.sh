#!/bin/bash
set -euo pipefail

# ============================================================================
# 模型服务启动脚本
# 功能: 启动模型服务并自动注册到 SGLang Router
# ============================================================================

# ------------- 使用说明 -------------
usage() {
  cat <<EOF
用法: $0 --config <配置文件> --router-ip <IP> --router-port <端口> [选项]

必需参数:
  --config FILE         模型配置文件 (YAML 格式)
  --router-ip IP        SGLang Router 的 IP 地址
  --router-port PORT    SGLang Router 的端口

可选参数:
  --local-port PORT     本地服务端口 (默认: 8000)
  --model-name NAME     覆盖配置文件中的模型名称
  --help                显示此帮助信息

配置文件示例:
  qwen3_vl_235b_a22b_instruct:
    tp: 8
    dp: 1
    pp: 1
    worker_count: 1
    memory_per_task: 1200
    cpus_per_task: 32
    model_path: "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-235B-A22B-Instruct/snapshots/xxx"

示例:
  $0 --config model_config.yaml --router-ip 100.102.249.23 --router-port 21001

EOF
  exit 1
}

# ------------- 参数解析 -------------
CONFIG_FILE=""
ROUTER_IP=""
ROUTER_PORT=""
LOCAL_PORT=8000
MODEL_NAME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
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
    --local-port)
      LOCAL_PORT="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME_OVERRIDE="$2"
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
if [ -z "$CONFIG_FILE" ] || [ -z "$ROUTER_IP" ] || [ -z "$ROUTER_PORT" ]; then
  echo "错误: 缺少必需参数"
  usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

# ------------- 读取配置文件 -------------
echo "正在解析配置文件: $CONFIG_FILE"

read_config() {
  python3 <<EOF
import yaml
import sys

try:
    with open("$CONFIG_FILE", "r") as f:
        config = yaml.safe_load(f)
    
    # 获取第一个模型配置
    if not config:
        print("错误: 配置文件为空", file=sys.stderr)
        sys.exit(1)
    
    model_name = list(config.keys())[0]
    model_config = config[model_name]
    
    # 如果有覆盖的模型名称，使用它
    if "$MODEL_NAME_OVERRIDE":
        model_name = "$MODEL_NAME_OVERRIDE"
    
    print(f"MODEL_NAME={model_name}")
    print(f"TP={model_config.get('tp', 1)}")
    print(f"DP={model_config.get('dp', 1)}")
    print(f"PP={model_config.get('pp', 1)}")
    print(f"MODEL_PATH={model_config['model_path']}")
    
except Exception as e:
    print(f"错误: 解析配置文件失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# 解析配置并导出变量
eval $(read_config)

# ------------- 显示配置信息 -------------
echo ""
echo "=========================================="
echo "         模型服务启动配置"
echo "=========================================="
echo "模型名称:     $MODEL_NAME"
echo "模型路径:     $MODEL_PATH"
echo "张量并行(TP): $TP"
echo "数据并行(DP): $DP"
echo "流水线并行:   $PP"
echo "本地端口:     $LOCAL_PORT"
echo "Router地址:   $ROUTER_IP:$ROUTER_PORT"
echo "=========================================="
echo ""

# ------------- 检查模型路径 -------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "错误: 模型路径不存在: $MODEL_PATH"
  exit 1
fi

# ------------- 环境配置 -------------
echo "激活conda环境..."
source /mnt/shared-storage-user/ailab-hx/wulianyi/miniconda3/etc/profile.d/conda.sh
conda activate sglang_conda

export TIKTOKEN_ENCODINGS_BASE=/mnt/shared-storage-user/ailab-hx/wulianyi/tiktoken_encodings

# ------------- 检查端口占用 -------------
if lsof -i :$LOCAL_PORT >/dev/null 2>&1; then
  echo "警告: 端口 $LOCAL_PORT 被占用，尝试释放..."
  lsof -t -i :$LOCAL_PORT | xargs -r kill -9 || true
  sleep 3
fi

# ------------- 工具函数 -------------
get_ip() {
  local ip
  ip=$(hostname -I 2>/dev/null | awk '{print $1}' | head -n1 || true)
  if [ -z "${ip}" ]; then
    ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' || true)
  fi
  if [ -n "${SERVER_IP:-}" ]; then
    ip="$SERVER_IP"  # 允许人工覆盖
  fi
  echo "$ip"
}

# ------------- 后台注册到 Router -------------
register_to_router() {
  (
    set -euo pipefail
    LOG="/tmp/sglang_register_${LOCAL_PORT}.log"
    echo "[$(date '+%F %T')] 启动注册后台任务..." > "$LOG"
    echo "[$(date '+%F %T')] Router 地址: http://${ROUTER_IP}:${ROUTER_PORT}" >> "$LOG"
    
    # 等待健康检查: 最多 40 次，每次 30s，总计 ~20 分钟
    RETRIES=40
    INTERVAL=30
    
    echo "[$(date '+%F %T')] 开始健康检查轮询..." >> "$LOG"
    
    for ((i=1; i<=RETRIES; i++)); do
      if curl -s --connect-timeout 5 http://localhost:$LOCAL_PORT/health >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] ✓ 健康检查通过!" >> "$LOG"
        break
      fi
      echo "[$(date '+%F %T')] 等待服务启动... ($i/$RETRIES)" >> "$LOG"
      sleep "$INTERVAL"
      if [ $i -eq $RETRIES ]; then
        echo "[$(date '+%F %T')] ✗ 健康检查超时，放弃注册" >> "$LOG"
        exit 0
      fi
    done
    
    # 获取本机IP
    IP="$(get_ip)"
    if [ -z "$IP" ]; then
      echo "[$(date '+%F %T')] ✗ 获取本机IP失败，放弃注册" >> "$LOG"
      exit 0
    fi
    
    WORKER_URL="http://${IP}:${LOCAL_PORT}"
    ROUTER_URL="http://${ROUTER_IP}:${ROUTER_PORT}"
    
    echo "[$(date '+%F %T')] 尝试注册 Worker: ${WORKER_URL}" >> "$LOG"
    
    # 尝试注册到 Router
    if curl -s -X POST "${ROUTER_URL}/add_worker?url=${WORKER_URL}" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] ✓ 成功注册到 Router!" >> "$LOG"
      echo "[$(date '+%F %T')]   Worker URL: ${WORKER_URL}" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "=========================================="
      echo "✓ 服务已成功注册到 Router!"
      echo "  Worker: ${WORKER_URL}"
      echo "  Router: ${ROUTER_URL}"
      echo "=========================================="
    else
      echo "[$(date '+%F %T')] ⚠ 注册失败，请检查 Router 是否正常运行" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "⚠ 警告: 注册到 Router 失败，请检查 ${ROUTER_URL}"
    fi
  ) &
  
  local pid=$!
  echo $pid > /tmp/sglang_register_${LOCAL_PORT}.pid
  echo "后台注册任务已启动 (PID: $pid)"
  echo "日志文件: /tmp/sglang_register_${LOCAL_PORT}.log"
}

# ------------- 启动注册后台任务 -------------
echo "启动后台注册任务..."
register_to_router
echo ""

# ------------- 启动模型服务 -------------
echo "=========================================="
echo "以前台方式启动模型服务..."
echo "服务将在端口 $LOCAL_PORT 上监听"
echo "按 Ctrl+C 可终止服务器进程"
echo "=========================================="
echo ""

# 使用 exec 以前台方式运行，确保信号能正确传递
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$LOCAL_PORT" \
  --max-prefill-tokens 32768 \
  --context-length 32768 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.92


