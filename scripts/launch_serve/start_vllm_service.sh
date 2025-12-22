#!/bin/bash
set -euo pipefail

# ============================================================================
# VLLM 模型服务启动脚本
# 功能: 启动 VLLM 模型服务并自动注册到 SGLang Router
# ============================================================================

# ------------- 使用说明 -------------
usage() {
  cat <<EOF
用法: $0 --config <配置文件> --model <模型名称> --router-ip <IP> --router-port <端口> [选项]

必需参数:
  --config FILE         模型配置文件 (YAML 格式，包含多个模型)
  --model NAME          要启动的模型名称 (必须在配置文件中定义)
  --router-ip IP        SGLang Router 的 IP 地址
  --router-port PORT    SGLang Router 的端口

可选参数:
  --local-port PORT     本地服务端口 (默认: 8000)
  --model-name NAME     覆盖使用的模型名称 (用于注册到 Router)
  --help                显示此帮助信息

配置文件示例 (model_config.yaml):
  qwen3_vl_235b_instruct:
    tp: 8
    dp: 1
    max_model_len: 128000  # 可选: 最大模型长度
    model_path: "/path/to/model"
    vllm_extra_args:       # 可选: vllm 额外参数
      gpu-memory-utilization: 0.85
      enable-expert-parallel: true  # MoE 模型需要
      mm-encoder-tp-mode: data      # VL 模型需要
  
  qwen2_72b_instruct:
    tp: 4
    dp: 1
    # max_model_len 不设置则使用模型默认值
    model_path: "/path/to/another/model"
    vllm_extra_args:       # 可选: vllm 额外参数
      gpu-memory-utilization: 0.90
      max-num-seqs: 256

示例:
  # 启动 qwen3_vl_235b_instruct 模型
  $0 --config model_config.yaml --model qwen3_vl_235b_instruct --router-ip 100.102.249.23 --router-port 21001
  
  # 启动 qwen2_72b_instruct 模型，使用自定义端口
  $0 --config model_config.yaml --model qwen2_72b_instruct --router-ip 100.102.249.23 --router-port 21001 --local-port 8001

EOF
  exit 1
}

# ------------- 参数解析 -------------
CONFIG_FILE=""
MODEL_KEY=""
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
    --model)
      MODEL_KEY="$2"
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
if [ -z "$CONFIG_FILE" ] || [ -z "$MODEL_KEY" ] || [ -z "$ROUTER_IP" ] || [ -z "$ROUTER_PORT" ]; then
  echo "错误: 缺少必需参数"
  usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

# ------------- 读取配置文件 -------------
echo "正在解析配置文件: $CONFIG_FILE"
echo "目标模型: $MODEL_KEY"

read_config() {
  python3 <<EOF
import yaml
import sys
import shlex

try:
    with open("$CONFIG_FILE", "r") as f:
        config = yaml.safe_load(f)
    
    # 检查配置文件是否为空
    if not config:
        print("错误: 配置文件为空", file=sys.stderr)
        sys.exit(1)
    
    # 检查指定的模型是否存在
    model_key = "$MODEL_KEY"
    if model_key not in config:
        available_models = ', '.join(config.keys())
        print(f"错误: 模型 '{model_key}' 不存在于配置文件中", file=sys.stderr)
        print(f"可用的模型: {available_models}", file=sys.stderr)
        sys.exit(1)
    
    # 获取指定模型的配置
    model_config = config[model_key]
    
    # 使用模型名称（如果有覆盖则使用覆盖的名称）
    model_name = "$MODEL_NAME_OVERRIDE" if "$MODEL_NAME_OVERRIDE" else model_key
    
    # 输出基础配置
    print(f"MODEL_NAME={model_name}")
    print(f"TP={model_config.get('tp', 1)}")
    print(f"DP={model_config.get('dp', 1)}")
    print(f"PP={model_config.get('pp', 1)}")
    print(f"MODEL_PATH={model_config['model_path']}")
    print(f"MAX_MODEL_LEN={model_config.get('max_model_len', '')}")
    
    # 处理 vllm 额外参数
    extra_args = model_config.get('vllm_extra_args', {})
    vllm_args_list = []
    
    for key, value in extra_args.items():
        # 将下划线转为破折号（Python dict key 风格 -> CLI 参数风格）
        cli_key = key.replace('_', '-')
        
        # 处理不同类型的值
        if isinstance(value, bool):
            if value:  # 只在为 True 时添加参数
                vllm_args_list.append(f"--{cli_key}")
        elif isinstance(value, (int, float)):
            vllm_args_list.append(f"--{cli_key} {value}")
        elif isinstance(value, str):
            # 字符串需要适当转义
            escaped_value = shlex.quote(value)
            vllm_args_list.append(f"--{cli_key} {escaped_value}")
        else:
            print(f"警告: 跳过不支持的参数类型: {key}={value}", file=sys.stderr)
    
    # 输出额外参数（用空格连接），整个字符串需要用引号括起来供 bash eval 使用
    vllm_extra_args_str = ' '.join(vllm_args_list)
    # 使用 shlex.quote 确保整个字符串在 bash eval 中被正确解析
    print(f"VLLM_EXTRA_ARGS={shlex.quote(vllm_extra_args_str)}")
    
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
echo "         VLLM 服务启动配置"
echo "=========================================="
echo "模型名称:     $MODEL_NAME"
echo "模型路径:     $MODEL_PATH"
echo "张量并行(TP): $TP"
echo "数据并行(DP): $DP"
echo "流水线并行:   $PP"
if [ -n "$MAX_MODEL_LEN" ]; then
  echo "最大长度:     $MAX_MODEL_LEN"
else
  echo "最大长度:     使用模型默认值"
fi
echo "本地端口:     $LOCAL_PORT"
echo "Router地址:   $ROUTER_IP:$ROUTER_PORT"
if [ -n "$VLLM_EXTRA_ARGS" ]; then
  echo "额外参数:     $VLLM_EXTRA_ARGS"
fi
echo "=========================================="
echo ""

# ------------- 检查模型路径 -------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "错误: 模型路径不存在: $MODEL_PATH"
  exit 1
fi

# ------------- 环境配置 -------------
echo "检查 Python 环境..."
# VLLM 通常不需要特定的 conda 环境，但如果需要可以取消注释下面的行
# source /mnt/shared-storage-user/ailab-hx/wulianyi/miniconda3/etc/profile.d/conda.sh
# conda activate vllm_env

# ------------- 计算实例数和 GPU 分配 -------------
GPUS_PER_INSTANCE=$((TP * DP * PP))
TOTAL_GPUS=8

echo "GPU 分配计算:"
echo "  TP=$TP, DP=$DP, PP=$PP"
echo "  每个实例需要: $GPUS_PER_INSTANCE 个GPU"
echo "  总GPU数: $TOTAL_GPUS"

# 检查是否能整除
if [ $((TOTAL_GPUS % GPUS_PER_INSTANCE)) -ne 0 ]; then
  echo ""
  echo "=========================================="
  echo "错误: 配置不合理！"
  echo "=========================================="
  echo "总GPU数($TOTAL_GPUS)不能被(TP*DP*PP=$GPUS_PER_INSTANCE)整除"
  echo "请调整配置使其满足: 8 % (TP*DP*PP) == 0"
  echo ""
  echo "合理的配置示例:"
  echo "  - TP=1, DP=1, PP=1  (启动8个实例)"
  echo "  - TP=2, DP=1, PP=1  (启动4个实例)"
  echo "  - TP=4, DP=1, PP=1  (启动2个实例)"
  echo "  - TP=8, DP=1, PP=1  (启动1个实例)"
  echo "=========================================="
  exit 1
fi

# 计算实例数
NUM_INSTANCES=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
echo "  将启动实例数: $NUM_INSTANCES"
echo ""

# ------------- 检查端口占用 -------------
echo "检查端口占用情况..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  PORT=$((LOCAL_PORT + i))
  if lsof -i :$PORT >/dev/null 2>&1; then
    echo "警告: 端口 $PORT 被占用，尝试释放..."
    lsof -t -i :$PORT | xargs -r kill -9 || true
  fi
done
if [ $NUM_INSTANCES -gt 1 ]; then
  echo "等待端口释放..."
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
  local INSTANCE_PORT=$1
  local INSTANCE_ID=$2
  
  (
    set -euo pipefail
    LOG="/tmp/vllm_register_${INSTANCE_PORT}.log"
    echo "[$(date '+%F %T')] 启动注册后台任务 [实例 $INSTANCE_ID]..." > "$LOG"
    echo "[$(date '+%F %T')] Router 地址: http://${ROUTER_IP}:${ROUTER_PORT}" >> "$LOG"
    
    # 等待健康检查: 最多 120 次，每次 30s，总计 ~60 分钟
    RETRIES=120
    INTERVAL=30
    
    echo "[$(date '+%F %T')] 开始健康检查轮询..." >> "$LOG"
    
    for ((i=1; i<=RETRIES; i++)); do
      # VLLM 的健康检查端点
      HEALTH_CMD="curl -s --connect-timeout 5 http://localhost:$INSTANCE_PORT/health"
      MODELS_CMD="curl -s --connect-timeout 5 http://localhost:$INSTANCE_PORT/v1/models"
      echo "[$(date '+%F %T')] 执行健康检查: $HEALTH_CMD" >> "$LOG"
      
      if $HEALTH_CMD >/dev/null 2>&1 || $MODELS_CMD >/dev/null 2>&1; then
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
    
    WORKER_URL="http://${IP}:${INSTANCE_PORT}"
    ROUTER_URL="http://${ROUTER_IP}:${ROUTER_PORT}"
    
    echo "[$(date '+%F %T')] 尝试注册 Worker [实例 $INSTANCE_ID]: ${WORKER_URL}" >> "$LOG"
    
    # 尝试注册到 Router
    REGISTER_CMD="curl -s -X POST \"${ROUTER_URL}/add_worker?url=${WORKER_URL}\""
    echo "[$(date '+%F %T')] 执行注册命令: $REGISTER_CMD" >> "$LOG"
    
    if curl -s -X POST "${ROUTER_URL}/add_worker?url=${WORKER_URL}" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] ✓ 成功注册到 Router!" >> "$LOG"
      echo "[$(date '+%F %T')]   Worker URL: ${WORKER_URL}" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "=========================================="
      echo "✓ 服务已成功注册到 Router! [实例 $INSTANCE_ID]"
      echo "  Worker: ${WORKER_URL}"
      echo "  Router: ${ROUTER_URL}"
      echo "=========================================="
    else
      echo "[$(date '+%F %T')] ⚠ 注册失败，请检查 Router 是否正常运行" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "⚠ 警告: 注册到 Router 失败 [实例 $INSTANCE_ID]，请检查 ${ROUTER_URL}"
    fi
  ) &
  
  local pid=$!
  echo $pid > /tmp/vllm_register_${INSTANCE_PORT}.pid
}

# ------------- 启动 VLLM 服务实例 -------------
echo "=========================================="
echo "启动 $NUM_INSTANCES 个 VLLM 服务实例"
echo "每个实例使用 $GPUS_PER_INSTANCE 个GPU"
echo "=========================================="
echo ""

# 构建 vllm serve 基础命令（不包含端口）
VLLM_CMD_BASE="vllm serve \"$MODEL_PATH\" \
  --served-model-name \"$MODEL_NAME\" \
  --tensor-parallel-size $TP \
  --data-parallel-size $DP \
  --pipeline-parallel-size $PP \
  --host 0.0.0.0"

# 如果配置了 max_model_len，则添加该参数
if [ -n "$MAX_MODEL_LEN" ]; then
  VLLM_CMD_BASE="$VLLM_CMD_BASE --max-model-len $MAX_MODEL_LEN"
fi

# 添加配置文件中的额外参数
if [ -n "$VLLM_EXTRA_ARGS" ]; then
  VLLM_CMD_BASE="$VLLM_CMD_BASE $VLLM_EXTRA_ARGS"
fi

# 启动多个实例
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  # 计算 GPU 范围
  GPU_START=$((i * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  
  # 构建 CUDA_VISIBLE_DEVICES
  if [ $GPUS_PER_INSTANCE -eq 1 ]; then
    GPU_IDS="$GPU_START"
  else
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)
  fi
  
  # 计算端口
  INSTANCE_PORT=$((LOCAL_PORT + i))
  
  # 日志文件
  LOG_FILE="/tmp/vllm_${INSTANCE_PORT}.log"
  
  echo "----------------------------------------"
  echo "启动实例 $i:"
  echo "  GPU: $GPU_IDS"
  echo "  端口: $INSTANCE_PORT"
  echo "  日志: $LOG_FILE"
  echo "----------------------------------------"
  
  # 构建完整命令（添加端口）
  VLLM_CMD="$VLLM_CMD_BASE --port $INSTANCE_PORT"
  
  # 打印完整命令
  echo "  完整命令: CUDA_VISIBLE_DEVICES=$GPU_IDS $VLLM_CMD"
  
  # 启动 vllm serve（后台运行,输出到日志）
  CUDA_VISIBLE_DEVICES=$GPU_IDS \
    nohup bash -c "$VLLM_CMD" > "$LOG_FILE" 2>&1 &
  # 记录 PID
  VLLM_PID=$!
  echo $VLLM_PID > "/tmp/vllm_${INSTANCE_PORT}.pid"
  echo "  PID: $VLLM_PID"
  
  # 启动注册任务
  register_to_router $INSTANCE_PORT $i
  echo "  注册任务已启动"
  echo ""
  
  # 短暂等待，避免启动过快
  sleep 2
done

echo "=========================================="
echo "✓ 所有实例已启动！"
echo "=========================================="
echo ""
echo "实例信息:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  INSTANCE_PORT=$((LOCAL_PORT + i))
  GPU_START=$((i * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  if [ $GPUS_PER_INSTANCE -eq 1 ]; then
    GPU_IDS="$GPU_START"
  else
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)
  fi
  echo "  实例 $i: 端口 $INSTANCE_PORT, GPU $GPU_IDS"
  echo "    日志: /tmp/vllm_${INSTANCE_PORT}.log"
  echo "    注册日志: /tmp/vllm_register_${INSTANCE_PORT}.log"
done

echo ""
echo "=========================================="
echo "查看第一个实例的日志 (按 Ctrl+C 退出日志查看)..."
echo "注意: 退出日志查看不会停止服务"
echo "=========================================="
echo ""

# 等待第一个实例的日志文件创建
FIRST_LOG="/tmp/vllm_${LOCAL_PORT}.log"
for i in {1..30}; do
  if [ -f "$FIRST_LOG" ]; then
    break
  fi
  sleep 1
done

# tail -f 第一个实例的日志
if [ -f "$FIRST_LOG" ]; then
  tail -f "$FIRST_LOG"
else
  echo "警告: 第一个实例日志文件未创建: $FIRST_LOG"
  echo "进入等待模式..."
  sleep infinity
fi

