#!/bin/bash
set -euo pipefail

# ============================================================================
# vLLM service launcher
# Purpose: start vLLM and auto-register instances to SGLang Router
# ============================================================================

# ------------- Usage -------------
usage() {
  cat <<EOF
Usage: $0 --config <CONFIG> --model <MODEL_NAME> --router-ip <IP> --router-port <PORT> [options]

Required:
  --config FILE         Model config file (YAML, contains multiple models)
  --model NAME          Model name to launch (must exist in config)
  --router-ip IP        SGLang Router IP
  --router-port PORT    SGLang Router port

Optional:
  --local-port PORT     Local service port (default: 8000)
  --model-name NAME     Override model name (used for router registration)
  --help                Show this help

Config example (model_config.yaml):
  qwen3_vl_235b_instruct:
    tp: 8
    dp: 1
    max_model_len: 128000  # Optional: max model length
    model_path: "/path/to/model"
    vllm_extra_args:       # Optional: extra vLLM args
      gpu-memory-utilization: 0.85
      enable-expert-parallel: true  # For MoE models
      mm-encoder-tp-mode: data      # For VL models
  
  qwen2_72b_instruct:
    tp: 4
    dp: 1
    # If max_model_len not set, uses model default
    model_path: "/path/to/another/model"
    vllm_extra_args:       # Optional: extra vLLM args
      gpu-memory-utilization: 0.90
      max-num-seqs: 256

Examples:
  # Launch qwen3_vl_235b_instruct
  $0 --config model_config.yaml --model qwen3_vl_235b_instruct --router-ip 100.102.249.23 --router-port 21001
  
  # Launch qwen2_72b_instruct with custom port
  $0 --config model_config.yaml --model qwen2_72b_instruct --router-ip 100.102.249.23 --router-port 21001 --local-port 8001

EOF
  exit 1
}

# ------------- Argument parsing -------------
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
      echo "Error: unknown argument $1"
      usage
      ;;
  esac
done

# ------------- Validate required args -------------
if [ -z "$CONFIG_FILE" ] || [ -z "$MODEL_KEY" ] || [ -z "$ROUTER_IP" ] || [ -z "$ROUTER_PORT" ]; then
  echo "Error: missing required arguments"
  usage
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

# ------------- Read config file -------------
echo "Parsing config file: $CONFIG_FILE"
echo "Target model: $MODEL_KEY"

read_config() {
  python3 <<EOF
import yaml
import sys
import shlex

try:
    with open("$CONFIG_FILE", "r") as f:
        config = yaml.safe_load(f)
    
    # Check if config file is empty
    if not config:
        print("Error: config file is empty", file=sys.stderr)
        sys.exit(1)
    
    # Check that the requested model exists
    model_key = "$MODEL_KEY"
    if model_key not in config:
        available_models = ', '.join(config.keys())
        print(f"Error: model '{model_key}' not found in config", file=sys.stderr)
        print(f"Available models: {available_models}", file=sys.stderr)
        sys.exit(1)
    
    # Get model config
    model_config = config[model_key]
    
    # Use model name (override if provided)
    model_name = "$MODEL_NAME_OVERRIDE" if "$MODEL_NAME_OVERRIDE" else model_key
    
    # Emit base config
    print(f"MODEL_NAME={model_name}")
    print(f"TP={model_config.get('tp', 1)}")
    print(f"DP={model_config.get('dp', 1)}")
    print(f"PP={model_config.get('pp', 1)}")
    print(f"MODEL_PATH={model_config['model_path']}")
    print(f"MAX_MODEL_LEN={model_config.get('max_model_len', '')}")
    
    # Process extra vLLM args
    extra_args = model_config.get('vllm_extra_args', {})
    vllm_args_list = []
    
    for key, value in extra_args.items():
        # Replace underscores with dashes (dict keys -> CLI args)
        cli_key = key.replace('_', '-')
        
        # Handle different value types
        if isinstance(value, bool):
            if value:  # Only add when True
                vllm_args_list.append(f"--{cli_key}")
        elif isinstance(value, (int, float)):
            vllm_args_list.append(f"--{cli_key} {value}")
        elif isinstance(value, str):
            # Escape strings
            escaped_value = shlex.quote(value)
            vllm_args_list.append(f"--{cli_key} {escaped_value}")
        else:
            print(f"Warning: skipping unsupported param type: {key}={value}", file=sys.stderr)
    
    # Emit extra args (space-joined), quote for bash eval
    vllm_extra_args_str = ' '.join(vllm_args_list)
    # Use shlex.quote to keep the string safe for bash eval
    print(f"VLLM_EXTRA_ARGS={shlex.quote(vllm_extra_args_str)}")
    
except Exception as e:
    print(f"Error: failed to parse config file: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# Parse config and export variables
eval $(read_config)

# ------------- Show configuration -------------
echo ""
echo "=========================================="
echo "         vLLM Service Launch Config"
echo "=========================================="
echo "Model name:    $MODEL_NAME"
echo "Model path:    $MODEL_PATH"
echo "Tensor (TP):   $TP"
echo "Data (DP):     $DP"
echo "Pipeline (PP): $PP"
if [ -n "$MAX_MODEL_LEN" ]; then
  echo "Max length:    $MAX_MODEL_LEN"
else
  echo "Max length:    model default"
fi
echo "Local port:    $LOCAL_PORT"
echo "Router addr:   $ROUTER_IP:$ROUTER_PORT"
if [ -n "$VLLM_EXTRA_ARGS" ]; then
  echo "Extra args:    $VLLM_EXTRA_ARGS"
fi
echo "=========================================="
echo ""

# ------------- Check model path -------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: model path not found: $MODEL_PATH"
  exit 1
fi

# ------------- Environment setup -------------
echo "Checking Python environment..."
# vLLM usually does not require a specific conda env; uncomment if needed
# source /mnt/shared-storage-user/ailab-hx/wulianyi/miniconda3/etc/profile.d/conda.sh
# conda activate vllm_env

# ------------- Compute instances and GPU allocation -------------
GPUS_PER_INSTANCE=$((TP * DP * PP))
TOTAL_GPUS=8

echo "GPU allocation:"
echo "  TP=$TP, DP=$DP, PP=$PP"
echo "  GPUs per instance: $GPUS_PER_INSTANCE"
echo "  Total GPUs: $TOTAL_GPUS"

# Check divisibility
if [ $((TOTAL_GPUS % GPUS_PER_INSTANCE)) -ne 0 ]; then
  echo ""
  echo "=========================================="
  echo "Error: invalid configuration!"
  echo "=========================================="
  echo "Total GPUs ($TOTAL_GPUS) must be divisible by (TP*DP*PP=$GPUS_PER_INSTANCE)"
  echo "Adjust config so: 8 % (TP*DP*PP) == 0"
  echo ""
  echo "Valid examples:"
  echo "  - TP=1, DP=1, PP=1  (start 8 instances)"
  echo "  - TP=2, DP=1, PP=1  (start 4 instances)"
  echo "  - TP=4, DP=1, PP=1  (start 2 instances)"
  echo "  - TP=8, DP=1, PP=1  (start 1 instance)"
  echo "=========================================="
  exit 1
fi

# Compute instance count
NUM_INSTANCES=$((TOTAL_GPUS / GPUS_PER_INSTANCE))
echo "  Instances to start: $NUM_INSTANCES"
echo ""

# ------------- Check port usage -------------
echo "Checking port usage..."
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  PORT=$((LOCAL_PORT + i))
  if lsof -i :$PORT >/dev/null 2>&1; then
    echo "Warning: port $PORT is in use, trying to free it..."
    lsof -t -i :$PORT | xargs -r kill -9 || true
  fi
done
if [ $NUM_INSTANCES -gt 1 ]; then
  echo "Waiting for ports to be released..."
  sleep 3
fi

# ------------- Helper functions -------------
get_ip() {
  local ip
  ip=$(hostname -I 2>/dev/null | awk '{print $1}' | head -n1 || true)
  if [ -z "${ip}" ]; then
    ip=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+' || true)
  fi
  if [ -n "${SERVER_IP:-}" ]; then
    ip="$SERVER_IP"  # Allow manual override
  fi
  echo "$ip"
}

# ------------- Background registration to Router -------------
register_to_router() {
  local INSTANCE_PORT=$1
  local INSTANCE_ID=$2
  
  (
    set -euo pipefail
    LOG="/tmp/vllm_register_${INSTANCE_PORT}.log"
    echo "[$(date '+%F %T')] Starting registration task [instance $INSTANCE_ID]..." > "$LOG"
    echo "[$(date '+%F %T')] Router URL: http://${ROUTER_IP}:${ROUTER_PORT}" >> "$LOG"
    
    # Health check: up to 120 tries, 30s each (~60 minutes)
    RETRIES=120
    INTERVAL=30
    
    echo "[$(date '+%F %T')] Starting health check polling..." >> "$LOG"
    
    for ((i=1; i<=RETRIES; i++)); do
      # vLLM health endpoints
      HEALTH_CMD="curl -s --connect-timeout 5 http://localhost:$INSTANCE_PORT/health"
      MODELS_CMD="curl -s --connect-timeout 5 http://localhost:$INSTANCE_PORT/v1/models"
      echo "[$(date '+%F %T')] Running health check: $HEALTH_CMD" >> "$LOG"
      
      if $HEALTH_CMD >/dev/null 2>&1 || $MODELS_CMD >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] ✓ Health check passed!" >> "$LOG"
        break
      fi
      echo "[$(date '+%F %T')] Waiting for service... ($i/$RETRIES)" >> "$LOG"
      sleep "$INTERVAL"
      if [ $i -eq $RETRIES ]; then
        echo "[$(date '+%F %T')] ✗ Health check timed out, giving up registration" >> "$LOG"
        exit 0
      fi
    done
    
    # Get local IP
    IP="$(get_ip)"
    if [ -z "$IP" ]; then
      echo "[$(date '+%F %T')] ✗ Failed to get local IP, giving up registration" >> "$LOG"
      exit 0
    fi
    
    WORKER_URL="http://${IP}:${INSTANCE_PORT}"
    ROUTER_URL="http://${ROUTER_IP}:${ROUTER_PORT}"
    
    echo "[$(date '+%F %T')] Registering worker [instance $INSTANCE_ID]: ${WORKER_URL}" >> "$LOG"
    
    # Try to register with Router
    REGISTER_CMD="curl -s -X POST \"${ROUTER_URL}/add_worker?url=${WORKER_URL}\""
    echo "[$(date '+%F %T')] Run registration command: $REGISTER_CMD" >> "$LOG"
    
    if curl -s -X POST "${ROUTER_URL}/add_worker?url=${WORKER_URL}" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] ✓ Registered to Router!" >> "$LOG"
      echo "[$(date '+%F %T')]   Worker URL: ${WORKER_URL}" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "=========================================="
      echo "✓ Service registered to Router! [instance $INSTANCE_ID]"
      echo "  Worker: ${WORKER_URL}"
      echo "  Router: ${ROUTER_URL}"
      echo "=========================================="
    else
      echo "[$(date '+%F %T')] ⚠ Registration failed; check Router status" >> "$LOG"
      echo "[$(date '+%F %T')]   Router URL: ${ROUTER_URL}" >> "$LOG"
      echo ""
      echo "⚠ Warning: failed to register to Router [instance $INSTANCE_ID]; check ${ROUTER_URL}"
    fi
  ) &
  
  local pid=$!
  echo $pid > /tmp/vllm_register_${INSTANCE_PORT}.pid
}

# ------------- Start vLLM service instances -------------
echo "=========================================="
echo "Starting $NUM_INSTANCES vLLM service instances"
echo "GPUs per instance: $GPUS_PER_INSTANCE"
echo "=========================================="
echo ""

# Build base vllm serve command (without port)
VLLM_CMD_BASE="vllm serve \"$MODEL_PATH\" \
  --served-model-name \"$MODEL_NAME\" \
  --tensor-parallel-size $TP \
  --data-parallel-size $DP \
  --pipeline-parallel-size $PP \
  --host 0.0.0.0"

# Add max_model_len if configured
if [ -n "$MAX_MODEL_LEN" ]; then
  VLLM_CMD_BASE="$VLLM_CMD_BASE --max-model-len $MAX_MODEL_LEN"
fi

# Add extra args from config
if [ -n "$VLLM_EXTRA_ARGS" ]; then
  VLLM_CMD_BASE="$VLLM_CMD_BASE $VLLM_EXTRA_ARGS"
fi

# Start multiple instances
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  # Compute GPU range
  GPU_START=$((i * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  
  # Build CUDA_VISIBLE_DEVICES
  if [ $GPUS_PER_INSTANCE -eq 1 ]; then
    GPU_IDS="$GPU_START"
  else
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)
  fi
  
  # Compute port
  INSTANCE_PORT=$((LOCAL_PORT + i))
  
  # Log file
  LOG_FILE="/tmp/vllm_${INSTANCE_PORT}.log"
  
  echo "----------------------------------------"
  echo "Starting instance $i:"
  echo "  GPU: $GPU_IDS"
  echo "  Port: $INSTANCE_PORT"
  echo "  Log:  $LOG_FILE"
  echo "----------------------------------------"
  
  # Build full command (add port)
  VLLM_CMD="$VLLM_CMD_BASE --port $INSTANCE_PORT"
  
  # Print full command
  echo "  Full command: CUDA_VISIBLE_DEVICES=$GPU_IDS $VLLM_CMD"
  
  # Start vllm serve (background, log to file)
  CUDA_VISIBLE_DEVICES=$GPU_IDS \
    nohup bash -c "$VLLM_CMD" > "$LOG_FILE" 2>&1 &
  # Record PID
  VLLM_PID=$!
  echo $VLLM_PID > "/tmp/vllm_${INSTANCE_PORT}.pid"
  echo "  PID: $VLLM_PID"
  
  # Start registration task
  register_to_router $INSTANCE_PORT $i
  echo "  Registration task started"
  echo ""
  
  # Brief sleep to avoid starting too fast
  sleep 2
done

echo "=========================================="
echo "✓ All instances started!"
echo "=========================================="
echo ""
echo "Instance info:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  INSTANCE_PORT=$((LOCAL_PORT + i))
  GPU_START=$((i * GPUS_PER_INSTANCE))
  GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
  if [ $GPUS_PER_INSTANCE -eq 1 ]; then
    GPU_IDS="$GPU_START"
  else
    GPU_IDS=$(seq -s, $GPU_START $GPU_END)
  fi
  echo "  Instance $i: port $INSTANCE_PORT, GPU $GPU_IDS"
  echo "    Log: /tmp/vllm_${INSTANCE_PORT}.log"
  echo "    Register log: /tmp/vllm_register_${INSTANCE_PORT}.log"
done

echo ""
echo "=========================================="
echo "Tailing the first instance log (Ctrl+C to stop tail)..."
echo "Note: stopping tail does not stop the service"
echo "=========================================="
echo ""

# Wait for the first instance log file
FIRST_LOG="/tmp/vllm_${LOCAL_PORT}.log"
for i in {1..30}; do
  if [ -f "$FIRST_LOG" ]; then
    break
  fi
  sleep 1
done

# tail -f the first instance log
if [ -f "$FIRST_LOG" ]; then
  tail -f "$FIRST_LOG"
else
  echo "Warning: first instance log not created: $FIRST_LOG"
  echo "Entering wait mode..."
  sleep infinity
fi
