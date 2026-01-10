#!/bin/bash
set -euo pipefail

# ============================================================================
# Submit multiple model service instances via rjob
# Each instance runs on a separate GPU node
# ============================================================================

# ------------- Usage -------------
usage() {
  cat <<EOF
Usage: $0 -n <NUM_INSTANCES> --config <CONFIG_FILE> --model <MODEL_NAME> --router-ip <IP> --router-port <PORT> [options]

Required:
  -n NUM                Number of instances (submit N rjob tasks)
  --config FILE         Model config file (YAML)
  --model NAME          Model name to launch
  --router-ip IP        SGLang Router IP
  --router-port PORT    SGLang Router port

Optional:
  --start-port PORT     Start port (default: 8000)
  --namespace NS        rjob namespace (default: \$RJOB_NAMESPACE or empty)
  --charged-group GRP   Charged group (default: \$RJOB_CHARGED_GROUP or puyullm_gpu)
  --image IMG           Docker image
  --help                Show this help

Example:
  $0 -n 4 --config model_config_example.yaml --model qwen3_vl_235b_a22b_thinking --router-ip 100.102.249.23 --router-port 21001

EOF
  exit 1
}

# ------------- Argument parsing -------------
NUM_INSTANCES=""
CONFIG_FILE=""
MODEL_NAME=""
ROUTER_IP=""
ROUTER_PORT=""
START_PORT=8000
NAMESPACE="${RJOB_NAMESPACE:-}"
CHARGED_GROUP="${RJOB_CHARGED_GROUP:-puyullm_gpu}"
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
      echo "Error: unknown argument $1"
      usage
      ;;
  esac
done

# ------------- Validate required args -------------
if [ -z "$NUM_INSTANCES" ] || [ -z "$CONFIG_FILE" ] || [ -z "$MODEL_NAME" ] || [ -z "$ROUTER_IP" ] || [ -z "$ROUTER_PORT" ]; then
  echo "Error: missing required arguments"
  usage
fi

if ! [[ "$NUM_INSTANCES" =~ ^[0-9]+$ ]] || [ "$NUM_INSTANCES" -lt 1 ]; then
  echo "Error: NUM_INSTANCES must be a positive integer"
  exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check config file
if [ ! -f "${SCRIPT_DIR}/${CONFIG_FILE}" ]; then
  echo "Error: config file not found: ${CONFIG_FILE}"
  exit 1
fi

# Check rjob command
if ! command -v rjob &> /dev/null; then
  echo "Error: rjob command not available; ensure correct environment"
  exit 1
fi

# ------------- Read model config to compute instances -------------
echo "Reading model config..."
read_model_config() {
  python3 <<EOF
import yaml
import sys

try:
    with open("${SCRIPT_DIR}/${CONFIG_FILE}", "r") as f:
        config = yaml.safe_load(f)
    
    if "${MODEL_NAME}" not in config:
        print("Error: model not found in config file", file=sys.stderr)
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
    print(f"Error: failed to parse config file: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

eval $(read_model_config)

# Compute expected total instances (for health check)
EXPECTED_TOTAL_INSTANCES=$((NUM_INSTANCES * INSTANCES_PER_JOB))

# ------------- Show configuration -------------
echo ""
echo "=========================================="
echo "Submit rjob tasks"
echo "=========================================="
echo "Model:        ${MODEL_NAME}"
echo "Model config: TP=${TP}, DP=${DP}, PP=${PP}"
echo "GPUs/task:    ${GPUS_PER_INSTANCE}"
echo "rjob tasks:   ${NUM_INSTANCES}"
echo "Inst/task:    ${INSTANCES_PER_JOB}"
echo "Total inst:   ${EXPECTED_TOTAL_INSTANCES}"
echo "Start port:   ${START_PORT}"
echo "Router:       ${ROUTER_IP}:${ROUTER_PORT}"
echo "Namespace:    ${NAMESPACE}"
echo "Charged grp:  ${CHARGED_GROUP}"
echo "=========================================="
echo ""

# ------------- Submit rjob tasks -------------
SUBMITTED_JOBS=()
SUBMITTED_PORTS=()

for ((i=0; i<NUM_INSTANCES; i++)); do
  # PORT=$((START_PORT + i))
  PORT=$START_PORT
  TIMESTAMP=$(date +%H%M%S)
  # Truncate model name to keep total length <= 64 chars
  MODEL_NAME_SHORT="${MODEL_NAME:0:50}"
  JOB_NAME="vllm-${MODEL_NAME_SHORT}-${TIMESTAMP}"
  
  echo "[$((i+1))/${NUM_INSTANCES}] Submitting task: ${JOB_NAME} (port ${PORT})..."
  
  # Build startup command
 STARTUP_CMD="cd /mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/scripts/launch_serve && \
bash start_vllm_service.sh \
  --config ${CONFIG_FILE} \
  --model ${MODEL_NAME} \
  --router-ip ${ROUTER_IP} \
  --router-port ${ROUTER_PORT} \
  --local-port ${PORT}"

  echo $STARTUP_CMD

  # Submit rjob task
  # Build base command
  RJOB_CMD="rjob submit \
    -e DISTRIBUTED_JOB=true \
    -e NCCL_DEBUG_SUBSYS=ALL \
    --image=\"${IMAGE}\""
  
  # Add --namespace only when NAMESPACE is non-empty
  if [ -n "$NAMESPACE" ]; then
    RJOB_CMD="$RJOB_CMD \
    --namespace \"${NAMESPACE}\""
  fi
  
  RJOB_CMD="$RJOB_CMD \
    --charged-group \"${CHARGED_GROUP}\" \
    --host-network=true \
    --name \"${JOB_NAME}\" \
    -P 1 \
    --gpu 8 \
    --cpu 80 \
    --memory 800000 \
    --private-machine='group' \
    --gang-start=true \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/ailab-hx:/mnt/shared-storage-user/ailab-hx \
    --mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
    --custom-resources rdma/mlnx_shared=8 \
    --mount=gpfs://gpfs2/intern-pretrain-shared02:/mnt/shared-storage-user/intern-pretrain-shared02 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --enable-sshd \
    -- bash -c \"${STARTUP_CMD}\""
  
  echo $RJOB_CMD
  
  if SUBMIT_OUTPUT=$(eval "$RJOB_CMD" 2>&1); then
    echo "  ✓ Task submitted: ${JOB_NAME}"
    SUBMITTED_JOBS+=("${JOB_NAME}")
    SUBMITTED_PORTS+=("${PORT}")
  else
    status=$?
    echo "  ✗ Task submission failed (exit code ${status})"
    echo "${SUBMIT_OUTPUT}"
  fi
  
  # Brief wait to avoid name collisions
  sleep 2
done

# ------------- Submission summary -------------
echo ""
echo "=========================================="
if [ ${#SUBMITTED_JOBS[@]} -eq 0 ]; then
  echo "✗ All tasks failed to submit!"
  echo "=========================================="
  exit 1
else
  echo "✓ Submitted ${#SUBMITTED_JOBS[@]}/${NUM_INSTANCES} tasks"
fi
echo "=========================================="
echo ""

echo "Submitted tasks:"
for ((i=0; i<${#SUBMITTED_JOBS[@]}; i++)); do
  echo "  - ${SUBMITTED_JOBS[$i]} (port ${SUBMITTED_PORTS[$i]})"
done
echo ""

# ------------- Wait for startup -------------
# Note: rjob list is unreliable; skip and poll router port directly
echo "=========================================="
echo "Skipping task status check (assuming tasks started)..."
echo "=========================================="
echo ""
echo "✓ Skipped rjob list check; moving to service health checks"
echo ""

# ------------- Wait for service health -------------
echo ""
echo "=========================================="
echo "Waiting for service health and registration..."
echo "=========================================="
echo ""

echo "Note: this may take 5-20 minutes (model load time)"
echo "Will keep checking SGLang Router until all services register..."
echo "Router URL: http://${ROUTER_IP}:${ROUTER_PORT}/list_workers"
echo "Timeout: 30 minutes"
echo ""

HEALTH_TIMEOUT=1800  # 30 minutes
START_TIME=$(date +%s)

# Registration checker - count registered workers
check_registration() {
  local workers=$(curl -sf --connect-timeout 3 "http://${ROUTER_IP}:${ROUTER_PORT}/list_workers" 2>/dev/null || echo "")
  if [ -z "$workers" ]; then
    echo 0
    return 0
  fi
  
  # Count registered instances (lines containing http)
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
    echo "⚠ Health check timed out (${HEALTH_TIMEOUT}s)"
    echo "Some services may still be starting"
    break
  fi
  
  # Check registration status
  REGISTERED_COUNT=$(check_registration)
  
  if [ "$REGISTERED_COUNT" -ge "$EXPECTED_TOTAL_INSTANCES" ]; then
    ALL_REGISTERED=true
    echo ""
    echo "✓ All services registered to Router!"
    echo "  Registered instances: ${REGISTERED_COUNT}"
    echo "  Elapsed: ${ELAPSED_MIN}m${ELAPSED_SEC}s"
    break
  fi
  
  # Show detailed progress
  printf "[Check #%d] Waiting for Router registration... (%d/%d registered, %dm%ds elapsed)\r" \
    "$CHECK_ITERATION" "$REGISTERED_COUNT" "$EXPECTED_TOTAL_INSTANCES" "$ELAPSED_MIN" "$ELAPSED_SEC"
  
  sleep 10
done

# ------------- Final result -------------
echo ""
echo "=========================================="
if [ "$ALL_REGISTERED" = true ]; then
  echo "✓✓✓ All services are ready! ✓✓✓"
  echo "Registered instances: ${REGISTERED_COUNT}"
else
  echo "⚠ Some services may not be ready"
  echo "Registered: ${REGISTERED_COUNT}/${EXPECTED_TOTAL_INSTANCES}"
fi
echo "=========================================="
echo ""

echo "Submitted tasks:"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "  - ${JOB}"
done
echo ""

echo "Router URL: http://${ROUTER_IP}:${ROUTER_PORT}"
echo ""

echo "Management commands:"
echo "  Task status: rjob list | grep vllm"
echo "  Task logs:   rjob logs <job-name>"
echo "  Stop all tasks:"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "    rjob stop ${JOB}"
done
echo ""

# Save task info to file
JOBS_FILE="/tmp/vllm_rjobs_${ROUTER_PORT}.txt"
echo "# vLLM rjob task list" > "${JOBS_FILE}"
echo "# Created at: $(date)" >> "${JOBS_FILE}"
echo "# Router: ${ROUTER_IP}:${ROUTER_PORT}" >> "${JOBS_FILE}"
for JOB in "${SUBMITTED_JOBS[@]}"; do
  echo "${JOB}" >> "${JOBS_FILE}"
done
echo ""
echo "Task info saved to: ${JOBS_FILE}"

echo "=========================================="

# Return status
if [ "$ALL_REGISTERED" = true ]; then
  exit 0
else
  exit 1
fi
