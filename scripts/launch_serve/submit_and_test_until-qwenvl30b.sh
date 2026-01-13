#!/bin/bash
set -euo pipefail

# ============================================================================
# Full workflow example: start services via rjob -> run tasks
# ============================================================================

echo "=========================================="
echo ""

# ------------- Config -------------
ROUTER_IP="YOUR_ROUTER_IP"
# ROUTER_PORT="21001"
# MODEL_NAME="qwen3_vl_235b_a22b_thinking"

ROUTER_PORT="21002"
MODEL_NAME="qwen3_vl_30b_a3b_thinking"

# ROUTER_PORT="21003"
# MODEL_NAME="qwen25_32b_instruct"

# ⭐ Important: NUM_INSTANCES meaning has changed!
# - For TP=8 large models (e.g., 235B): NUM_INSTANCES=8 (8 rjob tasks, 1 vLLM each)
# - For TP=1 small models (e.g., 30B):  NUM_INSTANCES=1 (1 rjob task, auto starts 8 vLLM)
NUM_INSTANCES=1

# Optional: rjob namespace/charged group (can be overridden via env vars)
RJOB_NAMESPACE="${RJOB_NAMESPACE:-}"
RJOB_CHARGED_GROUP="${RJOB_CHARGED_GROUP:-}"

# ------------- Parse args (rjob options only) -------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --namespace)
      RJOB_NAMESPACE="$2"
      shift 2
      ;;
    --charged-group)
      RJOB_CHARGED_GROUP="$2"
      shift 2
      ;;
    *)
      echo "Error: unknown argument $1"
      echo "Usage: $0 [--namespace NS] [--charged-group GRP]"
      exit 1
      ;;
  esac
done

export RJOB_NAMESPACE RJOB_CHARGED_GROUP

RJOB_ARGS=()
if [ -n "${RJOB_NAMESPACE}" ]; then
  RJOB_ARGS+=(--namespace "${RJOB_NAMESPACE}")
fi
if [ -n "${RJOB_CHARGED_GROUP}" ]; then
  RJOB_ARGS+=(--charged-group "${RJOB_CHARGED_GROUP}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Step 1: submit ${NUM_INSTANCES} model service tasks via rjob..."
echo ""

# Submit rjob tasks (blocks until all services are ready)
bash "${SCRIPT_DIR}/submit_rjob_instances.sh" \
  -n ${NUM_INSTANCES} \
  --config model_config_example.yaml \
  --model ${MODEL_NAME} \
  --router-ip ${ROUTER_IP} \
  --router-port ${ROUTER_PORT} \
  "${RJOB_ARGS[@]}"

# If the command returned successfully, all services are started and registered.
echo ""
echo "=========================================="
echo "Step 2: run tasks that use the services..."
echo "=========================================="
echo ""

# Now it's safe to run tasks that depend on these services.
echo "Services are ready. Starting tasks..."
echo "Router URL: http://${ROUTER_IP}:${ROUTER_PORT}"
echo ""

# Example: test service availability
echo "Testing service connectivity..."
if curl -sf "http://${ROUTER_IP}:${ROUTER_PORT}/health" >/dev/null; then
  echo "✓ Router reachable"
  echo ""
  echo "Health status:"
  curl -s "http://${ROUTER_IP}:${ROUTER_PORT}/health"
  echo ""
  echo ""
  echo "Available models:"
  curl -s "http://${ROUTER_IP}:${ROUTER_PORT}/v1/models" | python3 -m json.tool || true
else
  echo "✗ Router connection failed"
  exit 1
fi

echo "=========================================="
echo "✓ Launch complete!"
echo "=========================================="
echo ""
echo "Note: clean up rjob tasks after completion"
echo "  View tasks: cat /tmp/vllm_rjobs_${ROUTER_PORT}.txt"
echo "  Stop tasks: use rjob stop <job-name> (see names in the file above)"
