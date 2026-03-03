#!/bin/bash
set -euo pipefail

# ============================================================================
# Join a new machine to an existing Ray cluster as a CPU worker
# Intended to run on the newly created machine (e.g. via rjob submit --bash)
# ============================================================================

usage() {
  cat <<EOF
Usage: $0 --head-ip <HEAD_IP> [options]

Required:
  - none if RAY_HEAD_IP is set or a head-ip file exists

Optional:
  --head-ip IP        Ray head node IP (or set env RAY_HEAD_IP)
  --head-ip-file FILE Read head IP from file (default: scripts/launch_serve/.ray_head_ip)
  --head-port PORT    Ray head port (default: 6379, or env RAY_HEAD_PORT)
  --num-cpus N        CPUs to register (default: nproc, or env RAY_WORKER_CPUS)
  --no-wait           Exit after joining (default keeps process alive)
  --help              Show this help

Examples:
  # On head machine, capture head IP first and write it to shared storage:
  #   hostname -i | awk '{print \$1}' > scripts/launch_serve/.ray_head_ip
  # Then worker jobs can run this script directly without extra args.

  bash $0 --head-ip 10.0.0.1
  bash $0 --head-ip 10.0.0.1 --head-port 6379 --num-cpus 64
EOF
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEAD_IP="${RAY_HEAD_IP:-}"
HEAD_IP_FILE="${RAY_HEAD_IP_FILE:-${SCRIPT_DIR}/.ray_head_ip}"
HEAD_PORT="${RAY_HEAD_PORT:-6379}"
NUM_CPUS="${RAY_WORKER_CPUS:-$(nproc)}"
KEEP_ALIVE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --head-ip)
      HEAD_IP="$2"
      shift 2
      ;;
    --head-ip-file)
      HEAD_IP_FILE="$2"
      shift 2
      ;;
    --head-port)
      HEAD_PORT="$2"
      shift 2
      ;;
    --num-cpus)
      NUM_CPUS="$2"
      shift 2
      ;;
    --no-wait)
      KEEP_ALIVE=0
      shift 1
      ;;
    --help|-h)
      usage
      ;;
    *)
      echo "Error: unknown argument $1"
      usage
      ;;
  esac
done

if [ -z "$HEAD_IP" ] && [ -f "$HEAD_IP_FILE" ]; then
  HEAD_IP="$(awk 'NF {print $1; exit}' "$HEAD_IP_FILE")"
fi

if [ -z "$HEAD_IP" ]; then
  echo "Error: head IP is required (--head-ip or RAY_HEAD_IP)."
  echo "Tip 1: export RAY_HEAD_IP=\$(hostname -i | awk '{print \$1}') on head machine."
  echo "Tip 2: write shared head-ip file:"
  echo "       hostname -i | awk '{print \$1}' > ${HEAD_IP_FILE}"
  exit 1
fi

WORKER_IP="$(hostname -i | awk '{print $1}')"
MEMORY_GB="$(awk '/MemTotal/ {printf "%.1f", $2 / 1024 / 1024}' /proc/meminfo)"

echo "=========================================="
echo "Joining Ray cluster as CPU worker"
echo "=========================================="
echo "Head:      ${HEAD_IP}:${HEAD_PORT}"
echo "Worker IP: ${WORKER_IP}"
echo "CPUs:      ${NUM_CPUS}"
echo "Memory:    ${MEMORY_GB} GB"
echo "=========================================="

echo "Stopping any existing local Ray processes..."
ray stop --force >/dev/null 2>&1 || true

echo "Starting Ray worker and joining cluster..."
ray start \
  --address "${HEAD_IP}:${HEAD_PORT}" \
  --num-cpus "${NUM_CPUS}" \
  --disable-usage-stats

echo "[OK] Ray worker joined. Cluster status:"
ray status || true

if [ "$KEEP_ALIVE" -eq 1 ]; then
  echo ""
  echo "Worker is now contributing CPU/memory resources."
  echo "Keeping this job alive; Ctrl+C or SIGTERM will stop Ray worker."

  cleanup() {
    echo ""
    echo "Stopping Ray worker..."
    ray stop --force >/dev/null 2>&1 || true
    exit 0
  }

  trap cleanup INT TERM
  while true; do
    sleep 300
  done
fi
