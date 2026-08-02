#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
BACKEND=${BACKEND:?BACKEND is required}
GPU=${GPU:?GPU is required}
WAIT_FOR_FILE=${WAIT_FOR_FILE:?WAIT_FOR_FILE is required}
POLL_SECONDS=${POLL_SECONDS:-30}

gpu_idle() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, -v wanted="$GPU" '
      $1 + 0 == wanted + 0 {
        seen=1; memory=$2+0; utilization=$3+0
        if (memory <= 100 && utilization <= 10) ok=1
      }
      END { exit !(seen && ok) }
    '
}

while [[ ! -s "$WAIT_FOR_FILE" ]]; do
  sleep "$POLL_SECONDS"
done

stable=0
while (( stable < 3 )); do
  if gpu_idle; then stable=$((stable + 1)); else stable=0; fi
  (( stable == 3 )) || sleep "$POLL_SECONDS"
done

exec env REPO="$REPO" FORMAL_ROOT="$FORMAL_ROOT" BACKEND="$BACKEND" GPU="$GPU" \
  bash "$REPO/scripts/stage6_codriving_backend_tail_queue_v1.sh"
