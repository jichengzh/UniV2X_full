#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
BACKEND=${BACKEND:?BACKEND is required}
ARM=${ARM:?ARM is required}
BATCH=${BATCH:?BATCH is required}
GPU=${GPU:?GPU is required}
WAIT_FOR_FILES=${WAIT_FOR_FILES:-}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
POLL_SECONDS=${POLL_SECONDS:-30}

case "$BACKEND" in
  tvm) TASK=S5-COD-TVM ;;
  trt) TASK=S5-COD-TRT ;;
  *) echo "BACKEND must be tvm or trt" >&2; exit 2 ;;
esac

terminal="$FORMAL_ROOT/$BACKEND/$ARM/formal_batch_$BATCH/$TASK/round_$BATCH/final/stage6_fixed_batch_terminal.json"

wait_dependencies() {
  local path
  IFS=',' read -r -a paths <<<"$WAIT_FOR_FILES"
  for path in "${paths[@]}"; do
    [[ -n "$path" ]] || continue
    while [[ ! -s "$path" ]]; do sleep "$POLL_SECONDS"; done
  done
}

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

wait_stable_idle() {
  local stable=0
  while (( stable < 3 )); do
    if gpu_idle; then stable=$((stable + 1)); else stable=0; fi
    (( stable == 3 )) || sleep "$POLL_SECONDS"
  done
}

wait_dependencies
[[ -s "$terminal" ]] && exit 0
for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
  wait_stable_idle
  echo "$(date -Is) attempt=$attempt backend=$BACKEND arm=$ARM batch=$BATCH gpu=$GPU"
  if env REPO="$REPO" FORMAL_ROOT="$FORMAL_ROOT" BACKEND="$BACKEND" ARM="$ARM" \
      BATCH="$BATCH" GPU="$GPU" bash "$REPO/scripts/stage6_codriving_run_fixed_batch_v1.sh"; then
    [[ -s "$terminal" ]] || { echo "terminal missing after successful runner" >&2; exit 1; }
    exit 0
  fi
  [[ -s "$terminal" ]] && exit 0
  sleep "$POLL_SECONDS"
done
echo "retry budget exhausted: $BACKEND $ARM batch=$BATCH gpu=$GPU" >&2
exit 1
