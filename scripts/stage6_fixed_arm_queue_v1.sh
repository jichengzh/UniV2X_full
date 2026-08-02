#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
BACKEND=${BACKEND:?BACKEND must be tvm or trt}
GPU=${GPU:?GPU is required}
WAIT_PID=${WAIT_PID:-}

case "$BACKEND" in
  tvm) TASK=S5-PYR-TVM ;;
  trt) TASK=S5-PYR-TRT ;;
  *) echo "BACKEND must be tvm or trt" >&2; exit 2 ;;
esac

cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller"
QUEUE_DONE="$FORMAL_ROOT/controller/${BACKEND}_fixed_arm_queue.done"
QUEUE_FAILED="$FORMAL_ROOT/controller/${BACKEND}_fixed_arm_queue.failed"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$QUEUE_FAILED"; fi' EXIT
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
  first_terminal="$FORMAL_ROOT/$BACKEND/compression_only/formal_batch_00/$TASK/round_00/final/stage6_fixed_batch_terminal.json"
  [[ -s "$first_terminal" ]] || {
    echo "first formal batch did not complete successfully: $first_terminal" >&2
    exit 1
  }
fi

run_batch() {
  local arm=$1 batch=$2 trials=$3
  local root="$FORMAL_ROOT/$BACKEND/$arm/formal_batch_$batch"
  local round="$root/$TASK/round_$batch"
  local request="$FORMAL_ROOT/$BACKEND/$arm/batch_$batch/measurement_request.json"
  local terminal="$round/final/stage6_fixed_batch_terminal.json"
  [[ -s "$terminal" ]] && return 0
  [[ -s "$request" ]] || { echo "missing request: $request" >&2; return 2; }
  mkdir -p "$round"
  cp "$request" "$round/measurement_request.json"
  env REPO="$REPO" ROOT="$root" TASK="$TASK" ROUND_INDEX=$((10#$batch)) \
    GPU_POOL="$GPU" FEEDBACK_CONTRACT=actual_v3 FIXED_BATCH_MODE=1 \
    TVM_FP16_MAX_TRIALS="$trials" \
    bash scripts/stage5_task_round_controller_v3.sh
  [[ -s "$terminal" ]] || { echo "batch terminal missing: $terminal" >&2; return 1; }
}

for batch in 01 02 03; do
  run_batch compression_only "$batch" 0
done
run_batch compress_then_tune 00 64

date -Is >"$QUEUE_DONE"
rm -f "$QUEUE_FAILED"
trap - EXIT
