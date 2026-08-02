#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
BACKEND=${BACKEND:?BACKEND must be tvm or trt}
ARM=${ARM:?ARM must be compression_only or compress_then_tune}
BATCH=${BATCH:?BATCH is required, for example 02}
GPU=${GPU:?GPU is required}
TRIALS=${TRIALS:-}

case "$BACKEND" in
  tvm) TASK=S5-COD-TVM ;;
  trt) TASK=S5-COD-TRT ;;
  *) echo "BACKEND must be tvm or trt" >&2; exit 2 ;;
esac
case "$ARM" in
  compression_only) : "${TRIALS:=0}" ;;
  compress_then_tune) : "${TRIALS:=64}" ;;
  *) echo "ARM must be compression_only or compress_then_tune" >&2; exit 2 ;;
esac
[[ "$BATCH" =~ ^[0-9][0-9]$ ]] || { echo "BATCH must be two digits" >&2; exit 2; }

cd "$REPO"
root="$FORMAL_ROOT/$BACKEND/$ARM/formal_batch_$BATCH"
round="$root/$TASK/round_$BATCH"
request="$FORMAL_ROOT/$BACKEND/$ARM/batch_$BATCH/measurement_request.json"
terminal="$round/final/stage6_fixed_batch_terminal.json"
mkdir -p "$round" "$FORMAL_ROOT/controller/batch_locks"
exec 9>"$FORMAL_ROOT/controller/batch_locks/${BACKEND}_${ARM}_${BATCH}.lock"
flock 9
[[ -s "$terminal" ]] && exit 0
[[ -s "$request" ]] || { echo "missing frozen request: $request" >&2; exit 2; }
cp "$request" "$round/measurement_request.json"
env REPO="$REPO" ROOT="$root" TASK="$TASK" ROUND_INDEX=$((10#$BATCH)) \
  GPU_POOL="$GPU" FEEDBACK_CONTRACT=actual_v3 FIXED_BATCH_MODE=1 \
  TVM_FP16_MAX_TRIALS="$TRIALS" \
  bash scripts/stage5_task_round_controller_v3.sh
[[ -s "$terminal" ]] || { echo "batch terminal missing: $terminal" >&2; exit 1; }
