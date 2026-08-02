#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
TRT_CONTROLLER_PID=${TRT_CONTROLLER_PID:?TRT_CONTROLLER_PID is required}
TVM_CONTROLLER_PID=${TVM_CONTROLLER_PID:?TVM_CONTROLLER_PID is required}
ACTIVE_AP_PIDS=${ACTIVE_AP_PIDS:?ACTIVE_AP_PIDS is required}
TRT_FIXED_GPUS=${TRT_FIXED_GPUS:-3,6}
TVM_FIXED_GPUS=${TVM_FIXED_GPUS:-4,7}
TRT_TAIL_GPU=${TRT_TAIL_GPU:-6}
TVM_TAIL_GPU=${TVM_TAIL_GPU:-7}
POLL_SECONDS=${POLL_SECONDS:-20}

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]
cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller"
SUPERVISOR_DONE="$FORMAL_ROOT/controller/stage6_resume_parallel_queues_v1.done"
SUPERVISOR_FAILED="$FORMAL_ROOT/controller/stage6_resume_parallel_queues_v1.failed"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$SUPERVISOR_FAILED"; fi' EXIT

wait_for_exit() {
  local csv=$1 pid running state
  IFS=',' read -r -a pids <<<"$csv"
  while true; do
    running=0
    for pid in "${pids[@]}"; do
      [[ "$pid" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid PID in wait set: $pid" >&2
        return 2
      }
      if kill -0 "$pid" 2>/dev/null; then
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
        [[ "$state" == "Z" ]] || running=1
      fi
    done
    (( running == 0 )) && return 0
    sleep "$POLL_SECONDS"
  done
}

wait_for_terminal() {
  local backend=$1 task=$2 controller_pid=$3
  local round="$FORMAL_ROOT/$backend/compression_only/formal_batch_01/$task/round_01"
  local terminal="$round/final/stage6_fixed_batch_terminal.json"
  local failed="$FORMAL_ROOT/$backend/compression_only/formal_batch_01/controller/${task}_round01.failed"
  while [[ ! -s "$terminal" ]]; do
    [[ ! -s "$failed" ]] || {
      echo "batch01 controller failed: $failed" >&2
      return 1
    }
    kill -0 "$controller_pid" 2>/dev/null || {
      echo "batch01 controller exited without terminal: $controller_pid" >&2
      return 1
    }
    sleep "$POLL_SECONDS"
  done
}

# The controllers are intentionally SIGSTOPed while already-running external
# shards finish. Resuming only after their terminal records exist prevents a
# second executor from writing the same AP output directory.
wait_for_exit "$ACTIVE_AP_PIDS"
kill -CONT "$TRT_CONTROLLER_PID" "$TVM_CONTROLLER_PID"
wait_for_terminal trt S5-PYR-TRT "$TRT_CONTROLLER_PID" & trt_terminal_pid=$!
wait_for_terminal tvm S5-PYR-TVM "$TVM_CONTROLLER_PID" & tvm_terminal_pid=$!
wait "$trt_terminal_pid"
wait "$tvm_terminal_pid"

rm -f "$FORMAL_ROOT/controller/"{trt,tvm}_fixed_arm_queue.{done,failed}
env BACKEND=trt GPU="$TRT_FIXED_GPUS" FORMAL_ROOT="$FORMAL_ROOT" \
  bash scripts/stage6_fixed_arm_queue_v1.sh & trt_fixed_pid=$!
env BACKEND=tvm GPU="$TVM_FIXED_GPUS" FORMAL_ROOT="$FORMAL_ROOT" \
  bash scripts/stage6_fixed_arm_queue_v1.sh & tvm_fixed_pid=$!
wait "$trt_fixed_pid"
wait "$tvm_fixed_pid"

# Keep successful tail markers so rerunning this supervisor remains idempotent.
rm -f "$FORMAL_ROOT/controller/"{trt,tvm}_backend_tail_queue.failed
env BACKEND=trt GPU="$TRT_TAIL_GPU" FORMAL_ROOT="$FORMAL_ROOT" \
  bash scripts/stage6_backend_tail_queue_v1.sh & trt_tail_pid=$!
env BACKEND=tvm GPU="$TVM_TAIL_GPU" FORMAL_ROOT="$FORMAL_ROOT" \
  bash scripts/stage6_backend_tail_queue_v1.sh & tvm_tail_pid=$!
wait "$trt_tail_pid"
wait "$tvm_tail_pid"

date -Is >"$SUPERVISOR_DONE"
rm -f "$SUPERVISOR_FAILED"
trap - EXIT
