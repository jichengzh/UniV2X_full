#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_codriving_actual_v3_20260721}
TVM_GPUS=${TVM_GPUS:-0,1,3,5}
TRT_GPUS=${TRT_GPUS:-4,6,7}
MAX_INFRA_ATTEMPTS=${MAX_INFRA_ATTEMPTS:-3}
RETRY_DELAY_S=${RETRY_DELAY_S:-60}

mkdir -p "$ROOT/controller" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/codriving_actual_v3_independent_resume.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$ROOT/controller/codriving_independent_resume.failed"; fi' EXIT
cd "$REPO"

round_complete() {
  local task=$1 round=$2
  test -s "$ROOT/$task/round_$(printf '%02d' "$round")/actual_feedback/stage5_feedback_v3_actual.json"
}

wait_for_round() {
  local task=$1 round=$2
  until round_complete "$task" "$round"; do
    sleep 30
  done
}

run_round() {
  local task=$1 round=$2 gpus=$3 request request_sha attempt=1 rc
  request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
  [[ -s "$request" ]] || { echo "missing request: $request" >&2; return 1; }
  request_sha=$(sha256sum "$request" | awk '{print $1}')
  while true; do
    set +e
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" GPU_POOL="$gpus" \
      FEEDBACK_CONTRACT=actual_v3 bash scripts/stage5_task_round_controller_v3.sh
    rc=$?
    set -e
    [[ "$rc" -ne 0 ]] || return 0
    [[ "$(sha256sum "$request" | awk '{print $1}')" == "$request_sha" ]] || {
      echo "request changed during retry: $task round=$round" >&2
      return 1
    }
    [[ "$attempt" -lt "$MAX_INFRA_ATTEMPTS" ]] || return "$rc"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY_S"
  done
}

run_lane() {
  local task=$1 gpus=$2 wait_task=${3:-} wait_round=${4:-}
  if [[ -n "$wait_task" ]]; then
    wait_for_round "$wait_task" "$wait_round"
  fi
  for round in 2 3; do
    if round_complete "$task" "$round"; then
      continue
    fi
    run_round "$task" "$round" "$gpus"
  done
}

run_lane S5-COD-TVM "$TVM_GPUS" &
tvm_pid=$!
run_lane S5-COD-TRT "$TRT_GPUS" S5-COD-TRT 1 &
trt_pid=$!
wait "$tvm_pid"
wait "$trt_pid"

for task in S5-COD-TVM S5-COD-TRT; do
  jq -e '.status == "budget_exhausted" and .budget_consumed == 16 and
    .round_count == 4 and .feedback_contract == "actual_v3"' \
    "$ROOT/$task/task_budget_terminal.json" >/dev/null
done
date -Is >"$ROOT/controller/codriving_independent_resume.done"
rm -f "$ROOT/controller/codriving_independent_resume.failed"
