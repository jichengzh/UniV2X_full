#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_codriving_actual_v3_20260721}
MAX_INFRA_ATTEMPTS=${MAX_INFRA_ATTEMPTS:-3}
RETRY_DELAY_S=${RETRY_DELAY_S:-60}

mkdir -p "$ROOT/controller" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/codriving_gpu34567_closure_v2.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$ROOT/controller/codriving_gpu34567_closure_v2.failed"; fi' EXIT
cd "$REPO"

round_complete() {
  local task=$1 round=$2
  test -s "$ROOT/$task/round_$(printf '%02d' "$round")/actual_feedback/stage5_feedback_v3_actual.json"
}

run_round() {
  local task=$1 round=$2 gpus=$3 request request_sha attempt=1 rc
  request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
  [[ -s "$request" ]] || { echo "missing request: $request" >&2; return 1; }
  request_sha=$(sha256sum "$request" | awk '{print $1}')
  while true; do
    rm -f "$ROOT/controller/${task}_round$(printf '%02d' "$round").failed"
    set +e
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" GPU_POOL="$gpus" \
      FEEDBACK_CONTRACT=actual_v3 bash scripts/stage5_task_round_controller_v3.sh
    rc=$?
    set -e
    [[ "$(sha256sum "$request" | awk '{print $1}')" == "$request_sha" ]] || {
      echo "request SHA changed: task=$task round=$round" >&2
      return 1
    }
    [[ "$rc" -ne 0 ]] || return 0
    attempt=$((attempt + 1))
    [[ "$attempt" -le "$MAX_INFRA_ATTEMPTS" ]] || return "$rc"
    sleep "$RETRY_DELAY_S"
  done
}

# Round 2 may already be running on GPUs 3,4,5,6. Do not launch a duplicate.
until round_complete S5-COD-TRT 2; do sleep 30; done

# TVM's final full AP owns GPU 7. Round 3 starts only after that task releases it.
until round_complete S5-COD-TVM 3; do sleep 30; done

if ! round_complete S5-COD-TRT 3; then
  run_round S5-COD-TRT 3 3,4,5,6,7
fi

for task in S5-COD-TVM S5-COD-TRT; do
  jq -e '.status == "budget_exhausted" and .budget_consumed == 16 and
    .round_count == 4 and .feedback_contract == "actual_v3"' \
    "$ROOT/$task/task_budget_terminal.json" >/dev/null
done

date -Is >"$ROOT/controller/codriving_gpu34567_closure_v2.done"
rm -f "$ROOT/controller/codriving_gpu34567_closure_v2.failed"
