#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_codriving_actual_v3_20260721}
MAX_INFRA_ATTEMPTS=${MAX_INFRA_ATTEMPTS:-3}
RETRY_DELAY_S=${RETRY_DELAY_S:-60}
TVM_GPUS=${TVM_GPUS:-1,3,4}
TRT_GPUS=${TRT_GPUS:-5,6,7}

mkdir -p "$ROOT/controller" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/codriving_actual_v3_scheduler.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$ROOT/controller/codriving_actual_v3_scheduler.failed"; fi' EXIT
cd "$REPO"

round_complete() {
  local task=$1 round=$2
  test -s "$ROOT/$task/round_$(printf '%02d' "$round")/actual_feedback/stage5_feedback_v3_actual.json"
}

run_controller() {
  local task=$1 round=$2 gpus=$3 request request_sha attempt=1 rc
  request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
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
    jq -cn --arg timestamp "$(date -Is)" --arg task "$task" \
      --argjson round "$round" --argjson attempt "$attempt" \
      --argjson returncode "$rc" --arg request_sha256 "$request_sha" \
      '{schema_version:"stage5_codriving_actual_v3_retry_audit_v1",
        timestamp:$timestamp,task_id:$task,round_index:$round,attempt:$attempt,
        returncode:$returncode,measurement_request_sha256:$request_sha256}' \
      >>"$ROOT/controller/${task}_round$(printf '%02d' "$round").retry_audit.jsonl"
    [[ "$attempt" -lt "$MAX_INFRA_ATTEMPTS" ]] || return "$rc"
    rm -f "$ROOT/controller/${task}_round$(printf '%02d' "$round").failed"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY_S"
  done
}

for round in 0 1 2 3; do
  pending_tasks=()
  for task in S5-COD-TVM S5-COD-TRT; do
    if round_complete "$task" "$round"; then
      printf '%s\n' "$(date -Is) round=$round task=$task phase=skip_completed" \
        >>"$ROOT/controller/stage_timeline.log"
      continue
    fi
    request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
    [[ -s "$request" ]] || { echo "missing request: $request" >&2; exit 1; }
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" FEEDBACK_CONTRACT=actual_v3 \
      bash scripts/stage5_task_round_controller_v3.sh --validate-only
    pending_tasks+=("$task")
  done
  ((${#pending_tasks[@]} > 0)) || continue
  pids=()
  tasks=()
  for task in "${pending_tasks[@]}"; do
    gpus=$TVM_GPUS
    [[ "$task" == "S5-COD-TVM" ]] || gpus=$TRT_GPUS
    run_controller "$task" "$round" "$gpus" &
    pids+=("$!")
    tasks+=("$task")
  done
  controller_rc=0
  set +e
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || {
      echo "round controller failure: round=$round task=${tasks[$index]}" >&2
      controller_rc=1
    }
  done
  set -e
  [[ "$controller_rc" -eq 0 ]] || exit 1
  printf '%s\n' "$(date -Is) round=$round phase=actual_feedback_complete" \
    >>"$ROOT/controller/stage_timeline.log"
done

for task in S5-COD-TVM S5-COD-TRT; do
  jq -e '
    .status == "budget_exhausted" and .budget_consumed == 16 and
    .round_count == 4 and .feedback_contract == "actual_v3"
  ' "$ROOT/$task/task_budget_terminal.json" >/dev/null
done
jq -n --arg completed_at "$(date -Is)" \
  '{schema_version:"stage5_codriving_actual_v3_scheduler_terminal_v1",
    status:"budget_exhausted",task_count:2,round_count:8,
    fresh_round0_rows:8,new_online_rows:32,total_online_rows:32,
    feedback_contract:"actual_v3",completed_at:$completed_at}' \
  >"$ROOT/controller/codriving_actual_v3_scheduler_terminal.json"
date -Is >"$ROOT/controller/codriving_actual_v3_scheduler.done"
rm -f "$ROOT/controller/codriving_actual_v3_scheduler.failed"
