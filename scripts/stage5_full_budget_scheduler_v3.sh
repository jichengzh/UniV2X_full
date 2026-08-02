#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
GPU_POOL=${GPU_POOL:-4,5,6,7}
CURRENT_ROUND0_PID=${CURRENT_ROUND0_PID:-}
MAX_INFRA_ATTEMPTS=${MAX_INFRA_ATTEMPTS:-3}
RETRY_DELAY_S=${RETRY_DELAY_S:-60}
PLAN_ONLY=0

[[ "$MAX_INFRA_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || {
  echo "MAX_INFRA_ATTEMPTS must be a positive integer" >&2
  exit 2
}
[[ "$RETRY_DELAY_S" =~ ^[0-9]+$ ]] || {
  echo "RETRY_DELAY_S must be a non-negative integer" >&2
  exit 2
}

if [[ ${1:-} == "--plan-only" ]]; then
  PLAN_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "unknown argument: $1" >&2
  exit 2
fi

JOBS=(
  "S5-PYR-TVM:1" "S5-PYR-TVM:2" "S5-PYR-TVM:3"
  "S5-PYR-TRT:0" "S5-PYR-TRT:1" "S5-PYR-TRT:2" "S5-PYR-TRT:3"
  "S5-COD-TVM:0" "S5-COD-TVM:1" "S5-COD-TVM:2" "S5-COD-TVM:3"
  "S5-COD-TRT:0" "S5-COD-TRT:1" "S5-COD-TRT:2" "S5-COD-TRT:3"
)

for job in "${JOBS[@]}"; do
  task=${job%%:*}
  round=${job##*:}
  echo "RUN $task round=$round"
done
[[ "$PLAN_ONLY" -eq 1 ]] && exit 0

mkdir -p "$ROOT/controller" "$ROOT/logs"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$ROOT/controller/full_budget_scheduler.failed"; fi' EXIT

if [[ -n "$CURRENT_ROUND0_PID" ]]; then
  while kill -0 "$CURRENT_ROUND0_PID" 2>/dev/null; do sleep 300; done
  [[ -s "$ROOT/controller/S5-PYR-TVM_round00.done" ]] || {
    echo "active S5-PYR-TVM round 0 did not finish successfully" >&2
    exit 1
  }
fi

for job in "${JOBS[@]}"; do
  task=${job%%:*}
  round=${job##*:}
  tag="${task}_round$(printf '%02d' "$round")"
  if [[ -s "$ROOT/controller/$tag.done" ]]; then
    echo "SKIP completed $task round=$round"
    continue
  fi
  [[ ! -e "$ROOT/controller/$tag.failed" ]] || {
    echo "refusing to continue past failed $task round=$round" >&2
    exit 1
  }
  env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" GPU_POOL="$GPU_POOL" \
    bash "$REPO/scripts/stage5_task_round_controller_v3.sh" --validate-only
  request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
  request_sha=missing
  [[ ! -f "$request" ]] || request_sha=$(sha256sum "$request" | awk '{print $1}')
  attempt=1
  while true; do
    set +e
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" GPU_POOL="$GPU_POOL" \
      bash "$REPO/scripts/stage5_task_round_controller_v3.sh"
    rc=$?
    set -e
    [[ "$rc" -ne 0 ]] || break

    current_request_sha=missing
    [[ ! -f "$request" ]] || current_request_sha=$(sha256sum "$request" | awk '{print $1}')
    [[ "$current_request_sha" == "$request_sha" ]] || {
      echo "measurement request changed during infrastructure retry: $task round=$round" >&2
      exit 1
    }
    failed_marker="$ROOT/controller/$tag.failed"
    failure_detail=$(tr '\n' ' ' <"$failed_marker" 2>/dev/null || true)
    jq -cn \
      --arg timestamp "$(date -Is)" --arg task "$task" --argjson round "$round" \
      --argjson attempt "$attempt" --argjson returncode "$rc" \
      --arg request_sha256 "$request_sha" --arg detail "$failure_detail" \
      '{schema_version:"stage5_infrastructure_retry_audit_v1",timestamp:$timestamp,
        task_id:$task,round_index:$round,attempt:$attempt,returncode:$returncode,
        measurement_request_sha256:$request_sha256,failure_detail:$detail}' \
      >>"$ROOT/controller/$tag.retry_audit.jsonl"
    if [[ "$attempt" -ge "$MAX_INFRA_ATTEMPTS" ]]; then
      echo "controller failed after $attempt infrastructure attempts: $task round=$round" >&2
      exit "$rc"
    fi
    mkdir -p "$ROOT/controller/archive"
    [[ ! -e "$failed_marker" ]] || \
      mv "$failed_marker" "$ROOT/controller/archive/$tag.attempt${attempt}.failed"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY_S"
  done
done

for task in S5-PYR-TVM S5-PYR-TRT S5-COD-TVM S5-COD-TRT; do
  jq -e '.status == "budget_exhausted" and .budget_consumed == 16 and .round_count == 4' \
    "$ROOT/$task/task_budget_terminal.json" >/dev/null
done
jq -n --arg completed_at "$(date -Is)" \
  '{schema_version:"stage5_full_budget_scheduler_terminal_v3",status:"budget_exhausted",
    task_count:4,round_count:16,formal_online_genomes:64,completed_at:$completed_at}' \
  >"$ROOT/controller/full_budget_scheduler_terminal.json"
date -Is >"$ROOT/controller/full_budget_scheduler.done"
rm -f "$ROOT/controller/full_budget_scheduler.failed"
