#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_pyramid_actual_v3_20260720}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
MAX_INFRA_ATTEMPTS=${MAX_INFRA_ATTEMPTS:-3}
RETRY_DELAY_S=${RETRY_DELAY_S:-60}
TVM_GPUS=${TVM_GPUS:-0,1,2,3}
TRT_GPUS=${TRT_GPUS:-4,5,6,7}
SOURCE_GPUS=${SOURCE_GPUS:-$TVM_GPUS,$TRT_GPUS}

mkdir -p "$ROOT/controller" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/pyramid_actual_v3_scheduler.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$ROOT/controller/pyramid_actual_v3_scheduler.failed"; fi' EXIT
cd "$REPO"

materialize_round_sources() {
  local round=$1 task_csv=$2 plan
  local source_gpu_array=()
  IFS=',' read -r -a source_gpu_array <<<"$SOURCE_GPUS"
  ((${#source_gpu_array[@]} > 0)) || {
    echo "SOURCE_GPUS must contain at least one GPU" >&2
    return 2
  }
  plan="$ROOT/controller/round_$(printf '%02d' "$round")_source_plan.tsv"
  "$PY" - "$ROOT" "$round" "$plan" "$task_csv" <<'PY'
import json,sys
from pathlib import Path
root,round_index,out=Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3])
tasks=tuple(task for task in sys.argv[4].split(",") if task)
seen=set(); rows=[]
for task in tasks:
    request=root/task/f"round_{round_index:02d}"/"measurement_request.json"
    payload=json.loads(request.read_text())
    for row in payload["rows"]:
        group=str(row["group_id"])
        if group not in seen:
            seen.add(group); rows.append((str(request),group))
out.write_text("".join(f"{request}\t{group}\n" for request,group in rows))
PY
  local pids=() index=0
  while IFS=$'\t' read -r request group; do
    local gpu=${source_gpu_array[$((index % ${#source_gpu_array[@]}))]} safe=${group//|/_}
    bash scripts/stage5_materialize_round_sources_v1.sh \
      --request "$request" --model pyramid --group-id "$group" --gpu "$gpu" \
      >"$ROOT/logs/round$(printf '%02d' "$round")_source_${safe}_gpu${gpu}.log" 2>&1 &
    pids+=("$!"); index=$((index + 1))
  done <"$plan"
  for pid in "${pids[@]}"; do wait "$pid"; done
}

round_complete() {
  local task=$1 round=$2
  test -s "$ROOT/$task/round_$(printf '%02d' "$round")/actual_feedback/stage5_feedback_v3_actual.json"
}

run_controller() {
  local task=$1 round=$2 gpus=$3
  local tag="${task}_round$(printf '%02d' "$round")" request
  request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
  local request_sha attempt=1 rc
  request_sha=$(sha256sum "$request" | awk '{print $1}')
  while true; do
    set +e
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" GPU_POOL="$gpus" \
      FEEDBACK_CONTRACT=actual_v3 \
      bash scripts/stage5_task_round_controller_v3.sh
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
      '{schema_version:"stage5_actual_v3_retry_audit",timestamp:$timestamp,
        task_id:$task,round_index:$round,attempt:$attempt,returncode:$returncode,
        measurement_request_sha256:$request_sha256}' \
      >>"$ROOT/controller/$tag.retry_audit.jsonl"
    [[ "$attempt" -lt "$MAX_INFRA_ATTEMPTS" ]] || return "$rc"
    rm -f "$ROOT/controller/$tag.failed"
    attempt=$((attempt + 1))
    sleep "$RETRY_DELAY_S"
  done
}

for round in 1 2 3; do
  pending_tasks=()
  for task in S5-PYR-TVM S5-PYR-TRT; do
    if round_complete "$task" "$round"; then
      printf '%s\n' "$(date -Is) round=$round task=$task phase=skip_completed" \
        >>"$ROOT/controller/stage_timeline.log"
      continue
    fi
    pending_tasks+=("$task")
    request="$ROOT/$task/round_$(printf '%02d' "$round")/measurement_request.json"
    [[ -s "$request" ]] || { echo "missing request: $request" >&2; exit 1; }
    env TASK="$task" ROUND_INDEX="$round" ROOT="$ROOT" FEEDBACK_CONTRACT=actual_v3 \
      bash scripts/stage5_task_round_controller_v3.sh --validate-only
  done
  ((${#pending_tasks[@]} > 0)) || continue
  task_csv=$(IFS=,; echo "${pending_tasks[*]}")
  printf '%s\n' "$(date -Is) round=$round phase=source_materialization_start" \
    >>"$ROOT/controller/stage_timeline.log"
  materialize_round_sources "$round" "$task_csv"
  printf '%s\n' "$(date -Is) round=$round phase=source_materialization_end" \
    >>"$ROOT/controller/stage_timeline.log"
  controller_pids=()
  controller_tasks=()
  for task in "${pending_tasks[@]}"; do
    gpus=$TVM_GPUS
    [[ "$task" == "S5-PYR-TVM" ]] || gpus=$TRT_GPUS
    run_controller "$task" "$round" "$gpus" &
    controller_pids+=("$!")
    controller_tasks+=("$task")
  done
  set +e
  controller_rc=0
  for index in "${!controller_pids[@]}"; do
    wait "${controller_pids[$index]}" || {
      echo "round controller failure: round=$round task=${controller_tasks[$index]}" >&2
      controller_rc=1
    }
  done
  set -e
  [[ "$controller_rc" -eq 0 ]] || exit 1
  printf '%s\n' "$(date -Is) round=$round phase=atomic_feedback_end" \
    >>"$ROOT/controller/stage_timeline.log"
done

for task in S5-PYR-TVM S5-PYR-TRT; do
  jq -e '
    .status == "budget_exhausted" and .budget_consumed == 16 and
    .round_count == 4 and .feedback_contract == "actual_v3"
  ' "$ROOT/$task/task_budget_terminal.json" >/dev/null
done
jq -n --arg completed_at "$(date -Is)" \
  '{schema_version:"stage5_pyramid_actual_v3_scheduler_terminal",
    status:"budget_exhausted",task_count:2,round_count:8,
    imported_round0_rows:8,new_online_rows:24,total_online_rows:32,
    feedback_contract:"actual_v3",completed_at:$completed_at}' \
  >"$ROOT/controller/pyramid_actual_v3_scheduler_terminal.json"
date -Is >"$ROOT/controller/pyramid_actual_v3_scheduler.done"
rm -f "$ROOT/controller/pyramid_actual_v3_scheduler.failed"
