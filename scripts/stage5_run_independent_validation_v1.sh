#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
CLOSURE_ROOT=${CLOSURE_ROOT:-$FORMAL_ROOT/closure_v3}
VALIDATION_ROOT=${VALIDATION_ROOT:-$FORMAL_ROOT/independent_validation_v1}
COLDSTART_ROOT=${COLDSTART_ROOT:-$REPO/results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1}
ROWS=${ROWS:-$COLDSTART_ROOT/gold176_final.json}
GRAPHS=${GRAPHS:-$COLDSTART_ROOT/graph_features.json}
ROWS_SHA=${ROWS_SHA:-9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19}
GRAPHS_SHA=${GRAPHS_SHA:-c5f03e19daba4779cb187f036d7c4bf612d3479a00453149f4eaf213412536cd}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU_POOL=${GPU_POOL:-0,1,2,3}
START_TASK_INDEX=${START_TASK_INDEX:-0}
GPU_IDLE_CHECK_INTERVAL_S=${GPU_IDLE_CHECK_INTERVAL_S:-10}
GPU_STABLE_IDLE_CHECKS=${GPU_STABLE_IDLE_CHECKS:-3}
PLAN_ONLY=0

[[ "$START_TASK_INDEX" =~ ^[0-3]$ ]]
[[ "$GPU_IDLE_CHECK_INTERVAL_S" =~ ^[1-9][0-9]*$ ]]
[[ "$GPU_STABLE_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]]

if [[ ${1:-} == "--plan-only" ]]; then
  PLAN_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "unknown argument: $1" >&2
  exit 2
fi

TASKS=(S5-PYR-TVM S5-PYR-TRT S5-COD-TVM S5-COD-TRT)
for task in "${TASKS[@]}"; do
  for repeat in 0 1 2; do echo "PERFORMANCE $task repeat=$repeat"; done
  echo "FULL_AP $task"
done
echo "FINAL_CLOSURE"
[[ "$PLAN_ONLY" -eq 1 ]] && exit 0

mkdir -p "$VALIDATION_ROOT/logs" "$CLOSURE_ROOT"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$VALIDATION_ROOT/independent_validation.failed"; fi' EXIT

"$PY" "$REPO/scripts/stage5_finalize_full_budget_v3.py" \
  --formal-root "$FORMAL_ROOT" --coldstart-rows-json "$ROWS" \
  --coldstart-graph-features-json "$GRAPHS" \
  --expected-coldstart-rows-sha256 "$ROWS_SHA" \
  --expected-coldstart-graph-features-sha256 "$GRAPHS_SHA" \
  --output-dir "$CLOSURE_ROOT"

IFS=',' read -r -a GPUS <<<"$GPU_POOL"
[[ ${#GPUS[@]} -ge 1 ]]
GPU_COUNT=${#GPUS[@]}

gpus_idle() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, -v wanted="$GPU_POOL" '
    BEGIN {
      count = split(wanted, ids, ",")
      for (i = 1; i <= count; i++) required[ids[i] + 0] = 1
    }
    {
      gpu = $1 + 0; memory = $2 + 0; utilization = $3 + 0
      if (gpu in required) {
        seen[gpu] = 1
        if (memory > 100 || utilization > 10) busy[gpu] = 1
      }
    }
    END {
      for (gpu in required) if (!(gpu in seen) || (gpu in busy)) exit 1
    }
  '
}

wait_for_stable_idle() {
  local stable=0
  while (( stable < GPU_STABLE_IDLE_CHECKS )); do
    if gpus_idle; then stable=$((stable + 1)); else stable=0; fi
    (( stable == GPU_STABLE_IDLE_CHECKS )) || sleep "$GPU_IDLE_CHECK_INTERVAL_S"
  done
}

run_ap_stage() {
  local task=$1 ap_root=$2 stage=$3
  local pids=() shard=0
  while [[ -s "$ap_root/ap_plan_shard_${shard}.jsonl" ]]; do
    "$PY" "$REPO/scripts/stage3_execute_ap_plan_v3.py" \
      --ap-plan-jsonl "$ap_root/ap_plan_shard_${shard}.jsonl" --stage "$stage" \
      --state-jsonl "$ap_root/ap_state_shard_${shard}.jsonl" \
      --gpu "${GPUS[$((shard % ${#GPUS[@]}))]}" --univ2x-python "$PY" \
      --artifact-root "$VALIDATION_ROOT/$task/ap_execution/$stage/shard_${shard}" &
    pids+=("$!")
    shard=$((shard + 1))
    if (( ${#pids[@]} == GPU_COUNT )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

for ((task_index=START_TASK_INDEX; task_index<${#TASKS[@]}; task_index++)); do
  task=${TASKS[$task_index]}
  task_root="$VALIDATION_ROOT/$task"
  "$PY" "$REPO/scripts/stage5_prepare_independent_validation_v1.py" \
    --task-root "$FORMAL_ROOT/$task" \
    --closure-json "$CLOSURE_ROOT/$task/stage5_task_closure_audit_v3.json" \
    --coldstart-rows-json "$ROWS" \
    --output-dir "$task_root" --remote-artifact-root "$task_root/performance_execution" \
    --gpus "$GPU_POOL"
  for repeat in 0 1 2; do
    repeat_root="$task_root/repeat_${repeat}"
    wait_for_stable_idle
    "$PY" "$REPO/scripts/stage3_execute_performance_plan_v3.py" \
      --jobs-jsonl "$repeat_root/performance_jobs.jsonl" \
      --state-jsonl "$repeat_root/performance_state.jsonl" \
      --gpus "$GPU_POOL" --max-workers "${#GPUS[@]}"
  done

  ap_root="$task_root/ap"
  mkdir -p "$ap_root"
  "$PY" "$REPO/scripts/stage5_ap_plan_v2.py" \
    --manifest-json "$task_root/repeat_2/performance_manifest.json" \
    --performance-jobs-jsonl "$task_root/repeat_2/performance_jobs.jsonl" \
    --performance-state-jsonl "$task_root/repeat_2/performance_state.jsonl" \
    --output-root "$task_root/ap_execution" --output-json "$ap_root/ap_plan.json" \
    --output-jsonl "$ap_root/ap_plan.jsonl"
  row_count=$(wc -l <"$ap_root/ap_plan.jsonl")
  for ((shard=0; shard<row_count; shard++)); do
    sed -n "$((shard + 1))p" "$ap_root/ap_plan.jsonl" >"$ap_root/ap_plan_shard_${shard}.jsonl"
  done
  run_ap_stage "$task" "$ap_root" sanity
  run_ap_stage "$task" "$ap_root" full
  : >"$ap_root/ap_state.jsonl"
  for ((shard=0; shard<row_count; shard++)); do
    cat "$ap_root/ap_state_shard_${shard}.jsonl" >>"$ap_root/ap_state.jsonl"
  done
done

AUDIT="$VALIDATION_ROOT/stage5_independent_validation_audit_v1.json"
"$PY" "$REPO/scripts/stage5_finalize_independent_validation_v1.py" \
  --closure-root "$CLOSURE_ROOT" --validation-root "$VALIDATION_ROOT" \
  --output-json "$AUDIT"
"$PY" "$REPO/scripts/stage5_finalize_full_budget_v3.py" \
  --formal-root "$FORMAL_ROOT" --coldstart-rows-json "$ROWS" \
  --coldstart-graph-features-json "$GRAPHS" \
  --expected-coldstart-rows-sha256 "$ROWS_SHA" \
  --expected-coldstart-graph-features-sha256 "$GRAPHS_SHA" \
  --independent-validation-audit "$AUDIT" --output-dir "$CLOSURE_ROOT"

date -Is >"$VALIDATION_ROOT/independent_validation.done"
rm -f "$VALIDATION_ROOT/independent_validation.failed"
