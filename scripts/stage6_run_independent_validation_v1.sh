#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
PLAN_JSON=${PLAN_JSON:?PLAN_JSON is required}
GPU_POOL=${GPU_POOL:-3,4,5,6,7}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
AUDIT_JSON=${AUDIT_JSON:?AUDIT_JSON is required}

cd "$REPO"
IFS=',' read -r -a GPUS <<<"$GPU_POOL"
GPU_COUNT=${#GPUS[@]}
[[ "$GPU_COUNT" -gt 0 ]]

run_ap_stage() {
  local ap_root=$1 stage=$2
  local pids=() shard=0
  while [[ -s "$ap_root/ap_plan_shard_${shard}.jsonl" ]]; do
    "$PY" scripts/stage3_execute_ap_plan_v3.py \
      --ap-plan-jsonl "$ap_root/ap_plan_shard_${shard}.jsonl" --stage "$stage" \
      --state-jsonl "$ap_root/ap_state_shard_${shard}.jsonl" \
      --gpu "${GPUS[$((shard % GPU_COUNT))]}" --univ2x-python "$PY" \
      --artifact-root "$ap_root/execution/$stage/shard_${shard}" &
    pids+=("$!")
    shard=$((shard + 1))
    if (( ${#pids[@]} == GPU_COUNT )); then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

plan_count=$(jq -r '.plan_count' "$PLAN_JSON")
for ((plan_index=0; plan_index<plan_count; plan_index++)); do
  arm_root=$(jq -r ".plans[$plan_index].root" "$PLAN_JSON")
  for repeat in 0 1 2; do
    repeat_root="$arm_root/repeat_$repeat"
    "$PY" scripts/stage3_execute_performance_plan_v3.py \
      --jobs-jsonl "$repeat_root/performance_jobs.jsonl" \
      --state-jsonl "$repeat_root/performance_state.jsonl" \
      --gpus "$GPU_POOL" --max-workers "$GPU_COUNT"
  done

  ap_root="$arm_root/ap"
  mkdir -p "$ap_root"
  "$PY" scripts/stage5_ap_plan_v2.py \
    --manifest-json "$arm_root/repeat_2/performance_manifest.json" \
    --performance-jobs-jsonl "$arm_root/repeat_2/performance_jobs.jsonl" \
    --performance-state-jsonl "$arm_root/repeat_2/performance_state.jsonl" \
    --output-root "$ap_root/execution" --output-json "$ap_root/ap_plan.json" \
    --output-jsonl "$ap_root/ap_plan.jsonl"
  row_count=$(wc -l <"$ap_root/ap_plan.jsonl")
  for ((shard=0; shard<row_count; shard++)); do
    sed -n "$((shard + 1))p" "$ap_root/ap_plan.jsonl" >"$ap_root/ap_plan_shard_${shard}.jsonl"
  done
  run_ap_stage "$ap_root" sanity
  run_ap_stage "$ap_root" full
  : >"$ap_root/ap_state.jsonl.tmp"
  for ((shard=0; shard<row_count; shard++)); do
    cat "$ap_root/ap_state_shard_${shard}.jsonl" >>"$ap_root/ap_state.jsonl.tmp"
  done
  mv "$ap_root/ap_state.jsonl.tmp" "$ap_root/ap_state.jsonl"
done

"$PY" scripts/stage6_finalize_independent_validation_v1.py \
  --plan-json "$PLAN_JSON" --output-json "$AUDIT_JSON"
