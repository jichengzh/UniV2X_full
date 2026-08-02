#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_single_target_search_v2_20260718}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU_POOL=${GPU_POOL:-4,5,6,7}
TASK=S5-PYR-TVM
ROUND="$ROOT/$TASK/round_00"
CTRL="$ROOT/controller"
LOGS="$ROOT/logs"
REQUEST="$ROUND/measurement_request.json"

mkdir -p "$CTRL" "$LOGS" "$ROUND"
exec > >(tee -a "$LOGS/${TASK}_round0_controller.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$CTRL/${TASK}_round0.failed"; fi' EXIT
cd "$REPO"

jq -e '
  .schema_version == "stage5_measurement_request_v2" and
  .task_id == "S5-PYR-TVM" and .batch_size == 4 and
  (.rows | length) == 4 and
  ([.rows[].model] | unique) == ["pyramid"] and
  ([.rows[].dispatch_key] | unique) == ["tvm_auto"]
' "$REQUEST" >/dev/null

IFS=',' read -r -a GPUS <<<"$GPU_POOL"
mapfile -t SOURCE_GROUP_IDS < <(jq -r '[.rows[].group_id] | unique[]' "$REQUEST")
[[ ${#SOURCE_GROUP_IDS[@]} -le ${#GPUS[@]} ]]
pids=()
for index in "${!SOURCE_GROUP_IDS[@]}"; do
  group_id=${SOURCE_GROUP_IDS[$index]}
  gpu=${GPUS[$index]}
  bash scripts/stage5_materialize_round_sources_v1.sh \
    --request "$REQUEST" --model pyramid --group-id "$group_id" --gpu "$gpu" \
    >"$LOGS/source_${group_id#pyramid|}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

QUANT="$ROUND/quant_contracts"
quant_pids=()
quant_index=0
while IFS= read -r row; do
  width=$(jq -r '.width | map(tostring) | join("x")' <<<"$row")
  onnx=$(jq -r '.source_contract.onnx_path' <<<"$row")
  calibration=$(jq -r '.source_contract.calibration_npz' <<<"$row")
  summary=$(jq -r '.source_contract.calibration_summary' <<<"$row")
  output="$QUANT/$width/tensor_quant_params.json"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES="${GPUS[$quant_index]}" "$PY" \
    scripts/stage3_tvm_int8_quant_contract_v3.py \
    --onnx "$onnx" --calibration-npz "$calibration" \
    --calibration-summary "$summary" --output-json "$output" \
    >"$LOGS/quant_contract_${width}.log" 2>&1 &
  quant_pids+=("$!")
  quant_index=$((quant_index + 1))
done < <(jq -c '.rows[] | select(.dispatch_key=="tvm_auto" and .q_mode=="int8")' "$REQUEST")
for pid in "${quant_pids[@]}"; do
  wait "$pid"
done

PERF="$ROUND/performance"
mkdir -p "$PERF"
"$PY" scripts/stage5_build_performance_plan_v2.py \
  --request-json "$REQUEST" \
  --remote-artifact-root "$ROUND/performance_execution" \
  --output-dir "$PERF" --quant-contract-root "$QUANT" --gpus "$GPU_POOL"
"$PY" scripts/stage3_execute_performance_plan_v3.py \
  --jobs-jsonl "$PERF/performance_jobs.jsonl" \
  --state-jsonl "$PERF/performance_state.jsonl" \
  --gpus "$GPU_POOL" --max-workers 4

AP="$ROUND/ap"
mkdir -p "$AP"
"$PY" scripts/stage5_ap_plan_v2.py \
  --manifest-json "$PERF/performance_manifest.json" \
  --performance-jobs-jsonl "$PERF/performance_jobs.jsonl" \
  --performance-state-jsonl "$PERF/performance_state.jsonl" \
  --output-root "$ROUND/ap_execution" \
  --output-json "$AP/ap_plan.json" --output-jsonl "$AP/ap_plan.jsonl"

for shard in 0 1 2 3; do
  awk -v shard="$shard" 'NR-1 == shard' "$AP/ap_plan.jsonl" >"$AP/ap_plan_shard_${shard}.jsonl"
  if [[ ! -s "$AP/ap_state_shard_${shard}.jsonl" && -s "$AP/ap_state_sanity_${shard}.jsonl" ]]; then
    cp "$AP/ap_state_sanity_${shard}.jsonl" "$AP/ap_state_shard_${shard}.jsonl"
  fi
done
run_ap_stage() {
  local stage=$1 rc=0
  local stage_pids=()
  for shard in 0 1 2 3; do
    "$PY" scripts/stage3_execute_ap_plan_v3.py \
      --ap-plan-jsonl "$AP/ap_plan_shard_${shard}.jsonl" --stage "$stage" \
      --state-jsonl "$AP/ap_state_shard_${shard}.jsonl" --gpu "${GPUS[$shard]}" \
      --univ2x-python "$PY" --artifact-root "$ROUND/ap_execution/$stage/shard_${shard}" &
    stage_pids+=("$!")
  done
  for pid in "${stage_pids[@]}"; do wait "$pid" || true; done
  for shard in 0 1 2 3; do
    jq -s -e --arg stage "$stage" '
      any(.[]; .record_type == "job_terminal" and .stage == $stage and .status == "success")
    ' "$AP/ap_state_shard_${shard}.jsonl" >/dev/null || rc=1
  done
  return "$rc"
}
run_ap_stage sanity
run_ap_stage full
cat "$AP"/ap_state_shard_*.jsonl >"$AP/ap_state.jsonl.tmp"
mv "$AP/ap_state.jsonl.tmp" "$AP/ap_state.jsonl"
FINAL="$ROUND/final"
"$PY" scripts/stage5_finalize_feedback_v2.py \
  --manifest-json "$PERF/performance_manifest.json" \
  --measurement-request-json "$REQUEST" \
  --ap-plan-jsonl "$AP/ap_plan.jsonl" \
  --performance-state-jsonl "$PERF/performance_state.jsonl" \
  --ap-state-jsonl "$AP/ap_state.jsonl" --output-dir "$FINAL"
"$PY" scripts/stage5_advance_task_round_v2.py \
  --task-id "$TASK" --feedback-json "$FINAL/stage5_feedback_v2.json" \
  --output-dir "$ROOT" --round-index 1
date -Is >"$CTRL/${TASK}_round0.done"
