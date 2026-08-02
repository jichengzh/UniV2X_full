#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU_POOL=${GPU_POOL:-4,5,6,7}
TASK=${TASK:-}
ROUND_INDEX=${ROUND_INDEX:-0}
FEEDBACK_CONTRACT=${FEEDBACK_CONTRACT:-surrogate_v2}
FIXED_BATCH_MODE=${FIXED_BATCH_MODE:-0}
TVM_FP16_MAX_TRIALS=${TVM_FP16_MAX_TRIALS:-64}
VALIDATE_ONLY=0

if [[ ${1:-} == "--validate-only" ]]; then
  VALIDATE_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "unknown argument: $1" >&2
  exit 2
fi

case "$TASK" in
  S5-PYR-TVM) EXPECTED_MODEL=pyramid; EXPECTED_DISPATCH=tvm_auto ;;
  S5-PYR-TRT) EXPECTED_MODEL=pyramid; EXPECTED_DISPATCH=trt_engine ;;
  S5-COD-TVM) EXPECTED_MODEL=codriving; EXPECTED_DISPATCH=tvm_auto ;;
  S5-COD-TRT) EXPECTED_MODEL=codriving; EXPECTED_DISPATCH=trt_engine ;;
  *) echo "TASK must be one of the four frozen Stage5 task ids" >&2; exit 2 ;;
esac
[[ "$ROUND_INDEX" =~ ^[0-3]$ ]] || { echo "ROUND_INDEX must be 0..3" >&2; exit 2; }
[[ "$FEEDBACK_CONTRACT" == "surrogate_v2" || "$FEEDBACK_CONTRACT" == "actual_v3" ]] || {
  echo "FEEDBACK_CONTRACT must be surrogate_v2 or actual_v3" >&2
  exit 2
}

ROUND="$ROOT/$TASK/round_$(printf '%02d' "$ROUND_INDEX")"
REQUEST="$ROUND/measurement_request.json"
[[ -s "$REQUEST" ]] || { echo "missing measurement request: $REQUEST" >&2; exit 2; }
jq -e \
  --arg task "$TASK" --arg model "$EXPECTED_MODEL" --arg dispatch "$EXPECTED_DISPATCH" \
  --argjson round "$ROUND_INDEX" '
  .schema_version == "stage5_measurement_request_v2" and
  .task_id == $task and .round_index == $round and .batch_size == 4 and
  (.rows | length) == 4 and
  ([.rows[].task_id] | unique) == [$task] and
  ([.rows[].model] | unique) == [$model] and
  ([.rows[].dispatch_key] | unique) == [$dispatch]
' "$REQUEST" >/dev/null

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
  echo "REQUEST_VALID task=$TASK round=$ROUND_INDEX model=$EXPECTED_MODEL dispatch=$EXPECTED_DISPATCH"
  exit 0
fi

CTRL="$ROOT/controller"
LOGS="$ROOT/logs"
TAG="${TASK}_round$(printf '%02d' "$ROUND_INDEX")"
mkdir -p "$CTRL" "$LOGS" "$ROUND"
exec > >(tee -a "$LOGS/${TAG}_controller.log") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$CTRL/${TAG}.failed"; fi' EXIT
cd "$REPO"

IFS=',' read -r -a GPUS <<<"$GPU_POOL"
GPU_COUNT=${#GPUS[@]}
[[ "$GPU_COUNT" -ge 1 ]] || { echo "GPU_POOL must contain at least one GPU" >&2; exit 2; }
mapfile -t SOURCE_GROUP_IDS < <(jq -r '[.rows[].group_id] | unique[]' "$REQUEST")

source_pids=()
for index in "${!SOURCE_GROUP_IDS[@]}"; do
  group_id=${SOURCE_GROUP_IDS[$index]}
  gpu=${GPUS[$((index % GPU_COUNT))]}
  safe_group=${group_id//|/_}
  bash scripts/stage5_materialize_round_sources_v1.sh \
    --request "$REQUEST" --model "$EXPECTED_MODEL" --group-id "$group_id" --gpu "$gpu" \
    >"$LOGS/${TAG}_source_${safe_group}_gpu${gpu}.log" 2>&1 &
  source_pids+=("$!")
  if (( (index + 1) % GPU_COUNT == 0 )); then
    for pid in "${source_pids[@]}"; do wait "$pid"; done
    source_pids=()
  fi
done
for pid in "${source_pids[@]}"; do wait "$pid"; done

QUANT="$ROUND/quant_contracts"
mkdir -p "$QUANT"
quant_pids=()
quant_index=0
while IFS= read -r row; do
  width=$(jq -r '.width | map(tostring) | join("x")' <<<"$row")
  onnx=$(jq -r '.source_contract.onnx_path' <<<"$row")
  calibration=$(jq -r '.source_contract.calibration_npz' <<<"$row")
  summary=$(jq -r '.source_contract.calibration_summary' <<<"$row")
  output="$QUANT/$width/tensor_quant_params.json"
  mkdir -p "$(dirname "$output")"
  CUDA_VISIBLE_DEVICES="${GPUS[$((quant_index % GPU_COUNT))]}" "$PY" \
    scripts/stage3_tvm_int8_quant_contract_v3.py \
    --onnx "$onnx" --calibration-npz "$calibration" \
    --calibration-summary "$summary" --output-json "$output" \
    >"$LOGS/${TAG}_quant_contract_${width}.log" 2>&1 &
  quant_pids+=("$!")
  quant_index=$((quant_index + 1))
  if (( quant_index % GPU_COUNT == 0 )); then
    for pid in "${quant_pids[@]}"; do wait "$pid"; done
    quant_pids=()
  fi
done < <(jq -c '.rows[] | select(.dispatch_key=="tvm_auto" and .q_mode=="int8")' "$REQUEST")
for pid in "${quant_pids[@]}"; do wait "$pid"; done

PERF="$ROUND/performance"
mkdir -p "$PERF"
"$PY" scripts/stage5_build_performance_plan_v2.py \
  --request-json "$REQUEST" --remote-artifact-root "$ROUND/performance_execution" \
  --output-dir "$PERF" --quant-contract-root "$QUANT" --gpus "$GPU_POOL" \
  --tvm-fp16-max-trials "$TVM_FP16_MAX_TRIALS"
"$PY" scripts/stage3_execute_performance_plan_v3.py \
  --jobs-jsonl "$PERF/performance_jobs.jsonl" \
  --state-jsonl "$PERF/performance_state.jsonl" \
  --gpus "$GPU_POOL" --max-workers "$GPU_COUNT"

if [[ "$FIXED_BATCH_MODE" == "1" ]] && jq -s -e 'any(.[]; .status == "confirmed_failure")' \
  "$PERF/performance_state.jsonl" >/dev/null; then
  jq -s '{schema_version:"stage6_performance_failure_quarantine_v1",
    status:"quarantined_unresolved_performance_failure",
    failures:[.[] | select(.status == "confirmed_failure")]}' \
    "$PERF/performance_state.jsonl" >"$PERF/stage6_performance_failure_quarantine.json"
  echo "formal Stage6 performance failure quarantined before feedback release" >&2
  exit 1
fi

AP="$ROUND/ap"
mkdir -p "$AP"
if ! "$PY" scripts/stage5_ap_plan_v2.py \
  --manifest-json "$PERF/performance_manifest.json" \
  --performance-jobs-jsonl "$PERF/performance_jobs.jsonl" \
  --performance-state-jsonl "$PERF/performance_state.jsonl" \
  --output-root "$ROUND/ap_execution" \
  --output-json "$AP/ap_plan.json" --output-jsonl "$AP/ap_plan.jsonl"; then
  jq -s -e 'all(.[]; .ap_terminal == "ready" or .ap_terminal == "feasibility_failure")' \
    "$AP/ap_plan.jsonl" >/dev/null
fi

for shard in 0 1 2 3; do
  sed -n "$((shard + 1))p" "$AP/ap_plan.jsonl" | jq -c 'select(.ap_terminal == "ready")' \
    >"$AP/ap_plan_shard_${shard}.jsonl"
done

run_ap_stage() {
  local stage=$1 rc=0
  local stage_pids=() active_shards=()
  for shard in 0 1 2 3; do
    [[ -s "$AP/ap_plan_shard_${shard}.jsonl" ]] || continue
    local gpu=${GPUS[${#stage_pids[@]}]}
    "$PY" scripts/stage3_execute_ap_plan_v3.py \
      --ap-plan-jsonl "$AP/ap_plan_shard_${shard}.jsonl" --stage "$stage" \
      --state-jsonl "$AP/ap_state_shard_${shard}.jsonl" --gpu "$gpu" \
      --univ2x-python "$PY" --artifact-root "$ROUND/ap_execution/$stage/shard_${shard}" &
    stage_pids+=("$!")
    active_shards+=("$shard")
    if (( ${#stage_pids[@]} == GPU_COUNT )); then
      for pid in "${stage_pids[@]}"; do wait "$pid" || rc=1; done
      stage_pids=()
    fi
  done
  for pid in "${stage_pids[@]}"; do wait "$pid" || rc=1; done
  for shard in "${active_shards[@]}"; do
    jq -s -e --arg stage "$stage" '
      any(.[]; .record_type == "job_terminal" and .stage == $stage and .status == "success") or
      ($stage == "sanity" and any(.[];
        .record_type == "job_terminal" and .stage == "sanity" and .status == "failed" and
        .failure_reason == "numerical_feasibility_failure"
      )) or
      ($stage == "full" and any(.[];
        .record_type == "job_terminal" and .stage == "full" and
        .status == "skipped_numerical_feasibility" and
        .failure_reason == "numerical_feasibility_failure"
      ))
    ' "$AP/ap_state_shard_${shard}.jsonl" >/dev/null || rc=1
  done
  return "$rc"
}
run_ap_stage sanity
run_ap_stage full
: >"$AP/ap_state.jsonl.tmp"
for shard in 0 1 2 3; do
  [[ -f "$AP/ap_state_shard_${shard}.jsonl" ]] && cat "$AP/ap_state_shard_${shard}.jsonl" >>"$AP/ap_state.jsonl.tmp"
done
mv "$AP/ap_state.jsonl.tmp" "$AP/ap_state.jsonl"

FINAL="$ROUND/final"
"$PY" scripts/stage5_finalize_feedback_v2.py \
  --manifest-json "$PERF/performance_manifest.json" \
  --measurement-request-json "$REQUEST" --ap-plan-jsonl "$AP/ap_plan.jsonl" \
  --performance-state-jsonl "$PERF/performance_state.jsonl" \
  --ap-state-jsonl "$AP/ap_state.jsonl" --output-dir "$FINAL"
jq -e '.feedback_released == true and .budget_consumed == 4 and .batch_quarantined == false' \
  "$FINAL/atomic_batch_audit.json" >/dev/null

BATCH_FEEDBACK="$FINAL/stage5_feedback_v2_final.json"
if [[ "$FEEDBACK_CONTRACT" == "actual_v3" ]]; then
  ACTUAL="$ROUND/actual_feedback"
  "$PY" scripts/stage5_promote_actual_feedback_v3.py \
    --measurement-request-json "$REQUEST" \
    --feedback-json "$FINAL/stage5_feedback_v2_final.json" \
    --output-dir "$ACTUAL"
  jq -e '
    .promoted_row_count == 4 and .silent_surrogate_fallback_count == 0
  ' "$ACTUAL/actual_feedback_batch_audit_v3.json" >/dev/null
  BATCH_FEEDBACK="$ACTUAL/stage5_feedback_v3_actual.json"
fi

if [[ "$FIXED_BATCH_MODE" == "1" ]]; then
  jq -n --arg task "$TASK" --arg request "$REQUEST" --arg feedback "$BATCH_FEEDBACK" \
    --arg completed_at "$(date -Is)" --argjson round "$ROUND_INDEX" \
    --argjson tvm_trials "$TVM_FP16_MAX_TRIALS" \
    '{schema_version:"stage6_fixed_batch_terminal_v1",task_id:$task,round_index:$round,
      measurement_request:$request,feedback_path:$feedback,budget_consumed:4,
      tvm_fp16_max_trials:$tvm_trials,status:"complete",completed_at:$completed_at}' \
    >"$FINAL/stage6_fixed_batch_terminal.json"
  date -Is >"$CTRL/${TAG}.done"
  rm -f "$CTRL/${TAG}.failed"
  exit 0
fi

HISTORY="$ROOT/$TASK/feedback_history_through_round_$(printf '%02d' "$ROUND_INDEX").json"
"$PY" - "$ROOT" "$TASK" "$ROUND_INDEX" "$HISTORY" "$EXPECTED_MODEL" "$EXPECTED_DISPATCH" "$REQUEST" "$FEEDBACK_CONTRACT" <<'PY'
import json,sys
from pathlib import Path
root,task,round_index,out=Path(sys.argv[1]),sys.argv[2],int(sys.argv[3]),Path(sys.argv[4])
model,dispatch,request_path=sys.argv[5],sys.argv[6],Path(sys.argv[7])
feedback_contract=sys.argv[8]
request=json.loads(request_path.read_text())
expected={
    "task_id":task,
    "model":model,
    "hardware_id":"h800",
    "capability_profile_id":request["rows"][0]["capability_profile_id"],
    "dispatch_key":dispatch,
    "task_sha256":request["task_sha256"],
    "training_source":"online_feedback",
}
if feedback_contract == "actual_v3":
    expected["feedback_feature_contract"]="actual_feedback_v3"
rows=[]
for index in range(round_index+1):
    if feedback_contract == "actual_v3":
        path=root/task/f"round_{index:02d}"/"actual_feedback/stage5_feedback_v3_actual.json"
    else:
        path=root/task/f"round_{index:02d}"/"final/stage5_feedback_v2_final.json"
    payload=json.loads(path.read_text())
    if not isinstance(payload,list) or len(payload)!=4:
        raise ValueError(f"invalid feedback batch: {path}")
    rows.extend(payload)
if len(rows)!=(round_index+1)*4 or len({row["manifest_job_id"] for row in rows})!=len(rows):
    raise ValueError("cumulative feedback identity/count drift")
for row in rows:
    for field,value in expected.items():
        if str(row.get(field)) != str(value):
            raise ValueError(f"cumulative feedback {field} drift")
    if row.get("terminal_status") not in {
        "measured_success_gold", "feasibility_failure", "numerical_feasibility_failure"
    }:
        raise ValueError("cumulative feedback contains a public/non-terminal failure")
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
PY

if [[ "$ROUND_INDEX" -lt 3 ]]; then
  next_round=$((ROUND_INDEX + 1))
  "$PY" scripts/stage5_advance_task_round_v2.py \
    --task-id "$TASK" --feedback-json "$HISTORY" --output-dir "$ROOT" \
    --round-index "$next_round"
else
  jq -n --arg task "$TASK" --arg feedback "$HISTORY" \
    --arg feedback_contract "$FEEDBACK_CONTRACT" \
    --arg completed_at "$(date -Is)" \
    '{schema_version:"stage5_task_budget_terminal_v3",task_id:$task,
      budget_consumed:16,round_count:4,feedback_history:$feedback,
      feedback_contract:$feedback_contract,
      status:"budget_exhausted",completed_at:$completed_at}' \
    >"$ROOT/$TASK/task_budget_terminal.json"
fi

date -Is >"$CTRL/${TAG}.done"
rm -f "$CTRL/${TAG}.failed"
