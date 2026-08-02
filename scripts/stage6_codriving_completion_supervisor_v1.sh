#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
JOINT_ROOT=${JOINT_ROOT:-$REPO/results/stage5_codriving_actual_v3_20260721}
COLDSTART_ROWS=${COLDSTART_ROWS:-$REPO/results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1/gold176_final.json}
GPU_POOL=${GPU_POOL:-1,3,5,6}
POLL_SECONDS=${POLL_SECONDS:-60}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}

cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller" "$FORMAL_ROOT/paper_evidence" \
  "$FORMAL_ROOT/paper_tables" "$FORMAL_ROOT/independent_validation_controls"

wait_file() {
  local path=$1
  while [[ ! -s "$path" ]]; do sleep "$POLL_SECONDS"; done
}

wait_gpu_idle() {
  local gpu=$1 stable=0
  while (( stable < 3 )); do
    if nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits | awk -F, -v wanted="$gpu" '
        $1 + 0 == wanted + 0 {
          seen=1; memory=$2+0; utilization=$3+0
          if (memory <= 100 && utilization <= 10) ok=1
        }
        END { exit !(seen && ok) }
      '; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    (( stable == 3 )) || sleep "$POLL_SECONDS"
  done
}

dependencies=()
for backend in tvm trt; do
  if [[ "$backend" == tvm ]]; then task=S5-COD-TVM; else task=S5-COD-TRT; fi
  for batch in 00 01 02 03; do
    dependencies+=("$FORMAL_ROOT/$backend/compression_only/formal_batch_$batch/$task/round_$batch/final/stage6_fixed_batch_terminal.json")
  done
  dependencies+=("$FORMAL_ROOT/$backend/compress_then_tune/formal_batch_00/$task/round_00/final/stage6_fixed_batch_terminal.json")
  dependencies+=("$FORMAL_ROOT/controller/${backend}_codriving_backend_tail_queue.done")
done
dependencies+=(
  "$FORMAL_ROOT/independent_validation_joint/stage6_joint_independent_validation_audit_tvm_v1.json"
  "$FORMAL_ROOT/independent_validation_joint/stage6_joint_independent_validation_audit_trt_v1.json"
  "$FORMAL_ROOT/independent_validation_schedule/stage6_schedule_independent_validation_audit_v1.json"
  "$FORMAL_ROOT/controller/stage6_controls_parallel.done"
)
for path in "${dependencies[@]}"; do wait_file "$path"; done

joint_tvm="$FORMAL_ROOT/independent_validation_joint/stage6_joint_independent_validation_audit_tvm_v1.json"
joint_trt="$FORMAL_ROOT/independent_validation_joint/stage6_joint_independent_validation_audit_trt_v1.json"
schedule="$FORMAL_ROOT/independent_validation_schedule/stage6_schedule_independent_validation_audit_v1.json"
intermediate="$FORMAL_ROOT/paper_evidence/stage6_codriving_paper_evidence_bundle_pre_validation_v1.json"

"$PY" scripts/stage6_collect_codriving_paper_evidence_v1.py \
  --formal-root "$FORMAL_ROOT" --joint-root "$JOINT_ROOT" \
  --coldstart-rows-json "$COLDSTART_ROWS" \
  --independent-audit "$joint_tvm" --independent-audit "$joint_trt" \
  --independent-audit "$schedule" --output-json "$intermediate"

validation_root="$FORMAL_ROOT/independent_validation_controls"
existing_control_audits=("$FORMAL_ROOT"/independent_validation_controls_parallel/stage6_independent_validation_audit_*.json)
if [[ ! -e "${existing_control_audits[0]}" ]]; then existing_control_audits=(); fi
prepare=("$PY" scripts/stage6_prepare_codriving_independent_validation_v1.py \
  --formal-root "$FORMAL_ROOT" --evidence-bundle "$intermediate" \
  --output-root "$validation_root" --independent-audit "$joint_tvm" \
  --independent-audit "$joint_trt" --independent-audit "$schedule" \
  --gpus "$GPU_POOL")
for audit in "${existing_control_audits[@]}"; do prepare+=(--independent-audit "$audit"); done
"${prepare[@]}"

plan="$validation_root/stage6_independent_validation_plan_v1.json"
plan_count=$(jq -r '.plan_count' "$plan")
IFS=',' read -r -a gpus <<<"$GPU_POOL"
for ((index=0; index<plan_count; index++)); do
  wait_gpu_idle "${gpus[$((index % ${#gpus[@]}))]}"
done
pids=()
control_audits=("${existing_control_audits[@]}")
for ((index=0; index<plan_count; index++)); do
  scoped="$validation_root/stage6_independent_validation_plan_$(printf '%02d' "$index").json"
  audit="$validation_root/stage6_independent_validation_audit_$(printf '%02d' "$index").json"
  gpu=${gpus[$((index % ${#gpus[@]}))]}
  control_audits+=("$audit")
  env REPO="$REPO" PLAN_JSON="$scoped" GPU_POOL="$gpu" AUDIT_JSON="$audit" \
    bash scripts/stage6_run_independent_validation_v1.sh \
    >"$validation_root/runner_$(printf '%02d' "$index").log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for audit in "${control_audits[@]}"; do wait_file "$audit"; done

final_bundle="$FORMAL_ROOT/paper_evidence/stage6_codriving_paper_evidence_bundle_v1.json"
collect=(
  "$PY" scripts/stage6_collect_codriving_paper_evidence_v1.py
  --formal-root "$FORMAL_ROOT" --joint-root "$JOINT_ROOT"
  --coldstart-rows-json "$COLDSTART_ROWS"
  --independent-audit "$joint_tvm" --independent-audit "$joint_trt"
  --independent-audit "$schedule"
)
for audit in "${control_audits[@]}"; do collect+=(--independent-audit "$audit"); done
"${collect[@]}" --output-json "$final_bundle"

"$PY" scripts/stage6_build_codriving_paper_main_tables_v1.py \
  --evidence-bundle "$final_bundle" --output-dir "$FORMAL_ROOT/paper_tables" \
  --delta-ap-max 0.05 0.10
"$PY" scripts/stage6_audit_codriving_completion_v1.py \
  --formal-root "$FORMAL_ROOT" --joint-root "$JOINT_ROOT" \
  --evidence-bundle "$final_bundle" --table-dir "$FORMAL_ROOT/paper_tables" \
  --output-json "$FORMAL_ROOT/paper_evidence/stage6_codriving_completion_audit_v1.json"

date -Is >"$FORMAL_ROOT/controller/stage6_codriving_completion_supervisor.done"
