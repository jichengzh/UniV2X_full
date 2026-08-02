#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
JOINT_ROOT=${JOINT_ROOT:-$REPO/results/stage5_pyramid_actual_v3_20260720}
JOINT_VALIDATION_ROOT=${JOINT_VALIDATION_ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
COLDSTART_ROWS=${COLDSTART_ROWS:-$REPO/results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1/gold176_final.json}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
POLL_SECONDS=${POLL_SECONDS:-30}

CONTROLLER=$FORMAL_ROOT/controller
QUEUE_DONE=$CONTROLLER/stage6_resume_parallel_queues_v1.done
QUEUE_FAILED=$CONTROLLER/stage6_resume_parallel_queues_v1.failed
OUTPUT_ROOT=$FORMAL_ROOT/paper_evidence
INITIAL_BUNDLE=$OUTPUT_ROOT/stage6_paper_evidence_pre_validation_v1.json
FINAL_BUNDLE=$OUTPUT_ROOT/stage6_paper_evidence_bundle_v1.json
VALIDATION_ROOT=$FORMAL_ROOT/independent_validation_v1
PLAN=$VALIDATION_ROOT/stage6_independent_validation_plan_v1.json
SCHEDULE_AUDIT=$VALIDATION_ROOT/stage6_schedule_independent_validation_audit_v1.json
JOINT_SOURCE_AUDIT=$JOINT_VALIDATION_ROOT/independent_validation_v1/stage5_independent_validation_audit_v1.json
JOINT_AUDIT=$VALIDATION_ROOT/stage6_joint_pyramid_independent_validation_audit_v1.json
TABLE_ROOT=$FORMAL_ROOT/paper_tables
DONE=$CONTROLLER/stage6_paper_closure_supervisor_v1.done
FAILED=$CONTROLLER/stage6_paper_closure_supervisor_v1.failed

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]
cd "$REPO"
mkdir -p "$OUTPUT_ROOT" "$VALIDATION_ROOT" "$TABLE_ROOT"
rm -f "$DONE" "$FAILED"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$FAILED"; fi' EXIT

"$PY" scripts/stage6_scope_joint_validation_audit_v1.py \
  --source-audit "$JOINT_SOURCE_AUDIT" --output-json "$JOINT_AUDIT" \
  --task-id S5-PYR-TVM --task-id S5-PYR-TRT

audit_matches_plan() {
  local plan_json=$1 audit_json=$2
  [[ -s "$audit_json" ]] || return 1
  "$PY" - "$plan_json" "$audit_json" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
plans = plan.get("plans") or []
tasks = audit.get("tasks") or []
if len(plans) != 1 or len(tasks) != 1 or audit.get("all_tasks_passed") is not True:
    raise SystemExit(1)
expected, actual = plans[0], tasks[0]
expected_ids = sorted(str(value) for value in expected.get("configuration_ids") or [])
audit_ids = sorted(
    str(item.get("configuration_id") or "")
    for item in actual.get("configurations") or []
)
valid = (
    actual.get("backend") == expected.get("backend")
    and actual.get("arm_id") == expected.get("arm_id")
    and expected_ids == audit_ids
    and bool(expected_ids)
)
raise SystemExit(0 if valid else 1)
PY
}

while [[ ! -s "$QUEUE_DONE" ]]; do
  [[ ! -s "$QUEUE_FAILED" ]] || {
    echo "Stage6 raw queue failed: $QUEUE_FAILED" >&2
    exit 1
  }
  sleep "$POLL_SECONDS"
done

"$PY" scripts/stage6_collect_paper_evidence_v1.py \
  --formal-root "$FORMAL_ROOT" --joint-root "$JOINT_ROOT" \
  --coldstart-rows-json "$COLDSTART_ROWS" --independent-audit "$JOINT_AUDIT" \
  --output-json "$INITIAL_BUNDLE"

rm -f "$VALIDATION_ROOT"/stage6_independent_validation_plan_??.json
"$PY" scripts/stage6_prepare_independent_validation_v1.py \
  --formal-root "$FORMAL_ROOT" --evidence-bundle "$INITIAL_BUNDLE" \
  --output-root "$VALIDATION_ROOT" --independent-audit "$JOINT_AUDIT" \
  --gpus 3,4,5,6,7

plan_count=$(jq -r '.plan_count' "$PLAN")
if [[ "$plan_count" -gt 4 ]]; then
  echo "too many independent plans for the frozen GPU pool: $plan_count" >&2
  exit 1
fi

validation_pids=()
validation_audits=()
for ((index=0; index<plan_count; index++)); do
  split_plan=$(printf "%s/stage6_independent_validation_plan_%02d.json" "$VALIDATION_ROOT" "$index")
  audit=$(printf "%s/stage6_independent_validation_audit_%02d.json" "$VALIDATION_ROOT" "$index")
  gpu=$((3 + index))
  validation_audits+=("$audit")
  if audit_matches_plan "$split_plan" "$audit"; then
    echo "reuse matching independent audit: $audit"
    continue
  fi
  env PLAN_JSON="$split_plan" GPU_POOL="$gpu" AUDIT_JSON="$audit" \
    bash scripts/stage6_run_independent_validation_v1.sh \
    >"$VALIDATION_ROOT/validation_${index}.log" 2>&1 &
  validation_pids+=("$!")
done

schedule_pid=
if jq -e '[.backends[].schedule_only.status] | any(. == "complete")' \
  "$INITIAL_BUNDLE" >/dev/null; then
  schedule_gpu=$((3 + plan_count))
  env EVIDENCE_BUNDLE="$INITIAL_BUNDLE" AUDIT_JSON="$SCHEDULE_AUDIT" \
    TRT_GPU="$schedule_gpu" TVM_GPU="$schedule_gpu" \
    bash scripts/stage6_run_schedule_independent_v1.sh \
    >"$VALIDATION_ROOT/schedule_only.log" 2>&1 &
  schedule_pid=$!
fi

for pid in "${validation_pids[@]}"; do wait "$pid"; done
[[ -z "$schedule_pid" ]] || wait "$schedule_pid"

audit_args=(--independent-audit "$JOINT_AUDIT")
[[ -z "$schedule_pid" ]] || audit_args+=(--independent-audit "$SCHEDULE_AUDIT")
for audit in "${validation_audits[@]}"; do
  audit_args+=(--independent-audit "$audit")
done
"$PY" scripts/stage6_collect_paper_evidence_v1.py \
  --formal-root "$FORMAL_ROOT" --joint-root "$JOINT_ROOT" \
  --coldstart-rows-json "$COLDSTART_ROWS" "${audit_args[@]}" \
  --output-json "$FINAL_BUNDLE"

"$PY" scripts/stage6_write_terminal_summaries_v1.py \
  --evidence-bundle "$FINAL_BUNDLE" --formal-root "$FORMAL_ROOT"
"$PY" scripts/stage6_formal_runner_v1.py \
  --plan "$FORMAL_ROOT/stage6_formal_execution_plan_v1.json" \
  --output-root "$FORMAL_ROOT" \
  --state-out "$FORMAL_ROOT/stage6_formal_runner_state_v1.json"
"$PY" - "$FORMAL_ROOT/stage6_formal_runner_state_v1.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
assert state["slot_count"] == 11
assert state["complete_slot_count"] == 11
assert state["paper_table_ready"] is True
PY

"$PY" scripts/stage6_build_paper_main_tables_v1.py \
  --evidence-bundle "$FINAL_BUNDLE" --output-dir "$TABLE_ROOT"

date -Is >"$DONE"
rm -f "$FAILED"
trap - EXIT
