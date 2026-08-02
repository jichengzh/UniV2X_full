#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_ARGS=("$@")
TASK_ID=S5-FCO-TRT-V2
PILOT_FRAGMENT=fcooper_workpackage_a_20260723
SUCCESS=measured_success_gold
MAX_PARALLEL=4
REPEAT_GPU=7
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORMAL_ROOT=
SOURCE_REGISTRY_JSON=
OBSERVED_GRAPHS_JSON=
COLDSTART_ROWS_JSON=
COLDSTART_GRAPHS_JSON=
PROFILES_JSON=
FROZEN_CONTRACT_JSON=
ARTIFACT_ROOT=
HEAL_ROOT=
PYTHON=
SOURCE_CONFIG=
SOURCE_CHECKPOINT=
RECOVERY_CONTRACT=
CALIBRATION_DIR=
CALIBRATION_SUMMARY=
GPUS_CSV=0,1,2,3
MAX_RETRIES=2
DRY_RUN=0
REPEAT_GPU_WAIT_TIMEOUT_SECONDS=${FCOOPER_REPEAT_GPU_WAIT_TIMEOUT_SECONDS:-43200}
REPEAT_GPU_POLL_SECONDS=${FCOOPER_REPEAT_GPU_POLL_SECONDS:-30}
REPEAT_GPU_QUIET_SECONDS=${FCOOPER_REPEAT_GPU_QUIET_SECONDS:-15}
REPEAT_GPU_MONITOR_SECONDS=${FCOOPER_REPEAT_GPU_MONITOR_SECONDS:-0.5}
REPEAT_GPU_RUNTIME_TIMEOUT_SECONDS=${FCOOPER_REPEAT_GPU_RUNTIME_TIMEOUT_SECONDS:-7200}

usage() {
  cat <<'EOF'
Usage: fcooper_stage6_five_arm_supervisor_v2.sh [options]

Required:
  --formal-root PATH
  --source-registry-json PATH
  --observed-graphs-json PATH
  --coldstart-rows-json PATH
  --coldstart-graphs-json PATH
  --profiles-json PATH
  --frozen-contract-json PATH
  --artifact-root PATH
  --heal-root PATH
  --python PATH
  --source-config PATH
  --source-checkpoint PATH
  --recovery-contract PATH
  --calibration-dir PATH
  --calibration-summary PATH

Optional:
  --code-root PATH
  --gpus CSV              One to four control GPUs; GPU7 is reserved
  --max-retries N         Infrastructure retries per row, 0..2
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root) FORMAL_ROOT=$2; shift 2 ;;
    --source-registry-json) SOURCE_REGISTRY_JSON=$2; shift 2 ;;
    --observed-graphs-json) OBSERVED_GRAPHS_JSON=$2; shift 2 ;;
    --coldstart-rows-json) COLDSTART_ROWS_JSON=$2; shift 2 ;;
    --coldstart-graphs-json) COLDSTART_GRAPHS_JSON=$2; shift 2 ;;
    --profiles-json) PROFILES_JSON=$2; shift 2 ;;
    --frozen-contract-json) FROZEN_CONTRACT_JSON=$2; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT=$2; shift 2 ;;
    --code-root) CODE_ROOT=$2; shift 2 ;;
    --heal-root) HEAL_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --source-config) SOURCE_CONFIG=$2; shift 2 ;;
    --source-checkpoint) SOURCE_CHECKPOINT=$2; shift 2 ;;
    --recovery-contract) RECOVERY_CONTRACT=$2; shift 2 ;;
    --calibration-dir) CALIBRATION_DIR=$2; shift 2 ;;
    --calibration-summary) CALIBRATION_SUMMARY=$2; shift 2 ;;
    --gpus) GPUS_CSV=$2; shift 2 ;;
    --max-retries) MAX_RETRIES=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

REQUIRED_PATHS=(
  "$FORMAL_ROOT" "$SOURCE_REGISTRY_JSON" "$OBSERVED_GRAPHS_JSON"
  "$COLDSTART_ROWS_JSON" "$COLDSTART_GRAPHS_JSON" "$PROFILES_JSON"
  "$FROZEN_CONTRACT_JSON" "$ARTIFACT_ROOT" "$CODE_ROOT" "$HEAL_ROOT"
  "$PYTHON" "$SOURCE_CONFIG" "$SOURCE_CHECKPOINT" "$RECOVERY_CONTRACT"
  "$CALIBRATION_DIR" "$CALIBRATION_SUMMARY"
)
for path in "${REQUIRED_PATHS[@]}"; do
  [[ -n "$path" ]] || { echo "all required path arguments must be set" >&2; exit 2; }
  if [[ "$path" == *"$PILOT_FRAGMENT"* ]]; then
    echo "Stage6 formal controls reject old pilot path: $path" >&2
    exit 2
  fi
done
mkdir -p "$FORMAL_ROOT/controls"
SUPERVISOR_LOCK="$FORMAL_ROOT/controls/five_arm_supervisor.lock"
if [[ "${FCOOPER_STAGE6_SUPERVISOR_LOCKED:-0}" != "1" ]]; then
  exec flock -n -E 75 "$SUPERVISOR_LOCK" \
    env FCOOPER_STAGE6_SUPERVISOR_LOCKED=1 "$0" "${ORIGINAL_ARGS[@]}"
fi
printf '%s\n' "$$" >"$FORMAL_ROOT/controls/five_arm_supervisor.pid"
EXPECTED_FROZEN_CONTRACT_JSON="$FORMAL_ROOT/contracts/frozen_contract.json"
if [[ "$(realpath -m "$FROZEN_CONTRACT_JSON")" != \
      "$(realpath -m "$EXPECTED_FROZEN_CONTRACT_JSON")" ]]; then
  echo "--frozen-contract-json must be $EXPECTED_FROZEN_CONTRACT_JSON" >&2
  exit 2
fi
[[ "$MAX_RETRIES" =~ ^[0-2]$ ]] || {
  echo "--max-retries must be 0, 1, or 2" >&2
  exit 2
}
IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
if (( ${#GPUS[@]} < 1 || ${#GPUS[@]} > MAX_PARALLEL )); then
  echo "--gpus must provide between one and four GPU identifiers" >&2
  exit 2
fi
declare -A SEEN_GPUS=()
for gpu_index in "${!GPUS[@]}"; do
  gpu=${GPUS[$gpu_index]}
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU identifier: $gpu" >&2; exit 2; }
  gpu=$((10#$gpu))
  GPUS[$gpu_index]=$gpu
  [[ "$gpu" != "$REPEAT_GPU" ]] || {
    echo "GPU7 is reserved for independent selected-arm repeats" >&2
    exit 2
  }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || {
    echo "control GPU identifiers must be unique" >&2
    exit 2
  }
  SEEN_GPUS[$gpu]=1
done

CONTROL_TOOL="$CODE_ROOT/scripts/fcooper_stage6_control_request_v2.py"
PREPARE_TOOL="$CODE_ROOT/scripts/stage6_prepare_fcooper_five_arm_v2.py"
RUNNER="$CODE_ROOT/scripts/fcooper_execute_measurement_row_v2.py"
NATIVE_RUNNER="$CODE_ROOT/scripts/fcooper_native_scope_benchmark_v1.py"
TRT_RUNNER="$CODE_ROOT/framework/trt_baseline/trt_profile_v1.py"
AP_RUNNER="$CODE_ROOT/scripts/fcooper_trt_ap_bridge_v1.py"
FINALIZER="$CODE_ROOT/scripts/stage6_finalize_fcooper_table1_v2.py"
GPU_EXCLUSIVITY_GATE="$CODE_ROOT/scripts/fcooper_gpu_exclusivity_gate_v1.py"
PLAN_ROOT="$FORMAL_ROOT/controls/five_arm_plan"
T16_FEEDBACK="$FORMAL_ROOT/search/$TASK_ID/feedback_history_final_t16.json"
GEAR_SELECTION="$FORMAL_ROOT/search/$TASK_ID/gear_selected_after_final_t16.json"
LOG_ROOT="$FORMAL_ROOT/controls/logs"
GPU_EXCLUSIVITY_ROOT="$FORMAL_ROOT/controls/resource_audit/gpu7_exclusivity"

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

prepare_command() {
  PREPARE_COMMAND=(
    "$PYTHON" "$PREPARE_TOOL"
    --source-registry-json "$SOURCE_REGISTRY_JSON"
    --observed-graphs-json "$OBSERVED_GRAPHS_JSON"
    --coldstart-rows-json "$COLDSTART_ROWS_JSON"
    --coldstart-graphs-json "$COLDSTART_GRAPHS_JSON"
    --profiles-json "$PROFILES_JSON"
    --frozen-contract-json "$FROZEN_CONTRACT_JSON"
    --output-dir "$PLAN_ROOT"
  )
  if [[ -n "${1:-}" ]]; then
    PREPARE_COMMAND+=(
      --compress-then-tune-screen-feedback-json "$1"
    )
  fi
}

build_requests() {
  local candidate_plan=$1 arm_id=$2 phase=$3 builder_level=$4 output_dir=$5
  "$PYTHON" "$CONTROL_TOOL" build \
    --candidate-plan-json "$candidate_plan" \
    --five-arm-plan-json "$PLAN_ROOT/stage6_fcooper_five_arm_plan.json" \
    --arm-id "$arm_id" \
    --phase "$phase" \
    --builder-optimization-level "$builder_level" \
    --output-dir "$output_dir"
}

run_control_row() {
  local request=$1 row_index=$2 gpu=$3 phase_tag=$4
  local attempt
  for ((attempt = 0; attempt <= MAX_RETRIES; attempt++)); do
    local log="$LOG_ROOT/${phase_tag}_row_${row_index}_attempt_${attempt}.log"
    if "$PYTHON" "$RUNNER" \
      --request-kind stage6-control \
      --request-json "$request" \
      --row-index "$row_index" \
      --gpu "$gpu" \
      --artifact-root "$ARTIFACT_ROOT" \
      --code-root "$CODE_ROOT" \
      --heal-root "$HEAL_ROOT" \
      --python "$PYTHON" \
      --source-config "$SOURCE_CONFIG" \
      --source-checkpoint "$SOURCE_CHECKPOINT" \
      --recovery-contract "$RECOVERY_CONTRACT" \
      --calibration-dir "$CALIBRATION_DIR" \
      --calibration-summary "$CALIBRATION_SUMMARY" \
      --builder-optimization-level "$5" >"$log" 2>&1; then
      return 0
    fi
  done
  echo "control row exhausted retries: $phase_tag row=$row_index" >&2
  return 1
}

run_request_manifest() {
  local manifest=$1 phase_tag=$2 builder_level=$3
  local requests=()
  mapfile -t requests < <(
    "$PYTHON" "$CONTROL_TOOL" list-requests --request-manifest-json "$manifest"
  )
  [[ ${#requests[@]} -gt 0 ]] || {
    echo "request manifest is empty: $manifest" >&2
    return 1
  }
  local request_index=0
  for request in "${requests[@]}"; do
    if "$PYTHON" "$CONTROL_TOOL" validate-request-complete \
      --request-json "$request" \
      --artifact-root "$ARTIFACT_ROOT" \
      >"$LOG_ROOT/${phase_tag}_request_${request_index}_reuse_probe.log" 2>&1; then
      echo "reusing complete control request: $phase_tag request=$request_index"
      request_index=$((request_index + 1))
      continue
    fi
    local row_count
    row_count=$(
      "$PYTHON" "$CONTROL_TOOL" request-row-count --request-json "$request"
    )
    if (( row_count < 1 || row_count > MAX_PARALLEL )); then
      echo "control request row count is outside 1..4: $row_count" >&2
      return 1
    fi
    local worker_pids=()
    local row_index
    for ((row_index = 0; row_index < row_count; row_index++)); do
      if "$PYTHON" "$CONTROL_TOOL" validate-request-row-complete \
        --request-json "$request" \
        --row-index "$row_index" \
        --artifact-root "$ARTIFACT_ROOT" \
        >"$LOG_ROOT/${phase_tag}_request_${request_index}_row_${row_index}_reuse_probe.log" 2>&1; then
        echo "reusing complete control row: $phase_tag request=$request_index row=$row_index"
        continue
      fi
      local gpu=${GPUS[$((row_index % ${#GPUS[@]}))]}
      run_control_row \
        "$request" "$row_index" "$gpu" \
        "${phase_tag}_request_${request_index}" "$builder_level" &
      worker_pids+=("$!")
    done
    local failed=0 pid
    for pid in "${worker_pids[@]}"; do
      if ! wait "$pid"; then failed=1; fi
    done
    (( failed == 0 )) || return 1
    request_index=$((request_index + 1))
  done
}

aggregate_phase() {
  local manifest=$1 expected_count=$2 feedback_json=$3 audit_json=$4
  "$PYTHON" "$CONTROL_TOOL" aggregate \
    --request-manifest-json "$manifest" \
    --artifact-root "$ARTIFACT_ROOT" \
    --expected-count "$expected_count" \
    --feedback-json "$feedback_json" \
    --integrity-audit-json "$audit_json"
}

ensure_phase_feedback() {
  local manifest=$1 phase_tag=$2 builder_level=$3 expected_count=$4
  local feedback_json=$5 audit_json=$6
  if aggregate_phase \
    "$manifest" "$expected_count" "$feedback_json" "$audit_json" \
    >"$LOG_ROOT/${phase_tag}_reuse_probe.log" 2>&1; then
    echo "reusing complete control phase: $phase_tag"
    return 0
  fi
  run_request_manifest "$manifest" "$phase_tag" "$builder_level"
  aggregate_phase "$manifest" "$expected_count" "$feedback_json" "$audit_json"
}

select_arm() {
  local feedback_json=$1 expected_count=$2 task_id=$3 output_json=$4
  "$PYTHON" "$CONTROL_TOOL" select \
    --feedback-json "$feedback_json" \
    --expected-count "$expected_count" \
    --contract-json "$FROZEN_CONTRACT_JSON" \
    --required-task-id "$task_id" \
    --output-json "$output_json"
}

run_repeat_guarded() {
  local label=$1 log_file=$2
  shift 2
  "$PYTHON" "$GPU_EXCLUSIVITY_GATE" guard \
    --gpu-index "$REPEAT_GPU" \
    --timeout-seconds "$REPEAT_GPU_WAIT_TIMEOUT_SECONDS" \
    --poll-seconds "$REPEAT_GPU_POLL_SECONDS" \
    --quiet-seconds "$REPEAT_GPU_QUIET_SECONDS" \
    --monitor-seconds "$REPEAT_GPU_MONITOR_SECONDS" \
    --runtime-timeout-seconds "$REPEAT_GPU_RUNTIME_TIMEOUT_SECONDS" \
    --lock-file "$GPU_EXCLUSIVITY_ROOT/gpu7.guard.lock" \
    --audit-json "$GPU_EXCLUSIVITY_ROOT/${label}_guard.json" \
    --log-file "$log_file" \
    "$@"
}

run_native_repeats() {
  local repeat_root="$FORMAL_ROOT/controls/original_default/independent_repeats"
  local row_root="$repeat_root/fcooper-original-default"
  mkdir -p "$row_root"
  local repeat
  for repeat in 0 1 2; do
    local output="$row_root/same_gpu7_repeat_${repeat}.json"
    if [[ -f "$output" ]]; then
      [[ -f "$GPU_EXCLUSIVITY_ROOT/original_default_repeat_${repeat}_guard.json" ]] || {
        echo "native repeat exists without GPU7 guard audit: $output" >&2
        return 1
      }
      "$PYTHON" "$CONTROL_TOOL" bind-native-report \
        --report-json "$output" \
        --contract-json "$FROZEN_CONTRACT_JSON" \
        --gpu "$REPEAT_GPU" >/dev/null
      echo "reusing validated native repeat: $output"
      continue
    fi
    run_repeat_guarded \
      "original_default_repeat_${repeat}" \
      "$LOG_ROOT/original_native_repeat_${repeat}.log" \
      --env "CUDA_VISIBLE_DEVICES=$REPEAT_GPU" \
      --env "PYTHONPATH=$CODE_ROOT:$HEAL_ROOT" \
      -- "$PYTHON" "$NATIVE_RUNNER" \
      --config "$SOURCE_CONFIG" \
      --checkpoint "$SOURCE_CHECKPOINT" \
      --nvml-gpu "$REPEAT_GPU" \
      --output-json "$output"
    "$PYTHON" "$CONTROL_TOOL" bind-native-report \
      --report-json "$output" \
      --contract-json "$FROZEN_CONTRACT_JSON" \
      --gpu "$REPEAT_GPU"
  done
}

run_trt_repeats() {
  local selection_json=$1 repeat_root=$2 builder_level=$3 label=$4
  local spec=()
  mapfile -t spec < <(
    "$PYTHON" "$CONTROL_TOOL" repeat-spec --selection-json "$selection_json"
  )
  if [[ "${spec[0]}" != "$SUCCESS" ]]; then
    echo "$label selected a credible terminal failure; no TRT repeats required"
    return 0
  fi
  local row_id=${spec[1]} q_mode=${spec[2]} onnx_path=${spec[3]}
  local onnx_sha256=${spec[4]} config_path=${spec[5]} checkpoint_path=${spec[6]}
  local repeat
  for repeat in 0 1 2; do
    local output_dir="$repeat_root/$row_id/same_gpu7_repeat_${repeat}"
    local performance_json="$output_dir/performance.json"
    if [[ -f "$performance_json" ]]; then
      [[ -f "$GPU_EXCLUSIVITY_ROOT/${label}_repeat_${repeat}_guard.json" ]] || {
        echo "TRT repeat exists without GPU7 guard audit: $performance_json" >&2
        return 1
      }
      "$PYTHON" "$CONTROL_TOOL" validate-trt-repeat \
        --report-json "$performance_json" \
        --expected-onnx-sha256 "$onnx_sha256" \
        --expected-precision "$q_mode" \
        --expected-builder-level "$builder_level" \
        --gpu "$REPEAT_GPU" \
        --artifact-dir "$output_dir/engine" >/dev/null
      echo "reusing validated TRT repeat: $performance_json"
      continue
    fi
    [[ ! -e "$output_dir" ]] || {
      echo "refusing incomplete TRT repeat directory: $output_dir" >&2
      return 1
    }
    mkdir -p "$output_dir"
    run_repeat_guarded \
      "${label}_repeat_${repeat}" \
      "$LOG_ROOT/${label}_same_gpu7_repeat_${repeat}.log" \
      -- "$PYTHON" "$TRT_RUNNER" \
      --onnx "$onnx_path" \
      --precision "$q_mode" \
      --gpu "$REPEAT_GPU" \
      --calib-dir "$CALIBRATION_DIR" \
      --calibration-dataset OPV2V-validate \
      --builder-optimization-level "$builder_level" \
      --warmup 20 \
      --iters 300 \
      --repeat 5 \
      --energy-secs 5 \
      --artifact-dir "$output_dir/engine" \
      --out "$output_dir/performance.json"
  done
  local row_root="$repeat_root/$row_id"
  local ap_output="$row_root/independent_ap_report.json"
  if [[ -f "$ap_output" ]]; then
    [[ -f "$GPU_EXCLUSIVITY_ROOT/${label}_full_ap_guard.json" ]] || {
      echo "independent AP exists without GPU7 guard audit: $ap_output" >&2
      return 1
    }
    "$PYTHON" "$CONTROL_TOOL" validate-independent-ap \
      --report-json "$ap_output" \
      --selection-json "$selection_json" \
      --engine-path "$row_root/same_gpu7_repeat_0/engine/compiled.engine" \
      --config-path "$config_path" \
      --checkpoint-path "$checkpoint_path" >/dev/null
    echo "reusing validated independent AP: $ap_output"
    return 0
  fi
  run_repeat_guarded \
    "${label}_full_ap" \
    "$LOG_ROOT/${label}_independent_full_ap.log" \
    --cwd "$HEAL_ROOT" \
    --env "CUDA_VISIBLE_DEVICES=$REPEAT_GPU" \
    --env "PYTHONPATH=$CODE_ROOT:$HEAL_ROOT" \
    -- "$PYTHON" "$AP_RUNNER" \
      --config "$config_path" \
      --checkpoint-dir "$(dirname "$checkpoint_path")" \
      --checkpoint "$checkpoint_path" \
      --engine "$row_root/same_gpu7_repeat_0/engine/compiled.engine" \
      --output-json "$ap_output" \
      --num-workers 4
}

prepare_command
if (( DRY_RUN == 1 )); then
  print_command "$PYTHON" "$CONTROL_TOOL" validate-t16 \
    --feedback-json "$T16_FEEDBACK" \
    --contract-json "$FROZEN_CONTRACT_JSON" \
    --output-json "$GEAR_SELECTION"
  print_command "$PYTHON" "$CONTROL_TOOL" validate-original-contract \
    --contract-json "$FROZEN_CONTRACT_JSON" \
    --source-config "$SOURCE_CONFIG" \
    --source-checkpoint "$SOURCE_CHECKPOINT"
  print_command "${PREPARE_COMMAND[@]}"
  print_command "$PYTHON" "$GPU_EXCLUSIVITY_GATE" guard \
    --gpu-index "$REPEAT_GPU" \
    --timeout-seconds "$REPEAT_GPU_WAIT_TIMEOUT_SECONDS" \
    --poll-seconds "$REPEAT_GPU_POLL_SECONDS" \
    --quiet-seconds "$REPEAT_GPU_QUIET_SECONDS" \
    --monitor-seconds "$REPEAT_GPU_MONITOR_SECONDS" \
    --runtime-timeout-seconds "$REPEAT_GPU_RUNTIME_TIMEOUT_SECONDS" \
    --lock-file "$GPU_EXCLUSIVITY_ROOT/gpu7.guard.lock" \
    --audit-json "$GPU_EXCLUSIVITY_ROOT/<label>_guard.json" \
    --log-file "$LOG_ROOT/<label>.log" \
    -- '<repeat-command>'
  printf '# Original: three native repeats on physical GPU7; AP remains frozen-reference evidence\n'
  printf '# Schedule-only: 1 row, builder5, control task ID, then selected-row GPU7 repeats\n'
  printf '# Compression-only: 16 rows in 1..4-row requests, builder0, then selection and GPU7 repeats\n'
  printf '# Compress->Tune: complete 12-row builder0 screen, generate four, builder5, then selection and GPU7 repeats\n'
  printf '# GEAR: final T16 only, then selected-row builder5 GPU7 repeats\n'
  exit 0
fi

mkdir -p "$LOG_ROOT" "$ARTIFACT_ROOT"
[[ -f "$T16_FEEDBACK" ]] || {
  echo "Stage6 controls require completed final T16 feedback: $T16_FEEDBACK" >&2
  exit 1
}
"$PYTHON" "$CONTROL_TOOL" validate-original-contract \
  --contract-json "$FROZEN_CONTRACT_JSON" \
  --source-config "$SOURCE_CONFIG" \
  --source-checkpoint "$SOURCE_CHECKPOINT" \
  --output-json "$FORMAL_ROOT/controls/original_default/contract_preflight.json"
"$PYTHON" "$CONTROL_TOOL" validate-t16 \
  --feedback-json "$T16_FEEDBACK" \
  --contract-json "$FROZEN_CONTRACT_JSON" \
  --output-json "$GEAR_SELECTION"

"${PREPARE_COMMAND[@]}" >"$LOG_ROOT/prepare_initial.log" 2>&1
run_native_repeats

SCHEDULE_ROOT="$FORMAL_ROOT/controls/schedule_only"
build_requests \
  "$PLAN_ROOT/schedule_only/measure/candidate_plan.json" \
  schedule_only measure 5 "$SCHEDULE_ROOT/requests"
ensure_phase_feedback \
  "$SCHEDULE_ROOT/requests/request_manifest.json" schedule_only 5 1 \
  "$SCHEDULE_ROOT/feedback_history.json" \
  "$SCHEDULE_ROOT/file_integrity_audit.json"
select_arm \
  "$SCHEDULE_ROOT/feedback_history.json" 1 \
  S6-FCO-TRT-SCHEDULE-ONLY-MEASURE-V2 \
  "$SCHEDULE_ROOT/selected_after_complete_feedback.json"
run_trt_repeats \
  "$SCHEDULE_ROOT/selected_after_complete_feedback.json" \
  "$SCHEDULE_ROOT/independent_repeats" 5 schedule_only

COMPRESSION_ROOT="$FORMAL_ROOT/controls/compression_only"
build_requests \
  "$PLAN_ROOT/compression_only/measure/candidate_plan.json" \
  compression_only measure 0 "$COMPRESSION_ROOT/requests"
ensure_phase_feedback \
  "$COMPRESSION_ROOT/requests/request_manifest.json" compression_only 0 16 \
  "$COMPRESSION_ROOT/feedback_history.json" \
  "$COMPRESSION_ROOT/file_integrity_audit.json"
select_arm \
  "$COMPRESSION_ROOT/feedback_history.json" 16 \
  S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2 \
  "$COMPRESSION_ROOT/selected_after_complete_feedback.json"
run_trt_repeats \
  "$COMPRESSION_ROOT/selected_after_complete_feedback.json" \
  "$COMPRESSION_ROOT/independent_repeats" 0 compression_only

TUNE_ROOT="$FORMAL_ROOT/controls/compress_then_tune"
build_requests \
  "$PLAN_ROOT/compress_then_tune/screen/candidate_plan.json" \
  compress_then_tune screen 0 "$TUNE_ROOT/screen_requests"
ensure_phase_feedback \
  "$TUNE_ROOT/screen_requests/request_manifest.json" \
  compress_then_tune_screen 0 12 \
  "$TUNE_ROOT/screen_feedback_history.json" \
  "$TUNE_ROOT/screen_file_integrity_audit.json"

prepare_command "$TUNE_ROOT/screen_feedback_history.json"
"${PREPARE_COMMAND[@]}" >"$LOG_ROOT/prepare_tuned_four.log" 2>&1
build_requests \
  "$PLAN_ROOT/compress_then_tune/tuned_remeasurement/candidate_plan.json" \
  compress_then_tune tuned_remeasurement 5 "$TUNE_ROOT/tuned_requests"
ensure_phase_feedback \
  "$TUNE_ROOT/tuned_requests/request_manifest.json" \
  compress_then_tune_tuned 5 4 \
  "$TUNE_ROOT/tuned_feedback_history.json" \
  "$TUNE_ROOT/tuned_file_integrity_audit.json"
"$PYTHON" "$CONTROL_TOOL" combine-audits \
  --audit-json "$TUNE_ROOT/screen_file_integrity_audit.json" \
  --audit-json "$TUNE_ROOT/tuned_file_integrity_audit.json" \
  --output-json "$TUNE_ROOT/file_integrity_audit.json"
select_arm \
  "$TUNE_ROOT/tuned_feedback_history.json" 4 \
  S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2 \
  "$TUNE_ROOT/selected_after_complete_feedback.json"
run_trt_repeats \
  "$TUNE_ROOT/selected_after_complete_feedback.json" \
  "$TUNE_ROOT/independent_repeats" 5 compress_then_tune

run_trt_repeats \
  "$GEAR_SELECTION" \
  "$FORMAL_ROOT/search/$TASK_ID/independent_repeats" 5 gear

FINAL_AUDIT="$FORMAL_ROOT/final/fcooper_stage6_five_arm_audit_v2.json"
if [[ ! -f "$FINAL_AUDIT" ]]; then
  "$PYTHON" "$FINALIZER" \
    --root "$FORMAL_ROOT" \
    --output-dir "$FORMAL_ROOT/final" \
    >"$LOG_ROOT/finalize_five_arm.log" 2>&1
fi
"$PYTHON" "$CONTROL_TOOL" validate-final-audit \
  --audit-json "$FINAL_AUDIT" >/dev/null
"$PYTHON" "$CONTROL_TOOL" write-supervisor-complete \
  --audit-json "$FINAL_AUDIT" \
  --output-json "$FORMAL_ROOT/controls/five_arm_supervisor_complete.json" \
  >/dev/null
