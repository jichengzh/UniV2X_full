#!/usr/bin/env bash
set -euo pipefail

V2X_ROOT="${V2X_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
root="${V2X_RESULTS_DIR:-${V2X_ROOT}/results}/lane_c_orin_fcooper_stage6_five_config_20260724"
runner="${FCOOPER_TRT_AP_BRIDGE:-${V2X_ROOT}/scripts/fcooper_trt_ap_bridge_v1.py}"
python="${V2X_PYTHON:-python3}"
heal="${HEAL_ROOT:?set HEAL_ROOT to the HEAL checkout}"
export PYTHONPATH="${V2X_ROOT}:${heal}:${PYTHONPATH:-}"

wait_for_compression() {
  local output=$root/05_full2170_ap/compression_only
  while [[ ! -f $output/full.exit ]]; do
    sleep 30
  done
  [[ $(<"$output/full.exit") == 0 ]]
}

run_trt() {
  local arm=$1
  local widths=$2
  local source=$root/00_source_audit/recovered_h800/search_artifacts/sources/$widths
  local output=$root/05_full2170_ap/$arm
  mkdir -p "$output"
  local command=(
    "$python" -u "$runner"
    --config "$source/config.yaml"
    --checkpoint "$source/recovered_checkpoint.pth"
    --checkpoint-dir "$source"
    --engine "$root/02_engines/$arm/model.engine"
    --arm "$arm"
    --build-receipt "$root/02_engines/$arm/build.json"
    --manifest "$root/00_source_audit/canonical_manifest.json"
    --output-json "$output/metrics.json"
    --prediction-manifest "$output/prediction_manifest.json"
    --run-log-manifest "$output/run_manifest.json"
    --require-full2170
    --num-workers 4
  )
  printf '%q ' "${command[@]}" > "$output/full-command.txt"
  printf '\n' >> "$output/full-command.txt"
  set +e
  (
    cd "$heal"
    "${command[@]}"
  ) > "$output/full.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$output/full.exit"
  [[ $status == 0 ]]
}

run_native() {
  local arm=original_default
  local source=$root/00_source_audit/recovered_h800/search_artifacts/sources/64x128x256x128x256
  local output=$root/05_full2170_ap/$arm
  mkdir -p "$output"
  local command=(
    "$python" -u "$runner"
    --config "$source/config.yaml"
    --checkpoint "$source/recovered_checkpoint.pth"
    --checkpoint-dir "$source"
    --native
    --output-json "$output/metrics.json"
    --prediction-manifest "$output/prediction_manifest.json"
    --run-log-manifest "$output/run_manifest.json"
    --require-full2170
    --num-workers 4
  )
  printf '%q ' "${command[@]}" > "$output/full-command.txt"
  printf '\n' >> "$output/full-command.txt"
  set +e
  (
    cd "$heal"
    "${command[@]}"
  ) > "$output/full.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$output/full.exit"
  [[ $status == 0 ]]
}

wait_for_compression
run_trt schedule_only 64x128x256x128x256
run_trt compress_then_tune 64x64x64x32x64
run_trt joint_fp16_control 32x32x64x32x64
run_native
