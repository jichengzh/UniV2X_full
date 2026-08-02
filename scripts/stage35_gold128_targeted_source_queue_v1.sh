#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jichengzhi/V2X
OUT="$REPO/results/stage35_gold128_targeted_supplement_v1_20260714"
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
GPU=6
SHARD=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --shard) SHARD="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$SHARD" in
  0)
    SPECS=(
      "16x32x96|016x032x096|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_frontier_12_fresh_v1"
      "40x64x128|040x064x128|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_frontier_02_fresh_v1"
    )
    ;;
  1)
    SPECS=(
      "24x32x96|024x032x096|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_lhc_20_fresh_v1"
      "56x112x224|056x112x224|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_frontier_20_fresh_v1"
    )
    ;;
  *) echo "--shard must be 0 or 1" >&2; exit 2 ;;
esac

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local width="$1" padded="$2" ckpt_dir="$3"
  local checkpoint="$ckpt_dir/net_epoch_bestval_at1.pth"
  local source_dir="$OUT/pyramid_sources/$padded"
  local calibration_dir="$OUT/pyramid_calibration/$padded"
  local onnx="$source_dir/pyramid_${padded}_multiscale.onnx"
  local onnx_report="$source_dir/onnx_export_report.json"
  local calibration="$calibration_dir/spatial_features_train16.npz"
  local calibration_summary="$calibration_dir/summary.json"
  local trt_npy_dir="$calibration_dir/trt_npy"
  local done_marker="$OUT/source_prep/pyramid_${width}.done"
  local log="$OUT/logs/source_${width}_gpu${GPU}.log"
  local onnx_cmd=(
    "$PY" "$REPO/scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"
    --label "gold128_targeted_${width}" --ckpt-dir "$ckpt_dir"
    --checkpoint-path "$checkpoint" --out "$onnx" --report-json "$onnx_report"
    --gpu-id "$GPU" --eval-range 102.4,51.2 --input-shape 2,64,128,256
  )
  local calibration_cmd=(
    "$PY" "$REPO/scripts/stage3_pyramid_calibration_export_v3.py"
    --ckpt-dir "$ckpt_dir" --checkpoint-path "$checkpoint"
    --output-npz "$calibration" --summary-json "$calibration_summary"
    --num-samples 16 --gpu-id "$GPU" --eval-range 102.4,51.2
  )
  local trt_cmd=(
    "$PY" "$REPO/scripts/stage35_prepare_pyramid_trt_calibration_v1.py"
    --source-npz "$calibration" --output-dir "$trt_npy_dir"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'WIDTH=%s GPU=%s\n' "$width" "$GPU"
    print_command "${onnx_cmd[@]}"
    print_command "${calibration_cmd[@]}"
    print_command "${trt_cmd[@]}"
    return
  fi
  [[ -s "$checkpoint" ]]
  [[ -s "$ckpt_dir/config.yaml" ]]
  if [[ -f "$done_marker" ]]; then
    echo "SKIP $width already complete"
    return
  fi
  mkdir -p "$source_dir" "$calibration_dir" "$OUT/source_prep" "$OUT/logs"
  touch "$log"
  echo "START $width GPU$GPU $(date -Is)"
  if ! jq -e '.status == "success"' "$onnx_report" >/dev/null 2>&1; then
    PYTHONPATH=/home/jichengzhi/heal_research/HEAL:/home/jichengzhi/V2X \
      "${onnx_cmd[@]}" >>"$log" 2>&1
  fi
  if ! jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$calibration_summary" >/dev/null 2>&1; then
    PYTHONPATH=/home/jichengzhi/heal_research/HEAL:/home/jichengzhi/V2X \
      "${calibration_cmd[@]}" >>"$log" 2>&1
  fi
  if [[ "$(find "$trt_npy_dir" -maxdepth 1 -name 'batch2_*.npy' 2>/dev/null | wc -l)" -ne 15 ]]; then
    "${trt_cmd[@]}" >>"$log" 2>&1
  fi
  jq -e '.status == "success"' "$onnx_report" >/dev/null
  jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$calibration_summary" >/dev/null
  [[ "$(find "$trt_npy_dir" -maxdepth 1 -name 'batch2_*.npy' | wc -l)" -eq 15 ]]
  sha256sum "$checkpoint" "$onnx" "$calibration" > "$OUT/source_prep/pyramid_${width}_sha256.txt"
  touch "$done_marker"
  echo "DONE $width GPU$GPU $(date -Is)"
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r width padded ckpt_dir <<< "$spec"
  run_one "$width" "$padded" "$ckpt_dir"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$OUT/source_prep/shard_${SHARD}.done"
fi
