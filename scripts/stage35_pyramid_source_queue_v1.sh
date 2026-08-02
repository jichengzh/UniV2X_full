#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jichengzhi/V2X
OUT="$REPO/results/stage35_gold32_supplement_v1_20260713"
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
GPU=7
WAIT_PID=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --wait-pid) WAIT_PID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SPECS=(
  "16x32x64|016x032x064|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10|net_epoch_bestval_at31.pth"
  "24x64x128|024x064x128|/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_01_2026_06_28|net_epoch31.pth"
  "32x64x128|032x064x128|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10|net_epoch_bestval_at29.pth"
  "64x128x256|064x128x256|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29|net_epoch_bestval_at23.pth"
)

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

wait_for_gpu() {
  while true; do
    local util memory
    util=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    memory=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$util" -le 5 && "$memory" -le 1024 ]]; then
      return
    fi
    sleep 300
  done
}

run_one() {
  local width="$1" padded="$2" ckpt_dir="$3" ckpt_name="$4"
  local checkpoint="$ckpt_dir/$ckpt_name"
  local source_dir="$OUT/pyramid_sources/$padded"
  local calibration_dir="$OUT/pyramid_calibration/$padded"
  local onnx="$source_dir/pyramid_${padded}_multiscale.onnx"
  local onnx_report="$source_dir/onnx_export_report.json"
  local calibration="$calibration_dir/spatial_features_train16.npz"
  local calibration_summary="$calibration_dir/summary.json"
  local trt_npy_dir="$calibration_dir/trt_npy"
  local done_marker="$OUT/source_prep/pyramid_${width}.done"
  local log="$OUT/logs/pyramid_${width}_source_queue.log"
  local onnx_cmd=(
    "$PY" "$REPO/scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"
    --label "gold32_pyramid_${width}" --ckpt-dir "$ckpt_dir"
    --checkpoint-path "$checkpoint" --out "$onnx" --report-json "$onnx_report"
    --gpu-id "$GPU" --eval-range 102.4,51.2 --input-shape 2,64,128,256
  )
  local calibration_cmd=(
    "$PY" "$REPO/scripts/stage3_pyramid_calibration_export_v3.py"
    --ckpt-dir "$ckpt_dir" --checkpoint-path "$checkpoint"
    --output-npz "$calibration" --summary-json "$calibration_summary"
    --num-samples 16 --gpu-id "$GPU" --eval-range 102.4,51.2
  )
  local trt_calibration_cmd=(
    "$PY" "$REPO/scripts/stage35_prepare_pyramid_trt_calibration_v1.py"
    --source-npz "$calibration" --output-dir "$trt_npy_dir"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${onnx_cmd[@]}"
    print_command "${calibration_cmd[@]}"
    print_command "${trt_calibration_cmd[@]}"
    return
  fi
  if [[ -f "$done_marker" ]]; then
    return
  fi
  [[ -f "$checkpoint" ]]
  [[ -f "$ckpt_dir/config.yaml" ]]
  wait_for_gpu
  mkdir -p "$source_dir" "$calibration_dir" "$OUT/source_prep" "$OUT/logs"
  : > "$log"
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:/home/jichengzhi/V2X \
    "${onnx_cmd[@]}" >> "$log" 2>&1
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:/home/jichengzhi/V2X \
    "${calibration_cmd[@]}" >> "$log" 2>&1
  "${trt_calibration_cmd[@]}" >> "$log" 2>&1
  jq -e '.status == "success"' "$onnx_report" >/dev/null
  jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$calibration_summary" >/dev/null
  [[ $(find "$trt_npy_dir" -maxdepth 1 -name 'batch2_*.npy' | wc -l) -eq 15 ]]
  sha256sum "$checkpoint" "$onnx" "$calibration" > "$OUT/source_prep/pyramid_${width}_sha256.txt"
  touch "$done_marker"
}

if [[ "$DRY_RUN" -eq 0 && "$WAIT_PID" -gt 0 ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 300
  done
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r width padded ckpt_dir ckpt_name <<< "$spec"
  run_one "$width" "$padded" "$ckpt_dir" "$ckpt_name"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$OUT/source_prep/pyramid_source_queue.done"
fi
