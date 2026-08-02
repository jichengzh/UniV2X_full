#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jichengzhi/V2X
OUT="$REPO/results/stage35_gold32_supplement_v1_20260713"
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
REQUIRED_HOST=zs-nj-tap-gpu18
GPU=6
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

SPECS=(
  "16x32x64|016x032x064|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10|net_epoch_bestval_at31.pth"
  "24x64x128|024x064x128|/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_frontier_01_2026_06_28|net_epoch31.pth"
  "32x64x128|032x064x128|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10|net_epoch_bestval_at29.pth"
  "64x128x256|064x128x256|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29|net_epoch_bestval_at23.pth"
)

print_command() {
  local separator=""
  for argument in "$@"; do
    printf '%s%s' "$separator" "$argument"
    separator=" "
  done
  printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "required_host=$REQUIRED_HOST"
  echo "gpu=$GPU"
else
  [[ "$(hostname)" == "$REQUIRED_HOST" ]] || {
    echo "refusing to run on host $(hostname); expected $REQUIRED_HOST" >&2
    exit 3
  }
  gpu_name=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -n1)
  [[ "$gpu_name" == *H800* ]] || {
    echo "refusing to run on GPU $GPU ($gpu_name); expected H800" >&2
    exit 3
  }
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r width padded ckpt_dir ckpt_name <<< "$spec"
  checkpoint="$ckpt_dir/$ckpt_name"
  source_dir="$OUT/pyramid_sources_shape_repaired_v1/$padded"
  onnx="$source_dir/pyramid_${padded}_multiscale.onnx"
  report="$source_dir/onnx_export_report.json"
  calibration="$OUT/pyramid_calibration/$padded/spatial_features_train16.npz"
  trt_npy="$OUT/pyramid_calibration/$padded/trt_npy"
  export_command=(
    "$PY" "$REPO/scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"
    --label "gold32_shape_repair_${width}"
    --ckpt-dir "$ckpt_dir"
    --checkpoint-path "$checkpoint"
    --out "$onnx"
    --report-json "$report"
    --gpu-id "$GPU"
    --eval-range 102.4,51.2
    --input-shape 2,64,128,256
  )
  calibration_command=(
    "$PY" "$REPO/scripts/stage35_prepare_pyramid_trt_calibration_v1.py"
    --source-npz "$calibration"
    --output-dir "$trt_npy"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${export_command[@]}"
    print_command "${calibration_command[@]}"
    continue
  fi

  [[ -f "$checkpoint" ]]
  [[ -f "$ckpt_dir/config.yaml" ]]
  [[ -f "$calibration" ]]
  mkdir -p "$source_dir"
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:/home/jichengzhi/V2X \
    "${export_command[@]}"
  "${calibration_command[@]}"
  jq -e '.status == "success" and .input_shape == [2,64,128,256]' "$report" >/dev/null
  jq -e '.sample_shape == [2,64,128,256] and .sample_dtype == "float32" and .batch_count == 15' \
    "$trt_npy/trt_npy_manifest.json" >/dev/null
  [[ $(find "$trt_npy" -maxdepth 1 -name 'batch2_*.npy' | wc -l) -eq 15 ]]
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUT/repair_v1"
  touch "$OUT/repair_v1/pyramid_sources_shape_repaired.done"
fi
