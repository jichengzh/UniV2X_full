#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
V2X_ROOT=${V2X_ROOT:-/exdata/jichengzhi/V2Xverse_pyramid}
MODEL_ROOT=${MODEL_ROOT:-$V2X_ROOT/output/codriving_v2_gold_ap_20260709}
OUT=${OUT:-$REPO/results/stage35_gold144_targeted_supplement_v2_20260714}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU=${GPU:-6}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYRAMID_SPECS=(
  "24x32x64|024x032x064|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_frontier_10_fresh_v1"
  "40x96x192|040x096x192|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_frontier_14_fresh_v1"
  "56x128x256|056x128x256|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_s0_056_fresh_v1"
  "64x128x224|064x128x224|/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_o60_s2_224_fresh_v1"
)
CODRIVING_WIDTHS=(48x64x192 48x96x128 48x32x128 64x64x192)

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

wait_for_gpu() {
  while true; do
    local memory utilization
    memory=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    utilization=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$memory" -le 1024 && "$utilization" -le 5 ]]; then
      return
    fi
    sleep 300
  done
}

training_marker_valid() {
  local model_dir="$1" width="$2" marker checkpoint config checkpoint_sha config_sha bestval_count
  marker="$model_dir/stage35_gold176_training_complete.json"
  [[ -s "$marker" ]] || return 1
  jq -e --arg width "$width" \
    '.schema_version == "stage35_gold176_codriving_training_complete_v1" and .width == $width' \
    "$marker" >/dev/null || return 1
  checkpoint=$(jq -r '.checkpoint_path' "$marker")
  config=$(jq -r '.config_path' "$marker")
  checkpoint_sha=$(jq -r '.checkpoint_sha256' "$marker")
  config_sha=$(jq -r '.config_sha256' "$marker")
  [[ "$config" == "$model_dir/config.yaml" ]] || return 1
  [[ "$(dirname "$checkpoint")" == "$model_dir" ]] || return 1
  bestval_count=$(find "$model_dir" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' | wc -l)
  [[ "$bestval_count" -eq 1 ]] || return 1
  [[ "$(find "$model_dir" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' -print -quit)" == "$checkpoint" ]] || return 1
  [[ -s "$checkpoint" && -s "$config" ]] || return 1
  [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$checkpoint_sha" ]] || return 1
  [[ "$(sha256sum "$config" | awk '{print $1}')" == "$config_sha" ]]
}

pyramid_source_valid() {
  local done_marker="$1" onnx_report="$2" calibration_summary="$3" trt_dir="$4" sha_file="$5"
  [[ -f "$done_marker" && -s "$sha_file" ]] || return 1
  jq -e '.status == "success"' "$onnx_report" >/dev/null 2>&1 || return 1
  jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$calibration_summary" >/dev/null 2>&1 || return 1
  [[ "$(find "$trt_dir" -maxdepth 1 -name 'batch2_*.npy' 2>/dev/null | wc -l)" -eq 15 ]] || return 1
  sha256sum -c "$sha_file" >/dev/null 2>&1
}

codriving_source_valid() {
  local done_marker="$1" onnx="$2" calibration="$3" calibration_summary="$4" trt_dir="$5" sha_file="$6"
  [[ -f "$done_marker" && -s "$onnx" && -s "$calibration" && -s "$calibration_summary" && -s "$sha_file" ]] || return 1
  [[ "$(find "$trt_dir" -maxdepth 1 -name 'sample_*.npy' 2>/dev/null | wc -l)" -eq 16 ]] || return 1
  sha256sum -c "$sha_file" >/dev/null 2>&1
}

prepare_pyramid() {
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
  local sha_file="$OUT/source_prep/pyramid_${width}_sha256.txt"
  local log="$OUT/logs/source_pyramid_${width}_gpu${GPU}.log"
  local onnx_cmd=("$PY" "$REPO/scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"
    --label "gold176_${width}" --ckpt-dir "$ckpt_dir" --checkpoint-path "$checkpoint"
    --out "$onnx" --report-json "$onnx_report" --gpu-id "$GPU"
    --eval-range 102.4,51.2 --input-shape 2,64,128,256)
  local calibration_cmd=("$PY" "$REPO/scripts/stage3_pyramid_calibration_export_v3.py"
    --ckpt-dir "$ckpt_dir" --checkpoint-path "$checkpoint" --output-npz "$calibration"
    --summary-json "$calibration_summary" --num-samples 16 --gpu-id "$GPU"
    --eval-range 102.4,51.2)
  local trt_cmd=("$PY" "$REPO/scripts/stage35_prepare_pyramid_trt_calibration_v1.py"
    --source-npz "$calibration" --output-dir "$trt_npy_dir")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${onnx_cmd[@]}"; print_command "${calibration_cmd[@]}"; print_command "${trt_cmd[@]}"; return
  fi
  [[ -s "$checkpoint" && -s "$ckpt_dir/config.yaml" ]]
  pyramid_source_valid "$done_marker" "$onnx_report" "$calibration_summary" "$trt_npy_dir" "$sha_file" && return
  mkdir -p "$source_dir" "$calibration_dir" "$OUT/source_prep" "$OUT/logs"
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:"$REPO" "${onnx_cmd[@]}" >>"$log" 2>&1
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:"$REPO" "${calibration_cmd[@]}" >>"$log" 2>&1
  "${trt_cmd[@]}" >>"$log" 2>&1
  jq -e '.status == "success"' "$onnx_report" >/dev/null
  jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$calibration_summary" >/dev/null
  [[ "$(find "$trt_npy_dir" -maxdepth 1 -name 'batch2_*.npy' | wc -l)" -eq 15 ]]
  sha256sum "$checkpoint" "$onnx" "$onnx_report" "$calibration" "$calibration_summary" >"$sha_file"
  touch "$done_marker"
}

prepare_codriving() {
  local width="$1" model_dir="$MODEL_ROOT/$width"
  local calibration="$model_dir/stage3_calib_train_n16_float32.npz"
  local calibration_summary="$model_dir/stage3_calib_train_n16_float32_summary.json"
  local onnx="$model_dir/resnet_multiscale_${width}_final_fp32.onnx"
  local trt_dir="$model_dir/trt_calibration_npy"
  local done_marker="$OUT/source_prep/codriving_${width}.done"
  local sha_file="$OUT/source_prep/codriving_${width}_sha256.txt"
  local log="$OUT/logs/source_codriving_${width}_gpu${GPU}.log"
  local calibration_cmd=("$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py"
    --repo-root "$V2X_ROOT" --width "$width" --model-dir "$model_dir"
    --output "$calibration" --summary "$calibration_summary" --n-samples 16
    --num-workers 0 --progress-every 4)
  local onnx_cmd=("$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"
    --repo-root "$V2X_ROOT" --width "$width" --mode fp16 --model-dir "$model_dir"
    --onnx "$onnx" --n-samples 1 --num-workers 0 --progress-every 1 --tvm-gpu "$GPU"
    --force-fallback --eval-dir "$model_dir/source_export_smoke_fp16_eval"
    --out-json "$model_dir/source_export_smoke_fp16.json")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${calibration_cmd[@]}"; print_command "${onnx_cmd[@]}"; return
  fi
  training_marker_valid "$model_dir" "$width"
  codriving_source_valid "$done_marker" "$onnx" "$calibration" "$calibration_summary" "$trt_dir" "$sha_file" && return
  mkdir -p "$OUT/source_prep" "$OUT/logs" "$trt_dir"
  PYTHONPATH="$V2X_ROOT:$REPO" "${calibration_cmd[@]}" >>"$log" 2>&1
  PYTHONPATH="$V2X_ROOT:$REPO" "${onnx_cmd[@]}" >>"$log" 2>&1
  "$PY" -c 'import sys; from pathlib import Path; import numpy as np; z=np.load(sys.argv[1], allow_pickle=False); x=z["spatial_features"].astype("float32", copy=False); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); assert x.shape[0] == 16; [np.save(out/f"sample_{i:03d}.npy", x[i], allow_pickle=False) for i in range(16)]' "$calibration" "$trt_dir"
  [[ -s "$onnx" && "$(find "$trt_dir" -maxdepth 1 -name 'sample_*.npy' | wc -l)" -eq 16 ]]
  sha256sum "$model_dir"/net_epoch_bestval_at*.pth "$onnx" "$calibration" "$calibration_summary" >"$sha_file"
  touch "$done_marker"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  exec 9>"$MODEL_ROOT/.stage35_gold176_gpu${GPU}.lock"
  flock 9
  wait_for_gpu
fi

for spec in "${PYRAMID_SPECS[@]}"; do
  IFS='|' read -r width padded ckpt_dir <<<"$spec"
  prepare_pyramid "$width" "$padded" "$ckpt_dir"
done
for width in "${CODRIVING_WIDTHS[@]}"; do
  prepare_codriving "$width"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$OUT/source_prep/all_sources.done"
fi
