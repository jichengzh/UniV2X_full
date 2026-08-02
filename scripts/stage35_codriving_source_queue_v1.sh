#!/usr/bin/env bash
set -euo pipefail

REPO=/home/jichengzhi/V2X
V2X=/exdata/jichengzhi/V2Xverse_pyramid
MODEL_ROOT="$V2X/output/codriving_v2_gold_ap_20260709"
OUT="$REPO/results/stage35_gold32_supplement_v1_20260713"
PY=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python
GPU=7
WAIT_PID=0
DRY_RUN=0
WIDTHS=(16x64x128 24x56x128 32x32x256 40x80x160)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --wait-pid) WAIT_PID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

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
  local width="$1" model_dir="$MODEL_ROOT/$width"
  local calibration="$model_dir/stage3_calib_train_n16_float32.npz"
  local calibration_summary="$model_dir/stage3_calib_train_n16_float32_summary.json"
  local onnx="$model_dir/resnet_multiscale_${width}_final_fp32.onnx"
  local trt_dir="$model_dir/trt_calibration_npy"
  local done_marker="$OUT/source_prep/codriving_${width}.done"
  local log="$OUT/logs/codriving_${width}_source_queue.log"
  local calibration_cmd=(
    "$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py"
    --repo-root "$V2X" --width "$width" --model-dir "$model_dir"
    --output "$calibration" --summary "$calibration_summary"
    --n-samples 16 --num-workers 0 --progress-every 4
  )
  local onnx_cmd=(
    "$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"
    --repo-root "$V2X" --width "$width" --mode fp16 --model-dir "$model_dir"
    --onnx "$onnx" --n-samples 1 --num-workers 0 --progress-every 1
    --tvm-gpu "$GPU" --force-fallback
    --eval-dir "$model_dir/source_export_smoke_fp16_eval"
    --out-json "$model_dir/source_export_smoke_fp16.json"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${calibration_cmd[@]}"
    print_command "${onnx_cmd[@]}"
    return
  fi
  if [[ -f "$done_marker" ]]; then
    return
  fi
  grep -q 'Training Finished' "$MODEL_ROOT/logs/train_gold32_${width}_gpu7.log"
  wait_for_gpu
  mkdir -p "$OUT/source_prep" "$OUT/logs" "$trt_dir"
  : > "$log"
  PYTHONPATH="$V2X:$REPO" "${calibration_cmd[@]}" >> "$log" 2>&1
  PYTHONPATH="$V2X:$REPO" "${onnx_cmd[@]}" >> "$log" 2>&1
  "$PY" -c 'import sys; from pathlib import Path; import numpy as np; src=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); z=np.load(src, allow_pickle=False); x=z["spatial_features"].astype("float32", copy=False); assert x.shape[0] == 16; [np.save(out / f"sample_{i:03d}.npy", x[i], allow_pickle=False) for i in range(16)]' "$calibration" "$trt_dir"
  [[ -f "$onnx" ]]
  [[ $(find "$trt_dir" -maxdepth 1 -name 'sample_*.npy' | wc -l) -eq 16 ]]
  sha256sum "$model_dir"/net_epoch_bestval_at*.pth "$onnx" "$calibration" > "$OUT/source_prep/codriving_${width}_sha256.txt"
  touch "$done_marker"
}

if [[ "$DRY_RUN" -eq 0 && "$WAIT_PID" -gt 0 ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 300
  done
fi

for width in "${WIDTHS[@]}"; do
  run_one "$width"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$OUT/source_prep/codriving_source_queue.done"
fi
