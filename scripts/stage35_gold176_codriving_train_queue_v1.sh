#!/usr/bin/env bash
set -euo pipefail

V2X_ROOT=${V2X_ROOT:-/exdata/jichengzhi/V2Xverse_pyramid}
MODEL_ROOT=${MODEL_ROOT:-$V2X_ROOT/output/codriving_v2_gold_ap_20260709}
PYTHON_BIN=${PYTHON_BIN:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU=${GPU:-6}
DRY_RUN=0
WIDTHS=(48x64x192 48x96x128 48x32x128 64x64x192)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$MODEL_ROOT/logs"

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' \
    ! -name 'net_epoch_bestval_at0.pth' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

training_marker_valid() {
  local marker="$1" width="$2" model_dir="$3" checkpoint config checkpoint_sha config_sha bestval_count
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
  bestval_count=$(find "$model_dir" -maxdepth 1 -type f \
    -name 'net_epoch_bestval_at*.pth' | wc -l)
  [[ "$bestval_count" -eq 1 ]] || return 1
  [[ "$(find "$model_dir" -maxdepth 1 -type f \
    -name 'net_epoch_bestval_at*.pth' -print -quit)" == "$checkpoint" ]] || return 1
  [[ -s "$checkpoint" && -s "$config" ]] || return 1
  [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$checkpoint_sha" ]] || return 1
  [[ "$(sha256sum "$config" | awk '{print $1}')" == "$config_sha" ]]
}

write_training_marker() {
  local marker="$1" width="$2" config="$3" checkpoint="$4"
  jq -n --arg width "$width" --arg checkpoint "$checkpoint" --arg config "$config" \
    --arg checkpoint_sha "$(sha256sum "$checkpoint" | awk '{print $1}')" \
    --arg config_sha "$(sha256sum "$config" | awk '{print $1}')" \
    '{schema_version:"stage35_gold176_codriving_training_complete_v1", width:$width,
      checkpoint_path:$checkpoint, checkpoint_sha256:$checkpoint_sha,
      config_path:$config, config_sha256:$config_sha}' >"$marker.tmp"
  mv "$marker.tmp" "$marker"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  exec 9>"$MODEL_ROOT/.stage35_gold176_gpu${GPU}.lock"
  flock 9
fi

for width in "${WIDTHS[@]}"; do
  model_dir="$MODEL_ROOT/$width"
  config="$model_dir/config.yaml"
  log="$MODEL_ROOT/logs/train_stage35_gold176_${width}_gpu${GPU}.log"
  marker="$model_dir/stage35_gold176_training_complete.json"
  [[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 1; }
  if training_marker_valid "$marker" "$width" "$model_dir"; then
    continue
  fi
  command=(
    "$PYTHON_BIN" "$V2X_ROOT/opencood/tools/train.py"
    --hypes_yaml "$config" --model_dir "$model_dir" --fusion_method intermediate
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$GPU" "$V2X_ROOT:/home/jichengzhi/V2X"
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONPATH="$V2X_ROOT:/home/jichengzhi/V2X" \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee -a "$log"
  checkpoint=$(latest_checkpoint "$model_dir")
  [[ -n "$checkpoint" && -s "$checkpoint" ]]
  [[ "$(find "$model_dir" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' | wc -l)" -eq 1 ]]
  write_training_marker "$marker" "$width" "$config" "$checkpoint"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$MODEL_ROOT/logs/train_stage35_gold176_gpu${GPU}.done"
fi
