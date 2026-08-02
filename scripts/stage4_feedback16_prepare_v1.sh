#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
V2X_ROOT=${V2X_ROOT:-/exdata/jichengzhi/V2Xverse_pyramid}
MODEL_ROOT=${MODEL_ROOT:-$V2X_ROOT/output/codriving_v2_gold_ap_20260709}
PYTHON_BIN=${PYTHON_BIN:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU=${GPU:-6}
CANDIDATE=
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CANDIDATE" && -s "$CANDIDATE" ]] || {
  echo "--candidate must point to a non-empty JSON file" >&2
  exit 2
}

GOLD176="$REPO/results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1/gold176_final.json"
GOLD176_MANIFEST="${GOLD176%/*}/gold176_manifest.json"
GRAPH_FEATURES="${GOLD176%/*}/graph_features.json"
jq -e '
  .schema_version == "stage4_feedback16_candidate_v1" and
  .group_count == 4 and .row_count == 16 and
  ([.groups[].group_id] | length) == 4 and
  ([.groups[].group_id] | unique | length) == 4 and
  ([.groups[] | select(.model == "codriving")] | length == 2) and
  ([.groups[] | select(.model == "pyramid")] | length == 2)
' "$CANDIDATE" >/dev/null

expected_gold=$(jq -r '.source_gold176_sha256' "$CANDIDATE")
expected_manifest=$(jq -r '.source_gold176_manifest_sha256' "$CANDIDATE")
expected_graph=$(jq -r '.source_graph_features_sha256' "$CANDIDATE")
[[ "$(sha256sum "$GOLD176" | awk '{print $1}')" == "$expected_gold" ]] || {
  echo "Gold176 SHA mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$GOLD176_MANIFEST" | awk '{print $1}')" == "$expected_manifest" ]] || {
  echo "Gold176 manifest SHA mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$GRAPH_FEATURES" | awk '{print $1}')" == "$expected_graph" ]] || {
  echo "Gold176 graph features SHA mismatch" >&2
  exit 1
}

while IFS=$'\t' read -r group_id; do
  if jq -e --arg group_id "$group_id" '.jobs[] | select(.group_id == $group_id)' \
    "$GOLD176_MANIFEST" >/dev/null; then
    echo "candidate overlaps Gold176: $group_id" >&2
    exit 1
  fi
done < <(jq -r '.groups[].group_id' "$CANDIDATE")

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$MODEL_ROOT/logs"
fi

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' \
    ! -name 'net_epoch_bestval_at0.pth' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

training_marker_valid() {
  local marker="$1" width="$2" model_dir="$3" checkpoint config checkpoint_sha config_sha
  [[ -s "$marker" ]] || return 1
  jq -e --arg width "$width" \
    '.schema_version == "stage4_feedback16_training_complete_v1" and .width == $width' \
    "$marker" >/dev/null || return 1
  checkpoint=$(jq -r '.checkpoint_path' "$marker")
  config=$(jq -r '.config_path' "$marker")
  checkpoint_sha=$(jq -r '.checkpoint_sha256' "$marker")
  config_sha=$(jq -r '.config_sha256' "$marker")
  [[ "$config" == "$model_dir/config.yaml" ]] || return 1
  [[ "$(dirname "$checkpoint")" == "$model_dir" ]] || return 1
  [[ -s "$checkpoint" && -s "$config" ]] || return 1
  [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" == "$checkpoint_sha" ]] || return 1
  [[ "$(sha256sum "$config" | awk '{print $1}')" == "$config_sha" ]]
}

prepare_report_valid() {
  local report="$1" width="$2"
  [[ -s "$report" ]] || return 1
  jq -e --arg width "$width" \
    '.status == "success" and .width == $width and .sanity.missing == 0' \
    "$report" >/dev/null
}

write_training_marker() {
  local marker="$1" width="$2" config="$3" checkpoint="$4" log="$5"
  jq -n --arg width "$width" --arg checkpoint "$checkpoint" --arg config "$config" \
    --arg log "$log" \
    --arg checkpoint_sha "$(sha256sum "$checkpoint" | awk '{print $1}')" \
    --arg config_sha "$(sha256sum "$config" | awk '{print $1}')" \
    --arg log_sha "$(sha256sum "$log" | awk '{print $1}')" \
    '{schema_version:"stage4_feedback16_training_complete_v1", width:$width,
      checkpoint_path:$checkpoint, checkpoint_sha256:$checkpoint_sha,
      config_path:$config, config_sha256:$config_sha,
      training_log_path:$log, training_log_sha256:$log_sha}' >"$marker.tmp"
  mv "$marker.tmp" "$marker"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  exec 9>"$MODEL_ROOT/.stage4_feedback16_gpu${GPU}.lock"
  flock 9
fi

while IFS=$'\t' read -r width model_dir; do
  model_dir=${model_dir/#\/exdata\/jichengzhi\/V2Xverse_pyramid/$V2X_ROOT}
  config="$model_dir/config.yaml"
  marker="$model_dir/stage4_feedback16_training_complete.json"
  log="$MODEL_ROOT/logs/train_stage4_feedback16_${width}_gpu${GPU}.log"
  prepare=(
    "$PYTHON_BIN" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_ap_queue.py"
    --repo-root "$V2X_ROOT" --out-root "$MODEL_ROOT"
    prepare-one --width "$width" --device "cuda:0"
  )
  train=(
    "$PYTHON_BIN" "$V2X_ROOT/opencood/tools/train.py"
    --hypes_yaml "$config" --model_dir "$model_dir" --fusion_method intermediate
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$GPU" "$V2X_ROOT:$REPO"
    printf '%q ' "${prepare[@]}"
    printf '\n'
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$GPU" "$V2X_ROOT:$REPO"
    printf '%q ' "${train[@]}"
    printf '\n'
    continue
  fi
  if training_marker_valid "$marker" "$width" "$model_dir"; then
    continue
  fi
  if ! prepare_report_valid "$model_dir/prepare_report.json" "$width"; then
    prepared=0
    for attempt in 1 2; do
      if CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$V2X_ROOT:$REPO" \
        "${prepare[@]}" >>"$log" 2>&1 \
        && prepare_report_valid "$model_dir/prepare_report.json" "$width"; then
        prepared=1
        break
      fi
      if [[ -s "$model_dir/prepare_report.json" ]]; then
        cp "$model_dir/prepare_report.json" \
          "$model_dir/prepare_report.failed_attempt${attempt}.json"
      fi
    done
    [[ "$prepared" -eq 1 ]] || {
      echo "prepare failed twice for $width" >&2
      exit 1
    }
  fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$V2X_ROOT:$REPO" PYTHONUNBUFFERED=1 \
    "${train[@]}" 2>&1 | tee -a "$log"
  checkpoint=$(latest_checkpoint "$model_dir")
  [[ -n "$checkpoint" && -s "$checkpoint" ]]
  write_training_marker "$marker" "$width" "$config" "$checkpoint" "$log"
done < <(
  jq -r '.groups[] | select(.model == "codriving") |
    [(.width | map(tostring) | join("x")), .model_dir] | @tsv' "$CANDIDATE"
)

if [[ "$DRY_RUN" -eq 0 ]]; then
  touch "$MODEL_ROOT/logs/train_stage4_feedback16_gpu${GPU}.done"
fi
