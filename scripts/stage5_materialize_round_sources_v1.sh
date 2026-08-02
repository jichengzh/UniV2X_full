#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
V2X_ROOT=${V2X_ROOT:-/exdata/jichengzhi/V2Xverse_pyramid}
HEAL_ROOT=${HEAL_ROOT:-/home/jichengzhi/heal_research/HEAL}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
REQUEST=
MODEL=
GROUP_ID=
GPU=6
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --request) REQUEST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --group-id) GROUP_ID="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -s "$REQUEST" ]] || { echo "--request must be a non-empty JSON file" >&2; exit 2; }
[[ "$MODEL" == "pyramid" || "$MODEL" == "codriving" ]] || {
  echo "--model must be pyramid or codriving" >&2
  exit 2
}
SCHEMA=$(jq -r '.schema_version' "$REQUEST")
if [[ "$SCHEMA" == "stage5_measurement_request_v1" ]]; then
  jq -e '
    .group_count == 2 and .row_count == 8 and
    (.rows | length) == 8 and
    ([.rows[].group_id] | unique | length) == 2
  ' "$REQUEST" >/dev/null
  jq -e --arg model "$MODEL" '
    [.rows[] | select(.model == $model)] as $rows |
    ($rows | length) == 4 and
    ([$rows[] | [.dispatch_key, .q_mode]] | unique | sort) ==
      [["trt_engine","fp16"],["trt_engine","int8"],["tvm_auto","fp16"],["tvm_auto","int8"]]
  ' "$REQUEST" >/dev/null
  ROW=$(jq -c --arg model "$MODEL" '[.rows[] | select(.model == $model)][0]' "$REQUEST")
  GROUP_ID=$(jq -r '.group_id' <<<"$ROW")
elif [[ "$SCHEMA" == "stage5_measurement_request_v2" ]]; then
  [[ -n "$GROUP_ID" ]] || { echo "--group-id is required for a v2 request" >&2; exit 2; }
  "$PY" - "$REQUEST" <<'PY'
import json
import sys
from framework.stage5.measurement_plan_v2 import _validate_request

with open(sys.argv[1], encoding="utf-8") as handle:
    _validate_request(json.load(handle))
PY
  jq -e --arg model "$MODEL" --arg group_id "$GROUP_ID" '
    .batch_size == 4 and (.rows | length) == 4 and
    ([.rows[].model] | unique) == [$model] and
    ([.rows[].task_id] | unique | length) == 1 and
    ([.rows[].capability_profile_id] | unique | length) == 1 and
    any(.rows[]; .group_id == $group_id)
  ' "$REQUEST" >/dev/null
  ROW=$(jq -c --arg group_id "$GROUP_ID" '[.rows[] | select(.group_id == $group_id)][0]' "$REQUEST")
else
  echo "unsupported request schema: $SCHEMA" >&2
  exit 2
fi
WIDTH=$(jq -r '.width | map(tostring) | join("x")' <<<"$ROW")
SOURCE_PLAN_SHA=$(jq -r '.source_evidence_sha256' <<<"$ROW")

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

sha() {
  sha256sum "$1" | awk '{print $1}'
}

write_evidence() {
  local evidence="$1" checkpoint="$2" onnx="$3" calibration="$4" summary="$5" marker="$6"
  mkdir -p "$(dirname "$evidence")" "$(dirname "$marker")"
  jq -n \
    --arg schema "stage5_source_materialization_evidence_v1" \
    --arg group_id "$GROUP_ID" \
    --arg model "$MODEL" \
    --arg width "$WIDTH" \
    --arg source_plan_sha256 "$SOURCE_PLAN_SHA" \
    --arg checkpoint_path "$checkpoint" --arg checkpoint_sha256 "$(sha "$checkpoint")" \
    --arg onnx_path "$onnx" --arg onnx_sha256 "$(sha "$onnx")" \
    --arg calibration_path "$calibration" --arg calibration_sha256 "$(sha "$calibration")" \
    --arg calibration_summary_path "$summary" --arg calibration_summary_sha256 "$(sha "$summary")" \
    '{schema_version:$schema, group_id:$group_id, model:$model, width:$width,
      source_plan_sha256:$source_plan_sha256,
      checkpoint_path:$checkpoint_path, checkpoint_sha256:$checkpoint_sha256,
      onnx_path:$onnx_path, onnx_sha256:$onnx_sha256,
      calibration_path:$calibration_path, calibration_sha256:$calibration_sha256,
      calibration_summary_path:$calibration_summary_path,
      calibration_summary_sha256:$calibration_summary_sha256,
      status:"ready"}' >"$evidence.tmp"
  mv "$evidence.tmp" "$evidence"
  touch "$marker"
}

evidence_valid() {
  local evidence="$1" marker="$2"
  [[ -s "$evidence" && -f "$marker" ]] || return 1
  jq -e --arg group_id "$GROUP_ID" \
    --arg source_plan_sha256 "$SOURCE_PLAN_SHA" \
    '.schema_version == "stage5_source_materialization_evidence_v1" and
     .group_id == $group_id and .source_plan_sha256 == $source_plan_sha256 and
     .status == "ready"' "$evidence" >/dev/null || return 1
  while IFS=$'\t' read -r path digest; do
    [[ -s "$path" && "$(sha "$path")" == "$digest" ]] || return 1
  done < <(jq -r '[.checkpoint_path,.checkpoint_sha256],
                    [.onnx_path,.onnx_sha256],
                    [.calibration_path,.calibration_sha256],
                    [.calibration_summary_path,.calibration_summary_sha256] | @tsv' "$evidence")
}

materialize_pyramid() {
  local checkpoint checkpoint_dir checkpoint_sha onnx report calibration calibration_root summary trt_dir marker evidence log training_required export_checkpoint epoches
  checkpoint=$(jq -r '.source_contract.checkpoint_path' <<<"$ROW")
  checkpoint_dir=$(jq -r '.source_contract.checkpoint_dir' <<<"$ROW")
  checkpoint_sha=$(jq -r '.source_contract.checkpoint_sha256' <<<"$ROW")
  training_required=$(jq -r '.source_contract.training_required // false' <<<"$ROW")
  onnx=$(jq -r '.source_contract.onnx_path' <<<"$ROW")
  report=$(jq -r '.source_contract.onnx_report_path' <<<"$ROW")
  calibration=$(jq -r '.source_contract.calibration_npz' <<<"$ROW")
  calibration_root=$(jq -r '.source_contract.calibration_root' <<<"$ROW")
  summary=$(jq -r '.source_contract.calibration_summary' <<<"$ROW")
  trt_dir=$(jq -r '.source_contract.trt_calibration_dir' <<<"$ROW")
  marker=$(jq -r '.source_contract.source_done_marker' <<<"$ROW")
  evidence="${marker%.done}_evidence.json"
  log="$(dirname "$marker")/../logs/source_pyramid_${WIDTH}_gpu${GPU}.log"
  export_checkpoint="$checkpoint"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$training_required" == "true" ]]; then
      materialize_pyramid_checkpoint
    fi
  else
    if [[ "$training_required" == "true" ]]; then
      materialize_pyramid_checkpoint
      epoches=$(jq -r '.source_contract.training_epoches' <<<"$ROW")
      export_checkpoint=$(latest_pyramid_checkpoint_at_or_before "$checkpoint_dir" "$epoches")
      [[ -n "$export_checkpoint" && -s "$export_checkpoint" ]]
    else
      [[ -s "$checkpoint" && "$(sha "$checkpoint")" == "$checkpoint_sha" ]]
    fi
    evidence_valid "$evidence" "$marker" && return
  fi
  local onnx_cmd=("$PY" "$REPO/scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"
    --label "stage5_${WIDTH}" --ckpt-dir "$checkpoint_dir" --checkpoint-path "$export_checkpoint"
    --out "$onnx" --report-json "$report" --gpu-id "$GPU"
    --eval-range 102.4,51.2 --input-shape 2,64,128,256)
  local calibration_cmd=("$PY" "$REPO/scripts/stage3_pyramid_calibration_export_v3.py"
    --ckpt-dir "$checkpoint_dir" --checkpoint-path "$export_checkpoint"
    --output-npz "$calibration" --summary-json "$summary" --num-samples 16
    --gpu-id "$GPU" --eval-range 102.4,51.2)
  local trt_cmd=("$PY" "$REPO/scripts/stage35_prepare_pyramid_trt_calibration_v1.py"
    --source-npz "$calibration" --output-dir "$trt_dir")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${onnx_cmd[@]}"
    print_command "${calibration_cmd[@]}"
    print_command "${trt_cmd[@]}"
    return
  fi
  mkdir -p "$(dirname "$onnx")" "$calibration_root" "$trt_dir" "$(dirname "$log")"
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:"$REPO" "${onnx_cmd[@]}" >>"$log" 2>&1
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL:"$REPO" "${calibration_cmd[@]}" >>"$log" 2>&1
  "${trt_cmd[@]}" >>"$log" 2>&1
  jq -e '.status == "success"' "$report" >/dev/null
  jq -e '.schema == "stage3_pyramid_calibration_export_v3"' "$summary" >/dev/null
  [[ "$(find "$trt_dir" -maxdepth 1 -name 'batch2_*.npy' | wc -l)" -eq 15 ]]
  write_evidence "$evidence" "$checkpoint" "$onnx" "$calibration" "$summary" "$marker"
}

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'net_epoch_bestval_at*.pth' \
    ! -name 'net_epoch_bestval_at0.pth' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-
}

latest_pyramid_checkpoint_at_or_before() {
  "$PY" - "$1" "$2" <<'PY'
import re
import sys
from pathlib import Path

root, cutoff = Path(sys.argv[1]), int(sys.argv[2])
candidates = []
for path in root.glob("net_epoch_bestval_at*.pth"):
    match = re.fullmatch(r"net_epoch_bestval_at(\d+)\.pth", path.name)
    if match and 0 < int(match.group(1)) <= cutoff:
        candidates.append((int(match.group(1)), path))
if candidates:
    print(max(candidates)[1])
PY
}

write_pyramid_training_marker() {
  local marker="$1" checkpoint="$2" config="$3" base_sha="$4" train_log="$5"
  jq -n --arg width "$WIDTH" --arg checkpoint "$checkpoint" --arg config "$config" \
    --arg checkpoint_sha "$(sha "$checkpoint")" --arg config_sha "$(sha "$config")" \
    --arg base_checkpoint_sha256 "$base_sha" --arg train_log "$train_log" \
    '{schema_version:"stage5_pyramid_training_complete_v1",width:$width,
      checkpoint_path:$checkpoint,checkpoint_sha256:$checkpoint_sha,
      config_path:$config,config_sha256:$config_sha,
      base_checkpoint_sha256:$base_checkpoint_sha256,training_log_path:$train_log}' \
    >"$marker.tmp"
  mv "$marker.tmp" "$marker"
}

pyramid_training_marker_valid() {
  local marker="$1" width="$2" checkpoint config base_sha
  [[ -s "$marker" ]] || return 1
  jq -e --arg width "$width" \
    '.schema_version == "stage5_pyramid_training_complete_v1" and .width == $width' \
    "$marker" >/dev/null || return 1
  checkpoint=$(jq -r '.checkpoint_path' "$marker")
  config=$(jq -r '.config_path' "$marker")
  base_sha=$(jq -r '.base_checkpoint_sha256' "$marker")
  [[ -s "$checkpoint" && -s "$config" ]] || return 1
  [[ "$(sha "$checkpoint")" == "$(jq -r '.checkpoint_sha256' "$marker")" ]] || return 1
  [[ "$(sha "$config")" == "$(jq -r '.config_sha256' "$marker")" ]] || return 1
  [[ "$base_sha" == "$(jq -r '.source_contract.base_checkpoint_sha256' <<<"$ROW")" ]]
}

materialize_pyramid_checkpoint() {
  local checkpoint checkpoint_dir config marker base_checkpoint base_dir base_sha init_checkpoint
  local epoches width_per_group groups master_port prune_log train_log best target_checkpoint
  checkpoint=$(jq -r '.source_contract.checkpoint_path' <<<"$ROW")
  checkpoint_dir=$(jq -r '.source_contract.checkpoint_dir' <<<"$ROW")
  config=$(jq -r '.source_contract.config_path' <<<"$ROW")
  marker=$(jq -r '.source_contract.training_done_marker' <<<"$ROW")
  base_checkpoint=$(jq -r '.source_contract.base_checkpoint_path' <<<"$ROW")
  base_dir=$(jq -r '.source_contract.base_checkpoint_dir' <<<"$ROW")
  base_sha=$(jq -r '.source_contract.base_checkpoint_sha256' <<<"$ROW")
  epoches=$(jq -r '.source_contract.training_epoches' <<<"$ROW")
  width_per_group=$(jq -r '.source_contract.width_per_group' <<<"$ROW")
  groups=$(jq -r '.source_contract.groups' <<<"$ROW")
  init_checkpoint="$checkpoint_dir/net_epoch_bestval_at23.pth"
  prune_log="$checkpoint_dir/stage5_structural_prune.log"
  train_log="$checkpoint_dir/stage5_train.log"
  master_port=$((31000 + GPU))
  local prune_cmd=("$PY" "$REPO/tools/structural_prune_pyramid.py"
    --orig-dir "$base_dir" --out-dir "$checkpoint_dir"
    --num-filters-new "${WIDTH//x/,}" --width-per-group "$width_per_group" --groups "$groups")
  local flatten_cmd=("$PY" -c 'import sys,torch; p=sys.argv[1]; x=torch.load(p,map_location="cpu"); torch.save(x["model_state_dict"] if isinstance(x,dict) and "model_state_dict" in x else x,p)' "$init_checkpoint")
  local patch_cmd=("$PY" -c 'import re,sys; p=sys.argv[1]; n=sys.argv[2]; s=open(p).read(); s,count=re.subn(r"(^\s*epoches:\s*)\d+",lambda m:m.group(1)+n,s,flags=re.M); assert count==1, count; open(p,"w").write(s)' "$config" "$epoches")
  local train_cmd=("$PY" -m torch.distributed.launch --nproc_per_node=1 --use_env
    "--master_port=$master_port" "$HEAL_ROOT/opencood/tools/train_ddp.py"
    --hypes_yaml "$config" --model_dir "$checkpoint_dir" --half)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${prune_cmd[@]}"
    print_command "${flatten_cmd[@]}"
    print_command "${patch_cmd[@]}"
    print_command "${train_cmd[@]}"
    return
  fi
  [[ -s "$base_checkpoint" && "$(sha "$base_checkpoint")" == "$base_sha" ]]
  pyramid_training_marker_valid "$marker" "$WIDTH" && return
  mkdir -p "$checkpoint_dir"
  target_checkpoint="$checkpoint_dir/net_epoch${epoches}.pth"
  if [[ -s "$target_checkpoint" && -s "$config" ]]; then
    "${patch_cmd[@]}" >>"$prune_log" 2>&1
    best=$(latest_pyramid_checkpoint_at_or_before "$checkpoint_dir" "$epoches")
    [[ -n "$best" && -s "$best" ]] || best="$target_checkpoint"
    cp -f "$best" "$checkpoint"
    write_pyramid_training_marker "$marker" "$checkpoint" "$config" "$base_sha" "$train_log"
    return
  fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=/home/jichengzhi/heal_research/HEAL:"$REPO" \
    "${prune_cmd[@]}" >"$prune_log" 2>&1
  [[ -s "$init_checkpoint" && -s "$config" ]]
  "${flatten_cmd[@]}" >>"$prune_log" 2>&1
  "${patch_cmd[@]}" >>"$prune_log" 2>&1
  (cd "$HEAL_ROOT" && \
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$HEAL_ROOT:$REPO" PYTHONUNBUFFERED=1 \
      "${train_cmd[@]}") >"$train_log" 2>&1
  best=$(latest_pyramid_checkpoint_at_or_before "$checkpoint_dir" "$epoches")
  [[ -n "$best" && -s "$best" ]] || best="$target_checkpoint"
  [[ -n "$best" && -s "$best" ]]
  cp -f "$best" "$checkpoint"
  write_pyramid_training_marker "$marker" "$checkpoint" "$config" "$base_sha" "$train_log"
}

training_marker_valid() {
  local marker="$1" width="$2" checkpoint config
  [[ -s "$marker" ]] || return 1
  jq -e --arg width "$width" \
    '.schema_version == "stage5_codriving_training_complete_v1" and .width == $width' \
    "$marker" >/dev/null || return 1
  checkpoint=$(jq -r '.checkpoint_path' "$marker")
  config=$(jq -r '.config_path' "$marker")
  [[ -s "$checkpoint" && -s "$config" ]] || return 1
  [[ "$(sha "$checkpoint")" == "$(jq -r '.checkpoint_sha256' "$marker")" ]] || return 1
  [[ "$(sha "$config")" == "$(jq -r '.config_sha256' "$marker")" ]]
}

write_training_marker() {
  local marker="$1" checkpoint="$2" config="$3" log="$4"
  jq -n --arg width "$WIDTH" --arg checkpoint "$checkpoint" --arg config "$config" \
    --arg checkpoint_sha "$(sha "$checkpoint")" --arg config_sha "$(sha "$config")" \
    --arg log "$log" \
    '{schema_version:"stage5_codriving_training_complete_v1",width:$width,
      checkpoint_path:$checkpoint,checkpoint_sha256:$checkpoint_sha,
      config_path:$config,config_sha256:$config_sha,training_log_path:$log}' >"$marker.tmp"
  mv "$marker.tmp" "$marker"
}

materialize_codriving() {
  local model_dir onnx calibration summary trt_dir marker training_marker evidence log config checkpoint
  model_dir=$(jq -r '.source_contract.model_dir' <<<"$ROW")
  onnx=$(jq -r '.source_contract.onnx_path' <<<"$ROW")
  calibration=$(jq -r '.source_contract.calibration_npz' <<<"$ROW")
  summary=$(jq -r '.source_contract.calibration_summary' <<<"$ROW")
  trt_dir=$(jq -r '.source_contract.trt_calibration_dir' <<<"$ROW")
  marker=$(jq -r '.source_contract.source_done_marker' <<<"$ROW")
  training_marker=$(jq -r '.source_contract.training_done_marker' <<<"$ROW")
  evidence="${marker%.done}_evidence.json"
  log="$(dirname "$marker")/../logs/source_codriving_${WIDTH}_gpu${GPU}.log"
  config="$model_dir/config.yaml"
  local prepare_cmd=("$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_ap_queue.py"
    --repo-root "$V2X_ROOT" --out-root "$(dirname "$model_dir")"
    prepare-one --width "$WIDTH" --device cuda:0)
  local train_cmd=("$PY" "$V2X_ROOT/opencood/tools/train.py"
    --hypes_yaml "$config" --model_dir "$model_dir" --fusion_method intermediate)
  local calibration_cmd=("$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py"
    --repo-root "$V2X_ROOT" --width "$WIDTH" --model-dir "$model_dir"
    --output "$calibration" --summary "$summary" --n-samples 16
    --num-workers 0 --progress-every 4)
  local onnx_cmd=("$PY" "$REPO/scripts/stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"
    --repo-root "$V2X_ROOT" --width "$WIDTH" --mode fp16 --model-dir "$model_dir"
    --onnx "$onnx" --n-samples 1 --num-workers 0 --progress-every 1 --tvm-gpu "$GPU"
    --force-fallback --eval-dir "$model_dir/source_export_smoke_fp16_eval"
    --out-json "$model_dir/source_export_smoke_fp16.json")
  local trt_cmd=("$PY" -c 'import sys; from pathlib import Path; import numpy as np; z=np.load(sys.argv[1],allow_pickle=False); x=z["spatial_features"].astype("float32",copy=False); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); assert x.shape[0]==16; [np.save(out/f"sample_{i:03d}.npy",x[i],allow_pickle=False) for i in range(16)]' "$calibration" "$trt_dir")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "${prepare_cmd[@]}"
    print_command "$V2X_ROOT/opencood/tools/train.py" "${train_cmd[@]:2}"
    print_command "${calibration_cmd[@]}"
    print_command "${onnx_cmd[@]}"
    print_command "${trt_cmd[@]}"
    return
  fi
  evidence_valid "$evidence" "$marker" && return
  mkdir -p "$model_dir" "$trt_dir" "$(dirname "$log")"
  if ! training_marker_valid "$training_marker" "$WIDTH"; then
    if [[ ! -s "$model_dir/prepare_report.json" ]] || \
       ! jq -e --arg width "$WIDTH" '.status == "success" and .width == $width and .sanity.missing == 0' "$model_dir/prepare_report.json" >/dev/null; then
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$V2X_ROOT:$REPO" \
        "${prepare_cmd[@]}" >>"$log" 2>&1
    fi
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$V2X_ROOT:$REPO" PYTHONUNBUFFERED=1 \
      "${train_cmd[@]}" >>"$log" 2>&1
    checkpoint=$(latest_checkpoint "$model_dir")
    [[ -n "$checkpoint" && -s "$checkpoint" ]]
    write_training_marker "$training_marker" "$checkpoint" "$config" "$log"
  fi
  checkpoint=$(jq -r '.checkpoint_path' "$training_marker")
  PYTHONPATH="$V2X_ROOT:$REPO" "${calibration_cmd[@]}" >>"$log" 2>&1
  PYTHONPATH="$V2X_ROOT:$REPO" "${onnx_cmd[@]}" >>"$log" 2>&1
  "${trt_cmd[@]}" >>"$log" 2>&1
  [[ -s "$onnx" && -s "$calibration" && -s "$summary" ]]
  [[ "$(find "$trt_dir" -maxdepth 1 -name 'sample_*.npy' | wc -l)" -eq 16 ]]
  write_evidence "$evidence" "$checkpoint" "$onnx" "$calibration" "$summary" "$marker"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  lock_root=$(jq -r '.source_contract.model_dir // .source_contract.calibration_root' <<<"$ROW")
  mkdir -p "$lock_root"
  exec 9>"$lock_root/.stage5_source_group.lock"
  flock 9
  marker=$(jq -r '.source_contract.source_done_marker' <<<"$ROW")
  evidence="${marker%.done}_evidence.json"
  evidence_valid "$evidence" "$marker" && exit 0
  wait_for_gpu
fi

if [[ "$MODEL" == "pyramid" ]]; then
  materialize_pyramid
else
  materialize_codriving
fi
