#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
BACKEND=${BACKEND:?BACKEND must be tvm or trt}
GPU=${GPU:?GPU is required}
WAIT_PID=${WAIT_PID:-}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}
TVM_SITE=${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}

cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller" "$FORMAL_ROOT/$BACKEND/schedule_only" \
  "$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts"
QUEUE_DONE="$FORMAL_ROOT/controller/${BACKEND}_backend_tail_queue.done"
QUEUE_FAILED="$FORMAL_ROOT/controller/${BACKEND}_backend_tail_queue.failed"
exec 9>"$FORMAL_ROOT/controller/${BACKEND}_backend_tail_queue.lock"
flock 9
[[ ! -s "$QUEUE_DONE" ]] || exit 0
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$QUEUE_FAILED"; fi' EXIT
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  [[ -s "$FORMAL_ROOT/controller/${BACKEND}_fixed_arm_queue.done" ]] || {
    echo "fixed-arm queue did not complete successfully" >&2
    exit 1
  }
fi

if [[ "$BACKEND" == "tvm" ]]; then
  TVM_RUN_ROOT=/exdata/jichengzhi/stage6_pyramid_formal_20260720/tvm_schedule_only
  BASE_ONNX="$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx"
  CKPT_PATH=/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/performance_result.json" ]]; then
    env V2X_TVM_RUN_ROOT="$TVM_RUN_ROOT" V2X_TVM_MAX_TRIALS=64 V2X_TVM_TUNE_SEED=0 \
      "$PY" framework/measure_config.py --width 64,128,256 --precision fp32 --gpu "$GPU" \
      >"$FORMAL_ROOT/tvm/schedule_only/runner.log" 2>&1
    cp "$TVM_RUN_ROOT/results/smbo_64x128x256_fp32/measure_config_result.json" \
      "$FORMAL_ROOT/tvm/schedule_only/performance_result.json"
  fi
  "$PY" - "$FORMAL_ROOT/tvm/schedule_only/performance_result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get("build_success") is True and d.get("max_trials")==64
PY
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" || \
        ! -s "$FORMAL_ROOT/tvm/schedule_only/artifact_report.json" ]]; then
    TVM_NVLIBS=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
      --onnx "$BASE_ONNX" \
      --work-dir "$TVM_RUN_ROOT/workdirs/smbo_64x128x256_fp32" --gpu "$GPU" \
      --artifact "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" \
      --report "$FORMAL_ROOT/tvm/schedule_only/artifact_report.json"
  fi
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" ]]; then
    "$PY" scripts/stage2_h800_fp16_rewritten_activation_bridge.py \
      --label stage6_schedule_only_tvm_fp32 \
      --ckpt-dir /exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
      --checkpoint-path "$CKPT_PATH" \
      --raw-dir "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789" --eval-range 102.4,51.2 \
      --artifact-path "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" \
      --artifact-input-dtype float32 --persistent-worker --num-samples 1789 \
      --full-ap-min-samples 1789 \
      --export-report-json "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" \
      --gpu-id "$GPU"
  fi
  "$PY" - "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("status") == "success"
assert d.get("processed_samples") == 1789 and d.get("failed_samples") == 0
assert d.get("smoke_gate_passed") is True and int(d.get("pred_nonempty_count") or 0) > 0
assert isinstance(d.get("ap70"), (int, float))
PY
  BASE_WORK="$TVM_RUN_ROOT/workdirs/smbo_64x128x256_fp32"
else
  BASE_ONNX="$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx"
  CKPT_PATH=/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth
  "$PY" framework/trt_baseline/trt_profile_v1.py --onnx "$BASE_ONNX" --precision fp32 \
    --gpu "$GPU" --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
    --artifact-dir "$FORMAL_ROOT/trt/schedule_only/artifacts" \
    --out "$FORMAL_ROOT/trt/schedule_only/performance_result.json" \
    >"$FORMAL_ROOT/trt/schedule_only/runner.log" 2>&1
  "$PY" - "$FORMAL_ROOT/trt/schedule_only/performance_result.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get("build_success") is True and d.get("precision")=="fp32"
PY
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/stage3_trt_multiscale_ap_bridge_v3.py \
    --label stage6_schedule_only_trt_fp32 \
    --ckpt-dir /exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
    --engine "$FORMAL_ROOT/trt/schedule_only/artifacts/compiled.engine" \
    --precision-tag fp32 --num-samples 1789 --full-ap-min-samples 1789 \
    --eval-range 102.4,51.2 --raw-dir "$FORMAL_ROOT/trt/schedule_only/ap/full_1789" \
    --report-json "$FORMAL_ROOT/trt/schedule_only/ap/full_1789/full_ap_eval_report.json"
  BASE_CACHE="$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/trt_base_timing.cache"
fi

PLAN="$FORMAL_ROOT/$BACKEND/tune_then_compress_candidate_plan.json"
"$PY" - "$PLAN" >"$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts.tsv" <<'PY'
import json,sys
for index,row in enumerate(json.load(open(sys.argv[1]))["rows"]):
    width=",".join(map(str,row["width"]))
    print(index, width, row["q_mode"], row["source_contract"]["onnx_path"], sep="\t")
PY

while IFS=$'\t' read -r index width q_mode onnx; do
  out="$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts/attempt_$(printf '%02d' "$index").json"
  if [[ "$BACKEND" == "tvm" ]]; then
    TVM_NVLIBS=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage6_tvm_schedule_transfer_probe_v1.py --onnx "$onnx" \
      --base-work-dir "$BASE_WORK" --compressed-width "$width" \
      --intended-q-mode "$q_mode" --gpu "$GPU" --out "$out"
  else
    "$PY" scripts/stage6_trt_timing_cache_transfer_probe_v1.py \
      --base-onnx "$BASE_ONNX" --compressed-onnx "$onnx" --cache-in "$BASE_CACHE" \
      --compressed-width "$width" --intended-q-mode "$q_mode" --gpu "$GPU" --out "$out"
  fi
done <"$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts.tsv"

date -Is >"$QUEUE_DONE"
rm -f "$QUEUE_FAILED"
trap - EXIT
