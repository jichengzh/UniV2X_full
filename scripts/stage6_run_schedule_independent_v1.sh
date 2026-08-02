#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
OUTPUT_ROOT=${OUTPUT_ROOT:-$FORMAL_ROOT/independent_validation_v1/schedule_only}
EVIDENCE_BUNDLE=${EVIDENCE_BUNDLE:?EVIDENCE_BUNDLE is required}
AUDIT_JSON=${AUDIT_JSON:?AUDIT_JSON is required}
TRT_GPU=${TRT_GPU:-3}
TVM_GPU=${TVM_GPU:-4}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}
TVM_SITE=${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}

cd "$REPO"
mkdir -p "$OUTPUT_ROOT/trt" "$OUTPUT_ROOT/tvm"
BASE_ONNX="$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx"
CKPT=/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29
CKPT_PATH="$CKPT/net_epoch_bestval_at23.pth"

run_trt() {
for repeat in 0 1 2; do
  root="$OUTPUT_ROOT/trt/repeat_$repeat"
  mkdir -p "$root/artifacts"
  "$PY" framework/trt_baseline/trt_profile_v1.py --onnx "$BASE_ONNX" --precision fp32 \
    --gpu "$TRT_GPU" --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
    --artifact-dir "$root/artifacts" --out "$root/performance_result.json"
done
mkdir -p "$OUTPUT_ROOT/trt/ap/full_1789"
CUDA_VISIBLE_DEVICES="$TRT_GPU" "$PY" scripts/stage3_trt_multiscale_ap_bridge_v3.py \
  --label stage6_schedule_only_trt_fp32_independent --ckpt-dir "$CKPT" \
  --engine "$OUTPUT_ROOT/trt/repeat_2/artifacts/compiled.engine" --precision-tag fp32 \
  --num-samples 1789 --full-ap-min-samples 1789 --eval-range 102.4,51.2 \
  --raw-dir "$OUTPUT_ROOT/trt/ap/full_1789" \
  --report-json "$OUTPUT_ROOT/trt/ap/full_1789/full_ap_eval_report.json"
}

run_tvm() {
for repeat in 0 1 2; do
  root="$OUTPUT_ROOT/tvm/repeat_$repeat"
  run_root="/exdata/jichengzhi/stage6_pyramid_formal_20260720/tvm_schedule_independent/repeat_$repeat"
  mkdir -p "$root"
  env V2X_TVM_RUN_ROOT="$run_root" V2X_TVM_MAX_TRIALS=64 V2X_TVM_TUNE_SEED=0 \
    "$PY" framework/measure_config.py --width 64,128,256 --precision fp32 --gpu "$TVM_GPU"
  cp "$run_root/results/smbo_64x128x256_fp32/measure_config_result.json" \
    "$root/performance_result.json"
done
TVM_RUN_ROOT=/exdata/jichengzhi/stage6_pyramid_formal_20260720/tvm_schedule_independent/repeat_2
TVM_NVLIBS=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
  "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
  --onnx "$TVM_RUN_ROOT/models/smbo_64x128x256_backbone.onnx" \
  --work-dir "$TVM_RUN_ROOT/workdirs/smbo_64x128x256_fp32" --gpu "$TVM_GPU" \
  --artifact "$OUTPUT_ROOT/tvm/repeat_2/tvm_fp32_schedule.so" \
  --report "$OUTPUT_ROOT/tvm/repeat_2/artifact_report.json"
mkdir -p "$OUTPUT_ROOT/tvm/ap/full_1789"
"$PY" scripts/stage2_h800_fp16_rewritten_activation_bridge.py \
  --label stage6_schedule_only_tvm_fp32_independent --ckpt-dir "$CKPT" \
  --checkpoint-path "$CKPT_PATH" \
  --raw-dir "$OUTPUT_ROOT/tvm/ap/full_1789" --eval-range 102.4,51.2 \
  --artifact-path "$OUTPUT_ROOT/tvm/repeat_2/tvm_fp32_schedule.so" \
  --artifact-input-dtype float32 --persistent-worker --num-samples 1789 \
  --full-ap-min-samples 1789 \
  --export-report-json "$OUTPUT_ROOT/tvm/ap/full_1789/full_ap_eval_report.json" \
  --gpu-id "$TVM_GPU"
}

if [[ "$TRT_GPU" == "$TVM_GPU" ]]; then
  run_trt
  run_tvm
else
  run_trt & trt_pid=$!
  run_tvm & tvm_pid=$!
  wait "$trt_pid"
  wait "$tvm_pid"
fi

"$PY" scripts/stage6_finalize_schedule_independent_v1.py \
  --root "$OUTPUT_ROOT" --evidence-bundle "$EVIDENCE_BUNDLE" \
  --output-json "$AUDIT_JSON"
