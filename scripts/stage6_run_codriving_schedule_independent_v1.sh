#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
OUTPUT_ROOT=${OUTPUT_ROOT:-$FORMAL_ROOT/independent_validation_schedule}
EVIDENCE_BUNDLE=${EVIDENCE_BUNDLE:?EVIDENCE_BUNDLE is required}
AUDIT_JSON=${AUDIT_JSON:?AUDIT_JSON is required}
TRT_GPU=${TRT_GPU:-3}
TVM_GPU=${TVM_GPU:-4}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}
TVM_SITE=${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}
MODEL_DIR=${MODEL_DIR:-/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/64x128x256}
BASE_ONNX=${BASE_ONNX:-$MODEL_DIR/resnet_multiscale_64x128x256_final_fp32.onnx}
TVM_NVLIBS=${TVM_NVLIBS:-$(cat /exdata/jichengzhi/tvm_nvlibs.path)}
TVM_REFERENCE_WORK_DIR=${TVM_REFERENCE_WORK_DIR:-/exdata/jichengzhi/stage6_codriving_formal_20260722/tvm_schedule_only/workdirs/codriving_64x128x256_fp32}

cd "$REPO"
mkdir -p "$OUTPUT_ROOT/trt" "$OUTPUT_ROOT/tvm"

run_trt() {
  for repeat in 0 1 2; do
    root="$OUTPUT_ROOT/trt/repeat_$repeat"
    mkdir -p "$root/artifacts"
    if [[ ! -s "$root/performance_result.json" ]]; then
      "$PY" framework/trt_baseline/trt_profile_v1.py --onnx "$BASE_ONNX" --precision fp32 \
        --gpu "$TRT_GPU" --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
        --artifact-dir "$root/artifacts" --out "$root/performance_result.json"
    fi
  done
  mkdir -p "$OUTPUT_ROOT/trt/ap/full_1789"
  if [[ ! -s "$OUTPUT_ROOT/trt/ap/full_1789/full_ap_eval_report.json" ]]; then
    CUDA_VISIBLE_DEVICES="$TRT_GPU" "$PY" scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py \
      --model-dir "$MODEL_DIR" \
      --engine "$OUTPUT_ROOT/trt/repeat_2/artifacts/compiled.engine" \
      --precision-tag fp32 --gate full --n-samples 1789 \
      --eval-dir "$OUTPUT_ROOT/trt/ap/full_1789/eval" \
      --out-json "$OUTPUT_ROOT/trt/ap/full_1789/full_ap_eval_report.json"
  fi
}

run_tvm_repeat() {
  local repeat=$1
  local root="$OUTPUT_ROOT/tvm/repeat_$repeat"
  local run_root="/exdata/jichengzhi/stage6_codriving_formal_20260722/tvm_schedule_independent/repeat_$repeat"
  local work_dir="$run_root/workdir"
  local raw_root="$run_root/results"
  mkdir -p "$root/logs" "$work_dir" "$raw_root"
  [[ -s "$root/performance_result.json" ]] && return 0
  printf '%s\n' \
    "{\"gpu\":$TVM_GPU,\"onnx_path\":\"$BASE_ONNX\",\"tvm_work_dir\":\"$work_dir\",\"database_workload_path\":\"$work_dir/database_workload.json\",\"database_tuning_record_path\":\"$work_dir/database_tuning_record.json\",\"label\":\"codriving_schedule_independent_r$repeat\",\"candidate_id\":\"stage6:codriving:schedule_only:fp32:r$repeat\",\"job_id\":\"stage6_codriving_schedule_only_tvm_fp32_r$repeat\",\"width\":[64,128,256]}" \
    >"$root/tune_plan.jsonl"
  if [[ ! -s "$TVM_REFERENCE_WORK_DIR/database_workload.json" || ! -s "$TVM_REFERENCE_WORK_DIR/database_tuning_record.json" ]]; then
    echo "frozen TVM schedule database is missing: $TVM_REFERENCE_WORK_DIR" >&2
    return 1
  fi
  if [[ ! -s "$work_dir/database_workload.json" || ! -s "$work_dir/database_tuning_record.json" ]]; then
    cp "$TVM_REFERENCE_WORK_DIR/database_workload.json" "$work_dir/database_workload.json"
    cp "$TVM_REFERENCE_WORK_DIR/database_tuning_record.json" "$work_dir/database_tuning_record.json"
    "$PY" - "$TVM_REFERENCE_WORK_DIR" "$work_dir" "$root/frozen_schedule_database.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
source=Path(sys.argv[1]); target=Path(sys.argv[2])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
payload={
  "schema_version":"stage6_frozen_schedule_database_binding_v1",
  "policy":"reuse_formal_64_trial_database_for_independent_runtime_repeats",
  "source_work_dir":str(source),
  "workload_sha256":sha(target/"database_workload.json"),
  "tuning_record_sha256":sha(target/"database_tuning_record.json"),
}
Path(sys.argv[3]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  fi
  if [[ ! -s "$root/latency_row.jsonl" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$REPO:$TVM_SITE:${PYTHONPATH:-}" \
    "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
      --kind latency --model codriving --label codriving_schedule_independent_r$repeat \
      --phase stage6_codriving_independent --gpu "$TVM_GPU" --onnx "$BASE_ONNX" \
      --work-dir "$work_dir" --width 64,128,256 \
      --candidate-id stage6:codriving:schedule_only:fp32:r$repeat \
      --software-point-id stage6:codriving:64x128x256:q=fp32:profile=h800-tvm-schedule-independent \
      --config-id-tuned stage6_codriving_tvm_schedule_independent_r${repeat}_tuned \
      --config-id-default stage6_codriving_tvm_schedule_independent_r${repeat}_default \
      --run-id stage6_codriving_schedule_independent_r${repeat}_latency \
      --precision fp32 --quant-policy fp32 --quant-method none \
      --measurement-source stage6_codriving_schedule_independent_latency \
      --full-network-claim false --tune-budget base_graph_64_trials \
      --raw-root "$raw_root/latency" --out-jsonl "$root/latency_row.jsonl"
  fi
  if [[ ! -s "$root/energy_row.jsonl" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$REPO:$TVM_SITE:${PYTHONPATH:-}" \
    "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
      --kind energy --model codriving --label codriving_schedule_independent_r$repeat \
      --phase stage6_codriving_independent --gpu "$TVM_GPU" --onnx "$BASE_ONNX" \
      --work-dir "$work_dir" --width 64,128,256 \
      --candidate-id stage6:codriving:schedule_only:fp32:r$repeat \
      --software-point-id stage6:codriving:64x128x256:q=fp32:profile=h800-tvm-schedule-independent \
      --config-id-tuned stage6_codriving_tvm_schedule_independent_r${repeat}_tuned \
      --config-id-default stage6_codriving_tvm_schedule_independent_r${repeat}_default \
      --run-id stage6_codriving_schedule_independent_r${repeat}_energy \
      --precision fp32 --quant-policy fp32 --quant-method none \
      --measurement-source stage6_codriving_schedule_independent_energy \
      --full-network-claim false --tune-budget base_graph_64_trials \
      --energy-schedule-policy metaschedule_tuned \
      --raw-root "$raw_root/energy" --out-jsonl "$root/energy_row.jsonl"
  fi
  "$PY" - "$root/latency_row.jsonl" "$root/energy_row.jsonl" "$root/performance_result.json" <<'PY'
import json,sys
lat=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
eng=[json.loads(line) for line in open(sys.argv[2]) if line.strip()]
tuned=next((row for row in lat if row.get("schedule_policy")=="metaschedule_tuned"),lat[-1])
energy=eng[-1]
latency=float(tuned["latency_p50_us"])/1000.0
watt=energy.get("watt_avg") or energy.get("watt_p50")
payload={"build_success":True,"framework":"tvm","precision":"fp32","width":[64,128,256],
         "lat_tuned_ms":latency,"latency_ms":latency,"energy_j":float(watt)*latency/1000.0,
         "watt_avg":watt,"schema_version":"stage6_codriving_schedule_independent_performance_v1"}
open(sys.argv[3],"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
}

run_tvm() {
  for repeat in 0 1 2; do run_tvm_repeat "$repeat"; done
  local run_root=/exdata/jichengzhi/stage6_codriving_formal_20260722/tvm_schedule_independent/repeat_2
  if [[ ! -s "$OUTPUT_ROOT/tvm/repeat_2/tvm_fp32_schedule.so" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
        --onnx "$BASE_ONNX" --work-dir "$run_root/workdir" --gpu "$TVM_GPU" \
        --artifact "$OUTPUT_ROOT/tvm/repeat_2/tvm_fp32_schedule.so" \
        --report "$OUTPUT_ROOT/tvm/repeat_2/artifact_report.json"
  fi
  mkdir -p "$OUTPUT_ROOT/tvm/ap/full_1789"
  if [[ ! -s "$OUTPUT_ROOT/tvm/ap/full_1789/full_ap_eval_report.json" ]]; then
    "$PY" scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py \
      --compiled-artifact "$OUTPUT_ROOT/tvm/repeat_2/tvm_fp32_schedule.so" \
      --precision-tag stage6_schedule_only_tvm_fp32_independent \
      --num-samples 1789 --full-ap-min-samples 1789 \
      --output-dir "$OUTPUT_ROOT/tvm/ap/full_1789" --model-dir "$MODEL_DIR" \
      --report-json "$OUTPUT_ROOT/tvm/ap/full_1789/full_ap_eval_report.json" \
      --gpu-id "$TVM_GPU"
  fi
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
  --output-json "$AUDIT_JSON" --target-model codriving
