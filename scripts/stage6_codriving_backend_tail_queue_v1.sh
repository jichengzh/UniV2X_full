#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
BACKEND=${BACKEND:?BACKEND must be tvm or trt}
GPU=${GPU:?GPU is required}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}
TVM_SITE=${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}

MODEL_DIR=${MODEL_DIR:-/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/64x128x256}
BASE_ONNX=${BASE_ONNX:-$MODEL_DIR/resnet_multiscale_64x128x256_final_fp32.onnx}
BASE_WIDTH=${BASE_WIDTH:-64,128,256}

cd "$REPO"
mkdir -p "$FORMAL_ROOT/controller" "$FORMAL_ROOT/$BACKEND/schedule_only" \
  "$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts"
QUEUE_DONE="$FORMAL_ROOT/controller/${BACKEND}_codriving_backend_tail_queue.done"
QUEUE_FAILED="$FORMAL_ROOT/controller/${BACKEND}_codriving_backend_tail_queue.failed"
exec 9>"$FORMAL_ROOT/controller/${BACKEND}_codriving_backend_tail_queue.lock"
flock 9
[[ ! -s "$QUEUE_DONE" ]] || exit 0
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$QUEUE_FAILED"; fi' EXIT

if [[ "$BACKEND" == "tvm" ]]; then
  TVM_ROOT=/exdata/jichengzhi/stage6_codriving_formal_20260722/tvm_schedule_only
  WORK_DIR="$TVM_ROOT/workdirs/codriving_64x128x256_fp32"
  RAW_ROOT="$TVM_ROOT/results/codriving_64x128x256_fp32"
  LOG_DIR="$FORMAL_ROOT/tvm/schedule_only/logs"
  mkdir -p "$WORK_DIR" "$RAW_ROOT" "$LOG_DIR" "$FORMAL_ROOT/tvm/schedule_only/artifacts"
  if [[ ! -s "$WORK_DIR/database_workload.json" || ! -s "$WORK_DIR/database_tuning_record.json" ]]; then
    cat >"$LOG_DIR/tune_plan.jsonl" <<JSON
{"gpu": $GPU, "onnx_path": "$BASE_ONNX", "tvm_work_dir": "$WORK_DIR", "database_workload_path": "$WORK_DIR/database_workload.json", "database_tuning_record_path": "$WORK_DIR/database_tuning_record.json", "label": "codriving_64x128x256", "candidate_id": "stage6:codriving:schedule_only:fp32", "job_id": "stage6_codriving_schedule_only_tvm_fp32", "width": [64, 128, 256]}
JSON
    env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO:$TVM_SITE:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage2_original60_tvm_artifact_worker.py \
      --job-plan "$LOG_DIR/tune_plan.jsonl" \
      --job-state "$LOG_DIR/tune_state.json" \
      --manifest-out "$LOG_DIR/tune_manifest.json" \
      --log-dir "$LOG_DIR" --max-trials 64 --seed 0 \
      >"$LOG_DIR/tune.log" 2>&1
  fi
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/latency_row.jsonl" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$REPO:$TVM_SITE:${PYTHONPATH:-}" \
      "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
      --kind latency --model codriving --label codriving_64x128x256 --phase stage6_codriving \
      --gpu "$GPU" --onnx "$BASE_ONNX" --work-dir "$WORK_DIR" --width "$BASE_WIDTH" \
      --candidate-id stage6:codriving:schedule_only:fp32 \
      --software-point-id stage6:codriving:64x128x256:q=fp32:profile=h800-tvm-schedule-only \
      --config-id-tuned stage6_codriving_tvm_64x128x256_fp32_tuned \
      --config-id-default stage6_codriving_tvm_64x128x256_fp32_default \
      --run-id stage6_codriving_schedule_only_tvm_fp32_latency \
      --precision fp32 --quant-policy fp32 --quant-method none \
      --measurement-source stage6_codriving_schedule_only_latency_h800_tvm \
      --full-network-claim false --tune-budget base_graph_64_trials \
      --raw-root "$RAW_ROOT/latency" --out-jsonl "$FORMAL_ROOT/tvm/schedule_only/latency_row.jsonl" \
      >"$LOG_DIR/latency.log" 2>&1
  fi
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/energy_row.jsonl" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$REPO:$TVM_SITE:${PYTHONPATH:-}" \
      "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
      --kind energy --model codriving --label codriving_64x128x256 --phase stage6_codriving \
      --gpu "$GPU" --onnx "$BASE_ONNX" --work-dir "$WORK_DIR" --width "$BASE_WIDTH" \
      --candidate-id stage6:codriving:schedule_only:fp32 \
      --software-point-id stage6:codriving:64x128x256:q=fp32:profile=h800-tvm-schedule-only \
      --config-id-tuned stage6_codriving_tvm_64x128x256_fp32_tuned \
      --config-id-default stage6_codriving_tvm_64x128x256_fp32_default \
      --run-id stage6_codriving_schedule_only_tvm_fp32_energy \
      --precision fp32 --quant-policy fp32 --quant-method none \
      --measurement-source stage6_codriving_schedule_only_energy_h800_tvm \
      --full-network-claim false --tune-budget base_graph_64_trials \
      --energy-schedule-policy metaschedule_tuned \
      --raw-root "$RAW_ROOT/energy" --out-jsonl "$FORMAL_ROOT/tvm/schedule_only/energy_row.jsonl" \
      >"$LOG_DIR/energy.log" 2>&1
  fi
  "$PY" - "$FORMAL_ROOT/tvm/schedule_only/latency_row.jsonl" "$FORMAL_ROOT/tvm/schedule_only/energy_row.jsonl" "$FORMAL_ROOT/tvm/schedule_only/performance_result.json" <<'PY'
import json, sys
lat_rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
eng_rows=[json.loads(line) for line in open(sys.argv[2]) if line.strip()]
lat_tuned=next((r for r in lat_rows if r.get("schedule_policy")=="metaschedule_tuned"), lat_rows[-1])
eng=eng_rows[-1]
lat_ms=float(lat_tuned["latency_p50_us"])/1000.0
watt=eng.get("watt_avg") or eng.get("watt_p50")
payload={
  "schema_version":"stage6_codriving_tvm_schedule_only_performance_v1",
  "framework":"tvm","precision":"fp32","build_success":True,
  "width":[64,128,256],"latency_ms":lat_ms,"lat_p50_ms":lat_ms,
  "energy_j": (float(watt)*lat_ms/1000.0) if watt else None,
  "watt_avg": watt, "latency_row_count":len(lat_rows), "energy_row_count":len(eng_rows),
}
open(sys.argv[3],"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
      --onnx "$BASE_ONNX" --work-dir "$WORK_DIR" --gpu "$GPU" \
      --artifact "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" \
      --report "$FORMAL_ROOT/tvm/schedule_only/artifact_report.json" \
      >"$LOG_DIR/export_artifact.log" 2>&1
  fi
  if [[ ! -s "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" ]]; then
    "$PY" scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py \
      --compiled-artifact "$FORMAL_ROOT/tvm/schedule_only/artifacts/tvm_fp32_schedule.so" \
      --precision-tag stage6_schedule_only_tvm_fp32 \
      --num-samples 1789 --full-ap-min-samples 1789 \
      --output-dir "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789" \
      --model-dir "$MODEL_DIR" \
      --report-json "$FORMAL_ROOT/tvm/schedule_only/ap/full_1789/full_ap_eval_report.json" \
      --gpu-id "$GPU" >"$LOG_DIR/ap_full_1789.log" 2>&1
  fi
  BASE_WORK="$WORK_DIR"
else
  LOG_DIR="$FORMAL_ROOT/trt/schedule_only/logs"
  mkdir -p "$LOG_DIR" "$FORMAL_ROOT/trt/schedule_only/artifacts"
  if [[ ! -s "$FORMAL_ROOT/trt/schedule_only/performance_result.json" ]]; then
    "$PY" framework/trt_baseline/trt_profile_v1.py --onnx "$BASE_ONNX" --precision fp32 \
      --gpu "$GPU" --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
      --artifact-dir "$FORMAL_ROOT/trt/schedule_only/artifacts" \
      --out "$FORMAL_ROOT/trt/schedule_only/performance_result.json" \
      >"$LOG_DIR/runner.log" 2>&1
  fi
  if [[ ! -s "$FORMAL_ROOT/trt/schedule_only/ap/full_1789/full_ap_eval_report.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py \
      --model-dir "$MODEL_DIR" \
      --engine "$FORMAL_ROOT/trt/schedule_only/artifacts/compiled.engine" \
      --precision-tag fp32 --gate full --n-samples 1789 \
      --eval-dir "$FORMAL_ROOT/trt/schedule_only/ap/full_1789/eval" \
      --out-json "$FORMAL_ROOT/trt/schedule_only/ap/full_1789/full_ap_eval_report.json" \
      >"$LOG_DIR/ap_full_1789.log" 2>&1
  fi
fi

PLAN="$FORMAL_ROOT/$BACKEND/tune_then_compress_candidate_plan.json"
"$PY" - "$PLAN" >"$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts.tsv" <<'PY'
import json,sys
for index,row in enumerate(json.load(open(sys.argv[1]))["rows"]):
    width=",".join(map(str,row["width"]))
    source = row["source_contract"]
    print(index, width, row["q_mode"], source["onnx_path"], source["source_done_marker"], sep="\t")
PY

wait_source_ready() {
  local onnx="$1" marker="$2" waited=0 max_wait=${SOURCE_WAIT_SECS:-43200}
  local evidence="${marker%.done}_evidence.json"
  while [[ "$waited" -lt "$max_wait" ]]; do
    if [[ -s "$onnx" && -e "$marker" && -s "$evidence" ]]; then
      local expected actual status
      expected=$(jq -r '.onnx_sha256 // empty' "$evidence")
      status=$(jq -r '.status // empty' "$evidence")
      actual=$(sha256sum "$onnx" | awk '{print $1}')
      [[ "$status" == ready && -n "$expected" && "$actual" == "$expected" ]] || {
        echo "source evidence mismatch: onnx=$onnx marker=$marker evidence=$evidence" >&2
        return 1
      }
      return 0
    fi
    sleep 300
    waited=$((waited + 300))
  done
  echo "source artifact not ready after ${max_wait}s: onnx=$onnx marker=$marker evidence=$evidence" >&2
  return 1
}

CACHE="$FORMAL_ROOT/trt/tune_then_compress/base_timing.cache"
while IFS=$'\t' read -r index width q_mode onnx marker; do
  out="$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts/attempt_$(printf '%02d' "$index").json"
  [[ -s "$out" ]] && continue
  wait_source_ready "$onnx" "$marker"
  if [[ "$BACKEND" == "tvm" ]]; then
    env LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$(cat /exdata/jichengzhi/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}" \
      "$TVM_PY" scripts/stage6_tvm_schedule_transfer_probe_v1.py --onnx "$onnx" \
      --base-work-dir "$BASE_WORK" --base-width "$BASE_WIDTH" --compressed-width "$width" \
      --intended-q-mode "$q_mode" --gpu "$GPU" --out "$out"
  else
    if [[ -s "$CACHE" ]]; then
      "$PY" scripts/stage6_trt_timing_cache_transfer_probe_v1.py \
        --base-onnx "$BASE_ONNX" --compressed-onnx "$onnx" --cache-in "$CACHE" \
        --base-width "$BASE_WIDTH" --compressed-width "$width" \
        --intended-q-mode "$q_mode" --gpu "$GPU" --out "$out"
    else
      "$PY" scripts/stage6_trt_timing_cache_transfer_probe_v1.py \
        --base-onnx "$BASE_ONNX" --compressed-onnx "$onnx" --cache-out "$CACHE" \
        --base-width "$BASE_WIDTH" --compressed-width "$width" \
        --intended-q-mode "$q_mode" --gpu "$GPU" --out "$out"
    fi
  fi
done <"$FORMAL_ROOT/$BACKEND/tune_then_compress/attempts.tsv"

date -Is >"$QUEUE_DONE"
rm -f "$QUEUE_FAILED"
trap - EXIT
