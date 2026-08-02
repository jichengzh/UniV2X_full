#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
OUTPUT_ROOT=${OUTPUT_ROOT:-$FORMAL_ROOT/tvm/default_schedule_epoch23_repair_v1}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
TVM_PY=${TVM_PY:-/exdata/jichengzhi/tvm310/bin/python}
TVM_SITE=${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}
BASE_ONNX=${BASE_ONNX:-$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$(dirname "$CHECKPOINT_PATH")}
TUNED_WORK_DIR=${TUNED_WORK_DIR:-/exdata/jichengzhi/stage6_pyramid_formal_20260720/tvm_schedule_only/workdirs/smbo_64x128x256_fp32}
BUILD_GPU=${BUILD_GPU:-0}
DEFAULT_AP_GPU=${DEFAULT_AP_GPU:-3}
TUNED_AP_GPU=${TUNED_AP_GPU:-4}
REPEAT_GPUS=${REPEAT_GPUS:-0,2,5}

cd "$REPO"
mkdir -p "$OUTPUT_ROOT/artifacts" "$OUTPUT_ROOT/ap/default" "$OUTPUT_ROOT/ap/tuned" \
  "$OUTPUT_ROOT/performance"

"$PY" - "$BASE_ONNX" "$CHECKPOINT_PATH" "$TUNED_WORK_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

onnx, checkpoint, work_dir = map(Path, sys.argv[1:])
expected = {
    onnx: "3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff",
    checkpoint: "4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2",
}
for path, digest in expected.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise ValueError(f"SHA mismatch for {path}: {observed}")
for name in ("database_workload.json", "database_tuning_record.json"):
    path = work_dir / name
    if not path.is_file():
        raise FileNotFoundError(path)
PY

TVM_NVLIBS=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
TVM_ENV=(
  env
  "LD_LIBRARY_PATH=$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib:$TVM_NVLIBS:${LD_LIBRARY_PATH:-}"
)

"${TVM_ENV[@]}" "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
  --onnx "$BASE_ONNX" --work-dir "$OUTPUT_ROOT/zero_trial_no_database" \
  --gpu "$BUILD_GPU" --skip-database \
  --artifact "$OUTPUT_ROOT/artifacts/tvm_fp32_default.so" \
  --report "$OUTPUT_ROOT/artifacts/tvm_fp32_default_report.json"

"${TVM_ENV[@]}" "$TVM_PY" scripts/stage6_export_tvm_fp32_schedule_artifact_v1.py \
  --onnx "$BASE_ONNX" --work-dir "$TUNED_WORK_DIR" --gpu "$BUILD_GPU" \
  --artifact "$OUTPUT_ROOT/artifacts/tvm_fp32_tuned64.so" \
  --report "$OUTPUT_ROOT/artifacts/tvm_fp32_tuned64_report.json"

run_ap() {
  local policy=$1
  local gpu=$2
  local artifact=$3
  local root="$OUTPUT_ROOT/ap/$policy"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/stage2_h800_fp16_rewritten_activation_bridge.py \
    --label "stage6_pyramid_tvm_fp32_${policy}_epoch23" \
    --ckpt-dir "$CHECKPOINT_DIR" --checkpoint-path "$CHECKPOINT_PATH" \
    --raw-dir "$root" --eval-range 102.4,51.2 \
    --artifact-path "$artifact" --artifact-input-dtype float32 \
    --persistent-worker --num-samples 1789 --full-ap-min-samples 1789 \
    --export-report-json "$root/full_ap_eval_report.json" --gpu-id "$gpu" \
    >"$root/runner.log" 2>&1
}

run_repeat() {
  local index=$1
  local gpu=$2
  local root="$OUTPUT_ROOT/performance/repeat_$index"
  local common=(
    --model pyramid_lidar
    --label stage6_pyramid_base_fp32
    --phase stage6_pyramid_tvm_default_schedule_epoch23_repair_v1
    --gpu "$gpu"
    --onnx "$BASE_ONNX"
    --work-dir "$TUNED_WORK_DIR"
    --width 64,128,256
    --candidate-id stage6:pyramid:base:fp32
    --software-point-id stage6:pyramid:w64x128x256
    --config-id-tuned stage6_pyramid_tvm_fp32_tuned64
    --config-id-default stage6_pyramid_tvm_fp32_default
    --precision fp32
    --quant-policy fp32
    --quant-method h800_tvm_relax_fp32
    --optimized-scope backbone_only
    --full-network-claim false
    --measurement-source stage6_pyramid_tvm_default_schedule_epoch23_repair_v1
    --tune-budget reuse_frozen_64_trial_database
    --warmup-iters 20
    --measure-iters 300
    --repeat 5
    --energy-warmup-iters 20
    --energy-measure-iters 300
    --energy-sampling-mode threaded_window
    --energy-min-active-s 5
  )
  mkdir -p "$root"
  "${TVM_ENV[@]}" "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
    --kind latency "${common[@]}" \
    --run-id "repeat_${index}_latency" \
    --raw-root "$root/latency_raw" --out-jsonl "$root/latency_rows.jsonl" \
    >"$root/latency.log" 2>&1
  "${TVM_ENV[@]}" "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
    --kind energy "${common[@]}" --energy-schedule-policy default \
    --config-id-tuned stage6_pyramid_tvm_fp32_default \
    --latency-run-id "repeat_${index}_latency_default" \
    --run-id "repeat_${index}_energy_default" \
    --raw-root "$root/energy_default_raw" --out-jsonl "$root/energy_default_rows.jsonl" \
    >"$root/energy_default.log" 2>&1
  "${TVM_ENV[@]}" "$TVM_PY" scripts/stage2_h800_run_measurement_job.py \
    --kind energy "${common[@]}" --energy-schedule-policy metaschedule_tuned \
    --latency-run-id "repeat_${index}_latency_tuned" \
    --run-id "repeat_${index}_energy_tuned" \
    --raw-root "$root/energy_tuned_raw" --out-jsonl "$root/energy_tuned_rows.jsonl" \
    >"$root/energy_tuned.log" 2>&1
}

run_ap default "$DEFAULT_AP_GPU" "$OUTPUT_ROOT/artifacts/tvm_fp32_default.so" &
default_ap_pid=$!
run_ap tuned "$TUNED_AP_GPU" "$OUTPUT_ROOT/artifacts/tvm_fp32_tuned64.so" &
tuned_ap_pid=$!

IFS=',' read -r -a repeat_gpus <<<"$REPEAT_GPUS"
repeat_pids=()
for index in 0 1 2; do
  run_repeat "$index" "${repeat_gpus[$index]}" &
  repeat_pids+=("$!")
done

wait "$default_ap_pid"
wait "$tuned_ap_pid"
for pid in "${repeat_pids[@]}"; do
  wait "$pid"
done

"$PY" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

ap = {}
for policy in ("default", "tuned"):
    path = root / f"ap/{policy}/full_ap_eval_report.json"
    payload = read(path)
    if payload.get("status") != "success":
        raise ValueError(f"{policy} AP failed")
    if payload.get("processed_samples") != 1789 or payload.get("failed_samples") != 0:
        raise ValueError(f"{policy} AP completeness failure")
    if payload.get("resume_epoch") != 23:
        raise ValueError(f"{policy} used checkpoint epoch {payload.get('resume_epoch')}")
    if abs(float(payload["ap70"]) - 0.6311015785876983) > 0.01:
        raise ValueError(f"{policy} AP70 violates baseline contract: {payload['ap70']}")
    ap[policy] = {
        "ap30": float(payload["ap30"]),
        "ap50": float(payload["ap50"]),
        "ap70": float(payload["ap70"]),
        "path": str(path),
        "sha256": sha(path),
    }

metrics = {"default": {"latency": [], "energy": []}, "tuned": {"latency": [], "energy": []}}
repeat_evidence = []
for index in range(3):
    repeat = root / f"performance/repeat_{index}"
    latency = read(repeat / f"latency_raw/repeat_{index}_latency/latency_result.json")
    if latency.get("status") != "success":
        raise ValueError(f"repeat {index} latency failed")
    metrics["default"]["latency"].append(float(latency["default_us"]) / 1000.0)
    metrics["tuned"]["latency"].append(float(latency["tuned_us"]) / 1000.0)
    item = {"repeat_index": index, "files": []}
    for policy in ("default", "tuned"):
        path = repeat / f"energy_{policy}_raw/repeat_{index}_energy_{policy}/energy_result.json"
        energy = read(path)
        if energy.get("status") != "success":
            raise ValueError(f"repeat {index} {policy} energy failed")
        latency_ms = metrics[policy]["latency"][-1]
        metrics[policy]["energy"].append(float(energy["watt_avg"]) * latency_ms / 1000.0)
        item["files"].append({"path": str(path), "sha256": sha(path)})
    latency_path = repeat / f"latency_raw/repeat_{index}_latency/latency_result.json"
    item["files"].append({"path": str(latency_path), "sha256": sha(latency_path)})
    repeat_evidence.append(item)

arms = {}
for policy in ("default", "tuned"):
    latencies = metrics[policy]["latency"]
    energies = metrics[policy]["energy"]
    arms[policy] = {
        **ap[policy],
        "latency_median_ms": statistics.median(latencies),
        "energy_median_j": statistics.median(energies),
        "latency_values_ms": latencies,
        "energy_values_j": energies,
        "latency_cv": statistics.pstdev(latencies) / statistics.mean(latencies),
        "energy_cv": statistics.pstdev(energies) / statistics.mean(energies),
        "tuning_trials": 0 if policy == "default" else 64,
        "schedule_policy": "tvm_default_zero_trial" if policy == "default" else "tvm_metaschedule_64",
    }

audit = {
    "schema_version": "stage6_pyramid_tvm_default_schedule_epoch23_repair_v1",
    "status": "passed",
    "model": "pyramid",
    "backend": "tvm_h800",
    "width": [64, 128, 256],
    "q_mode": "fp32",
    "checkpoint_epoch": 23,
    "arms": arms,
    "schedule_speedup": arms["default"]["latency_median_ms"] / arms["tuned"]["latency_median_ms"],
    "repeat_evidence": repeat_evidence,
}
path = root / "repair_audit.json"
path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, sort_keys=True))
PY
