#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_pyramid_formal_20260720}
OUTPUT_ROOT=${OUTPUT_ROOT:-$FORMAL_ROOT/trt/schedule_only_epoch23_repair_v1}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
AP_GPU=${AP_GPU:-6}
PERF_GPU=${PERF_GPU:-7}
BASE_ONNX=${BASE_ONNX:-$REPO/results/stage6_pyramid_launch_gate_20260720/artifacts/pyramid_base_multiscale.onnx}
CKPT_DIR=${CKPT_DIR:-$REPO/results/stage35_gold128_label_repair_v2_20260714/checkpoint_bundles/pyramid_64x128x256_epoch23}

cd "$REPO"
mkdir -p "$OUTPUT_ROOT/ap/full_1789" "$OUTPUT_ROOT/performance"

"$PY" - "$BASE_ONNX" "$CKPT_DIR/net_epoch23.pth" <<'PY'
import hashlib
import sys
from pathlib import Path

expected = {
    "onnx": "3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff",
    "checkpoint": "4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2",
}
for label, path_value in zip(("onnx", "checkpoint"), sys.argv[1:]):
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected[label]:
        raise ValueError(f"{label} SHA mismatch: {observed}")
PY

run_performance() {
  for repeat in 0 1 2; do
    root="$OUTPUT_ROOT/performance/repeat_$repeat"
    mkdir -p "$root/artifacts"
    "$PY" framework/trt_baseline/trt_profile_v1.py \
      --onnx "$BASE_ONNX" --precision fp32 --gpu "$PERF_GPU" \
      --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
      --artifact-dir "$root/artifacts" --out "$root/performance_result.json" \
      >"$root/runner.log" 2>&1
  done
}

run_ap() {
  CUDA_VISIBLE_DEVICES="$AP_GPU" "$PY" scripts/stage3_trt_multiscale_ap_bridge_v3.py \
    --label stage6_schedule_only_trt_fp32_epoch23_repair \
    --ckpt-dir "$CKPT_DIR" \
    --engine "$FORMAL_ROOT/trt/schedule_only/artifacts/compiled.engine" \
    --precision-tag fp32 --num-samples 1789 --full-ap-min-samples 1789 \
    --eval-range 102.4,51.2 --raw-dir "$OUTPUT_ROOT/ap/full_1789" \
    --report-json "$OUTPUT_ROOT/ap/full_1789/full_ap_eval_report.json" \
    >"$OUTPUT_ROOT/ap/full_1789/runner.log" 2>&1
}

run_performance & performance_pid=$!
run_ap & ap_pid=$!
wait "$performance_pid"
wait "$ap_pid"

"$PY" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
ap_path = root / "ap/full_1789/full_ap_eval_report.json"
ap = json.loads(ap_path.read_text())
if ap.get("status") != "success" or ap.get("processed_samples") != 1789:
    raise ValueError("full AP did not complete")
if ap.get("failed_samples") != 0 or ap.get("fallback_samples") != 0:
    raise ValueError("full AP used failure or fallback samples")
if ap.get("checkpoint_epoch") != 23:
    raise ValueError(f"wrong checkpoint epoch: {ap.get('checkpoint_epoch')}")
if abs(float(ap["ap70"]) - 0.6311015785876983) > 0.01:
    raise ValueError(f"AP70 violates native baseline contract: {ap['ap70']}")

repeats = []
for index in range(3):
    path = root / f"performance/repeat_{index}/performance_result.json"
    result = json.loads(path.read_text())
    if result.get("build_success") is not True:
        raise ValueError(f"performance repeat {index} failed")
    repeats.append(result)

latencies = [float(item["lat_p50_ms"]) for item in repeats]
energies = [float(item["energy_j"]) for item in repeats]
summary = {
    "schema_version": "stage6_pyramid_trt_schedule_only_epoch23_repair_v1",
    "status": "passed",
    "checkpoint_epoch": 23,
    "ap30": float(ap["ap30"]),
    "ap50": float(ap["ap50"]),
    "ap70": float(ap["ap70"]),
    "processed_samples": 1789,
    "latency_median_ms": statistics.median(latencies),
    "energy_median_j": statistics.median(energies),
    "latency_cv": statistics.pstdev(latencies) / statistics.mean(latencies),
    "energy_cv": statistics.pstdev(energies) / statistics.mean(energies),
    "ap_report_path": str(ap_path),
    "ap_report_sha256": hashlib.sha256(ap_path.read_bytes()).hexdigest(),
    "performance_repeats": [
        {
            "path": str(root / f"performance/repeat_{index}/performance_result.json"),
            "sha256": hashlib.sha256(
                (root / f"performance/repeat_{index}/performance_result.json").read_bytes()
            ).hexdigest(),
        }
        for index in range(3)
    ],
}
out = root / "repair_audit.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
