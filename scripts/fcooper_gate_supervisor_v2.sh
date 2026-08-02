#!/usr/bin/env bash
set -euo pipefail

ROOT=
PY=${PY:-/exdata/jichengzhi/conda_envs/UniV2X_2.0/bin/python}
HEAL=${HEAL:-/exdata/jichengzhi/heal_research/HEAL}
CODE_ROOT=${CODE_ROOT:-/home/jichengzhi/V2X}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "invalid --root" >&2; exit 2; }

wait_for_training() {
  local tag="$1"
  local pid_file="$ROOT/pids/gate_${tag}_train.pid"
  local report="$ROOT/gates/$tag/recovery_training_report.json"
  local source="$ROOT/gates/$tag"
  while [[ ! -s "$report" ]]; do
    local pid
    pid=$(cat "$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "training exited without report: $tag" >&2
      return 1
    fi
    sleep 60
  done
  PYTHONPATH="$CODE_ROOT" "$PY" - \
    "$report" \
    "$ROOT/contracts/recovery_training_contract.json" \
    "$source/config.yaml" \
    "$source/net_epoch_bestval_at23.pth" \
    "$source/recovered_checkpoint.pth" <<'PY'
import sys
from pathlib import Path

from scripts.fcooper_execute_measurement_row_v2 import (
    validate_recovery_training_evidence,
)

audit = validate_recovery_training_evidence(
    report_path=Path(sys.argv[1]),
    recovery_contract_path=Path(sys.argv[2]),
    config_path=Path(sys.argv[3]),
    initial_checkpoint_path=Path(sys.argv[4]),
    recovered_checkpoint_path=Path(sys.argv[5]),
)
assert audit["passed"] is True
PY
}

run_gate() {
  local gpu="$1" tag="$2" width="$3"
  local source="$ROOT/gates/$tag"
  local output="$source/formal_gate"
  mkdir -p "$output"
  PYTHONPATH="$CODE_ROOT:$HEAL" \
    "$PY" "$CODE_ROOT/scripts/fcooper_export_source_v2.py" \
      --config "$source/config.yaml" \
      --checkpoint "$source/recovered_checkpoint.pth" \
      --training-report "$source/recovery_training_report.json" \
      --width "$width" --formal \
      --onnx "$output/fcooper_dense.onnx" \
      --report "$output/source_export_report.json" \
      >"$ROOT/logs/gate_${tag}_export_formal.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$CODE_ROOT:$HEAL" \
    "$PY" "$CODE_ROOT/framework/trt_baseline/trt_profile_v1.py" \
      --onnx "$output/fcooper_dense.onnx" --precision fp16 --gpu 0 \
      --calib-dir "$ROOT/calibration_validate/npy" \
      --calibration-dataset OPV2V-validate \
      --builder-optimization-level 5 \
      --warmup 20 --iters 300 --repeat 5 --energy-secs 5 \
      --artifact-dir "$output/engine" --out "$output/performance.json" \
      >"$ROOT/logs/gate_${tag}_performance.log" 2>&1
  (
    cd "$HEAL"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$CODE_ROOT:$HEAL" \
      "$PY" "$CODE_ROOT/scripts/fcooper_trt_ap_bridge_v1.py" \
        --config "$source/config.yaml" --checkpoint-dir "$source" \
        --checkpoint "$source/recovered_checkpoint.pth" \
        --engine "$output/engine/compiled.engine" \
        --output-json "$output/ap_report.json" --num-workers 4 \
        >"$ROOT/logs/gate_${tag}_full_ap.log" 2>&1
  )
}

wait_for_training 64x96x192x96x192
wait_for_training 32x64x128x64x128

run_gate 5 64x96x192x96x192 64,96,192,96,192 &
pid_a=$!
run_gate 6 32x64x128x64x128 32,64,128,64,128 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

"$PY" - "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for tag in ("64x96x192x96x192", "32x64x128x64x128"):
    gate = root / "gates" / tag / "formal_gate"
    ap_path = gate / "ap_report.json"
    performance_path = gate / "performance.json"
    ap = json.loads(ap_path.read_text())
    performance = json.loads(performance_path.read_text())
    rows.append(
        {
            "width": [int(value) for value in tag.split("x")],
            "status": ap["status"],
            "dataset_samples": ap["dataset_samples"],
            "processed_samples": ap["processed_samples"],
            "fallback_samples": ap["fallback_samples"],
            "ap30": ap["ap30"],
            "ap50": ap["ap50"],
            "ap70": ap["ap70"],
            "latency_ms": performance["lat_p50_ms"],
            "energy_j": performance["energy_j"],
            "ap_report_sha256": hashlib.sha256(ap_path.read_bytes()).hexdigest(),
            "performance_sha256": hashlib.sha256(
                performance_path.read_bytes()
            ).hexdigest(),
        }
    )
passed = all(
    row["status"] == "success_full"
    and row["processed_samples"] == 2170
    and row["fallback_samples"] == 0
    and row["ap70"] >= 0.10
    for row in rows
)
summary = {
    "schema_version": "fcooper_recovery_numeric_gate_v2",
    "status": "passed" if passed else "failed",
    "non_collapse_ap70_floor": 0.10,
    "rows": rows,
    "t16_search_allowed": passed,
}
path = root / "gates" / "recovery_numeric_gate_summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
if not passed:
    raise SystemExit("F-Cooper recovery numeric gate failed")
(root / "gates" / "gate.ready").touch()
print(json.dumps(summary, indent=2, sort_keys=True))
PY
