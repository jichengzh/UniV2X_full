#!/usr/bin/env bash
set -euo pipefail

ROOT=
GPU=
LANE=
PY=${PY:-/exdata/jichengzhi/conda_envs/UniV2X_2.0/bin/python}
HEAL=${HEAL:-/exdata/jichengzhi/heal_research/HEAL}
CODE_ROOT=${CODE_ROOT:-/home/jichengzhi/V2X}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --lane) LANE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "invalid --root" >&2; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "--gpu must be 0..7" >&2; exit 2; }
[[ "$LANE" =~ ^[1-3]$ ]] || { echo "--lane must be 1..3" >&2; exit 2; }

CALIBRATION="$ROOT/calibration_validate/npy"

run_probe() {
  local probe_id="$1" source_name="$2" precision="$3" optimization="$4"
  local output="$ROOT/probes/$probe_id"
  local artifact result
  if [[ "$precision" == "int8" ]]; then
    artifact="$output/artifact_detailed"
    result="$output/result_detailed.json"
  else
    artifact="$output/artifact"
    result="$output/result.json"
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT:$HEAL" \
    "$PY" "$CODE_ROOT/framework/trt_baseline/trt_profile_v1.py" \
      --onnx "$ROOT/probe_sources/$source_name/fcooper_dense.onnx" \
      --precision "$precision" --gpu 0 \
      --calib-dir "$CALIBRATION" --calibration-dataset OPV2V-validate \
      --builder-optimization-level "$optimization" \
      --warmup 10 --iters 50 --repeat 3 --energy-secs 3 \
      --artifact-dir "$artifact" --out "$result" \
      >"$ROOT/logs/probe_${probe_id}_profile.log" 2>&1
  (
    cd "$HEAL"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT:$HEAL" \
      "$PY" "$CODE_ROOT/scripts/fcooper_trt_ap_bridge_v1.py" \
        --config "$ROOT/probe_sources/$source_name/config.yaml" \
        --checkpoint-dir "$ROOT/probe_sources/$source_name" \
        --engine "$artifact/compiled.engine" \
        --output-json "$output/numeric_sanity16.json" \
        --num-workers 0 --max-samples 16 \
        >"$ROOT/logs/probe_${probe_id}_numeric.log" 2>&1
  )
}

case "$LANE" in
  1)
    run_probe fp16_base_default base fp16 0
    run_probe fp16_boundary_tuned boundary fp16 5
    ;;
  2)
    run_probe fp16_base_tuned base fp16 5
    run_probe int8_base base int8 5
    ;;
  3)
    run_probe fp16_boundary_default boundary fp16 0
    run_probe int8_boundary boundary int8 5
    ;;
esac
