#!/usr/bin/env bash
set -u

ROOT=/home/jichengzhi/V2X
PY=/exdata/jichengzhi/tvm310/bin/python
OUT=$ROOT/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627
RUN_ROOT=$OUT/raw/fp16_rewritten_tensorcore_remeasure/20260701_full60
QUEUE=$RUN_ROOT/launcher/queue.jsonl
STATUS=$RUN_ROOT/launcher/status.jsonl
LAT_ROWS=$OUT/rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl
ENE_ROWS=$OUT/rows/fp16_rewritten_tensorcore_full60_energy_rows_v1.jsonl

mkdir -p "$RUN_ROOT/launcher" "$RUN_ROOT/logs" "$OUT/rows"

export PYTHONPATH="$ROOT:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages:${PYTHONPATH:-}"
NVLIBS=""
if [ -f /exdata/jichengzhi/tvm_nvlibs.path ]; then
  NVLIBS=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
fi
export LD_LIBRARY_PATH="/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib:${NVLIBS}:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/cuda-12.2/bin:${PATH:-}"

run_item() {
  item_json=$1
  lane_gpu=$2
  eval "$($PY - <<'PY' "$item_json"
import json, shlex, sys
item = json.loads(sys.argv[1])
for key, value in item.items():
    print(f'{key}={shlex.quote(str(value))}')
PY
)"
  ts=$(date +%Y%m%d_%H%M%S)
  lat_run="fp16_rewritten_tensorcore_full60_${label}_latency_${ts}_gpu${lane_gpu}"
  ene_run="fp16_rewritten_tensorcore_full60_${label}_energy_threaded_${ts}_gpu${lane_gpu}"
  lat_stdout="$RUN_ROOT/logs/${lat_run}.stdout"
  lat_stderr="$RUN_ROOT/logs/${lat_run}.stderr"
  ene_stdout="$RUN_ROOT/logs/${ene_run}.stdout"
  ene_stderr="$RUN_ROOT/logs/${ene_run}.stderr"
  printf '{"event":"latency_start","label":"%s","gpu":"%s","run_id":"%s","ts":"%s"}\n' "$label" "$lane_gpu" "$lat_run" "$(date -Iseconds)" >> "$STATUS"
  "$PY" "$ROOT/scripts/stage2_measure_fp16_rewritten_artifact.py" \
    --kind latency \
    --label "$label" \
    --gpu "$lane_gpu" \
    --artifact "$artifact" \
    --rewrite-report "$rewrite_report" \
    --route "$route" \
    --width "$width" \
    --config-id "$config_id" \
    --run-id "$lat_run" \
    --raw-root "$RUN_ROOT" \
    --out-jsonl "$LAT_ROWS" \
    --warmup-iters 20 \
    --measure-iters 300 \
    --repeat 5 \
    > "$lat_stdout" 2> "$lat_stderr"
  lat_rc=$?
  printf '{"event":"latency_finish","label":"%s","gpu":"%s","run_id":"%s","rc":%s,"ts":"%s"}\n' "$label" "$lane_gpu" "$lat_run" "$lat_rc" "$(date -Iseconds)" >> "$STATUS"
  if [ "$lat_rc" -ne 0 ]; then
    return "$lat_rc"
  fi
  printf '{"event":"energy_start","label":"%s","gpu":"%s","run_id":"%s","latency_run_id":"%s","ts":"%s"}\n' "$label" "$lane_gpu" "$ene_run" "$lat_run" "$(date -Iseconds)" >> "$STATUS"
  "$PY" "$ROOT/scripts/stage2_measure_fp16_rewritten_artifact.py" \
    --kind energy \
    --label "$label" \
    --gpu "$lane_gpu" \
    --artifact "$artifact" \
    --rewrite-report "$rewrite_report" \
    --route "$route" \
    --width "$width" \
    --config-id "$config_id" \
    --latency-run-id "$lat_run" \
    --run-id "$ene_run" \
    --raw-root "$RUN_ROOT" \
    --out-jsonl "$ENE_ROWS" \
    --energy-warmup-iters 20 \
    --energy-measure-iters 300 \
    --energy-min-active-s 5 \
    --energy-sync-interval-iters 50 \
    > "$ene_stdout" 2> "$ene_stderr"
  ene_rc=$?
  printf '{"event":"energy_finish","label":"%s","gpu":"%s","run_id":"%s","rc":%s,"ts":"%s"}\n' "$label" "$lane_gpu" "$ene_run" "$ene_rc" "$(date -Iseconds)" >> "$STATUS"
  return "$ene_rc"
}

run_lane() {
  lane_gpu=$1
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    item_gpu=$("$PY" - <<'PY' "$line"
import json, sys
print(json.loads(sys.argv[1])["gpu"])
PY
)
    if [ "$item_gpu" = "$lane_gpu" ]; then
      if ! run_item "$line" "$lane_gpu"; then
        printf '{"event":"lane_item_failed","gpu":"%s","item":%s,"ts":"%s"}\n' "$lane_gpu" "$line" "$(date -Iseconds)" >> "$STATUS"
      fi
    fi
  done < "$QUEUE"
}

printf '{"event":"launcher_start","ts":"%s"}\n' "$(date -Iseconds)" >> "$STATUS"
run_lane 0 & pid0=$!
run_lane 6 & pid6=$!
echo "$pid0" > "$RUN_ROOT/launcher/lane_gpu0.pid"
echo "$pid6" > "$RUN_ROOT/launcher/lane_gpu6.pid"
printf '{"event":"lane_pids","gpu0":%s,"gpu6":%s,"ts":"%s"}\n' "$pid0" "$pid6" "$(date -Iseconds)" >> "$STATUS"
rc=0
wait "$pid0" || rc=1
wait "$pid6" || rc=1
printf '{"event":"launcher_finish","rc":%s,"ts":"%s"}\n' "$rc" "$(date -Iseconds)" >> "$STATUS"
exit "$rc"
