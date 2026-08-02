#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-results/stage5_two_model_search_v1_20260717}
PY=${PY:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}
GPU_POOL=${GPU_POOL:-5,7}
POLL_SECONDS=${POLL_SECONDS:-120}

cd "$REPO"
ROUND="$ROOT/round_00"
CTRL="$ROOT/controller"
LOGS="$ROOT/logs"
mkdir -p "$CTRL" "$LOGS"

status() {
  printf '%s %s\n' "$(date -Is)" "$*"
}

wait_for_marker_or_fail() {
  local marker="$1" pid_file="$2" label="$3" pid
  while [[ ! -f "$marker" ]]; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      status "$label exited without marker: $marker" >&2
      return 1
    fi
    status "waiting for $label pid=$pid"
    sleep "$POLL_SECONDS"
  done
  status "$label marker ready"
}

wait_for_process() {
  local pid_file="$1" label="$2" pid
  pid=$(cat "$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    status "waiting for $label pid=$pid"
    sleep "$POLL_SECONDS"
  done
  status "$label exited"
}

run_ap_shards() {
  local stage="$1" ap_dir="$2" artifact_root="$3" rc=0
  local gpu_a gpu_b pid_a pid_b
  IFS=',' read -r gpu_a gpu_b <<<"$GPU_POOL"
  [[ -n "$gpu_a" && -n "$gpu_b" ]]
  "$PY" scripts/stage3_execute_ap_plan_v3.py \
    --ap-plan-jsonl "$ap_dir/ap_plan_shard_0.jsonl" --stage "$stage" \
    --state-jsonl "$ap_dir/ap_state_shard_0.jsonl" --gpu "$gpu_a" \
    --univ2x-python "$PY" --artifact-root "$artifact_root/shard_0" &
  pid_a=$!
  "$PY" scripts/stage3_execute_ap_plan_v3.py \
    --ap-plan-jsonl "$ap_dir/ap_plan_shard_1.jsonl" --stage "$stage" \
    --state-jsonl "$ap_dir/ap_state_shard_1.jsonl" --gpu "$gpu_b" \
    --univ2x-python "$PY" --artifact-root "$artifact_root/shard_1" &
  pid_b=$!
  wait "$pid_a" || rc=1
  wait "$pid_b" || rc=1
  cat "$ap_dir/ap_state_shard_0.jsonl" "$ap_dir/ap_state_shard_1.jsonl" \
    >"$ap_dir/ap_state.jsonl.tmp"
  mv "$ap_dir/ap_state.jsonl.tmp" "$ap_dir/ap_state.jsonl"
  return "$rc"
}

if [[ -s "$ROOT/round_01/round_state.json" ]]; then
  "$PY" - \
    "$ROOT/round_01/round_state.json" \
    "$ROUND/measurement_request.json" \
    "$ROOT/round_01/measurement_request.json" \
    "$ROOT/round_01/feedback_audit.json" \
    "$ROOT/round_01/candidate_manifest.json" \
    "$ROOT/round_01/acquisition.json" \
    "$ROOT/round_01/model_bundle_manifest.json" \
    "$ROOT/round_01/predicted_candidates.json" \
    "$ROOT/round_01/feedback_rows.json" <<'PY'
import hashlib
import json
import sys

def load(index):
    with open(sys.argv[index], encoding="utf-8") as handle:
        return json.load(handle)

def digest(payload):
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

state = load(1)
previous_request = load(2)
next_request = load(3)
feedback_audit = load(4)
candidate_manifest = load(5)
acquisition = load(6)
model_bundle = load(7)
load(8)
load(9)
assert state["previous_measurement_request_sha256"] == digest(previous_request)
assert state["measurement_request_sha256"] == digest(next_request)
assert state["feedback_audit_sha256"] == digest(feedback_audit)
assert state["candidate_manifest_sha256"] == digest(candidate_manifest)
assert state["selection_sha256"] == digest(acquisition)
assert state["model_bundle_config_sha256"] == model_bundle["bundle_config_sha256"]
assert state["model_bundle_manifest_sha256"] == digest(model_bundle)
assert state["previous_round_feedback_verified"] is True
PY
  status "round-1 already exists; nothing to do"
  exit 0
fi

wait_for_marker_or_fail \
  "$ROOT/source_prep/codriving_16x48x64.done" \
  "$CTRL/source_codriving_gpu6.pid" \
  "CoDriving source"
wait_for_process "$CTRL/ap_pyramid_gpu5_7.pid" "Pyramid AP"

COD_REQ="$ROUND/codriving_measurement_request.json"
jq '{schema_version:.schema_version,group_count:1,row_count:4,
     rows:[.rows[]|select(.model=="codriving")]}' \
  "$ROUND/measurement_request.json" >"$COD_REQ.tmp"
mv "$COD_REQ.tmp" "$COD_REQ"

COD_PERF="$ROUND/codriving_performance"
mkdir -p "$COD_PERF"
"$PY" scripts/stage5_build_performance_plan_v1.py \
  --request-json "$COD_REQ" \
  --remote-artifact-root "$REPO/$ROOT/performance_execution" \
  --output-dir "$COD_PERF" --gpus "$GPU_POOL"
"$PY" scripts/stage3_execute_performance_plan_v3.py \
  --jobs-jsonl "$COD_PERF/performance_jobs.jsonl" \
  --state-jsonl "$COD_PERF/performance_state.jsonl" \
  --gpus "$GPU_POOL" --max-workers 2

COD_AP="$ROUND/codriving_ap"
mkdir -p "$COD_AP"
"$PY" scripts/stage5_ap_plan_v1.py \
  --manifest-json "$COD_PERF/performance_manifest.json" \
  --performance-jobs-jsonl "$COD_PERF/performance_jobs.jsonl" \
  --performance-state-jsonl "$COD_PERF/performance_state.jsonl" \
  --output-root "$REPO/$ROOT/ap_execution/codriving" \
  --output-json "$COD_AP/ap_plan.json" \
  --output-jsonl "$COD_AP/ap_plan.jsonl" || true
awk 'NR % 2 == 1' "$COD_AP/ap_plan.jsonl" >"$COD_AP/ap_plan_shard_0.jsonl"
awk 'NR % 2 == 0' "$COD_AP/ap_plan.jsonl" >"$COD_AP/ap_plan_shard_1.jsonl"
run_ap_shards sanity "$COD_AP" "$ROOT/ap_execution/codriving" || true
run_ap_shards full "$COD_AP" "$ROOT/ap_execution/codriving" || true

PYR_PERF="$ROUND/pyramid_performance"
PYR_AP="$ROUND/pyramid_ap"
PYR_FINAL="$ROUND/pyramid_final"
COD_FINAL="$ROUND/codriving_final"
"$PY" scripts/stage5_finalize_feedback_v1.py \
  --manifest-json "$PYR_PERF/performance_manifest.json" \
  --measurement-request-json "$ROUND/measurement_request.json" \
  --ap-plan-jsonl "$PYR_AP/ap_plan.jsonl" \
  --performance-state-jsonl "$PYR_PERF/performance_state_merged.jsonl" \
  --ap-state-jsonl "$PYR_AP/ap_state.jsonl" --output-dir "$PYR_FINAL"
"$PY" scripts/stage5_finalize_feedback_v1.py \
  --manifest-json "$COD_PERF/performance_manifest.json" \
  --measurement-request-json "$ROUND/measurement_request.json" \
  --ap-plan-jsonl "$COD_AP/ap_plan.jsonl" \
  --performance-state-jsonl "$COD_PERF/performance_state.jsonl" \
  --ap-state-jsonl "$COD_AP/ap_state.jsonl" --output-dir "$COD_FINAL"

jq -s 'add' \
  "$PYR_FINAL/stage5_feedback_final.json" \
  "$COD_FINAL/stage5_feedback_final.json" \
  >"$ROUND/feedback_rows_combined.json.tmp"
mv "$ROUND/feedback_rows_combined.json.tmp" "$ROUND/feedback_rows_combined.json"
jq -e 'length == 8 and ([.[].group_id] | unique | length) == 2' \
  "$ROUND/feedback_rows_combined.json" >/dev/null

"$PY" scripts/stage5_advance_round_v1.py \
  --feedback-rows-json "$ROUND/feedback_rows_combined.json" \
  --previous-round-state-json "$ROUND/round_state.json" \
  --previous-measurement-request-json "$ROUND/measurement_request.json" \
  --output-dir "$ROOT" --round-index 1
status "Stage5 round-0 real feedback closed and round-1 emitted"
