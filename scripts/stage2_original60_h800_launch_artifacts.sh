#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jichengzhi/V2X}"
BASE="${BASE:-multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/exdata/jichengzhi/s2_tvm}"
TVM_PYTHON="${TVM_PYTHON:-/exdata/jichengzhi/tvm310/bin/python}"
TVM_SITE="${TVM_SITE:-/exdata/jichengzhi/tvm310/lib/python3.10/site-packages}"
CUDA_BIN="${CUDA_BIN:-/usr/local/cuda-12.2/bin}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5}"
MAX_TRIALS="${MAX_TRIALS:-32}"
MAX_IDLE_MEM_MB="${MAX_IDLE_MEM_MB:-128}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"
APPLY_DATABASE_SMOKE="${APPLY_DATABASE_SMOKE:-1}"
POST_JOB_COOLDOWN_SECONDS="${POST_JOB_COOLDOWN_SECONDS:-5}"

cd "$REPO_ROOT"

LOG_DIR="$BASE/logs/original60_artifacts"
STATE_DIR="$BASE/jobs"
SHARD_DIR="$BASE/jobs/original60_artifact_shards"
ARTIFACT_DIR="$BASE/artifacts"
EXPORT_DIR="$BASE/exports"
mkdir -p "$LOG_DIR" "$STATE_DIR" "$SHARD_DIR" "$ARTIFACT_DIR" "$EXPORT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/launcher_${STAMP}.log"
RUN_PID="$STATE_DIR/artifact_build_original60_launcher.pid"
echo "$$" > "$RUN_PID"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_LOG"
}

gpu_mem_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$1" | awk '{print int($1)}'
}

gpu_has_compute_process() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v g="$gpu" '$1 == g && $2 != "-" { found=1 } END { exit found ? 0 : 1 }'
}

gpu_idle() {
  local gpu="$1"
  local mem
  mem="$(gpu_mem_mb "$gpu")"
  if [ "$mem" -gt "$MAX_IDLE_MEM_MB" ]; then
    return 1
  fi
  if gpu_has_compute_process "$gpu"; then
    return 1
  fi
  return 0
}

all_gpus_idle() {
  IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
  local gpu
  for gpu in "${GPUS[@]}"; do
    if ! gpu_idle "$gpu"; then
      return 1
    fi
  done
  return 0
}

log_gpu_status() {
  log "GPU status:"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tee -a "$RUN_LOG"
  nvidia-smi pmon -c 1 | sed -n '1,20p' | tee -a "$RUN_LOG"
}

wait_for_idle_gate() {
  local start
  start="$(date +%s)"
  while true; do
    if all_gpus_idle; then
      log "idle gate passed for GPUs $GPUS_CSV"
      log_gpu_status
      return 0
    fi
    local now elapsed
    now="$(date +%s)"
    elapsed=$((now - start))
    log "idle gate not passed after ${elapsed}s; waiting ${POLL_SECONDS}s"
    log_gpu_status
    if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
      log "ERROR: idle gate timeout after ${elapsed}s"
      return 2
    fi
    sleep "$POLL_SECONDS"
  done
}

remaining_jobs_for_gpu() {
  local gpu="$1"
  python3 - "$SHARD_DIR/artifact_build_queue_original60_gpu${gpu}.jsonl" "$STATE_DIR/artifact_build_state_original60_gpu${gpu}.jsonl" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
state_path = Path(sys.argv[2])
plan = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
latest = {}
if state_path.exists():
    for line in state_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["job_id"]] = row
remaining = [row for row in plan if latest.get(row["job_id"], {}).get("status") != "succeeded"]
print(len(remaining))
PY
}

run_gpu_controller() {
  local gpu="$1"
  local shard="$SHARD_DIR/artifact_build_queue_original60_gpu${gpu}.jsonl"
  local state="$STATE_DIR/artifact_build_state_original60_gpu${gpu}.jsonl"
  local manifest="$ARTIFACT_DIR/artifact_build_manifest_original60_gpu${gpu}.jsonl"
  local controller_log="$LOG_DIR/gpu${gpu}_controller_${STAMP}.out"
  local extra_smoke=()
  if [ "$APPLY_DATABASE_SMOKE" = "1" ]; then
    extra_smoke=(--apply-database-smoke)
  fi

  while true; do
    local remaining
    remaining="$(remaining_jobs_for_gpu "$gpu")"
    if [ "$remaining" -le 0 ]; then
      log "controller gpu=${gpu} completed all shard jobs"
      return 0
    fi
    log "controller gpu=${gpu} remaining=${remaining}; starting one-job worker" | tee -a "$controller_log"
    "$TVM_PYTHON" scripts/stage2_original60_tvm_artifact_worker.py \
      --job-plan "$shard" \
      --job-state "$state" \
      --manifest-out "$manifest" \
      --log-dir "$LOG_DIR/gpu${gpu}_${STAMP}" \
      --max-trials "$MAX_TRIALS" \
      --max-jobs 1 \
      --resume \
      --require-gpu-idle \
      "${extra_smoke[@]}" \
      >> "$controller_log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
      log "controller gpu=${gpu} one-job worker rc=${rc}; will retry after cooldown" | tee -a "$controller_log"
    fi
    sleep "$POST_JOB_COOLDOWN_SECONDS"
  done
}

start_workers() {
  local nvlibs=""
  if [ -f /exdata/jichengzhi/tvm_nvlibs.path ]; then
    nvlibs="$(cat /exdata/jichengzhi/tvm_nvlibs.path)"
  fi
  export PYTHONPATH="$REPO_ROOT:$TVM_SITE:${PYTHONPATH:-}"
  export PATH="$CUDA_BIN:${PATH:-}"
  export LD_LIBRARY_PATH="$TVM_SITE/nvidia/cuda_runtime/lib:$TVM_SITE/tvm/lib${nvlibs:+:$nvlibs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
  local gpu
  for gpu in "${GPUS[@]}"; do
    log "starting controller gpu=${gpu} shard=$SHARD_DIR/artifact_build_queue_original60_gpu${gpu}.jsonl"
    run_gpu_controller "$gpu" &
    echo "$!" > "$STATE_DIR/artifact_build_original60_gpu${gpu}.pid"
  done
}

wait_workers() {
  IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
  local failed=0
  local gpu pid
  for gpu in "${GPUS[@]}"; do
    pid="$(cat "$STATE_DIR/artifact_build_original60_gpu${gpu}.pid")"
    if wait "$pid"; then
      log "worker gpu=${gpu} completed"
    else
      local rc=$?
      log "ERROR: worker gpu=${gpu} failed rc=${rc}"
      failed=1
    fi
  done
  return "$failed"
}

run_planner_and_readiness() {
  python3 scripts/stage2_plan_artifact_tasks.py \
    --candidate-queue "$BASE/candidates/candidate_queue.jsonl" \
    --quarantine-file "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
    --artifact-root "$ARTIFACT_ROOT" \
    --artifact-tasks-out "$ARTIFACT_DIR/artifact_tasks_original60_v1.jsonl" \
    --artifact-state-out "$ARTIFACT_DIR/artifact_state_original60_v1.jsonl" \
    --artifact-registry-out "$ARTIFACT_DIR/artifact_registry_original60_v1.jsonl" \
    --missing-artifact-out "$BASE/quarantine/missing_artifact_original60_v1.jsonl" \
    --created-at 2026-06-26T00:00:00Z \
    | tee -a "$RUN_LOG"

  python3 scripts/stage2_original60_artifact_readiness.py \
    --candidate-queue "$BASE/candidates/candidate_queue.jsonl" \
    --artifact-root "$ARTIFACT_ROOT" \
    --job-state-glob "$STATE_DIR/artifact_build_state_original60_gpu*.jsonl" \
    --quarantine-file "$BASE/quarantine/missing_artifact_original60_v1.jsonl" \
    --json-out "$EXPORT_DIR/original60_artifact_readiness_latest.json" \
    --md-out "$EXPORT_DIR/original60_artifact_readiness_latest.md" \
    --require-complete \
    | tee -a "$RUN_LOG"
}

main() {
  log "original60 artifact launcher started"
  log "repo=${REPO_ROOT} base=${BASE} artifact_root=${ARTIFACT_ROOT} gpus=${GPUS_CSV} max_trials=${MAX_TRIALS}"
  if ! wait_for_idle_gate; then
    exit 2
  fi
  start_workers
  local worker_rc=0
  if ! wait_workers; then
    worker_rc=1
  fi
  if ! run_planner_and_readiness; then
    log "ERROR: readiness gate failed"
    exit 1
  fi
  if [ "$worker_rc" -ne 0 ]; then
    log "ERROR: worker failures occurred even though readiness ran"
    exit "$worker_rc"
  fi
  log "original60 artifact launcher completed"
}

main "$@"
