#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
ROOT=${ROOT:-$REPO/results/stage5_single_target_search_v2_gold176_20260718}
GPU_POOL=${GPU_POOL:-0,3,6,7}
WAIT_INTERVAL_S=${WAIT_INTERVAL_S:-60}
STABLE_IDLE_CHECKS=${STABLE_IDLE_CHECKS:-3}

CPU_DONE="$ROOT/controller/stage5_postsearch_cpu_closure_v1.done"
CPU_FAILED="$ROOT/controller/stage5_postsearch_cpu_closure_v1.failed"
VALIDATION_DONE="$ROOT/independent_validation_v1/independent_validation.done"
DONE="$ROOT/controller/stage5_postsearch_independent_launcher_v1.done"
FAILED="$ROOT/controller/stage5_postsearch_independent_launcher_v1.failed"

[[ "$WAIT_INTERVAL_S" =~ ^[1-9][0-9]*$ ]]
[[ "$STABLE_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]]

trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "%s rc=%s\n" "$(date -Is)" "$rc" >"$FAILED"; fi' EXIT

while [[ ! -s "$CPU_DONE" ]]; do
  [[ ! -s "$CPU_FAILED" ]] || {
    echo "CPU closure failed; refusing to start independent validation" >&2
    exit 1
  }
  sleep "$WAIT_INTERVAL_S"
done

gpus_idle() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, -v wanted="$GPU_POOL" '
    BEGIN {
      count = split(wanted, ids, ",")
      for (i = 1; i <= count; i++) required[ids[i] + 0] = 1
    }
    {
      gpu = $1 + 0
      memory = $2 + 0
      utilization = $3 + 0
      if (gpu in required) {
        seen[gpu] = 1
        if (memory > 100 || utilization > 10) busy[gpu] = 1
      }
    }
    END {
      for (gpu in required) {
        if (!(gpu in seen) || (gpu in busy)) exit 1
      }
    }
  '
}

stable=0
while (( stable < STABLE_IDLE_CHECKS )); do
  if gpus_idle; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  (( stable == STABLE_IDLE_CHECKS )) || sleep "$WAIT_INTERVAL_S"
done

if [[ ! -s "$VALIDATION_DONE" ]]; then
  cd "$REPO"
  env GPU_POOL="$GPU_POOL" ROOT="$ROOT" FORMAL_ROOT="$ROOT" \
    bash scripts/stage5_run_independent_validation_v1.sh
fi

date -Is >"$DONE"
rm -f "$FAILED"
trap - EXIT
