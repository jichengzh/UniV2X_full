#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
VALIDATION_ROOT=${VALIDATION_ROOT:-$FORMAL_ROOT/independent_validation_controls_parallel}
GPU_POOL=${GPU_POOL:-3,4,5,6}

cd "$REPO"
plan="$VALIDATION_ROOT/stage6_independent_validation_plan_v1.json"
plan_count=$(jq -r '.plan_count' "$plan")
IFS=',' read -r -a gpus <<<"$GPU_POOL"
(( plan_count <= ${#gpus[@]} )) || {
  echo "plan count exceeds available isolated GPUs: plans=$plan_count gpus=${#gpus[@]}" >&2
  exit 2
}

pids=()
for ((index=0; index<plan_count; index++)); do
  scoped="$VALIDATION_ROOT/stage6_independent_validation_plan_$(printf '%02d' "$index").json"
  audit="$VALIDATION_ROOT/stage6_independent_validation_audit_$(printf '%02d' "$index").json"
  log="$VALIDATION_ROOT/runner_$(printf '%02d' "$index").log"
  env REPO="$REPO" PLAN_JSON="$scoped" GPU_POOL="${gpus[$index]}" AUDIT_JSON="$audit" \
    bash scripts/stage6_run_independent_validation_v1.sh >"$log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for ((index=0; index<plan_count; index++)); do
  test -s "$VALIDATION_ROOT/stage6_independent_validation_audit_$(printf '%02d' "$index").json"
done
date -Is >"$FORMAL_ROOT/controller/stage6_controls_parallel.done"
