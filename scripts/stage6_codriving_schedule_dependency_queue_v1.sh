#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/jichengzhi/V2X}
FORMAL_ROOT=${FORMAL_ROOT:-$REPO/results/stage6_codriving_formal_20260722}
GPU=${GPU:?GPU is required}
WAIT_FOR_FILE=${WAIT_FOR_FILE:?WAIT_FOR_FILE is required}
POLL_SECONDS=${POLL_SECONDS:-30}
EVIDENCE_BUNDLE=${EVIDENCE_BUNDLE:-$FORMAL_ROOT/stage6_codriving_paper_evidence_bundle_preliminary.json}
AUDIT_JSON=${AUDIT_JSON:-$FORMAL_ROOT/independent_validation_schedule/stage6_schedule_independent_validation_audit_v1.json}

gpu_idle() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F, -v wanted="$GPU" '
      $1 + 0 == wanted + 0 {
        seen=1; memory=$2+0; utilization=$3+0
        if (memory <= 100 && utilization <= 10) ok=1
      }
      END { exit !(seen && ok) }
    '
}

while [[ ! -s "$WAIT_FOR_FILE" ]]; do
  sleep "$POLL_SECONDS"
done

stable=0
while (( stable < 3 )); do
  if gpu_idle; then stable=$((stable + 1)); else stable=0; fi
  (( stable == 3 )) || sleep "$POLL_SECONDS"
done

exec env REPO="$REPO" FORMAL_ROOT="$FORMAL_ROOT" TVM_GPU="$GPU" TRT_GPU="$GPU" \
  EVIDENCE_BUNDLE="$EVIDENCE_BUNDLE" AUDIT_JSON="$AUDIT_JSON" \
  bash "$REPO/scripts/stage6_run_codriving_schedule_independent_v1.sh"
