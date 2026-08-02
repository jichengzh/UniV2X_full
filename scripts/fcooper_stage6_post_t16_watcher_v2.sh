#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=
PYTHON=
CODE_ROOT=
BASE_PROBE_ONNX=
POLL_SECONDS=60
SUPERVISOR_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root) WORK_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --code-root) CODE_ROOT=$2; shift 2 ;;
    --base-probe-onnx) BASE_PROBE_ONNX=$2; shift 2 ;;
    --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
    --) shift; SUPERVISOR_ARGS=("$@"); break ;;
    *) echo "unknown watcher argument: $1" >&2; exit 2 ;;
  esac
done

for value in "$WORK_ROOT" "$PYTHON" "$CODE_ROOT" "$BASE_PROBE_ONNX"; do
  [[ -n "$value" ]] || { echo "missing required watcher argument" >&2; exit 2; }
  [[ "$value" != *"fcooper_workpackage_a_20260723"* ]] || {
    echo "watcher rejects pilot path" >&2
    exit 2
  }
done
[[ ${#SUPERVISOR_ARGS[@]} -gt 0 ]] || {
  echo "missing Stage6 supervisor arguments after --" >&2
  exit 2
}

T16="$WORK_ROOT/search/S5-FCO-TRT-V2/feedback_history_final_t16.json"
OBSERVED="$WORK_ROOT/contracts/stage6_observed_graph_evidence_v2.json"
while [[ ! -s "$T16" ]]; do
  sleep "$POLL_SECONDS"
done

"$PYTHON" "$CODE_ROOT/scripts/fcooper_prepare_stage6_graph_evidence_v2.py" \
  --t16-feedback-json "$T16" \
  --base-probe-onnx "$BASE_PROBE_ONNX" \
  --output-json "$OBSERVED"

exec "$CODE_ROOT/scripts/fcooper_stage6_five_arm_supervisor_v2.sh" \
  "${SUPERVISOR_ARGS[@]}"
