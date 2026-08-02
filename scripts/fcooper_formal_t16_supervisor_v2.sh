#!/usr/bin/env bash
set -euo pipefail

TASK_ID=S5-FCO-TRT-V2
PILOT_FRAGMENT=fcooper_workpackage_a_20260723
CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORMAL_ROOT=
GATE_READY=
STAGE4_DIR=
COLDSTART_ROOT=
PROFILES_JSON=
SOURCE_REGISTRY_JSON=
FORMAL_CONTRACT_JSON=
PROBE_AUDIT_JSON=
PROBE_ISOLATION_AUDIT_JSON=
NUMERIC_GATE_SUMMARY_JSON=
SCANNER_EXECUTION_JSON=
ARTIFACT_ROOT=
HEAL_ROOT=
PYTHON=
SOURCE_CONFIG=
SOURCE_CHECKPOINT=
RECOVERY_CONTRACT=
CALIBRATION_DIR=
CALIBRATION_SUMMARY=
GPUS_CSV=1,2,3,4
POLL_SECONDS=60
MAX_RETRIES=2
SEED=20260723
BUILDER_OPTIMIZATION_LEVEL=5
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --formal-root) FORMAL_ROOT=$2; shift 2 ;;
    --gate-ready) GATE_READY=$2; shift 2 ;;
    --stage4-dir) STAGE4_DIR=$2; shift 2 ;;
    --coldstart-root) COLDSTART_ROOT=$2; shift 2 ;;
    --profiles-json) PROFILES_JSON=$2; shift 2 ;;
    --source-registry-json) SOURCE_REGISTRY_JSON=$2; shift 2 ;;
    --formal-contract-json) FORMAL_CONTRACT_JSON=$2; shift 2 ;;
    --probe-audit-json) PROBE_AUDIT_JSON=$2; shift 2 ;;
    --probe-isolation-audit-json) PROBE_ISOLATION_AUDIT_JSON=$2; shift 2 ;;
    --numeric-gate-summary-json) NUMERIC_GATE_SUMMARY_JSON=$2; shift 2 ;;
    --scanner-execution-json) SCANNER_EXECUTION_JSON=$2; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT=$2; shift 2 ;;
    --code-root) CODE_ROOT=$2; shift 2 ;;
    --heal-root) HEAL_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --source-config) SOURCE_CONFIG=$2; shift 2 ;;
    --source-checkpoint) SOURCE_CHECKPOINT=$2; shift 2 ;;
    --recovery-contract) RECOVERY_CONTRACT=$2; shift 2 ;;
    --calibration-dir) CALIBRATION_DIR=$2; shift 2 ;;
    --calibration-summary) CALIBRATION_SUMMARY=$2; shift 2 ;;
    --gpus) GPUS_CSV=$2; shift 2 ;;
    --poll-seconds) POLL_SECONDS=$2; shift 2 ;;
    --max-retries) MAX_RETRIES=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --builder-optimization-level) BUILDER_OPTIMIZATION_LEVEL=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ALL_PATHS=(
  "$FORMAL_ROOT" "$GATE_READY" "$STAGE4_DIR" "$COLDSTART_ROOT"
  "$PROFILES_JSON" "$SOURCE_REGISTRY_JSON" "$FORMAL_CONTRACT_JSON"
  "$PROBE_AUDIT_JSON" "$PROBE_ISOLATION_AUDIT_JSON"
  "$NUMERIC_GATE_SUMMARY_JSON" "$SCANNER_EXECUTION_JSON" "$ARTIFACT_ROOT"
  "$CODE_ROOT" "$HEAL_ROOT" "$PYTHON" "$SOURCE_CONFIG" "$SOURCE_CHECKPOINT"
  "$RECOVERY_CONTRACT" "$CALIBRATION_DIR" "$CALIBRATION_SUMMARY"
)
for path in "${ALL_PATHS[@]}"; do
  if [[ "$path" == *"$PILOT_FRAGMENT"* ]]; then
    echo "formal supervisor rejects pilot path: $path" >&2
    exit 2
  fi
done
for path in "${ALL_PATHS[@]}"; do
  [[ -n "$path" ]] || { echo "all path arguments are required" >&2; exit 2; }
done
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--poll-seconds must be a positive integer" >&2
  exit 2
}
[[ "$MAX_RETRIES" =~ ^[0-2]$ ]] || {
  echo "--max-retries must be 0, 1, or 2" >&2
  exit 2
}
IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
[[ ${#GPUS[@]} -eq 4 ]] || {
  echo "--gpus must contain exactly four GPU identifiers" >&2
  exit 2
}
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU identifier: $gpu" >&2; exit 2; }
done

INITIALIZE_COMMAND=(
  "$PYTHON" "$CODE_ROOT/scripts/stage5_initialize_fcooper_actual_v2.py"
  --stage4-dir "$STAGE4_DIR"
  --coldstart-root "$COLDSTART_ROOT"
  --profiles-json "$PROFILES_JSON"
  --source-registry-json "$SOURCE_REGISTRY_JSON"
  --formal-contract-json "$FORMAL_CONTRACT_JSON"
  --probe-audit-json "$PROBE_AUDIT_JSON"
  --probe-isolation-audit-json "$PROBE_ISOLATION_AUDIT_JSON"
  --numeric-gate-summary-json "$NUMERIC_GATE_SUMMARY_JSON"
  --scanner-execution-json "$SCANNER_EXECUTION_JSON"
  --output-dir "$FORMAL_ROOT"
  --seed "$SEED"
)

row_command() {
  local round_index=$1 row_index=$2 gpu=$3
  local request="$FORMAL_ROOT/round_$(printf '%02d' "$round_index")/measurement_request.json"
  ROW_COMMAND=(
    "$PYTHON" "$CODE_ROOT/scripts/fcooper_execute_measurement_row_v2.py"
    --request-json "$request"
    --row-index "$row_index"
    --gpu "$gpu"
    --artifact-root "$ARTIFACT_ROOT"
    --code-root "$CODE_ROOT"
    --heal-root "$HEAL_ROOT"
    --python "$PYTHON"
    --source-config "$SOURCE_CONFIG"
    --source-checkpoint "$SOURCE_CHECKPOINT"
    --recovery-contract "$RECOVERY_CONTRACT"
    --calibration-dir "$CALIBRATION_DIR"
    --calibration-summary "$CALIBRATION_SUMMARY"
    --builder-optimization-level "$BUILDER_OPTIMIZATION_LEVEL"
  )
}

finalize_command() {
  local round_index=$1
  local tag
  tag=$(printf '%02d' "$round_index")
  local round_dir="$FORMAL_ROOT/round_$tag"
  FINALIZE_COMMAND=(
    "$PYTHON" "$CODE_ROOT/scripts/fcooper_finalize_formal_round_v2.py"
    --request-json "$round_dir/measurement_request.json"
    --artifact-root "$ARTIFACT_ROOT"
    --round-feedback-json "$round_dir/actual_feedback.json"
    --atomic-audit-json "$round_dir/atomic_batch_audit.json"
    --history-output-json "$FORMAL_ROOT/feedback_history_through_round_$tag.json"
    --failure-evidence-json "$round_dir/formal_round_failure.json"
    --round-index "$round_index"
  )
  if (( round_index > 0 )); then
    FINALIZE_COMMAND+=(
      --history-input-json
      "$FORMAL_ROOT/feedback_history_through_round_$(printf '%02d' "$((round_index - 1))").json"
    )
  fi
}

advance_command() {
  local next_round=$1
  local previous
  previous=$(printf '%02d' "$((next_round - 1))")
  ADVANCE_COMMAND=(
    "$PYTHON" "$CODE_ROOT/scripts/stage5_advance_fcooper_round_v2.py"
    --feedback-json "$FORMAL_ROOT/feedback_history_through_round_$previous.json"
    --atomic-audit-json "$FORMAL_ROOT/round_$previous/atomic_batch_audit.json"
    --output-dir "$FORMAL_ROOT"
    --round-index "$next_round"
    --source-registry-json "$SOURCE_REGISTRY_JSON"
    --probe-isolation-audit-json "$PROBE_ISOLATION_AUDIT_JSON"
    --seed "$SEED"
    --coldstart-rows-json "$COLDSTART_ROOT/gold176_final.json"
    --coldstart-graph-features-json "$COLDSTART_ROOT/graph_features.json"
    --profiles-json "$PROFILES_JSON"
  )
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if (( DRY_RUN == 1 )); then
  print_command "${INITIALIZE_COMMAND[@]}"
  for round_index in 0 1 2 3; do
    printf '# round %d: four row commands run concurrently\n' "$round_index"
    for row_index in 0 1 2 3; do
      row_command "$round_index" "$row_index" "${GPUS[$row_index]}"
      print_command "${ROW_COMMAND[@]}"
    done
    finalize_command "$round_index"
    print_command "${FINALIZE_COMMAND[@]}"
    if (( round_index < 3 )); then
      advance_command "$((round_index + 1))"
      print_command "${ADVANCE_COMMAND[@]}"
    fi
  done
  print_command "$PYTHON" "$CODE_ROOT/scripts/fcooper_close_formal_t16_v2.py" \
    --formal-root "$FORMAL_ROOT" \
    --contract-json "$FORMAL_CONTRACT_JSON" \
    --probe-isolation-json "$PROBE_ISOLATION_AUDIT_JSON"
  exit 0
fi

CONTROLLER="$FORMAL_ROOT/controller"
LOGS="$FORMAL_ROOT/logs"
ATTEMPTS="$CONTROLLER/attempts"
STATE="$CONTROLLER/fcooper_formal_t16_supervisor_state.json"
mkdir -p "$CONTROLLER" "$LOGS" "$ATTEMPTS" "$ARTIFACT_ROOT"
SUPERVISOR_STARTED=$(date +%s)
CURRENT_ROUND=-1
STATUS=starting

write_state() {
  local status=$1
  "$PYTHON" - "$STATE" "$status" "$CURRENT_ROUND" "$SUPERVISOR_STARTED" \
    "$GPUS_CSV" "$MAX_RETRIES" "$$" <<'PY'
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
now = time.time()
payload = {
    "schema_version": "fcooper_formal_t16_supervisor_state_v2",
    "task_id": "S5-FCO-TRT-V2",
    "status": sys.argv[2],
    "current_round": int(sys.argv[3]),
    "started_epoch_seconds": int(sys.argv[4]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds": now - int(sys.argv[4]),
    "gpus": [int(value) for value in sys.argv[5].split(",")],
    "max_infrastructure_retries": int(sys.argv[6]),
    "supervisor_pid": int(sys.argv[7]),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    write_state failed || true
  fi
}
trap on_exit EXIT

write_state waiting_for_gate
while [[ ! -f "$GATE_READY" ]]; do
  sleep "$POLL_SECONDS"
done
write_state initializing
"${INITIALIZE_COMMAND[@]}" >"$LOGS/initialize.log" 2>&1

validate_request() {
  local request=$1 round_index=$2
  "$PYTHON" - "$request" "$round_index" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1], encoding="utf-8"))
rows = request.get("rows") or []
identities = [str(row.get("row_id") or row.get("manifest_job_id") or "") for row in rows]
assert request["schema_version"] == "stage5_measurement_request_v2"
assert request["task_id"] == "S5-FCO-TRT-V2"
assert request["round_index"] == int(sys.argv[2])
assert request["batch_size"] == 4 and len(rows) == 4
assert len(set(identities)) == 4 and all(identities)
assert all(row["task_id"] == "S5-FCO-TRT-V2" for row in rows)
PY
}

record_attempt() {
  local path=$1 round_index=$2 row_index=$3 gpu=$4 attempt=$5 pid=$6
  local started=$7 ended=$8 rc=$9
  "$PYTHON" - "$path" "$round_index" "$row_index" "$gpu" "$attempt" "$pid" \
    "$started" "$ended" "$rc" <<'PY'
import json
import sys

payload = {
    "schema_version": "fcooper_formal_row_attempt_v2",
    "round_index": int(sys.argv[2]),
    "row_index": int(sys.argv[3]),
    "gpu": int(sys.argv[4]),
    "attempt": int(sys.argv[5]),
    "pid": int(sys.argv[6]),
    "started_epoch_seconds": float(sys.argv[7]),
    "ended_epoch_seconds": float(sys.argv[8]),
    "elapsed_seconds": float(sys.argv[8]) - float(sys.argv[7]),
    "returncode": int(sys.argv[9]),
}
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

write_exhausted_failure() {
  local round_index=$1 row_index=$2 gpu=$3 attempts_jsonl=$4 output=$5
  "$PYTHON" - "$round_index" "$row_index" "$gpu" "$attempts_jsonl" "$output" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

attempts_path = Path(sys.argv[4])
attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
payload = {
    "schema_version": "fcooper_formal_infrastructure_retry_failure_v2",
    "task_id": "S5-FCO-TRT-V2",
    "status": "failed",
    "failure_class": "exhausted_infrastructure_retry",
    "classified_as_feasibility": False,
    "terminal_status": "infrastructure_failure_unreleased",
    "budget_consumed": 0,
    "round_index": int(sys.argv[1]),
    "row_index": int(sys.argv[2]),
    "gpu": int(sys.argv[3]),
    "attempts": attempts,
}
output = Path(sys.argv[5])
output.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY
}

run_row_with_retries() {
  local round_index=$1 row_index=$2 gpu=$3
  local tag
  tag=$(printf 'round_%02d_row_%02d' "$round_index" "$row_index")
  local attempts_jsonl="$ATTEMPTS/$tag.jsonl"
  : >"$attempts_jsonl"
  local attempt rc pid started ended
  for ((attempt = 0; attempt <= MAX_RETRIES; attempt++)); do
    row_command "$round_index" "$row_index" "$gpu"
    started=$("$PYTHON" -c 'import time; print(time.time())')
    "${ROW_COMMAND[@]}" >"$LOGS/${tag}_attempt_${attempt}.log" 2>&1 &
    pid=$!
    if wait "$pid"; then rc=0; else rc=$?; fi
    ended=$("$PYTHON" -c 'import time; print(time.time())')
    record_attempt "$attempts_jsonl" "$round_index" "$row_index" "$gpu" \
      "$attempt" "$pid" "$started" "$ended" "$rc"
    if (( rc == 0 )); then
      return 0
    fi
  done
  write_exhausted_failure "$round_index" "$row_index" "$gpu" "$attempts_jsonl" \
    "$FORMAL_ROOT/round_$(printf '%02d' "$round_index")/infrastructure_failure_row_$(printf '%02d' "$row_index").json"
  return 1
}

for round_index in 0 1 2 3; do
  CURRENT_ROUND=$round_index
  write_state measuring
  round_tag=$(printf '%02d' "$round_index")
  request="$FORMAL_ROOT/round_$round_tag/measurement_request.json"
  validate_request "$request" "$round_index"

  worker_pids=()
  for row_index in 0 1 2 3; do
    run_row_with_retries "$round_index" "$row_index" "${GPUS[$row_index]}" &
    worker_pids+=("$!")
  done
  round_failed=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then round_failed=1; fi
  done
  if (( round_failed != 0 )); then
    echo "round $round_index exhausted an infrastructure retry; feedback remains unreleased" >&2
    exit 1
  fi

  write_state finalizing
  finalize_command "$round_index"
  "${FINALIZE_COMMAND[@]}" >"$LOGS/finalize_round_$round_tag.log" 2>&1
  if (( round_index < 3 )); then
    write_state advancing
    advance_command "$((round_index + 1))"
    "${ADVANCE_COMMAND[@]}" >"$LOGS/advance_to_round_$(printf '%02d' "$((round_index + 1))").log" 2>&1
  fi
done

write_state closing
"$PYTHON" "$CODE_ROOT/scripts/fcooper_close_formal_t16_v2.py" \
  --formal-root "$FORMAL_ROOT" \
  --contract-json "$FORMAL_CONTRACT_JSON" \
  --probe-isolation-json "$PROBE_ISOLATION_AUDIT_JSON" \
  >"$LOGS/close_formal_t16.log" 2>&1
write_state complete
trap - EXIT
