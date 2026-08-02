#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_SCHEMA="stage7_task_round_controller_config_v1"
readonly STATE_SCHEMA="stage7_task_round_controller_state_v1"
readonly INFRA_RETRY_EXIT=75
readonly INTERRUPT_EXIT=99

OUTPUT_ROOT=""
VARIANT=""
SEED=""
ROUND_INDEX=""
REQUEST_SHA256=""

usage() {
  cat >&2 <<'EOF'
usage: stage7_task_round_controller_v1.sh
  --output-root <absolute formal root>
  --variant <frozen variant>
  --seed <frozen seed>
  --round-index <0..3>
  --request-sha256 <64hex>

STAGE7_CONTROLLER_CONFIG_JSON must name an absolute frozen JSON config.
EOF
}

die() {
  echo "stage7 controller: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --output-root)
      (($# >= 2)) || die "missing value for --output-root"
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --variant)
      (($# >= 2)) || die "missing value for --variant"
      VARIANT=$2
      shift 2
      ;;
    --seed)
      (($# >= 2)) || die "missing value for --seed"
      SEED=$2
      shift 2
      ;;
    --round-index)
      (($# >= 2)) || die "missing value for --round-index"
      ROUND_INDEX=$2
      shift 2
      ;;
    --request-sha256)
      (($# >= 2)) || die "missing value for --request-sha256"
      REQUEST_SHA256=$2
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ $OUTPUT_ROOT == /* ]] || die "--output-root must be absolute"
case "$VARIANT" in
  full|without_surrogate|without_measured_feedback|backend_blind|without_capability_scan) ;;
  *) die "unknown frozen Stage7 variant" ;;
esac
case "$SEED" in
  20260718|20260719|20260720) ;;
  *) die "unknown frozen Stage7 seed" ;;
esac
[[ $ROUND_INDEX =~ ^[0-3]$ ]] || die "--round-index must be 0..3"
[[ $REQUEST_SHA256 =~ ^[0-9a-f]{64}$ ]] || die "--request-sha256 must be 64 lowercase hex characters"

CONFIG_JSON=${STAGE7_CONTROLLER_CONFIG_JSON:-}
[[ $CONFIG_JSON == /* ]] || die "STAGE7_CONTROLLER_CONFIG_JSON must be absolute"
[[ -f $CONFIG_JSON ]] || die "controller config does not exist: $CONFIG_JSON"

BOOTSTRAP_PYTHON=${STAGE7_CONFIG_PYTHON:-/usr/bin/python3}
[[ $BOOTSTRAP_PYTHON == /* && -x $BOOTSTRAP_PYTHON ]] ||
  die "STAGE7_CONFIG_PYTHON must be an absolute executable"
PYTHON_BIN=$(
  "$BOOTSTRAP_PYTHON" - "$CONFIG_JSON" "$CONFIG_SCHEMA" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or payload.get("schema_version") != sys.argv[2]:
    raise SystemExit("invalid Stage7 controller config schema")
value = payload.get("python_executable")
executable = Path(value) if isinstance(value, str) else None
if executable is None:
    raise SystemExit("python_executable must be configured")
if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
    raise SystemExit("python_executable must be an absolute executable file")
print(executable)
PY
) || die "unable to validate controller config"
[[ $PYTHON_BIN == /* ]] || die "validated executable path is not absolute"

validate_config() {
  "$PYTHON_BIN" - "$CONFIG_JSON" "$CONFIG_SCHEMA" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or payload.get("schema_version") != sys.argv[2]:
    raise SystemExit("invalid Stage7 controller config schema")
python_value = payload.get("python_executable")
python_executable = Path(python_value) if isinstance(python_value, str) else None
if (
    python_executable is None
    or not python_executable.is_absolute()
    or not python_executable.is_file()
    or not os.access(python_executable, os.X_OK)
):
    raise SystemExit("python_executable must be an absolute executable file")

required_paths = (
    "trajectory_contract_json",
    "key_dimensions_json",
    "cache_json",
    "full_plan_json",
)
for key in required_paths:
    value = payload.get(key)
    target = Path(value) if isinstance(value, str) else None
    if target is None or not target.is_absolute():
        raise SystemExit(f"{key} must be an absolute path")
    if key not in {"trajectory_contract_json", "full_plan_json"} and not target.is_file():
        raise SystemExit(f"{key} input is missing: {target}")

required_argv = (
    "unified_cli_argv",
    "build_full_plan_command",
    "materialize_sources_argv",
    "execute_performance_argv",
    "execute_ap_argv",
)
for key in (*required_argv, "release_lease_argv", "pre_stage_validator_argv"):
    value = payload.get(key)
    if key in {"release_lease_argv", "pre_stage_validator_argv"} and value is None:
        continue
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item and "\0" not in item and "\n" not in item
        for item in value
    ):
        raise SystemExit(f"{key} must be a non-empty string array")
    executable = Path(value[0])
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise SystemExit(f"{key} executable must be an absolute executable file")
PY
}
validate_config || die "controller config validation failed"

config_path() {
  local key=$1
  "$PYTHON_BIN" - "$CONFIG_JSON" "$key" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2])
if not isinstance(value, str) or not Path(value).is_absolute():
    raise SystemExit(f"invalid absolute config path: {sys.argv[2]}")
print(value)
PY
}

load_argv() {
  local key=$1
  local destination=$2
  local -a loaded=()
  mapfile -d '' -t loaded < <(
    "$PYTHON_BIN" - "$CONFIG_JSON" "$key" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get(sys.argv[2])
if value is None and sys.argv[2] in {"release_lease_argv", "pre_stage_validator_argv"}:
    raise SystemExit(0)
if not isinstance(value, list):
    raise SystemExit(f"invalid argv config: {sys.argv[2]}")
for item in value:
    sys.stdout.write(item)
    sys.stdout.write("\0")
PY
  )
  declare -n target=$destination
  target=("${loaded[@]}")
}

TRAJECTORY_JSON=$(config_path trajectory_contract_json)
KEY_DIMENSIONS_JSON=$(config_path key_dimensions_json)
CACHE_JSON=$(config_path cache_json)
FULL_PLAN_JSON=$(config_path full_plan_json)

declare -a UNIFIED_CLI FULL_PLAN_CMD MATERIALIZE_CMD PERFORMANCE_CMD AP_CMD RELEASE_CMD
load_argv unified_cli_argv UNIFIED_CLI
load_argv build_full_plan_command FULL_PLAN_CMD
load_argv materialize_sources_argv MATERIALIZE_CMD
load_argv execute_performance_argv PERFORMANCE_CMD
load_argv execute_ap_argv AP_CMD
load_argv release_lease_argv RELEASE_CMD
declare -a PRE_STAGE_VALIDATOR_CMD
load_argv pre_stage_validator_argv PRE_STAGE_VALIDATOR_CMD

load_pre_stage_validator_from_environment() {
  local raw=${STAGE7_PRE_STAGE_VALIDATOR_ARGV_JSON:-}
  [[ -n $raw ]] || return 0
  local -a loaded=()
  local validated
  validated=$("$PYTHON_BIN" - "$raw" <<'PY'
import json
import os
import sys
from pathlib import Path

value = json.loads(sys.argv[1])
if not isinstance(value, list) or not value or not all(
    isinstance(item, str) and item and "\0" not in item and "\n" not in item
    for item in value
):
    raise SystemExit("pre-stage validator must be a non-empty string array")
executable = Path(value[0])
if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
    raise SystemExit("pre-stage validator executable must be absolute and executable")
for item in value:
    sys.stdout.write(item)
    sys.stdout.write("\n")
PY
  ) || die "invalid STAGE7_PRE_STAGE_VALIDATOR_ARGV_JSON"
  mapfile -t loaded <<<"$validated"
  PRE_STAGE_VALIDATOR_CMD=("${loaded[@]}")
}
load_pre_stage_validator_from_environment

RELEASE_CALLED=0
release_owned_lease() {
  local original_rc=$?
  if ((RELEASE_CALLED == 0)) && ((${#RELEASE_CMD[@]})); then
    RELEASE_CALLED=1
    set +e
    "${RELEASE_CMD[@]}"
    set -e
  fi
  return "$original_rc"
}
trap release_owned_lease EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

readonly TRAJECTORY_DIR="$OUTPUT_ROOT/variants/$VARIANT/seed_$SEED"
readonly ROUND_DIR="$TRAJECTORY_DIR/round_$(printf '%02d' "$ROUND_INDEX")"
readonly REQUEST_JSON="$ROUND_DIR/measurement_request.json"
readonly BINDING_JSON="$ROUND_DIR/stage7_request_binding.json"
readonly CACHE_REVEAL_JSON="$ROUND_DIR/cache_reveal.json"
readonly PRE_CACHE_AUDIT_JSON="$ROUND_DIR/pre_cache_request_audit.json"
readonly MATERIALIZE_JSON="$ROUND_DIR/materialize_result.json"
readonly MISS_PLAN_JSON="$ROUND_DIR/miss_plan.json"
readonly PERFORMANCE_RESULTS_JSON="$ROUND_DIR/miss_performance_results.json"
readonly MISS_RESULTS_JSON="$ROUND_DIR/miss_results.json"
readonly FINALIZATION_JSON="$ROUND_DIR/round_finalization.json"
readonly FINAL_FEEDBACK_JSON="$ROUND_DIR/final_feedback.json"
readonly PROMOTION_WORK_DIR="$ROUND_DIR/promotion"
readonly ROUND_AUDIT_JSON="$ROUND_DIR/round_audit.json"
readonly STATE_JSON="$OUTPUT_ROOT/status/round_$(printf '%02d' "$ROUND_INDEX")_stage.json"

[[ $TRAJECTORY_JSON == "$TRAJECTORY_DIR/trajectory_contract.json" ]] ||
  die "trajectory contract path must be canonical for the selected trajectory"
[[ -d $ROUND_DIR ]] || die "canonical round directory does not exist: $ROUND_DIR"
EXPECTED_CWD=$(
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$ROUND_DIR" <<'PY'
import os
import sys
if os.path.realpath(sys.argv[1]) != sys.argv[1]:
    raise SystemExit("output root must be canonical")
print(os.path.realpath(sys.argv[2]))
PY
) || die "output root is not canonical"
ACTUAL_CWD=$(pwd -P)
[[ $ACTUAL_CWD == "$EXPECTED_CWD" ]] || die "cwd must equal canonical round directory"

IFS=',' read -r -a GPU_UUID_ARRAY <<<"${CUDA_VISIBLE_DEVICES:-}"
((${#GPU_UUID_ARRAY[@]} == 4)) || die "CUDA_VISIBLE_DEVICES must contain exactly four leased UUIDs"
"$PYTHON_BIN" - "${GPU_UUID_ARRAY[@]}" <<'PY' || die "invalid leased GPU UUID set"
import re
import sys

values = sys.argv[1:]
if len(values) != 4 or len(set(values)) != 4:
    raise SystemExit("four unique GPU UUIDs are required")
if any(re.fullmatch(r"GPU-[A-Za-z0-9-]+", value) is None for value in values):
    raise SystemExit("leased devices must be GPU UUIDs")
PY

state_start() {
  "$PYTHON_BIN" - \
    "$STATE_JSON" "$STATE_SCHEMA" "$ROUND_INDEX" "$VARIANT" "$SEED" \
    "$REQUEST_SHA256" "$CONFIG_JSON" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
schema, round_index, variant, seed, request_sha, config_path = sys.argv[2:]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

identity = {
    "schema_version": schema,
    "round_index": int(round_index),
    "variant": variant,
    "seed": int(seed),
    "request_sha256": request_sha,
}
if state_path.is_file():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if any(state.get(key) != value for key, value in identity.items()):
        raise SystemExit("existing controller state identity drift")
    stages = state.get("completed_stages")
    if not isinstance(stages, list):
        raise SystemExit("existing controller stage history is invalid")
    expected_stages = (
        "validate_controller",
        "select_request",
        "validate_request",
        "reveal_cache",
        "materialize_sources",
        "build_miss_plan",
        "execute_performance",
        "execute_ap",
        "finalize_round",
        "audit_round",
    )
    recorded_stages = [entry.get("stage") for entry in stages if isinstance(entry, dict)]
    if recorded_stages != list(expected_stages[: len(recorded_stages)]):
        raise SystemExit("existing controller stage history is not a valid prefix")
    for entry in stages:
        for record in [*entry.get("inputs", []), *entry.get("outputs", [])]:
            path = Path(str(record.get("path") or ""))
            if not path.is_file() or digest(path) != record.get("sha256"):
                raise SystemExit(
                    f"completed stage artifact SHA drift: {entry.get('stage')}:{path}"
                )
    state = {**state, "attempt": int(state.get("attempt", 0)) + 1}
    if state.get("status") != "complete":
        state = {**state, "status": "running", "failure": None}
else:
    state = {
        **identity,
        "attempt": 1,
        "status": "running",
        "stage": None,
        "completed_stages": [],
        "budget_consumed": 0,
        "gpu_runner_calls": 0,
        "failure": None,
    }
write_atomic(state_path, state)
PY
}
state_start || die "controller state validation failed"

stage_is_done() {
  local stage=$1
  "$PYTHON_BIN" - "$STATE_JSON" "$stage" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
completed = [entry.get("stage") for entry in state.get("completed_stages", [])]
raise SystemExit(0 if sys.argv[2] in completed else 1)
PY
}

run_pre_stage_validator() {
  local stage=$1
  ((${#PRE_STAGE_VALIDATOR_CMD[@]})) || return 0
  set +e
  "${PRE_STAGE_VALIDATOR_CMD[@]}" "$stage" "$REQUEST_SHA256"
  local rc=$?
  set -e
  if ((rc != 0)); then
    mark_failure terminal_invalid "$stage" "$rc"
    die "pre-stage validator rejected $stage"
  fi
}

begin_pending_stage() {
  local stage=$1
  if stage_is_done "$stage"; then
    return 1
  fi
  run_pre_stage_validator "$stage"
  return 0
}

mark_stage() {
  local stage=$1
  local stage_status=$2
  local gpu_increment=$3
  shift 3
  "$PYTHON_BIN" - "$STATE_JSON" "$stage" "$stage_status" "$gpu_increment" "$@" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
stage, stage_status, gpu_increment = sys.argv[2:5]
paths = sys.argv[5:]
if "--" not in paths:
    raise SystemExit("stage path separator is missing")
separator = paths.index("--")
input_paths = paths[:separator]
output_paths = paths[separator + 1 :]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def records(values):
    result = []
    for value in values:
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise SystemExit(f"verified stage artifact is missing: {path}")
        result.append({"path": str(path), "sha256": digest(path)})
    return result

def write_atomic(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

state = json.loads(state_path.read_text(encoding="utf-8"))
completed = list(state.get("completed_stages", []))
if any(entry.get("stage") == stage for entry in completed):
    raise SystemExit(f"stage was already recorded: {stage}")
expected_stages = (
    "validate_controller",
    "select_request",
    "validate_request",
    "reveal_cache",
    "materialize_sources",
    "build_miss_plan",
    "execute_performance",
    "execute_ap",
    "finalize_round",
    "audit_round",
)
if len(completed) >= len(expected_stages) or expected_stages[len(completed)] != stage:
    raise SystemExit(f"stage order violation: {stage}")
entry = {
    "stage": stage,
    "stage_status": stage_status,
    "attempt": state["attempt"],
    "inputs": records(input_paths),
    "outputs": records(output_paths),
}
completed.append(entry)
status = "complete" if stage == "audit_round" else "running"
state = {
    **state,
    "stage": stage,
    "status": status,
    "completed_stages": completed,
    "gpu_runner_calls": int(state.get("gpu_runner_calls", 0))
    + int(gpu_increment),
    "budget_consumed": 4 if stage == "audit_round" else int(
        state.get("budget_consumed", 0)
    ),
    "failure": None,
}
write_atomic(state_path, state)
PY
}

mark_failure() {
  local status=$1
  local stage=$2
  local exit_code=$3
  "$PYTHON_BIN" - "$STATE_JSON" "$status" "$stage" "$exit_code" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
state = json.loads(path.read_text(encoding="utf-8"))
state = {
    **state,
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "budget_consumed": 0,
    "failure": {
        "stage": sys.argv[3],
        "exit_code": int(sys.argv[4]),
        "request_sha256": state["request_sha256"],
    },
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(state, handle, ensure_ascii=True, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

interrupt_if_requested() {
  local stage=$1
  local requested=${STAGE7_INTERRUPT_AFTER_STAGE:-}
  if [[ -n $requested && $requested == "$stage" ]]; then
    exit "$INTERRUPT_EXIT"
  fi
}

complete_stage() {
  local stage=$1
  local stage_status=$2
  local gpu_increment=$3
  shift 3
  mark_stage "$stage" "$stage_status" "$gpu_increment" "$@"
  interrupt_if_requested "$stage"
}

run_external_stage() {
  local stage=$1
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if ((rc == INFRA_RETRY_EXIT)); then
    mark_failure "infrastructure_retry_required" "$stage" "$rc"
    exit "$INFRA_RETRY_EXIT"
  fi
  if ((rc != 0)); then
    mark_failure "terminal_invalid" "$stage" "$rc"
    return "$rc"
  fi
}

run_control_stage() {
  local stage=$1
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if ((rc != 0)); then
    mark_failure "terminal_invalid" "$stage" "$rc"
    return "$rc"
  fi
}

validate_request_chain() {
  "$PYTHON_BIN" - \
    "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON" \
    "$TRAJECTORY_DIR" "$VARIANT" "$SEED" "$ROUND_INDEX" "$REQUEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    trajectory_path,
    request_path,
    binding_path,
    trajectory_dir,
    variant,
    seed,
    round_index,
    expected_request_sha,
) = sys.argv[1:]

def load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload

def sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

def embedded(payload, field, label):
    copied = {key: value for key, value in payload.items() if key != field}
    if payload.get(field) != sha(copied):
        raise SystemExit(f"{label} SHA drift")
    return payload[field]

trajectory = load(trajectory_path)
request = load(request_path)
binding = load(binding_path)
trajectory_sha = embedded(
    trajectory, "trajectory_contract_sha256", "trajectory contract"
)
request_sha = embedded(request, "measurement_request_sha256", "measurement request")
binding_sha = embedded(binding, "request_binding_sha256", "request binding")
if (
    trajectory.get("task_id") != "S7-PYR-TVM"
    or trajectory.get("variant") != variant
    or int(trajectory.get("seed", -1)) != int(seed)
    or trajectory.get("trajectory_dir") != trajectory_dir
):
    raise SystemExit("trajectory identity drift")
if (
    request_sha != expected_request_sha
    or request.get("task_id") != "S7-PYR-TVM"
    or int(request.get("round_index", -1)) != int(round_index)
    or request.get("batch_size") != 4
):
    raise SystemExit("request flag or round identity drift")
rows = request.get("rows")
if not isinstance(rows, list) or len(rows) != 4:
    raise SystemExit("request must contain exactly four selected rows")
row_ids = [
    str(row.get("manifest_job_id") or row.get("row_id") or "")
    if isinstance(row, dict)
    else ""
    for row in rows
]
if any(not value for value in row_ids) or len(set(row_ids)) != 4:
    raise SystemExit("request must contain four unique selected row IDs")
if any(row.get("task_id") != "S7-PYR-TVM" for row in rows):
    raise SystemExit("selected row task identity drift")
if (
    binding.get("measurement_request_sha256") != request_sha
    or binding.get("trajectory_contract_sha256") != trajectory_sha
    or int(binding.get("round_index", -1)) != int(round_index)
    or binding.get("selected_row_ids") != row_ids
    or len(binding.get("selected_row_ids") or []) != 4
    or len(set(binding.get("selected_row_ids") or [])) != 4
    or not binding_sha
):
    raise SystemExit("request binding identity drift")
PY
}

validate_reveal() {
  "$PYTHON_BIN" - "$CACHE_REVEAL_JSON" "$REQUEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hits = payload.get("exact_hit_count")
misses = payload.get("miss_count")
rows = payload.get("rows")
if (
    payload.get("measurement_request_sha256") != sys.argv[2]
    or not isinstance(hits, int)
    or isinstance(hits, bool)
    or not isinstance(misses, int)
    or isinstance(misses, bool)
    or hits < 0
    or misses < 0
    or hits + misses != 4
    or not isinstance(rows, list)
    or len(rows) != 4
):
    raise SystemExit("cache reveal identity/count drift")
print(misses)
PY
}

validate_miss_plan() {
  local expected_misses=$1
  "$PYTHON_BIN" - "$MISS_PLAN_JSON" "$REQUEST_SHA256" "$expected_misses" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
miss_ids = payload.get("miss_row_ids")
manifest = payload.get("manifest")
jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
performance = payload.get("performance_jobs")

def row_ids(rows):
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("manifest_job_id") or row.get("row_id") or "")
        if isinstance(row, dict)
        else ""
        for row in rows
    ]

if (
    payload.get("original_measurement_request_sha256") != sys.argv[2]
    or not isinstance(miss_ids, list)
    or len(miss_ids) != int(sys.argv[3])
    or len(set(miss_ids)) != len(miss_ids)
    or not isinstance(manifest, dict)
    or manifest.get("row_count") != int(sys.argv[3])
    or manifest.get("genome_count") != int(sys.argv[3])
    or row_ids(jobs) != miss_ids
    or row_ids(performance) != miss_ids
):
    raise SystemExit("miss-only plan identity/count drift")
PY
}

validate_full_plan() {
  "$PYTHON_BIN" - "$FULL_PLAN_JSON" "$REQUEST_JSON" "$REQUEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
request = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
manifest = plan.get("manifest") if isinstance(plan, dict) else None
jobs = manifest.get("jobs") if isinstance(manifest, dict) else None
performance = plan.get("performance_jobs") if isinstance(plan, dict) else None
request_rows = request.get("rows") if isinstance(request, dict) else None

def row_ids(rows):
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("manifest_job_id") or row.get("row_id") or "")
        if isinstance(row, dict)
        else ""
        for row in rows
    ]

expected_ids = row_ids(request_rows)
if (
    request.get("measurement_request_sha256") != sys.argv[3]
    or not isinstance(manifest, dict)
    or manifest.get("source_request_sha256") != sys.argv[3]
    or manifest.get("row_count") != 4
    or manifest.get("genome_count") != 4
    or len(expected_ids) != 4
    or len(set(expected_ids)) != 4
    or row_ids(jobs) != expected_ids
    or row_ids(performance) != expected_ids
):
    raise SystemExit("full performance plan request identity drift")
PY
}

write_empty_miss_results() {
  "$PYTHON_BIN" - "$MISS_RESULTS_JSON" "$REQUEST_SHA256" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "stage7_empty_miss_results_v1",
    "measurement_request_sha256": sys.argv[2],
    "rows": [],
}
content = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
if path.is_file():
    if path.read_text(encoding="utf-8") != content:
        raise SystemExit("empty miss results drift")
    raise SystemExit(0)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(content, encoding="utf-8")
os.replace(temporary, path)
PY
}

validate_finalization() {
  "$PYTHON_BIN" - "$FINALIZATION_JSON" "$FINAL_FEEDBACK_JSON" "$REQUEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
feedback = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rows = feedback.get("rows")
if (
    result.get("feedback_released") is not True
    or result.get("measurement_request_sha256") != sys.argv[3]
    or not isinstance(rows, list)
    or len(rows) != 4
):
    raise SystemExit("finalized feedback is not an atomic four-row release")
PY
}

if begin_pending_stage validate_controller; then
  complete_stage validate_controller verified 0 "$CONFIG_JSON" -- "$CONFIG_JSON"
fi

if begin_pending_stage select_request; then
  [[ -e $REQUEST_JSON && ! -f $REQUEST_JSON ]] && die "measurement request is not a regular file"
  [[ -e $BINDING_JSON && ! -f $BINDING_JSON ]] && die "request binding is not a regular file"
  if [[ ! -f $REQUEST_JSON ]]; then
    [[ ! -e $BINDING_JSON ]] || die "binding exists without request; refusing reselection"
    if ((ROUND_INDEX == 0)); then
      run_control_stage select_request "${UNIFIED_CLI[@]}" init-round \
        --output-root "$OUTPUT_ROOT" \
        --variant "$VARIANT" \
        --seed "$SEED" \
        --round-index "$ROUND_INDEX"
      complete_stage select_request selected 0 \
        "$CONFIG_JSON" -- "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON"
    else
      previous_index=$((ROUND_INDEX - 1))
      previous_round="$TRAJECTORY_DIR/round_$(printf '%02d' "$previous_index")"
      previous_feedback="$previous_round/final_feedback.json"
      previous_state="$OUTPUT_ROOT/status/round_$(printf '%02d' "$previous_index")_stage.json"
      "$PYTHON_BIN" - "$previous_state" "$previous_feedback" "$STATE_SCHEMA" <<'PY' ||
import json
import sys
from pathlib import Path

state_path, feedback_path, schema = map(Path, sys.argv[1:])
if not state_path.is_file() or not feedback_path.is_file():
    raise SystemExit("previous round barrier evidence is missing")
state = json.loads(state_path.read_text(encoding="utf-8"))
if (
    state.get("schema_version") != str(schema)
    or state.get("status") != "complete"
    or state.get("stage") != "audit_round"
    or int(state.get("budget_consumed", 0)) != 4
):
    raise SystemExit("previous round barrier has not completed")
PY
      die "previous round barrier is not complete"
      run_control_stage select_request "${UNIFIED_CLI[@]}" advance-round \
        --output-root "$OUTPUT_ROOT" \
        --variant "$VARIANT" \
        --seed "$SEED" \
        --completed-round-index "$previous_index" \
        --feedback-json "$previous_feedback"
      complete_stage select_request selected 0 \
        "$CONFIG_JSON" "$previous_state" "$previous_feedback" -- \
        "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON"
    fi
  else
    [[ -f $BINDING_JSON && -f $TRAJECTORY_JSON ]] ||
      die "existing request is missing binding or trajectory contract"
    complete_stage select_request preexisting_request 0 \
      "$CONFIG_JSON" -- "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON"
  fi
fi

if begin_pending_stage validate_request; then
  validate_request_chain || {
    mark_failure terminal_invalid validate_request 2
    die "trajectory/request/binding validation failed"
  }
  run_control_stage validate_request "${UNIFIED_CLI[@]}" audit-trajectory \
    --output-root "$OUTPUT_ROOT" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --trajectory-contract-json "$TRAJECTORY_JSON" \
    --expected-request-sha256 "$REQUEST_SHA256" \
    --output-json "$PRE_CACHE_AUDIT_JSON"
  complete_stage validate_request verified 0 \
    "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON" -- "$PRE_CACHE_AUDIT_JSON"
fi

if begin_pending_stage reveal_cache; then
  run_control_stage reveal_cache "${UNIFIED_CLI[@]}" reveal-cache \
    --output-root "$OUTPUT_ROOT" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --trajectory-contract-json "$TRAJECTORY_JSON" \
    --key-dimensions-json "$KEY_DIMENSIONS_JSON" \
    --cache-json "$CACHE_JSON" \
    --output-json "$CACHE_REVEAL_JSON"
  MISS_COUNT=$(validate_reveal)
  complete_stage reveal_cache verified 0 \
    "$REQUEST_JSON" "$BINDING_JSON" "$TRAJECTORY_JSON" \
    "$KEY_DIMENSIONS_JSON" "$CACHE_JSON" -- "$CACHE_REVEAL_JSON"
else
  MISS_COUNT=$(validate_reveal)
fi

if begin_pending_stage materialize_sources; then
  if ((MISS_COUNT == 0)); then
    complete_stage materialize_sources skipped_all_hit 0 \
      "$CACHE_REVEAL_JSON" --
  else
    run_external_stage materialize_sources \
      "${MATERIALIZE_CMD[@]}" \
      --request-json "$REQUEST_JSON" \
      --request-binding-json "$BINDING_JSON" \
      --cache-reveal-json "$CACHE_REVEAL_JSON" \
      --output-json "$MATERIALIZE_JSON"
    complete_stage materialize_sources verified 1 \
      "$REQUEST_JSON" "$BINDING_JSON" "$CACHE_REVEAL_JSON" -- "$MATERIALIZE_JSON"
  fi
fi

if begin_pending_stage build_miss_plan; then
  run_external_stage build_miss_plan \
    "${FULL_PLAN_CMD[@]}" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --output-json "$FULL_PLAN_JSON"
  validate_full_plan || {
    mark_failure terminal_invalid build_miss_plan 2
    die "full performance plan request identity drift"
  }
  run_control_stage build_miss_plan "${UNIFIED_CLI[@]}" build-miss-plan \
    --output-root "$OUTPUT_ROOT" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --full-plan-json "$FULL_PLAN_JSON" \
    --cache-reveal-json "$CACHE_REVEAL_JSON" \
    --output-json "$MISS_PLAN_JSON"
  validate_miss_plan "$MISS_COUNT" || {
    mark_failure terminal_invalid build_miss_plan 2
    die "miss-only plan identity/count drift"
  }
  complete_stage build_miss_plan verified 0 \
    "$REQUEST_JSON" "$BINDING_JSON" "$CACHE_REVEAL_JSON" -- \
    "$FULL_PLAN_JSON" "$MISS_PLAN_JSON"
else
  validate_full_plan
  validate_miss_plan "$MISS_COUNT"
fi

if begin_pending_stage execute_performance; then
  if ((MISS_COUNT == 0)); then
    complete_stage execute_performance skipped_all_hit 0 \
      "$MISS_PLAN_JSON" --
  else
    run_external_stage execute_performance \
      "${PERFORMANCE_CMD[@]}" \
      --miss-plan-json "$MISS_PLAN_JSON" \
      --output-json "$PERFORMANCE_RESULTS_JSON"
    complete_stage execute_performance verified 1 \
      "$MISS_PLAN_JSON" -- "$PERFORMANCE_RESULTS_JSON"
  fi
fi

if begin_pending_stage execute_ap; then
  if ((MISS_COUNT == 0)); then
    write_empty_miss_results
    complete_stage execute_ap skipped_all_hit 0 \
      "$MISS_PLAN_JSON" -- "$MISS_RESULTS_JSON"
  else
    [[ -f $PERFORMANCE_RESULTS_JSON ]] ||
      die "performance results are missing before AP"
    run_external_stage execute_ap \
      "${AP_CMD[@]}" \
      --miss-plan-json "$MISS_PLAN_JSON" \
      --performance-results-json "$PERFORMANCE_RESULTS_JSON" \
      --output-json "$MISS_RESULTS_JSON"
    complete_stage execute_ap verified 1 \
      "$MISS_PLAN_JSON" "$PERFORMANCE_RESULTS_JSON" -- "$MISS_RESULTS_JSON"
  fi
fi

if begin_pending_stage finalize_round; then
  run_control_stage finalize_round "${UNIFIED_CLI[@]}" finalize-round \
    --output-root "$OUTPUT_ROOT" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --cache-reveal-json "$CACHE_REVEAL_JSON" \
    --miss-results-json "$MISS_RESULTS_JSON" \
    --promotion-work-dir "$PROMOTION_WORK_DIR" \
    --released-feedback-json "$FINAL_FEEDBACK_JSON" \
    --output-json "$FINALIZATION_JSON"
  validate_finalization
  complete_stage finalize_round verified 0 \
    "$REQUEST_JSON" "$BINDING_JSON" "$CACHE_REVEAL_JSON" "$MISS_RESULTS_JSON" -- \
    "$FINALIZATION_JSON" "$FINAL_FEEDBACK_JSON"
fi

if begin_pending_stage audit_round; then
  run_control_stage audit_round "${UNIFIED_CLI[@]}" audit-trajectory \
    --output-root "$OUTPUT_ROOT" \
    --request-json "$REQUEST_JSON" \
    --request-binding-json "$BINDING_JSON" \
    --trajectory-contract-json "$TRAJECTORY_JSON" \
    --expected-request-sha256 "$REQUEST_SHA256" \
    --output-json "$ROUND_AUDIT_JSON"
  complete_stage audit_round verified 0 \
    "$TRAJECTORY_JSON" "$REQUEST_JSON" "$BINDING_JSON" \
    "$FINALIZATION_JSON" "$FINAL_FEEDBACK_JSON" -- "$ROUND_AUDIT_JSON"
fi

exit 0
