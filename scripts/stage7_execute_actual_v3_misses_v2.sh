#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
# Stage7 owns only admission and argv-safe orchestration.  The existing
# Stage5/Stage3 programs remain the sole owners of source creation,
# quantization, hardware measurement, AP and actual-feedback semantics.
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEFAULT_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly REPO_ROOT="${STAGE7_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
readonly BUNDLE_CODE_ROOT="${STAGE7_BUNDLE_CODE_ROOT:-$REPO_ROOT}"
readonly FROZEN_REPO_ROOT="${STAGE7_FROZEN_REPO_ROOT:-$REPO_ROOT}"
readonly PYTHON="${STAGE7_CONFIG_PYTHON:-/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python}"
PHYSICAL_PLAN_JSON= FORMAL_V2_ROOT= LOGICAL_REQUEST_JSON=
SELECTION_BINDING_JSON= CACHE_REVEAL_JSON= CACHE_SNAPSHOT_JSON=
NO_GPU_RECEIPT_JSON= OUTPUT_DIR= ADMISSION_JSON=
EXECUTOR_ADMISSION_JSON= SOURCE_PLAN_JSON= SOURCE_RESULT_JSON=
SOURCE_PLAN_SHA256=
SOURCE_RESULT_SHA256=
RESOLVED_SOURCE_LOCK_SHA256=
CONTRACT_SHA256=
REQUEST_SHA256=
GPU_UUIDS=
SCHEDULER_STATE_JSON=
CONTROLLER_ID=
EXPECTED_RELEASE_SHA256=
EXPECTED_MANIFEST_FILE_SHA256=
DRY_RUN=0
usage() {
  printf '%s\n' \
    "usage: $0 --physical-plan-json PATH --output-dir PATH" \
    "  --formal-v2-root PATH --logical-request-json PATH" \
    "  --expected-release-sha256 SHA256" \
    "  --expected-manifest-file-sha256 SHA256" \
    "  --selection-binding-json PATH --cache-reveal-json PATH" \
    "  --cache-snapshot-json PATH" \
    "  --no-gpu-receipt-json PATH" \
    "  --admission-json PATH --request-sha256 SHA256" \
    "  --executor-admission-json PATH --contract-sha256 SHA256" \
    "  --source-plan-json PATH --source-result-json PATH" \
    "  --source-plan-sha256 SHA256 --source-result-sha256 SHA256" \
    "  --resolved-source-lock-sha256 SHA256" \
    "  --gpu-uuids UUID,UUID,UUID,UUID" \
    "  [--scheduler-state-json PATH --controller-id ID] [--dry-run]" >&2
}
while (($#)); do
  case "$1" in
    --physical-plan-json)
      PHYSICAL_PLAN_JSON=$2
      shift 2
      ;;
    --formal-v2-root)
      FORMAL_V2_ROOT=$2
      shift 2
      ;;
    --expected-release-sha256)
      EXPECTED_RELEASE_SHA256=$2
      shift 2
      ;;
    --expected-manifest-file-sha256)
      EXPECTED_MANIFEST_FILE_SHA256=$2
      shift 2
      ;;
    --logical-request-json)
      LOGICAL_REQUEST_JSON=$2
      shift 2
      ;;
    --selection-binding-json)
      SELECTION_BINDING_JSON=$2
      shift 2
      ;;
    --cache-reveal-json)
      CACHE_REVEAL_JSON=$2
      shift 2
      ;;
    --cache-snapshot-json)
      CACHE_SNAPSHOT_JSON=$2
      shift 2
      ;;
    --no-gpu-receipt-json)
      NO_GPU_RECEIPT_JSON=$2
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR=$2
      shift 2
      ;;
    --admission-json)
      ADMISSION_JSON=$2
      shift 2
      ;;
    --executor-admission-json)
      EXECUTOR_ADMISSION_JSON=$2
      shift 2
      ;;
    --source-plan-json)
      SOURCE_PLAN_JSON=$2
      shift 2
      ;;
    --source-result-json)
      SOURCE_RESULT_JSON=$2
      shift 2
      ;;
    --source-plan-sha256)
      SOURCE_PLAN_SHA256=$2
      shift 2
      ;;
    --source-result-sha256)
      SOURCE_RESULT_SHA256=$2
      shift 2
      ;;
    --resolved-source-lock-sha256)
      RESOLVED_SOURCE_LOCK_SHA256=$2
      shift 2
      ;;
    --contract-sha256)
      CONTRACT_SHA256=$2
      shift 2
      ;;
    --request-sha256)
      REQUEST_SHA256=$2
      shift 2
      ;;
    --gpu-uuids)
      GPU_UUIDS=$2
      shift 2
      ;;
    --scheduler-state-json)
      SCHEDULER_STATE_JSON=$2
      shift 2
      ;;
    --controller-id)
      CONTROLLER_ID=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage
      exit 2
      ;;
  esac
done
[[ -f "$PHYSICAL_PLAN_JSON" ]] || {
  echo "--physical-plan-json must name a regular file" >&2
  exit 2
}
[[ -d "$FORMAL_V2_ROOT" ]] || {
  echo "--formal-v2-root must name the initialized v2 root" >&2
  exit 2
}
for required_json in \
  "$LOGICAL_REQUEST_JSON" "$SELECTION_BINDING_JSON" \
  "$CACHE_REVEAL_JSON" "$CACHE_SNAPSHOT_JSON" \
  "$SOURCE_PLAN_JSON" "$SOURCE_RESULT_JSON" "$NO_GPU_RECEIPT_JSON"; do
  [[ -f "$required_json" ]] || {
    echo "Task4 authenticated request artifacts are incomplete" >&2
    exit 2
  }
done
for required_sha in \
  "$SOURCE_PLAN_SHA256" "$SOURCE_RESULT_SHA256" \
  "$RESOLVED_SOURCE_LOCK_SHA256"; do
  [[ "$required_sha" =~ ^[0-9a-f]{64}$ ]] || {
    echo "formal source SHA arguments must be lowercase SHA256" >&2
    exit 2
  }
done
[[ -n "$OUTPUT_DIR" && -n "$ADMISSION_JSON" ]] || {
  echo "--output-dir and --admission-json are required" >&2
  exit 2
}
[[ -f "$EXECUTOR_ADMISSION_JSON" ]] || {
  echo "--executor-admission-json must name the immutable Task4 admission" >&2
  exit 2
}
[[ "$CONTRACT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "--contract-sha256 must be a lowercase SHA256" >&2
  exit 2
}
[[ "$REQUEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "--request-sha256 must be a lowercase SHA256" >&2
  exit 2
}
for external_pin in \
  "$EXPECTED_RELEASE_SHA256" "$EXPECTED_MANIFEST_FILE_SHA256"; do
  [[ "$external_pin" =~ ^[0-9a-f]{64}$ ]] || {
    echo "external deployment SHA arguments must be lowercase SHA256" >&2
    exit 2
  }
done
[[ -x "$PYTHON" ]] || {
  echo "configured Python is not executable" >&2
  exit 2
}
if ((DRY_RUN == 0)) && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is empty; refusing actual GPU execution" >&2
  exit 2
fi
if ((DRY_RUN == 0)) && { [[ ! -f "$SCHEDULER_STATE_JSON" ]] || [[ -z "$CONTROLLER_ID" ]]; }; then
  echo "actual execution requires scheduler state and controller identity" >&2
  exit 2
fi
readonly -a ADMISSION_COMMAND=(
  "$PYTHON" "-"
  "$BUNDLE_CODE_ROOT"
  "$FROZEN_REPO_ROOT"
  "$EXPECTED_RELEASE_SHA256"
  "$EXPECTED_MANIFEST_FILE_SHA256"
  "$PHYSICAL_PLAN_JSON"
  "$FORMAL_V2_ROOT"
  "$LOGICAL_REQUEST_JSON"
  "$SELECTION_BINDING_JSON"
  "$CACHE_REVEAL_JSON"
  "$CACHE_SNAPSHOT_JSON"
  "$NO_GPU_RECEIPT_JSON"
  "$OUTPUT_DIR"
  "$ADMISSION_JSON"
  "$EXECUTOR_ADMISSION_JSON"
  "$SOURCE_PLAN_JSON"
  "$SOURCE_RESULT_JSON"
  "$SOURCE_PLAN_SHA256"
  "$SOURCE_RESULT_SHA256"
  "$RESOLVED_SOURCE_LOCK_SHA256"
  "$CONTRACT_SHA256"
  "$REQUEST_SHA256"
  "$GPU_UUIDS"
  "$SCHEDULER_STATE_JSON"
  "$CONTROLLER_ID"
  "$DRY_RUN"
  "${CUDA_VISIBLE_DEVICES:-}"
)
"${ADMISSION_COMMAND[@]}" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
import socket
import sys
from functools import partial
from pathlib import Path
(
    bundle_text,
    frozen_text,
    expected_release_sha,
    expected_manifest_file_sha,
    plan_text,
    formal_root_text,
    logical_request_text,
    selection_binding_text,
    cache_reveal_text,
    cache_snapshot_text,
    no_gpu_receipt_text,
    output_text,
    admission_text,
    executor_admission_text,
    source_plan_text,
    source_result_text,
    source_plan_sha,
    source_result_sha,
    resolved_source_lock_sha,
    contract_sha,
    request_sha,
    gpu_uuid_text,
    scheduler_state_text,
    controller_id,
    dry_run_text,
    cuda_visible_devices,
) = sys.argv[1:]
bundle_root = Path(bundle_text).resolve()
frozen_root = Path(frozen_text).resolve()
if bundle_root == frozen_root or not (bundle_root / "framework/stage7/deployment_bundle_v2.py").is_file():
    raise ValueError("bundle code root and frozen repo root must be distinct authenticated roots")
plan_path = Path(plan_text).resolve()
formal_root = Path(formal_root_text).resolve()
logical_request_path = Path(logical_request_text).resolve()
selection_binding_path = Path(selection_binding_text).resolve()
cache_reveal_path = Path(cache_reveal_text).resolve()
cache_snapshot_path = Path(cache_snapshot_text).resolve()
no_gpu_receipt_path = Path(no_gpu_receipt_text).resolve()
output_dir = Path(output_text).resolve()
admission_path = Path(admission_text).resolve()
executor_admission_path = Path(executor_admission_text).resolve()
source_plan_path = Path(source_plan_text).resolve()
source_result_path = Path(source_result_text).resolve()
dry_run = dry_run_text == "1"
sys.path.insert(0, str(frozen_root))
sys.path.insert(0, str(bundle_root))
from framework.stage7 import core_ablation_v2 as core_contract
from framework.stage7.deployment_bundle_v2 import EXPECTED_PRIMITIVE_SHA256, validate_deployment_bundle
from framework.stage7.actual_mode_v2 import (
    execute_recoverable_attempt,
    validate_no_gpu_receipt,
)
from framework.stage7.actual_pipeline_v2 import (
    build_concrete_stage_runner,
    validate_actual_runtime_admission,
)
from framework.stage7 import actual_pipeline_support_v2 as actual_support
from framework.stage7.physical_execution_v2 import build_independent_projection
from framework.stage7 import source_resolution_v2
from scripts import stage7_core_online_ablation_v2 as task4_online
from scripts import stage7_scheduler_requests_v2 as scheduler_requests
verification = validate_deployment_bundle(
    formal_root,
    frozen_repo_root=frozen_root,
    expected_release_sha256=expected_release_sha,
    expected_manifest_file_sha256=expected_manifest_file_sha,
)
if verification["bundle_code_root"] != str(bundle_root):
    raise SystemExit("authenticated bundle code root drift")
try:
    no_gpu_receipt_path.relative_to(formal_root)
    if no_gpu_receipt_path != (
        formal_root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    ):
        raise ValueError("receipt path is noncanonical")
    no_gpu_receipt = json.loads(no_gpu_receipt_path.read_text(encoding="utf-8"))
    validated_no_gpu_receipt = validate_no_gpu_receipt(
    no_gpu_receipt,
    root=formal_root,
    frozen_repo_root=frozen_root,
    deployment_bundle_sha256=verification["deployment_bundle_sha256"],
    )
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit("authenticated no-GPU receipt validation failed") from error
FROZEN_TASK4_CLI_SHA256 = (
    "5a01d9e1952568fa26539f5129026aa4eff514cc33e45f804b3ae988728a5c21"
)
FROZEN_PRIMITIVE_SHA256 = {
    "scripts/stage5_materialize_round_sources_v1.sh":
        "d978e6287afc239c63471cadf729b37b4617bf46d1b8b8ea13cb4e2433bf4abe",
    "scripts/stage3_tvm_int8_quant_contract_v3.py":
        "788556b775a1378e75458623c9c3186a417d30609200979694079ec628536fe5",
    "scripts/stage5_build_performance_plan_v2.py":
        "99126d0a3637f9cb51d8210b265c4d3133973a68bb3f11858966889771950912",
    "scripts/stage3_execute_performance_plan_v3.py":
        "3c0af3d857570a2bdba6b6ff5fe50a532abb5e079370119923ff0dae16f57335",
    "scripts/stage5_ap_plan_v2.py":
        "eaf175f0ce918a72924ea067d55ef1a2f8e8df3f0b54fa4afd984df9ba44c636",
    "scripts/stage3_execute_ap_plan_v3.py":
        "da71d1ba94480b9034c9dc3b16011ba0bba83166e478f865934b80b50dbecaff",
    "scripts/stage5_finalize_feedback_v2.py":
        "9c22be5270b70b5dd7cf3e3e8eede3e3ea8aecfe89b5c8d86893e42340b37393",
    "scripts/stage5_promote_actual_feedback_v3.py":
        "fbbd295cc881ebac6db50ee3764eb667bc921168a1981f96ec7c9d3fcb440187",
}
def canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def require_sha(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise SystemExit(f"{label} is not a lowercase SHA256")
    return text
task4_cli_path = frozen_root / "scripts/stage7_core_online_ablation_v2.py"
if (
    not task4_cli_path.is_file()
    or file_sha(task4_cli_path) != FROZEN_TASK4_CLI_SHA256
):
    raise SystemExit("frozen Task4 CLI SHA drift")
contract_path = formal_root / "contracts/core_ablation_v2.json"
try:
    serialized_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validated_contract = core_contract.validate_v2_contract(
        serialized_contract
    )
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit("formal v2 contract validation failed") from error
if (
    Path(str(validated_contract.get("v2_root") or "")).resolve()
    != formal_root
    or validated_contract.get("contract_sha256") != contract_sha
):
    raise SystemExit("formal v2 root/contract SHA binding drift")
for record in core_contract.FROZEN_ACTUAL_V3_EXECUTORS:
    path = frozen_root / str(record["path"])
    if not path.is_file() or file_sha(path) != record["sha256"]:
        raise SystemExit(
            f"frozen actual-feedback-v3 core SHA drift: {record['path']}"
        )
round_dir = plan_path.parent
for name, path in (
    ("physical plan", plan_path),
    ("logical request", logical_request_path),
    ("selection binding", selection_binding_path),
    ("cache reveal", cache_reveal_path),
    ("cache snapshot", cache_snapshot_path),
    ("executor admission", executor_admission_path),
    ("source plan", source_plan_path),
    ("source result", source_result_path),
):
    try:
        path.relative_to(formal_root)
    except ValueError as error:
        raise SystemExit(f"{name} escapes the formal v2 root") from error
    if path.parent != round_dir:
        raise SystemExit("Task4 admission artifacts must share one canonical round")
def load_mapping(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{label} is unavailable or invalid JSON") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return payload
try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit("physical plan is unavailable or invalid JSON") from error
if not isinstance(plan, dict):
    raise SystemExit("physical plan must be a JSON object")
recorded_physical_sha = require_sha(
    plan.get("physical_request_sha256"), "physical request SHA"
)
unsigned_plan = {
    key: value
    for key, value in plan.items()
    if key not in {"physical_request_sha256", "logical_row_bindings"}
}
if canonical_sha(unsigned_plan) != recorded_physical_sha:
    raise SystemExit("physical request SHA drift")
if plan.get("schema_version") != "stage7_actual_v3_miss_only_physical_request_v2":
    raise SystemExit("physical request schema drift")
if plan.get("logical_request_sha256") != request_sha:
    raise SystemExit("logical request SHA does not match scheduler request SHA")
logical_request = load_mapping(logical_request_path, "logical request")
selection_binding = load_mapping(selection_binding_path, "selection binding")
cache_reveal = load_mapping(cache_reveal_path, "cache reveal")
cache_snapshot = load_mapping(cache_snapshot_path, "cache snapshot")
source_plan = load_mapping(source_plan_path, "source plan")
source_result = load_mapping(source_result_path, "source result")
if source_result.get("schema_version") == "stage7_source_resolution_synthetic_dryrun_v2":
    raise SystemExit("synthetic source result cannot admit measurement")
try:
    validated_source_plan = source_resolution_v2.validate_source_resolution_plan(
        source_plan, logical_request=logical_request
    )
    validated_source_result = (
        source_resolution_v2.validate_formal_source_resolution_result(
            source_result, validated_source_plan
        )
    )
except (OSError, ValueError) as error:
    raise SystemExit("formal source result verification failed") from error
if (
    validated_source_plan["source_resolution_plan_sha256"] != source_plan_sha
    or validated_source_result["source_resolution_result_sha256"]
    != source_result_sha
):
    raise SystemExit("formal source plan/result SHA drift")
try:
    scheduler_requests.validate_resolved_source_lock(
        resolved_source_lock_sha,
        plan,
        validated_source_result,
    )
    resolved_sources = scheduler_requests.resolved_source_bindings(
        plan, validated_source_result
    )
except ValueError as error:
    raise SystemExit("resolved source lock verification failed") from error
task4_admission = load_mapping(
    executor_admission_path, "Task4 executor admission"
)
recorded_admission_sha = require_sha(
    task4_admission.get("admission_sha256"), "Task4 admission SHA"
)
unsigned_admission = {
    key: value
    for key, value in task4_admission.items()
    if key != "admission_sha256"
}
if (
    canonical_sha(unsigned_admission) != recorded_admission_sha
    or task4_admission.get("schema_version")
    != "stage7_actual_v3_executor_admission_v2"
    or task4_admission.get("admission_passed") is not True
    or task4_admission.get("contract_sha256") != contract_sha
    or task4_admission.get("logical_request_sha256") != request_sha
    or task4_admission.get("physical_request_sha256")
    != recorded_physical_sha
    or task4_admission.get("execution_primitive")
    != "existing_stage5_stage3_actual_feedback_v3"
    or task4_admission.get("gpu_jobs_launched") != 0
):
    raise SystemExit("Task4 executor admission does not authenticate this plan")
try:
    recomputed_task4_admission = task4_online.validate_executor_admission(
        plan,
        logical_request=logical_request,
        selection_binding=selection_binding,
        cache_reveal=cache_reveal,
        cache_snapshot=cache_snapshot,
        contract_sha256=contract_sha,
    )
except (OSError, ValueError) as error:
    raise SystemExit("real Task4 executor admission rejected this plan") from error
if recomputed_task4_admission != task4_admission:
    raise SystemExit("Task4 executor admission is not reproducible")
rows = plan.get("rows")
bindings = plan.get("logical_row_bindings")
physical_count = plan.get("physical_row_count")
if (
    plan.get("logical_row_count") != 4
    or not isinstance(rows, list)
    or isinstance(physical_count, bool)
    or physical_count != len(rows)
    or physical_count not in range(5)
    or not isinstance(bindings, list)
    or len(bindings) != 4
):
    raise SystemExit("miss plan must preserve one indivisible four-row logical round")
if (
    task4_admission.get("logical_row_count") != 4
    or task4_admission.get("physical_row_count") != physical_count
):
    raise SystemExit("Task4 executor admission row count drift")
row_sha = plan.get("row_sha256")
if not isinstance(row_sha, dict):
    raise SystemExit("physical row SHA map is missing")
row_ids: list[str] = []
for row in rows:
    if not isinstance(row, dict):
        raise SystemExit("physical row is not an object")
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    if (
        not row_id
        or row.get("task_id") != "S7-PYR-TVM"
        or row.get("model") != "pyramid"
        or row.get("hardware_id") != "h800"
        or row.get("dispatch_key") != "tvm_auto"
        or row.get("q_mode") not in {"fp16", "int8"}
        or row_sha.get(row_id) != canonical_sha(row)
    ):
        raise SystemExit(f"physical row identity or SHA drift: {row_id or '<empty>'}")
    row_ids.append(row_id)
if len(set(row_ids)) != len(row_ids) or set(row_sha) != set(row_ids):
    raise SystemExit("physical row identities are duplicate or incomplete")
miss_bindings = []
for expected_index, binding in enumerate(bindings):
    if not isinstance(binding, dict) or binding.get("logical_row_index") != expected_index:
        raise SystemExit("logical row binding order drift")
    disposition = binding.get("disposition")
    if disposition not in {"hit", "miss"}:
        raise SystemExit("logical row disposition is invalid")
    for field in (
        "logical_row_sha256",
        "logical_exact_binding_sha256",
        "exact_cache_key_sha256",
    ):
        require_sha(binding.get(field), f"logical binding {field}")
    if disposition == "miss":
        miss_bindings.append(binding)
if [str(item.get("candidate_id") or "") for item in miss_bindings] != row_ids:
    raise SystemExit("physical rows do not exactly match selected miss bindings")
if any(
    binding.get("physical_row_sha256")
    != row_sha.get(str(binding.get("candidate_id") or ""))
    for binding in miss_bindings
):
    raise SystemExit("physical row/binding SHA drift")
gpu_uuids = tuple(value.strip() for value in gpu_uuid_text.split(",") if value.strip())
if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4:
    raise SystemExit("one logical round requires four unique immutable GPU UUID leases")
if dry_run and cuda_visible_devices:
    raise SystemExit("dry-run requires CUDA_VISIBLE_DEVICES to be empty")
if not dry_run and cuda_visible_devices != ",".join(gpu_uuids):
    raise SystemExit("actual CUDA_VISIBLE_DEVICES differs from ordered leases")
primitive_specs = (
    ("stage3_v3_quant_contract", "scripts/stage3_tvm_int8_quant_contract_v3.py"),
    ("stage5_v2_performance_plan_builder", "scripts/stage5_build_performance_plan_v2.py"),
    ("stage3_v3_performance_executor", "scripts/stage3_execute_performance_plan_v3.py"),
    ("stage5_v2_ap_plan_builder", "scripts/stage5_ap_plan_v2.py"),
    ("stage3_v3_ap_executor", "scripts/stage3_execute_ap_plan_v3.py"),
    ("stage5_v2_finalizer_input", "scripts/stage5_finalize_feedback_v2.py"),
    ("stage5_actual_feedback_v3_promoter_input", "scripts/stage5_promote_actual_feedback_v3.py"),
)
execution_steps = []
for primitive, relative in primitive_specs:
    path = (frozen_root / relative).resolve()
    try:
        path.relative_to(frozen_root)
    except ValueError as error:
        raise SystemExit(f"primitive escapes repository: {relative}") from error
    expected_sha = FROZEN_PRIMITIVE_SHA256.get(relative)
    actual_sha = file_sha(path) if path.is_file() else None
    if actual_sha != expected_sha:
        raise SystemExit(f"frozen execution primitive SHA drift: {relative}")
    execution_steps.append(
        {
            "primitive": primitive,
            "path": str(path),
            "sha256": actual_sha,
            "exists": True,
            "mode": "input_contract_only" if primitive.endswith("_input") else "existing_primitive",
        }
    )
try:
    independent_projection = build_independent_projection(
        logical_request=logical_request,
        selection_binding=selection_binding,
        cache_reveal=cache_reveal,
        physical_plan=plan,
        source_plan=validated_source_plan,
        source_result=validated_source_result,
        executor_admission=task4_admission,
        deployment_manifest=verification,
        deployment_root=formal_root,
        frozen_repo_root=frozen_root,
        expected_release_sha256=expected_release_sha,
        expected_manifest_file_sha256=expected_manifest_file_sha,
    )
except (OSError, ValueError) as error:
    raise SystemExit("Phase5B independent projection admission failed") from error
projection_sha = (
    independent_projection.get("measurement_request_sha256")
    or independent_projection.get("empty_physical_terminal_sha256")
)
require_sha(projection_sha, "independent projection SHA")
actual_result = None
scheduler_lease_binding = None
if not dry_run:
    identity = actual_support.inspect_linux_process(os.getpid())
    if identity is None:
        raise SystemExit("actual executor process identity is unavailable")
    scheduler_lease_binding = actual_support.validate_scheduler_lease(
        scheduler_state_path=Path(scheduler_state_text).absolute(),
        controller_id=controller_id,
        logical_request_sha256=request_sha,
        ordered_lease_uuids=gpu_uuids,
        ancestor_pids=identity["ancestor_pids"],
    )
    owner = scheduler_lease_binding["expected_lock_owner"]
    controller_pid = scheduler_lease_binding["controller_pid"]
    inventory_probe = actual_support.query_h800_inventory
    lock_probe = actual_support.build_scheduler_lock_probe(
        scheduler_state_path=Path(scheduler_state_text).absolute(),
        controller_id=controller_id, logical_request_sha256=request_sha,
        ordered_lease_uuids=gpu_uuids, ancestor_pids=identity["ancestor_pids"],
    )
    process_probe = lambda: actual_support.query_related_processes(
        leased_uuids=gpu_uuids,
        controller_pid=controller_pid,
        formal_v2_root=formal_root,
        expected_lock_owner=owner,
        process_probe=actual_support.inspect_linux_process,
    )
    primitive_probe = lambda relative: file_sha(frozen_root / relative)
    runtime_admission = (
        validate_actual_runtime_admission(
            host_name=socket.gethostname(),
            formal_v2_root=formal_root,
            frozen_repo_root=frozen_root,
            deployment_bundle_sha256=verification["deployment_bundle_sha256"],
            no_gpu_receipt=validated_no_gpu_receipt,
            physical_bindings=miss_bindings,
            ordered_lease_uuids=gpu_uuids,
            parent_cuda_visible_devices=cuda_visible_devices,
            expected_lock_owner=owner,
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=primitive_probe,
        )
        if physical_count
        else {}
    )
    runner = build_concrete_stage_runner(
        projection=independent_projection,
        logical_request=logical_request,
        selection_binding=selection_binding,
        physical_plan=plan,
        source_plan=validated_source_plan,
        source_result=validated_source_result,
        runtime_admission=runtime_admission,
        python_executable=Path(sys.executable),
        remote_artifact_root=round_dir / "actual_v3_execution" / recorded_physical_sha / "remote_artifacts",
        quant_plan_builder=actual_support.build_quant_plan,
        ap_plan_builder=actual_support.build_ap_plan,
        finalizer_evidence_builder=partial(
            actual_support.build_finalizer_evidence,
            deployment_bundle_sha256=verification["deployment_bundle_sha256"],
            primitive_sha256=EXPECTED_PRIMITIVE_SHA256,
        ),
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        primitive_sha_probe=primitive_probe,
        run_argv=actual_support.run_argv,
    )
    actual_result = execute_recoverable_attempt(
        round_dir=round_dir,
        physical_request_sha256=recorded_physical_sha,
        logical_request_sha256=request_sha,
        projection=independent_projection,
        deployment_bundle_sha256=verification["deployment_bundle_sha256"],
        no_gpu_receipt=validated_no_gpu_receipt,
        frozen_repo_root=frozen_root,
        dry_run=False,
        stage_runner=runner,
    )
payload = {
    "schema_version": "stage7_actual_v3_miss_execution_admission_v2",
    "admission_passed": True,
    "dry_run": dry_run,
    "logical_round_indivisible": True,
    "logical_row_count": 4,
    "physical_row_count": physical_count,
    "logical_request_sha256": request_sha,
    "physical_request_sha256": recorded_physical_sha,
    "physical_plan_path": str(plan_path),
    "physical_plan_file_sha256": file_sha(plan_path),
    "task4_executor_admission_path": str(executor_admission_path),
    "task4_executor_admission_file_sha256": file_sha(
        executor_admission_path
    ),
    "task4_executor_admission_sha256": recorded_admission_sha,
    "contract_sha256": contract_sha,
    "formal_v2_root": str(formal_root),
    "bundle_code_root": str(bundle_root),
    "frozen_repo_root": str(frozen_root),
    "deployment_manifest_file_sha256": verification["deployment_manifest_file_sha256"],
    "expected_deployment_manifest_file_sha256": expected_manifest_file_sha,
    "deployment_release_sha256": verification["deployment_release_sha256"],
    "expected_deployment_release_sha256": expected_release_sha,
    "deployment_bundle_sha256": verification["deployment_bundle_sha256"],
    "no_gpu_receipt_path": str(no_gpu_receipt_path),
    "no_gpu_receipt_file_sha256": file_sha(no_gpu_receipt_path),
    "no_gpu_receipt_sha256": validated_no_gpu_receipt[
        "dry_run_receipt_sha256"
    ],
    "formal_contract_file_sha256": file_sha(contract_path),
    "task4_cli_sha256": FROZEN_TASK4_CLI_SHA256,
    "source_resolution_plan_sha256": source_plan_sha,
    "source_resolution_result_sha256": source_result_sha,
    "resolved_source_lock_sha256": resolved_source_lock_sha,
    "resolved_source_binding_count": len(resolved_sources),
    "source_verification_count": 1,
    "source_materializer_invocation_count": 0,
    "source_evidence_producer": {
        "path": "scripts/stage5_materialize_round_sources_v1.sh",
        "sha256": FROZEN_PRIMITIVE_SHA256[
            "scripts/stage5_materialize_round_sources_v1.sh"
        ],
        "mode": "preverified_source_input",
        "invocation_count": 0,
    },
    "gpu_uuid_leases": list(gpu_uuids),
    "cuda_visible_devices": cuda_visible_devices,
    "gpu_jobs_launched": 0 if dry_run else physical_count,
    "selected_only_physical_request": True,
    "measurement_semantics_copied": False,
    "execution_steps": execution_steps,
    "independent_projection_schema": independent_projection["schema_version"],
    "independent_projection_sha256": projection_sha,
    "actual_mode_recovery_engine": (
        "framework.stage7.actual_mode_v2.execute_recoverable_attempt"
    ),
    "output_dir": str(output_dir),
    "scheduler_lease_binding": scheduler_lease_binding,
    "actual_execution_result": actual_result,
}
payload["admission_sha256"] = canonical_sha(payload)
encoded = (
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
).encode("utf-8")
if admission_path.is_file():
    if admission_path.read_bytes() != encoded:
        raise SystemExit("request-SHA recovery admission drift")
else:
    if admission_path.exists():
        raise SystemExit("admission output path is not a regular file")
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = admission_path.with_name(
        f".{admission_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_bytes(encoded)
    os.replace(temporary, admission_path)
print(json.dumps(payload, sort_keys=True))
PY
