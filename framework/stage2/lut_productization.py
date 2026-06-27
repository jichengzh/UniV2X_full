"""Stage2 LUT productization row schemas and validation helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LATENCY_ROW_SCHEMA = "latency_lut_row_v1"
AP_ROW_SCHEMA = "ap_anchor_row_v1"
ENERGY_ROW_SCHEMA = "energy_lut_row_v1"
LUT_ROW_SCHEMAS = {LATENCY_ROW_SCHEMA, AP_ROW_SCHEMA, ENERGY_ROW_SCHEMA}
JOB_PLAN_SCHEMA = "lut_job_plan_row_v1"
JOB_STATE_SCHEMA = "lut_job_state_row_v1"
QUARANTINE_SCHEMA = "lut_bad_db_quarantine_row_v1"

MEASURED_LATENCY_BACKEND = "h800_tvm"
MEASURED_AP_BACKEND = "model_eval"
MEASURED_ENERGY_BACKEND = "h800_tvm_power_telemetry"

VALID_MEASUREMENT_STATUSES = {
    "measured",
    "estimated",
    "proxy",
    "historical",
    "predicted",
    "failed",
    "not_done",
}
VALID_LUT_KINDS = {"latency", "ap", "energy"}
VALID_JOB_TYPES = {
    "generate_latency_lut",
    "generate_ap_lut",
    "generate_energy_lut",
    "import_existing_latency_lut",
    "import_existing_ap_lut",
    "import_existing_energy_lut",
}
VALID_JOB_STATUSES = {
    "queued",
    "running",
    "payload_ready",
    "salvaged",
    "succeeded",
    "failed",
    "skipped",
    "preflight_blocked",
    "stale_lock_requeued",
}
VALID_QUARANTINE_STATUSES = {"active", "resolved"}

TVM310_SITE_PACKAGES = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
REPO_ROOT = Path(__file__).resolve().parents[2]
TVM_RUNTIME_PYTHONPATH_PREFIX = [str(REPO_ROOT), TVM310_SITE_PACKAGES]
TVM_RUNTIME_LD_PREFIX = [
    f"{TVM310_SITE_PACKAGES}/nvidia/cuda_runtime/lib",
    f"{TVM310_SITE_PACKAGES}/tvm/lib",
]
TVM_RUNTIME_PATH_PREFIX = "/usr/local/cuda-12.2/bin"
TVM_NVLIBS_PATH = "/exdata/jichengzhi/tvm_nvlibs.path"
QUARANTINE_FAILURE_PATTERNS = (
    "cuda_error_illegal_address",
    "cuda illegal memory access",
    "cuda_error_illegal_memory_access",
    "cudaerrorillegaladdress",
    "illegal memory access",
    "device-side assert",
)


class LutProductizationError(ValueError):
    """Raised when Stage2 LUT rows or jobs violate the productization contract."""


def _safe(value: str) -> str:
    return str(value).replace(":", "-").replace("/", "-").replace(" ", "_")


def stable_config_id(
    *,
    model: str,
    candidate_id: str,
    software_point_id: str,
    quant_policy: str,
    schedule_policy: str,
) -> str:
    return (
        f"{_safe(model)}__{_safe(candidate_id)}__{_safe(software_point_id)}"
        f"__q_{_safe(quant_policy)}__s_{_safe(schedule_policy)}"
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_width_csv(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise LutProductizationError("width must contain at least one integer")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise LutProductizationError(f"width contains a non-integer value: {value}") from exc


def run_json_command(command_json: str) -> dict[str, Any]:
    try:
        command = json.loads(command_json)
    except json.JSONDecodeError as exc:
        raise LutProductizationError("command json is not valid JSON") from exc
    if not isinstance(command, list) or not all(
        isinstance(item, str) for item in command
    ):
        raise LutProductizationError("command json must be a JSON string array")
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise LutProductizationError(
            detail or f"command failed with exit code {proc.returncode}"
        )
    stdout = proc.stdout.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LutProductizationError(f"command stdout is not JSON: {stdout[:200]}") from exc
    if not isinstance(payload, dict):
        raise LutProductizationError("command stdout JSON must be an object")
    return payload


def _base_row(
    *,
    schema: str,
    config_id: str,
    model: str,
    manifest_digest: str,
    candidate_id: str,
    software_point_id: str,
    dense_stage: str,
    width: list[int],
    quant_policy: str,
    schedule_policy: str,
    backend: str,
    measurement_status: str,
    run_id: str,
    created_at: str,
    **extra: Any,
) -> dict[str, Any]:
    hardware_target = (
        "H800 Hopper"
        if str(backend).startswith("h800")
        else "not_hardware_specific"
    )
    return {
        "schema": schema,
        "row_id": f"{schema.removesuffix('_row_v1')}:{model}:{config_id}:{backend}:{run_id}",
        "config_id": config_id,
        "model": model,
        "manifest_digest": manifest_digest,
        "search_space_schema": "stage2_search_space_v1",
        "candidate_id": candidate_id,
        "software_point_id": software_point_id,
        "dense_stage": dense_stage,
        "optimized_scope": extra.pop("optimized_scope", "rsu_dense_core"),
        "width": [int(item) for item in width],
        "quant_policy": quant_policy,
        "schedule_policy": schedule_policy,
        "hardware_target": extra.pop("hardware_target", hardware_target),
        "backend": backend,
        "measurement_status": measurement_status,
        "provenance": extra.pop("provenance", "stage2_lut_productization"),
        "run_id": run_id,
        "created_at": created_at,
        "source_files": extra.pop("source_files", []),
        "raw_artifact": extra.pop("raw_artifact", None),
        "failure_reason": extra.pop("failure_reason", None),
        "notes": extra.pop("notes", ""),
        **extra,
    }


def latency_lut_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(schema=LATENCY_ROW_SCHEMA, latency_unit="us", **kwargs)


def ap_anchor_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(schema=AP_ROW_SCHEMA, **kwargs)


def energy_lut_row(**kwargs: Any) -> dict[str, Any]:
    return _base_row(
        schema=ENERGY_ROW_SCHEMA,
        energy_unit="joule_per_inference",
        **kwargs,
    )


def _require(row: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise LutProductizationError(f"missing required fields: {missing}")


def _require_non_null(row: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if row.get(field) is None]
    if missing:
        raise LutProductizationError(f"missing required measured fields: {missing}")


def validate_lut_row(row: dict[str, Any]) -> None:
    required = [
        "schema",
        "row_id",
        "config_id",
        "model",
        "manifest_digest",
        "search_space_schema",
        "candidate_id",
        "software_point_id",
        "dense_stage",
        "optimized_scope",
        "width",
        "quant_policy",
        "schedule_policy",
        "hardware_target",
        "backend",
        "measurement_status",
        "provenance",
        "run_id",
        "created_at",
        "source_files",
        "raw_artifact",
        "failure_reason",
        "notes",
    ]
    _require(row, required)

    schema = str(row["schema"])
    if schema not in LUT_ROW_SCHEMAS:
        raise LutProductizationError(f"unknown lut row schema: {schema}")
    status = str(row["measurement_status"])
    if status not in VALID_MEASUREMENT_STATUSES:
        raise LutProductizationError(f"unknown measurement_status: {status}")
    if not isinstance(row["width"], list) or not all(
        isinstance(item, int) for item in row["width"]
    ):
        raise LutProductizationError("width must be a list[int]")
    if not isinstance(row["source_files"], list):
        raise LutProductizationError("source_files must be a list")
    if status == "failed" and not row.get("failure_reason"):
        raise LutProductizationError("failed rows must include failure_reason")

    if status != "measured":
        return

    if schema == LATENCY_ROW_SCHEMA:
        if row["backend"] != MEASURED_LATENCY_BACKEND:
            raise LutProductizationError("measured latency rows must use backend=h800_tvm")
        _require_non_null(row, ["latency_unit", "latency_p50_us"])
    elif schema == AP_ROW_SCHEMA:
        if row["backend"] != MEASURED_AP_BACKEND:
            raise LutProductizationError("measured AP rows must use backend=model_eval")
        _require_non_null(
            row,
            ["metric", "metric_value", "dataset", "eval_split", "ckpt_path"],
        )
    elif schema == ENERGY_ROW_SCHEMA:
        if row["backend"] != MEASURED_ENERGY_BACKEND:
            raise LutProductizationError(
                "measured energy rows must use backend=h800_tvm_power_telemetry"
            )
        _require_non_null(
            row,
            [
                "energy_unit",
                "joule_per_inference",
                "telemetry_source",
                "idle_baseline_policy",
                "latency_run_id",
                "measurement_run_id",
                "raw_artifact",
            ],
        )


def job_plan_row(
    *,
    job_id: str,
    model: str,
    lut_kind: str,
    job_type: str,
    priority: int,
    config_id: str,
    manifest_path: str,
    registry_path: str,
    expected_output: str,
    command: list[str],
    max_attempts: int,
    timeout_s: int,
    candidate_id: str | None = None,
    software_point_id: str | None = None,
    depends_on: list[str] | None = None,
    resource: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema": JOB_PLAN_SCHEMA,
        "job_id": job_id,
        "model": model,
        "lut_kind": lut_kind,
        "job_type": job_type,
        "priority": int(priority),
        "config_id": config_id,
        "manifest_path": manifest_path,
        "registry_path": registry_path,
        "candidate_id": candidate_id,
        "software_point_id": software_point_id,
        "depends_on": [] if depends_on is None else list(depends_on),
        "expected_output": expected_output,
        "command": list(command),
        "max_attempts": int(max_attempts),
        "timeout_s": int(timeout_s),
        "resource": {} if resource is None else dict(resource),
        "created_at": created_at or utc_timestamp(),
    }
    validate_job_plan_row(row)
    return row


def validate_job_plan_row(row: dict[str, Any]) -> None:
    _require(
        row,
        [
            "schema",
            "job_id",
            "model",
            "lut_kind",
            "job_type",
            "priority",
            "config_id",
            "manifest_path",
            "registry_path",
            "expected_output",
            "command",
            "max_attempts",
            "timeout_s",
        ],
    )
    if row["schema"] != JOB_PLAN_SCHEMA:
        raise LutProductizationError(f"unexpected job plan schema: {row['schema']}")
    if row["lut_kind"] not in VALID_LUT_KINDS:
        raise LutProductizationError(f"unknown lut_kind: {row['lut_kind']}")
    if row["job_type"] not in VALID_JOB_TYPES:
        raise LutProductizationError(f"unknown job_type: {row['job_type']}")
    expected_generate_type = f"generate_{row['lut_kind']}_lut"
    if (
        str(row["job_type"]).startswith("generate_")
        and row["job_type"] != expected_generate_type
    ):
        raise LutProductizationError(
            f"{row['lut_kind']} jobs must use job_type={expected_generate_type}"
        )
    if not isinstance(row["command"], list) or not all(
        isinstance(item, str) for item in row["command"]
    ):
        raise LutProductizationError("command must be a list[str]")
    if int(row["max_attempts"]) < 1:
        raise LutProductizationError("max_attempts must be >= 1")
    if int(row["timeout_s"]) <= 0:
        raise LutProductizationError("timeout_s must be > 0")


def validate_job_state_row(row: dict[str, Any]) -> None:
    _require(row, ["schema", "job_id", "status", "attempt"])
    if row["schema"] != JOB_STATE_SCHEMA:
        raise LutProductizationError(f"unexpected job state schema: {row['schema']}")
    if row["status"] not in VALID_JOB_STATUSES:
        raise LutProductizationError(f"unknown job status: {row['status']}")
    if int(row["attempt"]) < 0:
        raise LutProductizationError("attempt must be >= 0")


def quarantine_row(
    *,
    job_id: str,
    config_id: str,
    model: str,
    lut_kind: str,
    job_type: str,
    failure_reason: str,
    status: str = "active",
    created_at: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema": QUARANTINE_SCHEMA,
        "job_id": job_id,
        "config_id": config_id,
        "model": model,
        "lut_kind": lut_kind,
        "job_type": job_type,
        "status": status,
        "failure_reason": failure_reason,
        "created_at": created_at or utc_timestamp(),
    }
    validate_quarantine_row(row)
    return row


def validate_quarantine_row(row: dict[str, Any]) -> None:
    _require(
        row,
        [
            "schema",
            "job_id",
            "config_id",
            "model",
            "lut_kind",
            "job_type",
            "status",
            "failure_reason",
            "created_at",
        ],
    )
    if row["schema"] != QUARANTINE_SCHEMA:
        raise LutProductizationError(f"unexpected quarantine schema: {row['schema']}")
    if row["lut_kind"] not in VALID_LUT_KINDS:
        raise LutProductizationError(f"unknown lut_kind: {row['lut_kind']}")
    if row["job_type"] not in VALID_JOB_TYPES:
        raise LutProductizationError(f"unknown job_type: {row['job_type']}")
    if row["status"] not in VALID_QUARANTINE_STATUSES:
        raise LutProductizationError(f"unknown quarantine status: {row['status']}")
    if not row["failure_reason"]:
        raise LutProductizationError("quarantine rows must include failure_reason")


def failure_requires_quarantine(failure_reason: str | None) -> bool:
    text = str(failure_reason or "").lower().replace("-", "_")
    return any(pattern in text for pattern in QUARANTINE_FAILURE_PATTERNS)


def is_job_quarantined(
    quarantine_rows: list[dict[str, Any]],
    job: dict[str, Any],
) -> bool:
    for row in quarantine_rows:
        if row.get("schema") != QUARANTINE_SCHEMA:
            continue
        if row.get("status") != "active":
            continue
        if row.get("job_id") == job.get("job_id"):
            return True
        if row.get("config_id") != job.get("config_id"):
            continue
        if row.get("lut_kind") not in (None, job.get("lut_kind")):
            continue
        if row.get("job_type") not in (None, job.get("job_type")):
            continue
        return True
    return False


def build_tvm_runtime_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or {})
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [*TVM_RUNTIME_PYTHONPATH_PREFIX]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ld_parts = [*TVM_RUNTIME_LD_PREFIX]
    nvlibs_path = Path(TVM_NVLIBS_PATH)
    if nvlibs_path.is_file():
        nvlibs = nvlibs_path.read_text(encoding="utf-8").strip()
        if nvlibs:
            ld_parts.append(nvlibs)
    if existing_ld:
        ld_parts.append(existing_ld)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{TVM_RUNTIME_PATH_PREFIX}:{existing_path}"
        if existing_path
        else TVM_RUNTIME_PATH_PREFIX
    )
    return env


def latest_job_status(state_rows: list[dict[str, Any]], job_id: str) -> str:
    matches = [row for row in state_rows if row.get("job_id") == job_id]
    return str(matches[-1].get("status", "queued")) if matches else "queued"


def next_queued_jobs(
    plan_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pending = [
        row
        for row in plan_rows
        if latest_job_status(state_rows, str(row["job_id"]))
        not in {"succeeded", "skipped", "running"}
    ]
    return sorted(pending, key=lambda row: (-int(row["priority"]), str(row["job_id"])))


def energy_claim_allowed_from_rows(
    rows: list[dict[str, Any]],
    *,
    telemetry_source: str | None = None,
) -> bool:
    for row in rows:
        if row.get("schema") != ENERGY_ROW_SCHEMA:
            continue
        if row.get("measurement_status") != "measured":
            continue
        try:
            validate_lut_row(row)
        except LutProductizationError:
            continue
        if row.get("backend") != MEASURED_ENERGY_BACKEND:
            continue
        if (
            telemetry_source is not None
            and row.get("telemetry_source") != telemetry_source
        ):
            continue
        return True
    return False


def coverage_from_rows(
    rows: list[dict[str, Any]],
    *,
    expected_cells: int | None = None,
) -> dict[str, int]:
    measured = sum(1 for row in rows if row.get("measurement_status") == "measured")
    failed = sum(1 for row in rows if row.get("measurement_status") == "failed")
    proxy = sum(
        1
        for row in rows
        if row.get("measurement_status") in {"proxy", "estimated", "historical", "predicted"}
    )
    expected = len(rows) if expected_cells is None else int(expected_cells)
    return {
        "expected_cells": expected,
        "measured_cells": measured,
        "failed_cells": failed,
        "proxy_cells": proxy,
    }


def validate_row(row: dict[str, Any]) -> None:
    schema = str(row.get("schema", ""))
    if schema in LUT_ROW_SCHEMAS:
        validate_lut_row(row)
        return
    if schema == JOB_PLAN_SCHEMA:
        validate_job_plan_row(row)
        return
    if schema == JOB_STATE_SCHEMA:
        validate_job_state_row(row)
        return
    if schema == QUARANTINE_SCHEMA:
        validate_quarantine_row(row)
        return
    raise LutProductizationError(f"unknown schema: {schema}")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_row(row)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
