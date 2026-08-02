"""Formal-evidence constants and semantic audit validators for Stage7 v2."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import core_ablation_v2 as contracts
from framework.stage7 import core_cache_v2
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage7_core_online_ablation_v2 as online


SCHEMA_VERSION = "stage7_finalize_core_ablation_v2"
FORMAL_EVIDENCE_SCHEMA = "stage7_core_formal_evidence_v2"
EXECUTION_CLOSURE_SCHEMA = "stage7_core_execution_closure_v2"
INTEGRATION_BLOCKING_REASON = "stage7_to_actual_feedback_v3_integration_not_yet_closed"
CORE_VARIANTS = contracts.CORE_VARIANTS
ROUND_COUNT = 4
BATCH_SIZE = 4
TOTAL_TRAJECTORIES = len(CORE_VARIANTS) * len(SEEDS)
TOTAL_EVENTS = TOTAL_TRAJECTORIES * ROUND_COUNT * BATCH_SIZE
SUCCESS_STATUSES = {"measured_success", stage5_search.SUCCESS_STATUS}
TRUE_FAILURE_STATUSES = set(stage5_search.TRUE_FAILURE_STATUSES)
REQUIRED_AUDIT_FLAGS = (
    "actual_v3_terminal_evidence",
    "budget",
    "cache",
    "deployment",
    "gpu_exclusive",
    "label_leakage",
    "lock",
    "numerical",
    "path_isolation",
    "single_variable_isolation",
)
REQUIRED_AUDIT_SCHEMAS = {
    name: f"stage7_core_{name}_audit_v2" for name in REQUIRED_AUDIT_FLAGS
}
AUDIT_SOURCE_SCHEMAS = {
    "actual_v3_terminal_evidence": {
        "stage7_actual_v3_atomic_feedback_barrier_v2",
        "stage7_core_cache_v2",
    },
    "budget": {
        "stage7_actual_v3_atomic_feedback_barrier_v2",
        "stage5_atomic_batch_audit_v2",
    },
    "cache": {"stage7_core_cache_v2", "stage7_actual_v3_selection_binding_v2"},
    "deployment": {"stage7_deploy_manifest_v1"},
    "gpu_exclusive": {
        "stage7_gpu_lease_audit_v1",
        "stage7_gpu_occupancy_snapshot_v1",
        "stage7_h800_runtime_admission_v2",
    },
    "label_leakage": {"stage7_actual_v3_selector_audit_v2"},
    "lock": {"stage7_ablation_scheduler_state_v1", "stage7_gpu_lease_audit_v1"},
    "numerical": {"stage7_actual_v3_terminal_wrapper_v2", "stage7_core_cache_v2"},
    "path_isolation": {"stage7_core_ablation_v2"},
    "single_variable_isolation": {"stage7_actual_v3_selector_audit_v2"},
}
FORMAL_OUTPUT_PATHS = (
    "aggregate/stage7_core_events_raw.json",
    "aggregate/stage7_core_events_raw.csv",
    "aggregate/stage7_core_trajectories_raw.json",
    "aggregate/stage7_core_trajectories_raw.csv",
    "aggregate/stage7_core_audit_bundle.json",
    "aggregate/stage7_core_descriptive_paired_statistics.json",
    "aggregate/stage7_core_root_cause_summary.md",
    "paper/table_stage7_core_component_ablation.csv",
    "paper/table_stage7_core_component_ablation.md",
    "status/finalization_status.json",
)
EVIDENCE_KINDS = {
    "actual_v3_success",
    "actual_v3_candidate_failure",
    "cross_request_exact_hit",
}
JSON = dict[str, Any]


class FinalizationError(ValueError):
    """The evidence tree does not satisfy the frozen formal-v2 contract."""


def fail(message: str) -> None:
    raise FinalizationError(message)


def sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"{label} is unavailable or invalid: {error}")


def read_mapping(path: Path, label: str) -> JSON:
    payload = read_json(path, label)
    if not isinstance(payload, Mapping):
        fail(f"{label} must be a JSON object")
    return copy.deepcopy(dict(payload))


def read_source_rows(path: Path, label: str) -> list[JSON]:
    if path.suffix != ".jsonl":
        payload = read_json(path, label)
        rows = payload if isinstance(payload, list) else [payload]
    else:
        try:
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            rows = [json.loads(line) for line in lines]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            fail(f"{label} is unavailable or invalid: {error}")
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        fail(f"{label} must contain one or more object rows")
    return [copy.deepcopy(dict(row)) for row in rows]


def validate_deploy_manifest(payload: Mapping[str, Any]) -> None:
    records = payload.get("files")
    if (
        not isinstance(records, list)
        or not records
        or not str(payload.get("python_environment") or "")
        or not isinstance(payload.get("git"), Mapping)
        or not str(payload["git"].get("commit") or "")
        or not isinstance(payload.get("frozen_evidence"), Mapping)
        or not payload["frozen_evidence"]
    ):
        fail("deployment source manifest is incomplete")
    for record in records:
        if not isinstance(record, Mapping):
            fail("deployment source manifest file record is invalid")
        expected = str(record.get("sha256") or "")
        source = Path(str(record.get("source") or ""))
        destination = Path(str(record.get("destination") or ""))
        if (
            not source.is_absolute()
            or not destination.is_absolute()
            or not source.is_file()
            or not destination.is_file()
            or not is_sha(expected)
            or file_sha(source) != expected
            or file_sha(destination) != expected
        ):
            fail("deployment source manifest local/remote SHA drift")


def validate_selector_audit(payload: Mapping[str, Any]) -> None:
    selected = payload.get("selected_row_ids")
    if (
        payload.get("variant") not in CORE_VARIANTS
        or payload.get("round_index") not in range(ROUND_COUNT)
        or payload.get("ordered_pre_scan_sha256")
        != contracts.EXPECTED_ORDERED_PRE_SCAN_SHA256
        or payload.get("candidate_pool") != "pre_scan"
        or payload.get("scanner_deferred") is not True
        or payload.get("cache_visible_during_selection") is not False
        or payload.get("cache_input_accepted") is not False
        or not isinstance(selected, list)
        or len(selected) != BATCH_SIZE
        or len(set(selected)) != BATCH_SIZE
        or payload.get("selected_row_ids_sha256") != sha(selected)
        or not is_sha(payload.get("logical_request_sha256"))
    ):
        fail("selector audit source semantics are invalid")


def validate_scheduler_state(payload: Mapping[str, Any]) -> None:
    controllers = payload.get("controllers")
    if (
        not isinstance(controllers, Mapping)
        or payload.get("selected_event_budget_consumed") != TOTAL_EVENTS
        or not str(payload.get("scheduler_status") or "")
        or not isinstance(payload.get("updated_wall_time"), (int, float))
    ):
        fail("scheduler state source semantics are invalid")


def validate_gpu_audit_rows(schema: str, rows: Sequence[JSON]) -> None:
    for row in rows:
        if not isinstance(row.get("wall_time"), (int, float)) or not str(
            row.get("event") or ""
        ):
            fail("GPU audit source event identity is invalid")
        if schema == "stage7_gpu_lease_audit_v1":
            if not any(
                key in row
                for key in (
                    "gpu_uuid",
                    "gpu_uuids",
                    "idle_gpu_uuids",
                    "controller_id",
                )
            ):
                fail("GPU lease source lacks UUID/controller evidence")
        elif schema == "stage7_gpu_occupancy_snapshot_v1":
            gpus = row.get("gpus")
            if not isinstance(gpus, list) or not gpus:
                fail("GPU occupancy source lacks snapshots")
            for gpu in gpus:
                if (
                    not isinstance(gpu, Mapping)
                    or not str(gpu.get("uuid") or "")
                    or not isinstance(gpu.get("compute_pids"), (list, tuple))
                ):
                    fail("GPU occupancy snapshot is invalid")
        elif (
            row.get("hostname") != "zs-nj-tap-gpu18"
            or not isinstance(row.get("gpu_models"), Mapping)
            or not row["gpu_models"]
            or any("H800" not in str(model) for model in row["gpu_models"].values())
            or not isinstance(row.get("reservations"), Mapping)
            or not isinstance(row.get("processes"), list)
        ):
            fail("H800 runtime admission source semantics are invalid")


def validate_stage5_atomic_audit(payload: Mapping[str, Any]) -> None:
    released = payload.get("released_feedback_rows")
    if (
        payload.get("feedback_released") is not True
        or payload.get("batch_quarantined") is not False
        or payload.get("budget_consumed") != BATCH_SIZE
        or not isinstance(released, list)
        or len(released) != BATCH_SIZE
        or payload.get("resume_row_ids") != []
    ):
        fail("Stage5 atomic audit source semantics are invalid")


def validate_semantic_audit_source(
    root: Path,
    *,
    kind: str,
    path: Path,
    schema: str,
    rows: Sequence[JSON],
) -> None:
    """Re-run the strongest validator; a schema-only shell is never evidence."""
    try:
        if schema == core_cache_v2.CACHE_SCHEMA:
            if len(rows) != 1:
                fail("formal cache source must be one object")
            cache = core_cache_v2.validate_actual_v3_cache(rows[0])
            if not cache["entries"] or not cache["lineage"]:
                fail("formal cache source has no authenticated entries")
            for terminal in cache["entries"].values():
                actual_adapter.validate_actual_v3_terminal_wrapper(terminal)
        elif schema == actual_adapter.TERMINAL_WRAPPER_SCHEMA:
            for row in rows:
                actual_adapter.validate_actual_v3_terminal_wrapper(row)
        elif schema == online.BARRIER_SCHEMA:
            if len(rows) != 1 or path.name != "atomic_feedback_barrier.json":
                fail("atomic barrier source path is not canonical")
            online.validate_committed_barrier(path.parent)
        elif schema == "stage5_atomic_batch_audit_v2":
            for row in rows:
                validate_stage5_atomic_audit(row)
        elif schema == core_cache_v2.SELECTION_BINDING_SCHEMA:
            if len(rows) != 1:
                fail("selection binding source must be one object")
            request = read_mapping(
                path.parent / "logical_request.json", "bound logical request"
            )
            core_cache_v2.validate_selection_binding(request, rows[0])
        elif schema == "stage7_actual_v3_selector_audit_v2":
            for row in rows:
                validate_selector_audit(row)
        elif schema == "stage7_deploy_manifest_v1":
            if len(rows) != 1:
                fail("deployment source manifest must be one object")
            validate_deploy_manifest(rows[0])
        elif schema in {
            "stage7_gpu_lease_audit_v1",
            "stage7_gpu_occupancy_snapshot_v1",
            "stage7_h800_runtime_admission_v2",
        }:
            validate_gpu_audit_rows(schema, rows)
        elif schema == "stage7_ablation_scheduler_state_v1":
            if len(rows) != 1:
                fail("scheduler state source must be one object")
            validate_scheduler_state(rows[0])
        elif schema == contracts.SCHEMA_VERSION:
            if len(rows) != 1:
                fail("formal contract source must be one object")
            contracts.validate_v2_contract(rows[0])
        else:
            fail(f"{kind} source schema has no semantic validator")
    except (OSError, TypeError, ValueError, KeyError) as error:
        fail(f"{kind} source semantic validation failed: {error}")


def validate_bound_audit_artifacts(
    root: Path,
    records: Any,
    *,
    semantic_validator: Callable[..., None] = validate_semantic_audit_source,
) -> list[JSON]:
    if not isinstance(records, list) or len(records) != len(REQUIRED_AUDIT_FLAGS):
        fail("formal execution closure lacks bound audit artifacts")
    validated: list[JSON] = []
    discovered: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            fail("formal audit artifact record is invalid")
        relative = Path(str(record.get("path") or ""))
        kind = str(record.get("kind") or "")
        if (
            kind not in REQUIRED_AUDIT_SCHEMAS
            or kind in discovered
            or relative.is_absolute()
            or ".." in relative.parts
            or "no_gpu_dry_run" in relative.parts
        ):
            fail("formal audit artifact path escapes the v2 root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            fail("formal audit artifact path escapes the v2 root")
        expected = str(record.get("sha256") or "")
        if not path.is_file() or not is_sha(expected) or file_sha(path) != expected:
            fail(f"formal audit artifact drift at index {index}")
        payload = read_mapping(path, f"{kind} formal audit")
        if (
            payload.get("schema_version") != REQUIRED_AUDIT_SCHEMAS[kind]
            or payload.get("passed") is not True
            or payload.get("evidence_origin") != "formal_h800_actual_v3"
            or payload.get("synthetic_non_measurement") is not False
            or payload.get("eligible_for_formal_finalization") is not True
        ):
            fail(f"{kind} formal audit is synthetic, self-signed or incomplete")
        sources = payload.get("source_artifacts")
        if (
            not isinstance(sources, list)
            or not sources
            or payload.get("source_artifact_count") != len(sources)
        ):
            fail(f"{kind} formal audit lacks source lineage")
        for source in sources:
            if not isinstance(source, Mapping):
                fail(f"{kind} source lineage record is invalid")
            source_relative = Path(str(source.get("path") or ""))
            source_path = (root / source_relative).resolve()
            source_sha = str(source.get("sha256") or "")
            source_schema = str(source.get("schema_version") or "")
            try:
                source_path.relative_to(root)
            except ValueError:
                fail(f"{kind} source lineage escapes the formal root")
            if (
                source_relative.is_absolute()
                or ".." in source_relative.parts
                or "no_gpu_dry_run" in source_relative.parts
                or source_path == path
                or not source_path.is_file()
                or not is_sha(source_sha)
                or file_sha(source_path) != source_sha
                or source_schema not in AUDIT_SOURCE_SCHEMAS[kind]
            ):
                fail(f"{kind} source lineage identity drift")
            source_rows = read_source_rows(source_path, f"{kind} source lineage")
            if any(
                str(row.get("schema_version") or row.get("schema") or "")
                != source_schema
                for row in source_rows
            ):
                fail(f"{kind} source lineage schema drift")
            semantic_validator(
                root,
                kind=kind,
                path=source_path,
                schema=source_schema,
                rows=source_rows,
            )
        discovered.add(kind)
        validated.append({"kind": kind, "path": str(relative), "sha256": expected})
    if discovered != set(REQUIRED_AUDIT_FLAGS):
        fail("formal audit artifact kind coverage drift")
    return validated
