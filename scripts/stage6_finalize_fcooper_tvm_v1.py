#!/usr/bin/env python3
"""Finalize F-Cooper TVM five-arm evidence and the three-model TVM table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SUCCESS = "measured_success_gold"
FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
BASE_WIDTH = (64, 128, 256, 128, 256)
ARM_ORDER = (
    "Original/default",
    "Compression only",
    "Schedule only",
    "Compress -> Tune",
    "GEAR",
)
POOL_CONTRACT = {
    "compression_only": (16, 0),
    "schedule_only": (1, 64),
    "compress_then_tune_screen": (12, 0),
    "compress_then_tune_tuned": (4, 64),
    "gear": (16, 64),
}
POOL_TASK_ID = {
    "compression_only": "S6-FCO-TVM-COMPRESSION-ONLY-MEASURE-V1",
    "schedule_only": "S6-FCO-TVM-SCHEDULE-ONLY-MEASURE-V1",
    "compress_then_tune_screen": "S6-FCO-TVM-COMPRESS-THEN-TUNE-SCREEN-V1",
    "compress_then_tune_tuned": "S6-FCO-TVM-COMPRESS-THEN-TUNE-TUNED-V1",
    "gear": "S5-FCO-TVM-V1",
}
FORBIDDEN_TRT_KEYS = {
    "engine",
    "engine_path",
    "engine_sha256",
    "tactic",
    "tactic_cache",
    "tactic_cache_path",
    "builder_level",
    "trt_latency_ms",
    "trt_energy_j",
    "trt_performance",
}
OUTPUT_NAMES = (
    "fcooper_stage6_tvm_delta_ap_0.10_v1.csv",
    "fcooper_stage6_tvm_audit_v1.json",
    "fcooper_stage6_tvm_evidence_bundle_v1.json",
    "tvm_three_model_stage6_delta_ap_0.10_v1.csv",
    "tvm_three_model_stage6_table_v1.json",
    "paper_ready.json",
)
AP_REPEAT_ABS_TOL = 1e-3

class EvidenceError(ValueError):
    """Raised when formal evidence cannot support paper-ready promotion."""
def _read(path: Path) -> Any:
    if not path.is_file():
        raise EvidenceError(f"evidence file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _path_sha(path: Path) -> str:
    if path.is_file():
        return _file_sha(path)
    if path.is_dir():
        rows = [
            {
                "path": str(item.relative_to(path)),
                "sha256": _file_sha(item),
            }
            for item in sorted(path.rglob("*"))
            if item.is_file()
        ]
        if not rows:
            raise EvidenceError(f"evidence directory is empty: {path}")
        return _canonical_sha(rows)
    raise EvidenceError(f"evidence path is missing: {path}")
def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not a measured number") from exc
    if not math.isfinite(number):
        raise EvidenceError(f"{label} is non-finite")
    return number


def validate_ap_repeat(
    selected: Mapping[str, Any],
    repeated: Mapping[str, Any],
    *,
    absolute_tolerance: float = AP_REPEAT_ABS_TOL,
) -> dict[str, float]:
    drift = {
        metric: _finite(repeated.get(metric), f"repeated {metric}")
        - _finite(selected.get(metric), f"selected {metric}")
        for metric in ("ap30", "ap50", "ap70")
    }
    if any(abs(value) > absolute_tolerance for value in drift.values()):
        raise EvidenceError(
            f"winner AP repeat drift exceeds {absolute_tolerance}: {drift}"
        )
    return drift
def _resolve(value: Any, root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{label} path is missing")
    path = Path(value)
    return path if path.is_absolute() else root / path
def _verify_artifact(
    binding: Mapping[str, Any], *, root: Path, label: str
) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise EvidenceError(f"{label} artifact binding is missing")
    path = _resolve(binding.get("path"), root, label)
    expected = binding.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise EvidenceError(f"{label} SHA is missing")
    actual = _path_sha(path)
    if actual != expected:
        raise EvidenceError(f"{label} SHA mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual}
def _load_bound_json(
    binding: Mapping[str, Any], *, root: Path, label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    verified = _verify_artifact(binding, root=root, label=label)
    payload = _read(Path(verified["path"]))
    if not isinstance(payload, Mapping):
        raise EvidenceError(f"{label} must contain a JSON object")
    return dict(payload), verified
def _reject_trt_performance(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_TRT_KEYS or normalized.startswith(
                ("trt_latency", "trt_energy", "trt_engine", "trt_tactic")
            ):
                raise EvidenceError(
                    f"{context} contains forbidden TRT performance field: {key}"
                )
            _reject_trt_performance(item, context=context)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _reject_trt_performance(item, context=context)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_ap_reference(
    manifest_ap70_ref: Any,
    original: Mapping[str, Any],
) -> None:
    declared = _finite(manifest_ap70_ref, "manifest AP reference")
    measured = _finite(original.get("ap70"), "measured original AP reference")
    if not math.isclose(declared, measured, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError(
            "AP reference does not match the measured original/default evidence"
        )


def validate_paper_arm_constraints(
    winners: Mapping[str, Mapping[str, Any]],
) -> None:
    violations = [
        name
        for name, row in winners.items()
        if row.get("terminal_status") == SUCCESS
        and row.get("ap_constraint_satisfied") is not True
    ]
    if violations:
        raise EvidenceError(
            "successful paper rows violate the AP constraint: "
            + ", ".join(sorted(violations))
        )


def validate_search_initialization(
    bindings: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    names = {
        "task_contract",
        "initialization_summary",
        "capability_profile",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != names:
        raise EvidenceError(
            "three SHA-bound search initialization artifacts are required"
        )
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        payload, artifact = _load_bound_json(
            bindings[name],
            root=root,
            label=f"{name} search initialization",
        )
        payloads[name] = payload
        artifacts[name] = artifact

    profile_id = "h800-tvm-fcooper-probe-conditioned-v1"
    genome_schema = [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
        "q_mode",
    ]
    task = payloads["task_contract"]
    capability_digest = task.get("capability_digest")
    if (
        task.get("schema_version") != "stage5_search_task_contract_v2"
        or task.get("task_id") != "S5-FCO-TVM-V1"
        or task.get("target_model") != "fcooper"
        or task.get("hardware_id") != "h800"
        or task.get("dispatch_key") != "tvm_auto"
        or task.get("capability_profile_id") != profile_id
        or not _is_sha256(capability_digest)
        or list(task.get("genome_schema") or []) != genome_schema
        or int(task.get("batch_size", -1)) != 4
        or int(task.get("round_count", -1)) != 4
        or int(task.get("sample_budget", -1)) != 16
        or task.get("main_search_early_stopping") is not False
    ):
        raise EvidenceError("search initialization task contract drift")

    initialization = payloads["initialization_summary"]
    coldstart = initialization.get("coldstart_audit")
    selected_ids = initialization.get("selected_row_ids")
    provenance = initialization.get("source_registry_provenance_audit")
    if (
        initialization.get("schema_version")
        != "stage5_fcooper_formal_v2_initialization_summary"
        or initialization.get("task_id") != "S5-FCO-TVM-V1"
        or int(initialization.get("coldstart_rows", -1)) != 176
        or not isinstance(coldstart, Mapping)
        or not _is_sha256(coldstart.get("rows_sha256"))
        or not _is_sha256(coldstart.get("graph_features_sha256"))
        or initialization.get("training_view_policy") != "initial_coldstart_only"
        or initialization.get("pilot_online_labels_loaded") is not False
        or initialization.get("cross_model_online_labels_loaded") is not False
        or initialization.get("probe_metrics_loaded_as_labels") is not False
        or int(initialization.get("probe_rows_excluded_from_online_budget", 0))
        <= 0
        or int(initialization.get("eligible_genomes", 0)) <= 0
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 4
        or len(set(map(str, selected_ids))) != 4
        or any(f"profile={profile_id}" not in str(row_id) for row_id in selected_ids)
        or not isinstance(provenance, Mapping)
        or int(provenance.get("pilot_references", -1)) != 0
    ):
        raise EvidenceError("search initialization summary drift")

    capability = payloads["capability_profile"]
    profiles = capability.get("capability_profiles")
    if (
        capability.get("schema_version")
        != "fcooper_tvm_capability_profiles_v1"
        or not isinstance(profiles, list)
        or len(profiles) != 1
        or not isinstance(profiles[0], Mapping)
    ):
        raise EvidenceError("search initialization capability profile drift")
    profile = profiles[0]
    features = profile.get("features")
    if (
        profile.get("schema_version") != "stage2_capability_profile_v3"
        or profile.get("capability_profile_id") != profile_id
        or profile.get("capability_digest") != capability_digest
        or profile.get("dispatch_key") != "tvm_auto"
        or profile.get("hardware_target") != "h800"
        or not isinstance(features, Mapping)
        or int(features.get("structure_axis_count", -1)) != 5
        or int(features.get("probe_count", -1)) != 5
        or _finite(
            features.get("int8_automatic_route_coverage"),
            "INT8 automatic route coverage",
        )
        != 1.0
        or int(features.get("int8_fallback_conv_count", -1)) != 0
    ):
        raise EvidenceError("search initialization capability profile drift")

    return {
        "passed": True,
        "artifacts": artifacts,
        "profile_id": profile_id,
        "capability_digest": capability_digest,
        "genome_schema": genome_schema,
        "coldstart_rows": 176,
        "eligible_genomes": int(initialization["eligible_genomes"]),
        "initial_selected_rows": 4,
        "training_view_policy": "initial_coldstart_only",
    }


def validate_precondition_audits(
    bindings: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    names = {
        "capability_probe",
        "probe_isolation",
        "recovery_numeric_gate",
        "control_provenance",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != names:
        raise EvidenceError("four SHA-bound precondition audits are required")
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for name in sorted(names):
        payload, artifact = _load_bound_json(
            bindings[name],
            root=root,
            label=f"{name} precondition audit",
        )
        payloads[name] = payload
        artifacts[name] = artifact

    capability = payloads["capability_probe"]
    probes = capability.get("probes")
    expected_probes = {
        ("fp16", "base"),
        ("fp16", "boundary"),
        ("fp32", "original"),
        ("int8", "base"),
        ("int8", "boundary"),
    }
    if (
        capability.get("schema_version")
        != "fcooper_tvm_capability_probe_audit_v1"
        or capability.get("passed") is not True
        or capability.get("all_probes_terminal") is not True
        or capability.get("label_free") is not True
        or capability.get("dispatch_key") != "tvm_auto"
        or capability.get("profile_id")
        != "h800-tvm-fcooper-probe-conditioned-v1"
        or int(capability.get("probe_count", -1)) != 5
        or not isinstance(probes, list)
        or len(probes) != 5
        or {
            (str(probe.get("precision")), str(probe.get("role")))
            for probe in probes
            if isinstance(probe, Mapping)
        }
        != expected_probes
        or any(
            not isinstance(probe, Mapping)
            or probe.get("build_run_passed") is not True
            or probe.get("correctness_passed") is not True
            or (
                probe.get("precision") == "int8"
                and (
                    probe.get("automatic_route_passed") is not True
                    or int(probe.get("fallback_conv_count", -1)) != 0
                )
            )
            for probe in probes
        )
    ):
        raise EvidenceError("capability probe audit contract drift")

    isolation = payloads["probe_isolation"]
    policy = isolation.get("policy")
    if (
        isolation.get("schema_version")
        != "fcooper_tvm_probe_isolation_audit_v1"
        or isolation.get("passed") is not True
        or isolation.get("status") != "passed"
        or isolation.get("profile_id")
        != "h800-tvm-fcooper-probe-conditioned-v1"
        or isolation.get("probe_metrics_allowed_as_cost_model_labels") is not False
        or isolation.get("probe_rows_allowed_as_winner") is not False
        or isolation.get("probe_rows_allowed_in_t16_budget") is not False
        or not isinstance(policy, Mapping)
        or any(
            policy.get(field) is not False
            for field in (
                "probe_labels_allowed_in_training",
                "probe_performance_fields_exported_to_profile",
                "probe_rows_allowed_as_winner",
                "probe_rows_allowed_in_t16",
            )
        )
    ):
        raise EvidenceError("probe isolation audit contract drift")

    recovery = payloads["recovery_numeric_gate"]
    if (
        recovery.get("schema_version")
        != "fcooper_backend_neutral_recovery_numeric_gate_v1"
        or recovery.get("passed") is not True
        or recovery.get("status") != "passed"
        or recovery.get("t16_search_allowed") is not True
        or not {
            "backend_performance",
            "backend_accuracy",
            "TRT_artifacts",
            "probe_labels",
        }.issubset(set(recovery.get("excludes") or []))
    ):
        raise EvidenceError("recovery numeric gate audit contract drift")

    controls = payloads["control_provenance"]
    if (
        controls.get("schema_version")
        != "stage6_fcooper_tvm_request_rebind_audit_v1"
        or controls.get("backend") != "tvm"
        or controls.get("dispatch_key") != "tvm_auto"
        or controls.get("capability_profile_id")
        != "h800-tvm-fcooper-probe-conditioned-v1"
        or controls.get("candidate_counts")
        != {
            "compression_only": 16,
            "compress_then_tune": 12,
            "schedule_only": 1,
        }
        or controls.get("genome_sequence_equal") is not True
        or controls.get("input_manifests_modified") is not False
        or controls.get("old_measurement_results_read") is not False
        or controls.get("source_contract_sequence_equal") is not True
        or controls.get("source_evidence_sequence_equal") is not True
        or list(controls.get("trt_performance_artifact_fields") or [])
    ):
        raise EvidenceError("control provenance audit contract drift")
    return {
        "passed": True,
        "payloads": payloads,
        "artifacts": artifacts,
    }


def validate_control_provenance(
    provenance: Mapping[str, Any],
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    mappings = provenance.get("row_identity_mappings")
    expected = {
        "compression_only": "compression_only",
        "compress_then_tune": "compress_then_tune_screen",
        "schedule_only": "schedule_only",
    }
    if not isinstance(mappings, Mapping) or set(mappings) != set(expected):
        raise EvidenceError("control provenance row mapping is incomplete")
    for mapping_name, pool_name in expected.items():
        entries = mappings.get(mapping_name)
        if not isinstance(entries, list):
            raise EvidenceError("control provenance row mapping is invalid")
        rebound = [
            str(entry.get("rebound_row_id") or "")
            for entry in entries
            if isinstance(entry, Mapping)
        ]
        pool_ids = [str(row.get("row_id") or "") for row in pools[pool_name]]
        if (
            len(rebound) != len(entries)
            or len(set(rebound)) != len(rebound)
            or set(rebound) != set(pool_ids)
        ):
            raise EvidenceError(
                f"control provenance does not cover formal pool: {mapping_name}"
            )


def validate_legacy_trial_budget(
    row: Mapping[str, Any],
    raw_feedback: Mapping[str, Any],
) -> None:
    trial_derived = row.get("tvm_trials_source_field") == "tvm_max_trials"
    backend_derived = row.get("backend_source_field") == "dispatch_key"
    if not trial_derived and not backend_derived:
        return
    recorded = str(raw_feedback.get("actual_feedback_row_sha256") or "")
    unsigned = {
        key: value
        for key, value in raw_feedback.items()
        if key != "actual_feedback_row_sha256"
    }
    if (
        raw_feedback.get("row_id") != row.get("row_id")
        or (
            trial_derived
            and (
                int(raw_feedback.get("tvm_max_trials", -1))
                != int(row.get("tvm_trials", -2))
                or int(row.get("tvm_max_trials", -1))
                != int(row.get("tvm_trials", -2))
            )
        )
        or recorded != _canonical_sha(unsigned)
        or (
            backend_derived
            and (
                row.get("backend") != "tvm_auto"
                or raw_feedback.get("dispatch_key") != "tvm_auto"
            )
        )
    ):
        raise EvidenceError("legacy TVM trial budget evidence is invalid")


def _validate_formal_row_identity(row: Mapping[str, Any]) -> None:
    if (
        row.get("model") != "fcooper"
        or row.get("hardware_id") != "h800"
        or row.get("dispatch_key") != "tvm_auto"
        or row.get("capability_profile_id")
        != "h800-tvm-fcooper-probe-conditioned-v1"
        or row.get("training_source") != "online_feedback"
    ):
        raise EvidenceError("formal row identity contract drift")


def validate_success_evidence(
    source: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Validate a successful TVM row and all byte-bound artifacts."""
    row = dict(source)
    if row.get("terminal_status") != SUCCESS:
        raise EvidenceError("success validator received a non-success row")
    if str(row.get("backend")) not in {"tvm", "tvm_auto"}:
        raise EvidenceError("successful row is not from TVM automatic")
    _validate_formal_row_identity(row)
    _reject_trt_performance(row, context=str(row.get("row_id") or "row"))
    for metric in ("ap30", "ap50", "ap70", "latency_ms", "energy_j"):
        row[metric] = _finite(row.get(metric), metric)
    width = tuple(int(value) for value in row.get("width") or ())
    if len(width) != 5 or any(value <= 0 for value in width):
        raise EvidenceError("F-Cooper row must contain five positive widths")
    q_mode = str(row.get("q_mode") or "")
    if q_mode not in {"fp16", "fp32", "int8"}:
        raise EvidenceError("unsupported q_mode")

    artifacts = row.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceError("successful row has no artifact bindings")
    required = [
        "checkpoint",
        "onnx",
        "tvm_module",
        "tvm_database",
        "performance_report",
        "ap_report",
        "source_provenance",
    ]
    if q_mode == "int8":
        required.append("quant_contract")
    if (
        row.get("tvm_trials_source_field") == "tvm_max_trials"
        or row.get("backend_source_field") == "dispatch_key"
    ):
        required.append("feedback_row")
    verified = {
        name: _verify_artifact(artifacts.get(name), root=root, label=name)
        for name in required
    }
    performance = _read(Path(verified["performance_report"]["path"]))
    ap_report = _read(Path(verified["ap_report"]["path"]))
    provenance = _read(Path(verified["source_provenance"]["path"]))
    if not all(isinstance(payload, Mapping) for payload in (performance, ap_report, provenance)):
        raise EvidenceError("performance, AP, and provenance reports must be objects")
    _reject_trt_performance(provenance, context="source provenance")
    if (
        row.get("tvm_trials_source_field") == "tvm_max_trials"
        or row.get("backend_source_field") == "dispatch_key"
    ):
        raw_feedback = _read(Path(verified["feedback_row"]["path"]))
        if not isinstance(raw_feedback, Mapping):
            raise EvidenceError("legacy feedback row must be an object")
        validate_legacy_trial_budget(row, raw_feedback)
    if performance.get("row_id") != row.get("row_id") or any(
        not math.isclose(
            _finite(performance.get(metric), f"performance report {metric}"),
            row[metric],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for metric in ("latency_ms", "energy_j")
    ):
        raise EvidenceError("performance report metrics do not match terminal row")
    performance_source = _validate_performance_source(
        performance,
        row=row,
        verified_artifacts=verified,
        root=root,
    )
    if ap_report.get("row_id") != row.get("row_id") or any(
        not math.isclose(
            _finite(ap_report.get(metric), f"AP report {metric}"),
            row[metric],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for metric in ("ap30", "ap50", "ap70")
    ):
        raise EvidenceError("AP report metrics do not match terminal row")
    if (
        ap_report.get("schema_version") != "fcooper_tvm_normalized_ap_report_v1"
        or not isinstance(ap_report.get("source_report"), Mapping)
    ):
        raise EvidenceError("normalized AP report lacks a bound full source report")
    source_ap_payload, source_ap_binding = _load_bound_json(
        ap_report["source_report"],
        root=root,
        label="2170-sample full AP source report",
    )
    _validate_pool_full_ap_source(
        source_ap_payload,
        row=row,
        verified_artifacts=verified,
    )
    if q_mode == "int8":
        quant_payload = _read(Path(verified["quant_contract"]["path"]))
        if not isinstance(quant_payload, Mapping):
            raise EvidenceError("INT8 quant contract must be a JSON object")
        _reject_trt_performance(quant_payload, context="INT8 quant contract")
    return {
        **row,
        "width": list(width),
        "credible_terminal_status": True,
        "verified_artifacts": {
            **verified,
            "performance_source_report": performance_source,
            "full_ap_source_report": source_ap_binding,
        },
    }


def _validate_performance_source(
    report: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    verified_artifacts: Mapping[str, Mapping[str, str]],
    root: Path,
) -> dict[str, str]:
    if (
        report.get("schema_version")
        != "fcooper_tvm_normalized_performance_report_v1"
        or not isinstance(report.get("source_report"), Mapping)
    ):
        raise EvidenceError(
            "normalized performance report lacks a bound source report"
        )
    source, binding = _load_bound_json(
        report["source_report"],
        root=root,
        label="TVM performance source report",
    )
    q_mode = str(row.get("q_mode") or "")
    module = verified_artifacts["tvm_module"]
    database = verified_artifacts["tvm_database"]
    artifact_sha = source.get("artifact_sha256") or source.get("artifact_digest")
    latency = source.get("latency")
    energy = source.get("energy")
    source_latency = latency.get("latency_ms_p50") if isinstance(latency, Mapping) else None
    source_energy = None
    if isinstance(energy, Mapping):
        source_energy = energy.get("energy_J")
        if source_energy is None:
            source_energy = energy.get("joule_per_inference")
        if source_energy is None:
            source_energy = energy.get("joules_per_inference")
    if (
        source.get("status") != "success"
        or source.get("build_success") is not True
        or str(Path(str(source.get("artifact_path") or "")).resolve())
        != str(Path(module["path"]).resolve())
        or artifact_sha != module["sha256"]
        or not math.isclose(
            _finite(source_latency, "performance source latency"),
            _finite(row.get("latency_ms"), "row latency"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite(source_energy, "performance source energy"),
            _finite(row.get("energy_j"), "row energy"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise EvidenceError("performance source identity does not match formal row")

    module_parent = Path(module["path"]).resolve().parent
    database_path = Path(database["path"]).resolve()
    if q_mode == "int8":
        tuning = source.get("tuning")
        quant = verified_artifacts.get("quant_contract")
        if (
            source.get("schema") != "route_b_int8_auto_decomp_result_v1"
            or source.get("correctness_all_exact") is not True
            or not isinstance(tuning, Mapping)
            or str(Path(str(tuning.get("database_path") or "")).resolve())
            != str(database_path)
            or not isinstance(quant, Mapping)
            or str(
                Path(str(source.get("tensor_quant_params_path") or "")).resolve()
            )
            != str(Path(quant["path"]).resolve())
            or source.get("tensor_quant_params_sha256") != quant["sha256"]
        ):
            raise EvidenceError(
                "INT8 performance source identity does not match formal row"
            )
        for path_field, sha_field in (
            ("database_tuning_record_path", "database_tuning_record_sha256"),
            ("database_workload_path", "database_workload_sha256"),
        ):
            _verify_artifact(
                {
                    "path": tuning.get(path_field),
                    "sha256": tuning.get(sha_field),
                },
                root=root,
                label=f"INT8 performance source {path_field}",
            )
    else:
        trials = int(row.get("tvm_trials", -1))
        database_valid = database_path == module_parent / "ms_work_dir"
        if trials == 0:
            database_valid = (
                database_path
                == module_parent / "zero_trial_database_contract.json"
            )
            contract = _read(database_path)
            database_valid = database_valid and (
                isinstance(contract, Mapping)
                and contract.get("schema_version")
                == "fcooper_tvm_zero_trial_database_contract_v1"
                and contract.get("row_id") == row.get("row_id")
                and int(contract.get("tvm_trials", -1)) == 0
            )
        if (
            source.get("schema") != "route_b_fp16_auto_result_v1"
            or source.get("precision") != q_mode
            or source.get("gold_measurement_complete") is not True
            or not database_valid
        ):
            raise EvidenceError(
                "FP performance source identity does not match formal row"
            )
    return binding


def _validate_pool_full_ap_source(
    report: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    verified_artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    q_mode = str(row.get("q_mode") or "")
    common_valid = (
        report.get("dataset") == "OPV2V"
        and report.get("split") == "test"
        and int(report.get("requested_samples", -1)) == 2170
        and int(report.get("processed_samples", -1)) == 2170
        and int(report.get("failed_samples", -1)) == 0
        and int(report.get("fallback_samples", -1)) == 0
        and all(
            math.isclose(
                _finite(report.get(metric), f"full AP source {metric}"),
                _finite(row.get(metric), f"row {metric}"),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for metric in ("ap30", "ap50", "ap70")
        )
    )
    if q_mode == "int8":
        mode_valid = (
            report.get("schema_version") == "fcooper_tvm_int8_ap_report_v1"
            and report.get("status") == "success"
            and report.get("ap_measured") is True
            and int(report.get("vm_calls", -1)) == 2170
            and (report.get("gates") or {}).get("full_2170") is True
            and (report.get("gates") or {}).get("passed") is True
        )
    else:
        expected_schema = {
            "fp16": "fcooper_tvm_fp16_ap_report_v1",
            "fp32": "fcooper_tvm_fp32_ap_report_v1",
        }.get(q_mode)
        mode_valid = (
            expected_schema is not None
            and report.get("schema_version") == expected_schema
            and report.get("status") == "success_full"
            and int(report.get("dataset_samples", -1)) == 2170
            and int(report.get("backend_calls", -1)) == 2170
            and (report.get("gates") or {}).get("full") is True
        )
    if not common_valid or not mode_valid:
        raise EvidenceError("2170-sample full AP source report contract drift")
    module = verified_artifacts["tvm_module"]
    checkpoint = verified_artifacts["checkpoint"]
    declared_sha = report.get("sha256") or {}
    if (
        str(Path(str(report.get("artifact_path") or "")).resolve())
        != str(Path(module["path"]).resolve())
        or report.get("artifact_sha256") != module["sha256"]
        or report.get("checkpoint_sha256") != checkpoint["sha256"]
        or (
            q_mode != "int8"
            and (
                declared_sha.get("artifact") != module["sha256"]
                or declared_sha.get("checkpoint") != checkpoint["sha256"]
            )
        )
    ):
        raise EvidenceError("full AP source identity does not match formal row")
def validate_terminal_row(
    source: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    _validate_formal_row_identity(source)
    status = source.get("terminal_status")
    if status == SUCCESS:
        return validate_success_evidence(source, root=root)
    if status not in FAILURES:
        raise EvidenceError(f"non-credible terminal status: {status!r}")
    if any(source.get(metric) is not None for metric in ("ap70", "latency_ms", "energy_j")):
        raise EvidenceError("failure row contains fabricated numerical metrics")
    reason = str(source.get("failure_reason") or "").strip()
    if not reason:
        raise EvidenceError("failure row has no failure_reason")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceError("failure row has no failure contract")
    failure = _verify_artifact(
        artifacts.get("failure_contract"), root=root, label="failure_contract"
    )
    if (
        source.get("tvm_trials_source_field") == "tvm_max_trials"
        or source.get("backend_source_field") == "dispatch_key"
    ):
        raw_feedback = _read(Path(failure["path"]))
        if not isinstance(raw_feedback, Mapping):
            raise EvidenceError("legacy failure feedback row must be an object")
        validate_legacy_trial_budget(source, raw_feedback)
    return {
        **dict(source),
        "ap30": None,
        "ap50": None,
        "ap70": None,
        "latency_ms": None,
        "energy_j": None,
        "credible_terminal_status": True,
        "verified_artifacts": {"failure_contract": failure},
    }
def select_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    if not rows:
        raise EvidenceError("arm has no terminal rows")
    success = [dict(row) for row in rows if row.get("terminal_status") == SUCCESS]
    if not success:
        return {
            **dict(sorted(rows, key=lambda row: str(row.get("row_id")))[0]),
            "selection_status": "all_candidates_terminal_failure",
            "ap_constraint_satisfied": False,
        }
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in success if _finite(row.get("ap70"), "ap70") >= floor]
    pool = feasible or success
    minimum = min(_finite(row.get("latency_ms"), "latency_ms") for row in pool)
    tied = [
        row
        for row in pool
        if _finite(row.get("latency_ms"), "latency_ms") <= minimum * 1.01
    ]
    winner = min(
        tied,
        key=lambda row: (
            _finite(row.get("energy_j"), "energy_j"),
            _finite(row.get("latency_ms"), "latency_ms"),
            str(row.get("row_id") or ""),
        ),
    )
    return {
        **winner,
        "ap70_floor": floor,
        "ap_constraint_satisfied": bool(feasible),
        "selection_status": (
            "selected_feasible"
            if feasible
            else "selected_fastest_ap_constraint_violation"
        ),
    }


def apply_validated_winner(
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    updated = {
        **dict(selected),
        **{
            metric: validation[metric]
            for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        },
    }
    replay_rows = [
        updated if row.get("row_id") == selected.get("row_id") else dict(row)
        for row in rows
    ]
    replayed = select_winner(
        replay_rows,
        ap70_ref=ap70_ref,
        max_ap_drop=max_ap_drop,
    )
    if replayed.get("row_id") != selected.get("row_id"):
        raise EvidenceError(
            "independent repeat changes arm winner identity: "
            f"{selected.get('row_id')} -> {replayed.get('row_id')}"
        )
    return replayed


def _validate_pool(
    name: str,
    binding: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    payload, verified = _load_bound_json(binding, root=root, label=f"{name} pool")
    if (
        payload.get("schema_version") != "fcooper_tvm_stage6_evidence_pool_v1"
        or payload.get("pool_name") != name
    ):
        raise EvidenceError(f"{name} pool identity contract is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise EvidenceError(f"{name} pool has no rows")
    expected_count, expected_trials = POOL_CONTRACT[name]
    if len(rows) != expected_count:
        raise EvidenceError(
            f"{name} requires {expected_count} rows, got {len(rows)}"
        )
    normalized = [validate_terminal_row(row, root=root) for row in rows]
    if any(row.get("task_id") != POOL_TASK_ID[name] for row in normalized):
        raise EvidenceError(f"{name} formal task identity contract drift")
    identities = [str(row.get("row_id") or "") for row in normalized]
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise EvidenceError(f"{name} has missing or duplicate row IDs")
    if any(int(row.get("tvm_trials", -1)) != expected_trials for row in normalized):
        raise EvidenceError(f"{name} TVM trial budget drift")
    return normalized, verified, payload
def _validate_tuned_selection(
    screen: Sequence[Mapping[str, Any]],
    tuned: Sequence[Mapping[str, Any]],
    *,
    screen_binding: Mapping[str, str],
    tuned_payload: Mapping[str, Any],
    root: Path,
) -> None:
    selection = tuned_payload.get("automatic_selection")
    if not isinstance(selection, Mapping) or selection.get("automatic") is not True:
        raise EvidenceError("Compress -> Tune lacks automatic four-point selection")
    tuned_ids = [str(row["row_id"]) for row in tuned]
    if list(selection.get("selected_row_ids") or []) != tuned_ids:
        raise EvidenceError("tuned rows do not match automatic selected_row_ids")
    if selection.get("source_pool_sha256") != screen_binding["sha256"]:
        raise EvidenceError("tuned selection is not bound to screen12")
    normalized_screen = _verify_artifact(
        selection.get("source_normalized_screen_pool"),
        root=root,
        label="automatic selection normalized screen pool",
    )
    if normalized_screen != dict(screen_binding):
        raise EvidenceError("automatic selection normalized screen binding drift")
    selector, _ = _load_bound_json(
        selection.get("source_automatic_selection"),
        root=root,
        label="automatic selection artifact",
    )
    pre_normalization = _verify_artifact(
        selection.get("source_pre_normalization_screen_pool"),
        root=root,
        label="pre-normalization screen pool",
    )
    if (
        selector.get("automatic") is not True
        or list(selector.get("selected_row_ids") or []) != tuned_ids
        or selector.get("source_pool_sha256") != pre_normalization["sha256"]
        or str(Path(str(selector.get("source_pool_path") or "")).resolve())
        != pre_normalization["path"]
    ):
        raise EvidenceError("automatic selection artifact contract drift")
    screen_ids = {str(row["row_id"]) for row in screen}
    if not set(tuned_ids).issubset(screen_ids):
        raise EvidenceError("tuned four are not members of screen12")
def _validate_gear(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    *,
    round_bindings: Sequence[Mapping[str, Any]],
    round_state_bindings: Sequence[Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    if (
        int(payload.get("outer_budget", -1)) != 16
        or int(payload.get("batch_size", -1)) != 4
        or int(payload.get("rounds", -1)) != 4
        or payload.get("atomic_feedback") is not True
    ):
        raise EvidenceError("GEAR outer-budget contract is invalid")
    if any(
        row.get("task_id") != "S5-FCO-TVM-V1"
        or row.get("training_source") != "online_feedback"
        for row in rows
    ):
        raise EvidenceError("GEAR rows are not formal S5-FCO-TVM-V1 feedback")
    if not isinstance(round_bindings, Sequence) or len(round_bindings) != 4:
        raise EvidenceError("GEAR requires four SHA-bound atomic round audits")
    rows_by_id = {str(row.get("row_id") or ""): row for row in rows}
    if len(rows_by_id) != 16 or "" in rows_by_id:
        raise EvidenceError("GEAR formal row identity set is incomplete")
    seen: set[str] = set()
    verified_rounds: list[dict[str, str]] = []
    round_row_counts: list[int] = []
    round_row_ids: list[list[str]] = []
    for round_index, binding in enumerate(round_bindings):
        audit, verified = _load_bound_json(
            binding,
            root=root,
            label=f"GEAR round {round_index:02d} atomic audit",
        )
        released = audit.get("released_feedback_rows")
        if (
            audit.get("schema_version") != "stage5_atomic_batch_audit_v2"
            or int(audit.get("budget_consumed", -1)) != 4
            or audit.get("feedback_released") is not True
            or audit.get("batch_quarantined") is not False
            or not isinstance(released, list)
            or len(released) != 4
        ):
            raise EvidenceError(
                f"GEAR round {round_index:02d} atomic audit contract drift"
            )
        round_ids: list[str] = []
        for released_row in released:
            if not isinstance(released_row, Mapping):
                raise EvidenceError("GEAR atomic feedback row is invalid")
            row_id = str(released_row.get("row_id") or "")
            normalized = rows_by_id.get(row_id)
            if (
                normalized is None
                or row_id in seen
                or released_row.get("actual_feedback_row_sha256")
                != normalized.get("actual_feedback_row_sha256")
            ):
                raise EvidenceError(
                    "GEAR atomic round rows do not match normalized formal pool"
                )
            round_ids.append(row_id)
            seen.add(row_id)
        if len(set(round_ids)) != 4:
            raise EvidenceError("GEAR atomic round contains duplicate rows")
        verified_rounds.append(verified)
        round_row_counts.append(len(round_ids))
        round_row_ids.append(round_ids)
    if seen != set(rows_by_id):
        raise EvidenceError("GEAR atomic round coverage is incomplete")
    if (
        not isinstance(round_state_bindings, Sequence)
        or len(round_state_bindings) != 3
    ):
        raise EvidenceError("GEAR requires round states 01, 02, and 03")
    verified_states: list[dict[str, str]] = []
    for round_index, binding in enumerate(round_state_bindings, start=1):
        state, verified = _load_bound_json(
            binding,
            root=root,
            label=f"GEAR round {round_index:02d} state",
        )
        release = state.get("atomic_release_audit")
        feedback = state.get("feedback_evidence_audit")
        expected_feedback_rows = 4 * round_index
        if (
            state.get("schema_version")
            != "stage5_fcooper_formal_round_state_v2"
            or state.get("task_id") != "S5-FCO-TVM-V1"
            or int(state.get("round_index", -1)) != round_index
            or int(state.get("completed_feedback_rows", -1))
            != expected_feedback_rows
            or int(state.get("actual_graph_feedback_rows", -1))
            != expected_feedback_rows
            or int(state.get("budget_consumed", -1)) != expected_feedback_rows
            or state.get("cross_model_online_labels_loaded") is not False
            or state.get("pilot_online_labels_loaded") is not False
            or state.get("probe_metrics_loaded_as_labels") is not False
            or set(state.get("selected_row_ids") or [])
            != set(round_row_ids[round_index])
            or not isinstance(release, Mapping)
            or release.get("schema_version")
            != "stage5_fcooper_atomic_release_validation_v2"
            or int(release.get("round_index", -1)) != round_index - 1
            or set(release.get("released_row_ids") or [])
            != set(round_row_ids[round_index - 1])
            or not isinstance(feedback, Mapping)
            or feedback.get("schema_version")
            != "stage5_fcooper_formal_feedback_evidence_audit_v2"
            or int(feedback.get("feedback_rows", -1))
            != expected_feedback_rows
            or int(feedback.get("verified_actual_feedback_rows", -1))
            != expected_feedback_rows
            or int(feedback.get("recovered_pruned_rows", -1))
            != expected_feedback_rows
            or int(feedback.get("prefix_only_measurement_rows", -1)) != 0
        ):
            raise EvidenceError(
                f"GEAR round {round_index:02d} feedback-chain state drift"
            )
        verified_states.append(verified)
    return {
        "passed": True,
        "round_row_counts": round_row_counts,
        "verified_round_audits": verified_rounds,
        "verified_round_states": verified_states,
        "covered_row_count": len(seen),
    }
def _validate_original(
    contract_binding: Mapping[str, Any], *, root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    contract, verified = _load_bound_json(
        contract_binding, root=root, label="original native FP32 contract"
    )
    row = contract.get("row")
    if (
        contract.get("same_scope_sha_admission") is not True
        or not isinstance(row, Mapping)
        or tuple(row.get("width") or ()) != BASE_WIDTH
        or row.get("q_mode") != "fp32"
        or row.get("backend") != "pytorch_cuda_cudnn"
        or row.get("optimized_scope") != "post_scatter_backbone_shrinker"
        or row.get("terminal_status") != SUCCESS
    ):
        raise EvidenceError(
            "original/default scope or native FP32 contract is invalid"
        )
    normalized = {
        **dict(row),
        **{
            metric: _finite(row.get(metric), metric)
            for metric in ("ap30", "ap50", "ap70", "latency_ms", "energy_j")
        },
        "credible_terminal_status": True,
        "selection_status": "fixed_original_default",
        "ap_constraint_satisfied": True,
    }
    artifacts = normalized.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvidenceError("original/default artifact bindings are missing")
    normalized["verified_artifacts"] = {
        name: _verify_artifact(artifacts.get(name), root=root, label=f"original {name}")
        for name in ("checkpoint", "config", "ap_report", "evaluation_report")
    }
    ap_report = _read(Path(normalized["verified_artifacts"]["ap_report"]["path"]))
    if not isinstance(ap_report, Mapping):
        raise EvidenceError("original/default AP report must be an object")
    for metric in ("ap30", "ap50", "ap70"):
        if not math.isclose(
            _finite(ap_report.get(metric), f"original AP report {metric}"),
            normalized[metric],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvidenceError("original/default AP report metric drift")
    for name in ("checkpoint", "config"):
        expected = normalized["verified_artifacts"][name]["sha256"]
        if ap_report.get(f"{name}_sha256") != expected:
            raise EvidenceError(f"original/default AP report {name} SHA drift")
    if int(ap_report.get("dataset_samples", -1)) != 2170:
        raise EvidenceError("original/default AP report is not the 2170-sample test")
    return normalized, verified


def _validate_native_winner_package(
    selected: Mapping[str, Any],
    payload: Mapping[str, Any],
    verified: Mapping[str, str],
    *,
    root: Path,
) -> dict[str, Any]:
    if payload.get("schema_version") != "fcooper_native_gpu7_validation_admission_v1":
        raise EvidenceError("native FP32 validation schema is invalid")
    if (
        payload.get("row_id") != selected.get("row_id")
        or int(payload.get("gpu_index", -1)) != 7
    ):
        raise EvidenceError("native FP32 GPU7 validation identity mismatch")
    artifacts = selected.get("verified_artifacts") or {}
    repeats = payload.get("performance_repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise EvidenceError("native FP32 winner requires three GPU7 repeats")
    latency: list[float] = []
    energy: list[float] = []
    for index, repeat in enumerate(repeats):
        if not isinstance(repeat, Mapping) or int(repeat.get("gpu_index", -1)) != 7:
            raise EvidenceError("native FP32 repeat is not bound to GPU7")
        report_binding = _verify_artifact(
            repeat.get("report"), root=root, label=f"native repeat {index}"
        )
        report = _read(Path(report_binding["path"]))
        if not isinstance(report, Mapping) or int(report.get("gpu_abs", -1)) != 7:
            raise EvidenceError("native FP32 repeat report is not a GPU7 result")
        repeat_latency = _finite(repeat.get("latency_ms"), "native repeat latency")
        repeat_energy = _finite(repeat.get("energy_j"), "native repeat energy")
        if any(
            not math.isclose(
                _finite(report.get(metric), f"native repeat report {metric}"),
                measured,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for metric, measured in (
                ("latency_ms", repeat_latency),
                ("energy_j", repeat_energy),
            )
        ):
            raise EvidenceError("native FP32 repeat report metric drift")
        for name in ("checkpoint", "config"):
            expected = artifacts.get(name, {}).get("sha256")
            if (
                repeat.get(f"{name}_sha256") != expected
                or report.get(f"{name}_sha256") != expected
            ):
                raise EvidenceError(f"native FP32 repeat {name} SHA drift")
        latency.append(repeat_latency)
        energy.append(repeat_energy)

    full_ap = payload.get("full_ap")
    if not isinstance(full_ap, Mapping) or int(full_ap.get("gpu_index", -1)) != 7:
        raise EvidenceError("native FP32 full AP is not bound to GPU7")
    ap_binding = _verify_artifact(
        full_ap.get("report"), root=root, label="native FP32 full AP"
    )
    evaluation_binding = _verify_artifact(
        full_ap.get("evaluation_report"),
        root=root,
        label="native FP32 evaluation report",
    )
    if (
        ap_binding["sha256"] != artifacts.get("ap_report", {}).get("sha256")
        or evaluation_binding["sha256"]
        != artifacts.get("evaluation_report", {}).get("sha256")
    ):
        raise EvidenceError("native FP32 AP artifacts drift from admitted baseline")
    for name in ("checkpoint", "config"):
        if full_ap.get(f"{name}_sha256") != artifacts.get(name, {}).get("sha256"):
            raise EvidenceError(f"native FP32 full AP {name} SHA drift")
    for metric in ("ap30", "ap50", "ap70"):
        if not math.isclose(
            _finite(full_ap.get(metric), f"native full AP {metric}"),
            _finite(selected.get(metric), f"native selected {metric}"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvidenceError("native FP32 full AP metric drift")
    return {
        "required": True,
        "passed": True,
        "evidence_kind": "reused_native_fp32_gpu7",
        "prediction_required": False,
        "prediction_omission_reason": (
            "native baseline predates backend prediction-JSONL contract; "
            "the SHA-bound 2170-sample evaluation report is retained"
        ),
        "manifest": dict(verified),
        "repeat_count": 3,
        "latency_ms": statistics.median(latency),
        "energy_j": statistics.median(energy),
        "ap30": float(full_ap["ap30"]),
        "ap50": float(full_ap["ap50"]),
        "ap70": float(full_ap["ap70"]),
        "evaluation_report": evaluation_binding,
    }


def _validate_winner_package(
    selected: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if selected.get("terminal_status") != SUCCESS:
        return {"required": False, "passed": True}
    payload, verified = _load_bound_json(
        binding,
        root=root,
        label=f"{selected.get('row_id')} independent validation",
    )
    if selected.get("backend") == "pytorch_cuda_cudnn":
        return _validate_native_winner_package(
            selected, payload, verified, root=root
        )
    try:
        validation_gpu = int(payload.get("gpu_index", -1))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("independent validation GPU identity is invalid") from exc
    if payload.get("row_id") != selected.get("row_id") or not 0 <= validation_gpu <= 7:
        raise EvidenceError("independent validation winner identity mismatch")
    if validation_gpu != 7:
        exclusive_wait = payload.get("exclusive_wait")
        if (
            not isinstance(exclusive_wait, Mapping)
            or exclusive_wait.get("exclusive") is not True
            or int(exclusive_wait.get("physical_gpu_id", -1)) != validation_gpu
            or int(exclusive_wait.get("checks", 0)) < 1
            or _finite(
                exclusive_wait.get("waited_seconds"),
                "independent validation exclusive wait",
            )
            < 0
        ):
            raise EvidenceError(
                "migrated independent validation lacks exclusive GPU evidence"
            )
    repeats = payload.get("performance_repeats")
    if not isinstance(repeats, list) or len(repeats) != 3:
        raise EvidenceError("winner requires three independent performance repeats")
    latency: list[float] = []
    energy: list[float] = []
    selected_artifacts = selected.get("verified_artifacts") or {}
    binding_names = ["checkpoint", "onnx"]
    if selected.get("backend") in {"tvm", "tvm_auto"}:
        binding_names.extend(["tvm_module", "tvm_database"])
        if selected.get("q_mode") == "int8":
            binding_names.append("quant_contract")
    for index, repeat in enumerate(repeats):
        if (
            not isinstance(repeat, Mapping)
            or int(repeat.get("gpu_index", -1)) != validation_gpu
        ):
            raise EvidenceError(
                "performance repeat is not bound to the validation GPU"
            )
        report_binding = _verify_artifact(
            repeat.get("report"), root=root, label=f"performance repeat {index}"
        )
        repeat_latency = _finite(repeat.get("latency_ms"), "repeat latency_ms")
        repeat_energy = _finite(repeat.get("energy_j"), "repeat energy_j")
        report = _read(Path(report_binding["path"]))
        if (
            not isinstance(report, Mapping)
            or report.get("schema_version")
            != "fcooper_tvm_independent_performance_v1"
            or report.get("row_id") != selected.get("row_id")
            or int(report.get("gpu_index", -1)) != validation_gpu
        ):
            raise EvidenceError("performance repeat report identity drift")
        report_sha_fields = {
            "checkpoint": "checkpoint_sha256",
            "onnx": "onnx_sha256",
            "tvm_module": "module_sha256",
            "tvm_database": "database_sha256",
            "quant_contract": "quant_contract_sha256",
        }
        if any(
            report.get(report_sha_fields[name])
            != selected_artifacts.get(name, {}).get("sha256")
            for name in binding_names
        ):
            raise EvidenceError("performance repeat report artifact drift")
        if any(
            not math.isclose(
                _finite(report.get(metric), f"repeat report {metric}"),
                measured,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for metric, measured in (
                ("latency_ms", repeat_latency),
                ("energy_j", repeat_energy),
            )
        ):
            raise EvidenceError("performance repeat report metric drift")
        latency.append(repeat_latency)
        energy.append(repeat_energy)
        for name in binding_names:
            if repeat.get(f"{name}_sha256") != selected_artifacts.get(name, {}).get(
                "sha256"
            ):
                raise EvidenceError(f"repeat {name} SHA does not match winner")
    full_ap = payload.get("full_ap")
    if (
        not isinstance(full_ap, Mapping)
        or int(full_ap.get("gpu_index", -1)) != validation_gpu
    ):
        raise EvidenceError("winner full AP is not bound to the validation GPU")
    ap_binding = _verify_artifact(
        full_ap.get("report"), root=root, label="winner full AP"
    )
    prediction_binding = _verify_artifact(
        full_ap.get("prediction"), root=root, label="winner prediction"
    )
    ap_payload = _read(Path(ap_binding["path"]))
    if not isinstance(ap_payload, Mapping):
        raise EvidenceError("winner full AP report must be an object")
    validate_full_ap_report_identity(
        ap_payload,
        selected=selected,
        prediction_sha256=prediction_binding["sha256"],
        physical_gpu_id=validation_gpu,
    )
    for metric in ("ap30", "ap50", "ap70"):
        value = _finite(full_ap.get(metric), f"full AP {metric}")
        if not math.isclose(
            _finite(ap_payload.get(metric), f"full AP report {metric}"),
            value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvidenceError("winner full AP report metric drift")
    ap_repeat_drift = validate_ap_repeat(selected, full_ap)
    for name in binding_names:
        if full_ap.get(f"{name}_sha256") != selected_artifacts.get(name, {}).get(
            "sha256"
        ):
            raise EvidenceError(f"full AP {name} SHA does not match winner")
    if full_ap.get("prediction_sha256") != prediction_binding["sha256"]:
        raise EvidenceError("full AP prediction SHA does not match bound artifact")
    return {
        "required": True,
        "passed": True,
        "physical_gpu_id": validation_gpu,
        "manifest": verified,
        "repeat_count": 3,
        "latency_ms": statistics.median(latency),
        "energy_j": statistics.median(energy),
        "ap30": float(full_ap["ap30"]),
        "ap50": float(full_ap["ap50"]),
        "ap70": float(full_ap["ap70"]),
        "ap_repeat_drift": ap_repeat_drift,
        "ap_repeat_absolute_tolerance": AP_REPEAT_ABS_TOL,
        "prediction": prediction_binding,
    }


def validate_full_ap_report_identity(
    report: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    prediction_sha256: str,
    physical_gpu_id: int = 7,
) -> None:
    if not 0 <= int(physical_gpu_id) <= 7:
        raise EvidenceError("full AP validation GPU is outside H800 GPU0-7")
    artifacts = selected.get("verified_artifacts") or {}
    module = artifacts.get("tvm_module") or {}
    checkpoint = artifacts.get("checkpoint") or {}
    q_mode = str(selected.get("q_mode") or "")
    numerical = report.get("numerical_contract") or {}
    declared_sha = report.get("sha256") or {}
    execution_device = report.get("execution_device") or {}
    accepted_schema = {
        "int8": "fcooper_tvm_int8_ap_report_v1",
        "fp16": "fcooper_tvm_fp16_ap_report_v1",
        "fp32": "fcooper_tvm_fp32_ap_report_v1",
    }.get(q_mode)
    numerical_contract_valid = (
        numerical.get("fallback_forbidden") is True
        and numerical.get("graph_input_dtype") == "uint8"
        and numerical.get("graph_output_dtype") == "uint8"
        if q_mode == "int8"
        else numerical.get("silent_fallback_forbidden") is True
        and numerical.get("artifact_compute_dtype")
        == {"fp16": "float16", "fp32": "float32"}.get(q_mode)
    )
    sample_contract_valid = (
        report.get("status") == "success"
        and report.get("ap_measured") is True
        and int(report.get("vm_calls", -1)) == 2170
        and (report.get("gates") or {}).get("full_2170") is True
        and (report.get("gates") or {}).get("passed") is True
        if q_mode == "int8"
        else report.get("status") == "success_full"
        and int(report.get("dataset_samples", -1)) == 2170
        and int(report.get("backend_calls", -1)) == 2170
    )
    if (
        accepted_schema is None
        or report.get("schema_version") != accepted_schema
        or report.get("dataset") != "OPV2V"
        or report.get("split") != "test"
        or not sample_contract_valid
        or int(report.get("processed_samples", -1)) != 2170
        or int(report.get("requested_samples", -1)) != 2170
        or int(report.get("failed_samples", -1)) != 0
        or int(report.get("fallback_samples", -1)) != 0
        or int(execution_device.get("physical_gpu_id", -1))
        != int(physical_gpu_id)
        or execution_device.get("cuda_visible_devices")
        != str(int(physical_gpu_id))
        or not numerical_contract_valid
        or str(Path(str(report.get("artifact_path") or "")).resolve())
        != str(Path(str(module.get("path") or "")).resolve())
        or report.get("artifact_sha256") != module.get("sha256")
        or report.get("checkpoint_sha256") != checkpoint.get("sha256")
        or report.get("prediction_sha256") != prediction_sha256
        or (
            q_mode != "int8"
            and (
                declared_sha.get("artifact") != module.get("sha256")
                or declared_sha.get("checkpoint") != checkpoint.get("sha256")
                or declared_sha.get("prediction") != prediction_sha256
            )
        )
    ):
        raise EvidenceError("winner full AP report identity contract drift")

def _canonical_method(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())
    aliases = {
        "original/default": "Original/default",
        "original default": "Original/default",
        "compression only": "Compression only",
        "schedule only": "Schedule only",
        "compress then tune": "Compress -> Tune",
        "compress -> tune": "Compress -> Tune",
        "joint shcosearch": "GEAR",
        "gear": "GEAR",
        "gear (ours)": "GEAR",
    }
    return aliases.get(normalized)

def extract_reference_rows(
    payload: Mapping[str, Any], *, model: str
) -> list[dict[str, Any]]:
    if payload.get("paper_ready") is not True:
        raise EvidenceError(f"{model}/TVM reference is not paper_ready")
    backend = (payload.get("backends") or {}).get("tvm")
    if not isinstance(backend, Mapping) or backend.get("paper_ready") is not True:
        raise EvidenceError(f"{model}/TVM backend artifact is not paper_ready")
    rows = ((backend.get("tables") or {}).get("delta_0.10"))
    if not isinstance(rows, list):
        raise EvidenceError(f"{model}/TVM delta_0.10 table is missing")
    selected: dict[str, dict[str, Any]] = {}
    for source in rows:
        source_model = str(source.get("model") or "").strip().lower()
        if source_model and source_model != model:
            raise EvidenceError(f"{model}/TVM reference row model identity drift")
        method = _canonical_method(source.get("method"))
        if method is None:
            continue
        if method in selected:
            raise EvidenceError(f"{model}/TVM has duplicate method {method}")
        selected[method] = {
            **dict(source),
            "model": model,
            "backend": "tvm",
            "method": method,
            "AP70": source.get("AP70", source.get("ap70")),
        }
    if set(selected) != set(ARM_ORDER):
        raise EvidenceError(f"{model}/TVM does not provide the required five arms")
    return [selected[method] for method in ARM_ORDER]


def validate_reference_artifact_identity(
    binding: Mapping[str, Any],
    artifact: Mapping[str, str],
    *,
    model: str,
) -> None:
    expected_token = f"stage6_{model}_formal"
    if (
        binding.get("model") != model
        or expected_token not in artifact["path"].lower()
    ):
        raise EvidenceError(f"{model}/TVM reference artifact model identity drift")


def validate_resource_audit(
    binding: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload, verified = _load_bound_json(
        binding,
        root=root,
        label="stage6 resource audit",
    )
    formal = payload.get("formal_measurement")
    source = payload.get("source_reuse")
    scheduler = payload.get("scheduler")
    per_gpu = payload.get("per_gpu")
    row_trial_pairs = formal.get("row_trial_pairs") if isinstance(formal, Mapping) else None
    expected_pairs = {
        (str(item.get("row_id") or ""), int(item.get("tvm_trials", -1)))
        for item in (row_trial_pairs if isinstance(row_trial_pairs, list) else [])
        if isinstance(item, Mapping)
    }
    coverage = scheduler.get("formal_pair_coverage") if isinstance(scheduler, Mapping) else None
    components = scheduler.get("components") if isinstance(scheduler, Mapping) else None
    if (
        payload.get("schema_version") != "fcooper_tvm_stage6_resource_audit_v1"
        or payload.get("passed") is not True
        or int(payload.get("outer_budget", -1)) != 16
        or not isinstance(formal, Mapping)
        or int(formal.get("row_observations", -1)) != 49
        or int(formal.get("unique_row_trial_pairs", -1)) != 49
        or int(formal.get("tvm_trials", -1)) != 1344
        or set((formal.get("pool_bindings") or {})) != set(POOL_CONTRACT)
        or not expected_pairs
        or len(expected_pairs) != 49
        or len(expected_pairs) != len(row_trial_pairs)
        or not isinstance(source, Mapping)
        or int(source.get("audited_success_rows", -1))
        != int(source.get("reused_from_trt_v2_rows", -2))
        + int(source.get("generated_in_tvm_v1_rows", -2))
        or not isinstance(scheduler, Mapping)
        or int(scheduler.get("component_count", 0)) <= 0
        or not isinstance(components, list)
        or len(components) != int(scheduler.get("component_count", -1))
        or not isinstance(coverage, Mapping)
        or int(coverage.get("expected", -1)) != len(expected_pairs)
        or int(coverage.get("successful", -1)) != len(expected_pairs)
        or list(coverage.get("missing") or [])
        or _finite(scheduler.get("gpu_hours"), "resource gpu_hours") <= 0
        or _finite(scheduler.get("wall_clock_seconds"), "resource wall_clock") <= 0
        or list(scheduler.get("nonterminal_or_abandoned_components") or [])
        or not isinstance(scheduler.get("effective_parallelism"), Mapping)
        or _finite(
            scheduler["effective_parallelism"].get("average_gpu_jobs"),
            "resource average parallelism",
        )
        < 0
        or int(scheduler["effective_parallelism"].get("peak_running_jobs", 0)) <= 0
        or not isinstance(scheduler.get("queue"), Mapping)
        or int(scheduler["queue"].get("component_count", -1))
        != int(scheduler.get("component_count", -2))
        or int(scheduler["queue"].get("queued_jobs", -1)) < 0
        or _finite(
            scheduler["queue"].get("maximum_start_delay_seconds"),
            "resource maximum queue delay",
        )
        < 0
        or not isinstance(per_gpu, Mapping)
        or set(per_gpu) != {str(index) for index in range(8)}
    ):
        raise EvidenceError("stage6 resource audit contract is incomplete")
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise EvidenceError("stage6 resource audit component is invalid")
        for name in ("audit", "state", "manifest"):
            _verify_artifact(
                {
                    "path": component.get(f"{name}_path"),
                    "sha256": component.get(f"{name}_sha256"),
                },
                root=root,
                label=f"resource scheduler component {index} {name}",
            )
    job_count = 0
    attempted_pairs: set[tuple[str, int]] = set()
    for gpu_index, record in per_gpu.items():
        if not isinstance(record, Mapping):
            raise EvidenceError("stage6 resource audit GPU record is invalid")
        attempts = record.get("job_attempts")
        if (
            not isinstance(attempts, list)
            or int(record.get("jobs_started", -1)) != len(attempts)
            or _finite(record.get("busy_seconds"), "resource GPU busy seconds") < 0
        ):
            raise EvidenceError("stage6 resource audit GPU accounting drift")
        for attempt in attempts:
            pair = (
                str(attempt.get("row_id") or ""),
                int(attempt.get("tvm_trials", -1)),
            )
            if pair not in expected_pairs:
                raise EvidenceError("resource scheduler attempt is not a formal row")
            attempted_pairs.add(pair)
        job_count += len(attempts)
    if job_count <= 0:
        raise EvidenceError("stage6 resource audit has no formal GPU attempts")
    if attempted_pairs != expected_pairs:
        raise EvidenceError("resource GPU attempt coverage is incomplete")
    return payload, verified


def build_three_model_rows(
    pyramid: Sequence[Mapping[str, Any]],
    codriving: Sequence[Mapping[str, Any]],
    fcooper: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    combined = [dict(row) for rows in (pyramid, codriving, fcooper) for row in rows]
    counts = {
        model: sum(row.get("model") == model for row in combined)
        for model in ("pyramid", "codriving", "fcooper")
    }
    if len(combined) != 15 or counts != {
        "pyramid": 5,
        "codriving": 5,
        "fcooper": 5,
    }:
        raise EvidenceError(f"three-model table is incomplete: {counts}")
    return combined

def _table_row(method: str, row: Mapping[str, Any], budget: Any) -> dict[str, Any]:
    width = row.get("width")
    q_mode = row.get("q_mode")
    configuration = (
        f"({','.join(str(value) for value in width)},{q_mode})"
        if isinstance(width, list) and q_mode
        else None
    )
    return {
        "model": "fcooper",
        "backend": "tvm",
        "method": method,
        "terminal_status": row.get("terminal_status"),
        "selection_status": row.get("selection_status"),
        "ap_constraint_satisfied": row.get("ap_constraint_satisfied"),
        "configuration": configuration,
        "AP70": row.get("ap70"),
        "latency_ms": row.get("latency_ms"),
        "energy_j": row.get("energy_j"),
        "budget": budget,
        "failure_rate": 0.0 if row.get("terminal_status") == SUCCESS else 1.0,
        "failure_reason": row.get("failure_reason"),
    }

def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvidenceError("cannot write an empty result table")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        keys = list(dict.fromkeys(key for row in rows for key in row))
        fieldnames = keys
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def finalize(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _read(manifest_path)
    if manifest.get("schema_version") != "fcooper_tvm_stage6_finalize_manifest_v1":
        raise EvidenceError("unsupported finalization manifest")
    root = _resolve(
        manifest.get("evidence_root", str(manifest_path.parent)),
        manifest_path.parent,
        "evidence_root",
    ).resolve()
    ap70_ref = _finite(manifest.get("ap70_ref"), "ap70_ref")
    max_drop = 0.10

    original, original_contract = _validate_original(
        manifest.get("original_contract"), root=root
    )
    validate_ap_reference(ap70_ref, original)
    pools = manifest.get("pools")
    if not isinstance(pools, Mapping) or set(pools) != set(POOL_CONTRACT):
        raise EvidenceError("finalization manifest does not bind all five pools")
    validated: dict[str, list[dict[str, Any]]] = {}
    pool_artifacts: dict[str, dict[str, str]] = {}
    pool_payloads: dict[str, dict[str, Any]] = {}
    for name in POOL_CONTRACT:
        rows, artifact, payload = _validate_pool(name, pools[name], root=root)
        validated[name] = rows
        pool_artifacts[name] = artifact
        pool_payloads[name] = payload
    search_initialization = validate_search_initialization(
        manifest.get("search_initialization"),
        root=root,
    )
    preconditions = validate_precondition_audits(
        manifest.get("precondition_audits"),
        root=root,
    )
    validate_control_provenance(
        preconditions["payloads"]["control_provenance"],
        validated,
    )
    resource_payload, resource_artifact = validate_resource_audit(
        manifest.get("resource_audit"),
        root=root,
    )
    actual_pairs = [
        (str(row.get("row_id") or ""), int(row.get("tvm_trials", -1)))
        for name in POOL_CONTRACT
        for row in validated[name]
    ]
    resource_pairs = {
        (str(row.get("row_id") or ""), int(row.get("tvm_trials", -1)))
        for row in resource_payload["formal_measurement"]["row_trial_pairs"]
    }
    if (
        len(actual_pairs) != 49
        or len(set(actual_pairs)) != 49
        or set(actual_pairs) != resource_pairs
    ):
        raise EvidenceError(
            "resource audit measurement identities do not match formal pools"
        )
    for name, pool_artifact in pool_artifacts.items():
        recorded = resource_payload["formal_measurement"]["pool_bindings"][name]
        if (
            recorded.get("path") != pool_artifact["path"]
            or recorded.get("sha256") != pool_artifact["sha256"]
        ):
            raise EvidenceError(f"resource audit pool binding drift: {name}")
    schedule = validated["schedule_only"][0]
    if tuple(schedule.get("width") or ()) != BASE_WIDTH or schedule.get("q_mode") != "fp32":
        raise EvidenceError("Schedule only must be original-width FP32")
    _validate_tuned_selection(
        validated["compress_then_tune_screen"],
        validated["compress_then_tune_tuned"],
        screen_binding=pool_artifacts["compress_then_tune_screen"],
        tuned_payload=pool_payloads["compress_then_tune_tuned"],
        root=root,
    )
    gear_atomic_audit = _validate_gear(
        validated["gear"],
        pool_payloads["gear"],
        round_bindings=manifest.get("gear_round_audits"),
        round_state_bindings=manifest.get("gear_round_states"),
        root=root,
    )

    winners = {
        "original_default": original,
        "compression_only": select_winner(
            validated["compression_only"],
            ap70_ref=ap70_ref,
            max_ap_drop=max_drop,
        ),
        "schedule_only": {
            **schedule,
            "selection_status": "fixed_schedule_only",
            "ap_constraint_satisfied": (
                schedule.get("terminal_status") == SUCCESS
                and float(schedule["ap70"]) >= ap70_ref - max_drop
            ),
        },
        "compress_then_tune": select_winner(
            validated["compress_then_tune_tuned"],
            ap70_ref=ap70_ref,
            max_ap_drop=max_drop,
        ),
        "gear": select_winner(
            validated["gear"], ap70_ref=ap70_ref, max_ap_drop=max_drop
        ),
    }
    validations = manifest.get("winner_validations")
    if not isinstance(validations, Mapping) or set(validations) != set(winners):
        raise EvidenceError("winner validation bindings are incomplete")
    winner_audits = {
        name: _validate_winner_package(row, validations[name], root=root)
        for name, row in winners.items()
    }
    for name in ("original_default", "schedule_only"):
        audit = winner_audits[name]
        if audit.get("required"):
            winners[name] = {
                **winners[name],
                **{
                    metric: audit[metric]
                    for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
                },
            }
    if winner_audits["schedule_only"].get("required"):
        winners["schedule_only"]["ap_constraint_satisfied"] = (
            float(winners["schedule_only"]["ap70"]) >= ap70_ref - max_drop
        )
    for name, pool_name in (
        ("compression_only", "compression_only"),
        ("compress_then_tune", "compress_then_tune_tuned"),
        ("gear", "gear"),
    ):
        audit = winner_audits[name]
        if audit.get("required"):
            winners[name] = apply_validated_winner(
                validated[pool_name],
                winners[name],
                audit,
                ap70_ref=ap70_ref,
                max_ap_drop=max_drop,
            )

    validate_paper_arm_constraints(winners)
    arm_specs = (
        ("Original/default", winners["original_default"], 1),
        ("Compression only", winners["compression_only"], 16),
        ("Schedule only", winners["schedule_only"], 1),
        ("Compress -> Tune", winners["compress_then_tune"], "12+4"),
        ("GEAR", winners["gear"], 16),
    )
    fcooper_rows = [_table_row(*spec) for spec in arm_specs]

    references = manifest.get("reference_artifacts")
    if not isinstance(references, Mapping) or set(references) != {
        "pyramid_tvm",
        "codriving_tvm",
    }:
        raise EvidenceError("two paper-ready TVM reference artifacts are required")
    pyramid_payload, pyramid_artifact = _load_bound_json(
        references["pyramid_tvm"], root=root, label="Pyramid/TVM reference"
    )
    codriving_payload, codriving_artifact = _load_bound_json(
        references["codriving_tvm"], root=root, label="CoDriving/TVM reference"
    )
    validate_reference_artifact_identity(
        references["pyramid_tvm"],
        pyramid_artifact,
        model="pyramid",
    )
    validate_reference_artifact_identity(
        references["codriving_tvm"],
        codriving_artifact,
        model="codriving",
    )
    pyramid_rows = extract_reference_rows(pyramid_payload, model="pyramid")
    codriving_rows = extract_reference_rows(codriving_payload, model="codriving")
    combined = build_three_model_rows(pyramid_rows, codriving_rows, fcooper_rows)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        _write_csv(staging / OUTPUT_NAMES[0], fcooper_rows)
        _write_csv(staging / OUTPUT_NAMES[3], combined)
        audit = {
            "schema_version": "stage6_fcooper_tvm_five_arm_audit_v1",
            "task_id": "S5-FCO-TVM-V1",
            "target_model": "fcooper",
            "hardware_id": "h800",
            "backend": "tvm_auto",
            "ap70_ref": ap70_ref,
            "ap70_floor": ap70_ref - max_drop,
            "outer_budget": 16,
            "formal_rounds": 4,
            "batch_size": 4,
            "pool_counts": {
                name: len(rows) for name, rows in validated.items()
            },
            "winner_validation": winner_audits,
            "gear_atomic_feedback_audit": gear_atomic_audit,
            "search_initialization_audit": search_initialization,
            "precondition_audits": {
                "passed": True,
                "artifacts": preconditions["artifacts"],
            },
            "resource_audit": resource_payload,
            "rows": fcooper_rows,
            "paper_ready": True,
        }
        _write_json(staging / OUTPUT_NAMES[1], audit)
        table = {
            "schema_version": "stage6_tvm_three_model_table_v1",
            "hardware_id": "h800",
            "backend": "tvm_auto",
            "delta_ap_max": 0.10,
            "model_count": 3,
            "row_count": 15,
            "rows": combined,
            "paper_ready": True,
        }
        _write_json(staging / OUTPUT_NAMES[4], table)
        bundle = {
            "schema_version": "stage6_fcooper_tvm_evidence_bundle_v1",
            "manifest": {
                "path": str(manifest_path),
                "sha256": _file_sha(manifest_path),
            },
            "original_contract": original_contract,
            "terminal_pools": pool_artifacts,
            "reference_artifacts": {
                "pyramid_tvm": pyramid_artifact,
                "codriving_tvm": codriving_artifact,
            },
            "resource_audit": resource_artifact,
            "gear_atomic_feedback_audit": gear_atomic_audit,
            "search_initialization_audit": search_initialization,
            "precondition_audits": preconditions["artifacts"],
            "selected_evidence": {
                name: {
                    "row_id": row.get("row_id"),
                    "terminal_status": row.get("terminal_status"),
                    "verified_artifacts": row.get("verified_artifacts"),
                    "independent_validation": winner_audits[name],
                }
                for name, row in winners.items()
            },
            "outputs": {
                name: {
                    "path": str(output_dir / name),
                    "sha256": _file_sha(staging / name),
                }
                for name in (
                    OUTPUT_NAMES[0],
                    OUTPUT_NAMES[1],
                    OUTPUT_NAMES[3],
                    OUTPUT_NAMES[4],
                )
            },
            "paper_ready": True,
        }
        _write_json(staging / OUTPUT_NAMES[2], bundle)
        marker = {
            "schema_version": "stage6_tvm_three_model_paper_ready_v1",
            "paper_ready": True,
            "row_count": 15,
            "fcooper_audit_sha256": _file_sha(staging / OUTPUT_NAMES[1]),
            "evidence_bundle_sha256": _file_sha(staging / OUTPUT_NAMES[2]),
            "three_model_table_sha256": _file_sha(staging / OUTPUT_NAMES[4]),
        }
        _write_json(staging / OUTPUT_NAMES[5], marker)
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    audit = finalize(args.manifest, args.output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["paper_ready"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
