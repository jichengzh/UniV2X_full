from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage5 import measurement_plan_v2 as stage5_measurement_plan
from framework.stage7 import actual_v3_selector_v2 as selector
from framework.stage7 import core_cache_v2 as cache_v2
from scripts import stage3_execute_ap_plan_v3 as stage3_ap_executor
from scripts import stage3_gold96_ap_plan_v3 as stage3_ap_plan
from scripts import stage5_ap_plan_v2 as stage5_ap_plan


def _adapter() -> Any:
    return importlib.import_module("framework.stage7.actual_v3_adapter_v2")


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _request() -> dict[str, Any]:
    rows = [
        {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": f"candidate-{index}",
            "manifest_job_id": f"candidate-{index}",
            "model": "pyramid",
            "genome": [16 + index * 8, 32, 64, "fp16"],
            "q_mode": "fp16",
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-auto-v1",
            "dispatch_key": "tvm_auto",
        }
        for index in range(4)
    ]
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _dimensions(candidate_id: str, genome: list[Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "model": "pyramid",
        "capability_profile_id": "h800-tvm-auto-v1",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "measurement_scope": "formal_h800_latency_energy_ap",
        "input_protocol_sha256": "1" * 64,
        "batch_size": 1,
        "genome": genome,
        "q_mode": "fp16",
        "source_checkpoint_sha256": "2" * 64,
        "onnx_sha256": "3" * 64,
        "build_protocol_sha256": "4" * 64,
        "tuning_protocol_sha256": "5" * 64,
        "measurement_protocol_sha256": "6" * 64,
        "ap_protocol_sha256": "7" * 64,
        "runtime_contract_sha256": "8" * 64,
    }


def _frozen() -> dict[str, Any]:
    adapter = _adapter()
    request = _request()
    dimensions = {
        row["row_id"]: _dimensions(row["row_id"], row["genome"])
        for row in request["rows"]
    }
    return adapter.bind_selector_output(
        {
            "acquisition": {
                "selected_row_ids": [row["row_id"] for row in request["rows"]]
            },
            "measurement_request": request,
        },
        exact_dimensions_by_candidate=dimensions,
    )


def _empty_cache() -> dict[str, Any]:
    return {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }


def _artifact(
    root: Path,
    name: str,
    content: dict[str, Any],
    *,
    artifact_kind: str | None = None,
) -> dict[str, str]:
    path = root / f"{name}.json"
    encoded = json.dumps(
        content, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "artifact_kind": artifact_kind or name,
        "path": str(path),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _normalized_stage3_terminal_inputs(
    candidate_id: str,
    request: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    graph = {
        "row_id": candidate_id,
        "graph_feature_provenance": "materialized_onnx_extracted_v1",
        "conv_count": 21,
    }
    terminal = {
        "row_id": candidate_id,
        "terminal_status": "measured_success_gold",
        "measurement_request_row_sha256": request["row_sha256"][
            candidate_id
        ],
        "latency_ms": 1.0,
        "energy_j": 0.5,
        "ap30": 0.9,
        "ap50": 0.8,
        "ap70": 0.7,
        "materialized_graph_features_sha256": _sha(graph),
    }
    terminal["actual_feedback_row_sha256"] = _sha(terminal)
    row_binding = next(
        row
        for row in plan["logical_row_bindings"]
        if row["candidate_id"] == candidate_id
    )
    common = {
        "candidate_id": candidate_id,
        "logical_request_sha256": request["measurement_request_sha256"],
        "physical_request_sha256": plan["physical_request_sha256"],
        "physical_row_sha256": row_binding["physical_row_sha256"],
    }
    performance = {
        "schema_version": "stage3_performance_result_v3",
        **common,
        "latency_ms": terminal["latency_ms"],
        "energy_j": terminal["energy_j"],
    }
    ap = {
        "schema_version": "stage3_ap_report_v3",
        **common,
        "ap30": terminal["ap30"],
        "ap50": terminal["ap50"],
        "ap70": terminal["ap70"],
    }
    return {
        "stage5_terminal": terminal,
        "stage5_terminal_artifact": _artifact(
            root,
            f"{candidate_id}-stage5_terminal",
            terminal,
            artifact_kind="stage5_terminal",
        ),
        "stage3_performance_artifact": _artifact(
            root,
            f"{candidate_id}-stage3-performance",
            performance,
            artifact_kind="stage3_performance",
        ),
        "stage3_ap_artifact": _artifact(
            root,
            f"{candidate_id}-stage3-ap",
            ap,
            artifact_kind="stage3_ap",
        ),
        "actual_graph_features": graph,
    }


def _write_json(root: Path, name: str, content: Any) -> tuple[Path, str]:
    path = root / f"{name}.json"
    encoded = json.dumps(
        content, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def _write_jsonl(root: Path, name: str, rows: list[dict[str, Any]]) -> tuple[Path, str]:
    path = root / f"{name}.jsonl"
    encoded = (
        "\n".join(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def _raw_stage3_terminal_inputs(
    candidate_id: str,
    request: dict[str, Any],
    plan: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    dimensions = _dimensions(
        candidate_id,
        next(
            row["genome"]
            for row in request["rows"]
            if row["row_id"] == candidate_id
        ),
    )
    graph = {
        "row_id": candidate_id,
        "graph_feature_provenance": "materialized_onnx_extracted_v1",
        "conv_count": 21,
    }
    performance_raw = {
        "schema": "route_b_fp16_auto_result_v1",
        "status": "success",
        "build_success": True,
        "correctness_vs_default_fp16": [{"cosine": 1.0}],
        "latency": {"latency_ms_p50": 1.0},
        "energy": {"energy_j": 0.5},
    }
    compiled_path = root / f"{candidate_id}-route_b_fp16_auto.so"
    compiled_path.write_bytes(b"real-stage5-builder-shape-fixture")
    performance_raw["compiled_artifact_path"] = str(compiled_path)
    performance_raw["artifact_digest"] = hashlib.sha256(
        compiled_path.read_bytes()
    ).hexdigest()
    performance_path, performance_sha = _write_json(
        root, f"{candidate_id}-raw-performance", performance_raw
    )
    ap_raw = {
        "status": "success",
        "processed_samples": 1789,
        "fallback_samples": 0,
        "failed_samples": 0,
        "ap_measured": True,
        "smoke_gate_passed": True,
        "ap30": 0.9,
        "ap50": 0.8,
        "ap70": 0.7,
    }
    ap_path, ap_sha = _write_json(
        root, f"{candidate_id}-raw-ap", ap_raw
    )
    performance_manifest_row = {
        "schema_version": "stage5_performance_manifest_row_v1",
        "manifest_job_id": candidate_id,
        "job_id": candidate_id,
        "row_id": candidate_id,
        "group_id": f"pyramid|{'x'.join(map(str, dimensions['genome'][:3]))}",
        "split": "online_feedback",
        "model": "pyramid",
        "genome": dimensions["genome"],
        "width": dimensions["genome"][:3],
        "precision": dimensions["q_mode"],
        "q_mode": "fp16",
        "capability_profile_id": dimensions["capability_profile_id"],
        "hardware_id": dimensions["hardware_id"],
        "dispatch_key": "tvm_auto",
        "measurement_scope": dimensions["measurement_scope"],
        "input_protocol_sha256": dimensions["input_protocol_sha256"],
        "batch_size": dimensions["batch_size"],
        "build_protocol_sha256": dimensions["build_protocol_sha256"],
        "tuning_protocol_sha256": dimensions["tuning_protocol_sha256"],
        "measurement_protocol_sha256": dimensions[
            "measurement_protocol_sha256"
        ],
        "ap_protocol_sha256": dimensions["ap_protocol_sha256"],
        "runtime_contract_sha256": dimensions[
            "runtime_contract_sha256"
        ],
        "source_contract": {
            "checkpoint_sha256": dimensions[
                "source_checkpoint_sha256"
            ],
            "onnx_sha256": dimensions["onnx_sha256"],
            "onnx_path": str(root / f"{candidate_id}.onnx"),
            "calibration_root": str(root / "calibration"),
        },
        "required_metrics": ["latency", "energy", "ap"],
    }
    performance_manifest = {
        "schema_version": "stage5_performance_manifest_v2",
        "source_request_sha256": request["measurement_request_sha256"],
        "jobs": [performance_manifest_row],
    }
    performance_manifest_path, performance_manifest_sha = _write_json(
        root,
        f"{candidate_id}-performance-manifest",
        performance_manifest,
    )
    performance_job = {
        **stage5_measurement_plan._build_job(
            performance_manifest_row,
            batch_index=1,
            row_index=0,
            remote_artifact_root=root / "remote",
            gpus=[0],
        ),
        "schema_version": stage5_measurement_plan.JOB_SCHEMA,
    }
    performance_jobs_path, performance_jobs_sha = _write_jsonl(
        root,
        f"{candidate_id}-performance-jobs",
        [performance_job],
    )
    performance_state = {
        "schema_version": "stage3_execute_performance_plan_v3_state",
        "job_id": performance_job["job_id"],
        "attempt": 1,
        "status": "success",
        "returncode": 0,
        "result_json": str(performance_path),
        "result_sha256": performance_sha,
        "failure_reasons": [],
    }
    performance_state_path, performance_state_sha = _write_jsonl(
        root,
        f"{candidate_id}-performance-state",
        [performance_state],
    )
    underlying_ap_rows = stage3_ap_plan.build_ap_plan(
        performance_manifest,
        performance_jobs=[performance_job],
        performance_state_rows=[performance_state],
        pilot_root=root / "ap-pilot",
        manifest_schema="stage5_performance_manifest_v2",
        expected_row_count=1,
    )
    ap_plan_json = root / f"{candidate_id}-ap-plan.json"
    ap_plan_path = root / f"{candidate_id}-ap-plan.jsonl"
    stage5_ap_plan.write_stage5_outputs(
        underlying_ap_rows,
        ap_plan_json,
        ap_plan_path,
    )
    ap_plan_sha = hashlib.sha256(ap_plan_path.read_bytes()).hexdigest()
    ap_plan = json.loads(ap_plan_path.read_text(encoding="utf-8"))
    ap_state = {
        "record_type": "job_terminal",
        "job_id": candidate_id,
        "model": "pyramid",
        "stage": "full",
        "status": "success",
        "report_path": str(ap_path),
        "report_sha256": ap_sha,
        "ap": {"ap30": 0.9, "ap50": 0.8, "ap70": 0.7},
        "plan_fingerprint": stage3_ap_executor.plan_fingerprint(
            ap_plan, "full"
        ),
    }
    ap_state_path, ap_state_sha = _write_jsonl(
        root, f"{candidate_id}-ap-state", [ap_state]
    )
    terminal = {
        "row_id": candidate_id,
        "manifest_job_id": candidate_id,
        "terminal_status": "measured_success_gold",
        "measurement_request_row_sha256": request["row_sha256"][
            candidate_id
        ],
        "performance_result_json": str(performance_path),
        "performance_result_sha256": performance_sha,
        "ap_report_path": str(ap_path),
        "ap_report_sha256": ap_sha,
        "latency_ms": 1.0,
        "energy_j": 0.5,
        "ap30": 0.9,
        "ap50": 0.8,
        "ap70": 0.7,
        "materialized_graph_features_sha256": _sha(graph),
    }
    terminal["actual_feedback_row_sha256"] = _sha(terminal)
    return {
        "stage5_terminal": terminal,
        "stage5_terminal_artifact": _artifact(
            root,
            f"{candidate_id}-raw-stage5-terminal",
            terminal,
            artifact_kind="stage5_terminal",
        ),
        "stage3_performance_artifact": {
            "artifact_kind": "stage3_performance_raw_v3",
            "raw_artifact_path": str(performance_path),
            "raw_artifact_sha256": performance_sha,
            "executor_state_path": str(performance_state_path),
            "executor_state_sha256": performance_state_sha,
            "job_manifest_path": str(performance_manifest_path),
            "job_manifest_sha256": performance_manifest_sha,
            "performance_jobs_path": str(performance_jobs_path),
            "performance_jobs_sha256": performance_jobs_sha,
        },
        "stage3_ap_artifact": {
            "artifact_kind": "stage3_ap_raw_v3",
            "raw_artifact_path": str(ap_path),
            "raw_artifact_sha256": ap_sha,
            "executor_state_path": str(ap_state_path),
            "executor_state_sha256": ap_state_sha,
            "job_manifest_path": str(ap_plan_path),
            "job_manifest_sha256": ap_plan_sha,
        },
        "actual_graph_features": graph,
    }


def _rewrite_authenticated_json(
    reference: dict[str, Any],
    *,
    path_field: str,
    sha_field: str,
    payload: Any,
    jsonl: bool = False,
) -> None:
    encoded = (
        (
            "\n".join(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for row in payload
            )
            + "\n"
        ).encode("utf-8")
        if jsonl
        else json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    Path(reference[path_field]).write_bytes(encoded)
    reference[sha_field] = hashlib.sha256(encoded).hexdigest()


def test_ap_terminal_rejects_real_plan_fingerprint_drift(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache
    )
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(
        candidate_id, request, plan, tmp_path
    )
    artifact = inputs["stage3_ap_artifact"]
    state = json.loads(
        Path(artifact["executor_state_path"]).read_text(encoding="utf-8")
    )
    state["plan_fingerprint"] = "0" * 64
    _rewrite_authenticated_json(
        artifact,
        path_field="executor_state_path",
        sha_field="executor_state_sha256",
        payload=[state],
        jsonl=True,
    )

    with pytest.raises(ValueError, match="plan fingerprint"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


@pytest.mark.parametrize(
    ("field_path", "mutated"),
    (
        (("manifest_job_id",), "other-candidate"),
        (("job_id",), "other-candidate"),
        (("row_id",), "other-candidate"),
        (("model",), "codriving"),
        (("genome",), [999, 32, 64, "fp16"]),
        (("width",), [999, 32, 64]),
        (("precision",), "int8"),
        (("q_mode",), "int8"),
        (("capability_profile_id",), "other-profile"),
        (("hardware_id",), "other-hardware"),
        (("dispatch_key",), "trt_engine"),
        (
            ("source_contract", "checkpoint_sha256"),
            "a" * 64,
        ),
        (("source_contract", "onnx_sha256"), "b" * 64),
        (("measurement_scope",), "other-scope"),
        (("input_protocol_sha256",), "c" * 64),
        (("batch_size",), 2),
        (("build_protocol_sha256",), "d" * 64),
        (("tuning_protocol_sha256",), "e" * 64),
        (("measurement_protocol_sha256",), "f" * 64),
        (("ap_protocol_sha256",), "0" * 64),
        (("runtime_contract_sha256",), "9" * 64),
    ),
)
def test_performance_manifest_rejects_present_exact_dimension_drift(
    field_path: tuple[str, ...],
    mutated: Any,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache
    )
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(
        candidate_id, request, plan, tmp_path
    )
    artifact = inputs["stage3_performance_artifact"]
    manifest = json.loads(
        Path(artifact["job_manifest_path"]).read_text(encoding="utf-8")
    )
    target = manifest["jobs"][0]
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = mutated
    _rewrite_authenticated_json(
        artifact,
        path_field="job_manifest_path",
        sha_field="job_manifest_sha256",
        payload=manifest,
    )

    with pytest.raises(
        ValueError, match="Stage3 performance manifest"
    ):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


def test_real_stage5_builder_shape_binds_distinct_execution_job_id(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache
    )
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(
        candidate_id, request, plan, tmp_path
    )
    performance_ref = inputs["stage3_performance_artifact"]
    performance_manifest = json.loads(
        Path(performance_ref["job_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    performance_job = json.loads(
        Path(performance_ref["performance_jobs_path"]).read_text(
            encoding="utf-8"
        )
    )
    ap_plan = json.loads(
        Path(inputs["stage3_ap_artifact"]["job_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )

    assert performance_manifest["jobs"][0]["job_id"] == candidate_id
    assert performance_job["manifest_job_id"] == candidate_id
    assert performance_job["job_id"] != candidate_id
    assert ap_plan["schema_version"] == "stage5_ap_plan_v2"
    assert ap_plan["performance_job_id"] == performance_job["job_id"]

    terminal = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        **inputs,
    )
    assert terminal["candidate_id"] == candidate_id


@pytest.mark.parametrize(
    ("field_path", "mutated"),
    (
        (("manifest_job_id",), "other-candidate"),
        (("model",), "codriving"),
        (("q_mode",), "int8"),
        (("dispatch_key",), "trt_engine"),
        (
            ("source_contract", "checkpoint_sha256"),
            "a" * 64,
        ),
        (("source_contract", "onnx_sha256"), "b" * 64),
    ),
)
def test_performance_execution_job_rejects_present_identity_drift(
    field_path: tuple[str, ...],
    mutated: Any,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache
    )
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(
        candidate_id, request, plan, tmp_path
    )
    artifact = inputs["stage3_performance_artifact"]
    performance_job = json.loads(
        Path(artifact["performance_jobs_path"]).read_text(encoding="utf-8")
    )
    target = performance_job
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = mutated
    _rewrite_authenticated_json(
        artifact,
        path_field="performance_jobs_path",
        sha_field="performance_jobs_sha256",
        payload=[performance_job],
        jsonl=True,
    )

    with pytest.raises(ValueError, match="Stage3 performance"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


@pytest.mark.parametrize(
    ("field_path", "mutated"),
    (
        (("manifest_job_id",), "other-candidate"),
        (("model",), "codriving"),
        (("width",), [999, 32, 64]),
        (("q",), "int8"),
        (("profile",), "other-profile"),
        (
            ("source_contract", "checkpoint_sha256"),
            "a" * 64,
        ),
        (("source_contract", "onnx_sha256"), "b" * 64),
        (("performance_job_id",), "other-execution-job"),
        (("performance_result_json",), "/other/result.json"),
        (("performance_result_sha256",), "c" * 64),
        (("runner_key",), "pyramid_tvm_int8_numeric_gate"),
    ),
)
def test_stage5_ap_plan_rejects_resigned_present_identity_swap(
    field_path: tuple[str, ...],
    mutated: Any,
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache
    )
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(
        candidate_id, request, plan, tmp_path
    )
    artifact = inputs["stage3_ap_artifact"]
    ap_plan = json.loads(
        Path(artifact["job_manifest_path"]).read_text(encoding="utf-8")
    )
    target = ap_plan
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = mutated
    _rewrite_authenticated_json(
        artifact,
        path_field="job_manifest_path",
        sha_field="job_manifest_sha256",
        payload=[ap_plan],
        jsonl=True,
    )
    ap_state = json.loads(
        Path(artifact["executor_state_path"]).read_text(encoding="utf-8")
    )
    ap_state["plan_fingerprint"] = stage3_ap_executor.plan_fingerprint(
        ap_plan, "full"
    )
    _rewrite_authenticated_json(
        artifact,
        path_field="executor_state_path",
        sha_field="executor_state_sha256",
        payload=[ap_state],
        jsonl=True,
    )

    with pytest.raises(ValueError, match="Stage3 AP"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


def test_selector_public_api_never_accepts_cache() -> None:
    parameters = inspect.signature(
        selector.select_actual_v3_pre_scan_round
    ).parameters
    assert all("cache" not in name.lower() for name in parameters)


def test_only_self_authenticated_logical_request_and_binding_can_reveal() -> None:
    frozen = _frozen()
    request = copy.deepcopy(frozen["logical_request"])
    binding = copy.deepcopy(frozen["selection_binding"])
    request["rows"][0]["q_mode"] = "int8"

    class ExplodingCache(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError(f"cache opened before authentication: {key}")

    with pytest.raises(ValueError, match="logical request SHA mismatch"):
        cache_v2.reveal_v2_cache_after_selection(
            request, binding, ExplodingCache(_empty_cache())
        )


def test_exact_key_binds_every_formal_dimension_including_runtime() -> None:
    frozen = _frozen()
    dimensions = frozen["selection_binding"]["selected_candidates"][0][
        "exact_key_dimensions"
    ]
    baseline = cache_v2.build_v2_exact_cache_key(dimensions)

    for field in cache_v2.EXACT_CACHE_KEY_DIMENSIONS:
        mutated = copy.deepcopy(dimensions)
        if field == "batch_size":
            mutated[field] = 2
        elif field == "genome":
            mutated[field] = [24, 48, 96, "fp16"]
        elif field == "q_mode":
            mutated[field] = "int8"
        elif field.endswith("_sha256"):
            mutated[field] = "f" * 64
        else:
            mutated[field] = str(mutated[field]) + "-changed"
        assert cache_v2.build_v2_exact_cache_key(mutated) != baseline


@pytest.mark.parametrize("hit_count", range(5))
def test_partial_hits_derive_zero_to_four_misses_with_four_logical_bindings(
    hit_count: int, tmp_path: Path
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    for candidate in binding["selected_candidates"][:hit_count]:
        reveal = cache_v2.reveal_v2_cache_after_selection(
            request, binding, cache
        )
        plan = adapter.build_miss_only_physical_plan(
            request, binding, reveal, cache_snapshot=cache
        )
        candidate_id = candidate["candidate_id"]
        terminal = adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **_raw_stage3_terminal_inputs(
                candidate_id, request, plan, tmp_path
            ),
        )
        cache = adapter.append_terminal_wrapper(cache, terminal)

    reveal = cache_v2.reveal_v2_cache_after_selection(request, binding, cache)
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )

    assert len(request["rows"]) == 4
    assert len(plan["logical_row_bindings"]) == 4
    assert len(plan["rows"]) == 4 - hit_count
    assert plan["physical_row_count"] == 4 - hit_count
    assert [row["row_id"] for row in plan["rows"]] == [
        row["candidate_id"]
        for row in plan["logical_row_bindings"]
        if row["disposition"] == "miss"
    ]


def test_terminal_wrapper_binds_all_lineage_and_append_is_immutable(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = _empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(request, binding, cache)
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    candidate_id = request["rows"][0]["row_id"]

    terminal = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        **_raw_stage3_terminal_inputs(candidate_id, request, plan, tmp_path),
    )
    appended = adapter.append_terminal_wrapper(cache, terminal)

    assert cache == _empty_cache()
    assert "logical_request_sha256" not in terminal
    assert "physical_request_sha256" not in terminal
    assert "selection_binding_sha256" not in terminal
    assert plan["cache_snapshot_sha256"] == reveal[
        "cache_snapshot_sha256"
    ]
    assert terminal["stage5_terminal_artifact"]["artifact_kind"] == "stage5_terminal"
    assert terminal["stage3_performance_artifact"]["artifact_kind"] == "stage3_performance_raw_v3"
    assert terminal["stage3_ap_artifact"]["artifact_kind"] == "stage3_ap_raw_v3"
    assert terminal["actual_graph_features_sha256"] == _sha(
        _raw_stage3_terminal_inputs(
            candidate_id, request, plan, tmp_path
        )["actual_graph_features"]
    )
    assert len(appended["entries"]) == 1
    assert appended["lineage"][-1]["evidence_sha256"] == terminal[
        "terminal_evidence_sha256"
    ]


def test_terminal_rejects_mutated_logical_row_binding_and_stage5_artifact(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]
    mutated_plan = copy.deepcopy(plan)
    mutated_plan["logical_row_bindings"][0]["disposition"] = "hit"
    with pytest.raises(ValueError, match="physical request authentication"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            mutated_plan,
            candidate_id=candidate_id,
            **_raw_stage3_terminal_inputs(
                candidate_id, request, plan, tmp_path
            ),
        )

    inputs = _raw_stage3_terminal_inputs(candidate_id, request, plan, tmp_path)
    inputs["stage5_terminal_artifact"] = _artifact(
        tmp_path,
        "wrong-stage5-terminal",
        {"candidate_id": candidate_id, "wrong": True},
        artifact_kind="stage5_terminal",
    )
    with pytest.raises(ValueError, match="Stage5 terminal artifact payload"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


def test_candidate_failure_consumes_budget_but_infra_and_evidence_retry_same_request() -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]

    candidate = adapter.finalize_actual_v3_failure(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        failure_class="candidate",
        reason="build_failure",
    )
    infra = adapter.finalize_actual_v3_failure(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        failure_class="infrastructure",
        reason="gpu_unavailable",
    )
    evidence = adapter.finalize_actual_v3_failure(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        failure_class="evidence",
        reason="terminal_evidence_sha_mismatch",
    )

    assert candidate["selected_event_budget_delta"] == 1
    assert candidate["physical_request_sha256"] == plan[
        "physical_request_sha256"
    ]
    assert len(candidate["failure_wrapper_sha256"]) == 64
    assert all(candidate[field] is None for field in cache_v2.METRIC_FIELDS)
    for retry in (infra, evidence):
        assert retry["selected_event_budget_delta"] == 0
        assert retry["retry_logical_request_sha256"] == request[
            "measurement_request_sha256"
        ]
        assert retry["logical_request_sha256"] == request[
            "measurement_request_sha256"
        ]
        assert retry["physical_request_sha256"] == plan[
            "physical_request_sha256"
        ]

    forged = copy.deepcopy(candidate)
    forged["selected_event_budget_delta"] = 0
    forged["consumes_selected_event_budget"] = False
    forged["failure_wrapper_sha256"] = _sha(
        {
            key: value
            for key, value in forged.items()
            if key != "failure_wrapper_sha256"
        }
    )
    with pytest.raises(ValueError, match="failure classification budget"):
        adapter.validate_actual_v3_failure(
            forged, request, binding, plan
        )


def test_formal_append_rejects_legacy_terminal_schema() -> None:
    adapter = _adapter()
    from framework.tests.test_stage7_core_cache_v2 import _terminal_success

    with pytest.raises(ValueError, match="formal actual-v3 terminal schema"):
        adapter.append_terminal_wrapper(_empty_cache(), _terminal_success())


@pytest.mark.parametrize(
    ("row_field", "exact_field", "mutated"),
    (
        ("genome", "genome", [999, 32, 64, "fp16"]),
        ("q_mode", "q_mode", "int8"),
        ("model", "model", "codriving"),
        (
            "capability_profile_id",
            "capability_profile_id",
            "wrong-profile",
        ),
        ("hardware_id", "hardware_id", "wrong-hardware"),
        ("dispatch_key", "dispatch_key", "wrong-backend"),
    ),
)
def test_bind_rejects_logical_row_exact_identity_mismatch(
    row_field: str, exact_field: str, mutated: Any
) -> None:
    adapter = _adapter()
    request = _request()
    dimensions = {
        row["row_id"]: _dimensions(row["row_id"], row["genome"])
        for row in request["rows"]
    }
    dimensions[request["rows"][0]["row_id"]][exact_field] = mutated

    with pytest.raises(ValueError, match="logical row exact identity mismatch"):
        adapter.bind_selector_output(
            {
                "acquisition": {
                    "selected_row_ids": [
                        row["row_id"] for row in request["rows"]
                    ]
                },
                "measurement_request": request,
            },
            exact_dimensions_by_candidate=dimensions,
        )


def test_complete_exact_dimensions_are_cross_bound_to_logical_row_sha() -> None:
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = copy.deepcopy(frozen["selection_binding"])
    selected = binding["selected_candidates"][0]
    selected["exact_key_dimensions"]["source_checkpoint_sha256"] = "f" * 64
    selected["exact_cache_key_sha256"] = cache_v2.build_v2_exact_cache_key(
        selected["exact_key_dimensions"]
    )
    binding["selection_binding_sha256"] = _sha(
        {
            key: value
            for key, value in binding.items()
            if key != "selection_binding_sha256"
        }
    )

    with pytest.raises(ValueError, match="logical/exact row binding SHA"):
        cache_v2.reveal_v2_cache_after_selection(
            request, binding, _empty_cache()
        )


def test_physical_plan_rejects_forged_hit_without_actual_v3_terminal() -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    reveal["entries"][0]["disposition"] = "hit"
    reveal["entries"][0]["hardware_measurement_required"] = False
    unsigned = {
        key: value
        for key, value in reveal.items()
        if key != "cache_reveal_sha256"
    }
    reveal["cache_reveal_sha256"] = _sha(unsigned)

    with pytest.raises(ValueError, match="cache snapshot/reveal"):
        adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())


def test_wrap_rejects_resigned_mutated_physical_row(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]
    mutated = copy.deepcopy(plan)
    mutated["rows"][0]["genome"] = [999, 32, 64, "fp16"]
    physical_row_sha = _sha(mutated["rows"][0])
    mutated["row_sha256"][candidate_id] = physical_row_sha
    mutated["logical_row_bindings"][0][
        "physical_row_sha256"
    ] = physical_row_sha
    physical_payload = {
        key: value
        for key, value in mutated.items()
        if key not in {"physical_request_sha256", "logical_row_bindings"}
    }
    mutated["physical_request_sha256"] = _sha(physical_payload)

    with pytest.raises(ValueError, match="physical row differs from logical"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            mutated,
            candidate_id=candidate_id,
            **_raw_stage3_terminal_inputs(
                candidate_id, request, plan, tmp_path
            ),
        )


def test_formal_output_cannot_enter_legacy_cache_append_path(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]
    terminal = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        **_raw_stage3_terminal_inputs(candidate_id, request, plan, tmp_path),
    )

    assert terminal["schema_version"] == (
        cache_v2.ACTUAL_V3_TERMINAL_EVIDENCE_SCHEMA
    )
    with pytest.raises(ValueError, match="legacy cache append"):
        cache_v2.append_v2_terminal_evidence(_empty_cache(), terminal)


def test_stage3_artifact_file_and_semantic_binding_are_enforced(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]
    inputs = _raw_stage3_terminal_inputs(candidate_id, request, plan, tmp_path)
    performance_ref = inputs["stage3_performance_artifact"]
    performance_state_path = Path(performance_ref["executor_state_path"])
    performance_state = json.loads(
        performance_state_path.read_text(encoding="utf-8")
    )
    performance_state["job_id"] = "other-candidate"
    encoded_state = (
        json.dumps(
            performance_state,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    performance_state_path.write_bytes(encoded_state)
    performance_ref["executor_state_sha256"] = hashlib.sha256(
        encoded_state
    ).hexdigest()
    with pytest.raises(
        ValueError, match="Stage3 performance executor terminal"
    ):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )

    inputs = _raw_stage3_terminal_inputs(candidate_id, request, plan, tmp_path)
    Path(inputs["stage3_ap_artifact"]["raw_artifact_path"]).write_text(
        '{"tampered":true}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="raw Stage3 artifact file SHA"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **inputs,
        )


def test_formal_cache_recovery_rejects_corrupt_existing_lineage(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    first_id = request["rows"][0]["row_id"]
    first = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=first_id,
        **_raw_stage3_terminal_inputs(first_id, request, plan, tmp_path),
    )
    cache = adapter.append_terminal_wrapper(_empty_cache(), first)
    cache["lineage"][0]["lineage_record_sha256"] = "0" * 64
    second_id = request["rows"][1]["row_id"]
    second = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=second_id,
        **_raw_stage3_terminal_inputs(second_id, request, plan, tmp_path),
    )

    with pytest.raises(ValueError, match="cache lineage"):
        adapter.append_terminal_wrapper(cache, second)


def test_miss_plan_rejects_reveal_from_different_cache_snapshot(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    empty = _empty_cache()
    empty_reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, empty
    )
    empty_plan = adapter.build_miss_only_physical_plan(
        request, binding, empty_reveal, cache_snapshot=empty
    )
    candidate_id = request["rows"][0]["row_id"]
    terminal = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        empty_plan,
        candidate_id=candidate_id,
        **_raw_stage3_terminal_inputs(candidate_id, request, empty_plan, tmp_path),
    )
    cache_with_truth = adapter.append_terminal_wrapper(empty, terminal)
    genuine_hit_reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, cache_with_truth
    )

    with pytest.raises(ValueError, match="cache snapshot"):
        adapter.build_miss_only_physical_plan(
            request,
            binding,
            genuine_hit_reveal,
            cache_snapshot=empty,
        )


def test_prior_round_intrinsic_truth_is_reused_for_same_exact_key_only(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    first = _frozen()
    first_request = first["logical_request"]
    first_binding = first["selection_binding"]
    empty = _empty_cache()
    first_reveal = cache_v2.reveal_v2_cache_after_selection(
        first_request, first_binding, empty
    )
    first_plan = adapter.build_miss_only_physical_plan(
        first_request,
        first_binding,
        first_reveal,
        cache_snapshot=empty,
    )
    candidate_id = first_request["rows"][0]["row_id"]
    terminal = adapter.wrap_actual_v3_terminal(
        first_request,
        first_binding,
        first_plan,
        candidate_id=candidate_id,
        **_raw_stage3_terminal_inputs(
            candidate_id, first_request, first_plan, tmp_path
        ),
    )
    cache = adapter.append_terminal_wrapper(empty, terminal)

    second_request = copy.deepcopy(first_request)
    second_request["round_index"] = 1
    second_payload = {
        key: value
        for key, value in second_request.items()
        if key != "measurement_request_sha256"
    }
    second_request["measurement_request_sha256"] = _sha(second_payload)
    dimensions = {
        row["candidate_id"]: row["exact_key_dimensions"]
        for row in first_binding["selected_candidates"]
    }
    second = adapter.bind_selector_output(
        {
            "acquisition": {
                "selected_row_ids": [
                    row["row_id"] for row in second_request["rows"]
                ]
            },
            "measurement_request": second_request,
        },
        exact_dimensions_by_candidate=dimensions,
    )
    second_reveal = cache_v2.reveal_v2_cache_after_selection(
        second["logical_request"],
        second["selection_binding"],
        cache,
    )
    second_plan = adapter.build_miss_only_physical_plan(
        second["logical_request"],
        second["selection_binding"],
        second_reveal,
        cache_snapshot=cache,
    )

    assert [row["disposition"] for row in second_reveal["entries"]] == [
        "hit",
        "miss",
        "miss",
        "miss",
    ]
    assert [row["row_id"] for row in second_plan["rows"]] == [
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert second_reveal["entries"][0]["reveal_use_binding_sha256"]


def test_failure_validator_rejects_resigned_arbitrary_context() -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]
    failure = adapter.finalize_actual_v3_failure(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        failure_class="candidate",
        reason="build_failure",
    )
    forged = copy.deepcopy(failure)
    forged["physical_request_sha256"] = "f" * 64
    forged["failure_wrapper_sha256"] = _sha(
        {
            key: value
            for key, value in forged.items()
            if key != "failure_wrapper_sha256"
        }
    )

    with pytest.raises(ValueError, match="failure context"):
        adapter.validate_actual_v3_failure(
            forged,
            request,
            binding,
            plan,
        )


def test_resigned_physical_row_permutation_is_rejected(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    permuted = copy.deepcopy(plan)
    permuted["rows"][0], permuted["rows"][1] = (
        permuted["rows"][1],
        permuted["rows"][0],
    )
    payload = {
        key: value
        for key, value in permuted.items()
        if key not in {"physical_request_sha256", "logical_row_bindings"}
    }
    permuted["physical_request_sha256"] = _sha(payload)

    with pytest.raises(ValueError, match="strict logical order"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            permuted,
            candidate_id=request["rows"][0]["row_id"],
            **_raw_stage3_terminal_inputs(
                request["rows"][0]["row_id"],
                request,
                plan,
                tmp_path,
            ),
        )


def test_raw_stage3_executor_lineage_is_accepted(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]

    terminal = adapter.wrap_actual_v3_terminal(
        request,
        binding,
        plan,
        candidate_id=candidate_id,
        **_raw_stage3_terminal_inputs(
            candidate_id, request, plan, tmp_path
        ),
    )

    assert terminal["stage3_performance_artifact"][
        "raw_artifact_path"
    ].endswith("-raw-performance.json")
    assert terminal["stage3_ap_artifact"]["executor_state_path"].endswith(
        "-ap-state.jsonl"
    )


def test_invented_normalized_stage3_payload_is_rejected(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    frozen = _frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, binding, _empty_cache()
    )
    plan = adapter.build_miss_only_physical_plan(request, binding, reveal, cache_snapshot=_empty_cache())
    candidate_id = request["rows"][0]["row_id"]

    with pytest.raises(ValueError, match="raw Stage3"):
        adapter.wrap_actual_v3_terminal(
            request,
            binding,
            plan,
            candidate_id=candidate_id,
            **_normalized_stage3_terminal_inputs(
                candidate_id, request, plan, tmp_path
            ),
        )
