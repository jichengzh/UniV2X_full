from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha


MATERIALIZER_PATH = (
    "/home/jichengzhi/V2X/scripts/stage5_materialize_round_sources_v1.sh"
)
MATERIALIZER_SHA256 = "d978e6287afc239c63471cadf729b37b4617bf46d1b8b8ea13cb4e2433bf4abe"


def _module() -> Any:
    return importlib.import_module("framework.stage7.source_resolution_v2")


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_contract(root: Path, group_tag: str) -> dict[str, Any]:
    source_root = root / group_tag
    return {
        "checkpoint_path": str(source_root / "checkpoint.pth"),
        "checkpoint_sha256": None,
        "onnx_path": str(source_root / "model.onnx"),
        "onnx_sha256": None,
        "calibration_npz": str(source_root / "calibration.npz"),
        "calibration_summary": str(source_root / "summary.json"),
        "source_done_marker": str(source_root / "source.done"),
        "trt_calibration_dir": str(source_root / "trt_npy"),
    }


def _request(root: Path) -> dict[str, Any]:
    row_specs = (
        ("16x32x64", [16, 32, 64], "fp16"),
        ("16x32x64", [16, 32, 64], "int8"),
        ("24x48x96", [24, 48, 96], "fp16"),
        ("24x48x96", [24, 48, 96], "int8"),
    )
    rows = []
    for index, (group_tag, width, q_mode) in enumerate(row_specs):
        candidate_id = (
            f"pyramid|{group_tag}|q={q_mode}|" "profile=h800-tvm-probe-conditioned-v3"
        )
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
            "strategy_id": f"candidate-{index}",
            "model": "pyramid",
            "group_id": f"pyramid|{group_tag}",
            "width": width,
            "genome": [*width, q_mode],
            "q_mode": q_mode,
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
            "capability_digest": "b" * 64,
            "dispatch_key": "tvm_auto",
            "source_status": "planned",
            "source_contract": _source_contract(root, group_tag),
            "graph_features": {"conv_count": 51},
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
        rows.append(row)
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": [
            "latency_ms",
            "energy_j",
            "ap30",
            "ap50",
            "ap70",
        ],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _resign_request(request: dict[str, Any]) -> None:
    request["row_sha256"] = {row["row_id"]: _sha(row) for row in request["rows"]}
    payload = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(payload)


def _write_ready_evidence(
    root: Path,
    request: dict[str, Any],
) -> dict[str, str]:
    evidence_by_group: dict[str, str] = {}
    for row in request["rows"]:
        group_id = row["group_id"]
        if group_id in evidence_by_group:
            continue
        source = row["source_contract"]
        artifacts = {
            "checkpoint_path": b"checkpoint",
            "onnx_path": b"onnx",
            "calibration_npz": b"calibration",
            "calibration_summary": b'{"status":"success"}',
        }
        for field, content in artifacts.items():
            path = Path(source[field])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content + group_id.encode("utf-8"))
        evidence_path = Path(
            str(source["source_done_marker"]).removesuffix(".done") + "_evidence.json"
        )
        evidence = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "group_id": group_id,
            "model": row["model"],
            "width": "x".join(map(str, row["width"])),
            "source_plan_sha256": row["source_evidence_sha256"],
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_sha256": _file_sha(Path(source["checkpoint_path"])),
            "onnx_path": source["onnx_path"],
            "onnx_sha256": _file_sha(Path(source["onnx_path"])),
            "calibration_path": source["calibration_npz"],
            "calibration_sha256": _file_sha(Path(source["calibration_npz"])),
            "calibration_summary_path": source["calibration_summary"],
            "calibration_summary_sha256": _file_sha(
                Path(source["calibration_summary"])
            ),
            "status": "ready",
        }
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        evidence_by_group[group_id] = str(evidence_path)
    return {
        row["row_id"]: evidence_by_group[row["group_id"]] for row in request["rows"]
    }


def _formal(
    root: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = _module()
    request = _request(root)
    plan = module.build_source_resolution_plan(request)
    evidence_paths = _write_ready_evidence(root, request)
    result = module.build_formal_source_resolution_result(
        plan,
        evidence_paths_by_candidate=evidence_paths,
    )
    return module, request, plan, result


def _resign_plan(plan: dict[str, Any]) -> None:
    payload = {
        key: value
        for key, value in plan.items()
        if key != "source_resolution_plan_sha256"
    }
    plan["source_resolution_plan_sha256"] = _sha(payload)


def _resign_result(result: dict[str, Any], sha_field: str) -> None:
    payload = {key: value for key, value in result.items() if key != sha_field}
    result[sha_field] = _sha(payload)


def test_plan_binds_authenticated_request_and_allows_unresolved_source_sha(
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)

    plan = module.build_source_resolution_plan(request)

    assert plan["schema_version"] == "stage7_source_resolution_plan_v2"
    assert plan["status"] == "frozen"
    assert plan["logical_request_sha256"] == request["measurement_request_sha256"]
    assert plan["ordered_row_ids"] == [row["row_id"] for row in request["rows"]]
    assert plan["materializer"] == {
        "path": MATERIALIZER_PATH,
        "sha256": MATERIALIZER_SHA256,
    }
    assert plan["row_count"] == 4
    assert all(row["checkpoint_sha256"] is None for row in plan["rows"])
    assert all(row["onnx_sha256"] is None for row in plan["rows"])
    assert len({row["source_key_sha256"] for row in plan["rows"]}) == 4
    assert module.validate_source_resolution_plan(plan, logical_request=request) == plan


def test_plan_rejects_request_authentication_and_source_plan_drift(
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    request["rows"][0]["width"][0] = 999
    with pytest.raises(ValueError):
        module.build_source_resolution_plan(request)

    request = _request(tmp_path)
    request["rows"][0]["source_evidence_sha256"] = "c" * 64
    _resign_request(request)
    with pytest.raises(ValueError, match="source plan"):
        module.build_source_resolution_plan(request)


@pytest.mark.parametrize(
    "forbidden_field",
    (
        "latency_ms",
        "energy_j",
        "ap50",
        "terminal_status",
        "cache_hit",
        "failure_reason",
        "objective",
        "objectives",
        "cache",
    ),
)
def test_plan_recursively_rejects_objective_cache_and_terminal_labels(
    forbidden_field: str,
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    request["rows"][0]["source_contract"]["nested"] = {forbidden_field: 1.0}
    request["rows"][0]["source_evidence_sha256"] = _source_plan_sha(request["rows"][0])
    _resign_request(request)

    with pytest.raises(ValueError, match="forbidden source-resolution"):
        module.build_source_resolution_plan(request)


def test_plan_rejects_placeholder_source_sha_and_resigned_row_reordering(
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    request["rows"][0]["source_evidence_sha256"] = "0" * 64
    _resign_request(request)
    with pytest.raises(ValueError, match="placeholder"):
        module.build_source_resolution_plan(request)

    request = _request(tmp_path)
    plan = module.build_source_resolution_plan(request)
    plan["rows"][0], plan["rows"][1] = plan["rows"][1], plan["rows"][0]
    _resign_plan(plan)
    with pytest.raises(ValueError, match="ordered"):
        module.validate_source_resolution_plan(plan, logical_request=request)


def test_formal_result_reads_real_evidence_and_closes_atomic_four_rows(
    tmp_path: Path,
) -> None:
    module, request, plan, result = _formal(tmp_path)

    assert result["schema_version"] == "stage7_source_resolution_result_v2"
    assert result["status"] == "all_ready"
    assert result["synthetic_nonfinal"] is False
    assert result["eligible_for_exact_cache_reveal"] is True
    assert result["eligible_for_cache_append"] is False
    assert result["row_count"] == 4
    assert [row["candidate_id"] for row in result["rows"]] == plan["ordered_row_ids"]
    assert (
        result["rows"][0]["source_evidence_path"]
        == result["rows"][1]["source_evidence_path"]
    )
    assert (
        result["rows"][0]["resolved_source_sha256"]
        != result["rows"][1]["resolved_source_sha256"]
    )
    assert module.validate_formal_source_resolution_result(result, plan) == result
    assert request["rows"][0]["q_mode"] != request["rows"][1]["q_mode"]


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("status", "pending"),
        ("group_id", "pyramid|999x999x999"),
        ("model", "codriving"),
        ("width", "999x999x999"),
        ("source_plan_sha256", "c" * 64),
        ("checkpoint_sha256", "0" * 64),
    ),
)
def test_formal_result_rejects_evidence_contract_drift(
    field: str,
    mutated: Any,
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    plan = module.build_source_resolution_plan(request)
    evidence_paths = _write_ready_evidence(tmp_path, request)
    evidence_path = Path(evidence_paths[request["rows"][0]["row_id"]])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[field] = mutated
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError):
        module.build_formal_source_resolution_result(
            plan,
            evidence_paths_by_candidate=evidence_paths,
        )


def test_formal_result_rejects_file_sha_mismatch_and_path_escape(
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    plan = module.build_source_resolution_plan(request)
    evidence_paths = _write_ready_evidence(tmp_path, request)
    first_id = request["rows"][0]["row_id"]
    Path(request["rows"][0]["source_contract"]["onnx_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.build_formal_source_resolution_result(
            plan,
            evidence_paths_by_candidate=evidence_paths,
        )

    request = _request(tmp_path / "escape-case")
    plan = module.build_source_resolution_plan(request)
    evidence_paths = _write_ready_evidence(tmp_path / "escape-case", request)
    outside = tmp_path / "outside-checkpoint.pth"
    outside.write_bytes(b"outside")
    evidence_path = Path(evidence_paths[request["rows"][0]["row_id"]])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["checkpoint_path"] = str(outside)
    evidence["checkpoint_sha256"] = _file_sha(outside)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        module.build_formal_source_resolution_result(
            plan,
            evidence_paths_by_candidate=evidence_paths,
        )


def test_formal_result_binds_non_null_planned_checkpoint_and_onnx_sha(
    tmp_path: Path,
) -> None:
    module = _module()
    request = _request(tmp_path)
    plan = module.build_source_resolution_plan(request)
    evidence_paths = _write_ready_evidence(tmp_path, request)
    plan["rows"][0]["checkpoint_sha256"] = "12" * 32
    plan["rows"][0]["onnx_sha256"] = "34" * 32
    _resign_plan(plan)

    with pytest.raises(ValueError, match="planned source SHA"):
        module.build_formal_source_resolution_result(
            plan,
            evidence_paths_by_candidate=evidence_paths,
        )


@pytest.mark.parametrize("mutation", ("reorder", "partial", "duplicate"))
def test_formal_result_rejects_resigned_non_atomic_rows(
    mutation: str,
    tmp_path: Path,
) -> None:
    module, _, plan, result = _formal(tmp_path)
    if mutation == "reorder":
        result["rows"][0], result["rows"][1] = (
            result["rows"][1],
            result["rows"][0],
        )
    elif mutation == "partial":
        result["rows"].pop()
        result["row_count"] = 3
    else:
        result["rows"][1] = copy.deepcopy(result["rows"][0])
    _resign_result(result, "source_resolution_result_sha256")

    with pytest.raises(ValueError, match="atomic|ordered|duplicate"):
        module.validate_formal_source_resolution_result(result, plan)


def test_retry_result_preserves_identity_consumes_zero_and_reveals_nothing(
    tmp_path: Path,
) -> None:
    module = _module()
    plan = module.build_source_resolution_plan(_request(tmp_path))

    retry = module.build_source_resolution_retry_result(
        plan,
        retry_candidate_ids=[plan["ordered_row_ids"][1]],
        reason_code="evidence_unavailable",
    )

    assert retry["schema_version"] == "stage7_source_resolution_retry_v2"
    assert retry["status"] == "retry_required"
    assert retry["selected_event_budget_delta"] == 0
    assert retry["partial_reveal_allowed"] is False
    assert retry["eligible_for_exact_cache_reveal"] is False
    assert retry["ordered_row_ids"] == plan["ordered_row_ids"]
    assert module.validate_source_resolution_retry_result(retry, plan) == retry


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("logical_request_sha256", "c" * 64),
        ("source_resolution_plan_sha256", "d" * 64),
        ("selected_event_budget_delta", 1),
        ("partial_reveal_allowed", True),
        ("eligible_for_exact_cache_reveal", True),
    ),
)
def test_retry_result_rejects_resigned_contract_drift(
    field: str,
    mutated: Any,
    tmp_path: Path,
) -> None:
    module = _module()
    plan = module.build_source_resolution_plan(_request(tmp_path))
    retry = module.build_source_resolution_retry_result(
        plan,
        retry_candidate_ids=[plan["ordered_row_ids"][0]],
        reason_code="source_interrupted",
    )
    retry[field] = mutated
    _resign_result(retry, "source_resolution_retry_sha256")

    with pytest.raises(ValueError):
        module.validate_source_resolution_retry_result(retry, plan)


def test_synthetic_dryrun_is_isolated_and_formal_validator_rejects_it(
    tmp_path: Path,
) -> None:
    module = _module()
    plan = module.build_source_resolution_plan(_request(tmp_path))

    synthetic = module.build_synthetic_dryrun_source_resolution_result(plan)

    assert synthetic["schema_version"] == "stage7_source_resolution_synthetic_dryrun_v2"
    assert synthetic["synthetic_nonfinal"] is True
    assert synthetic["actual_v3_hardware_evidence"] is False
    assert synthetic["eligible_for_protocol_exact_reveal"] is True
    assert synthetic["eligible_for_exact_cache_reveal"] is False
    assert synthetic["eligible_for_cache_append"] is False
    assert synthetic["eligible_for_finalization"] is False
    assert (
        module.validate_synthetic_dryrun_source_resolution_result(synthetic, plan)
        == synthetic
    )
    with pytest.raises(ValueError, match="formal"):
        module.validate_formal_source_resolution_result(synthetic, plan)


def test_synthetic_dryrun_rejects_resigned_promotion_to_final_truth(
    tmp_path: Path,
) -> None:
    module = _module()
    plan = module.build_source_resolution_plan(_request(tmp_path))
    synthetic = module.build_synthetic_dryrun_source_resolution_result(plan)
    synthetic["eligible_for_cache_append"] = True
    _resign_result(synthetic, "synthetic_dryrun_result_sha256")

    with pytest.raises(ValueError):
        module.validate_synthetic_dryrun_source_resolution_result(synthetic, plan)


def test_public_api_surface_is_explicit() -> None:
    module = _module()
    assert set(module.__all__) == {
        "build_source_resolution_plan",
        "validate_source_resolution_plan",
        "build_formal_source_resolution_result",
        "validate_formal_source_resolution_result",
        "build_source_resolution_retry_result",
        "validate_source_resolution_retry_result",
        "build_synthetic_dryrun_source_resolution_result",
        "validate_synthetic_dryrun_source_resolution_result",
    }
