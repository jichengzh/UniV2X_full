from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage5 import measurement_plan_v1
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import deployment_bundle_v2
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage7.core_ablation_v2 import canonical_sha256
from scripts.stage7_scheduler_requests_v2 import resolved_source_lock_sha256
from scripts import stage35_gold32_performance_plan_v1


def _module() -> Any:
    from framework.stage7 import physical_execution_v2

    return physical_execution_v2


@pytest.fixture(autouse=True)
def _validated_terminal_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adjacent adapter suite owns raw Stage3 artifact characterization."""

    def validate(value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or value.get("formal_actual_v3_terminal") is not True
            or not isinstance(value.get("candidate_id"), str)
            or not isinstance(value.get("exact_cache_key_sha256"), str)
            or not isinstance(value.get("exact_key_dimensions"), dict)
            or not isinstance(value.get("terminal_evidence_sha256"), str)
        ):
            raise ValueError("formal actual-v3 terminal fixture required")
        return copy.deepcopy(value)

    monkeypatch.setattr(actual_adapter, "validate_actual_v3_terminal_wrapper", validate)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_contract(root: Path, tag: str) -> dict[str, Any]:
    source_root = root / "sources" / tag
    calibration_root = source_root / "trt_npy"
    return {
        "checkpoint_path": str(source_root / "checkpoint.pth"),
        "checkpoint_sha256": None,
        "onnx_path": str(source_root / "model.onnx"),
        "onnx_sha256": None,
        "calibration_npz": str(source_root / "calibration.npz"),
        "calibration_summary": str(source_root / "summary.json"),
        "source_done_marker": str(source_root / "source.done"),
        "trt_calibration_dir": str(calibration_root),
        "calibration_root": str(calibration_root),
    }


def _request(
    root: Path,
    q_modes: tuple[str, str, str, str] = ("fp16", "fp16", "fp16", "fp16"),
) -> dict[str, Any]:
    if len(q_modes) != 4:
        raise ValueError("four fixture q_modes are required")
    rows = []
    for index, (width, q_mode) in enumerate(
        zip(
            ([16, 32, 64], [24, 48, 96], [32, 64, 128], [40, 80, 160]),
            q_modes,
        )
    ):
        tag = "x".join(map(str, width))
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": f"candidate-{index}",
            "manifest_job_id": f"candidate-{index}",
            "strategy_id": f"strategy-{index}",
            "model": "pyramid",
            "group_id": f"pyramid|{tag}",
            "width": list(width),
            "genome": [*width, q_mode],
            "q_mode": q_mode,
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
            "capability_digest": "b" * 64,
            "dispatch_key": "tvm_auto",
            "source_status": "planned",
            "source_contract": _source_contract(root, tag),
            "graph_features": {"conv_count": 51 + index},
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


def _write_source_evidence(
    request: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in request["rows"]:
        source = row["source_contract"]
        group = row["group_id"]
        for field, content in (
            ("checkpoint_path", b"checkpoint"),
            ("onnx_path", b"onnx"),
            ("calibration_npz", b"calibration"),
            ("calibration_summary", b'{"status":"success"}'),
        ):
            path = Path(source[field])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content + group.encode("utf-8"))
        trt_root = Path(source["trt_calibration_dir"])
        trt_root.mkdir(parents=True, exist_ok=True)
        (trt_root / "sample_000.npy").write_bytes(b"npy")
        evidence_path = Path(
            source["source_done_marker"].removesuffix(".done") + "_evidence.json"
        )
        evidence = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "group_id": group,
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
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        result[row["row_id"]] = str(evidence_path)
    return result


def _selection_binding(
    request: dict[str, Any], source_result: dict[str, Any]
) -> dict[str, Any]:
    resolved = {row["candidate_id"]: row for row in source_result["rows"]}
    dimensions = {}
    for row in request["rows"]:
        candidate_id = row["row_id"]
        source = resolved[candidate_id]
        dimensions[candidate_id] = {
            "candidate_id": candidate_id,
            "model": "pyramid",
            "capability_profile_id": row["capability_profile_id"],
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "measurement_scope": "formal_h800_latency_energy_ap",
            "input_protocol_sha256": "1" * 64,
            "batch_size": 1,
            "genome": copy.deepcopy(row["genome"]),
            "q_mode": row["q_mode"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "onnx_sha256": source["onnx_sha256"],
            "build_protocol_sha256": "4" * 64,
            "tuning_protocol_sha256": "5" * 64,
            "measurement_protocol_sha256": "6" * 64,
            "ap_protocol_sha256": "7" * 64,
            "runtime_contract_sha256": "8" * 64,
        }
    return actual_adapter.bind_selector_output(
        {
            "acquisition": {
                "selected_row_ids": [row["row_id"] for row in request["rows"]]
            },
            "measurement_request": request,
        },
        exact_dimensions_by_candidate=dimensions,
    )["selection_binding"]


def _reveal_and_physical(
    request: dict[str, Any],
    selection: dict[str, Any],
    miss_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = selection["selected_candidates"]
    miss_ids = {entry["candidate_id"] for entry in selected[-miss_count:]}
    if miss_count == 0:
        miss_ids = set()
    snapshot_sha = "9" * 64
    lineage_sha = "d" * 64
    entries = []
    for entry in selected:
        is_miss = entry["candidate_id"] in miss_ids
        terminal_sha = None if is_miss else "e" * 64
        use = {
            "candidate_id": entry["candidate_id"],
            "logical_request_sha256": request["measurement_request_sha256"],
            "selection_binding_sha256": selection["selection_binding_sha256"],
            "logical_row_sha256": entry["logical_row_sha256"],
            "logical_exact_binding_sha256": entry["logical_exact_binding_sha256"],
            "exact_cache_key_sha256": entry["exact_cache_key_sha256"],
            "terminal_evidence_sha256": terminal_sha,
            "disposition": "miss" if is_miss else "hit",
            "cache_snapshot_sha256": snapshot_sha,
            "lineage_head_sha256": lineage_sha,
        }
        reveal_entry = {
            "candidate_id": entry["candidate_id"],
            "logical_row_sha256": entry["logical_row_sha256"],
            "logical_exact_binding_sha256": entry["logical_exact_binding_sha256"],
            "exact_cache_key_sha256": entry["exact_cache_key_sha256"],
            "disposition": "miss" if is_miss else "hit",
            "selected_event_budget_delta": 1,
            "hardware_measurement_required": is_miss,
            "terminal_evidence_sha256": terminal_sha,
            "reveal_use_binding_sha256": _sha(use),
        }
        if not is_miss:
            reveal_entry["terminal_evidence"] = {
                "formal_actual_v3_terminal": True,
                "candidate_id": entry["candidate_id"],
                "exact_cache_key_sha256": entry["exact_cache_key_sha256"],
                "exact_key_dimensions": copy.deepcopy(entry["exact_key_dimensions"]),
                "terminal_evidence_sha256": terminal_sha,
            }
        entries.append(reveal_entry)
    reveal_payload = {
        "schema_version": "stage7_actual_v3_selected_cache_reveal_v2",
        "formal_actual_v3_cache_reveal": True,
        "logical_request_sha256": request["measurement_request_sha256"],
        "selection_binding_sha256": selection["selection_binding_sha256"],
        "cache_snapshot_sha256": snapshot_sha,
        "lineage_head_sha256": lineage_sha,
        "entries": entries,
        "selected_event_budget_delta": 4,
    }
    reveal = {**reveal_payload, "cache_reveal_sha256": _sha(reveal_payload)}
    logical_rows = {row["row_id"]: row for row in request["rows"]}
    physical_rows = [
        copy.deepcopy(logical_rows[entry["candidate_id"]])
        for entry in entries
        if entry["disposition"] == "miss"
    ]
    physical_sha = {row["row_id"]: _sha(row) for row in physical_rows}
    physical_payload = {
        "schema_version": "stage7_actual_v3_miss_only_physical_request_v2",
        "logical_request_sha256": request["measurement_request_sha256"],
        "selection_binding_sha256": selection["selection_binding_sha256"],
        "cache_snapshot_sha256": snapshot_sha,
        "lineage_head_sha256": lineage_sha,
        "cache_reveal_sha256": reveal["cache_reveal_sha256"],
        "logical_row_count": 4,
        "physical_row_count": miss_count,
        "row_sha256": physical_sha,
        "rows": physical_rows,
    }
    bindings = [
        {
            "logical_row_index": index,
            "candidate_id": selected_row["candidate_id"],
            "logical_row_sha256": selected_row["logical_row_sha256"],
            "logical_exact_binding_sha256": selected_row[
                "logical_exact_binding_sha256"
            ],
            "exact_cache_key_sha256": selected_row["exact_cache_key_sha256"],
            "disposition": reveal_row["disposition"],
            "physical_row_sha256": physical_sha.get(selected_row["candidate_id"]),
        }
        for index, (selected_row, reveal_row) in enumerate(zip(selected, entries))
    ]
    return reveal, {
        **physical_payload,
        "logical_row_bindings": bindings,
        "physical_request_sha256": _sha(physical_payload),
    }


def _admission(request: dict[str, Any], physical: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "stage7_actual_v3_executor_admission_v2",
        "admission_passed": True,
        "contract_sha256": "c" * 64,
        "logical_request_sha256": request["measurement_request_sha256"],
        "physical_request_sha256": physical["physical_request_sha256"],
        "logical_row_count": 4,
        "physical_row_count": physical["physical_row_count"],
        "execution_primitive": "existing_stage5_stage3_actual_feedback_v3",
        "gpu_jobs_launched": 0,
    }
    return {**payload, "admission_sha256": canonical_sha256(payload)}


def _deployment(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    repository = Path(__file__).resolve().parents[2]
    frozen = root / "frozen-repo"
    for relative in (
        *deployment_bundle_v2.DEFAULT_BUNDLE_FILES,
        *deployment_bundle_v2.DEFAULT_PRIMITIVE_FILES,
    ):
        source = repository / relative
        destination = frozen / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        destination.chmod(0o640)
    deployment_root = root / "formal-v2"
    verified = deployment_bundle_v2.build_deployment_bundle(
        v2_root=deployment_root,
        frozen_repo_root=frozen,
    )
    return deployment_root, frozen, verified


def _formal_inputs(
    tmp_path: Path,
    miss_count: int,
    q_modes: tuple[str, str, str, str] = ("fp16", "fp16", "fp16", "fp16"),
) -> dict[str, dict[str, Any]]:
    request = _request(tmp_path, q_modes=q_modes)
    source_plan = source_resolution.build_source_resolution_plan(request)
    evidence = _write_source_evidence(request)
    source_result = source_resolution.build_formal_source_resolution_result(
        source_plan, evidence_paths_by_candidate=evidence
    )
    selection = _selection_binding(request, source_result)
    reveal, physical = _reveal_and_physical(request, selection, miss_count)
    deployment_root, frozen_repo_root, deployment = _deployment(tmp_path)
    return {
        "logical_request": request,
        "selection_binding": selection,
        "cache_reveal": reveal,
        "physical_plan": physical,
        "source_plan": source_plan,
        "source_result": source_result,
        "executor_admission": _admission(request, physical),
        "deployment_manifest": deployment,
        "deployment_root": deployment_root,
        "frozen_repo_root": frozen_repo_root,
        "expected_release_sha256": deployment["deployment_release_sha256"],
        "expected_manifest_file_sha256": deployment[
            "deployment_manifest_file_sha256"
        ],
    }


def _resign_reveal_chain(inputs: dict[str, dict[str, Any]]) -> None:
    reveal_payload = {
        key: value
        for key, value in inputs["cache_reveal"].items()
        if key != "cache_reveal_sha256"
    }
    inputs["cache_reveal"]["cache_reveal_sha256"] = _sha(reveal_payload)
    inputs["physical_plan"]["cache_reveal_sha256"] = inputs["cache_reveal"][
        "cache_reveal_sha256"
    ]
    physical_payload = {
        key: value
        for key, value in inputs["physical_plan"].items()
        if key not in {"physical_request_sha256", "logical_row_bindings"}
    }
    inputs["physical_plan"]["physical_request_sha256"] = _sha(physical_payload)
    inputs["executor_admission"] = _admission(
        inputs["logical_request"], inputs["physical_plan"]
    )


@pytest.mark.parametrize("miss_count", range(5))
def test_projection_preserves_ordered_miss_rows_and_full_lineage(
    miss_count: int, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, miss_count)
    row_bytes = {
        row["row_id"]: json.dumps(
            row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        for row in inputs["physical_plan"]["rows"]
    }

    result = module.build_independent_projection(**inputs)

    if miss_count == 0:
        assert result["schema_version"] == (
            "stage7_actual_v3_empty_physical_terminal_v2"
        )
        assert result["rows"] == []
        assert result["lineage_inputs"] == []
        assert result["gpu_subprocess_count"] == 0
        return
    assert result["schema_version"] == ("stage5_independent_validation_request_v1")
    assert result["batch_size"] == miss_count
    assert result["independent_from_search_measurement"] is True
    assert [row["row_id"] for row in result["rows"]] == [
        row["row_id"] for row in inputs["physical_plan"]["rows"]
    ]
    assert set(result["row_sha256"]) == set(row_bytes)
    assert all(
        json.dumps(
            row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
        == row_bytes[row["row_id"]]
        for row in result["rows"]
    )
    lineage = result["stage7_projection_lineage"]
    assert (
        lineage["logical_request_sha256"]
        == inputs["logical_request"]["measurement_request_sha256"]
    )
    assert (
        lineage["selection_binding_sha256"]
        == inputs["selection_binding"]["selection_binding_sha256"]
    )
    assert (
        lineage["cache_reveal_sha256"] == inputs["cache_reveal"]["cache_reveal_sha256"]
    )
    assert (
        lineage["physical_request_sha256"]
        == inputs["physical_plan"]["physical_request_sha256"]
    )
    assert (
        lineage["source_resolution_plan_sha256"]
        == inputs["source_plan"]["source_resolution_plan_sha256"]
    )
    assert (
        lineage["source_resolution_result_sha256"]
        == inputs["source_result"]["source_resolution_result_sha256"]
    )
    assert lineage["resolved_source_lock_sha256"] == resolved_source_lock_sha256(
        inputs["physical_plan"], inputs["source_result"]
    )
    assert (
        lineage["executor_admission_sha256"]
        == inputs["executor_admission"]["admission_sha256"]
    )
    assert (
        lineage["deployment_bundle_sha256"]
        == inputs["deployment_manifest"]["deployment_bundle_sha256"]
    )
    assert [row["candidate_id"] for row in lineage["rows"]] == [
        row["row_id"] for row in result["rows"]
    ]


@pytest.mark.parametrize("miss_count", range(5))
def test_projection_and_performance_artifacts_never_include_cache_hits(
    miss_count: int, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, miss_count)
    projection = module.build_independent_projection(**inputs)
    planned = module.build_authenticated_performance_artifacts(
        projection=projection,
        source_plan=inputs["source_plan"],
        source_result=inputs["source_result"],
        quant_contract_paths={},
        remote_artifact_root=tmp_path / "performance",
        gpus=[0, 1, 2, 3],
    )
    miss_ids = {row["row_id"] for row in inputs["physical_plan"]["rows"]}
    hit_ids = {row["row_id"] for row in inputs["logical_request"]["rows"]} - miss_ids

    assert not hit_ids.intersection(planned["planned_candidate_ids"])
    assert planned["manifest_row_count"] == miss_count
    if miss_count == 0:
        assert planned["performance_jobs"] == []
        assert planned["planner_call_count"] == 0
    else:
        assert planned["manifest"]["row_count"] == miss_count
        assert (
            planned["manifest"]["source_request_sha256"]
            == projection["measurement_request_sha256"]
        )


@pytest.mark.parametrize(
    "drift",
    ("selection_sha", "physical_order", "source_result", "admission"),
)
def test_projection_fails_closed_on_resigned_binding_or_order_drift(
    drift: str, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 3)
    if drift == "selection_sha":
        inputs["selection_binding"]["selected_candidates"][0]["logical_row_sha256"] = (
            "0" * 64
        )
        unsigned = {
            key: value
            for key, value in inputs["selection_binding"].items()
            if key != "selection_binding_sha256"
        }
        inputs["selection_binding"]["selection_binding_sha256"] = _sha(unsigned)
    elif drift == "physical_order":
        inputs["physical_plan"]["rows"].reverse()
        payload = {
            key: value
            for key, value in inputs["physical_plan"].items()
            if key not in {"physical_request_sha256", "logical_row_bindings"}
        }
        inputs["physical_plan"]["physical_request_sha256"] = _sha(payload)
        inputs["executor_admission"] = _admission(
            inputs["logical_request"], inputs["physical_plan"]
        )
    elif drift == "source_result":
        inputs["source_result"]["rows"][-1]["onnx_sha256"] = "0" * 64
        payload = {
            key: value
            for key, value in inputs["source_result"].items()
            if key != "source_resolution_result_sha256"
        }
        inputs["source_result"]["source_resolution_result_sha256"] = _sha(payload)
    else:
        inputs["executor_admission"]["physical_row_count"] = 2
        payload = {
            key: value
            for key, value in inputs["executor_admission"].items()
            if key != "admission_sha256"
        }
        inputs["executor_admission"]["admission_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError):
        module.build_independent_projection(**inputs)


def test_direct_framework_planner_gets_authenticated_source_result_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 2)
    projection = module.build_independent_projection(**inputs)
    original_rows = copy.deepcopy(projection["rows"])
    observed: dict[str, Any] = {}
    real_planner = module.measurement_plan_v2.build_performance_plan

    def fake_planner(
        request: dict[str, Any],
        *,
        source_evidence_paths: dict[str, Path],
        quant_contract_paths: dict[str, Path],
        remote_artifact_root: Path,
        gpus: list[int],
    ) -> dict[str, Any]:
        observed.update(
            {
                "request": copy.deepcopy(request),
                "source_evidence_paths": copy.deepcopy(source_evidence_paths),
                "quant_contract_paths": copy.deepcopy(quant_contract_paths),
                "remote_artifact_root": remote_artifact_root,
                "gpus": copy.deepcopy(gpus),
            }
        )
        return real_planner(
            request,
            source_evidence_paths=source_evidence_paths,
            quant_contract_paths=quant_contract_paths,
            remote_artifact_root=remote_artifact_root,
            gpus=gpus,
        )

    monkeypatch.setattr(
        module.measurement_plan_v2, "build_performance_plan", fake_planner
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Stage5 planner CLI was observed"),
    )

    result = module.build_authenticated_performance_artifacts(
        projection=projection,
        source_plan=inputs["source_plan"],
        source_result=inputs["source_result"],
        quant_contract_paths={},
        remote_artifact_root=tmp_path / "performance",
        gpus=[3, 5],
    )

    result_rows = {row["candidate_id"]: row for row in inputs["source_result"]["rows"]}
    expected_paths = {
        row["group_id"]: Path(result_rows[row["row_id"]]["source_evidence_path"])
        for row in projection["rows"]
    }
    assert observed["source_evidence_paths"] == expected_paths
    assert observed["request"]["rows"] == original_rows
    assert projection["rows"] == original_rows
    assert result["manifest_row_count"] == 2
    assert result["manifest"]["row_count"] == 2
    assert result["source_binding_audit"]["source_evidence_paths"] == {
        group: str(path) for group, path in expected_paths.items()
    }


def test_performance_planning_rejects_source_result_path_drift_without_rewriting_rows(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 1)
    projection = module.build_independent_projection(**inputs)
    original_rows = copy.deepcopy(projection["rows"])
    inputs["source_result"]["rows"][-1]["source_evidence_path"] = str(
        tmp_path / "wrong.json"
    )
    payload = {
        key: value
        for key, value in inputs["source_result"].items()
        if key != "source_resolution_result_sha256"
    }
    inputs["source_result"]["source_resolution_result_sha256"] = _sha(payload)

    with pytest.raises(ValueError):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={},
            remote_artifact_root=tmp_path / "performance",
            gpus=[0],
        )

    assert projection["rows"] == original_rows


def test_projection_rejects_resigned_fabricated_hit_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 3)
    hit = inputs["cache_reveal"]["entries"][0]
    hit["terminal_evidence"] = {
        "terminal_evidence_sha256": hit["terminal_evidence_sha256"],
        "fabricated": True,
    }
    _resign_reveal_chain(inputs)
    monkeypatch.setattr(
        actual_adapter,
        "validate_actual_v3_terminal_wrapper",
        lambda _value: (_ for _ in ()).throw(
            ValueError("fabricated actual-v3 terminal")
        ),
    )

    with pytest.raises(ValueError, match="terminal"):
        module.build_independent_projection(**inputs)


@pytest.mark.parametrize(
    "field",
    (
        "candidate_id",
        "exact_cache_key_sha256",
        "exact_key_dimensions",
        "terminal_evidence_sha256",
    ),
)
def test_projection_cross_binds_validated_hit_terminal_to_selection(
    field: str, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 3)
    hit = inputs["cache_reveal"]["entries"][0]["terminal_evidence"]
    if field == "exact_key_dimensions":
        hit[field] = {
            **hit[field],
            "runtime_contract_sha256": "0" * 64,
        }
    else:
        hit[field] = "0" * 64
    _resign_reveal_chain(inputs)

    with pytest.raises(ValueError, match="terminal"):
        module.build_independent_projection(**inputs)


def test_projection_rejects_self_asserted_nonexistent_deployment_record(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 4)
    inputs["deployment_root"] = Path("/formal-v2")
    inputs["frozen_repo_root"] = Path("/home/jichengzhi/V2X")

    with pytest.raises(ValueError, match="deployment"):
        module.build_independent_projection(**inputs)


@pytest.mark.parametrize(
    "missing_pin",
    ("expected_release_sha256", "expected_manifest_file_sha256"),
)
def test_projection_requires_both_external_deployment_pins_before_admission(
    missing_pin: str, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 1)
    inputs.pop(missing_pin)

    with pytest.raises(ValueError, match="expected.*SHA256"):
        module.build_independent_projection(**inputs)


@pytest.mark.parametrize(
    "drifted_pin",
    ("expected_release_sha256", "expected_manifest_file_sha256"),
)
def test_projection_rejects_external_deployment_pin_mismatch(
    drifted_pin: str, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 1)
    inputs[drifted_pin] = "0" * 64

    with pytest.raises(ValueError, match="external.*pin|canonical deployment"):
        module.build_independent_projection(**inputs)


@pytest.mark.parametrize("artifact", ("manifest", "state", "code", "primitive"))
def test_projection_replays_canonical_bundle_and_rejects_disk_drift(
    artifact: str, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 4)
    root = inputs["deployment_root"]
    if artifact == "manifest":
        path = root / "deployment/deployment_manifest_v2.json"
    elif artifact == "state":
        path = root / "deployment/deployment_state_v2.json"
    elif artifact == "code":
        path = root / "deployment/code/framework/stage7/physical_execution_v2.py"
    else:
        path = (
            inputs["frozen_repo_root"] / deployment_bundle_v2.DEFAULT_PRIMITIVE_FILES[0]
        )
    path.write_bytes(path.read_bytes() + b"\n# authenticated-drift\n")

    with pytest.raises(ValueError, match="deployment"):
        module.build_independent_projection(**inputs)


def test_performance_artifacts_reject_wrong_execution_job_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 2)
    projection = module.build_independent_projection(**inputs)

    def wrong_jobs(request: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        return {
            "manifest": {
                "schema_version": "stage5_performance_manifest_v2",
                "source_request_schema": request["schema_version"],
                "source_request_sha256": request["measurement_request_sha256"],
                "row_count": 2,
                "jobs": copy.deepcopy(request["rows"]),
            },
            "performance_jobs": [
                {"manifest_job_id": "wrong-a"},
                {"manifest_job_id": "wrong-b"},
            ],
        }

    monkeypatch.setattr(
        module.measurement_plan_v2, "build_performance_plan", wrong_jobs
    )

    with pytest.raises(ValueError, match="job"):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={},
            remote_artifact_root=tmp_path / "performance",
            gpus=[0, 1],
        )


@pytest.mark.parametrize(
    "drift",
    (
        "wrong_id",
        "duplicate_id",
        "swapped_order",
        "manifest_disagreement",
        "runner_key",
        "assigned_gpu",
        "gpu_pool",
        "command",
        "source_contract",
        "calibration_root",
        "remote_artifact_root",
        "expected_result_json",
    ),
)
def test_performance_artifacts_cross_bind_every_execution_job(
    drift: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 2)
    projection = module.build_independent_projection(**inputs)
    real_planner = module.measurement_plan_v2.build_performance_plan

    def drifted_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        planned = real_planner(*args, **kwargs)
        jobs = planned["performance_jobs"]
        if drift == "wrong_id":
            jobs[0]["manifest_job_id"] = "wrong"
        elif drift == "duplicate_id":
            jobs[1]["manifest_job_id"] = jobs[0]["manifest_job_id"]
        elif drift == "swapped_order":
            jobs.reverse()
        elif drift == "manifest_disagreement":
            jobs[0]["group_id"] = "pyramid|not-the-manifest-group"
        elif drift == "runner_key":
            jobs[0]["runner_key"] = "tvm_int8"
        elif drift == "assigned_gpu":
            jobs[0]["assigned_gpu"] = 7
        elif drift == "gpu_pool":
            jobs[0]["gpu_pool"] = "7"
        elif drift == "command":
            jobs[0]["command"] = [*jobs[0]["command"], "--forged"]
        elif drift == "source_contract":
            jobs[0]["source_contract"] = {
                **jobs[0]["source_contract"],
                "onnx_sha256": "0" * 64,
            }
        elif drift == "calibration_root":
            jobs[0]["calibration_root"] = "/forged/calibration"
        elif drift == "remote_artifact_root":
            jobs[0]["remote_artifact_root"] = "/forged/output"
        else:
            jobs[0]["expected_result_json"] = "/forged/result.json"
        return planned

    monkeypatch.setattr(
        module.measurement_plan_v2, "build_performance_plan", drifted_plan
    )

    with pytest.raises(ValueError, match="job"):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={},
            remote_artifact_root=tmp_path / "performance",
            gpus=[0, 1],
        )


@pytest.mark.parametrize(
    "drift",
    (
        "onnx_path",
        "calibration_root",
        "group_id",
        "width",
        "genome",
        "source_evidence_path",
    ),
)
def test_performance_artifacts_reject_consistent_manifest_and_job_drift(
    drift: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 2)
    projection = module.build_independent_projection(**inputs)
    real_planner = module.measurement_plan_v2.build_performance_plan
    artifact_root = tmp_path / "performance"
    gpus = [0, 1]

    def drifted_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        planned = real_planner(*args, **kwargs)
        row = planned["manifest"]["jobs"][0]
        if drift in {"onnx_path", "calibration_root"}:
            row["source_contract"] = {
                **row["source_contract"],
                drift: f"/forged/{drift}",
            }
        elif drift == "group_id":
            row[drift] = "pyramid|forged"
        elif drift == "width":
            row[drift] = [1, 2, 3]
        elif drift == "genome":
            row[drift] = [1, 2, 3, "fp16"]
        else:
            row[drift] = "/forged/source-evidence.json"
        planned["performance_jobs"] = module._rebuild_performance_jobs(
            planned["manifest"]["jobs"],
            quant_contract_paths={},
            remote_artifact_root=artifact_root,
            gpus=gpus,
        )
        return planned

    monkeypatch.setattr(
        module.measurement_plan_v2, "build_performance_plan", drifted_plan
    )

    with pytest.raises(ValueError, match="manifest|job"):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={},
            remote_artifact_root=artifact_root,
            gpus=gpus,
        )


def test_performance_planning_rejects_runtime_job_builder_alias_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(tmp_path, 1)
    projection = module.build_independent_projection(**inputs)

    def forwarding_alias(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return stage35_gold32_performance_plan_v1._build_job(*args, **kwargs)

    assert measurement_plan_v1._build_job is (
        stage35_gold32_performance_plan_v1._build_job
    )
    monkeypatch.setattr(module.measurement_plan_v2, "_build_job", forwarding_alias)

    with pytest.raises(ValueError, match="import chain|job builder"):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={},
            remote_artifact_root=tmp_path / "performance",
            gpus=[0],
        )


def test_performance_artifacts_accept_independently_rebuilt_int8_prepared_row(
    tmp_path: Path,
) -> None:
    module = _module()
    inputs = _formal_inputs(
        tmp_path,
        1,
        q_modes=("fp16", "fp16", "fp16", "int8"),
    )
    projection = module.build_independent_projection(**inputs)
    quant_path = tmp_path / "quant" / "candidate-3.json"
    quant_path.parent.mkdir(parents=True)
    quant_path.write_text(
        json.dumps(
            {
                "schema": "stage3_tvm_int8_quant_contract_v3",
                "params": {"scale": 0.125},
            }
        ),
        encoding="utf-8",
    )

    result = module.build_authenticated_performance_artifacts(
        projection=projection,
        source_plan=inputs["source_plan"],
        source_result=inputs["source_result"],
        quant_contract_paths={"candidate-3": quant_path},
        remote_artifact_root=tmp_path / "performance",
        gpus=[0],
    )

    source_contract = result["manifest"]["jobs"][0]["source_contract"]
    assert source_contract["tensor_quant_params_json"] == str(quant_path)
    assert source_contract["tensor_quant_params_sha256"] == _file_sha(quant_path)
    command = result["performance_jobs"][0]["command"]
    assert command[-2:] == ["--tensor-quant-params-json", str(quant_path)]


@pytest.mark.parametrize("drift", ("remove", "replace", "reorder"))
def test_performance_artifacts_authenticate_int8_command_augmentation(
    drift: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    inputs = _formal_inputs(
        tmp_path,
        1,
        q_modes=("fp16", "fp16", "fp16", "int8"),
    )
    projection = module.build_independent_projection(**inputs)
    quant_path = tmp_path / "quant" / "candidate-3.json"
    quant_path.parent.mkdir(parents=True)
    quant_path.write_text(
        json.dumps(
            {
                "schema": "stage3_tvm_int8_quant_contract_v3",
                "params": {"scale": 0.125},
            }
        ),
        encoding="utf-8",
    )
    real_planner = module.measurement_plan_v2.build_performance_plan

    def drifted_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        planned = real_planner(*args, **kwargs)
        command = list(planned["performance_jobs"][0]["command"])
        flag_index = command.index("--tensor-quant-params-json")
        if drift == "remove":
            del command[flag_index : flag_index + 2]
        elif drift == "replace":
            command[flag_index + 1] = "/forged/quant.json"
        else:
            argument = command.pop(flag_index + 1)
            flag = command.pop(flag_index)
            command[0:0] = [flag, argument]
        planned["performance_jobs"][0]["command"] = command
        return planned

    monkeypatch.setattr(
        module.measurement_plan_v2, "build_performance_plan", drifted_plan
    )

    with pytest.raises(ValueError, match="job"):
        module.build_authenticated_performance_artifacts(
            projection=projection,
            source_plan=inputs["source_plan"],
            source_result=inputs["source_result"],
            quant_contract_paths={"candidate-3": quant_path},
            remote_artifact_root=tmp_path / "performance",
            gpus=[0],
        )
