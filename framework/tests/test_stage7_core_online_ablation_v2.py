from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from framework.stage7.core_ablation_v2 import CORE_VARIANTS
from framework.stage7.online_component_ablation_v1 import SEEDS
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage5.measurement_plan_v1 import _source_plan_sha
from scripts import stage7_core_online_ablation_v2 as online


EXTERNAL_PINS = {
    "expected_release_sha256": "a" * 64,
    "expected_manifest_file_sha256": "b" * 64,
}


def test_round_controller_state_covers_each_canonical_transition(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "round"
    directory.mkdir()
    assert online.round_controller_state(directory) == "UNINITIALIZED"

    for filename, state in (
        ("logical_request.json", None),
        ("selection_binding.json", "SELECTED_FROZEN"),
        ("source_resolution_plan.json", "SOURCE_PLAN_FROZEN"),
        ("source_resolution_result.json", "SOURCE_READY"),
        ("exact_selection_binding.json", "EXACT_BINDING_FROZEN"),
        ("cache_reveal.json", "CACHE_REVEALED"),
    ):
        (directory / filename).write_text("{}\n")
        if state is not None:
            assert online.round_controller_state(directory) == state


@pytest.mark.parametrize(
    "filenames",
    (
        ("cache_reveal.json",),
        ("exact_selection_binding.json",),
        ("source_resolution_result.json",),
        ("source_resolution_plan.json",),
        ("logical_request.json",),
        ("selection_binding.json",),
    ),
)
def test_round_controller_state_rejects_incomplete_lineage(
    filenames: tuple[str, ...], tmp_path: Path
) -> None:
    directory = tmp_path / "round"
    directory.mkdir()
    for filename in filenames:
        (directory / filename).write_text("{}\n")

    with pytest.raises(ValueError, match="lineage is incomplete"):
        online.round_controller_state(directory)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _request(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    for index in range(4):
        candidate_id = f"candidate-{index}"
        graph = {
            "group_id": f"pyramid|{16 + index}x32x64",
            "model": "pyramid",
            "width": [16 + index, 32, 64],
            "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
        }
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
            "group_id": graph["group_id"],
            "model": "pyramid",
            "width": graph["width"],
            "genome": [*graph["width"], "fp16"],
            "q_mode": "fp16",
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-auto-v1",
            "dispatch_key": "tvm_auto",
            "source_evidence_sha256": "b" * 64,
            "graph_features": graph,
        }
        source = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "status": "ready",
            "group_id": row["group_id"],
            "model": row["model"],
            "width": "x".join(map(str, row["width"])),
            "source_plan_sha256": row["source_evidence_sha256"],
            "onnx_path": str(root / f"{candidate_id}.onnx"),
            "onnx_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
        }
        (root / f"{candidate_id}.onnx").write_bytes(candidate_id.encode())
        source_path = root / f"{candidate_id}-source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        rows.append(row)
        feedback.append(
            {
                **copy.deepcopy(row),
                "terminal_status": "measured_success_gold",
                "measurement_request_row_sha256": "",
                "latency_ms": 1.0 + index,
                "energy_j": 0.1 + index,
                "ap30": 0.9,
                "ap50": 0.8,
                "ap70": 0.7,
                "materialized_source_evidence_path": str(source_path),
                "materialized_source_evidence_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
            }
        )
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
    request = {**payload, "measurement_request_sha256": _sha(payload)}
    for item in feedback:
        item["measurement_request_row_sha256"] = request["row_sha256"][item["row_id"]]
    return request, feedback


def _physical_terminal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": "stage7_actual_v3_physical_terminal_batch_v2",
        "rows": copy.deepcopy(rows),
        "lineage_inputs": [
            {
                "candidate_id": row["row_id"],
                "stage3_performance_artifact": {},
                "stage3_ap_artifact": {},
            }
            for row in rows
            if row.get("terminal_status") == "measured_success_gold"
        ],
        "failures": [],
        "projection_artifact": {
            "artifact_kind": "stage5_independent_projection",
            "measurement_request_sha256": "1" * 64,
            "projection_payload_sha256": "2" * 64,
            "deployment_bundle_sha256": "4" * 64,
        },
        "performance_artifacts_sha256": "3" * 64,
        "execution_attempt": {
            "attempt_id": "attempt_000",
            "deployment_bundle_sha256": "4" * 64,
            "primitive_sha256": {"stage3_finalizer": "5" * 64},
        },
        "barrier_release_allowed": True,
        "retry_logical_request_sha256": None,
    }
    attempt = payload["execution_attempt"]
    attempt["execution_attempt_sha256"] = _sha(attempt)
    return {**payload, "physical_terminal_batch_sha256": _sha(payload)}


def _extract(path: Path) -> dict[str, Any]:
    return {
        "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conv_count": 21,
    }


def _source_selection(root: Path) -> dict[str, Any]:
    request, _ = _request(root)
    for index, row in enumerate(request["rows"]):
        source_root = root / f"source-{index}"
        row["capability_digest"] = "d" * 64
        row["source_status"] = "planned"
        row["source_contract"] = {
            "checkpoint_path": str(source_root / "checkpoint.pth"),
            "checkpoint_sha256": None,
            "onnx_path": str(source_root / "model.onnx"),
            "onnx_sha256": None,
            "calibration_npz": str(source_root / "calibration.npz"),
            "calibration_summary": str(source_root / "summary.json"),
            "source_done_marker": str(source_root / "source.done"),
            "trt_calibration_dir": str(source_root / "trt_npy"),
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    unsigned["row_sha256"] = {row["row_id"]: _sha(row) for row in unsigned["rows"]}
    request = {**unsigned, "measurement_request_sha256": _sha(unsigned)}
    return {
        "measurement_request": request,
        "acquisition": {"selected_row_ids": [row["row_id"] for row in request["rows"]]},
    }


def _write_source_result(
    request: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    logical_request = request.get("measurement_request", request)
    evidence_paths: dict[str, str] = {}
    for row in logical_request["rows"]:
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
            path.write_bytes(content + row["row_id"].encode("utf-8"))
        evidence_path = Path(
            str(source["source_done_marker"]).removesuffix(".done") + "_evidence.json"
        )
        evidence = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "group_id": row["group_id"],
            "model": row["model"],
            "width": "x".join(map(str, row["width"])),
            "source_plan_sha256": row["source_evidence_sha256"],
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_sha256": hashlib.sha256(
                Path(source["checkpoint_path"]).read_bytes()
            ).hexdigest(),
            "onnx_path": source["onnx_path"],
            "onnx_sha256": hashlib.sha256(
                Path(source["onnx_path"]).read_bytes()
            ).hexdigest(),
            "calibration_path": source["calibration_npz"],
            "calibration_sha256": hashlib.sha256(
                Path(source["calibration_npz"]).read_bytes()
            ).hexdigest(),
            "calibration_summary_path": source["calibration_summary"],
            "calibration_summary_sha256": hashlib.sha256(
                Path(source["calibration_summary"]).read_bytes()
            ).hexdigest(),
            "status": "ready",
        }
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        evidence_paths[row["row_id"]] = str(evidence_path)
    return source_resolution.build_formal_source_resolution_result(
        plan,
        evidence_paths_by_candidate=evidence_paths,
    )


def _phase2_validated_root(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scope_path = root / "scope.json"
    scope_path.write_text(
        json.dumps(
            {
                "measurement_scope": "formal_h800_latency_energy_ap",
                "inference_batch": 1,
            }
        ),
        encoding="utf-8",
    )
    scope_record = {
        "path": str(scope_path.resolve()),
        "sha256": hashlib.sha256(scope_path.read_bytes()).hexdigest(),
    }
    measurement = {
        "exact_key_protocol_sha256": {
            "build_protocol_sha256": "4" * 64,
            "tuning_protocol_sha256": "5" * 64,
            "measurement_protocol_sha256": "6" * 64,
            "ap_protocol_sha256": "7" * 64,
            "runtime_contract_sha256": "8" * 64,
        }
    }
    return (
        {
            "root": root,
            "contract": {
                "contract_sha256": "c" * 64,
                "immutable_inputs": {"scope_input_batch": scope_record},
            },
            "pre_scan_pool": [{} for _ in range(686)],
        },
        {
            "initial_rows": [],
            "initial_graph_features": [],
            "capability_profiles": [{"profile_id": "p"}],
            "closure": {},
            "measurement_contract": measurement,
            "coldstart_freeze_binding": {
                "path": "framework/stage5/single_target_search_v2.py",
                "sha256": "f" * 64,
            },
        },
    )


def _install_phase2_selector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Path,
    selection: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated, inputs = _phase2_validated_root(root)
    monkeypatch.setattr(
        online,
        "validate_formal_v2_root",
        lambda *_args, **_kwargs: copy.deepcopy(validated),
    )
    monkeypatch.setattr(
        online,
        "_prior_selection_state",
        lambda *_args, **_kwargs: {
            "prior_selected_ids": [],
            "promoted_feedback": [],
            "a2_frozen": None,
        },
        raising=False,
    )
    monkeypatch.setattr(
        online,
        "_selector_inputs",
        lambda _validated: copy.deepcopy(inputs),
    )
    monkeypatch.setattr(
        online.search_policy, "build_stage7_task", lambda _profile: object()
    )
    monkeypatch.setattr(
        online.actual_selector,
        "select_actual_v3_pre_scan_round",
        lambda **_kwargs: SimpleNamespace(
            selection=copy.deepcopy(selection),
            audit={"schema_version": "audit"},
        ),
    )
    return validated, inputs


def _resign_source_result(result: dict[str, Any]) -> None:
    field = next(key for key in result if key.endswith("_result_sha256"))
    unsigned = {key: value for key, value in result.items() if key != field}
    result[field] = _sha(unsigned)


def test_immutable_write_validates_full_contract_before_every_write(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def validate() -> dict[str, Any]:
        calls.append("validated")
        return {"contract_sha256": "c" * 64}

    target = tmp_path / "receipt.json"
    online.write_immutable_json(target, {"ok": True}, validate=validate)
    online.write_immutable_json(target, {"ok": True}, validate=validate)
    assert calls == ["validated", "validated"]
    with pytest.raises(ValueError, match="overwrite"):
        online.write_immutable_json(target, {"ok": False}, validate=validate)
    assert calls == ["validated", "validated", "validated"]


def test_merge_promote_and_finalize_uses_real_v3_atomic_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, feedback = _request(tmp_path)
    request_path = tmp_path / "logical_request.json"
    feedback_path = tmp_path / "historical_feedback.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    calls: list[str] = []
    real_promote = online.stage5_promote.promote_feedback_batch
    real_finalize = online.stage5_search.finalize_atomic_batch

    def promote(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("promote")
        return real_promote(*args, **kwargs)

    def finalize(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("finalize")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(online.stage5_promote, "promote_feedback_batch", promote)
    monkeypatch.setattr(online.stage5_search, "finalize_atomic_batch", finalize)
    result = online.promote_and_finalize_atomic_barrier(
        request_path=request_path,
        historical_feedback_path=feedback_path,
        extractor=_extract,
    )

    assert calls == ["promote", "finalize"]
    assert result["barrier"]["feedback_released"] is True
    assert result["barrier"]["budget_consumed"] == 4
    assert result["promotion"]["audit"]["promoted_row_count"] == 4
    assert result["promotion"]["audit"]["silent_surrogate_fallback_count"] == 0


def test_merge_feedback_is_logical_order_and_requires_all_four(
    tmp_path: Path,
) -> None:
    request, feedback = _request(tmp_path)
    reversed_rows = list(reversed(feedback))
    merged = online.merge_feedback_in_logical_order(request, reversed_rows)
    assert [row["row_id"] for row in merged] == [
        row["row_id"] for row in request["rows"]
    ]
    with pytest.raises(ValueError, match="all four"):
        online.merge_feedback_in_logical_order(request, feedback[:3])


def test_request_retry_is_immutable_and_next_round_requires_committed_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = tmp_path / "round_00"
    request, _ = _request(tmp_path)
    online.freeze_request_artifacts(
        round_dir,
        {
            "logical_request": request,
            "selector_audit": {"schema_version": "audit"},
            "selection_binding": {"schema_version": "binding"},
            "cache_snapshot": {
                "schema_version": cache_v2.CACHE_SCHEMA,
                "entries": {},
                "lineage": [],
            },
            "cache_reveal": {"schema_version": "reveal"},
            "physical_plan": {
                "schema_version": "stage7_actual_v3_miss_only_physical_request_v2"
            },
            "executor_admission": {"admission_passed": True},
        },
        validate=lambda: {},
    )
    first = (round_dir / "logical_request.json").read_bytes()
    online.freeze_request_artifacts(
        round_dir,
        {
            "logical_request": request,
            "selector_audit": {"schema_version": "audit"},
            "selection_binding": {"schema_version": "binding"},
            "cache_snapshot": {
                "schema_version": cache_v2.CACHE_SCHEMA,
                "entries": {},
                "lineage": [],
            },
            "cache_reveal": {"schema_version": "reveal"},
            "physical_plan": {
                "schema_version": "stage7_actual_v3_miss_only_physical_request_v2"
            },
            "executor_admission": {"admission_passed": True},
        },
        validate=lambda: {},
    )
    assert (round_dir / "logical_request.json").read_bytes() == first
    assert online.next_round_allowed(round_dir) is False
    barrier = {
        "schema_version": "stage7_actual_v3_atomic_feedback_barrier_v2",
        "logical_request_sha256": request["measurement_request_sha256"],
        "feedback_released": True,
        "budget_consumed": 4,
    }
    barrier["barrier_receipt_sha256"] = _sha(barrier)
    online.write_immutable_json(
        round_dir / "atomic_feedback_barrier.json",
        barrier,
        validate=lambda: {},
    )
    monkeypatch.setattr(
        online,
        "validate_committed_barrier",
        lambda directory: (
            barrier
            if (directory / "atomic_feedback_barrier.json").is_file()
            else (_ for _ in ()).throw(ValueError("missing"))
        ),
    )
    assert online.next_round_allowed(round_dir) is True


def test_empty_cache_reveal_and_miss_plan_are_built_only_after_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    request = {"measurement_request_sha256": "a" * 64}
    selection = {"measurement_request": request, "acquisition": {}}
    frozen = {
        "logical_request": request,
        "selection_binding": {"selection_frozen": True},
    }
    reveal = {"entries": [{"disposition": "miss"}] * 4}
    plan = {"physical_row_count": 4}

    monkeypatch.setattr(
        online.actual_adapter,
        "bind_selector_output",
        lambda *_args, **_kwargs: calls.append("freeze") or frozen,
    )
    monkeypatch.setattr(
        online.cache_v2,
        "reveal_v2_cache_after_selection",
        lambda *_args: calls.append("reveal") or reveal,
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: calls.append("plan") or plan,
    )
    result = online.bind_reveal_and_plan(
        selection,
        exact_dimensions_by_candidate={},
        cache_snapshot={
            "schema_version": cache_v2.CACHE_SCHEMA,
            "entries": {},
            "lineage": [],
        },
    )
    assert calls == ["freeze", "reveal", "plan"]
    assert result["physical_plan"]["physical_row_count"] == 4
    assert result["cache_snapshot"]["entries"] == {}


def _formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "stage7_pyramid_tvm_online_core_ablation_quick_v2_20260725"
    rows = [{"row_id": f"candidate-{index}"} for index in range(686)]
    bound = tmp_path / "bound.json"
    bound.write_text("{}", encoding="utf-8")
    record = {"path": str(bound), "sha256": hashlib.sha256(b"{}").hexdigest()}
    contract = {
        "v2_root": str(root),
        "contract_sha256": "c" * 64,
        "ordered_pre_scan_sha256": online.canonical_sha256(rows),
        "immutable_inputs": {
            key: record
            for key in (
                "gold176",
                "graph_features",
                "capability_profiles",
                "source_registry",
                "scope_input_batch",
                "measurement_ap",
                "hv_reference",
            )
        },
    }
    (root / "contracts").mkdir(parents=True)
    (root / "contracts" / "core_ablation_v2.json").write_text("{}", encoding="utf-8")
    (root / "contracts" / "pre_scan_candidate_registry.json").write_text(
        json.dumps(
            {
                "schema_version": "stage7_core_ablation_v2_pre_scan_registry",
                "ordered_pre_scan_sha256": contract["ordered_pre_scan_sha256"],
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    (root / "contracts" / "measurement_cache_initial.json").write_text(
        json.dumps(
            {
                "schema_version": cache_v2.CACHE_SCHEMA,
                "entries": {},
                "lineage": [],
            }
        ),
        encoding="utf-8",
    )
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            directory = root / "variants" / variant / f"seed_{seed}"
            directory.mkdir(parents=True)
            trajectory = {
                "core_ablation_contract_sha256": contract["contract_sha256"],
                "scanner_claim_allowed": False,
            }
            trajectory["trajectory_contract_sha256"] = online.canonical_sha256(
                trajectory
            )
            (directory / "trajectory_contract.json").write_text(
                json.dumps(trajectory), encoding="utf-8"
            )
    monkeypatch.setattr(
        online, "validate_v2_contract", lambda _payload: copy.deepcopy(contract)
    )
    monkeypatch.setattr(
        online.prepare_v2,
        "verify_frozen_actual_v3_executors",
        lambda _root: [],
    )
    monkeypatch.setattr(
        online.deployment_bundle_v2,
        "validate_deployment_bundle",
        lambda _root, **_kwargs: {
            "deployment_release_sha256": EXTERNAL_PINS[
                "expected_release_sha256"
            ]
        },
    )
    return root, contract


def test_validate_formal_root_requires_external_pins_before_root_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="external deployment pins"):
        online.validate_formal_v2_root(tmp_path / "missing")


def test_validate_formal_root_authenticates_12_trajectories_and_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, contract = _formal_root(tmp_path, monkeypatch)
    result = online.validate_formal_v2_root(
        root, repo_root=tmp_path, **EXTERNAL_PINS
    )
    assert result["contract"]["contract_sha256"] == contract["contract_sha256"]
    assert len(result["pre_scan_pool"]) == 686

    one = next(root.glob("variants/*/seed_*/trajectory_contract.json"))
    one.unlink()
    with pytest.raises(ValueError, match="twelve"):
        online.validate_formal_v2_root(
            root, repo_root=tmp_path, **EXTERNAL_PINS
        )


def test_initialize_formal_v2_delegates_to_task1_then_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        online.prepare_v2,
        "initialize_v2_trajectories",
        lambda root, repo_root, **_kwargs: calls.append(f"initialize:{root}")
        or {"trajectory_count": 12},
    )
    monkeypatch.setattr(
        online,
        "validate_formal_v2_root",
        lambda root, repo_root, **_kwargs: calls.append(f"validate:{root}")
        or {"contract": {"contract_sha256": "c" * 64}},
    )
    result = online.initialize_formal_v2(
        tmp_path, repo_root=tmp_path, **EXTERNAL_PINS
    )
    assert [item.split(":")[0] for item in calls] == ["initialize", "validate"]
    assert result["contract_sha256"] == "c" * 64


def test_executor_admission_accepts_zero_to_four_and_rejects_bad_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "schema_version": online.actual_adapter.PHYSICAL_REQUEST_SCHEMA,
        "logical_request_sha256": "a" * 64,
        "physical_request_sha256": "b" * 64,
        "physical_row_count": 0,
        "rows": [],
        "logical_row_bindings": [{} for _ in range(4)],
    }
    monkeypatch.setattr(
        online.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: copy.deepcopy(plan),
    )
    admitted = online.validate_executor_admission(
        plan,
        logical_request={},
        selection_binding={},
        cache_reveal={},
        cache_snapshot={},
        contract_sha256="c" * 64,
    )
    assert admitted["admission_passed"] is True
    assert admitted["gpu_jobs_launched"] == 0
    with pytest.raises(ValueError, match="admission|authentication"):
        online.validate_executor_admission(
            {**plan, "logical_row_bindings": []},
            logical_request={},
            selection_binding={},
            cache_reveal={},
            cache_snapshot={},
            contract_sha256="c" * 64,
        )


def test_executor_admission_rejects_authenticated_plan_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "schema_version": online.actual_adapter.PHYSICAL_REQUEST_SCHEMA,
        "logical_request_sha256": "a" * 64,
        "selection_binding_sha256": "b" * 64,
        "cache_snapshot_sha256": "c" * 64,
        "lineage_head_sha256": "d" * 64,
        "cache_reveal_sha256": "e" * 64,
        "logical_row_count": 4,
        "physical_row_count": 1,
        "row_sha256": {"candidate": "f" * 64},
        "rows": [{"row_id": "candidate"}],
        "logical_row_bindings": [
            {
                "candidate_id": "candidate",
                "logical_row_sha256": "1" * 64,
                "logical_exact_binding_sha256": "2" * 64,
                "exact_cache_key_sha256": "3" * 64,
                "disposition": "miss",
                "physical_row_sha256": "f" * 64,
            },
            *[
                {
                    "candidate_id": f"hit-{index}",
                    "logical_row_sha256": "1" * 64,
                    "logical_exact_binding_sha256": "2" * 64,
                    "exact_cache_key_sha256": str(index + 4) * 64,
                    "disposition": "hit",
                    "physical_row_sha256": None,
                }
                for index in range(3)
            ],
        ],
        "physical_request_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        online.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: {**plan, "physical_request_sha256": "8" * 64},
    )
    with pytest.raises(ValueError, match="SHA|authentication"):
        online.validate_executor_admission(
            plan,
            logical_request={},
            selection_binding={},
            cache_reveal={
                "cache_snapshot_sha256": "c" * 64,
                "lineage_head_sha256": "d" * 64,
                "cache_reveal_sha256": "e" * 64,
                "entries": [
                    {
                        "candidate_id": binding["candidate_id"],
                        "disposition": binding["disposition"],
                    }
                    for binding in plan["logical_row_bindings"]
                ],
            },
            cache_snapshot={},
            contract_sha256="0" * 64,
        )


def test_bound_payload_rows_selector_inputs_and_exact_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def artifact(name: str, payload: Any) -> dict[str, str]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    closure = {"schema_version": "closure"}
    gold = artifact(
        "gold",
        {
            "rows": [
                {
                    "row_id": f"gold-{index:03d}",
                    "manifest_job_id": f"gold-{index:03d}",
                }
                for index in range(176)
            ],
            "closure": closure,
        },
    )
    graph = artifact("graph", {"rows": [{"group_id": "g"}]})
    profile = artifact("profile", {"rows": [{"profile_id": "p"}]})
    source_registry = artifact("source_registry", {"groups": []})
    scope = artifact(
        "scope",
        {
            "measurement_scope": "formal_h800_latency_energy_ap",
            "inference_batch": 1,
        },
    )
    measurement = artifact(
        "measurement",
        {
            "exact_key_protocol_sha256": {
                "build_protocol_sha256": "4" * 64,
                "tuning_protocol_sha256": "5" * 64,
                "measurement_protocol_sha256": "6" * 64,
                "ap_protocol_sha256": "7" * 64,
                "runtime_contract_sha256": "8" * 64,
            }
        },
    )
    validated = {
        "contract": {
            "actual_feedback_v3_executors": online.prepare_v2.core_executor_records(),
            "immutable_inputs": {
                "gold176": gold,
                "graph_features": graph,
                "capability_profiles": profile,
                "source_registry": source_registry,
                "scope_input_batch": scope,
                "measurement_ap": measurement,
            },
        }
    }
    monkeypatch.setattr(
        online,
        "_PINNED_SELECTOR_INPUT_IDENTITIES",
        {
            "gold176": gold,
            "graph_features": graph,
            "capability_profiles": profile,
            "source_registry": source_registry,
        },
    )
    inputs = online._selector_inputs(validated)
    assert inputs["closure"] == closure
    request, _ = _request(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    evidence = tmp_path / "materialization.json"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    evidence.write_text('{"status":"ready"}\n')
    for row in request["rows"]:
        row["source_contract"] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx_path": str(onnx.resolve()),
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "materialization_evidence_path": str(evidence.resolve()),
            "materialization_evidence_sha256": hashlib.sha256(
                evidence.read_bytes()
            ).hexdigest(),
        }
    dimensions = online._exact_dimensions(
        request,
        contract=validated["contract"],
        selector_inputs=inputs,
    )
    assert set(dimensions) == {row["row_id"] for row in request["rows"]}
    assert all(value["batch_size"] == 1 for value in dimensions.values())


def test_real_gold176_is_frozen_through_reviewed_stage5_coldstart_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def artifact(name: str, payload: Any) -> dict[str, str]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    rows = [
        {
            "manifest_job_id": f"gold-{index:03d}",
            "row_id": f"gold-{index:03d}",
            "model": "pyramid",
            "terminal_status": "measured_success_gold",
        }
        for index in range(176)
    ]
    closure = {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
    }
    records = {
        "gold176": artifact("gold176", {"rows": rows}),
        "graph_features": artifact("graphs", {"rows": []}),
        "capability_profiles": artifact("profiles", {"rows": []}),
        "source_registry": artifact("source_registry", {"groups": []}),
    }
    validated = {
        "contract": {
            "immutable_inputs": {
                **records,
                "measurement_ap": artifact(
                    "measurement_ap",
                    {"closure": closure, "exact_key_protocol_sha256": {}},
                ),
            },
            "actual_feedback_v3_executors": [
                {
                    "path": "framework/stage5/single_target_search_v2.py",
                    "sha256": online._file_sha256(
                        online.REPO_ROOT / "framework/stage5/single_target_search_v2.py"
                    ),
                }
            ],
        }
    }
    monkeypatch.setattr(
        online,
        "_PINNED_SELECTOR_INPUT_IDENTITIES",
        copy.deepcopy(records),
    )

    inputs = online._selector_inputs(validated)

    assert len(inputs["initial_rows"]) == 176
    assert [row["manifest_job_id"] for row in inputs["initial_rows"]] == [
        f"gold-{index:03d}" for index in range(176)
    ]
    assert {row["training_source"] for row in inputs["initial_rows"]} == {
        "initial_coldstart"
    }
    assert (
        inputs["coldstart_freeze_binding"]
        == validated["contract"]["actual_feedback_v3_executors"][0]
    )

    contaminated = copy.deepcopy(validated)
    contaminated_path = tmp_path / "gold176_contaminated.json"
    contaminated_rows = copy.deepcopy(rows)
    contaminated_rows[0]["training_source"] = "online_feedback"
    contaminated_path.write_text(
        json.dumps({"rows": contaminated_rows}),
        encoding="utf-8",
    )
    contaminated["contract"]["immutable_inputs"]["gold176"] = {
        "path": str(contaminated_path.resolve()),
        "sha256": hashlib.sha256(contaminated_path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(
        online,
        "_PINNED_SELECTOR_INPUT_IDENTITIES",
        {
            **copy.deepcopy(records),
            "gold176": copy.deepcopy(
                contaminated["contract"]["immutable_inputs"]["gold176"]
            ),
        },
    )
    with pytest.raises(ValueError, match="initial_coldstart"):
        online._selector_inputs(contaminated)


def test_selector_rejects_self_signed_gold176_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gold176.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "row_id": f"gold-{index:03d}",
                        "manifest_job_id": f"gold-{index:03d}",
                    }
                    for index in range(176)
                ],
                "closure": {},
            }
        ),
        encoding="utf-8",
    )
    record = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    validated = {
        "contract": {
            "immutable_inputs": {
                "gold176": record,
                "graph_features": record,
                "capability_profiles": record,
                "measurement_ap": record,
            },
            "actual_feedback_v3_executors": online.prepare_v2.core_executor_records(),
        }
    }

    with pytest.raises(ValueError, match="frozen.*identity|Gold176.*SHA"):
        online._selector_inputs(validated)


@pytest.mark.parametrize("missing_field", ("checkpoint_sha256", "onnx_sha256"))
def test_round0_real_source_preflight_blocks_null_source_sha_before_cache_reveal(
    tmp_path: Path,
    missing_field: str,
) -> None:
    def artifact(name: str, payload: Any) -> dict[str, str]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    scope = artifact(
        "scope_input_batch",
        {
            "schema_version": "stage7_scope_input_batch_v2",
            "task_id": "S7-PYR-TVM",
            "model": "pyramid",
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "measurement_scope": "stage5_actual_v3_end_to_end_latency_energy_full_ap",
            "inference_batch": 1,
        },
    )
    measurement_payload = {
        "schema_version": "stage7_measurement_ap_v2",
        "exact_key_protocol_sha256": {
            "build_protocol_sha256": "4" * 64,
            "tuning_protocol_sha256": "5" * 64,
            "measurement_protocol_sha256": "6" * 64,
            "ap_protocol_sha256": "7" * 64,
            "runtime_contract_sha256": "8" * 64,
        },
    }
    contract = {
        "immutable_inputs": {
            "scope_input_batch": scope,
            "measurement_ap": artifact("measurement_ap", measurement_payload),
        }
    }
    request, _ = _request(tmp_path)
    checkpoint = tmp_path / "candidate.pth"
    onnx = tmp_path / "candidate.onnx"
    evidence = tmp_path / "materialization.json"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    evidence.write_text('{"status":"ready"}\n')
    for row in request["rows"]:
        row["source_contract"] = {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "onnx_path": str(onnx.resolve()),
            "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "materialization_evidence_path": str(evidence.resolve()),
            "materialization_evidence_sha256": hashlib.sha256(
                evidence.read_bytes()
            ).hexdigest(),
        }
    request["rows"][0]["source_contract"][missing_field] = None

    with pytest.raises(
        online.SourceResolutionRequiredBeforeExactCacheReveal,
        match="round0 selected source contract.*checkpoint.*ONNX.*cache reveal",
    ) as caught:
        online._exact_dimensions(
            request,
            contract=contract,
            selector_inputs={"measurement_contract": measurement_payload},
        )
    assert caught.value.audit["status"] == (
        "source_resolution_required_before_exact_cache_reveal"
    )
    assert caught.value.audit["selected_candidate_count"] == 4
    assert caught.value.audit["unresolved_candidate_count"] == 1
    assert caught.value.audit["unresolved_candidates"][0]["candidate_id"] == (
        "candidate-0"
    )


@pytest.mark.parametrize(
    "source_mutation",
    (
        {"checkpoint_sha256": "0" * 64},
        {"onnx_sha256": "a" * 64},
        {"materialization_evidence_path": None},
    ),
)
def test_round0_source_preflight_rejects_placeholder_or_unverified_files(
    tmp_path: Path,
    source_mutation: dict[str, object],
) -> None:
    checkpoint = tmp_path / "candidate.pth"
    onnx = tmp_path / "candidate.onnx"
    evidence = tmp_path / "materialization.json"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    evidence.write_text('{"status":"ready"}\n')
    source = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(onnx.resolve()),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        "materialization_evidence_path": str(evidence.resolve()),
        "materialization_evidence_sha256": hashlib.sha256(
            evidence.read_bytes()
        ).hexdigest(),
    }
    source.update(source_mutation)
    request, _ = _request(tmp_path)
    for row in request["rows"]:
        row["source_contract"] = copy.deepcopy(source)

    with pytest.raises(online.SourceResolutionRequiredBeforeExactCacheReveal) as caught:
        online._preflight_selected_sources(request)

    assert caught.value.audit["cache_reveal_allowed"] is False
    assert caught.value.audit["placeholder_sha_inserted"] is False
    assert caught.value.audit["unresolved_candidate_count"] == 4


def test_prior_state_requires_barrier_and_collects_feedback_cache_and_a2(
    tmp_path: Path,
) -> None:
    root = tmp_path
    initial = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    contracts = root / "contracts"
    contracts.mkdir()
    (contracts / "measurement_cache_initial.json").write_text(
        json.dumps(initial), encoding="utf-8"
    )
    assert online._prior_state(root, "full", 20260718, 0)["cache"] == initial
    with pytest.raises(ValueError, match="barrier"):
        online._prior_state(root, "full", 20260718, 1)


def test_feedback_for_reveal_merges_cache_hit_and_physical_miss(
    tmp_path: Path,
) -> None:
    hit = {"row_id": "hit"}
    hit_path = tmp_path / "hit.json"
    hit_path.write_text(json.dumps(hit), encoding="utf-8")
    reveal = {
        "entries": [
            {
                "candidate_id": "hit",
                "disposition": "hit",
                "terminal_evidence": {
                    "stage5_terminal_artifact": {
                        "path": str(hit_path),
                        "artifact_sha256": hashlib.sha256(
                            hit_path.read_bytes()
                        ).hexdigest(),
                    }
                },
            },
            {"candidate_id": "miss", "disposition": "miss"},
        ]
    }
    assert [
        row["row_id"]
        for row in online._feedback_for_reveal(reveal, [{"row_id": "miss"}])
    ] == ["hit", "miss"]
    with pytest.raises(ValueError, match="incomplete"):
        online._feedback_for_reveal(reveal, [])


def test_cache_append_accepts_only_wrappers_bound_to_selected_successful_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = [
        {
            "candidate_id": "success",
            "exact_cache_key_sha256": "1" * 64,
            "exact_key_dimensions": {"candidate_id": "success"},
        },
        {
            "candidate_id": "failure",
            "exact_cache_key_sha256": "2" * 64,
            "exact_key_dimensions": {"candidate_id": "failure"},
        },
    ]
    monkeypatch.setattr(
        online.cache_v2,
        "validate_selection_binding",
        lambda *_args: {"selected_candidates": selected},
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "validate_actual_v3_terminal_wrapper",
        lambda wrapper: copy.deepcopy(wrapper),
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "append_terminal_wrapper",
        lambda cache, wrapper: {
            **cache,
            "appended": [*(cache.get("appended") or []), wrapper["candidate_id"]],
        },
    )
    plan = {
        "logical_row_bindings": [
            {"candidate_id": "success", "disposition": "miss"},
            {"candidate_id": "failure", "disposition": "miss"},
        ]
    }
    feedback = [
        {
            "row_id": "success",
            "terminal_status": online.stage5_search.SUCCESS_STATUS,
        },
        {
            "row_id": "failure",
            "terminal_status": "numerical_feasibility_failure",
        },
    ]
    wrapper = {
        "candidate_id": "success",
        "exact_cache_key_sha256": "1" * 64,
        "exact_key_dimensions": {"candidate_id": "success"},
        "stage5_terminal_artifact": {},
    }
    monkeypatch.setattr(
        online,
        "_read_stage5_terminal_artifact",
        lambda _wrapper: copy.deepcopy(feedback[0]),
    )
    result = online.append_selected_terminal_wrappers(
        {},
        request={},
        selection_binding={},
        physical_plan=plan,
        merged_feedback=feedback,
        terminal_wrappers=[wrapper],
    )
    assert result["appended"] == ["success"]
    with pytest.raises(ValueError, match="outside selected"):
        online.append_selected_terminal_wrappers(
            {},
            request={},
            selection_binding={},
            physical_plan=plan,
            merged_feedback=feedback,
            terminal_wrappers=[
                {
                    "candidate_id": "failure",
                    "exact_cache_key_sha256": "2" * 64,
                    "exact_key_dimensions": {"candidate_id": "failure"},
                    "stage5_terminal_artifact": {},
                }
            ],
        )
    with pytest.raises(ValueError, match="incomplete"):
        online.append_selected_terminal_wrappers(
            {},
            request={},
            selection_binding={},
            physical_plan=plan,
            merged_feedback=feedback,
            terminal_wrappers=[],
        )


def test_cache_append_rejects_stale_wrapper_for_same_exact_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {
        "candidate_id": "candidate",
        "exact_cache_key_sha256": "1" * 64,
        "exact_key_dimensions": {"candidate_id": "candidate"},
    }
    wrapper = {
        **selected,
        "stage5_terminal_artifact": {},
    }
    feedback = {
        "row_id": "candidate",
        "terminal_status": online.stage5_search.SUCCESS_STATUS,
        "latency_ms": 1.0,
    }
    monkeypatch.setattr(
        online.cache_v2,
        "validate_selection_binding",
        lambda *_args: {"selected_candidates": [selected]},
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "validate_actual_v3_terminal_wrapper",
        lambda value: value,
    )
    monkeypatch.setattr(
        online,
        "_read_stage5_terminal_artifact",
        lambda _wrapper: {**feedback, "latency_ms": 99.0},
    )
    with pytest.raises(ValueError, match="feedback row"):
        online.append_selected_terminal_wrappers(
            {},
            request={},
            selection_binding={},
            physical_plan={
                "logical_row_bindings": [
                    {"candidate_id": "candidate", "disposition": "miss"}
                ]
            },
            merged_feedback=[feedback],
            terminal_wrappers=[wrapper],
        )


def test_next_round_rejects_forged_or_incomplete_barrier(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "round_00"
    directory.mkdir()
    barrier = {
        "schema_version": online.BARRIER_SCHEMA,
        "logical_request_sha256": "a" * 64,
        "feedback_released": True,
        "budget_consumed": 4,
    }
    barrier["barrier_receipt_sha256"] = _sha(barrier)
    (directory / "atomic_feedback_barrier.json").write_text(
        json.dumps(barrier), encoding="utf-8"
    )
    assert online.next_round_allowed(directory) is False


def test_validate_committed_barrier_binds_request_promotion_atomic_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "round_00"
    directory.mkdir()
    request, feedback = _request(tmp_path)
    request_path = directory / "logical_request.json"
    historical_path = directory / "historical_feedback.json"
    request_path.write_bytes(online._json_bytes(request))
    historical_path.write_bytes(online._json_bytes(feedback))
    result = online.promote_and_finalize_atomic_barrier(
        request_path=request_path,
        historical_feedback_path=historical_path,
        extractor=_extract,
    )
    promoted_path = directory / "promoted_feedback.json"
    audit_path = directory / "promotion_audit.json"
    atomic_path = directory / "stage5_atomic_audit.json"
    cache_path = directory / "cache_after_round.json"
    promoted_path.write_bytes(online._json_bytes(result["promotion"]["rows"]))
    audit_path.write_bytes(online._json_bytes(result["promotion"]["audit"]))
    atomic_path.write_bytes(online._json_bytes(result["barrier"]))
    snapshot_cache = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    terminal_wrapper = {"candidate_id": feedback[0]["row_id"]}
    cache = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {"exact-key": terminal_wrapper},
        "lineage": [{"candidate_id": feedback[0]["row_id"]}],
    }
    cache_path.write_bytes(online._json_bytes(cache))
    pre_source_binding = {"binding_kind": "pre_source_identity"}
    exact_binding = {"binding_kind": "exact_source_identity"}
    (directory / "selection_binding.json").write_bytes(
        online._json_bytes(pre_source_binding)
    )
    (directory / "source_resolution_plan.json").write_text("{}", encoding="utf-8")
    (directory / "source_resolution_result.json").write_text("{}", encoding="utf-8")
    (directory / "exact_selection_binding.json").write_bytes(
        online._json_bytes(exact_binding)
    )
    (directory / "cache_snapshot_before_reveal.json").write_bytes(
        online._json_bytes(snapshot_cache)
    )
    reveal = {
        "entries": [
            {
                "candidate_id": row["row_id"],
                "disposition": "miss" if index == 0 else "hit",
            }
            for index, row in enumerate(request["rows"])
        ]
    }
    (directory / "cache_reveal.json").write_bytes(online._json_bytes(reveal))
    (directory / "miss_only_physical_request.json").write_text("{}", encoding="utf-8")
    admission = {"contract_sha256": "c" * 64}
    (directory / "executor_admission.json").write_bytes(online._json_bytes(admission))
    monkeypatch.setattr(
        online,
        "validate_executor_admission",
        lambda *_args, **_kwargs: admission,
    )
    validated_bindings: list[dict[str, Any]] = []

    def validate_exact_binding(
        _request: dict[str, Any], binding: dict[str, Any]
    ) -> dict[str, Any]:
        validated_bindings.append(copy.deepcopy(binding))
        return {
            "selected_candidates": [
                {
                    "candidate_id": feedback[0]["row_id"],
                    "exact_cache_key_sha256": "exact-key",
                }
            ]
        }

    monkeypatch.setattr(
        online.cache_v2,
        "validate_selection_binding",
        validate_exact_binding,
    )
    monkeypatch.setattr(
        online,
        "_feedback_for_reveal",
        lambda _reveal, _physical: copy.deepcopy(feedback),
    )
    monkeypatch.setattr(
        online.cache_v2,
        "validate_actual_v3_cache",
        lambda value: copy.deepcopy(value),
    )
    monkeypatch.setattr(
        online,
        "_read_stage5_terminal_artifact",
        lambda _terminal: copy.deepcopy(result["promotion"]["rows"][0]),
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "append_terminal_wrapper",
        lambda _cache, _terminal: copy.deepcopy(cache),
    )
    monkeypatch.setattr(
        online.stage5_promote,
        "promote_feedback_batch",
        lambda *_args, **_kwargs: copy.deepcopy(result["promotion"]),
    )
    physical_terminal = _physical_terminal([feedback[0]])
    physical_terminal_path = directory / "physical_terminal.json"
    physical_terminal_path.write_bytes(online._json_bytes(physical_terminal))
    payload = {
        "schema_version": online.BARRIER_SCHEMA,
        "logical_request_sha256": request["measurement_request_sha256"],
        "feedback_released": True,
        "budget_consumed": 4,
        "successful_rows": 4,
        "feasibility_terminal_rows": 0,
        "silent_surrogate_fallback_count": 0,
        "historical_feedback_file_sha256": hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest(),
        "promoted_feedback_file_sha256": hashlib.sha256(
            promoted_path.read_bytes()
        ).hexdigest(),
        "promotion_audit_file_sha256": hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        "stage5_atomic_audit_file_sha256": hashlib.sha256(
            atomic_path.read_bytes()
        ).hexdigest(),
        "cache_after_round_file_sha256": hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest(),
        "physical_terminal_batch_sha256": physical_terminal[
            "physical_terminal_batch_sha256"
        ],
        "physical_terminal_file_sha256": hashlib.sha256(
            physical_terminal_path.read_bytes()
        ).hexdigest(),
        "stage5_atomic_audit": result["barrier"],
    }
    barrier = {
        **payload,
        "barrier_receipt_sha256": _sha(payload),
    }
    (directory / "atomic_feedback_barrier.json").write_bytes(
        online._json_bytes(barrier)
    )
    assert online.validate_committed_barrier(directory)["feedback_released"] is True
    assert validated_bindings == [exact_binding]
    forged_atomic = {
        **result["barrier"],
        "successful_rows": 3,
    }
    atomic_path.write_bytes(online._json_bytes(forged_atomic))
    forged_payload = {
        **payload,
        "stage5_atomic_audit": forged_atomic,
        "stage5_atomic_audit_file_sha256": hashlib.sha256(
            atomic_path.read_bytes()
        ).hexdigest(),
    }
    forged = {
        **forged_payload,
        "barrier_receipt_sha256": _sha(forged_payload),
    }
    (directory / "atomic_feedback_barrier.json").write_bytes(online._json_bytes(forged))
    assert online.next_round_allowed(directory) is False
    atomic_path.write_bytes(online._json_bytes(result["barrier"]))
    (directory / "atomic_feedback_barrier.json").write_bytes(
        online._json_bytes(barrier)
    )
    physical_terminal_path.write_text("{}", encoding="utf-8")
    assert online.next_round_allowed(directory) is False
    physical_terminal_path.write_bytes(online._json_bytes(physical_terminal))
    promoted = _read_json_for_test(promoted_path)
    promoted[0]["latency_ms"] = 999.0
    promoted_path.write_bytes(online._json_bytes(promoted))
    assert online.next_round_allowed(directory) is False


def test_freeze_selection_relocates_source_outputs_before_plan_without_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _source_selection(tmp_path)
    _install_phase2_selector(monkeypatch, root=tmp_path, selection=selection)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cache/exact binding is forbidden before source-ready")

    monkeypatch.setattr(online, "_global_cache_snapshot", forbidden)
    monkeypatch.setattr(online.cache_v2, "reveal_v2_cache_after_selection", forbidden)
    monkeypatch.setattr(online.actual_adapter, "bind_selector_output", forbidden)

    receipt = online.freeze_selection_and_source_plan(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
    )

    directory = tmp_path / "variants/full/seed_20260718/round_00"
    assert receipt["controller_state"] == "SOURCE_PLAN_FROZEN"
    assert receipt["logical_request_sha256"] != selection[
        "measurement_request"
    ]["measurement_request_sha256"]
    request = json.loads(
        (directory / "logical_request.json").read_text(encoding="utf-8")
    )
    assert request["measurement_request_sha256"] == receipt[
        "logical_request_sha256"
    ]
    assert [row["row_id"] for row in request["rows"]] == [
        row["row_id"] for row in selection["measurement_request"]["rows"]
    ]
    assert all(
        Path(row["source_contract"]["checkpoint_path"]).is_relative_to(
            tmp_path / "sources/pyramid"
        )
        for row in request["rows"]
    )
    relocation = json.loads(
        (directory / "source_contract_relocation.json").read_text(
            encoding="utf-8"
        )
    )
    assert relocation["pre_relocation_request_sha256"] == selection[
        "measurement_request"
    ]["measurement_request_sha256"]
    assert relocation["relocated_request_sha256"] == receipt[
        "logical_request_sha256"
    ]
    selector_output = json.loads(
        (directory / "selector_output.json").read_text(encoding="utf-8")
    )
    assert selector_output["measurement_request"] == request
    assert (directory / "selection_binding.json").is_file()
    assert (directory / "source_resolution_plan.json").is_file()
    for forbidden_name in (
        "exact_selection_binding.json",
        "cache_snapshot_before_reveal.json",
        "cache_reveal.json",
        "miss_only_physical_request.json",
        "executor_admission.json",
    ):
        assert not (directory / forbidden_name).exists()
    assert online.round_controller_state(directory) == "SOURCE_PLAN_FROZEN"


def test_bind_reveal_after_real_four_row_source_ready_preserves_request_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _source_selection(tmp_path)
    _install_phase2_selector(monkeypatch, root=tmp_path, selection=selection)
    online.freeze_selection_and_source_plan(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
    )
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    request_before = (directory / "logical_request.json").read_bytes()
    plan = json.loads(
        (directory / "source_resolution_plan.json").read_text(encoding="utf-8")
    )
    source_result = _write_source_result(
        json.loads((directory / "logical_request.json").read_text(encoding="utf-8")),
        plan,
    )
    result_path = tmp_path / "source_result.json"
    result_path.write_text(json.dumps(source_result), encoding="utf-8")
    empty_cache = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    monkeypatch.setattr(
        online, "_global_cache_snapshot", lambda _root: copy.deepcopy(empty_cache)
    )

    receipt = online.bind_reveal_after_source_ready(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        source_result_path=result_path,
        repo_root=tmp_path,
    )

    assert receipt["controller_state"] == "CACHE_REVEALED"
    assert receipt["synthetic_nonfinal"] is False
    assert (directory / "logical_request.json").read_bytes() == request_before
    binding = json.loads(
        (directory / "exact_selection_binding.json").read_text(encoding="utf-8")
    )
    by_id = {row["candidate_id"]: row for row in source_result["rows"]}
    for selected in binding["selected_candidates"]:
        dimensions = selected["exact_key_dimensions"]
        resolved = by_id[selected["candidate_id"]]
        assert dimensions["source_checkpoint_sha256"] == resolved["checkpoint_sha256"]
        assert dimensions["onnx_sha256"] == resolved["onnx_sha256"]
    reveal = json.loads((directory / "cache_reveal.json").read_text(encoding="utf-8"))
    assert [entry["disposition"] for entry in reveal["entries"]] == ["miss"] * 4
    assert online.round_controller_state(directory) == "CACHE_REVEALED"


@pytest.mark.parametrize("mutation", ("partial", "reordered", "null_sha", "plan_drift"))
def test_bind_reveal_rejects_non_atomic_or_drifted_source_result_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    selection = _source_selection(tmp_path)
    _install_phase2_selector(monkeypatch, root=tmp_path, selection=selection)
    online.freeze_selection_and_source_plan(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
    )
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    plan = json.loads(
        (directory / "source_resolution_plan.json").read_text(encoding="utf-8")
    )
    result = _write_source_result(
        json.loads((directory / "logical_request.json").read_text(encoding="utf-8")),
        plan,
    )
    if mutation == "partial":
        result["rows"] = result["rows"][:3]
    elif mutation == "reordered":
        result["rows"][0], result["rows"][1] = result["rows"][1], result["rows"][0]
    elif mutation == "null_sha":
        result["rows"][0]["onnx_sha256"] = None
    else:
        result["source_resolution_plan_sha256"] = "9" * 64
    _resign_source_result(result)
    result_path = tmp_path / f"{mutation}.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cache must remain unread on source-result failure")

    monkeypatch.setattr(online, "_global_cache_snapshot", forbidden)
    with pytest.raises(ValueError):
        online.bind_reveal_after_source_ready(
            tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            source_result_path=result_path,
            repo_root=tmp_path,
        )
    assert not (directory / "cache_reveal.json").exists()
    assert online.round_controller_state(directory) == "SOURCE_PLAN_FROZEN"


def test_source_retry_consumes_zero_budget_and_resumes_same_frozen_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _source_selection(tmp_path)
    _install_phase2_selector(monkeypatch, root=tmp_path, selection=selection)
    online.freeze_selection_and_source_plan(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
    )
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    request_before = (directory / "logical_request.json").read_bytes()
    plan_before = (directory / "source_resolution_plan.json").read_bytes()
    plan = json.loads(plan_before)
    retry = source_resolution.build_source_resolution_retry_result(
        plan,
        retry_candidate_ids=[plan["ordered_row_ids"][0]],
        reason_code="source_interrupted",
    )
    retry_path = tmp_path / "source_retry.json"
    retry_path.write_text(json.dumps(retry), encoding="utf-8")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("retry must not read cache")

    monkeypatch.setattr(online, "_global_cache_snapshot", forbidden)
    receipt = online.bind_reveal_after_source_ready(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        source_result_path=retry_path,
        repo_root=tmp_path,
    )

    assert receipt["controller_state"] == "SOURCE_PLAN_FROZEN"
    assert receipt["selected_event_budget_delta"] == 0
    assert (directory / "logical_request.json").read_bytes() == request_before
    assert (directory / "source_resolution_plan.json").read_bytes() == plan_before
    assert not (directory / "cache_reveal.json").exists()


def test_prior_selection_state_authenticates_feedback_without_cache_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    request, _ = _request(tmp_path)
    (directory / "logical_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    (directory / "promoted_feedback.json").write_text("[]", encoding="utf-8")
    (directory / "selector_output.json").write_text(
        json.dumps({"a2_frozen": {"bundle": "frozen"}}), encoding="utf-8"
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("selection phase cannot validate/read cache")

    monkeypatch.setattr(online, "validate_committed_barrier", forbidden)
    monkeypatch.setattr(online, "_global_cache_snapshot", forbidden)
    monkeypatch.setattr(
        online,
        "_validate_prior_feedback_barrier",
        lambda _directory: {"feedback_released": True},
        raising=False,
    )

    state = online._prior_selection_state(tmp_path, "full", 20260718, 1)

    assert state["prior_selected_ids"] == [row["row_id"] for row in request["rows"]]
    assert state["a2_frozen"] == {"bundle": "frozen"}
    assert "cache" not in state


def test_synthetic_source_result_requires_explicit_no_gpu_and_is_nonfinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _source_selection(tmp_path)
    _install_phase2_selector(monkeypatch, root=tmp_path, selection=selection)
    online.freeze_selection_and_source_plan(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
    )
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    plan = json.loads(
        (directory / "source_resolution_plan.json").read_text(encoding="utf-8")
    )
    synthetic = source_resolution.build_synthetic_dryrun_source_resolution_result(plan)
    source_path = tmp_path / "synthetic.json"
    source_path.write_text(json.dumps(synthetic), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic.*no-GPU"):
        online.bind_reveal_after_source_ready(
            tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            source_result_path=source_path,
            repo_root=tmp_path,
        )

    def forbidden_global_cache(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("synthetic dry-run must not read the formal global cache")

    monkeypatch.setattr(online, "_global_cache_snapshot", forbidden_global_cache)
    receipt = online.bind_reveal_after_source_ready(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        source_result_path=source_path,
        synthetic_no_gpu_dryrun=True,
        repo_root=tmp_path,
    )
    assert receipt["synthetic_nonfinal"] is True
    assert receipt["eligible_for_cache_append"] is False
    assert receipt["eligible_for_finalization"] is False
    assert receipt["cache_membership_observed"] is False
    for formal_artifact in (
        "source_resolution_result.json",
        "exact_selection_binding.json",
        "cache_snapshot_before_reveal.json",
        "cache_reveal.json",
        "miss_only_physical_request.json",
        "executor_admission.json",
        "exact_binding_reveal_receipt.json",
    ):
        assert not (directory / formal_artifact).exists()
    for synthetic_artifact in (
        "synthetic_source_resolution_result.json",
        "synthetic_exact_selection_binding.json",
        "synthetic_physical_protocol.json",
        "synthetic_protocol_receipt.json",
    ):
        assert (directory / synthetic_artifact).is_file()


def test_precommit_lineage_validator_rejects_drift_before_any_commit_write(
    tmp_path: Path,
) -> None:
    request, _ = _request(tmp_path)
    exact_binding = {"logical_request": {"logical_request_sha256": "a" * 64}}
    snapshot = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    reveal = {"entries": [{"candidate_id": "forged", "disposition": "miss"}]}
    physical_plan = {"logical_row_bindings": []}
    stored_admission = {"contract_sha256": "c" * 64}

    with pytest.raises(ValueError, match="precommit lineage drift"):
        online.feedback_barrier.validate_precommit_lineage(
            request=request,
            exact_binding=exact_binding,
            cache_snapshot=snapshot,
            cache_reveal=reveal,
            physical_plan=physical_plan,
            stored_admission=stored_admission,
            validate_admission=online.validate_executor_admission,
        )


def test_real_multi_round_exact_cache_reuse_closes_precommit_lineage(
    tmp_path: Path,
) -> None:
    from framework.tests import test_stage7_actual_v3_adapter_v2 as fixtures

    adapter = fixtures._adapter()
    frozen_round0 = fixtures._frozen()
    request0 = frozen_round0["logical_request"]
    binding0 = frozen_round0["selection_binding"]
    cache = fixtures._empty_cache()
    reveal0 = cache_v2.reveal_v2_cache_after_selection(request0, binding0, cache)
    plan0 = adapter.build_miss_only_physical_plan(
        request0, binding0, reveal0, cache_snapshot=cache
    )
    for row in request0["rows"]:
        candidate_id = row["row_id"]
        terminal = adapter.wrap_actual_v3_terminal(
            request0,
            binding0,
            plan0,
            candidate_id=candidate_id,
            **fixtures._raw_stage3_terminal_inputs(
                candidate_id, request0, plan0, tmp_path
            ),
        )
        cache = adapter.append_terminal_wrapper(cache, terminal)

    unsigned1 = {
        key: copy.deepcopy(value)
        for key, value in request0.items()
        if key != "measurement_request_sha256"
    }
    unsigned1["round_index"] = 1
    request1 = {**unsigned1, "measurement_request_sha256": _sha(unsigned1)}
    dimensions1 = {
        row["candidate_id"]: copy.deepcopy(row["exact_key_dimensions"])
        for row in binding0["selected_candidates"]
    }
    frozen_round1 = adapter.bind_selector_output(
        {
            "acquisition": {
                "selected_row_ids": [row["row_id"] for row in request1["rows"]]
            },
            "measurement_request": request1,
        },
        exact_dimensions_by_candidate=dimensions1,
    )
    binding1 = frozen_round1["selection_binding"]
    reveal1 = cache_v2.reveal_v2_cache_after_selection(request1, binding1, cache)
    plan1 = adapter.build_miss_only_physical_plan(
        request1, binding1, reveal1, cache_snapshot=cache
    )
    admission1 = online.validate_executor_admission(
        plan1,
        logical_request=request1,
        selection_binding=binding1,
        cache_reveal=reveal1,
        cache_snapshot=cache,
        contract_sha256="c" * 64,
    )

    validated = online.feedback_barrier.validate_precommit_lineage(
        request=request1,
        exact_binding=binding1,
        cache_snapshot=cache,
        cache_reveal=reveal1,
        physical_plan=plan1,
        stored_admission=admission1,
        validate_admission=online.validate_executor_admission,
    )

    assert [row["disposition"] for row in reveal1["entries"]] == ["hit"] * 4
    assert plan1["physical_row_count"] == 0
    assert validated["executor_admission"] == admission1


def test_finalize_precommit_drift_has_zero_commit_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from framework.tests import test_stage7_actual_v3_adapter_v2 as fixtures

    frozen = fixtures._frozen()
    request = frozen["logical_request"]
    binding = frozen["selection_binding"]
    cache = fixtures._empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(request, binding, cache)
    plan = online.actual_adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    admission = online.validate_executor_admission(
        plan,
        logical_request=request,
        selection_binding=binding,
        cache_reveal=reveal,
        cache_snapshot=cache,
        contract_sha256="c" * 64,
    )
    drifted_reveal = copy.deepcopy(reveal)
    drifted_reveal["entries"][0]["disposition"] = "hit"
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    artifacts = {
        "logical_request.json": request,
        "selection_binding.json": {},
        "source_resolution_plan.json": {},
        "source_resolution_result.json": {},
        "exact_selection_binding.json": binding,
        "cache_snapshot_before_reveal.json": cache,
        "cache_reveal.json": drifted_reveal,
        "miss_only_physical_request.json": plan,
        "executor_admission.json": admission,
    }
    for filename, artifact in artifacts.items():
        (directory / filename).write_bytes(online._json_bytes(artifact))
    terminal = tmp_path / "terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        online.feedback_barrier.source_resolution,
        "validate_source_resolution_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online.feedback_barrier.source_resolution,
        "validate_formal_source_resolution_result",
        lambda *_args, **_kwargs: {},
    )
    committed_writes: list[Path] = []

    def forbidden_after_lineage(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("precommit drift must stop promotion/cache append")

    with pytest.raises(ValueError, match="precommit lineage drift"):
        online.feedback_barrier.finalize_round(
            tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            terminal_payload_path=terminal,
            repo_root=tmp_path,
            validate_root=lambda *_args, **_kwargs: {"root": tmp_path},
            round_directory=online._round_dir,
            validate_identity=lambda *_args, **_kwargs: {
                "logical_request_sha256": request["measurement_request_sha256"]
            },
            merge_logical=online.merge_feedback_in_logical_order,
            promote_atomic=forbidden_after_lineage,
            append_wrappers=forbidden_after_lineage,
            validate_admission=online.validate_executor_admission,
            write_json=lambda path, *_args, **_kwargs: committed_writes.append(path),
        )
    assert committed_writes == []
    for filename in (
        "historical_feedback.json",
        "promoted_feedback.json",
        "promotion_audit.json",
        "stage5_atomic_audit.json",
        "cache_after_round.json",
        "atomic_feedback_barrier.json",
    ):
        assert not (directory / filename).exists()


@pytest.mark.parametrize(
    "argv,command",
    (
        (["initialize"], "initialize"),
        (
            [
                "freeze-selection-and-source-plan",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
            ],
            "freeze-selection-and-source-plan",
        ),
        (
            [
                "bind-reveal-after-source-ready",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
                "--source-result-json",
                "/tmp/source-result.json",
            ],
            "bind-reveal-after-source-ready",
        ),
        (
            [
                "prepare-round",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
            ],
            "prepare-round",
        ),
        (
            [
                "finalize-round",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
                "--terminal-payload-json",
                "/tmp/terminal.json",
            ],
            "finalize-round",
        ),
        (["validate"], "validate"),
    ),
)
def test_parser_exposes_only_canonical_commands(argv: list[str], command: str) -> None:
    assert online.build_parser().parse_args(argv).command == command


def test_prepare_round_compatibility_alias_stops_before_exact_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        online,
        "freeze_selection_and_source_plan",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or {
            "command": "freeze-selection-and-source-plan",
            "controller_state": "SOURCE_PLAN_FROZEN",
        },
    )

    receipt = online.prepare_round(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        repo_root=tmp_path,
        **EXTERNAL_PINS,
    )
    assert receipt["controller_state"] == "SOURCE_PLAN_FROZEN"
    assert len(calls) == 1
    assert calls[0][1]["expected_release_sha256"] == "a" * 64
    assert calls[0][1]["expected_manifest_file_sha256"] == "b" * 64


def test_finalize_round_commits_cache_and_barrier_last_and_is_retry_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    directory = root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    request, feedback = _request(tmp_path)
    (directory / "logical_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    (directory / "cache_reveal.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"candidate_id": row["row_id"], "disposition": "miss"}
                    for row in request["rows"]
                ]
            }
        ),
        encoding="utf-8",
    )
    (directory / "selection_binding.json").write_text("{}", encoding="utf-8")
    (directory / "source_resolution_plan.json").write_text("{}", encoding="utf-8")
    (directory / "source_resolution_result.json").write_text("{}", encoding="utf-8")
    exact_binding = {
        "logical_request": {
            "logical_request_sha256": request["measurement_request_sha256"]
        }
    }
    (directory / "exact_selection_binding.json").write_text(
        json.dumps(exact_binding), encoding="utf-8"
    )
    (directory / "miss_only_physical_request.json").write_text("{}", encoding="utf-8")
    empty = {
        "schema_version": cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    (directory / "cache_snapshot_before_reveal.json").write_text(
        json.dumps(empty), encoding="utf-8"
    )
    admission = {"contract_sha256": "c" * 64}
    (directory / "executor_admission.json").write_text(
        json.dumps(admission), encoding="utf-8"
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(_physical_terminal(feedback)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        online,
        "validate_formal_v2_root",
        lambda *_args, **_kwargs: {
            "root": root,
            "contract": {"contract_sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        online,
        "_prior_state",
        lambda *_args, **_kwargs: {"cache": empty},
    )
    monkeypatch.setattr(
        online,
        "validate_selection_identity",
        lambda *_args, **_kwargs: {
            "logical_request_sha256": request["measurement_request_sha256"]
        },
    )
    monkeypatch.setattr(
        online.source_resolution,
        "validate_source_resolution_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online.source_resolution,
        "validate_formal_source_resolution_result",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online.cache_v2,
        "validate_selection_binding",
        lambda *_args, **_kwargs: {
            "logical_request": copy.deepcopy(exact_binding["logical_request"]),
            "binding": copy.deepcopy(exact_binding),
            "selected_candidates": [],
        },
    )
    monkeypatch.setattr(
        online.cache_v2,
        "reveal_v2_cache_after_selection",
        lambda *_args, **_kwargs: _read_json_for_test(directory / "cache_reveal.json"),
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online,
        "validate_executor_admission",
        lambda *_args, **_kwargs: copy.deepcopy(admission),
    )
    monkeypatch.setattr(
        online,
        "promote_and_finalize_atomic_barrier",
        lambda **_kwargs: {
            "promotion": {
                "rows": feedback,
                "audit": {
                    "promoted_row_count": 4,
                    "silent_surrogate_fallback_count": 0,
                },
            },
            "barrier": {
                "feedback_released": True,
                "budget_consumed": 4,
                "successful_rows": 4,
                "feasibility_terminal_rows": 0,
            },
        },
    )
    monkeypatch.setattr(
        online,
        "append_selected_terminal_wrappers",
        lambda cache, **_kwargs: {**cache, "appended": True},
    )
    monkeypatch.setattr(
        online.feedback_barrier,
        "_write_promoted_terminal_artifacts",
        lambda *_args, **_kwargs: {row["row_id"]: {} for row in feedback},
    )
    monkeypatch.setattr(
        online.feedback_barrier.physical_feedback,
        "build_post_promotion_terminal_wrappers",
        lambda **_kwargs: [{"wrapper": 1}],
    )

    barrier = online.finalize_round(
        root,
        variant="full",
        seed=20260718,
        round_index=0,
        terminal_payload_path=terminal,
        repo_root=root,
    )
    assert barrier["feedback_released"] is True
    monkeypatch.setattr(
        online,
        "validate_committed_barrier",
        lambda _directory: barrier,
    )
    assert online.next_round_allowed(directory) is True
    assert _read_json_for_test(directory / "cache_after_round.json")["appended"] is True
    assert (
        online.finalize_round(
            root,
            variant="full",
            seed=20260718,
            round_index=0,
            terminal_payload_path=terminal,
            repo_root=root,
        )
        == barrier
    )


def _read_json_for_test(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_finalize_round_does_not_commit_when_atomic_barrier_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    directory = root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    request, feedback = _request(tmp_path)
    (directory / "logical_request.json").write_text(
        json.dumps(request), encoding="utf-8"
    )
    (directory / "cache_reveal.json").write_text(
        json.dumps(
            {
                "entries": [
                    {"candidate_id": row["row_id"], "disposition": "miss"}
                    for row in request["rows"]
                ]
            }
        ),
        encoding="utf-8",
    )
    (directory / "selection_binding.json").write_text("{}", encoding="utf-8")
    (directory / "source_resolution_plan.json").write_text("{}", encoding="utf-8")
    (directory / "source_resolution_result.json").write_text("{}", encoding="utf-8")
    exact_binding = {
        "logical_request": {
            "logical_request_sha256": request["measurement_request_sha256"]
        }
    }
    (directory / "exact_selection_binding.json").write_text(
        json.dumps(exact_binding), encoding="utf-8"
    )
    (directory / "miss_only_physical_request.json").write_text("{}", encoding="utf-8")
    (directory / "cache_snapshot_before_reveal.json").write_text(
        json.dumps(
            {
                "schema_version": cache_v2.CACHE_SCHEMA,
                "entries": {},
                "lineage": [],
            }
        ),
        encoding="utf-8",
    )
    admission = {"contract_sha256": ""}
    (directory / "executor_admission.json").write_text(
        json.dumps(admission), encoding="utf-8"
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps(_physical_terminal(feedback)), encoding="utf-8")
    monkeypatch.setattr(
        online,
        "validate_formal_v2_root",
        lambda *_args, **_kwargs: {"root": root, "contract": {}},
    )
    monkeypatch.setattr(
        online,
        "validate_selection_identity",
        lambda *_args, **_kwargs: {
            "logical_request_sha256": request["measurement_request_sha256"]
        },
    )
    monkeypatch.setattr(
        online.source_resolution,
        "validate_source_resolution_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online.source_resolution,
        "validate_formal_source_resolution_result",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online.cache_v2,
        "validate_selection_binding",
        lambda *_args, **_kwargs: {
            "logical_request": copy.deepcopy(exact_binding["logical_request"]),
            "binding": copy.deepcopy(exact_binding),
            "selected_candidates": [],
        },
    )
    monkeypatch.setattr(
        online.cache_v2,
        "reveal_v2_cache_after_selection",
        lambda *_args, **_kwargs: _read_json_for_test(directory / "cache_reveal.json"),
    )
    monkeypatch.setattr(
        online.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        online,
        "validate_executor_admission",
        lambda *_args, **_kwargs: copy.deepcopy(admission),
    )
    monkeypatch.setattr(
        online,
        "promote_and_finalize_atomic_barrier",
        lambda **_kwargs: {
            "promotion": {
                "rows": feedback,
                "audit": {"silent_surrogate_fallback_count": 0},
            },
            "barrier": {
                "feedback_released": False,
                "budget_consumed": 0,
            },
        },
    )
    with pytest.raises(ValueError, match="did not release"):
        online.finalize_round(
            root,
            variant="full",
            seed=20260718,
            round_index=0,
            terminal_payload_path=terminal,
            repo_root=root,
        )
    assert not (directory / "historical_feedback.json").exists()
    assert not (directory / "stage5_atomic_audit.json").exists()
    assert not (directory / "cache_after_round.json").exists()
    assert not (directory / "atomic_feedback_barrier.json").exists()


@pytest.mark.parametrize(
    "argv,patched,expected",
    (
        (["initialize"], "initialize_formal_v2", "initialize"),
        (
            [
                "prepare-round",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
            ],
            "freeze_selection_and_source_plan",
            "freeze",
        ),
        (
            [
                "finalize-round",
                "--variant",
                "full",
                "--seed",
                "20260718",
                "--round-index",
                "0",
                "--terminal-payload-json",
                "/tmp/terminal.json",
            ],
            "finalize_round",
            "finalize",
        ),
    ),
)
def test_main_dispatches_canonical_mutating_commands(
    argv: list[str],
    patched: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        online, patched, lambda *_args, **_kwargs: {"command": expected}
    )
    assert online.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["command"] == expected


def test_main_validate_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        online,
        "validate_formal_v2_root",
        lambda *_args, **_kwargs: {"contract": {"contract_sha256": "c" * 64}},
    )
    assert online.main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["trajectory_count"] == 12
