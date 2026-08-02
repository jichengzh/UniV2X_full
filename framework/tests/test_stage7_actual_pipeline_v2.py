from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from framework.stage7 import (
    actual_pipeline_v2,
    physical_execution_v2,
    physical_feedback_v2,
    physical_runtime_v2,
)


SHA = "a" * 64
LEASES = tuple(f"GPU-{index}" for index in range(4))


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _receipt(root: Path, frozen_repo_root: Path) -> dict[str, Any]:
    unsigned = {
        "schema_version": "stage7_core_no_gpu_dry_run_v2",
        "formal_v2_root": str(root.resolve()),
        "frozen_repo_root": str(frozen_repo_root.resolve()),
        "formal_v2_gpu_jobs_launched": 0,
        "cuda_visible_devices": "",
        "deployment_bundle_sha256": SHA,
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
    }
    return {**unsigned, "dry_run_receipt_sha256": _sha(unsigned)}


def _inventory() -> list[dict[str, Any]]:
    return [
        {"global_index": index + 2, "uuid": uuid, "name": "NVIDIA H800 80GB HBM3"}
        for index, uuid in enumerate(LEASES)
    ]


def _bindings(count: int) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"candidate-{index}",
            "logical_row_index": index,
            "disposition": "miss",
        }
        for index in range(count)
    ]


def _projection(count: int) -> dict[str, Any]:
    return {
        "schema_version": "stage5_independent_validation_request_v1",
        "measurement_request_sha256": "b" * 64,
        "rows": [
            {
                "row_id": f"candidate-{index}",
                "candidate_id": f"candidate-{index}",
                "group_id": f"group-{index}",
                "q_mode": "int8" if index % 2 == 0 else "fp16",
            }
            for index in range(count)
        ],
    }


def _admission(
    tmp_path: Path,
    count: int,
    *,
    primitive_sha_probe: Any = None,
    inventory_probe: Any = None,
) -> dict[str, Any]:
    frozen = tmp_path / "frozen"
    formal = tmp_path / "formal"
    frozen.mkdir(exist_ok=True)
    formal.mkdir(exist_ok=True)
    return actual_pipeline_v2.validate_actual_runtime_admission(
        host_name="zs-nj-tap-gpu18",
        formal_v2_root=formal,
        frozen_repo_root=frozen,
        deployment_bundle_sha256=SHA,
        no_gpu_receipt=_receipt(formal, frozen),
        physical_bindings=_bindings(count),
        ordered_lease_uuids=LEASES,
        parent_cuda_visible_devices=",".join(LEASES),
        expected_lock_owner="stage7-owner",
        inventory_probe=inventory_probe or _inventory,
        lock_probe=lambda _uuid: "stage7-owner",
        process_probe=lambda: [],
        primitive_sha_probe=(
            primitive_sha_probe
            or (lambda relative: physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256[relative])
        ),
    )


def test_public_defaults_are_the_reviewed_phase5b_c_d_callables() -> None:
    signature = inspect.signature(actual_pipeline_v2.build_concrete_stage_runner)
    admission_signature = inspect.signature(
        actual_pipeline_v2.validate_actual_runtime_admission
    )

    assert (
        signature.parameters["performance_artifact_builder"].default
        is physical_execution_v2.build_authenticated_performance_artifacts
    )
    assert (
        admission_signature.parameters["uuid_assignment_builder"].default
        is physical_runtime_v2.build_uuid_assignments
    )
    assert (
        signature.parameters["quant_executor"].default
        is physical_runtime_v2.execute_bound_quant_row
    )
    assert (
        signature.parameters["performance_executor"].default
        is physical_runtime_v2.execute_bound_performance_row
    )
    assert (
        signature.parameters["ap_executor"].default
        is physical_runtime_v2.execute_bound_ap_row
    )
    assert (
        signature.parameters["physical_finalizer"].default
        is physical_feedback_v2.finalize_physical_rows
    )


def test_actual_admission_authenticates_h800_uuid_mask_locks_and_pins(
    tmp_path: Path,
) -> None:
    admission = _admission(tmp_path, 4)

    assert admission["host_name"] == "zs-nj-tap-gpu18"
    assert admission["actual_cuda_visible_devices"] == ",".join(LEASES)
    assert admission["physical_row_count"] == 4
    assert [
        row["lease_slot"] for row in admission["uuid_assignments"]["assignments"]
    ] == [
        0,
        1,
        2,
        3,
    ]
    assert admission["no_gpu_truth"]["synthetic_non_measurement"] is True
    assert admission["actual_execution_truth"] == {
        "synthetic_non_measurement": False,
        "actual_v3_hardware_evidence": True,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": True,
    }
    assert admission["primitive_sha256"] == physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256
    assert admission["actual_runtime_admission_sha256"] == _sha(
        {
            key: value
            for key, value in admission.items()
            if key != "actual_runtime_admission_sha256"
        }
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host_name", "other-host", "host"),
        ("parent_cuda_visible_devices", "", "nonempty"),
        (
            "parent_cuda_visible_devices",
            ",".join(reversed(LEASES)),
            "mask",
        ),
    ],
)
def test_actual_admission_rejects_wrong_host_or_cuda_mask(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    frozen = tmp_path / "frozen"
    formal = tmp_path / "formal"
    frozen.mkdir()
    formal.mkdir()
    arguments = {
        "host_name": "zs-nj-tap-gpu18",
        "formal_v2_root": formal,
        "frozen_repo_root": frozen,
        "deployment_bundle_sha256": SHA,
        "no_gpu_receipt": _receipt(formal, frozen),
        "physical_bindings": _bindings(1),
        "ordered_lease_uuids": LEASES,
        "parent_cuda_visible_devices": ",".join(LEASES),
        "expected_lock_owner": "stage7-owner",
        "inventory_probe": _inventory,
        "lock_probe": lambda _uuid: "stage7-owner",
        "process_probe": lambda: [],
        "primitive_sha_probe": (
            lambda relative: physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256[relative]
        ),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        actual_pipeline_v2.validate_actual_runtime_admission(**arguments)


def test_actual_admission_rejects_no_gpu_truth_drift(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen"
    formal = tmp_path / "formal"
    frozen.mkdir()
    formal.mkdir()
    receipt = _receipt(formal, frozen)
    receipt["eligible_for_formal_finalization"] = True
    unsigned = {
        key: value for key, value in receipt.items() if key != "dry_run_receipt_sha256"
    }
    receipt["dry_run_receipt_sha256"] = _sha(unsigned)

    with pytest.raises(ValueError, match="no-GPU truth"):
        actual_pipeline_v2.validate_actual_runtime_admission(
            host_name="zs-nj-tap-gpu18",
            formal_v2_root=formal,
            frozen_repo_root=frozen,
            deployment_bundle_sha256=SHA,
            no_gpu_receipt=receipt,
            physical_bindings=_bindings(1),
            ordered_lease_uuids=LEASES,
            parent_cuda_visible_devices=",".join(LEASES),
            expected_lock_owner="stage7-owner",
            inventory_probe=_inventory,
            lock_probe=lambda _uuid: "stage7-owner",
            process_probe=lambda: [],
            primitive_sha_probe=(
                lambda relative: physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256[relative]
            ),
        )


def test_primitive_drift_fails_before_inventory_probe(tmp_path: Path) -> None:
    inventory_calls = 0

    def inventory_probe() -> list[dict[str, Any]]:
        nonlocal inventory_calls
        inventory_calls += 1
        return _inventory()

    with pytest.raises(RuntimeError, match="primitive SHA drift"):
        _admission(
            tmp_path,
            1,
            primitive_sha_probe=lambda _relative: "0" * 64,
            inventory_probe=inventory_probe,
        )

    assert inventory_calls == 0


class _PipelineFakes:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def quant_plan(
        self,
        *,
        row: Mapping[str, Any],
        output_path: Path,
        row_root: Path,
    ) -> Mapping[str, Any]:
        del row_root
        return {
            "arguments": ["--output-json", str(output_path)],
            "input_paths": [],
        }

    def quant_execute(self, **kwargs: Any) -> dict[str, Any]:
        candidate = str(kwargs["candidate_id"])
        self.calls.append(("quant", candidate))
        if kwargs["q_mode"] == "int8":
            arguments = list(kwargs["quant_arguments"])
            output = Path(arguments[arguments.index("--output-json") + 1])
            output.write_text(
                json.dumps(
                    {
                        "schema": "stage3_tvm_int8_quant_contract_v3",
                        "params": {"tensor": {"scale": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            status = "success"
        else:
            status = "skipped_fp16"
        return {"candidate_id": candidate, "status": status}

    def performance_artifacts(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("planner", "batch"))
        rows = kwargs["projection"]["rows"]
        ids = [row["candidate_id"] for row in rows]
        return {
            "schema_version": "stage7_authenticated_performance_artifacts_v2",
            "performance_artifacts_sha256": "c" * 64,
            "manifest": {
                "row_count": len(rows),
                "jobs": [
                    {
                        **dict(row),
                        "manifest_job_id": row["candidate_id"],
                    }
                    for row in rows
                ],
            },
            "performance_jobs": [
                {
                    "candidate_id": candidate,
                    "manifest_job_id": candidate,
                    "job_id": candidate,
                    "command": ["python", "runner.py", "--gpu", "0"],
                }
                for candidate in ids
            ],
        }

    def performance_execute(self, **kwargs: Any) -> dict[str, Any]:
        candidate = str(kwargs["authenticated_job"]["candidate_id"])
        self.calls.append(("performance", candidate))
        return {
            "candidate_id": candidate,
            "stage3_state_row": {
                "manifest_job_id": candidate,
                "status": "success",
            },
        }

    def ap_plan(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(("ap_plan", "batch"))
        output_root = Path(kwargs["output_root"])
        jobs = kwargs["performance_artifacts"]["manifest"]["jobs"]
        rows = []
        shards = {}
        for job in jobs:
            candidate = str(job["manifest_job_id"])
            row = {
                "schema_version": "stage5_ap_plan_v2",
                "candidate_id": candidate,
                "manifest_job_id": candidate,
                "performance_job_id": candidate,
            }
            shard = output_root / "shards" / f"{candidate}.jsonl"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows.append(row)
            shards[candidate] = shard
        return {"rows": rows, "shards": shards}

    def ap_execute(self, **kwargs: Any) -> dict[str, Any]:
        candidate = str(kwargs["candidate_id"])
        stage = str(kwargs["stage"])
        self.calls.append((stage, candidate))
        return {
            "candidate_id": candidate,
            "manifest_job_id": candidate,
            "stage": stage,
            "status": "success",
        }

    def finalizer_evidence(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(("finalizer_evidence", "batch"))
        ids = kwargs["performance_artifacts"]["manifest"]["jobs"]
        candidate_ids = [str(row["manifest_job_id"]) for row in ids]
        return {
            "performance_state_rows": [
                {"manifest_job_id": candidate, "status": "success"}
                for candidate in candidate_ids
            ],
            "ap_plan_rows": list(kwargs["ap_plan_rows"]),
            "ap_state_rows": [
                {"manifest_job_id": candidate, "status": "success"}
                for candidate in candidate_ids
            ],
            "structured_failure_reports": [],
            "lineage_inputs": [
                {"candidate_id": candidate, "artifact": f"raw-{candidate}"}
                for candidate in candidate_ids
            ],
            "execution_attempt": {
                "attempt_id": Path(kwargs["output_root"]).parents[0].name,
                "primitive_sha256": dict(physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256),
            },
        }

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("finalizer", "batch"))
        ids = [row["candidate_id"] for row in kwargs["projection"]["rows"]]
        return {
            "schema_version": "stage7_actual_v3_physical_terminal_batch_v2",
            "rows": [{"candidate_id": candidate} for candidate in ids],
            "physical_terminal_batch_sha256": "d" * 64,
        }


def _build_runner(
    tmp_path: Path,
    count: int,
    fakes: _PipelineFakes,
    *,
    projection: Mapping[str, Any] | None = None,
    source_result: Mapping[str, Any] | None = None,
    forbidden_components: Mapping[str, Any] | None = None,
    remote_artifact_root: Path | None = None,
):
    selected_projection = dict(projection or _projection(count))
    return actual_pipeline_v2.build_concrete_stage_runner(
        projection=selected_projection,
        logical_request={"rows": selected_projection["rows"]},
        selection_binding={"schema_version": "selection"},
        physical_plan={"logical_row_bindings": _bindings(count)},
        source_plan={"schema_version": "source-plan"},
        source_result=source_result or {"schema_version": "source-result", "rows": []},
        runtime_admission=_admission(tmp_path, max(count, 1)) if count else {},
        python_executable=Path(os.sys.executable).resolve(),
        remote_artifact_root=remote_artifact_root
        or tmp_path / "formal" / "remote-artifacts",
        quant_plan_builder=fakes.quant_plan,
        ap_plan_builder=fakes.ap_plan,
        finalizer_evidence_builder=fakes.finalizer_evidence,
        inventory_probe=_inventory,
        lock_probe=lambda _uuid: "stage7-owner",
        process_probe=lambda: [],
        primitive_sha_probe=(
            lambda relative: physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256[relative]
        ),
        performance_artifact_builder=fakes.performance_artifacts,
        quant_executor=fakes.quant_execute,
        performance_executor=fakes.performance_execute,
        ap_executor=fakes.ap_execute,
        physical_finalizer=fakes.finalize,
        forbidden_components=forbidden_components,
    )


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_runner_executes_five_isolated_stages_and_returns_physical_terminal(
    tmp_path: Path, count: int
) -> None:
    fakes = _PipelineFakes()
    runner = _build_runner(tmp_path, count, fakes)
    attempt = tmp_path / "execution" / "attempt_000"

    receipts = {stage: runner(stage, attempt) for stage in actual_pipeline_v2.STAGES}

    assert all(receipt["status"] == "success" for receipt in receipts.values())
    assert receipts["final"]["terminal_payload"]["rows"] == [
        {"candidate_id": f"candidate-{index}"} for index in range(count)
    ]
    assert (attempt / "quant/pipeline_output.json").is_file()
    assert (attempt / "performance/pipeline_output.json").is_file()
    assert (attempt / "ap/sanity_pipeline_output.json").is_file()
    assert (attempt / "ap/full_pipeline_output.json").is_file()
    assert (attempt / "final/pipeline_output.json").is_file()
    assert fakes.calls.count(("planner", "batch")) == 1
    assert fakes.calls.count(("ap_plan", "batch")) == 1
    assert fakes.calls.count(("finalizer", "batch")) == 1
    assert sum(call[0] == "performance" for call in fakes.calls) == count
    assert sum(call[0] == "sanity" for call in fakes.calls) == count
    assert sum(call[0] == "full" for call in fakes.calls) == count


def test_new_runner_resumes_from_immutable_prior_stage_output(tmp_path: Path) -> None:
    first_fakes = _PipelineFakes()
    attempt = tmp_path / "execution" / "attempt_000"
    _build_runner(tmp_path, 2, first_fakes)("quant", attempt)
    second_fakes = _PipelineFakes()

    receipt = _build_runner(tmp_path, 2, second_fakes)("performance", attempt)

    assert receipt["status"] == "success"
    assert ("quant", "candidate-0") not in second_fakes.calls
    assert ("planner", "batch") in second_fakes.calls


def test_zero_rows_never_probe_or_invoke_pipeline_dependencies(tmp_path: Path) -> None:
    fakes = _PipelineFakes()

    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("zero-row path called a physical dependency")

    projection = {
        "schema_version": "stage7_actual_v3_empty_physical_terminal_v2",
        "rows": [],
        "gpu_subprocess_count": 0,
        "empty_physical_terminal_sha256": "e" * 64,
    }
    runner = actual_pipeline_v2.build_concrete_stage_runner(
        projection=projection,
        logical_request={"rows": []},
        selection_binding={},
        physical_plan={"logical_row_bindings": []},
        source_plan={},
        source_result={},
        runtime_admission={},
        python_executable=Path(os.sys.executable),
        remote_artifact_root=tmp_path / "remote",
        quant_plan_builder=explode,
        ap_plan_builder=explode,
        finalizer_evidence_builder=explode,
        inventory_probe=explode,
        lock_probe=explode,
        process_probe=explode,
        primitive_sha_probe=explode,
        performance_artifact_builder=explode,
        quant_executor=explode,
        performance_executor=explode,
        ap_executor=explode,
        physical_finalizer=explode,
    )
    attempt = tmp_path / "execution" / "attempt_000"

    results = {stage: runner(stage, attempt) for stage in actual_pipeline_v2.STAGES}

    assert all(result["status"] == "success" for result in results.values())
    assert results["final"]["terminal_payload"] == projection
    assert fakes.calls == []


def test_forbidden_components_are_never_invoked(tmp_path: Path) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("forbidden component was invoked")

    forbidden = {
        name: explode
        for name in (
            "source_materializer",
            "selector",
            "advance",
            "promoter",
            "online_finalizer",
        )
    }
    fakes = _PipelineFakes()
    runner = _build_runner(tmp_path, 1, fakes, forbidden_components=forbidden)
    attempt = tmp_path / "execution" / "attempt_000"

    for stage in actual_pipeline_v2.STAGES:
        runner(stage, attempt)


def test_stage_attempt_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target" / "attempt_000"
    target.mkdir(parents=True)
    attempt = tmp_path / "attempt_000"
    attempt.symlink_to(target, target_is_directory=True)
    runner = _build_runner(tmp_path, 1, _PipelineFakes())

    with pytest.raises(ValueError, match="symlink"):
        runner("quant", attempt)


def test_quant_output_hardlink_is_rejected(tmp_path: Path) -> None:
    fakes = _PipelineFakes()
    original_execute = fakes.quant_execute

    def hardlinking_quant(**kwargs: Any) -> dict[str, Any]:
        result = original_execute(**kwargs)
        if kwargs["q_mode"] == "int8":
            arguments = list(kwargs["quant_arguments"])
            output = Path(arguments[arguments.index("--output-json") + 1])
            alias = tmp_path / "quant-contract-alias.json"
            os.link(output, alias)
        return result

    fakes.quant_execute = hardlinking_quant  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 1, fakes)

    with pytest.raises(ValueError, match="hardlink"):
        runner("quant", tmp_path / "execution" / "attempt_000")


def test_quant_output_escape_is_rejected_before_executor(tmp_path: Path) -> None:
    fakes = _PipelineFakes()

    def escape_plan(**_kwargs: Any) -> Mapping[str, Any]:
        return {
            "arguments": [
                "--output-json",
                str(tmp_path / "outside-contract.json"),
            ],
            "input_paths": [],
        }

    fakes.quant_plan = escape_plan  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 1, fakes)

    with pytest.raises(ValueError, match="output path"):
        runner("quant", tmp_path / "execution" / "attempt_000")


def test_source_evidence_symlink_is_rejected_before_planner(tmp_path: Path) -> None:
    evidence = tmp_path / "source-evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    link = tmp_path / "source-link.json"
    link.symlink_to(evidence)
    source_result = {
        "rows": [
            {
                "candidate_id": "candidate-0",
                "source_evidence_path": str(link),
                "source_evidence_file_sha256": hashlib.sha256(b"{}").hexdigest(),
            }
        ]
    }
    fakes = _PipelineFakes()
    runner = _build_runner(tmp_path, 1, fakes, source_result=source_result)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)

    with pytest.raises(ValueError, match="symlink"):
        runner("performance", attempt)
    assert ("planner", "batch") not in fakes.calls


def test_unknown_or_out_of_order_stage_is_rejected(tmp_path: Path) -> None:
    runner = _build_runner(tmp_path, 1, _PipelineFakes())
    attempt = tmp_path / "execution" / "attempt_000"

    with pytest.raises(ValueError, match="stage"):
        runner("materialize", attempt)
    with pytest.raises(ValueError, match="prior stage"):
        runner("performance", attempt)


def test_performance_failure_stops_before_ap(tmp_path: Path) -> None:
    fakes = _PipelineFakes()

    def failed_performance(**kwargs: Any) -> dict[str, Any]:
        return {
            "candidate_id": kwargs["authenticated_job"]["candidate_id"],
            "status": "failed",
        }

    fakes.performance_execute = failed_performance  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 1, fakes)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)

    result = runner("performance", attempt)

    assert result["status"] == "infrastructure_failure"
    assert ("ap_plan", "batch") not in fakes.calls
    with pytest.raises(ValueError, match="prior stage"):
        runner("sanity", attempt)


def test_quant_failure_cannot_unlock_performance(tmp_path: Path) -> None:
    fakes = _PipelineFakes()
    fakes.quant_execute = (  # type: ignore[method-assign]
        lambda **kwargs: {
            "candidate_id": kwargs["candidate_id"],
            "status": "failed",
        }
    )
    runner = _build_runner(tmp_path, 1, fakes)
    attempt = tmp_path / "execution" / "attempt_000"

    assert runner("quant", attempt)["status"] == "infrastructure_failure"
    with pytest.raises(ValueError, match="prior stage"):
        runner("performance", attempt)


@pytest.mark.parametrize(
    ("failed_stage", "next_stage"), [("sanity", "full"), ("full", "final")]
)
def test_ap_failure_cannot_unlock_next_stage(
    tmp_path: Path, failed_stage: str, next_stage: str
) -> None:
    fakes = _PipelineFakes()
    original = fakes.ap_execute

    def fail_selected_stage(**kwargs: Any) -> dict[str, Any]:
        result = original(**kwargs)
        if kwargs["stage"] == failed_stage:
            result["status"] = "failed"
        return result

    fakes.ap_execute = fail_selected_stage  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 1, fakes)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)
    runner("performance", attempt)
    runner("sanity", attempt)
    if failed_stage == "full":
        runner("full", attempt)

    with pytest.raises(ValueError, match="prior stage"):
        runner(next_stage, attempt)


@pytest.mark.parametrize("candidate_id", ["../../outside", "__pycache__", "a/b"])
def test_candidate_identity_cannot_become_a_path(
    tmp_path: Path, candidate_id: str
) -> None:
    projection = _projection(1)
    projection["rows"][0]["candidate_id"] = candidate_id
    projection["rows"][0]["row_id"] = candidate_id

    with pytest.raises(ValueError, match="candidate identity"):
        _build_runner(tmp_path, 1, _PipelineFakes(), projection=projection)


def test_runner_rejects_resigned_runtime_admission_drift(tmp_path: Path) -> None:
    admission = _admission(tmp_path, 1)
    admission["actual_cuda_visible_devices"] = ",".join(reversed(LEASES))
    unsigned = {
        key: value
        for key, value in admission.items()
        if key != "actual_runtime_admission_sha256"
    }
    admission["actual_runtime_admission_sha256"] = _sha(unsigned)
    fakes = _PipelineFakes()

    with pytest.raises(ValueError, match="runtime admission"):
        actual_pipeline_v2.build_concrete_stage_runner(
            projection=_projection(1),
            logical_request={"rows": _projection(1)["rows"]},
            selection_binding={},
            physical_plan={"logical_row_bindings": _bindings(1)},
            source_plan={},
            source_result={"rows": []},
            runtime_admission=admission,
            python_executable=Path(os.sys.executable),
            remote_artifact_root=tmp_path / "remote",
            quant_plan_builder=fakes.quant_plan,
            ap_plan_builder=fakes.ap_plan,
            finalizer_evidence_builder=fakes.finalizer_evidence,
            inventory_probe=_inventory,
            lock_probe=lambda _uuid: "stage7-owner",
            process_probe=lambda: [],
            primitive_sha_probe=(
                lambda relative: physical_runtime_v2.RUNTIME_PRIMITIVE_SHA256[relative]
            ),
        )


def test_performance_job_order_drift_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    fakes = _PipelineFakes()
    original = fakes.performance_artifacts

    def reversed_jobs(**kwargs: Any) -> dict[str, Any]:
        artifacts = original(**kwargs)
        artifacts["performance_jobs"] = list(reversed(artifacts["performance_jobs"]))
        return artifacts

    fakes.performance_artifacts = reversed_jobs  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 2, fakes)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)

    with pytest.raises(ValueError, match="job identity"):
        runner("performance", attempt)
    assert not any(call[0] == "performance" for call in fakes.calls)


def test_performance_manifest_order_drift_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    fakes = _PipelineFakes()
    original = fakes.performance_artifacts

    def reversed_manifest(**kwargs: Any) -> dict[str, Any]:
        artifacts = original(**kwargs)
        artifacts["manifest"]["jobs"] = list(reversed(artifacts["manifest"]["jobs"]))
        return artifacts

    fakes.performance_artifacts = reversed_manifest  # type: ignore[method-assign]
    runner = _build_runner(tmp_path, 2, fakes)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)

    with pytest.raises(ValueError, match="manifest identity"):
        runner("performance", attempt)
    assert not any(call[0] == "performance" for call in fakes.calls)


def test_remote_artifact_root_symlink_is_rejected_before_planner(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    target = formal / "target"
    target.mkdir(parents=True)
    link = formal / "remote-link"
    link.symlink_to(target, target_is_directory=True)
    fakes = _PipelineFakes()
    runner = _build_runner(tmp_path, 1, fakes, remote_artifact_root=link)
    attempt = tmp_path / "execution" / "attempt_000"
    runner("quant", attempt)

    with pytest.raises(ValueError, match="symlink"):
        runner("performance", attempt)
    assert ("planner", "batch") not in fakes.calls
