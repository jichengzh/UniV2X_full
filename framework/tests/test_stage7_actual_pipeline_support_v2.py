from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from framework.stage7 import actual_pipeline_support_v2 as support


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_quant_plan_uses_only_frozen_source_contract_inputs(tmp_path: Path) -> None:
    onnx = tmp_path / "model.onnx"
    calibration = tmp_path / "calibration.npz"
    summary = tmp_path / "calibration.json"
    for path in (onnx, calibration, summary):
        path.write_bytes(path.name.encode("utf-8"))
    output = tmp_path / "attempt/quant/contract.json"

    plan = support.build_quant_plan(
        row={
            "q_mode": "int8",
            "source_contract": {
                "onnx_path": str(onnx),
                "calibration_npz": str(calibration),
                "calibration_summary": str(summary),
            },
        },
        output_path=output,
        row_root=output.parent,
    )

    assert plan["arguments"] == [
        "--onnx",
        str(onnx),
        "--calibration-npz",
        str(calibration),
        "--calibration-summary",
        str(summary),
        "--output-json",
        str(output),
    ]
    assert plan["input_paths"] == [str(onnx), str(calibration), str(summary)]


def test_fp16_quant_plan_does_not_invent_quant_inputs(tmp_path: Path) -> None:
    output = tmp_path / "attempt/quant/contract.json"

    plan = support.build_quant_plan(
        row={"q_mode": "fp16", "source_contract": {}},
        output_path=output,
        row_root=output.parent,
    )

    assert plan == {"arguments": ["--output-json", str(output)], "input_paths": []}

    with pytest.raises(ValueError, match="fp16 or int8"):
        support.build_quant_plan(
            row={"q_mode": "fp32"},
            output_path=output,
            row_root=output.parent,
        )
    with pytest.raises(ValueError, match="incomplete"):
        support.build_quant_plan(
            row={"q_mode": "int8", "source_contract": {"onnx_path": "/model"}},
            output_path=output,
            row_root=output.parent,
        )


def test_inventory_probe_parses_exact_uuid_index_and_h800_model() -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout=("0, GPU-a, NVIDIA H800 80GB HBM3\n" "3, GPU-b, NVIDIA H800 PCIe\n"),
        stderr="",
    )

    rows = support.query_h800_inventory(run_command=lambda *_args, **_kwargs: completed)

    assert rows == [
        {
            "global_index": 0,
            "uuid": "GPU-a",
            "name": "NVIDIA H800 80GB HBM3",
        },
        {"global_index": 3, "uuid": "GPU-b", "name": "NVIDIA H800 PCIe"},
    ]


def test_inventory_probe_rejects_command_failure_or_malformed_rows() -> None:
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        support.query_h800_inventory(
            run_command=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="failed"
            )
        )
    with pytest.raises(ValueError, match="malformed"):
        support.query_h800_inventory(
            run_command=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0, stdout="not,a,valid,row\n", stderr=""
            )
        )


def test_scheduler_controller_admission_binds_ancestor_request_and_leases(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "scheduler_state.json"
    state = {
        "schema_version": "stage7_ablation_scheduler_state_v1",
        "controllers": {
            "controller-a": {
                "controller_id": "controller-a",
                "pid": 4321,
                "status": "running",
                "request_sha256": "a" * 64,
                "gpu_uuids": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    binding = support.validate_scheduler_lease(
        scheduler_state_path=state_path,
        controller_id="controller-a",
        logical_request_sha256="a" * 64,
        ordered_lease_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        ancestor_pids=(9999, 4321, 1),
    )

    assert binding["expected_lock_owner"] == "controller-a"
    assert binding["controller_pid"] == 4321
    assert binding["scheduler_state_file_sha256"] == _sha_file(state_path)

    lock_probe = support.build_scheduler_lock_probe(
        scheduler_state_path=state_path,
        controller_id="controller-a",
        logical_request_sha256="a" * 64,
        ordered_lease_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        ancestor_pids=(9999, 4321, 1),
    )
    assert lock_probe("GPU-c") == "controller-a"
    state["controllers"]["controller-a"]["status"] = "succeeded"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler lease"):
        lock_probe("GPU-c")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "succeeded"),
        ("request_sha256", "b" * 64),
        ("gpu_uuids", ["GPU-a", "GPU-b", "GPU-c", "GPU-x"]),
        ("pid", 9876),
    ],
)
def test_scheduler_controller_admission_rejects_stale_or_foreign_lease(
    tmp_path: Path, field: str, value: Any
) -> None:
    state_path = tmp_path / "scheduler_state.json"
    controller = {
        "controller_id": "controller-a",
        "pid": 4321,
        "status": "running",
        "request_sha256": "a" * 64,
        "gpu_uuids": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
    }
    controller[field] = value
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "stage7_ablation_scheduler_state_v1",
                "controllers": {"controller-a": controller},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scheduler lease"):
        support.validate_scheduler_lease(
            scheduler_state_path=state_path,
            controller_id="controller-a",
            logical_request_sha256="a" * 64,
            ordered_lease_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
            ancestor_pids=(9999, 4321, 1),
        )


def test_scheduler_controller_admission_rejects_hardlinked_state(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "scheduler_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "stage7_ablation_scheduler_state_v1",
                "controllers": {},
            }
        ),
        encoding="utf-8",
    )
    os.link(state_path, tmp_path / "scheduler_state_alias.json")

    with pytest.raises(ValueError, match="canonical"):
        support.validate_scheduler_lease(
            scheduler_state_path=state_path,
            controller_id="controller-a",
            logical_request_sha256="a" * 64,
            ordered_lease_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
            ancestor_pids=(4321, 1),
        )


def test_ap_plan_builder_uses_existing_stage5_plan_function_and_shards(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def build_ap_plan(manifest: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        calls.append({"manifest": manifest, **kwargs})
        return [
            {
                "schema_version": "stage5_ap_plan_v2",
                "manifest_job_id": "candidate-0",
                "performance_job_id": "physical-job-0",
            }
        ]

    result = support.build_ap_plan(
        performance_artifacts={
            "manifest": {"row_count": 1, "jobs": [{"job_id": "candidate-0"}]},
            "performance_jobs": [
                {"job_id": "physical-job-0", "manifest_job_id": "candidate-0"}
            ],
        },
        performance_executions=[
            {
                "stage3_state_row": {
                    "job_id": "physical-job-0",
                    "status": "success",
                }
            }
        ],
        output_root=tmp_path / "ap-plan",
        ap_plan_function=build_ap_plan,
    )

    assert len(calls) == 1
    assert calls[0]["expected_row_count"] == 1
    shard = Path(result["shards"]["candidate-0"])
    assert shard.is_file()
    assert json.loads(shard.read_text())["manifest_job_id"] == "candidate-0"


def test_ap_plan_builder_rejects_symlinked_shard_parent_without_external_write(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ap-plan"
    outside = tmp_path / "outside"
    outside.mkdir()
    output.mkdir()
    (output / "shards").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        support.build_ap_plan(
            performance_artifacts={
                "manifest": {"row_count": 1, "jobs": [{"job_id": "candidate-0"}]},
                "performance_jobs": [
                    {
                        "job_id": "physical-job-0",
                        "manifest_job_id": "candidate-0",
                    }
                ],
            },
            performance_executions=[
                {
                    "stage3_state_row": {
                        "job_id": "physical-job-0",
                        "status": "success",
                    }
                }
            ],
            output_root=output,
            ap_plan_function=lambda *_args, **_kwargs: [
                {
                    "schema_version": "stage5_ap_plan_v2",
                    "manifest_job_id": "candidate-0",
                    "performance_job_id": "physical-job-0",
                }
            ],
        )

    assert list(outside.iterdir()) == []


def test_subprocess_runner_is_argv_only_and_preserves_environment() -> None:
    observed: dict[str, Any] = {}

    def run_command(argv: list[str], **kwargs: Any) -> Any:
        observed.update({"argv": argv, **kwargs})
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    result = support.run_argv(
        ["python", "worker.py", "--value", "hostile ; $(touch nope)"],
        {**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-a"},
        run_command=run_command,
    )

    assert observed["argv"][-1] == "hostile ; $(touch nope)"
    assert observed["shell"] is False
    assert result == {"returncode": 0, "stdout": "ok", "stderr": ""}


def test_finalizer_evidence_reads_stage3_state_and_binds_raw_artifacts(
    tmp_path: Path,
) -> None:
    performance_result = tmp_path / "performance.json"
    performance_result.write_text('{"latency_ms": 1.0, "energy_j": 2.0}')
    ap_report = tmp_path / "full_ap.json"
    ap_report.write_text('{"ap30": 0.8, "ap50": 0.7, "ap70": 0.6}')
    sanity_state = tmp_path / "sanity.jsonl"
    sanity_state.write_text(
        json.dumps(
            {
                "manifest_job_id": "candidate-0",
                "stage": "sanity",
                "status": "success",
            }
        )
        + "\n"
    )
    full_state = tmp_path / "full.jsonl"
    full_state.write_text(
        json.dumps(
            {
                "manifest_job_id": "candidate-0",
                "stage": "full",
                "status": "success",
                "report_path": str(ap_report),
                "report_sha256": _sha_file(ap_report),
            }
        )
        + "\n"
    )
    deployment_sha = "d" * 64

    evidence = support.build_finalizer_evidence(
        performance_artifacts={
            "manifest": {"jobs": [{"job_id": "candidate-0"}]},
        },
        performance_executions=[
            {
                "candidate_id": "candidate-0",
                "stage3_state_row": {
                    "job_id": "candidate-0",
                    "status": "success",
                    "result_json": str(performance_result),
                    "result_sha256": _sha_file(performance_result),
                },
            }
        ],
        ap_plan_rows=[
            {
                "manifest_job_id": "candidate-0",
                "performance_job_id": "candidate-0",
            }
        ],
        sanity_executions=[
            {
                "candidate_id": "candidate-0",
                "argv": ["python", "ap.py", "--state-jsonl", str(sanity_state)],
            }
        ],
        full_executions=[
            {
                "candidate_id": "candidate-0",
                "argv": ["python", "ap.py", "--state-jsonl", str(full_state)],
            }
        ],
        output_root=tmp_path / "attempt_000/final",
        deployment_bundle_sha256=deployment_sha,
        primitive_sha256={"primitive.py": "e" * 64},
    )

    assert evidence["performance_state_rows"][0]["status"] == "success"
    assert [row["stage"] for row in evidence["ap_state_rows"]] == [
        "sanity",
        "full",
    ]
    assert evidence["structured_failure_reports"] == []
    assert evidence["lineage_inputs"] == [
        {
            "candidate_id": "candidate-0",
            "stage3_performance_artifact": {
                "path": str(performance_result),
                "artifact_sha256": _sha_file(performance_result),
            },
            "stage3_ap_artifact": {
                "path": str(ap_report),
                "artifact_sha256": _sha_file(ap_report),
            },
        }
    ]
    attempt = evidence["execution_attempt"]
    assert attempt["attempt_id"] == "attempt_000"
    assert attempt["deployment_bundle_sha256"] == deployment_sha
    assert attempt["execution_attempt_sha256"] == support.canonical_sha256(
        {
            key: value
            for key, value in attempt.items()
            if key != "execution_attempt_sha256"
        }
    )


def test_related_process_probe_marks_leased_foreign_process_as_related() -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout="GPU-a, 1234\nGPU-x, 5678\n",
        stderr="",
    )

    rows = support.query_related_processes(
        leased_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        controller_pid=99,
        formal_v2_root=Path("/formal"),
        run_command=lambda *_args, **_kwargs: completed,
        process_probe=lambda pid: {
            "pid": pid,
            "owner": "other",
            "command": ("python", "/foreign/job.py"),
            "ancestor_pids": (pid, 1),
        },
    )

    assert rows[0]["related"] is True
    assert rows[0]["controller_owned"] is False
    assert rows[1]["related"] is False

    with pytest.raises(RuntimeError, match="process query"):
        support.query_related_processes(
            leased_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
            controller_pid=99,
            formal_v2_root=Path("/formal"),
            run_command=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="unavailable"
            ),
            process_probe=lambda _pid: None,
        )


def test_linux_process_identity_reads_owner_command_and_ancestry(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    for pid, parent in ((30, 20), (20, 1), (1, 0)):
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text(
            f"Name:\tpython\nPPid:\t{parent}\nUid:\t1000\t1000\t1000\t1000\n"
        )
        (directory / "cmdline").write_bytes(
            f"python\0/formal/job-{pid}.py\0".encode("utf-8")
        )

    identity = support.inspect_linux_process(
        30,
        proc_root=proc,
        uid_name_probe=lambda uid: f"user-{uid}",
    )

    assert identity == {
        "pid": 30,
        "owner": "user-1000",
        "command": ("python", "/formal/job-30.py"),
        "ancestor_pids": (30, 20, 1),
    }
