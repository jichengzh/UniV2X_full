from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.fcooper_execute_measurement_row_v2 import validate_control_request
from scripts.fcooper_stage6_control_request_v2 import (
    CONTROL_PHASES,
    aggregate_feedback,
    build_control_requests,
    canonical_sha256,
    validate_final_audit,
    validate_independent_ap_report,
    validate_candidate_plan_against_five_arm_plan,
    validate_bound_feedback_row,
    validate_request_complete_command,
    validate_trt_repeat_report,
    write_supervisor_complete,
    select_complete_arm_winner,
    repeat_spec,
    validate_original_contract,
)
from scripts.stage6_prepare_fcooper_five_arm_v2 import fixed_original_candidate


ROOT = Path(__file__).resolve().parents[2]


def candidate(index: int, *, width: list[int] | None = None, q_mode: str = "int8") -> dict:
    row_id = f"candidate-{index:02d}"
    selected_width = width or [32, 64, 128, 64, 128]
    return {
        "row_id": row_id,
        "manifest_job_id": row_id,
        "task_id": "S5-FCO-TRT-V2",
        "task_sha256": "a" * 64,
        "model": "fcooper",
        "hardware_id": "h800",
        "group_id": f"fcooper|{row_id}",
        "width": selected_width,
        "width_schema": [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ],
        "q_mode": q_mode,
        "source_evidence_sha256": f"{index + 1:064x}",
    }


def successful_feedback(row: dict, *, latency: float = 1.0) -> dict:
    feedback = {
        **row,
        "terminal_status": "measured_success_gold",
        "training_source": "online_feedback",
        "latency_ms": latency,
        "energy_j": 0.2,
        "ap70": 0.61,
        "checkpoint_sha256": "b" * 64,
        "recovery_training_report_sha256": "c" * 64,
    }
    feedback["actual_feedback_row_sha256"] = canonical_sha256(feedback)
    return feedback


def test_control_requests_are_content_addressed_and_chunked_4_plus_1() -> None:
    requests = build_control_requests(
        [candidate(index) for index in range(5)],
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )

    assert [request["batch_size"] for request in requests] == [4, 1]
    assert {request["task_id"] for request in requests} == {
        "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2"
    }
    assert all(request["arm_id"] == "compression_only" for request in requests)
    assert all(request["phase"] == "measure" for request in requests)
    assert all(
        request["measurement_request_sha256"]
        == canonical_sha256(
            {
                key: value
                for key, value in request.items()
                if key != "measurement_request_sha256"
            }
        )
        for request in requests
    )
    for request in requests:
        for row_index, row in enumerate(request["rows"]):
            assert validate_control_request(request, row_index=row_index) == row


def test_control_script_resolves_repo_modules_outside_repo_cwd(
    tmp_path: Path,
) -> None:
    request = build_control_requests(
        [candidate(0)],
        arm_id="schedule_only",
        phase="measure",
        builder_optimization_level=5,
    )[0]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fcooper_stage6_control_request_v2.py"),
            "request-row-count",
            "--request-json",
            str(request_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1"


def test_control_requests_reject_duplicate_rows_builder_drift_and_pilot() -> None:
    row = candidate(0)
    with pytest.raises(ValueError, match="unique"):
        build_control_requests(
            [row, row],
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )
    with pytest.raises(ValueError, match="builder"):
        build_control_requests(
            [row],
            arm_id="schedule_only",
            phase="measure",
            builder_optimization_level=0,
        )
    pilot = {
        **row,
        "source_contract": {
            "path": "/tmp/fcooper_workpackage_a_20260723/checkpoint.pth"
        },
    }
    with pytest.raises(ValueError, match="pilot"):
        build_control_requests(
            [pilot],
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )


def test_candidate_plan_must_match_frozen_five_arm_selection() -> None:
    rows = [candidate(index) for index in range(16)]
    plan = {
        "schema_version": "stage6_fcooper_five_arm_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arms": {
            "compression_only": {
                "selected_row_ids": [row["row_id"] for row in rows],
                "builder_optimization_level": 0,
            }
        },
    }
    candidate_plan = {
        "schema_version": "stage6_fcooper_arm_candidate_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arm_id": "compression_only",
        "phase": "measure",
        "builder_optimization_level": 0,
        "row_count": 16,
        "rows": rows,
    }

    validated = validate_candidate_plan_against_five_arm_plan(
        candidate_plan,
        plan,
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )

    assert [row["row_id"] for row in validated] == [
        row["row_id"] for row in rows
    ]
    injected = {**candidate_plan, "rows": [*rows[:-1], candidate(99)]}
    with pytest.raises(ValueError, match="five-arm plan"):
        validate_candidate_plan_against_five_arm_plan(
            injected,
            plan,
            arm_id="compression_only",
            phase="measure",
            builder_optimization_level=0,
        )


def test_schedule_candidate_is_bound_to_frozen_derivation_identity() -> None:
    row = {
        **candidate(
            0,
            width=[64, 128, 256, 128, 256],
            q_mode="fp32",
        ),
        "row_id": "schedule-row",
        "manifest_job_id": "schedule-row",
        "schedule_baseline_derivation": (
            "scanner_unique_original_structure_to_fixed_fp32"
        ),
        "schedule_baseline_derivation_sha256": "d" * 64,
    }
    arm = {
        "fixed_row_id": row["row_id"],
        "fixed_manifest_job_id": row["manifest_job_id"],
        "schedule_baseline_derivation": row["schedule_baseline_derivation"],
        "schedule_baseline_derivation_sha256": row[
            "schedule_baseline_derivation_sha256"
        ],
    }
    plan = {
        "schema_version": "stage6_fcooper_five_arm_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arms": {"schedule_only": arm},
    }
    candidate_plan = {
        "schema_version": "stage6_fcooper_arm_candidate_plan_v2",
        "task_id": "S5-FCO-TRT-V2",
        "arm_id": "schedule_only",
        "phase": "measure",
        "builder_optimization_level": 5,
        "row_count": 1,
        "rows": [row],
    }

    assert validate_candidate_plan_against_five_arm_plan(
        candidate_plan,
        plan,
        arm_id="schedule_only",
        phase="measure",
        builder_optimization_level=5,
    ) == [row]
    with pytest.raises(ValueError, match="schedule-only"):
        validate_candidate_plan_against_five_arm_plan(
            {
                **candidate_plan,
                "rows": [
                    {
                        **row,
                        "schedule_baseline_derivation_sha256": "e" * 64,
                    }
                ],
            },
            plan,
            arm_id="schedule_only",
            phase="measure",
            builder_optimization_level=5,
        )


def test_every_control_phase_has_an_explicit_task_id() -> None:
    assert CONTROL_PHASES == {
        ("schedule_only", "measure"): (
            "S6-FCO-TRT-SCHEDULE-ONLY-MEASURE-V2",
            5,
        ),
        ("compression_only", "measure"): (
            "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2",
            0,
        ),
        ("compress_then_tune", "screen"): (
            "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
            0,
        ),
        ("compress_then_tune", "tuned_remeasurement"): (
            "S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2",
            5,
        ),
    }


def test_feedback_aggregation_requires_every_content_bound_row(tmp_path: Path) -> None:
    requests = build_control_requests(
        [candidate(0), candidate(1)],
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )
    request = requests[0]
    artifact_root = tmp_path / "artifacts"
    first = request["rows"][0]
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{first['row_id']}".encode()
    ).hexdigest()[:16]
    feedback_path = artifact_root / "execution" / row_tag / "feedback_row.json"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(
        json.dumps(
            successful_feedback(
                {
                    **first,
                    "measurement_request_row_sha256": request["row_sha256"][
                        first["row_id"]
                    ],
                }
            )
        )
    )

    with pytest.raises(ValueError, match="complete"):
        aggregate_feedback(requests, artifact_root=artifact_root, expected_count=2)

    second = request["rows"][1]
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{second['row_id']}".encode()
    ).hexdigest()[:16]
    feedback_path = artifact_root / "execution" / row_tag / "feedback_row.json"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_text(
        json.dumps(
            successful_feedback(
                {
                    **second,
                    "measurement_request_row_sha256": request["row_sha256"][
                        second["row_id"]
                    ],
                },
                latency=0.9,
            )
        )
    )

    rows, audit = aggregate_feedback(
        requests, artifact_root=artifact_root, expected_count=2
    )

    assert [row["row_id"] for row in rows] == ["candidate-00", "candidate-01"]
    assert audit["passed"] is True
    assert audit["row_count"] == 2
    assert len(audit["feedback_sha256"]) == 2


def test_request_resume_reuses_only_complete_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = build_control_requests(
        [candidate(index) for index in range(5)],
        arm_id="compression_only",
        phase="measure",
        builder_optimization_level=0,
    )
    artifact_root = tmp_path / "artifacts"
    request_paths = []
    for request_index, request in enumerate(requests):
        request_path = tmp_path / f"request-{request_index}.json"
        request_path.write_text(json.dumps(request))
        request_paths.append(request_path)
    for row in requests[0]["rows"]:
        row_id = row["row_id"]
        row_tag = hashlib.sha256(
            f"{requests[0]['task_id']}:{row_id}".encode()
        ).hexdigest()[:16]
        feedback_path = artifact_root / "execution" / row_tag / "feedback_row.json"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(
            json.dumps(
                successful_feedback(
                    {
                        **row,
                        "measurement_request_row_sha256": requests[0]["row_sha256"][
                            row_id
                        ],
                    }
                )
            )
        )

    validate_request_complete_command(
        argparse.Namespace(
            request_json=request_paths[0],
            artifact_root=artifact_root,
        )
    )
    assert json.loads(capsys.readouterr().out)["request_complete"] is True
    with pytest.raises(ValueError, match="incomplete"):
        validate_request_complete_command(
            argparse.Namespace(
                request_json=request_paths[1],
                artifact_root=artifact_root,
            )
        )
    first, first_path = validate_bound_feedback_row(
        requests[0],
        row_index=0,
        artifact_root=artifact_root,
    )
    assert first["row_id"] == requests[0]["rows"][0]["row_id"]
    assert first_path.is_file()
    with pytest.raises(ValueError, match="missing bound row"):
        validate_bound_feedback_row(
            requests[1],
            row_index=0,
            artifact_root=artifact_root,
        )


def test_winner_selection_waits_for_complete_arm_feedback() -> None:
    rows = [successful_feedback(candidate(index), latency=2.0 - index / 10) for index in range(4)]
    with pytest.raises(ValueError, match="complete"):
        select_complete_arm_winner(
            rows[:3],
            expected_count=4,
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

    selected = select_complete_arm_winner(
        rows,
        expected_count=4,
        ap70_ref=0.63,
        max_ap_drop=0.10,
    )

    assert selected["row_id"] == "candidate-03"
    assert selected["selection_status"] == "selected_feasible"


def test_fixed_original_candidate_is_scanner_derived_fp32() -> None:
    original = candidate(
        0,
        width=[64, 128, 256, 128, 256],
        q_mode="fp32",
    )

    selected = fixed_original_candidate([candidate(1), original])

    assert selected["row_id"] == original["row_id"]
    with pytest.raises(ValueError, match="scanner-derived original"):
        fixed_original_candidate([candidate(1)])


def test_original_contract_preflight_binds_native_source_and_ap(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    ap_report = tmp_path / "ap.json"
    config.write_text("model: fcooper\n")
    checkpoint.write_bytes(b"checkpoint")
    ap_report.write_text('{"status": "success_full"}\n')
    contract = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ap_reference_report_path": str(ap_report),
        "ap_reference_report_sha256": hashlib.sha256(
            ap_report.read_bytes()
        ).hexdigest(),
    }

    audit = validate_original_contract(
        contract,
        source_config=config,
        source_checkpoint=checkpoint,
    )

    assert audit["passed"] is True
    drifted = {**contract, "ap_reference_report_sha256": "0" * 64}
    with pytest.raises(ValueError, match="AP reference"):
        validate_original_contract(
            drifted,
            source_config=config,
            source_checkpoint=checkpoint,
        )


def test_supervisor_uses_control_mode_four_lanes_and_dynamic_gpu7_repeats() -> None:
    source = (
        ROOT / "scripts/fcooper_stage6_five_arm_supervisor_v2.sh"
    ).read_text(encoding="utf-8")

    assert "--request-kind" in source
    assert "stage6-control" in source
    assert "MAX_PARALLEL=4" in source
    assert "REPEAT_GPU=7" in source
    assert "same_gpu7_repeat_" in source
    assert "feedback_history_final_t16.json" in source
    assert "fcooper_workpackage_a_20260723" in source
    assert "fcooper_execute_measurement_row_v1.py" not in source
    assert "stage5_measurement_request_v2" not in source
    assert "fcooper_trt_ap_bridge_v1.py" in source
    assert "stage6_finalize_fcooper_table1_v2.py" in source
    assert "fcooper_gpu_exclusivity_gate_v1.py" in source
    assert "run_repeat_guarded" in source
    assert " guard " in source
    assert "--lock-file" in source
    assert "--monitor-seconds" in source
    assert "gpu7_exclusivity" in source
    assert "SUCCESS=measured_success_gold" in source
    assert source.count('-- "$PYTHON" "$AP_RUNNER"') == 1
    assert "reusing validated native repeat" in source
    assert "reusing complete control phase" in source
    assert "reusing complete control request" in source
    assert "reusing complete control row" in source
    assert "reusing validated TRT repeat" in source
    assert "reusing validated independent AP" in source


def test_native_repeat_binding_is_idempotent() -> None:
    contract = {
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    report = {
        "status": "success",
        "precision": "fp32",
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "config_sha256": contract["config_sha256"],
    }
    from scripts.fcooper_stage6_control_request_v2 import bind_native_report

    first = bind_native_report(report, contract=contract, gpu_abs=7)
    second = bind_native_report(first, contract=contract, gpu_abs=7)

    assert second == first


def test_repeat_and_ap_reuse_require_bound_complete_evidence(
    tmp_path: Path,
) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    artifact_paths = {
        "compiled_engine": engine_dir / "compiled.engine",
        "engine_build_config": engine_dir / "engine_build_config.json",
        "engine_inspector": engine_dir / "engine_inspector.json",
        "calibration_cache": engine_dir / "calibration.cache",
        "calibration_manifest": engine_dir / "calibration_manifest.json",
    }
    for name, path in artifact_paths.items():
        path.write_bytes(name.encode())
    calibration_manifest_sha256 = hashlib.sha256(
        artifact_paths["calibration_manifest"].read_bytes()
    ).hexdigest()
    artifact_paths["engine_build_config"].write_text(
        json.dumps(
            {
                "builder_optimization_level": 5,
                "calibration_dataset": "OPV2V-validate",
                "calibration_manifest_sha256": calibration_manifest_sha256,
                "precision": "int8",
                "source_onnx_sha256": "a" * 64,
            }
        )
    )
    repeat = {
        "gpu_abs": 7,
        "precision": "int8",
        "builder_optimization_level": 5,
        "lat_p50_ms": 0.7,
        "energy_j": 0.3,
        "artifact_sha256": {
            "source_onnx": "a" * 64,
            **{
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in artifact_paths.items()
            },
        },
    }

    validated = validate_trt_repeat_report(
        repeat,
        expected_onnx_sha256="a" * 64,
        expected_precision="int8",
        expected_builder_level=5,
        expected_gpu=7,
        artifact_dir=engine_dir,
    )
    assert validated["passed"] is True

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    config.write_bytes(b"config")
    checkpoint.write_bytes(b"checkpoint")
    selected = {
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    ap_report = {
        "schema_version": "fcooper_trt_ap_report_v1",
        "status": "success_full",
        "dataset_samples": 2170,
        "processed_samples": 2170,
        "fallback_samples": 0,
        "failed_samples": 0,
        "engine_samples": 2170,
        "engine_calls": 2170,
        "engine_sha256": hashlib.sha256(
            artifact_paths["compiled_engine"].read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "numerical_contract": {
            "requested_samples": 2170,
            "fallback_samples": 0,
            "full_dataset_engine_execution": True,
            "silent_fallback_forbidden": True,
        },
        "ap30": selected["ap30"] + 1.0e-4,
        "ap50": selected["ap50"] - 1.0e-4,
        "ap70": selected["ap70"] + 1.0e-4,
    }
    assert validate_independent_ap_report(
        ap_report,
        expected=selected,
        engine_path=artifact_paths["compiled_engine"],
        config_path=config,
        checkpoint_path=checkpoint,
    )["passed"] is True

    with pytest.raises(ValueError, match="precision"):
        validate_trt_repeat_report(
            {**repeat, "precision": "fp16"},
            expected_onnx_sha256="a" * 64,
            expected_precision="int8",
            expected_builder_level=5,
            expected_gpu=7,
            artifact_dir=engine_dir,
        )
    artifact_paths["compiled_engine"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="artifact"):
        validate_trt_repeat_report(
            repeat,
            expected_onnx_sha256="a" * 64,
            expected_precision="int8",
            expected_builder_level=5,
            expected_gpu=7,
            artifact_dir=engine_dir,
        )
    artifact_paths["compiled_engine"].write_bytes(b"compiled_engine")
    build_config = json.loads(artifact_paths["engine_build_config"].read_text())
    artifact_paths["engine_build_config"].write_text(
        json.dumps({**build_config, "precision": "fp16"})
    )
    semantic_drift = {
        **repeat,
        "artifact_sha256": {
            **repeat["artifact_sha256"],
            "engine_build_config": hashlib.sha256(
                artifact_paths["engine_build_config"].read_bytes()
            ).hexdigest(),
        },
    }
    with pytest.raises(ValueError, match="build-config semantic"):
        validate_trt_repeat_report(
            semantic_drift,
            expected_onnx_sha256="a" * 64,
            expected_precision="int8",
            expected_builder_level=5,
            expected_gpu=7,
            artifact_dir=engine_dir,
        )
    artifact_paths["engine_build_config"].write_text(json.dumps(build_config))
    with pytest.raises(ValueError, match="full-2170"):
        validate_independent_ap_report(
            {**ap_report, "processed_samples": 2169},
            expected=selected,
            engine_path=artifact_paths["compiled_engine"],
            config_path=config,
            checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_independent_ap_report(
            {**ap_report, "checkpoint_sha256": "0" * 64},
            expected=selected,
            engine_path=artifact_paths["compiled_engine"],
            config_path=config,
            checkpoint_path=checkpoint,
        )
    with pytest.raises(ValueError, match="metric drift"):
        validate_independent_ap_report(
            {**ap_report, "ap70": selected["ap70"] + 0.01},
            expected=selected,
            engine_path=artifact_paths["compiled_engine"],
            config_path=config,
            checkpoint_path=checkpoint,
        )


def test_final_audit_gate_and_complete_marker_are_idempotent(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "final_audit.json"
    csv_path = tmp_path / "fcooper_stage6_trt_delta_ap_0.10_v2.csv"
    bundle_path = tmp_path / "fcooper_stage6_evidence_bundle_v2.json"
    audit = {
        "schema_version": "stage6_fcooper_five_arm_audit_v2",
        "task_id": "S5-FCO-TRT-V2",
        "paper_ready": True,
        "all_five_arms_credible_terminal": True,
        "successful_repeats_verified": True,
        "resource_guards_verified": True,
        "rows": [{}, {}, {}, {}, {}],
    }
    audit_path.write_text(json.dumps(audit))
    csv_path.write_text("method\n")
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "fcooper_stage6_evidence_bundle_v2",
                "task_id": "S5-FCO-TRT-V2",
                "audit_path": str(audit_path),
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "csv_path": str(csv_path),
                "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }
        )
    )

    validated = validate_final_audit(audit_path)
    assert validated["paper_ready"] is True

    marker = tmp_path / "complete.json"
    first = write_supervisor_complete(marker, final_audit_path=audit_path)
    second = write_supervisor_complete(marker, final_audit_path=audit_path)
    assert first == second
    assert json.loads(marker.read_text()) == first
    assert first["final_audit_sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()

    audit_path.write_text(json.dumps({**audit, "paper_ready": False}))
    with pytest.raises(ValueError, match="paper-ready"):
        validate_final_audit(audit_path)
    with pytest.raises(ValueError, match="drifted"):
        write_supervisor_complete(marker, final_audit_path=audit_path)


def test_supervisor_rejects_leading_zero_gpu7_identifier(tmp_path: Path) -> None:
    supervisor = ROOT / "scripts/fcooper_stage6_five_arm_supervisor_v2.sh"
    required = {
        "--formal-root": tmp_path / "formal",
        "--source-registry-json": tmp_path / "registry.json",
        "--observed-graphs-json": tmp_path / "graphs.json",
        "--coldstart-rows-json": tmp_path / "rows.json",
        "--coldstart-graphs-json": tmp_path / "coldgraphs.json",
        "--profiles-json": tmp_path / "profiles.json",
        "--frozen-contract-json": tmp_path / "formal/contracts/frozen_contract.json",
        "--artifact-root": tmp_path / "artifacts",
        "--heal-root": tmp_path / "heal",
        "--python": Path("/usr/bin/python3"),
        "--source-config": tmp_path / "config.yaml",
        "--source-checkpoint": tmp_path / "checkpoint.pth",
        "--recovery-contract": tmp_path / "recovery.json",
        "--calibration-dir": tmp_path / "calibration",
        "--calibration-summary": tmp_path / "calibration.json",
    }
    arguments = [str(supervisor)]
    for flag, value in required.items():
        arguments.extend([flag, str(value)])
    arguments.extend(["--gpus", "07", "--dry-run"])

    completed = subprocess.run(arguments, capture_output=True, text=True)

    assert completed.returncode == 2
    assert "GPU7 is reserved" in completed.stderr


def test_repeat_spec_binds_config_checkpoint_and_onnx(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    for path, content in (
        (config, b"config"),
        (checkpoint, b"checkpoint"),
        (onnx, b"onnx"),
    ):
        path.write_bytes(content)
    evidence = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(onnx),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }
    evidence_path = tmp_path / "source_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    selection = {
        "row_id": "winner",
        "terminal_status": "measured_success_gold",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp16",
        "materialized_source_evidence_path": str(evidence_path),
        "materialized_source_evidence_sha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "graph_features": {"onnx_sha256": evidence["onnx_sha256"]},
    }

    spec = repeat_spec(selection)

    assert spec["config_path"] == str(config)
    assert spec["checkpoint_path"] == str(checkpoint)


def test_repeat_spec_rejects_pruned_source_without_recovery_contract(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    onnx = tmp_path / "model.onnx"
    for path in (config, checkpoint, onnx):
        path.write_bytes(path.name.encode())
    evidence = {
        "config_path": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_path": str(onnx),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }
    evidence_path = tmp_path / "source_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    selection = {
        "row_id": "pruned",
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "fp16",
        "materialized_source_evidence_path": str(evidence_path),
        "materialized_source_evidence_sha256": hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "graph_features": {"onnx_sha256": evidence["onnx_sha256"]},
    }

    with pytest.raises((KeyError, ValueError), match="recovery"):
        repeat_spec(selection)
