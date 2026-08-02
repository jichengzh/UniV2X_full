from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage7 import source_resolution_v2
from framework.stage7 import source_round_orchestration_v2 as source_round
from scripts import stage7_core_no_gpu_dry_run_v2 as dry_run


RELEASE_PIN = "d" * 64
MANIFEST_FILE_PIN = "e" * 64


def test_isolated_json_writer_recovers_only_identical_stale_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text('{\n  "ok": true\n}\n', encoding="utf-8")

    dry_run._write_isolated_json(target, {"ok": True})

    assert json.loads(target.read_text()) == {"ok": True}
    assert not temporary.exists()


def test_isolated_json_writer_rejects_drifted_stale_temp(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.with_suffix(".json.tmp").write_text('{"forged":true}\n')

    with pytest.raises(ValueError, match="temporary artifact drift"):
        dry_run._write_isolated_json(target, {"ok": True})


def test_isolated_json_writer_rejects_existing_symlink_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    backing = tmp_path / "backing.json"
    dry_run._write_isolated_json(backing, {"ok": True})
    target.symlink_to(backing)

    with pytest.raises(ValueError, match="not a regular file"):
        dry_run._write_isolated_json(target, {"ok": True})


def test_isolated_json_writer_rejects_world_writable_stale_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "receipt.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text('{\n  "ok": true\n}\n', encoding="utf-8")
    temporary.chmod(0o666)

    with pytest.raises(ValueError, match="temporary artifact drift"):
        dry_run._write_isolated_json(target, {"ok": True})


def test_cli_prepares_fresh_root_once_before_no_gpu_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs_path = tmp_path / "prepare-inputs.json"
    manifest_path = tmp_path / "embedded-source-manifest.json"
    onnx_path = tmp_path / "frozen.onnx"
    inputs_path.write_text('{"pre_scan_registry":[]}\n')
    manifest_path.write_text('{"schema_version":"frozen"}\n')
    onnx_path.write_bytes(b"frozen")
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        dry_run,
        "prepare_and_initialize_no_gpu_root",
        lambda **kwargs: calls.append(("prepare", kwargs))
        or {"trajectory_count": 12},
    )
    monkeypatch.setattr(
        dry_run,
        "run_no_gpu_dry_run",
        lambda root, **kwargs: calls.append(("run", {"root": root, **kwargs}))
        or {"paper_ready": False, "formal_v2_gpu_jobs_launched": 0},
    )
    base_args = [
        "--v2-root", str(tmp_path / "formal-v2"),
        "--repo-root", str(tmp_path),
        "--frozen-onnx", str(onnx_path),
        "--frozen-onnx-sha256", "a" * 64,
        "--expected-release-sha256", RELEASE_PIN,
        "--expected-manifest-file-sha256", MANIFEST_FILE_PIN,
        "--v1-root", str(tmp_path / "v1"),
        "--prepare-inputs-json", str(inputs_path),
        "--embedded-source-manifest-json", str(manifest_path),
    ]
    with pytest.raises(ValueError, match="persistent --orchestrator-pid"):
        dry_run.main(base_args)
    result = dry_run.main([*base_args, "--orchestrator-pid", "4242"])

    assert result == 0
    assert [name for name, _payload in calls] == ["prepare", "run"]
    assert calls[0][1]["embedded_source_manifest"] == {"schema_version": "frozen"}
    assert calls[0][1]["orchestrator_pid"] == 4242
    assert json.loads(capsys.readouterr().out)["formal_v2_gpu_jobs_launched"] == 0


def test_no_gpu_api_fails_closed_when_deployment_bundle_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path / "formal-v2"
    root.mkdir()
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"frozen")

    with pytest.raises(ValueError, match="authenticated deployment bundle"):
        dry_run.run_no_gpu_dry_run(
            root,
            repo_root=tmp_path,
            frozen_onnx=onnx,
            expected_release_sha256=RELEASE_PIN,
            expected_manifest_file_sha256=MANIFEST_FILE_PIN,
            initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
            validate_root=lambda *_args, **_kwargs: {},
            prepare_round=lambda *_args, **_kwargs: {},
        )


def _real_source_request(root: Path) -> dict[str, object]:
    rows = []
    specs = (
        ([16, 32, 64], "fp16"),
        ([16, 32, 64], "int8"),
        ([24, 48, 96], "fp16"),
        ([24, 48, 96], "int8"),
    )
    for index, (width, q_mode) in enumerate(specs):
        group = "x".join(map(str, width))
        candidate_id = f"pyramid|{group}|q={q_mode}|candidate={index}"
        source_root = root / f"source-{index}"
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
            "strategy_id": f"candidate-{index}",
            "model": "pyramid",
            "group_id": f"pyramid|{group}",
            "width": width,
            "genome": [*width, q_mode],
            "q_mode": q_mode,
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
            "capability_digest": "b" * 64,
            "dispatch_key": "tvm_auto",
            "source_status": "planned",
            "source_contract": {
                "checkpoint_path": str(source_root / "checkpoint.pth"),
                "checkpoint_sha256": None,
                "onnx_path": str(source_root / "model.onnx"),
                "onnx_sha256": None,
                "calibration_npz": str(source_root / "calibration.npz"),
                "calibration_summary": str(source_root / "summary.json"),
                "source_done_marker": str(source_root / "source.done"),
                "trt_calibration_dir": str(source_root / "trt_npy"),
            },
            "graph_features": {"conv_count": 51},
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
        rows.append(row)
    unsigned = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {
            row["row_id"]: dry_run.canonical_sha256(row) for row in rows
        },
        "rows": rows,
    }
    return {
        **unsigned,
        "measurement_request_sha256": dry_run.canonical_sha256(unsigned),
    }


def test_real_synthetic_bind_path_closes_fresh_round_without_formal_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-v2"
    directory = root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    request = _real_source_request(tmp_path / "planned")
    plan = source_resolution_v2.build_source_resolution_plan(request)
    selection = {
        "measurement_request": request,
        "acquisition": {
            "selected_row_ids": [row["row_id"] for row in request["rows"]]
        },
    }
    identity = source_round.freeze_selection_identity(selection)
    for name, payload in {
        "logical_request.json": request,
        "selector_output.json": selection,
        "selection_binding.json": identity,
        "source_resolution_plan.json": plan,
    }.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    validated = {"root": root, "contract": {"contract_sha256": "c" * 64}}

    def exact_dimensions(
        logical_request: dict[str, object],
        *,
        resolved_sources_by_candidate: dict[str, dict[str, object]],
        **_kwargs: object,
    ) -> dict[str, dict[str, object]]:
        dimensions = {}
        for row in logical_request["rows"]:
            candidate_id = row["row_id"]
            source = resolved_sources_by_candidate[candidate_id]
            dimensions[candidate_id] = {
                "candidate_id": candidate_id,
                "model": row["model"],
                "capability_profile_id": row["capability_profile_id"],
                "hardware_id": row["hardware_id"],
                "measurement_scope": "synthetic_protocol_only",
                "input_protocol_sha256": "1" * 64,
                "batch_size": 1,
                "genome": row["genome"],
                "q_mode": row["q_mode"],
                "source_checkpoint_sha256": source["synthetic_checkpoint_sha256"],
                "onnx_sha256": source["synthetic_onnx_sha256"],
                "build_protocol_sha256": "2" * 64,
                "tuning_protocol_sha256": "3" * 64,
                "measurement_protocol_sha256": "4" * 64,
                "ap_protocol_sha256": "5" * 64,
                "dispatch_key": row["dispatch_key"],
                "runtime_contract_sha256": "6" * 64,
            }
        return dimensions

    def real_bind(target: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs.pop("expected_release_sha256") == "d" * 64
        assert kwargs.pop("expected_manifest_file_sha256") == "e" * 64
        return source_round.bind_reveal_after_source_ready(
            target,
            validate_root=lambda *_args, **_kwargs: validated,
            write_json=lambda path, payload, **_kwargs: dry_run._write_isolated_json(
                path, payload
            ),
            selector_inputs_builder=lambda _validated: {},
            exact_dimensions_builder=exact_dimensions,
            cache_snapshot_builder=lambda _root: pytest.fail(
                "synthetic bind must not inspect formal cache"
            ),
            **kwargs,
        )

    result = dry_run.close_synthetic_source_protocol(
        root,
        variant="full",
        prepared={
            "logical_request_sha256": request["measurement_request_sha256"],
            "source_resolution_plan_sha256": plan[
                "source_resolution_plan_sha256"
            ],
            "cache_membership_observed": False,
            "exact_binding_frozen": False,
        },
        repo_root=tmp_path,
        audit_dir=root / "audits/no_gpu_dry_run",
        expected_release_sha256="d" * 64,
        expected_manifest_file_sha256="e" * 64,
        online_module=SimpleNamespace(
            source_resolution=source_resolution_v2,
            bind_reveal_after_source_ready=real_bind,
        ),
    )

    assert result["closure"]["actual_v3_hardware_evidence"] is False
    assert result["closure"]["formal_miss_plan_allowed"] is False
    assert result["closure"]["executor_admission_allowed"] is False
    assert result["receipt"]["controller_state"] == "SYNTHETIC_PROTOCOL_CLOSED"
    assert not (directory / "source_resolution_result.json").exists()
    assert not (directory / "cache_reveal.json").exists()
    assert not (directory / "miss_only_physical_request.json").exists()
    assert not (directory / "executor_admission.json").exists()
